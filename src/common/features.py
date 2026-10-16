"""Feature engineering shared by training AND streaming (so there is no train/serve skew).

For each service we look at the last 15 minutes (excluding the current one) and compute a
ROBUST baseline: median and MAD (median absolute deviation). Median/MAD are not distorted by a
fault that has already been running for a few minutes, unlike mean/std.
"""
import numpy as np
import pandas as pd

from src.common.config import SERVICE_IDS

METRICS = ["req_count", "error_rate", "avg_latency", "p95_latency"]
ABS_FLOOR = {"req_count": 5.0, "error_rate": 0.004, "avg_latency": 2.0, "p95_latency": 3.0}
WINDOW, MIN_PERIODS = 15, 3

PEER_COLS = ["peer_p95_z_max", "peer_error_z_max"]
RISK_FEATURES = [f for c in METRICS for f in (c, f"{c}_base", f"{c}_ratio", f"{c}_z", f"{c}_d2")] \
    + ["service_id"] + PEER_COLS
IF_FEATURES = [f"{c}_{s}" for c in METRICS for s in ("ratio", "z")]


def _mad(x):
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def build_features(df, service):
    """df: ONE service, ascending time, columns = METRICS. Returns one feature row per input row."""
    df = df.reset_index(drop=True)
    prior = df[METRICS].shift(1).rolling(WINDOW, min_periods=MIN_PERIODS)
    base = prior.median()
    scale = prior.apply(_mad, raw=True)
    out = pd.DataFrame(index=df.index)
    for c in METRICS:
        floor = np.maximum(0.05 * base[c].abs(), ABS_FLOOR[c])
        sigma = np.maximum(scale[c], floor)
        eps = ABS_FLOOR[c] * 0.5
        out[c] = df[c]
        out[f"{c}_base"] = base[c]
        out[f"{c}_ratio"] = (df[c] + eps) / (base[c].abs() + eps)
        out[f"{c}_z"] = (df[c] - base[c]) / sigma
        out[f"{c}_d2"] = df[c] - df[c].shift(2)
    out["service_id"] = SERVICE_IDS[service]
    return out


def rule_signals(f):
    """Classic statistical rules on the robust z-scores. Returns a list of signal names."""
    s = []
    if f["error_rate_z"] > 4 and f["error_rate"] > 0.03:
        s.append("error_spike")
    if f["p95_latency_z"] > 4 and f["p95_latency_ratio"] > 1.6:
        s.append("latency_high")
    if f["req_count_z"] > 4 and f["req_count_ratio"] > 1.8:
        s.append("traffic_burst")
    if f["req_count_z"] < -4 and f["req_count_ratio"] < 0.4:
        s.append("traffic_drop")
    return s


def label_next_minutes(incident_active, horizon=5):
    """1 if an incident is active at ANY of the next `horizon` minutes (t+1 .. t+horizon)."""
    ia = pd.Series(np.asarray(incident_active, dtype=float))
    fut = ia.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1].shift(-1)
    return fut.fillna(0).astype(int).to_numpy()


def _clip_z(x):
    return float(np.clip(x, -20, 50))


def peer_features_frame(data):
    """Training: for every (minute, service) row, the worst latency/error z-score among the OTHER services."""
    out = data.copy()
    for src, dst in (("p95_latency_z", "peer_p95_z_max"), ("error_rate_z", "peer_error_z_max")):
        wide = data.pivot(index="window_start", columns="service", values=src).clip(-20, 50)
        peer = {s: wide.drop(columns=s).max(axis=1, skipna=True) for s in wide.columns}
        out[dst] = [peer[s].get(t, np.nan) for s, t in zip(data["service"], data["window_start"])]
        out[dst] = out[dst].fillna(0.0)
    return out


def peer_features_row(service, minute_feats):
    """Streaming: same quantity from {service: feature-row} of the current minute."""
    others = [f for s, f in minute_feats.items() if s != service]
    p95 = [_clip_z(f["p95_latency_z"]) for f in others if pd.notna(f["p95_latency_z"])]
    err = [_clip_z(f["error_rate_z"]) for f in others if pd.notna(f["error_rate_z"])]
    return {"peer_p95_z_max": max(p95) if p95 else 0.0, "peer_error_z_max": max(err) if err else 0.0}