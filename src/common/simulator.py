"""Statistical model of 5 microservices, used by BOTH the offline training-data generator
and the live Kafka producer, so the ML model is trained on the same kind of data it sees live.

Each service has a normal profile (traffic, error rate, latency). A Fault changes that profile
gradually (ramp-up) or suddenly, and also tells us the ground truth ("is an incident active?").
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.config import SERVICES

# events per second, probability of a 5xx error, median latency in ms
PROFILES = {
    "gateway":  {"rate": 12.0, "err": 0.010, "lat": 45.0},
    "auth":     {"rate": 6.0,  "err": 0.008, "lat": 30.0},
    "orders":   {"rate": 8.0,  "err": 0.012, "lat": 80.0},
    "payments": {"rate": 4.0,  "err": 0.015, "lat": 120.0},
    "database": {"rate": 10.0, "err": 0.005, "lat": 15.0},
}
LATENCY_SIGMA = 0.35                      # log-normal spread of latency
RAMP_SECONDS = {"error_spike": 150, "latency_degradation": 180,
                "traffic_burst": 30, "service_down": 20}
MODES = list(RAMP_SECONDS)
INCIDENT_K = 0.5                          # fault intensity >= 0.5 counts as an active incident


@dataclass
class Fault:
    service: str
    mode: str          # error_spike | latency_degradation | traffic_burst | service_down | blip
    start: float       # epoch seconds
    end: float
    ramp: float        # seconds to reach full intensity
    severity: float = 1.0

    def intensity(self, t):
        if t < self.start or t > self.end:
            return 0.0
        return min(1.0, (t - self.start) / max(self.ramp, 1e-6)) * self.severity


def make_faults(scenario, service, start, duration):
    """Turn a scenario name into one or more Fault objects."""
    if scenario == "cascading_failure":      # database slows down -> upstream services fail
        end = start + duration
        return [Fault("database", "latency_degradation", start, end, 150),
                Fault("payments", "error_spike", start + 120, end, 90),
                Fault("orders", "error_spike", start + 180, end, 90),
                Fault("gateway", "latency_degradation", start + 240, end, 90, severity=0.7)]
    return [Fault(service, scenario, start, start + duration, RAMP_SECONDS[scenario])]


def diurnal(t):
    """Traffic is higher in the afternoon (UTC) than at night."""
    hour = (t % 86400) / 3600.0
    return 1.0 + 0.35 * np.sin(2 * np.pi * (hour - 8) / 24.0)


def service_state(service, t, faults):
    """Traffic rate, error probability, median latency and ground-truth intensity at time t."""
    p = PROFILES[service]
    rate, err, lat = p["rate"] * diurnal(t), p["err"], p["lat"]
    k, mode = 0.0, "none"
    for f in faults:
        if f.service != service:
            continue
        kk = f.intensity(t)
        if f.mode == "blip":                  # harmless traffic bump (no incident)
            rate *= 1.0 + 0.5 * kk
        elif kk > k:
            k, mode = kk, f.mode
    if mode == "error_spike":
        err += 0.35 * k; lat *= 1 + 0.3 * k
    elif mode == "latency_degradation":
        lat *= 1 + 5 * k; err += 0.03 * k
    elif mode == "traffic_burst":
        rate *= 1 + 4 * k; lat *= 1 + 0.8 * k; err += 0.02 * k
    elif mode == "service_down":
        rate *= 1 - 0.85 * k; err += 0.9 * k; lat *= 1 + 5 * k
    return {"rate": max(rate, 0.0), "err": min(err, 0.98), "lat": lat, "k": k, "mode": mode}


def random_schedule(rng, t0, t1):
    """Random incidents (one every ~30-90 min) plus harmless blips, for training data."""
    faults, t = [], t0 + 3600 * rng.uniform(0.3, 1.0)
    while t < t1 - 900:
        dur = rng.uniform(420, 720)
        if rng.random() < 0.2:
            faults += make_faults("cascading_failure", None, t, dur)
        else:
            faults += make_faults(str(rng.choice(MODES)), str(rng.choice(SERVICES)), t, dur)
        t += dur + rng.uniform(1800, 5400)
    for _ in range(int((t1 - t0) / 3600 * 2)):
        s = rng.uniform(t0, t1 - 300)
        faults.append(Fault(str(rng.choice(SERVICES)), "blip", s, s + rng.uniform(60, 180), 10))
    return faults


def sample_minute(rng, st):
    """Aggregate one minute of traffic: (req_count, error_rate, avg_latency, p95_latency)."""
    n = int(rng.poisson(st["rate"] * 60))
    if n == 0:
        return None
    errors = rng.binomial(n, st["err"])
    lats = st["lat"] * np.exp(LATENCY_SIGMA * rng.standard_normal(n))
    return n, errors / n, float(lats.mean()), float(np.percentile(lats, 95))


def generate_metrics(days=7, seed=42, start="2026-01-01"):
    """Offline training data: one row per service per minute, with ground-truth labels."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start).timestamp()
    minutes = days * 1440
    faults = random_schedule(rng, t0, t0 + minutes * 60)
    rows = []
    for m in range(minutes):
        t = t0 + m * 60
        for s in SERVICES:
            st = service_state(s, t + 30, faults)
            agg = sample_minute(rng, st)
            if agg is None:
                continue
            rows.append((t, s, agg[0], agg[1], agg[2], agg[3], st["k"], st["mode"]))
    df = pd.DataFrame(rows, columns=["ts", "service", "req_count", "error_rate", "avg_latency",
                                     "p95_latency", "truth_k", "truth_mode"])
    df["window_start"] = pd.to_datetime(df.pop("ts"), unit="s")
    df["incident_active"] = (df["truth_k"] >= INCIDENT_K).astype(int)
    return df


def normal_history(service, n, now_ts, seed=0):
    """n minutes of fault-free metrics ending just before now_ts (used to warm up the stream)."""
    rng = np.random.default_rng(seed + hash(service) % 1000)
    rows = []
    for i in range(n, 0, -1):
        t = now_ts - i * 60
        agg = sample_minute(rng, service_state(service, t, []))
        rows.append({"window_start": pd.to_datetime(t, unit="s"), "service": service,
                     "req_count": agg[0], "error_rate": agg[1],
                     "avg_latency": agg[2], "p95_latency": agg[3]})
    return pd.DataFrame(rows)