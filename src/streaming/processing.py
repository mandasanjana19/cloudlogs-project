"""The ML + incident logic that runs on every finished 1-minute window.

Spark does the heavy lifting (windows, aggregation, p95). Each finished window is tiny
(5 services x 1 row), so scoring it with XGBoost / Isolation Forest in pandas is cheap.
"""
import time

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.common import config as C
from src.common.features import METRICS, build_features, peer_features_row, rule_signals
from src.common.incidents import IncidentTracker
from src.common.simulator import normal_history
from src.common.store import RollingTable, read_json


class MinuteProcessor:
    def __init__(self, out_dir=None):
        out = out_dir or C.STREAM_OUT
        meta = read_json(C.MODELS / "incident_meta.json")
        if not meta:
            raise SystemExit("models/incident_meta.json not found - run Phase 3 (train_incident_model) first")
        self.meta = meta
        self.xgb = XGBClassifier()
        self.xgb.load_model(str(C.MODELS / "incident_risk_xgb.json"))
        self.iso = joblib.load(C.MODELS / "incident_iforest.joblib")
        # warm start: pretend we already saw 20 normal minutes, so detection works immediately
        now = time.time()
        self.history = {s: normal_history(s, 20, now) for s in C.SERVICES}
        self.tracker = IncidentTracker(gap_minutes=2)
        self.first_minute = None
        self.minutes = RollingTable(out / "minute_scores.parquet", 6000)
        self.anomalies = RollingTable(out / "anomalies.parquet", 6000)
        self.incidents = RollingTable(out / "incidents.parquet", 1000)

    def process(self, pdf):
        """pdf: finished 1-minute windows. Columns: window_start, service, req_count,
        error_count, avg_latency, p95_latency, truth_k"""
        if self.first_minute is None and len(pdf):
            self.first_minute = pdf["window_start"].min()      # first window of a run is usually partial
        pdf = pdf[pdf["window_start"] > self.first_minute]
        all_rows, all_anoms = [], []
        for minute, grp in pdf.sort_values(["window_start", "service"]).groupby("window_start"):
            feats = {}
            for r in grp.itertuples():
                row = {"window_start": r.window_start, "service": r.service, "req_count": float(r.req_count),
                       "error_rate": r.error_count / max(r.req_count, 1), "avg_latency": float(r.avg_latency),
                       "p95_latency": float(r.p95_latency)}
                h = pd.concat([self.history[r.service], pd.DataFrame([row])], ignore_index=True).tail(20)
                self.history[r.service] = h
                f = build_features(h[METRICS], r.service).iloc[-1].to_dict()
                f["truth_k"] = float(r.truth_k)
                feats[r.service] = f

            rows = []
            for svc, f in feats.items():
                f.update(peer_features_row(svc, feats))
                rows.append((svc, f))
            X = pd.DataFrame([f for _, f in rows])[self.meta["risk_features"]].fillna(0.0)
            risk = self.xgb.predict_proba(X)[:, 1]
            iso_score = -self.iso.score_samples(pd.DataFrame([f for _, f in rows])[self.meta["if_features"]].fillna(0.0))

            minute_anoms = []
            for (svc, f), rk, sc in zip(rows, risk, iso_score):
                signals = rule_signals(f)
                ml_flag = bool(sc > self.meta["if_threshold"])
                if ml_flag:
                    signals.append("ml_outlier")
                out = {"window_start": minute, "service": svc, "req_count": f["req_count"],
                       "error_rate": f["error_rate"], "avg_latency": f["avg_latency"],
                       "p95_latency": f["p95_latency"], "error_rate_z": f["error_rate_z"],
                       "p95_latency_z": f["p95_latency_z"], "req_count_z": f["req_count_z"],
                       "risk": float(rk), "anomaly_score": float(sc), "is_anomaly": bool(signals),
                       "signals": ",".join(signals), "truth_k": f["truth_k"]}
                all_rows.append(out)
                if signals:
                    a = {"service": svc, "signals": signals, "score": float(sc), "risk": float(rk)}
                    minute_anoms.append(a)
                    all_anoms.append(out)
            self.tracker.update(minute, minute_anoms)

        self.minutes.append(pd.DataFrame(all_rows))
        self.anomalies.append(pd.DataFrame(all_anoms))
        self.incidents.replace(self.tracker.to_frame())
        return len(all_rows), len(all_anoms)