"""PHASE 3b - ML for the LIVE pipeline (per-service, per-minute).

Training data = simulated microservice metrics with ground-truth incidents.
  * XGBoost risk model : P(incident active at any of the next 5 minutes | recent behaviour)
  * Isolation Forest   : unsupervised anomaly score, trained on NORMAL minutes only
  * Logistic Regression: baseline to compare against
Evaluation is TEMPORAL: train on days 1-4, pick thresholds on day 5, test on days 6-7.
"""
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, \
    precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.common import config as C
from src.common.features import IF_FEATURES, METRICS, RISK_FEATURES, build_features, \
    label_next_minutes, peer_features_frame, rule_signals
from src.common.simulator import generate_metrics
from src.common.store import write_json, write_parquet


def prf(y, pred):
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0))}


def best_f1_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.argmax(f1))])


def main():
    t0 = time.time()
    raw = generate_metrics(days=7, seed=42)
    print(f"simulated {len(raw):,} service-minutes ({time.time() - t0:.0f}s)")

    parts = []
    for svc, g in raw.groupby("service"):
        g = g.sort_values("window_start").reset_index(drop=True)
        f = build_features(g[METRICS], svc)
        f["label"] = label_next_minutes(g["incident_active"])
        f["incident_active"], f["truth_k"] = g["incident_active"], g["truth_k"]
        f["window_start"], f["service"] = g["window_start"], svc
        parts.append(f)
    data = pd.concat(parts, ignore_index=True).dropna(subset=["req_count_base"])
    data = peer_features_frame(data)              # cross-service features
    for c in RISK_FEATURES:                       # no NaN left after warm-up rows are dropped
        data[c] = data[c].fillna(0.0)

    t_start = data["window_start"].min()
    day = (data["window_start"] - t_start).dt.total_seconds() // 86400
    train, val, test = data[day < 4], data[day == 4], data[day >= 5]
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,} | "
          f"positive rate {data['label'].mean():.1%}")

    results = {}
    # ---- baseline: logistic regression
    sc = StandardScaler().fit(train[RISK_FEATURES])
    lr = LogisticRegression(max_iter=500, class_weight="balanced").fit(sc.transform(train[RISK_FEATURES]), train["label"])
    lr_val, lr_te = (lr.predict_proba(sc.transform(d[RISK_FEATURES]))[:, 1] for d in (val, test))
    t_lr = best_f1_threshold(val["label"], lr_val)

    # ---- main model: XGBoost
    spw = float((train["label"] == 0).sum() / max(train["label"].sum(), 1))
    xgb = XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                        eval_metric="aucpr", early_stopping_rounds=30, n_jobs=4, random_state=42)
    xgb.fit(train[RISK_FEATURES], train["label"], eval_set=[(val[RISK_FEATURES], val["label"])], verbose=False)
    x_val, x_te = (xgb.predict_proba(d[RISK_FEATURES])[:, 1] for d in (val, test))
    thr = best_f1_threshold(val["label"], x_val)

    yt = test["label"].values
    for name, s, th in [("Logistic Regression", lr_te, t_lr), ("XGBoost", x_te, thr)]:
        results[name] = {**prf(yt, (s >= th).astype(int)), "roc_auc": float(roc_auc_score(yt, s)),
                         "pr_auc": float(average_precision_score(yt, s))}

    # early-warning: minutes that are still healthy NOW but an incident starts within 5 minutes
    ew = (test["incident_active"] == 0) & (test["label"] == 1)
    early = {"n_events": int(ew.sum()),
             "xgb_recall": float((x_te[ew.values] >= thr).mean()) if ew.any() else None,
             "logreg_recall": float((lr_te[ew.values] >= t_lr).mean()) if ew.any() else None}

    # ---- Isolation Forest on NORMAL minutes only (novelty detection), compared with rules
    normal = train[train["truth_k"] < 0.05]
    iso = IsolationForest(n_estimators=150, random_state=42, n_jobs=-1).fit(normal[IF_FEATURES])
    # the model never sees labels; only its alert threshold is tuned on the validation days
    iso_thr = best_f1_threshold(val["incident_active"], -iso.score_samples(val[IF_FEATURES]))
    iso_flag = (-iso.score_samples(test[IF_FEATURES]) > iso_thr)
    rule_flag = np.array([len(rule_signals(r)) > 0 for r in test.to_dict("records")])
    truth = test["incident_active"].values
    detection = {"Statistical rules (robust z-score)": prf(truth, rule_flag.astype(int)),
                 "Isolation Forest": prf(truth, iso_flag.astype(int)),
                 "Rules OR Isolation Forest": prf(truth, (rule_flag | iso_flag).astype(int))}

    # ---- save models + metadata + dashboard tables
    C.MODELS.mkdir(parents=True, exist_ok=True)
    xgb.save_model(str(C.MODELS / "incident_risk_xgb.json"))
    joblib.dump(iso, C.MODELS / "incident_iforest.joblib")
    write_json({"risk_features": RISK_FEATURES, "if_features": IF_FEATURES, "risk_threshold": thr,
                "if_threshold": iso_thr}, C.MODELS / "incident_meta.json")
    gain = xgb.get_booster().get_score(importance_type="gain")
    write_parquet(pd.DataFrame({"feature": list(gain), "gain": list(gain.values())})
                  .sort_values("gain", ascending=False), C.ML_OUT / "incident_feature_importance.parquet")
    out = test[["window_start", "service", "label", "incident_active", "truth_k"]].copy()
    out["risk"], out["anomaly_score"] = x_te, -iso.score_samples(test[IF_FEATURES])
    write_parquet(out.reset_index(drop=True), C.ML_OUT / "incident_test_scores.parquet")
    write_json({"models": results, "early_warning": early, "detection": detection,
                "risk_threshold": thr, "if_threshold": iso_thr, "horizon_minutes": 5,
                "rows": {"train": len(train), "val": len(val), "test": len(test)},
                "positive_rate": float(data["label"].mean())}, C.ML_OUT / "incident_metrics.json")

    print("\nRISK MODEL (test days 6-7)")
    print(pd.DataFrame(results).T.round(3).to_string())
    print("early warning:", early)
    print("\nANOMALY DETECTION vs active incidents")
    print(pd.DataFrame(detection).T.round(3).to_string())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()