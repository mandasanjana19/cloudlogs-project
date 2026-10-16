"""PHASE 3a - Supervised + unsupervised ML on the labeled HDFS_v1 blocks.

Task: given the event-count vector of one HDFS block, is the block Normal or Anomalous?
Models compared:  Logistic Regression (baseline) | Isolation Forest (unsupervised) | XGBoost (main)
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.common import config as C
from src.common.store import read_parquet, write_json, write_parquet


def metrics_at(y, score, thr):
    pred = (score >= thr).astype(int)
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, score)),
            "pr_auc": float(average_precision_score(y, score))}


def best_f1_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-9, None)
    return float(t[int(np.argmax(f1))])


def main():
    t0 = time.time()
    df = read_parquet(C.BATCH_OUT / "block_features.parquet")
    if df.empty:
        raise SystemExit("block_features.parquet not found - run the batch job first")
    feature_cols = [c for c in df.columns if c not in ("block_id", "label", "start_ts")]
    X, y = df[feature_cols].astype("float32"), df["label"].astype(int)
    print(f"{len(df):,} blocks, {y.sum():,} anomalous ({y.mean():.2%}), {len(feature_cols)} features")

    # 60% train / 20% validation (choose the threshold) / 20% test (final report), stratified
    idx_tr, idx_te = train_test_split(df.index, test_size=0.2, stratify=y, random_state=42)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.25, stratify=y[idx_tr], random_state=42)
    Xtr, Xva, Xte = X.loc[idx_tr], X.loc[idx_va], X.loc[idx_te]
    ytr, yva, yte = y.loc[idx_tr], y.loc[idx_va], y.loc[idx_te]

    results = {}

    # ---- 1. Logistic regression baseline
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=300, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    s_va, s_te = lr.predict_proba(scaler.transform(Xva))[:, 1], lr.predict_proba(scaler.transform(Xte))[:, 1]
    results["Logistic Regression"] = metrics_at(yte, s_te, best_f1_threshold(yva, s_va))

    # ---- 2. Isolation Forest: unsupervised, never sees the labels
    iso = IsolationForest(n_estimators=100, contamination=float(ytr.mean()), random_state=42, n_jobs=-1)
    iso.fit(Xtr)
    a_va, a_te = -iso.score_samples(Xva), -iso.score_samples(Xte)
    results["Isolation Forest (unsupervised)"] = metrics_at(yte, a_te, best_f1_threshold(yva, a_va))

    # ---- 3. XGBoost: the main model
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))      # handle class imbalance
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                        eval_metric="aucpr", early_stopping_rounds=30, n_jobs=4, random_state=42)
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    x_va, x_te = xgb.predict_proba(Xva)[:, 1], xgb.predict_proba(Xte)[:, 1]
    thr = best_f1_threshold(yva, x_va)
    results["XGBoost"] = metrics_at(yte, x_te, thr)
    tn, fp, fn, tp = confusion_matrix(yte, (x_te >= thr).astype(int)).ravel()

    # ---- 4. Robustness check: train on the EARLIER 70% of blocks, test on the LATER 30%
    order = df["start_ts"].sort_values().index
    cut = int(len(order) * 0.7)
    tr_i, te_i = order[:cut], order[cut:]
    temporal = None
    if y.loc[te_i].sum() >= 20 and y.loc[tr_i].sum() >= 20:
        m2 = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.8,
                           colsample_bytree=0.8, scale_pos_weight=float((y.loc[tr_i] == 0).sum() / y.loc[tr_i].sum()),
                           tree_method="hist", n_jobs=4, random_state=42).fit(X.loc[tr_i], y.loc[tr_i])
        temporal = metrics_at(y.loc[te_i], m2.predict_proba(X.loc[te_i])[:, 1], thr)

    # ---- save everything the dashboard needs
    C.MODELS.mkdir(parents=True, exist_ok=True)
    xgb.save_model(str(C.MODELS / "hdfs_block_xgb.json"))
    write_json({"features": feature_cols, "threshold": thr}, C.MODELS / "hdfs_block_meta.json")

    gain = xgb.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": list(gain), "gain": list(gain.values())}).sort_values("gain", ascending=False)
    write_parquet(imp, C.ML_OUT / "hdfs_feature_importance.parquet")

    test = pd.DataFrame({"block_id": df.loc[idx_te, "block_id"].values, "y_true": yte.values,
                         "xgb_score": x_te, "iforest_score": a_te})
    write_parquet(test, C.ML_OUT / "hdfs_test_scores.parquet")

    # scores for the blocks in the dashboard's block explorer (marks which were held-out)
    sample = read_parquet(C.BATCH_OUT / "block_logs_sample.parquet")
    if not sample.empty:
        ids = sample["block_id"].unique()
        sub = df[df["block_id"].isin(ids)]
        expl = pd.DataFrame({"block_id": sub["block_id"].values, "label": sub["label"].values,
                             "xgb_score": xgb.predict_proba(sub[feature_cols].astype("float32"))[:, 1],
                             "n_events": sub["n_events"].values, "duration_s": sub["duration_s"].values,
                             "split": np.where(sub.index.isin(idx_te), "test (unseen)", "train")})
        write_parquet(expl, C.ML_OUT / "hdfs_block_explorer.parquet")

    write_json({"n_blocks": len(df), "n_features": len(feature_cols), "threshold": thr,
                "models": results, "temporal_check": temporal,
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                "best_iteration": int(xgb.best_iteration)}, C.ML_OUT / "hdfs_metrics.json")
    print("\nTEST-SET RESULTS")
    print(pd.DataFrame(results).T.round(4).to_string())
    print("confusion matrix:", dict(tn=tn, fp=fp, fn=fn, tp=tp))
    if temporal:
        print("temporal check:", {k: round(v, 4) for k, v in temporal.items()})
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()