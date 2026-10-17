"""Quick look at what the streaming job has produced:   python -m src.utils.peek"""
import pandas as pd

from src.common import config as C
from src.common.store import read_json

pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_colwidth", 60)


def load(name):
    p = C.STREAM_OUT / name
    return pd.read_parquet(p) if p.exists() else None


def main():
    print("PRODUCER STATUS:", read_json(C.CONTROL_DIR / "producer_status.json") or "no status file yet")

    live = load("live_10s.parquet")
    print("\n== live_10s.parquet (10-second windows) ==")
    if live is None:
        print("not created yet (wait ~40 s after both jobs are running)")
    else:
        print(f"{len(live):,} rows, newest window {live['window_start'].max()}")
        print(live.sort_values("window_start").groupby("service").tail(1)
              [["window_start", "service", "req_count", "error_rate", "avg_latency", "p95_latency"]].round(3).to_string(index=False))

    mins = load("minute_scores.parquet")
    print("\n== minute_scores.parquet (1-minute windows scored by the ML models) ==")
    if mins is None:
        print("not created yet (wait ~2 minutes)")
    else:
        print(f"{len(mins):,} rows, newest minute {mins['window_start'].max()}")
        print(mins.sort_values("window_start").groupby("service").tail(1)
              [["window_start", "service", "error_rate", "p95_latency", "risk", "anomaly_score", "signals"]].round(3).to_string(index=False))

    inc = load("incidents.parquet")
    print("\n== incidents.parquet ==")
    if inc is None or inc.empty:
        print("no confirmed incidents yet")
    else:
        print(inc[["incident_id", "status", "severity", "start", "duration_min", "services", "root_cause", "signals"]].to_string(index=False))

    raw = load("raw_events.parquet")
    if raw is not None:
        print(f"\n== raw_events.parquet: {len(raw):,} stored lines ==")
        print(raw.groupby(["service", "level"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()