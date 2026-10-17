"""Correlates anomalies from different services/signals into incidents.

Rule: anomalous minutes that are at most `gap_minutes` apart belong to the same incident.
An incident is "confirmed" when it spans >= 2 anomalous minutes or >= 2 services.
Root-cause candidate = the service that became anomalous first.
"""
import pandas as pd


def severity(n_services, n_signals, minutes):
    pts = 2 * (n_services - 1) + (n_signals - 1) + (1 if minutes >= 3 else 0) + (1 if minutes >= 6 else 0)
    return "CRITICAL" if pts >= 6 else "HIGH" if pts >= 4 else "MEDIUM" if pts >= 2 else "LOW"


class IncidentTracker:
    def __init__(self, gap_minutes=2):
        self.gap = pd.Timedelta(minutes=gap_minutes)
        self.current = None
        self.finished = []
        self.counter = 0

    def update(self, minute, anomalies):
        """Call once per minute, in time order. anomalies: list of dicts with service, signals, score."""
        minute = pd.Timestamp(minute)
        if anomalies:
            if self.current is None:
                self.counter += 1
                self.current = {"incident_id": f"INC-{self.counter:03d}", "start": minute,
                                "end": minute, "first_seen": {}, "services": set(),
                                "signals": set(), "minutes": set(), "peak_score": 0.0,
                                "peak_risk": 0.0}
            c = self.current
            c["end"] = minute
            c["minutes"].add(minute)
            for a in anomalies:
                c["services"].add(a["service"])
                c["signals"].update(a["signals"])
                c["first_seen"].setdefault(a["service"], (minute, -a["score"]))
                c["peak_score"] = max(c["peak_score"], a["score"])
                c["peak_risk"] = max(c["peak_risk"], a.get("risk", 0.0))
        elif self.current is not None and minute - self.current["end"] > self.gap:
            self.finished.append(self.current)
            self.current = None

    def to_frame(self):
        rows = []
        for inc, status in [(i, "closed") for i in self.finished] + \
                           ([(self.current, "open")] if self.current else []):
            confirmed = len(inc["minutes"]) >= 2 or len(inc["services"]) >= 2
            if not confirmed:
                continue
            duration = max(1, int((inc["end"] - inc["start"]).total_seconds() // 60) + 1)
            root = min(inc["first_seen"].items(), key=lambda kv: kv[1])[0]
            rows.append({"incident_id": inc["incident_id"], "status": status,
                         "start": inc["start"], "end": inc["end"], "duration_min": duration,
                         "severity": severity(len(inc["services"]), len(inc["signals"]), duration),
                         "services": ",".join(sorted(inc["services"])),
                         "n_services": len(inc["services"]),
                         "signals": ",".join(sorted(inc["signals"])),
                         "root_cause": root, "peak_anomaly_score": round(inc["peak_score"], 3),
                         "peak_risk": round(inc["peak_risk"], 3)})
        cols = ["incident_id", "status", "start", "end", "duration_min", "severity", "services",
                "n_services", "signals", "root_cause", "peak_anomaly_score", "peak_risk"]
        return pd.DataFrame(rows, columns=cols)