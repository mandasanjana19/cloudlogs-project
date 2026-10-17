"""Turns the simulator's statistical state into individual JSON log events."""
from datetime import datetime, timezone

import numpy as np

from src.common.config import SERVICES
from src.common.simulator import LATENCY_SIGMA, PROFILES, service_state

ENDPOINTS = {"gateway": ["/api/route", "/api/health"], "auth": ["/login", "/token/refresh", "/logout"],
             "orders": ["/orders", "/orders/{id}", "/cart"], "payments": ["/charge", "/refund", "/invoice"],
             "database": ["SELECT orders", "INSERT payment", "UPDATE inventory", "SELECT users"]}
ERROR_MSG = {"gateway": ["upstream timed out", "no healthy upstream"],
             "auth": ["token store unreachable", "session lookup failed"],
             "orders": ["order service error", "inventory check failed"],
             "payments": ["payment provider timeout", "charge declined: gateway error"],
             "database": ["connection pool exhausted", "query timeout", "deadlock detected"]}


def fmt_ts(epoch):
    d = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}"


def generate_second(t, faults, rng):
    """All events for the one-second interval [t, t+1) across every service."""
    events = []
    for svc in SERVICES:
        st = service_state(svc, t + 0.5, faults)
        for _ in range(int(rng.poisson(st["rate"]))):
            latency = st["lat"] * float(np.exp(LATENCY_SIGMA * rng.standard_normal()))
            if rng.random() < st["err"]:
                status, level, msg = int(rng.choice([500, 502, 503, 504])), "ERROR", str(rng.choice(ERROR_MSG[svc]))
            elif rng.random() < 0.015:
                status, level, msg = int(rng.choice([400, 401, 404, 429])), "WARN", "client error"
            else:
                slow = latency > 3 * PROFILES[svc]["lat"]
                status, level = int(rng.choice([200, 200, 200, 201])), "WARN" if slow else "INFO"
                msg = "slow response" if slow else "request completed"
            events.append({"ts": fmt_ts(t + float(rng.random())), "service": svc, "level": level,
                           "status_code": status, "latency_ms": round(latency, 2),
                           "host": f"{svc}-{int(rng.integers(1, 4))}",
                           "endpoint": str(rng.choice(ENDPOINTS[svc])), "message": msg,
                           "truth_k": round(st["k"], 3)})
    return events