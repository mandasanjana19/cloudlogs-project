"""Command-line failure injection:  python -m src.generator.inject <scenario> [service] [duration_s]
scenarios: error_spike | latency_degradation | traffic_burst | service_down | cascading_failure"""
import sys
import time

from src.common import config as C
from src.common.store import write_json

scenario = sys.argv[1] if len(sys.argv) > 1 else "cascading_failure"
service = sys.argv[2] if len(sys.argv) > 2 else "database"
duration = float(sys.argv[3]) if len(sys.argv) > 3 else 420
write_json({"id": str(time.time()), "scenario": scenario, "service": service, "duration_s": duration},
           C.CONTROL_DIR / "fault_request.json")
print(f"requested {scenario} on {service} for {duration:.0f}s")