"""PHASE 4a - Live log generator. Sends JSON log events to Kafka, ~40 events/second.

Run:   python -m src.generator.log_producer            (add --auto for random failures)
Inject a failure from the dashboard sidebar, or:  python -m src.generator.inject cascading_failure
"""
import argparse
import json
import time

import numpy as np
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

from src.common import config as C
from src.common.simulator import MODES, make_faults
from src.common.store import read_json, write_json
from src.generator.events import fmt_ts, generate_second

REQUEST_FILE = C.CONTROL_DIR / "fault_request.json"
STATUS_FILE = C.CONTROL_DIR / "producer_status.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default=C.KAFKA_BOOTSTRAP)
    ap.add_argument("--auto", action="store_true", help="inject a random failure every 6-12 minutes")
    args = ap.parse_args()

    try:
        producer = KafkaProducer(bootstrap_servers=args.bootstrap, linger_ms=50, acks=1,
                                 key_serializer=lambda k: k.encode(),
                                 value_serializer=lambda v: json.dumps(v).encode())
    except NoBrokersAvailable:
        raise SystemExit(f"Cannot reach Kafka at {args.bootstrap}. Is the 'kafka' container running?")

    rng = np.random.default_rng()
    faults, sent, last_status = [], 0, 0.0
    seen_id = read_json(REQUEST_FILE).get("id")     # ignore a failure request left over from a previous run
    next_auto = time.time() + float(rng.uniform(360, 720))
    next_tick = float(int(time.time()) + 1)
    print(f"producing to topic '{C.KAFKA_TOPIC}' at {args.bootstrap}  (Ctrl+C to stop)")
    try:
        while True:
            now = time.time()
            req = read_json(REQUEST_FILE)                      # failure request from the dashboard
            if req and req.get("id") != seen_id:
                seen_id = req["id"]
                faults += make_faults(req["scenario"], req.get("service"), now, float(req["duration_s"]))
                print(f"[{fmt_ts(now)}] injected {req['scenario']} on {req.get('service')} for {req['duration_s']}s")
            if args.auto and now > next_auto:
                scen = str(rng.choice(MODES + ["cascading_failure"]))
                faults += make_faults(scen, str(rng.choice(C.SERVICES)), now, float(rng.uniform(360, 600)))
                print(f"[{fmt_ts(now)}] auto-injected {scen}")
                next_auto = now + float(rng.uniform(600, 900))
            faults = [f for f in faults if f.end > now - 60]

            events = generate_second(next_tick - 1.0, faults, rng)   # events of the second that just ended
            for e in events:
                producer.send(C.KAFKA_TOPIC, key=e["service"], value=e)
            sent += len(events)

            if now - last_status > 2:
                active = sorted({f"{f.service}:{f.mode}" for f in faults if f.mode != "blip" and f.start <= now <= f.end})
                write_json({"running": True, "updated": fmt_ts(now), "events_sent": sent,
                            "events_last_second": len(events), "active_faults": active}, STATUS_FILE)
                last_status = now
            next_tick += 1.0
            time.sleep(max(0.0, next_tick - time.time()))
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        write_json({"running": False, "updated": fmt_ts(time.time()), "events_sent": sent,
                    "active_faults": []}, STATUS_FILE)


if __name__ == "__main__":
    main()