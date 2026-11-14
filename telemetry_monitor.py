"""Follow a training session live from Aiven Kafka (instructor station or robot bridge).

    python telemetry_monitor.py            # print each hand's instrument and the next step

Uses the same settings as the simulator (AIVEN_KAFKA_BOOTSTRAP and the certificates in
kafka/, see aiven_store.py).
"""

import json
from pathlib import Path
import sys
import uuid

from aiven_store import EVENTS_TOPIC, ROOT, TELEMETRY_TOPIC, _env


def main():
    bootstrap = _env("AIVEN_KAFKA_BOOTSTRAP")
    if not bootstrap:
        sys.exit("Set AIVEN_KAFKA_BOOTSTRAP (and put ca.pem, service.cert, service.key in kafka/)")
    from confluent_kafka import Consumer
    folder = Path(_env("AIVEN_KAFKA_CERTS") or ROOT / "kafka")
    consumer = Consumer({
        "bootstrap.servers": bootstrap, "security.protocol": "SSL",
        "ssl.ca.location": str(folder / "ca.pem"),
        "ssl.certificate.location": str(folder / "service.cert"),
        "ssl.key.location": str(folder / "service.key"),
        "group.id": f"monitor-{uuid.uuid4().hex[:8]}", "auto.offset.reset": "latest",
    })
    consumer.subscribe([TELEMETRY_TOPIC, EVENTS_TOPIC])
    print(f"Following {TELEMETRY_TOPIC} and {EVENTS_TOPIC}. Ctrl+C stops.", flush=True)
    try:
        while True:
            message = consumer.poll(1.0)
            if message is None or message.error():
                continue
            value = json.loads(message.value())
            if message.topic() == EVENTS_TOPIC:
                if value["kind"] != "rejected":
                    detail = {k: v for k, v in value.items() if k not in ("time", "attempt", "site", "kind")}
                    print(f"  * {value['site']}: {value['kind']} {detail}", flush=True)
                continue
            hands = "  ".join(f"{side[0].upper()} {h['tool']:<13} {'closed' if h['closed'] else 'open  '}"
                              f" ({h['tip'][0]:+.3f} {h['tip'][1]:+.3f} {h['tip'][2]:.3f})"
                              for side, h in sorted(value["hands"].items()))
            mode = "AUTOPILOT " if value["autopilot"] else ""
            print(f"{mode}{value['site']} {value['stitches']} st | {hands} | next: "
                  f"{value['next']['hand'] or ''} {value['next']['step']}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
