"""Training records on Aiven: attempts and their scores in PostgreSQL, live telemetry in Kafka.

PostgreSQL (AIVEN_DATABASE_URL, CA certificate in aiven-ca.pem):
  sim_attempts    one row per attempt: site, score, duration, summary (JSONB)
  sim_tasks       one row per cut or stitch of an attempt, with its score breakdown
  sim_recordings  the full two-hand recording of every complete attempt
  sim_ideal_runs  view: the best complete attempt per site (what the autopilot replays)
  sim_task_best   view: the best result per site and task (what the coach compares with)

Every attempt is first written to training-data/ and then synchronised in the background,
so nothing is lost when the network or the database is unavailable; the best runs of other
trainees are downloaded from Aiven into the same folder.

Kafka (AIVEN_KAFKA_BOOTSTRAP plus the service's ca.pem / service.cert / service.key in
kafka/): every frame's instrument state at 10 Hz on `surgery-telemetry` and every step
event on `surgery-events`, so an instructor station or a robot can follow the procedure
live (see telemetry_monitor.py).
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "training-data"
TELEMETRY_TOPIC = "surgery-telemetry"
EVENTS_TOPIC = "surgery-events"
LEGACY = {"skin": "skin-suture", "heart": "heart-suture", "bypass": "heart-bypass"}   # see procedures.py

SCHEMA = """
CREATE TABLE IF NOT EXISTS sim_attempts (
    id uuid PRIMARY KEY,
    site text NOT NULL,
    mode text NOT NULL,
    started_at timestamptz NOT NULL,
    complete boolean NOT NULL,
    score real NOT NULL,
    duration_s real NOT NULL,
    summary jsonb NOT NULL
);
CREATE TABLE IF NOT EXISTS sim_tasks (
    attempt_id uuid REFERENCES sim_attempts(id) ON DELETE CASCADE,
    task text NOT NULL,
    seconds real, travel_cm real, rejected integer, needle_error_mm real, score real,
    PRIMARY KEY (attempt_id, task)
);
CREATE TABLE IF NOT EXISTS sim_recordings (
    attempt_id uuid PRIMARY KEY REFERENCES sim_attempts(id) ON DELETE CASCADE,
    frames bytea NOT NULL
);
CREATE OR REPLACE VIEW sim_ideal_runs AS
    SELECT DISTINCT ON (a.site) a.site, a.id, a.score, a.duration_s, a.started_at
    FROM sim_attempts a JOIN sim_recordings r ON r.attempt_id = a.id
    WHERE a.complete ORDER BY a.site, a.score DESC, a.duration_s;
CREATE OR REPLACE VIEW sim_task_best AS
    SELECT a.site, t.task, max(t.score) AS score, min(t.seconds) AS seconds,
           min(t.travel_cm) AS travel_cm, count(*) AS attempts
    FROM sim_tasks t JOIN sim_attempts a ON a.id = t.attempt_id GROUP BY a.site, t.task;
"""


def _env(name):
    """A setting from the environment or the project's .env file."""
    if os.environ.get(name, "").strip():
        return os.environ[name].strip()
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip("\"'")
    return None


class TrainingStore:
    """Local-first attempt storage, synchronised with Aiven PostgreSQL."""

    def __init__(self, root=DATA, uri=None, ca_file=ROOT / "aiven-ca.pem"):
        self.root = Path(root)
        self.attempts, self.runs, self.pending = (self.root / name for name in ("attempts", "runs", "pending"))
        for folder in (self.attempts, self.runs, self.pending):
            folder.mkdir(parents=True, exist_ok=True)
        self.uri = uri if uri is not None else _env("AIVEN_DATABASE_URL")
        self.ca_file = Path(ca_file)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.status = "Aiven: not configured (saved locally)" if not self.uri else "Aiven: connecting"
        self.task_best = {}       # (site, task) -> best result, from Aiven
        if self.uri:
            self.worker.submit(self._sync)

    def close(self):
        self.worker.shutdown(wait=True, cancel_futures=False)

    # --- saving ------------------------------------------------------------------------
    def save(self, attempt):
        """Write the attempt locally, then upload it in the background."""
        summary = attempt.summary
        (self.attempts / f"{summary['id']}.json").write_text(json.dumps(summary), encoding="utf-8")
        if summary["complete"]:
            (self.runs / f"{summary['id']}.npz").write_bytes(attempt.blob)
        (self.pending / summary["id"]).touch()
        if self.uri:
            self.worker.submit(self._sync)

    def _connect(self):
        import psycopg
        options = {"connect_timeout": 6, "options": "-c statement_timeout=15000"}
        if self.ca_file.is_file():
            options.update(sslmode="verify-ca", sslrootcert=str(self.ca_file))
        return psycopg.connect(self.uri, **options)

    def _sync(self):
        try:
            import psycopg
        except ImportError:
            self.status = "Aiven: install requirements-ai.txt (psycopg)"
            return
        try:
            with self._connect() as db:
                db.execute(SCHEMA)
                for marker in sorted(self.pending.iterdir()):
                    self._upload(db, marker.name)
                    db.commit()
                    marker.unlink(missing_ok=True)
                self._download_ideal(db)
                self.task_best = {(site, task): {"score": score, "seconds": seconds, "travel_cm": travel}
                                  for site, task, score, seconds, travel in db.execute(
                                      "SELECT site, task, score, seconds, travel_cm FROM sim_task_best")}
                count = db.execute("SELECT count(*) FROM sim_attempts").fetchone()[0]
            self.status = f"Aiven PostgreSQL: {count} attempts synced"
        except (psycopg.Error, OSError) as error:
            self.status = f"Aiven unavailable, saved locally ({type(error).__name__})"

    def _upload(self, db, attempt_id):
        from psycopg.types.json import Jsonb
        path = self.attempts / f"{attempt_id}.json"
        if not path.is_file():
            return
        s = json.loads(path.read_text(encoding="utf-8"))
        db.execute("INSERT INTO sim_attempts VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                   "ON CONFLICT (id) DO NOTHING",
                   (s["id"], s["site"], s["mode"], s["started_at"], s["complete"], s["score"],
                    s["duration_s"], Jsonb(s)))
        for t in s.get("tasks", []):
            db.execute("INSERT INTO sim_tasks VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                       (s["id"], t["task"], t["seconds"], t["travel_cm"], t["rejected"],
                        t["needle_error_mm"], t["score"]))
        run = self.runs / f"{attempt_id}.npz"
        if run.is_file():
            db.execute("INSERT INTO sim_recordings VALUES (%s, %s) ON CONFLICT DO NOTHING",
                       (s["id"], run.read_bytes()))

    def _download_ideal(self, db):
        """Fetch each site's best run (possibly another trainee's) if it is not here yet."""
        rows = db.execute("SELECT i.id, a.summary, r.frames FROM sim_ideal_runs i "
                          "JOIN sim_attempts a ON a.id = i.id JOIN sim_recordings r ON r.attempt_id = i.id "
                          "WHERE NOT (i.id = ANY(%s))",
                          ([p.stem for p in self.runs.glob("*.npz")],)).fetchall()
        for attempt_id, summary, frames in rows:
            (self.attempts / f"{attempt_id}.json").write_text(json.dumps(summary), encoding="utf-8")
            (self.runs / f"{attempt_id}.npz").write_bytes(bytes(frames))

    def refresh(self):
        if self.uri:
            self.worker.submit(self._sync)

    # --- reading -----------------------------------------------------------------------
    def summaries(self, site=None):
        out = []
        for path in self.attempts.glob("*.json"):
            try:
                s = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            s["site"] = LEGACY.get(s.get("site"), s.get("site"))   # pre-procedure recordings
            if site is None or s["site"] == site:
                out.append(s)
        return sorted(out, key=lambda s: s["started_at"], reverse=True)

    def best(self, site, scene=None):
        """(summary, recording) of the best complete attempt at `site`, or None. With `scene`,
        only attempts recorded in that scene (older ones are kept but not replayed)."""
        runs = [s for s in self.summaries(site)
                if s["complete"] and (self.runs / f"{s['id']}.npz").is_file()
                and (scene is None or s.get("scene") == scene)]
        if not runs:
            return None
        top = max(runs, key=lambda s: (s["score"], -s["duration_s"]))
        return top, (self.runs / f"{top['id']}.npz").read_bytes()

    def benchmark(self, site, task):
        """Best known result for a task: Aiven's (all trainees) or the local one."""
        local = [t for s in self.summaries(site) for t in s.get("tasks", []) if t["task"] == task]
        best = self.task_best.get((site, task))
        if local:
            mine = max(local, key=lambda t: t["score"])
            if best is None or mine["score"] > best["score"]:
                best = mine
        return best


class Telemetry:
    """Live instrument telemetry and step events to Aiven Kafka (a no-op when unset)."""

    RATE = 10.0

    def __init__(self):
        self.producer = None
        self.status = "Kafka: not configured"
        self.next_frame = 0.0
        bootstrap = _env("AIVEN_KAFKA_BOOTSTRAP")
        if not bootstrap:
            return
        folder = Path(_env("AIVEN_KAFKA_CERTS") or ROOT / "kafka")
        try:
            from confluent_kafka import Producer
        except ImportError:
            self.status = "Kafka: install requirements-ai.txt (confluent-kafka)"
            return
        self.producer = Producer({
            "bootstrap.servers": bootstrap, "security.protocol": "SSL",
            "ssl.ca.location": str(folder / "ca.pem"),
            "ssl.certificate.location": str(folder / "service.cert"),
            "ssl.key.location": str(folder / "service.key"),
            "linger.ms": 50, "message.timeout.ms": 10000,
        })
        self.status = "Kafka: streaming"

    def _send(self, topic, key, value):
        if self.producer is None:
            return
        try:
            self.producer.produce(topic, key=key.encode(), value=json.dumps(value).encode(),
                                  on_delivery=self._delivered)
            self.producer.poll(0)
        except BufferError:
            self.status = "Kafka: send buffer full"

    def _delivered(self, error, _message):
        self.status = f"Kafka: {error}" if error else "Kafka: streaming"

    def frame(self, now, attempt_id, station, autopilot=False):
        if self.producer is None or now < self.next_frame:
            return
        self.next_frame = now + 1/self.RATE
        hands = {}
        for side, hand in station.hands.items():
            if hand.pose is None:
                continue
            tip = station.fingertip(hand) if hand.held is None else station.active_tip(hand)
            hands[side] = {"tool": hand.tool, "tip": [round(float(v), 5) for v in tip],
                           "closed": hand.grip is not None,
                           "depth_mm": round(hand.depth*1000, 2)}
        side, text = station.plan()
        self._send(TELEMETRY_TOPIC, attempt_id or "autopilot", {
            "time": time.time(), "attempt": attempt_id, "site": station.site.name,
            "autopilot": autopilot, "next": {"hand": side, "step": text},
            "stitches": len(station.stitches), "hands": hands})

    def events(self, attempt_id, site, events):
        for kind, detail in events:
            self._send(EVENTS_TOPIC, attempt_id or "autopilot",
                       {"time": time.time(), "attempt": attempt_id, "site": site, "kind": kind, **detail})

    def close(self):
        if self.producer is not None:
            self.producer.flush(3)
