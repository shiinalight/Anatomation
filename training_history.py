"""Record simulated practice and sync finished attempts to Aiven PostgreSQL."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from math import dist
import os
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb


ROOT = Path(__file__).resolve().parent
SAMPLE_SECONDS = .1
SCHEMA = """
CREATE TABLE IF NOT EXISTS surgisense_attempts (
    id uuid PRIMARY KEY,
    started_at timestamptz NOT NULL,
    input_mode text NOT NULL,
    report jsonb NOT NULL
)
"""


class TrainingAttempt:
    def __init__(self, now, mode, site):
        self.id = str(uuid4())
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.mode = mode
        self.site = site
        self.started = now
        self.next_sample = now
        self.events = []
        self.samples = []
        self.path = 0.0
        self.previous = None
        self.tracking = None
        self.tied = self.trimmed = 0

    def event(self, now, name, detail):
        self.events.append({"seconds": round(now - self.started, 2), "event": name, "detail": detail})

    def observe(self, now, station, tool, status):
        if status != self.tracking:
            self.event(now, "tracking", status)
            self.tracking = status
        for name, detail in station.activity:
            self.event(now, name, detail)
        self.tied = len(station.stitches)
        self.trimmed = sum(trimmed for _, _, trimmed in station.stitches)
        if tool is None:
            self.previous = None
            return
        if now < self.next_sample:
            return
        self.next_sample = now + SAMPLE_SECONDS
        tip = tuple(round(float(value), 5) for value in station.active_tip())
        assisted = station.guide_tip is not None
        self.samples.append({"seconds": round(now - self.started, 2), "tool": tool, "tip_m": tip,
                             "aligned": assisted, "depth_mm": round(station.depth * 1000, 2)})
        if self.previous and self.previous[0] == tool and now - self.previous[2] <= .25:
            self.path += dist(tip, self.previous[1])
        self.previous = (tool, tip, now)

    def report(self, now):
        if not any(item["event"] != "tracking" for item in self.events) and not self.samples:
            return None
        return {
            "id": self.id, "started_at": self.started_at, "input_mode": self.mode, "site": self.site,
            "duration_seconds": round(now - self.started, 2),
            "sample_hz": round(1 / SAMPLE_SECONDS),
            "sample_count": len(self.samples),
            "stitches_tied": self.tied, "stitches_trimmed": self.trimmed,
            "rejected_actions": sum(item["event"] == "rejected" for item in self.events),
            "tip_path_cm": round(self.path * 100, 1),
            "events": self.events, "samples": self.samples,
        }


def configured_uri():
    if os.environ.get("AIVEN_DATABASE_URL"):
        return os.environ["AIVEN_DATABASE_URL"]
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "AIVEN_DATABASE_URL" and value.strip():
                return value.strip().strip('"\'')
    return None


class TrainingHistory:
    def __init__(self, uri, directory=None, ca_file=None):
        self.uri = uri
        self.directory = directory or ROOT / "training-data"
        self.pending = self.directory / "pending"
        self.ca_file = ca_file or ROOT / "aiven-ca.pem"
        self.worker = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self.worker.shutdown()

    def save(self, report):
        self.pending.mkdir(parents=True, exist_ok=True)
        target = self.pending / (report["id"] + ".json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report), encoding="utf-8")
        temporary.replace(target)
        return self.refresh(report["input_mode"])

    def refresh(self, mode):
        return self.worker.submit(self._sync_and_read, mode)

    def _sync_and_read(self, mode):
        files = sorted(self.pending.glob("*.json"))
        queued = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in files]
        if self.uri and self.ca_file.is_file():
            try:
                return self._from_aiven(queued, mode), "Aiven PostgreSQL"
            except psycopg.Error:
                status = "Aiven unavailable • attempts saved locally"
        else:
            status = "Local history • Aiven connection and CA certificate required"
        local = sorted((r for _, r in queued if r["input_mode"] == mode),
                       key=lambda report: report["started_at"], reverse=True)
        return local[:10], status

    def _from_aiven(self, queued, mode):
        # Use the project's CA. libpq hostname verification on Windows rejects
        # the IPv6 SAN in this Aiven certificate.
        with psycopg.connect(self.uri, sslmode="verify-ca", sslrootcert=str(self.ca_file),
                             connect_timeout=5, options="-c statement_timeout=5000") as connection:
            connection.execute(SCHEMA)
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO surgisense_attempts VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    [(r["id"], r["started_at"], r["input_mode"], Jsonb(r)) for _, r in queued],
                )
            rows = connection.execute(
                "SELECT (report - 'samples') || "
                "jsonb_build_object('sample_count', jsonb_array_length(report->'samples')) "
                "FROM surgisense_attempts WHERE input_mode = %s "
                "ORDER BY started_at DESC, id DESC LIMIT 10", (mode,),
            ).fetchall()
        for path, _ in queued:
            path.unlink(missing_ok=True)
        return [row[0] for row in rows]


def describe_attempt(report):
    return (f'{report["duration_seconds"]:.0f}s · {report["stitches_tied"]} tied · '
            f'{report["stitches_trimmed"]} trimmed · {report["rejected_actions"]} rejected · '
            f'{report["tip_path_cm"]:.1f} cm sampled tip travel')
