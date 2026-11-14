"""Record two-handed attempts, score them, and replay the best one (autopilot).

A recording keeps everything the simulator received each frame (both gloves' joints,
the keys pressed and the held-key moves), so replaying it into a fresh CuttingStation
reproduces the attempt exactly: the same cuts, needle passes and knots. It also keeps what
each hand's instrument did (tool, tip position, direction, closed or open), which is the
trajectory a robot arm would follow; `robot_trajectory` exports it.

Scoring splits an attempt into its tasks (each cut and each stitch) using the station's
event log, and scores each task on time, instrument travel, refused actions and needle
accuracy. The best complete attempt for a site becomes its ideal run.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
from uuid import uuid4

import numpy as np

SIDES = ("left", "right")
FORMAT = 1
IDLE_KEEP = .4              # autopilot: idle stretches longer than this are shortened to it
STILL = .0005               # m per frame: instrument tips moving less than this are still


@dataclass
class Attempt:
    """A finished recording: `summary` is small (JSON), `blob` holds the frames."""
    summary: dict
    blob: bytes = field(repr=False)


class Recorder:
    """Collects the frames of one attempt at a site (from a fresh station to the next reset)."""

    def __init__(self, site, mode, joints, anchors, now):
        self.id = str(uuid4())
        self.site, self.mode = site, mode
        self.joints = list(joints)
        self.anchors = {side: np.asarray(anchors[side], float) for side in SIDES}
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.start = now
        self.times, self.poses, self.present = [], [], []
        self.keys, self.held = [], []
        self.tools, self.tips, self.dirs, self.closed = [], [], [], []
        self.depth = []
        self.events = []
        self.steps = []          # (frame, plan side, plan text) whenever the prompt changes
        self._step = None

    def __len__(self):
        return len(self.times)

    def frame(self, now, poses, keys, held, station, events):
        """Record one simulator frame: inputs (`poses` before offsets, keys, held moves)
        and what the station did with them (`events` from station.log, instrument poses)."""
        index = len(self.times)
        self.times.append(now - self.start)
        pose_row = np.zeros((2, len(self.joints), 3), np.float32)
        present = np.zeros(2, bool)
        for i, side in enumerate(SIDES):
            pose = poses.get(side)
            if pose is None:
                continue
            present[i] = True
            rel = np.array([pose[name] for name in self.joints]) - self.anchors[side]
            pose_row[i] = rel     # relative to the hand's anchor: full float32 precision
        self.poses.append(pose_row)
        self.present.append(present)
        self.keys.append([int(k) for k in keys])
        self.held.append([(int(code), float(amount)) for code, amount in held])
        tools, tips, dirs, closed = [], [], [], []
        for side in SIDES:
            hand = station.hands[side]
            if hand.held is None or hand.tool_pose is None:
                tools.append(-1)
                tips.append(np.zeros(3) if hand.pose is None else station.fingertip(hand))
                dirs.append(np.zeros(3))
            else:
                tools.append(hand.held)
                tips.append(station.active_tip(hand))
                dirs.append(hand.tool_pose[1])
            closed.append(hand.grip is not None or
                          (hand.held == 2 and station.needle_grasp is not None))
        self.tools.append(tools)
        self.tips.append(tips)
        self.dirs.append(dirs)
        self.closed.append(closed)
        self.depth.append(station.depth)
        for kind, detail in events:
            self.events.append({"frame": index, "t": round(self.times[-1], 3), "kind": kind, **detail})
        step = station.plan()
        if step != self._step:
            self._step = step
            self.steps.append((index, step[0], step[1]))

    def finish(self, now, complete, station_summary):
        """The Attempt, or None if nothing happened."""
        if not any(e["kind"] not in ("rejected",) for e in self.events):
            return None
        summary = {
            "id": self.id, "format": FORMAT, "site": self.site, "mode": self.mode,
            "started_at": self.started_at, "duration_s": round(now - self.start, 2),
            "frames": len(self.times), "complete": bool(complete), **station_summary,
        }
        summary["score"], summary["tasks"] = score(self.events, np.array(self.times),
                                                   np.array(self.tips, float), self.site)
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            times=np.array(self.times, np.float32), poses=np.array(self.poses, np.float32),
            present=np.array(self.present, bool), tools=np.array(self.tools, np.int8),
            tips=np.array(self.tips, np.float32), dirs=np.array(self.dirs, np.float32),
            closed=np.array(self.closed, bool), depth=np.array(self.depth, np.float32),
            meta=np.frombuffer(json.dumps({
                "joints": self.joints,
                "anchors": {side: self.anchors[side].tolist() for side in SIDES},
                "keys": self.keys, "held": self.held, "events": self.events,
                "steps": self.steps, "summary": summary,
            }).encode(), np.uint8),
        )
        return Attempt(summary, buffer.getvalue())


def load(blob):
    """Frames and metadata of a recorded attempt."""
    with np.load(io.BytesIO(blob)) as data:
        run = {name: data[name] for name in data.files}
    run["meta"] = json.loads(run["meta"].tobytes().decode())
    return run


# --- scoring -------------------------------------------------------------------------------

def _travel(tips, a, b):
    """Instrument-tip travel of both hands between frames a and b, in cm (jumps from tool
    pick-ups and set-downs excluded)."""
    if b - a < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(tips[a:b], axis=0), axis=2)
    return float(steps[steps < .02].sum())*100


def score(events, times, tips, site):
    """(overall 0-100, tasks): each cut or stitch scored on time, travel, refused actions
    and needle accuracy. The overall score is the mean task score."""
    times = np.asarray(times)
    targets = {"skin": (45, 40), "heart": (45, 25), "bypass": (40, 20)}   # (s, cm) per stitch
    seconds, travel_cm = targets.get(site, (45, 30))
    tasks, start = [], 0
    pending = {"rejected": 0, "errors": []}
    for event in events:
        kind = event["kind"]
        if kind == "rejected":
            pending["rejected"] += 1
        elif kind == "puncture" and event.get("error_mm") is not None:
            pending["errors"].append(event["error_mm"])
        done = (kind == "stitch" or
                (kind == "procedure_step" and event.get("action") == "cut"))
        if not done:
            continue
        end = event["frame"]
        name = (f"stitch {event['number']}" if kind == "stitch"
                else f"cut {event.get('vessel', '')}".strip())
        duration = float(times[end] - times[start]) if end < len(times) else 0.0
        travel = _travel(tips, start, end + 1)
        error = float(np.mean(pending["errors"])) if pending["errors"] else 0.0
        task_time = seconds if kind == "stitch" else seconds*.6
        points = 100.0
        points -= max(0.0, duration/task_time - 1)*25          # slower than the target time
        points -= max(0.0, travel/travel_cm - 1)*20             # wandering instruments
        points -= pending["rejected"]*4                         # refused actions
        points -= error*6                                       # mm off the needle target
        tasks.append({"task": name, "seconds": round(duration, 1), "travel_cm": round(travel, 1),
                      "rejected": pending["rejected"], "needle_error_mm": round(error, 2),
                      "score": round(float(np.clip(points, 0, 100)), 1),
                      "frames": [int(start), int(end)]})
        start = end + 1
        pending = {"rejected": 0, "errors": []}
    overall = round(float(np.mean([t["score"] for t in tasks])), 1) if tasks else 0.0
    return overall, tasks


# --- autopilot ----------------------------------------------------------------------------

def active_frames(run):
    """Frame indices to replay: every frame with input, instrument motion, cutting or a
    logged event, with still stretches shortened to IDLE_KEEP seconds. Skipped frames change
    nothing in the station, so the replay reproduces the attempt."""
    times = run["times"]
    keys, held = run["meta"]["keys"], run["meta"]["held"]
    marked = {event["frame"] for event in run["meta"]["events"]}
    changed = np.zeros(len(times), bool)
    for name in ("present", "tools", "closed"):   # glove pinches change these without keys
        values = run[name].reshape(len(times), -1)
        changed[1:] |= (values[1:] != values[:-1]).any(axis=1)
    motion = np.zeros(len(times))
    motion[1:] = np.linalg.norm(np.diff(run["tips"], axis=0), axis=2).max(axis=1)
    cutting = run["depth"] > 0
    keep, idle_since = [], None
    for i in range(len(times)):
        busy = (bool(keys[i] or held[i]) or i in marked or changed[i] or i + 1 in marked
                or motion[i] > STILL or cutting[i] or (i and cutting[i - 1]))
        if busy:
            idle_since = None
            keep.append(i)
            continue
        if idle_since is None:
            idle_since = times[i]
        if times[i] - idle_since <= IDLE_KEEP:
            keep.append(i)
    return keep


class Autopilot:
    """Feeds a recorded attempt back to the simulator, frame by frame, as if it were the
    gloves and keyboard."""

    def __init__(self, blob):
        self.run = load(blob)
        meta = self.run["meta"]
        self.summary = meta["summary"]
        self.site = self.summary["site"]
        self.joints = meta["joints"]
        self.anchors = {side: np.array(meta["anchors"][side]) for side in SIDES}
        self.order = active_frames(self.run)
        self.position = 0
        # Idle glove motion while skipping a frame is dropped, but the key and held-key
        # input of skipped frames is empty by construction (see active_frames).

    @property
    def done(self):
        return self.position >= len(self.order)

    @property
    def progress(self):
        return self.position / max(len(self.order), 1)

    def next(self):
        """(poses, keys, held) for the next frame."""
        i = self.order[self.position]
        self.position += 1
        poses = {}
        for k, side in enumerate(SIDES):
            if not self.run["present"][i][k]:
                poses[side] = None
                continue
            rel = self.run["poses"][i][k].astype(float)
            poses[side] = {name: self.anchors[side] + rel[j] for j, name in enumerate(self.joints)}
        meta = self.run["meta"]
        return poses, list(meta["keys"][i]), [tuple(h) for h in meta["held"][i]]


def robot_trajectory(blob, rate=None):
    """The instrument trajectory of a recorded attempt, for a two-arm robot: per kept frame
    and hand, the tool, its tip position (m, scene frame), its pointing direction and
    whether it is closed (forceps gripping / holding the needle). Idle time is shortened
    as in the autopilot and time restarts from the kept frames."""
    run = load(blob)
    names = ("scalpel", "needle_holder", "forceps", "scissors")
    frames, clock, previous = [], 0.0, None
    for i in active_frames(run):
        t = float(run["times"][i])
        clock += 0.0 if previous is None else min(t - previous, 1/30)
        previous = t
        hands = {}
        for k, side in enumerate(SIDES):
            tool = int(run["tools"][i][k])
            hands[side] = {"tool": None if tool < 0 else names[tool],
                           "tip": [round(float(v), 5) for v in run["tips"][i][k]],
                           "direction": [round(float(v), 4) for v in run["dirs"][i][k]],
                           "closed": bool(run["closed"][i][k])}
        frames.append({"t": round(clock, 4), **hands})
    meta = run["meta"]
    return {"attempt": meta["summary"]["id"], "site": meta["summary"]["site"],
            "score": meta["summary"]["score"], "frame": "scene: +x right, +y away from surgeon, +z up",
            "units": "metres", "frames": frames}
