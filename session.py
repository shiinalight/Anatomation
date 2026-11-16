"""One trainee at the table: the station for the chosen procedure, the recording and scoring
of each attempt, its upload (Aiven), the coach, and the autopilot."""

from collections import deque
import json
import time

import motion_log
import or_scene
from environment.or_controls import heart_home
from glove_input import HAND_ANCHOR, JOINTS
from instruments import TOOL_NAMES
from or_scene import SITES, make_tissue
from station import CuttingStation


def completed_text(kind, detail):
    """What the trainee just completed, for the overlay (None for events not worth showing)."""
    hand = detail.get("hand", "").capitalize()
    if kind == "pickup":
        return f"{hand} hand took the {TOOL_NAMES[detail['tool']].lower()}"
    if kind == "grip":
        return f"{hand} hand: tissue {'held with the forceps' if detail.get('tool') == 2 else 'stretched'}"
    if kind == "puncture":
        error = detail.get("error_mm")
        off = f" ({error:.1f} mm off target)" if error else ""
        return f"Needle {'through the first' if detail['edge'] == 'entry' else 'out through the second'} edge{off}"
    if kind == "grasp":
        return "Forceps grasped the needle"
    if kind == "pulled":
        return "Suture pulled through"
    if kind == "throw":
        return f"Throw {detail['throw']}/3 tied"
    if kind == "stitch":
        return f"Stitch {detail['number']} complete"
    if kind == "procedure_step":
        return f"Step complete: {'cut' if detail['action'] == 'cut' else 'sutured'} {detail['vessel']}"
    if kind == "trim":
        return "Thread trimmed"
    return None


class Session:
    """One trainee at the table: the station, the recording of the current attempt, its
    upload to Aiven (PostgreSQL, Kafka), the coach, and the autopilot.

    An attempt starts with a fresh station (start, site switch, R, Clear repair) and ends
    when the task is complete or the trainee starts over; it is then scored and saved. The
    autopilot replays the best complete attempt at the site into a fresh station.
    """

    def __init__(self, model, data, procedure, store, telemetry=None, coach=None, render=None, now=None):
        self.model, self.data = model, data
        self.store, self.telemetry, self.coach = store, telemetry, coach
        self.render = render or (lambda: None)
        self.autopilot = None
        self._notice, self._notice_at = "", -1e9
        self.last = None             # summary of the last saved attempt
        self.done = deque(maxlen=3)  # (time, text) of the last completed actions
        self.begin(procedure, time.monotonic() if now is None else now)

    def begin(self, procedure, now, record=True):
        """A fresh station for `procedure` (a procedures.Procedure)."""
        self.procedure = procedure
        self.site = SITES[procedure.site]
        with heart_home():         # targets are placed with the heart in the chest (it may be lifted out)
            self.station = CuttingStation(make_tissue(self.model, self.data, self.site, procedure.task),
                                          procedure.task)
        self.done.clear()
        self.recorder = (motion_log.Recorder(procedure.key, "keyboard", sorted(JOINTS), HAND_ANCHOR, now)
                         if record else None)

    @property
    def notice(self):
        return self._notice

    @notice.setter
    def notice(self, text):
        self._notice, self._notice_at = text, time.monotonic()

    def recent_done(self, now, seconds=6):
        return [text for at, text in self.done if now - at <= seconds]

    def recent_notice(self, seconds=8):
        return self._notice if time.monotonic() - self._notice_at < seconds else ""

    @property
    def attempt_id(self):
        return self.recorder.id if self.recorder is not None else None

    def replay_scene(self):
        """Skin attempts replay only if recorded on the room's chest skin (the heart did not move)."""
        return or_scene.SCENE if self.site.name == "skin" else None

    def complete(self):
        st = self.station
        if st.procedure is not None:
            return st.procedure.done
        if st.guide is not None:
            return st.guide.done
        if st.task in ("free", "stitch"):
            return False       # free practice and free stitching end when the trainee starts over
        return len(st.stitches) >= st.site.stitch_slots

    def end(self, now):
        """Score and save the current attempt (if anything happened)."""
        if self.recorder is None:
            return None
        attempt = self.recorder.finish(now, self.complete(), {"stitches": len(self.station.stitches),
                                                              "scene": or_scene.SCENE})
        self.recorder = None
        if attempt is None:
            return None
        self.store.save(attempt)
        self.last = attempt.summary
        best = self.store.best(self.procedure.key, self.replay_scene())
        ideal = " - new best, the autopilot will replay it" if (
            best and best[0]["id"] == attempt.summary["id"]) else ""
        self.notice = (f"Attempt saved: {attempt.summary['score']:.0f}/100 "
                       f"in {attempt.summary['duration_s']:.0f} s{ideal}")
        print(self.notice, flush=True)
        if self.coach is not None:
            self.coach.attempt_finished(attempt.summary, self.render)
        return attempt

    def restart(self, now, procedure=None):
        self.end(now)
        self.stop_autopilot(now, restart=False)
        self.station.park_all()
        self.begin(procedure or self.procedure, now)

    # --- autopilot ------------------------------------------------------------------------
    def start_autopilot(self, now):
        best = self.store.best(self.procedure.key, self.replay_scene())
        if best is None:
            self.notice = (f"{self.procedure.label} has no ideal run: choose a guided procedure "
                           "for the autopilot" if self.procedure.task in ("free", "stitch") else
                           f"No complete {self.procedure.label.lower()} attempt yet: finish one first")
            return False
        summary, blob = best
        self.end(now)
        self.station.park_all()
        self.begin(self.procedure, now, record=False)
        self.autopilot = motion_log.Autopilot(blob)
        robot = self.store.root / "robot" / f"{self.procedure.key}-{summary['id']}.json"
        robot.parent.mkdir(parents=True, exist_ok=True)
        if not robot.exists():
            robot.write_text(json.dumps(motion_log.robot_trajectory(blob)), encoding="utf-8")
        self.notice = (f"Autopilot: replaying the best attempt at {self.procedure.label} "
                       f"({summary['score']:.0f}/100, {summary['started_at'][:10]}); "
                       f"robot trajectory in {robot.relative_to(self.store.root.parent)}")
        print(self.notice, flush=True)
        return True

    def stop_autopilot(self, now, restart=True):
        if self.autopilot is None:
            return
        self.autopilot = None
        if restart:
            self.station.park_all()
            self.begin(self.procedure, now)
            self.notice = "Autopilot off: your turn"

    # --- per frame -------------------------------------------------------------------------
    def step(self, now, poses, keys, held, live=False):
        """Advance one frame with the trainee's input (ignored while the autopilot runs)."""
        if self.autopilot is not None:
            if self.autopilot.done:
                poses, keys, held = self._held_pose, [], []
            else:
                poses, keys, held = self.autopilot.next()
                self._held_pose = poses
        drawn = self.station.update(poses, keys, held)
        events, self.station.log = self.station.log, []
        if self.recorder is not None:
            if live:
                self.recorder.mode = "glove"
            self.recorder.frame(now, poses, keys, held, self.station, events)
        if self.telemetry is not None:
            self.telemetry.events(self.attempt_id, self.site.name, events)
            self.telemetry.frame(now, self.attempt_id, self.station, self.autopilot is not None)
        if self.coach is not None:
            self.coach.note_events(events)
        for kind, detail in events:
            text = completed_text(kind, detail)
            if text:
                self.done.append((now, text))
        if self.recorder is not None and self.complete():
            self.end(now)
        return drawn
