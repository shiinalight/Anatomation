"""Hands-only suturing: the gloves drive the whole repair, no keys.

The trainee starts with the needle holder in the right hand and the forceps in the left,
both just above the cut on the chest. Six stitches go along the laceration, one at a time:

  1. reach  the next entry hole glows yellow; lower the needle toward it. Within MAGNET_XY the
            tip is drawn over the hole; in the catch zone (SNAP_XY around it, SNAP_Z above) it
            snaps in and the needle goes in.
  2. drive  move the right hand a little (DRIVE_TRAVEL, any direction; across the cut is the
            natural one): the needle follows its curve under the skin and comes out of the exit
            hole on the far side (yellow dots show the way).
  3. pull   lift the right hand (PULL_TRAVEL): the thread is pulled through and the stitch is
            tied.

Then the next hole lights up. The left hand's forceps are carried along (they are not needed
for a step). Everything comes from the glove poses, so a recording replays exactly and a
complete attempt becomes an ideal run for the autopilot like any other procedure.
"""

import numpy as np
import mujoco as mj

from or_scene import add_capsule, add_geom

STITCHES = 6
# The catch is generous so the stitches take little effort with real gloves:
SNAP_XY = .03            # needle tip within 3 cm sideways of the yellow hole ...
SNAP_Z = .025            # ... and less than 2.5 cm above it: it snaps in
MAGNET_XY = .07          # within 7 cm sideways the tip is drawn over the hole as the hand lowers
MAGNET_PULL = .06        # fraction of the remaining sideways gap closed each frame (at the hole)
DRIVE_TRAVEL = .025      # hand movement (any direction) that takes the needle through
PULL_TRAVEL = .03        # hand lift that pulls the thread through
NEEDLE_DEPTH = .005      # deepest point of the needle's path under the skin
START_LIFT = .07         # the tools start this far above the cut
UP = np.array([0., 0., 1.])
TOOLS = {"right": 1, "left": 2}     # needle holder, forceps
PHASES = ("reach", "drive", "pull")


class GloveSuture:
    def __init__(self, station):
        self.st = station
        site = station.site
        self.holes = []
        for along in np.linspace(-site.half*.8, site.half*.8, STITCHES):
            entry = station.surface(site.point(along, -site.bite_guide))
            exit_ = station.surface(site.point(along, site.bite_guide))
            self.holes.append((entry, exit_))
        self.k = 0                    # stitch in progress
        self.phase = "reach"
        self.progress = 0.0           # of the drive or pull
        self.distance = None          # needle tip to the yellow hole (reach)
        self.anchor = None            # raw glove tip when the phase began
        self.placed = set()           # hands already brought above the cut
        self.finished_at = None

    @property
    def done(self):
        return self.k >= STITCHES

    # --- the hands ---------------------------------------------------------------------
    def start_point(self, side):
        site = self.st.site
        if side == "right":
            xy = site.point(-site.half*.8, -site.bite_guide*2.5)
        else:
            xy = site.point(-site.half*.8 - .03, site.bite_guide*2.5)
        return self.st.surface(xy) + UP*START_LIFT

    def place(self, hand):
        """The tool tip just above the start of the cut, wherever the glove is."""
        st = self.st
        hand.tool_pose = st.pose_for_tool(hand)
        hand.offset += self.start_point(hand.side) - st.active_tip(hand)
        hand.tool_pose = st.pose_for_tool(hand)
        self.placed.add(hand.side)

    def equip(self):
        """Both tools in hand from the start (and again after a glove drops out)."""
        st = self.st
        for side, tool in TOOLS.items():
            hand = st.hands[side]
            if hand.pose is None:
                continue
            if hand.held is None and st.holder(tool) is None:
                hand.held = hand.selected = tool
                hand.pinch_finger = None        # held whatever the fingers do
                hand.tool_pose = st.pose_for_tool(hand)
            if side not in self.placed and hand.held is not None:
                self.place(hand)

    def lock(self, hand, point):
        """Move the drawn hand so the needle tip sits on `point` (the raw glove keeps going)."""
        st = self.st
        hand.offset += point - st.active_tip(hand)
        hand.tool_pose = st.pose_for_tool(hand)

    def raw_tip(self, hand):
        return self.st.active_tip(hand) - hand.offset

    def path(self, t):
        """The needle tip's curve from the entry to the exit hole, under the skin."""
        entry, exit_ = self.holes[self.k]
        normal = self.st.tissue.normal(*((entry + exit_)/2)[:2])
        return entry + (exit_ - entry)*t - normal*NEEDLE_DEPTH*np.sin(np.pi*t)

    # --- each frame ----------------------------------------------------------------------
    def update(self):
        st = self.st
        self.equip()
        hand = st.hands["right"]
        if self.done or hand.pose is None or hand.held != 1 or hand.tool_pose is None:
            return
        entry, exit_ = self.holes[self.k]
        if self.phase == "reach":
            tip = st.active_tip(hand)
            gap = entry - tip
            sideways, height = float(np.linalg.norm(gap[:2])), float(-gap[2])
            if sideways <= MAGNET_XY:          # the magnet: drawn over the hole, height stays yours
                hand.offset[:2] += gap[:2]*MAGNET_PULL*(1 - sideways/MAGNET_XY)
                hand.tool_pose = st.pose_for_tool(hand)
            self.distance = float(np.linalg.norm(st.active_tip(hand) - entry))
            if sideways <= SNAP_XY and height <= SNAP_Z:
                # the snap is part of the design, so its distance is kept but not scored as error
                st.note("puncture", hand="right", edge="entry", error_mm=None,
                        snap_mm=round(self.distance*1000, 2))
                self.lock(hand, entry)
                self.anchor = self.raw_tip(hand)
                self.phase, self.progress = "drive", 0.0
                st.bleed(entry, .0011)
                st.events.append("puncture")
        elif self.phase == "drive":
            across = exit_ - entry
            across[2] = 0
            across /= max(np.linalg.norm(across), 1e-9)
            # any movement counts (across the cut is the natural one); hand tremor stays below it
            moved = float(np.linalg.norm(self.raw_tip(hand) - self.anchor)) / DRIVE_TRAVEL
            self.progress = max(self.progress, min(max(moved, 0.0), 1.0))   # never slips back
            self.lock(hand, self.path(self.progress))
            if self.progress >= 1:
                st.note("puncture", hand="right", edge="exit", error_mm=None)
                self.anchor = self.raw_tip(hand)
                self.phase, self.progress = "pull", 0.0
                st.bleed(exit_, .0011)
                st.events.append("puncture")
        else:   # pull: the tool follows the hand freely again, out of the hole
            lifted = float((self.raw_tip(hand) - self.anchor)[2]) / PULL_TRAVEL
            self.progress = max(self.progress, min(max(lifted, 0.0), 1.0))
            if self.progress >= 1:
                st.stitches.append((entry, exit_, True))
                st.note("stitch", number=self.k + 1)
                st.events.append("tie")
                self.k += 1
                self.phase, self.progress, self.distance = "reach", 0.0, None
                if self.done:
                    self.finished_at = st.clock
                    st.events.append("complete")

    def reset_hands(self):
        """Reset hands: both tools back above the cut (the stitch in progress starts again)."""
        self.placed.clear()
        if self.phase != "reach":
            self.phase, self.progress = "reach", 0.0

    # --- guidance -----------------------------------------------------------------------
    def plan(self):
        if self.done:
            return None, f"All {STITCHES} stitches done: showcase complete. Clear repair to start again"
        if self.st.hands["right"].pose is None:
            return "right", "Right glove not tracked: hold your right hand where the gloves see it"
        n = f"Stitch {self.k + 1}/{STITCHES}"
        if self.phase == "reach":
            far = "" if self.distance is None else f" ({self.distance*1000:.0f} mm)"
            return "right", f"{n}: lower the needle onto the yellow point{far}"
        if self.phase == "drive":
            return "right", (f"{n}: push the needle through: move your right hand across the cut "
                             f"({self.progress:.0%})")
        return "right", f"{n}: pull the thread through: lift your right hand up ({self.progress:.0%})"

    def checklist(self):
        rows = []
        for k in range(STITCHES):
            if k < self.k:
                rows.append(("x", f"Stitch {k + 1}"))
            elif k == self.k:
                rows.append((">", f"Stitch {k + 1} of {STITCHES}"))
                at = PHASES.index(self.phase)
                for i, text in enumerate(("Needle tip onto the yellow point",
                                          "Across the cut: needle out the far side",
                                          "Lift: thread pulled through")):
                    rows.append(("x" if i < at else ">" if i == at else " ", "  " + text))
            else:
                rows.append((" ", f"Stitch {k + 1}"))
        return rows

    # --- drawing ------------------------------------------------------------------------
    def draw(self, scene, clock):
        sphere, eye = mj.mjtGeom.mjGEOM_SPHERE, np.eye(3)
        for k in range(self.k + 1, STITCHES):            # the holes still to come
            for p in self.holes[k]:
                add_geom(scene, sphere, [.0012, 0, 0], p, eye, "target")
        if self.done:
            return
        entry, exit_ = self.holes[self.k]
        pulse = 1 + .25*np.sin(clock*6)
        hand = self.st.hands["right"]
        tip = self.st.active_tip(hand) if hand.tool_pose is not None and hand.held == 1 else None
        if self.phase == "reach":
            add_geom(scene, sphere, [.0028*pulse, 0, 0], entry, eye, "highlight")
            add_geom(scene, mj.mjtGeom.mjGEOM_CYLINDER, [SNAP_XY, SNAP_Z/2, 0],
                     entry + UP*SNAP_Z/2, eye, "highlight_zone")     # the catch zone
            add_geom(scene, sphere, [.0014, 0, 0], exit_, eye, "target")
            return
        if self.phase == "drive":
            add_geom(scene, sphere, [.0028*pulse, 0, 0], exit_, eye, "highlight")
            for t in np.linspace(.1, .9, 5):              # the way across, on the skin
                dot = entry + (exit_ - entry)*t + UP*.002
                add_geom(scene, sphere, [.0011, 0, 0], dot, eye, "highlight")
            if tip is not None:
                add_capsule(scene, entry, tip, .0008, "thread")
            return
        for t in np.linspace(.2, 1, 5):                   # pull: the way up
            add_geom(scene, sphere, [.0011, 0, 0], exit_ + UP*PULL_TRAVEL*t, eye, "highlight")
        if tip is not None:
            add_capsule(scene, entry, exit_, .0008, "thread")
            add_capsule(scene, exit_, tip, .0008, "thread")
