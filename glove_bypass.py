"""Hands-only coronary bypass: the gloves drive it with snap points, no keys.

The right hand does it all, in six steps (the left hand rests):

  1. cut     the scalpel is in hand; a yellow point marks where the blade goes on the coronary
             artery, and the whole line lights up as the blade comes close. In the catch zone
             the blade snaps in; moving the hand runs it along the line.
  2. pick    the scalpel goes back to the tray; the end of the vein graft glows yellow. Reach
             for it: it snaps and glides into the hand.
  3. attach  carry it to the yellow point on the opening: it snaps on and is sewn on by itself.
  4. cut     the scalpel is back in hand: the same on the ascending aorta.
  5. pick    the graft's free end glows yellow: reach for it.
  6. attach  carry it to the opening in the aorta: it is sewn on and the graft fills with blood.

As in glove_suture.py, the gloves only turn about the wrist: each step starts with the tip
(and the graft, when carried) gliding to just above its yellow point, so every step is a
wrist movement; nothing starts before the glove is live and its direction locked. The catch
works as in glove_suture.py: a magnet draws the tip over the target sideways while
the trainee controls the height, and the catch zone is shown as a soft yellow disc. The geometry
(marked lines, graft path, anastomosis hoods and stitch rings) comes from bypass.BypassProcedure.
Everything is driven by the glove poses, so recordings and the autopilot replay it exactly.
"""

import numpy as np
import mujoco as mj

from bypass import AORTA, CORONARY, GRAFT_RADIUS, STITCHES_PER_ANASTOMOSIS, BypassProcedure
from glove_suture import settle
from or_scene import HEART_RATE, add_capsule, add_geom

UP = np.array([0., 0., 1.])
SNAP_XY = .015           # tip within 1.5 cm sideways of the yellow point ...
SNAP_Z = .02             # ... and less than 2 cm above it: it snaps
MAGNET_XY = .05          # within 5 cm sideways the tip is drawn over the point as the hand lowers
MAGNET_PULL = .06
SHOW_LINE = .05          # the cut line lights up when the blade is this close
CUT_TRAVEL = .025        # hand movement (any direction) that runs the blade along the whole line
CUT_DEPTH = .0015
READY_LIFT = .035        # each step starts with the tip glided this far above its yellow point
ARM = 1.5                # a snap point arms once the tip has been this many catch zones away
GRAB_TIME = .4           # the graft end glides into the hand over this time (s)
SEW_TIME = 1.2           # an anastomosis is sewn on over this time (s)

STAGES = (
    ("cut", CORONARY, "Cut the coronary artery"),
    ("pick", "heel", "Pick up the bypass graft"),
    ("attach", CORONARY, "Attach the graft to the coronary artery"),
    ("cut", AORTA, "Cut the ascending aorta"),
    ("pick", "free", "Pick up the free end of the graft"),
    ("attach", AORTA, "Attach the graft to the aorta"),
)


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u*u*(3 - 2*u)


def _bezier(p0, p1, p2, p3, n=22):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t)**3*p0 + 3*(1 - t)**2*t*p1 + 3*(1 - t)*t**2*p2 + t**3*p3


class GloveBypass:
    def __init__(self, station):
        self.st = station
        self.bp = BypassProcedure(station.tissue)        # geometry only
        self.blade_home = (station.rest[0][0].copy(), station.rest[0][1].copy())
        # The graft lies beside the aorta: its free end where BypassProcedure rests it, the heel
        # (the end sewn to the coronary artery first) 4.5 cm toward the coronary artery.
        self.free_rest = self.bp._surface(AORTA.centre, 0) + np.array([.018, -.032, .018])
        self.heel_rest = self.free_rest + np.array([.045, -.010, 0.])
        self.stage = 0
        self.phase = "reach"          # cut steps: reach, then cut
        self.progress = 0.0
        self.distance = None
        self.anchor = None
        self.last_cut = None          # previous point of the blade on the line (wound segments)
        self.segments = 0
        self.armed = False            # the hand has been outside the catch zone in this step
        self.settling = True          # the tip glides above the step's yellow point first
        self.glide = None
        self.ready = None             # glove live and locked? None: not told yet (the display says)
        self.started = False
        self.carried = None           # "heel" or "free" while the graft end is in the hand
        self.grabbed_at, self.grab_from = None, None
        self.sewn_at = {}             # vessel name -> clock when the graft was attached
        self.finished_at = None

    @property
    def done(self):
        return self.stage >= len(STAGES)

    # --- the hand ----------------------------------------------------------------------
    def tip(self, hand):
        """The blade tip with the scalpel in hand, otherwise the fingertips (pinch point)."""
        return self.st.active_tip(hand) if hand.held == 0 else self.st.fingertip(hand)

    def refresh(self, hand):
        if hand.held is not None:
            hand.tool_pose = self.st.pose_for_tool(hand)

    def shift(self, hand, delta):
        hand.offset += delta
        self.refresh(hand)

    def equip(self, hand):
        """Scalpel in hand for a cut step, an empty hand for the graft."""
        st = self.st
        want = 0 if not self.done and STAGES[self.stage][0] == "cut" else None
        if hand.held == want:
            return
        if hand.held == 0:                                # the scalpel goes back to the tray
            st.rest[0] = (self.blade_home[0].copy(), self.blade_home[1].copy())
            hand.held = hand.tool_pose = None
            st.events.append("setdown")
        if want == 0 and st.holder(0) is None:
            hand.held = hand.selected = 0
            hand.pinch_finger = None
            self.refresh(hand)
            st.events.append("pickup")

    def target(self):
        """Where the current step's yellow point is."""
        kind, what, _ = STAGES[self.stage]
        if kind == "cut":
            return self.bp._surface(what.point(-what.half))
        if kind == "pick":
            return self.heel_rest if what == "heel" else self.free_rest
        return self.bp._surface(what.centre, GRAFT_RADIUS*.6)


    def catch(self, hand, target):
        """The magnet and the catch zone: True when the tip is caught on `target`."""
        gap = target - self.tip(hand)
        sideways = float(np.linalg.norm(gap[:2]))
        if sideways <= MAGNET_XY:
            hand.offset[:2] += gap[:2]*MAGNET_PULL*(1 - sideways/MAGNET_XY)
            self.refresh(hand)
        gap = target - self.tip(hand)
        self.distance = float(np.linalg.norm(gap))
        sideways, height = float(np.linalg.norm(gap[:2])), float(-gap[2])
        inside = sideways <= SNAP_XY and height <= SNAP_Z
        if sideways > SNAP_XY*ARM or height > SNAP_Z*ARM:
            self.armed = True
        return inside and self.armed   # a step never catches where the last one left the hand

    def raw(self, hand):
        return self.tip(hand) - hand.offset

    # --- each frame --------------------------------------------------------------------
    def update(self):
        hand = self.st.hands["right"]
        if hand.pose is None:
            return
        self.equip(hand)
        if self.done:
            return
        kind, what, _ = STAGES[self.stage]
        if self.phase == "reach" and self.settling:
            # The gloves only turn about the wrist: each step starts with the tip (and anything
            # in the hand) brought just above its yellow point; bending the wrist does the rest.
            self.glide = 0 if self.glide is None else self.glide + 1
            self.settling = not settle(hand, self.tip(hand), self.target() + UP*READY_LIFT, self.st,
                                       self.glide)
            if not self.settling:      # it ended above the catch zone: the snap point is armed
                self.glide, self.armed = None, True
            self.distance = float(np.linalg.norm(self.target() - self.tip(hand)))
            return
        if not self.ready:
            return
        self.started = True
        if kind == "cut":
            self.update_cut(hand, what)
        elif self.catch(hand, self.target()):
            if kind == "pick":
                self.carried, self.grabbed_at, self.grab_from = what, self.st.clock, self.target()
                self.st.note("grasp", hand="right", what=f"graft {what} end")
                self.st.events.append("pickup")
            else:
                self.carried = None
                self.sewn_at[what.name] = self.st.clock
                self.st.note("stitch", number=len(self.sewn_at))
                self.st.events.append("tie")
            self.advance()

    def update_cut(self, hand, cut):
        st = self.st
        if self.phase == "reach":
            if self.catch(hand, self.target()):
                self.phase, self.progress = "cut", 0.0
                self.anchor = self.raw(hand)
                self.last_cut = None
                st.note("puncture", hand="right", edge="blade", error_mm=None)
                st.events.append("puncture")
            else:
                return
        moved = float(np.linalg.norm(self.raw(hand) - self.anchor)) / CUT_TRAVEL
        self.progress = max(self.progress, min(max(moved, 0.0), 1.0))
        point = self.bp._surface(cut.point(-cut.half + 2*cut.half*self.progress))
        normal = st.tissue.any_normal(*point[:2])
        self.shift(hand, point - normal*CUT_DEPTH - self.tip(hand))   # the blade rides the line
        mark = point + normal*.0007
        if self.last_cut is None:
            self.last_cut = mark
        elif np.linalg.norm(mark - self.last_cut) >= .0008:
            st.marks.append(st.wound_segment(self.last_cut, mark, CUT_DEPTH, normal))
            self.segments += 1
            if self.segments % 3 == 0:                   # the opened vessel oozes
                st.bleed(mark, .0010)
            self.last_cut = mark
        if self.progress >= 1:
            st.note("procedure_step", done=self.stage + 1, action="cut", vessel=cut.name)
            self.advance()

    def advance(self):
        self.stage += 1
        self.phase, self.progress, self.distance = "reach", 0.0, None
        self.armed = False
        self.settling = True
        if self.done:
            self.finished_at = self.st.clock
            self.st.events.append("complete")
        else:
            self.st.events.append("switch")

    def reset_hands(self):
        """Reset hands: the hand back above the current target (a cut starts its line again)."""
        self.settling = True
        if self.phase == "cut":
            self.phase, self.progress = "reach", 0.0

    # --- guidance ----------------------------------------------------------------------
    def plan(self):
        if self.done:
            return None, "Bypass complete: the graft is perfused. Clear repair to start again"
        if self.st.hands["right"].pose is None:
            return "right", "Right glove not tracked: hold your right hand where the gloves see it"
        if not self.ready:
            return "right", ("Rest your right hand flat, fingers to the screen, to "
                             + ("continue" if self.started else "start"))
        if self.settling:
            return "right", "Relax your wrist: moving into place"
        kind, what, _ = STAGES[self.stage]
        n = f"Step {self.stage + 1}/{len(STAGES)}"
        far = "" if self.distance is None else f" ({self.distance*1000:.0f} mm)"
        if kind == "cut" and self.phase == "cut":
            return "right", f"{n}: cut along the yellow line: turn your wrist ({self.progress:.0%})"
        if kind == "cut":
            return "right", f"{n}: bend your wrist down: scalpel onto the {what.name}{far}"
        if kind == "pick":
            end = "end of the vein graft" if what == "heel" else "free end of the graft"
            return "right", f"{n}: bend your wrist down: take the yellow {end}{far}"
        return "right", f"{n}: bend your wrist down: graft onto the {what.name} opening{far}"

    def checklist(self):
        rows = []
        for i, (kind, _, title) in enumerate(STAGES):
            mark = "x" if i < self.stage else ">" if i == self.stage else " "
            rows.append((mark, f"{i + 1}. {title}"))
            if i == self.stage and kind == "cut":
                rows.append(("x" if self.phase == "cut" else ">", "  Scalpel on the yellow point"))
                rows.append((">" if self.phase == "cut" else " ",
                             f"  Cut along the line: {self.progress:.0%}"))
        return rows

    # --- drawing -----------------------------------------------------------------------
    def carry_point(self):
        hand = self.st.hands["right"]
        if hand.pose is None:
            return self.grab_from
        u = _smooth((self.st.clock - self.grabbed_at)/GRAB_TIME)   # glides into the hand
        return self.grab_from + (self.st.fingertip(hand) - self.grab_from)*u

    def graft_path(self):
        bp = self.bp
        coronary_on, aorta_on = CORONARY.name in self.sewn_at, AORTA.name in self.sewn_at
        if aorta_on:
            bp.step = 3
            return bp._graft_path()
        if coronary_on and self.carried != "free":
            bp.step = 1
            return bp._graft_path()
        if coronary_on:                                   # free end in the hand
            heel = bp._surface(CORONARY.point(-CORONARY.half - .002), GRAFT_RADIUS)
            end = self.carry_point()
            return _bezier(heel, heel + np.array([*-CORONARY.dir, 0.])*.014 + UP*.016,
                           end + UP*.022, end)
        heel = self.carry_point() if self.carried == "heel" else self.heel_rest
        free = self.free_rest
        sag = UP*(.022 if self.carried == "heel" else .002)
        return _bezier(free, free + np.array([.3, -1., 0.])*.016 + sag, heel + sag, heel)

    def draw(self, scene, clock):
        sphere, eye = mj.mjtGeom.mjGEOM_SPHERE, np.eye(3)
        ellipse = mj.mjtGeom.mjGEOM_ELLIPSOID
        perfused = self.done
        pulse = 1.0
        if perfused:
            beat = (clock*HEART_RATE/60) % 1.0
            pulse = 1 + .12*np.exp(-((beat - .3)/.08)**2)
        material = "graft_perfused" if perfused else "graft"
        path = self.graft_path()
        for a, b in zip(path, path[1:]):
            add_capsule(scene, a, b, GRAFT_RADIUS*pulse, material)
        for cut in (CORONARY, AORTA):                     # sewn-on ends, stitch by stitch
            if cut.name not in self.sewn_at:
                continue
            size, pos, matrix = self.bp.hoods[cut.name]
            add_geom(scene, ellipse, size*[1, pulse, 1], pos, matrix, material)
            sewn = int((clock - self.sewn_at[cut.name])/SEW_TIME*STITCHES_PER_ANASTOMOSIS) + 1
            for entry, exit_ in self.bp.rings[cut.name][:max(1, min(sewn, STITCHES_PER_ANASTOMOSIS))]:
                add_capsule(scene, entry, exit_, .0005, "thread")
                add_geom(scene, sphere, [.0009, 0, 0], exit_, eye, "thread")
        if self.done:
            return
        kind, what, _ = STAGES[self.stage]
        target = self.target()
        glow = 1 + .25*np.sin(clock*6)
        if kind == "cut":
            near = self.phase == "cut" or (self.distance is not None and self.distance <= SHOW_LINE)
            for size, pos, matrix, line_material in self.bp.lines[what.name]:
                add_geom(scene, ellipse, size, pos, matrix, "highlight" if near else line_material)
            if self.phase == "cut":
                return
        add_geom(scene, sphere, [.0022*glow, 0, 0], target, eye, "highlight")
        add_geom(scene, mj.mjtGeom.mjGEOM_CYLINDER, [SNAP_XY, SNAP_Z/2, 0], target + UP*SNAP_Z/2,
                 eye, "highlight_zone")
