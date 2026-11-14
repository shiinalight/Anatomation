"""Coronary artery bypass graft (CABG) on the heart model, as a guided four-step procedure.

1. Cut the coronary artery (arteriotomy) along the marked line.
2. Suture the vein graft onto the coronary artery (distal anastomosis).
3. Make a small incision in the ascending aorta (aortotomy).
4. Suture the graft onto the ascending aorta (proximal anastomosis); the graft then fills
   with blood.

The marked lines were placed by ray-casting the heart model in the scene: the coronary cut
lies on the straight arm of the visible coronary artery on the anterior surface, and the
aortotomy on the top ridge of the ascending aorta. All heights come from the meshes at
run time (`HeartTissue.any_point`), so the targets sit exactly on the vessel surfaces.
"""

import numpy as np
import mujoco as mj

from or_scene import HEART_RATE, add_capsule, add_geom, surface_strip

UP = np.array([0., 0., 1.])
STITCHES_PER_ANASTOMOSIS = 6
COVERAGE_NEEDED = .85          # fraction of the marked line that must be cut
CUT_WIDTH = .003               # the blade must stay this close to the marked line
PUNCTURE_REACH = .003          # how close (horizontally) a needle puncture must be to its target
GRAFT_RADIUS = .0028


class Incision:
    """A marked straight cut on a vessel: centre (xy), direction (xy), half length."""

    def __init__(self, name, centre, direction, half):
        self.name = name
        self.centre = np.asarray(centre, float)
        d = np.asarray(direction, float)
        self.dir = d / np.linalg.norm(d)
        self.side = np.array([-self.dir[1], self.dir[0]])
        self.half = half

    def point(self, along, side=0.0):
        return self.centre + self.dir*along + self.side*side

    def coords(self, xy):
        rel = np.asarray(xy[:2]) - self.centre
        return float(rel @ self.dir), float(rel @ self.side)


# Placed on the model as positioned in or_scene (HEART_POS / HEART_AXES).
CORONARY = Incision("coronary artery", (-.0432, .0062), (.686, .727), .006)
AORTA = Incision("ascending aorta", (-.125, .006), (1., 0.), .004)

STEPS = (
    ("cut", CORONARY, "Cut the coronary artery"),
    ("suture", CORONARY, "Suture the graft onto the coronary artery"),
    ("cut", AORTA, "Make a small incision in the ascending aorta"),
    ("suture", AORTA, "Suture the graft onto the ascending aorta"),
)


class BypassProcedure:
    def __init__(self, tissue):
        self.tissue = tissue
        self.step = 0
        self.samples = 15
        self.covered = np.zeros(self.samples, bool)
        self.tied = 0
        self.finished_at = None
        self.rings = {cut.name: self._ring(cut) for cut in (CORONARY, AORTA)}
        self.lines = {cut.name: self._line_marks(cut) for cut in (CORONARY, AORTA)}
        self.hoods = {cut.name: self._hood(cut) for cut in (CORONARY, AORTA)}

    # --- geometry ------------------------------------------------------------------
    def _surface(self, xy, lift=0.0):
        return self.tissue.any_point(xy, lift)

    def _ring(self, cut):
        """(graft entry, artery exit) points around the incision for each stitch."""
        pairs = []
        for k in range(STITCHES_PER_ANASTOMOSIS):
            a = 2*np.pi*k/STITCHES_PER_ANASTOMOSIS
            rim = cut.point(np.cos(a)*(cut.half + .0025), np.sin(a)*.0035)
            outward = rim - cut.centre
            outward /= np.linalg.norm(outward)
            exit_ = self._surface(rim, .0010)
            entry = self._surface(rim + outward*.0022, GRAFT_RADIUS*1.2)
            pairs.append((entry, exit_))
        return pairs

    def _line_marks(self, cut):
        """White surgical-marker dashes along the line to cut."""
        geoms = []
        for along in np.linspace(-cut.half, cut.half, 7):
            p = self._surface(cut.point(along), .0004)
            n = self.tissue.any_normal(*p[:2])
            a = self._surface(cut.point(along - .0006), .0004)
            b = self._surface(cut.point(along + .0006), .0004)
            geoms.append(surface_strip(a, b, .0005, .0002, n, "target"))
        return geoms

    def _hood(self, cut):
        """The graft's spatulated end lying over the incision."""
        c = self._surface(cut.centre, GRAFT_RADIUS*.6)
        n = self.tissue.any_normal(*cut.centre)
        a = self._surface(cut.point(-cut.half - .002), GRAFT_RADIUS*.6)
        b = self._surface(cut.point(cut.half + .002), GRAFT_RADIUS*.6)
        size, _, matrix, _ = surface_strip(a, b, .0042, GRAFT_RADIUS*.55, n, "graft")
        return size, c, matrix

    def _graft_path(self):
        """Centre line of the graft from the coronary heel to the aorta (or its loose end)."""
        heel = self._surface(CORONARY.point(-CORONARY.half - .002), GRAFT_RADIUS)
        if self.step >= 3:
            end = self._surface(AORTA.point(AORTA.half + .002), GRAFT_RADIUS)
            end_tangent = np.array([*AORTA.dir, 0.])
        else:   # not yet attached: the free end rests beside the aorta
            end = self._surface(AORTA.centre, 0) + np.array([.018, -.032, .018])
            end_tangent = np.array([.3, -1., 0.])
        p0, p3 = heel, end
        p1 = p0 + np.array([*-CORONARY.dir, 0.])*.014 + UP*.016
        p2 = p3 + end_tangent*.016 + UP*.022
        t = np.linspace(0, 1, 22)[:, None]
        return (1 - t)**3*p0 + 3*(1 - t)**2*t*p1 + 3*(1 - t)*t**2*p2 + t**3*p3

    # --- state -----------------------------------------------------------------------
    @property
    def done(self):
        return self.step >= len(STEPS)

    def current(self):
        return None if self.done else STEPS[self.step]

    def _advance(self, station):
        self.step += 1
        self.covered[:] = False
        self.tied = 0
        station.cut_line = None
        if self.done:
            self.finished_at = station.clock
            station.events.append("complete")
        else:
            station.events.append("switch")

    def cut_allowed(self, xy):
        """True where the blade may open a vessel: on the marked line of a cut step."""
        step = self.current()
        if step is None or step[0] != "cut":
            return False
        along, side = step[1].coords(xy)
        return abs(side) <= CUT_WIDTH and abs(along) <= step[1].half + .003

    def on_cut(self, station, point):
        step = self.current()
        if step is None or step[0] != "cut" or not self.cut_allowed(point):
            return
        cut = step[1]
        along, _ = cut.coords(point)
        samples = np.linspace(-cut.half, cut.half, self.samples)
        self.covered |= np.abs(samples - along) <= .0016
        if self.covered.mean() >= COVERAGE_NEEDED:
            self._advance(station)

    # --- tools -----------------------------------------------------------------------
    def align(self, station):
        """N key. Returns True when the procedure positioned the held tool."""
        step = self.current()
        if step is None:
            return False
        kind, cut, _ = step
        if station.held == 0 and kind == "cut":
            start = self._surface(cut.point(-cut.half - .001))
            station.offset += start + UP*.003 - station.active_tip()
            station.cut_line = cut
            return True
        if station.held == 1 and kind == "suture":
            if station.exit is not None:
                return True
            entry, exit_ = self.rings[cut.name][self.tied]
            target = entry if station.entry is None else exit_
            station.guide_tip = target + UP*.004
            station.guide_lower_steps = 2
            return True
        return False

    def puncture(self, station, tip):
        """F key with the needle holder during a suture step. Returns True if handled."""
        step = self.current()
        if step is None or step[0] != "suture":
            return False
        if station.exit is not None:
            station.message = "T (tie this stitch)"
            return True
        entry, exit_ = self.rings[step[1].name][self.tied]
        target = entry if station.entry is None else exit_
        close = np.linalg.norm(tip[:2] - target[:2]) <= PUNCTURE_REACH
        depth_ok = target[2] - .012 <= tip[2] <= target[2] + .001
        if not (close and depth_ok):
            station.message = "N to align on the next target, then E E and F"
            return True
        if station.entry is None:
            station.entry = entry
        else:
            station.exit = exit_
        station.bleed(target, .0008)
        station.events.append("puncture")
        return True

    def tie(self, station):
        """T key during a suture step. Returns True if handled."""
        step = self.current()
        if step is None or step[0] != "suture":
            return False
        if station.exit is None:
            station.message = "Pass the needle through graft and artery first"
            return True
        station.stitches.append((station.entry, station.exit, False))
        station.entry = station.exit = station.guide_tip = None
        station.events.append("tie")
        self.tied += 1
        if self.tied >= STITCHES_PER_ANASTOMOSIS:
            self._advance(station)
        return True

    def prompt(self, station):
        if self.done:
            return "Bypass complete: the graft is perfused. R or Clear repair to restart"
        kind, cut, title = STEPS[self.step]
        head = f"Step {self.step + 1}/4 {title.lower()}: "
        if kind == "cut":
            if station.held != 0:
                return head + ("G set down, then 1 + G (scalpel)" if station.held is not None
                               else "1 then G (scalpel)")
            if station.cut_line is None:
                return head + "N (put the blade on the marked line)"
            if station.depth <= 0:
                return head + "E (lower the blade into the vessel)"
            return head + f"left / right arrows cut along the line ({self.covered.mean():.0%})"
        n = f"stitch {self.tied + 1}/{STITCHES_PER_ANASTOMOSIS}"
        if station.held != 1:
            return head + ("G set down, then 2 + G (needle holder)" if station.held is not None
                           else "2 then G (needle holder)")
        if station.exit is not None:
            return head + f"T (tie {n})"
        if station.guide_tip is None:
            return head + f"N (align {n} on the {'graft' if station.entry is None else 'artery'})"
        if station.guide_lower_steps:
            return head + f"{station.guide_lower_steps}x E (lower needle)"
        return head + f"F (pass {n} through the {'graft' if station.entry is None else 'artery'})"

    # --- drawing ---------------------------------------------------------------------
    def draw(self, scene, clock):
        ellipse = mj.mjtGeom.mjGEOM_ELLIPSOID
        step = self.current()
        if step is not None and step[0] == "cut":
            for size, pos, matrix, material in self.lines[step[1].name]:
                add_geom(scene, ellipse, size, pos, matrix, material)
        if self.step >= 1:
            perfused = self.done
            pulse = 1.0
            if perfused:
                beat = (clock*HEART_RATE/60) % 1.0
                pulse = 1 + .12*np.exp(-((beat - .3)/.08)**2)
            material = "graft_perfused" if perfused else "graft"
            path = self._graft_path()
            for a, b in zip(path, path[1:]):
                add_capsule(scene, a, b, GRAFT_RADIUS*pulse, material)
            for cut in (CORONARY,) + ((AORTA,) if self.step >= 3 else ()):
                size, pos, matrix = self.hoods[cut.name]
                add_geom(scene, ellipse, size*[1, pulse, 1], pos, matrix, material)
        if step is not None and step[0] == "suture":
            for k, (entry, exit_) in enumerate(self.rings[step[1].name]):
                if k < self.tied:
                    continue
                material = "button_next" if k == self.tied else "target"
                for p in (entry, exit_):
                    add_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, [.0007, 0, 0], p, np.eye(3), material)
