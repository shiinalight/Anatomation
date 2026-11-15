"""The procedures a trainee can choose (Procedures window or P), and the incision task.

Each procedure is a site (where: skin or heart) plus a task (what):
  incision  cut along a marked line, the other hand steadying the tissue
  suture    repair the laceration, stitch by stitch
  bypass    the guided coronary artery bypass (heart only, see bypass.py)
  stitch    free stitching: stitches anywhere along the laceration, any number, with the
            stitch technique prompted step by step
  glove     hands-only suturing: tools already in hand, six highlighted stitches done with the
            glove movements alone, no keys (see glove_suture.py)
  free      no prompts and no forced order; cutting needs no steadying hand
"""

from dataclasses import dataclass

import numpy as np

from bypass import BypassProcedure, Incision
from or_scene import INCISION_CENTER, PAD_CENTER, SITES


@dataclass(frozen=True)
class Procedure:
    key: str
    label: str
    site: str        # key in or_scene.SITES (geometry, visible groups, camera)
    task: str        # incision / suture / stitch / glove / bypass / free


PROCEDURES = {p.key: p for p in (
    Procedure("skin-incision", "Skin: incision", "skin", "incision"),
    Procedure("skin-suture", "Skin: suture the laceration", "skin", "suture"),
    Procedure("skin-stitch", "Skin: free stitching", "skin", "stitch"),
    Procedure("skin-gloves", "Skin: hands-only suturing (gloves, no keys)", "skin", "glove"),
    Procedure("skin-free", "Skin: free practice", "skin", "free"),
    Procedure("heart-incision", "Heart: incision on the left ventricle", "heart", "incision"),
    Procedure("heart-suture", "Heart: suture the laceration", "heart", "suture"),
    Procedure("heart-bypass", "Heart: coronary artery bypass", "bypass", "bypass"),
    Procedure("heart-stitch", "Heart: free stitching", "heart", "stitch"),
    Procedure("heart-free", "Heart: free practice", "heart", "free"),
)}
DEFAULT = "skin-suture"
# Recordings made before procedures existed were labelled by site.
LEGACY = {"skin": "skin-suture", "heart": "heart-suture", "bypass": "heart-bypass"}


def from_site(name):
    """The procedure for an old --site value or recording label."""
    return PROCEDURES[LEGACY.get(name, name)]


class IncisionProcedure(BypassProcedure):
    """One marked incision: steady the tissue beside the line, then cut along it."""
    steady_cut = True

    def __init__(self, tissue, line, title, steady_side, steady_reach, steady_tools):
        self.tissue = tissue
        self.title = title
        self.steady_side, self.steady_reach, self.steady_tools = steady_side, steady_reach, steady_tools
        self.steps = (("cut", line, "Cut along the marked line"),)
        self.step = 0
        self.samples = 15
        self.covered = np.zeros(self.samples, bool)
        self.tied = 0
        self.finished_at = None
        self.rings, self.hoods = {}, {}
        self.lines = {line.name: self._line_marks(line)}

    def prompt_done(self):
        return f"{self.title} complete. R or Clear repair to start over"

    def draw(self, scene, clock):
        from or_scene import add_geom
        import mujoco as mj
        if self.done:
            return
        for size, pos, matrix, material in self.lines[self.steps[0][1].name]:
            add_geom(scene, mj.mjtGeom.mjGEOM_ELLIPSOID, size, pos, matrix, material)


# Skin: 8 cm line on the pad, between the upper marker line and the pad edge (clear of the
# laceration); fingers or forceps stretch the skin 1.5 cm beside it.
SKIN_LINE = Incision("skin incision", (INCISION_CENTER[0], PAD_CENTER[1] + .047), (1., 0.), .04)
# Heart: 2.4 cm along the laceration line on the left ventricle (>= 7 mm from the vessels);
# the forceps hold the muscle 5 mm beside it.
HEART_LINE = Incision("left ventricle incision", INCISION_CENTER, SITES["heart"].direction, .012)


def make_procedure(task, tissue):
    """The guided procedure object for a task, or None (suture, stitch and free are run by the station)."""
    if task == "bypass":
        return BypassProcedure(tissue)
    if task == "incision":
        if tissue.site.name == "skin":
            return IncisionProcedure(tissue, SKIN_LINE, "Skin incision", .015, .012, (None, 2))
        return IncisionProcedure(tissue, HEART_LINE, "Heart incision", .005, .004, (2,))
    return None
