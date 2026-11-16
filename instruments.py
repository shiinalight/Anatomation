"""The four instruments (scalpel, needle holder, forceps, suture scissors): where they rest,
where their tips are, and how they and the gloved hands are drawn."""

import mujoco as mj
import numpy as np

from glove_input import CHAINS
from or_scene import TRAY_TOP, add_bar, add_capsule, add_geom


REST_Z = TRAY_TOP + .004
TRAY_GRIP = np.array([-.221, .066, REST_Z])
TRAY_DIRECTION = np.array([0., 1., 0.])
PICKUP_RADIUS = .045
TOOL_NAMES = ("Scalpel", "Needle holder", "Forceps", "Suture scissors")
TOOL_HOME = (
    TRAY_GRIP,
    np.array([.27, .08, REST_Z]),
    np.array([-.271, .072, REST_Z]),
    np.array([.33, .082, REST_Z]),
)
TOOL_TIP = (.051, .084, .062, .050)


GLOVE = "glove"


def draw_hand(scene, points, active=False):
    if points is None:
        return

    def capsule(start, end, radius, color):
        add_capsule(scene, start, end, radius, color)

    wrist = points["wrist"]
    knuckles = np.array([points[CHAINS[name][1]] for name in ("index", "middle", "ring", "little")])
    across = knuckles[0] - knuckles[-1]
    forward = knuckles.mean(axis=0) - wrist
    length = np.linalg.norm(forward)
    width = np.linalg.norm(across)
    if length < 1e-6 or width < 1e-6:
        return
    forward /= length
    normal = np.cross(across, forward)
    normal_length = np.linalg.norm(normal)
    if normal_length < 1e-6:
        return
    normal /= normal_length
    across = np.cross(forward, normal)
    basis = np.column_stack((across, forward, normal))
    # Gown sleeve, with the glove's rolled cuff pulled over it.
    # 45 cm: forearm plus a little upper arm, enough to leave the bottom of the default view.
    capsule(wrist - forward*.45, wrist - forward*.07, .031, "gown")
    capsule(wrist - forward*.095, wrist + forward*.02, .0255, GLOVE)
    add_capsule(scene, wrist - forward*.092, wrist - forward*.082, .0322, GLOVE,
                mj.mjtGeom.mjGEOM_CYLINDER)
    if active:   # green band on the cuff of the hand the keyboard controls
        add_capsule(scene, wrist - forward*.125, wrist - forward*.112, .0328, "button_next",
                    mj.mjtGeom.mjGEOM_CYLINDER)
    add_geom(scene, mj.mjtGeom.mjGEOM_ELLIPSOID, [width*.57, length*.58, .013],
             wrist + forward*length*.51, basis, GLOVE)

    for finger, chain in CHAINS.items():
        thumb = finger == "thumb"
        base = points[chain[0]]
        knuckle = points[chain[1]]
        palm_start = wrist*.25 + base*.75 if thumb else wrist + (knuckle-wrist)*.62
        capsule(palm_start, knuckle, .013 if thumb else .011, GLOVE)
        for index, (root, tip) in enumerate(zip(chain[1:], chain[2:])):
            radius = (.0105 if thumb else .0095) - index*.0009
            capsule(points[root], points[tip], radius, GLOVE)


def draw_path(scene, vertices, radius, color):
    for start, end in zip(vertices, vertices[1:]):
        add_capsule(scene, start, end, radius, color)


def instrument_axes(direction):
    across = np.cross(direction, [0, 0, 1])
    if np.linalg.norm(across) < .01:
        across = np.cross(direction, [1, 0, 0])
    across /= np.linalg.norm(across)
    up = np.cross(across, direction)
    if up[2] < 0:
        up = -up
    return across, up


def _ring(scene, centre, radius, a, b, material, wire=.0016):
    points = [centre + radius*(np.cos(t)*a + np.sin(t)*b) for t in np.linspace(0, 2*np.pi, 13)]
    draw_path(scene, points, wire, material)


def draw_instrument(scene, index, grip, direction):
    d = direction
    across, up = instrument_axes(d)
    basis = np.column_stack((np.cross(up, d), up, d))
    box, ellipsoid = mj.mjtGeom.mjGEOM_BOX, mj.mjtGeom.mjGEOM_ELLIPSOID

    if index == 0:
        # No. 3 scalpel handle (knurled grip) with a No. 15 blade, cutting edge down.
        add_bar(scene, grip - d*.082, grip + d*.003, .0082, .0024, across, "satin")
        add_bar(scene, grip - d*.046, grip - d*.010, .0088, .0029, across, "knurl")
        add_bar(scene, grip + d*.002, grip + d*.013, .0042, .0019, across, "polished")
        add_bar(scene, grip + d*.011 + up*.0006, grip + d*.040 + up*.0006, .0062, .0005, across, "blade")
        add_geom(scene, ellipsoid, [.0003, .0037, .0135], grip + d*.038 - up*.0004, basis, "blade")
        return

    if index == 2:
        # Adson tissue forceps: two flat spring leaves welded at the back, toothed tips.
        back = grip - d*.050
        for side in (-1, 1):
            mid = grip - d*.004 + side*across*.0055
            tip = grip + d*TOOL_TIP[2] + side*across*.0008
            add_bar(scene, back + side*across*.0012, mid, .0068, .0013, across, "satin")
            add_bar(scene, mid, tip, .0026, .0012, across, "satin")
            for k in range(5):
                at = grip - d*(.026 - k*.005) + side*across*.0062
                add_bar(scene, at - up*.003, at + up*.003, .0007, .0006, d, "knurl")
        add_capsule(scene, back - d*.002, back + d*.004, .0034, "satin")
        return

    # Ring-handled instruments: Mayo-Hegar needle holder (gold rings) or Metzenbaum scissors.
    holder = index == 1
    pivot = grip + d*(.030 if holder else .010)
    ring_material = "gold" if holder else "satin"
    for side in (-1, 1):
        centre = grip - d*.050 + side*across*.011
        _ring(scene, centre, .0078, across, d, ring_material)
        shank_start = centre + d*.0078
        add_capsule(scene, shank_start, pivot - d*.004 + side*across*.0012, .0021, "polished")
        if holder:
            ratchet = grip - d*.036
            add_capsule(scene, ratchet + side*across*.0072, ratchet + side*across*.0018, .0011, "satin")
            add_capsule(scene, pivot + d*.004 + side*up*.0007, grip + d*.058 + side*up*.0004,
                         .0017, "polished")
        else:
            curve = up*.0016
            mid = pivot + d*.022 + side*across*.0012 + curve*.4
            tip = grip + d*TOOL_TIP[3] - side*across*.0018 + curve
            add_bar(scene, pivot, mid, .0048, .0011, across, "polished")
            add_bar(scene, mid, tip, .0026, .0010, across, "polished")
    add_geom(scene, box, [.0035, .0026, .0050], pivot, basis, "satin")
    add_capsule(scene, pivot - up*.0032, pivot + up*.0032, .0018, "polished", mj.mjtGeom.mjGEOM_CYLINDER)
    if holder:
        # Curved needle clamped near its midpoint, with the violet suture trailing.
        centre = grip + d*.059
        needle = [centre + d*(.025*np.sin(t)) + up*(.015*(np.cos(t)-1))
                  for t in np.linspace(-1.45, np.pi/2, 12)]
        draw_path(scene, needle, .00095, "polished")
        tail = needle[0]
        draw_path(scene, [tail, tail - across*.018 - up*.014,
                          tail - across*.034 - up*.019, tail - across*.05], .0006, "thread")
