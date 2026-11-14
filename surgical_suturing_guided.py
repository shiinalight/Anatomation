"""Practice incisions and sutures on skin or on a heart in a glove-controlled MuJoCo operating room."""

import argparse
from collections import deque
from queue import SimpleQueue
from pathlib import Path
import subprocess
import sys
import threading
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

from or_scene import (SITES, TABLE_MAX, TABLE_MIN, TRAY_TOP, WORK_TOP, Look, Sounds, add_bar,
                      add_capsule, add_geom, build_model, draw_monitor, make_tissue, show_site,
                      surface_strip, SiteSwitch)

SDK = Path.home() / ".local/share/rokoko-device-sdk/examples/python"
ANCHOR = np.array([0, -0.10, 1.015])
FINGERS = {
    "thumb": ("metacarpal", "proximal", "distal", "tip"),
    **{name: ("metacarpal", "proximal", "intermediate", "distal", "tip")
       for name in ("index", "middle", "ring", "little")},
}
CHAINS = {
    finger: [f"{finger}_{joint}" for joint in joints]
    for finger, joints in FINGERS.items()
}
JOINTS = {"wrist", "palm", *(joint for chain in CHAINS.values() for joint in chain)}
MAX_DEPTH = .006
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
# GLFW key code -> (axis, metres per tap). Holding a key moves at MOVE_SPEED * step / s.
MOVES = {263: (0, -.006), 262: (0, .006), 264: (1, -.006), 265: (1, .006),
         ord("Q"): (2, .003), ord("E"): (2, -.003)}
MOVE_SPEED = 12      # held keys travel 12 taps per second (7 cm/s across, 3.6 cm/s up/down)
HOLD_DELAY = .18     # a quick tap stays a single precise step
MAC_KEYCODES = {263: 123, 262: 124, 264: 125, 265: 126, ord("Q"): 12, ord("E"): 14}



def _mac_key_state():
    """Return a function telling whether a macOS key is physically down, or None."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        lib = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        query = lib.CGEventSourceKeyState
        query.argtypes, query.restype = [ctypes.c_int32, ctypes.c_uint16], ctypes.c_bool
    except (OSError, AttributeError):
        return None
    return lambda code: code in MAC_KEYCODES and query(0, MAC_KEYCODES[code])


ZOOM_IN = {ord("="), 334}    # = (or +) and keypad +
ZOOM_OUT = {ord("-"), 333}   # - and keypad -
RESET_VIEW = ord("0")


def bench_view(model, data, cam, zoom=1.0):
    """Point the free (mouse-controllable) camera where the bench camera looks; `zoom`
    below 1 starts closer (the heart site is small)."""
    cid = model.camera("bench").id
    position = data.cam_xpos[cid]
    forward = -data.cam_xmat[cid].reshape(3, 3)[:, 2]
    distance = (WORK_TOP - position[2]) / forward[2]  # aim at the repair site
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = position + forward*distance
    cam.distance = distance*zoom
    cam.azimuth = np.degrees(np.arctan2(forward[1], forward[0]))
    cam.elevation = np.degrees(np.arcsin(forward[2]))


class HeldKeys:
    """Turn the viewer's press-only key events into continuous movement.

    MuJoCo's passive viewer never reports key releases, so a key counts as held
    while macOS says it is down, or while the OS keeps sending repeat presses.
    Movement only starts from a press inside the viewer window.
    """

    def __init__(self):
        self.is_down = _mac_key_state()
        self.keys = {}  # code -> [pressed_at, last_event_at]

    def press(self, code, now):
        if code in MOVES:
            if code in self.keys:
                self.keys[code][1] = now
            else:
                self.keys[code] = [now, now]

    def moves(self, now, dt):
        for code, (pressed_at, last_event) in list(self.keys.items()):
            down = (self.is_down is not None and self.is_down(code)) or now - last_event < .12
            if not down and now - pressed_at > .05:
                del self.keys[code]
            elif now - pressed_at >= HOLD_DELAY:
                yield code, MOVES[code][1] * MOVE_SPEED * dt


class Frames:
    def __init__(self):
        self.latest = None

    def read(self, stream):
        timestamp, positions = None, {}
        for line in stream:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8 or parts[1] != "joints_openxr" or parts[4] != "TRANSFORM":
                continue
            stamp, joint = parts[0], parts[2].removeprefix("xr_")
            if joint not in JOINTS:
                continue
            if stamp != timestamp:
                timestamp, positions = stamp, {}
            try:
                point = np.array([float(x) for x in parts[7].split(",")[:3]])
            except ValueError:
                continue
            if point.shape != (3,) or not np.isfinite(point).all():
                continue
            positions[joint] = point
            if positions.keys() >= JOINTS:
                self.latest = (time.monotonic(), positions.copy())


def preview_pose(phase):
    points = {"wrist": np.zeros(3), "palm": np.array([0, .048, 0])}
    for finger, x, length, spread in (
        ("thumb", .027, .097, .065), ("index", .030, .169, .016),
        ("middle", .010, .183, .003), ("ring", -.013, .172, -.007),
        ("little", -.033, .145, -.017),
    ):
        chain = CHAINS[finger]
        for i, joint in enumerate(chain):
            t = i / (len(chain) - 1)
            curl = (.5 + .5 * np.sin(phase - x * 22)) * max(t - .40, 0)**1.7
            points[joint] = np.array([x + spread*t, .014 + length*t - .095*curl, -.11*curl])
    return {joint: point + ANCHOR for joint, point in points.items()}


def hand_basis(points):
    """Right-handed (across, forward, back-of-hand) axes of a left hand, or None."""
    knuckles = np.array([points[CHAINS[name][1]] for name in ("index", "middle", "ring", "little")])
    forward = knuckles.mean(axis=0) - points["wrist"]
    across = knuckles[0] - knuckles[-1]
    normal = np.cross(across, forward)
    if np.linalg.norm(forward) < .025 or np.linalg.norm(normal) < 1e-6:
        return None
    forward /= np.linalg.norm(forward)
    normal /= np.linalg.norm(normal)
    return np.column_stack((np.cross(forward, normal), forward, normal))


def _yaw(angle):
    c, s_ = np.cos(angle), np.sin(angle)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


class Heading:
    """Yaw-only glove calibration.

    The Rokoko solver's pitch and roll are gravity-referenced, so they are used
    exactly as measured: tilt your real hand and the virtual one tilts the same
    way. Only the heading (yaw) has an arbitrary per-session zero, so it is the
    one thing calibrated: when the hand rests roughly flat and still for
    LOCK_TIME seconds, its averaged heading becomes "toward the screen".
    """
    LOCK_TIME = .5

    def __init__(self):
        self.reset()

    def reset(self):
        self.angle = None          # locked yaw correction, radians
        self.samples = []          # (time, heading) while the hand is flat and steady

    @property
    def locked(self):
        return self.angle is not None

    def rotation(self, points, now):
        basis = hand_basis(points)
        if basis is None:
            return None
        forward, normal = basis[:, 1], basis[:, 2]
        flat_length = np.linalg.norm(forward[:2])
        heading = np.arctan2(forward[1], forward[0])
        if self.angle is not None:
            return _yaw(self.angle)

        flat = flat_length > .8 and abs(normal[2]) > .75
        if flat and self.samples:
            mean = np.angle(np.mean([np.exp(1j*h) for _, h in self.samples]))
            if abs(np.angle(np.exp(1j*(heading - mean)))) > np.radians(12):
                self.samples.clear()      # the hand is still turning: start over
        if flat:
            self.samples.append((now, heading))
            if now - self.samples[0][0] >= self.LOCK_TIME:
                mean = np.angle(np.mean([np.exp(1j*h) for _, h in self.samples]))
                self.angle = np.pi/2 - mean
                self.samples.clear()
                return _yaw(self.angle)
        else:
            self.samples.clear()
        # Until locked, keep the fingers pointing into the scene; tilt is still live.
        return _yaw(np.pi/2 - heading if flat_length > .3 else 0.0)


def world_pose(points, heading, now):
    """Map glove joints (OpenXR xr_base: +X right, +Y up, -Z forward) into the
    scene (+X right, +Y away from the camera, +Z up), wrist pinned at ANCHOR."""
    wrist = points["wrist"]
    rebased = {}
    for name, position in points.items():
        x, y, z = position - wrist
        rebased[name] = np.array([x, -z, y])   # fixed +90 deg about X: Y-up -> Z-up
    rotation = heading.rotation(rebased, now)
    if rotation is None:
        return None
    return {key: ANCHOR + rotation @ value for key, value in rebased.items()}


def draw_hand(scene, points):
    scene.ngeom = 0
    if points is None:
        return

    def capsule(start, end, radius, color):
        draw_capsule(scene, start, end, radius, color)

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
    capsule(wrist - forward*.60, wrist - forward*.07, .031, "gown")
    capsule(wrist - forward*.095, wrist + forward*.02, .0255, GLOVE)
    add_capsule(scene, wrist - forward*.092, wrist - forward*.082, .0322, GLOVE,
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


GLOVE = "glove"
draw_geom = add_geom
draw_capsule = add_capsule


def draw_path(scene, vertices, radius, color):
    for start, end in zip(vertices, vertices[1:]):
        draw_capsule(scene, start, end, radius, color)


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
        draw_geom(scene, ellipsoid, [.0003, .0037, .0135], grip + d*.038 - up*.0004, basis, "blade")
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
        draw_capsule(scene, back - d*.002, back + d*.004, .0034, "satin")
        return

    # Ring-handled instruments: Mayo-Hegar needle holder (gold rings) or Metzenbaum scissors.
    holder = index == 1
    pivot = grip + d*(.030 if holder else .010)
    ring_material = "gold" if holder else "satin"
    for side in (-1, 1):
        centre = grip - d*.050 + side*across*.011
        _ring(scene, centre, .0078, across, d, ring_material)
        shank_start = centre + d*.0078
        draw_capsule(scene, shank_start, pivot - d*.004 + side*across*.0012, .0021, "polished")
        if holder:
            ratchet = grip - d*.036
            draw_capsule(scene, ratchet + side*across*.0072, ratchet + side*across*.0018, .0011, "satin")
            draw_capsule(scene, pivot + d*.004 + side*up*.0007, grip + d*.058 + side*up*.0004,
                         .0017, "polished")
        else:
            curve = up*.0016
            mid = pivot + d*.022 + side*across*.0012 + curve*.4
            tip = grip + d*TOOL_TIP[3] - side*across*.0018 + curve
            add_bar(scene, pivot, mid, .0048, .0011, across, "polished")
            add_bar(scene, mid, tip, .0026, .0010, across, "polished")
    draw_geom(scene, box, [.0035, .0026, .0050], pivot, basis, "satin")
    draw_capsule(scene, pivot - up*.0032, pivot + up*.0032, .0018, "polished", mj.mjtGeom.mjGEOM_CYLINDER)
    if holder:
        # Curved needle clamped near its midpoint, with the violet suture trailing.
        centre = grip + d*.059
        needle = [centre + d*(.025*np.sin(t)) + up*(.015*(np.cos(t)-1))
                  for t in np.linspace(-1.45, np.pi/2, 12)]
        draw_path(scene, needle, .00095, "polished")
        tail = needle[0]
        draw_path(scene, [tail, tail - across*.018 - up*.014,
                          tail - across*.034 - up*.019, tail - across*.05], .0006, "thread")


class CuttingStation:
    """Instrument handling, cutting and suturing on the tissue surface (via `tissue`),
    with the incision and stitch geometry of the chosen site."""

    def __init__(self, tissue):
        self.tissue = tissue
        self.site = tissue.site
        self.marks = deque(maxlen=450)
        self.stitches = []
        self.entry = None
        self.exit = None
        self.tissue_grip = None
        self.offset = np.zeros(3)
        self.rest = [(position.copy(), TRAY_DIRECTION.copy()) for position in TOOL_HOME]
        self.selected = 0
        self.held = None
        self.tool_pose = None
        self.pinch_finger = None
        self.pinching = False
        self.guide_tip = None
        self.guide_lower_steps = 0
        self.previous_tip = None
        self.depth = 0.0
        self.message = "2 (select needle holder) / G (pick up selected tool)"
        self.events = []                 # sound cues for the display loop
        self.blood = deque(maxlen=220)   # (position, born, final radius)
        self.clock = 0.0

    def park(self):
        if self.held is None:
            return
        grip, direction = self.tool_pose
        xy = np.clip(grip[:2], TABLE_MIN, TABLE_MAX)
        flat = np.array([direction[0], direction[1], 0.])
        length = np.linalg.norm(flat)
        flat = flat / length if length > .001 else TRAY_DIRECTION.copy()
        x, y = xy
        self.rest[self.held] = (np.array([x, y, self.tissue.rest_height(x, y) + .004]), flat)
        self.events.append("setdown")
        self.held = self.tool_pose = self.previous_tip = self.tissue_grip = None
        self.pinch_finger = None
        self.guide_tip = None
        self.guide_lower_steps = 0
        self.message = "G (pick up selected tool)"

    def update(self, pose, keys, held=()):
        if pose is None:
            self.park()
            self.pinching = False
            return None
        thumb = pose["thumb_tip"]
        fingers = ("index", "middle", "ring")
        closest = min(fingers, key=lambda name: np.linalg.norm(pose[f"{name}_tip"] - thumb))
        gap = np.linalg.norm(pose[f"{self.pinch_finger or closest}_tip"] - thumb)
        was_pinching = self.pinching
        self.pinching = gap < (.042 if was_pinching else .032)
        if self.pinch_finger and not self.pinching:
            self.park()

        for key in keys:
            self.key(key, pose)
        for code, amount in held:
            if not self.guided_lowering(code):
                self.move(code, amount, pose)

        if self.held is None and self.pinching and not was_pinching:
            point = self.grip(pose, closest) + self.offset
            nearest = min(range(4), key=lambda index: np.linalg.norm(self.rest[index][0]-point))
            if np.linalg.norm(self.rest[nearest][0] - point) <= PICKUP_RADIUS:
                self.pick_up(nearest, pose, closest)
        self.update_tool_pose(pose)
        if self.held == 0:
            grip, direction = self.tool_pose
            self.cut(grip + direction*TOOL_TIP[0])
        else:
            self.cut(None)
        return {name: point + self.offset for name, point in pose.items()}

    @staticmethod
    def grip(pose, finger):
        return (pose["thumb_tip"] + pose[f"{finger}_tip"]) / 2

    def pose_for_tool(self, pose):
        grip = self.grip(pose, self.pinch_finger or "index") + self.offset
        direction = pose["middle_proximal"] - pose["wrist"]
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        direction[2] -= .36
        return grip, direction / max(np.linalg.norm(direction), 1e-9)

    def update_tool_pose(self, pose):
        if self.held is None:
            return
        self.tool_pose = self.pose_for_tool(pose)
        if self.held == 1 and self.guide_tip is not None:
            # Keep the assisted needle tip at its target as the glove rotates.
            self.offset += self.guide_tip - self.active_tip()
            self.tool_pose = self.pose_for_tool(pose)

    def action_prompt(self):
        if self.held != 1:
            return self.message
        if self.exit is not None:
            return "T (tighten and tie stitch)"
        if self.guide_tip is None:
            return ("N (align over opposite edge)" if self.entry is not None
                    else "N (align needle for next stitch)")
        if self.guide_lower_steps:
            count = self.guide_lower_steps
            key = f"{count}xE" if count > 1 else "E"
            return f"{key} (lower needle to the {self.site.noun})"
        return "F (puncture opposite edge)" if self.entry is not None else "F (puncture first edge)"

    def active_tip(self):
        grip, direction = self.tool_pose
        tip = grip + direction*TOOL_TIP[self.held]
        if self.held == 1:
            _, up = instrument_axes(direction)
            tip -= up*.015
        return tip

    def pick_up(self, index, pose, finger=None):
        self.held = self.selected = index
        self.pinch_finger = finger
        self.offset[:] = self.rest[index][0] - self.grip(pose, finger or "index")
        self.tool_pose = self.pose_for_tool(pose)
        self.events.append("pickup")
        self.message = {
            0: "Arrows (move) / E (lower blade) / G (set down)",
            2: "N (align forceps) / F (grip edge)",
            3: "N (align scissors) / X (trim thread)",
        }.get(index, "N (align needle)")

    def move(self, code, amount, pose):
        axis = MOVES[code][0]
        self.update_tool_pose(pose)
        self.guide_tip = None
        self.guide_lower_steps = 0
        limit = .75 if axis < 2 else .45
        self.offset[axis] = np.clip(self.offset[axis]+amount, -limit, limit)

    def guided_lowering(self, code):
        return self.held == 1 and self.guide_tip is not None and code == ord("E")

    def key(self, code, pose):
        if code in MOVES:
            amount = MOVES[code][1]
            if self.guided_lowering(code):
                # Guided needle lowering stays in discrete taps.
                if self.guide_lower_steps:
                    self.guide_tip[2] += amount
                    self.guide_lower_steps -= 1
            else:
                self.move(code, amount, pose)
        elif ord("1") <= code <= ord("4"):
            self.selected = code - ord("1")
            self.message = f"G (pick up {TOOL_NAMES[self.selected]})"
        elif code == ord("G"):
            if self.held is not None:
                self.update_tool_pose(pose)
                self.park()
            else:
                self.pick_up(self.selected, pose)
        elif code in (ord("F"), ord("T"), ord("X")):
            self.update_tool_pose(pose)
            self.stitch(code)
        elif code == ord("N"):
            if self.held == 2:
                self.tool_pose = self.pose_for_tool(pose)
                target = self.tissue.point(self.site.point(*self.site.forceps_target), .003)
                self.offset += target - self.active_tip()
                self.message = "F (grip or release the wound edge)"
                return
            if self.held == 3 and self.stitches:
                self.tool_pose = self.pose_for_tool(pose)
                self.offset += self.stitches[-1][1] + np.array([0., 0., .008]) - self.active_tip()
                self.message = "X (trim last knot)"
                return
            if self.held != 1:
                self.message = "G (set down tool) / 2 then G (pick up needle holder)"
                return
            if self.exit is not None:
                return
            if self.entry is None:
                along = self.site.slot(len(self.stitches) % self.site.stitch_slots)
                side = -self.site.bite_guide
            else:
                along, entry_side = self.site.coords(self.entry)
                side = -self.site.bite_guide*np.sign(entry_side)
            surface = self.tissue.point(self.site.point(along, side))
            self.guide_tip = surface + np.array([0., 0., .004])
            self.guide_lower_steps = 2
        elif code == ord("C"):
            if self.held is not None:
                self.tool_pose = self.pose_for_tool(pose)
            self.park()
            self.offset.fill(0)
        elif code == ord("R"):
            self.marks.clear()
            self.stitches.clear()
            self.entry = self.exit = self.previous_tip = self.tissue_grip = None
            self.guide_tip = None
            self.guide_lower_steps = 0
            self.blood.clear()
            self.message = "G (pick up selected tool or set it down)"

    def stitch(self, action):
        if self.held == 2 and action == ord("F"):
            if self.tissue_grip is not None:
                self.tissue_grip = None
                self.message = "F (grip practice edge)"
                return
            tip = self.active_tip()
            surface = self.tissue.height(*tip[:2])
            along, side = self.site.coords(tip)
            if (surface is not None and abs(tip[2] - surface) <= .009
                    and abs(side) <= self.site.grip_side and abs(along) <= self.site.grip_along):
                self.tissue_grip = np.array([*tip[:2], surface])
                self.message = "Q (lift edge) / F (release)"
            else:
                self.message = "N (align forceps with the wound edge)"
            return

        if action == ord("X"):
            if self.held != 3 or not self.stitches:
                self.message = "4 then G (pick up scissors after tying)"
                return
            tip = self.active_tip()
            entry, exit, trimmed = self.stitches[-1]
            if np.linalg.norm(tip - exit) > .035:
                self.message = "N (align scissors) / X (trim thread)"
                return
            self.stitches[-1] = (entry, exit, True)
            self.events.append("trim")
            self.message = "G (set down scissors; thread trimmed)"
            return

        if self.held != 1:
            self.message = "G (set down tool) / 2 then G (pick up needle holder)"
            return
        if action == ord("T"):
            if self.exit is None:
                self.message = "Pass the needle through both sides first"
                return
            if len(self.stitches) >= 48:
                self.message = "Repair full - press R to reset"
                return
            self.stitches.append((self.entry, self.exit, False))
            self.events.append("tie")
            self.entry = self.exit = None
            self.guide_tip = None
            return

        if self.exit is not None:
            self.message = "Press T to tighten and tie"
            return
        if self.guide_tip is not None and self.guide_lower_steps:
            return
        tip = self.active_tip()
        if self.tissue.on_vessel(*tip[:2]):
            self.message = "Keep the needle off the coronary vessels"
            return
        surface = self.tissue.height(*tip[:2])
        if surface is None or not surface - .012 <= tip[2] <= surface + .001:
            self.message = f"Lower the needle tip to the {self.site.noun}"
            return
        point = self.tissue.point(tip[:2], .0015)
        along, side = self.site.coords(point)
        site = self.site
        if not (site.bite_min <= abs(side) <= site.bite_max and abs(along) <= site.along_limit):
            self.message = (f"Place the needle {site.bite_min*1000:.0f}-{site.bite_max*1000:.0f} mm "
                            "beside the wound")
            return
        if self.entry is None:
            self.entry = point
            self.guide_tip = None
            self.bleed(point, .0011)
            self.events.append("puncture")
            return
        entry_along, entry_side = self.site.coords(self.entry)
        if side*entry_side < 0 and abs(along - entry_along) <= site.exit_tolerance:
            self.exit = point
            self.bleed(point, .0011)
            self.events.append("puncture")
            self.message = "Press T to tighten and tie"
        else:
            self.message = "Place the exit straight across from the entry"

    def cut(self, tip):
        self.depth = 0.0
        if tip is None:
            self.previous_tip = None
            return
        surface = self.tissue.height(*tip[:2])
        if surface is None or tip[2] > surface:
            if self.held == 0 and self.tissue.on_vessel(*tip[:2]):
                vessel_top = self.tissue.vessel_top(*tip[:2])
                if tip[2] <= vessel_top:
                    self.message = "Careful - blade on a coronary vessel. Lift (Q) and move away"
            self.previous_tip = None
            return
        self.depth = min(MAX_DEPTH, surface - tip[2])
        normal = self.tissue.normal(*tip[:2])
        point = np.array([*tip[:2], surface]) + normal*.0007
        if self.previous_tip is not None:
            distance = np.linalg.norm(point-self.previous_tip)
            if .0012 <= distance <= .025:
                self.marks.append(self.wound_segment(self.previous_tip, point, self.depth, normal))
                # Deeper cuts ooze: drops appear along the wound and slowly spread.
                if self.depth > .0012 and np.random.random() < .18 + 40*self.depth:
                    along = (point - self.previous_tip) / distance
                    beside = np.cross(along, normal)*np.random.uniform(-.0015, .0015)
                    self.bleed(point + beside, np.random.uniform(.0014, .0018) + .35*self.depth)
            elif distance < .0012:
                return
        self.previous_tip = point

    def wound_segment(self, start, end, depth, normal):
        """Flat ellipses lying on the tissue surface for one cut segment: everted edges
        around a wet dark gap that widens with depth."""
        radius = .0006 + .24*depth
        lip = surface_strip(start - normal*.0004, end - normal*.0004, radius*2.0, .0003, normal,
                            self.site.lip_material)
        gap = surface_strip(start, end, radius*.85, .0004, normal, "wound")
        return lip, gap

    def bleed(self, point, radius):
        xy = point[:2]
        surface = self.tissue.point(xy, .0004)
        if surface is None:
            return
        _, _, matrix, _ = surface_strip(surface, surface, 0, 0, self.tissue.normal(*xy), "blood")
        self.blood.append((surface, matrix, self.clock, radius))

    def draw(self, scene, now=None):
        if now is not None:
            self.clock = now
        for index, pose in enumerate(self.rest):
            draw_instrument(scene, index, *(self.tool_pose if index == self.held else pose))
        if self.tissue_grip is not None and self.held == 2:
            tip = self.active_tip()
            displacement = np.clip(tip - self.tissue_grip, -.024, .024)
            draw_capsule(scene, self.tissue_grip, self.tissue_grip + displacement, .003, "lip")
        ellipse = mj.mjtGeom.mjGEOM_ELLIPSOID
        for size, pos, matrix, material in self.tissue.wound:
            draw_geom(scene, ellipse, size, pos, matrix, material)
        for segment in self.marks:
            for size, pos, matrix, material in segment:
                draw_geom(scene, ellipse, size, pos, matrix, material)
        for position, matrix, born, radius in self.blood:
            grow = min(1.0, max(self.clock - born, 0) / 2.5)**.5
            r = radius*(.35 + .65*grow)
            draw_geom(scene, ellipse, [r, r*.9, r*.28], position, matrix, "blood")
        for entry, exit, trimmed in self.stitches:
            self.draw_thread(scene, entry, exit, True, trimmed)
        if self.entry is not None:
            end = self.exit
            if end is None and self.held == 1:
                end = self.active_tip()
            if end is not None:
                self.draw_thread(scene, self.entry, end, False, False)

    def draw_thread(self, scene, entry, exit, tied, trimmed):
        mid = (entry+exit)/2 + np.array([0., 0., .003 if tied else .013])
        draw_path(scene, [entry, (entry+mid)/2, mid, (mid+exit)/2, exit], .0008, "thread")
        if tied:
            knot = exit + np.array([0., 0., .001])
            draw_geom(scene, mj.mjtGeom.mjGEOM_ELLIPSOID, [.0019, .0014, .0012], knot, np.eye(3), "thread")
            length = self.site.tail_length*(.3 if trimmed else 1)
            for spread in (.6, -.2):
                draw_capsule(scene, knot, knot + np.array([length, length*spread, .0004]), .0006, "thread")


SWITCH_SITE = ord("P")


def draw_user(scene, points, station, now):
    """Everything that moves: glove, instruments, wounds, sutures and the patient monitor."""
    draw_hand(scene, points)
    station.draw(scene, now)
    draw_monitor(scene, now)


def display(frames, preview, site):
    model = build_model()
    Look.model = model
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    switch = SiteSwitch(model, data, site.name)
    heading = Heading()
    last_status = None
    keys = SimpleQueue()
    station = CuttingStation(make_tissue(model, data, site))
    held_keys = HeldKeys()
    sounds = Sounds()
    last_frame = time.monotonic()
    print("1 scalpel | 2 needle holder | 3 forceps | 4 scissors | G pick up/set down\n"
          "Arrows move, Q/E raise/lower (tap for a small step, hold to glide).\n"
          "N aligns tools, F punctures or grips tissue, T ties, X trims. R clears the repair.\n"
          "View: scroll or +/- to zoom, left-drag to orbit, right-drag to pan,\n"
          "      double-right-click a spot (tissue, tray, room) to zoom around it, 0 resets the view.\n"
          "H re-aims the glove: press H, then rest your hand flat with fingers toward the screen for a moment.\n"
          "M mutes or unmutes the OR sounds.\n"
          "Switch site: Skin / Heart buttons in the Equality panel on the right, or press P.",
          flush=True)

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        with viewer.lock():
            show_site(model, viewer.opt, site)
            bench_view(model, data, viewer.cam, site.zoom)
        overlay = getattr(viewer, "set_texts", None)
        next_overlay = 0.0
        while viewer.is_running():
            now = time.monotonic()
            dt = min(now - last_frame, .1)
            last_frame = now
            pressed = []
            while not keys.empty():
                code = keys.get_nowait()
                if code in MOVES and code in held_keys.keys:
                    held_keys.press(code, now)  # OS auto-repeat: continuous motion handles it
                    continue
                held_keys.press(code, now)
                pressed.append(code)
            held = list(held_keys.moves(now, dt))

            # Site switching: the Skin / Heart buttons in the viewer's Equality panel, or P.
            with viewer.lock():
                choice = switch.poll()
                if SWITCH_SITE in pressed:
                    choice = "heart" if station.site.name == "skin" else "skin"
                    switch.select(choice)
            pressed = [code for code in pressed if code != SWITCH_SITE]
            if choice is not None and choice != station.site.name:
                site = SITES[choice]
                station.park()
                station = CuttingStation(make_tissue(model, data, site))
                with viewer.lock():
                    show_site(model, viewer.opt, site)
                    bench_view(model, data, viewer.cam, site.zoom)
                sounds.play("switch")
                print(f"Operating site: {site.name}", flush=True)

            if ord("M") in pressed:
                sounds.muted = not sounds.muted
                pressed = [code for code in pressed if code != ord("M")]
            zoom = [code for code in pressed if code in ZOOM_IN | ZOOM_OUT | {RESET_VIEW}]
            if zoom:
                pressed = [code for code in pressed if code not in zoom]
                with viewer.lock():
                    for code in zoom:
                        if code == RESET_VIEW:
                            bench_view(model, data, viewer.cam, site.zoom)
                        else:
                            if viewer.cam.type != mj.mjtCamera.mjCAMERA_FREE:
                                bench_view(model, data, viewer.cam, site.zoom)
                            factor = .8 if code in ZOOM_IN else 1.25
                            viewer.cam.distance = float(np.clip(viewer.cam.distance*factor, .03, 6))

            sample = frames.latest
            if preview:
                status, points = "Keyboard practice", preview_pose(-1.2)
            elif sample is None or now - sample[0] > .5:
                status, points = "Waiting for the Rokoko hand stream", None
            else:
                if ord("H") in pressed:
                    heading.reset()
                    pressed = [key for key in pressed if key != ord("H")]
                points = world_pose(sample[1], heading, now)
                status = ("Rokoko live" if heading.locked
                          else "Rest your hand flat, fingers toward the screen")

            points = station.update(points, pressed, held)
            with viewer.lock():
                draw_user(viewer.user_scn, points, station, now)
            for event in station.events:
                sounds.play(event)
            station.events.clear()
            sounds.heartbeat(now)

            if status != last_status:
                print(status, flush=True)
                last_status = status
            if overlay is not None and now >= next_overlay:
                tool_status = (TOOL_NAMES[station.selected] + " selected"
                               if station.held is None else TOOL_NAMES[station.held] + " held")
                overlay((int(mj.mjtFont.mjFONT_NORMAL), int(mj.mjtGridPos.mjGRID_BOTTOMLEFT),
                         "SURGICAL TRAINING\nSite\nInstrument\nStitches / depth\nAction",
                         f"{status}\n{station.site.name.capitalize()}  (switch: Equality panel "
                         f"on the right, or P)\n{tool_status}\n{len(station.stitches)} / "
                         f"{station.depth*1000:.1f} mm\n{station.action_prompt()}"))
                next_overlay = now + .10
            viewer.sync()
            time.sleep(max(0, 1/60 - (time.monotonic() - now)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Practice cutting with the keyboard and no glove")
    parser.add_argument("--site", choices=sorted(SITES), default="skin",
                        help="What to operate on: the skin pad (default) or the heart")
    args = parser.parse_args()
    site = SITES[args.site]
    print(f"Operating site: {site.name}", flush=True)
    frames = Frames()
    if args.preview:
        display(frames, True, site)
        return
    stream_script = SDK / "rgmp_stream.py"
    if not stream_script.is_file():
        parser.error(f"Missing Rokoko SDK example: {stream_script}")
    print("Connect only the left glove. Tilt and roll follow your hand directly; to set the\n"
          "direction, rest your hand flat with fingers toward the screen for half a second.\n"
          "Press H any time to set the direction again.", flush=True)
    with subprocess.Popen(
        [sys.executable, "-u", str(stream_script), "--port", "12277",
         "--group-name", "joints_openxr", "--measure-type", "TRANSFORM"],
        stdout=subprocess.PIPE, text=True,
    ) as process:
        threading.Thread(target=frames.read, args=(process.stdout,), daemon=True).start()
        try:
            display(frames, False, site)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
