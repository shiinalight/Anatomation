"""Practice incisions and sutures with both hands, on skin or on a heart, in a glove-controlled
MuJoCo operating room."""

import argparse
from collections import deque
import json
from queue import SimpleQueue
import re
import socket
import struct
import sys
import threading
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

from aiven_store import Telemetry, TrainingStore
from bypass import STITCHES_PER_ANASTOMOSIS
import motion_log
from procedures import DEFAULT, PROCEDURES, from_site, make_procedure
from or_scene import (SITES, TABLE_MAX, TABLE_MIN, TRAY_TOP, WORK_TOP, Look, Sounds, add_bar,
                      add_capsule, add_geom, build_model, draw_monitor, make_tissue, show_site,
                      surface_strip, view_groups, ASSISTANT_BUTTON, AUTOPILOT_BUTTON, CLEAR_BUTTON,
                      BOTH_HANDS_BUTTON, GUIDE_BUTTON, PROCEDURES_BUTTON, RESET_HANDS_BUTTON,
                      REVIEW_BUTTON, VOICE_BUTTON,
                      PanelButton,
                      PanelToggle)

GLOVE_PORT = 12277     # rkk-hand-solver's RGMP v2 output (both gloves, one device each)
SIDES = ("left", "right")
OTHER = {"left": "right", "right": "left"}
ANCHOR = np.array([0, -0.10, 1.015])
# Where each wrist is pinned before the keyboard moves it (the gloves track rotation only).
HAND_ANCHOR = {"left": ANCHOR + [-.13, 0, 0], "right": ANCHOR + [.13, 0, 0]}
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
SWITCH_HAND = ord("Z")    # keyboard control moves to the other hand (it also follows the steps)
RESET_HANDS = 0x10000     # from the panel's Reset hands button (a code no keyboard key sends)
BOTH_HANDS = ord("Y")     # toggles moving both hands together (also the Both hands button)
SELECT_HAND = {"left": 0x10002, "right": 0x10003}   # double-click on a hand in the viewer
PULL_HEIGHT = .009        # forceps lift that pulls the suture through after the needle exits
THROWS = ("double wrap", "single wrap, reversed", "single wrap, squares the knot")
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


_RGMP_SIZES = {"INT32": 4, "UINT32": 4, "INT64": 8, "UINT64": 8, "FLOAT": 4, "DOUBLE": 8}


def _rgmp_size(data_type):
    m = re.match(r"^(INT32|UINT32|INT64|UINT64|FLOAT|DOUBLE)(?:\[(\d+)(?:,\s*(\d+))?\])?$",
                 data_type.strip())
    if not m:
        raise ValueError(f"unknown RGMP data type {data_type!r}")
    return _RGMP_SIZES[m.group(1)]*int(m.group(2) or 1)*int(m.group(3) or 1)


class GloveStream:
    """Both Smartgloves from rkk-hand-solver (RGMP v2 over TCP).

    The solver announces each glove as its own device, with `device_info.hand` naming the
    side, so the joints of each hand are kept apart by device id. `latest[side]` is
    (receive time, {joint: xr_base position}).
    """

    def __init__(self, port=GLOVE_PORT):
        self.port = port
        self.latest = {}
        self.devices = {}    # device id -> (side, openxr group index, [(byte offset, joint)])

    def run(self):
        while True:
            try:
                self._session()
            except (OSError, ValueError, KeyError, struct.error, json.JSONDecodeError):
                pass
            time.sleep(1)

    def _session(self):
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.settimeout(10)
            self.devices.clear()

            def receive(n):
                buf = bytearray()
                while len(buf) < n:
                    chunk = sock.recv(n - len(buf))
                    if not chunk:
                        raise ConnectionError("glove stream closed")
                    buf += chunk
                return bytes(buf)

            while True:
                kind, length = struct.unpack("<II", receive(8))
                payload = receive(length)
                if kind == 1:
                    self._define(json.loads(payload))
                elif kind == 2:
                    self._data(payload)
                elif kind == 3 and len(payload) >= 4:
                    self.devices.pop(struct.unpack_from("<I", payload)[0], None)

    def _define(self, definition):
        groups = definition.get("groups", [])
        side = (definition.get("device_info") or {}).get("hand")
        if side not in SIDES:
            frames = [s.get("target_frame", "") for g in groups for s in g.get("streams", [])]
            side = next((name for name in SIDES if any(f.startswith(name + "_") for f in frames)), None)
        if side is None:
            return
        for index, group in enumerate(groups):
            if group.get("name") != "joints_openxr":
                continue
            offset, joints = 0, []
            for stream in group.get("streams", []):
                size = _rgmp_size(stream["data_type"])
                name = stream.get("target_frame", "").removeprefix("xr_")
                if (name in JOINTS and stream.get("measure_type") == "TRANSFORM"
                        and stream["data_type"].startswith("FLOAT") and size >= 12):
                    joints.append((offset, name))
                offset += size
            self.devices[int(definition["device_id"])] = (side, index, joints)

    def _data(self, payload):
        device, group, _ = struct.unpack_from("<IIQ", payload)
        entry = self.devices.get(device)
        if entry is None or group != entry[1]:
            return
        side, _, joints = entry
        positions = {}
        for offset, name in joints:
            point = np.array(struct.unpack_from("<3f", payload, 16 + offset))
            if np.isfinite(point).all():
                positions[name] = point
        if positions.keys() >= JOINTS:
            self.latest[side] = (time.monotonic(), positions)


ZOOM_IN = {ord("="), 334}    # = (or +) and keypad +
ZOOM_OUT = {ord("-"), 333}   # - and keypad -
RESET_VIEW = ord("0")


# The surgeon's view of the table. Not a camera in the model: MuJoCo's viewer can draw
# model cameras as markers (its Q shortcut), which would float in the scene.
BENCH_POS = np.array([.28, -.62, 1.43])
BENCH_FORWARD = np.cross([-.202, .492, .847], [.925, .380, 0])   # -(x axis cross y axis)
BENCH_FORWARD /= np.linalg.norm(BENCH_FORWARD)


def bench_view(model, data, cam, zoom=1.0, focus=None):
    """Point the free (mouse-controllable) camera at the repair site from the surgeon's
    position; `zoom` below 1 starts closer (the heart site is small)."""
    position, forward = BENCH_POS, BENCH_FORWARD
    distance = (WORK_TOP - position[2]) / forward[2]  # aim at the repair site
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = position + forward*distance
    if focus is not None:
        cam.lookat[:2] = focus
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


def preview_pose(phase, side="left"):
    """A relaxed hand for keyboard practice (and for a glove that is not streaming)."""
    mirror = -1 if side == "right" else 1
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
            points[joint] = np.array([mirror*(x + spread*t), .014 + length*t - .095*curl, -.11*curl])
    return {joint: point + HAND_ANCHOR[side] for joint, point in points.items()}


def hand_basis(points):
    """(across, forward, palm normal) axes of a hand, or None."""
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


def world_pose(points, heading, now, anchor):
    """Map glove joints (OpenXR xr_base: +X right, +Y up, -Z forward) into the
    scene (+X right, +Y away from the camera, +Z up), wrist pinned at `anchor`."""
    wrist = points["wrist"]
    rebased = {}
    for name, position in points.items():
        x, y, z = position - wrist
        rebased[name] = np.array([x, -z, y])   # fixed +90 deg about X: Y-up -> Z-up
    rotation = heading.rotation(rebased, now)
    if rotation is None:
        return None
    return {key: anchor + rotation @ value for key, value in rebased.items()}


def draw_hand(scene, points, active=False):
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


UP = np.array([0., 0., 1.])


class Hand:
    """One gloved hand: its keyboard offset, the tool it holds and what it is holding on to."""

    def __init__(self, side):
        self.side = side
        self.label = side.capitalize()
        self.heading = Heading()
        self.pose = None          # glove joints in the scene, before `offset`
        self.offset = np.zeros(3)
        self.selected = 2 if side == "left" else 1
        self.held = None
        self.tool_pose = None
        self.pinch_finger = None
        self.pinching = False
        self.guide_tip = None     # assisted needle tip target (N)
        self.guide_lower_steps = 0
        self.previous_tip = None
        self.depth = 0.0
        self.cut_line = None      # marked incision the scalpel is guided along (bypass)
        self.grip = None          # tissue point held by the forceps or pressed by the fingers
        self.grip_pass = None     # which needle pass the grip was taken for (see `passes`)

    @property
    def tool(self):
        return "empty" if self.held is None else TOOL_NAMES[self.held].lower()


class CuttingStation:
    """Both hands' instruments, cutting and suturing on the tissue surface (via `tissue`),
    with the incision and stitch geometry of the chosen site.

    The technique follows theatre practice. The right hand cuts and drives the needle, and
    the left hand works against it:
      - cutting: the left hand steadies the tissue next to the blade (fingers stretching the
        skin, or forceps holding the vessel wall); the blade does not open unsteadied tissue;
      - each needle pass: the forceps first lift the edge the needle goes through;
      - after the exit: the forceps grasp the needle tip and pull the suture through (Q);
      - knot: instrument tie with the needle holder, three throws (T).
    Any tool can be taken by either hand; the prompts name the usual one. Keyboard control
    follows whichever hand the next step needs, and Z swaps it by hand.
    """

    def __init__(self, tissue, task=None):
        self.tissue = tissue
        self.site = tissue.site
        self.task = task or ("bypass" if self.site.procedure == "cabg" else "suture")
        self.procedure = make_procedure(self.task, tissue)
        self.hands = {side: Hand(side) for side in SIDES}
        self.active = "right"
        self.both = False             # arrows and Q/E move both hands together
        self.need = None              # hand the current step needs (keyboard follows it)
        self.marks = deque(maxlen=450)
        self.stitches = []
        self.entry = None
        self.exit = None
        self.passes = 0               # needle passes so far: each needs a fresh forceps grip
        self.needle_grasp = None      # forceps tip height when it took the needle at the exit
        self.pulled = False
        self.throws = 0
        self.rest = [(position.copy(), TRAY_DIRECTION.copy()) for position in TOOL_HOME]
        self._message, self._message_at = "", -1e9
        self.log = []                    # (kind, detail) of each step, for recording and scoring
        self.events = []                 # sound cues for the display loop
        self.blood = deque(maxlen=220)   # (position, born, final radius)
        self.clock = 0.0
        reach = .004 if self.procedure is not None else max(.004, self.site.bite_guide*.9)
        self.edge_reach = reach                                   # forceps grip to needle pass
        self.steady_reach = max(.012, self.site.half*.8)          # steadying hand to blade

    # --- small helpers ---------------------------------------------------------------
    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, text):
        """A warning or a refused action (logged as rejected). See also inform()."""
        repeat = text == self._message
        self._message, self._message_at = text, time.monotonic()
        if text and not repeat:      # a warning that stays up (blade on a vessel) counts once
            self.note("rejected", reason=text)

    def inform(self, text):
        self._message, self._message_at = text, time.monotonic()

    def note(self, kind, **detail):
        self.log.append((kind, detail))

    def moving_hands(self):
        """The hands the arrows and Q/E move: the active one, or both."""
        hands = self.hands.values() if self.both else [self.hands[self.active]]
        return [hand for hand in hands if hand.pose is not None]

    def other(self, hand):
        return self.hands[OTHER[hand.side]]

    def holder(self, index):
        return next((hand for hand in self.hands.values() if hand.held == index), None)

    @property
    def depth(self):
        return max(hand.depth for hand in self.hands.values())

    def surface(self, xy, lift=0.0):
        point = self.tissue.point(xy[:2], lift)
        if point is None and hasattr(self.tissue, "any_point"):
            point = self.tissue.any_point(xy[:2], lift)
        return point

    def surface_height(self, x, y):
        z = self.tissue.height(x, y)
        if z is None and hasattr(self.tissue, "any_height"):
            z = self.tissue.any_height(x, y)
        return z

    @staticmethod
    def grip_point(pose, finger):
        return (pose["thumb_tip"] + pose[f"{finger}_tip"]) / 2

    def fingertip(self, hand):
        """Where an empty hand touches the tissue: between thumb and index tips."""
        return self.grip_point(hand.pose, "index") + hand.offset

    def contact(self, hand):
        return self.fingertip(hand) if hand.held is None else self.active_tip(hand)

    # --- holding and moving tools ------------------------------------------------------
    def park(self, hand):
        if hand.held is not None:
            grip, direction = hand.tool_pose
            xy = np.clip(grip[:2], TABLE_MIN, TABLE_MAX)
            flat = np.array([direction[0], direction[1], 0.])
            length = np.linalg.norm(flat)
            flat = flat / length if length > .001 else TRAY_DIRECTION.copy()
            x, y = xy
            self.rest[hand.held] = (np.array([x, y, self.tissue.rest_height(x, y) + .004]), flat)
            self.events.append("setdown")
            if hand.held == 2 and self.needle_grasp is not None and self.throws == 0:
                self.needle_grasp, self.pulled = None, False   # the needle slips back
            self.note("setdown", hand=hand.side, tool=hand.held)
            self.inform(f"{hand.label} hand: G (pick up selected tool)")
        hand.held = hand.tool_pose = hand.previous_tip = hand.grip = None
        hand.cut_line = hand.pinch_finger = hand.guide_tip = None
        hand.guide_lower_steps = 0
        hand.depth = 0.0

    def park_all(self):
        for hand in self.hands.values():
            if hand.held is not None and hand.pose is not None:
                hand.tool_pose = self.pose_for_tool(hand)
            self.park(hand)

    def update(self, poses, keys, held=()):
        """`poses`: side -> scene joints (or None); returns side -> drawn joints."""
        for side, hand in self.hands.items():
            pose = poses.get(side)
            hand.pose = pose
            if pose is None:
                self.park(hand)
                hand.pinching = False
                continue
            thumb = pose["thumb_tip"]
            closest = min(("index", "middle", "ring"),
                          key=lambda name: np.linalg.norm(pose[f"{name}_tip"] - thumb))
            hand.closest = closest
            gap = np.linalg.norm(pose[f"{hand.pinch_finger or closest}_tip"] - thumb)
            hand.was_pinching = hand.pinching
            hand.pinching = gap < (.042 if hand.was_pinching else .032)
            if hand.pinch_finger and not hand.pinching:
                self.park(hand)

        for key in keys:
            self.key(key)
        for moving in self.moving_hands():
            for code, amount in held:
                if not self.guided_lowering(moving, code):
                    self.move(moving, code, amount)

        for hand in self.hands.values():
            if hand.pose is None:
                continue
            if hand.held is None and hand.pinching and not hand.was_pinching:
                point = self.grip_point(hand.pose, hand.closest) + hand.offset
                free = [i for i in range(4) if self.holder(i) is None]
                if free:
                    nearest = min(free, key=lambda i: np.linalg.norm(self.rest[i][0] - point))
                    if np.linalg.norm(self.rest[nearest][0] - point) <= PICKUP_RADIUS:
                        self.pick_up(hand, nearest, hand.closest)
            self.update_tool_pose(hand)
            if hand.grip is not None:   # a grip lets go when the hand moves away from it
                gap = self.contact(hand) - hand.grip
                if np.linalg.norm(gap[:2]) > .03 or gap[2] > .045:
                    hand.grip = None
            if hand.held == 0:
                grip, direction = hand.tool_pose
                self.cut(hand, grip + direction*TOOL_TIP[0])
            else:
                self.cut(hand, None)
        self.check_pull()

        side, _ = self.plan()
        if side != self.need:          # keyboard control follows the step's hand
            self.need = side
            if side is not None:
                self.active = side
        return {side: None if hand.pose is None else
                {name: point + hand.offset for name, point in hand.pose.items()}
                for side, hand in self.hands.items()}

    def pose_for_tool(self, hand):
        grip = self.grip_point(hand.pose, hand.pinch_finger or "index") + hand.offset
        direction = hand.pose["middle_proximal"] - hand.pose["wrist"]
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        direction[2] -= .36
        return grip, direction / max(np.linalg.norm(direction), 1e-9)

    def update_tool_pose(self, hand):
        if hand.held is None or hand.pose is None:
            return
        hand.tool_pose = self.pose_for_tool(hand)
        if hand.held == 1 and hand.guide_tip is not None:
            # Keep the assisted needle tip at its target as the glove rotates.
            hand.offset += hand.guide_tip - self.active_tip(hand)
            hand.tool_pose = self.pose_for_tool(hand)

    def active_tip(self, hand):
        grip, direction = hand.tool_pose
        tip = grip + direction*TOOL_TIP[hand.held]
        if hand.held == 1:
            _, up = instrument_axes(direction)
            tip -= up*.015
        return tip

    def pick_up(self, hand, index, finger=None):
        owner = self.holder(index)
        if owner is not None:
            self.message = f"The {owner.side} hand holds the {TOOL_NAMES[index].lower()}"
            return
        hand.held = hand.selected = index
        hand.pinch_finger = finger
        hand.grip = None
        hand.offset[:] = self.rest[index][0] - self.grip_point(hand.pose, finger or "index")
        hand.tool_pose = self.pose_for_tool(hand)
        self.events.append("pickup")
        self.note("pickup", hand=hand.side, tool=index)
        self.inform("")

    def move(self, hand, code, amount):
        axis = MOVES[code][0]
        self.update_tool_pose(hand)
        if hand.cut_line is not None and hand.held == 0 and axis != 1:
            if axis == 0:
                self.slide_along_cut(hand, amount)
            else:
                hand.offset[2] += amount
            return
        hand.cut_line = None
        hand.guide_tip = None
        hand.guide_lower_steps = 0
        limit = .75 if axis < 2 else .45
        hand.offset[axis] = np.clip(hand.offset[axis] + amount, -limit, limit)

    def slide_along_cut(self, hand, amount):
        """Move the blade along the marked line, keeping its depth below the surface."""
        line, tip = hand.cut_line, self.active_tip(hand)
        along, _ = line.coords(tip)
        below = (self.tissue.any_height(*tip[:2]) or tip[2]) - tip[2]
        along = float(np.clip(along + amount*.5, -line.half - .003, line.half + .003))
        xy = line.point(along)
        target = np.array([*xy, self.tissue.any_height(*xy) - below])
        hand.offset += target - tip

    @staticmethod
    def guided_lowering(hand, code):
        return hand.held == 1 and hand.guide_tip is not None and code == ord("E")

    # --- keys ----------------------------------------------------------------------------
    def key(self, code):
        if code == SWITCH_HAND:
            self.active = OTHER[self.active]
            self.inform(f"Keyboard moves the {self.active} hand")
            return
        hand = self.hands[self.active]
        if code == ord("R"):
            self.reset()
            return
        if code == RESET_HANDS:
            self.reset_hands()
            return
        if code in SELECT_HAND.values():
            self.active = next(side for side, c in SELECT_HAND.items() if c == code)
            self.both = False
            self.inform(f"Arrows and Q/E move the {self.active} hand")
            return
        if code == BOTH_HANDS:
            self.both = not self.both
            self.inform("Arrows and Q/E move both hands" if self.both else
                        f"Arrows and Q/E move the {self.active} hand (Z swaps)")
            return
        if hand.pose is None:
            return
        if code in MOVES:
            for moving in self.moving_hands():
                if self.guided_lowering(moving, code):
                    # Guided needle lowering stays in discrete taps.
                    if moving.guide_lower_steps:
                        moving.guide_tip[2] += MOVES[code][1]
                        moving.guide_lower_steps -= 1
                else:
                    self.move(moving, code, MOVES[code][1])
        elif ord("1") <= code <= ord("4"):
            hand.selected = code - ord("1")
            self.inform(f"{hand.label} hand: G (pick up {TOOL_NAMES[hand.selected].lower()})")
        elif code == ord("G"):
            if hand.held is not None:
                self.update_tool_pose(hand)
                self.park(hand)
            else:
                self.pick_up(hand, hand.selected)
        elif code == ord("F"):
            self.update_tool_pose(hand)
            self.act(hand)
        elif code == ord("T"):
            self.throw(hand)
        elif code == ord("X"):
            self.update_tool_pose(hand)
            self.trim(hand)
        elif code == ord("N"):
            self.update_tool_pose(hand)
            self.align(hand)
        elif code == ord("C"):
            self.park(hand)
            hand.offset.fill(0)

    def reset_hands(self):
        """Both hands back to where they start, in front of the surgeon, keeping their tools."""
        for hand in self.hands.values():
            hand.offset.fill(0)
            hand.grip = hand.guide_tip = hand.cut_line = hand.previous_tip = None
            hand.guide_lower_steps = 0
            hand.depth = 0.0
            if hand.held is not None and hand.pose is not None:
                hand.tool_pose = self.pose_for_tool(hand)
        if self.needle_grasp is not None and not self.pulled:
            self.needle_grasp = None       # the forceps let go of the needle
        self.note("reset_hands")
        self.inform("Hands back in the surgeon's view")

    def reset(self):
        self.marks.clear()
        self.stitches.clear()
        self.blood.clear()
        self.entry = self.exit = self.needle_grasp = None
        self.pulled, self.throws = False, 0
        for hand in self.hands.values():
            hand.previous_tip = hand.grip = hand.guide_tip = hand.cut_line = None
            hand.guide_lower_steps = 0
        if self.procedure is not None:
            self.procedure = make_procedure(self.task, self.tissue)
        self.note("reset")
        self.inform("Repair cleared")

    # --- targets ------------------------------------------------------------------------
    def cutting_step(self):
        """True when the step is a cut: a bypass cut, or a free cut with the scalpel held."""
        if self.procedure is not None:
            return self.procedure.cutting()
        return self.holder(0) is not None

    def needle_target(self):
        """Surface point of the next needle pass (guided placement), or None."""
        if self.procedure is not None:
            return self.procedure.needle_target(self)
        if self.exit is not None:
            return None
        site = self.site
        if self.entry is None:
            along = site.slot(len(self.stitches) % site.stitch_slots)
            side = -site.bite_guide
        else:
            along, entry_side = site.coords(self.entry)
            side = -site.bite_guide*np.sign(entry_side)
        return self.tissue.point(site.point(along, side))

    def edge_point(self, target):
        """Where the forceps hold the edge for a needle pass at `target`."""
        if self.procedure is not None:
            return self.procedure.edge_point(target)
        along, side = self.site.coords(target)
        return self.tissue.point(self.site.point(along, side*.45))

    def steady_point(self, hand):
        """Where `hand` steadies the tissue for the other hand's blade."""
        if self.procedure is not None and self.procedure.cutting():
            return self.procedure.steady_point()
        blade = self.holder(0)
        if blade is not None and blade is not hand and blade.tool_pose is not None:
            tip = self.active_tip(blade)
            reach = min(.025, self.steady_reach*.5)
            for sign in (-1, 1):
                point = self.surface(tip[:2] + self.site.across*reach*sign)
                if point is not None:
                    return point
        return self.surface(self.site.point(0, -self.site.bite_guide*2))

    def edge_held(self, forceps, target):
        """True if `forceps` hold the edge for the needle pass at `target` (taken for this pass)."""
        if forceps is None or forceps.held != 2 or forceps.grip is None or target is None:
            return False
        if forceps.grip_pass != self.passes:
            return False
        if np.linalg.norm(forceps.grip[:2] - target[:2]) > self.edge_reach:
            return False
        if self.procedure is None:
            return self.site.coords(forceps.grip)[1]*self.site.coords(target)[1] > 0
        return True

    def steadied(self, blade):
        """True if the other hand steadies the tissue where `blade` cuts. In a bypass the
        forceps must hold the vessel wall beside the marked line."""
        other = self.other(blade)
        if other.grip is None:
            return False
        if self.procedure is not None and self.procedure.cutting():
            return self.procedure.steady_ok(other)
        if blade.tool_pose is None:
            return False
        tip = self.active_tip(blade)
        return np.linalg.norm(other.grip[:2] - tip[:2]) <= self.steady_reach

    # --- actions -----------------------------------------------------------------------
    def align(self, hand):
        """N: put the hand's tool (or fingertips) where the next step needs it."""
        if hand.held is None:
            target = self.steady_point(hand)
            if target is None:
                return
            hand.grip = None
            hand.offset += target + UP*.006 - self.fingertip(hand)
            self.inform("")
            return
        if hand.held == 0:
            if self.procedure is not None and self.procedure.align_blade(self, hand):
                return
            self.inform("Arrows move, E lowers the blade")
            return
        if hand.held == 3:
            if self.stitches:
                hand.offset += self.stitches[-1][1] + UP*.008 - self.active_tip(hand)
                self.inform("")
            return
        if hand.held == 2:
            hand.grip = None
            self.inform("")
            if self.exit is not None and self.needle_grasp is None:
                target = self.exit + UP*.003
            elif self.needle_grasp is not None:
                return
            elif self.cutting_step():
                target = self.steady_point(hand)
            else:
                needle = self.needle_target()
                target = None if needle is None else self.edge_point(needle)
            if target is None:
                target = self.tissue.point(self.site.point(*self.site.forceps_target))
            if target is not None:
                hand.offset += target + UP*.003 - self.active_tip(hand)
            return
        # Needle holder: guided placement above the next pass, lowered with E E.
        if self.exit is not None:
            return
        target = self.needle_target()
        if target is None:
            self.message = "The needle is not needed in this step"
            return
        hand.guide_tip = target + UP*.004
        hand.guide_lower_steps = 2
        self.inform("")

    def act(self, hand):
        """F: forceps or fingers grip / release; the needle holder passes the needle."""
        if hand.held == 1:
            self.puncture(hand)
            return
        if hand.held not in (None, 2):
            self.message = "F works with the needle holder, forceps or an empty hand"
            return
        tip = self.contact(hand)
        if (hand.held == 2 and self.exit is not None and self.needle_grasp is None
                and np.linalg.norm(tip - self.exit) <= .009):
            hand.grip = None
            self.needle_grasp = tip[2]
            self.events.append("pickup")
            self.note("grasp", hand=hand.side)
            self.inform("")
            return
        if hand.grip is not None:
            hand.grip = None
            self.note("release", hand=hand.side)
            self.inform("Released")
            return
        surface = self.surface_height(*tip[:2])
        reach = .009 if hand.held == 2 else .015
        if surface is None or abs(tip[2] - surface) > reach:
            self.message = "Lower onto the tissue first (E), then F"
            return
        hand.grip = np.array([*tip[:2], surface])
        hand.grip_pass = self.passes
        self.note("grip", hand=hand.side, tool=hand.held, at=[round(float(v), 4) for v in hand.grip])
        self.inform("")

    def puncture(self, hand):
        if self.exit is not None:
            self.message = "Forceps: take the needle and pull the suture through"
            return
        if hand.guide_tip is not None and hand.guide_lower_steps:
            return
        tip = self.active_tip(hand)
        if self.procedure is not None:
            target = self.procedure.needle_target(self)
            if target is None:
                self.message = "Not a suturing step"
                return
            point = self.procedure.puncture_point(target, tip)
            if point is None:
                self.message = "N to align on the next target, then E E and F"
                return
        else:
            point = self.site_puncture(tip)
            if point is None:
                return
        forceps = self.holder(2)
        if not self.edge_held(forceps, point):
            label = OTHER[hand.side].capitalize() if forceps is None else forceps.label
            self.message = f"{label} hand: lift this edge with the forceps first (N, F)"
            return
        guided = self.needle_target()
        error = None if guided is None else float(np.linalg.norm(point[:2] - guided[:2]))*1000
        if self.entry is None:
            self.entry = point
            hand.guide_tip = None
        else:
            self.exit = point
        self.note("puncture", hand=hand.side, edge="exit" if self.exit is not None else "entry",
                  error_mm=None if error is None else round(error, 2))
        self.passes += 1
        self.inform("")
        self.bleed(point, .0008 if self.procedure is not None else .0011)
        self.events.append("puncture")

    def site_puncture(self, tip):
        """Free needle placement on the skin or heart laceration: the point, or None."""
        if self.tissue.on_vessel(*tip[:2]):
            self.message = "Keep the needle off the coronary vessels"
            return None
        surface = self.tissue.height(*tip[:2])
        if surface is None or not surface - .012 <= tip[2] <= surface + .001:
            self.message = f"Lower the needle tip to the {self.site.noun}"
            return None
        point = self.tissue.point(tip[:2], .0015)
        along, side = self.site.coords(point)
        site = self.site
        if not (site.bite_min <= abs(side) <= site.bite_max and abs(along) <= site.along_limit):
            self.message = (f"Place the needle {site.bite_min*1000:.0f}-{site.bite_max*1000:.0f} mm "
                            "beside the wound")
            return None
        if self.entry is not None:
            entry_along, entry_side = site.coords(self.entry)
            if not (side*entry_side < 0 and abs(along - entry_along) <= site.exit_tolerance):
                self.message = "Exit must be straight across the entry"
                return None
        return point

    def check_pull(self):
        """The forceps holding the needle pull the suture through as they rise."""
        if self.needle_grasp is None or self.pulled:
            return
        forceps = self.holder(2)
        if forceps is None or forceps.tool_pose is None:
            return
        if self.active_tip(forceps)[2] - self.needle_grasp >= PULL_HEIGHT:
            self.pulled = True
            self.note("pulled", hand=forceps.side)
            self.inform("")
            self.events.append("trim")

    def throw(self, hand):
        """T: one throw of an instrument tie; the third completes the knot."""
        if hand.held != 1:
            self.message = "Tie with the needle holder (T)"
            return
        if self.exit is None:
            self.message = "Pass the needle through both edges first"
            return
        if not self.pulled:
            self.message = "Forceps: grasp the needle (N, F) and pull it through (Q)"
            return
        if len(self.stitches) >= 48:
            self.message = "Repair full - press R to reset"
            return
        self.throws += 1
        self.note("throw", hand=hand.side, throw=self.throws)
        self.inform("")
        self.events.append("tie")
        if self.throws < len(THROWS):
            return
        self.stitches.append((self.entry, self.exit, False))
        self.note("stitch", number=len(self.stitches))
        self.entry = self.exit = self.needle_grasp = hand.guide_tip = None
        self.pulled, self.throws = False, 0
        forceps = self.holder(2)
        if forceps is not None:
            forceps.grip = None
        if self.procedure is not None:
            self.procedure.stitch_tied(self)

    def trim(self, hand):
        if hand.held != 3 or not self.stitches:
            self.message = "4 then G (pick up scissors after tying)"
            return
        entry, exit, trimmed = self.stitches[-1]
        if np.linalg.norm(self.active_tip(hand) - exit) > .035:
            self.message = "N (align scissors) / X (trim thread)"
            return
        self.stitches[-1] = (entry, exit, True)
        self.events.append("trim")
        self.note("trim", stitch=len(self.stitches))
        self.inform("Thread trimmed")

    # --- guidance ----------------------------------------------------------------------
    def plan(self):
        """(hand side, prompt) for the next step of the technique."""
        def take(hand, index):
            name = TOOL_NAMES[index].lower()
            if hand.held is not None:
                return f"G set down the {hand.tool}, then {index + 1} + G ({name})"
            return f"{index + 1} then G ({name})"

        right = self.hands["right"]
        head = "" if self.procedure is None else self.procedure.head()
        if self.procedure is not None and self.procedure.done:
            return None, self.procedure.prompt_done()
        if self.task == "free":
            return None, "Free practice: any order. Z switches hands; the two-hand rules still apply"

        if self.cutting_step():
            blade = self.holder(0) or right
            steady = self.other(blade)
            if blade.held != 0:
                return blade.side, head + take(blade, 0)
            if self.procedure is not None:
                tools = self.procedure.steady_tools
                if steady.held not in tools:
                    if steady.held is not None and None in tools:
                        return steady.side, head + f"G set down the {steady.tool} (that hand steadies)"
                    return steady.side, head + take(steady, 2)
                if not self.steadied(blade):
                    what = "forceps hold" if steady.held == 2 else "fingers stretch"
                    return steady.side, head + f"N then F ({what} the tissue beside the line)"
                if blade.cut_line is None:
                    return blade.side, head + "N (put the blade on the marked line)"
                if blade.depth <= 0:
                    return blade.side, head + "E (lower the blade into the tissue)"
                return blade.side, head + (f"left / right arrows cut along the line "
                                           f"({self.procedure.covered.mean():.0%})")
            if steady.held not in (None, 2):
                return steady.side, f"G set down the {steady.tool} (that hand steadies the tissue)"
            if steady.grip is None:
                what = "forceps hold" if steady.held == 2 else "fingers stretch"
                return steady.side, f"N then F ({what} the tissue beside the blade)"
            if not self.steadied(blade):
                return blade.side, f"Arrows: bring the blade next to the {steady.side} hand"
            return blade.side, "E lowers the blade, arrows cut (keep the tissue steady)"

        scissors = self.holder(3)
        if scissors is not None and self.exit is None:
            return scissors.side, "N (align scissors) / X (trim last knot)"
        needle = self.holder(1) or right
        forceps = self.holder(2) or self.other(needle)
        if needle.held != 1:
            return needle.side, head + take(needle, 1)
        if forceps.held != 2:
            return forceps.side, head + take(forceps, 2)
        if self.procedure is not None:
            n = f"stitch {self.procedure.tied + 1}/{STITCHES_PER_ANASTOMOSIS}"
            edge = "graft" if self.entry is None else "artery"
        else:
            n = f"stitch {len(self.stitches) + 1}"
            edge = "first edge" if self.entry is None else "opposite edge"
        if self.exit is None:
            if not self.edge_held(forceps, self.needle_target()):
                return forceps.side, head + f"N then F (forceps lift the {edge}, {n})"
            if needle.guide_tip is None:
                return needle.side, head + f"N (align the needle on the {edge})"
            if needle.guide_lower_steps:
                return needle.side, head + f"{needle.guide_lower_steps}x E (lower the needle)"
            return needle.side, head + f"F (pass the needle through the {edge})"
        if self.needle_grasp is None:
            return forceps.side, head + "N then F (forceps grasp the needle tip)"
        if not self.pulled:
            pulled = 0
            if forceps.tool_pose is not None:
                pulled = (self.active_tip(forceps)[2] - self.needle_grasp) / PULL_HEIGHT
            return forceps.side, head + f"Q (pull the suture through, {min(max(pulled, 0), 1):.0%})"
        return needle.side, head + f"T (instrument tie, throw {self.throws + 1}/3: {THROWS[self.throws]})"

    def checklist(self):
        """The procedure's steps for the overlay: [(mark, text)] with mark "x" done,
        ">" in progress (the first step not done) and " " still to come."""
        def marks(items, indent=""):
            out, current = [], False
            for text, done in items:
                mark = "x" if done else (" " if current else ">")
                current = current or not done
                out.append((mark, indent + text))
            return out

        right = self.hands["right"]
        if self.task == "free":
            return [(" ", "No set order: cut and stitch anywhere"),
                    (" ", f"{len(self.stitches)} stitches, {len(self.marks)} cut segments")]
        p = self.procedure
        if p is not None and p.done:
            return [("x", p.prompt_done().split(".")[0])]

        def cut_items():
            blade = self.holder(0) or right
            placed = blade.cut_line is not None or p.covered.any()
            who = "forceps or fingers" if None in p.steady_tools else "forceps"
            return [("Scalpel in the right hand", blade.held == 0),
                    (f"Left hand steadies ({who})", self.steadied(blade) or p.covered.any()),
                    ("Blade on the line (N, E)", placed and (blade.depth > 0 or p.covered.any())),
                    (f"Cut along the line: {p.covered.mean():.0%}", False)]

        def stitch_items(edge_names):
            needle = self.holder(1) or right
            forceps = self.holder(2) or self.other(needle)
            target = self.needle_target()
            first, second = edge_names
            return [("Needle holder (R) and forceps (L)", needle.held == 1 and forceps.held == 2),
                    (f"Lift the {first} (forceps)", self.entry is not None or
                     self.edge_held(forceps, target)),
                    (f"Needle through the {first}", self.entry is not None),
                    (f"Lift the {second}", self.exit is not None or
                     (self.entry is not None and self.edge_held(forceps, target))),
                    (f"Needle out through the {second}", self.exit is not None),
                    ("Grasp the needle tip (forceps)", self.needle_grasp is not None),
                    ("Pull the suture through (Q)", self.pulled),
                    (f"Instrument tie: throw {min(self.throws + 1, 3)}/3", False)]

        if p is None:       # suture the laceration
            slots = self.site.stitch_slots
            done = len(self.stitches)
            if done >= slots:
                return [("x", f"All {slots} stitches tied")]
            return ([(">", f"Stitch {done + 1} of {slots} ({done} tied)")] +
                    marks(stitch_items(("first edge", "opposite edge")), "  "))
        rows = []
        for i, (kind, line, title) in enumerate(p.steps):
            mark = "x" if i < p.step else (">" if i == p.step else " ")
            rows.append((mark, f"{i + 1}. {title}"))
            if i == p.step:
                if kind == "cut":
                    rows += marks(cut_items(), "  ")
                else:
                    rows.append((">", f"  Stitch {p.tied + 1} of {STITCHES_PER_ANASTOMOSIS}"))
                    rows += marks(stitch_items(("graft", "artery")), "  ")
        return rows

    def action_prompt(self, short=False):
        """The next action; `short` leaves out the step title (the checklist shows it)."""
        if self._message and time.monotonic() - self._message_at < 2.5:
            return self._message
        side, text = self.plan()
        if short and self.procedure is not None and self.procedure.head():
            text = text.replace(self.procedure.head(), "", 1)
        return text if side is None else f"{side.capitalize()} hand: {text}"

    # --- cutting -------------------------------------------------------------------------
    def cut(self, hand, tip):
        hand.depth = 0.0
        if tip is None:
            hand.previous_tip = None
            return
        surface = self.tissue.height(*tip[:2])
        on_line = self.procedure is not None and self.procedure.cut_allowed(tip)
        if on_line:
            surface = self.tissue.any_height(*tip[:2])
        if surface is None or tip[2] > surface:
            if self.tissue.on_vessel(*tip[:2]):
                vessel_top = self.tissue.vessel_top(*tip[:2])
                if tip[2] <= vessel_top:
                    self.message = "Blade on a vessel! Lift (Q), move away"
            hand.previous_tip = None
            return
        if not self.steadied(hand):
            self.message = f"Steady the tissue first: {OTHER[hand.side]} hand N then F"
            hand.previous_tip = None
            return
        hand.depth = min(MAX_DEPTH, surface - tip[2])
        normal = self.tissue.any_normal(*tip[:2]) if on_line else self.tissue.normal(*tip[:2])
        point = np.array([*tip[:2], surface]) + normal*.0007
        if hand.previous_tip is not None:
            distance = np.linalg.norm(point - hand.previous_tip)
            if .0012 <= distance <= .025:
                self.marks.append(self.wound_segment(hand.previous_tip, point, hand.depth, normal))
                if on_line:
                    self.procedure.on_cut(self, point)
                # Deeper cuts ooze: drops appear along the wound and slowly spread.
                if hand.depth > .0012 and np.random.random() < .18 + 40*hand.depth:
                    along = (point - hand.previous_tip) / distance
                    beside = np.cross(along, normal)*np.random.uniform(-.0015, .0015)
                    self.bleed(point + beside, np.random.uniform(.0014, .0018) + .35*hand.depth)
            elif distance < .0012:
                return
        hand.previous_tip = point

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
        surface, normal = self.tissue.point(xy, .0004), None
        if surface is None and hasattr(self.tissue, "any_point"):   # on a vessel
            surface, normal = self.tissue.any_point(xy, .0004), self.tissue.any_normal(*xy)
        if surface is None:
            return
        normal = self.tissue.normal(*xy) if normal is None else normal
        _, _, matrix, _ = surface_strip(surface, surface, 0, 0, normal, "blood")
        self.blood.append((surface, matrix, self.clock, radius))

    # --- drawing -------------------------------------------------------------------------
    def draw(self, scene, now=None):
        if now is not None:
            self.clock = now
        for index, pose in enumerate(self.rest):
            hand = self.holder(index)
            draw_instrument(scene, index, *(hand.tool_pose if hand is not None else pose))
        lift = min(.024, self.site.bite_guide*1.2)
        forceps = self.holder(2)
        if forceps is not None and forceps.grip is not None:
            displacement = np.clip(self.active_tip(forceps) - forceps.grip, -lift, lift)
            draw_capsule(scene, forceps.grip, forceps.grip + displacement,
                         min(.003, self.site.bite_guide*.25), "lip")
        ellipse = mj.mjtGeom.mjGEOM_ELLIPSOID
        for size, pos, matrix, material in self.tissue.wound:
            draw_geom(scene, ellipse, size, pos, matrix, material)
        if self.procedure is not None:
            self.procedure.draw(scene, self.clock)
        for segment in self.marks:
            for size, pos, matrix, material in segment:
                draw_geom(scene, ellipse, size, pos, matrix, material)
        for position, matrix, born, radius in self.blood:
            grow = min(1.0, max(self.clock - born, 0) / 2.5)**.5
            r = radius*(.35 + .65*grow)
            draw_geom(scene, ellipse, [r, r*.9, r*.28], position, matrix, "blood")
        for entry, exit, trimmed in self.stitches:
            self.draw_thread(scene, entry, exit, True, trimmed)
        if self.entry is None:
            return
        if self.exit is None:
            holder = self.holder(1)
            if holder is not None and holder.tool_pose is not None:
                self.draw_thread(scene, self.entry, self.active_tip(holder), False, False)
            return
        self.draw_thread(scene, self.entry, self.exit, self.throws > 0, False, self.throws)
        if self.needle_grasp is not None and forceps is not None and forceps.tool_pose is not None:
            # The long end, from the exit up to the forceps holding the needle.
            tip = self.active_tip(forceps)
            draw_path(scene, [self.exit, (self.exit + tip)/2 + UP*.002, tip], .0008, "thread")
            draw_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, [.0012, 0, 0], tip, np.eye(3), "polished")

    def draw_thread(self, scene, entry, exit, tied, trimmed, throws=3):
        mid = (entry + exit)/2 + np.array([0., 0., .003 if tied else .013])
        draw_path(scene, [entry, (entry + mid)/2, mid, (mid + exit)/2, exit], .0008, "thread")
        if tied:
            knot = exit + np.array([0., 0., .001])
            grow = .6 + .4*min(throws, 3)/3
            draw_geom(scene, mj.mjtGeom.mjGEOM_ELLIPSOID, [.0019*grow, .0014*grow, .0012*grow],
                      knot, np.eye(3), "thread")
            if throws >= 3:
                length = self.site.tail_length*(.3 if trimmed else 1)
                for spread in (.6, -.2):
                    draw_capsule(scene, knot, knot + np.array([length, length*spread, .0004]),
                                 .0006, "thread")



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

    def complete(self):
        st = self.station
        if st.procedure is not None:
            return st.procedure.done
        if st.task == "free":
            return False       # free practice ends when the trainee starts over
        return len(st.stitches) >= st.site.stitch_slots

    def end(self, now):
        """Score and save the current attempt (if anything happened)."""
        if self.recorder is None:
            return None
        attempt = self.recorder.finish(now, self.complete(), {"stitches": len(self.station.stitches)})
        self.recorder = None
        if attempt is None:
            return None
        self.store.save(attempt)
        self.last = attempt.summary
        best = self.store.best(self.procedure.key)
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
        best = self.store.best(self.procedure.key)
        if best is None:
            self.notice = ("Free practice has no ideal run: choose a procedure for the autopilot"
                           if self.procedure.task == "free" else
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

SWITCH_SITE = ord("P")
TOGGLE_GUIDE = ord("K")
START_OVER = ord("R")
VOICE_KEY, REVIEW_KEY, ASSISTANT_KEY = ord("V"), ord("B"), ord("I")

# Shown at the bottom right, matching the status box at the bottom left (six rows each).
# K or the Key guide button hides it.
KEY_GUIDE = (
    ("KEYS  Hands", "Z other hand  Y both hands  G take / drop"),
    ("Tools", "1 scalpel 2 needle 3 forceps 4 scissors"),
    ("Move", "arrows move   Q/E up/down   N align"),
    ("Act", "F grip / pass needle   T throw   X trim"),
    ("AI", "V voice  B Gemma review  I assistant  (panel: Autopilot)"),
    ("More", "R start over  0 view  H re-aim  P procedure  M sound  K keys"),
)

# MuJoCo's viewer toggles display options on single keys that overlap ours (Q camera
# markers, T transparency, X textures, 0-5 object groups, ...). It keeps its own copy of
# the options and only takes ours when they change, so after such a key we briefly send
# the toggled value and then the normal one, which puts the display back.
VIEWER_FLAG_KEYS = {ord(mj.mjVISSTRING[i][2]): i for i in range(mj.mjtVisFlag.mjNVISFLAG)
                    if len(mj.mjVISSTRING[i][2]) == 1 and mj.mjVISSTRING[i][2].isalpha()}
VIEWER_GROUP_KEYS = {ord(str(i)): i for i in range(6)}


def undo_viewer_shortcuts(option, key_events, flags, groups):
    """Send `option` the toggled value of every shortcut pressed an odd number of times
    in `key_events` (including auto-repeats); the next frame restores `flags`/`groups`."""
    option.flags[:] = flags
    option.geomgroup[:] = groups
    for code in set(key_events):
        if key_events.count(code) % 2 == 0:
            continue
        if code in VIEWER_FLAG_KEYS:
            i = VIEWER_FLAG_KEYS[code]
            option.flags[i] = 1 - flags[i]
        if code in VIEWER_GROUP_KEYS:
            i = VIEWER_GROUP_KEYS[code]
            option.geomgroup[i] = 1 - groups[i]


def draw_user(scene, hands, station, now):
    """Everything that moves: gloves, instruments, wounds, sutures and the patient monitor."""
    scene.ngeom = 0
    for side, points in hands.items():
        draw_hand(scene, points, station.both or side == station.active)
    station.draw(scene, now)
    draw_monitor(scene, now)


def glove_poses(gloves, hands, preview, now, reset_heading):
    """side -> scene joints, and a status line. A glove that is not streaming leaves a
    keyboard-driven hand in its place."""
    poses, live, waiting = {}, [], []
    for side, hand in hands.items():
        sample = None if preview else gloves.latest.get(side)
        if reset_heading:
            hand.heading.reset()
        pose = None
        if sample is not None and now - sample[0] <= .5:
            pose = world_pose(sample[1], hand.heading, now, HAND_ANCHOR[side])
            live.append(side)
            if not hand.heading.locked:
                waiting.append(side)
        poses[side] = pose if pose is not None else preview_pose(-1.2, side)
    if preview:
        status = "Keyboard practice"
    elif not live:
        status = "Waiting for the gloves (keyboard hands)"
    elif waiting:
        status = f"Rest {' and '.join(waiting)} hand flat, fingers to the screen"
    elif len(live) == 1:
        status = f"{live[0].capitalize()} glove live, {OTHER[live[0]]} by keyboard"
    else:
        status = "Both gloves live"
    return poses, status


def render_frame(renderer, data, camera, option, drawn, station, now):
    """An offscreen picture of the operating field as the trainee sees it (for Gemma)."""
    renderer.update_scene(data, camera=camera, scene_option=option)
    extra = mj.MjvScene(renderer.model, 8000) if not hasattr(renderer, "_extra") else renderer._extra
    renderer._extra = extra
    draw_user(extra, drawn, station, now)
    scene = renderer.scene
    for i in range(extra.ngeom):
        if scene.ngeom >= scene.maxgeom:
            break
        target, source = scene.geoms[scene.ngeom], extra.geoms[i]
        mj.mjv_initGeom(target, source.type, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.ones(4, np.float32))
        for name in ("size", "pos", "mat", "rgba", "matid", "emission", "specular", "shininess",
                     "reflectance", "category", "dataid", "objtype", "objid"):
            setattr(target, name, getattr(source, name))
        scene.ngeom += 1
    return renderer.render()


def compact_checklist(rows, before=1, after=3):
    """Main steps in full; of the indented sub-steps only the last finished one, the current
    one and the next few, so the box stays small (MuJoCo overlays hold about 500 characters)."""
    sub = [i for i, (_, text) in enumerate(rows)
           if text.startswith("  ") and not text.startswith("  Stitch")]
    active = [i for i in sub if rows[i][0] == ">"]
    current = active[-1] if active else (sub[0] if sub else 0)
    keep = set(i for i in range(len(rows)) if i not in sub)
    keep |= {i for i in sub if current - before <= i <= current + after}
    return [rows[i] for i in sorted(keep)]


def checklist_text(session, now):
    """The checklist as plain text (Procedures window)."""
    rows = [f"[{mark}] {text}" for mark, text in session.station.checklist()]
    done = session.recent_done(now)
    return "\n".join(rows + ([""] + [f"Done: {text}" for text in done] if done else []))


def display(gloves, preview, procedure):
    model = build_model()
    Look.model = model
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    site = SITES[procedure.site]
    procedures_button = PanelButton(model, data, PROCEDURES_BUTTON)
    reset_hands_button = PanelButton(model, data, RESET_HANDS_BUTTON)
    both_toggle = PanelToggle(model, data, BOTH_HANDS_BUTTON)
    pick_bodies = {side: model.body(f"pick_{side}").id for side in SIDES}
    pick_mocap = {side: model.body(f"pick_{side}").mocapid[0] for side in SIDES}
    clear_button = PanelButton(model, data, CLEAR_BUTTON)
    review_button = PanelButton(model, data, REVIEW_BUTTON)
    assistant_button = PanelButton(model, data, ASSISTANT_BUTTON)
    autopilot_toggle = PanelToggle(model, data, AUTOPILOT_BUTTON)
    voice_toggle = PanelToggle(model, data, VOICE_BUTTON)
    guide_id = model.equality(GUIDE_BUTTON).id
    last_status = None
    keys = SimpleQueue()
    held_keys = HeldKeys()
    sounds = Sounds()
    store, telemetry = TrainingStore(), Telemetry()
    from coach import Coach
    coach = Coach(store)
    picture = {"renderer": None, "camera": mj.MjvCamera(), "option": mj.MjvOption(), "drawn": {}}

    def render():
        """Offscreen frame of the current view for Gemma (None if rendering fails)."""
        try:
            if picture["renderer"] is None:
                picture["renderer"] = mj.Renderer(model, 540, 768)
            return render_frame(picture["renderer"], data, picture["camera"], picture["option"],
                                picture["drawn"], session.station, time.monotonic())
        except Exception as error:     # no offscreen GL context: review the data only
            print(f"Offscreen render unavailable ({error})", flush=True)
            return None

    last_frame = time.monotonic()
    session = Session(model, data, procedure, store, telemetry, coach, render, last_frame)
    picker = None

    def open_picker():
        nonlocal picker
        if picker is None or not picker.alive:
            try:
                from procedure_window import ProcedureClient
                picker = ProcedureClient(PROCEDURES.values(), session.procedure.key)
            except OSError as error:
                print(f"Procedures window unavailable ({error}); use P to cycle", flush=True)

    open_picker()
    print("Two hands: the keyboard moves the hand the next step needs (green cuff); Z swaps it.\n"
          "1 scalpel | 2 needle holder | 3 forceps | 4 scissors | G pick up/set down\n"
          "Arrows move, Q/E raise/lower (tap for a small step, hold to glide).\n"
          "N aligns the tool, F grips (forceps/fingers) or passes the needle, T throws a knot,\n"
          "X trims. R starts over (the attempt so far is scored and saved).\n"
          "View: scroll or +/- to zoom, left-drag to orbit, right-drag to pan,\n"
          "      double-right-click a spot (tissue, tray, room) to zoom around it, 0 resets the view.\n"
          "H re-aims the gloves: press H, then rest your hands flat with fingers toward the screen.\n"
          "M mutes or unmutes the OR sounds.\n"
          "Procedures window (dropdown): pick incision, suture, bypass or free practice; P cycles.\n"
          "Double-click a hand to move it with the arrows; Y (or the Both hands button) moves both.\n"
          "Panel (Equality, right): Procedures (reopens it), Reset hands (hands back to their\n"
          "      start), Clear repair, Key guide (or K),\n"
          "      Autopilot (replays the best recorded attempt), AI Assistant (I), AI Voice (V),\n"
          "      Gemma review (B).\n"
          f"{store.status}. {telemetry.status}. {coach.status()}.",
          flush=True)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
            with viewer.lock():
                show_site(model, viewer.opt, site)
                bench_view(model, data, viewer.cam, site.zoom, site.focus)
                display_flags = viewer.opt.flags.copy()   # normal display; see VIEWER_FLAG_KEYS
            overlay = getattr(viewer, "set_texts", None)
            next_overlay = 0.0
            while viewer.is_running():
                now = time.monotonic()
                dt = min(now - last_frame, .1)
                last_frame = now
                pressed, key_events = [], []
                while not keys.empty():
                    code = keys.get_nowait()
                    key_events.append(code)
                    if code in MOVES and code in held_keys.keys:
                        held_keys.press(code, now)  # OS auto-repeat: continuous motion handles it
                        continue
                    held_keys.press(code, now)
                    pressed.append(code)
                held = list(held_keys.moves(now, dt))
                station = session.station

                # Panel buttons (Equality section) and their keys.
                with viewer.lock():
                    undo_viewer_shortcuts(viewer.opt, key_events, display_flags, view_groups(station.site))
                    picked = viewer.perturb.select       # double-click on a hand selects it
                    for side, body in pick_bodies.items():
                        if picked == body:
                            pressed.append(SELECT_HAND[side])   # recorded like a key
                            viewer.perturb.select = 0
                            viewer.perturb.active = 0
                    clear = clear_button.pressed() or START_OVER in pressed
                    review = review_button.pressed() or REVIEW_KEY in pressed
                    assistant = assistant_button.pressed() or ASSISTANT_KEY in pressed
                    voice = voice_toggle.changed() or VOICE_KEY in pressed
                    autopilot = autopilot_toggle.changed()
                    if TOGGLE_GUIDE in pressed:
                        data.eq_active[guide_id] = not data.eq_active[guide_id]
                    show_guide = bool(data.eq_active[guide_id])
                    if procedures_button.pressed():
                        open_picker()
                    if both_toggle.changed() and both_toggle.on != session.station.both:
                        pressed.append(BOTH_HANDS)       # recorded like a key, so replays stay exact
                    if reset_hands_button.pressed():
                        pressed.append(RESET_HANDS)      # recorded like a key, so replays stay exact
                    choice = picker.chosen() if picker is not None else None
                    if SWITCH_SITE in pressed:
                        names = list(PROCEDURES)
                        choice = names[(names.index(session.procedure.key) + 1) % len(names)]
                    picture["camera"].lookat[:] = viewer.cam.lookat
                    picture["camera"].distance = viewer.cam.distance
                    picture["camera"].azimuth = viewer.cam.azimuth
                    picture["camera"].elevation = viewer.cam.elevation
                    picture["option"].geomgroup[:] = station.site.groups
                taken = {SWITCH_SITE, TOGGLE_GUIDE, START_OVER, REVIEW_KEY, ASSISTANT_KEY, VOICE_KEY}
                pressed = [code for code in pressed if code not in taken]
                if clear and (choice is None or choice == session.procedure.key):
                    session.restart(now)
                    sounds.play("setdown")
                if choice is not None and choice != session.procedure.key:
                    session.restart(now, PROCEDURES[choice])
                    if session.site is not site:
                        site = session.site
                        with viewer.lock():
                            show_site(model, viewer.opt, site)
                            bench_view(model, data, viewer.cam, site.zoom, site.focus)
                    sounds.play("switch")
                    print(f"Procedure: {session.procedure.label}", flush=True)
                if autopilot:
                    if autopilot_toggle.on:
                        if not session.start_autopilot(now):
                            with viewer.lock():
                                autopilot_toggle.set(False)
                    else:
                        session.stop_autopilot(now)
                    sounds.play("switch")
                elif session.autopilot is None and autopilot_toggle.on:
                    with viewer.lock():
                        autopilot_toggle.set(False)   # a restart or site switch ended it

                if ord("M") in pressed:
                    sounds.muted = not sounds.muted
                    pressed = [code for code in pressed if code != ord("M")]
                zoom = [code for code in pressed if code in ZOOM_IN | ZOOM_OUT | {RESET_VIEW}]
                if zoom:
                    pressed = [code for code in pressed if code not in zoom]
                    with viewer.lock():
                        for code in zoom:
                            if code == RESET_VIEW:
                                bench_view(model, data, viewer.cam, site.zoom, site.focus)
                            else:
                                if viewer.cam.type != mj.mjtCamera.mjCAMERA_FREE:
                                    bench_view(model, data, viewer.cam, site.zoom, site.focus)
                                factor = .8 if code in ZOOM_IN else 1.25
                                viewer.cam.distance = float(np.clip(viewer.cam.distance*factor, .03, 6))

                reset_heading = ord("H") in pressed
                pressed = [key for key in pressed if key != ord("H")]
                poses, status = glove_poses(gloves, session.station.hands, preview, now, reset_heading)
                live = status.endswith("live") or "glove live" in status
                drawn = session.step(now, poses, pressed, held, live)
                picture["drawn"] = drawn
                coach.update(session.station, session.attempt_id or "autopilot",
                             session.autopilot is not None, voice, review, assistant, render)
                with viewer.lock():
                    draw_user(viewer.user_scn, drawn, session.station, now)
                    for side, points in drawn.items():     # keep the pick spheres on the palms
                        data.mocap_pos[pick_mocap[side]] = (
                            (points["wrist"] + points["middle_proximal"]) / 2 if points else [0, 0, -5])
                    mj.mj_kinematics(model, data)
                    if voice_toggle.on != coach.voice_on:
                        voice_toggle.set(coach.voice_on)
                    if both_toggle.on != session.station.both:
                        both_toggle.set(session.station.both)    # Y, or a fresh station
                for event in session.station.events:
                    sounds.play(event)
                session.station.events.clear()
                sounds.heartbeat(now)

                if status != last_status:
                    print(status, flush=True)
                    last_status = status
                if overlay is not None and now >= next_overlay:
                    station = session.station
                    left, right = station.hands["left"], station.hands["right"]
                    keys_for = "both hands, Y" if station.both else f"{station.active}, Z"
                    tool_status = f"L {left.tool}   R {right.tool}   (keys: {keys_for})"
                    if session.autopilot is not None:
                        pilot = session.autopilot
                        action = ("AUTOPILOT finished: Autopilot button off to take over" if pilot.done else
                                  f"AUTOPILOT {pilot.progress:.0%} (best run {pilot.summary['score']:.0f}/100): "
                                  f"{station.action_prompt(short=True)}")
                    else:
                        action = station.action_prompt(short=True)
                    best = store.best(session.procedure.key)
                    scores = (f"last {session.last['score']:.0f}" if session.last else "no attempt yet") + (
                        f", best {best[0]['score']:.0f}" if best else "")
                    coach_line = f"{scores}  |  {session.recent_notice() or coach.status()}"
                    rows = compact_checklist(station.checklist())
                    done = session.recent_done(now)[-2:]
                    steps = [(int(mj.mjtFont.mjFONT_NORMAL), int(mj.mjtGridPos.mjGRID_TOPLEFT),
                              "\n".join(["PROCEDURE"] + [f"[{m}]" for m, _ in rows] +
                                         ["DONE"]*bool(done) + [""]*(len(done) - 1)),
                              "\n".join([session.procedure.label] + [t for _, t in rows] + done))]
                    if picker is not None:
                        picker.update(session.procedure.key, checklist_text(session, now))
                    texts = steps + [(int(mj.mjtFont.mjFONT_NORMAL), int(mj.mjtGridPos.mjGRID_BOTTOMLEFT),
                              "SURGICAL TRAINING\nProcedure\nHands\nStitches / depth\nAction\nScore / coach",
                              f"{status}\n{session.procedure.label}  (P or Procedures window)\n"
                              f"{tool_status}\n{len(station.stitches)} / {station.depth*1000:.1f} mm\n"
                              f"{action}\n{coach_line}")]
                    if show_guide:
                        texts.append((int(mj.mjtFont.mjFONT_NORMAL), int(mj.mjtGridPos.mjGRID_BOTTOMRIGHT),
                                      "\n".join(left for left, _ in KEY_GUIDE),
                                      "\n".join(right for _, right in KEY_GUIDE)))
                    overlay(texts)
                    next_overlay = now + .10
                viewer.sync()
                time.sleep(max(0, 1/60 - (time.monotonic() - now)))
    finally:
        session.end(time.monotonic())
        if picker is not None:
            picker.close()
        coach.close()
        telemetry.close()
        store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Practice with keyboard hands and no gloves")
    parser.add_argument("--procedure", choices=list(PROCEDURES), default=DEFAULT,
                        help="Procedure to start with (also chosen in the Procedures window)")
    parser.add_argument("--site", choices=sorted(SITES),
                        help="Older option: skin or heart (suture) or bypass")
    args = parser.parse_args()
    procedure = from_site(args.site) if args.site else PROCEDURES[args.procedure]
    print(f"Procedure: {procedure.label}", flush=True)
    gloves = GloveStream()
    if not args.preview:
        print("Connect both gloves (rokoko-sdk and rkk-hand-solver --emit-openxr). Tilt and roll\n"
              "follow your hands directly; to set the direction, rest each hand flat with the\n"
              "fingers toward the screen for half a second. Press H to set it again.\n"
              "A glove that is not streaming is replaced by a keyboard hand.", flush=True)
        threading.Thread(target=gloves.run, daemon=True).start()
    display(gloves, args.preview, procedure)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
