"""Rokoko Smartglove input: the RGMP v2 stream from rkk-hand-solver, yaw calibration, and the
mapping of glove joints into the operating room (wrist pinned, rotation and fingers live).

A glove that is not streaming is replaced by a relaxed keyboard-driven hand (preview_pose).
"""

import json
import re
import socket
import struct
import time

import numpy as np


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
