"""Practice virtual incisions and sutures in a glove-controlled MuJoCo scene."""

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
GLOVE = [.20, .48, .70, 1]
SLEEVE = [.10, .29, .29, 1]
PAD_MIN = np.array([-.098, -.016])
PAD_MAX = np.array([.108, .126])
PAD_TOP = .955
MAX_DEPTH = .006
METAL = [.70, .78, .83, 1]
HANDLE = [.19, .30, .35, 1]
TRAY_GRIP = np.array([-.221, .066, .942])
TRAY_DIRECTION = np.array([0., 1., 0.])
PICKUP_RADIUS = .045
THREAD = [.13, .19, .36, 1]
INCISION_Y = .055
TOOL_NAMES = ("Scalpel", "Needle holder", "Forceps", "Suture scissors")
TOOL_HOME = (
    TRAY_GRIP,
    np.array([.27, .08, .938]),
    np.array([-.271, .072, .941]),
    np.array([.33, .082, .938]),
)
TOOL_TIP = (.051, .084, .062, .050)
TABLE_MIN = np.array([-.54, -.28])
TABLE_MAX = np.array([.54, .44])

LAB = """
<mujoco model="Surgical training station">
  <visual>
    <global offwidth="1280" offheight="800"/>
    <headlight ambient=".36 .38 .40" diffuse=".52 .54 .56" specular=".20 .22 .24"/>
    <map znear=".008"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" width="128" height="128"
             rgb1=".27 .33 .34" rgb2=".31 .37 .38"/>
    <material name="floor" texture="floor" texrepeat="6 6"/>
    <material name="steel" rgba=".61 .69 .71 1" specular=".8" shininess=".65"/>
    <material name="drape" rgba=".045 .28 .26 1" specular=".08"/>
  </asset>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="-.23 -.16 1.65" dir=".12 .12 -1" cutoff="58" diffuse=".78 .80 .81"/>
    <geom type="plane" size="4 4 .01" material="floor"/>
    <geom type="box" pos="0 .68 1.18" size="1.6 .03 1.18" rgba=".55 .67 .67 1"/>
    <geom type="box" pos="-1.5 -.12 1.18" size=".025 .84 1.18" rgba=".46 .57 .58 1"/>
    <geom type="box" pos="0 .642 1.12" size="1.45 .009 .075" rgba=".24 .38 .38 1"/>
    <!-- Operating table, sterile drape and silicone practice pad -->
    <geom type="box" pos="0 .08 .876" size=".55 .37 .023" material="steel"/>
    <geom type="box" pos="0 .09 .46" size=".31 .18 .37" rgba=".22 .33 .34 1"/>
    <geom type="box" pos="0 .065 .909" size=".326 .285 .011" material="drape"/>
    <geom type="box" pos="0 -.218 .856" size=".326 .008 .055" material="drape"/>
    <geom type="box" pos="-.322 .065 .866" size=".008 .28 .044" material="drape"/>
    <geom type="box" pos=".322 .065 .866" size=".008 .28 .044" material="drape"/>
    <geom type="box" pos=".005 .055 .924" size=".116 .086 .006" rgba=".58 .60 .47 1"/>
    <geom type="box" pos=".005 .055 .942" size=".103 .071 .013" rgba=".67 .41 .34 1"/>
    <geom type="box" pos=".005 .055 .936" size=".105 .073 .003" rgba=".78 .58 .42 1"/>
    <geom type="box" pos="0 .030 .9555" size=".085 .001 .0005" rgba=".91 .77 .66 1"/>
    <geom type="box" pos="0 .081 .9555" size=".085 .001 .0005" rgba=".91 .77 .66 1"/>
    <geom type="cylinder" pos="-.080 .055 .956" size=".004 .001" rgba=".92 .82 .71 1"/>
    <geom type="cylinder" pos=".080 .055 .956" size=".004 .001" rgba=".92 .82 .71 1"/>
    <!-- Stainless instrument trays -->
    <geom type="box" pos="-.236 .081 .927" size=".066 .108 .005" material="steel"/>
    <geom type="box" pos="-.299 .081 .938" size=".003 .108 .010" material="steel"/>
    <geom type="box" pos="-.173 .081 .938" size=".003 .108 .010" material="steel"/>
    <geom type="box" pos="-.236 -.024 .938" size=".064 .003 .010" material="steel"/>
    <geom type="box" pos="-.236 .186 .938" size=".064 .003 .010" material="steel"/>
    <geom type="box" pos=".297 .085 .927" size=".077 .104 .005" material="steel"/>
    <geom type="box" pos=".222 .085 .937" size=".003 .104 .010" material="steel"/>
    <geom type="box" pos=".373 .085 .937" size=".003 .104 .010" material="steel"/>
    <geom type="box" pos=".297 -.016 .937" size=".074 .003 .010" material="steel"/>
    <geom type="box" pos=".297 .186 .937" size=".074 .003 .010" material="steel"/>
    <geom type="capsule" fromto="-.076 .055 .9559 .076 .055 .9559"
          size=".00125" rgba=".37 .105 .097 1"/>
    <!-- Surgical lamp and equipment mast -->
    <geom type="capsule" fromto="-.34 .27 .92 -.34 .27 1.34" size=".013" material="steel"/>
    <geom type="capsule" fromto="-.34 .27 1.34 -.20 .28 1.34" size=".013" material="steel"/>
    <geom type="cylinder" pos="-.185 .28 1.309" size=".094 .023" rgba=".69 .77 .76 1"/>
    <geom type="cylinder" pos="-.185 .28 1.283" size=".082 .005" rgba=".85 .93 .91 1"/>
    <geom type="sphere" pos="-.219 .275 1.277" size=".016" rgba=".75 1 .96 1"/>
    <geom type="sphere" pos="-.153 .270 1.277" size=".016" rgba=".75 1 .96 1"/>
    <geom type="sphere" pos="-.181 .314 1.277" size=".016" rgba=".75 1 .96 1"/>
    <!-- Patient monitor with a static training waveform -->
    <geom type="box" pos=".287 .277 1.113" size=".124 .024 .086" rgba=".18 .27 .28 1"/>
    <geom type="box" pos=".287 .251 1.113" size=".111 .003 .073" rgba=".025 .086 .082 1"/>
    <geom type="capsule" fromto=".194 .247 1.105 .236 .247 1.105" size=".0018" rgba=".32 .94 .67 1"/>
    <geom type="capsule" fromto=".236 .247 1.105 .246 .247 1.116" size=".0018" rgba=".32 .94 .67 1"/>
    <geom type="capsule" fromto=".246 .247 1.116 .258 .247 1.097" size=".0018" rgba=".32 .94 .67 1"/>
    <geom type="capsule" fromto=".258 .247 1.097 .270 .247 1.150" size=".0018" rgba=".32 .94 .67 1"/>
    <geom type="capsule" fromto=".270 .247 1.150 .284 .247 1.088" size=".0018" rgba=".32 .94 .67 1"/>
    <geom type="capsule" fromto=".284 .247 1.088 .298 .247 1.105" size=".0018" rgba=".32 .94 .67 1"/>
    <geom type="capsule" fromto=".298 .247 1.105 .377 .247 1.105" size=".0018" rgba=".32 .94 .67 1"/>
    <geom type="box" pos=".286 .278 .988" size=".024 .022 .042" material="steel"/>
    <geom type="box" pos=".286 .278 .922" size=".077 .043 .006" material="steel"/>
    <camera name="bench" pos=".28 -.62 1.43" xyaxes=".925 .380 0 -.202 .492 .847" fovy="45"/>
  </worldbody>
</mujoco>
"""


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


def world_pose(points, heading):
    wrist = points["wrist"]
    rebased = {}
    for name, position in points.items():
        x, y, z = position - wrist
        rebased[name] = np.array([x, -z, y])
    if heading is None:
        forward = rebased["middle_proximal"]
        if np.linalg.norm(forward[:2]) < .025:
            return None, None
        angle = np.pi / 2 - np.arctan2(forward[1], forward[0])
        c, s = np.cos(angle), np.sin(angle)
        heading = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return {key: ANCHOR + heading @ value for key, value in rebased.items()}, heading


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
    sleeve_end = wrist - forward*.017
    capsule(wrist - forward*.14, sleeve_end, .027, SLEEVE)
    capsule(sleeve_end, wrist + forward*.019, .023, GLOVE)
    palm = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(
        palm, mj.mjtGeom.mjGEOM_ELLIPSOID,
        np.array([width*.57, length*.58, .013]),
        wrist + forward*length*.51, basis.ravel(), np.asarray(GLOVE, dtype=np.float32),
    )
    scene.ngeom += 1

    for finger, chain in CHAINS.items():
        thumb = finger == "thumb"
        base = points[chain[0]]
        knuckle = points[chain[1]]
        palm_start = wrist*.25 + base*.75 if thumb else wrist + (knuckle-wrist)*.62
        capsule(palm_start, knuckle, .013 if thumb else .011, GLOVE)
        for index, (root, tip) in enumerate(zip(chain[1:], chain[2:])):
            radius = (.0105 if thumb else .0095) - index*.0009
            capsule(points[root], points[tip], radius, GLOVE)


def draw_geom(scene, kind, size, pos, matrix, color):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    mj.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), pos, matrix.ravel(),
                    np.asarray(color, dtype=np.float32))
    scene.ngeom += 1
    return geom


def draw_capsule(scene, start, end, radius, color):
    if np.linalg.norm(end - start) < 1e-6:
        return
    geom = draw_geom(scene, mj.mjtGeom.mjGEOM_CAPSULE, [1, 1, 1],
                     np.zeros(3), np.eye(3), color)
    if geom is not None:
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, radius, start, end)


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


def draw_instrument(scene, index, grip, direction):
    across, up = instrument_axes(direction)
    if index == 0:
        basis = np.column_stack((-across, np.cross(direction, -across), direction))
        draw_capsule(scene, grip - direction*.060, grip + direction*.015, .0048, HANDLE)
        draw_geom(scene, mj.mjtGeom.mjGEOM_BOX, [.0011, .0038, .011],
                  grip + direction*.025, basis, METAL)
        draw_geom(scene, mj.mjtGeom.mjGEOM_ELLIPSOID, [.0009, .0032, .013],
                  grip + direction*.038, basis, METAL)
        return

    jaw_length = .058 if index == 1 else TOOL_TIP[index]
    jaw_gap = .003 if index == 3 else .001
    for side in (-1, 1):
        start = grip - direction*.038 + side*across*.007
        end = grip + direction*jaw_length + side*across*jaw_gap
        draw_capsule(scene, start, end, .0025, METAL)
        if index in (1, 3):
            angles = np.linspace(0, 2*np.pi, 9)
            center = start - direction*.006 + side*across*.003
            ring = [center + .007*(np.cos(t)*across + np.sin(t)*direction) for t in angles]
            draw_path(scene, ring, .0016, METAL)
    if index == 1:
        # Curved needle clamped near its midpoint, with a visible thread tail.
        center = grip + direction*.059
        needle = [center + direction*(.025*np.sin(t)) + up*(.015*(np.cos(t)-1))
                  for t in np.linspace(-1.45, np.pi/2, 10)]
        draw_path(scene, needle, .00115, METAL)
        tail = needle[0]
        draw_path(scene, [tail, tail - across*.018 - up*.014,
                          tail - across*.034 - up*.019, tail - across*.05], .00085, THREAD)


class CuttingStation:
    def __init__(self):
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

    def park(self):
        if self.held is None:
            return
        grip, direction = self.tool_pose
        xy = np.clip(grip[:2], TABLE_MIN, TABLE_MAX)
        flat = np.array([direction[0], direction[1], 0.])
        length = np.linalg.norm(flat)
        flat = flat / length if length > .001 else TRAY_DIRECTION.copy()
        x, y = xy
        height = .899
        if -.326 <= x <= .326 and -.220 <= y <= .350:
            height = .920
        if (-.302 <= x <= -.170 or .220 <= x <= .374) and -.022 <= y <= .188:
            height = .932
        if np.all(xy >= PAD_MIN) and np.all(xy <= PAD_MAX):
            height = PAD_TOP
        self.rest[self.held] = (np.array([x, y, height + .010]), flat)
        self.held = self.tool_pose = self.previous_tip = self.tissue_grip = None
        self.pinch_finger = None
        self.guide_tip = None
        self.guide_lower_steps = 0
        self.message = "G (pick up selected tool)"

    def update(self, pose, keys):
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
            return f"{key} (lower needle to pad)"
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
        self.message = {
            0: "Arrows (move) / E (lower blade) / G (set down)",
            2: "N (align forceps) / F (grip edge)",
            3: "N (align scissors) / X (trim thread)",
        }.get(index, "N (align needle)")

    def key(self, code, pose):
        moves = {263: (0, -.006), 262: (0, .006), 264: (1, -.006), 265: (1, .006),
                 ord("Q"): (2, .003), ord("E"): (2, -.003)}
        if code in moves:
            axis, amount = moves[code]
            if self.held == 1 and self.guide_tip is not None and code == ord("E"):
                if self.guide_lower_steps:
                    self.guide_tip[2] += amount
                    self.guide_lower_steps -= 1
            else:
                self.update_tool_pose(pose)
                self.guide_tip = None
                self.guide_lower_steps = 0
                limit = .75 if axis < 2 else .45
                self.offset[axis] = np.clip(self.offset[axis]+amount, -limit, limit)
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
                target = np.array([-.02, INCISION_Y-.012, PAD_TOP+.003])
                self.offset += target - self.active_tip()
                self.message = "F (grip or release practice edge)"
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
            slot = len(self.stitches) % 5
            x = -.064 + slot*.032 if self.entry is None else self.entry[0]
            side = -.014 if self.entry is None else -.014 * np.sign(self.entry[1]-INCISION_Y)
            self.guide_tip = np.array([x, INCISION_Y + side, PAD_TOP+.004])
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
            self.message = "G (pick up selected tool or set it down)"

    def stitch(self, action):
        if self.held == 2 and action == ord("F"):
            if self.tissue_grip is not None:
                self.tissue_grip = None
                self.message = "F (grip practice edge)"
                return
            tip = self.active_tip()
            if (abs(tip[2]-PAD_TOP) <= .009 and abs(tip[1]-INCISION_Y) <= .025
                    and PAD_MIN[0] <= tip[0] <= PAD_MAX[0]):
                self.tissue_grip = np.array([*tip[:2], PAD_TOP])
                self.message = "Q (lift edge) / F (release)"
            else:
                self.message = "N (align forceps with practice edge)"
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
                self.message = "Pad full - press R to reset"
                return
            self.stitches.append((self.entry, self.exit, False))
            self.entry = self.exit = None
            self.guide_tip = None
            return

        if self.exit is not None:
            self.message = "Press T to tighten and tie"
            return
        if self.guide_tip is not None and self.guide_lower_steps:
            return
        tip = self.active_tip()
        if not (PAD_TOP-.012 <= tip[2] <= PAD_TOP+.001
                and np.all(tip[:2] >= PAD_MIN) and np.all(tip[:2] <= PAD_MAX)):
            self.message = "Lower the needle tip to the pad"
            return
        point = np.array([*tip[:2], PAD_TOP+.002])
        side = point[1] - INCISION_Y
        if not .006 <= abs(side) <= .030:
            self.message = "Place the needle just beside the incision"
            return
        if self.entry is None:
            self.entry = point
            self.guide_tip = None
        elif side*(self.entry[1]-INCISION_Y) < 0 and abs(point[0]-self.entry[0]) <= .027:
            self.exit = point
            self.message = "Press T to tighten and tie"
        else:
            self.message = "Place the exit across from the entry"

    def cut(self, tip):
        self.depth = 0.0
        if tip is None or tip[2] > PAD_TOP or np.any(tip[:2] < PAD_MIN) or np.any(tip[:2] > PAD_MAX):
            self.previous_tip = None
            return
        self.depth = min(MAX_DEPTH, PAD_TOP - tip[2])
        point = np.array([*tip[:2], PAD_TOP + .0007])
        if self.previous_tip is not None:
            distance = np.linalg.norm(point-self.previous_tip)
            if .0012 <= distance <= .025:
                self.marks.append((self.previous_tip, point, self.depth))
            elif distance < .0012:
                return
        self.previous_tip = point

    def draw(self, scene):
        for index, pose in enumerate(self.rest):
            draw_instrument(scene, index, *(self.tool_pose if index == self.held else pose))
        if self.tissue_grip is not None and self.held == 2:
            tip = self.active_tip()
            displacement = np.clip(tip - self.tissue_grip, -.024, .024)
            draw_capsule(scene, self.tissue_grip, self.tissue_grip + displacement,
                         .003, [.74, .44, .38, 1])
        for start, end, depth in self.marks:
            draw_capsule(scene, start, end, .0006 + .24*depth, [.22, .055, .049, 1])
        for entry, exit, trimmed in self.stitches:
            self.draw_thread(scene, entry, exit, True, trimmed)
        if self.entry is not None:
            end = self.exit
            if end is None and self.held == 1:
                end = self.active_tip()
            if end is not None:
                self.draw_thread(scene, self.entry, end, False, False)

    @staticmethod
    def draw_thread(scene, entry, exit, tied, trimmed):
        mid = (entry+exit)/2 + np.array([0., 0., .003 if tied else .013])
        draw_path(scene, [entry, (entry+mid)/2, mid, (mid+exit)/2, exit], .0012, THREAD)
        if tied:
            knot = exit + np.array([0., 0., .001])
            draw_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, [.002, 0, 0], knot, np.eye(3), THREAD)
            length = .003 if trimmed else .012
            draw_capsule(scene, knot, knot + np.array([length, length*.6, 0.]), .001, THREAD)


def display(frames, preview):
    model = mj.MjModel.from_xml_string(LAB)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    heading = None
    last_status = None
    keys = SimpleQueue()
    station = CuttingStation()
    print("1 scalpel | 2 needle holder | 3 forceps | 4 scissors | G pick up/set down\n"
          "Arrows move, Q/E raise/lower. N aligns tools, F punctures or grips tissue, T ties, X trims. R clears pad.",
          flush=True)

    with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
        with viewer.lock():
            viewer.cam.type = mj.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("bench").id
        overlay = getattr(viewer, "set_texts", None)
        next_overlay = 0.0
        while viewer.is_running():
            now = time.monotonic()
            pressed = []
            while not keys.empty():
                pressed.append(keys.get_nowait())

            sample = frames.latest
            if preview:
                status, points = "Keyboard practice", preview_pose(-1.2)
            elif sample is None or now - sample[0] > .5:
                status, points = "Waiting for the Rokoko hand stream", None
                heading = None
            else:
                points, heading = world_pose(sample[1], heading)
                status = "Rokoko live" if points else "Hold your hand flat to align"

            points = station.update(points, pressed)
            with viewer.lock():
                draw_hand(viewer.user_scn, points)
                station.draw(viewer.user_scn)

            if status != last_status:
                print(status, flush=True)
                last_status = status
            if overlay is not None and now >= next_overlay:
                tool_status = (TOOL_NAMES[station.selected] + " selected"
                               if station.held is None else TOOL_NAMES[station.held] + " held")
                overlay((int(mj.mjtFont.mjFONT_NORMAL), int(mj.mjtGridPos.mjGRID_BOTTOMLEFT),
                         "SURGICAL TRAINING\nInstrument\nStitches / depth\nAction",
                         f"{status}\n{tool_status}\n{len(station.stitches)} / "
                         f"{station.depth*1000:.1f} mm\n{station.action_prompt()}"))
                next_overlay = now + .10
            viewer.sync()
            time.sleep(max(0, 1/60 - (time.monotonic() - now)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Practice cutting with the keyboard and no glove")
    args = parser.parse_args()
    frames = Frames()
    if args.preview:
        display(frames, True)
        return
    stream_script = SDK / "rgmp_stream.py"
    if not stream_script.is_file():
        parser.error(f"Missing Rokoko SDK example: {stream_script}")
    print("Connect only the left glove and hold your hand flat at startup.", flush=True)
    with subprocess.Popen(
        [sys.executable, "-u", str(stream_script), "--port", "12277",
         "--group-name", "joints_openxr", "--measure-type", "TRANSFORM"],
        stdout=subprocess.PIPE, text=True,
    ) as process:
        threading.Thread(target=frames.read, args=(process.stdout,), daemon=True).start()
        try:
            display(frames, False)
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
