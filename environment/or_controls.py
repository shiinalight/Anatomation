"""Operating-room controls for the hand simulator (needs only mujoco + numpy).

Exported as environment/or_controls.py (source: surgery-sim/surgical_env/sim_controls.py; edit
there). or_scene.py / surgical_suturing_guided.py use it like this:

    controls = EnvironmentControls(model, data)          # after building the model
    ...each frame of the viewer loop:
    actions = controls.frame(viewer, now)                # buttons, camera, heart, animations
    with controls.heart_at_home(viewer):                 # tissue queries + drawing on the heart
        ...station.update(...); draw_user(...)
    controls.follow_heart(viewer.user_scn, *station.drawn)

What it does:
- Buttons in the viewer's Equality panel (welds between two bodies, never simulated):
  "Reset view", "Reset hands", "Reset instruments", "Lift heart out" (on/off), "Load heart (STL)".
  frame() returns the names of the buttons pressed this frame.
- The free camera never leaves the room (numeric or_room_box).
- The heart (a mocap body): double-click it to select it - it lifts out of the chest - then
  Ctrl + right-drag moves it, Ctrl + left-drag turns it. "Lift heart out" off (or Reset
  instruments) puts it back. Wounds and stitches on the heart move with it.
- The environment's extra instruments can be moved the same way (Ctrl + drag); Reset instruments
  puts them back.
- Screens (CT viewer, heart model, ventilator, perfusion) and the heart-lung machine's roller pumps
  animate. Frames come baked in the XML (texture <name>_frames, numeric or_anim_<name>).
"""

from __future__ import annotations

import contextlib
import queue
import subprocess
import sys
import threading
from pathlib import Path

import mujoco as mj
import numpy as np

BUTTONS = ("Reset view", "Reset hands", "Reset instruments", "Lift heart out", "Load heart (STL)")
LIFT_BUTTON = "Lift heart out"
LIFT = np.array([0.0, -0.04, 0.20])      # where "lift out" takes the heart (from its place in the chest)
LIFT_SECONDS = 0.8
PUMP_RPM = 34.0
CAMERA_MARGIN = 0.15                     # keep the eye this far from walls, floor and ceiling


def button_xml(body1: str, body2: str) -> str:
    """<weld> lines for the panel buttons (put them inside the model's <equality>)."""
    return "".join(f'<weld name="{name}" body1="{body1}" body2="{body2}" active="false"/>' for name in BUTTONS)


def _quat_mul(a, b):
    out = np.zeros(4)
    mj.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _quat_mat(q):
    out = np.zeros(9)
    mj.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


class EnvironmentControls:
    def __init__(self, model: mj.MjModel, data: mj.MjData, heart_body: str = "heart"):
        self.model, self.data = model, data
        m = model
        # buttons
        self.buttons = {name: m.equality(name).id for name in BUTTONS
                        if mj.mj_name2id(m, mj.mjtObj.mjOBJ_EQUALITY, name) >= 0}
        for i in self.buttons.values():
            data.eq_active[i] = False
        # camera box
        nid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_NUMERIC, "or_room_box")
        box = m.numeric_data[m.numeric_adr[nid]:][:6] if nid >= 0 else np.array([-50, -50, 0, 50, 50, 50.0])
        self.lower, self.upper = box[:3] + CAMERA_MARGIN, box[3:] - CAMERA_MARGIN
        # heart (mocap)
        self.heart = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, heart_body)
        self.heart_mocap = int(m.body_mocapid[self.heart]) if self.heart >= 0 else -1
        if self.heart_mocap >= 0:
            self.home = (m.body_pos[self.heart].copy(), m.body_quat[self.heart].copy())
        self.lift = None                         # (start time, from pos, to pos) while animating
        self.selected = 0
        lines = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "or_bypass_lines")
        self.bypass_lines = [g for g in range(m.ngeom) if lines >= 0 and m.geom_bodyid[g] == lines]
        self.bypass_groups = m.geom_group[self.bypass_lines].copy()
        # environment instruments (free bodies named or_*)
        self.free_qpos = []
        for j in range(m.njnt):
            if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE and m.body(m.jnt_bodyid[j]).name.startswith("or_"):
                adr = m.jnt_qposadr[j]
                self.free_qpos.append(slice(adr, adr + 7))
        # animations
        self.screens = []                        # (target tid, frames tid, count, cycle)
        for n in range(m.nnumeric):
            name = m.numeric(n).name
            if not name.startswith("or_anim_"):
                continue
            tex = name.removeprefix("or_anim_")
            tid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_TEXTURE, tex)
            fid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_TEXTURE, tex + "_frames")
            if tid < 0 or fid < 0:
                continue
            count, cycle = m.numeric_data[m.numeric_adr[n]:][:2]
            size = int(m.tex_height[tid] * m.tex_width[tid] * m.tex_nchannel[tid])
            if int(m.tex_height[fid] * m.tex_width[fid] * m.tex_nchannel[fid]) != size * int(count):
                continue
            self.screens.append((tid, fid, int(count), float(cycle), size))
        self.shown = {}
        self.rollers = []
        for b in range(m.nbody):
            name = m.body(b).name
            if name.startswith("or_hlm_roller_") and m.body_mocapid[b] >= 0:
                self.rollers.append((m.body_mocapid[b], m.body_quat[b].copy(), int(name.split("_")[-1])))
        self.uploads = queue.SimpleQueue()       # update_texture blocks until drawn: do it off the loop
        self.viewer = None

    # --- per frame ----------------------------------------------------------------------------
    def frame(self, viewer, now: float) -> list[str]:
        """Handle buttons, camera, heart and animations. Returns the pressed button names."""
        if self.viewer is not viewer:
            self.viewer = viewer
            threading.Thread(target=self._upload_loop, args=(viewer,), daemon=True).start()
        pressed = []
        with viewer.lock():
            d = self.data
            for name, i in self.buttons.items():
                if name == LIFT_BUTTON:
                    continue
                if d.eq_active[i]:
                    d.eq_active[i] = False
                    pressed.append(name)
            self._heart(viewer.perturb, now)
            if "Reset instruments" in pressed:
                self.reset_instruments(now)
            self._instruments(viewer.perturb)
            self.keep_inside_room(viewer.cam)
            changed = self._animate(now)
            mj.mj_kinematics(self.model, d)
        for tid in changed:
            self.uploads.put(tid)
        return pressed

    def _upload_loop(self, viewer):
        while True:
            tid = self.uploads.get()
            try:
                if not viewer.is_running():
                    return
                viewer.update_texture(tid)
            except Exception:
                return

    def keep_inside_room(self, cam) -> None:
        """Clamp the free camera so the eye (lookat - distance * forward) stays inside the room."""
        if cam.type != mj.mjtCamera.mjCAMERA_FREE:
            return
        lookat = np.clip(np.array(cam.lookat), self.lower + 0.05, self.upper - 0.05)
        cam.lookat[:] = lookat
        az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
        back = -np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        reach = np.inf
        for axis in range(3):
            if back[axis] > 1e-9:
                reach = min(reach, (self.upper[axis] - lookat[axis]) / back[axis])
            elif back[axis] < -1e-9:
                reach = min(reach, (self.lower[axis] - lookat[axis]) / back[axis])
        if cam.distance > reach:
            cam.distance = max(reach, 0.02)

    # --- heart ----------------------------------------------------------------------------------
    @property
    def heart_out(self) -> bool:
        if self.heart_mocap < 0:
            return False
        pos, quat = self.data.mocap_pos[self.heart_mocap], self.data.mocap_quat[self.heart_mocap]
        return np.linalg.norm(pos - self.home[0]) > 1e-6 or np.abs(quat - self.home[1]).max() > 1e-6

    def _set_lift_button(self, on: bool):
        self.lift_state = on
        if LIFT_BUTTON in self.buttons:
            self.data.eq_active[self.buttons[LIFT_BUTTON]] = on

    def _lift_out(self, now):
        k = self.heart_mocap
        self.lift = (now, self.data.mocap_pos[k].copy(), self.home[0] + LIFT, None)
        self._set_lift_button(True)

    def _put_back(self, now):
        k = self.heart_mocap
        self.lift = (now, self.data.mocap_pos[k].copy(), self.home[0].copy(), self.data.mocap_quat[k].copy())
        self._set_lift_button(False)

    def _heart(self, pert, now):
        if self.heart_mocap < 0:
            return
        d, k = self.data, self.heart_mocap
        state = getattr(self, "lift_state", False)
        button = bool(d.eq_active[self.buttons[LIFT_BUTTON]]) if LIFT_BUTTON in self.buttons else state
        newly_selected = pert.select == self.heart and self.selected != self.heart
        self.selected = int(pert.select)
        if button and not state:                             # panel: lift out
            self._lift_out(now)
        elif state and not button:                           # panel: put back
            self._put_back(now)
        elif newly_selected and not self.heart_out and self.lift is None:   # clicking the heart lifts it
            self._lift_out(now)
        if pert.active and pert.select == self.heart:        # Ctrl + drag moves / turns it
            self.lift = None
            mj.mjv_applyPerturbPose(self.model, d, pert, 0)
            self._set_lift_button(True)
        elif self.lift is not None:
            t0, a, b, q = self.lift
            u = min(1.0, (now - t0) / LIFT_SECONDS)
            u = u * u * (3 - 2 * u)
            d.mocap_pos[k] = a + (b - a) * u
            if q is not None:                                 # turn back while it goes home
                q1 = self.home[1] if np.dot(q, self.home[1]) >= 0 else -self.home[1]
                qu = (1 - u) * q + u * q1
                d.mocap_quat[k] = qu / np.linalg.norm(qu)
            if u >= 1.0:
                self.lift = None
                if q is not None:
                    d.mocap_pos[k], d.mocap_quat[k] = self.home
        out = self.heart_out
        for g, group in zip(self.bypass_lines, self.bypass_groups):   # cannulas stay in place: hide them
            self.model.geom_group[g] = 5 if out else group

    @contextlib.contextmanager
    def heart_at_home(self, viewer):
        """Holds the viewer lock; inside, the heart is in its place in the chest, so the simulator's
        ray-cast tissue (cutting, stitches, bypass) works in heart coordinates; drawing done inside
        is moved with the heart afterwards by follow_heart()."""
        with viewer.lock():
            if not self.heart_out:
                self.moved = None
                yield
                return
            k = self.heart_mocap
            pose = self.data.mocap_pos[k].copy(), self.data.mocap_quat[k].copy()
            self.data.mocap_pos[k], self.data.mocap_quat[k] = self.home
            mj.mj_kinematics(self.model, self.data)
            try:
                yield
            finally:
                self.data.mocap_pos[k], self.data.mocap_quat[k] = pose
                mj.mj_kinematics(self.model, self.data)
                self.moved = pose

    def follow_heart(self, scene, start: int, end: int) -> None:
        """Move user-scene geoms [start, end) (wounds, stitches, graft) from the heart's place in
        the chest to where it is now."""
        if getattr(self, "moved", None) is None:
            return
        pos, quat = self.moved
        home_pos, home_quat = self.home
        conj = home_quat * np.array([1, -1, -1, -1])
        rot = _quat_mat(_quat_mul(quat, conj))
        for i in range(start, min(end, scene.ngeom)):
            g = scene.geoms[i]
            g.pos[:] = pos + rot @ (np.array(g.pos) - home_pos)
            g.mat[:] = rot @ np.array(g.mat).reshape(3, 3)

    # --- instruments ----------------------------------------------------------------------------
    def _instruments(self, pert):
        sel = int(pert.select)
        if sel <= 0 or sel == self.heart or not pert.active:
            return
        m, d = self.model, self.data
        root = int(m.body_rootid[sel])
        if root != sel and m.body_jntnum[root] > 0:          # a jaw half: grab the whole instrument
            world = d.xpos[sel] + d.xmat[sel].reshape(3, 3) @ np.array(pert.localpos)
            pert.select = root
            pert.localpos[:] = d.xmat[root].reshape(3, 3).T @ (world - d.xpos[root])
        if m.body_jntnum[root] > 0 and m.body(root).name.startswith("or_"):
            mj.mjv_applyPerturbPose(m, d, pert, 1)

    def reset_instruments(self, now=0.0):
        """Environment instruments back on the Mayo stand, heart back in the chest."""
        for s in self.free_qpos:
            self.data.qpos[s] = self.model.qpos0[s]
        if self.heart_mocap >= 0 and (self.heart_out or self.lift is not None):
            self._put_back(now)

    # --- animations -----------------------------------------------------------------------------
    def _animate(self, now: float) -> list[int]:
        m, changed = self.model, []
        for tid, fid, count, cycle, size in self.screens:
            k = int((now % cycle) / cycle * count) % count
            if self.shown.get(tid) != k:
                src = m.tex_adr[fid] + k * size
                m.tex_data[m.tex_adr[tid]:m.tex_adr[tid] + size] = m.tex_data[src:src + size]
                self.shown[tid] = k
                changed.append(tid)
        for mocap, base, k in self.rollers:                   # heart-lung machine roller pumps
            angle = 2 * np.pi * PUMP_RPM / 60.0 * now * (1 if k % 2 else -1) + k
            mj.mju_mulQuat(self.data.mocap_quat[mocap], base,
                           np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)]))
        return changed


# --- loading a heart model --------------------------------------------------------------------

def choose_heart_file() -> str | None:
    """Ask for a heart model file (STL or OBJ) with the system's file dialog; None if cancelled.
    Runs the dialog in a separate process so it works next to the MuJoCo window (also mjpython)."""
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["osascript", "-e",
                                  'POSIX path of (choose file with prompt "Choose a heart model (STL or OBJ)")'],
                                 capture_output=True, text=True, timeout=600)
        else:
            code = ("import tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw(); "
                    "r.attributes('-topmost', True); print(filedialog.askopenfilename(title='Choose a heart model',"
                    " filetypes=[('Heart model', '*.stl *.STL *.obj *.OBJ'), ('All files', '*.*')]))")
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.stdout.strip()
    return path if out.returncode == 0 and path and Path(path).is_file() else None


def read_mesh(path) -> tuple[np.ndarray, np.ndarray]:
    """(vertices, triangles) of a binary/ASCII STL or an OBJ file."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix.lower() == ".obj":
        verts, faces = [], []
        for line in raw.decode("utf-8", "ignore").splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(v) for v in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) for p in parts[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
        return np.array(verts, float), np.array(faces, int)
    count = int.from_bytes(raw[80:84], "little") if len(raw) >= 84 else 0
    if len(raw) == 84 + 50 * count and count > 0:            # binary STL
        rec = np.frombuffer(raw[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        tri = rec["v"].astype(float)
    else:                                                    # ASCII STL
        nums = [line.split()[1:4] for line in raw.decode("utf-8", "ignore").splitlines()
                if line.strip().startswith("vertex")]
        tri = np.array(nums, float).reshape(-1, 3, 3)
    verts, inverse = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return verts, inverse.reshape(-1, 3)


def heart_obj(path, center) -> bytes:
    """The heart file as OBJ text in the heart body's frame: millimetres become metres and its
    bounding-box center goes to `center` (axes as in the file: x anterior, y left, z superior)."""
    verts, faces = read_mesh(path)
    if np.ptp(verts, axis=0).max() > 1.0:                    # in millimetres
        verts = verts * 0.001
    verts = verts - (verts.min(0) + verts.max(0)) / 2 + np.asarray(center, float)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    return ("\n".join(lines) + "\n").encode()


def obj_center(files) -> np.ndarray:
    """Bounding-box center of the vertices in OBJ files (bytes)."""
    pts = [line.split()[1:4] for data in files for line in data.decode("utf-8", "ignore").splitlines()
           if line.startswith("v ")]
    pts = np.array(pts, float)
    return (pts.min(0) + pts.max(0)) / 2
