# Integrating the Operating-Room Environment

The cardiac open-chest operating room comes in two forms. Both contain the same model, because the XML is generated from the Python code. Everything is plain MuJoCo (tested with **4.0.1**) + numpy + Pillow. There is no hand, glove or movement code in it.

| You get | Use it if |
|---|---|
| `environment/` (`environment.xml` + `assets/`) | your scene is MJCF XML (like the `LAB` string in `surgical_pickup.py`) |
| `surgical_env/` (Python package) | you build or extend your model in Python with `mujoco.MjSpec` |

Working example of all variants with your `surgical_pickup.py` scene: `examples/with_reference_scene.py`.

---

## Option A: MJCF folder, one `<include>` line

Copy `environment/` next to your code and add one line to your model, **after** your own `<visual>` block (the included file brings its own lighting/visual settings, and the later one wins):

```xml
<mujoco model="Surgical training station">
  <visual> ... </visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <camera name="bench" .../>
  </worldbody>
  <include file="environment/environment.xml"/>   <!-- the whole operating room -->
</mujoco>
```

Remove your old table, pad, tray, lamp and monitor geoms; the environment replaces them. For `from_xml_string`, give an absolute path (e.g. `Path("environment/environment.xml").resolve().as_posix()`).

Two things to know:
- `environment.xml` sets `<compiler angle="radian"/>`, and with `<include>` that applies to your whole model. Your current scene has no angles (it uses `xyaxes`, `dir`, `fromto`), so nothing changes. If you later write angles in degrees (`euler`, `axisangle`, joint `range`), use **Option A2**.
- Your `<default>` classes cannot switch off our collisions: our colliding geoms use `contype/conaffinity = 3`, which still collides with ordinary geoms (bit 1).

### Option A2: `<attach>` (keeps your compiler settings separate)
```xml
<asset> <model name="or_env" file="environment/environment.xml"/> </asset>
<worldbody> <attach model="or_env" prefix=""/> ... </worldbody>
```
MuJoCo prints two harmless warnings (empty prefix; `znear` conflict, where your value is kept).

---

## Option B: Python, `build_environment()`

Put `surgical_env/` on your Python path (copy the folder next to your script).

```python
import mujoco
from surgical_env import build_environment, constants as C

spec = build_environment()                 # the operating room as a mujoco.MjSpec
# add your own things:
spec.worldbody.add_camera(name="bench", pos=[.28, -.62, 1.43], xyaxes=[.925, .380, 0, -.202, .492, .847])
hand = spec.worldbody.add_body(name="hand_root", mocap=True, pos=list(C.ANCHOR))
# ...or bring in your whole existing XML scene:
# spec.attach(mujoco.MjSpec.from_string(LAB), frame=spec.worldbody.add_frame(), prefix="lab_")

model = spec.compile()
data = mujoco.MjData(model)
```

Options: `build_environment(include_placeholder_heart=True, quality="high" | "medium" | "low")`. Use `low` on integrated graphics (1 shadow-casting light instead of 2, smaller textures).
Angles in the spec are **radians** (`spec.compiler.degree = False`).

---

## Contract (what you can rely on)

**Coordinates** (meters, z up): surgeon on the **-y** side looking toward **+y**, **+x = surgeon's right**; patient supine along x, **head at -x**, feet at +x. Same as your prototypes.

| Constant (`surgical_env.constants`) | Value | Site in the model |
|---|---|---|
| `FIELD_CENTER` (sternotomy, skin level) | `(0, 0.05, 0.955)` | `or_field_center` |
| `ANCHOR` (your hand anchor) | `(0, -0.10, 1.015)` | `or_anchor` |
| `HEART_MOUNT` (heart bounding-box center) | `(0, 0.05, 0.885)` | `or_heart_mount` |
| `MAYO_CENTER` (instrument tray surface) | `(0.52, 0.05, 1.08)` | `or_mayo_center` |
| `TIMESTEP` | `0.002` s | `<option timestep>` |

Sites are in group 5 (hidden in the viewer by default).

**Names**: every element starts with `or_`, so nothing clashes with your names.

**Cameras**: `or_surgeon` (default), `or_overview`, `or_topdown`, `or_far_side`, `or_closeup_heart`, `or_closeup_mayo`.

**Room and equipment**: deliberately reduced to what a cardiac case needs near the table. All static bodies: `or_room`, `or_light_head_1/2`, `or_boom_central`, `or_heart_lung_machine`, `or_anesthesia_machine`, `or_anesthesia_pendant` (with `or_patient_monitor`), `or_surgical_pendant` (ESU, suction regulators and canisters), `or_iv_pole`. To speed up rendering, static decoration is merged into meshes named `<body>_mesh_<material>`. Screens stay separate geoms (e.g. `or_patient_monitor`, `or_boom_display`).
**Sterile field** (step 4, static): `or_mayo_stand` (tray over the legs, top of the blue towel at z = 1.088), `or_back_table` (at the foot: trays, basins, sponge packs, sutures, sternal wires, needle counter, sternal saw, internal defibrillator paddles), `or_ring_stand`, `or_kick_bucket`, `or_esu_holster` (pencil with CUT/COAG buttons, cord to the ESU). The ESU pencil and the Yankauer suction tip are static because their cords are fixed; make them free bodies if you need to pick them up.
**Lights**: `or_light_head_1_spot` and `or_light_head_2_spot` cast shadows (both on high, head 1 on medium, none on low), plus `or_headlamp` and `or_fill_1..3`. The visual settings (camera headlight dimmed, `shadowscale 1.5`, free-camera start pose) come with the environment. With `<include>` they replace yours, so put your `<visual>` block **after** the include line if you want to keep your own.
**Performance**: shadow-casting lights are the main cost. Measured on an Intel Iris Plus at 1280x800: low ≈ 55–75 FPS, medium ≈ 30, high ≈ 12. A dedicated GPU is much faster.

**Patient and field** (step 3): `or_operating_table`, `or_patient_drape` (one drape mesh with the field window), `or_anesthesia_screen` (curtain at x ≈ -0.42), `or_operative_field` (incise film, wound wall layers, pericardial well), `or_sternal_retractor`, `or_lap_pads`, `or_stay_sutures`, `or_cables` (bypass lines, IV lines, suction tubing, breathing circuit, ESU cord).
**Collisions**: walls, ceiling and floor collide, and the draped patient is approximated by hidden boxes `or_collision_*` (group 3), so dropped things rest on it. The field window has no box: things fall into the chest and rest on the heart's collision volume.
Your hands rest naturally at drape level: the drape top is at z ≈ 0.95–0.97 around the field, and at `ANCHOR` z = 1.015 your hand floats just above it.

## Heart (patient-specific)

The placeholder `or_heart_placeholder` sits with its bounding-box center on `or_heart_mount`. Replace it with the CT-reconstructed mesh:

```python
from surgical_env import build_environment, replace_heart
spec = build_environment()
replace_heart(spec, "patient_heart.stl", scale=0.001, orientation="LPS")   # mm, DICOM/ITK axes
model = spec.compile()
```
XML users can export a ready folder instead: `tools\export.py --heart-mesh patient_heart.stl --heart-frame LPS --out environment_patient`.

| Parameter | Meaning |
|---|---|
| `mesh_path` | `.stl` (binary/ASCII) or `.obj` surface mesh of the heart (and vessels) |
| `scale` | file units → meters. CT exports are usually **mm → 0.001** (default). The size is checked: it must come out 3–30 cm. |
| `orientation` | `"LPS"` (DICOM patient axes: x→Left, y→Posterior, z→Superior; ITK, most CT tools), `"RAS"` (3D Slicer default), `"world"`, or a 3×3 matrix |
| `collision` | `"hull"` (default, convex hull), `"box"` or `"none"` |
| `max_faces` | simplify to at most this many triangles (default 60 000; `None` keeps all) |

Placement: the mesh's **bounding-box center** goes to `or_heart_mount`, so absolute scanner coordinates don't matter. The new body is `or_heart_patient` (visual mesh plus hidden collision `or_heart_patient_collision`). The bypass cannulas in `or_cables` stay where the placeholder's aorta and right atrium were.

## Instruments (pickable)

16 free bodies, built to real dimensions. Fifteen lie on the Mayo towel in cardiac order (near → far from the surgeon) and the 4x4 gauze lies on the back table. No actuators, no hand code: grasping and driving the jaws is up to you.

- **Body**: `or_<name>` with free joint `or_<name>_free`. Visual meshes have no collision and no mass; hidden capsules (group 3) carry the mass and collide (`contype=conaffinity=3`, friction 0.8, condim 4).
- **Jaw joint**: `or_<name>_jaw` (hinge) on a child body (`or_<name>_jaw_half` for ring instruments, `or_<name>_upper_arm` for forceps-type). **Angle = opening in radians, 0 = closed**, range-limited, with a gentle spring:
  - Ring instruments (scissors, needle holder, clamps) rest closed. Hinge about the instrument's z axis at the box lock; spring 0.05 N·m/rad.
  - DeBakey and Castroviejo rest open (0.06 / 0.09 rad). Real forceps spring 1 N·m/rad, so squeezing the tips about 1–2 N closes them.
  - Bulldog clamps are sprung shut (preload).
  - Add your own actuator or drive `qpos` directly. The ratchet lock is not modelled.
- **Sites** (group 4, hidden): `_tip`, `_grip` on every instrument. Ring instruments add `_ring_finger` (body half) and `_ring_thumb` (moving half); forceps-type add `_tip_upper` (moving arm).
- **Frame**: length along local +x (tip at +x), lying flat, z up. On the Mayo the tips point toward the patient's feet and the rings toward the field.
- **Physics** (checked by `tools\physics_check.py`): every instrument settles within 1.5 mm and comes to rest when dropped from 20 cm onto the tray or the drape (no tunnelling). Timestep 2 ms.

| Instrument | Body | Mass | Jaw joint (range, rest) | Sites (`or_<name>_…`) |
|---|---|---|---|---|
| Scalpel #3 handle + #10 blade | `or_scalpel` | 25 g | – | grip, tip |
| DeBakey forceps 20 cm (×2) | `or_debakey_1`, `or_debakey_2` | 30 g | `…_jaw` 0–0.12, rest 0.06 | tip, tip_upper, grip |
| Metzenbaum scissors 18 cm | `or_metzenbaum` | 45 g | `or_metzenbaum_jaw` 0–0.50, rest 0 | tip, grip, ring_finger, ring_thumb |
| Potts scissors 18 cm (angled) | `or_potts` | 40 g | `or_potts_jaw` 0–0.40, rest 0 | tip, grip, ring_finger, ring_thumb |
| Mayo-Hegar needle holder 16 cm (TC, gold rings) | `or_needle_holder` | 65 g | `or_needle_holder_jaw` 0–0.30, rest 0 | tip, grip, ring_finger, ring_thumb |
| Castroviejo needle holder 18 cm (titanium) | `or_castroviejo` | 35 g | `or_castroviejo_jaw` 0–0.16, rest 0.09 | tip, tip_upper, grip |
| Satinsky clamp 25 cm (side-biting) | `or_satinsky` | 90 g | `or_satinsky_jaw` 0–0.35, rest 0 | tip, grip, ring_finger, ring_thumb |
| DeBakey aortic cross-clamp 24 cm | `or_cross_clamp` | 85 g | `or_cross_clamp_jaw` 0–0.35, rest 0 | tip, grip, ring_finger, ring_thumb |
| Kelly hemostat 14 cm (×2) | `or_kelly_1`, `or_kelly_2` | 45 g | `…_jaw` 0–0.35, rest 0 | tip, grip, ring_finger, ring_thumb |
| Bulldog clamp 5 cm (×2) | `or_bulldog_1`, `or_bulldog_2` | 12 g | `…_jaw` 0–0.35, rest 0 (sprung shut) | tip, tip_upper, grip |
| Lap sponge, rolled (×2) | `or_lap_1`, `or_lap_2` | 22 g | – | grip, tip |
| 4x4 gauze (back table) | `or_gauze` | 10 g | – | grip, tip |

Instrument masses, sizes, springs and the layout are in `surgical_env/instruments.py` (`RING`, `SPRING`, `MAYO_LAYOUT`).

---

## Regenerating `environment/`
In the environment repository: `tools\export.py` (or `.\view.cmd` / `.\snapshot.cmd`, which export first). The export checks that the XML compiles on its own and matches `build_environment()`.
