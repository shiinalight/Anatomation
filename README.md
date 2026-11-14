# Surgical Training Simulatorr

A MuJoCo hackathon demo with a Rokoko-controlled five-finger hand,
interactive surgical instruments, and guided suturing.

## Run on macOS

Create and activate a Python virtual environment, then install the dependencies.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Try it without a glove:

    mjpython surgical_suturing_guided.py --preview

Switch between sites inside the simulator with the **Skin** / **Heart** / **Bypass** buttons
in the **Equality** section of the viewer's right-hand panel (or press **P**). `--site` picks where to
start (works with or without `--preview`):

    mjpython surgical_suturing_guided.py --site skin     # skin pad (default)
    mjpython surgical_suturing_guided.py --site heart    # heart in an open chest
    mjpython surgical_suturing_guided.py --site bypass   # coronary artery bypass graft

For live tracking, start the Rokoko Device SDK driver in one Terminal:

    rokoko-sdk --auto-stream-usb

In another Terminal, start the hand solver:

    rkk-hand-solver --no-driver --emit-openxr

Then launch the simulator from this directory:

    mjpython surgical_suturing_guided.py

## Controls

| Action | Keys / mouse |
|---|---|
| Pick an instrument | 1 scalpel, 2 needle holder, 3 forceps, 4 scissors, then G to pick up / set down |
| Move the hand | Arrows and Q/E (tap for a small step, hold to glide) |
| Suturing | N aligns the tool, F punctures or grips tissue, T ties, X trims, R clears the pad |
| Camera | Two-finger scroll to zoom, left-drag to orbit, right-drag to pan, double-right-click a spot to zoom around it, 0 resets the view |
| Glove | H recalibrates: rest your hand palm-down, fingers toward the monitor, then press H |
| Sound | M mutes or unmutes the OR sounds (macOS) |
| Operating site | Skin / Heart / Bypass buttons in the right panel's Equality section, or press P to go to the next one |
| Key guide | Shown in a bar at the bottom right, next to the status box; K or the Key guide button in the Equality section hides it |
| Start over | Clear repair button in the same Equality section: removes all cuts, blood and stitches and puts the tools back in the trays (R clears cuts and stitches only) |

## Operating sites

**Skin** (`--site skin`, default): a betadine-prepped skin window in the drapes, with a
15 cm laceration, violet marker lines and five stitches 32 mm apart.

**Heart** (`--site heart`): a heart in an open chest (median sternotomy with a
retractor). The heart is the NIH 3D model in `assets/heart-nih3d/`, loaded from its
own `heart-nih3d.xml` so its meshes keep their relative paths and its materials keep
the file's own colours (pink muscle, red arteries, blue veins), converted from Blender's
linear colour space so they look the same as in Blender.

The repair is a 3 cm laceration on the front wall of the left ventricle, placed by
ray-casting the model where it stays at least 7 mm from any visible vessel. There are
five stitch targets (white dots), 6 mm apart, with 6 mm bites. Cutting, needle
puncture and forceps grip follow the curved heart surface. Placing the blade or
needle on a vessel gives a warning instead.

**Bypass** (`--site bypass`): a guided coronary artery bypass graft on the same heart.
The status bar shows the current step and the next key to press.

| Step | Tool | How |
|---|---|---|
| 1. Cut the coronary artery | Scalpel (1, G) | N puts the blade on the white marked line, E lowers it, left / right arrows cut along the line |
| 2. Suture the graft onto the coronary | Needle holder (2, G) | Six stitches around the opening: N aligns on the green target, E E lowers, F passes through the graft, then N, E E, F through the artery, T ties |
| 3. Incise the ascending aorta | Scalpel | Same as step 1, on the aorta |
| 4. Suture the graft onto the aorta | Needle holder | Same as step 2; when the last stitch is tied the graft fills with blood |

The marked lines were placed by ray-casting the heart model: the coronary cut lies on
the straight arm of the visible coronary artery, the aortotomy on the top of the
ascending aorta. Cutting any other vessel still gives a warning.

## Files

- `surgical_suturing_guided.py`: glove tracking, instruments and the suturing logic.
- `or_scene.py`: the operating room (layout, heart model import, heart-surface queries,
  generated textures, lighting, patient monitor, sounds).
- `bypass.py`: the guided coronary bypass procedure (targets, steps, graft).
- `assets/heart-nih3d/`: the heart model (MJCF, three OBJ meshes, MTL), unchanged.
