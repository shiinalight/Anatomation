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

Switch between skin and heart inside the simulator with the **Skin** / **Heart** buttons
in the **Equality** section of the viewer's right-hand panel (or press **P**). `--site` picks where to
start (works with or without `--preview`):

    mjpython surgical_suturing_guided.py --site skin     # skin pad (default)
    mjpython surgical_suturing_guided.py --site heart    # heart in an open chest

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
| Operating site | Skin / Heart buttons in the right panel's Equality section, or press P |

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

## Files

- `surgical_suturing_guided.py`: glove tracking, instruments and the suturing logic.
- `or_scene.py`: the operating room (layout, heart model import, heart-surface queries,
  generated textures, lighting, patient monitor, sounds).
- `assets/heart-nih3d/`: the heart model (MJCF, three OBJ meshes, MTL), unchanged.
