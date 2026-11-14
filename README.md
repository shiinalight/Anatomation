# Surgical Training Simulator

A MuJoCo hackathon demo with a Rokoko-controlled five-finger hand,
interactive surgical instruments, and guided suturing.

## Run the integrated dashboard

On Windows, connect one Rokoko Smartglove over USB and run from this directory:

    py -3.11 -m pip install -r requirements-mvp.txt
    py -3.11 surgical_suturing_guided.py --mvp

Add `--site heart` for heart practice or `--preview` for keyboard practice.
On macOS, run the script with `mjpython` after installing the requirements.
The live launcher starts the installed Rokoko driver and OpenXR solver when
their local ports are idle.

Open **Equality** in the native MuJoCo right sidebar. Click **AI Voice** to
toggle the microphone and ask Gemini about the exercise. Click **Review scene**
to get Gemma feedback. Click **AI Assistant** to open a small window for the
current cue, replies, and typed questions. You can close and reopen it while the
training scene stays open. **V** and **B** are shortcuts for voice and review.

Click **Save attempt** (**J**) to save your work and start a fresh attempt.
**AI history** loads recent attempts for the current input mode into the
assistant window. Click **Next attempt** to select one and ask Gemini in the
window or with **AI Voice**. Turn off **AI history** to return to live coaching.
**R** saves and resets the exercise.

For a new installation, set `SURGERY_GEMMA_API_KEY` and `AIVEN_DATABASE_URL`
in a local `.env` using `.env.example`, and save the Aiven CA certificate as
`aiven-ca.pem`. Attempts are queued locally and synchronized to PostgreSQL.
Stored records include the simulator site, input mode, actions, and virtual
tool-tip samples. Existing attempts are still available by input mode.
Use `--ai` for the native coach controls without session recording.

## Run on macOS

Create and activate a Python virtual environment, then install the dependencies.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Try it without a glove:

    mjpython surgical_suturing_guided.py --preview

Choose what to operate on with `--site` (works with or without `--preview`):

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
| Camera | Scroll to zoom, left-drag to orbit, right-drag to pan, 0 resets the view |
| Glove | H recalibrates: rest your hand palm-down, fingers toward the monitor, then press H |
| Sound | M mutes or unmutes the OR sounds (macOS) |

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
