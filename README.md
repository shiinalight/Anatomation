# Surgical Training Simulator

A MuJoCo hackathon demo with a Rokoko-controlled five-finger hand,
interactive surgical instruments, and guided suturing.

## Run on macOS

Create and activate a Python virtual environment, then install the dependencies.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Try it without a glove:

    mjpython surgical_suturing_guided.py --preview

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

## Files

- `surgical_suturing_guided.py`: glove tracking, instruments and the suturing logic.
- `or_scene.py`: the operating room (layout, generated textures, lighting, patient monitor, sounds).
