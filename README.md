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
