# Surgical Training Simulator: guide

A MuJoCo hackathon demo with two Rokoko-controlled hands, interactive surgical
instruments, and guided two-handed cutting and suturing.

## Run on macOS

Create and activate a Python virtual environment, then install the dependencies.

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Try it without gloves (both hands are keyboard-driven):

    mjpython surgical_suturing_guided.py --preview

Choose the procedure in the **Procedures** window, a dropdown that opens next to the
simulator (the Procedures button in the viewer's Equality panel reopens it), or press **P** to
go to the next one. `--procedure` picks where to start:

    mjpython surgical_suturing_guided.py --procedure skin-incision    # cut along a marked line
    mjpython surgical_suturing_guided.py --procedure skin-suture      # repair the laceration (default)
    mjpython surgical_suturing_guided.py --procedure heart-incision   # incision on the left ventricle
    mjpython surgical_suturing_guided.py --procedure heart-suture     # heart laceration
    mjpython surgical_suturing_guided.py --procedure heart-bypass     # coronary artery bypass graft
    mjpython surgical_suturing_guided.py --procedure skin-stitch      # free stitching, prompted (also heart-stitch)
    mjpython surgical_suturing_guided.py --procedure skin-gloves      # hands-only suturing: gloves, no keys
    mjpython surgical_suturing_guided.py --procedure heart-gloves-bypass  # hands-only bypass: gloves, no keys
    mjpython surgical_suturing_guided.py --procedure skin-free        # free practice (also heart-free)

For live tracking, start the Rokoko Device SDK driver in one Terminal:

    rokoko-sdk --auto-stream-usb

In another Terminal, start the hand solver:

    rkk-hand-solver --no-driver --emit-openxr

Then launch the simulator from this directory:

    mjpython surgical_suturing_guided.py

Connect both gloves. The simulator reads the solver's stream directly (port 12277) and
tells the hands apart by the side each glove reports. If one glove is not streaming,
a keyboard-driven hand takes its place, so you can also work with one glove.

## Controls

| Action | Keys / mouse |
|---|---|
| Which hand | Double-click a hand to move it with the arrows and Q/E (its cuff turns green). The keys also follow the hand the next step needs; Z swaps to the other hand |
| Pick an instrument | 1 scalpel, 2 needle holder, 3 forceps, 4 scissors, then G to pick up / set down |
| Move the hand | Arrows and Q/E (tap for a small step, hold to glide) |
| Suturing | N aligns the tool, F grips with the forceps (or presses with an empty hand) or passes the needle, T throws a knot, X trims, R clears the pad |
| Camera | Two-finger scroll to zoom, left-drag to orbit, right-drag to pan, double-right-click a spot to zoom around it, 0 resets the view |
| Gloves | H recalibrates: press H, then rest your hands palm-down, fingers toward the monitor |
| Sound | M mutes or unmutes the OR sounds (macOS) |
| Procedure | Procedures window (dropdown), or P for the next one. The checklist at the top left shows every step: [x] done, [>] in progress, [ ] still to come, and the last actions completed |
| Both hands | Y, or the Both hands button (Equality section): the arrows and Q/E move both hands together (both cuffs green); press again to go back to one hand |
| Reset hands | Reset hands button (Equality section): both hands go back to their start in front of you, keeping their tools; the camera stays where it is |
| Room | Reset view (camera back to the surgeon's view; the hands stay), Reset instruments (every tool back in its tray, the repair stays), Lift heart out (or double-click the heart; Ctrl + right-drag moves it, Ctrl + left-drag turns it), Load heart (STL) or `--heart file.stl` (another heart model). All in the Equality section |
| Key guide | Hidden at start; K or the Key guide button in the Equality section shows it at the bottom right, next to the status box |
| Start over | R, or the Clear repair button in the same Equality section: the attempt so far is scored and saved, then the site starts fresh with the tools back in the trays |
| Autopilot | Autopilot button (Equality section): replays the best recorded attempt at this site; turn it off to take over |
| AI coach | I opens the assistant window (typed questions), V turns the voice coach on or off, B asks Gemma for a review (also buttons in the Equality section) |

## Hands-only modes: wrist movements only

The gloves report how the hand turns and how the fingers bend, not where the arm is: the
wrist stays where it is. So in the two hands-only modes below, every step is done by turning
the wrist, never by moving the arm or pressing keys:

- Nothing starts until the right glove is live and its direction is locked (rest the hand flat,
  fingers to the screen). If the glove drops out, the steps pause and continue when it is back.
- At the start of every step the tool (and the graft, when carried) glides to just above the
  next yellow point while the prompt says "Relax your wrist: moving into place".
- Then: bend the wrist down to catch the point, turn it a little to cut or push the needle,
  bend it up to pull the thread. A wrist-only test trainee finishes both modes with at most
  about 8 degrees side to side and 12 degrees up or down.

## Hands-only suturing (gloves, no keys)

**Skin: hands-only suturing** (`--procedure skin-gloves`) is the telemetry showcase: the cut on
the chest is already there, the needle holder is in the right hand and the forceps in the left,
both just above the cut. Six stitches go along the cut, and no key is needed:

1. The next hole glows yellow, with a yellow catch zone (3 cm around it, 2.5 cm high). Lower
   the needle toward it: within 7 cm sideways a magnet draws the tip over the hole, and in the
   catch zone it snaps in and the needle goes in.
2. Move the right hand a little (2.5 cm, any direction; across the cut is the natural one):
   the needle follows its curve under the skin and comes out of the far hole (yellow dots
   show the way; the bottom line shows the percentage).
3. Lift the right hand 3 cm: the thread is pulled through and the stitch is tied.

Then the next hole lights up; after the sixth stitch the showcase is complete. The bottom
line always says the movement to make and how far the needle is from the hole. When a glove
comes in or its direction locks, the tools go back above the cut. Recordings, scoring and the
autopilot work as for the other procedures (code: `glove_suture.py`).

## Hands-only bypass (gloves, no keys)

**Heart: hands-only bypass** (`--procedure heart-gloves-bypass`): the coronary bypass with snap
points, done with the right hand alone and no key:

1. **Cut the coronary artery**: the scalpel is in hand; a yellow point marks where the blade
   goes, and the whole marked line turns yellow as the blade comes close. Lower the blade onto
   the point (a magnet draws it over; it snaps in the yellow catch zone), then move the hand
   a little: the blade runs along the line and opens the artery.
2. **Pick up the graft**: the scalpel goes back to the tray and the hand lifts off the heart;
   the end of the vein graft glows yellow. Reach for it: it snaps and glides into the hand.
3. **Attach it to the coronary artery**: carry it to the yellow point on the opening; it snaps
   on and is sewn on, stitch by stitch.
4. **Cut the ascending aorta**: the scalpel is back in hand; the same as step 1.
5. **Pick up the free end of the graft** (yellow).
6. **Attach it to the aorta**: it is sewn on and the graft fills with blood and pulses.

A snap point only catches after the hand has come toward it, never where the previous step
left the hand. Code: `glove_bypass.py` (geometry from `bypass.py`).

## Two-handed technique

The right hand cuts and drives the needle; the left hand works against it, as in theatre.
The status bar always names the hand and the key for the next step.

- **Cutting**: the left hand steadies the tissue next to the blade (an empty hand stretches
  the skin, or the forceps hold it). The blade does not open unsteadied tissue. Free practice
  and the bypass cuts need no steadying hand.
- **Each stitch**:
  1. Left forceps lift the first edge (N, F).
  2. Right needle holder passes the needle through it (N, E E, F).
  3. Left forceps lift the opposite edge (N, F).
  4. Needle through the opposite edge (N, E E, F).
  5. Left forceps grasp the needle tip (N, F) and pull the suture through (Q Q Q).
  6. Instrument tie with the needle holder: three throws (T T T): a double wrap, a single
     wrap the other way, and a single wrap that squares the knot.
- Each needle pass needs a fresh forceps grip on the edge it goes through. Dropping the
  forceps before the knot lets the needle slip back.

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
| 1. Cut the coronary artery | Right scalpel | N puts the blade on the white marked line, E lowers it, left / right arrows cut along it (no steadying hand needed) |
| 2. Suture the graft onto the coronary | Right needle holder, left forceps | Six stitches around the opening, each as above: forceps lift the graft, needle through the graft (green target), forceps lift the artery, needle through the artery, forceps pull the suture through, T T T |
| 3. Incise the ascending aorta | Right scalpel | Same as step 1, on the aorta |
| 4. Suture the graft onto the aorta | Right needle holder, left forceps | Same as step 2; when the last stitch is tied the graft fills with blood |

The marked lines were placed by ray-casting the heart model: the coronary cut lies on
the straight arm of the visible coronary artery, the aortotomy on the top of the
ascending aorta. Cutting any other vessel still gives a warning.

## Recording, scoring, autopilot and AI coach

Every attempt (from a fresh site until it is complete, or until R, Clear repair or a site
switch) is recorded at 60 frames per second: both gloves' joints, every key, and what each
instrument did. When it ends it is scored and saved.

- **Scoring** (`motion_log.py`): the attempt is split into its tasks (each cut and each stitch).
  Each task is scored out of 100 on time, instrument travel, refused actions and needle
  accuracy. The status box shows the last and best scores.
- **Ideal run and autopilot**: the best complete attempt at a site is its ideal run. The
  **Autopilot** button replays it into a fresh site, exactly: the same cuts, needle passes and
  knots, with idle time shortened. Starting the autopilot also writes the instrument trajectory
  for a two-arm robot to `training-data/robot/<site>-<attempt>.json`: per frame and hand, the
  tool, tip position, direction and whether it is closed.
- **Aiven** (`aiven_store.py`): attempts are saved in `training-data/` first and then synced in
  the background to **Aiven PostgreSQL**:
  - `sim_attempts`, `sim_tasks` and `sim_recordings` hold the attempts, task scores and recordings;
  - the `sim_ideal_runs` view gives each site's best run, from any trainee, which is downloaded
    for the autopilot;
  - the `sim_task_best` view gives the per-task benchmarks the coach compares against.

  Live instrument telemetry (10 Hz) and step events stream to **Aiven Kafka** (`surgery-telemetry`,
  `surgery-events`). `python telemetry_monitor.py` follows a session from another computer,
  for example an instructor station or a robot bridge.
- **Gemini** (`gemini_voice.py`, `coach.py`) answers spoken or typed questions from the live
  state: the next step and hand, recent events, and the last attempt against the best results.
  Answers are read aloud.
- **Gemma** (`gemma_review.py`) writes a short debrief from a rendered frame and the task scores
  after each complete attempt, or when you press B. It runs on **Modal** with open weights
  (`modal_gemma.py`), or on Google AI Studio when Modal is not set up.

Set-up: `pip install -r requirements-ai.txt`, then copy `.env.example` to `.env` and fill in
what you have (AI Studio key, Modal URL and token, Aiven PostgreSQL URI, Kafka bootstrap).
Put the Aiven CA certificate in `aiven-ca.pem` and the Kafka `ca.pem`, `service.cert` and
`service.key` in `kafka/`. Create the two Kafka topics in the Aiven console. Without any of
this, recording, scoring and the autopilot still work locally.

## Files

- `surgical_suturing_guided.py`: glove tracking (both gloves), instruments and the
  two-handed cutting and suturing logic.
- `or_scene.py`: the scene (heart model import, skin and heart surface queries, trays,
  patient monitor, sounds); it includes the operating room from `environment/`.
- `environment/`: saimn80's cardiac operating room (`environment_simulator.xml` and its
  assets: the draped patient with curved chest skin, the opened chest, heart-lung and
  anaesthesia machines, lights, animated screens) and `or_controls.py` (Reset view, Reset
  instruments, Lift heart out, Load heart, the camera kept inside the room). See
  `environment/README.txt`. Recordings made since the room carry `"scene": "or-room"`;
  older skin recordings (flat pad) are kept but not replayed by the autopilot.
- `bypass.py`: the guided coronary bypass procedure (targets, steps, graft).
- `glove_suture.py`: hands-only suturing (six highlighted stitches driven by the gloves alone).
- `glove_bypass.py`: the hands-only bypass (snap points for the scalpel, graft and anastomoses).
- `procedures.py`: the procedures in the dropdown and the incision task.
- `motion_log.py`: recording, task scoring, autopilot replay and the robot trajectory export.
- `aiven_store.py`: local-first storage synced to Aiven PostgreSQL, and Kafka telemetry;
  `telemetry_monitor.py` follows the telemetry live.
- `coach.py`, `gemini_voice.py`, `gemma_review.py`, `assistant_window.py`: the AI coach
  (Gemini voice and typed questions, Gemma debriefs, the assistant window). The voice code and
  window layout are ported from hadi-ce04's `feature/ai-assistant-aiven` branch.
- `modal_gemma.py`: Gemma served on a Modal GPU (vLLM, OpenAI-compatible API).
- `assets/heart-nih3d/`: the heart model (MJCF, three OBJ meshes, MTL), unchanged.
