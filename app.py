"""The MuJoCo viewer loop: input from the gloves, keys and panel buttons each frame, the
session step, drawing, sounds, the overlay text, and the side windows."""

from queue import SimpleQueue
import time

import mujoco as mj
import mujoco.viewer
import numpy as np

import or_scene
from aiven_store import Telemetry, TrainingStore
from environment.or_controls import EnvironmentControls   # the operating room
from glove_input import SIDES, glove_poses
from instruments import draw_hand
from keymap import (ASSISTANT_KEY, BOTH_HANDS, GLOVES_READY, GLOVES_WAIT, MOVES, RESET_HANDS,
                    RESET_TOOLS, RESET_VIEW, REVIEW_KEY, SELECT_HAND, START_OVER, SWITCH_SITE,
                    TOGGLE_GUIDE, VOICE_KEY, ZOOM_IN, ZOOM_OUT, HeldKeys)
from or_scene import (SITES, WORK_TOP, Look, Sounds, build_model, draw_monitor, show_site,
                      view_groups, ASSISTANT_BUTTON, AUTOPILOT_BUTTON, CLEAR_BUTTON,
                      BOTH_HANDS_BUTTON, GUIDE_BUTTON, PROCEDURES_BUTTON, RESET_HANDS_BUTTON,
                      REVIEW_BUTTON, VOICE_BUTTON, PanelButton, PanelToggle)
from procedures import PROCEDURES
from session import Session


# The surgeon's view of the table. Not a camera in the model: MuJoCo's viewer can draw
# model cameras as markers (its Q shortcut), which would float in the scene.
BENCH_POS = np.array([.28, -.62, 1.43])
BENCH_FORWARD = np.cross([-.202, .492, .847], [.925, .380, 0])   # -(x axis cross y axis)
BENCH_FORWARD /= np.linalg.norm(BENCH_FORWARD)


def bench_view(model, data, cam, zoom=1.0, focus=None):
    """Point the free (mouse-controllable) camera at the repair site from the surgeon's
    position; `zoom` below 1 starts closer (the heart site is small)."""
    position, forward = BENCH_POS, BENCH_FORWARD
    distance = (WORK_TOP - position[2]) / forward[2]  # aim at the repair site
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = position + forward*distance
    if focus is not None:
        cam.lookat[:2] = focus
    cam.distance = distance*zoom
    cam.azimuth = np.degrees(np.arctan2(forward[1], forward[0]))
    cam.elevation = np.degrees(np.arcsin(forward[2]))


# Shown at the bottom right, matching the status box at the bottom left (six rows each).
# K or the Key guide button hides it.
KEY_GUIDE = (
    ("KEYS  Hands", "Z other hand  Y both hands  G take / drop"),
    ("Tools", "1 scalpel 2 needle 3 forceps 4 scissors"),
    ("Move", "arrows move   Q/E up/down   N align"),
    ("Act", "F grip / pass needle   T throw   X trim"),
    ("AI", "V voice  B Gemma review  I assistant  (panel: Autopilot)"),
    ("More", "R start over  0 view  H re-aim  P procedure  M sound  K keys"),
)

# MuJoCo's viewer toggles display options on single keys that overlap ours (Q camera
# markers, T transparency, X textures, 0-5 object groups, ...). It keeps its own copy of
# the options and only takes ours when they change, so after such a key we briefly send
# the toggled value and then the normal one, which puts the display back.
VIEWER_FLAG_KEYS = {ord(mj.mjVISSTRING[i][2]): i for i in range(mj.mjtVisFlag.mjNVISFLAG)
                    if len(mj.mjVISSTRING[i][2]) == 1 and mj.mjVISSTRING[i][2].isalpha()}
VIEWER_GROUP_KEYS = {ord(str(i)): i for i in range(6)}


def undo_viewer_shortcuts(option, key_events, flags, groups):
    """Send `option` the toggled value of every shortcut pressed an odd number of times
    in `key_events` (including auto-repeats); the next frame restores `flags`/`groups`."""
    option.flags[:] = flags
    option.geomgroup[:] = groups
    for code in set(key_events):
        if key_events.count(code) % 2 == 0:
            continue
        if code in VIEWER_FLAG_KEYS:
            i = VIEWER_FLAG_KEYS[code]
            option.flags[i] = 1 - flags[i]
        if code in VIEWER_GROUP_KEYS:
            i = VIEWER_GROUP_KEYS[code]
            option.geomgroup[i] = 1 - groups[i]


def draw_user(scene, hands, station, now):
    """Everything that moves: gloves, instruments, wounds, sutures and the patient monitor."""
    scene.ngeom = 0
    for side, points in hands.items():
        draw_hand(scene, points, station.both or side == station.active)
    station.draw(scene, now)
    draw_monitor(scene, now)


def render_frame(renderer, data, camera, option, drawn, station, now):
    """An offscreen picture of the operating field as the trainee sees it (for Gemma)."""
    renderer.update_scene(data, camera=camera, scene_option=option)
    extra = mj.MjvScene(renderer.model, 8000) if not hasattr(renderer, "_extra") else renderer._extra
    renderer._extra = extra
    draw_user(extra, drawn, station, now)
    scene = renderer.scene
    for i in range(extra.ngeom):
        if scene.ngeom >= scene.maxgeom:
            break
        target, source = scene.geoms[scene.ngeom], extra.geoms[i]
        mj.mjv_initGeom(target, source.type, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.ones(4, np.float32))
        for name in ("size", "pos", "mat", "rgba", "matid", "emission", "specular", "shininess",
                     "reflectance", "category", "dataid", "objtype", "objid"):
            setattr(target, name, getattr(source, name))
        scene.ngeom += 1
    return renderer.render()


def compact_checklist(rows, before=1, after=3):
    """Main steps in full; of the indented sub-steps only the last finished one, the current
    one and the next few, so the box stays small (MuJoCo overlays hold about 500 characters)."""
    sub = [i for i, (_, text) in enumerate(rows)
           if text.startswith("  ") and not text.startswith("  Stitch")]
    active = [i for i in sub if rows[i][0] == ">"]
    current = active[-1] if active else (sub[0] if sub else 0)
    keep = set(i for i in range(len(rows)) if i not in sub)
    keep |= {i for i in sub if current - before <= i <= current + after}
    return [rows[i] for i in sorted(keep)]


def checklist_text(session, now):
    """The checklist as plain text (Procedures window)."""
    rows = [f"[{mark}] {text}" for mark, text in session.station.checklist()]
    done = session.recent_done(now)
    return "\n".join(rows + ([""] + [f"Done: {text}" for text in done] if done else []))


def overlay_texts(session, status, best, coach_status, show_guide, now):
    """The viewer's three text boxes: the checklist (top left), the status box (bottom left)
    and, when shown, the key guide (bottom right). `best` is the site's best run or None."""
    font = int(mj.mjtFont.mjFONT_NORMAL)
    station = session.station
    left, right = station.hands["left"], station.hands["right"]
    keys_for = "both hands, Y" if station.both else f"{station.active}, Z"
    tool_status = f"L {left.tool}   R {right.tool}   (keys: {keys_for})"
    if session.autopilot is not None:
        pilot = session.autopilot
        action = ("AUTOPILOT finished: Autopilot button off to take over" if pilot.done else
                  f"AUTOPILOT {pilot.progress:.0%} (best run {pilot.summary['score']:.0f}/100): "
                  f"{station.action_prompt(short=True)}")
    else:
        action = station.action_prompt(short=True)
    scores = (f"last {session.last['score']:.0f}" if session.last else "no attempt yet") + (
        f", best {best[0]['score']:.0f}" if best else "")
    coach_line = f"{scores}  |  {session.recent_notice() or coach_status}"
    rows = compact_checklist(station.checklist())
    done = session.recent_done(now)[-2:]
    texts = [
        (font, int(mj.mjtGridPos.mjGRID_TOPLEFT),
         "\n".join(["PROCEDURE"] + [f"[{m}]" for m, _ in rows] + ["DONE"]*bool(done) + [""]*(len(done) - 1)),
         "\n".join([session.procedure.label] + [t for _, t in rows] + done)),
        (font, int(mj.mjtGridPos.mjGRID_BOTTOMLEFT),
         "SURGICAL TRAINING\nProcedure\nHands\nStitches / depth\nAction\nScore / coach",
         f"{status}\n{session.procedure.label}  (P or Procedures window)\n"
         f"{tool_status}\n{len(station.stitches)} / {station.depth*1000:.1f} mm\n"
         f"{action}\n{coach_line}"),
    ]
    if show_guide:
        texts.append((font, int(mj.mjtGridPos.mjGRID_BOTTOMRIGHT),
                      "\n".join(left for left, _ in KEY_GUIDE),
                      "\n".join(right for _, right in KEY_GUIDE)))
    return texts


def display(gloves, preview, procedure):
    model = build_model()
    Look.model = model
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    site = SITES[procedure.site]
    procedures_button = PanelButton(model, data, PROCEDURES_BUTTON)
    reset_hands_button = PanelButton(model, data, RESET_HANDS_BUTTON)
    both_toggle = PanelToggle(model, data, BOTH_HANDS_BUTTON)
    pick_bodies = {side: model.body(f"pick_{side}").id for side in SIDES}
    pick_mocap = {side: model.body(f"pick_{side}").mocapid[0] for side in SIDES}
    clear_button = PanelButton(model, data, CLEAR_BUTTON)
    review_button = PanelButton(model, data, REVIEW_BUTTON)
    assistant_button = PanelButton(model, data, ASSISTANT_BUTTON)
    autopilot_toggle = PanelToggle(model, data, AUTOPILOT_BUTTON)
    voice_toggle = PanelToggle(model, data, VOICE_BUTTON)
    guide_id = model.equality(GUIDE_BUTTON).id
    last_status = None
    keys = SimpleQueue()
    held_keys = HeldKeys()
    sounds = Sounds()
    controls = EnvironmentControls(model, data)   # room: panel buttons, camera, heart, screens
    # Reset instruments goes through the keys like Reset hands, so recordings replay exactly.
    controls.reset_tools = lambda station: [RESET_TOOLS]
    store, telemetry = TrainingStore(), Telemetry()
    from coach import Coach
    coach = Coach(store)
    picture = {"renderer": None, "camera": mj.MjvCamera(), "option": mj.MjvOption(), "drawn": {}}

    def render():
        """Offscreen frame of the current view for Gemma (None if rendering fails)."""
        try:
            if picture["renderer"] is None:
                picture["renderer"] = mj.Renderer(model, 540, 768)
            return render_frame(picture["renderer"], data, picture["camera"], picture["option"],
                                picture["drawn"], session.station, time.monotonic())
        except Exception as error:     # no offscreen GL context: review the data only
            print(f"Offscreen render unavailable ({error})", flush=True)
            return None

    last_frame = time.monotonic()
    session = Session(model, data, procedure, store, telemetry, coach, render, last_frame)
    picker = None

    def open_picker():
        nonlocal picker
        if picker is None or not picker.alive:
            try:
                from procedure_window import ProcedureClient
                picker = ProcedureClient(PROCEDURES.values(), session.procedure.key)
            except OSError as error:
                print(f"Procedures window unavailable ({error}); use P to cycle", flush=True)
        else:
            picker.show()               # already open (maybe behind the viewer): bring it forward

    open_picker()
    print("Two hands: the keyboard moves the hand the next step needs (green cuff); Z swaps it.\n"
          "1 scalpel | 2 needle holder | 3 forceps | 4 scissors | G pick up/set down\n"
          "Arrows move, Q/E raise/lower (tap for a small step, hold to glide).\n"
          "N aligns the tool, F grips (forceps/fingers) or passes the needle, T throws a knot,\n"
          "X trims. R starts over (the attempt so far is scored and saved).\n"
          "View: scroll or +/- to zoom, left-drag to orbit, right-drag to pan,\n"
          "      double-right-click a spot (tissue, tray, room) to zoom around it, 0 resets the view.\n"
          "H re-aims the gloves: press H, then rest your hands flat with fingers toward the screen.\n"
          "M mutes or unmutes the OR sounds.\n"
          "Procedures window (dropdown): pick incision, suture, free stitching, bypass or free practice; P cycles.\n"
          "Double-click a hand to move it with the arrows; Y (or the Both hands button) moves both.\n"
          "Panel (Equality, right): Procedures (reopens it), Reset hands (hands back to their\n"
          "      start), Clear repair, Key guide (or K),\n"
          "      Autopilot (replays the best recorded attempt), AI Assistant (I), AI Voice (V),\n"
          "      Gemma review (B).\n"
          f"{store.status}. {telemetry.status}. {coach.status()}.",
          flush=True)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=keys.put) as viewer:
            with viewer.lock():
                show_site(model, viewer.opt, site)
                bench_view(model, data, viewer.cam, site.zoom, site.focus)
                display_flags = viewer.opt.flags.copy()   # normal display; see VIEWER_FLAG_KEYS
            overlay = getattr(viewer, "set_texts", None)
            next_overlay = 0.0
            while viewer.is_running():
                now = time.monotonic()
                dt = min(now - last_frame, .1)
                last_frame = now
                pressed, key_events = [], []
                while not keys.empty():
                    code = keys.get_nowait()
                    key_events.append(code)
                    if code in MOVES and code in held_keys.keys:
                        held_keys.press(code, now)  # OS auto-repeat: continuous motion handles it
                        continue
                    held_keys.press(code, now)
                    pressed.append(code)
                held = list(held_keys.moves(now, dt))
                pressed += controls.step(viewer, now, session.station)   # room buttons (Reset view: 0)
                if controls.new_heart:                     # Load heart (STL): reopen with it
                    return controls.new_heart, session.procedure
                station = session.station

                # Panel buttons (Equality section) and their keys.
                with viewer.lock():
                    undo_viewer_shortcuts(viewer.opt, key_events, display_flags, view_groups(station.site))
                    picked = viewer.perturb.select       # double-click on a hand selects it
                    for side, body in pick_bodies.items():
                        if picked == body:
                            pressed.append(SELECT_HAND[side])   # recorded like a key
                            viewer.perturb.select = 0
                            viewer.perturb.active = 0
                    clear = clear_button.pressed() or START_OVER in pressed
                    review = review_button.pressed() or REVIEW_KEY in pressed
                    assistant = assistant_button.pressed() or ASSISTANT_KEY in pressed
                    voice = voice_toggle.changed() or VOICE_KEY in pressed
                    autopilot = autopilot_toggle.changed()
                    if TOGGLE_GUIDE in pressed:
                        data.eq_active[guide_id] = not data.eq_active[guide_id]
                    show_guide = bool(data.eq_active[guide_id])
                    if procedures_button.pressed():
                        open_picker()
                    if both_toggle.changed() and both_toggle.on != session.station.both:
                        pressed.append(BOTH_HANDS)       # recorded like a key, so replays stay exact
                    if reset_hands_button.pressed():
                        pressed.append(RESET_HANDS)      # recorded like a key, so replays stay exact
                    choice = picker.chosen() if picker is not None else None
                    if SWITCH_SITE in pressed:
                        names = list(PROCEDURES)
                        choice = names[(names.index(session.procedure.key) + 1) % len(names)]
                    picture["camera"].lookat[:] = viewer.cam.lookat
                    picture["camera"].distance = viewer.cam.distance
                    picture["camera"].azimuth = viewer.cam.azimuth
                    picture["camera"].elevation = viewer.cam.elevation
                    picture["option"].geomgroup[:] = station.site.groups
                taken = {SWITCH_SITE, TOGGLE_GUIDE, START_OVER, REVIEW_KEY, ASSISTANT_KEY, VOICE_KEY}
                pressed = [code for code in pressed if code not in taken]
                if clear and (choice is None or choice == session.procedure.key):
                    session.restart(now)
                    sounds.play("setdown")
                if choice is not None and choice != session.procedure.key:
                    session.restart(now, PROCEDURES[choice])
                    if session.site is not site:
                        site = session.site
                        with viewer.lock():
                            show_site(model, viewer.opt, site)
                            bench_view(model, data, viewer.cam, site.zoom, site.focus)
                    sounds.play("switch")
                    print(f"Procedure: {session.procedure.label}", flush=True)
                if autopilot:
                    if autopilot_toggle.on:
                        if not session.start_autopilot(now):
                            with viewer.lock():
                                autopilot_toggle.set(False)
                    else:
                        session.stop_autopilot(now)
                    sounds.play("switch")
                elif session.autopilot is None and autopilot_toggle.on:
                    with viewer.lock():
                        autopilot_toggle.set(False)   # a restart or site switch ended it

                if ord("M") in pressed:
                    sounds.muted = not sounds.muted
                    pressed = [code for code in pressed if code != ord("M")]
                zoom = [code for code in pressed if code in ZOOM_IN | ZOOM_OUT | {RESET_VIEW}]
                if zoom:
                    pressed = [code for code in pressed if code not in zoom]
                    with viewer.lock():
                        for code in zoom:
                            if code == RESET_VIEW:
                                bench_view(model, data, viewer.cam, site.zoom, site.focus)
                            else:
                                if viewer.cam.type != mj.mjtCamera.mjCAMERA_FREE:
                                    bench_view(model, data, viewer.cam, site.zoom, site.focus)
                                factor = .8 if code in ZOOM_IN else 1.25
                                viewer.cam.distance = float(np.clip(viewer.cam.distance*factor, .03, 6))

                reset_heading = ord("H") in pressed
                pressed = [key for key in pressed if key != ord("H")]
                poses, status = glove_poses(gloves, session.station.hands, preview, now, reset_heading)
                live = status.endswith("live") or "glove live" in status
                if session.station.guide is not None and (status != last_status or
                                                          session.station.guide.ready is None):
                    # Hands-only modes: a glove came in or its direction locked, so the hand
                    # jumped: bring the tools back into place, and run the steps only while the
                    # right glove is live and locked (recorded keys: replays match).
                    ready = (status in ("Both gloves live", "Keyboard practice")
                             or status.startswith("Right glove live"))
                    pressed += [RESET_HANDS, GLOVES_READY if ready else GLOVES_WAIT]
                with controls.heart_at_home(viewer):     # tissue queries with the heart in place
                    drawn = session.step(now, poses, pressed, held, live)
                picture["drawn"] = drawn
                coach.update(session.station, session.attempt_id or "autopilot",
                             session.autopilot is not None, voice, review, assistant, render)
                with viewer.lock():
                    draw_user(viewer.user_scn, drawn, session.station, now)
                    controls.follow_heart(viewer.user_scn)   # wounds move with a lifted heart
                    for side, points in drawn.items():     # keep the pick spheres on the palms
                        data.mocap_pos[pick_mocap[side]] = (
                            (points["wrist"] + points["middle_proximal"]) / 2 if points else [0, 0, -5])
                    mj.mj_kinematics(model, data)
                    if voice_toggle.on != coach.voice_on:
                        voice_toggle.set(coach.voice_on)
                    if both_toggle.on != session.station.both:
                        both_toggle.set(session.station.both)    # Y, or a fresh station
                for event in session.station.events:
                    sounds.play(event)
                session.station.events.clear()
                sounds.heartbeat(now)

                if status != last_status:
                    print(status, flush=True)
                    last_status = status
                if overlay is not None and now >= next_overlay:
                    if picker is not None:
                        picker.update(session.procedure.key, checklist_text(session, now))
                    overlay(overlay_texts(session, status, store.best(session.procedure.key,
                                                                      session.replay_scene()),
                                          coach.status(), show_guide, now))
                    next_overlay = now + .10
                viewer.sync()
                time.sleep(max(0, 1/60 - (time.monotonic() - now)))
    finally:
        session.end(time.monotonic())
        if picker is not None:
            picker.close()
        coach.close()
        telemetry.close()
        store.close()


def run(gloves, preview, procedure):
    """The viewer; opened again with the new model when another heart is loaded."""
    while (result := display(gloves, preview, procedure)) is not None:
        or_scene.HEART_FILE, procedure = result
