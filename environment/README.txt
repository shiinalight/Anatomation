README: OPERATING-ROOM ENVIRONMENT (XML)
=========================================

A complete cardiac operating room for MuJoCo (>= 4.0.1), as plain XML. No extra Python code needed.

  environment_simulator.xml  used by the hand simulator (or_scene.py): intact prepped skin for the
                             practice pad in geom group 1, the opened chest (sternotomy, retractor,
                             pericardial well) in group 3, fitted around the simulator's own heart
                             (HEART_POS) - the simulator brings the heart (group 2)
  environment.xml            stand-alone: open chest with the CT-reconstructed patient heart
  environment_suturing.xml   stand-alone: closed prepped skin only (for a practice pad)
  assets/                    textures and meshes for all three (keep it next to the XML files)
  or_controls.py             for the simulator: panel buttons, camera limit, heart lift / move /
                             load, moving screens (mujoco + numpy only)
  README.txt                 this file

This folder is generated. Do not edit the files by hand; when a new version arrives, replace the
whole environment/ folder and commit it (git add environment). Exported in high quality.


HOW THE SIMULATOR USES IT (already done in or_scene.py / app.py)
-----------------------------------------------------------------

1) ENV_XML = Path(__file__).resolve().parent / "environment" / "environment_simulator.xml"
2) In _scene_xml(), directly after </visual>:   <include file="{ENV_XML.as_posix()}"/>
3) The old room, table, drapes, torso, practice-pad box, open chest, lights, anaesthesia
   workstation and back table were removed; the environment provides them. Kept from the
   simulator: heart (group 2), Mayo trays with the four tools, patient monitor, hand pick spheres
   (group 5), Procedures window, panel buttons, sounds, both hands, recording, autopilot, AI coach
   and all game logic.
4) Skin: SkinTissue ray-casts the environment's intact skin (group 1) inside the pad rectangle,
   so cutting and suturing follow the chest; laceration and marker lines are drawn on it.
   make_tissue() builds a site inside or_controls.heart_home() (works with the heart lifted out);
   With another heart model (HEART_FILE) the bypass marks (placed for the NIH heart) are projected
   onto it.
5) The hook in app.py, display() (search for "controls"):
       controls = EnvironmentControls(model, data)
       pressed += controls.step(viewer, now, session.station)    # room buttons; Reset view = key 0
       if controls.new_heart: return controls.new_heart, session.procedure
       with controls.heart_at_home(viewer): drawn = session.step(...)
       controls.follow_heart(viewer.user_scn)                     # after draw_user
       run(): reopens display() with or_scene.HEART_FILE; --heart file.stl
   What or_controls does:
   - Equality-panel buttons: Reset view, Reset instruments, Lift heart out, Load heart (STL)
     (button_xml() adds them to <equality>; Reset hands is the simulator's own button)
   - the free camera stays inside the room (numeric or_room_box)
   - the heart is a mocap body: double-click lifts it out, Ctrl + right-drag moves, Ctrl + left-drag
     turns; wounds and stitches (tissue materials near the heart) move with it; the bypass
     cannulas hide (group 4) while it is out
   - the extra instruments can be moved with Ctrl + drag too; Reset instruments puts them back
   - screens animate from frame strips in the XML (texture <name>_frames + numeric
     or_anim_<name> = frame count, cycle seconds); the heart-lung machine's pumps turn
   - choose_heart_file() / heart_obj(): another heart (STL/OBJ, x anterior, y left, z superior,
     mm or m) centred where the NIH heart is; the viewer reopens with it

Sites (chosen through the Procedures window) work unchanged: show_site() sets the visible geom groups
  skin   -> groups 0, 1      room + intact skin
  heart  -> groups 0, 2, 3   room + heart + opened chest + bypass lines (aorta, right atrium)
Group 4 holds the environment's invisible helper geoms (never shown). Nothing of the environment
in group 2 (the simulator's heart) or group 5 (the simulator's hand pick spheres).

Screens (no duplicates): the simulator's own monitor shows the vitals; the pendant monitor is a
CT viewer scrolling through the slices, the boom display a beating 3D heart model, the
heart-lung machine its perfusion data, the anaesthesia machine the ventilator, the wall a case
board (procedure, blood, heparin/ACT, checklist).

Run:   mjpython surgical_suturing_guided.py --preview      (macOS; python ... elsewhere)

Tested (MuJoCo 4.0.1, Windows, headless): the scene compiles; all three sites render; picking up,
puncturing, tying, trimming and setting down give the same events as before; the buttons, camera
limit, heart lift (stitches follow) and Reset instruments work; another STL heart loads. Not tested:
the live Rokoko glove stream, the macOS sounds, and reopening the window after loading a heart
under mjpython on macOS.


COORDINATES AND NUMBERS
-----------------------

Meters, z up. Surgeon at -y looking toward +y, +x = surgeon's right, patient's head at -x.

  Hand anchor        ANCHOR = (0, -0.10, 1.015)        identical to the simulator
  Field center       (0, 0.05, 0.955)                  site or_field_center (skin level)
  Practice pad       flat skin plateau at z = 0.9435 under the pad (PAD_TOP 0.955, 11.5 mm thick)
  Simulator opening  sternotomy centered at x = -0.066, y = 0.023, 29 x 17.6 cm, around HEART_POS
                     (-0.067, 0.025, 0.902). Moving HEART_POS needs a new export (ask).
  Drape height       about 0.95 over the chest, 0.92 over the abdomen, 0.86 over the legs.
  Mayo stand         extra instruments on a towel at z = 1.088, center (0.62, 0.05), clear of
                     the simulator's own trays (TRAY_TOP 1.012, right tray ends at x = 0.374).
  Step platform      the surgeon stands on a 15 cm step at (-0.10, -0.62).

Cameras: or_surgeon, or_overview, or_topdown, or_far_side, or_closeup_heart, or_closeup_mayo
(the simulator ignores them and uses its own free bench view). All names start with "or_".


WHAT IS IN THE ROOM
-------------------

Static: room, two LED light heads + boom display, heart-lung machine, anesthesia machine with
monitor and pendant, surgical pendant (ESU, suction), IV pole, back table, ring stand, Mayo stand,
draped patient, anesthesia screen, cables/tubing, headlamp light aimed at the field.

Extra instruments (free bodies with real masses, hinge jaws with a gentle spring, no actuators):
or_debakey_1, or_potts (the only scissors), or_castroviejo, or_satinsky, or_cross_clamp,
or_bulldog_1/2, or_lap_1/2 (Mayo stand), or_gauze (back table). Also on the Mayo stand (static):
internal defibrillator paddles and suture packets. Sites or_<name>_tip / or_<name>_grip; jaw joint
or_<name>_jaw (0 = closed). The simulator only calls mj_forward, so they stay where they are
unless moved with Ctrl + drag; they collide (contype 3) if a future version steps the physics.

Mocap bodies (move only if code sets data.mocap_pos/quat): or_hlm_roller_1..4, or_headlamp_mount.


IF SOMETHING LOOKS WRONG
------------------------

- "file not found": environment/ is missing or assets/ is not next to the XML files.
- XML errors about unknown attributes: update MuJoCo (pip install -U "mujoco>=4.0.1").
- Angles: the environment sets <compiler angle="radian"/> for the whole model. or_scene.py uses
  no angles. If you add degrees later, use <attach> instead of <include>:
      <asset> <model name="or_env" file="{ENV_XML.as_posix()}"/> </asset>
      <worldbody> <attach model="or_env" prefix=""/> ... </worldbody>
- Lights and visual settings (shadows, headlight) come from the environment; the simulator's
  own <visual> values that the environment does not set (haze, fov) still apply.
- Low frame rate: shadows cost the most. Set castshadow="false" on or_light_head_2_spot (and
  or_light_head_1_spot) in the XML, or ask for a lower-quality export.
