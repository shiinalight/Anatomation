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
  README.txt                 this file

This folder is generated. Do not edit the files by hand; when a new version arrives, replace the
whole environment/ folder and commit it (git add environment). Exported in high quality.


HOW THE SIMULATOR USES IT (already done in or_scene.py)
--------------------------------------------------------

1) ENV_XML = Path(__file__).resolve().parent / "environment" / "environment_simulator.xml"
2) In _scene_xml(), directly after </visual>:   <include file="{ENV_XML.as_posix()}"/>
3) The old room, OR table, drapes, torso, open-chest geometry, surgical lights, anaesthesia
   workstation and back table were removed from _scene_xml(); the environment provides them.
   Kept from the simulator: practice pad (group 1), heart (group 2), Mayo trays with the four
   tools, patient monitor, site-switch buttons, sounds, hand, instruments and all game logic.

Site switch (Skin / Heart / Bypass) works unchanged: show_site() sets the visible geom groups
  skin   -> groups 0, 1      room + intact skin + practice pad
  heart  -> groups 0, 2, 3   room + heart + opened chest
Group 5 holds the environment's invisible helper geoms (never shown). Nothing in groups 2 and 4.

Run:   mjpython surgical_suturing_guided.py --preview      (macOS; python ... elsewhere)

Tested (MuJoCo 4.0.1): the scene compiles; all three sites render from the bench view with the
hand, the tools on the trays and the monitor; picking up, puncturing, tying, trimming and setting
down give the same results and sound events as before the change; the heart does not touch the
skin, the wound walls or the pericardium; the trays, posts and monitor do not intersect anything.
The live Rokoko glove stream was not available for the test.


COORDINATES AND NUMBERS
-----------------------

Meters, z up. Surgeon at -y looking toward +y, +x = surgeon's right, patient's head at -x.

  Hand anchor        ANCHOR = (0, -0.10, 1.015)        identical to the simulator
  Field center       (0, 0.05, 0.955)                  site or_field_center (skin level)
  Practice pad       flat skin plateau at z = 0.9435 under the pad (PAD_TOP 0.955, 11.5 mm thick)
  Simulator opening  sternotomy centered at x = -0.066, y = 0.023, 29 x 17.6 cm, around HEART_POS
                     (-0.067, 0.025, 0.902). Moving HEART_POS needs a new export (ask).
  Drape height       about 0.95 over the chest, 0.92 over the abdomen, 0.86 over the legs.
  Mayo stand         extra instruments on a towel at z = 1.088, center (0.52, 0.05). The
                     simulator's own trays (TRAY_TOP 1.012) stand next to it.
  Step platform      the surgeon stands on a 15 cm step at (-0.10, -0.62).

Cameras: or_surgeon, or_overview, or_topdown, or_far_side, or_closeup_heart, or_closeup_mayo
(the simulator ignores them and uses its own free bench view). All names start with "or_".


WHAT IS IN THE ROOM
-------------------

Static: room, two LED light heads + boom display, heart-lung machine, anesthesia machine with
monitor and pendant, surgical pendant (ESU, suction), IV pole, back table, ring stand, Mayo stand,
draped patient, anesthesia screen, cables/tubing, headlamp light aimed at the field.

Extra instruments (free bodies with real masses, hinge jaws with a gentle spring, no actuators):
or_scalpel, or_debakey_1/2, or_metzenbaum, or_potts, or_needle_holder, or_castroviejo,
or_satinsky, or_cross_clamp, or_kelly_1/2, or_bulldog_1/2, or_lap_1/2 (Mayo stand), or_gauze
(back table). Sites or_<name>_tip / or_<name>_grip; jaw joint or_<name>_jaw (0 = closed).
The simulator only calls mj_forward, so they stay where they are; they collide (contype 3) if a
future version steps the physics.

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
