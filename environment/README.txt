README: OPERATING-ROOM ENVIRONMENT (XML)
=========================================

environment.xml + assets/ = a complete cardiac operating room for MuJoCo.
Keep assets/ next to environment.xml.


QUICK START (3 changes in or_scene.py)
--------------------------------------

1) Add this line near the top of or_scene.py (absolute path is required,
   because the model is loaded from a string):

       ENV = (Path(__file__).parent / "environment" / "environment.xml").resolve().as_posix()

2) In scene_xml(), in the big <mujoco> string, add this line directly AFTER
   the closing </visual> tag (the string is an f-string, so {ENV} works):

       <include file="{ENV}"/>

3) Remove the old room so it does not overlap with the new one. In
   scene_xml(), delete (or comment out) the add(...) lines for:
     - floor, walls, ceiling, door, cabinets
     - OR table, drapes, patient torso
     - Mayo stands / trays
     - surgical lights and the three <light> entries
     - anaesthesia workstation, IV pole, monitor geoms

   KEEP: the practice pad (skin window, marker lines, lip/wound geoms) and
   <camera name="bench" .../> (the game script looks it up by name).

Then run:   python surgical_suturing_guided.py --preview

If MuJoCo says a file is missing, the path in <include> is wrong or assets/ is
not next to environment.xml.


WHAT CAN BE CHANGED AFTERWARDS
------------------------------

Coordinates (meters, z up): surgeon at -y looking toward +y, +x = surgeon's
right, head of patient at -x.

- Pad height: the environment's field center is (0, 0.05, 0.955). The pad
  (PADTOP = 0.955, PADCENTER = 0.005/0.055) is already close. If the pad
  clips into the drape, raise PADTOP slightly.
- Trays / tools: the environment's Mayo tray surface is at about z = 1.08
  (center x = 0.52). The old TRAYTOP is 1.012. Update TRAYS, TRAYTOP,
  TRAYGRIP, TOOLHOME and surfaceheight() if the tools should rest on the new
  Mayo stand.
- Monitor: drawmonitor() draws the ECG/SpO2 at MONITORCENTER, MONITORNORMAL,
  MONITORHALF. Move these to the new monitor (or_patient_monitor), or keep
  the old monitor geoms.
- Hand anchor: ANCHOR (0, -0.10, 1.015) is identical in both. No change.
- Angles: environment.xml sets <compiler angle="radian"/> for the whole model.
  or_scene.py uses no angles, so nothing changes. If degrees are used later,
  use <attach> instead of <include>:
      <asset> <model name="or_env" file="{ENV}"/> </asset>
      <worldbody> <attach model="or_env" prefix=""/> ... </worldbody>
- Performance: shadow-casting lights cost the most. If the frame rate is low,
  set castshadow="false" on one of the or_light_head_*_spot lights in
  environment.xml.
- Cameras available: or_surgeon, or_overview, or_topdown, or_far_side,
  or_closeup_heart, or_closeup_mayo.
- All names in the environment start with "or_", so nothing clashes with
  existing names.
