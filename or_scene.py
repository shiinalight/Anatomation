"""Operating-room scene for the suturing simulator: layout, textures, lights and feedback.

Everything here is visual. The gameplay code imports the layout constants
(pad, trays, monitor) so the scene and the interaction always agree.
"""

import shutil
import struct
import subprocess
import sys
import tempfile
import wave
import zlib
from pathlib import Path

import mujoco as mj
import numpy as np

# --- Layout shared with the gameplay code (metres, world frame, z up) ---------------
PAD_CENTER = np.array([.005, .055])
PAD_HALF = np.array([.103, .071])
PAD_MIN = PAD_CENTER - PAD_HALF
PAD_MAX = PAD_CENTER + PAD_HALF
PAD_TOP = .955
INCISION_Y = .055

TRAY_TOP = 1.012                    # Mayo-stand tray floors, above the draped patient
TRAYS = (                           # (centre xy, half size xy)
    (np.array([-.236, .081]), np.array([.066, .108])),
    (np.array([.297, .085]), np.array([.077, .104])),
)
TABLE_MIN = np.array([-.54, -.28])  # where a dropped tool may come to rest
TABLE_MAX = np.array([.54, .44])
TABLE_TOP = .812                    # drape over the OR table mattress
TORSO = (np.array([-.02, .06, .84]), np.array([.50, .23, .112]))  # draped patient

MONITOR_CENTER = np.array([.10, 1.12, 1.00])
MONITOR_HALF = np.array([.20, .125])   # screen half width / height
_toward = np.array([.28, -.62, 1.25]) - MONITOR_CENTER
MONITOR_NORMAL = _toward / np.linalg.norm(_toward)
MONITOR_RIGHT = np.cross([0, 0, 1], MONITOR_NORMAL)
MONITOR_RIGHT /= np.linalg.norm(MONITOR_RIGHT)
MONITOR_UP = np.cross(MONITOR_NORMAL, MONITOR_RIGHT)
HEART_RATE = 72


def surface_height(x, y):
    """Height of whatever a tool set down at (x, y) would rest on."""
    if PAD_MIN[0] <= x <= PAD_MAX[0] and PAD_MIN[1] <= y <= PAD_MAX[1]:
        return PAD_TOP
    for centre, half in TRAYS:
        if np.all(np.abs(np.array([x, y]) - centre) <= half):
            return TRAY_TOP
    centre, size = TORSO
    r = ((x - centre[0]) / size[0])**2 + ((y - centre[1]) / size[1])**2
    if r < 1:
        return centre[2] + size[2]*np.sqrt(1 - r)
    return TABLE_TOP


# --- Procedural textures (seamless, generated at startup; no image files needed) -----
def _png(rgb):
    rgb = np.clip(rgb*255 + .5, 0, 255).astype(np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, body):
        return (struct.pack(">I", len(body)) + kind + body
                + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _noise(rng, size, cells, octaves=4):
    """Tileable fractal value noise in [-1, 1]."""
    h, w = size
    total, amp, norm = np.zeros(size), 1.0, 0.0
    cy, cx = cells
    for _ in range(octaves):
        grid = rng.uniform(-1, 1, (cy, cx))
        y = np.linspace(0, cy, h, endpoint=False)
        x = np.linspace(0, cx, w, endpoint=False)
        y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
        fy, fx = y - y0, x - x0
        fy, fx = fy*fy*(3 - 2*fy), fx*fx*(3 - 2*fx)
        y1, x1 = (y0 + 1) % cy, (x0 + 1) % cx
        top = grid[y0][:, x0]*(1 - fx) + grid[y0][:, x1]*fx
        bottom = grid[y1][:, x0]*(1 - fx) + grid[y1][:, x1]*fx
        total += amp*(top*(1 - fy)[:, None] + bottom*fy[:, None])
        norm += amp
        amp *= .5
        cy, cx = cy*2, cx*2
    return total / norm


def _textures():
    rng = np.random.default_rng(7)
    tex = {}
    col = lambda *c: np.array(c)[None, None, :]

    # Seamless speckled vinyl floor, 2 m per tile of texture.
    n = 512
    base = col(.50, .55, .56) + .035*_noise(rng, (n, n), (4, 4))[..., None]
    fleck = rng.random((n, n))
    base = base + np.where(fleck > .985, .10, 0)[..., None] - np.where(fleck < .012, .08, 0)[..., None]
    base[:2] *= .82
    base[:, :2] *= .82
    tex["floor.png"] = base

    # Glazed wall tiles, 4 x 4 tiles of 20 cm per texture.
    tile = col(.78, .86, .83) + .012*_noise(rng, (n, n), (8, 8))[..., None]
    shade = rng.normal(0, .012, (4, 4))
    tile += np.kron(shade, np.ones((n // 4, n // 4)))[..., None]
    grout = np.zeros((n, n), bool)
    for k in range(4):
        edge = k*n // 4
        grout[edge:edge + 3] = grout[:, edge:edge + 3] = True
    tile[grout] = (.66, .68, .66)
    tex["tile.png"] = tile

    # Acoustic ceiling panels (60 cm) with pinholes.
    m = 256
    ceil = col(.90, .91, .90) + .015*_noise(rng, (m, m), (4, 4))[..., None]
    ceil[rng.random((m, m)) > .93] *= .9
    ceil[:3] = ceil[:, :3] = (.70, .72, .72)
    tex["ceiling.png"] = ceil

    # Woven non-woven drape and gown fabric.
    weave = .035*_noise(rng, (m, m), (128, 128), 1) + .012*_noise(rng, (m, m), (4, 4), 2)
    tex["drape.png"] = col(.21, .40, .56) + weave[..., None]
    tex["gown.png"] = col(.17, .31, .47) + 1.3*weave[..., None]

    # Betadine-prepped skin: skin tone stained orange-brown, patchy, with pores.
    skin = col(.80, .58, .47)
    iodine = col(.60, .30, .14)
    stain = np.clip(.62 + .35*_noise(rng, (n, n), (3, 3)), 0, 1)[..., None]
    pores = (rng.random((n, n)) > .97)[..., None]*.05
    tex["skin.png"] = skin*(1 - stain) + iodine*stain + .03*_noise(rng, (n, n), (40, 40), 2)[..., None] - pores

    # Brushed stainless steel: streaks along one axis.
    streak = _noise(rng, (m, m), (128, 2), 3)
    tex["steel.png"] = col(.74, .76, .78) + .06*streak[..., None]
    return {name: _png(image) for name, image in tex.items()}


# --- Scene description --------------------------------------------------------------
def _fmt(v):
    return " ".join(f"{float(x):.4g}" for x in np.ravel(v))


def _lookat(position, target):
    d = np.asarray(target, float) - position
    return d / np.linalg.norm(d)


def _scene_xml():
    main_light = np.array([-.30, .26, 1.86])
    main_dir = _lookat(main_light, [0, .055, PAD_TOP])
    side_light = np.array([.46, -.18, 1.92])
    side_dir = _lookat(side_light, [0, .055, PAD_TOP])
    g = []   # static geometry, one line each
    add = g.append

    # Room shell: floor, tiled walls with stainless bumper rail, panel ceiling.
    add('<geom type="plane" pos="0 .3 0" size="3.3 3 .05" material="floor"/>')
    for pos, size in (("0 3.05 1.5", "3.3 .05 1.5"), ("0 -2.65 1.5", "3.3 .05 1.5"),
                      ("-3.25 .2 1.5", ".05 2.9 1.5"), ("3.25 .2 1.5", ".05 2.9 1.5")):
        add(f'<geom type="box" pos="{pos}" size="{size}" material="tile"/>')
    add('<geom type="box" pos="0 .2 3.03" size="3.3 2.9 .03" material="ceiling"/>')
    for pos, size in (("0 2.985 .95", "3.2 .02 .035"), ("-3.185 .2 .95", ".02 2.8 .035"),
                      ("3.185 .2 .95", ".02 2.8 .035"), ("0 2.99 .07", "3.2 .02 .07")):
        add(f'<geom type="box" pos="{pos}" size="{size}" material="satin"/>')
    for x in (-1.6, 1.6):
        for y in (-.8, 1.6):
            add(f'<geom type="box" pos="{x} {y} 2.995" size=".6 .3 .01" material="panel"/>')

    # Sliding OR door with a vision panel, wall-mounted PACS display, storage cabinets.
    add('<geom type="box" pos="1.75 2.985 1.1" size=".8 .03 1.1" material="satin"/>')
    add('<geom type="box" pos="1.75 2.96 1.45" size=".18 .01 .25" material="glass"/>')
    add('<geom type="box" pos="-1.0 2.975 1.65" size=".62 .04 .36" material="black"/>')
    add('<geom type="box" pos="-1.0 2.93 1.65" size=".58 .005 .32" material="xray"/>')
    for y in (-1.6, -.5, .6):
        add(f'<geom type="box" pos="-3.0 {y} 1.55" size=".2 .52 .55" material="satin"/>')
        add(f'<geom type="box" pos="-2.795 {y} 1.55" size=".004 .48 .5" material="glass"/>')
        add(f'<geom type="box" pos="-3.0 {y} .45" size=".2 .52 .42" material="satin"/>')

    # OR table: pedestal, column and black mattress under the drape.
    add('<geom type="box" pos=".05 .06 .04" size=".42 .24 .04" material="plastic_dark"/>')
    add('<geom type="box" pos=".05 .06 .40" size=".16 .12 .34" material="satin"/>')
    add('<geom type="box" pos=".05 .06 .765" size="1.1 .27 .035" material="black"/>')

    # Sterile drapes over the patient: hanging sides, body, legs, anaesthesia screen.
    add(f'<geom type="box" pos=".05 .06 {TABLE_TOP - .006}" size="1.12 .29 .006" material="drape"/>')
    for y in (-.235, .355):
        add(f'<geom type="box" pos=".05 {y} .58" size="1.12 .004 .24" material="drape"/>')
    add('<geom type="box" pos="1.172 .06 .58" size=".004 .30 .24" material="drape"/>')
    centre, size = TORSO
    add(f'<geom type="ellipsoid" pos="{_fmt(centre)}" size="{_fmt(size)}" material="drape"/>')
    add('<geom type="ellipsoid" pos="-.40 .06 .835" size=".30 .25 .095" material="drape"/>')
    add('<geom type="ellipsoid" pos=".62 .06 .815" size=".56 .24 .075" material="drape"/>')
    add('<geom type="ellipsoid" pos="1.10 .00 .845" size=".04 .05 .06" material="drape"/>')
    add('<geom type="ellipsoid" pos="1.10 .12 .845" size=".04 .05 .06" material="drape"/>')
    add('<geom type="box" pos="-.68 .06 1.14" size=".004 .36 .32" material="drape"/>')
    add('<geom type="capsule" fromto="-.68 -.31 1.47 -.68 .43 1.47" size=".008" material="steel"/>')
    for y in (-.31, .43):
        add(f'<geom type="capsule" fromto="-.68 {y} .80 -.68 {y} 1.47" size=".008" material="steel"/>')

    # Skin window: betadine-prepped skin framed by the adhesive edge of the incise drape.
    (cx, cy), (hx, hy) = PAD_CENTER, PAD_HALF
    add(f'<geom type="box" pos="{cx} {cy} {PAD_TOP - .0055}" size="{hx} {hy} .0055" material="skin"/>')
    w = .016
    for pos, size in (((cx, cy - hy - w + .004), (hx + 2*w, w)), ((cx, cy + hy + w - .004), (hx + 2*w, w)),
                      ((cx - hx - w + .004, cy), (w, hy)), ((cx + hx + w - .004, cy), (w, hy))):
        add(f'<geom type="box" pos="{pos[0]:.4f} {pos[1]:.4f} {PAD_TOP + .0006}" '
            f'size="{size[0]:.4f} {size[1]:.4f} .0012" material="drape"/>')
    # Surgical-marker guide lines (gentian violet) where entry and exit bites go.
    for y in (INCISION_Y - .025, INCISION_Y + .026):
        for x in np.arange(-.084, .086, .012):
            add(f'<geom type="box" pos="{x:.3f} {y} {PAD_TOP + .0002}" size=".0035 .0006 .0002" '
                f'material="marker"/>')
    # The laceration to repair: swollen pink lips around a wet dark-red gap.
    add(f'<geom type="ellipsoid" pos="0 {INCISION_Y} {PAD_TOP + .0001}" size=".079 .0045 .0004" '
        f'material="lip"/>')
    add(f'<geom type="ellipsoid" pos="0 {INCISION_Y} {PAD_TOP + .0003}" size=".075 .0017 .0005" '
        f'material="wound"/>')

    # Mayo stands: stainless trays on a cantilever arm, post and wheeled base behind the table.
    for (tx, ty), (hx, hy) in TRAYS:
        z = TRAY_TOP
        add(f'<geom type="box" pos="{tx} {ty} {z - .004}" size="{hx} {hy} .004" material="steel"/>')
        for pos, size in (((tx - hx, ty), (.002, hy)), ((tx + hx, ty), (.002, hy)),
                          ((tx, ty - hy), (hx, .002)), ((tx, ty + hy), (hx, .002))):
            add(f'<geom type="box" pos="{pos[0]:.3f} {pos[1]:.3f} {z + .006}" '
                f'size="{size[0]:.3f} {size[1]:.3f} .010" material="steel"/>')
        # Sterile towel lining the tray.
        add(f'<geom type="box" pos="{tx} {ty} {z - .0005}" size="{hx - .006} {hy - .006} .0006" '
            f'material="towel"/>')
        add(f'<geom type="capsule" fromto="{tx} {ty + hy} {z - .01} {tx} .52 {z - .01}" size=".009" '
            f'material="steel"/>')
        add(f'<geom type="capsule" fromto="{tx} .52 .08 {tx} .52 {z - .01}" size=".014" material="steel"/>')
        for a in np.linspace(0, 2*np.pi, 5)[:-1]:
            fx, fy = tx + .22*np.cos(a + .6), .52 + .22*np.sin(a + .6)
            add(f'<geom type="capsule" fromto="{tx} .52 .07 {fx:.3f} {fy:.3f} .05" size=".012" '
                f'material="satin"/>')
            add(f'<geom type="sphere" pos="{fx:.3f} {fy:.3f} .025" size=".025" material="black"/>')

    # Ceiling-mounted surgical lights on articulated booms.
    for centre, direction, radius in ((main_light, main_dir, .30), (side_light, side_dir, .22)):
        add(f'<geom type="capsule" fromto="{centre[0]} {centre[1]} 3.0 {centre[0]} {centre[1]} 2.35" '
            f'size=".035" material="plastic_white"/>')
        elbow = centre - direction*.05 + np.array([0, 0, .35])
        add(f'<geom type="capsule" fromto="{centre[0]} {centre[1]} 2.35 {_fmt(elbow)}" size=".028" '
            f'material="plastic_white"/>')
        add(f'<geom type="capsule" fromto="{_fmt(elbow)} {_fmt(centre - direction*.05)}" size=".022" '
            f'material="plastic_white"/>')
        add(f'<geom type="cylinder" pos="{_fmt(centre)}" zaxis="{_fmt(-direction)}" '
            f'size="{radius} .035" material="plastic_white"/>')
        lens = centre + direction*.036
        add(f'<geom type="cylinder" pos="{_fmt(lens)}" zaxis="{_fmt(-direction)}" '
            f'size="{radius*.93} .002" material="lamp_lens"/>')
        right = np.cross(direction, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, direction)
        for k in range(7):
            ring = 0 if k == 0 else radius*.55
            a = k*2*np.pi/6
            pod = lens + direction*.003 + ring*(np.cos(a)*right + np.sin(a)*up)
            add(f'<geom type="cylinder" pos="{_fmt(pod)}" zaxis="{_fmt(-direction)}" '
                f'size="{radius*.15:.3f} .004" material="lamp_led"/>')
        add(f'<geom type="capsule" fromto="{_fmt(lens + direction*.004)} {_fmt(lens + direction*.09)}" '
            f'size=".018" material="plastic_white"/>')

    # Patient monitor on a pole, facing the surgeon. Traces are drawn live each frame.
    n, r, u = MONITOR_NORMAL, MONITOR_RIGHT, MONITOR_UP
    c = MONITOR_CENTER
    add(f'<geom type="box" pos="{_fmt(c - n*.025)}" xyaxes="{_fmt(r)} {_fmt(u)}" '
        f'size="{MONITOR_HALF[0] + .022} {MONITOR_HALF[1] + .03} .025" material="plastic_dark"/>')
    add(f'<geom type="box" pos="{_fmt(c + n*.0005)}" xyaxes="{_fmt(r)} {_fmt(u)}" '
        f'size="{MONITOR_HALF[0]} {MONITOR_HALF[1]} .0005" material="screen"/>')
    add(f'<geom type="capsule" fromto="{c[0]} {c[1] + .06} .06 {c[0]} {c[1] + .06} {c[2] - .1}" '
        f'size=".018" material="satin"/>')
    add(f'<geom type="box" pos="{c[0]} {c[1] + .06} .03" size=".22 .22 .03" material="plastic_dark"/>')

    # Anaesthesia workstation and IV pole at the head end, behind the screen.
    add('<geom type="box" pos="-1.25 .75 .62" size=".32 .30 .58" material="plastic_white"/>')
    add('<geom type="box" pos="-1.25 .44 .95" size=".28 .012 .17" material="plastic_dark"/>')
    add('<geom type="box" pos="-1.25 .43 .95" size=".25 .004 .14" material="screen"/>')
    add('<geom type="box" pos="-1.25 .55 1.45" size=".24 .14 .16" material="plastic_dark"/>')
    for x, colour in ((-1.38, "gas_o2"), (-1.25, "gas_air"), (-1.12, "gas_n2o")):
        add(f'<geom type="cylinder" pos="{x} .43 .42" zaxis="0 -1 0" size=".03 .01" material="{colour}"/>')
    add('<geom type="capsule" fromto="-.95 .55 .05 -.95 .55 1.95" size=".012" material="steel"/>')
    add('<geom type="ellipsoid" pos="-.95 .55 1.72" size=".05 .025 .1" material="iv_bag"/>')
    add('<geom type="capsule" fromto="-.95 .55 1.62 -.70 .30 .95" size=".003" material="iv_bag"/>')
    add('<geom type="box" pos="-.95 .55 .03" size=".22 .22 .02" material="satin"/>')

    # Back table with sterile drape and kick bucket for the scrub nurse.
    add('<geom type="box" pos="1.6 .3 .45" size=".45 .75 .45" material="satin"/>')
    add('<geom type="box" pos="1.6 .3 .905" size=".47 .77 .006" material="drape"/>')
    add('<geom type="box" pos="1.12 .3 .70" size=".006 .77 .21" material="drape"/>')
    add('<geom type="cylinder" pos=".95 -.7 .22" size=".17 .22" material="satin"/>')

    lights = f"""
    <light name="surgical_main" type="spot" pos="{_fmt(main_light)}" dir="{_fmt(main_dir)}"
           cutoff="70" exponent="12" castshadow="true" bulbradius=".16"
           diffuse=".95 .95 .92" specular=".55 .55 .55" ambient="0 0 0"/>
    <light name="surgical_side" type="spot" pos="{_fmt(side_light)}" dir="{_fmt(side_dir)}"
           cutoff="70" exponent="10" castshadow="false"
           diffuse=".35 .35 .35" specular=".2 .2 .2" ambient="0 0 0"/>
    <light name="room" type="directional" pos="0 0 2.9" dir=".15 .35 -1" castshadow="false"
           diffuse=".36 .37 .37" specular=".05 .05 .05" ambient=".26 .27 .27"/>
    """
    return f"""
<mujoco model="Operating room">
  <visual>
    <global offwidth="1600" offheight="1000" fovy="45"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient=".08 .08 .08" diffuse=".14 .14 .14" specular=".05 .05 .05"/>
    <map znear=".006" zfar="30" haze=".2" shadowclip="2.5" shadowscale=".6"/>
    <rgba haze=".72 .78 .78 1"/>
  </visual>
  <asset>
    <texture name="floor" type="2d" file="floor.png"/>
    <texture name="tile" type="2d" file="tile.png"/>
    <texture name="ceiling" type="2d" file="ceiling.png"/>
    <texture name="drape" type="2d" file="drape.png"/>
    <texture name="gown" type="2d" file="gown.png"/>
    <texture name="skin" type="2d" file="skin.png"/>
    <texture name="steel" type="2d" file="steel.png"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat=".5 .5"
              specular=".35" shininess=".35" reflectance=".07"/>
    <material name="tile" texture="tile" texuniform="true" texrepeat="1.25 1.25"
              specular=".45" shininess=".7"/>
    <material name="ceiling" texture="ceiling" texuniform="true" texrepeat="1.66 1.66" specular=".05"/>
    <material name="drape" texture="drape" texuniform="true" texrepeat="4 4"
              specular=".04" shininess=".05"/>
    <material name="towel" texture="drape" texuniform="true" texrepeat="6 6" rgba=".72 .80 .86 1"
              specular=".02"/>
    <material name="gown" texture="gown" texuniform="true" texrepeat="8 8" specular=".05"/>
    <material name="skin" texture="skin" specular=".35" shininess=".45"/>
    <material name="steel" texture="steel" texuniform="true" texrepeat="10 10" rgba="1 1 1 1"
              specular=".9" shininess=".85" metallic="1" roughness=".25"/>
    <material name="satin" texture="steel" texuniform="true" texrepeat="4 4" rgba=".82 .84 .85 1"
              specular=".55" shininess=".5" metallic="1" roughness=".45"/>
    <material name="polished" rgba=".86 .88 .90 1" specular="1" shininess="1" metallic="1" roughness=".1"/>
    <material name="blade" rgba=".80 .82 .84 1" specular="1" shininess="1" metallic="1" roughness=".05"/>
    <material name="gold" rgba=".84 .67 .30 1" specular=".9" shininess=".85" metallic="1" roughness=".2"/>
    <material name="knurl" rgba=".42 .45 .47 1" specular=".6" shininess=".4" metallic="1"/>
    <material name="plastic_white" rgba=".90 .91 .90 1" specular=".35" shininess=".6"/>
    <material name="plastic_dark" rgba=".22 .24 .25 1" specular=".3" shininess=".5"/>
    <material name="black" rgba=".07 .07 .08 1" specular=".2" shininess=".3"/>
    <material name="glass" rgba=".55 .66 .68 .55" specular=".9" shininess="1"/>
    <material name="xray" rgba=".55 .58 .60 1" emission=".5"/>
    <material name="screen" rgba=".015 .02 .025 1" specular=".6" shininess=".95"/>
    <material name="indicator" rgba=".15 .65 .35 1" emission="1"/>
    <material name="panel" rgba=".97 .98 1 1" emission="1"/>
    <material name="lamp_lens" rgba=".86 .90 .92 1" emission=".45" specular="1" shininess="1"/>
    <material name="lamp_led" rgba="1 1 .97 1" emission="1.6"/>
    <material name="glove" rgba=".96 .88 .73 1" specular=".5" shininess=".7"/>
    <material name="marker" rgba=".38 .16 .55 1" specular=".1"/>
    <material name="lip" rgba=".80 .42 .38 1" specular=".45" shininess=".6"/>
    <material name="wound" rgba=".36 .03 .04 1" specular=".95" shininess=".95"/>
    <material name="blood" rgba=".46 .02 .03 1" specular="1" shininess="1"/>
    <material name="thread" rgba=".36 .20 .62 1" specular=".5" shininess=".6"/>
    <material name="iv_bag" rgba=".85 .90 .95 .45" specular=".9" shininess=".9"/>
    <material name="gas_o2" rgba=".9 .92 .9 1" specular=".5"/>
    <material name="gas_air" rgba=".95 .85 .15 1" specular=".5"/>
    <material name="gas_n2o" rgba=".15 .3 .75 1" specular=".5"/>
    <material name="ecg" rgba=".25 1 .45 1" emission="1"/>
    <material name="pleth" rgba=".25 .85 1 1" emission="1"/>
    <material name="nibp" rgba="1 .9 .35 1" emission="1"/>
    <material name="grid" rgba=".08 .16 .14 1" emission=".6"/>
  </asset>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    {lights}
    {chr(10).join("    " + line for line in g)}
    <camera name="bench" pos=".28 -.62 1.43" xyaxes=".925 .380 0 -.202 .492 .847" fovy="45"/>
  </worldbody>
</mujoco>
"""


def build_model():
    return mj.MjModel.from_xml_string(_scene_xml(), _textures())


# --- Drawing helpers for per-frame (user scene) geometry ----------------------------
class Look:
    """Material lookup so per-frame geoms get the same textures and shine as the room."""
    model = None
    _cache = {}

    @classmethod
    def apply(cls, geom, material):
        if cls.model is None:
            return
        props = cls._cache.get((id(cls.model), material))
        if props is None:
            m = cls.model
            mid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, material)
            props = None if mid < 0 else (mid, m.mat_rgba[mid].copy(), float(m.mat_specular[mid]),
                                          float(m.mat_shininess[mid]), float(m.mat_reflectance[mid]),
                                          float(m.mat_emission[mid]))
            cls._cache[(id(cls.model), material)] = props
        if props is None:
            return
        geom.matid, geom.rgba, geom.specular, geom.shininess, geom.reflectance, geom.emission = props


_WHITE = np.ones(4, dtype=np.float32)


def add_geom(scene, kind, size, pos, matrix, material):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    rgba = _WHITE if isinstance(material, str) else np.asarray(material, dtype=np.float32)
    mj.mjv_initGeom(geom, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                    np.asarray(matrix, dtype=float).ravel(), rgba)
    if isinstance(material, str):
        Look.apply(geom, material)
    scene.ngeom += 1
    return geom


def add_capsule(scene, start, end, radius, material, kind=mj.mjtGeom.mjGEOM_CAPSULE):
    if np.linalg.norm(end - start) < 1e-7:
        return
    geom = add_geom(scene, kind, [1, 1, 1], np.zeros(3), np.eye(3), material)
    if geom is not None:
        mj.mjv_connector(geom, kind, radius, start, end)


def add_bar(scene, start, end, width, thickness, up, material):
    """Flat bar (box) from start to end, its wide face perpendicular to `up`."""
    axis = end - start
    length = np.linalg.norm(axis)
    if length < 1e-7:
        return
    axis = axis / length
    side = np.cross(up, axis)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross([1, 0, 0], axis)
    side /= np.linalg.norm(side)
    normal = np.cross(axis, side)
    add_geom(scene, mj.mjtGeom.mjGEOM_BOX, [width/2, thickness/2, length/2], (start + end)/2,
             np.column_stack((side, normal, axis)), material)


# Seven-segment digits for the monitor read-outs.
_SEGMENTS = {"0": "abcfed", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
             "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg", "/": "s"}
_SEG_LINES = {"a": ((0, 1), (1, 1)), "b": ((1, 1), (1, .5)), "c": ((1, .5), (1, 0)),
              "d": ((0, 0), (1, 0)), "e": ((0, .5), (0, 0)), "f": ((0, 1), (0, .5)),
              "g": ((0, .5), (1, .5)), "s": ((.1, 0), (.9, 1))}


def _screen(u, v, lift=.0015):
    """Point on the monitor screen; u, v in [-1, 1]."""
    return (MONITOR_CENTER + MONITOR_NORMAL*lift
            + MONITOR_RIGHT*u*MONITOR_HALF[0] + MONITOR_UP*v*MONITOR_HALF[1])


def _digits(scene, text, u, v, height, material):
    width = height*.55
    for i, ch in enumerate(text):
        x0 = u + i*width*1.45
        for seg in _SEGMENTS.get(ch, ""):
            (a, b), (c, d) = _SEG_LINES[seg]
            start = _screen(x0 + a*width, v + b*height)
            end = _screen(x0 + c*width, v + d*height)
            add_capsule(scene, start, end, .0016, material)


def _ecg(phase):
    """Lead II-like waveform, phase in [0, 1) over one heartbeat."""
    bump = lambda centre, width, amp: amp*np.exp(-((phase - centre)/width)**2)
    return (bump(.12, .035, .12) - bump(.235, .008, .10) + bump(.255, .010, 1.0)
            - bump(.275, .010, .25) + bump(.48, .06, .25))


def _pleth(phase):
    return (np.exp(-((phase - .30)/.09)**2) + .35*np.exp(-((phase - .55)/.07)**2)) * .9


def draw_monitor(scene, t):
    """Scrolling ECG and SpO2 traces with numeric read-outs, synced to HEART_RATE."""
    beat = 60 / HEART_RATE
    sweep = 4.0                                  # seconds of signal across the trace area
    count = 72
    times = t - sweep + np.linspace(0, sweep, count)
    us = np.linspace(-.95, .38, count)
    for v0 in (.42, -.18):
        add_capsule(scene, _screen(-.95, v0 - .25, .001), _screen(.38, v0 - .25, .001), .0006, "grid")
    for trace, v0, scale, material in ((_ecg, .42, .32, "ecg"), (_pleth, -.18, .28, "pleth")):
        points = [_screen(u, v0 - .2 + scale*trace((tt % beat)/beat)) for u, tt in zip(us, times)]
        # Fade the oldest part of the sweep like a real erase bar.
        for k in range(len(points) - 1):
            if k > 3:
                add_capsule(scene, points[k], points[k + 1], .0011, material)
    _digits(scene, str(HEART_RATE), .52, .30, .26, "ecg")
    _digits(scene, "98", .52, -.30, .26, "pleth")
    _digits(scene, "118/76", .40, -.88, .14, "nibp")
    pulse = (t % beat)/beat < .12
    if pulse:
        add_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, [.006, 0, 0], _screen(.93, .82), np.eye(3), "ecg")


# --- Sound feedback (macOS afplay; silently skipped elsewhere) ----------------------
class Sounds:
    SYSTEM = Path("/System/Library/Sounds")
    EVENTS = {"pickup": ("Tink", .35), "setdown": ("Bottle", .18), "puncture": ("Pop", .45),
              "tie": ("Purr", .5), "trim": ("Tink", .5), "incise": ("Frog", .10)}

    def __init__(self):
        self.enabled = sys.platform == "darwin" and shutil.which("afplay") is not None
        self.muted = False
        self.next_beat = 0.0
        self.beep = None
        if self.enabled:
            self.beep = Path(tempfile.gettempdir()) / "or_monitor_beep.wav"
            if not self.beep.exists():
                rate = 22050
                t = np.arange(int(rate*.09)) / rate
                tone = np.sin(2*np.pi*988*t) * np.minimum(1, (.09 - t)*60) * np.minimum(1, t*400)
                with wave.open(str(self.beep), "wb") as out:
                    out.setnchannels(1)
                    out.setsampwidth(2)
                    out.setframerate(rate)
                    out.writeframes((tone*12000).astype("<i2").tobytes())

    def _play(self, path, volume):
        if self.enabled and not self.muted and path.exists():
            subprocess.Popen(["afplay", "-v", f"{volume:.2f}", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def play(self, event):
        name, volume = self.EVENTS.get(event, (None, 0))
        if name:
            self._play(self.SYSTEM / f"{name}.aiff", volume)

    def heartbeat(self, t):
        """Pulse-oximeter beep in time with the monitor."""
        beat = 60 / HEART_RATE
        if t >= self.next_beat:
            self.next_beat = (np.floor(t/beat) + 1)*beat + .255*beat
            if self.beep is not None:
                self._play(self.beep, .06)
