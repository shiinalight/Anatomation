"""Glove input: yaw calibration, the joint mapping into the room, and the RGMP stream format."""

import json
import struct

import numpy as np
import pytest

from glove_input import (CHAINS, HAND_ANCHOR, JOINTS, GloveStream, Heading, _rgmp_size,
                         glove_poses, preview_pose, world_pose)
from station import Hand


def openxr_hand(yaw_deg=0.0):
    """A flat right hand in OpenXR coordinates (+Y up, -Z forward), turned by `yaw_deg`."""
    scene = preview_pose(0.0, "right")
    wrist = scene["wrist"]
    angle = np.radians(yaw_deg)
    turn = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    points = {}
    for name, p in scene.items():
        x, y, z = turn @ (p - wrist)
        points[name] = np.array([x, z, -y])     # scene (x, y forward, z up) -> OpenXR
    return points


def test_heading_locks_after_the_hand_rests_flat():
    heading = Heading()
    points = openxr_hand(yaw_deg=40)
    now = 0.0
    while not heading.locked and now < 2:
        world_pose(points, heading, now, HAND_ANCHOR["right"])
        now += 1/60
    assert heading.locked
    assert now == pytest.approx(Heading.LOCK_TIME, abs=.05)


def test_locked_hand_points_into_the_scene_whatever_its_real_heading():
    for yaw in (-60, 0, 75):
        heading = Heading()
        for frame in range(40):
            pose = world_pose(openxr_hand(yaw), heading, frame/60, HAND_ANCHOR["right"])
        forward = pose["middle_proximal"] - pose["wrist"]
        assert forward[1] > .9*np.linalg.norm(forward), f"yaw {yaw}: fingers not toward the screen"
        np.testing.assert_allclose(pose["wrist"], HAND_ANCHOR["right"])


def test_rgmp_sizes():
    assert _rgmp_size("FLOAT") == 4
    assert _rgmp_size("FLOAT[3]") == 12
    assert _rgmp_size("DOUBLE[4, 4]") == 128
    with pytest.raises(ValueError):
        _rgmp_size("STRING")


def test_glove_stream_decodes_a_right_glove():
    joints = sorted(JOINTS)
    stream = GloveStream()
    stream._define({
        "device_id": 7, "device_info": {"hand": "right"},
        "groups": [{"name": "joints_openxr", "streams": [
            {"target_frame": f"xr_{name}", "measure_type": "TRANSFORM", "data_type": "FLOAT[7]"}
            for name in joints]}]})
    values = b"".join(struct.pack("<7f", i, i + .5, -i, 0, 0, 0, 1) for i in range(len(joints)))
    stream._data(struct.pack("<IIQ", 7, 0, 0) + values)
    _, positions = stream.latest["right"]
    np.testing.assert_allclose(positions[joints[3]], [3, 3.5, -3])
    assert json.dumps(sorted(positions)) == json.dumps(joints)


def test_keyboard_hands_stand_in_for_missing_gloves():
    stream = GloveStream()
    hands = {side: Hand(side) for side in ("left", "right")}
    poses, status = glove_poses(stream, hands, preview=False, now=0.0, reset_heading=False)
    assert status == "Waiting for the gloves (keyboard hands)"
    assert set(poses["left"]) == JOINTS
    assert set(CHAINS) == {"thumb", "index", "middle", "ring", "little"}
