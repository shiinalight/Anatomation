"""Attempts are recorded, scored, and replayed exactly by the autopilot."""

import numpy as np
import pytest

import motion_log
from conftest import next_keys
from glove_input import HAND_ANCHOR, JOINTS, SIDES, preview_pose
from or_scene import SITES, make_tissue
from station import CuttingStation


def record(world, site):
    """An attempt with glove-like jitter on the hands and idle pauses between actions."""
    st = world.station(site)
    recorder = motion_log.Recorder(site, "keyboard", sorted(JOINTS), HAND_ANCHOR, 0.0)
    rng = np.random.default_rng(1)
    now = 0.0

    def frame(poses, keys):
        nonlocal now
        now += 1/60
        st.update(poses, keys, ())
        recorder.frame(now, poses, keys, [], st, st.log)
        st.log.clear()

    for _ in range(4000):
        side, keys = next_keys(st)
        if side is None or (st.procedure is None and len(st.stitches) >= 5):
            break
        poses = {s: {j: p + rng.normal(0, .0004, 3) for j, p in preview_pose(-1.2, s).items()}
                 for s in SIDES}
        for _ in range(int(rng.integers(0, 40))):
            frame(poses, [])
        frame(poses, keys)
    return st, recorder.finish(now, True, {"stitches": len(st.stitches)})


@pytest.mark.parametrize("site", ["skin", "bypass"])
def test_autopilot_replays_the_attempt_exactly(world, site):
    original, attempt = record(world, site)
    assert attempt.summary["score"] > 95

    pilot = motion_log.Autopilot(attempt.blob)
    replay = CuttingStation(make_tissue(world.model, world.data, SITES[pilot.site]))
    while not pilot.done:
        replay.update(*pilot.next())

    assert len(replay.stitches) == len(original.stitches) > 0
    for (a_in, a_out, _), (b_in, b_out, _) in zip(original.stitches, replay.stitches):
        np.testing.assert_allclose(a_in, b_in)
        np.testing.assert_allclose(a_out, b_out)
    if replay.procedure is not None:
        assert replay.procedure.done

    trajectory = motion_log.robot_trajectory(attempt.blob)
    assert trajectory["frames"] and set(trajectory["frames"][0]) >= {"left", "right"}
