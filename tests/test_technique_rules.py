"""The surgical technique is enforced: steps done out of order are refused with a message."""

import numpy as np


def press(world, station, side, *keys):
    """Press `keys` with the keyboard on `side` (as if the trainee pressed Z when needed)."""
    for key in keys:
        station.active = side
        station.need = station.plan()[0]
        station.update(world.rest_poses, [ord(key) if isinstance(key, str) else key], ())


def test_suture_steps_must_follow_the_technique(world):
    st = world.station("skin")
    press(world, st, "right", "2", "G", "N", "E", "E", "F")
    assert st.entry is None, "needle passed without the forceps lifting the edge"
    assert "forceps" in st.message

    press(world, st, "left", "3", "G", "N", "F")
    press(world, st, "right", "F")
    assert st.entry is not None

    press(world, st, "right", "N", "E", "E", "F")
    assert st.exit is None, "exit allowed without a fresh grip on the opposite edge"
    press(world, st, "left", "N", "F")
    press(world, st, "right", "F")
    assert st.exit is not None

    press(world, st, "right", "T")
    assert st.throws == 0, "knot tied before the suture was pulled through"

    press(world, st, "left", "N", "F")
    assert st.needle_grasp is not None
    press(world, st, "left", "G")
    assert st.needle_grasp is None, "setting the forceps down should let the needle slip back"

    press(world, st, "left", "G", "N", "F", "Q", "Q", "Q")
    assert st.pulled

    press(world, st, "right", "T", "T")
    assert len(st.stitches) == 0 and st.throws == 2
    press(world, st, "right", "T")
    assert len(st.stitches) == 1, "the third throw completes the stitch"


def test_skin_cut_needs_a_steadying_hand(world):
    st = world.station("skin")
    press(world, st, "right", "1", "G")
    blade = st.hands["right"]
    st.update_tool_pose(blade)
    blade.offset += np.array([.04, .03, .958]) - st.active_tip(blade)   # just above the skin
    for _ in range(3):
        press(world, st, "right", "E", 262, 262)
    assert len(st.marks) == 0, "the blade opened unsteadied skin"
    assert "Steady" in st.message

    press(world, st, "left", "N", "F")
    for _ in range(4):
        press(world, st, "right", "E", 262)
    assert len(st.marks) > 0, "the blade did not cut with the skin stretched"


def test_bypass_cuts_need_no_steadying_hand(world):
    st = world.station("bypass")
    assert not st.needs_steadying()
