"""Hands-only modes: a trainee who only moves the gloved hand (no keys) finishes them, nothing
happens before the glove is ready, and the attempts replay."""

import numpy as np
import pytest

import glove_bypass
from conftest import moved_hands
from glove_input import SIDES
from keymap import GLOVES_READY
from procedures import PROCEDURES
from session import Session

UP = np.array([0., 0., 1.])
SPEED = .003      # metres per frame a hand moves (18 cm/s)


def suture_move(st):
    """Toward the yellow hole, then across the cut, then up."""
    guide, hand = st.guide, st.hands["right"]
    entry, exit_ = guide.holes[guide.k]
    if guide.phase == "reach":
        step = entry - st.active_tip(hand)
    elif guide.phase == "drive":
        step = exit_ - entry
        step[2] = 0
        step = step/np.linalg.norm(step)*SPEED + UP*.0005
    else:
        step = UP*SPEED
    return step


def bypass_move(st):
    """Over the target until the magnet has it, then lower; any wave of the hand cuts."""
    guide, hand = st.guide, st.hands["right"]
    if guide.done or (hand.tool_pose is None and hand.held is not None):
        return np.zeros(3)
    if glove_bypass.STAGES[guide.stage][0] == "cut" and guide.phase == "cut":
        return np.array([.002, 0, 0])
    gap = guide.target() - guide.tip(hand)
    sideways = np.linalg.norm(gap[:2])
    if sideways > glove_bypass.MAGNET_XY*.8:
        return np.r_[gap[:2]/sideways*SPEED, 0]
    return -UP*.002 if (gap[2] < 0 and guide.armed) else UP*SPEED


def drive(world, step_fn, st=None, session=None, jitter=0.0, seed=0, frames=4000):
    """Move the right hand by `step_fn` plus tremor until the guide is done. Returns frames used."""
    rng = np.random.default_rng(seed)
    delta = {side: np.zeros(3) for side in SIDES}
    now = 0.0
    for frame in range(frames):
        station = session.station if session is not None else st
        poses = moved_hands(delta, world.rest_poses)
        keys = [GLOVES_READY] if frame == 0 and session is not None else []
        if session is not None:
            now += 1/60
            session.step(now, poses, keys, [], live=True)
        else:
            station.update(poses, keys, ())
            station.draw(world.scene, frame/60)
        if station.guide.done:
            return frame
        step = step_fn(station)
        if np.linalg.norm(step) > SPEED:
            step = step/np.linalg.norm(step)*SPEED
        delta["right"] += step + rng.normal(0, jitter/3, 3)
        delta["left"] += rng.normal(0, jitter/3, 3)
    pytest.fail("the hands-only procedure did not finish")


@pytest.mark.parametrize("jitter", [0.0, .002])
def test_hands_only_suturing(world, jitter):
    st = world.station("skin", "glove")
    st.guide.ready = True
    drive(world, suture_move, st, jitter=jitter, seed=1)
    assert len(st.stitches) == 6


@pytest.mark.parametrize("jitter", [0.0, .002, .004])
def test_hands_only_bypass(world, jitter):
    st = world.station("bypass", "glove-bypass")
    st.guide.ready = True
    drive(world, bypass_move, st, jitter=jitter, seed=1)
    assert set(st.guide.sewn_at) == {"coronary artery", "ascending aorta"}
    assert len(st.marks) > 0                   # both cuts were made


def test_hands_only_bypass_does_not_progress_on_tremor_alone(world):
    st = world.station("bypass", "glove-bypass")
    st.guide.ready = True
    rng = np.random.default_rng(0)
    delta = {side: np.zeros(3) for side in SIDES}
    for _ in range(900):     # 15 s of a 4 mm tremor around the start
        delta["right"] = np.clip(delta["right"] + rng.normal(0, .004/3, 3), -.004, .004)
        st.update(moved_hands(delta, world.rest_poses), [], ())
    assert st.guide.stage == 0 and len(st.marks) == 0


@pytest.mark.parametrize("site,task", [("skin", "glove"), ("bypass", "glove-bypass")])
def test_nothing_happens_until_the_glove_is_ready(world, site, task):
    st = world.station(site, task)
    rng = np.random.default_rng(0)
    delta = {side: np.zeros(3) for side in SIDES}
    for _ in range(600):     # 10 s of waving the hand before the glove is locked
        delta["right"] = rng.normal(0, .03, 3)
        st.update(moved_hands(delta, world.rest_poses), [], ())
    assert not st.stitches and not st.marks
    assert "Rest your right hand flat" in st.action_prompt()
    st.update(world.rest_poses, [GLOVES_READY], ())
    assert "Rest your right hand flat" not in st.action_prompt()


@pytest.mark.parametrize("key,step_fn", [("skin-gloves", suture_move), ("heart-gloves-bypass", bypass_move)])
def test_hands_only_attempt_is_saved_and_replayed(world, store, key, step_fn):
    session = Session(world.model, world.data, PROCEDURES[key], store, now=0.0)
    drive(world, step_fn, session=session, jitter=.002, seed=2)
    assert session.last is not None and session.last["complete"]

    assert session.start_autopilot(100.0)
    now = 100.0
    for _ in range(8000):
        if session.autopilot.done:
            break
        now += 1/60
        session.step(now, {side: None for side in SIDES}, [], [])
    assert session.station.guide.done
