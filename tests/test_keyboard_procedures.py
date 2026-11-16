"""A trainee who follows every prompt with the keyboard finishes each guided procedure."""

import pytest

from conftest import next_keys
from procedures import PROCEDURES
from session import Session


def follow_prompts(world, station, stop, max_frames=3000):
    """Press the prompted keys until `stop(station)`; fail if a prompt repeats without progress."""
    last, repeats = None, 0
    for frame in range(max_frames):
        side, keys = next_keys(station)
        if side is None or stop(station):
            return frame
        assert station.active == side, f"keys on the {station.active} hand, step needs the {side}"
        station.update(world.rest_poses, keys, ())
        station.draw(world.scene, frame/60)
        prompt = station.plan()
        repeats = repeats + 1 if prompt == last else 0
        assert repeats < 40, f"stuck at {prompt}: {station.message}"
        last = prompt
    pytest.fail("ran out of frames")


@pytest.mark.parametrize("site", ["skin", "heart"])
def test_suture_the_laceration(world, site):
    station = world.station(site)
    follow_prompts(world, station, lambda st: len(st.stitches) >= st.site.stitch_slots)
    assert len(station.stitches) == station.site.stitch_slots == 5


def test_coronary_bypass(world):
    station = world.station("bypass")
    follow_prompts(world, station, lambda st: st.procedure.done)
    assert station.procedure.done
    assert len(station.stitches) == 12      # six per anastomosis


def test_free_stitching_follows_the_forceps(world):
    """Free stitching: each stitch goes where the trainee moved the forceps, not to fixed slots."""
    station = world.station("skin", "stitch")
    moves = (0, 6, 6, -3)     # forceps nudges (arrow frames) before each stitch
    for _ in range(6000):
        side, text = station.plan()
        assert side is not None, text
        if text.startswith("arrows to"):
            step = moves[len(station.stitches) % len(moves)]
            for _ in range(abs(step)):
                station.update(world.rest_poses, [262 if step > 0 else 263], ())
            keys = [ord("N"), ord("F")]
        else:
            keys = next_keys(station)[1]
        station.update(world.rest_poses, keys, ())
        if len(station.stitches) >= 4:
            break
    alongs = [station.site.coords(entry)[0] for entry, _, _ in station.stitches]
    assert len(alongs) == 4
    assert len({round(a, 3) for a in alongs}) == 4      # four different places


@pytest.mark.parametrize("key", [k for k, p in PROCEDURES.items() if p.task in ("incision", "suture", "bypass")])
def test_session_scores_a_complete_attempt(world, store, key):
    now = 0.0
    session = Session(world.model, world.data, PROCEDURES[key], store, now=now)
    for _ in range(4000):
        if session.recorder is None:
            break
        _, keys = next_keys(session.station)
        now += 1/60
        session.step(now, world.rest_poses, keys, [])
    summary = session.last
    assert summary is not None and summary["complete"], f"{key} did not finish"
    assert summary["score"] > 90
    assert summary["tasks"]


def test_every_procedure_starts_with_a_prompt_and_checklist(world, store):
    session = Session(world.model, world.data, PROCEDURES["skin-suture"], store, now=0.0)
    for procedure in PROCEDURES.values():
        session.restart(0.0, procedure)
        _, prompt = session.station.plan()
        assert prompt
        assert session.station.checklist()
