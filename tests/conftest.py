"""Shared fixtures: one compiled operating-room model for the whole run (building it takes a
few seconds), and helpers that play a trainee: by keyboard, or by moving a gloved hand."""

from pathlib import Path
import sys

import mujoco as mj
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glove_input import SIDES, preview_pose  # noqa: E402
from or_scene import SITES, Look, build_model, make_tissue  # noqa: E402
from station import CuttingStation  # noqa: E402


class World:
    def __init__(self):
        self.model = build_model()
        Look.model = self.model
        self.data = mj.MjData(self.model)
        self.scene = mj.MjvScene(self.model, 10000)
        self.rest_poses = {side: preview_pose(-1.2, side) for side in SIDES}

    def station(self, site, task=None):
        """A fresh station at `site` ("skin", "heart" or "bypass") for `task`."""
        return CuttingStation(make_tissue(self.model, self.data, SITES[site], task), task)


@pytest.fixture(scope="session")
def world():
    return World()


@pytest.fixture
def store(tmp_path):
    from aiven_store import TrainingStore
    store = TrainingStore(root=tmp_path, uri="")   # local only: no Aiven
    yield store
    store.close()


def keys_for(prompt):
    """The keys a trainee presses for a prompt (without the procedure's step title)."""
    if prompt.startswith(("1 then G", "2 then G", "3 then G", "4 then G")):
        return [ord(prompt[0]), ord("G")]
    if prompt.startswith("G set down"):
        return [ord("G")]
    if prompt.startswith("N then F"):
        return [ord("N"), ord("F")]
    if prompt.startswith("N ("):
        return [ord("N")]
    if "x E" in prompt[:4] or prompt.startswith("E ("):
        return [ord("E")]
    if prompt.startswith("F ("):
        return [ord("F")]
    if prompt.startswith("Q ("):
        return [ord("Q")]
    if prompt.startswith("T ("):
        return [ord("T")]
    if prompt.startswith("left / right"):
        return [262]
    raise ValueError(f"no keys for prompt {prompt!r}")


def next_keys(station):
    """(side, keys) for the station's next step, as a trainee following the prompts."""
    side, text = station.plan()
    if side is None:
        return None, []
    if station.procedure is not None:
        text = text[len(station.procedure.head()):]
    return side, keys_for(text)


def moved_hands(delta, base):
    """Glove poses: each hand's resting pose moved by `delta[side]` (a real hand moving)."""
    return {side: {joint: point + delta[side] for joint, point in base[side].items()} for side in SIDES}
