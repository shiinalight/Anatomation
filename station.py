"""The operating station: both hands, their instruments, and the cutting and suturing rules.

CuttingStation holds the state of one attempt at one site. Each frame, `update()` takes the
hand poses and keys and advances the technique; `plan()` names the next action, `checklist()`
the procedure's steps, and `draw()` adds everything that moves to the scene. Guided procedures
(bypass, incision) live in `procedure`; hands-only modes in `guide`.
"""

from collections import deque
import time

import mujoco as mj
import numpy as np

from bypass import STITCHES_PER_ANASTOMOSIS
from glove_bypass import GloveBypass
from glove_suture import GloveSuture
from glove_input import OTHER, SIDES, Heading
from instruments import (PICKUP_RADIUS, TOOL_HOME, TOOL_NAMES, TOOL_TIP, TRAY_DIRECTION,
                         draw_instrument, draw_path, instrument_axes)
from keymap import (BOTH_HANDS, GLOVES_READY, GLOVES_WAIT, MOVES, RESET_HANDS, RESET_TOOLS,
                    SELECT_HAND, SWITCH_HAND)
from or_scene import TABLE_MAX, TABLE_MIN, add_capsule, add_geom, surface_strip
from procedures import make_procedure


MAX_DEPTH = .006
PULL_HEIGHT = .009        # forceps lift that pulls the suture through after the needle exits
THROWS = ("double wrap", "single wrap, reversed", "single wrap, squares the knot")
UP = np.array([0., 0., 1.])


class Hand:
    """One gloved hand: its keyboard offset, the tool it holds and what it is holding on to."""

    def __init__(self, side):
        self.side = side
        self.label = side.capitalize()
        self.heading = Heading()
        self.pose = None          # glove joints in the scene, before `offset`
        self.offset = np.zeros(3)
        self.selected = 2 if side == "left" else 1
        self.held = None
        self.tool_pose = None
        self.pinch_finger = None
        self.pinching = False
        self.guide_tip = None     # assisted needle tip target (N)
        self.guide_lower_steps = 0
        self.previous_tip = None
        self.depth = 0.0
        self.cut_line = None      # marked incision the scalpel is guided along (bypass)
        self.grip = None          # tissue point held by the forceps or pressed by the fingers
        self.grip_pass = None     # which needle pass the grip was taken for (see `passes`)

    @property
    def tool(self):
        return "empty" if self.held is None else TOOL_NAMES[self.held].lower()


GUIDES = {"glove": GloveSuture, "glove-bypass": GloveBypass}   # hands-only (telemetry) modes


class CuttingStation:
    """Both hands' instruments, cutting and suturing on the tissue surface (via `tissue`),
    with the incision and stitch geometry of the chosen site.

    The technique follows theatre practice. The right hand cuts and drives the needle, and
    the left hand works against it:
      - cutting: the left hand steadies the tissue next to the blade (fingers stretching the
        skin, or forceps holding the tissue); the blade does not open unsteadied tissue (not in
        free practice or the bypass, see needs_steadying);
      - each needle pass: the forceps first lift the edge the needle goes through;
      - after the exit: the forceps grasp the needle tip and pull the suture through (Q);
      - knot: instrument tie with the needle holder, three throws (T).
    Any tool can be taken by either hand; the prompts name the usual one. Keyboard control
    follows whichever hand the next step needs, and Z swaps it by hand.
    """

    def __init__(self, tissue, task=None):
        self.tissue = tissue
        self.site = tissue.site
        self.task = task or ("bypass" if self.site.procedure == "cabg" else "suture")
        self.procedure = make_procedure(self.task, tissue)
        self.hands = {side: Hand(side) for side in SIDES}
        self.active = "right"
        self.both = False             # arrows and Q/E move both hands together
        self.need = None              # hand the current step needs (keyboard follows it)
        self.marks = deque(maxlen=450)
        self.stitches = []
        self.entry = None
        self.exit = None
        self.passes = 0               # needle passes so far: each needs a fresh forceps grip
        self.needle_grasp = None      # forceps tip height when it took the needle at the exit
        self.pulled = False
        self.throws = 0
        self.rest = [(position.copy(), TRAY_DIRECTION.copy()) for position in TOOL_HOME]
        self._message, self._message_at = "", -1e9
        self.log = []                    # (kind, detail) of each step, for recording and scoring
        self.events = []                 # sound cues for the display loop
        self.blood = deque(maxlen=220)   # (position, born, final radius)
        self.clock = 0.0
        reach = .004 if self.procedure is not None else max(.004, self.site.bite_guide*.9)
        self.edge_reach = reach                                   # forceps grip to needle pass
        self.steady_reach = max(.012, self.site.half*.8)          # steadying hand to blade
        self.guide = GUIDES[self.task](self) if self.task in GUIDES else None   # hands-only modes

    # --- small helpers ---------------------------------------------------------------
    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, text):
        """A warning or a refused action (logged as rejected). See also inform()."""
        repeat = text == self._message
        self._message, self._message_at = text, time.monotonic()
        if text and not repeat:      # a warning that stays up (blade on a vessel) counts once
            self.note("rejected", reason=text)

    def inform(self, text):
        self._message, self._message_at = text, time.monotonic()

    def note(self, kind, **detail):
        self.log.append((kind, detail))

    def moving_hands(self):
        """The hands the arrows and Q/E move: the active one, or both."""
        hands = self.hands.values() if self.both else [self.hands[self.active]]
        return [hand for hand in hands if hand.pose is not None]

    def other(self, hand):
        return self.hands[OTHER[hand.side]]

    def holder(self, index):
        return next((hand for hand in self.hands.values() if hand.held == index), None)

    @property
    def depth(self):
        return max(hand.depth for hand in self.hands.values())

    def surface(self, xy, lift=0.0):
        point = self.tissue.point(xy[:2], lift)
        if point is None and hasattr(self.tissue, "any_point"):
            point = self.tissue.any_point(xy[:2], lift)
        return point

    def surface_height(self, x, y):
        z = self.tissue.height(x, y)
        if z is None and hasattr(self.tissue, "any_height"):
            z = self.tissue.any_height(x, y)
        return z

    @staticmethod
    def grip_point(pose, finger):
        return (pose["thumb_tip"] + pose[f"{finger}_tip"]) / 2

    def fingertip(self, hand):
        """Where an empty hand touches the tissue: between thumb and index tips."""
        return self.grip_point(hand.pose, "index") + hand.offset

    def contact(self, hand):
        return self.fingertip(hand) if hand.held is None else self.active_tip(hand)

    # --- holding and moving tools ------------------------------------------------------
    def park(self, hand):
        if hand.held is not None:
            grip, direction = hand.tool_pose
            xy = np.clip(grip[:2], TABLE_MIN, TABLE_MAX)
            flat = np.array([direction[0], direction[1], 0.])
            length = np.linalg.norm(flat)
            flat = flat / length if length > .001 else TRAY_DIRECTION.copy()
            x, y = xy
            self.rest[hand.held] = (np.array([x, y, self.tissue.rest_height(x, y) + .004]), flat)
            self.events.append("setdown")
            if hand.held == 2 and self.needle_grasp is not None and self.throws == 0:
                self.needle_grasp, self.pulled = None, False   # the needle slips back
            self.note("setdown", hand=hand.side, tool=hand.held)
            self.inform(f"{hand.label} hand: G (pick up selected tool)")
        hand.held = hand.tool_pose = hand.previous_tip = hand.grip = None
        hand.cut_line = hand.pinch_finger = hand.guide_tip = None
        hand.guide_lower_steps = 0
        hand.depth = 0.0

    def park_all(self):
        for hand in self.hands.values():
            if hand.held is not None and hand.pose is not None:
                hand.tool_pose = self.pose_for_tool(hand)
            self.park(hand)

    def update(self, poses, keys, held=()):
        """`poses`: side -> scene joints (or None); returns side -> drawn joints."""
        for side, hand in self.hands.items():
            pose = poses.get(side)
            hand.pose = pose
            if pose is None:
                self.park(hand)
                hand.pinching = False
                continue
            thumb = pose["thumb_tip"]
            closest = min(("index", "middle", "ring"),
                          key=lambda name: np.linalg.norm(pose[f"{name}_tip"] - thumb))
            hand.closest = closest
            gap = np.linalg.norm(pose[f"{hand.pinch_finger or closest}_tip"] - thumb)
            hand.was_pinching = hand.pinching
            hand.pinching = gap < (.042 if hand.was_pinching else .032)
            if hand.pinch_finger and not hand.pinching:
                self.park(hand)

        for key in keys:
            self.key(key)
        for moving in self.moving_hands():
            for code, amount in held:
                if not self.guided_lowering(moving, code):
                    self.move(moving, code, amount)

        for hand in self.hands.values():
            if hand.pose is None:
                continue
            if hand.held is None and hand.pinching and not hand.was_pinching:
                point = self.grip_point(hand.pose, hand.closest) + hand.offset
                free = [i for i in range(4) if self.holder(i) is None]
                if free:
                    nearest = min(free, key=lambda i: np.linalg.norm(self.rest[i][0] - point))
                    if np.linalg.norm(self.rest[nearest][0] - point) <= PICKUP_RADIUS:
                        self.pick_up(hand, nearest, hand.closest)
            self.update_tool_pose(hand)
            if hand.grip is not None:   # a grip lets go when the hand moves away from it
                gap = self.contact(hand) - hand.grip
                if np.linalg.norm(gap[:2]) > .03 or gap[2] > .045:
                    hand.grip = None
            if hand.held == 0 and self.guide is None:     # (a hands-only guide cuts by itself)
                grip, direction = hand.tool_pose
                self.cut(hand, grip + direction*TOOL_TIP[0])
            else:
                self.cut(hand, None)
        self.check_pull()
        if self.guide is not None:
            self.guide.update()

        side, _ = self.plan()
        if side != self.need:          # keyboard control follows the step's hand
            self.need = side
            if side is not None:
                self.active = side
        return {side: None if hand.pose is None else
                {name: point + hand.offset for name, point in hand.pose.items()}
                for side, hand in self.hands.items()}

    def pose_for_tool(self, hand):
        grip = self.grip_point(hand.pose, hand.pinch_finger or "index") + hand.offset
        direction = hand.pose["middle_proximal"] - hand.pose["wrist"]
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        direction[2] -= .36
        return grip, direction / max(np.linalg.norm(direction), 1e-9)

    def update_tool_pose(self, hand):
        if hand.held is None or hand.pose is None:
            return
        hand.tool_pose = self.pose_for_tool(hand)
        if hand.held == 1 and hand.guide_tip is not None:
            # Keep the assisted needle tip at its target as the glove rotates.
            hand.offset += hand.guide_tip - self.active_tip(hand)
            hand.tool_pose = self.pose_for_tool(hand)

    def active_tip(self, hand):
        grip, direction = hand.tool_pose
        tip = grip + direction*TOOL_TIP[hand.held]
        if hand.held == 1:
            _, up = instrument_axes(direction)
            tip -= up*.015
        return tip

    def pick_up(self, hand, index, finger=None):
        owner = self.holder(index)
        if owner is not None:
            self.message = f"The {owner.side} hand holds the {TOOL_NAMES[index].lower()}"
            return
        hand.held = hand.selected = index
        hand.pinch_finger = finger
        hand.grip = None
        hand.offset[:] = self.rest[index][0] - self.grip_point(hand.pose, finger or "index")
        hand.tool_pose = self.pose_for_tool(hand)
        self.events.append("pickup")
        self.note("pickup", hand=hand.side, tool=index)
        self.inform("")

    def move(self, hand, code, amount):
        axis = MOVES[code][0]
        self.update_tool_pose(hand)
        if hand.cut_line is not None and hand.held == 0 and axis != 1:
            if axis == 0:
                self.slide_along_cut(hand, amount)
            else:
                hand.offset[2] += amount
            return
        hand.cut_line = None
        hand.guide_tip = None
        hand.guide_lower_steps = 0
        limit = .75 if axis < 2 else .45
        hand.offset[axis] = np.clip(hand.offset[axis] + amount, -limit, limit)

    def slide_along_cut(self, hand, amount):
        """Move the blade along the marked line, keeping its depth below the surface."""
        line, tip = hand.cut_line, self.active_tip(hand)
        along, _ = line.coords(tip)
        below = (self.tissue.any_height(*tip[:2]) or tip[2]) - tip[2]
        along = float(np.clip(along + amount*.5, -line.half - .003, line.half + .003))
        xy = line.point(along)
        target = np.array([*xy, self.tissue.any_height(*xy) - below])
        hand.offset += target - tip

    @staticmethod
    def guided_lowering(hand, code):
        return hand.held == 1 and hand.guide_tip is not None and code == ord("E")

    # --- keys ----------------------------------------------------------------------------
    def key(self, code):
        if code == SWITCH_HAND:
            self.active = OTHER[self.active]
            self.inform(f"Keyboard moves the {self.active} hand")
            return
        hand = self.hands[self.active]
        if code == ord("R"):
            self.reset()
            return
        if code in (GLOVES_READY, GLOVES_WAIT):
            if self.guide is not None:
                self.guide.ready = code == GLOVES_READY
            return
        if code == RESET_HANDS or (code == RESET_TOOLS and self.guide is not None):
            self.reset_hands()
            return
        if self.guide is not None and code not in (BOTH_HANDS, *SELECT_HAND.values(), *MOVES):
            return      # hands-only suturing: the tools stay in hand (arrows still nudge a hand)
        if code == RESET_TOOLS:
            self.park_all()
            self.rest = [(position.copy(), TRAY_DIRECTION.copy()) for position in TOOL_HOME]
            self.inform("Instruments back in their trays")
            return
        if code in SELECT_HAND.values():
            self.active = next(side for side, c in SELECT_HAND.items() if c == code)
            self.both = False
            self.inform(f"Arrows and Q/E move the {self.active} hand")
            return
        if code == BOTH_HANDS:
            self.both = not self.both
            self.inform("Arrows and Q/E move both hands" if self.both else
                        f"Arrows and Q/E move the {self.active} hand (Z swaps)")
            return
        if hand.pose is None:
            return
        if code in MOVES:
            for moving in self.moving_hands():
                if self.guided_lowering(moving, code):
                    # Guided needle lowering stays in discrete taps.
                    if moving.guide_lower_steps:
                        moving.guide_tip[2] += MOVES[code][1]
                        moving.guide_lower_steps -= 1
                else:
                    self.move(moving, code, MOVES[code][1])
        elif ord("1") <= code <= ord("4"):
            hand.selected = code - ord("1")
            self.inform(f"{hand.label} hand: G (pick up {TOOL_NAMES[hand.selected].lower()})")
        elif code == ord("G"):
            if hand.held is not None:
                self.update_tool_pose(hand)
                self.park(hand)
            else:
                self.pick_up(hand, hand.selected)
        elif code == ord("F"):
            self.update_tool_pose(hand)
            self.act(hand)
        elif code == ord("T"):
            self.throw(hand)
        elif code == ord("X"):
            self.update_tool_pose(hand)
            self.trim(hand)
        elif code == ord("N"):
            self.update_tool_pose(hand)
            self.align(hand)
        elif code == ord("C"):
            self.park(hand)
            hand.offset.fill(0)

    def reset_hands(self):
        """Both hands back to where they start, in front of the surgeon, keeping their tools."""
        for hand in self.hands.values():
            hand.offset.fill(0)
            hand.grip = hand.guide_tip = hand.cut_line = hand.previous_tip = None
            hand.guide_lower_steps = 0
            hand.depth = 0.0
            if hand.held is not None and hand.pose is not None:
                hand.tool_pose = self.pose_for_tool(hand)
        if self.needle_grasp is not None and not self.pulled:
            self.needle_grasp = None       # the forceps let go of the needle
        if self.guide is not None:
            self.guide.reset_hands()
        self.note("reset_hands")
        self.inform("Hands back to their start")

    def reset(self):
        self.marks.clear()
        self.stitches.clear()
        self.blood.clear()
        self.entry = self.exit = self.needle_grasp = None
        self.pulled, self.throws = False, 0
        for hand in self.hands.values():
            hand.previous_tip = hand.grip = hand.guide_tip = hand.cut_line = None
            hand.guide_lower_steps = 0
        if self.procedure is not None:
            self.procedure = make_procedure(self.task, self.tissue)
        if self.guide is not None:
            self.guide = type(self.guide)(self)
            self.reset_hands()
        self.note("reset")
        self.inform("Repair cleared")

    # --- targets ------------------------------------------------------------------------
    def cutting_step(self):
        """True when the step is a cut: a bypass cut, or a free cut with the scalpel held."""
        if self.procedure is not None:
            return self.procedure.cutting()
        return self.holder(0) is not None

    def needle_target(self):
        """Surface point of the next needle pass (guided placement), or None."""
        if self.procedure is not None:
            return self.procedure.needle_target(self)
        if self.exit is not None:
            return None
        site = self.site
        if self.entry is None and self.task == "stitch":
            along, side = self.chosen_spot()
        elif self.entry is None:
            along = site.slot(len(self.stitches) % site.stitch_slots)
            side = -site.bite_guide
        else:
            along, entry_side = site.coords(self.entry)
            side = -site.bite_guide*np.sign(entry_side)
        return self.tissue.point(site.point(along, side))

    def chosen_spot(self):
        """Free stitching: (along, side) of the next first bite, where the trainee chose it.
        The forceps' grip on an edge (taken for this pass) fixes it; before that it follows
        the forceps (or else the needle holder) along the wound."""
        site = self.site
        forceps = self.holder(2)
        if forceps is not None and forceps.grip is not None and forceps.grip_pass == self.passes:
            along, side = site.coords(forceps.grip)
        else:
            along = side = None
            for tool in (forceps, self.holder(1)):
                if tool is not None and tool.tool_pose is not None:
                    a, b = site.coords(self.active_tip(tool))
                    if abs(a) <= site.along_limit and abs(b) <= site.bite_max*3:   # over the wound
                        along, side = a, b
                        break
            if along is None:      # not over the wound yet: the next of the usual places
                return site.slot(len(self.stitches) % site.stitch_slots), -site.bite_guide
        along = float(np.clip(along, -site.along_limit*.9, site.along_limit*.9))
        return along, site.bite_guide*(1.0 if side > 0 else -1.0)

    def edge_point(self, target):
        """Where the forceps hold the edge for a needle pass at `target`."""
        if self.procedure is not None:
            return self.procedure.edge_point(target)
        along, side = self.site.coords(target)
        return self.tissue.point(self.site.point(along, side*.45))

    def steady_point(self, hand):
        """Where `hand` steadies the tissue for the other hand's blade."""
        if self.procedure is not None and self.procedure.cutting():
            return self.procedure.steady_point()
        blade = self.holder(0)
        if blade is not None and blade is not hand and blade.tool_pose is not None:
            tip = self.active_tip(blade)
            reach = min(.025, self.steady_reach*.5)
            for sign in (-1, 1):
                point = self.surface(tip[:2] + self.site.across*reach*sign)
                if point is not None:
                    return point
        return self.surface(self.site.point(0, -self.site.bite_guide*2))

    def edge_held(self, forceps, target):
        """True if `forceps` hold the edge for the needle pass at `target` (taken for this pass)."""
        if forceps is None or forceps.held != 2 or forceps.grip is None or target is None:
            return False
        if forceps.grip_pass != self.passes:
            return False
        if np.linalg.norm(forceps.grip[:2] - target[:2]) > self.edge_reach:
            return False
        if self.procedure is None:
            return self.site.coords(forceps.grip)[1]*self.site.coords(target)[1] > 0
        return True

    def needs_steadying(self):
        """True if the blade cuts only with the other hand steadying the tissue: guided incisions
        and the laceration tasks, not free practice or the bypass."""
        if self.task == "free":
            return False
        return self.procedure is None or self.procedure.steady_cut

    def steadied(self, blade):
        """True if the other hand steadies the tissue where `blade` cuts. In a bypass the
        forceps must hold the vessel wall beside the marked line."""
        other = self.other(blade)
        if other.grip is None:
            return False
        if self.procedure is not None and self.procedure.cutting():
            return self.procedure.steady_ok(other)
        if blade.tool_pose is None:
            return False
        tip = self.active_tip(blade)
        return np.linalg.norm(other.grip[:2] - tip[:2]) <= self.steady_reach

    # --- actions -----------------------------------------------------------------------
    def align(self, hand):
        """N: put the hand's tool (or fingertips) where the next step needs it."""
        if hand.held is None:
            target = self.steady_point(hand)
            if target is None:
                return
            hand.grip = None
            hand.offset += target + UP*.006 - self.fingertip(hand)
            self.inform("")
            return
        if hand.held == 0:
            if self.procedure is not None and self.procedure.align_blade(self, hand):
                return
            self.inform("Arrows move, E lowers the blade")
            return
        if hand.held == 3:
            if self.stitches:
                hand.offset += self.stitches[-1][1] + UP*.008 - self.active_tip(hand)
                self.inform("")
            return
        if hand.held == 2:
            hand.grip = None
            self.inform("")
            if self.exit is not None and self.needle_grasp is None:
                target = self.exit + UP*.003
            elif self.needle_grasp is not None:
                return
            elif self.cutting_step():
                target = self.steady_point(hand)
            else:
                needle = self.needle_target()
                target = None if needle is None else self.edge_point(needle)
            if target is None:
                target = self.tissue.point(self.site.point(*self.site.forceps_target))
            if target is not None:
                hand.offset += target + UP*.003 - self.active_tip(hand)
            return
        # Needle holder: guided placement above the next pass, lowered with E E.
        if self.exit is not None:
            return
        target = self.needle_target()
        if target is None:
            self.message = "The needle is not needed in this step"
            return
        hand.guide_tip = target + UP*.004
        hand.guide_lower_steps = 2
        self.inform("")

    def act(self, hand):
        """F: forceps or fingers grip / release; the needle holder passes the needle."""
        if hand.held == 1:
            self.puncture(hand)
            return
        if hand.held not in (None, 2):
            self.message = "F works with the needle holder, forceps or an empty hand"
            return
        tip = self.contact(hand)
        if (hand.held == 2 and self.exit is not None and self.needle_grasp is None
                and np.linalg.norm(tip - self.exit) <= .009):
            hand.grip = None
            self.needle_grasp = tip[2]
            self.events.append("pickup")
            self.note("grasp", hand=hand.side)
            self.inform("")
            return
        if hand.grip is not None:
            hand.grip = None
            self.note("release", hand=hand.side)
            self.inform("Released")
            return
        surface = self.surface_height(*tip[:2])
        reach = .009 if hand.held == 2 else .015
        if surface is None or abs(tip[2] - surface) > reach:
            self.message = "Lower onto the tissue first (E), then F"
            return
        hand.grip = np.array([*tip[:2], surface])
        hand.grip_pass = self.passes
        self.note("grip", hand=hand.side, tool=hand.held, at=[round(float(v), 4) for v in hand.grip])
        self.inform("")

    def puncture(self, hand):
        if self.exit is not None:
            self.message = "Forceps: take the needle and pull the suture through"
            return
        if hand.guide_tip is not None and hand.guide_lower_steps:
            return
        tip = self.active_tip(hand)
        if self.procedure is not None:
            target = self.procedure.needle_target(self)
            if target is None:
                self.message = "Not a suturing step"
                return
            point = self.procedure.puncture_point(target, tip)
            if point is None:
                self.message = "N to align on the next target, then E E and F"
                return
        else:
            point = self.site_puncture(tip)
            if point is None:
                return
        forceps = self.holder(2)
        if not self.edge_held(forceps, point):
            label = OTHER[hand.side].capitalize() if forceps is None else forceps.label
            self.message = f"{label} hand: lift this edge with the forceps first (N, F)"
            return
        guided = self.needle_target()
        error = None if guided is None else float(np.linalg.norm(point[:2] - guided[:2]))*1000
        if self.entry is None:
            self.entry = point
            hand.guide_tip = None
        else:
            self.exit = point
        self.note("puncture", hand=hand.side, edge="exit" if self.exit is not None else "entry",
                  error_mm=None if error is None else round(error, 2))
        self.passes += 1
        self.inform("")
        self.bleed(point, .0008 if self.procedure is not None else .0011)
        self.events.append("puncture")

    def site_puncture(self, tip):
        """Free needle placement on the skin or heart laceration: the point, or None."""
        if self.tissue.on_vessel(*tip[:2]):
            self.message = "Keep the needle off the coronary vessels"
            return None
        surface = self.tissue.height(*tip[:2])
        if surface is None or not surface - .012 <= tip[2] <= surface + .001:
            self.message = f"Lower the needle tip to the {self.site.noun}"
            return None
        point = self.tissue.point(tip[:2], .0015)
        along, side = self.site.coords(point)
        site = self.site
        if not (site.bite_min <= abs(side) <= site.bite_max and abs(along) <= site.along_limit):
            self.message = (f"Place the needle {site.bite_min*1000:.0f}-{site.bite_max*1000:.0f} mm "
                            "beside the wound")
            return None
        if self.entry is not None:
            entry_along, entry_side = site.coords(self.entry)
            if not (side*entry_side < 0 and abs(along - entry_along) <= site.exit_tolerance):
                self.message = "Exit must be straight across the entry"
                return None
        return point

    def check_pull(self):
        """The forceps holding the needle pull the suture through as they rise."""
        if self.needle_grasp is None or self.pulled:
            return
        forceps = self.holder(2)
        if forceps is None or forceps.tool_pose is None:
            return
        if self.active_tip(forceps)[2] - self.needle_grasp >= PULL_HEIGHT:
            self.pulled = True
            self.note("pulled", hand=forceps.side)
            self.inform("")
            self.events.append("trim")

    def throw(self, hand):
        """T: one throw of an instrument tie; the third completes the knot."""
        if hand.held != 1:
            self.message = "Tie with the needle holder (T)"
            return
        if self.exit is None:
            self.message = "Pass the needle through both edges first"
            return
        if not self.pulled:
            self.message = "Forceps: grasp the needle (N, F) and pull it through (Q)"
            return
        if len(self.stitches) >= 48:
            self.message = "Repair full - press R to reset"
            return
        self.throws += 1
        self.note("throw", hand=hand.side, throw=self.throws)
        self.inform("")
        self.events.append("tie")
        if self.throws < len(THROWS):
            return
        self.stitches.append((self.entry, self.exit, False))
        self.note("stitch", number=len(self.stitches))
        self.entry = self.exit = self.needle_grasp = hand.guide_tip = None
        self.pulled, self.throws = False, 0
        forceps = self.holder(2)
        if forceps is not None:
            forceps.grip = None
        if self.procedure is not None:
            self.procedure.stitch_tied(self)

    def trim(self, hand):
        if hand.held != 3 or not self.stitches:
            self.message = "4 then G (pick up scissors after tying)"
            return
        entry, exit, trimmed = self.stitches[-1]
        if np.linalg.norm(self.active_tip(hand) - exit) > .035:
            self.message = "N (align scissors) / X (trim thread)"
            return
        self.stitches[-1] = (entry, exit, True)
        self.events.append("trim")
        self.note("trim", stitch=len(self.stitches))
        self.inform("Thread trimmed")

    # --- guidance ----------------------------------------------------------------------
    def plan(self):
        """(hand side, prompt) for the next step of the technique."""
        def take(hand, index):
            name = TOOL_NAMES[index].lower()
            if hand.held is not None:
                return f"G set down the {hand.tool}, then {index + 1} + G ({name})"
            return f"{index + 1} then G ({name})"

        if self.guide is not None:
            return self.guide.plan()
        right = self.hands["right"]
        head = "" if self.procedure is None else self.procedure.head()
        if self.procedure is not None and self.procedure.done:
            return None, self.procedure.prompt_done()
        if self.task == "free":
            return None, "Free practice: cut and stitch anywhere, in any order (Z switches hands)"

        if self.cutting_step():
            blade = self.holder(0) or right
            steady = self.other(blade)
            if blade.held != 0:
                return blade.side, head + take(blade, 0)
            if self.procedure is not None:
                tools = self.procedure.steady_tools
                if not self.procedure.steady_cut:
                    pass               # the bypass: the blade cuts without the other hand
                elif steady.held not in tools:
                    if steady.held is not None and None in tools:
                        return steady.side, head + f"G set down the {steady.tool} (that hand steadies)"
                    return steady.side, head + take(steady, 2)
                elif not self.steadied(blade):
                    what = "forceps hold" if steady.held == 2 else "fingers stretch"
                    return steady.side, head + f"N then F ({what} the tissue beside the line)"
                if blade.cut_line is None:
                    return blade.side, head + "N (put the blade on the marked line)"
                if blade.depth <= 0:
                    return blade.side, head + "E (lower the blade into the tissue)"
                return blade.side, head + (f"left / right arrows cut along the line "
                                           f"({self.procedure.covered.mean():.0%})")
            if steady.held not in (None, 2):
                return steady.side, f"G set down the {steady.tool} (that hand steadies the tissue)"
            if steady.grip is None:
                what = "forceps hold" if steady.held == 2 else "fingers stretch"
                return steady.side, f"N then F ({what} the tissue beside the blade)"
            if not self.steadied(blade):
                return blade.side, f"Arrows: bring the blade next to the {steady.side} hand"
            return blade.side, "E lowers the blade, arrows cut (keep the tissue steady)"

        scissors = self.holder(3)
        if scissors is not None and self.exit is None:
            return scissors.side, "N (align scissors) / X (trim last knot)"
        needle = self.holder(1) or right
        forceps = self.holder(2) or self.other(needle)
        if needle.held != 1:
            return needle.side, head + take(needle, 1)
        if forceps.held != 2:
            return forceps.side, head + take(forceps, 2)
        if self.procedure is not None:
            n = f"stitch {self.procedure.tied + 1}/{STITCHES_PER_ANASTOMOSIS}"
            edge = "graft" if self.entry is None else "artery"
        else:
            n = f"stitch {len(self.stitches) + 1}"
            edge = "first edge" if self.entry is None else "opposite edge"
        if self.exit is None:
            if not self.edge_held(forceps, self.needle_target()):
                if self.task == "stitch" and self.entry is None:
                    return forceps.side, f"arrows to where {n} goes, then N, F (lift the edge)"
                return forceps.side, head + f"N then F (forceps lift the {edge}, {n})"
            if needle.guide_tip is None:
                return needle.side, head + f"N (align the needle on the {edge})"
            if needle.guide_lower_steps:
                return needle.side, head + f"{needle.guide_lower_steps}x E (lower the needle)"
            return needle.side, head + f"F (pass the needle through the {edge})"
        if self.needle_grasp is None:
            return forceps.side, head + "N then F (forceps grasp the needle tip)"
        if not self.pulled:
            pulled = 0
            if forceps.tool_pose is not None:
                pulled = (self.active_tip(forceps)[2] - self.needle_grasp) / PULL_HEIGHT
            return forceps.side, head + f"Q (pull the suture through, {min(max(pulled, 0), 1):.0%})"
        return needle.side, head + f"T (instrument tie, throw {self.throws + 1}/3: {THROWS[self.throws]})"

    def checklist(self):
        """The procedure's steps for the overlay: [(mark, text)] with mark "x" done,
        ">" in progress (the first step not done) and " " still to come."""
        def marks(items, indent=""):
            out, current = [], False
            for text, done in items:
                mark = "x" if done else (" " if current else ">")
                current = current or not done
                out.append((mark, indent + text))
            return out

        right = self.hands["right"]
        if self.guide is not None:
            return self.guide.checklist()
        if self.task == "free":
            return [(" ", "No set order: cut and stitch anywhere"),
                    (" ", f"{len(self.stitches)} stitches, {len(self.marks)} cut segments")]
        p = self.procedure
        if p is not None and p.done:
            return [("x", p.prompt_done().split(".")[0])]

        def cut_items():
            blade = self.holder(0) or right
            placed = blade.cut_line is not None or p.covered.any()
            who = "forceps or fingers" if None in p.steady_tools else "forceps"
            steady = ([(f"Left hand steadies ({who})", self.steadied(blade) or p.covered.any())]
                      if p.steady_cut else [])
            return ([("Scalpel in the right hand", blade.held == 0)] + steady +
                    [("Blade on the line (N, E)", placed and (blade.depth > 0 or p.covered.any())),
                     (f"Cut along the line: {p.covered.mean():.0%}", False)])

        def stitch_items(edge_names):
            needle = self.holder(1) or right
            forceps = self.holder(2) or self.other(needle)
            target = self.needle_target()
            first, second = edge_names
            return [("Needle holder (R) and forceps (L)", needle.held == 1 and forceps.held == 2),
                    (f"Lift the {first} (forceps)", self.entry is not None or
                     self.edge_held(forceps, target)),
                    (f"Needle through the {first}", self.entry is not None),
                    (f"Lift the {second}", self.exit is not None or
                     (self.entry is not None and self.edge_held(forceps, target))),
                    (f"Needle out through the {second}", self.exit is not None),
                    ("Grasp the needle tip (forceps)", self.needle_grasp is not None),
                    ("Pull the suture through (Q)", self.pulled),
                    (f"Instrument tie: throw {min(self.throws + 1, 3)}/3", False)]

        if self.task == "stitch":
            done = len(self.stitches)
            return ([(">", f"Stitch {done + 1}, anywhere along the wound ({done} tied)")] +
                    marks(stitch_items(("first edge", "opposite edge")), "  "))
        if p is None:       # suture the laceration
            slots = self.site.stitch_slots
            done = len(self.stitches)
            if done >= slots:
                return [("x", f"All {slots} stitches tied")]
            return ([(">", f"Stitch {done + 1} of {slots} ({done} tied)")] +
                    marks(stitch_items(("first edge", "opposite edge")), "  "))
        rows = []
        for i, (kind, _line, title) in enumerate(p.steps):
            mark = "x" if i < p.step else (">" if i == p.step else " ")
            rows.append((mark, f"{i + 1}. {title}"))
            if i == p.step:
                if kind == "cut":
                    rows += marks(cut_items(), "  ")
                else:
                    rows.append((">", f"  Stitch {p.tied + 1} of {STITCHES_PER_ANASTOMOSIS}"))
                    rows += marks(stitch_items(("graft", "artery")), "  ")
        return rows

    def action_prompt(self, short=False):
        """The next action; `short` leaves out the step title (the checklist shows it)."""
        if self._message and time.monotonic() - self._message_at < 2.5:
            return self._message
        side, text = self.plan()
        if short and self.procedure is not None and self.procedure.head():
            text = text.replace(self.procedure.head(), "", 1)
        if side is not None and side != self.active and not self.both and self.guide is None:
            # Z or a double-click put the keys on the other hand: the step's keys would go there
            return (f"{side.capitalize()} hand next, but the keys move the {self.active} hand: "
                    f"press Z (or double-click the {side} hand)")
        return text if side is None else f"{side.capitalize()} hand: {text}"

    # --- cutting -------------------------------------------------------------------------
    def cut(self, hand, tip):
        hand.depth = 0.0
        if tip is None:
            hand.previous_tip = None
            return
        surface = self.tissue.height(*tip[:2])
        on_line = self.procedure is not None and self.procedure.cut_allowed(tip)
        if on_line:
            surface = self.tissue.any_height(*tip[:2])
        if surface is None or tip[2] > surface:
            if self.tissue.on_vessel(*tip[:2]):
                vessel_top = self.tissue.vessel_top(*tip[:2])
                if tip[2] <= vessel_top:
                    self.message = "Blade on a vessel! Lift (Q), move away"
            hand.previous_tip = None
            return
        if self.needs_steadying() and not self.steadied(hand):
            self.message = f"Steady the tissue first: {OTHER[hand.side]} hand N then F"
            hand.previous_tip = None
            return
        hand.depth = min(MAX_DEPTH, surface - tip[2])
        normal = self.tissue.any_normal(*tip[:2]) if on_line else self.tissue.normal(*tip[:2])
        point = np.array([*tip[:2], surface]) + normal*.0007
        if hand.previous_tip is not None:
            distance = np.linalg.norm(point - hand.previous_tip)
            if .0012 <= distance <= .025:
                self.marks.append(self.wound_segment(hand.previous_tip, point, hand.depth, normal))
                if on_line:
                    self.procedure.on_cut(self, point)
                # Deeper cuts ooze: drops appear along the wound and slowly spread.
                if hand.depth > .0012 and np.random.random() < .18 + 40*hand.depth:
                    along = (point - hand.previous_tip) / distance
                    beside = np.cross(along, normal)*np.random.uniform(-.0015, .0015)
                    self.bleed(point + beside, np.random.uniform(.0014, .0018) + .35*hand.depth)
            elif distance < .0012:
                return
        hand.previous_tip = point

    def wound_segment(self, start, end, depth, normal):
        """Flat ellipses lying on the tissue surface for one cut segment: everted edges
        around a wet dark gap that widens with depth."""
        radius = .0006 + .24*depth
        lip = surface_strip(start - normal*.0004, end - normal*.0004, radius*2.0, .0003, normal,
                            self.site.lip_material)
        gap = surface_strip(start, end, radius*.85, .0004, normal, "wound")
        return lip, gap

    def bleed(self, point, radius):
        xy = point[:2]
        surface, normal = self.tissue.point(xy, .0004), None
        if surface is None and hasattr(self.tissue, "any_point"):   # on a vessel
            surface, normal = self.tissue.any_point(xy, .0004), self.tissue.any_normal(*xy)
        if surface is None:
            return
        normal = self.tissue.normal(*xy) if normal is None else normal
        _, _, matrix, _ = surface_strip(surface, surface, 0, 0, normal, "blood")
        self.blood.append((surface, matrix, self.clock, radius))

    # --- drawing -------------------------------------------------------------------------
    def draw(self, scene, now=None):
        if now is not None:
            self.clock = now
        for index, pose in enumerate(self.rest):
            hand = self.holder(index)
            draw_instrument(scene, index, *(hand.tool_pose if hand is not None else pose))
        lift = min(.024, self.site.bite_guide*1.2)
        forceps = self.holder(2)
        if forceps is not None and forceps.grip is not None:
            displacement = np.clip(self.active_tip(forceps) - forceps.grip, -lift, lift)
            add_capsule(scene, forceps.grip, forceps.grip + displacement,
                         min(.003, self.site.bite_guide*.25), "lip")
        ellipse = mj.mjtGeom.mjGEOM_ELLIPSOID
        for size, pos, matrix, material in self.tissue.wound:
            add_geom(scene, ellipse, size, pos, matrix, material)
        if self.procedure is not None:
            self.procedure.draw(scene, self.clock)
        if self.guide is not None:
            self.guide.draw(scene, self.clock)
        for segment in self.marks:
            for size, pos, matrix, material in segment:
                add_geom(scene, ellipse, size, pos, matrix, material)
        for position, matrix, born, radius in self.blood:
            grow = min(1.0, max(self.clock - born, 0) / 2.5)**.5
            r = radius*(.35 + .65*grow)
            add_geom(scene, ellipse, [r, r*.9, r*.28], position, matrix, "blood")
        for entry, exit, trimmed in self.stitches:
            self.draw_thread(scene, entry, exit, True, trimmed)
        if self.entry is None:
            return
        if self.exit is None:
            holder = self.holder(1)
            if holder is not None and holder.tool_pose is not None:
                self.draw_thread(scene, self.entry, self.active_tip(holder), False, False)
            return
        self.draw_thread(scene, self.entry, self.exit, self.throws > 0, False, self.throws)
        if self.needle_grasp is not None and forceps is not None and forceps.tool_pose is not None:
            # The long end, from the exit up to the forceps holding the needle.
            tip = self.active_tip(forceps)
            draw_path(scene, [self.exit, (self.exit + tip)/2 + UP*.002, tip], .0008, "thread")
            add_geom(scene, mj.mjtGeom.mjGEOM_SPHERE, [.0012, 0, 0], tip, np.eye(3), "polished")

    def draw_thread(self, scene, entry, exit, tied, trimmed, throws=3):
        mid = (entry + exit)/2 + np.array([0., 0., .003 if tied else .013])
        draw_path(scene, [entry, (entry + mid)/2, mid, (mid + exit)/2, exit], .0008, "thread")
        if tied:
            knot = exit + np.array([0., 0., .001])
            grow = .6 + .4*min(throws, 3)/3
            add_geom(scene, mj.mjtGeom.mjGEOM_ELLIPSOID, [.0019*grow, .0014*grow, .0012*grow],
                      knot, np.eye(3), "thread")
            if throws >= 3:
                length = self.site.tail_length*(.3 if trimmed else 1)
                for spread in (.6, -.2):
                    add_capsule(scene, knot, knot + np.array([length, length*spread, .0004]),
                                 .0006, "thread")
