"""Key codes and continuous key movement.

Besides GLFW key codes, the simulator uses synthetic codes (0x10000 and up) for panel buttons and
glove state. They travel through the same path as keys, so recordings replay exactly.
"""

import sys


# --- Station keys (recorded with each attempt) ---
SWITCH_HAND = ord("Z")    # keyboard control moves to the other hand (it also follows the steps)
RESET_HANDS = 0x10000     # from the panel's Reset hands button (a code no keyboard key sends)
BOTH_HANDS = ord("Y")     # toggles moving both hands together (also the Both hands button)
SELECT_HAND = {"left": 0x10002, "right": 0x10003}   # double-click on a hand in the viewer
RESET_TOOLS = 0x10004     # the room's Reset instruments button: every tool back in its tray
GLOVES_READY = 0x10005    # hands-only modes: the right glove is live with its direction locked
GLOVES_WAIT = 0x10006     # ... or no longer (the steps pause); both sent by the display


# --- Moving a hand ---
# GLFW key code -> (axis, metres per tap). Holding a key moves at MOVE_SPEED * step / s.
MOVES = {263: (0, -.006), 262: (0, .006), 264: (1, -.006), 265: (1, .006),
         ord("Q"): (2, .003), ord("E"): (2, -.003)}
MOVE_SPEED = 12      # held keys travel 12 taps per second (7 cm/s across, 3.6 cm/s up/down)
HOLD_DELAY = .18     # a quick tap stays a single precise step
MAC_KEYCODES = {263: 123, 262: 124, 264: 125, 265: 126, ord("Q"): 12, ord("E"): 14}


# --- Viewer keys (handled by the display loop, not recorded) ---
SWITCH_SITE = ord("P")
TOGGLE_GUIDE = ord("K")
START_OVER = ord("R")
VOICE_KEY, REVIEW_KEY, ASSISTANT_KEY = ord("V"), ord("B"), ord("I")
ZOOM_IN = {ord("="), 334}    # = (or +) and keypad +
ZOOM_OUT = {ord("-"), 333}   # - and keypad -
RESET_VIEW = ord("0")


def _mac_key_state():
    """Return a function telling whether a macOS key is physically down, or None."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes
        lib = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        query = lib.CGEventSourceKeyState
        query.argtypes, query.restype = [ctypes.c_int32, ctypes.c_uint16], ctypes.c_bool
    except (OSError, AttributeError):
        return None
    return lambda code: code in MAC_KEYCODES and query(0, MAC_KEYCODES[code])


class HeldKeys:
    """Turn the viewer's press-only key events into continuous movement.

    MuJoCo's passive viewer never reports key releases, so a key counts as held
    while macOS says it is down, or while the OS keeps sending repeat presses.
    Movement only starts from a press inside the viewer window.
    """

    def __init__(self):
        self.is_down = _mac_key_state()
        self.keys = {}  # code -> [pressed_at, last_event_at]

    def press(self, code, now):
        if code in MOVES:
            if code in self.keys:
                self.keys[code][1] = now
            else:
                self.keys[code] = [now, now]

    def moves(self, now, dt):
        for code, (pressed_at, last_event) in list(self.keys.items()):
            down = (self.is_down is not None and self.is_down(code)) or now - last_event < .12
            if not down and now - pressed_at > .05:
                del self.keys[code]
            elif now - pressed_at >= HOLD_DELAY:
                yield code, MOVES[code][1] * MOVE_SPEED * dt
