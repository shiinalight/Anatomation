"""Practice incisions and sutures with both hands, on skin or on a heart, in a glove-controlled
MuJoCo operating room.

Entry point: parses the options, starts the glove stream and opens the viewer (app.py).
"""

import argparse
import threading

import or_scene
from app import run
from glove_input import GloveStream
from or_scene import SITES
from procedures import DEFAULT, PROCEDURES, from_site


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true",
                        help="Practice with keyboard hands and no gloves")
    parser.add_argument("--procedure", choices=list(PROCEDURES), default=DEFAULT,
                        help="Procedure to start with (also chosen in the Procedures window)")
    parser.add_argument("--site", choices=sorted(SITES),
                        help="Older option: skin or heart (suture) or bypass")
    parser.add_argument("--heart", help="Another heart model (STL or OBJ; x anterior, y left, z superior)")
    args = parser.parse_args()
    or_scene.HEART_FILE = args.heart
    procedure = from_site(args.site) if args.site else PROCEDURES[args.procedure]
    print(f"Procedure: {procedure.label}", flush=True)
    gloves = GloveStream()
    if not args.preview:
        print("Connect both gloves (rokoko-sdk and rkk-hand-solver --emit-openxr). Tilt and roll\n"
              "follow your hands directly; to set the direction, rest each hand flat with the\n"
              "fingers toward the screen for half a second. Press H to set it again.\n"
              "A glove that is not streaming is replaced by a keyboard hand.", flush=True)
        threading.Thread(target=gloves.run, daemon=True).start()
    run(gloves, args.preview, procedure)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
