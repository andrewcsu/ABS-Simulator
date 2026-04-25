"""Launch the interactive pygame ABS simulator.

Usage:
    python -m scripts.run_interactive
    python -m scripts.run_interactive --track oval
    python -m scripts.run_interactive --track f1_like --cruise 35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abs_sim.viz.pygame_app import AppOptions, PygameApp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", default="f1_like",
                   help="track preset name (oval, figure_8, f1_like, curve_braking, "
                        "straight, random_surface_straight, split_mu_curves)")
    p.add_argument("--cruise", type=float, default=30.0,
                   help="driver target cruise speed in m/s")
    p.add_argument("--surface", default=None,
                   help="force a surface (dry/wet/snow/ice/sand)")
    p.add_argument("--no-follow", action="store_true", help="disable camera follow")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    opts = AppOptions(
        track_preset=args.track,
        v_cruise=args.cruise,
        follow_camera=not args.no_follow,
        initial_surface_override=args.surface,
    )
    PygameApp(opts).run()


if __name__ == "__main__":
    main()
