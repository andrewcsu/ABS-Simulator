"""Built-in track presets."""

from __future__ import annotations

import math
from typing import Dict, List

from abs_sim.track.track import SurfacePatch, Track


def straight_road(length: float = 400.0, surface: str = "dry") -> Track:
    """Plain long straight, ideal for ABS straight-line braking tests."""
    return Track.build(
        name=f"straight_{int(length)}m_{surface}",
        specs=[{"type": "straight", "length": length, "surface": surface}],
        width=10.0,
    )


def oval(straight: float = 200.0, radius: float = 50.0, surface: str = "dry") -> Track:
    """Closed oval with two straights and two 180-degree left arcs."""
    specs = [
        {"type": "straight", "length": straight, "surface": surface},
        {"type": "arc", "radius": radius, "angle": math.pi, "direction": "left", "surface": surface},
        {"type": "straight", "length": straight, "surface": surface},
        {"type": "arc", "radius": radius, "angle": math.pi, "direction": "left", "surface": surface},
    ]
    return Track.build(name="oval", specs=specs, width=10.0)


def figure_8(leg: float = 120.0, radius: float = 35.0, surface: str = "dry") -> Track:
    """Figure-8: two loops of opposite handedness."""
    specs = [
        {"type": "straight", "length": leg, "surface": surface},
        {"type": "arc", "radius": radius, "angle": math.pi, "direction": "left", "surface": surface},
        {"type": "straight", "length": leg, "surface": surface},
        {"type": "arc", "radius": radius, "angle": 2 * math.pi, "direction": "right", "surface": surface},
        {"type": "straight", "length": leg, "surface": surface},
        {"type": "arc", "radius": radius, "angle": math.pi, "direction": "left", "surface": surface},
    ]
    return Track.build(name="figure_8", specs=specs, width=10.0)


def f1_like(surface: str = "dry") -> Track:
    """Mixed sequence of straights and arcs that resembles a road course.

    Not a real circuit geometry, but has slow and fast corners of different
    radii so the 3-driver curve-braking demo has something interesting to do.
    """
    specs = [
        {"type": "straight", "length": 300.0, "surface": surface},
        {"type": "arc", "radius": 20.0, "angle": math.pi / 2, "direction": "right",
         "surface": surface},  # tight
        {"type": "straight", "length": 120.0, "surface": surface},
        {"type": "arc", "radius": 60.0, "angle": math.pi, "direction": "left",
         "surface": surface},  # medium
        {"type": "straight", "length": 150.0, "surface": surface},
        {"type": "arc", "radius": 35.0, "angle": math.pi * 0.75, "direction": "right",
         "surface": surface},
        {"type": "straight", "length": 100.0, "surface": surface},
        {"type": "arc", "radius": 25.0, "angle": math.pi * 0.75, "direction": "left",
         "surface": surface},
        {"type": "straight", "length": 80.0, "surface": surface},
        {"type": "arc", "radius": 50.0, "angle": math.pi * 0.55, "direction": "left",
         "surface": surface},
        {"type": "straight", "length": 60.0, "surface": surface},
        {"type": "arc", "radius": 15.0, "angle": math.pi * 0.85, "direction": "right",
         "surface": surface},
        {"type": "straight", "length": 80.0, "surface": surface},
    ]
    return Track.build(name="f1_like", specs=specs, width=12.0)


def curve_braking_scenario(
    straight: float = 350.0, radius: float = 35.0, surface: str = "dry",
) -> Track:
    """Long straight then a tight left corner. Used for 3-driver brake timing."""
    specs = [
        {"type": "straight", "length": straight, "surface": surface},
        {"type": "arc", "radius": radius, "angle": math.pi * 0.75, "direction": "left",
         "surface": surface},
        {"type": "straight", "length": 150.0, "surface": surface},
    ]
    return Track.build(name="curve_braking", specs=specs, width=12.0)


def split_mu_curves(
    surface_left: str = "ice", surface_right: str = "dry",
) -> Track:
    """Mixed straight + curve course with different surfaces per lane half.

    Classic split-mu scenario for ABS/ESC demos: the left wheels ride on one
    surface (ice by default) while the right wheels ride on another (dry
    asphalt by default). Alternating left- and right-hand curves let the
    viewer see ABS cycling asymmetrically (left wheels saturate on ice while
    right wheels still have grip) and the ESC yaw controller fight the
    mu-induced yaw moment whenever the driver brakes or accelerates.

    The geometry is intentionally gentle: large radii, < 90 deg arcs, and
    long straights between corners. Cornering on a split-mu road is hard --
    on a right-hander the OUTSIDE (left) tires that need to bear the
    cornering load are on ice, so even a tuned ESC can only do so much. With
    these dimensions the car has enough margin to stay in the 12 m wide lane
    while still demonstrating the asymmetric-mu behaviour clearly.
    """
    kw = {"surface_left": surface_left, "surface_right": surface_right}
    specs = [
        # Long entry straight: lets the user tap the brake here to feel
        # the asymmetric brake-yaw moment on split-mu.
        {"type": "straight", "length": 140.0, "surface": surface_right, **kw},
        # Gentle right opener (R=120, 30 deg). Outside-of-turn = LEFT = ice.
        {"type": "arc", "radius": 120.0, "angle": math.pi / 6,
         "direction": "right", "surface": surface_right, **kw},
        {"type": "straight", "length": 100.0, "surface": surface_right, **kw},
        # Sharper left (R=70, 60 deg). Outside = RIGHT = dry, easier line.
        {"type": "arc", "radius": 70.0, "angle": math.pi / 3,
         "direction": "left", "surface": surface_right, **kw},
        {"type": "straight", "length": 100.0, "surface": surface_right, **kw},
        # Tight right (R=50, 75 deg). Outside = LEFT = ice. The sharpest
        # corner on the course; the autopilot really has to crawl through
        # it (~5 m/s) because the outside tires are on a frozen pond.
        {"type": "arc", "radius": 50.0, "angle": math.radians(75.0),
         "direction": "right", "surface": surface_right, **kw},
        {"type": "straight", "length": 100.0, "surface": surface_right, **kw},
        # Medium left back the other way (R=80, 45 deg).
        {"type": "arc", "radius": 80.0, "angle": math.pi / 4,
         "direction": "left", "surface": surface_right, **kw},
        {"type": "straight", "length": 140.0, "surface": surface_right, **kw},
    ]
    track = Track.build(name="split_mu_curves", specs=specs, width=12.0)
    # Cruise at 13 m/s. The autonomous driver needs to be at corner-target
    # speed (~9 m/s on ice-outside corners) BEFORE turning -- braking and
    # turning at the same time on split-mu produces an unbalanced yaw
    # moment that ESC can't fully suppress. Cruising at 13 means the
    # required brake delta is small (13 -> 9 m/s) and the car can settle
    # well before each corner. Pro instructors give the same advice on
    # real split-mu skid pads: treat it like a low-grip course top to
    # bottom, not a dry road that has one slow corner. The slider can
    # still raise this if the user wants to push it harder.
    return track.with_recommended_cruise(13.0)


def random_surface_straight(length: float = 400.0, seed: int = 0) -> Track:
    """Long straight with randomly-placed low-mu patches for ABS demos."""
    import random

    rng = random.Random(seed)
    patches: List[SurfacePatch] = []
    s = 60.0
    while s < length - 40.0:
        surf = rng.choice(["ice", "wet", "snow", "sand"])
        patch_len = rng.uniform(10.0, 40.0)
        patches.append(SurfacePatch(start_s=s, end_s=s + patch_len, surface=surf))
        s += patch_len + rng.uniform(25.0, 75.0)
    return Track.build(
        name="random_surface_straight",
        specs=[{"type": "straight", "length": length, "surface": "dry"}],
        width=10.0,
        surface_patches=patches,
    )


PRESETS: Dict[str, callable] = {
    "straight": straight_road,
    "oval": oval,
    "figure_8": figure_8,
    "f1_like": f1_like,
    "curve_braking": curve_braking_scenario,
    "random_surface_straight": random_surface_straight,
    "split_mu_curves": split_mu_curves,
}


def get_preset(name: str) -> Track:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Known: {list(PRESETS)}")
    return PRESETS[name]()
