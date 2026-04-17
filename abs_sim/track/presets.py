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
}


def get_preset(name: str) -> Track:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Known: {list(PRESETS)}")
    return PRESETS[name]()
