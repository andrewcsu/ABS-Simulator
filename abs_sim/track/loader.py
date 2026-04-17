"""YAML track loader.

Track schema
------------
name: oval
width: 10.0
start: {x: 0.0, y: 0.0, heading: 0.0}  # optional, defaults to origin
segments:
  - {type: straight, length: 100.0, surface: dry}
  - {type: arc, radius: 30.0, angle: 3.14159, direction: left, surface: dry}
surface_patches:                        # optional
  - {start_s: 50.0, end_s: 80.0, surface: ice}
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from abs_sim.track.track import SurfacePatch, Track


def load_track(path: Union[str, Path]) -> Track:
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return load_track_from_dict(cfg, fallback_name=path.stem)


def load_track_from_dict(cfg: dict, fallback_name: str = "track") -> Track:
    name = cfg.get("name", fallback_name)
    width = float(cfg.get("width", 8.0))
    start = cfg.get("start", {})
    sx = float(start.get("x", 0.0))
    sy = float(start.get("y", 0.0))
    sh = float(start.get("heading", 0.0))
    specs = cfg.get("segments", [])
    patches_cfg = cfg.get("surface_patches", []) or []
    patches = [
        SurfacePatch(
            start_s=float(p["start_s"]),
            end_s=float(p["end_s"]),
            surface=p["surface"],
        )
        for p in patches_cfg
    ]
    return Track.build(
        name=name,
        specs=specs,
        start_x=sx,
        start_y=sy,
        start_heading=sh,
        width=width,
        surface_patches=patches,
    )
