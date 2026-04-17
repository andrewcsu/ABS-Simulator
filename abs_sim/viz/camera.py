"""Camera: world-to-screen transform, with follow / fit modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Camera:
    """Translates between world meters (x right / y up) and screen pixels.

    The screen uses the usual top-left origin with y pointing down, so we flip
    the y axis when transforming.
    """

    cx: float = 0.0        # world x at the center of the viewport
    cy: float = 0.0        # world y at the center of the viewport
    scale: float = 4.0     # pixels per meter
    w: int = 800           # viewport width in pixels
    h: int = 600           # viewport height in pixels
    viewport_x: int = 0    # viewport top-left x on the main surface
    viewport_y: int = 0    # viewport top-left y on the main surface

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = int(self.viewport_x + self.w / 2 + (x - self.cx) * self.scale)
        sy = int(self.viewport_y + self.h / 2 - (y - self.cy) * self.scale)
        return sx, sy

    def fit(self, points: List[Tuple[float, float]], padding: float = 30.0) -> None:
        """Set scale and center to show all `points` with pixel `padding`."""
        if not points:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = max_x - min_x
        dy = max_y - min_y
        self.cx = 0.5 * (min_x + max_x)
        self.cy = 0.5 * (min_y + max_y)
        if dx <= 0 or dy <= 0:
            return
        sx = (self.w - 2 * padding) / dx
        sy = (self.h - 2 * padding) / dy
        self.scale = max(min(sx, sy), 0.1)

    def follow(self, x: float, y: float, blend: float = 0.15) -> None:
        """Smooth-follow a point in world coordinates."""
        self.cx += (x - self.cx) * blend
        self.cy += (y - self.cy) * blend
