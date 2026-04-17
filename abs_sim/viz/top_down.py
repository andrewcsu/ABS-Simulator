"""Top-down track and car renderer."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pygame

from abs_sim.physics.tire import SURFACES
from abs_sim.sim.simulation import Car
from abs_sim.track.track import Segment, Track
from abs_sim.viz.camera import Camera


# --------------------------------------------------------------------------- #
# Track polygon caching
# --------------------------------------------------------------------------- #

def _segment_polygon(
    seg: Segment, width: float, steps_per_m: float = 1.0,
) -> List[Tuple[float, float]]:
    """Build the closed boundary polygon of a single segment."""
    n = max(2, int(seg.length * steps_per_m) + 1)
    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []
    half = width / 2.0
    for i in range(n):
        s = seg.length * i / (n - 1)
        x, y, h = seg.position(s)
        nx, ny = -math.sin(h), math.cos(h)  # left normal
        left.append((x + nx * half, y + ny * half))
        right.append((x - nx * half, y - ny * half))
    return left + list(reversed(right))


def _surface_color(surface_name: str) -> Tuple[int, int, int]:
    return SURFACES.get(surface_name, SURFACES["dry"]).color


class TrackRenderer:
    """Pre-computes per-segment polygons and draws the track + cars."""

    def __init__(self, track: Track) -> None:
        self.track = track
        self._seg_polygons: List[List[Tuple[float, float]]] = [
            _segment_polygon(seg, track.width) for seg in track.segments
        ]
        self._centerline_points: List[Tuple[float, float]] = [
            (p.x, p.y) for p in track.samples()
        ]
        # Per-sample surface for drawing coloured strips
        self._surface_segments: List[Tuple[str, List[Tuple[float, float]]]] = \
            self._build_surface_strips()

    def _build_surface_strips(self) -> List[Tuple[str, List[Tuple[float, float]]]]:
        """Group consecutive samples by current surface to color surface patches."""
        samples = self.track.samples()
        if not samples:
            return []
        strips: List[Tuple[str, List[Tuple[float, float]]]] = []
        current_surface = samples[0].surface
        left: List[Tuple[float, float]] = []
        right: List[Tuple[float, float]] = []
        half = self.track.width / 2.0
        for p in samples:
            nx, ny = -math.sin(p.heading), math.cos(p.heading)
            left.append((p.x + nx * half, p.y + ny * half))
            right.append((p.x - nx * half, p.y - ny * half))
            if p.surface != current_surface:
                strips.append((current_surface, left + list(reversed(right))))
                left = [left[-1]]
                right = [right[-1]]
                current_surface = p.surface
        strips.append((current_surface, left + list(reversed(right))))
        return strips

    def draw_background(self, surf: pygame.Surface, cam: Camera) -> None:
        pygame.draw.rect(
            surf, (30, 32, 38),
            pygame.Rect(cam.viewport_x, cam.viewport_y, cam.w, cam.h),
        )

    def draw_track(self, surf: pygame.Surface, cam: Camera) -> None:
        """Draw the track ribbon with per-surface colouring."""
        prev_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(cam.viewport_x, cam.viewport_y, cam.w, cam.h))
        try:
            for surface_name, poly in self._surface_segments:
                if len(poly) < 3:
                    continue
                pts = [cam.world_to_screen(x, y) for x, y in poly]
                pygame.draw.polygon(surf, _surface_color(surface_name), pts)
            self._draw_centerline(surf, cam)
            self._draw_edges(surf, cam)
        finally:
            surf.set_clip(prev_clip)

    def _draw_centerline(self, surf: pygame.Surface, cam: Camera) -> None:
        color = (230, 230, 160)
        if len(self._centerline_points) < 2:
            return
        # Dashed centerline: draw every 3rd pair of samples
        pts = self._centerline_points
        for i in range(0, len(pts) - 1, 4):
            j = min(i + 2, len(pts) - 1)
            a = cam.world_to_screen(*pts[i])
            b = cam.world_to_screen(*pts[j])
            pygame.draw.line(surf, color, a, b, 2)

    def _draw_edges(self, surf: pygame.Surface, cam: Camera) -> None:
        half = self.track.width / 2.0
        left: List[Tuple[int, int]] = []
        right: List[Tuple[int, int]] = []
        for p in self.track.samples():
            nx, ny = -math.sin(p.heading), math.cos(p.heading)
            lx, ly = p.x + nx * half, p.y + ny * half
            rx, ry = p.x - nx * half, p.y - ny * half
            left.append(cam.world_to_screen(lx, ly))
            right.append(cam.world_to_screen(rx, ry))
        if len(left) >= 2:
            pygame.draw.lines(surf, (250, 250, 250), False, left, 2)
        if len(right) >= 2:
            pygame.draw.lines(surf, (250, 250, 250), False, right, 2)

    # ---------------------------------------------------------------- #
    # Cars
    # ---------------------------------------------------------------- #
    def draw_car(
        self,
        surf: pygame.Surface,
        cam: Camera,
        car: Car,
        tire_mark_buffer: Optional[List[Tuple[float, float]]] = None,
        body_length: float = 4.4,
        body_width: float = 1.9,
    ) -> None:
        prev_clip = surf.get_clip()
        surf.set_clip(pygame.Rect(cam.viewport_x, cam.viewport_y, cam.w, cam.h))
        try:
            if tire_mark_buffer:
                for x, y in tire_mark_buffer:
                    px, py = cam.world_to_screen(x, y)
                    surf.set_at((px, py), (20, 20, 22))

            v = car.vehicle
            c, s = math.cos(v.psi), math.sin(v.psi)
            half_l = body_length / 2.0
            half_w = body_width / 2.0
            corners = [(half_l, -half_w), (half_l, half_w), (-half_l, half_w), (-half_l, -half_w)]
            body_pts: List[Tuple[int, int]] = []
            for cx_local, cy_local in corners:
                wx = v.x + c * cx_local - s * cy_local
                wy = v.y + s * cx_local + c * cy_local
                body_pts.append(cam.world_to_screen(wx, wy))
            pygame.draw.polygon(surf, car.color, body_pts)
            pygame.draw.polygon(surf, (10, 10, 10), body_pts, 2)

            # Brake lights (rear)
            brake_avg = sum(car.last_actuator_pressure) / 4.0
            if brake_avg > 0.05:
                for y_off in (-half_w * 0.7, half_w * 0.7):
                    lx = v.x + c * (-half_l) - s * y_off
                    ly = v.y + s * (-half_l) + c * y_off
                    px, py = cam.world_to_screen(lx, ly)
                    intensity = min(255, int(80 + 175 * brake_avg))
                    pygame.draw.circle(surf, (intensity, 20, 20), (px, py), 3)

            # Forward arrow indicating heading
            tip_w = (v.x + c * (half_l + 1.0), v.y + s * (half_l + 1.0))
            pygame.draw.line(
                surf, (250, 250, 250),
                cam.world_to_screen(v.x, v.y),
                cam.world_to_screen(*tip_w), 1,
            )

            # Car name above
            if car.name:
                font = _get_font(14)
                label = font.render(car.name, True, (235, 235, 235))
                lx_s, ly_s = cam.world_to_screen(v.x, v.y + 3.0)
                surf.blit(label, (lx_s - label.get_width() // 2, ly_s - label.get_height() // 2))
        finally:
            surf.set_clip(prev_clip)


# --------------------------------------------------------------------------- #
# Tire-mark buffer helper
# --------------------------------------------------------------------------- #

class TireMarkBuffer:
    """Keep a rolling buffer of world positions where tires left marks."""

    def __init__(self, max_points: int = 4000) -> None:
        self.points: List[Tuple[float, float]] = []
        self.max = max_points

    def add(self, x: float, y: float) -> None:
        self.points.append((x, y))
        if len(self.points) > self.max:
            self.points = self.points[-self.max:]

    def maybe_add_from_car(self, car: Car, slip_threshold: float = 0.25) -> None:
        kin = car.vehicle.wheel_kinematics()
        if any(abs(k.kappa) > slip_threshold for k in kin):
            v = car.vehicle
            self.add(v.x, v.y)


# --------------------------------------------------------------------------- #
# Tiny font cache
# --------------------------------------------------------------------------- #

_font_cache: Dict[int, pygame.font.Font] = {}


def _get_font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        pygame.font.init()
        _font_cache[size] = pygame.font.SysFont("dejavusansmono,monospace", size)
    return _font_cache[size]
