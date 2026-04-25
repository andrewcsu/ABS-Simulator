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
        # Polyline (world coords) of centerline sections where the left and
        # right halves of the lane have DIFFERENT surfaces, used to paint a
        # contrasting boundary marker on top of the ribbon.
        self._split_centerline: List[List[Tuple[float, float]]] = []
        # Per-sample surface for drawing coloured strips
        self._surface_segments: List[Tuple[str, List[Tuple[float, float]]]] = \
            self._build_surface_strips()

    @staticmethod
    def _resolve_sides(p) -> Tuple[str, str]:
        left_surf = p.surface_left if p.surface_left is not None else p.surface
        right_surf = p.surface_right if p.surface_right is not None else p.surface
        return left_surf, right_surf

    def _build_surface_strips(self) -> List[Tuple[str, List[Tuple[float, float]]]]:
        """Group consecutive samples by (left_surface, right_surface) pairs.

        When both sides share the same surface a single full-width polygon is
        emitted (back-compat with uniform tracks and full-width patches).
        When they differ, two half-width polygons are emitted so the ribbon
        is painted ice on one half and e.g. dry on the other.
        """
        samples = self.track.samples()
        if not samples:
            return []
        strips: List[Tuple[str, List[Tuple[float, float]]]] = []
        self._split_centerline = []

        half = self.track.width / 2.0

        def edges(p) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
            # Returns (left_edge, center, right_edge) world points.
            nx, ny = -math.sin(p.heading), math.cos(p.heading)
            lx, ly = p.x + nx * half, p.y + ny * half
            cx, cy = p.x, p.y
            rx, ry = p.x - nx * half, p.y - ny * half
            return (lx, ly), (cx, cy), (rx, ry)

        def flush(group_samples: List, lsurf: str, rsurf: str) -> None:
            if not group_samples:
                return
            left_edge: List[Tuple[float, float]] = []
            center_pts: List[Tuple[float, float]] = []
            right_edge: List[Tuple[float, float]] = []
            for p in group_samples:
                le, cp, re_ = edges(p)
                left_edge.append(le)
                center_pts.append(cp)
                right_edge.append(re_)
            if lsurf == rsurf:
                strips.append((lsurf, left_edge + list(reversed(right_edge))))
            else:
                strips.append((lsurf, left_edge + list(reversed(center_pts))))
                strips.append((rsurf, center_pts + list(reversed(right_edge))))
                self._split_centerline.append(list(center_pts))

        current_sides = self._resolve_sides(samples[0])
        group: List = [samples[0]]
        for p in samples[1:]:
            sides = self._resolve_sides(p)
            if sides != current_sides:
                # Include p as the closing sample of the outgoing group AND
                # the opening sample of the next group so adjacent strips
                # share an edge (no visible seams).
                group.append(p)
                flush(group, *current_sides)
                group = [p]
                current_sides = sides
            else:
                group.append(p)
        flush(group, *current_sides)
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
            self._draw_split_boundary(surf, cam)
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

    def _draw_split_boundary(self, surf: pygame.Surface, cam: Camera) -> None:
        """Overlay a bright solid line on sections where the left and right
        half-lane surfaces differ, so the split-mu boundary is obvious even
        when the two colours are similar in brightness.
        """
        color = (255, 90, 90)
        for segment in self._split_centerline:
            if len(segment) < 2:
                continue
            pts = [cam.world_to_screen(x, y) for x, y in segment]
            pygame.draw.lines(surf, color, False, pts, 3)

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
        viewport = pygame.Rect(cam.viewport_x, cam.viewport_y, cam.w, cam.h)
        surf.set_clip(viewport)
        try:
            if tire_mark_buffer:
                # set_at() does NOT respect the surface clip rect and will
                # raise IndexError for pixels outside the main surface, so
                # filter the points against the viewport ourselves. We draw
                # a 2x2 rect instead of a single pixel for visibility.
                vp_left = viewport.left
                vp_top = viewport.top
                vp_right = viewport.right
                vp_bottom = viewport.bottom
                mark_color = (20, 20, 22)
                for x, y in tire_mark_buffer:
                    px, py = cam.world_to_screen(x, y)
                    if vp_left <= px < vp_right and vp_top <= py < vp_bottom:
                        surf.fill(mark_color, (px, py, 2, 2))

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
