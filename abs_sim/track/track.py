"""Track: ordered sequence of straight / arc segments with per-region surfaces.

Public API
----------
* Track.build(segments, ...) class method accepts compact segment specs
  (dicts with {type, length|radius+angle+direction, surface}) and computes
  centerline geometry.
* Track.sample(s) -> (x, y, heading, curvature, surface_name)
* Track.closest(x, y, s_hint=None) -> (s, lateral_offset)
* Track.total_length
* Track.surface_at(s) (honors optional surface_patches overrides).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import math


def _wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass
class Segment:
    """A single track segment (straight or arc).

    Straight:   type='straight', length, start_{x,y,heading}
    Arc:        type='arc',      length = |R|*|angle|,
                signed_radius (+ = left turn / CCW),
                center_{x,y}, start_angle_from_center (rad).
    """

    type: str
    length: float
    surface: str
    start_x: float
    start_y: float
    start_heading: float
    end_x: float
    end_y: float
    end_heading: float
    signed_radius: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0

    def curvature(self) -> float:
        if self.type == "straight" or self.signed_radius == 0.0:
            return 0.0
        return 1.0 / self.signed_radius

    def position(self, s: float) -> Tuple[float, float, float]:
        """Return (x, y, heading) at local arc length s in [0, length]."""
        if s < 0.0:
            s = 0.0
        elif s > self.length:
            s = self.length

        if self.type == "straight":
            x = self.start_x + s * math.cos(self.start_heading)
            y = self.start_y + s * math.sin(self.start_heading)
            return x, y, self.start_heading

        R = self.signed_radius
        # Angle subtended from start, signed with radius sign.
        dtheta = s / R
        heading = self.start_heading + dtheta
        # Parametrize position relative to center: center points at -90 deg
        # to the initial heading on the "left" side (for R > 0).
        # position(s) = center + R * ( sin(heading), -cos(heading) )
        x = self.center_x + R * math.sin(heading)
        y = self.center_y - R * math.cos(heading)
        return x, y, _wrap_angle(heading)


@dataclass
class SurfacePatch:
    """Override the surface for a range of global arc length."""
    start_s: float
    end_s: float
    surface: str


@dataclass
class _CenterlineSample:
    s: float
    x: float
    y: float
    heading: float
    curvature: float
    surface: str
    seg_idx: int


@dataclass
class Track:
    """An ordered track made of segments plus optional surface patches."""

    name: str
    segments: List[Segment]
    width: float = 8.0
    surface_patches: List[SurfacePatch] = field(default_factory=list)
    samples_per_meter: float = 2.0
    _samples: List[_CenterlineSample] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._sample()

    @property
    def total_length(self) -> float:
        return sum(seg.length for seg in self.segments)

    def _sample(self) -> None:
        self._samples = []
        s_cum = 0.0
        for idx, seg in enumerate(self.segments):
            n = max(2, int(seg.length * self.samples_per_meter))
            for i in range(n):
                if i == 0 and idx > 0:
                    continue  # avoid duplicate at segment boundary
                local_s = seg.length * i / max(n - 1, 1)
                x, y, h = seg.position(local_s)
                g_s = s_cum + local_s
                self._samples.append(
                    _CenterlineSample(
                        s=g_s, x=x, y=y, heading=h,
                        curvature=seg.curvature(),
                        surface=seg.surface, seg_idx=idx,
                    )
                )
            s_cum += seg.length
        # Tag surface overrides
        for patch in self.surface_patches:
            for sam in self._samples:
                if patch.start_s <= sam.s <= patch.end_s:
                    sam.surface = patch.surface

    # ------------------------------------------------------------------ #
    # Sampling / lookup
    # ------------------------------------------------------------------ #
    def sample(self, s: float) -> Tuple[float, float, float, float, str]:
        """Return (x, y, heading, curvature, surface) at global arc length s."""
        L = self.total_length
        if L <= 0.0:
            return 0.0, 0.0, 0.0, 0.0, "dry"
        s = s % L
        s_cum = 0.0
        for seg in self.segments:
            if s <= s_cum + seg.length:
                local_s = s - s_cum
                x, y, h = seg.position(local_s)
                return x, y, h, seg.curvature(), self.surface_at(s)
            s_cum += seg.length
        last = self.segments[-1]
        return last.end_x, last.end_y, last.end_heading, 0.0, last.surface

    def surface_at(self, s: float) -> str:
        L = self.total_length
        if L <= 0.0:
            return "dry"
        s_mod = s % L
        for patch in self.surface_patches:
            if patch.start_s <= s_mod <= patch.end_s:
                return patch.surface
        s_cum = 0.0
        for seg in self.segments:
            if s_mod <= s_cum + seg.length:
                return seg.surface
            s_cum += seg.length
        return self.segments[-1].surface

    def closest(
        self, x: float, y: float, s_hint: Optional[float] = None, window: float = 20.0,
    ) -> Tuple[float, float]:
        """Return (s, lateral_offset) of the closest centerline point.

        lateral_offset is positive to the LEFT of the centerline direction (so
        a car drifting right has negative offset). Brute-force over samples,
        limited to a +/- window around s_hint when provided.
        """
        if not self._samples:
            return 0.0, 0.0
        if s_hint is None:
            candidates = self._samples
        else:
            L = self.total_length
            lo = (s_hint - window) % L
            hi = (s_hint + window) % L
            if lo <= hi:
                candidates = [p for p in self._samples if lo <= p.s <= hi]
            else:
                candidates = [p for p in self._samples if p.s >= lo or p.s <= hi]
            if not candidates:
                candidates = self._samples

        best = candidates[0]
        best_d2 = (best.x - x) ** 2 + (best.y - y) ** 2
        for p in candidates[1:]:
            d2 = (p.x - x) ** 2 + (p.y - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = p

        dx = x - best.x
        dy = y - best.y
        nx = -math.sin(best.heading)
        ny = math.cos(best.heading)
        e = dx * nx + dy * ny
        return best.s, e

    def centerline_points(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self._samples]

    def centerline_with_surface(self) -> List[Tuple[float, float, str]]:
        return [(p.x, p.y, p.surface) for p in self._samples]

    def samples(self) -> List[_CenterlineSample]:
        return list(self._samples)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        name: str,
        specs: Sequence[dict],
        start_x: float = 0.0,
        start_y: float = 0.0,
        start_heading: float = 0.0,
        width: float = 8.0,
        surface_patches: Optional[Sequence[SurfacePatch]] = None,
    ) -> "Track":
        """Build a Track from a list of segment spec dicts.

        A straight spec:     {"type": "straight", "length": L, "surface": "dry"}
        An arc spec:         {"type": "arc", "radius": R, "angle": A,
                              "direction": "left" | "right",
                              "surface": "wet"}
        Arc length = R * A; direction sets the sign of the curvature.
        """
        segments: List[Segment] = []
        cx, cy, ch = start_x, start_y, start_heading
        for spec in specs:
            t = spec["type"]
            surface = spec.get("surface", "dry")
            if t == "straight":
                L = float(spec["length"])
                ex = cx + L * math.cos(ch)
                ey = cy + L * math.sin(ch)
                segments.append(
                    Segment(
                        type="straight", length=L, surface=surface,
                        start_x=cx, start_y=cy, start_heading=ch,
                        end_x=ex, end_y=ey, end_heading=ch,
                    )
                )
                cx, cy = ex, ey
            elif t == "arc":
                R_abs = float(spec["radius"])
                angle = float(spec["angle"])
                direction = spec.get("direction", "left").lower()
                signed_R = R_abs if direction == "left" else -R_abs
                dtheta = angle if direction == "left" else -angle
                # center of arc: perpendicular to heading, on the turn side
                center_x = cx - signed_R * math.sin(ch)
                center_y = cy + signed_R * math.cos(ch)
                L = R_abs * angle
                end_heading = _wrap_angle(ch + dtheta)
                ex = center_x + signed_R * math.sin(end_heading)
                ey = center_y - signed_R * math.cos(end_heading)
                segments.append(
                    Segment(
                        type="arc", length=L, surface=surface,
                        start_x=cx, start_y=cy, start_heading=ch,
                        end_x=ex, end_y=ey, end_heading=end_heading,
                        signed_radius=signed_R,
                        center_x=center_x, center_y=center_y,
                    )
                )
                cx, cy, ch = ex, ey, end_heading
            else:
                raise ValueError(f"Unknown segment type '{t}'")
        return cls(
            name=name,
            segments=segments,
            width=width,
            surface_patches=list(surface_patches or []),
        )
