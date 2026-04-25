"""Track: ordered sequence of straight / arc segments with per-region surfaces.

Public API
----------
* Track.build(segments, ...) class method accepts compact segment specs
  (dicts with {type, length|radius+angle+direction, surface[, surface_left,
  surface_right]}) and computes centerline geometry.
* Track.sample(s) -> (x, y, heading, curvature, surface_name)
* Track.closest(x, y, s_hint=None) -> (s, lateral_offset)
* Track.total_length
* Track.surface_at(s, e=0.0) (honors optional surface_patches overrides and,
  when e is signed, per-side surfaces). The sign of e matches the value
  returned by Track.closest(): for the default wheel layout e<0 is the side
  of the FL/RL wheels (called "left") and e>0 is the FR/RR side ("right").
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

    Surface fields
    --------------
    * ``surface``       uniform fallback for the whole cross-section.
    * ``surface_left``  optional override for the left half-lane (e<0 in the
                        convention of :meth:`Track.closest`, which is the side
                        the FL/RL wheels sit on in the default vehicle layout).
    * ``surface_right`` optional override for the right half-lane (e>0).

    When both ``surface_left`` and ``surface_right`` are None the segment is
    uniform-mu and ``surface`` is used across the full width (back-compat).
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
    surface_left: Optional[str] = None
    surface_right: Optional[str] = None

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
    """Override the surface for a range of global arc length.

    Setting ``surface_left`` / ``surface_right`` produces a half-lane patch
    (ice on one side, dry on the other). When both are None the patch is
    full-width and overrides whichever side ``surface_at`` is queried for.
    """

    start_s: float
    end_s: float
    surface: str
    surface_left: Optional[str] = None
    surface_right: Optional[str] = None


@dataclass
class _CenterlineSample:
    s: float
    x: float
    y: float
    heading: float
    curvature: float
    surface: str
    seg_idx: int
    # Optional per-side overrides (None = use ``surface`` for that side).
    surface_left: Optional[str] = None
    surface_right: Optional[str] = None


@dataclass
class Track:
    """An ordered track made of segments plus optional surface patches."""

    name: str
    segments: List[Segment]
    width: float = 8.0
    surface_patches: List[SurfacePatch] = field(default_factory=list)
    samples_per_meter: float = 2.0
    # Optional preset-specific cruise speed (m/s). When set, the interactive
    # app will use this as the driver's v_cruise instead of AppOptions.v_cruise.
    # Useful for split-mu / low-friction presets where the global default
    # (30 m/s) would force the autonomous driver to brake too hard before
    # every corner and spin out on the asymmetric grip.
    recommended_v_cruise: Optional[float] = None
    _samples: List[_CenterlineSample] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._sample()

    @property
    def total_length(self) -> float:
        return sum(seg.length for seg in self.segments)

    @property
    def is_closed(self) -> bool:
        """True if the last segment's endpoint coincides (within ~1 m) with
        the first segment's start. Used by driver helpers to decide whether
        lookahead should wrap via `s % L` or clamp at `total_length`.
        """
        if not self.segments:
            return False
        first = self.segments[0]
        last = self.segments[-1]
        dx = last.end_x - first.start_x
        dy = last.end_y - first.start_y
        return (dx * dx + dy * dy) < 1.0

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
                        surface_left=seg.surface_left,
                        surface_right=seg.surface_right,
                    )
                )
            s_cum += seg.length
        # Tag surface overrides (patches win over segment fields).
        for patch in self.surface_patches:
            for sam in self._samples:
                if patch.start_s <= sam.s <= patch.end_s:
                    if patch.surface_left is None and patch.surface_right is None:
                        sam.surface = patch.surface
                        sam.surface_left = None
                        sam.surface_right = None
                    else:
                        if patch.surface_left is not None:
                            sam.surface_left = patch.surface_left
                        if patch.surface_right is not None:
                            sam.surface_right = patch.surface_right

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

    def surface_at(self, s: float, e: float = 0.0) -> str:
        """Return the surface name at (s, e).

        ``s`` is the global centerline arc length (wrapped by total_length).
        ``e`` is the signed lateral offset produced by :meth:`closest`. Note
        that because the vehicle body frame uses ``+y = right`` while the
        track's lateral normal uses ``+y = left`` (math convention), a
        default-layout FL/RL wheel maps to ``e < 0`` and FR/RR maps to
        ``e > 0``. Hence ``surface_left`` (the half the car's left wheels
        ride on) is selected for ``e < 0`` and ``surface_right`` for
        ``e > 0``. When ``e == 0`` or the segment/patch has no side-specific
        surface, the base ``surface`` is returned. Patches override segment
        surfaces.
        """
        L = self.total_length
        if L <= 0.0:
            return "dry"
        s_mod = s % L
        for patch in self.surface_patches:
            if patch.start_s <= s_mod <= patch.end_s:
                if e < 0.0 and patch.surface_left is not None:
                    return patch.surface_left
                if e > 0.0 and patch.surface_right is not None:
                    return patch.surface_right
                return patch.surface
        s_cum = 0.0
        for seg in self.segments:
            if s_mod <= s_cum + seg.length:
                if e < 0.0 and seg.surface_left is not None:
                    return seg.surface_left
                if e > 0.0 and seg.surface_right is not None:
                    return seg.surface_right
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

        Either spec may additionally specify ``surface_left`` and/or
        ``surface_right`` (surface-name strings) to make the two halves of the
        lane differ -- e.g. a split-mu road with ice on the left and dry
        asphalt on the right.
        """
        segments: List[Segment] = []
        cx, cy, ch = start_x, start_y, start_heading
        for spec in specs:
            t = spec["type"]
            surface = spec.get("surface", "dry")
            surface_left = spec.get("surface_left")
            surface_right = spec.get("surface_right")
            if t == "straight":
                L = float(spec["length"])
                ex = cx + L * math.cos(ch)
                ey = cy + L * math.sin(ch)
                segments.append(
                    Segment(
                        type="straight", length=L, surface=surface,
                        start_x=cx, start_y=cy, start_heading=ch,
                        end_x=ex, end_y=ey, end_heading=ch,
                        surface_left=surface_left,
                        surface_right=surface_right,
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
                        surface_left=surface_left,
                        surface_right=surface_right,
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

    def with_recommended_cruise(self, v_cruise: float) -> "Track":
        """Return self after setting the recommended cruise speed (chainable)."""
        self.recommended_v_cruise = float(v_cruise)
        return self
