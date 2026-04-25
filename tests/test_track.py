"""Tests for track geometry and loader."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from abs_sim.track.loader import load_track, load_track_from_dict
from abs_sim.track.presets import (
    PRESETS,
    curve_braking_scenario,
    figure_8,
    oval,
    split_mu_curves,
    straight_road,
)
from abs_sim.track.track import SurfacePatch, Track


def test_straight_segment_positions_and_total_length():
    t = straight_road(length=100.0)
    assert t.total_length == pytest.approx(100.0)
    x, y, h, k, s = t.sample(0.0)
    assert (x, y, h) == (0.0, 0.0, 0.0)
    assert k == 0.0
    assert s == "dry"
    x, y, h, k, _ = t.sample(50.0)
    assert x == pytest.approx(50.0)
    assert y == pytest.approx(0.0)


def test_arc_segment_endpoint_left_turn_90deg():
    # Left 90deg arc of radius 10 starting at origin pointing +x:
    # should end at (10, 10) pointing +y.
    t = Track.build(
        name="test_arc",
        specs=[{"type": "arc", "radius": 10.0, "angle": math.pi / 2,
                "direction": "left", "surface": "dry"}],
    )
    assert t.total_length == pytest.approx(10.0 * math.pi / 2)
    # Sample just before the wrap-around at total_length.
    x, y, h, k, _ = t.sample(t.total_length - 1e-6)
    assert x == pytest.approx(10.0, abs=1e-4)
    assert y == pytest.approx(10.0, abs=1e-4)
    assert h == pytest.approx(math.pi / 2, abs=1e-4)
    assert k == pytest.approx(1.0 / 10.0)  # positive = left


def test_arc_segment_right_turn_curvature_sign():
    t = Track.build(
        name="test_arc_right",
        specs=[{"type": "arc", "radius": 10.0, "angle": math.pi / 2,
                "direction": "right", "surface": "dry"}],
    )
    _, _, _, k, _ = t.sample(0.5 * t.total_length)
    assert k < 0.0


def test_oval_closes_approximately():
    t = oval(straight=100.0, radius=30.0)
    x0, y0, _, _, _ = t.sample(0.0)
    xe, ye, _, _, _ = t.sample(t.total_length - 0.01)
    # End of last arc should be near the start
    assert math.hypot(xe - x0, ye - y0) < 2.0


def test_closest_returns_zero_lateral_offset_on_centerline():
    t = straight_road(100.0)
    s, e = t.closest(50.0, 0.0)
    assert s == pytest.approx(50.0, abs=1.0)
    assert abs(e) < 1e-6


def test_closest_lateral_offset_sign_is_left_positive():
    # Heading +x, a point at y=+2 should be LEFT of centerline -> positive offset.
    t = straight_road(100.0)
    s, e = t.closest(50.0, 2.0)
    assert s == pytest.approx(50.0, abs=1.0)
    assert e > 0.0
    s, e = t.closest(50.0, -2.0)
    assert e < 0.0


def test_surface_patch_overrides_segment_surface():
    patches = [SurfacePatch(start_s=25.0, end_s=75.0, surface="ice")]
    t = Track.build(
        name="icy_middle",
        specs=[{"type": "straight", "length": 100.0, "surface": "dry"}],
        surface_patches=patches,
    )
    assert t.surface_at(10.0) == "dry"
    assert t.surface_at(50.0) == "ice"
    assert t.surface_at(80.0) == "dry"


def test_segment_roundtrips_left_right_surfaces_through_build():
    t = Track.build(
        name="split",
        specs=[{
            "type": "straight", "length": 50.0,
            "surface": "dry",
            "surface_left": "ice", "surface_right": "dry",
        }],
    )
    seg = t.segments[0]
    assert seg.surface == "dry"
    assert seg.surface_left == "ice"
    assert seg.surface_right == "dry"


def test_surface_at_with_e_sign_selects_half_lane():
    t = Track.build(
        name="split",
        specs=[{
            "type": "straight", "length": 100.0,
            "surface": "dry",
            "surface_left": "ice", "surface_right": "dry",
        }],
    )
    # e < 0 is the FL/RL side -> surface_left
    assert t.surface_at(50.0, e=-1.0) == "ice"
    # e > 0 is the FR/RR side -> surface_right
    assert t.surface_at(50.0, e=+1.0) == "dry"
    # e == 0 falls back to base surface (centerline ambiguous).
    assert t.surface_at(50.0, e=0.0) == "dry"
    # Back-compat: zero-arg call still works.
    assert t.surface_at(50.0) == "dry"


def test_surface_patch_half_lane_override():
    # Base dry + a left-only ice patch in the middle.
    patches = [SurfacePatch(
        start_s=20.0, end_s=80.0,
        surface="dry", surface_left="ice",
    )]
    t = Track.build(
        name="left_ice_patch",
        specs=[{"type": "straight", "length": 100.0, "surface": "dry"}],
        surface_patches=patches,
    )
    assert t.surface_at(50.0, e=-1.0) == "ice"
    # Right side inside the patch stays dry because surface_right is None.
    assert t.surface_at(50.0, e=+1.0) == "dry"
    # Outside the patch both sides are dry.
    assert t.surface_at(10.0, e=-1.0) == "dry"


def test_split_mu_curves_preset_has_asymmetric_surfaces():
    t = split_mu_curves()
    assert t.total_length > 0.0
    # Every sample should have left=ice, right=dry on the split preset.
    samples = t.samples()
    assert any(
        (p.surface_left == "ice" and p.surface_right == "dry") for p in samples
    )
    # At least one arc segment so there's a curve on this track.
    assert any(seg.type == "arc" for seg in t.segments)
    # surface_at reflects the split too.
    s_mid = 0.5 * t.total_length
    assert t.surface_at(s_mid, e=-1.0) == "ice"
    assert t.surface_at(s_mid, e=+1.0) == "dry"


def test_presets_loadable_and_have_positive_length():
    for name in PRESETS:
        t = PRESETS[name]()
        assert isinstance(t, Track)
        assert t.total_length > 0.0


def test_load_track_from_yaml_file():
    path = Path(__file__).parent.parent / "abs_sim" / "config" / "tracks" / "oval.yaml"
    t = load_track(path)
    assert t.name == "oval_wet_patch"
    assert t.total_length > 100.0
    # The wet patch should override the segment at s=165.
    assert t.surface_at(165.0) == "wet"


def test_load_track_from_dict_minimum():
    cfg = {
        "name": "mini",
        "segments": [
            {"type": "straight", "length": 50.0, "surface": "dry"},
        ],
    }
    t = load_track_from_dict(cfg)
    assert t.name == "mini"
    assert t.total_length == pytest.approx(50.0)


def test_curve_braking_preset_has_a_real_corner():
    t = curve_braking_scenario()
    curves = [s.curvature for s in t.samples() if s.curvature != 0.0]
    assert len(curves) > 0
    assert max(abs(c) for c in curves) > 0.01  # tighter than a 100m radius
