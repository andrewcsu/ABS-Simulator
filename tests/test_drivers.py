"""Tests for driver policies."""

from __future__ import annotations

import math

import pytest

from abs_sim.drivers.policies import (
    CruisePursuitDriver,
    CurveBrakeDelayDriver,
    DriverCommand,
    DriverContext,
    curvature_limited_target_speed,
    first_corner_start_s,
    pure_pursuit_steer,
)
from abs_sim.track.presets import curve_braking_scenario, straight_road


def _ctx(x=0.0, y=0.0, psi=0.0, vx=20.0, vy=0.0, r=0.0, wb=2.7) -> DriverContext:
    return DriverContext(x=x, y=y, psi=psi, vx=vx, vy=vy, r=r,
                         speed=math.hypot(vx, vy), wheelbase=wb)


def test_pure_pursuit_points_straight_on_straight_track():
    t = straight_road(200.0)
    steer = pure_pursuit_steer(_ctx(), t, s=10.0, lookahead=15.0)
    assert abs(steer) < 1e-6


def test_pure_pursuit_bends_toward_centerline_when_offset():
    t = straight_road(200.0)
    # Car sitting 2m left of centerline, heading along +x.
    steer = pure_pursuit_steer(_ctx(x=10.0, y=2.0), t, s=10.0, lookahead=15.0)
    # To return to the centerline (y=0) it should steer to the RIGHT: negative.
    assert steer < 0.0


def test_curvature_limited_target_speed_reduces_before_curve():
    t = curve_braking_scenario()
    # Before the corner, full cruise; approaching the corner, speed drops.
    v_far = curvature_limited_target_speed(t, s=0.0, v_cruise=40.0, mu_assumed=0.9)
    v_near = curvature_limited_target_speed(t, s=340.0, v_cruise=40.0, mu_assumed=0.9)
    assert v_far == pytest.approx(40.0)
    assert v_near < 40.0


def test_first_corner_s_on_curve_braking_track():
    t = curve_braking_scenario()
    s_c = first_corner_start_s(t)
    assert s_c is not None
    assert 300.0 < s_c < 370.0


def test_cruise_driver_steers_zero_on_straight_at_centerline():
    t = straight_road(200.0)
    d = CruisePursuitDriver(v_cruise=25.0)
    cmd = d.update(_ctx(x=20.0, y=0.0, vx=25.0), t, dt=0.01)
    assert abs(cmd.steer) < 1e-5


def test_cruise_driver_brakes_when_over_target_speed():
    t = straight_road(500.0)
    d = CruisePursuitDriver(v_cruise=20.0)
    for _ in range(5):
        cmd = d.update(_ctx(x=50.0, y=0.0, vx=30.0), t, dt=0.01)
    assert cmd.brake_demand > 0.0
    assert cmd.throttle == 0.0


def test_cruise_driver_throttles_when_below_target_speed():
    t = straight_road(500.0)
    d = CruisePursuitDriver(v_cruise=30.0)
    for _ in range(5):
        cmd = d.update(_ctx(x=50.0, y=0.0, vx=10.0), t, dt=0.01)
    assert cmd.throttle > 0.0
    assert cmd.brake_demand == 0.0


def test_curve_brake_delay_varies_brake_start():
    t = curve_braking_scenario()
    early = CurveBrakeDelayDriver(v_cruise=40.0, v_corner=13.0, delay_s=-1.0)
    late = CurveBrakeDelayDriver(v_cruise=40.0, v_corner=13.0, delay_s=+1.0)
    early._ensure_plan(t)  # type: ignore[attr-defined]
    late._ensure_plan(t)   # type: ignore[attr-defined]
    assert early._ideal_brake_s is not None  # type: ignore[attr-defined]
    # Early brakes at (ideal - v*1), late at (ideal + v*1).
    s_early_start = early._ideal_brake_s + early.delay_s * early.v_cruise  # type: ignore[attr-defined]
    s_late_start = late._ideal_brake_s + late.delay_s * late.v_cruise  # type: ignore[attr-defined]
    assert s_early_start < s_late_start
    assert (s_late_start - s_early_start) == pytest.approx(2.0 * 40.0, rel=1e-6)
