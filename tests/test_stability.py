"""Tests for the chassis yaw-rate stability controller."""

from __future__ import annotations

import math

import pytest

from abs_sim.control.stability import StabilityController


def test_desired_yaw_rate_linear_ackermann():
    # V * delta / L for K_us = 0
    r = StabilityController.desired_yaw_rate(speed=20.0, steer=0.1, wheelbase=2.7)
    assert r == pytest.approx(20.0 * 0.1 / 2.7, rel=1e-6)


def test_desired_yaw_rate_zero_at_standstill():
    r = StabilityController.desired_yaw_rate(speed=0.0, steer=0.2, wheelbase=2.7)
    assert r == 0.0


def test_stability_disabled_passes_cmds_unchanged():
    c = StabilityController(enabled=False)
    cmds_in = (0.5, 0.5, 0.5, 0.5)
    out, err, u = c.update(
        base_cmds=cmds_in, yaw_rate=0.0, speed=20.0, steer=0.5,
        wheelbase=2.7, dt=0.005,
    )
    assert out == cmds_in
    assert u == 0.0


def test_stability_intervenes_on_understeer_right_turn():
    # Driver steering right (+delta) -> r_des > 0. Actual r = 0 (no yaw) means
    # understeer; controller should increase right-side brake and decrease left.
    c = StabilityController(kp=1.0, ki=0.0, kd=0.0, dead_band=0.0)
    base = (0.3, 0.3, 0.3, 0.3)
    out, err, u = c.update(base_cmds=base, yaw_rate=0.0, speed=20.0,
                           steer=0.2, wheelbase=2.7, dt=0.005)
    assert err > 0.0  # yaw error positive
    assert u > 0.0
    assert out[1] > base[1]  # FR up
    assert out[3] > base[3]  # RR up
    assert out[0] < base[0]  # FL down
    assert out[2] < base[2]  # RL down


def test_stability_intervenes_on_oversteer_right_turn():
    # Desired r_des = 20*0.1/2.7 ~= 0.74 rad/s; if actual r = 2 we are way
    # oversteering: controller should bias toward LEFT wheels.
    c = StabilityController(kp=1.0, ki=0.0, kd=0.0, dead_band=0.0)
    base = (0.3, 0.3, 0.3, 0.3)
    out, err, u = c.update(base_cmds=base, yaw_rate=2.0, speed=20.0,
                           steer=0.1, wheelbase=2.7, dt=0.005)
    assert err < 0.0
    assert u < 0.0
    assert out[0] > base[0]  # FL up
    assert out[2] > base[2]  # RL up
    assert out[1] < base[1]  # FR down
    assert out[3] < base[3]  # RR down


def test_stability_dead_band_ignores_small_errors():
    c = StabilityController(dead_band=0.2)  # large dead band
    base = (0.3, 0.3, 0.3, 0.3)
    out, err, u = c.update(base_cmds=base, yaw_rate=0.01, speed=20.0,
                           steer=0.01, wheelbase=2.7, dt=0.005)
    assert out == base
    assert u == 0.0


def test_stability_override_magnitude_bounded():
    # Even under huge error, override should saturate at max_override.
    c = StabilityController(kp=100.0, ki=100.0, max_override=0.3, dead_band=0.0)
    base = (0.5, 0.5, 0.5, 0.5)
    for _ in range(50):
        out, _, u = c.update(base_cmds=base, yaw_rate=0.0, speed=20.0,
                             steer=1.0, wheelbase=2.7, dt=0.005)
    assert abs(u) <= 0.3 + 1e-9
    for v in out:
        assert 0.0 <= v <= 1.0
