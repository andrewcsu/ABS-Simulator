"""Tests for wheel, chassis, and vehicle dynamics."""

from __future__ import annotations

import math

import pytest

from abs_sim.physics.chassis import ChassisParams, G, load_transfer, static_loads
from abs_sim.physics.vehicle import Vehicle, VehicleInputs, FL, FR, RL, RR
from abs_sim.physics.wheel import (
    compute_slip,
    wheel_velocity_in_tire_frame,
)


# ------------------------- Chassis / load transfer -------------------------- #

def test_static_loads_sum_to_mg():
    p = ChassisParams()
    Fz = static_loads(p)
    assert sum(Fz) == pytest.approx(p.mass * G, rel=1e-6)


def test_load_transfer_zero_accel_equals_static():
    p = ChassisParams()
    Fz = load_transfer(p, ax_body=0.0, ay_body=0.0)
    assert Fz == pytest.approx(static_loads(p), rel=1e-6)


def test_load_transfer_under_braking_front_gains():
    p = ChassisParams()
    FzFL_s, FzFR_s, FzRL_s, FzRR_s = static_loads(p)
    FzFL, FzFR, FzRL, FzRR = load_transfer(p, ax_body=-5.0, ay_body=0.0)
    assert FzFL > FzFL_s
    assert FzFR > FzFR_s
    assert FzRL < FzRL_s
    assert FzRR < FzRR_s
    # Total weight preserved
    assert FzFL + FzFR + FzRL + FzRR == pytest.approx(p.mass * G, rel=1e-6)


def test_load_transfer_lateral_right_turn_to_left_wheels():
    # ay > 0 in body frame means right turn; load transfers to the LEFT (outside).
    p = ChassisParams()
    FL_s, FR_s, RL_s, RR_s = static_loads(p)
    FLw, FRw, RLw, RRw = load_transfer(p, ax_body=0.0, ay_body=+5.0)
    assert FLw > FL_s
    assert RLw > RL_s
    assert FRw < FR_s
    assert RRw < RR_s


# ----------------------------- Wheel helpers ------------------------------- #

def test_wheel_velocity_no_yaw_no_steer():
    vx_t, vy_t = wheel_velocity_in_tire_frame(
        vx_body=20.0, vy_body=0.0, yaw_rate=0.0,
        wheel_x=1.2, wheel_y=-0.75, steer=0.0,
    )
    assert vx_t == pytest.approx(20.0)
    assert vy_t == pytest.approx(0.0)


def test_wheel_velocity_yaw_contribution():
    # Pure yaw rate, zero body translation: wheel velocity comes from r x r_wheel
    vx_t, vy_t = wheel_velocity_in_tire_frame(
        vx_body=0.0, vy_body=0.0, yaw_rate=1.0,
        wheel_x=2.0, wheel_y=0.0, steer=0.0,
    )
    # For a wheel 2m forward of CoG, yaw rate 1 rad/s: v = r * x = +2 in body y.
    assert vx_t == pytest.approx(0.0, abs=1e-9)
    assert vy_t == pytest.approx(2.0)


def test_compute_slip_freerolling_is_zero():
    # Free rolling: omega*R == vx, so kappa == 0.
    R = 0.32
    vx = 20.0
    omega = vx / R
    kappa, alpha = compute_slip(omega, vx, 0.0, R)
    assert kappa == pytest.approx(0.0, abs=1e-9)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_compute_slip_locked_wheel_gives_minus_one():
    kappa, _ = compute_slip(omega=0.0, vx_tire=20.0, vy_tire=0.0, R=0.32)
    assert kappa == pytest.approx(-1.0)


# ------------------------------ Vehicle ------------------------------------ #

def test_vehicle_initialization_consistent():
    v = Vehicle()
    v.set_pose(0.0, 0.0, 0.0)
    v.set_speed(20.0)
    assert v.vx == pytest.approx(20.0)
    # Wheel spin matches free-rolling
    R = v.wheels[0].tire.R
    for w in v.wheel_speeds:
        assert w == pytest.approx(20.0 / R, rel=1e-6)


def test_vehicle_freerolling_decays_only_with_drag():
    # Free-rolling on a dry surface with no drive/brake: only aerodynamic drag
    # should slow the vehicle. It should decay but remain close to initial for
    # a short simulation.
    v = Vehicle()
    v.set_pose(0.0, 0.0, 0.0)
    v.set_speed(20.0)
    dt = 0.001
    for _ in range(100):
        v.step(dt, VehicleInputs(mu=(0.9,) * 4))
    assert v.vx < 20.0
    assert v.vx > 19.5  # should not lose more than 0.5 m/s in 0.1 s
    assert abs(v.vy) < 0.1
    assert abs(v.r) < 0.1


def test_vehicle_straight_line_braking_decelerates():
    # Decelerate from 20 m/s with moderate brake on dry. Peak physical decel on
    # dry is ~mu*g = 8.8 m/s^2, so in 1.0 s we should lose at least 4 m/s with
    # modest brake pressure.
    v = Vehicle()
    v.set_pose(0.0, 0.0, 0.0)
    v.set_speed(20.0)
    dt = 0.001
    inputs = VehicleInputs(
        steer=0.0, drive_torque=0.0,
        brake_pressure=(0.5, 0.5, 0.5, 0.5),
        mu=(0.9,) * 4,
    )
    for _ in range(1000):
        v.step(dt, inputs)
    assert v.vx < 16.0
    assert v.vx < 20.0  # clearly decelerated
    assert abs(v.vy) < 1.0


def test_vehicle_straight_line_braking_load_transfers():
    # After a sustained braking step, front loads should exceed rear.
    v = Vehicle()
    v.set_pose(0.0, 0.0, 0.0)
    v.set_speed(20.0)
    dt = 0.001
    inputs = VehicleInputs(
        brake_pressure=(0.4, 0.4, 0.4, 0.4),
        mu=(0.9,) * 4,
    )
    for _ in range(200):
        v.step(dt, inputs)
    kin = v.wheel_kinematics()
    assert kin[FL].Fz > kin[RL].Fz
    assert kin[FR].Fz > kin[RR].Fz


def test_vehicle_locks_on_ice_without_abs():
    # With high brake on ice and no ABS controller, wheels should saturate
    # slip towards locked; we check that kappa dips below -0.5 at least once.
    v = Vehicle()
    v.set_pose(0.0, 0.0, 0.0)
    v.set_speed(20.0)
    dt = 0.001
    inputs = VehicleInputs(
        brake_pressure=(1.0,) * 4,
        mu=(0.1,) * 4,
    )
    min_kappa = 0.0
    for _ in range(500):
        v.step(dt, inputs)
        for k in v.wheel_kinematics():
            min_kappa = min(min_kappa, k.kappa)
    assert min_kappa < -0.5
