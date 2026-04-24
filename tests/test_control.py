"""Tests for brake actuator and per-wheel ABS controller."""

from __future__ import annotations

import pytest

from abs_sim.control.brake_actuator import BrakeActuator
from abs_sim.control.pid import PID
from abs_sim.control.wheel_abs import ABSState, WheelABS
from abs_sim.physics.vehicle import Vehicle, VehicleInputs


# --------------------------------- PID ------------------------------------- #

def test_pid_proportional_only():
    pid = PID(kp=2.0, ki=0.0, kd=0.0)
    assert pid.update(0.5, 0.01) == pytest.approx(1.0)


def test_pid_integral_windup_clamp():
    pid = PID(kp=0.0, ki=1.0, kd=0.0, i_min=-0.2, i_max=0.2)
    for _ in range(1000):
        pid.update(1.0, 0.01)
    assert pid.integral == pytest.approx(0.2)


def test_pid_reset_clears_state():
    pid = PID(kp=1.0, ki=1.0, kd=0.0)
    pid.update(0.5, 0.01)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.prev_error == 0.0
    assert pid.has_prev is False


# ----------------------------- Brake actuator ------------------------------ #

def test_actuator_rises_toward_command():
    a = BrakeActuator(tau=0.03, rate_max=8.0)
    for _ in range(100):
        a.update(1.0, 0.001)
    assert a.pressure > 0.5
    assert a.pressure <= 1.0


def test_actuator_clamped_to_unit_interval():
    a = BrakeActuator(tau=0.03)
    a.update(5.0, 0.001)
    assert 0.0 <= a.pressure <= 1.0


def test_actuator_rate_limit_prevents_instant_steps():
    a = BrakeActuator(tau=0.001, rate_max=4.0)
    # Even with tiny tau, rate limit keeps dp <= rate_max * dt
    p0 = a.pressure
    a.update(1.0, 0.01)
    assert a.pressure - p0 <= 4.0 * 0.01 + 1e-9


# ---------------------------- WheelABS logic ------------------------------- #

def test_abs_disabled_passes_through_demand():
    abs_c = WheelABS()
    cmd, st = abs_c.update(driver_demand=0.8, kappa=-0.5, dt=0.005, enabled=False)
    assert cmd == pytest.approx(0.8)
    assert st == ABSState.APPLY


def test_abs_release_when_slip_exceeds_upper():
    abs_c = WheelABS(lambda_opt=0.15, delta=0.03)
    # Feed a high-slip sample during braking
    cmd, st = abs_c.update(driver_demand=1.0, kappa=-0.5, dt=0.005, enabled=True)
    assert st == ABSState.RELEASE
    assert cmd == pytest.approx(abs_c.release_pressure)


def test_actuator_dumps_pressure_within_abs_hold_window():
    # With default rate_max, an actuator commanded from 1.0 -> 0.0 must
    # fall below 0.05 within the ABS hold_duration (~30 ms). Otherwise
    # the ABS FSM would re-engage HOLD before the wheel ever unloads,
    # starving the wheel of the pressure relief that should let it spin
    # back up during RELEASE.
    a = BrakeActuator()
    # Saturate pressure first
    for _ in range(500):
        a.update(1.0, 0.001)
    assert a.pressure > 0.5
    hold_duration = 0.03
    dt = 0.0005
    steps = int(hold_duration / dt)
    for _ in range(steps):
        a.update(0.0, dt)
    assert a.pressure < 0.05, (
        f"Actuator failed to dump within {hold_duration*1000:.0f} ms "
        f"(pressure={a.pressure:.3f}); ABS cannot recover."
    )


def test_abs_recovers_and_reapplies():
    abs_c = WheelABS(lambda_opt=0.15, delta=0.03, hold_duration=0.02)
    # Lock phase
    abs_c.update(1.0, -0.5, 0.005)
    assert abs_c.state == ABSState.RELEASE
    # Slip recovers below lower threshold
    abs_c.update(1.0, -0.05, 0.005)
    assert abs_c.state == ABSState.HOLD
    # After HOLD duration elapses, back to APPLY
    for _ in range(5):
        abs_c.update(1.0, -0.05, 0.005)
    assert abs_c.state == ABSState.APPLY


# ---------- End-to-end: ABS vs no-ABS on low-mu braking ------------------- #

def _simulate_braking(
    mu: float,
    use_abs: bool,
    v0: float = 15.0,
    dt: float = 0.001,
    max_time: float = 10.0,
    driver_demand: float = 1.0,
) -> tuple[float, float]:
    """Return (stop_distance, final_speed) for a straight-line braking run."""
    v = Vehicle()
    v.set_pose(0.0, 0.0, 0.0)
    v.set_speed(v0)
    controllers = [WheelABS(lambda_opt=0.15, delta=0.03) for _ in range(4)]
    actuators = [BrakeActuator(tau=0.01, rate_max=20.0) for _ in range(4)]
    t = 0.0
    x0 = v.x
    controller_dt = 0.005
    controller_accum = 0.0
    cmds = [0.0] * 4
    while t < max_time and v.vx > 0.1:
        if controller_accum >= controller_dt:
            kin = v.wheel_kinematics()
            for i in range(4):
                cmd, _ = controllers[i].update(
                    driver_demand, kin[i].kappa, controller_dt, enabled=use_abs
                )
                cmds[i] = cmd
            controller_accum = 0.0
        ap = tuple(actuators[i].update(cmds[i], dt) for i in range(4))
        v.step(dt, VehicleInputs(brake_pressure=ap, mu=(mu,) * 4))
        t += dt
        controller_accum += dt
    return v.x - x0, v.vx


def test_abs_shortens_stopping_distance_on_low_mu():
    d_off, v_off = _simulate_braking(mu=0.3, use_abs=False)
    d_on, v_on = _simulate_braking(mu=0.3, use_abs=True)
    assert v_off < 1.0
    assert v_on < 1.0
    # ABS should be strictly better on low mu (post-peak friction drops).
    assert d_on < d_off
    # And the improvement should be meaningful (>=5% on snow).
    assert (d_off - d_on) / d_off >= 0.05


def test_abs_prevents_sustained_lockup_on_ice():
    v = Vehicle()
    v.set_pose(0.0, 0.0, 0.0)
    v.set_speed(20.0)
    controllers = [WheelABS() for _ in range(4)]
    actuators = [BrakeActuator() for _ in range(4)]
    dt = 0.001
    cmds = [0.0] * 4
    controller_dt = 0.005
    controller_accum = 0.0
    lockup_time = 0.0
    t = 0.0
    while t < 3.0 and v.vx > 0.5:
        if controller_accum >= controller_dt:
            kin = v.wheel_kinematics()
            for i in range(4):
                cmd, _ = controllers[i].update(1.0, kin[i].kappa, controller_dt, True)
                cmds[i] = cmd
            controller_accum = 0.0
        ap = tuple(actuators[i].update(cmds[i], dt) for i in range(4))
        v.step(dt, VehicleInputs(brake_pressure=ap, mu=(0.1,) * 4))
        kin = v.wheel_kinematics()
        if all(abs(k.kappa) > 0.9 for k in kin):
            lockup_time += dt
        t += dt
        controller_accum += dt
    # ABS should prevent ALL four wheels from sitting at >90% slip for long.
    assert lockup_time < 0.3
