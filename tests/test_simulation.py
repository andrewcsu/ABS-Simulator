"""End-to-end simulation integration tests."""

from __future__ import annotations

import math

import pytest

from abs_sim.drivers.policies import CruisePursuitDriver, CurveBrakeDelayDriver
from abs_sim.sim.events import force_brake, set_surface_override
from abs_sim.sim.simulation import Car, Simulation
from abs_sim.sim.telemetry import TelemetryLogger
from abs_sim.track.presets import curve_braking_scenario, f1_like, straight_road


def _run_straight_braking(use_abs: bool, mu_name: str = "snow", v0: float = 20.0):
    track = straight_road(length=400.0, surface=mu_name)
    car = Car.make_default(
        name="test", driver=CruisePursuitDriver(v_cruise=v0), color=(200, 200, 200),
    )
    car.vehicle.set_speed(v0)
    car.abs_enabled = use_abs
    sim = Simulation(track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True))
    # After 0.5 s of cruise, start hammering the brake
    sim.schedule(0.5, force_brake(1.0, duration=20.0), desc="slam")
    while sim.time < 15.0 and sim.cars[0].vehicle.vx > 0.5:
        sim.step()
    return sim


def test_simulation_runs_and_decelerates():
    sim = _run_straight_braking(use_abs=True, mu_name="snow", v0=20.0)
    assert sim.cars[0].vehicle.vx < 1.0


def test_simulation_abs_helps_on_snow():
    off = _run_straight_braking(use_abs=False, mu_name="snow", v0=20.0)
    on = _run_straight_braking(use_abs=True, mu_name="snow", v0=20.0)
    d_off = off.cars[0].vehicle.x
    d_on = on.cars[0].vehicle.x
    assert d_on < d_off


def test_surface_override_event_changes_mu():
    track = straight_road(length=200.0, surface="dry")
    car = Car.make_default(name="t", driver=CruisePursuitDriver(v_cruise=15.0))
    car.vehicle.set_speed(15.0)
    sim = Simulation(track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True))
    sim.schedule(0.1, set_surface_override("ice", duration=0.5))
    sim.schedule(0.3, force_brake(1.0, duration=5.0))
    saw_ice = False
    while sim.time < 2.0 and sim.cars[0].vehicle.vx > 0.2:
        sim.step()
        if sim.cars[0].last_surface == "ice":
            saw_ice = True
    assert saw_ice


def test_telemetry_captures_expected_fields():
    track = straight_road(length=100.0, surface="dry")
    car = Car.make_default(name="t", driver=CruisePursuitDriver(v_cruise=15.0))
    car.vehicle.set_speed(15.0)
    sim = Simulation(track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True))
    for _ in range(500):
        sim.step()
    rows = sim.telemetry.rows()
    assert len(rows) > 50
    required = {
        "t", "x", "y", "psi", "vx", "vy", "speed",
        "kappa_FL", "kappa_FR", "kappa_RL", "kappa_RR",
        "brake_actual_FL", "brake_actual_FR", "brake_actual_RL", "brake_actual_RR",
        "abs_state_FL", "abs_state_FR", "abs_state_RL", "abs_state_RR",
        "surface", "mu_FL", "ax", "ay",
    }
    missing = required - set(rows[-1].keys())
    assert not missing


def test_f1_like_car_clears_first_corner_without_spinning():
    """Regression test for interactive spin-out.

    On the `f1_like` preset, the CruisePursuitDriver brakes hard into a 20 m
    right-hander at ~12 m/s then, at corner exit, sees curvature drop to zero
    and wants to accelerate back to v_cruise. Without a driver throttle slew
    rate and a per-wheel drive-torque cap (simple TCS), the inside rear
    unloads, PI floors the throttle, and 2 kNm of drive torque per wheel
    blows past the ~400 Nm tire limit causing runaway wheelspin and yaw
    divergence. This test runs the same scenario headlessly and asserts the
    yaw rate stays bounded and slip stays well below lock.
    """
    track = f1_like()
    car = Car.make_default(
        name="ego", driver=CruisePursuitDriver(v_cruise=30.0),
    )
    car.vehicle.set_pose(0.0, 0.0, 0.0)
    car.vehicle.set_speed(30.0)
    sim = Simulation(track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True))

    # 18 s covers the straight -> 20m right -> short straight; past this the
    # car is well clear of the first corner and back on throttle.
    max_kappa = 0.0
    max_yaw_rate = 0.0
    while sim.time < 18.0:
        sim.step()
        kin = car.vehicle.wheel_kinematics()
        for k in kin:
            if abs(k.kappa) > max_kappa:
                max_kappa = abs(k.kappa)
        if abs(car.vehicle.r) > max_yaw_rate:
            max_yaw_rate = abs(car.vehicle.r)

    # Peak |kappa| during a healthy cycle is the ABS target (~lambda_opt=0.15)
    # plus ABS cycling overshoot. A value under ~0.6 means no wheel is near
    # lock; under ~1.0 means definitely not a fully-locked or fully-spun
    # wheel. Pre-fix the inside rear reached kappa > 2000 from wheelspin.
    assert max_kappa < 1.0, f"wheel-slip runaway: max |kappa|={max_kappa:.2f}"
    # 3.5 rad/s ~ 200 deg/s is an aggressive but controllable yaw; >>5 rad/s
    # indicates the body is spinning about its CG.
    assert max_yaw_rate < 3.5, f"yaw divergence: max |r|={max_yaw_rate:.2f} rad/s"
    # And it should actually be past the first corner (exit is ~ (320, -20)).
    assert car.vehicle.x > 310.0
    # Heading should be somewhere near -90 deg (i.e. not still spinning).
    # Wrap to [-pi, pi] first.
    psi = math.atan2(math.sin(car.vehicle.psi), math.cos(car.vehicle.psi))
    assert abs(psi - (-math.pi / 2.0)) < math.radians(45.0), (
        f"car heading after first corner = {math.degrees(psi):.1f} deg; expected near -90"
    )


def test_curve_braking_three_drivers_each_run():
    """Integration smoke test: 3 delay variants all run to completion."""
    track = curve_braking_scenario()
    cars = []
    for name, delay, col in [
        ("early", -0.5, (100, 230, 100)),
        ("ontime", 0.0, (200, 200, 200)),
        ("late", 0.5, (230, 100, 100)),
    ]:
        car = Car.make_default(
            name=name,
            driver=CurveBrakeDelayDriver(v_cruise=38.0, v_corner=14.0, delay_s=delay),
            color=col,
        )
        car.vehicle.set_speed(38.0)
        cars.append(car)
    sim = Simulation(track=track, cars=cars, telemetry=TelemetryLogger(in_memory=True))
    for _ in range(4000):  # 4 s
        sim.step()
    # All three cars should have travelled forward significantly.
    for c in sim.cars:
        assert c.vehicle.x > 60.0
