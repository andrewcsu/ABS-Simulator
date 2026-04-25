"""End-to-end simulation integration tests."""

from __future__ import annotations

import math

import pytest

from abs_sim.drivers.policies import CruisePursuitDriver, CurveBrakeDelayDriver
from abs_sim.sim.events import force_brake, set_surface_override
from abs_sim.sim.simulation import Car, Simulation
from abs_sim.sim.telemetry import TelemetryLogger
from abs_sim.physics.tire import SURFACES
from abs_sim.track.presets import (
    curve_braking_scenario,
    f1_like,
    split_mu_curves,
    straight_road,
)
from abs_sim.track.track import Track


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


def test_split_mu_track_yields_different_per_wheel_mu():
    """Split-mu roads: left wheels ice, right wheels dry asphalt.

    On the split_mu_curves preset the per-wheel mu lookup must see different
    surfaces for FL/RL vs FR/RR. We run a single controller tick with the
    car centered on the first straight and verify that
        - the left-wheel mus match SURFACES["ice"].mu
        - the right-wheel mus match SURFACES["dry"].mu
        - the per-wheel surface names are populated asymmetrically.
    """
    track = split_mu_curves()
    car = Car.make_default(name="ego", driver=CruisePursuitDriver(v_cruise=10.0))
    # Place the car well inside the first straight, centered, facing +x.
    car.vehicle.set_pose(60.0, 0.0, 0.0)
    car.vehicle.set_speed(10.0)
    sim = Simulation(track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True))
    # Step past at least one controller tick so _mu_for_car populates last_mu.
    for _ in range(20):
        sim.step()

    mu_ice = SURFACES["ice"].mu
    mu_dry = SURFACES["dry"].mu
    assert car.last_mu[0] == pytest.approx(mu_ice)  # FL
    assert car.last_mu[2] == pytest.approx(mu_ice)  # RL
    assert car.last_mu[1] == pytest.approx(mu_dry)  # FR
    assert car.last_mu[3] == pytest.approx(mu_dry)  # RR
    assert car.last_mu[0] < car.last_mu[1]
    assert car.last_mu[2] < car.last_mu[3]
    assert car.last_surface_per_wheel[0] == "ice"
    assert car.last_surface_per_wheel[1] == "dry"
    assert car.last_surface_per_wheel[2] == "ice"
    assert car.last_surface_per_wheel[3] == "dry"


def test_split_mu_curves_keeps_car_on_track_through_every_corner():
    """Regression: on the split_mu_curves preset the autopilot has to make
    it through all three corners without sliding off the lane.

    Previously the driver assumed mu=0.8 across the lane and entered
    corners hot enough that the asymmetric ice/dry brake yaw moment
    overwhelmed ESC, sending the car off the outside of every curve. The
    fix is twofold: the planner now probes per-half-lane mu (taking the
    minimum for lateral grip and a worst-side-biased blend for the
    longitudinal brake plan) and adds a ``settle_buffer`` so braking is
    finished before the corner mouth, and the preset advertises a
    sensible ``recommended_v_cruise`` (slow enough that the brake delta
    never demands aggressive pedal). This test locks both behaviours in.
    """
    track = split_mu_curves()
    assert track.recommended_v_cruise is not None
    car = Car.make_default(
        name="ego",
        driver=CruisePursuitDriver(v_cruise=track.recommended_v_cruise),
    )
    car.vehicle.set_pose(0.0, 0.0, 0.0)
    car.vehicle.set_speed(track.recommended_v_cruise)
    sim = Simulation(track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True))

    half_w = track.width / 2.0
    last_s = 0.0
    max_off = 0.0
    off_count = 0
    # The track is ~840 m and the autopilot crawls through the sharp corners
    # at ~5 m/s on the ice-side outsides, so it needs 100+ s of sim time
    # to reach the end. Give it 130 s of headroom.
    while sim.time < 130.0:
        sim.step()
        s, e = track.closest(car.vehicle.x, car.vehicle.y, s_hint=last_s)
        last_s = s
        if abs(e) > max_off:
            max_off = abs(e)
        if abs(e) > half_w:
            off_count += 1
        # Bail out early if we've already cleared every corner.
        if s > track.total_length - 5.0:
            break

    assert max_off <= half_w + 0.5, (
        f"car drifted to |e|={max_off:.2f} (lane half {half_w:.2f}); "
        "still sliding off split-mu corners"
    )
    assert off_count == 0, f"car was outside the lane for {off_count} steps"
    assert last_s > track.total_length - 50.0, (
        f"car stalled at s={last_s:.1f} of {track.total_length:.1f} -- it "
        "didn't make it through every corner"
    )


def test_split_mu_emergency_brake_keeps_car_on_track():
    """Slamming the brake at full pedal on a split-mu road must NOT
    fishtail the car off the lane.

    Real-world physics: the dry-side wheels can decelerate hard while the
    ice-side wheels saturate the moment ABS lets them, producing a strong
    yaw moment toward the high-mu side. ESC has to bias the brake away
    from the dry side hard enough to keep the car straight. With the old
    ``max_override = 0.4`` the controller couldn't take enough pedal off
    the dry side and the car slid off; we now require ``max_override``
    large enough to handle full-pedal split-mu input.
    """
    track = split_mu_curves()
    v0 = 13.0
    car = Car.make_default(name="ego", driver=CruisePursuitDriver(v_cruise=v0))
    car.vehicle.set_pose(20.0, 0.0, 0.0)
    car.vehicle.set_speed(v0)
    sim = Simulation(track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True))
    # Cruise for half a second, then slam the pedal for 2 s.
    sim.schedule(0.5, force_brake(1.0, duration=2.0), desc="ebrake")

    half_w = track.width / 2.0
    last_s = 20.0
    max_off = 0.0
    off_count = 0
    max_r = 0.0
    while sim.time < 5.0:
        sim.step()
        s, e = track.closest(car.vehicle.x, car.vehicle.y, s_hint=last_s)
        last_s = s
        max_off = max(max_off, abs(e))
        max_r = max(max_r, abs(car.vehicle.r))
        if abs(e) > half_w:
            off_count += 1

    assert off_count == 0, (
        f"car was off the lane for {off_count} steps under emergency brake "
        f"on split-mu (max |e|={max_off:.2f}, half-width {half_w:.2f})"
    )
    assert max_off <= half_w, (
        f"max |e|={max_off:.2f} reached the lane edge ({half_w:.2f})"
    )
    # Yaw should peak well below a full-on spin (which would be > 2 rad/s).
    assert max_r < 1.5, (
        f"max |yaw rate|={max_r:.2f} rad/s -- ESC isn't catching the "
        "split-mu brake yaw moment fast enough"
    )


def test_split_mu_surface_override_still_forces_uniform_mu():
    """The legacy set_surface_override event must still clamp all four wheels
    to one surface, regardless of the underlying split-mu track."""
    track = split_mu_curves()
    car = Car.make_default(name="ego", driver=CruisePursuitDriver(v_cruise=10.0))
    car.surface_override = "snow"
    car.vehicle.set_pose(60.0, 0.0, 0.0)
    car.vehicle.set_speed(10.0)
    sim = Simulation(track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True))
    for _ in range(20):
        sim.step()

    mu_snow = SURFACES["snow"].mu
    for m in car.last_mu:
        assert m == pytest.approx(mu_snow)
    assert car.last_surface == "snow"
    assert set(car.last_surface_per_wheel) == {"snow"}


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
