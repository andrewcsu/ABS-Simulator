"""Chassis geometry, parameters, and quasi-static load transfer."""

from __future__ import annotations

from dataclasses import dataclass


G = 9.81  # m/s^2


@dataclass
class ChassisParams:
    """Rigid-body chassis parameters.

    Coordinates: body frame has +x forward, +y right, +z down (SAE-like).
    CoG is at the origin of the body frame. Wheel offsets are measured from CoG.
    """

    mass: float = 1500.0        # kg, total vehicle mass including wheels
    Izz: float = 2500.0         # kg*m^2, yaw inertia
    wheelbase: float = 2.7      # L = a + b (m)
    a: float = 1.2              # m, CoG to front axle
    b: float = 1.5              # m, CoG to rear axle  (a+b should equal wheelbase)
    track: float = 1.55         # T, left-right track width (m)
    h_cg: float = 0.55          # CoG height above ground (m)
    Cd_A: float = 0.75          # drag area (m^2), for F_drag = 0.5 * rho * Cd_A * v^2
    rho_air: float = 1.225      # kg/m^3


def static_loads(p: ChassisParams) -> tuple[float, float, float, float]:
    """Return (Fz_FL, Fz_FR, Fz_RL, Fz_RR) with vehicle stationary on flat ground."""
    Fz_front = p.mass * G * p.b / p.wheelbase / 2.0
    Fz_rear = p.mass * G * p.a / p.wheelbase / 2.0
    return Fz_front, Fz_front, Fz_rear, Fz_rear


def load_transfer(
    p: ChassisParams,
    ax_body: float,
    ay_body: float,
) -> tuple[float, float, float, float]:
    """Quasi-static per-wheel normal loads given body-frame linear accelerations.

    Sign conventions
    ----------------
    ax > 0 : vehicle accelerating forward  -> load shifts to REAR.
    ax < 0 : vehicle braking / decelerating -> load shifts to FRONT.
    ay > 0 : accelerating to the right (right turn) -> load shifts to the LEFT.

    Returns (Fz_FL, Fz_FR, Fz_RL, Fz_RR), all >= 0 (clamped).
    """
    m = p.mass
    L = p.wheelbase
    T = p.track
    h = p.h_cg

    Fz_front_axle = m * G * p.b / L - m * ax_body * h / L
    Fz_rear_axle = m * G * p.a / L + m * ax_body * h / L
    dFy = m * ay_body * h / T  # total shift to the left (positive when ay>0)

    Fz_FL = Fz_front_axle / 2.0 + dFy / 2.0
    Fz_FR = Fz_front_axle / 2.0 - dFy / 2.0
    Fz_RL = Fz_rear_axle / 2.0 + dFy / 2.0
    Fz_RR = Fz_rear_axle / 2.0 - dFy / 2.0
    return (
        max(Fz_FL, 0.0),
        max(Fz_FR, 0.0),
        max(Fz_RL, 0.0),
        max(Fz_RR, 0.0),
    )
