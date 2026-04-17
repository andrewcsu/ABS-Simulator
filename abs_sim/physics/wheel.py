"""Wheel geometry, spin state, and slip computation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from abs_sim.physics.tire import DugoffTire


EPS_V = 0.5  # m/s, speed floor for slip denominator (avoids singularity at stop)


@dataclass
class WheelParams:
    """Per-wheel physical parameters."""

    x_offset: float                 # m, longitudinal offset from CoG (+forward)
    y_offset: float                 # m, lateral offset from CoG (+right)
    J: float = 1.2                  # kg*m^2, wheel+brake rotational inertia
    steerable: bool = False         # front wheels steer; rear typically don't
    driven: bool = False            # true if engine torque reaches this wheel
    brake_torque_max: float = 2000  # Nm at brake_pressure = 1.0
    tire: DugoffTire = field(default_factory=DugoffTire)


@dataclass
class WheelKinematics:
    """Computed per-wheel kinematics & forces for one integration sub-step."""

    vx_tire: float = 0.0      # wheel-frame longitudinal velocity (m/s)
    vy_tire: float = 0.0      # wheel-frame lateral velocity (m/s)
    kappa: float = 0.0        # longitudinal slip ratio
    alpha: float = 0.0        # slip angle (rad)
    Fz: float = 0.0           # normal load (N)
    Fx_tire: float = 0.0      # longitudinal tire force, tire frame (N)
    Fy_tire: float = 0.0      # lateral tire force, tire frame (N)
    mu: float = 0.9           # friction of road under this wheel


def wheel_velocity_in_tire_frame(
    vx_body: float,
    vy_body: float,
    yaw_rate: float,
    wheel_x: float,
    wheel_y: float,
    steer: float,
) -> tuple[float, float]:
    """Velocity of a wheel hub expressed in the wheel (tire) frame.

    Start from body-frame velocity at the wheel hub:
        v_body = v_cg + r x r_wheel
        vx_body_wheel = vx_body - r * wheel_y
        vy_body_wheel = vy_body + r * wheel_x
    Then rotate by -steer to express in wheel frame.
    """
    vx_b = vx_body - yaw_rate * wheel_y
    vy_b = vy_body + yaw_rate * wheel_x
    c = math.cos(steer)
    s = math.sin(steer)
    vx_t = c * vx_b + s * vy_b
    vy_t = -s * vx_b + c * vy_b
    return vx_t, vy_t


def compute_slip(
    omega: float, vx_tire: float, vy_tire: float, R: float
) -> tuple[float, float]:
    """Compute (kappa, alpha) from wheel spin + tire-frame velocity.

    kappa = (omega*R - vx) / max(|vx|, EPS_V)  (SAE long. slip ratio)
    alpha = atan2(-vy_tire, max(|vx_tire|, EPS_V))  (SAE slip angle)
    """
    denom = max(abs(vx_tire), EPS_V)
    kappa = (omega * R - vx_tire) / denom
    alpha = math.atan2(-vy_tire, denom)
    return kappa, alpha
