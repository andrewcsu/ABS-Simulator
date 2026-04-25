"""Chassis yaw-rate stability controller (simplified ESC-style override).

Monitors the chassis yaw rate against a desired yaw rate derived from the
Ackermann bicycle model. When the yaw error grows large (because a wheel has
saturated, or mu has dropped under one side), it overrides per-wheel brake
pressure to apply a corrective yaw moment via differential braking.

Sign conventions
----------------
* Steering angle delta > 0 steers the front wheels to the right.
* Yaw rate r > 0 means the vehicle nose is rotating to the right.
* A positive corrective yaw moment (Mz) pushes the nose further to the right.
  Braking a right-side wheel contributes +Mz (left-side contributes -Mz),
  because the friction force is aft on a wheel at a positive y offset.

Override scheme
---------------
Given a PID output u (sign = desired corrective yaw moment):

* u > 0, need more right-yaw: bias brake pressure toward RIGHT wheels.
* u < 0, need less right-yaw (or more left-yaw): bias toward LEFT wheels.

The override is applied as a small additive delta on top of the per-wheel
ABS command, then clamped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from abs_sim.control.pid import PID


@dataclass
class StabilityController:
    """Chassis yaw-rate stability / ESC-like differential braking override."""

    kp: float = 0.5
    ki: float = 0.5
    kd: float = 0.02
    # max |brake delta| added per wheel. 0.4 was too timid for split-mu
    # emergency braking: with one side on ice and one on dry asphalt, full
    # pedal demand produces a big yaw moment (the dry side brakes much
    # harder), and the controller has to be able to back the high-mu side
    # off significantly to keep the car straight. 0.8 gives near-full
    # authority while still leaving headroom; subtle-error interventions
    # are unaffected because the integrator and gains keep u small there.
    max_override: float = 0.8
    dead_band: float = 0.05         # rad/s error below which we don't intervene
    understeer_gradient: float = 0.0  # K_us (s^2/m-ish units)
    enabled: bool = True

    _pid: PID = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._pid is None:
            self._pid = PID(self.kp, self.ki, self.kd, i_min=-2.0, i_max=2.0)

    def reset(self) -> None:
        self._pid.reset()

    @staticmethod
    def desired_yaw_rate(
        speed: float, steer: float, wheelbase: float, K_us: float = 0.0
    ) -> float:
        """Linear bicycle-model target yaw rate from steer angle and speed.

        r_des = V * delta / (L + K_us * V^2)
        """
        denom = wheelbase + K_us * speed * speed
        if denom <= 0.0:
            return 0.0
        return speed * steer / denom

    def update(
        self,
        base_cmds: Tuple[float, float, float, float],
        yaw_rate: float,
        speed: float,
        steer: float,
        wheelbase: float,
        dt: float,
    ) -> Tuple[Tuple[float, float, float, float], float, float]:
        """Apply stability override on top of per-wheel ABS commands.

        Returns (adjusted_cmds, yaw_error, pid_out). yaw_error is in rad/s and
        pid_out is the corrective moment signal (arbitrary units), exposed for
        telemetry / HUD.
        """
        r_des = self.desired_yaw_rate(speed, steer, wheelbase, self.understeer_gradient)
        e_r = r_des - yaw_rate

        if (not self.enabled) or abs(e_r) < self.dead_band or speed < 3.0:
            self._pid.reset()
            return tuple(base_cmds), float(e_r), 0.0

        u = self._pid.update(e_r, dt)
        if u > self.max_override:
            u = self.max_override
        elif u < -self.max_override:
            u = -self.max_override

        fl, fr, rl, rr = base_cmds
        fl = max(0.0, min(1.0, fl - u))
        rl = max(0.0, min(1.0, rl - u))
        fr = max(0.0, min(1.0, fr + u))
        rr = max(0.0, min(1.0, rr + u))
        return (fl, fr, rl, rr), float(e_r), float(u)
