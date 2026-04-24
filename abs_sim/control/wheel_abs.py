"""Per-wheel ABS controller: FSM (APPLY / HOLD / RELEASE) + slip-target PID.

This is a classroom-scale version of a real ABS algorithm. A finite-state
machine produces the characteristic pump cycle (Bosch / NHTSA-style bang-bang
pressure modulation), and a PID on slip error is layered inside the APPLY
phase to softly converge slip to the optimal value (maximum of the mu-slip
curve, typically kappa ~= 0.15).

States
------
APPLY     Normal braking. Driver's demand is passed through, optionally reduced
          by the PID when slip is already at/above lambda_opt.
HOLD      After a release cycle, pressure is held constant for a short time to
          let slip settle before re-applying.
RELEASE   Slip has exceeded (lambda_opt + delta). Pressure is dumped to almost
          zero so the wheel can recover spin.

References
----------
* Mathworks, "Model an Anti-Lock Braking System" (bang-bang ABS demo).
* Predictive Performance of ABS with PID Controller Optimized by GSA,
  Automotive Experiences 2024 (PID parameter ranges).
* PathSim ABS example with zero-crossing events (lambda_opt, delta choices).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from abs_sim.control.pid import PID


class ABSState(Enum):
    APPLY = "A"
    HOLD = "H"
    RELEASE = "R"


@dataclass
class WheelABS:
    """Per-wheel ABS controller."""

    lambda_opt: float = 0.15          # target slip ratio (peak of mu-slip curve)
    delta: float = 0.03               # hysteresis around lambda_opt
    hold_duration: float = 0.03       # s, duration of HOLD before re-APPLY
    release_pressure: float = 0.0     # commanded pressure during RELEASE
    pid_authority: float = 0.6        # [0,1]: how much the PID can reduce driver demand
    hold_ratio: float = 0.6           # held_pressure = last_cmd * hold_ratio.
                                      # <1 prevents HOLD from re-applying the
                                      # exact pressure that caused overspin and
                                      # locking the FSM into a RELEASE<->HOLD
                                      # limit cycle.

    kp: float = 3.0
    ki: float = 40.0
    kd: float = 0.02

    _pid: PID = field(default=None)  # type: ignore[assignment]
    state: ABSState = ABSState.APPLY
    held_pressure: float = 0.0
    last_cmd: float = 0.0
    time_in_state: float = 0.0

    def __post_init__(self) -> None:
        if self._pid is None:
            self._pid = PID(self.kp, self.ki, self.kd, i_min=-1.0, i_max=1.0)

    def reset(self) -> None:
        self.state = ABSState.APPLY
        self.held_pressure = 0.0
        self.last_cmd = 0.0
        self.time_in_state = 0.0
        self._pid.reset()

    def _set_state(self, new_state: ABSState) -> None:
        if new_state != self.state:
            self.state = new_state
            self.time_in_state = 0.0

    def update(
        self,
        driver_demand: float,
        kappa: float,
        dt: float,
        enabled: bool = True,
    ) -> tuple[float, ABSState]:
        """Return (brake_pressure_command in [0,1], current FSM state).

        driver_demand  : raw brake intent from the driver, in [0, 1].
        kappa          : current longitudinal slip ratio for this wheel.
        dt             : controller timestep.
        enabled        : when False, the controller is bypassed and the driver
                         demand is forwarded as-is.
        """
        if driver_demand < 0.0:
            driver_demand = 0.0
        elif driver_demand > 1.0:
            driver_demand = 1.0

        if (not enabled) or driver_demand < 1e-6:
            self._set_state(ABSState.APPLY)
            self.held_pressure = 0.0
            self._pid.reset()
            self.last_cmd = driver_demand
            return driver_demand, self.state

        abs_k = abs(kappa)
        upper = self.lambda_opt + self.delta
        lower = self.lambda_opt - self.delta

        if self.state == ABSState.APPLY:
            if abs_k > upper:
                # Remember a REDUCED fraction of the pressure that just caused
                # overspin, not the pressure itself. Otherwise HOLD re-applies
                # exactly the command that locked the wheel and the FSM
                # oscillates RELEASE<->HOLD without ever entering controlled
                # APPLY.
                self.held_pressure = self.last_cmd * self.hold_ratio
                self._pid.reset()
                self._set_state(ABSState.RELEASE)
        elif self.state == ABSState.RELEASE:
            if abs_k < lower:
                self._set_state(ABSState.HOLD)
        elif self.state == ABSState.HOLD:
            if abs_k > upper:
                self._set_state(ABSState.RELEASE)
            elif self.time_in_state >= self.hold_duration:
                self._pid.reset()
                self._set_state(ABSState.APPLY)

        self.time_in_state += dt

        if self.state == ABSState.APPLY:
            error = self.lambda_opt - abs_k
            pid_out = self._pid.update(error, dt)
            scale = 1.0 + self.pid_authority * pid_out
            if scale > 1.0:
                scale = 1.0
            elif scale < 0.0:
                scale = 0.0
            cmd = driver_demand * scale
        elif self.state == ABSState.HOLD:
            cmd = self.held_pressure
        else:
            cmd = self.release_pressure

        if cmd > 1.0:
            cmd = 1.0
        elif cmd < 0.0:
            cmd = 0.0
        self.last_cmd = cmd
        return cmd, self.state
