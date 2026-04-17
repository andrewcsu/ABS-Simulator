"""Brake actuator: first-order hydraulic lag with a rate limiter.

Represents a simplified brake hydraulic line + caliper: the commanded brake
pressure (in [0, 1]) does not take effect instantly. A first-order lag with
time constant `tau` models the hydraulic fill + caliper response, and a rate
limit caps how fast the pressure can slew (both up and down). This is what
gives the system a realistic "low latency but non-zero" brake-press-to-actuate
behavior and is where the low-latency non-functional requirement is tunable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BrakeActuator:
    tau: float = 0.03         # s, first-order lag time constant
    rate_max: float = 12.0    # 1/s, maximum rate of pressure change
    pressure: float = 0.0     # current actual brake pressure (state)

    def update(self, cmd: float, dt: float) -> float:
        """Advance the actuator state by dt given the commanded pressure."""
        cmd = 0.0 if cmd < 0.0 else (1.0 if cmd > 1.0 else cmd)
        if self.tau <= 0.0:
            desired_rate = (cmd - self.pressure) / max(dt, 1e-9)
        else:
            desired_rate = (cmd - self.pressure) / self.tau
        if desired_rate > self.rate_max:
            desired_rate = self.rate_max
        elif desired_rate < -self.rate_max:
            desired_rate = -self.rate_max
        new_p = self.pressure + desired_rate * dt
        if new_p > 1.0:
            new_p = 1.0
        elif new_p < 0.0:
            new_p = 0.0
        self.pressure = new_p
        return self.pressure

    def reset(self) -> None:
        self.pressure = 0.0
