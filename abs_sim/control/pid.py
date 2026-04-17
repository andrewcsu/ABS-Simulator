"""Minimal PID with optional integrator clamping (anti-windup)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PID:
    """Classical PID controller with anti-windup clamp on the integral term."""

    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    i_min: float = -1.0
    i_max: float = 1.0
    integral: float = 0.0
    prev_error: float = 0.0
    has_prev: bool = False

    def update(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return self.kp * error
        self.integral += error * dt
        if self.integral > self.i_max:
            self.integral = self.i_max
        elif self.integral < self.i_min:
            self.integral = self.i_min
        if self.has_prev:
            deriv = (error - self.prev_error) / dt
        else:
            deriv = 0.0
        self.prev_error = error
        self.has_prev = True
        return self.kp * error + self.ki * self.integral + self.kd * deriv

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev = False
