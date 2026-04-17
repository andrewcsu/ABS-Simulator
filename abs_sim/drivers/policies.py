"""Driver policies: cruise pursuit, random brake events, and curve-braking delay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import random
from typing import List, Optional

from abs_sim.control.pid import PID
from abs_sim.track.track import Track


G = 9.81


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass
class DriverCommand:
    """Driver output for a single control step."""
    throttle: float = 0.0        # [0, 1]
    brake_demand: float = 0.0    # [0, 1]
    steer: float = 0.0           # rad (front-wheel steer)


@dataclass
class DriverContext:
    """Information the driver observes each step (no engine-control signals)."""
    x: float
    y: float
    psi: float
    vx: float
    vy: float
    r: float
    speed: float
    wheelbase: float


class Driver(ABC):
    """Base driver: reset() + update(ctx, track, dt) -> DriverCommand."""

    name: str = "driver"
    color: tuple = (60, 200, 255)

    @abstractmethod
    def update(self, ctx: DriverContext, track: Track, dt: float) -> DriverCommand: ...

    def reset(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def pure_pursuit_steer(
    ctx: DriverContext, track: Track, s: float, lookahead: float,
) -> float:
    """Pure-pursuit front steer angle targeting a point lookahead down the track."""
    tx, ty, _, _, _ = track.sample(s + lookahead)
    dx = tx - ctx.x
    dy = ty - ctx.y
    bearing = math.atan2(dy, dx)
    alpha = _wrap(bearing - ctx.psi)
    return math.atan2(2.0 * ctx.wheelbase * math.sin(alpha), max(lookahead, 0.5))


def curvature_limited_target_speed(
    track: Track,
    s: float,
    v_cruise: float,
    mu_assumed: float,
    horizon: float = 120.0,
    step: float = 5.0,
    margin: float = 0.8,
    a_plan: float = 5.0,
) -> float:
    """Look ahead on the track and return a speed target that keeps the lateral
    acceleration below mu*g*margin on any upcoming curve, with enough time to
    brake from v_cruise at deceleration a_plan.
    """
    v_target = v_cruise
    d = step
    while d < horizon:
        s_ahead = s + d
        _, _, _, curv, _ = track.sample(s_ahead)
        if curv != 0.0:
            R = 1.0 / abs(curv)
            v_curve = math.sqrt(max(mu_assumed * G * R, 0.0)) * margin
            v_allowed = math.sqrt(max(v_curve * v_curve + 2.0 * a_plan * d, 0.0))
            if v_allowed < v_target:
                v_target = v_allowed
        d += step
    return v_target


def first_corner_start_s(track: Track, min_curvature: float = 0.01) -> Optional[float]:
    """Return the arc length at which the first significant curve begins."""
    for sam in track.samples():
        if abs(sam.curvature) >= min_curvature:
            return sam.s
    return None


# --------------------------------------------------------------------------- #
# Cruise / pursuit driver
# --------------------------------------------------------------------------- #

@dataclass
class CruisePursuitDriver(Driver):
    """Pure-pursuit steering + PI longitudinal speed control to a
    curvature-limited target."""

    v_cruise: float = 30.0
    mu_assumed: float = 0.8
    lookahead_base: float = 8.0
    lookahead_k: float = 0.3
    kp_v: float = 0.6
    ki_v: float = 0.4
    name: str = "cruise"
    color: tuple = (60, 200, 255)

    _pid_v: PID = field(default=None)  # type: ignore[assignment]
    _last_s: float = 0.0

    def __post_init__(self) -> None:
        self._pid_v = PID(self.kp_v, self.ki_v, 0.0, i_min=-2.0, i_max=2.0)

    def reset(self) -> None:
        self._pid_v.reset()
        self._last_s = 0.0

    def _target_speed(self, track: Track, s: float) -> float:
        return curvature_limited_target_speed(
            track, s, self.v_cruise, self.mu_assumed,
        )

    def update(self, ctx: DriverContext, track: Track, dt: float) -> DriverCommand:
        s, _ = track.closest(ctx.x, ctx.y, s_hint=self._last_s)
        self._last_s = s

        lookahead = self.lookahead_base + self.lookahead_k * ctx.speed
        steer = pure_pursuit_steer(ctx, track, s, lookahead)
        v_target = self._target_speed(track, s)
        err = v_target - ctx.speed
        u = self._pid_v.update(err, dt)
        if u > 0.0:
            throttle = min(u, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-u, 1.0)
        return DriverCommand(throttle=throttle, brake_demand=brake, steer=steer)


# --------------------------------------------------------------------------- #
# Curve-brake-delay driver (3-driver comparison scenario)
# --------------------------------------------------------------------------- #

@dataclass
class CurveBrakeDelayDriver(CruisePursuitDriver):
    """Same steering and speed control as Cruise, but the driver brakes for
    the FIRST corner at (ideal - delay) seconds -- so negative delay means
    'brakes early', positive means 'brakes late'.

    The ideal brake start is computed from kinematics: it is the location where
    starting a constant-`a_brake_plan` deceleration from `v_cruise` would just
    bring the car to `v_corner` by the corner entry.
    """

    v_corner: float = 13.0           # m/s target speed in the corner
    delay_s: float = 0.0             # seconds, +late / -early
    a_brake_plan: float = 6.0        # m/s^2 planned deceleration
    name: str = "curve_brake"

    _corner_s: Optional[float] = None
    _ideal_brake_s: Optional[float] = None

    def _ensure_plan(self, track: Track) -> None:
        if self._corner_s is not None:
            return
        s_corner = first_corner_start_s(track)
        if s_corner is None:
            self._corner_s = -1.0
            self._ideal_brake_s = -1.0
            return
        v0 = self.v_cruise
        v1 = self.v_corner
        d_brake = max((v0 * v0 - v1 * v1) / (2.0 * self.a_brake_plan), 0.0)
        self._corner_s = s_corner
        self._ideal_brake_s = s_corner - d_brake

    def _target_speed(self, track: Track, s: float) -> float:
        self._ensure_plan(track)
        v_base = self.v_cruise
        if self._ideal_brake_s is None or self._corner_s is None:
            return v_base
        if self._corner_s < 0:
            return v_base

        brake_s = self._ideal_brake_s + self.delay_s * self.v_cruise

        if s < brake_s:
            return v_base
        if brake_s <= s <= self._corner_s:
            return self.v_corner
        return v_base


# --------------------------------------------------------------------------- #
# Random full-stop brake-event driver (straight demo)
# --------------------------------------------------------------------------- #

@dataclass
class RandomBrakeEventDriver(CruisePursuitDriver):
    """Cruise driver with random full-stop brake events overlaid.

    At random arc-length s values (sampled once at reset), the driver slams
    the brakes to 1.0 for `hold_duration` regardless of speed target.
    """

    rng_seed: int = 0
    min_gap: float = 60.0
    max_gap: float = 120.0
    hold_duration: float = 1.5
    name: str = "random_brake"

    _events: List[tuple] = field(default_factory=list)   # list of (s_trigger, released)
    _active_until: float = -1.0

    def _plan_events(self, track: Track) -> None:
        rng = random.Random(self.rng_seed)
        self._events = []
        s = self.min_gap
        L = track.total_length
        while s < L:
            self._events.append((s, False))
            s += rng.uniform(self.min_gap, self.max_gap)

    def reset(self) -> None:
        super().reset()
        self._events = []
        self._active_until = -1.0

    def update(self, ctx: DriverContext, track: Track, dt: float) -> DriverCommand:
        if not self._events:
            self._plan_events(track)
        cmd = super().update(ctx, track, dt)
        s, _ = track.closest(ctx.x, ctx.y, s_hint=self._last_s)

        t_now = self._last_s  # proxy: we don't have absolute time; use s
        for i, (s_trig, done) in enumerate(self._events):
            if done:
                continue
            if s >= s_trig:
                self._active_until = s + ctx.speed * self.hold_duration + 5.0
                self._events[i] = (s_trig, True)
                break

        if s < self._active_until:
            return DriverCommand(throttle=0.0, brake_demand=1.0, steer=cmd.steer)
        return cmd
