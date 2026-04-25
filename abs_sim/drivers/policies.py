"""Driver policies: cruise pursuit, random brake events, curve-braking delay,
and named persona archetypes (Pro / Cautious / Novice / Aggressive) that
combine turn-timing and brake-timing offsets on top of the cruise driver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import random
from typing import Callable, Dict, List, Optional, Tuple

from abs_sim.control.pid import PID
from abs_sim.physics.tire import SURFACES
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

def _lookahead_s(track: Track, s: float, lookahead: float) -> float:
    """Return the arc length to sample `lookahead` meters ahead of `s`.

    On a closed track the result can wrap past `total_length`; on an open
    track we clamp so pure-pursuit doesn't teleport its target back to the
    start of the track via `sample()`'s implicit `s % L`.
    """
    s_ahead = s + lookahead
    if track.is_closed:
        return s_ahead
    return min(s_ahead, track.total_length)


def pure_pursuit_steer(
    ctx: DriverContext, track: Track, s: float, lookahead: float,
) -> float:
    """Pure-pursuit front steer angle targeting a point lookahead down the track."""
    tx, ty, _, _, _ = track.sample(_lookahead_s(track, s, lookahead))
    dx = tx - ctx.x
    dy = ty - ctx.y
    bearing = math.atan2(dy, dx)
    alpha = _wrap(bearing - ctx.psi)
    return math.atan2(2.0 * ctx.wheelbase * math.sin(alpha), max(lookahead, 0.5))


def _mu_limit_at(track: Track, s: float, mu_fallback: float) -> float:
    """Return the conservative mu to plan against at arc length s.

    On a split-mu road the outside-of-turn tires (which carry most load under
    cornering) can sit on a much lower mu than the driver's nominal assumption.
    Probing both half-lanes and taking the minimum gives a lateral-grip
    estimate that's correct whether the surfaces are uniform, full-width
    patches, or half-lane split. Also caps by the driver's own ``mu_assumed``
    so drivers who want to stay conservative on unknown tracks still do.
    """
    left = SURFACES.get(track.surface_at(s, e=-1.0), SURFACES["dry"]).mu
    right = SURFACES.get(track.surface_at(s, e=+1.0), SURFACES["dry"]).mu
    return min(mu_fallback, left, right)


def _mu_brake_plan_at(track: Track, s: float, mu_fallback: float) -> float:
    """Mu to plan BRAKING deceleration against.

    Braking on split-mu produces an asymmetric yaw moment that ESC has to
    fight -- and that fight spends lateral grip on the outside wheels. If
    we plan braking by the average mu we'll demand too much brake force in
    the seconds before a corner, the asymmetric yaw moment fights the
    steering, and the car spins. We bias TOWARD the worst side: 40% min mu
    + 60% avg mu, capped at the driver's nominal mu_assumed. On uniform
    surfaces this collapses to mu_assumed (no behaviour change); on a true
    ice/dry split it gives mu ~ 0.34 -> a_plan ~ 2.7 m/s^2 so the driver
    starts braking earlier and applies less peak pedal.
    """
    left = SURFACES.get(track.surface_at(s, e=-1.0), SURFACES["dry"]).mu
    right = SURFACES.get(track.surface_at(s, e=+1.0), SURFACES["dry"]).mu
    blended = 0.4 * min(left, right) + 0.6 * 0.5 * (left + right)
    return min(mu_fallback, blended)


def curvature_limited_target_speed(
    track: Track,
    s: float,
    v_cruise: float,
    mu_assumed: float,
    horizon: float = 180.0,
    step: float = 5.0,
    margin: float = 0.8,
    a_plan: float = 5.0,
    settle_buffer: float = 25.0,
) -> float:
    """Look ahead on the track and return a speed target that keeps the lateral
    acceleration below mu*g*margin on any upcoming curve, with enough time to
    brake from v_cruise at deceleration a_plan.

    The lateral grip cap uses the MINIMUM mu across both half-lanes, so a
    split-mu road (e.g. ice left / dry right) forces a low corner-entry
    speed even though the centerline surface may read as dry. The
    deceleration cap (used to figure out *when* to start braking ahead of
    the corner) is biased toward the worst side (40% min + 60% avg), which
    on split-mu gives a softer pedal application than on uniform dry --
    this limits the asymmetric yaw moment that ESC has to fight on entry.
    Finally, ``settle_buffer`` adds a few meters of pre-corner runway: the
    planner asks for v_curve to be reached BEFORE the corner mouth, so any
    residual yaw from braking has time to die down before the steering
    input arrives. This is the single biggest fix for split-mu corner
    entry: braking and turning at the same time on asymmetric grip is
    what causes the spin.
    """
    v_target = v_cruise
    d = step
    L = track.total_length
    closed = track.is_closed
    while d < horizon:
        s_ahead = s + d
        if not closed and s_ahead > L:
            break  # don't wrap around the start of an open track
        _, _, _, curv, _ = track.sample(s_ahead)
        if curv != 0.0:
            R = 1.0 / abs(curv)
            mu_lat = _mu_limit_at(track, s_ahead, mu_assumed)
            mu_long = _mu_brake_plan_at(track, s_ahead, mu_assumed)
            v_curve = math.sqrt(max(mu_lat * G * R, 0.0)) * margin
            a_plan_eff = min(a_plan, mu_long * G * margin)
            d_eff = max(d - settle_buffer, 0.0)
            v_allowed = math.sqrt(max(v_curve * v_curve + 2.0 * a_plan_eff * d_eff, 0.0))
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
    # Pedal slew-rate (units of demand / second). A real driver's foot can
    # go from idle to full throttle in ~0.5 s; without this limit the PI
    # saturates to 1.0 the instant a curvature-limited v_target steps
    # from ~12 to ~30 m/s at corner exit, overwhelming TCS and producing
    # rear-wheelspin that cascades into a spin-out.
    throttle_rate: float = 2.5       # 0 -> 1 in 0.4 s
    brake_rate: float = 5.0          # brake can rise a bit faster
    name: str = "cruise"
    color: tuple = (60, 200, 255)

    _pid_v: PID = field(default=None)  # type: ignore[assignment]
    _last_s: float = 0.0
    _last_throttle: float = 0.0
    _last_brake: float = 0.0

    def __post_init__(self) -> None:
        self._pid_v = PID(self.kp_v, self.ki_v, 0.0, i_min=-2.0, i_max=2.0)

    def reset(self) -> None:
        self._pid_v.reset()
        self._last_s = 0.0
        self._last_throttle = 0.0
        self._last_brake = 0.0

    def _target_speed(self, track: Track, s: float) -> float:
        return curvature_limited_target_speed(
            track, s, self.v_cruise, self.mu_assumed,
        )

    @staticmethod
    def _slew(current: float, target: float, rate: float, dt: float) -> float:
        max_step = rate * dt
        delta = target - current
        if delta > max_step:
            return current + max_step
        if delta < -max_step:
            return current - max_step
        return target

    def update(self, ctx: DriverContext, track: Track, dt: float) -> DriverCommand:
        s, _ = track.closest(ctx.x, ctx.y, s_hint=self._last_s)
        self._last_s = s

        lookahead = self.lookahead_base + self.lookahead_k * ctx.speed
        steer = pure_pursuit_steer(ctx, track, s, lookahead)
        v_target = self._target_speed(track, s)
        err = v_target - ctx.speed
        u = self._pid_v.update(err, dt)
        if u > 0.0:
            throttle_target = min(u, 1.0)
            brake_target = 0.0
        else:
            throttle_target = 0.0
            brake_target = min(-u, 1.0)

        throttle = self._slew(self._last_throttle, throttle_target, self.throttle_rate, dt)
        brake = self._slew(self._last_brake, brake_target, self.brake_rate, dt)
        self._last_throttle = throttle
        self._last_brake = brake
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


# --------------------------------------------------------------------------- #
# Persona driver: configurable turn-timing and brake-timing on top of Cruise
# --------------------------------------------------------------------------- #

# Default settle buffer used by the base Cruise driver (matches the default
# in ``curvature_limited_target_speed``). Centralizing it here lets the
# persona subclass shift it per-driver without hard-coding the magic number
# in multiple places.
_DEFAULT_SETTLE_BUFFER: float = 25.0


@dataclass
class PersonaDriver(CruisePursuitDriver):
    """Cruise + pure-pursuit driver with independent turn- and brake-timing
    offsets, used to model driving-style archetypes (Pro, Cautious, Novice,
    Aggressive).

    Both offsets are expressed in SECONDS so that tuning is roughly
    speed-independent: a ``turn_lead_s`` of +0.4 shifts the pure-pursuit
    steering target ``0.4 * speed`` meters further down the road at any
    speed. Positive values mean the driver ANTICIPATES (turns/brakes
    early); negative values mean the driver REACTS (turns/brakes late).

    * ``turn_lead_s``: adds ``turn_lead_s * speed`` meters to the pure-
      pursuit lookahead distance. A positive value makes the driver pick
      a steering target further ahead, so they steer before the geometric
      corner mouth (an over-prepared driver). A negative value shrinks the
      lookahead, so the steering target lags the vehicle -- steer input
      only appears once the car is already in the corner.
    * ``brake_lead_s``: adds ``brake_lead_s * v_cruise`` meters to the
      planner's ``settle_buffer``. Positive = brakes earlier (the planner
      aims to reach v_curve well BEFORE the corner mouth); negative = the
      driver dives in hot.

    The rest of the behaviour (PI velocity control, pedal slew limits, per-
    half-lane mu probing, etc.) is inherited unchanged from
    :class:`CruisePursuitDriver`.
    """

    turn_lead_s: float = 0.0   # seconds; + early / - late
    brake_lead_s: float = 0.0  # seconds; + early / - late
    name: str = "persona"

    def _target_speed(self, track: Track, s: float) -> float:
        eff_settle = max(0.0, _DEFAULT_SETTLE_BUFFER + self.brake_lead_s * self.v_cruise)
        return curvature_limited_target_speed(
            track, s, self.v_cruise, self.mu_assumed, settle_buffer=eff_settle,
        )

    def update(self, ctx: DriverContext, track: Track, dt: float) -> DriverCommand:
        s, _ = track.closest(ctx.x, ctx.y, s_hint=self._last_s)
        self._last_s = s

        base_lookahead = self.lookahead_base + self.lookahead_k * ctx.speed
        # Positive turn_lead_s -> larger lookahead -> steer target sampled
        # further down the track -> driver turns EARLIER. Clamp to a small
        # positive floor so pure_pursuit_steer's sin/distance math stays
        # well-defined if someone picks a wildly negative value at low speed.
        lookahead = max(0.5, base_lookahead + self.turn_lead_s * ctx.speed)
        steer = pure_pursuit_steer(ctx, track, s, lookahead)

        v_target = self._target_speed(track, s)
        err = v_target - ctx.speed
        u = self._pid_v.update(err, dt)
        if u > 0.0:
            throttle_target = min(u, 1.0)
            brake_target = 0.0
        else:
            throttle_target = 0.0
            brake_target = min(-u, 1.0)

        throttle = self._slew(self._last_throttle, throttle_target, self.throttle_rate, dt)
        brake = self._slew(self._last_brake, brake_target, self.brake_rate, dt)
        self._last_throttle = throttle
        self._last_brake = brake
        return DriverCommand(throttle=throttle, brake_demand=brake, steer=steer)


# --------------------------------------------------------------------------- #
# Persona factory registry
# --------------------------------------------------------------------------- #

# RGB colors matched to each archetype's character: cyan = calm/expert,
# blue = deliberate, yellow = uncertain, red = fast/risky. Exposed so the
# CLI demo and the pygame HUD can paint each persona consistently.
PERSONA_COLORS: Dict[str, Tuple[int, int, int]] = {
    "pro":        ( 80, 220, 230),
    "cautious":   ( 80, 140, 255),
    "novice":     (245, 220,  80),
    "aggressive": (230,  80,  80),
}


def _make_pro(v_cruise: Optional[float] = None) -> PersonaDriver:
    """Clean, on-time baseline: same timing as the existing Cruise driver,
    with a slightly optimistic mu assumption (an experienced driver knows
    how much grip the road actually has)."""
    return PersonaDriver(
        v_cruise=18.0 if v_cruise is None else v_cruise,
        mu_assumed=0.85,
        turn_lead_s=0.0,
        brake_lead_s=0.0,
        name="pro",
        color=PERSONA_COLORS["pro"],
    )


def _make_cautious(v_cruise: Optional[float] = None) -> PersonaDriver:
    """Over-prepared: anticipates both steering and braking, cruises slower,
    and assumes less grip than is really available. Clean lines but losing
    time to the Pro on every straight."""
    return PersonaDriver(
        v_cruise=14.0 if v_cruise is None else v_cruise,
        mu_assumed=0.70,
        turn_lead_s=+0.4,
        brake_lead_s=+0.5,
        name="cautious",
        color=PERSONA_COLORS["cautious"],
    )


def _make_novice(v_cruise: Optional[float] = None) -> PersonaDriver:
    """Reactive, slightly overconfident: turns and brakes late because they
    aren't anticipating the corner, and assumes more grip than is really
    there. Expect a lot of understeer on tight corners."""
    return PersonaDriver(
        v_cruise=18.0 if v_cruise is None else v_cruise,
        mu_assumed=0.90,
        turn_lead_s=-0.3,
        brake_lead_s=-0.4,
        name="novice",
        color=PERSONA_COLORS["novice"],
    )


def _make_aggressive(v_cruise: Optional[float] = None) -> PersonaDriver:
    """Fast and late: cruises high, trail-brakes deep into corners, and
    assumes near-maximum grip. Quickest on open track but leaves no margin
    for mistakes or surface changes."""
    return PersonaDriver(
        v_cruise=22.0 if v_cruise is None else v_cruise,
        mu_assumed=0.95,
        turn_lead_s=-0.2,
        brake_lead_s=-0.6,
        name="aggressive",
        color=PERSONA_COLORS["aggressive"],
    )


# Public registry. Order is meaningful for the pygame 'P'-key cycle: Pro
# first (the baseline the user sees by default), then the two anticipating
# and reacting variants, then Aggressive last as the "push it" option.
PERSONAS: Dict[str, Callable[..., PersonaDriver]] = {
    "pro":        _make_pro,
    "cautious":   _make_cautious,
    "novice":     _make_novice,
    "aggressive": _make_aggressive,
}
