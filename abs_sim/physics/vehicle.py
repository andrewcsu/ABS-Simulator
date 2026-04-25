"""Vehicle: 4-wheel chassis + wheels + tire model + RK4 integrator.

State vector (numpy array, length 10):
    [0]  x      world-frame position (m)
    [1]  y      world-frame position (m)
    [2]  psi    yaw angle (rad, CCW from +x world axis)
    [3]  vx     body-frame longitudinal velocity (m/s)
    [4]  vy     body-frame lateral velocity (m/s, +y = right)
    [5]  r      yaw rate (rad/s)
    [6]  w_FL   front-left wheel spin rate (rad/s)
    [7]  w_FR   front-right wheel spin rate (rad/s)
    [8]  w_RL   rear-left wheel spin rate (rad/s)
    [9]  w_RR   rear-right wheel spin rate (rad/s)

Inputs (held constant through a single RK4 step):
    * brake_pressure[4] in [0, 1]     per-wheel brake actuator output
    * drive_torque           (Nm)     net longitudinal torque from engine;
                                      split between driven wheels
    * steer                  (rad)    front-wheel steer angle
    * mu[4]                           per-wheel road friction

Notes
-----
Load transfer is computed quasi-statically from body-frame SPECIFIC FORCES
(Fx/m, Fy/m -- the accelerometer-style signal) cached from the previous
step, which is the standard control-oriented approximation: avoids a
fixpoint inside RK4 and still captures the dominant weight-shift behavior
under braking AND cornering. Note: velocity derivatives (dvx/dt, dvy/dt)
are NOT the right input for load transfer -- during steady cornering
dvy/dt goes to zero while the physical lateral acceleration is r*vx.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple
import math

import numpy as np

from abs_sim.physics.chassis import ChassisParams, load_transfer, static_loads
from abs_sim.physics.tire import DugoffTire
from abs_sim.physics.wheel import (
    EPS_V,
    WheelKinematics,
    WheelParams,
    compute_slip,
    wheel_velocity_in_tire_frame,
)


FL, FR, RL, RR = 0, 1, 2, 3
WHEEL_NAMES = ("FL", "FR", "RL", "RR")
STATE_DIM = 10


@dataclass
class VehicleInputs:
    """Driver-plus-ABS inputs to the vehicle for one timestep."""

    steer: float = 0.0                         # rad, front steer angle
    drive_torque: float = 0.0                  # Nm, positive = accelerating
    brake_pressure: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    mu: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 0.9)
    tcs_enabled: bool = True                   # simple engine-side traction control


# Per-driven-wheel drive-torque cap as a fraction of tire peak grip
# (mu*Fz*R). Must stay STRICTLY BELOW 1.0 so the wheel equilibrates in
# the stable pre-peak region of the mu-slip curve; a value >= 1 parks the
# operating point at or past the friction peak, where the tire's Fx
# derivative vs kappa is zero or negative and any excess torque spins the
# wheel up without bound. 0.85 leaves enough margin for the wheel to
# actually accelerate the car while still preventing the inside rear from
# entering positive-feedback wheelspin on corner exit.
TCS_CAP_FRACTION: float = 0.85

# Kappa above which TCS aggressively cuts drive torque (primarily a
# backstop for transients the static cap alone can't catch fast enough).
TCS_CUT_KAPPA: float = 0.25

# Characteristic spin rate (rad/s) over which the brake-torque sign switches.
# ~1 rad/s corresponds to ~0.3 m/s wheel-edge speed, small enough to be
# indistinguishable from a locked wheel while still giving a continuous
# derivative for the RK4 integrator to stage against.
BRAKE_SMOOTH_W: float = 1.0


def default_wheels(tire: DugoffTire | None = None) -> List[WheelParams]:
    """Four-wheel params with a standard sedan layout.

    Only rear wheels are 'driven' in this default (RWD) so accelerating sets
    longitudinal wheel torque on the rear axle. Front wheels steer.
    """
    if tire is None:
        tire = DugoffTire()
    p = ChassisParams()
    return [
        WheelParams(x_offset=+p.a, y_offset=-p.track / 2.0, steerable=True, driven=False, tire=tire),
        WheelParams(x_offset=+p.a, y_offset=+p.track / 2.0, steerable=True, driven=False, tire=tire),
        WheelParams(x_offset=-p.b, y_offset=-p.track / 2.0, steerable=False, driven=True, tire=tire),
        WheelParams(x_offset=-p.b, y_offset=+p.track / 2.0, steerable=False, driven=True, tire=tire),
    ]


@dataclass
class Vehicle:
    """Full 4-wheel vehicle: chassis + wheels + tire model + integrator."""

    chassis: ChassisParams = field(default_factory=ChassisParams)
    wheels: List[WheelParams] = field(default_factory=default_wheels)
    state: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    _last_ax: float = 0.0
    _last_ay: float = 0.0
    _last_kin: List[WheelKinematics] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.wheels) != 4:
            raise ValueError("Vehicle requires exactly 4 wheels")
        if self._last_kin == []:
            self._last_kin = [WheelKinematics() for _ in range(4)]

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #
    @property
    def x(self) -> float: return float(self.state[0])
    @property
    def y(self) -> float: return float(self.state[1])
    @property
    def psi(self) -> float: return float(self.state[2])
    @property
    def vx(self) -> float: return float(self.state[3])
    @property
    def vy(self) -> float: return float(self.state[4])
    @property
    def r(self) -> float: return float(self.state[5])
    @property
    def speed(self) -> float: return math.hypot(self.vx, self.vy)
    @property
    def wheel_speeds(self) -> Tuple[float, float, float, float]:
        return (float(self.state[6]), float(self.state[7]),
                float(self.state[8]), float(self.state[9]))

    def set_pose(self, x: float, y: float, psi: float) -> None:
        self.state[0] = x
        self.state[1] = y
        self.state[2] = psi

    def set_speed(self, v: float) -> None:
        """Initialize forward motion: set body vx and spin rates consistent with rolling."""
        self.state[3] = v
        self.state[4] = 0.0
        self.state[5] = 0.0
        R = self.wheels[0].tire.R
        if R > 0:
            w = v / R
            self.state[6:10] = w

    # ------------------------------------------------------------------ #
    # Dynamics core
    # ------------------------------------------------------------------ #
    def _compute_wheel_kinematics(
        self,
        s: np.ndarray,
        inputs: VehicleInputs,
        ax_cache: float,
        ay_cache: float,
    ) -> List[WheelKinematics]:
        """Compute kinematics and tire forces for all 4 wheels given state."""
        vx, vy, r = s[3], s[4], s[5]
        w = s[6:10]

        Fz = load_transfer(self.chassis, ax_cache, ay_cache)

        out: List[WheelKinematics] = []
        for i, wp in enumerate(self.wheels):
            steer = inputs.steer if wp.steerable else 0.0
            vx_t, vy_t = wheel_velocity_in_tire_frame(
                vx, vy, r, wp.x_offset, wp.y_offset, steer
            )
            kappa, alpha = compute_slip(float(w[i]), vx_t, vy_t, wp.tire.R)
            Fx_tire, Fy_tire = wp.tire.forces(kappa, alpha, Fz[i], inputs.mu[i])
            out.append(
                WheelKinematics(
                    vx_tire=vx_t, vy_tire=vy_t,
                    kappa=kappa, alpha=alpha,
                    Fz=Fz[i], Fx_tire=Fx_tire, Fy_tire=Fy_tire,
                    mu=inputs.mu[i],
                )
            )
        return out

    def _derivatives(
        self,
        s: np.ndarray,
        inputs: VehicleInputs,
        ax_cache: float,
        ay_cache: float,
    ) -> Tuple[np.ndarray, float, float, List[WheelKinematics]]:
        """Compute state derivative vector plus fresh ax, ay for caching."""
        p = self.chassis
        psi, vx, vy, r = s[2], s[3], s[4], s[5]
        w = s[6:10]
        kin = self._compute_wheel_kinematics(s, inputs, ax_cache, ay_cache)

        n_driven = sum(1 for wp in self.wheels if wp.driven)
        T_drive_each = (inputs.drive_torque / n_driven) if n_driven > 0 else 0.0

        Fx_body_total = 0.0
        Fy_body_total = 0.0
        Mz_total = 0.0
        d_omega = np.zeros(4)

        for i, wp in enumerate(self.wheels):
            steer = inputs.steer if wp.steerable else 0.0
            c = math.cos(steer)
            ss = math.sin(steer)
            Fx_b = c * kin[i].Fx_tire - ss * kin[i].Fy_tire
            Fy_b = ss * kin[i].Fx_tire + c * kin[i].Fy_tire

            Fx_body_total += Fx_b
            Fy_body_total += Fy_b
            Mz_total += wp.x_offset * Fy_b - wp.y_offset * Fx_b

            if wp.driven:
                T_drive = T_drive_each
                if inputs.tcs_enabled:
                    tire_cap = max(
                        inputs.mu[i] * kin[i].Fz * wp.tire.R * TCS_CAP_FRACTION,
                        0.0,
                    )
                    # Static pre-peak cap keeps the equilibrium slip stable.
                    if T_drive > tire_cap:
                        T_drive = tire_cap
                    elif T_drive < -tire_cap:
                        T_drive = -tire_cap
                    # Dynamic cut if the wheel is already past peak slip
                    # (e.g. briefly during the load-transfer transient at
                    # corner exit): scale the drive torque down toward zero
                    # so the wheel can decelerate back into the pre-peak region.
                    if T_drive > 0.0 and kin[i].kappa > TCS_CUT_KAPPA:
                        cut = max(0.0, 1.0 - (kin[i].kappa - TCS_CUT_KAPPA) * 8.0)
                        T_drive *= cut
                    elif T_drive < 0.0 and kin[i].kappa < -TCS_CUT_KAPPA:
                        cut = max(0.0, 1.0 - (-kin[i].kappa - TCS_CUT_KAPPA) * 8.0)
                        T_drive *= cut
            else:
                T_drive = 0.0
            T_brake_i = inputs.brake_pressure[i] * wp.brake_torque_max
            # Smooth Coulomb brake friction: T_brake_signed = -T_brake_i * sign(w)
            # but with a linear transition through w=0. A hard sign flip causes
            # the RK4 stages (which re-evaluate w at s + 0.5*dt*k1 etc.) to
            # chatter between +T_brake and -T_brake when the wheel is near lock,
            # injecting spurious high-frequency torque.
            w_i = float(w[i])
            smooth = math.tanh(w_i / BRAKE_SMOOTH_W)
            T_brake_signed = -T_brake_i * smooth
            d_omega[i] = (T_drive + T_brake_signed - wp.tire.R * kin[i].Fx_tire) / wp.J

        # Aerodynamic drag (along body x, always opposing motion)
        F_drag = 0.5 * p.rho_air * p.Cd_A * vx * abs(vx)
        Fx_body_total -= F_drag

        # Body-frame specific forces (Fx/m, Fy/m). These are what an
        # accelerometer rigidly attached to the body would read, and what
        # load_transfer() needs for its d'Alembert inertial-force formula.
        # The velocity derivatives (dvx/dt, dvy/dt) include Coriolis terms
        # (r*vy, -r*vx) and are NOT the right inputs for load transfer --
        # during steady cornering dvy/dt goes to zero while the real
        # lateral accel is r*vx.
        ax_spec = Fx_body_total / p.mass
        ay_spec = Fy_body_total / p.mass
        dr = Mz_total / p.Izz

        cpsi = math.cos(psi)
        spsi = math.sin(psi)
        dx = cpsi * vx - spsi * vy
        dy = spsi * vx + cpsi * vy
        dpsi = r
        dvx = ax_spec + r * vy
        dvy = ay_spec - r * vx

        dstate = np.empty(STATE_DIM)
        dstate[0] = dx
        dstate[1] = dy
        dstate[2] = dpsi
        dstate[3] = dvx
        dstate[4] = dvy
        dstate[5] = dr
        dstate[6:10] = d_omega
        return dstate, float(ax_spec), float(ay_spec), kin

    # ------------------------------------------------------------------ #
    # Integrator
    # ------------------------------------------------------------------ #
    def step(self, dt: float, inputs: VehicleInputs) -> None:
        """Advance the vehicle state by dt using classical RK4.

        Load transfer uses cached ax, ay from the previous completed step;
        this is standard control-oriented practice and avoids a fixpoint.
        """
        s = self.state
        axc, ayc = self._last_ax, self._last_ay

        k1, ax1, ay1, _ = self._derivatives(s, inputs, axc, ayc)
        k2, _, _, _ = self._derivatives(s + 0.5 * dt * k1, inputs, ax1, ay1)
        k3, _, _, _ = self._derivatives(s + 0.5 * dt * k2, inputs, ax1, ay1)
        k4, ax4, ay4, kin4 = self._derivatives(s + dt * k3, inputs, ax1, ay1)

        self.state = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        for i in range(4):
            if self.state[6 + i] < 0.0:
                self.state[6 + i] = 0.0

        kin_final = self._compute_wheel_kinematics(self.state, inputs, ax4, ay4)
        self._last_kin = kin_final
        self._last_ax = 0.5 * (ax1 + ax4)
        self._last_ay = 0.5 * (ay1 + ay4)

    # ------------------------------------------------------------------ #
    # Exposed snapshots for controllers, viz, telemetry
    # ------------------------------------------------------------------ #
    def wheel_kinematics(self) -> List[WheelKinematics]:
        return list(self._last_kin)

    def ax_body(self) -> float:
        """Body-frame longitudinal specific force (Fx/m), i.e. what a body-
        mounted accelerometer would read. Same quantity fed to load_transfer."""
        return self._last_ax

    def ay_body(self) -> float:
        """Body-frame lateral specific force (Fy/m)."""
        return self._last_ay

    def static_loads(self) -> Tuple[float, float, float, float]:
        return static_loads(self.chassis)

    def wheel_world_positions(self) -> List[Tuple[float, float]]:
        """Return world-frame (x, y) for each tire contact, in order FL, FR, RL, RR.

        Applies the same body->world rotation used inside ``_derivatives``
        (psi CCW from world +x). Used by the simulation layer to look up
        per-wheel road surface, e.g. on split-mu tracks where the left and
        right half-lanes differ.
        """
        cpsi = math.cos(self.psi)
        spsi = math.sin(self.psi)
        x, y = self.x, self.y
        out: List[Tuple[float, float]] = []
        for wp in self.wheels:
            xw = x + cpsi * wp.x_offset - spsi * wp.y_offset
            yw = y + spsi * wp.x_offset + cpsi * wp.y_offset
            out.append((xw, yw))
        return out
