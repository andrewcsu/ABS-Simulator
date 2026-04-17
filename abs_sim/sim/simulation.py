"""Simulation loop that orchestrates vehicles + controllers + events + telemetry.

Design
------
* Physics ticks at dt_phys (default 1 ms).
* Driver + ABS + Stability tick at dt_ctrl (default 5 ms), accumulating between
  physics steps.
* One or more Cars run on the same track, each with its own Vehicle, driver,
  ABS, actuators, and stability controller.
* Events are time-triggered. Live events can be injected via inject(event_fn).
* Telemetry is logged at each controller tick for one named car by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from abs_sim.control.brake_actuator import BrakeActuator
from abs_sim.control.stability import StabilityController
from abs_sim.control.wheel_abs import ABSState, WheelABS
from abs_sim.drivers.policies import (
    CruisePursuitDriver,
    Driver,
    DriverCommand,
    DriverContext,
)
from abs_sim.physics.tire import SURFACES
from abs_sim.physics.vehicle import Vehicle, VehicleInputs
from abs_sim.sim.events import EventQueue
from abs_sim.sim.telemetry import TelemetryLogger
from abs_sim.track.track import Track


@dataclass
class Car:
    """One vehicle + its per-wheel ABS / actuators / stability / driver."""

    name: str
    vehicle: Vehicle
    driver: Driver
    abs_controllers: List[WheelABS] = field(default_factory=list)
    actuators: List[BrakeActuator] = field(default_factory=list)
    stability: StabilityController = field(default_factory=StabilityController)
    color: Tuple[int, int, int] = (60, 200, 255)

    abs_enabled: bool = True
    stability_enabled: bool = True

    # Override fields (set by events):
    mu_multiplier: float = 1.0
    surface_override: Optional[str] = None          # forces a surface under car
    brake_override: float = 0.0                     # additional brake demand
    brake_override_until: float = -1.0

    # Telemetry fields, updated per controller tick:
    last_cmd: DriverCommand = field(default_factory=DriverCommand)
    last_pre_abs: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_post_abs: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_post_stab: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_actuator_pressure: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_abs_states: Tuple[ABSState, ABSState, ABSState, ABSState] = (
        ABSState.APPLY, ABSState.APPLY, ABSState.APPLY, ABSState.APPLY,
    )
    last_yaw_error: float = 0.0
    last_stab_out: float = 0.0
    last_mu: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 0.9)
    last_surface: str = "dry"

    def reset(self) -> None:
        for a in self.abs_controllers:
            a.reset()
        for b in self.actuators:
            b.reset()
        self.stability.reset()
        self.driver.reset()
        self.brake_override = 0.0
        self.brake_override_until = -1.0
        self.surface_override = None
        self.mu_multiplier = 1.0

    @classmethod
    def make_default(
        cls,
        name: str,
        driver: Optional[Driver] = None,
        color: Tuple[int, int, int] = (60, 200, 255),
        abs_lambda_opt: float = 0.15,
    ) -> "Car":
        v = Vehicle()
        d = driver if driver is not None else CruisePursuitDriver()
        return cls(
            name=name,
            vehicle=v,
            driver=d,
            abs_controllers=[WheelABS(lambda_opt=abs_lambda_opt) for _ in range(4)],
            actuators=[BrakeActuator() for _ in range(4)],
            stability=StabilityController(),
            color=color,
        )


@dataclass
class Simulation:
    """Top-level simulation orchestrator.

    Construct with a track and one or more Cars; call step(dt_phys) repeatedly.
    """

    track: Track
    cars: List[Car] = field(default_factory=list)
    dt_phys: float = 0.001
    dt_ctrl: float = 0.005
    time: float = 0.0
    events: EventQueue = field(default_factory=EventQueue)
    telemetry: TelemetryLogger = field(default_factory=TelemetryLogger)
    telemetry_car_index: int = 0
    _ctrl_accum: float = 0.0
    _pending_inject: List[Callable] = field(default_factory=list)

    # --------------------------------------------------------------- #
    # Track driving helpers
    # --------------------------------------------------------------- #
    def _surface_under_car(self, car: Car) -> str:
        if car.surface_override:
            return car.surface_override
        s, _ = self.track.closest(car.vehicle.x, car.vehicle.y)
        return self.track.surface_at(s)

    def _mu_for_car(self, car: Car) -> Tuple[float, float, float, float]:
        """Return (mu_FL, mu_FR, mu_RL, mu_RR). For now, all wheels see the
        same track surface mu, times the car's mu_multiplier event override."""
        surface_name = self._surface_under_car(car)
        mu_base = SURFACES.get(surface_name, SURFACES["dry"]).mu
        mu = mu_base * car.mu_multiplier
        car.last_mu = (mu, mu, mu, mu)
        car.last_surface = surface_name
        return car.last_mu

    # --------------------------------------------------------------- #
    # Event injection
    # --------------------------------------------------------------- #
    def inject(self, event_fn: Callable[["Simulation"], None]) -> None:
        """Queue an event to run on the next step (thread-safe-ish)."""
        self._pending_inject.append(event_fn)

    def schedule(self, t: float, fn: Callable[["Simulation"], None], desc: str = "") -> None:
        self.events.schedule(t, fn, desc)

    # --------------------------------------------------------------- #
    # Controller tick: driver -> ABS -> stability -> actuator queue
    # --------------------------------------------------------------- #
    def _controller_tick(self, dt: float) -> None:
        for car in self.cars:
            mu = self._mu_for_car(car)

            ctx = DriverContext(
                x=car.vehicle.x, y=car.vehicle.y, psi=car.vehicle.psi,
                vx=car.vehicle.vx, vy=car.vehicle.vy, r=car.vehicle.r,
                speed=car.vehicle.speed, wheelbase=car.vehicle.chassis.wheelbase,
            )
            cmd = car.driver.update(ctx, self.track, dt)

            driver_brake = cmd.brake_demand
            if car.brake_override > 0.0 and self.time < car.brake_override_until:
                driver_brake = max(driver_brake, car.brake_override)
                cmd = DriverCommand(throttle=0.0, brake_demand=driver_brake, steer=cmd.steer)

            pre_abs = (driver_brake, driver_brake, driver_brake, driver_brake)

            kin = car.vehicle.wheel_kinematics()
            post_abs: List[float] = []
            abs_states: List[ABSState] = []
            for i, ctrl in enumerate(car.abs_controllers):
                c_i, st_i = ctrl.update(
                    driver_demand=driver_brake,
                    kappa=kin[i].kappa,
                    dt=dt,
                    enabled=car.abs_enabled,
                )
                post_abs.append(c_i)
                abs_states.append(st_i)

            post_stab, yaw_err, u_stab = car.stability.update(
                base_cmds=tuple(post_abs),  # type: ignore[arg-type]
                yaw_rate=car.vehicle.r,
                speed=car.vehicle.speed,
                steer=cmd.steer,
                wheelbase=car.vehicle.chassis.wheelbase,
                dt=dt,
            ) if car.stability_enabled else (tuple(post_abs), 0.0, 0.0)

            car.last_cmd = cmd
            car.last_pre_abs = pre_abs
            car.last_post_abs = tuple(post_abs)  # type: ignore[assignment]
            car.last_post_stab = tuple(post_stab)  # type: ignore[assignment]
            car.last_abs_states = tuple(abs_states)  # type: ignore[assignment]
            car.last_yaw_error = yaw_err
            car.last_stab_out = u_stab

    # --------------------------------------------------------------- #
    # Physics tick: actuators + vehicle integrator
    # --------------------------------------------------------------- #
    def _physics_tick(self, dt: float) -> None:
        for car in self.cars:
            ap = tuple(
                car.actuators[i].update(car.last_post_stab[i], dt) for i in range(4)
            )
            car.last_actuator_pressure = ap  # type: ignore[assignment]

            drive_torque = car.last_cmd.throttle * 4000.0
            inputs = VehicleInputs(
                steer=car.last_cmd.steer,
                drive_torque=drive_torque,
                brake_pressure=ap,
                mu=car.last_mu,
            )
            car.vehicle.step(dt, inputs)

    # --------------------------------------------------------------- #
    # Main step
    # --------------------------------------------------------------- #
    def step(self, dt_phys: Optional[float] = None) -> None:
        if dt_phys is None:
            dt_phys = self.dt_phys

        for ev_fn in self._pending_inject:
            ev_fn(self)
        self._pending_inject.clear()
        for scheduled in self.events.pop_due(self.time):
            scheduled.fn(self)

        self._ctrl_accum += dt_phys
        if self._ctrl_accum >= self.dt_ctrl - 1e-12:
            self._controller_tick(self.dt_ctrl)
            self._log_telemetry()
            self._ctrl_accum -= self.dt_ctrl

        self._physics_tick(dt_phys)
        self.time += dt_phys

    # --------------------------------------------------------------- #
    # Telemetry
    # --------------------------------------------------------------- #
    def _log_telemetry(self) -> None:
        if not self.cars:
            return
        idx = self.telemetry_car_index
        if idx < 0 or idx >= len(self.cars):
            return
        car = self.cars[idx]
        kin = car.vehicle.wheel_kinematics()
        row = {
            "t": self.time,
            "car": car.name,
            "x": car.vehicle.x, "y": car.vehicle.y, "psi": car.vehicle.psi,
            "vx": car.vehicle.vx, "vy": car.vehicle.vy,
            "speed": car.vehicle.speed, "r": car.vehicle.r,
            "ax": car.vehicle.ax_body(), "ay": car.vehicle.ay_body(),
            "throttle": car.last_cmd.throttle, "brake_demand": car.last_cmd.brake_demand,
            "steer": car.last_cmd.steer,
            "yaw_error": car.last_yaw_error, "stab_out": car.last_stab_out,
            "surface": car.last_surface, "abs_enabled": int(car.abs_enabled),
            "stab_enabled": int(car.stability_enabled),
        }
        for i, tag in enumerate(("FL", "FR", "RL", "RR")):
            row[f"kappa_{tag}"] = kin[i].kappa
            row[f"alpha_{tag}"] = kin[i].alpha
            row[f"Fz_{tag}"] = kin[i].Fz
            row[f"Fx_{tag}"] = kin[i].Fx_tire
            row[f"Fy_{tag}"] = kin[i].Fy_tire
            row[f"omega_{tag}"] = car.vehicle.wheel_speeds[i]
            row[f"brake_cmd_{tag}"] = car.last_post_stab[i]
            row[f"brake_actual_{tag}"] = car.last_actuator_pressure[i]
            row[f"mu_{tag}"] = car.last_mu[i]
            row[f"abs_state_{tag}"] = car.last_abs_states[i].value
        self.telemetry.log(row)
