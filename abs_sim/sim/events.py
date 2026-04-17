"""Event system: scheduled + live-injected events that mutate sim state.

An event is a callable taking the live Simulation object and returning None.
Scheduled events have a fire time; live events are dispatched immediately via
Simulation.inject(event_fn). Events can install cleanup callbacks to expire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Callable, List, Optional


@dataclass(order=True)
class ScheduledEvent:
    time: float
    seq: int = field(compare=True)
    fn: Callable[["object"], None] = field(compare=False)
    description: str = field(default="", compare=False)


class EventQueue:
    """Time-ordered queue of scheduled events."""

    def __init__(self) -> None:
        self._heap: List[ScheduledEvent] = []
        self._seq = 0

    def schedule(self, t: float, fn: Callable[["object"], None], desc: str = "") -> None:
        self._seq += 1
        heapq.heappush(self._heap, ScheduledEvent(t, self._seq, fn, desc))

    def pop_due(self, t_now: float) -> List[ScheduledEvent]:
        due: List[ScheduledEvent] = []
        while self._heap and self._heap[0].time <= t_now:
            due.append(heapq.heappop(self._heap))
        return due

    def __len__(self) -> int:
        return len(self._heap)


# --------------------------------------------------------------------------- #
# Canned event factories
# --------------------------------------------------------------------------- #

def set_global_mu(mu: float, duration: Optional[float] = None, car_index: int = 0):
    """Set the global friction multiplier for a car (optionally for `duration`
    seconds, after which it reverts)."""
    def apply(sim) -> None:
        car = sim.cars[car_index]
        prev = car.mu_multiplier
        car.mu_multiplier = mu
        if duration is not None:
            def revert(s2) -> None:
                sim.cars[car_index].mu_multiplier = prev
            sim.events.schedule(sim.time + duration, revert, desc="revert_mu")
    return apply


def force_brake(driver_demand: float, duration: float, car_index: int = 0):
    """Force this car's effective driver brake demand for `duration` seconds."""
    def apply(sim) -> None:
        car = sim.cars[car_index]
        car.brake_override = driver_demand
        car.brake_override_until = sim.time + duration
    return apply


def set_surface_override(surface_name: str, duration: Optional[float] = None, car_index: int = 0):
    """Override the surface under the car regardless of the track's actual
    surface. Useful for quick manual testing of ice/wet patches."""
    def apply(sim) -> None:
        car = sim.cars[car_index]
        prev = car.surface_override
        car.surface_override = surface_name
        if duration is not None:
            def revert(s2) -> None:
                sim.cars[car_index].surface_override = prev
            sim.events.schedule(sim.time + duration, revert, desc="revert_surface")
    return apply


def toggle_abs(car_index: int = 0):
    def apply(sim) -> None:
        sim.cars[car_index].abs_enabled = not sim.cars[car_index].abs_enabled
    return apply


def toggle_stability(car_index: int = 0):
    def apply(sim) -> None:
        c = sim.cars[car_index]
        c.stability_enabled = not c.stability_enabled
    return apply
