"""Straight-line ABS vs no-ABS stopping distance comparison on a chosen surface.

Usage:
    python scripts/run_abs_compare.py
    python scripts/run_abs_compare.py --surface ice --v 22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abs_sim.drivers.policies import CruisePursuitDriver
from abs_sim.sim.events import force_brake
from abs_sim.sim.simulation import Car, Simulation
from abs_sim.sim.telemetry import TelemetryLogger
from abs_sim.track.presets import straight_road
from abs_sim.viz.reports import (
    plot_abs_vs_noabs,
    plot_stopping_distances,
    plot_stopping_overview,
    rows_to_df,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--surface", default="snow")
    p.add_argument("--v", type=float, default=20.0)
    p.add_argument("--max-time", type=float, default=15.0)
    p.add_argument("--out", default="runs/abs_compare")
    return p.parse_args()


def run_one(use_abs: bool, surface: str, v0: float, max_time: float):
    track = straight_road(length=800.0, surface=surface)
    driver = CruisePursuitDriver(v_cruise=v0)
    car = Car.make_default(name="car", driver=driver)
    car.vehicle.set_speed(v0)
    car.abs_enabled = use_abs
    sim = Simulation(track=track, cars=[car],
                     telemetry=TelemetryLogger(in_memory=True))
    sim.schedule(0.5, force_brake(1.0, duration=max_time), desc="slam")
    x0 = car.vehicle.x
    while sim.time < max_time and car.vehicle.vx > 0.1:
        sim.step()
    distance = car.vehicle.x - x0
    return rows_to_df(sim.telemetry.rows()), distance


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df_on, d_on = run_one(True, args.surface, args.v, args.max_time)
    df_off, d_off = run_one(False, args.surface, args.v, args.max_time)

    df_on.to_csv(out / f"abs_on_{args.surface}.csv", index=False)
    df_off.to_csv(out / f"abs_off_{args.surface}.csv", index=False)

    plot_stopping_overview(df_on, out / f"abs_on_{args.surface}.png",
                           title=f"ABS ON ({args.surface}, v0={args.v})")
    plot_stopping_overview(df_off, out / f"abs_off_{args.surface}.png",
                           title=f"ABS OFF ({args.surface}, v0={args.v})")
    plot_abs_vs_noabs(df_on, df_off, out / f"compare_{args.surface}.png",
                      title=f"Stopping distance ({args.surface}, v0={args.v})")

    savings = (d_off - d_on) / d_off * 100.0 if d_off > 0 else 0.0
    print(f"Surface: {args.surface}, v0: {args.v} m/s")
    print(f"  ABS off: stopped in {d_off:.2f} m")
    print(f"  ABS on : stopped in {d_on:.2f} m")
    print(f"  Savings: {savings:.1f}%")

    plot_stopping_distances(
        {"abs_off": d_off, "abs_on": d_on},
        out / f"bar_{args.surface}.png",
        title=f"Stopping distance ({args.surface}, v0={args.v})",
    )


if __name__ == "__main__":
    main()
