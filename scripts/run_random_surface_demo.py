"""Headless random-surface straight-line demo: random surface patches plus
random full-stop brake events. Compares ABS on vs ABS off.

Usage:
    python scripts/run_random_surface_demo.py
    python scripts/run_random_surface_demo.py --seed 12 --cruise 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from abs_sim.drivers.policies import RandomBrakeEventDriver
from abs_sim.sim.simulation import Car, Simulation
from abs_sim.sim.telemetry import TelemetryLogger
from abs_sim.track.presets import random_surface_straight
from abs_sim.viz.reports import (
    plot_abs_vs_noabs,
    plot_stopping_distances,
    plot_stopping_overview,
    rows_to_df,
)


def run_one(use_abs: bool, seed: int, cruise: float, max_time: float = 30.0):
    track = random_surface_straight(length=500.0, seed=seed)
    driver = RandomBrakeEventDriver(
        v_cruise=cruise, rng_seed=seed, min_gap=80.0, max_gap=150.0, hold_duration=1.5,
    )
    car = Car.make_default(name="car", driver=driver)
    car.vehicle.set_speed(cruise)
    car.abs_enabled = use_abs
    sim = Simulation(track=track, cars=[car],
                     telemetry=TelemetryLogger(in_memory=True))
    while sim.time < max_time:
        sim.step()
    return rows_to_df(sim.telemetry.rows())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cruise", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/random_surface")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df_on = run_one(use_abs=True, seed=args.seed, cruise=args.cruise)
    df_off = run_one(use_abs=False, seed=args.seed, cruise=args.cruise)

    df_on.to_csv(out / "abs_on.csv", index=False)
    df_off.to_csv(out / "abs_off.csv", index=False)

    plot_stopping_overview(df_on, out / "abs_on_overview.png", title="ABS ON")
    plot_stopping_overview(df_off, out / "abs_off_overview.png", title="ABS OFF")
    plot_abs_vs_noabs(df_on, df_off, out / "abs_vs_noabs.png",
                      title="Random surface: ABS on vs off")

    # Simple 'distance covered during braking' heuristic: max x reached
    distances = {
        "abs_off": float(df_off["x"].iloc[-1]),
        "abs_on": float(df_on["x"].iloc[-1]),
    }
    plot_stopping_distances(distances, out / "distance_bar.png",
                            title="Total distance covered (30 s run)")
    print(f"Done. Outputs in {out.resolve()}.")


if __name__ == "__main__":
    main()
