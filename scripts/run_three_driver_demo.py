"""Headless 3-driver curve-braking comparison.

Three drivers hit the SAME corner at the SAME cruise speed but start braking
at different times (early, on-time, late). Runs one simulation per driver,
dumps telemetry, and produces a matplotlib comparison figure.

Usage:
    python scripts/run_three_driver_demo.py
    python scripts/run_three_driver_demo.py --cruise 40 --corner 14 --abs off
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from abs_sim.drivers.policies import CurveBrakeDelayDriver
from abs_sim.sim.simulation import Car, Simulation
from abs_sim.sim.telemetry import TelemetryLogger
from abs_sim.track.presets import curve_braking_scenario
from abs_sim.viz.reports import (
    plot_stopping_overview,
    plot_three_driver_comparison,
    rows_to_df,
)


SCENARIOS = [
    ("early", -0.8),
    ("ontime", 0.0),
    ("late", +1.0),
]


def run_one(
    name: str,
    delay_s: float,
    cruise: float,
    corner: float,
    use_abs: bool,
    max_time: float = 25.0,
    surface: str = "dry",
) -> pd.DataFrame:
    track = curve_braking_scenario(surface=surface)
    driver = CurveBrakeDelayDriver(
        v_cruise=cruise, v_corner=corner, delay_s=delay_s,
    )
    car = Car.make_default(name=name, driver=driver)
    car.vehicle.set_speed(cruise)
    car.abs_enabled = use_abs
    sim = Simulation(track=track, cars=[car],
                     telemetry=TelemetryLogger(in_memory=True))
    while sim.time < max_time:
        sim.step()
    return rows_to_df(sim.telemetry.rows())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cruise", type=float, default=40.0)
    p.add_argument("--corner", type=float, default=14.0)
    p.add_argument("--surface", default="wet")
    p.add_argument("--abs", choices=["on", "off", "both"], default="on")
    p.add_argument("--out", default="runs/three_driver")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for abs_mode in (["on"] if args.abs == "on" else ["off"] if args.abs == "off" else ["on", "off"]):
        dfs: Dict[str, pd.DataFrame] = {}
        for name, delay in SCENARIOS:
            df = run_one(name=name, delay_s=delay, cruise=args.cruise,
                         corner=args.corner, use_abs=abs_mode == "on",
                         surface=args.surface)
            csv_path = out / f"{name}_abs_{abs_mode}.csv"
            df.to_csv(csv_path, index=False)
            dfs[name] = df
            plot_stopping_overview(df, out / f"{name}_abs_{abs_mode}.png",
                                   title=f"{name} (ABS {abs_mode})")
            print(f"  {name} abs={abs_mode}: final v={df['speed'].iloc[-1]:.2f} m/s, "
                  f"x={df['x'].iloc[-1]:.1f} m")
        plot_three_driver_comparison(
            dfs, out / f"comparison_abs_{abs_mode}.png",
            title=f"3-driver comparison (ABS {abs_mode}, surface={args.surface})",
        )
    print(f"Done. Outputs in {out.resolve()}.")


if __name__ == "__main__":
    main()
