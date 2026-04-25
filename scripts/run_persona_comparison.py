"""Headless persona-archetype comparison.

Runs all four driver personas (Pro / Cautious / Novice / Aggressive) on
the same track and dumps per-persona telemetry plus a side-by-side
comparison plot. This is the offline counterpart to the interactive 'P'
key cycle in ``scripts/run_interactive.py``.

Usage::

    python scripts/run_persona_comparison.py
    python scripts/run_persona_comparison.py --track split_mu_curves --max-time 60
    python scripts/run_persona_comparison.py --only pro,aggressive

Outputs land in ``runs/personas/`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from abs_sim.drivers.policies import PERSONA_COLORS, PERSONAS
from abs_sim.sim.simulation import Car, Simulation
from abs_sim.sim.telemetry import TelemetryLogger
from abs_sim.track.presets import PRESETS
from abs_sim.viz.reports import (
    plot_persona_comparison,
    plot_stopping_overview,
    rows_to_df,
)


PERSONA_ORDER = ["pro", "cautious", "novice", "aggressive"]


def run_one(
    persona_name: str,
    track_name: str,
    max_time: float,
    v_cruise_override: Optional[float],
) -> pd.DataFrame:
    """Run a single persona on a fresh copy of the track; return telemetry df.

    ``v_cruise_override`` lets the CLI force every persona onto the same
    cruise speed (useful when we want to isolate timing differences from
    pace differences), otherwise each persona uses its own default.
    """
    track = PRESETS[track_name]()
    # Honor a preset-specific cruise hint (split-mu presets suggest a lower
    # speed) only when the caller didn't explicitly override it.
    default_v = (
        v_cruise_override
        if v_cruise_override is not None
        else track.recommended_v_cruise
    )
    driver = PERSONAS[persona_name](v_cruise=default_v)
    car = Car.make_default(
        name=persona_name, driver=driver, color=PERSONA_COLORS[persona_name],
    )
    car.vehicle.set_pose(0.0, 0.0, 0.0)
    car.vehicle.set_speed(driver.v_cruise)
    sim = Simulation(
        track=track, cars=[car], telemetry=TelemetryLogger(in_memory=True),
    )
    # Stop early if the car has already finished the track; otherwise keep
    # stepping until we hit max_time so slower personas still get a full
    # dataset for the trajectory plot.
    last_s = 0.0
    while sim.time < max_time:
        sim.step()
        s, _ = track.closest(car.vehicle.x, car.vehicle.y, s_hint=last_s)
        last_s = s
        if s > track.total_length - 5.0:
            break
    return rows_to_df(sim.telemetry.rows())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", default="f1_like",
                   choices=sorted(PRESETS.keys()),
                   help="Preset track to run the personas on.")
    p.add_argument("--max-time", type=float, default=90.0,
                   help="Simulation seconds per persona (upper bound).")
    p.add_argument("--v-cruise", type=float, default=None,
                   help="Override every persona's cruise speed (m/s). "
                        "If omitted, each persona uses its own default.")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of personas to run "
                        f"(default: all of {','.join(PERSONA_ORDER)}).")
    p.add_argument("--out", default="runs/personas",
                   help="Output directory for CSVs and plots.")
    return p.parse_args()


def _selected_personas(only: Optional[str]) -> List[str]:
    if not only:
        return list(PERSONA_ORDER)
    names = [n.strip().lower() for n in only.split(",") if n.strip()]
    unknown = [n for n in names if n not in PERSONAS]
    if unknown:
        raise SystemExit(
            f"Unknown persona(s): {unknown}. Known: {list(PERSONAS.keys())}"
        )
    return names


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    personas = _selected_personas(args.only)

    dfs: Dict[str, pd.DataFrame] = {}
    summary_rows: List[dict] = []
    for name in personas:
        print(f"[{name}] running on {args.track} ...")
        df = run_one(
            persona_name=name,
            track_name=args.track,
            max_time=args.max_time,
            v_cruise_override=args.v_cruise,
        )
        csv_path = out / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        dfs[name] = df
        plot_stopping_overview(df, out / f"{name}.png", title=f"{name} on {args.track}")

        # Quick at-a-glance summary so the user can read ordering from stdout
        # without having to open the plot.
        final_t = float(df["t"].iloc[-1])
        final_v = float(df["speed"].iloc[-1])
        mean_v = float(df["speed"].mean())
        summary_rows.append({
            "persona": name,
            "final_t": final_t,
            "final_speed": final_v,
            "mean_speed": mean_v,
        })
        print(f"  -> t_final={final_t:.2f}s, v_final={final_v:.2f} m/s, "
              f"v_mean={mean_v:.2f} m/s")

    plot_persona_comparison(
        dfs,
        out / f"comparison_{args.track}.png",
        title=f"Driver personas on {args.track}",
        colors=PERSONA_COLORS,
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out / f"summary_{args.track}.csv", index=False)
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nOutputs in {out.resolve()}.")


if __name__ == "__main__":
    main()
