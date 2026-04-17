"""matplotlib post-run reports from telemetry rows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # headless by default; override externally if desired
import matplotlib.pyplot as plt
import pandas as pd


def rows_to_df(rows: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Single-run plots
# --------------------------------------------------------------------------- #

def plot_stopping_overview(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    """Speed, slip, brake pressure, and yaw rate vs time for one run."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    t = df["t"].values

    axes[0].plot(t, df["speed"], label="speed (m/s)", color="C0")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    for tag, c in zip(("FL", "FR", "RL", "RR"), ("C1", "C2", "C3", "C4")):
        axes[1].plot(t, df[f"kappa_{tag}"], label=tag, color=c, linewidth=1)
    axes[1].axhline(-0.15, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
                    label="lambda_opt")
    axes[1].set_ylabel("slip ratio")
    axes[1].set_ylim(-1.05, 0.6)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", ncol=5, fontsize=8)

    for tag, c in zip(("FL", "FR", "RL", "RR"), ("C1", "C2", "C3", "C4")):
        axes[2].plot(t, df[f"brake_actual_{tag}"], label=tag, color=c, linewidth=1)
    axes[2].set_ylabel("brake pressure")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", ncol=4, fontsize=8)

    axes[3].plot(t, df["r"], label="yaw rate (rad/s)", color="C5")
    axes[3].plot(t, df["yaw_error"], label="yaw error", color="C6", alpha=0.6)
    axes[3].set_ylabel("yaw (rad/s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_abs_vs_noabs(
    df_on: pd.DataFrame,
    df_off: pd.DataFrame,
    out_path: Path,
    title: str = "ABS on vs off",
) -> None:
    """Compare two runs on the same x/speed axes."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(df_off["t"], df_off["speed"], label="ABS OFF", color="C3")
    axes[0].plot(df_on["t"], df_on["speed"], label="ABS ON", color="C0")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df_off["t"], df_off["kappa_FL"], label="FL OFF", color="C3", alpha=0.7)
    axes[1].plot(df_on["t"], df_on["kappa_FL"], label="FL ON", color="C0", alpha=0.7)
    axes[1].plot(df_off["t"], df_off["kappa_RL"], label="RL OFF", color="C1", alpha=0.7)
    axes[1].plot(df_on["t"], df_on["kappa_RL"], label="RL ON", color="C9", alpha=0.7)
    axes[1].axhline(-0.15, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("slip ratio")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Multi-car comparison
# --------------------------------------------------------------------------- #

def plot_three_driver_comparison(
    dfs: Dict[str, pd.DataFrame],
    out_path: Path,
    corner_s: Optional[float] = None,
    title: str = "3-driver curve braking",
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = {"early": "C2", "ontime": "C0", "late": "C3"}
    for name, df in dfs.items():
        c = colors.get(name, None)
        axes[0].plot(df["t"], df["speed"], label=name, color=c)
        axes[1].plot(df["t"], df["brake_demand"], label=f"{name} demand",
                     color=c, linewidth=1)
        axes[1].plot(df["t"], df[[f"brake_actual_{t}" for t in ("FL","FR","RL","RR")]]
                     .mean(axis=1), label=f"{name} actual avg", color=c, linestyle="--",
                     linewidth=1)
        axes[2].plot(df["x"], df["y"], label=name, color=c)

    axes[0].set_ylabel("speed (m/s)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_ylabel("brake")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_stopping_distances(
    distances: Dict[str, float],
    out_path: Path,
    title: str = "Stopping distance",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(distances)
    vals = [distances[n] for n in names]
    bars = ax.bar(names, vals, color=["C3" if "off" in n else "C0" for n in names])
    ax.set_ylabel("stop distance (m)")
    ax.set_title(title)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                ha="center", va="bottom")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
