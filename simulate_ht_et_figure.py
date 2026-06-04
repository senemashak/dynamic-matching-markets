# -*- coding: utf-8 -*-
"""Create a simulation figure for H_t and E_t dynamics.

Outputs:
  - simulation_outputs/ht_et_trajectories_*.png
  - simulation_outputs/ht_et_trajectories_*.csv

The PNG contains:
  1) H_t mean +/- std over time (all policies)
  2) E_t mean +/- std over time (all policies)
  3) Phase plot of (H_t, E_t) mean paths
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from simulate_pool_trajectories import (
    POLICIES,
    POLICY_COLORS,
    POLICY_LABELS,
    simulate_pool_path,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


def stem_name(m: int, d1: float, d2: float, horizon: float, n_runs: int) -> str:
    return f"ht_et_trajectories_m{m}_d1_{d1:g}_d2_{d2:g}_t{horizon:g}_runs{n_runs}"


def run_simulations(
    m: int,
    d1: float,
    d2: float,
    horizon: float,
    dt: float,
    n_runs: int,
    base_seed: int,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    times = np.arange(0.0, horizon + dt / 2.0, dt)
    out: dict[str, dict[str, np.ndarray]] = {}

    for p_idx, policy in enumerate(POLICIES):
        h_runs = []
        e_runs = []
        for run_idx in range(n_runs):
            seed = base_seed + 1000 * p_idx + run_idx
            h_path, e_path = simulate_pool_path(m, d1, d2, policy, times, seed)
            h_runs.append(h_path)
            e_runs.append(e_path)

        h_arr = np.vstack(h_runs)
        e_arr = np.vstack(e_runs)
        out[policy] = {
            "h_mean": h_arr.mean(axis=0),
            "h_std": h_arr.std(axis=0, ddof=1),
            "e_mean": e_arr.mean(axis=0),
            "e_std": e_arr.std(axis=0, ddof=1),
        }

    return times, out


def write_csv(path: Path, times: np.ndarray, stats: dict[str, dict[str, np.ndarray]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "time", "H_mean", "H_std", "E_mean", "E_std"])
        for policy in POLICIES:
            s = stats[policy]
            for i, t in enumerate(times):
                writer.writerow(
                    [
                        policy,
                        f"{t:.4f}",
                        f"{s['h_mean'][i]:.8f}",
                        f"{s['h_std'][i]:.8f}",
                        f"{s['e_mean'][i]:.8f}",
                        f"{s['e_std'][i]:.8f}",
                    ]
                )


def write_png(path: Path, times: np.ndarray, stats: dict[str, dict[str, np.ndarray]], title: str) -> None:
    if not HAS_MPL:
        print("matplotlib not installed; skipping PNG.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax_h = axes[0, 0]
    ax_e = axes[0, 1]
    ax_phase = axes[1, 0]
    ax_legend = axes[1, 1]
    ax_legend.axis("off")

    legend_handles = []
    legend_labels = []

    for policy in POLICIES:
        color = POLICY_COLORS[policy]
        label = POLICY_LABELS[policy]
        s = stats[policy]

        h_mean = s["h_mean"]
        h_std = s["h_std"]
        e_mean = s["e_mean"]
        e_std = s["e_std"]

        line_h = ax_h.plot(times, h_mean, color=color, linewidth=2.1)[0]
        ax_h.fill_between(times, np.maximum(0.0, h_mean - h_std), h_mean + h_std, color=color, alpha=0.18)

        ax_e.plot(times, e_mean, color=color, linewidth=2.1)
        ax_e.fill_between(times, np.maximum(0.0, e_mean - e_std), e_mean + e_std, color=color, alpha=0.18)

        ax_phase.plot(h_mean, e_mean, color=color, linewidth=2.2)
        ax_phase.scatter(h_mean[0], e_mean[0], color=color, s=25, marker="o")
        ax_phase.scatter(h_mean[-1], e_mean[-1], color=color, s=30, marker="x")

        legend_handles.append(line_h)
        legend_labels.append(label)

    ax_h.set_title(r"$H_t$ pool trajectory")
    ax_h.set_xlabel("time")
    ax_h.set_ylabel(r"$H_t$")
    ax_h.grid(alpha=0.25)

    ax_e.set_title(r"$E_t$ pool trajectory")
    ax_e.set_xlabel("time")
    ax_e.set_ylabel(r"$E_t$")
    ax_e.grid(alpha=0.25)

    ax_phase.set_title(r"Phase path: $(H_t, E_t)$")
    ax_phase.set_xlabel(r"$H_t$")
    ax_phase.set_ylabel(r"$E_t$")
    ax_phase.grid(alpha=0.25)

    ax_legend.legend(legend_handles, legend_labels, loc="center", frameon=False, fontsize=11, ncol=1)
    ax_legend.text(
        0.02,
        0.15,
        "Phase plot markers:\n"
        "  o = start (t=0)\n"
        "  x = end (t=T)\n"
        "Bands are mean +/- 1 std.",
        fontsize=10,
        va="bottom",
    )

    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate and plot H_t and E_t trajectories.")
    parser.add_argument("--m", type=int, default=5000)
    parser.add_argument("--d1", type=float, default=10.0)
    parser.add_argument("--d2", type=float, default=1000.0)
    parser.add_argument("--horizon", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out-dir", type=Path, default=Path("simulation_outputs"))
    args = parser.parse_args()

    if args.n_runs < 2:
        raise ValueError("--n-runs must be >= 2")
    if args.dt <= 0:
        raise ValueError("--dt must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_name(args.m, args.d1, args.d2, args.horizon, args.n_runs)
    times, stats = run_simulations(args.m, args.d1, args.d2, args.horizon, args.dt, args.n_runs, args.seed)

    csv_path = args.out_dir / f"{stem}.csv"
    png_path = args.out_dir / f"{stem}.png"
    write_csv(csv_path, times, stats)

    title = (
        rf"$H_t, E_t$ dynamics  (m={args.m}, d_1={args.d1:g}, d_2={args.d2:g}, "
        rf"T={args.horizon:g}, runs={args.n_runs})"
    )
    write_png(png_path, times, stats, title)

    print("Wrote:")
    print(f"  {csv_path.name}")
    if HAS_MPL:
        print(f"  {png_path.name}")


if __name__ == "__main__":
    main()
