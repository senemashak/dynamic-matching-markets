# -*- coding: utf-8 -*-
"""Exact CTMC perishing simulations for T=100 with five runs.

Outputs a combined total/H/E perishing plot. All plotted quantities are
normalized by the same fixed denominator m*T.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import plot_all_perishing_combined as plotter
from simulate_perishing_loss_comparison import simulate_loss_path, summarize_paths
from simulate_pool_trajectories import POLICIES, POLICY_LABELS


M = 10000
D1 = 10.0
D2_VALUES = (10.0, 100.0, 1000.0, 10000.0)
T_HORIZON = 100.0
DT = 1.0
N_RUNS = 5
BASE_SEED = 20260515
STEM = "all_perishing_combined_m10000_d1_10_d2_10_100_1000_10000_t100_runs5"


def run_policy_paths(d2: float, policy: str, times: np.ndarray, policy_index: int):
    paths = []
    experiment_seed = BASE_SEED + int(d2)
    for run_idx in range(N_RUNS):
        seed = experiment_seed + 1000 * policy_index + run_idx
        paths.append(simulate_loss_path(M, D1, d2, policy, times, seed))
    return summarize_paths(paths)


def main() -> None:
    out_dir = Path("simulation_outputs")
    out_dir.mkdir(exist_ok=True)
    times = np.arange(0.0, T_HORIZON + DT / 2.0, DT)
    scale = times / T_HORIZON

    series = {}
    for d2 in D2_VALUES:
        print(f"Running exact CTMC: m={M}, d1={D1:g}, d2={d2:g}, T={T_HORIZON:g}, runs={N_RUNS}")
        series[d2] = {}
        for policy_index, policy in enumerate(POLICIES):
            summary = run_policy_paths(d2, policy, times, policy_index)
            series[d2][policy] = plotter.PerishingSeries(
                times=times,
                total_mean=summary.total_mean * scale,
                total_std=summary.total_std * scale,
                h_mean=summary.h_mean * scale,
                h_std=summary.h_std * scale,
                e_mean=summary.e_mean * scale,
                e_std=summary.e_std * scale,
            )

    plotter.M = M
    plotter.D1 = D1
    plotter.T_HORIZON = T_HORIZON
    plotter.N_RUNS = N_RUNS
    plotter.D2_VALUES = D2_VALUES
    plotter.STEM = STEM

    plotter.write_combined_csv(out_dir / f"{STEM}.csv", series)
    plotter.write_svg(out_dir / f"{STEM}.svg", series)
    plotter.write_png(out_dir / f"{STEM}.png", series)

    print("\nFinal combined perishing, normalized by m*T:")
    for d2 in D2_VALUES:
        print(f"  d2={d2:g}")
        for policy in POLICIES:
            s = series[d2][policy]
            print(
                f"    {POLICY_LABELS[policy]:>10}: "
                f"total={s.total_mean[-1]:.4f} +/- {s.total_std[-1]:.4f}, "
                f"H={s.h_mean[-1]:.4f} +/- {s.h_std[-1]:.4f}, "
                f"E={s.e_mean[-1]:.4f} +/- {s.e_std[-1]:.4f}"
            )


if __name__ == "__main__":
    main()
