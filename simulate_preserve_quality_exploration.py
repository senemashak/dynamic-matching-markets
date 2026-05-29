# -*- coding: utf-8 -*-
"""Explore preserve_quality policy under key parameter sweeps.

This script focuses on the `preserve_quality` mechanism only and generates:
  1) Performance over (rho, lambda_ent) with fixed (d1, d2)
  2) Performance over (d1, d2) with fixed (rho, lambda_ent)

Outputs are written to simulation_outputs/:
  - two CSVs
  - two heatmap PNGs (if matplotlib installed)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from simulate_four_mechanism_policies import (
    LossResult,
    SummaryPoint,
    _simulate_preserve_quality,
    summarize,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


def parse_float_grid(text: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("Grid specification is empty.")
    return values


def run_cell(
    m: int,
    d1: float,
    d2: float,
    lambda_ent: float,
    rho: float,
    horizon: float,
    n_runs: int,
    seed_base: int,
) -> SummaryPoint:
    losses: list[LossResult] = []
    for run_idx in range(n_runs):
        losses.append(
            _simulate_preserve_quality(
                m=m,
                d1=d1,
                d2=d2,
                lambda_ent=lambda_ent,
                rho=rho,
                horizon=horizon,
                seed=seed_base + run_idx,
                share_quality_g=0.35,
                quality_boost=1.25,
                allow_g_fallback=False,
            )
        )
    return summarize(losses, w_h=1.0, w_e=1.0)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def matrix_from_summary(
    row_grid: tuple[float, ...],
    col_grid: tuple[float, ...],
    summary: dict[tuple[float, float], SummaryPoint],
    metric: str,
) -> np.ndarray:
    z = np.zeros((len(row_grid), len(col_grid)), dtype=float)
    for i, r in enumerate(row_grid):
        for j, c in enumerate(col_grid):
            z[i, j] = float(getattr(summary[(r, c)], metric))
    return z


def heatmap_four_metrics(
    out_path: Path,
    row_grid: tuple[float, ...],
    col_grid: tuple[float, ...],
    summary: dict[tuple[float, float], SummaryPoint],
    row_label: str,
    col_label: str,
    row_tick_fmt: str,
    col_tick_fmt: str,
    title: str,
) -> None:
    if not HAS_MPL:
        print(f"matplotlib unavailable; skipping {out_path.name}")
        return

    metrics = (
        ("objective_mean", "Objective loss"),
        ("total_mean", "Total perished/(mT)"),
        ("h_mean", "H perished/(mT)"),
        ("e_mean", "E perished/(mT)"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes_flat = list(axes.flat)
    row_ticks = np.arange(len(row_grid))
    col_ticks = np.arange(len(col_grid))
    row_tick_labels = [format(x, row_tick_fmt) for x in row_grid]
    col_tick_labels = [format(x, col_tick_fmt) for x in col_grid]

    for ax, (metric, subtitle) in zip(axes_flat, metrics):
        z = matrix_from_summary(row_grid, col_grid, summary, metric)
        im = ax.imshow(z, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(subtitle)
        ax.set_xticks(col_ticks)
        ax.set_xticklabels(col_tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(row_ticks)
        ax.set_yticklabels(row_tick_labels, fontsize=8)
        ax.set_xlabel(col_label)
        ax.set_ylabel(row_label)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore preserve_quality policy over two parameter grids.")
    parser.add_argument("--m", type=int, default=5000)
    parser.add_argument("--horizon", type=float, default=80.0)
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--out-dir", type=Path, default=Path("simulation_outputs"))

    # Sweep A: rho x lambda_ent with fixed d1,d2.
    parser.add_argument("--fixed-d1", type=float, default=10.0)
    parser.add_argument("--fixed-d2", type=float, default=1000.0)
    parser.add_argument(
        "--rho-grid",
        type=str,
        default="1,2,5,10,20,40,100",
    )
    parser.add_argument(
        "--lambda-ent-grid",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    )

    # Sweep B: d1 x d2 with fixed rho, lambda_ent.
    parser.add_argument("--fixed-rho", type=float, default=10.0)
    parser.add_argument("--fixed-lambda-ent", type=float, default=0.5)
    # Backward-compatible aliases.
    parser.add_argument("--lambda-e-grid", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fixed-lambda-e", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--d1-grid",
        type=str,
        default="2,5,10,20,40",
    )
    parser.add_argument(
        "--d2-grid",
        type=str,
        default="100,300,1000,3000,10000",
    )
    args = parser.parse_args()

    if args.n_runs < 2:
        raise ValueError("--n-runs must be >= 2")

    rho_grid = parse_float_grid(args.lambda_e_grid) if args.lambda_e_grid else parse_float_grid(args.rho_grid)
    lambda_ent_grid = parse_float_grid(args.lambda_ent_grid)
    fixed_rho = args.fixed_lambda_e if args.fixed_lambda_e is not None else args.fixed_rho
    d1_grid = parse_float_grid(args.d1_grid)
    d2_grid = parse_float_grid(args.d2_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stem_a = (
        f"preserve_quality_lambda_sweep_m{args.m}_d1_{args.fixed_d1:g}_d2_{args.fixed_d2:g}_"
        f"t{args.horizon:g}_runs{args.n_runs}"
    )
    stem_b = (
        f"preserve_quality_d_sweep_m{args.m}_rho_{fixed_rho:g}_"
        f"lambdaEnt_{args.fixed_lambda_ent:g}_t{args.horizon:g}_runs{args.n_runs}"
    )

    # Sweep A
    summary_a: dict[tuple[float, float], SummaryPoint] = {}
    rows_a: list[dict[str, object]] = []
    for i, rho in enumerate(rho_grid):
        for j, lam_ent in enumerate(lambda_ent_grid):
            seed_base = args.seed + 10000 * i + 500 * j
            point = run_cell(
                m=args.m,
                d1=args.fixed_d1,
                d2=args.fixed_d2,
                lambda_ent=lam_ent,
                rho=rho,
                horizon=args.horizon,
                n_runs=args.n_runs,
                seed_base=seed_base,
            )
            summary_a[(rho, lam_ent)] = point
            rows_a.append(
                {
                    "m": args.m,
                    "horizon": f"{args.horizon:g}",
                    "n_runs": args.n_runs,
                    "d1": f"{args.fixed_d1:g}",
                    "d2": f"{args.fixed_d2:g}",
                    "rho": f"{rho:g}",
                    "lambda_ent": f"{lam_ent:g}",
                    "objective_mean": f"{point.objective_mean:.8f}",
                    "objective_std": f"{point.objective_std:.8f}",
                    "total_mean": f"{point.total_mean:.8f}",
                    "total_std": f"{point.total_std:.8f}",
                    "H_mean": f"{point.h_mean:.8f}",
                    "H_std": f"{point.h_std:.8f}",
                    "E_mean": f"{point.e_mean:.8f}",
                    "E_std": f"{point.e_std:.8f}",
                }
            )

    write_csv(
        args.out_dir / f"{stem_a}.csv",
        rows_a,
        [
            "m",
            "horizon",
            "n_runs",
            "d1",
            "d2",
            "rho",
            "lambda_ent",
            "objective_mean",
            "objective_std",
            "total_mean",
            "total_std",
            "H_mean",
            "H_std",
            "E_mean",
            "E_std",
        ],
    )
    heatmap_four_metrics(
        out_path=args.out_dir / f"{stem_a}.png",
        row_grid=rho_grid,
        col_grid=lambda_ent_grid,
        summary=summary_a,
        row_label="rho (E criticality rate)",
        col_label="lambda_ent",
        row_tick_fmt="g",
        col_tick_fmt=".2g",
        title="Preserve-quality policy: sensitivity to E departure and E entry composition",
    )

    # Sweep B
    summary_b: dict[tuple[float, float], SummaryPoint] = {}
    rows_b: list[dict[str, object]] = []
    for i, d1 in enumerate(d1_grid):
        for j, d2 in enumerate(d2_grid):
            seed_base = args.seed + 200000 + 10000 * i + 500 * j
            point = run_cell(
                m=args.m,
                d1=d1,
                d2=d2,
                lambda_ent=args.fixed_lambda_ent,
                rho=fixed_rho,
                horizon=args.horizon,
                n_runs=args.n_runs,
                seed_base=seed_base,
            )
            summary_b[(d1, d2)] = point
            rows_b.append(
                {
                    "m": args.m,
                    "horizon": f"{args.horizon:g}",
                    "n_runs": args.n_runs,
                    "rho": f"{fixed_rho:g}",
                    "lambda_ent": f"{args.fixed_lambda_ent:g}",
                    "d1": f"{d1:g}",
                    "d2": f"{d2:g}",
                    "objective_mean": f"{point.objective_mean:.8f}",
                    "objective_std": f"{point.objective_std:.8f}",
                    "total_mean": f"{point.total_mean:.8f}",
                    "total_std": f"{point.total_std:.8f}",
                    "H_mean": f"{point.h_mean:.8f}",
                    "H_std": f"{point.h_std:.8f}",
                    "E_mean": f"{point.e_mean:.8f}",
                    "E_std": f"{point.e_std:.8f}",
                }
            )

    write_csv(
        args.out_dir / f"{stem_b}.csv",
        rows_b,
        [
            "m",
            "horizon",
            "n_runs",
            "rho",
            "lambda_ent",
            "d1",
            "d2",
            "objective_mean",
            "objective_std",
            "total_mean",
            "total_std",
            "H_mean",
            "H_std",
            "E_mean",
            "E_std",
        ],
    )
    heatmap_four_metrics(
        out_path=args.out_dir / f"{stem_b}.png",
        row_grid=d1_grid,
        col_grid=d2_grid,
        summary=summary_b,
        row_label="d1 (E-H link scale)",
        col_label="d2 (E-E link scale)",
        row_tick_fmt="g",
        col_tick_fmt="g",
        title="Preserve-quality policy: sensitivity to compatibility parameters d1 and d2",
    )

    print("Wrote:")
    print(f"  {stem_a}.csv")
    if HAS_MPL:
        print(f"  {stem_a}.png")
    print(f"  {stem_b}.csv")
    if HAS_MPL:
        print(f"  {stem_b}.png")


if __name__ == "__main__":
    main()
