# -*- coding: utf-8 -*-
"""Plot policy-loss and patient-advantage heatmaps from the entrance/departure grid CSV.

This script is a post-processing companion to `simulate_entrance_mix_sweep.py`.

It visualizes outcomes over the 2D parameter grid:
  - easy entrant share lambda_ent (x-axis)
  - easy criticality multiplier rho (y-axis)

Loss objective:
  objective = w_H * H_perish_over_mT_mean + w_E * E_perish_over_mT_mean

With defaults w_H = w_E = 1, this equals total perishing over mT.

Also writes a focused TA-Patient vs TA-Greedy figure to show whether
TA-Patient's advantage diminishes as rho increases and how this depends
on lambda_ent.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulate_pool_trajectories import POLICIES, POLICY_LABELS


@dataclass(frozen=True)
class Cell:
    total: float
    h: float
    e: float

    def objective(self, w_h: float, w_e: float) -> float:
        return w_h * self.h + w_e * self.e


Grid = dict[tuple[float, float], dict[str, Cell]]


def load_grid(csv_path: Path) -> tuple[tuple[float, ...], tuple[float, ...], Grid]:
    grid: Grid = {}
    rho_vals: set[float] = set()
    lambda_ent_vals: set[float] = set()

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {
            "easy_entrant_share",
            "policy",
            "total_perish_over_mT_mean",
            "H_perish_over_mT_mean",
            "E_perish_over_mT_mean",
        }
        missing = required - fields
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        if "rho" not in fields and "lambda_E" not in fields:
            raise ValueError("CSV must include either 'rho' or 'lambda_E' column.")

        for row in reader:
            rho = float(row["rho"]) if "rho" in row and row["rho"] != "" else float(row["lambda_E"])
            lam_ent = float(row["easy_entrant_share"])
            policy = row["policy"]
            if policy not in POLICIES:
                continue
            cell = Cell(
                total=float(row["total_perish_over_mT_mean"]),
                h=float(row["H_perish_over_mT_mean"]),
                e=float(row["E_perish_over_mT_mean"]),
            )
            key = (rho, lam_ent)
            if key not in grid:
                grid[key] = {}
            grid[key][policy] = cell
            rho_vals.add(rho)
            lambda_ent_vals.add(lam_ent)

    lambda_e = tuple(sorted(rho_vals))
    lambda_ent = tuple(sorted(lambda_ent_vals))
    return lambda_e, lambda_ent, grid


def objective_matrix(
    grid: Grid,
    policy: str,
    lambda_e: tuple[float, ...],
    lambda_ent: tuple[float, ...],
    w_h: float,
    w_e: float,
) -> np.ndarray:
    z = np.zeros((len(lambda_e), len(lambda_ent)), dtype=float)
    for i, lam_e in enumerate(lambda_e):
        for j, lam_ent in enumerate(lambda_ent):
            z[i, j] = grid[(lam_e, lam_ent)][policy].objective(w_h, w_e)
    return z


def render(
    out_path: Path,
    lambda_e: tuple[float, ...],
    lambda_ent: tuple[float, ...],
    matrices: list[np.ndarray],
    titles: list[str],
    main_title: str,
    colorbar_label: str,
    shared_scale: bool,
    cmap: str,
    center_zero: bool = False,
    zero_contour: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    ncols = len(matrices)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.8), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    if shared_scale:
        vmin = float(min(np.min(m) for m in matrices))
        vmax = float(max(np.max(m) for m in matrices))
        if center_zero:
            vmax_abs = max(abs(vmin), abs(vmax))
            norm = colors.TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-12)
    else:
        norm = None

    x_ticks = np.arange(len(lambda_ent))
    y_ticks = np.arange(len(lambda_e))
    x_labels = [f"{x:.2g}" for x in lambda_ent]
    y_labels = [f"{y:g}" for y in lambda_e]

    last_im = None
    for idx, (ax, mat, title) in enumerate(zip(axes, matrices, titles)):
        if norm is None:
            if center_zero:
                vmax_abs = max(abs(float(mat.min())), abs(float(mat.max())))
                local_norm = colors.TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)
            else:
                local_norm = colors.Normalize(
                    vmin=float(mat.min()),
                    vmax=float(mat.max()) if float(mat.max()) > float(mat.min()) else float(mat.min()) + 1e-12,
                )
        else:
            local_norm = norm
        last_im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, norm=local_norm)
        if zero_contour:
            # Draw indifference frontier (0 = policy tie) when requested.
            ax.contour(mat, levels=[0.0], colors="black", linewidths=1.0)
        ax.set_title(title)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(y_ticks)
        if idx == 0:
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_ylabel("rho (E criticality multiplier)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("lambda_ent (easy entrant share)")

    fig.suptitle(main_title)
    fig.colorbar(last_im, ax=axes, label=colorbar_label, shrink=0.9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def render_tap_vs_tag_dynamics(
    out_path: Path,
    lambda_e: tuple[float, ...],
    lambda_ent: tuple[float, ...],
    tap_advantage: np.ndarray,
    w_h: float,
    w_e: float,
    zero_contour: bool,
) -> None:
    """Create focused TA-Patient vs TA-Greedy visuals.

    `tap_advantage` is TAG loss - TAP loss, so positive values favor TA-Patient.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    ax_hm, ax_ln = axes

    vmax_abs = max(abs(float(tap_advantage.min())), abs(float(tap_advantage.max())))
    norm = colors.TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs if vmax_abs > 0 else 1e-12)

    # Left panel: 2D heatmap with indifference contour.
    im = ax_hm.imshow(tap_advantage, origin="lower", aspect="auto", cmap="RdBu_r", norm=norm)
    if zero_contour:
        ax_hm.contour(tap_advantage, levels=[0.0], colors="black", linewidths=1.0)
    ax_hm.set_title("TA-Patient advantage over TA-Greedy")
    ax_hm.set_xlabel("lambda_ent (easy entrant share)")
    ax_hm.set_ylabel("rho (E criticality multiplier)")
    ax_hm.set_xticks(np.arange(len(lambda_ent)))
    ax_hm.set_xticklabels([f"{x:.2g}" for x in lambda_ent], rotation=45, ha="right", fontsize=8)
    ax_hm.set_yticks(np.arange(len(lambda_e)))
    ax_hm.set_yticklabels([f"{y:g}" for y in lambda_e], fontsize=8)
    fig.colorbar(im, ax=ax_hm, shrink=0.9, label="TAG loss - TAP loss (per mT)")

    # Right panel: lines vs rho for each lambda_ent.
    cm = plt.get_cmap("viridis")
    lam_ent_arr = np.array(lambda_ent, dtype=float)
    lam_e_arr = np.array(lambda_e, dtype=float)
    for j, lam_ent in enumerate(lambda_ent):
        color = cm(j / max(1, len(lambda_ent) - 1))
        ax_ln.plot(
            lam_e_arr,
            tap_advantage[:, j],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            color=color,
            label=f"λ_ent={lam_ent:.2g}",
        )
    ax_ln.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax_ln.set_xscale("log")
    ax_ln.set_title("How TAP advantage changes as rho rises")
    ax_ln.set_xlabel("rho (log scale)")
    ax_ln.set_ylabel("TAG loss - TAP loss (per mT)")
    ax_ln.legend(ncol=2, fontsize=7, frameon=False)
    ax_ln.grid(True, which="both", alpha=0.22)

    fig.suptitle(
        f"TA-Patient vs TA-Greedy under objective = {w_h:g}*H_loss + {w_e:g}*E_loss",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot patient advantage heatmaps from entrance/departure sweep CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "simulation_outputs/"
            "entrance_departure_grid_m10000_d1_10_d2_1000_"
            "lambdaE_1_2_5_10_20_40_100_lambdaEnt_lin01to09_t100_runs5.csv"
        ),
        help="Input grid CSV from simulate_entrance_mix_sweep.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("simulation_outputs"),
        help="Directory for output images",
    )
    parser.add_argument(
        "--w-h",
        type=float,
        default=1.0,
        help="Weight on H perishing in loss objective",
    )
    parser.add_argument(
        "--w-e",
        type=float,
        default=1.0,
        help="Weight on E perishing in loss objective",
    )
    parser.add_argument(
        "--zero-contour",
        action="store_true",
        help="Overlay 0-level indifference contours on advantage heatmaps.",
    )
    args = parser.parse_args()

    lambda_e, lambda_ent, grid = load_grid(args.csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"wh_{args.w_h:g}_we_{args.w_e:g}".replace(".", "p")
    base = args.csv.stem

    # 1) Objective loss heatmaps by policy.
    policy_mats = [
        objective_matrix(grid, p, lambda_e, lambda_ent, args.w_h, args.w_e)
        for p in POLICIES
    ]
    render(
        out_path=args.out_dir / f"{base}_objective_loss_{suffix}_policy_heatmaps.png",
        lambda_e=lambda_e,
        lambda_ent=lambda_ent,
        matrices=policy_mats,
        titles=[POLICY_LABELS[p] for p in POLICIES],
        main_title=(
            f"Policy loss over (rho, lambda_ent); "
            f"objective = {args.w_h:g}*H_loss + {args.w_e:g}*E_loss"
        ),
        colorbar_label="Objective loss (per mT)",
        shared_scale=True,
        cmap="viridis",
        center_zero=False,
        zero_contour=False,
    )

    # 2) TA-Patient advantage heatmaps vs alternatives.
    tap = objective_matrix(grid, "TAP", lambda_e, lambda_ent, args.w_h, args.w_e)
    others = [p for p in POLICIES if p != "TAP"]
    advantage_mats = []
    titles = []
    for p in others:
        alt = objective_matrix(grid, p, lambda_e, lambda_ent, args.w_h, args.w_e)
        # Positive means TA-Patient has lower loss than alternative.
        advantage_mats.append(alt - tap)
        titles.append(f"TA-Patient advantage vs {POLICY_LABELS[p]}")

    render(
        out_path=args.out_dir / f"{base}_objective_advantage_{suffix}_ta_patient_vs_others.png",
        lambda_e=lambda_e,
        lambda_ent=lambda_ent,
        matrices=advantage_mats,
        titles=titles,
        main_title=(
            f"TA-Patient advantage over alternatives; "
            f"objective = {args.w_h:g}*H_loss + {args.w_e:g}*E_loss"
        ),
        colorbar_label="Alternative loss - TA-Patient loss (per mT)",
        shared_scale=True,
        cmap="RdBu_r",
        center_zero=True,
        zero_contour=args.zero_contour,
    )

    # 3) Focused TAP-vs-TAG dynamics for the specific hypothesis.
    tag = objective_matrix(grid, "TAG", lambda_e, lambda_ent, args.w_h, args.w_e)
    tap = objective_matrix(grid, "TAP", lambda_e, lambda_ent, args.w_h, args.w_e)
    tap_advantage = tag - tap  # positive => TA-Patient better than TA-Greedy
    render_tap_vs_tag_dynamics(
        out_path=args.out_dir / f"{base}_objective_tap_vs_tag_dynamics_{suffix}.png",
        lambda_e=lambda_e,
        lambda_ent=lambda_ent,
        tap_advantage=tap_advantage,
        w_h=args.w_h,
        w_e=args.w_e,
        zero_contour=args.zero_contour,
    )

    print("Wrote:")
    print(f"  {base}_objective_loss_{suffix}_policy_heatmaps.png")
    print(f"  {base}_objective_advantage_{suffix}_ta_patient_vs_others.png")
    print(f"  {base}_objective_tap_vs_tag_dynamics_{suffix}.png")
    print("Interpretation: positive (red) in advantage maps means TA-Patient is better.")


if __name__ == "__main__":
    main()
