# -*- coding: utf-8 -*-
"""Compare four proposed mechanism policies under outside options.

The four policies correspond to the proposed design ideas:
  1) priority_credits
  2) soft_commitment
  3) risk_adjusted
  4) preserve_quality

All policies are evaluated on a shared (rho, lambda_ent) grid.
Outputs:
  - cell-level CSV with mean/std losses and weighted objective
  - winner-by-cell CSV
  - optional heatmaps (if matplotlib is available)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulate_pool_trajectories import one_minus_power

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


POLICIES = (
    "priority_credits",
    "soft_commitment",
    "risk_adjusted",
    "preserve_quality",
)

POLICY_LABELS = {
    "priority_credits": "Priority credits",
    "soft_commitment": "Soft commitment",
    "risk_adjusted": "Risk-adjusted matching",
    "preserve_quality": "Preserve high-quality E",
}


@dataclass(frozen=True)
class LossResult:
    total: float
    h: float
    e: float

    def objective(self, w_h: float, w_e: float) -> float:
        return w_h * self.h + w_e * self.e


@dataclass(frozen=True)
class SummaryPoint:
    total_mean: float
    total_std: float
    h_mean: float
    h_std: float
    e_mean: float
    e_std: float
    objective_mean: float
    objective_std: float


def summarize(results: list[LossResult], w_h: float, w_e: float) -> SummaryPoint:
    total = np.array([x.total for x in results], dtype=float)
    h = np.array([x.h for x in results], dtype=float)
    e = np.array([x.e for x in results], dtype=float)
    obj = w_h * h + w_e * e
    return SummaryPoint(
        total_mean=float(total.mean()),
        total_std=float(total.std(ddof=1)),
        h_mean=float(h.mean()),
        h_std=float(h.std(ddof=1)),
        e_mean=float(e.mean()),
        e_std=float(e.std(ddof=1)),
        objective_mean=float(obj.mean()),
        objective_std=float(obj.std(ddof=1)),
    )


def _remove_one_from_bucket(arr: np.ndarray, strategy: str, rng: np.random.Generator) -> None:
    """Remove one element from non-empty integer bucket array in-place."""
    if arr.sum() <= 0:
        raise ValueError("Cannot remove from empty buckets.")
    if strategy == "highest":
        idx = int(np.max(np.nonzero(arr)[0]))
    elif strategy == "lowest":
        idx = int(np.min(np.nonzero(arr)[0]))
    elif strategy == "random":
        probs = arr / arr.sum()
        idx = int(rng.choice(len(arr), p=probs))
    else:
        raise ValueError(f"Unknown bucket removal strategy: {strategy}")
    arr[idx] -= 1


def _simulate_priority_credits(
    m: int,
    d1: float,
    d2: float,
    lambda_ent: float,
    rho: float,
    horizon: float,
    seed: int,
    credit_cap: int,
    gamma_credit: float,
) -> LossResult:
    p_he = d1 / (2.0 * m)
    p_ee = d2 / (2.0 * m)
    rng = np.random.default_rng(seed)

    rate_h = m * (1.0 - lambda_ent)
    rate_e = m * lambda_ent

    h_pool = 0
    e_credits = np.zeros(credit_cap + 1, dtype=int)
    perish_h = 0
    perish_e = 0
    t = 0.0

    while t < horizon:
        total_e = int(e_credits.sum())
        rate_h_crit = h_pool
        rate_e_crit = rho * total_e
        rate_credit = gamma_credit * int(e_credits[:-1].sum())
        rate_total = rate_h + rate_e + rate_h_crit + rate_e_crit + rate_credit
        if rate_total <= 0:
            break
        t_next = t + rng.exponential(1.0 / rate_total)
        if t_next > horizon:
            break
        t = t_next

        u = rng.random() * rate_total
        if u < rate_h:
            h_pool += 1
            continue
        if u < rate_h + rate_e:
            e_credits[0] += 1
            continue
        if u < rate_h + rate_e + rate_h_crit:
            # H critical: credits get priority for being selected.
            h_pool -= 1
            total_e = int(e_credits.sum())
            if total_e > 0 and rng.random() < one_minus_power(p_he, total_e):
                _remove_one_from_bucket(e_credits, "highest", rng)
            else:
                perish_h += 1
            continue
        if u < rate_h + rate_e + rate_h_crit + rate_e_crit:
            # E critical from random bucket.
            total_e = int(e_credits.sum())
            if total_e <= 0:
                continue
            _remove_one_from_bucket(e_credits, "random", rng)
            total_e_after = int(e_credits.sum())
            n_h = rng.binomial(h_pool, p_he)
            n_e = rng.binomial(total_e_after, p_ee)
            if n_h + n_e == 0:
                perish_e += 1
            elif rng.random() < n_h / (n_h + n_e):
                h_pool -= 1
            else:
                _remove_one_from_bucket(e_credits, "random", rng)
            continue

        # Credit accumulation transition.
        promotable = e_credits[:-1]
        if promotable.sum() > 0:
            probs = promotable / promotable.sum()
            c = int(rng.choice(credit_cap, p=probs))
            e_credits[c] -= 1
            e_credits[c + 1] += 1

    denom = m * horizon
    return LossResult((perish_h + perish_e) / denom, perish_h / denom, perish_e / denom)


def _simulate_soft_commitment(
    m: int,
    d1: float,
    d2: float,
    lambda_ent: float,
    rho: float,
    horizon: float,
    seed: int,
    tenure_cap: int,
    gamma_tenure: float,
    alpha_tenure: float,
) -> LossResult:
    p_he = d1 / (2.0 * m)
    p_ee = d2 / (2.0 * m)
    rng = np.random.default_rng(seed)

    rate_h = m * (1.0 - lambda_ent)
    rate_e = m * lambda_ent

    h_pool = 0
    e_tenure = np.zeros(tenure_cap + 1, dtype=int)
    perish_h = 0
    perish_e = 0
    t = 0.0

    while t < horizon:
        total_e = int(e_tenure.sum())
        rate_h_crit = h_pool
        # delta_E(c) = rho * exp(-alpha * c)
        tenure_rates = rho * np.exp(-alpha_tenure * np.arange(tenure_cap + 1))
        rate_e_crit = float(np.dot(e_tenure, tenure_rates))
        rate_age = gamma_tenure * int(e_tenure[:-1].sum())
        rate_total = rate_h + rate_e + rate_h_crit + rate_e_crit + rate_age
        if rate_total <= 0:
            break
        t_next = t + rng.exponential(1.0 / rate_total)
        if t_next > horizon:
            break
        t = t_next

        u = rng.random() * rate_total
        if u < rate_h:
            h_pool += 1
            continue
        if u < rate_h + rate_e:
            e_tenure[0] += 1
            continue
        if u < rate_h + rate_e + rate_h_crit:
            h_pool -= 1
            total_e = int(e_tenure.sum())
            if total_e > 0 and rng.random() < one_minus_power(p_he, total_e):
                _remove_one_from_bucket(e_tenure, "random", rng)
            else:
                perish_h += 1
            continue
        if u < rate_h + rate_e + rate_h_crit + rate_e_crit:
            if e_tenure.sum() <= 0:
                continue
            bucket_weights = e_tenure * tenure_rates
            c = int(rng.choice(tenure_cap + 1, p=bucket_weights / bucket_weights.sum()))
            e_tenure[c] -= 1
            total_e_after = int(e_tenure.sum())
            n_h = rng.binomial(h_pool, p_he)
            n_e = rng.binomial(total_e_after, p_ee)
            if n_h + n_e == 0:
                perish_e += 1
            elif rng.random() < n_h / (n_h + n_e):
                h_pool -= 1
            else:
                _remove_one_from_bucket(e_tenure, "random", rng)
            continue

        # Tenure accumulation.
        promotable = e_tenure[:-1]
        if promotable.sum() > 0:
            probs = promotable / promotable.sum()
            c = int(rng.choice(tenure_cap, p=probs))
            e_tenure[c] -= 1
            e_tenure[c + 1] += 1

    denom = m * horizon
    return LossResult((perish_h + perish_e) / denom, perish_h / denom, perish_e / denom)


def _simulate_risk_adjusted(
    m: int,
    d1: float,
    d2: float,
    lambda_ent: float,
    rho: float,
    horizon: float,
    seed: int,
    risk_lambda: float,
    risk_h_match: float,
    risk_ee_match: float,
) -> LossResult:
    p_he = d1 / (2.0 * m)
    p_ee = d2 / (2.0 * m)
    rng = np.random.default_rng(seed)

    # Utility of chosen edge type in critical matching.
    u_h = p_he - risk_lambda * risk_h_match
    u_ee = p_ee - risk_lambda * risk_ee_match

    rate_h = m * (1.0 - lambda_ent)
    rate_e = m * lambda_ent

    h_pool = 0
    e_pool = 0
    perish_h = 0
    perish_e = 0
    t = 0.0

    while t < horizon:
        rate_h_crit = h_pool
        rate_e_crit = rho * e_pool
        rate_total = rate_h + rate_e + rate_h_crit + rate_e_crit
        if rate_total <= 0:
            break
        t_next = t + rng.exponential(1.0 / rate_total)
        if t_next > horizon:
            break
        t = t_next

        u = rng.random() * rate_total
        if u < rate_h:
            h_pool += 1
            continue
        if u < rate_h + rate_e:
            e_pool += 1
            continue
        if u < rate_h + rate_e + rate_h_crit:
            h_pool -= 1
            if e_pool > 0 and u_h > 0 and rng.random() < one_minus_power(p_he, e_pool):
                e_pool -= 1
            else:
                perish_h += 1
            continue

        # E critical.
        if e_pool <= 0:
            continue
        e_pool -= 1
        n_h = rng.binomial(h_pool, p_he)
        n_e = rng.binomial(e_pool, p_ee)
        w_h = max(u_h, 0.0) * n_h
        w_e = max(u_ee, 0.0) * n_e
        if w_h + w_e <= 0:
            perish_e += 1
        elif rng.random() < w_h / (w_h + w_e):
            h_pool -= 1
        else:
            e_pool -= 1

    denom = m * horizon
    return LossResult((perish_h + perish_e) / denom, perish_h / denom, perish_e / denom)


def _simulate_preserve_quality(
    m: int,
    d1: float,
    d2: float,
    lambda_ent: float,
    rho: float,
    horizon: float,
    seed: int,
    share_quality_g: float,
    quality_boost: float,
    allow_g_fallback: bool,
) -> LossResult:
    p_he = d1 / (2.0 * m)
    p_ee = d2 / (2.0 * m)
    p_ge = min(1.0, p_ee * quality_boost)
    rng = np.random.default_rng(seed)

    rate_h = m * (1.0 - lambda_ent)
    rate_e = m * lambda_ent

    h_pool = 0
    e_r_pool = 0
    e_g_pool = 0
    perish_h = 0
    perish_e = 0
    t = 0.0

    while t < horizon:
        total_e = e_r_pool + e_g_pool
        rate_h_crit = h_pool
        rate_e_crit = rho * total_e
        rate_total = rate_h + rate_e + rate_h_crit + rate_e_crit
        if rate_total <= 0:
            break
        t_next = t + rng.exponential(1.0 / rate_total)
        if t_next > horizon:
            break
        t = t_next

        u = rng.random() * rate_total
        if u < rate_h:
            h_pool += 1
            continue
        if u < rate_h + rate_e:
            if rng.random() < share_quality_g:
                e_g_pool += 1
            else:
                e_r_pool += 1
            continue
        if u < rate_h + rate_e + rate_h_crit:
            # H critical: prefer using high-quality E_G when match exists.
            h_pool -= 1
            total_e = e_r_pool + e_g_pool
            if total_e > 0 and rng.random() < one_minus_power(p_he, total_e):
                if e_g_pool > 0:
                    e_g_pool -= 1
                else:
                    e_r_pool -= 1
            else:
                perish_h += 1
            continue

        # E critical.
        total_e = e_r_pool + e_g_pool
        if total_e <= 0:
            continue
        active_g = rng.random() < (e_g_pool / total_e) if total_e > 0 else False
        if active_g:
            e_g_pool -= 1
        else:
            e_r_pool -= 1

        n_h = rng.binomial(h_pool, p_he)
        n_r = rng.binomial(e_r_pool, p_ee)
        n_g = rng.binomial(e_g_pool, p_ge)

        if n_h > 0:
            h_pool -= 1
            continue
        if n_r > 0:
            e_r_pool -= 1
            continue
        if n_g > 0 and allow_g_fallback:
            e_g_pool -= 1
            continue
        perish_e += 1

    denom = m * horizon
    return LossResult((perish_h + perish_e) / denom, perish_h / denom, perish_e / denom)


def simulate_one(
    policy: str,
    m: int,
    d1: float,
    d2: float,
    lambda_ent: float,
    rho: float,
    horizon: float,
    seed: int,
) -> LossResult:
    if policy == "priority_credits":
        return _simulate_priority_credits(
            m=m,
            d1=d1,
            d2=d2,
            lambda_ent=lambda_ent,
            rho=rho,
            horizon=horizon,
            seed=seed,
            credit_cap=4,
            gamma_credit=0.30,
        )
    if policy == "soft_commitment":
        return _simulate_soft_commitment(
            m=m,
            d1=d1,
            d2=d2,
            lambda_ent=lambda_ent,
            rho=rho,
            horizon=horizon,
            seed=seed,
            tenure_cap=5,
            gamma_tenure=0.25,
            alpha_tenure=0.45,
        )
    if policy == "risk_adjusted":
        return _simulate_risk_adjusted(
            m=m,
            d1=d1,
            d2=d2,
            lambda_ent=lambda_ent,
            rho=rho,
            horizon=horizon,
            seed=seed,
            risk_lambda=0.45,
            risk_h_match=0.04,
            risk_ee_match=0.10,
        )
    if policy == "preserve_quality":
        return _simulate_preserve_quality(
            m=m,
            d1=d1,
            d2=d2,
            lambda_ent=lambda_ent,
            rho=rho,
            horizon=horizon,
            seed=seed,
            share_quality_g=0.35,
            quality_boost=1.25,
            allow_g_fallback=False,
        )
    raise ValueError(f"Unknown policy {policy!r}")


def stem_name(
    m: int,
    d1: float,
    d2: float,
    horizon: float,
    n_runs: int,
    rho_grid: tuple[float, ...],
    lambda_ent_grid: tuple[float, ...],
) -> str:
    tok_e = "_".join(f"{x:g}".replace(".", "p") for x in rho_grid)
    tok_ent = "_".join(f"{x:g}".replace(".", "p") for x in lambda_ent_grid)
    return (
        f"mechanism_policy_comparison_m{m}_d1_{d1:g}_d2_{d2:g}_"
        f"rho_{tok_e}_lambdaEnt_{tok_ent}_t{horizon:g}_runs{n_runs}"
    )


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "m",
        "d1",
        "d2",
        "T",
        "n_runs",
        "rho",
        "lambda_ent",
        "policy",
        "objective_mean",
        "objective_std",
        "total_mean",
        "total_std",
        "H_mean",
        "H_std",
        "E_mean",
        "E_std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_winner_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["rho", "lambda_ent", "best_policy", "best_objective", "runner_up_gap"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_heatmaps(
    out_dir: Path,
    stem: str,
    rho_grid: tuple[float, ...],
    lambda_ent_grid: tuple[float, ...],
    summary: dict[tuple[float, float], dict[str, SummaryPoint]],
) -> None:
    if not HAS_MPL:
        print("matplotlib not installed; skipping heatmaps.")
        return

    rho_vals = rho_grid
    lam_ent = lambda_ent_grid
    x_labels = [f"{x:.2g}" for x in lam_ent]
    y_labels = [f"{y:g}" for y in rho_vals]

    # Policy objective heatmaps (1x4).
    mats = []
    for policy in POLICIES:
        z = np.zeros((len(rho_vals), len(lam_ent)), dtype=float)
        for i, le in enumerate(rho_vals):
            for j, lent in enumerate(lam_ent):
                z[i, j] = summary[(le, lent)][policy].objective_mean
        mats.append(z)

    fig, axes = plt.subplots(1, len(POLICIES), figsize=(5.4 * len(POLICIES), 4.8), constrained_layout=True)
    vmin = float(min(np.min(m) for m in mats))
    vmax = float(max(np.max(m) for m in mats))
    norm = colors.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-12)
    last_im = None
    for idx, (ax, mat, policy) in enumerate(zip(axes, mats, POLICIES)):
        last_im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis", norm=norm)
        ax.set_title(POLICY_LABELS[policy])
        ax.set_xticks(np.arange(len(lam_ent)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(rho_vals)))
        if idx == 0:
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_ylabel("rho (E criticality rate)")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("lambda_ent")
    fig.suptitle("Objective loss by mechanism policy (lower is better)")
    fig.colorbar(last_im, ax=axes, label="Objective loss (per mT)")
    fig.savefig(out_dir / f"{stem}_objective_by_policy.png", dpi=180)
    plt.close(fig)

    # Winner map.
    winner_index = np.zeros((len(rho_vals), len(lam_ent)), dtype=int)
    for i, le in enumerate(rho_vals):
        for j, lent in enumerate(lam_ent):
            best = min(POLICIES, key=lambda p: summary[(le, lent)][p].objective_mean)
            winner_index[i, j] = POLICIES.index(best)

    cmap = colors.ListedColormap(["#4c78a8", "#f58518", "#54a24b", "#b279a2"])
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    im = ax.imshow(winner_index, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(POLICIES) - 0.5)
    ax.set_title("Best mechanism policy by (rho, lambda_ent)")
    ax.set_xticks(np.arange(len(lam_ent)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rho_vals)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("lambda_ent")
    ax.set_ylabel("rho (E criticality rate)")
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(POLICIES)))
    cbar.ax.set_yticklabels([POLICY_LABELS[p] for p in POLICIES])
    fig.savefig(out_dir / f"{stem}_best_policy_map.png", dpi=180)
    plt.close(fig)


def run() -> None:
    parser = argparse.ArgumentParser(description="Compare four proposed mechanism policies.")
    parser.add_argument("--m", type=int, default=5000)
    parser.add_argument("--d1", type=float, default=10.0)
    parser.add_argument("--d2", type=float, default=1000.0)
    parser.add_argument("--horizon", type=float, default=80.0)
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--w-h", type=float, default=1.0)
    parser.add_argument("--w-e", type=float, default=1.0)
    parser.add_argument("--dense-grid", action="store_true", help="Use denser rho/lambda_ent grids.")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--out-dir", type=Path, default=Path("simulation_outputs"))
    args = parser.parse_args()

    if args.n_runs < 2:
        raise ValueError("--n-runs must be >= 2")

    if args.dense_grid:
        rho_grid = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0)
        lambda_ent_grid = tuple(float(x) for x in np.linspace(0.05, 0.95, 19))
    else:
        rho_grid = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0)
        lambda_ent_grid = tuple(float(x) for x in np.linspace(0.1, 0.9, 9))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem_name(args.m, args.d1, args.d2, args.horizon, args.n_runs, rho_grid, lambda_ent_grid)

    summary: dict[tuple[float, float], dict[str, SummaryPoint]] = {}
    results_rows: list[dict[str, object]] = []
    winner_rows: list[dict[str, object]] = []

    for i, rho in enumerate(rho_grid):
        for j, lam_ent in enumerate(lambda_ent_grid):
            cell_key = (rho, lam_ent)
            summary[cell_key] = {}
            print(f"Cell rho={rho:g}, lambda_ent={lam_ent:g}")
            for k, policy in enumerate(POLICIES):
                losses = []
                for run_idx in range(args.n_runs):
                    seed = args.seed + 100000 * i + 4000 * j + 300 * k + run_idx
                    losses.append(
                        simulate_one(
                            policy=policy,
                            m=args.m,
                            d1=args.d1,
                            d2=args.d2,
                            lambda_ent=lam_ent,
                            rho=rho,
                            horizon=args.horizon,
                            seed=seed,
                        )
                    )
                point = summarize(losses, args.w_h, args.w_e)
                summary[cell_key][policy] = point
                results_rows.append(
                    {
                        "m": args.m,
                        "d1": f"{args.d1:g}",
                        "d2": f"{args.d2:g}",
                        "T": f"{args.horizon:g}",
                        "n_runs": args.n_runs,
                        "rho": f"{rho:g}",
                        "lambda_ent": f"{lam_ent:g}",
                        "policy": policy,
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

            ranked = sorted(POLICIES, key=lambda p: summary[cell_key][p].objective_mean)
            best, second = ranked[0], ranked[1]
            gap = summary[cell_key][second].objective_mean - summary[cell_key][best].objective_mean
            winner_rows.append(
                {
                    "rho": f"{rho:g}",
                    "lambda_ent": f"{lam_ent:g}",
                    "best_policy": best,
                    "best_objective": f"{summary[cell_key][best].objective_mean:.8f}",
                    "runner_up_gap": f"{gap:.8f}",
                }
            )

    write_results_csv(args.out_dir / f"{stem}.csv", results_rows)
    write_winner_csv(args.out_dir / f"{stem}_best_policy.csv", winner_rows)
    write_heatmaps(args.out_dir, stem, rho_grid, lambda_ent_grid, summary)

    print("\nWrote:")
    print(f"  {stem}.csv")
    print(f"  {stem}_best_policy.csv")
    if HAS_MPL:
        print(f"  {stem}_objective_by_policy.png")
        print(f"  {stem}_best_policy_map.png")

    # Show overall winner frequency.
    counts = {p: 0 for p in POLICIES}
    for row in winner_rows:
        counts[str(row['best_policy'])] += 1
    print("\nBest-policy cell counts:")
    for p in POLICIES:
        print(f"  {POLICY_LABELS[p]:>24}: {counts[p]}")


if __name__ == "__main__":
    run()
