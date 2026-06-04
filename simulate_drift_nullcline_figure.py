# -*- coding: utf-8 -*-
"""Estimate and visualize 2D drift/nullclines for (H_t, E_t).

This script provides simulation evidence for a rate-balance point in the
two-dimensional CTMC:
  - estimate empirical drift field components F_H, F_E
  - plot zero-contours (nullclines) F_H=0 and F_E=0
  - show occupancy concentration and sample trajectories

Outputs (simulation_outputs/):
  - drift_nullclines_<...>.png
  - drift_nullclines_<...>.csv  (bin-level estimates)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulate_pool_trajectories import POLICIES, one_minus_power

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False


@dataclass(frozen=True)
class PathStats:
    x_prev: np.ndarray
    y_prev: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    dt: np.ndarray
    traj_x: np.ndarray
    traj_y: np.ndarray


def _simulate_event_path(
    m: int,
    d1: float,
    d2: float,
    policy: str,
    lambda_ent: float,
    rho: float,
    horizon: float,
    burn_in: float,
    seed: int,
    thin_every: int = 20,
) -> PathStats:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}. Must be one of {POLICIES}.")
    if not (0.0 < lambda_ent < 1.0):
        raise ValueError("lambda_ent must be in (0,1).")
    if rho <= 0.0:
        raise ValueError("rho must be positive.")

    p_he = d1 / (2.0 * m)
    p_ee = d2 / (2.0 * m)
    if not (0.0 <= p_he <= 1.0 and 0.0 <= p_ee <= 1.0):
        raise ValueError("Need d1/(2m) and d2/(2m) in [0,1].")

    rng = np.random.default_rng(seed)
    rate_h = m * (1.0 - lambda_ent)
    rate_e = m * lambda_ent

    h_pool = 0
    e_pool = 0
    t = 0.0
    step_idx = 0

    x_prev_list: list[float] = []
    y_prev_list: list[float] = []
    dx_list: list[float] = []
    dy_list: list[float] = []
    dt_list: list[float] = []

    traj_x: list[float] = []
    traj_y: list[float] = []

    while t < horizon:
        rate_h_crit = h_pool
        rate_e_crit = rho * e_pool
        rate_total = rate_h + rate_e + rate_h_crit + rate_e_crit
        if rate_total <= 0.0:
            break
        dt = float(rng.exponential(1.0 / rate_total))
        t_next = t + dt
        if t_next > horizon:
            break

        # Pre-jump state
        h_before = h_pool
        e_before = e_pool

        u = rng.random() * rate_total
        if u < rate_h:
            # H arrival
            if policy in ("greedy", "TAG") and e_pool > 0:
                if rng.random() < one_minus_power(p_he, e_pool):
                    e_pool -= 1
                else:
                    h_pool += 1
            else:
                h_pool += 1

        elif u < rate_h + rate_e:
            # E arrival
            if policy == "greedy" and (h_pool > 0 or e_pool > 0):
                n_h = rng.binomial(h_pool, p_he)
                n_e = rng.binomial(e_pool, p_ee)
                if n_h + n_e > 0:
                    if rng.random() < n_h / (n_h + n_e):
                        h_pool -= 1
                    else:
                        e_pool -= 1
                else:
                    e_pool += 1
            elif policy == "TAG" and (h_pool > 0 or e_pool > 0):
                n_h = rng.binomial(h_pool, p_he)
                if n_h > 0:
                    h_pool -= 1
                else:
                    n_e = rng.binomial(e_pool, p_ee)
                    if n_e > 0:
                        e_pool -= 1
                    else:
                        e_pool += 1
            else:
                e_pool += 1

        else:
            # Criticality event
            u2 = u - rate_h - rate_e
            if u2 < rate_h_crit:
                # H critical
                if policy in ("greedy", "TAG"):
                    h_pool -= 1
                else:
                    if e_pool > 0 and rng.random() < one_minus_power(p_he, e_pool):
                        h_pool -= 1
                        e_pool -= 1
                    else:
                        h_pool -= 1
            else:
                # E critical
                if policy in ("greedy", "TAG"):
                    e_pool -= 1
                elif policy == "patient":
                    n_h = rng.binomial(h_pool, p_he)
                    n_e = rng.binomial(e_pool - 1, p_ee)
                    if n_h + n_e > 0:
                        if rng.random() < n_h / (n_h + n_e):
                            h_pool -= 1
                            e_pool -= 1
                        else:
                            e_pool -= 2
                    else:
                        e_pool -= 1
                else:  # TAP
                    n_h = rng.binomial(h_pool, p_he)
                    if n_h > 0:
                        h_pool -= 1
                        e_pool -= 1
                    else:
                        n_e = rng.binomial(e_pool - 1, p_ee)
                        if n_e > 0:
                            e_pool -= 2
                        else:
                            e_pool -= 1

        if t >= burn_in:
            x_prev_list.append(h_before / m)
            y_prev_list.append(e_before / m)
            dx_list.append((h_pool - h_before) / m)
            dy_list.append((e_pool - e_before) / m)
            dt_list.append(dt)
            if step_idx % thin_every == 0:
                traj_x.append(h_pool / m)
                traj_y.append(e_pool / m)

        step_idx += 1
        t = t_next

    return PathStats(
        x_prev=np.asarray(x_prev_list, dtype=float),
        y_prev=np.asarray(y_prev_list, dtype=float),
        dx=np.asarray(dx_list, dtype=float),
        dy=np.asarray(dy_list, dtype=float),
        dt=np.asarray(dt_list, dtype=float),
        traj_x=np.asarray(traj_x, dtype=float),
        traj_y=np.asarray(traj_y, dtype=float),
    )


def _hist2d_weighted(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    h, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=w)
    return h


def _write_csv(
    path: Path,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    occ: np.ndarray,
    drift_x: np.ndarray,
    drift_y: np.ndarray,
    time_in_bin: np.ndarray,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x_center",
                "y_center",
                "occupancy_share",
                "drift_x_estimate",
                "drift_y_estimate",
                "time_in_bin",
            ]
        )
        for i, x in enumerate(x_centers):
            for j, y in enumerate(y_centers):
                writer.writerow(
                    [
                        f"{x:.8f}",
                        f"{y:.8f}",
                        f"{occ[i, j]:.10f}",
                        f"{drift_x[i, j]:.10f}",
                        f"{drift_y[i, j]:.10f}",
                        f"{time_in_bin[i, j]:.10f}",
                    ]
                )


def _estimate_balance_point(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    occ: np.ndarray,
    drift_x: np.ndarray,
    drift_y: np.ndarray,
    time_in_bin: np.ndarray,
) -> tuple[int, int, float]:
    valid = time_in_bin > 0
    if not np.any(valid):
        raise RuntimeError("Cannot estimate balance point: all bins have zero occupancy time.")

    drift_norm = np.sqrt(drift_x**2 + drift_y**2)
    occ_pos = occ[valid]
    occ_threshold = float(np.quantile(occ_pos, 0.60))
    candidate = valid & (occ >= occ_threshold)
    if not np.any(candidate):
        candidate = valid

    score = np.where(candidate, drift_norm, np.inf)
    flat_idx = int(np.argmin(score))
    i_star, j_star = np.unravel_index(flat_idx, drift_norm.shape)
    return i_star, j_star, float(occ_threshold)


def _write_balance_table(
    out_path: Path,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    occ: np.ndarray,
    drift_x: np.ndarray,
    drift_y: np.ndarray,
    time_in_bin: np.ndarray,
    i_star: int,
    j_star: int,
    top_n: int = 15,
) -> None:
    drift_norm = np.sqrt(drift_x**2 + drift_y**2)
    valid = time_in_bin > 0
    # Balance score prefers heavily visited states with low local drift.
    balance_score = np.where(valid, occ / (drift_norm + 1e-10), 0.0)
    order = np.argsort(balance_score.ravel())[::-1]

    x_star = x_centers[i_star]
    y_star = y_centers[j_star]
    selected = []
    for flat_idx in order:
        i, j = np.unravel_index(int(flat_idx), balance_score.shape)
        if not valid[i, j]:
            continue
        selected.append((i, j))
        if len(selected) >= top_n:
            break

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "is_estimated_balance_bin",
                "x_center",
                "y_center",
                "occupancy_share",
                "time_in_bin",
                "drift_x_estimate",
                "drift_y_estimate",
                "drift_norm",
                "balance_score",
                "distance_to_balance_bin",
            ]
        )
        for rank, (i, j) in enumerate(selected, start=1):
            dist = float(np.hypot(x_centers[i] - x_star, y_centers[j] - y_star))
            writer.writerow(
                [
                    rank,
                    int(i == i_star and j == j_star),
                    f"{x_centers[i]:.8f}",
                    f"{y_centers[j]:.8f}",
                    f"{occ[i, j]:.10f}",
                    f"{time_in_bin[i, j]:.10f}",
                    f"{drift_x[i, j]:.10f}",
                    f"{drift_y[i, j]:.10f}",
                    f"{drift_norm[i, j]:.10f}",
                    f"{balance_score[i, j]:.10f}",
                    f"{dist:.10f}",
                ]
            )


def _write_fixed_point_summary(
    out_path: Path,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    occ: np.ndarray,
    drift_x: np.ndarray,
    drift_y: np.ndarray,
    time_in_bin: np.ndarray,
    i_star: int,
    j_star: int,
    occ_threshold: float,
) -> None:
    drift_norm = np.sqrt(drift_x**2 + drift_y**2)
    valid = time_in_bin > 0
    writer_header = [
        "x_star",
        "y_star",
        "occupancy_share_at_star",
        "time_in_bin_at_star",
        "drift_x_at_star",
        "drift_y_at_star",
        "drift_norm_at_star",
        "occupancy_threshold_for_candidate_bins",
        "share_of_bins_with_data",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(writer_header)
        writer.writerow(
            [
                f"{x_centers[i_star]:.8f}",
                f"{y_centers[j_star]:.8f}",
                f"{occ[i_star, j_star]:.10f}",
                f"{time_in_bin[i_star, j_star]:.10f}",
                f"{drift_x[i_star, j_star]:.10f}",
                f"{drift_y[i_star, j_star]:.10f}",
                f"{drift_norm[i_star, j_star]:.10f}",
                f"{occ_threshold:.10f}",
                f"{(np.sum(valid) / valid.size):.10f}",
            ]
        )


def _plot(
    out_path: Path,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    occ: np.ndarray,
    drift_x: np.ndarray,
    drift_y: np.ndarray,
    trajectories: list[tuple[np.ndarray, np.ndarray]],
    title: str,
) -> None:
    if not HAS_MPL:
        print("matplotlib not installed; skipping PNG.")
        return

    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    # Panel 1: F_H drift
    ax = axes[0]
    im1 = ax.pcolormesh(x_edges, y_edges, drift_x.T, shading="auto", cmap="RdBu_r")
    ax.contour(X, Y, drift_x, levels=[0.0], colors="black", linewidths=1.0)
    ax.set_title(r"Estimated drift $F_H(x,y)$")
    ax.set_xlabel(r"$x = H_t/m$")
    ax.set_ylabel(r"$y = E_t/m$")
    fig.colorbar(im1, ax=ax, shrink=0.9)

    # Panel 2: F_E drift
    ax = axes[1]
    im2 = ax.pcolormesh(x_edges, y_edges, drift_y.T, shading="auto", cmap="RdBu_r")
    ax.contour(X, Y, drift_y, levels=[0.0], colors="black", linewidths=1.0)
    ax.set_title(r"Estimated drift $F_E(x,y)$")
    ax.set_xlabel(r"$x = H_t/m$")
    ax.set_ylabel(r"$y = E_t/m$")
    fig.colorbar(im2, ax=ax, shrink=0.9)

    # Panel 3: occupancy + nullclines + trajectories
    ax = axes[2]
    im3 = ax.pcolormesh(x_edges, y_edges, occ.T, shading="auto", cmap="viridis")
    c1 = ax.contour(X, Y, drift_x, levels=[0.0], colors="#00FFFF", linewidths=1.5)
    c2 = ax.contour(X, Y, drift_y, levels=[0.0], colors="#FF00FF", linewidths=1.5)
    for tx, ty in trajectories:
        if len(tx) > 1:
            ax.plot(tx, ty, color="white", alpha=0.55, linewidth=0.9)
    ax.set_title("Occupancy + nullclines + trajectories")
    ax.set_xlabel(r"$x = H_t/m$")
    ax.set_ylabel(r"$y = E_t/m$")
    ax.clabel(c1, fmt={0.0: "F_H=0"}, fontsize=8, inline=True)
    ax.clabel(c2, fmt={0.0: "F_E=0"}, fontsize=8, inline=True)
    fig.colorbar(im3, ax=ax, shrink=0.9, label="Occupancy share")

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def _plot_interpretable(
    out_path: Path,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    occ: np.ndarray,
    drift_x: np.ndarray,
    drift_y: np.ndarray,
    time_in_bin: np.ndarray,
    i_star: int,
    j_star: int,
    title: str,
) -> None:
    if not HAS_MPL:
        print("matplotlib not installed; skipping interpretable PNG.")
        return

    X, Y = np.meshgrid(x_centers, y_centers, indexing="ij")
    drift_norm = np.sqrt(drift_x**2 + drift_y**2)
    valid = time_in_bin > 0
    occ_threshold = float(np.quantile(time_in_bin[valid], 0.60)) if np.any(valid) else 0.0
    quiver_mask = valid & (time_in_bin >= occ_threshold)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    x_star = x_centers[i_star]
    y_star = y_centers[j_star]

    # Panel 1: Occupancy + nullclines + local vector field.
    ax = axes[0]
    im1 = ax.pcolormesh(x_edges, y_edges, occ.T, shading="auto", cmap="viridis")
    c1 = ax.contour(X, Y, drift_x, levels=[0.0], colors="#00FFFF", linewidths=1.6)
    c2 = ax.contour(X, Y, drift_y, levels=[0.0], colors="#FF00FF", linewidths=1.6)
    stride = max(1, len(x_centers) // 18)
    qmask = quiver_mask[::stride, ::stride]
    if np.any(qmask):
        ax.quiver(
            X[::stride, ::stride][qmask],
            Y[::stride, ::stride][qmask],
            drift_x[::stride, ::stride][qmask],
            drift_y[::stride, ::stride][qmask],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="white",
            alpha=0.7,
            width=0.003,
        )
    ax.scatter([x_star], [y_star], marker="*", s=160, color="gold", edgecolor="black", linewidth=0.7)
    ax.text(
        x_star,
        y_star,
        f"  k*≈({x_star:.3f}, {y_star:.3f})",
        fontsize=8,
        color="white",
        va="bottom",
        ha="left",
    )
    ax.set_title("Where the chain spends time and how it drifts")
    ax.set_xlabel(r"$x = H_t/m$")
    ax.set_ylabel(r"$y = E_t/m$")
    ax.clabel(c1, fmt={0.0: "F_H=0"}, fontsize=8, inline=True)
    ax.clabel(c2, fmt={0.0: "F_E=0"}, fontsize=8, inline=True)
    fig.colorbar(im1, ax=ax, shrink=0.9, label="Occupancy share")

    # Panel 2: Drift magnitude (smaller is closer to local balance).
    ax = axes[1]
    masked_norm = np.where(valid, drift_norm, np.nan)
    im2 = ax.pcolormesh(x_edges, y_edges, masked_norm.T, shading="auto", cmap="magma_r")
    ax.contour(X, Y, drift_x, levels=[0.0], colors="#00FFFF", linewidths=1.3)
    ax.contour(X, Y, drift_y, levels=[0.0], colors="#FF00FF", linewidths=1.3)
    ax.scatter([x_star], [y_star], marker="*", s=160, color="gold", edgecolor="black", linewidth=0.7)
    ax.set_title(r"Local imbalance size $||F(x,y)||$")
    ax.set_xlabel(r"$x = H_t/m$")
    ax.set_ylabel(r"$y = E_t/m$")
    fig.colorbar(im2, ax=ax, shrink=0.9, label=r"$\sqrt{F_H^2 + F_E^2}$")

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=190)
    plt.close(fig)


def _plot_markov_arrows(
    out_path: Path,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    occ: np.ndarray,
    drift_x: np.ndarray,
    drift_y: np.ndarray,
    time_in_bin: np.ndarray,
    trajectories: list[tuple[np.ndarray, np.ndarray]],
    i_star: int,
    j_star: int,
    m: int,
    policy: str,
    title: str,
) -> None:
    if not HAS_MPL:
        print("matplotlib not installed; skipping Markov arrow PNG.")
        return

    # Convert state coordinates and drift units from shares to agent counts.
    x_edges_count = x_edges * m
    y_edges_count = y_edges * m
    x_centers_count = x_centers * m
    y_centers_count = y_centers * m
    drift_h = drift_x * m
    drift_e = drift_y * m

    X, Y = np.meshgrid(x_centers_count, y_centers_count, indexing="ij")
    valid = time_in_bin > 0
    if not np.any(valid):
        print("No visited bins; skipping Markov arrow PNG.")
        return

    # Keep only high-occupancy bins for cleaner arrows.
    occ_cut = float(np.quantile(occ[valid], 0.70))
    mask = valid & (occ >= occ_cut)

    # Downsample to keep plot legible.
    stride = max(1, len(x_centers) // 14)
    Xm = X[::stride, ::stride]
    Ym = Y[::stride, ::stride]
    Um = drift_h[::stride, ::stride]
    Vm = drift_e[::stride, ::stride]
    Mm = mask[::stride, ::stride]
    if not np.any(Mm):
        Mm = valid[::stride, ::stride]

    # Robustly scale arrows so outliers do not dominate the visual.
    Nm = np.sqrt(Um**2 + Vm**2)
    n_ref = float(np.quantile(Nm[Mm], 0.90)) if np.any(Mm) else 1.0
    if n_ref <= 1e-12:
        n_ref = 1.0
    direction_u = np.divide(Um, Nm, out=np.zeros_like(Um), where=Nm > 1e-12)
    direction_v = np.divide(Vm, Nm, out=np.zeros_like(Vm), where=Nm > 1e-12)
    rel = np.clip(Nm / n_ref, 0.0, 1.0)
    max_axis = max(float(x_edges_count[-1]), float(y_edges_count[-1]), 1.0)
    arrow_len = (0.009 * max_axis) + (0.024 * max_axis) * rel
    U_plot = direction_u * arrow_len
    V_plot = direction_v * arrow_len

    fig, ax = plt.subplots(1, 1, figsize=(8.1, 6.6), constrained_layout=True)

    im = ax.pcolormesh(x_edges_count, y_edges_count, occ.T, shading="auto", cmap="Greys", alpha=0.22)
    q = ax.quiver(
        Xm[Mm],
        Ym[Mm],
        U_plot[Mm],
        V_plot[Mm],
        rel[Mm],
        cmap="plasma",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        width=0.004,
        alpha=0.95,
    )
    ax.quiverkey(q, X=0.86, Y=1.03, U=0.02 * max_axis, label="relative drift scale", labelpos="E")

    c1 = ax.contour(X, Y, drift_h, levels=[0.0], colors="#00B3B3", linewidths=2.1)
    c2 = ax.contour(X, Y, drift_e, levels=[0.0], colors="#B300B3", linewidths=2.1)

    # Overlay sampled phase trajectories (H_t, E_t), similar to ht/et style.
    traj_color = (
        "#1f77b4" if policy == "greedy" else
        "#d62728" if policy == "patient" else
        "#2ca02c" if policy == "TAG" else
        "#9467bd"
    )
    max_paths = 6
    for tx, ty in trajectories[:max_paths]:
        if len(tx) < 2:
            continue
        tx_count = tx * m
        ty_count = ty * m
        ax.plot(tx_count, ty_count, color=traj_color, alpha=0.52, linewidth=1.5, zorder=4)
        ax.scatter(tx_count[0], ty_count[0], color=traj_color, s=20, marker="o", zorder=5)
        ax.scatter(tx_count[-1], ty_count[-1], color=traj_color, s=24, marker="x", zorder=5)

    x_star = x_centers_count[i_star]
    y_star = y_centers_count[j_star]
    ax.scatter([x_star], [y_star], marker="*", s=170, color="gold", edgecolor="black", linewidth=0.7, zorder=6)
    ax.text(
        x_star,
        y_star,
        f"  k*≈({x_star:.1f}, {y_star:.1f})",
        fontsize=8,
        color="black",
        va="bottom",
        ha="left",
    )

    # Minimal legend without contour labels on the plot body.
    ax.plot([], [], color="#00B3B3", linewidth=2.1, label=r"$F_H=0$")
    ax.plot([], [], color="#B300B3", linewidth=2.1, label=r"$F_E=0$")
    ax.plot([], [], color=traj_color, linewidth=1.7, label="sample phase trajectories")
    ax.scatter([], [], color=traj_color, s=20, marker="o", label="trajectory start")
    ax.scatter([], [], color=traj_color, s=24, marker="x", label="trajectory end")
    ax.scatter([], [], marker="*", s=120, color="gold", edgecolor="black", linewidth=0.7, label=r"estimated $k^\ast$")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8)

    ax.set_title("Markov-chain drift + phase trajectories")
    ax.set_xlabel(r"Number of hard-to-match agents ($H_t$)")
    ax.set_ylabel(r"Number of easy-to-match agents ($E_t$)")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Occupancy share")
    fig.colorbar(q, ax=ax, shrink=0.85, pad=0.02, label="Relative drift magnitude")
    fig.suptitle(title, fontsize=12.5)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_occupancy_star(
    out_path: Path,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    occ: np.ndarray,
    i_star: int,
    j_star: int,
    m: int,
    title: str,
) -> None:
    if not HAS_MPL:
        print("matplotlib not installed; skipping occupancy-star PNG.")
        return

    x_edges_count = x_edges * m
    y_edges_count = y_edges * m
    x_star = x_centers[i_star] * m
    y_star = y_centers[j_star] * m

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.1), constrained_layout=True)
    im = ax.pcolormesh(x_edges_count, y_edges_count, occ.T, shading="auto", cmap="viridis")
    ax.scatter(
        [x_star],
        [y_star],
        marker="*",
        s=210,
        color="gold",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )
    ax.text(
        x_star,
        y_star,
        f"  (h*, e*) ≈ ({x_star:.1f}, {y_star:.1f})",
        fontsize=9,
        color="white",
        va="bottom",
        ha="left",
    )
    ax.set_xlabel(r"Number of hard-to-match agents ($H_t$)")
    ax.set_ylabel(r"Number of easy-to-match agents ($E_t$)")
    ax.set_title(r"Occupancy heatmap: $\Pr(H_t=h, E_t=e)$")
    fig.colorbar(im, ax=ax, shrink=0.9, label=r"Estimated $\Pr(H_t=h, E_t=e)$")
    fig.suptitle(title, fontsize=12.5)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _run_for_policy(args: argparse.Namespace, policy: str) -> None:
    all_x_prev = []
    all_y_prev = []
    all_dx = []
    all_dy = []
    all_dt = []
    trajectories: list[tuple[np.ndarray, np.ndarray]] = []

    for r in range(args.n_runs):
        s = _simulate_event_path(
            m=args.m,
            d1=args.d1,
            d2=args.d2,
            policy=policy,
            lambda_ent=args.lambda_ent,
            rho=args.rho,
            horizon=args.horizon,
            burn_in=args.burn_in,
            seed=args.seed + 1000 * r,
            thin_every=20,
        )
        if len(s.x_prev) == 0:
            continue
        all_x_prev.append(s.x_prev)
        all_y_prev.append(s.y_prev)
        all_dx.append(s.dx)
        all_dy.append(s.dy)
        all_dt.append(s.dt)
        trajectories.append((s.traj_x, s.traj_y))

    if not all_x_prev:
        raise RuntimeError(f"No post-burn-in events collected for policy={policy}.")

    x_prev = np.concatenate(all_x_prev)
    y_prev = np.concatenate(all_y_prev)
    dx = np.concatenate(all_dx)
    dy = np.concatenate(all_dy)
    dt = np.concatenate(all_dt)

    # Bin ranges with small margin.
    x_max = float(np.quantile(x_prev, 0.995) * 1.08 + 1e-6)
    y_max = float(np.quantile(y_prev, 0.995) * 1.08 + 1e-6)
    x_edges = np.linspace(0.0, max(x_max, 1e-4), args.bins + 1)
    y_edges = np.linspace(0.0, max(y_max, 1e-4), args.bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    time_in_bin = _hist2d_weighted(x_prev, y_prev, dt, x_edges, y_edges)
    sum_dx = _hist2d_weighted(x_prev, y_prev, dx, x_edges, y_edges)
    sum_dy = _hist2d_weighted(x_prev, y_prev, dy, x_edges, y_edges)

    with np.errstate(divide="ignore", invalid="ignore"):
        drift_x = np.divide(sum_dx, time_in_bin, out=np.zeros_like(sum_dx), where=time_in_bin > 0)
        drift_y = np.divide(sum_dy, time_in_bin, out=np.zeros_like(sum_dy), where=time_in_bin > 0)
    occ = time_in_bin / np.sum(time_in_bin)

    stem = (
        f"drift_nullclines_policy_{policy}_m{args.m}_d1_{args.d1:g}_d2_{args.d2:g}_"
        f"rho_{args.rho:g}_lambdaEnt_{args.lambda_ent:g}_T{args.horizon:g}_runs{args.n_runs}_bins{args.bins}"
    )
    csv_path = args.out_dir / f"{stem}.csv"
    png_path = args.out_dir / f"{stem}.png"
    interp_png_path = args.out_dir / f"{stem}_interpretable.png"
    arrows_png_path = args.out_dir / f"{stem}_markov_arrows.png"
    occ_star_png_path = args.out_dir / f"{stem}_occupancy_star.png"
    summary_csv_path = args.out_dir / f"{stem}_fixed_point_summary.csv"
    table_csv_path = args.out_dir / f"{stem}_balance_table.csv"

    i_star, j_star, occ_threshold = _estimate_balance_point(
        x_centers=x_centers,
        y_centers=y_centers,
        occ=occ,
        drift_x=drift_x,
        drift_y=drift_y,
        time_in_bin=time_in_bin,
    )

    _write_csv(csv_path, x_centers, y_centers, occ, drift_x, drift_y, time_in_bin)
    _write_fixed_point_summary(
        summary_csv_path,
        x_centers,
        y_centers,
        occ,
        drift_x,
        drift_y,
        time_in_bin,
        i_star,
        j_star,
        occ_threshold,
    )
    _write_balance_table(
        table_csv_path,
        x_centers,
        y_centers,
        occ,
        drift_x,
        drift_y,
        time_in_bin,
        i_star,
        j_star,
    )
    title = (
        f"Drift/nullcline diagnostics for ({policy})  "
        f"m={args.m}, d1={args.d1:g}, d2={args.d2:g}, rho={args.rho:g}, "
        f"lambda_ent={args.lambda_ent:g}, T={args.horizon:g}"
    )
    _plot(png_path, x_edges, y_edges, x_centers, y_centers, occ, drift_x, drift_y, trajectories, title)
    _plot_interpretable(
        interp_png_path,
        x_edges,
        y_edges,
        x_centers,
        y_centers,
        occ,
        drift_x,
        drift_y,
        time_in_bin,
        i_star,
        j_star,
        title,
    )
    _plot_markov_arrows(
        arrows_png_path,
        x_edges,
        y_edges,
        x_centers,
        y_centers,
        occ,
        drift_x,
        drift_y,
        time_in_bin,
        trajectories,
        i_star,
        j_star,
        args.m,
        policy,
        title,
    )
    _plot_occupancy_star(
        occ_star_png_path,
        x_edges,
        y_edges,
        x_centers,
        y_centers,
        occ,
        i_star,
        j_star,
        args.m,
        title,
    )

    print(f"[{policy}] Wrote:")
    print(f"  {csv_path.name}")
    print(f"  {summary_csv_path.name}")
    print(f"  {table_csv_path.name}")
    if HAS_MPL:
        print(f"  {png_path.name}")
        print(f"  {interp_png_path.name}")
        print(f"  {arrows_png_path.name}")
        print(f"  {occ_star_png_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate drift/nullclines for (H_t, E_t) CTMC.")
    parser.add_argument(
        "--policy",
        type=str,
        default="TAP",
        help="One of {greedy, patient, TAG, TAP} or 'all'.",
    )
    parser.add_argument("--m", type=int, default=5000)
    parser.add_argument("--d1", type=float, default=10.0)
    parser.add_argument("--d2", type=float, default=1000.0)
    parser.add_argument("--lambda-ent", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=10.0)
    parser.add_argument("--horizon", type=float, default=120.0)
    parser.add_argument("--burn-in", type=float, default=20.0)
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--bins", type=int, default=35)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--out-dir", type=Path, default=Path("simulation_outputs"))
    args = parser.parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be >= 1")
    if args.bins < 8:
        raise ValueError("--bins should be >= 8")
    if args.horizon <= args.burn_in:
        raise ValueError("--horizon must exceed --burn-in")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    policy_token = args.policy.strip()
    if policy_token.lower() == "all":
        policies = list(POLICIES)
    elif policy_token in POLICIES:
        policies = [policy_token]
    else:
        raise ValueError(f"--policy must be one of {POLICIES} or 'all'.")

    for policy in policies:
        _run_for_policy(args, policy)


if __name__ == "__main__":
    main()
