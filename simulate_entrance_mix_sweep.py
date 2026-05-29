# -*- coding: utf-8 -*-
"""Sweep the easy (E) versus hard (H) entrant intensity split in the two-type CTMC.

Economic interpretation
------------------------
Entrants arrive as competing Poisson processes. In the symmetric baseline (`simulate_easy_departure_sweep.py`),
hard (H) and easy (E) types each arrive at rate m/2, so total entrant intensity is m per unit time.

Here we retain total intensity m but skew the composition: fix lambda_ent ∈ (0, 1), so the **rates** are:

  - H entrants: (1 − lambda_ent) * m
  - E entrants: lambda_ent * m

(This is shorthand for “fraction lambda_ent of the total rate m”: over an interval [0, T], the **expected**
count of easy entrants is lambda_ent·m·T—not a deterministic cap.) Thinning one Poisson(m) arrival stream
into easy vs hard is equivalent.

At lambda_ent = 1/2 this matches the symmetric m/2 + m/2 construction. Compatibility probabilities
p_he = d1/(2m), p_ee = d2/(2m) are unchanged—they still encode market size on the Erdős–Rényi graphs.

Departures mirror the easy-departure simulator: H criticality rate equals the H pool count; easy agents
criticality rate is lambda_E multiplied by the E pool count. This sweep **varies lambda_E jointly with
lambda_ent** on a Cartesian grid—see constants `E_DEPARTURE_RATES` and `EASY_ENTRANT_SHARES`.

Optional output with **matplotlib** installed: a heatmap PNG with the same basename as the grid CSV and the
suffix `_heatmap_means.png` (three rows: total/H/E mean perishing; four columns: policies; shared scale per row).

Normalization
--------------
Report perishing normalized by m*T (same denominator as other scripts) across both lambda_ent and lambda_E.
Total arrival intensity is always m per unit time.

Fluid limit (notebook)
----------------------
In `dynamic_matching_markets_from_when_to_whom.py`, symmetric entrant arrivals appear as 0.5 terms in drift;
replacing those with (1 − lambda_ent) and lambda_ent yields the analogous ODE analogue of this generalized entry mix.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulate_pool_trajectories import (
    POLICIES,
    POLICY_COLORS,
    POLICY_LABELS,
    font,
    hex_to_rgb,
    nice_upper_bound,
    one_minus_power,
    svg_polyline,
)

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - SVG output still works without Pillow.
    Image = None
    ImageDraw = None


M = 10000
D1 = 10.0
D2 = 1000.0
T_HORIZON = 100.0
N_RUNS = 20
BASE_SEED = 20260518
COARSE_E_DEPARTURE_RATES = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0)
COARSE_EASY_ENTRANT_SHARES = tuple(float(x) for x in np.linspace(0.1, 0.9, 9))
DENSE_E_DEPARTURE_RATES = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0)
DENSE_EASY_ENTRANT_SHARES = tuple(float(x) for x in np.linspace(0.05, 0.95, 19))

E_DEPARTURE_RATES = DENSE_E_DEPARTURE_RATES
EASY_ENTRANT_SHARES = DENSE_EASY_ENTRANT_SHARES


def _seq_token(values: tuple[float, ...]) -> str:
    return "_".join(f"{v:g}".replace(".", "p") for v in values)


def build_stem_grid() -> str:
    return (
        f"entrance_departure_grid_m{M}_d1_{D1:g}_d2_{D2:g}_"
        f"lambdaE_{_seq_token(E_DEPARTURE_RATES)}_"
        f"lambdaEnt_{_seq_token(EASY_ENTRANT_SHARES)}_"
        f"t{T_HORIZON:g}_runs{N_RUNS}"
    )


STEM_GRID = build_stem_grid()


@dataclass(frozen=True)
class FinalLoss:
    total: float
    h: float
    e: float


@dataclass(frozen=True)
class LossPoint:
    total_mean: float
    total_std: float
    h_mean: float
    h_std: float
    e_mean: float
    e_std: float


GridResults = dict[tuple[float, float], dict[str, LossPoint]]


def _fname_param(x: float) -> str:
    """Safe filename fragment for numeric parameters."""
    return f"{x:g}".replace(".", "p")


def slice_by_fixed_lambda_e(grid: GridResults, lam_e: float) -> dict[float, dict[str, LossPoint]]:
    return {lam_ent: grid[(lam_e, lam_ent)] for lam_ent in EASY_ENTRANT_SHARES}


def slice_by_fixed_lambda_ent(grid: GridResults, lam_ent: float) -> dict[float, dict[str, LossPoint]]:
    return {lam_e: grid[(lam_e, lam_ent)] for lam_e in E_DEPARTURE_RATES}


def simulate_final_loss(
    m: int,
    d1: float,
    d2: float,
    lambda_ent: float,
    e_departure_rate: float,
    policy: str,
    horizon: float,
    seed: int,
) -> FinalLoss:
    """Run one exact CTMC path and return final losses normalized by m*T."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}")
    if not 0.0 < lambda_ent < 1.0:
        raise ValueError("easy entrant share lambda_ent must lie strictly between 0 and 1.")
    if e_departure_rate <= 0.0:
        raise ValueError("E departure rate must be positive.")

    p_he = d1 / (2.0 * m)
    p_ee = d2 / (2.0 * m)
    if not (0.0 <= p_he <= 1.0 and 0.0 <= p_ee <= 1.0):
        raise ValueError("Need d1/(2m) and d2/(2m) in [0, 1].")

    rng = np.random.default_rng(seed)
    rate_h = m * (1.0 - lambda_ent)
    rate_e = m * lambda_ent

    h_pool = 0
    e_pool = 0
    perish_h = 0
    perish_e = 0
    t = 0.0

    while t < horizon:
        rate_h_crit = h_pool
        rate_e_crit = e_departure_rate * e_pool
        rate_total = rate_h + rate_e + rate_h_crit + rate_e_crit
        t_next = t + rng.exponential(1.0 / rate_total)
        if t_next > horizon:
            break
        t = t_next

        u = rng.random() * rate_total
        if u < rate_h:
            # H arrival.
            if policy in ("greedy", "TAG") and e_pool > 0:
                if rng.random() < one_minus_power(p_he, e_pool):
                    e_pool -= 1
                else:
                    h_pool += 1
            else:
                h_pool += 1

        elif u < rate_h + rate_e:
            # E arrival.
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
            # Criticality/departure event.
            u2 = u - rate_h - rate_e
            if u2 < rate_h_crit:
                # H critical.
                if policy in ("greedy", "TAG"):
                    h_pool -= 1
                    perish_h += 1
                else:
                    if e_pool > 0 and rng.random() < one_minus_power(p_he, e_pool):
                        h_pool -= 1
                        e_pool -= 1
                    else:
                        h_pool -= 1
                        perish_h += 1
            else:
                # E critical.
                if policy in ("greedy", "TAG"):
                    e_pool -= 1
                    perish_e += 1
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
                        perish_e += 1
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
                            perish_e += 1

    denominator = m * horizon
    return FinalLoss(
        total=(perish_h + perish_e) / denominator,
        h=perish_h / denominator,
        e=perish_e / denominator,
    )


def summarize(losses: list[FinalLoss]) -> LossPoint:
    total = np.array([loss.total for loss in losses])
    h = np.array([loss.h for loss in losses])
    e = np.array([loss.e for loss in losses])
    return LossPoint(
        total_mean=float(total.mean()),
        total_std=float(total.std(ddof=1)),
        h_mean=float(h.mean()),
        h_std=float(h.std(ddof=1)),
        e_mean=float(e.mean()),
        e_std=float(e.std(ddof=1)),
    )


def run_sweep() -> GridResults:
    """Full Cartesian sweep over `(lambda_E, lambda_ent)`."""
    results: GridResults = {}
    for e_idx, lam_e in enumerate(E_DEPARTURE_RATES):
        for share_index, lam_ent in enumerate(EASY_ENTRANT_SHARES):
            print(
                f"Grid run: m={M}, d1={D1:g}, d2={D2:g}, T={T_HORIZON:g}, "
                f"lambda_E={lam_e:g}, lambda_ent={lam_ent:g}, {N_RUNS} runs/policy"
            )
            key = (lam_e, lam_ent)
            results[key] = {}
            for policy_index, policy in enumerate(POLICIES):
                losses = []
                for run_idx in range(N_RUNS):
                    seed = (
                        BASE_SEED
                        + e_idx * 500_000
                        + share_index * 6_000
                        + policy_index * 110
                        + run_idx
                    )
                    losses.append(
                        simulate_final_loss(
                            M, D1, D2, lam_ent, lam_e, policy, T_HORIZON, seed
                        )
                    )
                results[key][policy] = summarize(losses)
    return results


def write_csv(path: Path, grid: GridResults) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "m",
                "d1",
                "d2",
                "T",
                "n_runs",
                "lambda_E",
                "easy_entrant_share",
                "policy",
                "total_perish_over_mT_mean",
                "total_perish_over_mT_std",
                "H_perish_over_mT_mean",
                "H_perish_over_mT_std",
                "E_perish_over_mT_mean",
                "E_perish_over_mT_std",
            ]
        )
        for lam_e in E_DEPARTURE_RATES:
            for lam_ent in EASY_ENTRANT_SHARES:
                for policy in POLICIES:
                    point = grid[(lam_e, lam_ent)][policy]
                    writer.writerow(
                        [
                            M,
                            f"{D1:g}",
                            f"{D2:g}",
                            f"{T_HORIZON:g}",
                            N_RUNS,
                            f"{lam_e:g}",
                            f"{lam_ent:g}",
                            policy,
                            f"{point.total_mean:.8f}",
                            f"{point.total_std:.8f}",
                            f"{point.h_mean:.8f}",
                            f"{point.h_std:.8f}",
                            f"{point.e_mean:.8f}",
                            f"{point.e_std:.8f}",
                        ]
                    )


PANEL_SPECS = (
    ("Total perishing", "total_mean", "total_std", "Total perished / (mT)"),
    ("Hard-to-match H perishing", "h_mean", "h_std", "H perished / (mT)"),
    ("Easy-to-match E perishing", "e_mean", "e_std", "E perished / (mT)"),
)


def panel_ymax_slice(
    x_keys: tuple[float, ...],
    results: dict[float, dict[str, LossPoint]],
    mean_name: str,
    std_name: str,
) -> float:
    values = []
    for xv in x_keys:
        for policy in POLICIES:
            point = results[xv][policy]
            values.append(getattr(point, mean_name) + getattr(point, std_name))
    return nice_upper_bound(max(values) * 1.12)


def x_positions_general(
    x0: float, width: float, x_values: tuple[float, ...], x_scale: str
) -> dict[float, float]:
    if len(x_values) == 1:
        return {x_values[0]: x0 + width / 2.0}

    if x_scale == "linear":
        x_min = float(min(x_values))
        x_span = float(max(x_values) - x_min)
        return {xv: x0 + width * (xv - x_min) / x_span for xv in x_values}

    if x_scale != "log":
        raise ValueError("x_scale must be 'log' or 'linear'.")

    log_x = np.log(np.array(x_values, dtype=float))
    ln_min = float(log_x.min())
    ln_span = float(log_x.max() - ln_min)
    return {xv: x0 + width * (float(np.log(xv)) - ln_min) / ln_span for xv in x_values}


def label_entrant_share_axis(x_scale: str) -> str:
    return "easy entrant share λ_ent (log scale)" if x_scale == "log" else "easy entrant share λ_ent"


def label_lambda_e_axis(x_scale: str) -> str:
    return "E departure rate λ_E (log scale)" if x_scale == "log" else "E departure rate λ_E"


def _write_svg_three_panel(
    path: Path,
    x_keys: tuple[float, ...],
    results: dict[float, dict[str, LossPoint]],
    x_scale: str,
    title: str,
    subtitle: str,
    x_tick_label: Callable[[float], str],
    axis_footnote: str,
) -> None:
    width = 1500
    height = 520
    left = 82
    top = 130
    panel_w = 390
    panel_h = 250
    gap = 72

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#17202a}"
        ".title{font-size:25px;font-weight:700}"
        ".subtitle{font-size:14px;fill:#52606d}"
        ".panel-title{font-size:16px;font-weight:700}"
        ".axis{stroke:#26323f;stroke-width:1}"
        ".grid{stroke:#e3e8ef;stroke-width:1}"
        ".tick{font-size:12px;fill:#667080}"
        ".legend{font-size:13px;fill:#17202a}",
        "</style>",
        f'<text x="{left}" y="42" class="title">{title}</text>',
        f'<text x="{left}" y="68" class="subtitle">{subtitle}</text>',
    ]

    legend_x = left + 930
    legend_y = 38
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 170
        y = legend_y + (i // 2) * 24
        color = POLICY_COLORS[policy]
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 36}" y="{y + 4}" class="legend">{POLICY_LABELS[policy]}</text>')

    for panel_idx, (pane_title, mean_name, std_name, y_label) in enumerate(PANEL_SPECS):
        x0 = left + panel_idx * (panel_w + gap)
        y0 = top
        ymax = panel_ymax_slice(x_keys, results, mean_name, std_name)
        xpos = x_positions_general(x0, panel_w, x_keys, x_scale)

        parts.append(f'<text x="{x0}" y="{y0 - 28}" class="panel-title">{pane_title}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" class="axis"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" class="axis"/>')
        parts.append(
            f'<text x="{x0 - 58}" y="{y0 + panel_h / 2}" transform="rotate(-90 {x0 - 58} {y0 + panel_h / 2})" '
            f'text-anchor="middle" class="subtitle">{y_label}</text>'
        )

        for frac in (0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h - panel_h * frac
            value = ymax * frac
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + panel_w}" y2="{y}" class="grid"/>')
            parts.append(f'<text x="{x0 - 12}" y="{y + 4}" text-anchor="end" class="tick">{value:.2f}</text>')

        for xv in x_keys:
            x = xpos[xv]
            parts.append(f'<line x1="{x}" y1="{y0 + panel_h}" x2="{x}" y2="{y0 + panel_h + 5}" class="axis"/>')
            parts.append(
                f'<text x="{x}" y="{y0 + panel_h + 22}" text-anchor="middle" class="tick">{x_tick_label(xv)}</text>'
            )
        parts.append(
            f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 48}" text-anchor="middle" class="subtitle">'
            f"{axis_footnote}</text>"
        )

        for policy in POLICIES:
            color = POLICY_COLORS[policy]
            upper = []
            lower = []
            line_pts = []
            for xv in x_keys:
                point = results[xv][policy]
                mean = getattr(point, mean_name)
                std = getattr(point, std_name)
                x = xpos[xv]
                y_mean = y0 + panel_h - panel_h * mean / ymax
                y_upper = y0 + panel_h - panel_h * (mean + std) / ymax
                y_lower = y0 + panel_h - panel_h * max(0.0, mean - std) / ymax
                line_pts.append((x, y_mean))
                upper.append((x, y_upper))
                lower.append((x, y_lower))
            band = upper + list(reversed(lower))
            parts.append(f'<polygon points="{svg_polyline(band)}" fill="{color}" opacity="0.22"/>')
            parts.append(f'<polyline points="{svg_polyline(line_pts)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            for x, y in line_pts:
                parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_png_three_panel(
    path: Path,
    x_keys: tuple[float, ...],
    results: dict[float, dict[str, LossPoint]],
    x_scale: str,
    title: str,
    subtitle: str,
    x_tick_label: Callable[[float], str],
    axis_footnote: str,
) -> None:
    if Image is None or ImageDraw is None:
        return

    scale_f = 2
    width = 1500
    height = 520
    img = Image.new("RGB", (width * scale_f, height * scale_f), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    def sc(v: float) -> int:
        return int(round(v * scale_f))

    def text(
        x: float,
        y: float,
        value: str,
        fill=(23, 32, 42),
        size=14,
        bold=False,
        anchor=None,
    ):
        draw.text((sc(x), sc(y)), value, fill=fill + (255,), font=font(size * scale_f, bold), anchor=anchor)

    def rotated_text(cx: float, cy: float, value: str, fill=(82, 96, 109), size=14):
        fnt = font(size * scale_f)
        bbox = draw.textbbox((0, 0), value, font=fnt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tile = Image.new("RGBA", (tw + sc(8), th + sc(8)), (255, 255, 255, 0))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((sc(4), sc(4)), value, fill=fill + (255,), font=fnt)
        rotated = tile.rotate(90, expand=True)
        img.paste(rotated, (sc(cx) - rotated.width // 2, sc(cy) - rotated.height // 2), rotated)

    left = 82
    top = 130
    panel_w = 390
    panel_h = 250
    gap = 72
    axis_rgb = (38, 50, 63, 255)
    grid_rgb = (227, 232, 239, 255)
    tick_rgb = (102, 112, 128)

    text(left, 24, title, size=25, bold=True)
    text(left, 54, subtitle, fill=(82, 96, 109), size=14)

    legend_x = left + 930
    legend_y = 38
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 170
        y = legend_y + (i // 2) * 24
        rgb_pt = hex_to_rgb(POLICY_COLORS[policy])
        draw.line((sc(x), sc(y), sc(x + 28), sc(y)), fill=rgb_pt + (255,), width=sc(3))
        text(x + 36, y - 8, POLICY_LABELS[policy], size=13)

    for panel_idx, (pane_title, mean_name, std_name, y_label) in enumerate(PANEL_SPECS):
        x0 = left + panel_idx * (panel_w + gap)
        y0 = top
        ymax = panel_ymax_slice(x_keys, results, mean_name, std_name)
        xpos = x_positions_general(x0, panel_w, x_keys, x_scale)

        text(x0, y0 - 42, pane_title, size=16, bold=True)
        draw.line((sc(x0), sc(y0 + panel_h), sc(x0 + panel_w), sc(y0 + panel_h)), fill=axis_rgb, width=sc(1))
        draw.line((sc(x0), sc(y0), sc(x0), sc(y0 + panel_h)), fill=axis_rgb, width=sc(1))
        rotated_text(x0 - 58, y0 + panel_h / 2, y_label)

        for frac in (0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h - panel_h * frac
            value_y = ymax * frac
            draw.line((sc(x0), sc(y), sc(x0 + panel_w), sc(y)), fill=grid_rgb, width=sc(1))
            text(x0 - 12, y - 8, f"{value_y:.2f}", fill=tick_rgb[:3], size=12, anchor="ra")

        for xv in x_keys:
            x = xpos[xv]
            draw.line((sc(x), sc(y0 + panel_h), sc(x), sc(y0 + panel_h + 5)), fill=axis_rgb, width=sc(1))
            text(x, y0 + panel_h + 8, x_tick_label(xv), fill=tick_rgb[:3], size=12, anchor="ma")
        text(x0 + panel_w / 2, y0 + panel_h + 34, axis_footnote, fill=(82, 96, 109), size=14, anchor="ma")

        for policy in POLICIES:
            rgb_pt = hex_to_rgb(POLICY_COLORS[policy])
            upper = []
            lower = []
            line_pts = []
            for xv in x_keys:
                point = results[xv][policy]
                mean = getattr(point, mean_name)
                std = getattr(point, std_name)
                x = xpos[xv]
                y_mean = y0 + panel_h - panel_h * mean / ymax
                y_upper = y0 + panel_h - panel_h * (mean + std) / ymax
                y_lower = y0 + panel_h - panel_h * max(0.0, mean - std) / ymax
                line_pts.append((x, y_mean))
                upper.append((x, y_upper))
                lower.append((x, y_lower))
            draw.polygon([(sc(a), sc(b)) for a, b in upper + list(reversed(lower))], fill=rgb_pt + (56,))
            draw.line([(sc(a), sc(b)) for a, b in line_pts], fill=rgb_pt + (255,), width=sc(2.4), joint="curve")
            for xp, yp in line_pts:
                draw.ellipse((sc(xp - 3.5), sc(yp - 3.5), sc(xp + 3.5), sc(yp + 3.5)), fill=rgb_pt + (255,))

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def _metric_matrix(grid: GridResults, policy: str, mean_attr: str) -> np.ndarray:
    """Rows: lambda_E (increasing down in `E_DEPARTURE_RATES` order), cols: lambda_ent."""
    lam_e_list = list(E_DEPARTURE_RATES)
    lam_ent_list = list(EASY_ENTRANT_SHARES)
    z = np.zeros((len(lam_e_list), len(lam_ent_list)), dtype=float)
    for i, le in enumerate(lam_e_list):
        for j, lent in enumerate(lam_ent_list):
            z[i, j] = float(getattr(grid[(le, lent)][policy], mean_attr))
    return z


def write_heatmap_figure(out_dir: Path, grid: GridResults) -> None:
    """3x4 heatmaps: rows = total / H / E mean perish; cols = policies. Optional matplotlib."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import colors
    except ImportError:  # pragma: no cover
        print("matplotlib not installed; skipping heatmap figure (pip install matplotlib).")
        return

    metrics = (
        ("total_mean", "Total perished / (mT)"),
        ("h_mean", "H perished / (mT)"),
        ("e_mean", "E perished / (mT)"),
    )
    n_e = len(E_DEPARTURE_RATES)
    n_ent = len(EASY_ENTRANT_SHARES)

    fig, axes = plt.subplots(3, len(POLICIES), figsize=(18, 10), constrained_layout=True)
    fig.suptitle(
        f"Perishing heatmaps  (m={M}, d1={D1:g}, d2={D2:g}, T={T_HORIZON:g}, {N_RUNS} runs/cell; "
        f"H crit rate 1, E crit rate λ_E×pool)",
        fontsize=14,
    )

    x_labels = [f"{x:.2g}" for x in EASY_ENTRANT_SHARES]
    y_labels = [f"{y:g}" for y in E_DEPARTURE_RATES]

    for r, (attr, row_label) in enumerate(metrics):
        mats = [_metric_matrix(grid, pol, attr) for pol in POLICIES]
        vmin = float(min(m.min() for m in mats))
        vmax = float(max(m.max() for m in mats))
        if vmin >= vmax:
            vmax = vmin + 1e-15
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        mpl_last = None
        for c, pol in enumerate(POLICIES):
            ax = axes[r, c]
            mpl_last = ax.imshow(mats[c], origin="lower", aspect="auto", cmap="viridis", norm=norm)
            ax.set_xticks(range(n_ent))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n_e))
            ax.set_yticklabels(y_labels, fontsize=8)
            if r == 0:
                ax.set_title(POLICY_LABELS[pol], fontsize=11)
            if r == len(metrics) - 1:
                ax.set_xlabel("easy entrant share λ_ent (columns)", fontsize=9)
            if c == 0:
                ax.set_ylabel(row_label + "\n(rows: λ_E)", fontsize=9)
            else:
                ax.set_yticklabels([])
        fig.colorbar(
            mpl_last,
            ax=list(axes[r, :].flat),
            shrink=0.82,
            label=row_label,
        )

    out_path = out_dir / f"{STEM_GRID}_heatmap_means.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote heatmap: {out_path.name}")


def write_slice_figures(out_dir: Path, grid: GridResults) -> None:
    subtitle_base = (
        f"final perished / (mT), m={M}, T={T_HORIZON:g}, d1={D1:g}, d2={D2:g}, {N_RUNS} runs/point; "
        f"entrant intensity total = m; H departure coefficient = 1"
    )

    prefix = STEM_GRID + "__slice"

    for lam_e in E_DEPARTURE_RATES:
        frag_e = _fname_param(lam_e)
        slim = slice_by_fixed_lambda_e(grid, lam_e)
        stem = out_dir / f"{prefix}_lambdaE_{frag_e}"
        subtitle = subtitle_base + f"; λ_E = {lam_e:g} fixed"

        title_ent = "Policy performance as easy entrant share rises"
        stem_lin = stem.with_name(stem.name + "_vs_lambdaEnt_lin")

        _write_svg_three_panel(
            stem_lin.with_suffix(".svg"),
            EASY_ENTRANT_SHARES,
            slim,
            "linear",
            title_ent,
            subtitle,
            lambda xv: f"{xv:.2g}",
            label_entrant_share_axis("linear"),
        )
        _write_png_three_panel(
            stem_lin.with_suffix(".png"),
            EASY_ENTRANT_SHARES,
            slim,
            "linear",
            title_ent,
            subtitle,
            lambda xv: f"{xv:.2g}",
            label_entrant_share_axis("linear"),
        )
        stem_log = stem.with_name(stem.name + "_vs_lambdaEnt_log_x")
        _write_svg_three_panel(
            stem_log.with_suffix(".svg"),
            EASY_ENTRANT_SHARES,
            slim,
            "log",
            title_ent,
            subtitle,
            lambda xv: f"{xv:.2g}",
            label_entrant_share_axis("log"),
        )
        _write_png_three_panel(
            stem_log.with_suffix(".png"),
            EASY_ENTRANT_SHARES,
            slim,
            "log",
            title_ent,
            subtitle,
            lambda xv: f"{xv:.2g}",
            label_entrant_share_axis("log"),
        )

    title_dep = "Policy performance as E departure rate rises"
    for lam_ent in EASY_ENTRANT_SHARES:
        frag_ent = _fname_param(lam_ent)
        slim = slice_by_fixed_lambda_ent(grid, lam_ent)
        stem = out_dir / f"{prefix}_lambdaEnt_{frag_ent}"
        subtitle = subtitle_base + f"; λ_ent = {lam_ent:g} fixed"

        stem_log_e = stem.with_name(stem.name + "_vs_lambdaE_log")

        _write_svg_three_panel(
            stem_log_e.with_suffix(".svg"),
            E_DEPARTURE_RATES,
            slim,
            "log",
            title_dep,
            subtitle,
            lambda xv: f"{xv:g}",
            label_lambda_e_axis("log"),
        )
        _write_png_three_panel(
            stem_log_e.with_suffix(".png"),
            E_DEPARTURE_RATES,
            slim,
            "log",
            title_dep,
            subtitle,
            lambda xv: f"{xv:g}",
            label_lambda_e_axis("log"),
        )
        stem_lin_e = stem.with_name(stem.name + "_vs_lambdaE_lin")
        _write_svg_three_panel(
            stem_lin_e.with_suffix(".svg"),
            E_DEPARTURE_RATES,
            slim,
            "linear",
            title_dep,
            subtitle,
            lambda xv: f"{xv:g}",
            label_lambda_e_axis("linear"),
        )
        _write_png_three_panel(
            stem_lin_e.with_suffix(".png"),
            E_DEPARTURE_RATES,
            slim,
            "linear",
            title_dep,
            subtitle,
            lambda xv: f"{xv:g}",
            label_lambda_e_axis("linear"),
        )


def main() -> None:
    global N_RUNS, E_DEPARTURE_RATES, EASY_ENTRANT_SHARES, STEM_GRID

    parser = argparse.ArgumentParser(
        description="Run entrance/departure grid sweep and write aggregate plots."
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=N_RUNS,
        help="Monte Carlo runs per (lambda_E, lambda_ent, policy) cell.",
    )
    parser.add_argument(
        "--coarse-grid",
        action="store_true",
        help="Use original coarse lambda_E/lambda_ent grids for faster tests.",
    )
    parser.add_argument(
        "--with-slices",
        action="store_true",
        help="Also write the large bundle of slice PNG/SVG figures (off by default).",
    )
    args = parser.parse_args()
    if args.n_runs < 2:
        raise ValueError("--n-runs must be >= 2 for standard-deviation estimates.")

    N_RUNS = args.n_runs
    if args.coarse_grid:
        E_DEPARTURE_RATES = COARSE_E_DEPARTURE_RATES
        EASY_ENTRANT_SHARES = COARSE_EASY_ENTRANT_SHARES
    else:
        E_DEPARTURE_RATES = DENSE_E_DEPARTURE_RATES
        EASY_ENTRANT_SHARES = DENSE_EASY_ENTRANT_SHARES
    STEM_GRID = build_stem_grid()

    out_dir = Path("simulation_outputs")
    out_dir.mkdir(exist_ok=True)
    grid = run_sweep()
    write_csv(out_dir / f"{STEM_GRID}.csv", grid)
    write_heatmap_figure(out_dir, grid)
    if args.with_slices:
        write_slice_figures(out_dir, grid)

    print(f"\nWrote grid CSV: {STEM_GRID}.csv")
    print(f"Heatmap summary (needs matplotlib): {STEM_GRID}_heatmap_means.png")
    if args.with_slices:
        print(f"Line slices under {STEM_GRID}__slice_lambdaE_* and __slice_lambdaEnt_*")
    else:
        print("Slices skipped (use --with-slices to generate them).")


if __name__ == "__main__":
    main()
