# -*- coding: utf-8 -*-
"""Sweep the E-agent departure/criticality rate in the two-type CTMC.

H agents have criticality rate 1. E agents have criticality rate lambda_E.
The output plots final cumulative perishing at horizon T, normalized by m*T,
as lambda_E increases.
"""

from __future__ import annotations

import csv
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
N_RUNS = 5
BASE_SEED = 20260517
E_DEPARTURE_RATES = (1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0)
STEM = "easy_departure_sweep_m10000_d1_10_d2_1000_lambdaE_1_2_5_10_20_40_100_t100_runs5"


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


def simulate_final_loss(
    m: int,
    d1: float,
    d2: float,
    e_departure_rate: float,
    policy: str,
    horizon: float,
    seed: int,
) -> FinalLoss:
    """Run one exact CTMC path and return final losses normalized by m*T."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}")
    if e_departure_rate <= 0.0:
        raise ValueError("E departure rate must be positive.")

    p_he = d1 / (2.0 * m)
    p_ee = d2 / (2.0 * m)
    if not (0.0 <= p_he <= 1.0 and 0.0 <= p_ee <= 1.0):
        raise ValueError("Need d1/(2m) and d2/(2m) in [0, 1].")

    rng = np.random.default_rng(seed)
    rate_arr = m / 2.0
    h_pool = 0
    e_pool = 0
    perish_h = 0
    perish_e = 0
    t = 0.0

    while t < horizon:
        rate_h_crit = h_pool
        rate_e_crit = e_departure_rate * e_pool
        rate_total = 2.0 * rate_arr + rate_h_crit + rate_e_crit
        t_next = t + rng.exponential(1.0 / rate_total)
        if t_next > horizon:
            break
        t = t_next

        u = rng.random() * rate_total
        if u < rate_arr:
            # H arrival.
            if policy in ("greedy", "TAG") and e_pool > 0:
                if rng.random() < one_minus_power(p_he, e_pool):
                    e_pool -= 1
                else:
                    h_pool += 1
            else:
                h_pool += 1

        elif u < 2.0 * rate_arr:
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
            u2 = u - 2.0 * rate_arr
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
                else:
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


def run_sweep() -> dict[float, dict[str, LossPoint]]:
    results: dict[float, dict[str, LossPoint]] = {}
    for rate_index, e_rate in enumerate(E_DEPARTURE_RATES):
        print(
            f"Running E departure sweep: m={M}, d1={D1:g}, d2={D2:g}, "
            f"T={T_HORIZON:g}, lambda_E={e_rate:g}, runs={N_RUNS}"
        )
        results[e_rate] = {}
        for policy_index, policy in enumerate(POLICIES):
            losses = []
            for run_idx in range(N_RUNS):
                seed = BASE_SEED + 10000 * rate_index + 1000 * policy_index + run_idx
                losses.append(simulate_final_loss(M, D1, D2, e_rate, policy, T_HORIZON, seed))
            results[e_rate][policy] = summarize(losses)
    return results


def write_csv(path: Path, results: dict[float, dict[str, LossPoint]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "m",
                "d1",
                "d2",
                "T",
                "n_runs",
                "E_departure_rate",
                "policy",
                "total_perish_over_mT_mean",
                "total_perish_over_mT_std",
                "H_perish_over_mT_mean",
                "H_perish_over_mT_std",
                "E_perish_over_mT_mean",
                "E_perish_over_mT_std",
            ]
        )
        for e_rate in E_DEPARTURE_RATES:
            for policy in POLICIES:
                point = results[e_rate][policy]
                writer.writerow(
                    [
                        M,
                        f"{D1:g}",
                        f"{D2:g}",
                        f"{T_HORIZON:g}",
                        N_RUNS,
                        f"{e_rate:g}",
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


def panel_ymax(results: dict[float, dict[str, LossPoint]], mean_name: str, std_name: str) -> float:
    values = []
    for e_rate in E_DEPARTURE_RATES:
        for policy in POLICIES:
            point = results[e_rate][policy]
            values.append(getattr(point, mean_name) + getattr(point, std_name))
    return nice_upper_bound(max(values) * 1.12)


def x_positions(x0: float, width: float, x_scale: str = "log") -> dict[float, float]:
    if len(E_DEPARTURE_RATES) == 1:
        return {E_DEPARTURE_RATES[0]: x0 + width / 2.0}

    if x_scale == "linear":
        rate_min = float(min(E_DEPARTURE_RATES))
        rate_range = float(max(E_DEPARTURE_RATES) - rate_min)
        return {
            e_rate: x0 + width * (e_rate - rate_min) / rate_range
            for e_rate in E_DEPARTURE_RATES
        }

    if x_scale != "log":
        raise ValueError("x_scale must be 'log' or 'linear'.")

    log_rates = np.log(np.array(E_DEPARTURE_RATES, dtype=float))
    log_min = float(log_rates.min())
    log_range = float(log_rates.max() - log_min)
    return {
        e_rate: x0 + width * (float(np.log(e_rate)) - log_min) / log_range
        for e_rate in E_DEPARTURE_RATES
    }


def x_axis_label(x_scale: str) -> str:
    return "E departure rate (log scale)" if x_scale == "log" else "E departure rate"


def write_svg(path: Path, results: dict[float, dict[str, LossPoint]], x_scale: str = "log") -> None:
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
        f'<text x="{left}" y="42" class="title">Policy performance as E departure rate increases</text>',
        f'<text x="{left}" y="68" class="subtitle">final perished / (mT), m={M}, T={T_HORIZON:g}, d1={D1:g}, d2={D2:g}, {N_RUNS} runs; H departure rate = 1</text>',
    ]

    legend_x = left + 930
    legend_y = 38
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 170
        y = legend_y + (i // 2) * 24
        color = POLICY_COLORS[policy]
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 36}" y="{y + 4}" class="legend">{POLICY_LABELS[policy]}</text>')

    for panel_idx, (title, mean_name, std_name, y_label) in enumerate(PANEL_SPECS):
        x0 = left + panel_idx * (panel_w + gap)
        y0 = top
        ymax = panel_ymax(results, mean_name, std_name)
        xpos = x_positions(x0, panel_w, x_scale)

        parts.append(f'<text x="{x0}" y="{y0 - 28}" class="panel-title">{title}</text>')
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

        for e_rate in E_DEPARTURE_RATES:
            x = xpos[e_rate]
            parts.append(f'<line x1="{x}" y1="{y0 + panel_h}" x2="{x}" y2="{y0 + panel_h + 5}" class="axis"/>')
            parts.append(f'<text x="{x}" y="{y0 + panel_h + 22}" text-anchor="middle" class="tick">{e_rate:g}</text>')
        parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 48}" text-anchor="middle" class="subtitle">{x_axis_label(x_scale)}</text>')

        for policy in POLICIES:
            color = POLICY_COLORS[policy]
            upper = []
            lower = []
            line = []
            for e_rate in E_DEPARTURE_RATES:
                point = results[e_rate][policy]
                mean = getattr(point, mean_name)
                std = getattr(point, std_name)
                x = xpos[e_rate]
                y_mean = y0 + panel_h - panel_h * mean / ymax
                y_upper = y0 + panel_h - panel_h * (mean + std) / ymax
                y_lower = y0 + panel_h - panel_h * max(0.0, mean - std) / ymax
                line.append((x, y_mean))
                upper.append((x, y_upper))
                lower.append((x, y_lower))
            band = upper + list(reversed(lower))
            parts.append(f'<polygon points="{svg_polyline(band)}" fill="{color}" opacity="0.22"/>')
            parts.append(f'<polyline points="{svg_polyline(line)}" fill="none" stroke="{color}" stroke-width="2.4"/>')
            for x, y in line:
                parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_png(path: Path, results: dict[float, dict[str, LossPoint]], x_scale: str = "log") -> None:
    if Image is None or ImageDraw is None:
        return

    scale = 2
    width = 1500
    height = 520
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    def sc(v: float) -> int:
        return int(round(v * scale))

    def text(x: float, y: float, value: str, fill=(23, 32, 42), size=14, bold=False, anchor=None):
        draw.text((sc(x), sc(y)), value, fill=fill + (255,), font=font(size * scale, bold), anchor=anchor)

    def rotated_text(cx: float, cy: float, value: str, fill=(82, 96, 109), size=14):
        fnt = font(size * scale)
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
    axis = (38, 50, 63, 255)
    grid = (227, 232, 239, 255)
    tick_fill = (102, 112, 128)

    text(left, 24, "Policy performance as E departure rate increases", size=25, bold=True)
    text(
        left,
        54,
        f"final perished / (mT), m={M}, T={T_HORIZON:g}, d1={D1:g}, d2={D2:g}, {N_RUNS} runs; H departure rate = 1",
        fill=(82, 96, 109),
        size=14,
    )

    legend_x = left + 930
    legend_y = 38
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 170
        y = legend_y + (i // 2) * 24
        rgb = hex_to_rgb(POLICY_COLORS[policy])
        draw.line((sc(x), sc(y), sc(x + 28), sc(y)), fill=rgb + (255,), width=sc(3))
        text(x + 36, y - 8, POLICY_LABELS[policy], size=13)

    for panel_idx, (title, mean_name, std_name, y_label) in enumerate(PANEL_SPECS):
        x0 = left + panel_idx * (panel_w + gap)
        y0 = top
        ymax = panel_ymax(results, mean_name, std_name)
        xpos = x_positions(x0, panel_w, x_scale)

        text(x0, y0 - 42, title, size=16, bold=True)
        draw.line((sc(x0), sc(y0 + panel_h), sc(x0 + panel_w), sc(y0 + panel_h)), fill=axis, width=sc(1))
        draw.line((sc(x0), sc(y0), sc(x0), sc(y0 + panel_h)), fill=axis, width=sc(1))
        rotated_text(x0 - 58, y0 + panel_h / 2, y_label)

        for frac in (0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h - panel_h * frac
            value = ymax * frac
            draw.line((sc(x0), sc(y), sc(x0 + panel_w), sc(y)), fill=grid, width=sc(1))
            text(x0 - 12, y - 8, f"{value:.2f}", fill=tick_fill[:3], size=12, anchor="ra")

        for e_rate in E_DEPARTURE_RATES:
            x = xpos[e_rate]
            draw.line((sc(x), sc(y0 + panel_h), sc(x), sc(y0 + panel_h + 5)), fill=axis, width=sc(1))
            text(x, y0 + panel_h + 8, f"{e_rate:g}", fill=tick_fill[:3], size=12, anchor="ma")
        text(x0 + panel_w / 2, y0 + panel_h + 34, x_axis_label(x_scale), fill=(82, 96, 109), size=14, anchor="ma")

        for policy in POLICIES:
            rgb = hex_to_rgb(POLICY_COLORS[policy])
            upper = []
            lower = []
            line = []
            for e_rate in E_DEPARTURE_RATES:
                point = results[e_rate][policy]
                mean = getattr(point, mean_name)
                std = getattr(point, std_name)
                x = xpos[e_rate]
                y_mean = y0 + panel_h - panel_h * mean / ymax
                y_upper = y0 + panel_h - panel_h * (mean + std) / ymax
                y_lower = y0 + panel_h - panel_h * max(0.0, mean - std) / ymax
                line.append((x, y_mean))
                upper.append((x, y_upper))
                lower.append((x, y_lower))
            draw.polygon([(sc(x), sc(y)) for x, y in upper + list(reversed(lower))], fill=rgb + (56,))
            draw.line([(sc(x), sc(y)) for x, y in line], fill=rgb + (255,), width=sc(2.4), joint="curve")
            for x, y in line:
                draw.ellipse((sc(x - 3.5), sc(y - 3.5), sc(x + 3.5), sc(y + 3.5)), fill=rgb + (255,))

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def main() -> None:
    out_dir = Path("simulation_outputs")
    out_dir.mkdir(exist_ok=True)
    results = run_sweep()
    write_csv(out_dir / f"{STEM}.csv", results)
    write_svg(out_dir / f"{STEM}.svg", results)
    write_png(out_dir / f"{STEM}.png", results)
    write_svg(out_dir / f"{STEM}_linear_x.svg", results, x_scale="linear")
    write_png(out_dir / f"{STEM}_linear_x.png", results, x_scale="linear")

    print("\nFinal perishing, normalized by m*T:")
    for e_rate in E_DEPARTURE_RATES:
        print(f"  lambda_E={e_rate:g}")
        for policy in POLICIES:
            p = results[e_rate][policy]
            print(
                f"    {POLICY_LABELS[policy]:>10}: "
                f"total={p.total_mean:.4f} +/- {p.total_std:.4f}, "
                f"H={p.h_mean:.4f} +/- {p.h_std:.4f}, "
                f"E={p.e_mean:.4f} +/- {p.e_std:.4f}"
            )


if __name__ == "__main__":
    main()
