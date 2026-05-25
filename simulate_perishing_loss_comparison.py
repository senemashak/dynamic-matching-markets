# -*- coding: utf-8 -*-
"""Compare cumulative perishing loss as heterogeneity increases.

The plotted statistic is the policy objective over time:

    cumulative number perished by time t / (m * t)

where total arrivals have expectation m * t.
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
T_HORIZON = 50.0
DT = 0.25
N_RUNS = 10
BASE_SEED = 20260513
EXPERIMENTS = (100.0, 1000.0)


@dataclass(frozen=True)
class LossSummary:
    total_mean: np.ndarray
    total_std: np.ndarray
    h_mean: np.ndarray
    h_std: np.ndarray
    e_mean: np.ndarray
    e_std: np.ndarray


@dataclass(frozen=True)
class ExperimentResult:
    d2: float
    summaries: dict[str, LossSummary]


def normalized_loss(perish_count: int, m: int, t: float) -> float:
    return 0.0 if t <= 0.0 else perish_count / (m * t)


def simulate_loss_path(
    m: int,
    d1: float,
    d2: float,
    policy: str,
    times: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one CTMC path and return normalized total/H/E perish losses."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}")

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
    sample_idx = 0

    total_loss = np.empty(len(times), dtype=float)
    h_loss = np.empty(len(times), dtype=float)
    e_loss = np.empty(len(times), dtype=float)

    while sample_idx < len(times):
        rate_total = 2.0 * rate_arr + h_pool + e_pool
        t_next = t + rng.exponential(1.0 / rate_total)

        while sample_idx < len(times) and times[sample_idx] <= t_next:
            sample_time = float(times[sample_idx])
            h_loss[sample_idx] = normalized_loss(perish_h, m, sample_time)
            e_loss[sample_idx] = normalized_loss(perish_e, m, sample_time)
            total_loss[sample_idx] = normalized_loss(perish_h + perish_e, m, sample_time)
            sample_idx += 1

        if sample_idx >= len(times):
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
            # Criticality event.
            u2 = u - 2.0 * rate_arr
            if u2 < h_pool:
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

    return total_loss, h_loss, e_loss


def summarize_paths(paths: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> LossSummary:
    total = np.vstack([p[0] for p in paths])
    h = np.vstack([p[1] for p in paths])
    e = np.vstack([p[2] for p in paths])
    return LossSummary(
        total_mean=total.mean(axis=0),
        total_std=total.std(axis=0, ddof=1),
        h_mean=h.mean(axis=0),
        h_std=h.std(axis=0, ddof=1),
        e_mean=e.mean(axis=0),
        e_std=e.std(axis=0, ddof=1),
    )


def run_experiment(d2: float, times: np.ndarray) -> ExperimentResult:
    summaries: dict[str, LossSummary] = {}
    experiment_seed = BASE_SEED + int(d2)

    for p_idx, policy in enumerate(POLICIES):
        paths = []
        for run_idx in range(N_RUNS):
            seed = experiment_seed + 1000 * p_idx + run_idx
            paths.append(simulate_loss_path(M, D1, d2, policy, times, seed))
        summaries[policy] = summarize_paths(paths)

    return ExperimentResult(d2=d2, summaries=summaries)


def write_csv(path: Path, times: np.ndarray, results: list[ExperimentResult]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "m",
                "d1",
                "d2",
                "policy",
                "time",
                "total_loss_mean",
                "total_loss_std",
                "H_loss_mean",
                "H_loss_std",
                "E_loss_mean",
                "E_loss_std",
            ]
        )
        for result in results:
            for policy in POLICIES:
                s = result.summaries[policy]
                for i, t in enumerate(times):
                    writer.writerow(
                        [
                            M,
                            f"{D1:g}",
                            f"{result.d2:g}",
                            policy,
                            f"{t:.2f}",
                            f"{s.total_mean[i]:.8f}",
                            f"{s.total_std[i]:.8f}",
                            f"{s.h_mean[i]:.8f}",
                            f"{s.h_std[i]:.8f}",
                            f"{s.e_mean[i]:.8f}",
                            f"{s.e_std[i]:.8f}",
                        ]
                    )


def line_coordinates(
    times: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    x0: float,
    y0: float,
    width: float,
    height: float,
    ymax: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    def sx(t: float) -> float:
        return x0 + width * t / T_HORIZON

    def sy(v: float) -> float:
        return y0 + height - height * max(0.0, min(ymax, v)) / ymax

    upper = mean + std
    lower = np.maximum(0.0, mean - std)
    band = [(sx(t), sy(v)) for t, v in zip(times, upper)]
    band += [(sx(t), sy(v)) for t, v in zip(times[::-1], lower[::-1])]
    line = [(sx(t), sy(v)) for t, v in zip(times, mean)]
    return band, line


def loss_ymax(results: list[ExperimentResult]) -> float:
    max_v = 0.0
    for result in results:
        for s in result.summaries.values():
            max_v = max(max_v, float(np.max(s.total_mean + s.total_std)))
    return nice_upper_bound(max_v * 1.08)


def write_svg(path: Path, times: np.ndarray, results: list[ExperimentResult]) -> None:
    width = 1420
    height = 520
    left = 82
    top = 128
    panel_w = 600
    panel_h = 285
    col_gap = 84
    ymax = loss_ymax(results)

    parts: list[str] = [
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
        f'<text x="{left}" y="42" class="title">Cumulative perishing loss by policy</text>',
        f'<text x="{left}" y="68" class="subtitle">loss(t) = perished by time t / (m t), m={M}, d1={D1:g}, {N_RUNS} runs, t=0..{T_HORIZON:g}</text>',
    ]

    legend_x = left + 710
    legend_y = 38
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 165
        y = legend_y + (i // 2) * 24
        color = POLICY_COLORS[policy]
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 36}" y="{y + 4}" class="legend">{POLICY_LABELS[policy]}</text>')

    for col_idx, result in enumerate(results):
        x0 = left + col_idx * (panel_w + col_gap)
        y0 = top
        parts.append(f'<text x="{x0}" y="{y0 - 24}" class="panel-title">d2 = {result.d2:g}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" class="axis"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" class="axis"/>')

        for frac in (0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h - panel_h * frac
            value = ymax * frac
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + panel_w}" y2="{y}" class="grid"/>')
            if col_idx == 0:
                parts.append(f'<text x="{x0 - 12}" y="{y + 4}" text-anchor="end" class="tick">{value:.2f}</text>')

        for t in (0, 10, 20, 30, 40, 50):
            x = x0 + panel_w * t / T_HORIZON
            parts.append(f'<line x1="{x}" y1="{y0 + panel_h}" x2="{x}" y2="{y0 + panel_h + 5}" class="axis"/>')
            parts.append(f'<text x="{x}" y="{y0 + panel_h + 22}" text-anchor="middle" class="tick">{t}</text>')
        parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 50}" text-anchor="middle" class="subtitle">time</text>')

        for policy in POLICIES:
            s = result.summaries[policy]
            color = POLICY_COLORS[policy]
            band, line = line_coordinates(times, s.total_mean, s.total_std, x0, y0, panel_w, panel_h, ymax)
            parts.append(f'<polygon points="{svg_polyline(band)}" fill="{color}" opacity="0.13"/>')
            parts.append(
                f'<polyline points="{svg_polyline(line)}" fill="none" stroke="{color}" '
                'stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_png(path: Path, times: np.ndarray, results: list[ExperimentResult]) -> None:
    if Image is None or ImageDraw is None:
        return

    scale = 2
    width = 1420
    height = 520
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img, "RGBA")

    def sc(v: float) -> int:
        return int(round(v * scale))

    def text(x: float, y: float, value: str, fill=(23, 32, 42), size=14, bold=False, anchor=None):
        draw.text((sc(x), sc(y)), value, fill=fill + (255,), font=font(size * scale, bold), anchor=anchor)

    left = 82
    top = 128
    panel_w = 600
    panel_h = 285
    col_gap = 84
    ymax = loss_ymax(results)
    axis = (38, 50, 63, 255)
    grid = (227, 232, 239, 255)
    tick_fill = (102, 112, 128)

    text(left, 24, "Cumulative perishing loss by policy", size=25, bold=True)
    text(
        left,
        54,
        f"loss(t) = perished by time t / (m t), m={M}, d1={D1:g}, {N_RUNS} runs, t=0..{T_HORIZON:g}",
        fill=(82, 96, 109),
        size=14,
    )

    legend_x = left + 710
    legend_y = 38
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 165
        y = legend_y + (i // 2) * 24
        rgb = hex_to_rgb(POLICY_COLORS[policy])
        draw.line((sc(x), sc(y), sc(x + 28), sc(y)), fill=rgb + (255,), width=sc(3))
        text(x + 36, y - 8, POLICY_LABELS[policy], size=13)

    for col_idx, result in enumerate(results):
        x0 = left + col_idx * (panel_w + col_gap)
        y0 = top
        text(x0, y0 - 44, f"d2 = {result.d2:g}", size=16, bold=True)
        draw.line((sc(x0), sc(y0 + panel_h), sc(x0 + panel_w), sc(y0 + panel_h)), fill=axis, width=sc(1))
        draw.line((sc(x0), sc(y0), sc(x0), sc(y0 + panel_h)), fill=axis, width=sc(1))

        for frac in (0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_h - panel_h * frac
            value = ymax * frac
            draw.line((sc(x0), sc(y), sc(x0 + panel_w), sc(y)), fill=grid, width=sc(1))
            if col_idx == 0:
                text(x0 - 12, y - 8, f"{value:.2f}", fill=tick_fill[:3], size=12, anchor="ra")

        for t in (0, 10, 20, 30, 40, 50):
            x = x0 + panel_w * t / T_HORIZON
            draw.line((sc(x), sc(y0 + panel_h), sc(x), sc(y0 + panel_h + 5)), fill=axis, width=sc(1))
            text(x, y0 + panel_h + 8, f"{t}", fill=tick_fill[:3], size=12, anchor="ma")
        text(x0 + panel_w / 2, y0 + panel_h + 36, "time", fill=(82, 96, 109), size=14, anchor="ma")

        for policy in POLICIES:
            s = result.summaries[policy]
            rgb = hex_to_rgb(POLICY_COLORS[policy])
            band, line = line_coordinates(times, s.total_mean, s.total_std, x0, y0, panel_w, panel_h, ymax)
            draw.polygon([(sc(x), sc(y)) for x, y in band], fill=rgb + (34,))
            draw.line([(sc(x), sc(y)) for x, y in line], fill=rgb + (255,), width=sc(2.3), joint="curve")

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def main() -> None:
    out_dir = Path("simulation_outputs")
    out_dir.mkdir(exist_ok=True)
    times = np.arange(0.0, T_HORIZON + DT / 2.0, DT)

    results = []
    for d2 in EXPERIMENTS:
        print(f"Running loss paths: m={M}, d1={D1:g}, d2={d2:g}, t={T_HORIZON:g}, runs={N_RUNS}")
        results.append(run_experiment(d2, times))

    stem = "perishing_loss_m10000_d1_10_d2_100_vs_1000_t50"
    write_csv(out_dir / f"{stem}.csv", times, results)
    write_svg(out_dir / f"{stem}.svg", times, results)
    write_png(out_dir / f"{stem}.png", times, results)

    final_idx = -1
    print("\nFinal cumulative loss per m*t at t=50:")
    for result in results:
        print(f"  d2={result.d2:g}")
        for policy in POLICIES:
            s = result.summaries[policy]
            print(
                f"    {POLICY_LABELS[policy]:>10}: "
                f"total={s.total_mean[final_idx]:.4f} +/- {s.total_std[final_idx]:.4f}, "
                f"H={s.h_mean[final_idx]:.4f}, E={s.e_mean[final_idx]:.4f}"
            )


if __name__ == "__main__":
    main()
