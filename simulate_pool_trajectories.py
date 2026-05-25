# -*- coding: utf-8 -*-
"""Simulate pool-size trajectories for the two-type dynamic matching market.

This script runs the finite-m CTMC directly and samples the pool state on a
regular time grid. It writes:

  - pool_trajectory_summary.csv
  - pool_trajectory_m2000_d1_10_d2_100.svg
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - SVG output still works without Pillow.
    Image = None
    ImageDraw = None
    ImageFont = None


M = 2000
D1 = 10.0
D2 = 100.0
N_RUNS = 10
T_HORIZON = 80.0
DT = 0.25
BASE_SEED = 20260511

POLICIES = ("greedy", "patient", "TAG", "TAP")
POLICY_LABELS = {
    "greedy": "Greedy",
    "patient": "Patient",
    "TAG": "TA-Greedy",
    "TAP": "TA-Patient",
}
POLICY_COLORS = {
    "greedy": "#1f77b4",
    "patient": "#d62728",
    "TAG": "#2ca02c",
    "TAP": "#9467bd",
}


def one_minus_power(p: float, n: int) -> float:
    """Stable computation of 1 - (1 - p)^n for integer n >= 0."""
    if n <= 0:
        return 0.0
    return float(-np.expm1(n * np.log1p(-p)))


def simulate_pool_path(
    m: int,
    d1: float,
    d2: float,
    policy: str,
    times: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one CTMC path and return H/E pool counts sampled at `times`."""
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
    t = 0.0
    sample_idx = 0
    h_path = np.empty(len(times), dtype=float)
    e_path = np.empty(len(times), dtype=float)

    while sample_idx < len(times):
        rate_total = 2.0 * rate_arr + h_pool + e_pool
        if rate_total <= 0:
            h_path[sample_idx:] = h_pool
            e_path[sample_idx:] = e_pool
            break

        t_next = t + rng.exponential(1.0 / rate_total)

        while sample_idx < len(times) and times[sample_idx] <= t_next:
            h_path[sample_idx] = h_pool
            e_path[sample_idx] = e_pool
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
                else:
                    if e_pool > 0 and rng.random() < one_minus_power(p_he, e_pool):
                        h_pool -= 1
                        e_pool -= 1
                    else:
                        h_pool -= 1
            else:
                # E critical.
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

    return h_path, e_path


@dataclass
class Summary:
    h_mean: np.ndarray
    h_std: np.ndarray
    e_mean: np.ndarray
    e_std: np.ndarray
    total_mean: np.ndarray
    total_std: np.ndarray


def run_experiment() -> tuple[np.ndarray, dict[str, Summary]]:
    times = np.arange(0.0, T_HORIZON + DT / 2.0, DT)
    summaries: dict[str, Summary] = {}

    for p_idx, policy in enumerate(POLICIES):
        h_runs = []
        e_runs = []
        for run_idx in range(N_RUNS):
            seed = BASE_SEED + 1000 * p_idx + run_idx
            h_path, e_path = simulate_pool_path(M, D1, D2, policy, times, seed)
            h_runs.append(h_path)
            e_runs.append(e_path)

        h_arr = np.vstack(h_runs)
        e_arr = np.vstack(e_runs)
        total_arr = h_arr + e_arr
        summaries[policy] = Summary(
            h_mean=h_arr.mean(axis=0),
            h_std=h_arr.std(axis=0, ddof=1),
            e_mean=e_arr.mean(axis=0),
            e_std=e_arr.std(axis=0, ddof=1),
            total_mean=total_arr.mean(axis=0),
            total_std=total_arr.std(axis=0, ddof=1),
        )

    return times, summaries


def write_csv(path: Path, times: np.ndarray, summaries: dict[str, Summary]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "time",
                "H_mean",
                "H_std",
                "E_mean",
                "E_std",
                "total_mean",
                "total_std",
            ]
        )
        for policy in POLICIES:
            s = summaries[policy]
            for i, t in enumerate(times):
                writer.writerow(
                    [
                        policy,
                        f"{t:.2f}",
                        f"{s.h_mean[i]:.6f}",
                        f"{s.h_std[i]:.6f}",
                        f"{s.e_mean[i]:.6f}",
                        f"{s.e_std[i]:.6f}",
                        f"{s.total_mean[i]:.6f}",
                        f"{s.total_std[i]:.6f}",
                    ]
                )


def svg_polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def panel_paths(
    times: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    x0: float,
    y0: float,
    width: float,
    height: float,
    ymax: float,
) -> tuple[str, str]:
    def sx(t: float) -> float:
        return x0 + width * t / T_HORIZON

    def sy(v: float) -> float:
        return y0 + height - height * max(0.0, min(ymax, v)) / ymax

    upper = mean + std
    lower = np.maximum(0.0, mean - std)
    band_points = [(sx(t), sy(v)) for t, v in zip(times, upper)]
    band_points += [(sx(t), sy(v)) for t, v in zip(times[::-1], lower[::-1])]
    line_points = [(sx(t), sy(v)) for t, v in zip(times, mean)]
    return svg_polyline(band_points), svg_polyline(line_points)


def nice_upper_bound(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10**exponent
    scaled = value / base
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return nice * base


def write_svg(path: Path, times: np.ndarray, summaries: dict[str, Summary]) -> None:
    width = 1180
    height = 900
    margin_left = 82
    panel_width = 990
    panel_height = 190
    panel_gap = 70
    top = 105
    panels = [
        ("Total pool", "total_mean", "total_std"),
        ("Hard-to-match H", "h_mean", "h_std"),
        ("Easy-to-match E", "e_mean", "e_std"),
    ]

    maxima = []
    for _, mean_name, std_name in panels:
        max_v = 0.0
        for s in summaries.values():
            max_v = max(max_v, float(np.max(getattr(s, mean_name) + getattr(s, std_name))))
        maxima.append(nice_upper_bound(max_v * 1.06))

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts.append("<style>")
    parts.append(
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#17202a}"
        ".title{font-size:25px;font-weight:700}"
        ".subtitle{font-size:14px;fill:#52606d}"
        ".panel-title{font-size:16px;font-weight:700}"
        ".axis{stroke:#26323f;stroke-width:1}"
        ".grid{stroke:#e3e8ef;stroke-width:1}"
        ".tick{font-size:12px;fill:#667080}"
        ".legend{font-size:13px;fill:#17202a}"
    )
    parts.append("</style>")
    parts.append(
        f'<text x="{margin_left}" y="42" class="title">Finite-m CTMC pool trajectories by policy</text>'
    )
    parts.append(
        f'<text x="{margin_left}" y="68" class="subtitle">'
        f"m={M}, d1={D1:g}, d2={D2:g}, {N_RUNS} runs, sampled every {DT:g} time units"
        "</text>"
    )

    legend_x = margin_left + 570
    legend_y = 40
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 165
        y = legend_y + (i // 2) * 24
        color = POLICY_COLORS[policy]
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+36}" y="{y+4}" class="legend">{POLICY_LABELS[policy]}</text>')

    for panel_idx, ((title, mean_name, std_name), ymax) in enumerate(zip(panels, maxima)):
        x0 = margin_left
        y0 = top + panel_idx * (panel_height + panel_gap)
        parts.append(f'<text x="{x0}" y="{y0-16}" class="panel-title">{title}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0+panel_height}" x2="{x0+panel_width}" y2="{y0+panel_height}" class="axis"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+panel_height}" class="axis"/>')

        for frac in (0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_height - panel_height * frac
            value = ymax * frac
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0+panel_width}" y2="{y}" class="grid"/>')
            parts.append(f'<text x="{x0-12}" y="{y+4}" text-anchor="end" class="tick">{value:.0f}</text>')

        for t in (0, 20, 40, 60, 80):
            x = x0 + panel_width * t / T_HORIZON
            parts.append(f'<line x1="{x}" y1="{y0+panel_height}" x2="{x}" y2="{y0+panel_height+5}" class="axis"/>')
            parts.append(f'<text x="{x}" y="{y0+panel_height+22}" text-anchor="middle" class="tick">{t}</text>')
        if panel_idx == len(panels) - 1:
            parts.append(
                f'<text x="{x0 + panel_width / 2}" y="{y0 + panel_height + 48}" '
                'text-anchor="middle" class="subtitle">time</text>'
            )

        for policy in POLICIES:
            s = summaries[policy]
            color = POLICY_COLORS[policy]
            band, line = panel_paths(
                times,
                getattr(s, mean_name),
                getattr(s, std_name),
                x0,
                y0,
                panel_width,
                panel_height,
                ymax,
            )
            parts.append(f'<polygon points="{band}" fill="{color}" opacity="0.13"/>')
            parts.append(
                f'<polyline points="{line}" fill="none" stroke="{color}" '
                'stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def font(size: int, bold: bool = False):
    if ImageFont is None:
        return None

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def write_png(path: Path, times: np.ndarray, summaries: dict[str, Summary]) -> None:
    if Image is None or ImageDraw is None:
        return

    scale = 2
    width = 1180
    height = 900
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image, "RGBA")

    def sc(v: float) -> int:
        return int(round(v * scale))

    def text(x: float, y: float, value: str, fill=(23, 32, 42), size=14, bold=False, anchor=None):
        draw.text((sc(x), sc(y)), value, fill=fill + (255,), font=font(size * scale, bold), anchor=anchor)

    margin_left = 82
    panel_width = 990
    panel_height = 190
    panel_gap = 70
    top = 105
    panels = [
        ("Total pool", "total_mean", "total_std"),
        ("Hard-to-match H", "h_mean", "h_std"),
        ("Easy-to-match E", "e_mean", "e_std"),
    ]

    maxima = []
    for _, mean_name, std_name in panels:
        max_v = 0.0
        for s in summaries.values():
            max_v = max(max_v, float(np.max(getattr(s, mean_name) + getattr(s, std_name))))
        maxima.append(nice_upper_bound(max_v * 1.06))

    text(margin_left, 24, "Finite-m CTMC pool trajectories by policy", size=25, bold=True)
    text(
        margin_left,
        54,
        f"m={M}, d1={D1:g}, d2={D2:g}, {N_RUNS} runs, sampled every {DT:g} time units",
        fill=(82, 96, 109),
        size=14,
    )

    legend_x = margin_left + 570
    legend_y = 40
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 165
        y = legend_y + (i // 2) * 24
        rgb = hex_to_rgb(POLICY_COLORS[policy])
        draw.line((sc(x), sc(y), sc(x + 28), sc(y)), fill=rgb + (255,), width=sc(3))
        text(x + 36, y - 8, POLICY_LABELS[policy], size=13)

    for panel_idx, ((title, mean_name, std_name), ymax) in enumerate(zip(panels, maxima)):
        x0 = margin_left
        y0 = top + panel_idx * (panel_height + panel_gap)
        text(x0, y0 - 36, title, size=16, bold=True)

        axis = (38, 50, 63, 255)
        grid = (227, 232, 239, 255)
        tick_fill = (102, 112, 128)
        draw.line((sc(x0), sc(y0 + panel_height), sc(x0 + panel_width), sc(y0 + panel_height)), fill=axis, width=sc(1))
        draw.line((sc(x0), sc(y0), sc(x0), sc(y0 + panel_height)), fill=axis, width=sc(1))

        for frac in (0.25, 0.5, 0.75, 1.0):
            y = y0 + panel_height - panel_height * frac
            value = ymax * frac
            draw.line((sc(x0), sc(y), sc(x0 + panel_width), sc(y)), fill=grid, width=sc(1))
            text(x0 - 12, y - 8, f"{value:.0f}", fill=tick_fill[:3], size=12, anchor="ra")

        for t in (0, 20, 40, 60, 80):
            x = x0 + panel_width * t / T_HORIZON
            draw.line((sc(x), sc(y0 + panel_height), sc(x), sc(y0 + panel_height + 5)), fill=axis, width=sc(1))
            text(x, y0 + panel_height + 8, f"{t}", fill=tick_fill[:3], size=12, anchor="ma")
        if panel_idx == len(panels) - 1:
            text(x0 + panel_width / 2, y0 + panel_height + 34, "time", fill=(82, 96, 109), size=14, anchor="ma")

        def sx(t: float) -> float:
            return x0 + panel_width * t / T_HORIZON

        def sy(v: float) -> float:
            return y0 + panel_height - panel_height * max(0.0, min(ymax, v)) / ymax

        for policy in POLICIES:
            s = summaries[policy]
            rgb = hex_to_rgb(POLICY_COLORS[policy])
            mean = getattr(s, mean_name)
            std = getattr(s, std_name)
            upper = mean + std
            lower = np.maximum(0.0, mean - std)
            band = [(sc(sx(t)), sc(sy(v))) for t, v in zip(times, upper)]
            band += [(sc(sx(t)), sc(sy(v))) for t, v in zip(times[::-1], lower[::-1])]
            line = [(sc(sx(t)), sc(sy(v))) for t, v in zip(times, mean)]
            draw.polygon(band, fill=rgb + (34,))
            draw.line(line, fill=rgb + (255,), width=sc(2.3), joint="curve")

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(path)


def main() -> None:
    out_dir = Path("simulation_outputs")
    out_dir.mkdir(exist_ok=True)
    times, summaries = run_experiment()
    write_csv(out_dir / "pool_trajectory_summary.csv", times, summaries)
    write_svg(out_dir / "pool_trajectory_m2000_d1_10_d2_100.svg", times, summaries)
    write_png(out_dir / "pool_trajectory_m2000_d1_10_d2_100.png", times, summaries)

    final_idx = -1
    print("Final sampled mean pool sizes at t=80:")
    for policy in POLICIES:
        s = summaries[policy]
        print(
            f"  {POLICY_LABELS[policy]:>10}: "
            f"total={s.total_mean[final_idx]:7.1f} +/- {s.total_std[final_idx]:5.1f}, "
            f"H={s.h_mean[final_idx]:7.1f} +/- {s.h_std[final_idx]:5.1f}, "
            f"E={s.e_mean[final_idx]:7.1f} +/- {s.e_std[final_idx]:5.1f}"
        )


if __name__ == "__main__":
    main()
