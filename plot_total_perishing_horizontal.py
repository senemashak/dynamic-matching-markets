# -*- coding: utf-8 -*-
"""Plot total cumulative perishing normalized by m*T across d2 levels."""

from __future__ import annotations

import csv
from collections import defaultdict
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
D2_VALUES = (10.0, 100.0, 1000.0, 10000.0)

INPUT_CSV = Path("simulation_outputs/perishing_loss_m10000_d1_10_d2_10_100_1000_10000_t50_stacked.csv")
STEM = "total_perishing_by_d2_horizontal_m10000_d1_10_d2_10_100_1000_10000_t50"


@dataclass(frozen=True)
class TotalSeries:
    times: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def load_transformed_series(path: Path) -> dict[float, dict[str, TotalSeries]]:
    rows_by_key: dict[tuple[float, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows_by_key[(float(row["d2"]), row["policy"])].append(row)

    series: dict[float, dict[str, TotalSeries]] = {}
    for d2 in D2_VALUES:
        series[d2] = {}
        for policy in POLICIES:
            rows = sorted(rows_by_key[(d2, policy)], key=lambda r: float(r["time"]))
            times = np.array([float(r["time"]) for r in rows])
            scale = times / T_HORIZON
            series[d2][policy] = TotalSeries(
                times=times,
                mean=np.array([float(r["total_loss_mean"]) for r in rows]) * scale,
                std=np.array([float(r["total_loss_std"]) for r in rows]) * scale,
            )
    return series


def write_transformed_csv(path: Path, series: dict[float, dict[str, TotalSeries]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "d1", "d2", "policy", "time", "total_perish_over_mT_mean", "total_perish_over_mT_std"])
        for d2 in D2_VALUES:
            for policy in POLICIES:
                s = series[d2][policy]
                for i, t in enumerate(s.times):
                    writer.writerow(
                        [
                            M,
                            f"{D1:g}",
                            f"{d2:g}",
                            policy,
                            f"{t:.2f}",
                            f"{s.mean[i]:.8f}",
                            f"{s.std[i]:.8f}",
                        ]
                    )


def y_max(series: dict[float, dict[str, TotalSeries]]) -> float:
    return 0.30


def panel_coordinates(
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


def write_svg(path: Path, series: dict[float, dict[str, TotalSeries]]) -> None:
    width = 1900
    height = 460
    left = 82
    top = 142
    panel_w = 405
    panel_h = 220
    col_gap = 45
    ymax = y_max(series)

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
        f'<text x="{left}" y="42" class="title">Total cumulative perishing, normalized by mT</text>',
        f'<text x="{left}" y="68" class="subtitle">all perished by time t / (mT), m={M}, T={T_HORIZON:g}, d1={D1:g}, 10 runs</text>',
    ]

    legend_x = left + 1040
    legend_y = 38
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 170
        y = legend_y + (i // 2) * 24
        color = POLICY_COLORS[policy]
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 36}" y="{y + 4}" class="legend">{POLICY_LABELS[policy]}</text>')

    for col_idx, d2 in enumerate(D2_VALUES):
        x0 = left + col_idx * (panel_w + col_gap)
        y0 = top
        parts.append(f'<text x="{x0}" y="{y0 - 24}" class="panel-title">d2 = {d2:g}</text>')
        if col_idx == 0:
            parts.append(
                f'<text x="{x0 - 62}" y="{y0 + panel_h / 2}" transform="rotate(-90 {x0 - 62} {y0 + panel_h / 2})" '
                'text-anchor="middle" class="subtitle">Total perished / (mT)</text>'
            )
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
        parts.append(f'<text x="{x0 + panel_w / 2}" y="{y0 + panel_h + 48}" text-anchor="middle" class="subtitle">Time</text>')

        for policy in POLICIES:
            s = series[d2][policy]
            color = POLICY_COLORS[policy]
            band, line = panel_coordinates(s.times, s.mean, s.std, x0, y0, panel_w, panel_h, ymax)
            parts.append(f'<polygon points="{svg_polyline(band)}" fill="{color}" opacity="0.13"/>')
            parts.append(
                f'<polyline points="{svg_polyline(line)}" fill="none" stroke="{color}" '
                'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_png(path: Path, series: dict[float, dict[str, TotalSeries]]) -> None:
    if Image is None or ImageDraw is None:
        return

    scale = 2
    width = 1900
    height = 460
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
    top = 142
    panel_w = 405
    panel_h = 220
    col_gap = 45
    ymax = y_max(series)
    axis = (38, 50, 63, 255)
    grid = (227, 232, 239, 255)
    tick_fill = (102, 112, 128)

    text(left, 24, "Total cumulative perishing, normalized by mT", size=25, bold=True)
    text(left, 54, f"all perished by time t / (mT), m={M}, T={T_HORIZON:g}, d1={D1:g}, 10 runs", fill=(82, 96, 109), size=14)

    legend_x = left + 1040
    legend_y = 38
    for i, policy in enumerate(POLICIES):
        x = legend_x + (i % 2) * 170
        y = legend_y + (i // 2) * 24
        rgb = hex_to_rgb(POLICY_COLORS[policy])
        draw.line((sc(x), sc(y), sc(x + 28), sc(y)), fill=rgb + (255,), width=sc(3))
        text(x + 36, y - 8, POLICY_LABELS[policy], size=13)

    for col_idx, d2 in enumerate(D2_VALUES):
        x0 = left + col_idx * (panel_w + col_gap)
        y0 = top
        text(x0, y0 - 44, f"d2 = {d2:g}", size=16, bold=True)
        if col_idx == 0:
            rotated_text(x0 - 62, y0 + panel_h / 2, "Total perished / (mT)")
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
        text(x0 + panel_w / 2, y0 + panel_h + 34, "Time", fill=(82, 96, 109), size=14, anchor="ma")

        for policy in POLICIES:
            s = series[d2][policy]
            rgb = hex_to_rgb(POLICY_COLORS[policy])
            band, line = panel_coordinates(s.times, s.mean, s.std, x0, y0, panel_w, panel_h, ymax)
            draw.polygon([(sc(x), sc(y)) for x, y in band], fill=rgb + (34,))
            draw.line([(sc(x), sc(y)) for x, y in line], fill=rgb + (255,), width=sc(2.2), joint="curve")

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    img.save(path)


def main() -> None:
    out_dir = Path("simulation_outputs")
    series = load_transformed_series(INPUT_CSV)
    write_transformed_csv(out_dir / f"{STEM}.csv", series)
    write_svg(out_dir / f"{STEM}.svg", series)
    write_png(out_dir / f"{STEM}.png", series)

    print("Final total perishing, normalized by m*T:")
    for d2 in D2_VALUES:
        print(f"  d2={d2:g}")
        for policy in POLICIES:
            s = series[d2][policy]
            print(f"    {POLICY_LABELS[policy]:>10}: total={s.mean[-1]:.4f} +/- {s.std[-1]:.4f}")


if __name__ == "__main__":
    main()
