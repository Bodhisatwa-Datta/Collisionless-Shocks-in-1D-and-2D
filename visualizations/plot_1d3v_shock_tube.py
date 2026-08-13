"""Run and plot the periodic 1D3V shock-tube analogue as dependency-free SVG."""

from html import escape
from pathlib import Path
import sys

sys.path.append("./")
sys.path.append("../")

import numpy as np

from runs.shock_tube_1d3v import (
    BULK_SPEED,
    ELECTRON_THERMAL_SPEED,
    ION_MASS,
    X_MAX,
    run,
)


OUTPUT = Path("Results/1D3V_shock_tube_plots")


def smooth_periodic(values, passes=4):
    result = np.asarray(values, dtype=float).copy()
    for _ in range(passes):
        result = (
            np.roll(result, 2)
            + 4 * np.roll(result, 1)
            + 6 * result
            + 4 * np.roll(result, -1)
            + np.roll(result, -2)
        ) / 16
    return result


def detect_front(x, density):
    """Detect the right-moving shock edge outside the central dense slab."""
    profile = smooth_periodic(density)
    gradient = np.gradient(profile, x)
    mask = (x > 0.52 * X_MAX) & (x < 0.92 * X_MAX)
    candidates = np.flatnonzero(mask)
    index = candidates[np.argmin(gradient[candidates])]
    return index, profile, gradient


def local_shock_metrics(x, density, front_index):
    dx = x[1] - x[0]
    direction = np.sign(np.gradient(smooth_periodic(density), x)[front_index])
    offset_near = max(4, int(round(0.5 / dx)))
    offset_far = max(9, int(round(1.2 / dx)))
    if direction < 0:
        downstream = np.arange(front_index - offset_far, front_index - offset_near)
        upstream = np.arange(front_index + offset_near, front_index + offset_far)
    else:
        downstream = np.arange(front_index + offset_near, front_index + offset_far)
        upstream = np.arange(front_index - offset_far, front_index - offset_near)
    downstream %= len(density)
    upstream %= len(density)
    n_down = float(np.mean(density[downstream]))
    n_up = float(np.mean(density[upstream]))
    return n_up, n_down, n_down / max(n_up, 1e-30), direction


def svg_header(width, height, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{escape(title)}</text>',
        f'<text x="{width/2}" y="53" text-anchor="middle" font-family="Arial" font-size="13">{escape(subtitle)}</text>',
    ]


def line_panel(parts, x, series, labels, colors, x0, y0, width, height, y_label, front=None):
    all_values = np.concatenate([np.asarray(v, float) for v in series])
    ymin, ymax = float(np.min(all_values)), float(np.max(all_values))
    pad = 0.08 * max(ymax - ymin, 1e-12)
    ymin -= pad
    ymax += pad
    xmap = lambda value: x0 + (value - x[0]) / (x[-1] - x[0]) * width
    ymap = lambda value: y0 + (ymax - value) / (ymax - ymin) * height
    parts.append(f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#222"/>')
    for q in range(5):
        value = ymin + q * (ymax - ymin) / 4
        py = ymap(value)
        parts.append(f'<line x1="{x0}" y1="{py:.2f}" x2="{x0+width}" y2="{py:.2f}" stroke="#e0e0e0"/>')
        parts.append(f'<text x="{x0-9}" y="{py+4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2g}</text>')
    for values, label, color in zip(series, labels, colors):
        points = " ".join(f"{xmap(float(a)):.2f},{ymap(float(b)):.2f}" for a, b in zip(x, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"/>')
    if front is not None:
        px = xmap(front)
        parts.append(f'<line x1="{px:.2f}" y1="{y0}" x2="{px:.2f}" y2="{y0+height}" stroke="#bd2b2b" stroke-width="2" stroke-dasharray="7 5"/>')
        parts.append(f'<text x="{px+7:.2f}" y="{y0+18}" font-family="Arial" font-size="12" fill="#9f2020">shock front x_s</text>')
    for q in range(5):
        value = x[0] + q * (x[-1] - x[0]) / 4
        parts.append(f'<text x="{xmap(value):.2f}" y="{y0+height+20}" text-anchor="middle" font-family="Arial" font-size="11">{value:.1f}</text>')
    parts.append(f'<text x="{x0+width/2}" y="{y0+height+42}" text-anchor="middle" font-family="Arial" font-size="13">x (c / omega_pe)</text>')
    parts.append(f'<text x="24" y="{y0+height/2}" text-anchor="middle" transform="rotate(-90 24 {y0+height/2})" font-family="Arial" font-size="13">{escape(y_label)}</text>')
    for k, (label, color) in enumerate(zip(labels, colors)):
        lx = x0 + 12 + k * 155
        parts.append(f'<line x1="{lx}" y1="{y0+height-14}" x2="{lx+25}" y2="{y0+height-14}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+32}" y="{y0+height-10}" font-family="Arial" font-size="12">{escape(label)}</text>')


def snapshots(path, results):
    indices = sorted(set([0, len(results.t)//3, 2*len(results.t)//3, len(results.t)-1]))
    width, height = 1120, 900
    parts = svg_header(width, height, "1D3V shock-tube density evolution", "Periodic analogue; reduced mass ratio m_i/m_e = 100")
    for row, index in enumerate(indices):
        x = np.asarray(results.x[index])
        ni = np.asarray(results.n_i[index])
        ne = np.asarray(results.n_e[index])
        front = None
        if index > 0:
            fi, _, _ = detect_front(x, ni)
            front = float(x[fi])
        parts.append(f'<text x="550" y="{82+row*205}" text-anchor="middle" font-family="Arial" font-size="15">t omega_pe = {results.t[index]:.2f}</text>')
        line_panel(parts, x, [ni, ne], ["ions", "electrons"], ["#1767a6", "#d06019"], 75, 92+row*205, 980, 135, "density / n0", front)
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def spacetime(path, results):
    density = np.asarray(results.n_i)
    low, high = float(np.min(density)), float(np.max(density))
    width, height = 1080, 630
    left, top, pw, ph = 105, 75, 880, 450
    parts = svg_header(width, height, "Ion density in space and time", "The propagating high-density edge traces the shock front")
    nt, nx = density.shape
    cw, ch = pw/nx, ph/nt
    for j in range(nt):
        for i in range(nx):
            z = np.clip((density[j,i]-low)/max(high-low,1e-30), 0, 1)
            rgb = (int(248-220*z), int(248-103*z), int(250-46*z))
            parts.append(f'<rect x="{left+i*cw:.2f}" y="{top+(nt-1-j)*ch:.2f}" width="{cw+0.1:.2f}" height="{ch+0.1:.2f}" fill="rgb{rgb}"/>')
    parts.extend([
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#222"/>',
        f'<text x="{left+pw/2}" y="570" text-anchor="middle" font-family="Arial" font-size="14">x (c / omega_pe)</text>',
        f'<text x="28" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 28 {top+ph/2})" font-family="Arial" font-size="14">time (1 / omega_pe)</text>',
        f'<text x="{left}" y="545" text-anchor="middle" font-family="Arial" font-size="12">0</text>',
        f'<text x="{left+pw}" y="545" text-anchor="middle" font-family="Arial" font-size="12">20</text>',
        f'<text x="{left-12}" y="{top+ph}" text-anchor="end" font-family="Arial" font-size="12">0</text>',
        f'<text x="{left-12}" y="{top+8}" text-anchor="end" font-family="Arial" font-size="12">{results.t[-1]:.1f}</text>',
        f'<text x="{left+pw}" y="603" text-anchor="end" font-family="Arial" font-size="12">range {low:.2f}–{high:.2f} n0</text>',
        "</svg>",
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def shock_instance(path, results):
    x = np.asarray(results.x[-1])
    ni = np.asarray(results.n_i[-1])
    ne = np.asarray(results.n_e[-1])
    ex = np.asarray(results.E[-1])[:,0]
    xi = np.asarray(results.x_i[-1])[:,0]
    vxi = np.asarray(results.v_i[-1])[:,0]
    front_index, _, _ = detect_front(x, ni)
    shock_x = float(x[front_index])
    n_up, n_down, ratio, direction = local_shock_metrics(x, ni, front_index)

    width, height = 1120, 970
    parts = svg_header(width, height, f"1D3V shock instance at t omega_pe = {results.t[-1]:.2f}", f"Detected compression edge x_s = {shock_x:.2f}; local n_down/n_up = {ratio:.2f}")
    parts.append('<text x="550" y="81" text-anchor="middle" font-family="Arial" font-size="16">Density jump</text>')
    line_panel(parts, x, [ni, ne], ["ions", "electrons"], ["#1767a6", "#d06019"], 80, 92, 970, 210, "density / n0", shock_x)
    parts.append('<text x="550" y="366" text-anchor="middle" font-family="Arial" font-size="16">Localized longitudinal electric field</text>')
    line_panel(parts, x, [ex], ["Ex"], ["#16825d"], 80, 377, 970, 185, "Ex", shock_x)

    left, top, pw, ph = 80, 650, 970, 235
    vlim = max(float(np.max(np.abs(vxi)))*1.08, BULK_SPEED*1.3)
    xmap = lambda value: left + value/X_MAX*pw
    ymap = lambda value: top + (vlim-value)/(2*vlim)*ph
    parts.extend([
        '<text x="550" y="635" text-anchor="middle" font-family="Arial" font-size="16">Ion phase space: incoming, slowed and reflected populations</text>',
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#222"/>',
        f'<line x1="{xmap(shock_x):.2f}" y1="{top}" x2="{xmap(shock_x):.2f}" y2="{top+ph}" stroke="#bd2b2b" stroke-width="2" stroke-dasharray="7 5"/>',
        f'<line x1="{left}" y1="{ymap(0):.2f}" x2="{left+pw}" y2="{ymap(0):.2f}" stroke="#aaa"/>',
    ])
    for xp, vp in zip(xi, vxi):
        parts.append(f'<circle cx="{xmap(float(xp)):.2f}" cy="{ymap(float(vp)):.2f}" r="1.15" fill="#1767a6" fill-opacity="0.48"/>')
    parts.extend([
        f'<text x="{left+pw/2}" y="935" text-anchor="middle" font-family="Arial" font-size="14">x (c / omega_pe)</text>',
        f'<text x="28" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 28 {top+ph/2})" font-family="Arial" font-size="14">ion vx / c</text>',
        "</svg>",
    ])
    path.write_text("\n".join(parts), encoding="utf-8")
    return shock_x, n_up, n_down, ratio


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = run(write_results=False)
    snapshots(OUTPUT / "density_evolution.svg", results)
    spacetime(OUTPUT / "ion_density_spacetime.svg", results)
    shock_x, n_up, n_down, ratio = shock_instance(OUTPUT / "shock_instance.svg", results)
    initial_energy = float(results.TE[0])
    drift = float(np.max(np.abs((np.asarray(results.TE)-initial_energy)/initial_energy)))
    transverse_max = max(
        float(np.max(np.abs(np.asarray(results.E)[...,1:]))),
        float(np.max(np.abs(np.asarray(results.B)))),
    )
    summary = (
        "PERIODIC 1D3V SHOCK-TUBE ANALOGUE\n"
        "Reduced-mass demonstration; shock identification remains preliminary.\n"
        f"mass_ratio_mi_me={ION_MASS:.0f}\n"
        f"bulk_speed_over_c={BULK_SPEED:.8e}\n"
        f"electron_thermal_speed_over_c={ELECTRON_THERMAL_SPEED:.8e}\n"
        f"final_saved_time={results.t[-1]:.8e}\n"
        f"detected_front_x={shock_x:.8e}\n"
        f"local_upstream_density={n_up:.8e}\n"
        f"local_downstream_density={n_down:.8e}\n"
        f"local_compression_ratio={ratio:.8e}\n"
        f"max_transverse_field={transverse_max:.8e}\n"
        f"max_relative_energy_drift={drift:.8e}\n"
        f"max_gauss_linf={max(results.gauss_linf):.8e}\n"
        f"max_continuity_linf={max(results.continuity_linf):.8e}\n"
    )
    (OUTPUT / "run_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
