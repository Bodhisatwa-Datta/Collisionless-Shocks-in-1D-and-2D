"""Run a validated periodic 2D2V case and write dependency-free SVG plots."""

from html import escape
from pathlib import Path
import numpy as np

from pic import solver_2d2v
from pic.config import BoundaryCondition, Parameters
from pic.constants import m_e, m_i, q_e, q_i
from pic.particles import Particles


OUTPUT = Path("Results/2D2V_plots")


def perturbed_plasma(particles_per_side=32, thermal_speed=0.02, seed=7):
    rng = np.random.default_rng(seed)
    coordinates = (np.arange(particles_per_side) + 0.5) / particles_per_side
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    count = len(positions)
    electrons = Particles(count, 2, 2, m_e, q_e)
    ions = Particles(count, 2, 2, m_i, q_i)
    electrons.x[:] = positions
    ions.x[:] = positions
    electrons.x[:, 0] = (
        electrons.x[:, 0]
        + 0.003
        * np.cos(2 * np.pi * electrons.x[:, 0])
        * np.cos(2 * np.pi * electrons.x[:, 1])
    ) % 1.0
    electrons.v[:] = rng.normal(0.0, thermal_speed, (count, 2))
    electrons.v -= np.mean(electrons.v, axis=0, keepdims=True)
    ions.v.fill(0.0)
    electrons.v_to_u()
    ions.v_to_u()
    return electrons, ions


def colour(value, limit, diverging=True):
    if diverging:
        z = np.clip(value / max(limit, 1e-30), -1, 1)
        if z < 0:
            a = -z
            rgb = (int(246 - 190 * a), int(248 - 105 * a), int(255 - 43 * a))
        else:
            a = z
            rgb = (int(250 - 28 * a), int(248 - 193 * a), int(243 - 205 * a))
    else:
        z = np.clip(value / max(limit, 1e-30), 0, 1)
        rgb = (int(245 - 218 * z), int(247 - 96 * z), int(250 - 48 * z))
    return f"rgb{rgb}"


def heatmap_svg(path, arrays, titles, figure_title, diverging):
    width, height = 1120, 700
    columns, rows = 3, 2
    panel_w, panel_h = 330, 270
    left, top = 72, 70
    gap_x, gap_y = 28, 55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{escape(figure_title)}</text>',
    ]
    for k, (array, title, is_diverging) in enumerate(zip(arrays, titles, diverging)):
        row, col = divmod(k, columns)
        x0 = left + col * (panel_w + gap_x)
        y0 = top + row * (panel_h + gap_y)
        n0, n1 = array.shape
        limit = np.max(np.abs(array)) if is_diverging else np.max(array)
        cell_w, cell_h = panel_w / n0, panel_h / n1
        parts.append(
            f'<text x="{x0+panel_w/2}" y="{y0-12}" text-anchor="middle" font-family="Arial" font-size="16">{escape(title)}</text>'
        )
        for i in range(n0):
            for j in range(n1):
                fill = colour(array[i, j], limit, is_diverging)
                parts.append(
                    f'<rect x="{x0+i*cell_w:.2f}" y="{y0+(n1-1-j)*cell_h:.2f}" width="{cell_w+0.15:.2f}" height="{cell_h+0.15:.2f}" fill="{fill}"/>'
                )
        parts.extend(
            [
                f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#222"/>',
                f'<text x="{x0+panel_w/2}" y="{y0+panel_h+28}" text-anchor="middle" font-family="Arial" font-size="13">x / L</text>',
                f'<text x="{x0-42}" y="{y0+panel_h/2}" text-anchor="middle" transform="rotate(-90 {x0-42} {y0+panel_h/2})" font-family="Arial" font-size="13">y / L</text>',
                f'<text x="{x0}" y="{y0+panel_h+16}" text-anchor="middle" font-family="Arial" font-size="11">0</text>',
                f'<text x="{x0+panel_w}" y="{y0+panel_h+16}" text-anchor="middle" font-family="Arial" font-size="11">1</text>',
                f'<text x="{x0-8}" y="{y0+panel_h}" text-anchor="end" font-family="Arial" font-size="11">0</text>',
                f'<text x="{x0-8}" y="{y0+8}" text-anchor="end" font-family="Arial" font-size="11">1</text>',
                f'<text x="{x0+panel_w-3}" y="{y0+17}" text-anchor="end" font-family="Arial" font-size="11" fill="#111">max |.| = {limit:.2e}</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def line_chart_svg(path, t, series, labels, title, log_y=False):
    width, height = 1000, 580
    left, right, top, bottom = 92, 35, 65, 75
    plot_w, plot_h = width-left-right, height-top-bottom
    palette = ["#1665a8", "#c44e12", "#16825d", "#8c3fa7"]
    transformed = []
    for values in series:
        values = np.asarray(values, float)
        transformed.append(np.log10(np.maximum(values, 1e-30)) if log_y else values)
    ymin = min(np.min(v) for v in transformed)
    ymax = max(np.max(v) for v in transformed)
    if ymax == ymin:
        ymax = ymin + 1
    pad = 0.08 * (ymax-ymin)
    ymin, ymax = ymin-pad, ymax+pad
    xmin, xmax = float(t[0]), float(t[-1])
    xmap = lambda x: left + (x-xmin)/(xmax-xmin) * plot_w
    ymap = lambda y: top + (ymax-y)/(ymax-ymin) * plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">{escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222"/>',
    ]
    for q in range(6):
        frac=q/5; y=ymin+frac*(ymax-ymin); py=ymap(y)
        parts.append(f'<line x1="{left}" y1="{py}" x2="{left+plot_w}" y2="{py}" stroke="#ddd"/>')
        label=f"10^{y:.1f}" if log_y else f"{y:.3e}"
        parts.append(f'<text x="{left-10}" y="{py+4}" text-anchor="end" font-family="Arial" font-size="12">{label}</text>')
    for q in range(6):
        frac=q/5; x=xmin+frac*(xmax-xmin); px=xmap(x)
        parts.append(f'<text x="{px}" y="{top+plot_h+24}" text-anchor="middle" font-family="Arial" font-size="12">{x:.3f}</text>')
    for values, label, color in zip(transformed, labels, palette):
        points=" ".join(f"{xmap(float(x)):.2f},{ymap(float(y)):.2f}" for x,y in zip(t,values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    parts.extend([
        f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" font-family="Arial" font-size="14">normalized time</text>',
        f'<text x="24" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 24 {top+plot_h/2})" font-family="Arial" font-size="14">{"log10 residual" if log_y else "normalized energy"}</text>',
    ])
    legend_x=left+15
    for i,(label,color) in enumerate(zip(labels,palette)):
        y=top+20+i*22
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+25}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+33}" y="{y+4}" font-family="Arial" font-size="13">{escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    electrons, ions = perturbed_plasma()
    params = Parameters(
        x_max=1.0,
        n_cells=24,
        t_max=0.5,
        max_iter=2_000,
        bc=BoundaryCondition.Periodic,
        dimX=2,
        dimV=2,
        num_particles=electrons.N + ions.N,
        n0=1.0,
    )
    results = solver_2d2v.simulate(electrons, ions, params, write_results=False)
    final = -1
    heatmap_svg(
        OUTPUT / "fields_and_charge.svg",
        [
            np.asarray(results.E[final])[:, :, 0],
            np.asarray(results.E[final])[:, :, 1],
            np.asarray(results.B[final])[:, :, 0],
            np.asarray(results.rho[final]),
            np.asarray(results.n_e[final]),
            np.asarray(results.n_i[final]),
        ],
        ["Electric field Ex", "Electric field Ey", "Magnetic field Bz", "Charge density rho", "Electron density", "Ion density"],
        f"Periodic 2D2V fields and densities at t = {results.t[final]:.3f}",
        [True, True, True, True, False, False],
    )
    initial_total = results.TE[0]
    line_chart_svg(
        OUTPUT / "energy_evolution.svg",
        np.asarray(results.t),
        [
            np.asarray(results.KE) / initial_total,
            np.asarray(results.PE) / initial_total,
            np.asarray(results.TE) / initial_total,
        ],
        ["particle kinetic", "field", "total"],
        "2D2V normalized energy evolution",
    )
    line_chart_svg(
        OUTPUT / "constraint_residuals.svg",
        np.asarray(results.t),
        [results.gauss_linf, results.continuity_linf],
        ["Gauss law", "charge continuity"],
        "2D2V numerical constraint residuals",
        log_y=True,
    )
    drift = max(abs((np.asarray(results.TE) - initial_total) / initial_total))
    summary = (
        f"snapshots={len(results.t)}\n"
        f"final_time={results.t[-1]:.8e}\n"
        f"max_relative_energy_drift={drift:.8e}\n"
        f"max_gauss_linf={max(results.gauss_linf):.8e}\n"
        f"max_continuity_linf={max(results.continuity_linf):.8e}\n"
    )
    (OUTPUT / "run_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
