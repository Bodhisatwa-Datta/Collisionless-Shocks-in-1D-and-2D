"""Run a preliminary periodic counter-streaming 2D2V shock experiment.

The run deliberately uses a reduced ion/electron mass ratio so that ion-scale
structure becomes visible in a short demonstration.  It produces SVG figures
without requiring matplotlib.
"""

from html import escape
from pathlib import Path
import sys

sys.path.append("./")
sys.path.append("../")

import numpy as np

from parameters import BoundaryCondition, Parameters
from particles import Particles
from physical_constants import m_e, q_e, q_i
import solver_2D


OUTPUT = Path("Results/2D2V_shock_plots")
X_MAX = 10.0
N_CELLS = 48
PARTICLES_PER_SIDE = 48
ION_MASS = 100.0
BULK_SPEED = 0.20
ELECTRON_THERMAL_SPEED = 0.04
ION_THERMAL_SPEED = 0.01
T_MAX = 60.0


def counterstreaming_plasma(seed=17):
    """Create two neutral, oppositely directed slabs in a periodic box."""
    rng = np.random.default_rng(seed)
    coordinates = (np.arange(PARTICLES_PER_SIDE) + 0.5) * (
        X_MAX / PARTICLES_PER_SIDE
    )
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    count = len(positions)

    electrons = Particles(count, 2, 2, m_e, q_e)
    ions = Particles(count, 2, 2, ION_MASS, q_i)
    electrons.x[:] = positions
    ions.x[:] = positions

    flow = np.where(positions[:, 0] < X_MAX / 2, BULK_SPEED, -BULK_SPEED)
    electrons.v[:] = rng.normal(0.0, ELECTRON_THERMAL_SPEED, (count, 2))
    ions.v[:] = rng.normal(0.0, ION_THERMAL_SPEED, (count, 2))
    electrons.v[:, 0] += flow
    ions.v[:, 0] += flow

    # Seed a weak transverse mode while retaining zero net transverse drift.
    seed_mode = 0.004 * np.sin(2 * np.pi * positions[:, 1] / X_MAX)
    electrons.v[:, 1] += seed_mode
    ions.v[:, 1] += seed_mode
    electrons.v[:, 1] -= np.mean(electrons.v[:, 1])
    ions.v[:, 1] -= np.mean(ions.v[:, 1])
    electrons.v_to_u()
    ions.v_to_u()
    return electrons, ions


def _colour(value, low, high, diverging=False):
    if diverging:
        bound = max(abs(low), abs(high), 1e-30)
        z = np.clip(value / bound, -1.0, 1.0)
        if z < 0:
            a = -z
            rgb = (int(245 - 190 * a), int(247 - 100 * a), int(252 - 28 * a))
        else:
            a = z
            rgb = (int(250 - 25 * a), int(247 - 190 * a), int(240 - 195 * a))
    else:
        z = np.clip((value - low) / max(high - low, 1e-30), 0.0, 1.0)
        rgb = (int(247 - 220 * z), int(248 - 100 * z), int(250 - 45 * z))
    return f"rgb{rgb}"


def _heatmap(parts, array, x0, y0, width, height, low, high, diverging=False):
    nx, ny = array.shape
    cw, ch = width / nx, height / ny
    for i in range(nx):
        for j in range(ny):
            parts.append(
                f'<rect x="{x0+i*cw:.2f}" y="{y0+(ny-1-j)*ch:.2f}" '
                f'width="{cw+0.12:.2f}" height="{ch+0.12:.2f}" '
                f'fill="{_colour(array[i,j], low, high, diverging)}"/>'
            )
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
        'fill="none" stroke="#222"/>'
    )


def density_snapshots(path, results):
    indices = sorted(set([0, len(results.t) // 2, len(results.t) - 1]))
    arrays = [np.asarray(results.n_i[k]) for k in indices]
    low = min(float(np.min(a)) for a in arrays)
    high = max(float(np.max(a)) for a in arrays)
    width, height = 1240, 450
    panel_w, panel_h = 350, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="620" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">Ion-density evolution: periodic counter-streaming 2D2V</text>',
        '<text x="620" y="54" text-anchor="middle" font-family="Arial" font-size="14">Preliminary run; reduced mass ratio m_i/m_e = 100</text>',
    ]
    for col, (k, array) in enumerate(zip(indices, arrays)):
        x0, y0 = 70 + col * 390, 98
        _heatmap(parts, array, x0, y0, panel_w, panel_h, low, high)
        parts.extend(
            [
                f'<text x="{x0+panel_w/2}" y="84" text-anchor="middle" font-family="Arial" font-size="16">t omega_pe = {results.t[k]:.2f}</text>',
                f'<text x="{x0+panel_w/2}" y="414" text-anchor="middle" font-family="Arial" font-size="13">x (c / omega_pe)</text>',
                f'<text x="{x0-43}" y="{y0+panel_h/2}" text-anchor="middle" transform="rotate(-90 {x0-43} {y0+panel_h/2})" font-family="Arial" font-size="13">y (c / omega_pe)</text>',
                f'<text x="{x0}" y="394" text-anchor="middle" font-family="Arial" font-size="11">0</text>',
                f'<text x="{x0+panel_w}" y="394" text-anchor="middle" font-family="Arial" font-size="11">10</text>',
            ]
        )
    parts.append(
        f'<text x="1200" y="423" text-anchor="end" font-family="Arial" font-size="12">common range: {low:.2f} to {high:.2f} n0</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def spacetime_density(path, results):
    data = np.asarray([np.mean(np.asarray(a), axis=1) for a in results.n_i])
    low, high = float(np.min(data)), float(np.max(data))
    width, height = 1050, 590
    x0, y0, pw, ph = 105, 70, 860, 430
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="525" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">Space-time ion density</text>',
        '<text x="525" y="51" text-anchor="middle" font-family="Arial" font-size="13">y-averaged; fronts appear as slanted high-density bands</text>',
    ]
    # Transpose so array axes map to x horizontally and time vertically.
    _heatmap(parts, data.T, x0, y0, pw, ph, low, high)
    parts.extend(
        [
            f'<text x="{x0+pw/2}" y="550" text-anchor="middle" font-family="Arial" font-size="14">x (c / omega_pe)</text>',
            f'<text x="28" y="{y0+ph/2}" text-anchor="middle" transform="rotate(-90 28 {y0+ph/2})" font-family="Arial" font-size="14">time (1 / omega_pe)</text>',
            f'<text x="{x0}" y="520" text-anchor="middle" font-family="Arial" font-size="12">0</text>',
            f'<text x="{x0+pw}" y="520" text-anchor="middle" font-family="Arial" font-size="12">10</text>',
            f'<text x="{x0-12}" y="{y0+ph}" text-anchor="end" font-family="Arial" font-size="12">0</text>',
            f'<text x="{x0-12}" y="{y0+8}" text-anchor="end" font-family="Arial" font-size="12">{results.t[-1]:.1f}</text>',
            f'<text x="965" y="570" text-anchor="end" font-family="Arial" font-size="12">density range: {low:.2f} to {high:.2f} n0</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def phase_space(path, results):
    x = np.asarray(results.x_i[-1])[:, 0]
    vx = np.asarray(results.v_i[-1])[:, 0]
    width, height = 1050, 620
    left, top, pw, ph = 100, 80, 870, 440
    vlim = max(float(np.max(np.abs(vx))) * 1.08, BULK_SPEED * 1.2)
    xmap = lambda value: left + value / X_MAX * pw
    ymap = lambda value: top + (vlim - value) / (2 * vlim) * ph
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="525" y="31" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">Ion phase space at t omega_pe = {results.t[-1]:.2f}</text>',
        '<text x="525" y="54" text-anchor="middle" font-family="Arial" font-size="13">Reflected or thermalized ions fill the region between the incoming beams</text>',
        f'<rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="none" stroke="#222"/>',
        f'<line x1="{left}" y1="{ymap(0):.2f}" x2="{left+pw}" y2="{ymap(0):.2f}" stroke="#aaa"/>',
    ]
    for xp, vp in zip(x, vx):
        parts.append(
            f'<circle cx="{xmap(float(xp)):.2f}" cy="{ymap(float(vp)):.2f}" r="1.45" fill="#1767a6" fill-opacity="0.55"/>'
        )
    for q in range(6):
        xx = X_MAX * q / 5
        parts.append(f'<text x="{xmap(xx):.2f}" y="544" text-anchor="middle" font-family="Arial" font-size="12">{xx:.0f}</text>')
    for q in range(5):
        vv = -vlim + 2 * vlim * q / 4
        parts.append(f'<text x="{left-12}" y="{ymap(vv)+4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{vv:.2f}</text>')
    parts.extend(
        [
            '<text x="535" y="585" text-anchor="middle" font-family="Arial" font-size="14">x (c / omega_pe)</text>',
            f'<text x="28" y="{top+ph/2}" text-anchor="middle" transform="rotate(-90 28 {top+ph/2})" font-family="Arial" font-size="14">ion vx / c</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def fields_and_profiles(path, results):
    ni = np.asarray(results.n_i[-1])
    ex = np.asarray(results.E[-1])[:, :, 0]
    bz = np.asarray(results.B[-1])[:, :, 0]
    arrays = [ex, bz]
    titles = ["Longitudinal electric field Ex", "Out-of-plane magnetic field Bz"]
    width, height = 1220, 740
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="610" y="30" text-anchor="middle" font-family="Arial" font-size="22" font-weight="bold">Fields and compression at t omega_pe = {results.t[-1]:.2f}</text>',
        '<text x="610" y="52" text-anchor="middle" font-family="Arial" font-size="13">Periodic preliminary run; reduced mass ratio m_i/m_e = 100</text>',
    ]
    for col, (array, title) in enumerate(zip(arrays, titles)):
        x0, y0, pw, ph = 75 + col * 570, 95, 500, 300
        lim = float(np.max(np.abs(array)))
        _heatmap(parts, array, x0, y0, pw, ph, -lim, lim, True)
        parts.extend(
            [
                f'<text x="{x0+pw/2}" y="82" text-anchor="middle" font-family="Arial" font-size="16">{escape(title)}</text>',
                f'<text x="{x0+pw/2}" y="425" text-anchor="middle" font-family="Arial" font-size="13">x (c / omega_pe)</text>',
                f'<text x="{x0+pw-4}" y="114" text-anchor="end" font-family="Arial" font-size="11">max |field| = {lim:.2e}</text>',
            ]
        )

    profile = np.mean(ni, axis=1)
    px0, py0, pw, ph = 100, 490, 1040, 180
    ymin, ymax = 0.0, max(1.1, float(np.max(profile)) * 1.12)
    xvals = np.linspace(0.0, X_MAX, len(profile), endpoint=False)
    xmap = lambda value: px0 + value / X_MAX * pw
    ymap = lambda value: py0 + (ymax - value) / (ymax - ymin) * ph
    points = " ".join(f"{xmap(float(x)):.2f},{ymap(float(y)):.2f}" for x, y in zip(xvals, profile))
    parts.extend(
        [
            '<text x="610" y="474" text-anchor="middle" font-family="Arial" font-size="16">y-averaged ion-density profile</text>',
            f'<rect x="{px0}" y="{py0}" width="{pw}" height="{ph}" fill="none" stroke="#222"/>',
            f'<line x1="{px0}" y1="{ymap(1):.2f}" x2="{px0+pw}" y2="{ymap(1):.2f}" stroke="#999" stroke-dasharray="5 4"/>',
            f'<polyline points="{points}" fill="none" stroke="#1767a6" stroke-width="2.5"/>',
            f'<text x="{px0+pw/2}" y="718" text-anchor="middle" font-family="Arial" font-size="14">x (c / omega_pe)</text>',
            f'<text x="35" y="{py0+ph/2}" text-anchor="middle" transform="rotate(-90 35 {py0+ph/2})" font-family="Arial" font-size="14">ion density / n0</text>',
            f'<text x="{px0+pw-5}" y="{ymap(float(np.max(profile)))-8:.2f}" text-anchor="end" font-family="Arial" font-size="12">peak = {np.max(profile):.2f} n0</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    electrons, ions = counterstreaming_plasma()
    params = Parameters(
        x_max=X_MAX,
        n_cells=N_CELLS,
        t_max=T_MAX,
        max_iter=20_000,
        bc=BoundaryCondition.Periodic,
        dimX=2,
        dimV=2,
        num_particles=electrons.N + ions.N,
        n0=1.0,
    )
    results = solver_2D.simulate(electrons, ions, params, write_results=False)

    density_snapshots(OUTPUT / "ion_density_snapshots.svg", results)
    spacetime_density(OUTPUT / "ion_density_spacetime.svg", results)
    phase_space(OUTPUT / "ion_phase_space.svg", results)
    fields_and_profiles(OUTPUT / "fields_and_density_profile.svg", results)

    profiles = np.asarray([np.mean(np.asarray(a), axis=1) for a in results.n_i])
    peak_density = float(np.max(profiles[-1]))
    initial_total = float(results.TE[0])
    drift = float(np.max(np.abs((np.asarray(results.TE) - initial_total) / initial_total)))
    summary = (
        "PRELIMINARY PERIODIC COUNTER-STREAMING 2D2V RUN\n"
        "This is a reduced-mass demonstration, not a converged physical shock claim.\n"
        f"mass_ratio_mi_me={ION_MASS:.0f}\n"
        f"bulk_speed_over_c={BULK_SPEED:.6e}\n"
        f"final_saved_time={results.t[-1]:.8e}\n"
        f"final_peak_y_averaged_ion_density_over_n0={peak_density:.8e}\n"
        f"max_relative_energy_drift={drift:.8e}\n"
        f"max_gauss_linf={max(results.gauss_linf):.8e}\n"
        f"max_continuity_linf={max(results.continuity_linf):.8e}\n"
    )
    (OUTPUT / "run_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
