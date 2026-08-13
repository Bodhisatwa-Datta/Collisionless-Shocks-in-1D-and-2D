"""Matplotlib diagnostics for the reflecting-wall 1D3V shock experiment."""

from pathlib import Path
import os
import sys

sys.path.append("./")
sys.path.append("../")

MPL_CACHE = Path("Results/.matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runs.reflecting_wall_1d3v import (
    INFLOW_SPEED,
    ION_MASS,
    LENGTH,
    run,
)


OUTPUT = Path("Results/1D3V_reflecting_wall_plots")


def smooth(values, width=31):
    kernel = np.ones(width) / width
    return np.convolve(np.asarray(values), kernel, mode="same")


def detect_front(x, density):
    """Find the midpoint crossing between downstream and upstream density."""
    profile = smooth(density)
    n2 = float(np.mean(profile[(x >= 0.4) & (x <= 1.2)]))
    n1 = float(np.mean(profile[(x >= 8.0) & (x <= 12.0)]))
    threshold = 0.5 * (n1 + n2)
    candidates = np.flatnonzero((x > 0.5) & (x < 8.0) & (profile < threshold))
    return int(candidates[0])


def metrics(x, density, index):
    downstream = (x >= max(0.2, x[index] - 1.2)) & (x <= x[index] - 0.4)
    upstream = (x >= x[index] + 0.8) & (x <= x[index] + 2.4)
    n2 = float(np.mean(density[downstream]))
    n1 = float(np.mean(density[upstream]))
    return n1, n2, n2 / max(n1, 1e-30)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = run()
    x = results.x_grid
    ni = np.asarray(results.n_i[-1])
    ne = np.asarray(results.n_e[-1])
    ex = np.asarray(results.ex[-1])
    ion_x = np.asarray(results.ion_x[-1])
    ion_vx = np.asarray(results.ion_v[-1])[:, 0]
    index = detect_front(x, ni)
    front = float(x[index])
    n1, n2, ratio = metrics(x, ni, index)
    near_upstream = (ion_x >= front + 0.8) & (ion_x <= front + 2.4)
    reflected_fraction = float(np.mean(ion_vx[near_upstream] > 0.0))

    xmax = min(14.0, LENGTH)
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)
    axes[0].plot(x, ni, color="tab:blue", lw=1.5, label="ions")
    axes[0].plot(x, ne, color="tab:orange", lw=1.2, label="electrons")
    axes[0].set_ylabel(r"density / $n_0$")
    axes[0].legend(frameon=False, ncol=2)
    axes[1].plot(x, ex, color="tab:green", lw=1.4)
    axes[1].axhline(0, color="0.7", lw=0.8)
    axes[1].set_ylabel(r"$E_x$")
    axes[2].scatter(ion_x, ion_vx, s=2, alpha=0.35, color="tab:blue", rasterized=True)
    axes[2].axhline(0, color="0.7", lw=0.8)
    axes[2].set_ylabel(r"ion $v_x/c$")
    axes[2].set_xlabel(r"$x\;(c/\omega_{pe})$")
    for axis in axes:
        axis.axvline(front, color="crimson", ls="--", lw=1.5)
        axis.set_xlim(0, xmax)
        axis.grid(alpha=0.18)
    axes[0].annotate(
        rf"shock front $x_s={front:.2f}$" + "\n" + rf"$n_2/n_1={ratio:.2f}$",
        xy=(front, ni[index]),
        xytext=(front + 1.0, max(ni[(x < xmax)]) * 0.8),
        color="crimson",
        arrowprops={"arrowstyle": "->", "color": "crimson"},
    )
    fig.suptitle(rf"Reflecting-wall 1D3V, $t\omega_{{pe}}={results.t[-1]:.0f}$", fontsize=14)
    fig.tight_layout()
    save(fig, "shock_diagnostic")

    density = np.asarray(results.n_i)
    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    image = axis.imshow(
        density,
        origin="lower",
        aspect="auto",
        extent=[0, LENGTH, results.t[0], results.t[-1]],
        cmap="viridis",
    )
    axis.set_xlim(0, xmax)
    axis.set_xlabel(r"$x\;(c/\omega_{pe})$")
    axis.set_ylabel(r"$t\omega_{pe}$")
    axis.set_title("Ion-density evolution from the reflecting wall")
    fig.colorbar(image, ax=axis, label=r"$n_i/n_0$")
    fig.tight_layout()
    save(fig, "shock_spacetime")

    energy = np.asarray(results.total_energy)
    energy_drift = float(np.max(np.abs((energy-energy[0])/energy[0])))
    shock_checks = {
        "compression": ratio > 1.2,
        "localized_field": abs(ex[index]) > 0.05 * np.max(np.abs(ex)),
        "reflected_ions": reflected_fraction > 0.01,
        "energy_ok": energy_drift < 0.02,
    }
    confirmed = all(shock_checks.values())
    summary = (
        "REFLECTING-WALL ELECTROSTATIC 1D3V\n"
        f"shock_diagnostic_pass={confirmed}\n"
        f"mass_ratio_mi_me={ION_MASS:.0f}\n"
        f"inflow_speed_over_c={INFLOW_SPEED:.8e}\n"
        f"final_time={results.t[-1]:.8e}\n"
        f"front_x={front:.8e}\n"
        f"upstream_density={n1:.8e}\n"
        f"downstream_density={n2:.8e}\n"
        f"compression_ratio={ratio:.8e}\n"
        f"upstream_reflected_ion_fraction={reflected_fraction:.8e}\n"
        f"max_relative_energy_drift={energy_drift:.8e}\n"
        f"max_gauss_linf={max(results.gauss_linf):.8e}\n"
        + "\n".join(f"check_{key}={value}" for key, value in shock_checks.items())
        + "\n"
    )
    (OUTPUT / "run_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
