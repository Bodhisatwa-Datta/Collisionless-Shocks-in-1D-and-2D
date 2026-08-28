"""Simple Matplotlib figures for the periodic 1D3V shock-tube experiment."""

from pathlib import Path
import os
MPL_CACHE = Path("Results/.matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runs.shock_tube_1d3v import BULK_SPEED, ION_MASS, X_MAX, run


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


def detect_right_front(x, density):
    """Locate the steepest falling edge of the central compressed region."""
    smoothed = smooth_periodic(density)
    gradient = np.gradient(smoothed, x)
    candidates = np.flatnonzero((x > 0.52 * X_MAX) & (x < 0.92 * X_MAX))
    return int(candidates[np.argmin(gradient[candidates])])


def shock_metrics(x, density, front_index):
    """Average density in small downstream/upstream windows around the edge."""
    dx = float(x[1] - x[0])
    near = max(4, int(round(0.5 / dx)))
    far = max(9, int(round(1.2 / dx)))
    downstream = np.arange(front_index - far, front_index - near) % len(density)
    upstream = np.arange(front_index + near, front_index + far) % len(density)
    n_down = float(np.mean(density[downstream]))
    n_up = float(np.mean(density[upstream]))
    return n_up, n_down, n_down / max(n_up, 1e-30)


def save_figure(fig, stem):
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_shock_instance(results):
    x = np.asarray(results.x[-1])
    ni = np.asarray(results.n_i[-1])
    ne = np.asarray(results.n_e[-1])
    ex = np.asarray(results.E[-1])[:, 0]
    ion_x = np.asarray(results.x_i[-1])[:, 0]
    ion_vx = np.asarray(results.v_i[-1])[:, 0]
    front_index = detect_right_front(x, ni)
    shock_x = float(x[front_index])
    n_up, n_down, ratio = shock_metrics(x, ni, front_index)

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)

    axes[0].plot(x, ni, label="ions", color="tab:blue", lw=1.7)
    axes[0].plot(x, ne, label="electrons", color="tab:orange", lw=1.3)
    axes[0].set_ylabel(r"density / $n_0$")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(x, ex, color="tab:green", lw=1.4)
    axes[1].axhline(0.0, color="0.7", lw=0.8)
    axes[1].set_ylabel(r"$E_x$")

    axes[2].scatter(ion_x, ion_vx, s=2, alpha=0.35, color="tab:blue", rasterized=True)
    axes[2].axhline(0.0, color="0.7", lw=0.8)
    axes[2].set_ylabel(r"ion $v_x/c$")
    axes[2].set_xlabel(r"$x\;(c/\omega_{pe})$")

    for axis in axes:
        axis.axvline(shock_x, color="crimson", ls="--", lw=1.5)
        axis.grid(alpha=0.18)
    axes[0].annotate(
        rf"compression edge  $x_c={shock_x:.2f}$" + "\n" + rf"$n_2/n_1={ratio:.2f}$",
        xy=(shock_x, ni[front_index]),
        xytext=(shock_x + 1.0, max(ni) * 0.82),
        arrowprops={"arrowstyle": "->", "color": "crimson"},
        color="crimson",
    )

    fig.suptitle(
        rf"Counter-streaming 1D3V snapshot, $t\omega_{{pe}}={results.t[-1]:.1f}$",
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, "shock_instance")
    return shock_x, n_up, n_down, ratio


def plot_density_evolution(results):
    indices = [0, len(results.t) // 2, len(results.t) - 1]
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True, sharey=True)
    for axis, index in zip(axes, indices):
        x = np.asarray(results.x[index])
        axis.plot(x, results.n_i[index], color="tab:blue", lw=1.6)
        axis.plot(x, results.n_e[index], color="tab:orange", lw=1.1, alpha=0.85)
        axis.set_ylabel(r"$n/n_0$")
        axis.text(0.02, 0.86, rf"$t\omega_{{pe}}={results.t[index]:.1f}$", transform=axis.transAxes)
        axis.grid(alpha=0.18)
    axes[0].legend(["ions", "electrons"], frameon=False, ncol=2)
    axes[-1].set_xlabel(r"$x\;(c/\omega_{pe})$")
    fig.suptitle("1D3V shock-tube density evolution", fontsize=14)
    fig.tight_layout()
    save_figure(fig, "density_evolution")


def plot_density_spacetime(results):
    density = np.asarray(results.n_i)
    fig, axis = plt.subplots(figsize=(8.0, 4.7))
    image = axis.imshow(
        density,
        origin="lower",
        aspect="auto",
        extent=[0.0, X_MAX, results.t[0], results.t[-1]],
        cmap="viridis",
    )
    axis.set_xlabel(r"$x\;(c/\omega_{pe})$")
    axis.set_ylabel(r"$t\omega_{pe}$")
    axis.set_title("Ion-density evolution")
    fig.colorbar(image, ax=axis, label=r"$n_i/n_0$")
    fig.tight_layout()
    save_figure(fig, "ion_density_spacetime")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = run(write_results=False)
    shock_x, n_up, n_down, ratio = plot_shock_instance(results)
    plot_density_evolution(results)
    plot_density_spacetime(results)

    initial_energy = float(results.TE[0])
    energy_drift = float(
        np.max(np.abs((np.asarray(results.TE) - initial_energy) / initial_energy))
    )
    summary = (
        "PERIODIC 1D3V SHOCK-TUBE ANALOGUE\n"
        "Reduced-mass demonstration; ballistic compression, not a mature shock.\n"
        f"mass_ratio_mi_me={ION_MASS:.0f}\n"
        f"bulk_speed_over_c={BULK_SPEED:.8e}\n"
        f"final_saved_time={results.t[-1]:.8e}\n"
        f"detected_front_x={shock_x:.8e}\n"
        f"local_upstream_density={n_up:.8e}\n"
        f"local_downstream_density={n_down:.8e}\n"
        f"local_compression_ratio={ratio:.8e}\n"
        f"max_relative_energy_drift={energy_drift:.8e}\n"
        f"max_gauss_linf={max(results.gauss_linf):.8e}\n"
        f"max_continuity_linf={max(results.continuity_linf):.8e}\n"
    )
    (OUTPUT / "run_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
