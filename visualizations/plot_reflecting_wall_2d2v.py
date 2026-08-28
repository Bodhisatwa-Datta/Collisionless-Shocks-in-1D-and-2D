"""Simple Matplotlib diagnostics and strict checks for reflecting-wall 2D2V."""

from pathlib import Path
import os

MPL_CACHE = Path("Results/.matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pic.diagnostics import analyze_shock, relative_energy_drift
from runs.reflecting_wall_2d2v import ION_MASS, LENGTH_X, LENGTH_Y, run


OUTPUT = Path("Results/2D2V_reflecting_wall_plots")


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = run()
    x = results.x_grid
    y = results.y_grid
    ni = np.asarray(results.n_i[-1])
    ne = np.asarray(results.n_e[-1])
    electric = np.asarray(results.electric[-1])
    ion_position = np.asarray(results.ion_x[-1])
    ion_velocity = np.asarray(results.ion_v[-1])
    times = np.asarray(results.t)
    shock = analyze_shock(
        x=x,
        times=times,
        density_history=results.n_i,
        ion_position=ion_position,
        ion_velocity=ion_velocity,
        electron_position=results.final_electron_x,
        electron_velocity=results.final_electron_v,
        ion_mass=ION_MASS,
        front_downstream_window=(0.4, 1.2),
        front_upstream_window=(7.0, 10.0),
        front_search_window=(0.5, 7.0),
        smoothing_width=21,
    )
    front = shock.final_front
    ni_profile = np.mean(ni, axis=1)
    energy_drift = relative_energy_drift(results.total_energy)

    # Clean presentation figure: no numerical callout text or arrows.
    xmax = 11.0
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    density_plot = axes[0, 0].imshow(
        ni.T,
        origin="lower",
        aspect="auto",
        extent=[0, LENGTH_X, 0, LENGTH_Y],
        cmap="viridis",
    )
    axes[0, 0].axvline(front, color="crimson", ls="--", lw=1.4)
    axes[0, 0].set_xlim(0, xmax)
    axes[0, 0].set_ylabel(r"$y\;(c/\omega_{pe})$")
    axes[0, 0].set_title("Ion density")
    fig.colorbar(density_plot, ax=axes[0, 0], label=r"$n_i/n_0$")

    axes[0, 1].plot(x, ni_profile, color="tab:blue", label="ions")
    axes[0, 1].plot(x, np.mean(ne, axis=1), color="tab:orange", label="electrons")
    axes[0, 1].axvline(front, color="crimson", ls="--", lw=1.4)
    axes[0, 1].set_xlim(0, xmax)
    axes[0, 1].set_ylabel(r"density / $n_0$")
    axes[0, 1].set_title("y-averaged density")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(x, np.mean(electric[:, :, 0], axis=1), color="tab:green")
    axes[1, 0].axvline(front, color="crimson", ls="--", lw=1.4)
    axes[1, 0].axhline(0, color="0.7", lw=0.8)
    axes[1, 0].set_xlim(0, xmax)
    axes[1, 0].set_xlabel(r"$x\;(c/\omega_{pe})$")
    axes[1, 0].set_ylabel(r"$\langle E_x\rangle_y$")
    axes[1, 0].set_title("Longitudinal electric field")

    axes[1, 1].scatter(
        ion_position[:, 0], ion_velocity[:, 0], s=2, alpha=0.35,
        color="tab:blue", rasterized=True,
    )
    axes[1, 1].axvline(front, color="crimson", ls="--", lw=1.4)
    axes[1, 1].axhline(0, color="0.7", lw=0.8)
    axes[1, 1].set_xlim(0, xmax)
    axes[1, 1].set_xlabel(r"$x\;(c/\omega_{pe})$")
    axes[1, 1].set_ylabel(r"ion $v_x/c$")
    axes[1, 1].set_title("Ion phase space")
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    fig.suptitle(rf"Reflecting-wall 2D2V, $t\omega_{{pe}}={times[-1]:.0f}$", fontsize=14)
    fig.tight_layout()
    save(fig, "shock_diagnostic")

    # Dedicated spatial-structure figure.  Subtracting the y average in the
    # second panel removes the dominant 1D compression and exposes variations
    # along the shock surface.
    density_variation = ni - np.mean(ni, axis=1, keepdims=True)
    delta_limit = float(np.percentile(np.abs(density_variation[x <= xmax]), 99))
    ex_limit = float(np.percentile(np.abs(electric[x <= xmax, :, 0]), 99))
    ey_limit = float(np.percentile(np.abs(electric[x <= xmax, :, 1]), 99))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), sharex=True, sharey=True)
    panels = [
        (ni, "Ion density", "viridis", None, None, r"$n_i/n_0$"),
        (
            density_variation,
            r"Transverse density variation $n_i-\langle n_i\rangle_y$",
            "RdBu_r",
            -delta_limit,
            delta_limit,
            r"$\delta n_i/n_0$",
        ),
        (electric[:, :, 0], r"$E_x(x,y)$", "RdBu_r", -ex_limit, ex_limit, r"$E_x$"),
        (electric[:, :, 1], r"$E_y(x,y)$", "RdBu_r", -ey_limit, ey_limit, r"$E_y$"),
    ]
    for axis, (values, title, cmap, vmin, vmax, label) in zip(axes.flat, panels):
        image = axis.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=[0, LENGTH_X, 0, LENGTH_Y],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.axvline(front, color="crimson", ls="--", lw=1.3)
        axis.set_xlim(0, xmax)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, label=label)
    axes[0, 0].set_ylabel(r"$y\;(c/\omega_{pe})$")
    axes[1, 0].set_ylabel(r"$y\;(c/\omega_{pe})$")
    axes[1, 0].set_xlabel(r"$x\;(c/\omega_{pe})$")
    axes[1, 1].set_xlabel(r"$x\;(c/\omega_{pe})$")
    fig.suptitle(rf"2D spatial structure, $t\omega_{{pe}}={times[-1]:.0f}$", fontsize=14)
    fig.tight_layout()
    save(fig, "transverse_structure")

    density_history = np.asarray([np.mean(value, axis=1) for value in results.n_i])
    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    image = axis.imshow(
        density_history,
        origin="lower",
        aspect="auto",
        extent=[0, LENGTH_X, times[0], times[-1]],
        cmap="viridis",
    )
    axis.set_xlim(0, xmax)
    axis.set_xlabel(r"$x\;(c/\omega_{pe})$")
    axis.set_ylabel(r"$t\omega_{pe}$")
    axis.set_title("y-averaged ion-density evolution")
    fig.colorbar(image, ax=axis, label=r"$\langle n_i\rangle_y/n_0$")
    fig.tight_layout()
    save(fig, "shock_spacetime")

    field_profile = np.mean(electric[:, :, 0], axis=1)
    front_index = int(np.argmin(np.abs(x-front)))
    checks = {
        "front_moves_outward": shock.front_fit.speed > 0,
        "front_motion_is_coherent": shock.front_fit.r_squared > 0.9,
        "density_is_compressed": shock.compression_ratio > 1.2,
        "upstream_is_supersonic": shock.upstream_mach > 1.0,
        "downstream_is_subsonic": shock.downstream_mach < 1.0,
        "mass_flux_is_steady": shock.mass_flux_mismatch < 0.2,
        "ions_are_heated": shock.ion_temperature_ratio > 1.5,
        "ions_are_reflected": shock.reflected_ion_fraction > 0.01,
        "field_is_localized": abs(field_profile[front_index]) > 0.05*np.max(np.abs(field_profile)),
        "energy_is_controlled": energy_drift < 0.02,
        "gauss_law_is_controlled": max(results.gauss_linf) < 1e-12,
    }
    confirmed = all(checks.values())
    summary = (
        "STRICT REFLECTING-WALL 2D2V SHOCK VALIDATION\n"
        f"confirmed_collisionless_shock={confirmed}\n"
        f"final_front_x={front:.8e}\n"
        f"shock_speed_over_c={shock.front_fit.speed:.8e}\n"
        f"front_trajectory_r_squared={shock.front_fit.r_squared:.8e}\n"
        f"compression_ratio={shock.compression_ratio:.8e}\n"
        f"upstream_mach={shock.upstream_mach:.8e}\n"
        f"downstream_mach={shock.downstream_mach:.8e}\n"
        f"mass_flux_relative_mismatch={shock.mass_flux_mismatch:.8e}\n"
        f"ion_temperature_ratio={shock.ion_temperature_ratio:.8e}\n"
        f"upstream_reflected_ion_fraction={shock.reflected_ion_fraction:.8e}\n"
        f"maximum_relative_energy_drift={energy_drift:.8e}\n"
        f"maximum_gauss_linf={max(results.gauss_linf):.8e}\n"
        + "\n".join(f"check_{name}={value}" for name, value in checks.items())
        + "\n"
    )
    (OUTPUT / "strict_validation.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
