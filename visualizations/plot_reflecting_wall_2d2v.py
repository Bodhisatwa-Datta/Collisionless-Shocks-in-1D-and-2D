"""Simple Matplotlib diagnostics and strict checks for reflecting-wall 2D2V."""

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

from runs.reflecting_wall_2d2v import ION_MASS, LENGTH_X, LENGTH_Y, run


OUTPUT = Path("Results/2D2V_reflecting_wall_plots")


def smooth(values, width=21):
    return np.convolve(np.asarray(values), np.ones(width) / width, mode="same")


def front_position(x, density_2d):
    profile = smooth(np.mean(density_2d, axis=1))
    downstream = (x >= 0.4) & (x <= 1.2)
    upstream = (x >= 7.0) & (x <= 10.0)
    n2 = float(np.mean(profile[downstream]))
    n1 = float(np.mean(profile[upstream]))
    if n2 < 1.1 * n1:
        return np.nan
    threshold = 0.5 * (n1 + n2)
    candidates = np.flatnonzero((x > 0.5) & (x < 7.0) & (profile < threshold))
    return float(x[candidates[0]]) if len(candidates) else np.nan


def species_moments(position, velocity, xlow, xhigh, mass):
    mask = (position[:, 0] >= xlow) & (position[:, 0] <= xhigh)
    vx = velocity[mask, 0]
    return float(np.mean(vx)), float(mass * np.var(vx)), mask


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
    front = front_position(x, ni)

    # Track the front only after a compressed downstream state has appeared.
    times = np.asarray(results.t)
    fronts = np.asarray([front_position(x, density) for density in results.n_i])
    fit_mask = np.isfinite(fronts) & (times >= 0.4 * times[-1])
    shock_speed, intercept = np.polyfit(times[fit_mask], fronts[fit_mask], 1)
    fitted = shock_speed * times[fit_mask] + intercept
    residual = np.sum((fronts[fit_mask] - fitted) ** 2)
    total = np.sum((fronts[fit_mask] - np.mean(fronts[fit_mask])) ** 2)
    trajectory_r2 = 1.0 - residual / max(total, 1e-30)

    downstream = (max(0.2, front - 1.2), front - 0.4)
    upstream = (front + 0.8, front + 2.4)
    ni_profile = np.mean(ni, axis=1)
    n2 = float(np.mean(ni_profile[(x >= downstream[0]) & (x <= downstream[1])]))
    n1 = float(np.mean(ni_profile[(x >= upstream[0]) & (x <= upstream[1])]))
    compression = n2 / n1
    vi2, ti2, ion_down = species_moments(ion_position, ion_velocity, *downstream, ION_MASS)
    vi1, ti1, ion_up = species_moments(ion_position, ion_velocity, *upstream, ION_MASS)
    electron_position = np.asarray(results.final_electron_x)
    electron_velocity = np.asarray(results.final_electron_v)
    _, te2, _ = species_moments(electron_position, electron_velocity, *downstream, 1.0)
    _, te1, _ = species_moments(electron_position, electron_velocity, *upstream, 1.0)
    u1 = vi1 - shock_speed
    u2 = vi2 - shock_speed
    cs1 = np.sqrt(max(te1 + 3 * ti1, 0.0) / ION_MASS)
    cs2 = np.sqrt(max(te2 + 3 * ti2, 0.0) / ION_MASS)
    mach1 = abs(u1) / max(cs1, 1e-30)
    mach2 = abs(u2) / max(cs2, 1e-30)
    flux_mismatch = abs(n1*u1 - n2*u2) / max(abs(n1*u1), abs(n2*u2), 1e-30)
    heating = ti2 / max(ti1, 1e-30)
    reflected_fraction = float(np.mean(ion_velocity[ion_up, 0] > 0.0))
    energy = np.asarray(results.total_energy)
    energy_drift = float(np.max(np.abs((energy-energy[0])/energy[0])))

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
        "front_moves_outward": shock_speed > 0,
        "front_motion_is_coherent": trajectory_r2 > 0.9,
        "density_is_compressed": compression > 1.2,
        "upstream_is_supersonic": mach1 > 1.0,
        "downstream_is_subsonic": mach2 < 1.0,
        "mass_flux_is_steady": flux_mismatch < 0.2,
        "ions_are_heated": heating > 1.5,
        "ions_are_reflected": reflected_fraction > 0.01,
        "field_is_localized": abs(field_profile[front_index]) > 0.05*np.max(np.abs(field_profile)),
        "energy_is_controlled": energy_drift < 0.02,
        "gauss_law_is_controlled": max(results.gauss_linf) < 1e-12,
    }
    confirmed = all(checks.values())
    summary = (
        "STRICT REFLECTING-WALL 2D2V SHOCK VALIDATION\n"
        f"confirmed_collisionless_shock={confirmed}\n"
        f"final_front_x={front:.8e}\n"
        f"shock_speed_over_c={shock_speed:.8e}\n"
        f"front_trajectory_r_squared={trajectory_r2:.8e}\n"
        f"compression_ratio={compression:.8e}\n"
        f"upstream_mach={mach1:.8e}\n"
        f"downstream_mach={mach2:.8e}\n"
        f"mass_flux_relative_mismatch={flux_mismatch:.8e}\n"
        f"ion_temperature_ratio={heating:.8e}\n"
        f"upstream_reflected_ion_fraction={reflected_fraction:.8e}\n"
        f"maximum_relative_energy_drift={energy_drift:.8e}\n"
        f"maximum_gauss_linf={max(results.gauss_linf):.8e}\n"
        + "\n".join(f"check_{name}={value}" for name, value in checks.items())
        + "\n"
    )
    (OUTPUT / "strict_validation.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
