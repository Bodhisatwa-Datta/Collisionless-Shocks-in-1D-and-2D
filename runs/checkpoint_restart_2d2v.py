"""Demonstrate exact checkpoint/restart reproducibility for the 2D2V run."""

from pathlib import Path
import json
import os

OUTPUT = Path("Results/2D2V_checkpoint_demo")
MPL_CACHE = Path("Results/.matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runs.reflecting_wall_2d2v import WallConfig2D, run


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = WallConfig2D(
        length_x=8.0,
        length_y=2.0,
        n_x=64,
        n_y=16,
        particles_x=80,
        particles_y=20,
        ion_mass=100.0,
        inflow_speed=-0.03,
        electron_thermal_speed=0.08,
        ion_thermal_speed=0.002,
        dt=0.01,
        t_max=2.0,
        save_interval=20,
        seed=41,
    )
    pause_time = 1.0
    checkpoint = OUTPUT / "state_at_t1.npz"
    uninterrupted = run(config)
    run(config, stop_time=pause_time, checkpoint_path=checkpoint)
    resumed = run(resume_from=checkpoint)

    density_difference = np.asarray(resumed.n_i) - uninterrupted.n_i
    field_difference = np.asarray(resumed.electric) - uninterrupted.electric
    energy_difference = np.asarray(resumed.total_energy) - uninterrupted.total_energy
    summary = {
        "pause_time": pause_time,
        "final_time": config.t_max,
        "snapshots": len(uninterrupted.t),
        "maximum_density_difference": float(np.max(np.abs(density_difference))),
        "maximum_field_difference": float(np.max(np.abs(field_difference))),
        "maximum_energy_difference": float(np.max(np.abs(energy_difference))),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    (OUTPUT / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    density = np.asarray(uninterrupted.n_i[-1])
    resumed_density = np.asarray(resumed.n_i[-1])
    data = {
        "x": np.round(uninterrupted.x_grid, 6).tolist(),
        "mean_density": np.round(np.mean(density, axis=1), 10).tolist(),
        "resumed_mean_density": np.round(np.mean(resumed_density, axis=1), 10).tolist(),
        "times": np.round(uninterrupted.t, 6).tolist(),
        "energy": np.round(uninterrupted.total_energy, 12).tolist(),
        "resumed_energy": np.round(resumed.total_energy, 12).tolist(),
    }
    (OUTPUT / "comparison_data.json").write_text(
        json.dumps(data), encoding="utf-8"
    )

    extent = [0, config.length_x, 0, config.length_y]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    image = axes[0, 0].imshow(
        density.T, origin="lower", aspect="auto", extent=extent, cmap="viridis"
    )
    axes[0, 0].set_title("Uninterrupted ion density")
    axes[0, 0].set_ylabel(r"$y\;(c/\omega_{pe})$")
    fig.colorbar(image, ax=axes[0, 0], label=r"$n_i/n_0$")

    difference_image = axes[0, 1].imshow(
        np.abs(resumed_density - density).T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=0.0,
        vmax=1.0e-15,
    )
    axes[0, 1].set_title("Absolute density difference")
    fig.colorbar(difference_image, ax=axes[0, 1], label=r"$|\Delta n_i|$")

    axes[1, 0].plot(
        uninterrupted.x_grid, np.mean(density, axis=1), label="uninterrupted"
    )
    axes[1, 0].plot(
        resumed.x_grid,
        np.mean(resumed_density, axis=1),
        "--",
        label="resumed at t=1",
    )
    axes[1, 0].set_xlabel(r"$x\;(c/\omega_{pe})$")
    axes[1, 0].set_ylabel(r"$\langle n_i\rangle_y/n_0$")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(uninterrupted.t, uninterrupted.total_energy, label="uninterrupted")
    axes[1, 1].plot(resumed.t, resumed.total_energy, "--", label="resumed at t=1")
    axes[1, 1].axvline(pause_time, color="0.4", ls=":", label="checkpoint")
    axes[1, 1].set_xlabel(r"time $t\omega_{pe}$")
    axes[1, 1].set_ylabel("total energy")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle("2D2V checkpoint/restart reproducibility")
    fig.tight_layout()
    fig.savefig(OUTPUT / "checkpoint_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
