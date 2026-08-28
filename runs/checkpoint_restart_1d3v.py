"""Demonstrate that a resumed 1D3V run reproduces an uninterrupted run."""

from pathlib import Path
import json
import os

OUTPUT = Path("Results/1D3V_checkpoint_demo")
MPL_CACHE = Path("Results/.matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runs.reflecting_wall_1d3v import WallConfig, run


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = WallConfig(
        length=10.0,
        n_cells=100,
        particles_per_species=2_000,
        ion_mass=100.0,
        inflow_speed=-0.03,
        electron_thermal_speed=0.08,
        ion_thermal_speed=0.002,
        dt=0.01,
        t_max=10.0,
        save_interval=50,
        seed=31,
    )
    checkpoint = OUTPUT / "state_at_t5.npz"
    uninterrupted = run(config)
    run(config, stop_time=5.0, checkpoint_path=checkpoint)
    resumed = run(resume_from=checkpoint)

    density_error = np.max(np.abs(np.asarray(resumed.n_i) - uninterrupted.n_i))
    field_error = np.max(np.abs(np.asarray(resumed.ex) - uninterrupted.ex))
    energy_error = np.max(
        np.abs(np.asarray(resumed.total_energy) - uninterrupted.total_energy)
    )
    summary = {
        "pause_time": 5.0,
        "final_time": config.t_max,
        "snapshots": len(uninterrupted.t),
        "maximum_density_difference": float(density_error),
        "maximum_field_difference": float(field_error),
        "maximum_energy_difference": float(energy_error),
        "checkpoint_bytes": checkpoint.stat().st_size,
    }
    (OUTPUT / "comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    x = uninterrupted.x_grid
    times = np.asarray(uninterrupted.t)
    full_energy = np.asarray(uninterrupted.total_energy)
    resumed_energy = np.asarray(resumed.total_energy)
    comparison_data = {
        "x": np.round(x, 6).tolist(),
        "uninterrupted_density": np.round(uninterrupted.n_i[-1], 10).tolist(),
        "resumed_density": np.round(resumed.n_i[-1], 10).tolist(),
        "times": np.round(times, 6).tolist(),
        "uninterrupted_energy": np.round(full_energy, 12).tolist(),
        "resumed_energy": np.round(resumed_energy, 12).tolist(),
    }
    (OUTPUT / "comparison_data.json").write_text(
        json.dumps(comparison_data), encoding="utf-8"
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5))
    axes[0].plot(x, uninterrupted.n_i[-1], lw=1.8, label="uninterrupted")
    axes[0].plot(x, resumed.n_i[-1], "--", lw=1.2, label="resumed at t=5")
    axes[0].set_ylabel(r"ion density / $n_0$")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(times, full_energy, lw=1.8, label="uninterrupted")
    axes[1].plot(times, resumed_energy, "--", lw=1.2, label="resumed at t=5")
    axes[1].axvline(5.0, color="0.4", ls=":", label="checkpoint")
    axes[1].set_xlabel(r"time $t\omega_{pe}$")
    axes[1].set_ylabel("total energy")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Checkpoint/restart reproducibility")
    fig.tight_layout()
    fig.savefig(OUTPUT / "checkpoint_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
