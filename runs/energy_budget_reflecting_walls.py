"""Compare resolved energy transfer in the 1D3V and 2D2V wall runs."""

from pathlib import Path
import json
import os

OUTPUT = Path("Results/energy_budget")
MPL_CACHE = Path("Results/.matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runs.reflecting_wall_1d3v import WallConfig, run as run_1d
from runs.reflecting_wall_2d2v import WallConfig2D, run as run_2d


def normalized_changes(results):
    electron = np.asarray(results.electron_kinetic_energy)
    ion = np.asarray(results.ion_kinetic_energy)
    field = np.asarray(results.field_energy)
    total = np.asarray(results.total_energy)
    scale = total[0]
    return {
        "electron": (electron - electron[0]) / scale,
        "ion": (ion - ion[0]) / scale,
        "field": (field - field[0]) / scale,
        "total": (total - total[0]) / scale,
    }


def summarize(results):
    electron = np.asarray(results.electron_kinetic_energy)
    ion = np.asarray(results.ion_kinetic_energy)
    field = np.asarray(results.field_energy)
    total = np.asarray(results.total_energy)
    closure = electron + ion + field - total
    return {
        "maximum_relative_total_energy_drift": float(
            np.max(np.abs((total - total[0]) / total[0]))
        ),
        "maximum_component_closure_error": float(np.max(np.abs(closure))),
        "final_electron_energy_fraction": float(electron[-1] / total[-1]),
        "final_ion_energy_fraction": float(ion[-1] / total[-1]),
        "final_field_energy_fraction": float(field[-1] / total[-1]),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    one_d = run_1d(
        WallConfig(
            length=10.0,
            n_cells=100,
            particles_per_species=2_000,
            electron_thermal_speed=0.08,
            dt=0.01,
            t_max=10.0,
            save_interval=50,
            seed=31,
        )
    )
    two_d = run_2d(
        WallConfig2D(
            length_x=8.0,
            length_y=2.0,
            n_x=64,
            n_y=16,
            particles_x=80,
            particles_y=20,
            electron_thermal_speed=0.08,
            dt=0.01,
            t_max=2.0,
            save_interval=20,
            seed=41,
        )
    )
    changes_1d = normalized_changes(one_d)
    changes_2d = normalized_changes(two_d)
    summary = {"1d3v": summarize(one_d), "2d2v": summarize(two_d)}
    (OUTPUT / "energy_budget_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    data = {
        "1d3v": {
            "time": np.round(one_d.t, 6).tolist(),
            **{key: np.round(value, 12).tolist() for key, value in changes_1d.items()},
        },
        "2d2v": {
            "time": np.round(two_d.t, 6).tolist(),
            **{key: np.round(value, 12).tolist() for key, value in changes_2d.items()},
        },
    }
    (OUTPUT / "energy_budget_data.json").write_text(
        json.dumps(data), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.0))
    labels = {
        "electron": "electron kinetic",
        "ion": "ion kinetic",
        "field": "electric field",
        "total": "total",
    }
    styles = {"electron": "-", "ion": "-", "field": "-", "total": "--"}
    for axis, results, changes, title in (
        (axes[0], one_d, changes_1d, "1D3V reflecting wall"),
        (axes[1], two_d, changes_2d, "2D2V reflecting wall"),
    ):
        for key in ("electron", "ion", "field", "total"):
            axis.plot(results.t, changes[key], styles[key], lw=1.5, label=labels[key])
        axis.axhline(0.0, color="0.65", lw=0.8)
        axis.set_ylabel(r"$\Delta E/E_0$")
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[1].set_xlabel(r"time $t\omega_{pe}$")
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Resolved energy transfer and total-energy control")
    fig.tight_layout()
    fig.savefig(OUTPUT / "energy_budget.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
