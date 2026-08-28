"""Convergence and robustness study for the reflecting-wall 2D2V experiment."""

from pathlib import Path
import csv
import json
import os
import time

MPL_CACHE = Path("Results/.matplotlib")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pic.diagnostics import analyze_shock, relative_energy_drift
from runs.reflecting_wall_2d2v import run


OUTPUT = Path("Results/2D2V_convergence")


def analyze(name, results, runtime):
    config = results.configuration
    times = np.asarray(results.t)
    metrics = analyze_shock(
        x=results.x_grid,
        times=times,
        density_history=results.n_i,
        ion_position=results.ion_x[-1],
        ion_velocity=results.ion_v[-1],
        electron_position=results.final_electron_x,
        electron_velocity=results.final_electron_v,
        ion_mass=config["ion_mass"],
        front_downstream_window=(0.4, 1.2),
        front_upstream_window=(7.0, 10.0),
        front_search_window=(0.5, 7.0),
        smoothing_width=21,
    )
    energy_drift = relative_energy_drift(results.total_energy)

    x = results.x_grid
    ni = np.asarray(results.n_i[-1])
    front = metrics.final_front
    shock_region = (x >= max(0.2, front - 1.5)) & (x <= front + 1.5)
    fluctuation = ni - np.mean(ni, axis=1, keepdims=True)
    transverse_rms = float(
        np.sqrt(np.mean(fluctuation[shock_region] ** 2))
        / max(np.mean(ni[shock_region]), 1e-30)
    )
    return {
        "case": name,
        "runtime_s": runtime,
        "dt": config["dt"],
        "n_y": config["n_y"],
        "particles_per_cell": config["particles_x"]*config["particles_y"]/(config["n_x"]*config["n_y"]),
        "seed": config["seed"],
        "mass_ratio": config["ion_mass"],
        "final_time": times[-1],
        "front_x": front,
        "shock_speed": metrics.front_fit.speed,
        "trajectory_r2": metrics.front_fit.r_squared,
        "compression": metrics.compression_ratio,
        "upstream_mach": metrics.upstream_mach,
        "downstream_mach": metrics.downstream_mach,
        "flux_mismatch": metrics.mass_flux_mismatch,
        "ion_heating": metrics.ion_temperature_ratio,
        "reflected_fraction": metrics.reflected_ion_fraction,
        "energy_drift": energy_drift,
        "gauss_linf": float(max(results.gauss_linf)),
        "transverse_rms": transverse_rms,
    }


def relative_change(value, reference):
    return abs(value-reference) / max(abs(reference), 1e-30)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [
        ("baseline", dict(t_max=80.0)),
        ("half_dt", dict(t_max=80.0, dt=0.005, save_interval=200)),
        ("particles_2x", dict(t_max=80.0, particles_y=80)),
        ("particles_4x", dict(t_max=80.0, particles_y=160)),
        ("y_resolution_2x", dict(t_max=80.0, n_y=40, particles_y=80)),
        ("seed_17", dict(t_max=80.0, seed=17)),
        ("seed_73", dict(t_max=80.0, seed=73)),
        ("mass_25", dict(t_max=40.0, ion_mass=25.0)),
        ("mass_400", dict(t_max=160.0, ion_mass=400.0)),
    ]
    partial_file = OUTPUT / "partial_results.json"
    rows = json.loads(partial_file.read_text(encoding="utf-8")) if partial_file.exists() else []
    completed = {row["case"] for row in rows}
    for index, (name, overrides) in enumerate(cases, 1):
        if name in completed:
            print(f"[{index}/{len(cases)}] reusing completed {name}", flush=True)
            continue
        print(f"[{index}/{len(cases)}] running {name}", flush=True)
        started = time.perf_counter()
        results = run(**overrides)
        row = analyze(name, results, time.perf_counter()-started)
        rows.append(row)
        partial_file.write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        print(
            f"  speed={row['shock_speed']:.4f}, compression={row['compression']:.3f}, "
            f"energy drift={100*row['energy_drift']:.2f}%",
            flush=True,
        )

    with (OUTPUT / "convergence_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_name = {row["case"]: row for row in rows}
    baseline = by_name["baseline"]
    key_metrics = ["shock_speed", "compression", "upstream_mach", "ion_heating"]
    timestep_changes = {
        key: relative_change(by_name["half_dt"][key], baseline[key]) for key in key_metrics
    }
    resolution_changes = {
        key: relative_change(by_name["y_resolution_2x"][key], baseline[key]) for key in key_metrics
    }
    seed_rows = [by_name[name] for name in ("baseline", "seed_17", "seed_73")]
    seed_cv = {
        key: float(np.std([row[key] for row in seed_rows], ddof=1) / abs(np.mean([row[key] for row in seed_rows])))
        for key in key_metrics
    }
    checks = {
        "half_dt_key_metrics_within_15_percent": max(timestep_changes.values()) < 0.15,
        "half_dt_energy_below_2_percent": by_name["half_dt"]["energy_drift"] < 0.02,
        "double_y_resolution_metrics_within_20_percent": max(resolution_changes.values()) < 0.20,
        "random_seed_cv_below_20_percent": max(seed_cv.values()) < 0.20,
        "particle_noise_decreases_at_2x": by_name["particles_2x"]["transverse_rms"] < baseline["transverse_rms"],
        "particle_noise_decreases_at_4x": by_name["particles_4x"]["transverse_rms"] < by_name["particles_2x"]["transverse_rms"],
        "all_gauss_residuals_below_1e-12": max(row["gauss_linf"] for row in rows) < 1e-12,
    }

    labels = [row["case"].replace("_", "\n") for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [
        ("shock_speed", r"shock speed / $c$"),
        ("compression", r"compression $n_2/n_1$"),
        ("energy_drift", "maximum relative energy drift"),
        ("transverse_rms", r"transverse density RMS / $\langle n_i\rangle$"),
    ]
    for axis, (key, ylabel) in zip(axes.flat, panels):
        values = [row[key] for row in rows]
        axis.plot(range(len(rows)), values, "o-", lw=1.2)
        axis.set_xticks(range(len(rows)), labels, fontsize=8)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    fig.suptitle("2D2V reflecting-wall convergence and robustness")
    fig.tight_layout()
    fig.savefig(OUTPUT / "convergence_summary.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "convergence_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    particle_rows = [by_name[name] for name in ("baseline", "particles_2x", "particles_4x")]
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(
        [row["particles_per_cell"] for row in particle_rows],
        [row["transverse_rms"] for row in particle_rows],
        "o-",
    )
    axis.set_xlabel("macro-particles per cell per species")
    axis.set_ylabel(r"transverse density RMS / $\langle n_i\rangle$")
    axis.set_title("Particle-noise convergence")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "particle_noise_convergence.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "particle_noise_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    report = ["2D2V REFLECTING-WALL CONVERGENCE REPORT", ""]
    report.extend(f"{name}={value}" for name, value in checks.items())
    report.append("")
    report.append("HALF-TIMESTEP RELATIVE CHANGES")
    report.extend(f"{key}={value:.8e}" for key, value in timestep_changes.items())
    report.append("")
    report.append("DOUBLE-Y-RESOLUTION RELATIVE CHANGES")
    report.extend(f"{key}={value:.8e}" for key, value in resolution_changes.items())
    report.append("")
    report.append("THREE-SEED COEFFICIENTS OF VARIATION")
    report.extend(f"{key}={value:.8e}" for key, value in seed_cv.items())
    report.append("")
    report.append(f"all_declared_convergence_checks_pass={all(checks.values())}")
    (OUTPUT / "convergence_report.txt").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
