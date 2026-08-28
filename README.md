# Collisionless shocks with particle-in-cell simulations

[![DOI](https://zenodo.org/badge/1311073124.svg)](https://doi.org/10.5281/zenodo.22148547)

This repository is my working implementation of electrostatic and electromagnetic
particle-in-cell (PIC) experiments in one and two spatial dimensions. The main
scientific target is a collisionless shock formed by reflecting an incoming
electron-ion plasma at a conducting wall. Periodic reference problems are kept
alongside the shock runs because they provide much cleaner tests of the field
solver, current deposition, particle pusher, and conservation laws.

The code uses normalized units (`c = eps_0 = mu_0 = 1`). It is deliberately a
small research code: the numerical steps are visible in Python, and every solver
that is described as validated has a corresponding regression test.

## What is currently supported

| Model | Fields and particles | Boundary used in validated runs | Status |
| --- | --- | --- | --- |
| 1D1V | electrostatic | periodic | legacy reference path |
| 1D3V | electromagnetic | periodic | tested |
| 1D3V shock | electrostatic reflecting-wall experiment with three velocity components | reflecting wall / upstream injection | tested experiment |
| 2D2V | TMz electromagnetic (`Ex`, `Ey`, `Bz`) | periodic | tested |
| 2D2V shock | electrostatic reflecting-wall experiment | reflecting wall / upstream injection | tested, with convergence study |

The reflecting-wall experiments produce shock-like density compression, a
travelling front, ion reflection, and ion heating. These are numerical research
results, not a claim that every kinetic shock regime has been reproduced. In
particular, transverse structure and detailed heating in the present 2D runs are
more sensitive to resolution than the measured front speed and compression.

## Set up the project

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Bodhisatwa-Datta/Collisionless-Shocks-in-1D-and-2D.git
cd Collisionless-Shocks-in-1D-and-2D
python -m venv .venv
```

Activate the environment, then install the project and its plotting tools:

```bash
python -m pip install -e ".[plots]"
```

## Run the checks

```bash
python -m unittest discover -s tests -v
```

The tests cover the Yee field updates, Gauss's law, discrete charge continuity,
CIC deposition, relativistic Boris pushers, reflecting-wall injection, and the
shock diagnostics.

## Reproduce the main experiments

Run commands from the repository root.

```bash
# Periodic electromagnetic reference cases
python -m runs.periodic_1d3v
python -m runs.periodic_2d2v

# Reflecting-wall shock experiments
python -m runs.reflecting_wall_1d3v
python -m runs.reflecting_wall_2d2v

# Quantitative 1D validation and 2D robustness study
python -m runs.validate_reflecting_wall_1d3v
python -m runs.convergence_reflecting_wall_2d2v

# Verify an interrupted 1D3V run against an uninterrupted run
python -m runs.checkpoint_restart_1d3v
```

Simulation output is written below `Results/`, which is intentionally ignored by
Git. The plotting programs in `visualizations/` read those saved results and
write presentation-ready PNG files.

The reflecting-wall 1D3V run accepts a serializable `WallConfig`. It can stop at
an intermediate time, write an atomic compressed checkpoint, and resume without
reinitializing the particles. The checkpoint contains the configuration,
particle phase space, current step, and saved diagnostic history.

## Repository map

- `pic/`: configuration, state containers, field updates, deposition, particle
  pushers, diagnostics storage, and dimension-specific solver drivers.
- `runs/`: reproducible numerical experiments rather than library code.
- `visualizations/`: plotting programs kept separate from simulation logic.
- `tests/`: short numerical regression tests.
- `docs/`: numerical notes, validation limits, and project history.

## Scope and limitations

The validated electromagnetic solvers currently assume periodic fields and
particles. Open and absorbing electromagnetic boundaries are not advertised as
supported because they still need consistent particle injection, field boundary
conditions, and independent validation. The reflecting-wall shock programs are
self-contained electrostatic experiments and should not be confused with those
periodic electromagnetic drivers.

The code favors readable algorithms over large-scale performance. Production
studies should repeat the convergence analysis with more particles, finer grids,
longer domains, and several random seeds.

## Project history

This work started from a group plasma-dynamics project and was subsequently
debugged, extended, and reorganized for my independent research. The exact code used
for the original report is preserved in the `Original_Report_Code` branch. The
`main` branch is the maintained research version. See
[`docs/project-history.md`](docs/project-history.md) for the authorship and branch
policy.

## References

- C. K. Birdsall and A. B. Langdon, *Plasma Physics via Computer Simulation*.
- R. W. Hockney and J. W. Eastwood, *Computer Simulation Using Particles*.
- D. R. Nicholson, *Introduction to Plasma Theory*.

The repository is licensed under Apache-2.0; see [`LICENSE`](LICENSE).
