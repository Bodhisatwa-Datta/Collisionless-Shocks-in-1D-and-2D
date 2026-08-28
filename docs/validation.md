# Validation status

This note distinguishes numerical checks from physical interpretation. Passing
a software test shows that a discrete algorithm behaves as intended; it does not
by itself prove that a plotted structure is a collisionless shock.

## Numerical checks

The automated suite checks:

- bounded electromagnetic vacuum-wave energy on the Yee grids;
- Gauss-law initialization and preservation;
- discrete charge continuity after current projection;
- correct CIC deposition and staggered-grid interpolation;
- speed preservation by the relativistic Boris pushers;
- reflecting-wall particle reflection and upstream injection;
- reproducibility of short 1D3V and 2D2V shock runs.

## Evidence used for a shock interpretation

The reflecting-wall diagnostics combine several observations: a propagating
density transition, density compression across that transition, upstream to
downstream flow deceleration, ion heating, and a reflected-ion population. A
vertical line on a density plot is only a visual aid; it is not evidence on its
own.

The 1D validation program reports the front trajectory, fitted front speed,
compression ratio, upstream and downstream Mach estimates, mass-flux mismatch,
ion-temperature ratio, reflected fraction, and numerical conservation errors.

For 2D, the convergence program varies grid resolution, particles per cell, and
random seed. In the current parameter range, front speed and bulk compression
are more robust than detailed heating and transverse structure. Claims made from
the 2D figures should reflect that distinction.

## Known limitations

- The electromagnetic 1D3V and 2D2V drivers are validated only for periodic
  boundaries.
- The reflecting-wall shock programs are electrostatic research experiments,
  even though their particles retain multiple velocity components.
- Longer domains and run times are needed to separate a mature shock from the
  initial wall interaction more convincingly.
- Quantitative physical claims need a documented convergence study and several
  independent random seeds.
