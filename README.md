# HDG Euler Gravity

A Julia solver for the compressible 2D x–z Euler equations with gravity. The main application cases are a rising warm bubble and flow over terrain. DGSEM and finite-volume (FV) spatial discretizations share the split-explicit MIS time-stepping layer.

The split is organized as

```text
spatial discretization:  q′ = N(q),  fast operator L(qⁿ), gravity, fluxes, boundaries
MIS time stepper:        slow residual S(q) = N(q) − L(qⁿ), large time steps
fast integrator:         acoustic/gravity substeps; DG and FV support HEVI
```

## Install and run

Run commands from the repository root. Install the example environment once:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

### Warm bubble: DGSEM or FV

```sh
# DGSEM: MIS4 outer stepper, HEVI acoustic/gravity substeps by default
julia --project=examples examples/euler2d_bubble.jl dg

# FV: MIS4 outer stepper, HEVI fast solver by default
julia --project=examples examples/euler2d_bubble.jl fv
```

The optional arguments are `[dg|fv] [T|-] [fast]`, where `T` is the final time in seconds. Both use MIS4 with 12 fast substeps by default. DG uses its HEVI column solver; FV uses horizontal Verlet updates around an implicit vertical Crank–Nicolson solve. The FV WRF solver is also available. The time step is selected from the fast-wave CFL estimate. Results are written in the current directory as VTKHDF time series and PNG plots.

Useful environment controls include `LX`, `H`, `NEX`, `NEZ`, `P`, `BUBBLE_X`, `BUBBLE_Z`, `BUBBLE_RX`, `BUBBLE_RZ`, `DELTA_THETA`, `N`, `U`, and `NSUB`. Lengths are in metres. For FV, the grid uses `P*NEX` by `P*NEZ` cells. The finite-volume grid supports the same Gal–Chen terrain map as DGSEM.

```sh
LX=30000 NEX=30 NEZ=12 julia --project=examples examples/euler2d_bubble.jl dg 300 hevi
julia --project=examples examples/euler2d_bubble.jl fv 300 wrf
```

### Gravity wave: DGSEM or FV

The Skamarock–Klemp gravity-wave case also runs with both discretizations:

```sh
julia --project=examples examples/euler2d_gravity_wave.jl dg hevi
julia --project=examples examples/euler2d_gravity_wave.jl fv wrf
```

DG and FV default to MIS4 with HEVI. The long-channel version is `examples/euler2d_gravity_wave_long.jl`.

### Flow over an Agnesi hill: DGSEM or FV

The production hill analysis in `examples/euler2d_hills.jl` uses DGSEM. The introductory driver also supports the terrain-following FV grid and all three hill widths:

```sh
julia --project=examples examples/intro_2d/agnesi_hills.jl a1
julia --project=examples examples/intro_2d/agnesi_hills.jl a10 - hevi fv
julia --project=examples examples/intro_2d/agnesi_hills.jl a100

# Production-resolution DGSEM hill analysis
julia --project=examples examples/euler2d_hills.jl a1
julia --project=examples examples/euler2d_hills.jl a10 3000 hevi

# Custom hill half-width a = 5 km and horizontal domain Lx = 150 km
HILL_A=5000 LX=150000 NEX=50 julia --project=examples examples/euler2d_hills.jl a10 - hevi
```

The introductory driver uses a 40a periodic x-region, MIS4, and defaults to 50 horizontal elements at degree 4 (nodal spacing 0.2a). `NEX`, `NEZ`, `P`, `LX`, `LFACTOR`, `H0`, `ZD`, and `NSUB` can be set in the environment. The DG production driver accepts `a1`, `a10`, or `a100`, and its default run uses MIS4/HEVI. Generated CSV, VTKHDF, PNG, and SVG output files are ignored by git.

## Spatial discretization and time stepping

`src/DGSEM/Euler2D` owns the 2D DGSEM state, geometry, nonlinear RHS `N(q)` (including gravity and boundary fluxes), frozen fast operator, and its column solver. `src/FV` owns the Arakawa C-grid state, RHS and WRF-style fast operators. `src/TimeStepping` is shared: it computes MIS stages and `S=N-L(qⁿ)`, then advances the fast system with the selected fast integrator. This keeps equation/mesh choices independent from the large-step tableau.

MIS4 is the default large-step method for both 2D discretizations. DG's `:hevi` fast integrator advances the horizontal fast tendencies explicitly and vertical tendencies implicitly with a column-local solve. FV's `:hevi` uses horizontal Verlet half steps around an implicit vertical Crank–Nicolson column solve. FV also retains the WRF-style column solver as an option.

## HDG traces, relaxation, and the planned long-hill method

The 1D `HDGRelaxTheta` module is the prototype for the HDG idea with retained face-trace unknowns. It evolves the trace relaxation equation

```math
\varepsilon\,\dot{\widehat q}=Hq-\widehat q,
```

so the traces are not exactly eliminated. Its relaxed-trace path currently uses IMEX integration. The 1D code is a vertical-solver study, not the target 2D application.

The 2D `Euler2D` fast operator eliminates the HDG traces. The `:verlet` and `:fb` paths use a stored face momentum interface variable; `:fb` can optionally relax that variable through `eps_if` (`EPS_IF` in the example scripts). A positive relaxation time takes effect with `fast=fb`; the default `fast=hevi` path uses eliminated traces and does not apply face relaxation. This is not yet the full retained trace system from `HDGRelaxTheta`. FV has no corresponding HDG trace variables.

The intended research direction for long hill runs is to retain the HDG interface variables and their relaxation dynamics while making the vertical fast solve HEVI. The supplied `Imex.tex` discusses linear and nonlinear Verlet variants and implicit solves; it is design context for future work on the retained-trace DG formulation. McLachlan and Stern analyze a related midpoint-fast/Verlet-slow IMEX integrator for highly oscillatory Hamiltonian systems, but that result does not establish stability of this Euler/HEVI composition. The current DG `:hevi` option is IMEX RK, and `:verlet` is a separate method.

The split matters in the 2D hill case: vertical grid spacing is much smaller than the horizontal scale, so explicit vertical acoustics would force very small steps. HEVI leaves horizontal fast terms explicit and solves the vertical part by columns. The DG implementation currently statically condenses element-local variables inside each vertical column and solves a tridiagonal system for the column-face momentum. A retained-trace formulation will need a compatible column system rather than reusing that eliminated-trace factorization.

## References and design notes

- Xu, van der Vegt & Bokhove (2008), *Discontinuous Hamiltonian Finite Element Method for Linear Hyperbolic Systems*, for the Hamiltonian DG/flux and Störmer–Verlet setting. [DOI: 10.1007/s10915-008-9191-y](https://doi.org/10.1007/s10915-008-9191-y).
- McLachlan & Stern (2014), *Modified trigonometric integrators*, on an IMEX method using midpoint for the fast linear part and Störmer–Verlet for the slow nonlinear part. [DOI: 10.1137/130921118](https://doi.org/10.1137/130921118).
- Baldauf (2021), *A horizontally explicit, vertically implicit (HEVI) discontinuous Galerkin scheme for the 2-dimensional Euler and Navier–Stokes equations using terrain-following coordinates*. [DOI: 10.1016/j.jcp.2021.110635](https://doi.org/10.1016/j.jcp.2021.110635).
- Giraldo et al. (2023), *A Performance Study of Horizontally Explicit Vertically Implicit (HEVI) Time-Integrators for Non-Hydrostatic Atmospheric Models*. [arXiv:2311.11425](https://arxiv.org/abs/2311.11425).

The source for this project is under `src/`; the runnable case drivers are under `examples/`. Optional VTKHDF output uses HDF5 and plots use CairoMakie from the examples environment.
