# 2D split-explicit MIS introductions

These small programs demonstrate the shared MIS time stepper with the two spatial
discretizations. The checked-in examples manifest was resolved with Julia 1.12. On a fresh Julia
depot, instantiate it once so binary dependencies such as the transitive `OpenSSL_jll` are
installed:

```sh
julia +1.12 --project=examples -e 'using Pkg; Pkg.instantiate()'
```

Then run the checks from the repository root:

```sh
julia --project=examples examples/intro_2d/check_bubble.jl
julia --project=examples examples/intro_2d/check_agnesi.jl
```

Run the examples themselves with:

```sh
julia --project=examples examples/intro_2d/warm_bubble.jl dg
EPS_IF=1 julia --project=examples examples/intro_2d/warm_bubble.jl dg - hevi_relax
julia --project=examples examples/intro_2d/warm_bubble.jl fv
julia --project=examples examples/intro_2d/warm_bubble.jl fvc
julia --project=examples examples/intro_2d/agnesi_hills.jl a1
julia --project=examples examples/intro_2d/agnesi_hills.jl a1 - hevi fvc
julia --project=examples examples/intro_2d/agnesi_hills.jl a1 - rosenbrock_amf fvc
julia --project=examples examples/intro_2d/agnesi_hills.jl a10
julia --project=examples examples/intro_2d/agnesi_hills.jl a100
```

All paths use MIS4 for the large step. Terrain-following FV uses horizontal Verlet substeps around
an implicit vertical Crank–Nicolson solve. Cut-cell FV uses Cartesian cells clipped by the terrain
and conservative aperture-weighted fluxes, with two fast solvers. `hevi` uses a per-column implicit
vertical acoustic solve and explicit horizontal substeps, so its step remains limited by the
smallest cut cell. `rosenbrock_amf` uses a three-stage Rosenbrock-W method with approximate matrix
factors: vertical acoustics over the domain plus horizontal acoustics on the cut-cell mask, followed
by the masked cut-cell advection Jacobian. Its step limit is based on the regular-grid horizontal
acoustic CFL. The masked advection Jacobian currently differentiates the cut-cell FV donor-cell
residual; the MATLAB third-order `AdvectionCutThird` reconstruction is a later fidelity step. In
`fvc` bubble runs the default cut terrain is a 500 m Agnesi hill (`CUT_H0`, `CUT_A` override it).
Set `EPS_IF` to a positive relaxation time and select `hevi_relax` to keep the HEVI vertical
column solve while relaxing the horizontal DG face momentum. Compare with `hevi` under the same
parameters; `EPS_IF=0` makes the variants coincide. `fb` remains available as a separate
forward-backward method with relaxed face momentum.

`check_bubble.jl` advances a warm bubble on a flat domain with both DGSEM/HDG and the
finite-volume C-grid, using HEVI for both. FV uses the terrain-aware vertical column solver and a
stable explicit horizontal Verlet update. WRF remains an available FV option.

`check_agnesi.jl` checks terrain-following and cut-cell FV conservation, the cut-cell column solve,
small-step cut-cell HEVI, and larger-step MIS with the Rosenbrock-W AMF fast solver.
`agnesi_hills.jl` covers `a = 1, 10,
100 km` with a 40a periodic x domain and 0.2a horizontal spacing. HEVI is the default for both
discretizations.

These are quick user checks, not the production-resolution experiments. The production bubble
and hill programs and their output options are documented in the example scripts.
