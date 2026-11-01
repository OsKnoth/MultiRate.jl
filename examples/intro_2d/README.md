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
julia --project=examples examples/intro_2d/agnesi_hills.jl a1
julia --project=examples examples/intro_2d/agnesi_hills.jl a10
julia --project=examples examples/intro_2d/agnesi_hills.jl a100
```

Both bubble paths use MIS4 for the large step and HEVI for the fast part. DG uses its vertical
column solver; FV uses horizontal Verlet substeps around an implicit vertical Crank–Nicolson solve.
Set `EPS_IF` to a positive relaxation time and select `hevi_relax` to keep the HEVI vertical
column solve while relaxing the horizontal DG face momentum. Compare with `hevi` under the same
parameters; `EPS_IF=0` makes the variants coincide. `fb` remains available as a separate
forward-backward method with relaxed face momentum.

`check_bubble.jl` advances a warm bubble on a flat domain with both DGSEM/HDG and the
finite-volume C-grid, using HEVI for both. FV uses the terrain-aware vertical column solver and a
stable explicit horizontal Verlet update. WRF remains an available FV option.

`check_agnesi.jl` is a short terrain-following run for DGSEM/HDG/HEVI and the FV C-grid/HEVI
solver, including the FV column-solve consistency check. `agnesi_hills.jl` covers `a = 1, 10,
100 km` with a 40a periodic x domain and 0.2a horizontal spacing. HEVI is the default for both
discretizations.

These are quick user checks, not the production-resolution experiments. The production bubble
and hill programs and their output options are documented in the example scripts.
