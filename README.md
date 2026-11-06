# MultiRate.jl

Partially implicit multirate infinitesimal step (MIS) methods for the compressible Euler equations
with gravity in 2D (x–z), tested on several spatial discretizations.

The project studies one question: **which parts of the fast linear operator should be treated
implicitly, and where?** The time integrator is the same for all discretizations. The spatial
discretizations supply a fast operator and a choice of its implicit part.

## The method class

The semi-discrete system is written as `q' = N(q)`. At the start of a large step the fast linear
operator `L = L(qⁿ)` is frozen, and the right-hand side is split by adding and subtracting it:

```text
q' = L q + (N(q) − L q) = L q + S(q),       L = L_E + L_I
```

- The **slow part** `S(q)` is advanced by the outer MIS tableau (MIS4 by default) with the large
  step Δt, which is limited by advection in the free atmosphere.
- In each MIS stage the **fast part** is integrated with small steps Δτ, with `L_E` treated explicitly
  and `L_I` implicitly. The stage tendency `S` is held constant.

`L` and its split `L_I` can be chosen locally. That choice is the subject of the project:

| Splitting | Typical choice of `L_I` | Removes the step limit from |
|---|---|---|
| vertical vs horizontal (HEVI) | vertical acoustics and buoyancy in every column | small Δz |
| acoustics vs advection | advection in `L` only where it is needed, otherwise acoustics only | — (advection stays explicit, Δt ∝ Δx/U) |
| cut cells | acoustics **and** advection in a mask around the cut cells | arbitrarily small cut cells |

The order of the method does not depend on what is used for `L` (W-method property). Stability
does. In cut cells the implicit advection can be a simplified Jacobian, for example first-order
upwind, while the right-hand side keeps the conservative third-order scheme. Outside the mask
`L` is purely acoustic, so the explicit-advection limit of the underlying split-explicit method
still applies there (Jebens, Knoth & Weiner 2011). Because the implicit region covers only a thin
layer near the ground, a cut-cell grid costs about as much as the split-explicit method on a
regular grid.

## Spatial discretizations and fast solvers

| Discretization | Terrain | Fast solvers (`fast=`) |
|---|---|---|
| DGSEM, split form (`src/DGSEM/Euler2D`) | terrain-following (Gal–Chen) | `hevi`, `hevi_relax`, `verlet`, `fb`, `imex` |
| FV, Arakawa C-grid (`src/FV`) | terrain-following | `hevi`, `wrf`, `verlet`, `fb` |
| FV, Arakawa C-grid with cut cells (`src/FV/cutcell_*`) | Cartesian cells cut by `zs(x)` | `hevi`, `rosenbrock_amf`, `verlet`, `fb` |

- **`hevi`:** horizontal fast terms explicit, vertical column solve implicit. On cut-cell grids
  this solver is still limited by the smallest cut cell.
- **`rosenbrock_amf`** (cut cells only): a three-stage Rosenbrock-W fast integrator with approximate
  matrix factors.
  - Implicit: vertical acoustics everywhere, plus horizontal acoustics and first-order masked
    advection in the cut-cell mask.
  - Explicit: horizontal acoustics outside the mask.
  - Its step limit therefore comes from the regular-grid horizontal acoustic CFL and is independent
    of the cut-cell size.
- **Cut-cell masks:** the mask is a heuristic (`CutCellFVGrid(...; mask=:diag)`, as in the MATLAB
  reference; variants `:cut`, `:cut4`, `:cut8`, `:all`).
- **Cut-cell advection:** the right-hand side uses the third-order `AdvectionCutThird` scheme by
  default (`advection=:third`); the first-order alternative is `advection=:upwind1`. The Θ flux
  reconstructs θ' = θ − θ̄(z) and adds θ̄ exactly at the face height, so the background
  stratification is not smeared where the flow crosses the grid lines along the terrain.

## Code layout

```text
src/TimeStepping   MIS tableaus and stages, slow residual S = N − L(qⁿ), generic fast loops
src/DGSEM          split-form DGSEM state, geometry, RHS, fast operator, column solver
src/FV             C-grid FV: terrain-following grid and fast operators; cut-cell geometry,
                   RHS, HEVI column solve and Rosenbrock-W/AMF fast solver
src/Common         gas law, reference states, output (VTKHDF, plots via extensions)
examples/          case drivers; examples/intro_2d holds the short introductory cases
```

A spatial discretization provides `rhs_nonlinear!`, `fast_operator` (freeze `L` at `qⁿ`) and the
pieces its fast solver needs (`fast_apply!`, column factors, model-specific integrators). The MIS
layer is independent of the mesh and the equations.

## Install and run

Run the commands from the repository root. Install the example environment once:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

### Introductory cases

These cover the warm bubble and the Agnesi hills (a = 1, 10, 100 km, h = 400 m, U = 10 m/s,
N = 0.01 s⁻¹) on all three discretizations:

```sh
julia --project=examples examples/intro_2d/warm_bubble.jl   dg|fv|fvc [T|-] [fast]
julia --project=examples examples/intro_2d/agnesi_hills.jl  a1|a10|a100 [T|-] [fast] [dg|fv|fvc]

# cut cells, horizontally explicit acoustics (step limited by the smallest cut cell)
julia --project=examples examples/intro_2d/agnesi_hills.jl a10 - hevi fvc
# cut cells, partially implicit fast solver (step independent of the cut-cell size)
julia --project=examples examples/intro_2d/agnesi_hills.jl a10 - rosenbrock_amf fvc
```

`examples/intro_2d/README.md` describes the options, and `check_bubble.jl` and `check_agnesi.jl`
are quick consistency checks. Environment controls include `NEX`, `NEZ`, `P`, `LX`, `LFACTOR`,
`H0`, `ZD`, `NSUB`, `U0` (hill inflow speed; `U0=0` is a resting atmosphere), `CUT_H0` and `CUT_A`. FV grids use `P*NEX × P*NEZ` cells.

### Production drivers

```sh
julia --project=examples examples/euler2d_bubble.jl        dg|fv [T|-] [fast]
julia --project=examples examples/euler2d_gravity_wave.jl  dg|fv [fast]
julia --project=examples examples/euler2d_hills.jl         a1|a10|a100 [T|-] [fast]   # DGSEM
```

Results are written to the current directory as VTKHDF time series and PNG plots. Generated output
files are ignored by git.

## Status and next steps

- **Cut-cell FV with MIS4:**
  - `hevi` and `rosenbrock_amf` give the same Agnesi solutions for all three hill widths.
  - `rosenbrock_amf` also runs at the large steps of the MATLAB reference: Δt = 20 s for a1,
    100 s for a10.
  - The AMF fast-step CFL is 0.65 of the regular-grid limit, because of the explicit horizontal
    acoustics inside the W-method.
- **DGSEM with `hevi_verlet`** (horizontal Störmer–Verlet around a vertical Crank–Nicolson column
  solve) runs all three Agnesi hills at the regular horizontal acoustic step (a100: dτ = 26 s,
  Δt = 313 s), about 16× fewer steps than `hevi`. This needs an `L` that matches `N`:
  - the vertical and horizontal fast parts are the linearization of the split-form `N` (two-point
    means), not pointwise products; otherwise `N − L` keeps an imaginary mode ≈ 0.02 (p+1) s⁻¹;
  - the vertical transport of momentum by the fast vertical mass flux, −∂(ρ̄w′ v̄)/∂z, is part of
    the implicit vertical `L`. Left in the explicit slow part, it couples the large step to the
    vertical acoustics with Δt |v̄| ∂z and makes a100 unstable.
- **Terrain-following FV** runs all three hills with `hevi` at the regular step (a100: Δt = 540 s).
  This needed a background-consistent `N` and `L` on the Gal-Chen grid: θ' = θ − θ̄(z) in the Θ
  flux with θ̄ ρ_t − ρw dθ̄/dz, and −∂p′/∂x|_z in a θ̄-weighted chain-rule form adjoint to the
  σ-corrected mass flux (otherwise a U mode grows at ≈ 1e-5 s⁻¹). As for DGSEM, the vertical
  transport of u by the fast vertical mass flux is part of the vertical `L`.
- **Agnesi hills, all discretizations** (MIS4, nsub = 12, 200 × 64 FV cells / 50 × 16 DG elements
  of degree 4, T = 21.6 a/U):

  | | DGSEM `hevi_verlet` | FV terrain-following `hevi` | FV cut cells `rosenbrock_amf` |
  |---|---|---|---|
  | large steps | 690 | 400 | 565 |
  | a1: max\|w\|, θ′ (m/s, K) | 2.77, −1.22…0.93 | 2.54, −1.09…0.90 | 2.17, −1.16…0.92 |
  | a10 | 0.70, −1.49…1.66 | 0.55, −1.37…1.52 | 0.59, −1.45…1.62 |
  | a100 | 0.064, −1.49…1.71 | 0.054, −1.37…1.53 | 0.057, −1.45…1.64 |

- **Planned fast solver:** a linear IMEX fast solver as in the MATLAB reference
  (`SplitExplicitLin`/`LinIMEX_RK`). It is explicit in `L_E` and implicit in `L_I`, with one sparse
  LU of `I − γΔτ L_I`. It would replace the product of approximate factors and allow a choice of
  whether the vertical direction is implicit everywhere or only in the mask.
- **Wider choices of `L_I`:** adaptive masks, for example advection implicit where the local
  advective CFL is large, and the same splittings for DGSEM.
- **Retained HDG traces:** the 1D `HDGRelaxTheta` prototype keeps face traces with a relaxation
  equation `ε dq̂/dt = Hq − q̂`. The 2D DGSEM fast operator currently eliminates the traces; `fb`
  can relax a stored face momentum (`EPS_IF`). Long hill runs are meant to keep the relaxed traces
  together with an HEVI vertical solve.

## References

- Wensch, Knoth & Galant (2009), *Multirate infinitesimal step methods for atmospheric flow
  simulation*, BIT 49, 449–473. [DOI: 10.1007/s10543-009-0222-3](https://doi.org/10.1007/s10543-009-0222-3)
- Jebens, Knoth & Weiner (2011), *Partially implicit peer methods for the compressible Euler
  equations*, J. Comput. Phys. 230, 4955–4974. [DOI: 10.1016/j.jcp.2011.03.015](https://doi.org/10.1016/j.jcp.2011.03.015)
- Knoth (2016), *Split-explicit methods and local linear splitting*, talk at ECMWF, 3 Oct 2016.
- Wicker & Skamarock (2002), *Time-splitting methods for elastic models using forward time schemes*,
  Mon. Wea. Rev. 130, 2088–2097.
- Gallus & Klemp (2000), *Behavior of flow over step orography*, Mon. Wea. Rev. 128, 1153–1164.
- Baldauf (2021), *A horizontally explicit, vertically implicit (HEVI) discontinuous Galerkin scheme
  for the 2-dimensional Euler and Navier–Stokes equations using terrain-following coordinates*.
  [DOI: 10.1016/j.jcp.2021.110635](https://doi.org/10.1016/j.jcp.2021.110635)
- Giraldo et al. (2023), *A performance study of horizontally explicit vertically implicit (HEVI)
  time-integrators for non-hydrostatic atmospheric models*. [arXiv:2311.11425](https://arxiv.org/abs/2311.11425)
- Xu, van der Vegt & Bokhove (2008), *Discontinuous Hamiltonian finite element method for linear
  hyperbolic systems*. [DOI: 10.1007/s10915-008-9191-y](https://doi.org/10.1007/s10915-008-9191-y)
- McLachlan & Stern (2014), *Modified trigonometric integrators*.
  [DOI: 10.1137/130921118](https://doi.org/10.1137/130921118)
