# Findings

Results of the numerical experiments, in the order they were obtained. Each entry says what was
measured, how, and what follows from it. Commands are run from the repository root.

## 1. Discretizations without a background state

- **DG** uses the pointwise fast operator `L`; **terrain-following FV** uses the `:flux` form of
  Θ; **cut cells** use `:donor`. None of them uses θ̄(z).
- **FV without a hydrostatic reference p̄, ρ̄** (`FV_HYDRO_REFERENCE[] = false`, the default): the
  momentum equations use full p and ρ. The analytic hydrostatic initial state is balanced to
  O(Δz²) (test: residual rate > 1.9 between nz = 6 and 12).
- At rest, 6 h on the a10 grid, max|w|: terrain FV 4.1e-3 m/s (1e-11 with the reference), cut
  cells 9e-3 m/s (3e-5 with the reference).
- **Ground pressure in the Gal-Chen correction −∂η(z_x p):** the ground-face p must be
  extrapolated from the face means (`fv_p_sigma`), so that its O(Δη²) error matches the interior
  means and cancels. Pointwise 2nd/3rd-order extrapolation is only first order and gave an O(1)
  error (DomainError at rest).

## 2. Agnesi hills (h = 400 m, U = 10 m/s, N = 0.01 s⁻¹)

max|w| (m/s) and θ′ range (K) at the final time:

| | DG | FV | cut cells |
|---|---|---|---|
| a1 | 2.77, −1.22…0.93 | 2.54, −1.09…0.92 | 2.14, −1.08…0.91 |
| a10 | 0.69, −1.48…1.65 | 0.56, −1.39…1.55 | 0.58, −1.43…1.58 |
| a100 | 0.064, −1.49…1.70 | 0.056, −1.40…1.57 | 0.057, −1.43…1.61 |

- **a1, terrain FV:** 2Δx noise near the ground at the default step. It disappears with half the
  slow step (`NSUB=6`, max|w| 2.41), so it is an MIS slow-step limit on terrain terms left in
  `N − L`, not a spatial defect. It is also present with the hydrostatic reference and with the
  `:advective` Θ form. At twice the resolution the default step blows up; `NSUB=6` is stable (2.68).
- **Cut cells:** a warm layer of ≈ 0.35 K in the lowest cells downstream of the hill (a10), open.
- U ≠ 0 runs without the reference drift in mass by ≈ 2.5e-5 (3.7e-7 with it); attributed to the
  sponge relaxing to a q0 that is not a discrete equilibrium, not investigated further.

## 3. Gravity waves (Skamarock & Klemp 1994, flat ground)

`examples/euler2d_gravity_wave.jl` (300 km) and `examples/euler2d_gravity_wave_long.jl` (6000 km,
a = 100 km, Δx = 20 km, Δz = 625 m, U = 20 m/s, T = 60000 s), MIS4, `hevi`, nsub = 12:

- 300 km: DG and FV symmetric (θ′ maxima 2.73e-3/2.72e-3 K DG, 3.78e-3/3.32e-3 K FV).
- **6000 km, FV at the default step** Δt = 540 s (NΔt ≈ 5.4, UΔt/Δx = 0.54, cΔt/Δx ≈ 9.5):
  asymmetric packets (θ′ maxima 2.2e-3 and 9.0e-3 K) and grid-scale noise between them. Identical
  with the hydrostatic reference and on the commit before the background was removed, and with
  `:hevi` and the default solver. With `NSUB=3` (Δt = 135 s, same dτ) the packets are symmetric
  (5.9e-3, 4.8e-3 K) and the noise is gone. DG is clean at the default step (4.9e-3/4.9e-3 K).
  Since dτ is the same in both runs, the error comes from the slow step, not from the fast solver.
- **Cause and fix (§6):** the vertical transport of horizontal momentum by the fast vertical
  velocity, −∂z(ūW′), was left in the slow part although UΔt/Δz ≈ 17. The Agnesi driver puts it
  into L for UΔt/Δz > 4, the production gravity-wave drivers did not apply that criterion. With
  `set_vadv!` (now in all drivers) the default step gives θ′ maxima 5.34e-3 / 5.29e-3 K and an
  asymmetry of 4.4e-4 K (6.9e-3 K before). The 300 km case (UΔt/Δz = 2.1) and DG (0.6) are unchanged.
- The uniform ≈ 1e-3 K warm offset of θ′ in these runs is present also for a 100× smaller
  perturbation and for U = 0: a diagnostic offset of θ′ against θ̄(z) at the cell centres, not part
  of the wave (not the k = 0 part of the perturbation, as first assumed).

## 4. MIS tableaus

`TimeStepping.mis_observed_order` (exact fast stages, L and S not commuting) and `rk_order`
(L = 0):

| | WSRK3 | MIS2 | MIS3C | MIS4 | MIS4a | MIS_KW3 | MIS54 |
|---|---|---|---|---|---|---|---|
| RK order (L = 0) | 2 | 2 | 3 | 3 | 3 | 3 | 4 |
| MIS order | 2 | 2 | **2** | 3 | 3 | 3 | 4 |
| fast time Σ dᵢ / Δt | 1.83 | 1.19 | 1.19 | 1.44 | 1.47 | 1.00 | 1.88 |

- **MIS3C** (coefficients as in ClimaTimeSteppers.jl) is only second order once L ≠ 0: a coupling
  condition fails. To be checked against Knoth & Wensch (2014).
- **MIS54** (Bauer & Knoth 2021, from MISDG) is fourth order; it runs in the FV gravity-wave test.

## 5. Linear acoustics with advection (`MultiRate.LinearAnalysis`)

The MISDG talk these experiments come from is in [talks/](talks/README.md).
`examples/linear_analysis/mis_stability.jl`; amplification matrices per Fourier mode, computed by
running the production `mis_step!` and fast loops on unit vectors.

- **Knoth–Wensch test (exact fast stages):** pure advection is stable to C_adv = 1.63 (3-stage
  methods) and 2.06 (MIS54). Undamped WSRK3 is unstable from C_sound ≈ 3.2, MIS_KW3 from ≈ 2
  (and for any C_sound at C_adv = 1). MIS4a and MIS54 are stable on the whole box C_sound ≤ 12.
- **MISDG grid** (4th-order staggered acoustics, 5th-order upwind advection, C_adv = C_sound/7),
  exact fast stages: WSRK3 stable to C_sound = 2.75, MIS4 beyond 14, MIS4a to 12, MIS54 to 14.
  MIS_KW3 is weakly unstable for every C_sound (ρ − 1 = 7e-6 at C_sound = 1). Same as in MISDG.
- **Fast loops with nsub = 12:** `:verlet` and `:fb` reproduce the exact-stage result as dτ → 0
  (orders 2 and 1). At C_sound = 10 the Verlet substep exceeds its own limit for every tableau.
  `:fb` makes MIS4 and MIS54 weakly unstable at C_sound = 3–7, `:verlet` does not. WRF divergence
  damping (`:wrf`) stabilizes WSRK3 up to C_sound = 7.
- **Pulse convergence** (exact fast stages): orders 2 (WSRK3), 3 (MIS4a), 4 (MIS54).
- Consequence: pure acoustics + advection does not explain the slow-step limits of §2 and §3; the
  MIS methods in use (MIS4, MIS4a) are stable far beyond the Courant numbers of those runs.

## 6. Linearized production models: the long-channel failure (`fv_gravity_wave_split.jl`)

`examples/linear_analysis/fv_gravity_wave_split.jl`. The flat FV model is invariant under shifts
in x. `LinearAnalysis.linearized_mode_model` builds, per horizontal Fourier mode, the column
blocks (63 × 63) of the Jacobian J of `rhs_nonlinear!` (central differences) and of the frozen fast
operator L and its vertical part; `step_matrix` runs the production MIS step on them. The
eigenvalues of all blocks together equal those of the full `fast_matrix` (test).

**Steps that led to the cause:**

1. The real FV run is linear in the amplitude: with Δθ = 1e-4 instead of 1e-2 the packet asymmetry
   scales by 1e-2 (7.0e-5 K). With U = 0 the asymmetry is 1e-5 K (1e-7 K for Δθ = 1e-4).
2. A constant-coefficient model with the FV grids (`boussinesq_symbols`: compressible Boussinesq,
   C-grid, Lorenz grid, rigid lid, buoyancy and acoustics in L, advection in S) is accurate at
   Δt = 540 s: max error 1.5e-4 of a 4.5e-3 K packet with the FV `:hevi` loop. Moving the advection
   of any rows between L and S, or anti-advecting u in L, changes this by at most 1.4%. So the
   failure is not in the generic acoustic-gravity-advection problem.
3. The linearized production FV reproduces the nonlinear run exactly (θ′ maxima 1.319e-3 /
   8.017e-3 K vs 1.320e-3 / 8.022e-3 K with the reference). The exact linear solution is
   symmetric (4.50e-3 / 4.46e-3 K). With exact fast stages the error remains (6.1e-3 / 2.8e-3 K), so
   it is the slow-step coupling, not the fast solver; the `:hevi` loop adds instability
   (max |λ| = 1.15, growth up to 0.15 per step at 4Δx–8Δx).
4. Moving S rows into L, exact fast stages, Δt = 540 s: the horizontal-momentum rows reduce the
   error from 1.67e-3 to 2.8e-5 K, the Θ rows do nothing. Writing L's mass and Θ fluxes with the
   perturbation momentum m − ūρ (L as primitive acoustics) does nothing either.
5. The stiff part of S's momentum rows is the vertical transport −∂z(ūW′) with coefficient
   ūΔt/Δz ≈ 17 — the term that `FV_VADV_FAST` already puts into L (§2, a100).

**Result** (MIS4, nsub = 12 / 3, modes j ≤ 60 of 300; 2.8e-5 K is the truncation floor):

| L | fast | Δt [s] | packets [K] | max err [K] | max \|λ\| |
|---|---|---|---|---|---|
| as in the driver (no vadv) | exact | 540 | 6.13e-3 / 2.80e-3 | 1.7e-3 | 1.0000 |
| | `:hevi` | 540 | 1.36e-3 / 7.99e-3 | 4.0e-3 | 1.151 |
| | `:hevi` | 135 | 4.94e-3 / 3.91e-3 | 5.6e-4 | 1.0034 |
| vertical u-transport in L | exact | 540 | 4.55e-3 / 4.38e-3 | 1.2e-4 | 1.0000 |
| | `:hevi` | 540 | 4.42e-3 / 4.38e-3 | 3.7e-4 | 1.0000 |
| | `:hevi` | 135 | 4.46e-3 / 4.47e-3 | 2.0e-4 | 1.0000 |
| all S momentum rows in L | `:hevi` | 540 | 4.42e-3 / 4.76e-3 | 4.5e-4 | 1.0015 |

exact linear: 4.50e-3 / 4.46e-3 K. Regression test: a 30-column grid with the same Δx, Δz, Δt has
max |λ| > 1.1 at 7.5Δx without vadv and < 1 with it.

**Consequences:**

- The vadv criterion (UΔt/Δz > 4) is a property of the step, not of a test case: it is now applied
  by every driver through `set_vadv!`.
- Constant-coefficient model problems (§5, step 2) are good for the tableaus and fast loops but
  missed this; the linearized production operator is the reliable diagnostic. It needs
  shift invariance in x, so terrain (the a1 noise of §2, UΔt/Δz = 0.2, a different mechanism) needs
  the matrix-free variant (Arnoldi on the linearized step) still to be written.
- With `:wrf` (WRF's off-centred vertical step and divergence damping) instead of `:hevi`, the
  default step is stable without vadv (max |λ| = 1.0014) but still inaccurate (3.6e-3 / 5.7e-3 K).
