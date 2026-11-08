# Findings

Results of the numerical experiments, in the order they were obtained. Each entry says what was
measured, how, and what follows from it. Commands are run from the repository root.

A LaTeX version with the figures is [findings/findings.tex](findings/findings.tex)
([PDF](findings/findings.pdf)).

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

## 7. Matrix-free analysis of the linearized step (preliminary, terrain paused)

`LinearAnalysis.linearized_step(model, qb, Δt, tab; nsub, fast)` applies the linearized
production MIS step (fast operator frozen at qb, central differences of `mis_step!`) to a vector;
it needs no shift invariance. `arnoldi_eigs` (restarted Arnoldi) finds the eigenvalues of largest
modulus of such a map.

- On the a1 terrain-FV grid (51k unknowns) Arnoldi on the step itself does not converge: almost
  all eigenvalues lie near the unit circle. Power iteration (repeated application) works.
- First result, a1 FV linearized about the initial state, MIS4, `:hevi`: growth 1.027 per step at
  the default step (`NSUB=12`, Δt = 5.4 s), 1.0001 (neutral) at `NSUB=6`. The growing mode is a
  2–3Δx oscillation of U and W in the lowest three levels on the steep flanks (slope ±0.25), where
  the noise of §2 appears. Not yet separated: fast loop vs slow-step coupling, and which terrain
  term of S = N − L is responsible. Paused (no orography tests for now).

## 8. Lawson RK with the MIS fast integrators as propagator

`TimeStepping.LawsonTableau` (`:LRK4` classical RK4, `:LRK3` Kutta, `:LRK2` midpoint; selectable by
name like the MIS tableaus, env `TABLE` in the drivers). With the fast flow φ_τ of the frozen L
(including its constant forcing) and S = N − L:
Y_i = φ_{c_iΔt}(qⁿ) + Δt Σ_j a_ij e^{(c_i−c_j)ΔtL} S(Y_j). The propagator is the fast loop of the MIS
step (`fast_solve!` without slow forcing, dτ ≈ Δt/nsub); φ_{cΔt}(qⁿ) is computed once per node.
`mis_step!` dispatches on the tableau type, so `run_mis`, the drivers and `LinearAnalysis` work
unchanged. Fast integration time per step: LRK4 2.5 Δt (MIS4 1.44 Δt).

- **Order** (exact propagators, non-commuting test): LRK2 2, LRK3 3, LRK4 4. With the Verlet loop
  the step converges to the exact one at order 2 in dτ.
- **Linear stability:** limited by the advection Courant number as for RK4 — C_adv ≤ 1.75 (LRK4),
  1.62 (LRK3) on the Knoth–Wensch test (MIS54 2.06); stable fraction of the box 0.78 (MIS54 0.99).
  On the MISDG grid (C_adv = C_sound/7) LRK4 with Verlet, nsub = 12, is stable up to C_sound = 10,
  where MIS4 and MIS54 with Verlet are already unstable (equal-length Lawson propagations vs the
  rounding of d_i nsub in MIS); |ρ| − 1 ≈ 1e-8 vs 1e-5 for MIS4.
- **Accuracy, exact propagators:** pulse error 190× smaller than MIS54 at the same step
  (5.3e-6 vs 9.9e-4), order 4. Lawson is exact for S = 0; MIS with α, γ ≠ 0 is not: for an
  unbalanced initial state (the gravity-wave θ′ at constant pressure) all MIS tableaus damp or
  dephase the fast acoustic part by a Δt-independent amount (4e-3 relative in the linear
  Boussinesq wave, also for the k = 0 mode), Lawson keeps it (5e-13).
- **Accuracy with the real fast loops:** the propagator error enters directly. Pulse with Verlet:
  LRK4 error = the 2nd-order Verlet error (rates 2.00), the same as MIS54. Nonlinear FV gravity
  wave (300 km, `:hevi`) against explicit RK4 on N: max θ′ error 1.6e-3 / 1.7e-3 / 3.7e-4 K at
  Δt = 67 / 34 / 17 s for LRK4, 1.1e-3 / 8.6e-4 / 6.0e-4 K for MIS4 (max θ′ 4.4e-3 K). All
  methods are dominated by the fast-loop error (vertical Crank–Nicolson at large vertical Courant
  numbers).
- **Consequence (revised in §9):** with second-order fast loops the Lawson error at small slow
  steps is the fast-loop error. The comparison above mixes the gravity wave with an oscillation of
  the horizontal mean (see §9); at large slow steps and with a fourth-order fast loop the picture
  is different.

## 9. ETD, high-order fast loops, MIS vs Lawson/ETD at large slow steps, optimal number of substeps

**ETD in stage-ODE form** (`TimeStepping.ETDTableau`, after the MATLAB `SplitExplicitETD.m` /
`MethodETD.m`): stage i starts from qⁿ and integrates Z' = L Z + Σ_j Σ_k P_ijk τ^{k−1} S(Y_j) over
c_iΔt, τ the normalized local time, P = a/((k−1)! c_i) from the φ_k coefficients a. Exactly
integrated this is the ETD stage e^{c_iΔtL}qⁿ + Δt Σ a_ijk φ_k(c_iΔtL) S_j. Tableaus :EB1, :ETD2
(Cox–Matthews), :EB4 (Krogstad ETDRK4); fast time 3Δt per step for EB4. `fast_solve_poly!` runs
the fast loops with the forcing taken at the midpoint of every base step, which keeps symmetric
loops symmetric in (q, t). Orders with exact fast stages: 1, 2, 4.

**High-order fast loops:** `:<base>_y4` (Yoshida triple jump, weights 1.35, −1.70), `:<base>_s4`
(Suzuki, 5 stages, |w| ≤ 0.66, stable at the regular substep), `:<base>_y6` (Yoshida, 7 stages)
compose a symmetric loop (`:verlet`, FV `:hevi`, `:hevi_verlet`); orders 4, 4, 6 also with the
time-dependent ETD forcing. Suzuki's error constant is 60× smaller than the triple jump's.

**Linear pulse at large steps** (`examples/linear_analysis/lawson_mis_pulse.jl`, h = 50 s,
C_sound = 7, C_adv = 1; relative error at T):

| | exact fast stages | Verlet, nsub = 48 | Suzuki, nsub = 8 |
|---|---|---|---|
| MIS4a | 9.9e-3 | 9.9e-3 | 9.9e-3 |
| MIS54 | 9.9e-4 | 8.4e-4 | 1.0e-3 |
| EB4 | 1.6e-3 | 1.6e-3 | 1.6e-3 |
| LRK4 | **5.3e-6** | 3.1e-4 | **5.8e-6** |

![](images/lawson_mis_pulse.png)

**Nonlinear FV gravity wave at large steps** (`examples/lawson_mis_gravity_wave.jl`, 300 km,
reference explicit RK4 on N). The initial θ′ at constant pressure is not balanced and excites a
vertical acoustic oscillation of the horizontal mean of θ′ (≈ 1e-3 K at z = 5 km). RK4, Lawson
and ETD keep it, MIS damps it (sections below: the MIS54 curve is shifted). The *wave error*
(horizontal mean removed) at the best number of substeps:

| wave error [K] | Δt = 67 s (UΔt/Δx = 0.53) | 130 s (1.04) | 200 s (1.6) |
|---|---|---|---|
| MIS4 `hevi` | 2.2e-4 | 5.6e-4 | unstable |
| MIS54 `hevi` | 7.5e-5 | **1.0e-4** | **3.2e-4** |
| LRK4 `hevi_s4` | 7.2e-5 | 2.0e-4 | unstable |
| EB4 `hevi_s4` | **6.3e-5** | 1.8e-4 | 1.1e-3 |

The total error (with the mean): MIS 0.9–1.4e-3 K at all steps; LRK4/EB4 with `hevi_s4`
6–9e-5 (67 s), 1.6–2.1e-4 (130 s); with `hevi` (second order) erratic, 1e-4 … 1.7e-3.

![](images/lawson_mis_gravity_wave.png)
![](images/lawson_mis_gravity_wave_sections.png)

**MIS vs Lawson/ETD:**

- MIS solves each stage as Z' = LZ + constant forcing, i.e. it is an ETD method with φ-functions
  built from the stage structure; with α, γ ≠ 0 it is not exact for S = 0 and damps or dephases
  fast modes at ωΔt ≫ 1 by a Δt-independent amount. Lawson and ETD are exact for S = 0 and keep
  the fast modes.
- Where the fast modes are the signal (the acoustic pulse) Lawson with an accurate propagator is
  far better (170× below MIS54). ETD (EB4) is not (1.6e-3); a plausible reason: its polynomial-in-time
  forcing interpolates S along the fast oscillation, Lawson interpolates in the frame moving with e^{tL}.
- Where the slow modes are the signal (gravity wave) MIS54 is as accurate as LRK4/EB4 at
  UΔt/Δx ≈ 0.5, more accurate at ≈ 1, and the only one of them stable at 1.6 (EB4 stable but 3×
  worse). MIS54 needs no high-order fast loop, so it is 5–8× cheaper per step.
- Stability limits in UΔt/Δx: MIS54 > 1.6; LRK4 and MIS4 < 1.6 (LRK4's RK4 advection limit 1.75
  on the linear test); EB4 1.6 but inaccurate.

**Optimal number of substeps:**

- *MIS:* the error does not depend on the fast loop once the substep is stable: the same error
  with `hevi`, `hevi_s4`, `hevi_y6` and for every nsub. The optimum is the smallest stable nsub,
  dτ at the acoustic limit (nsub = Δt/dτ_max); at most a margin of ~1.5 (MIS54 at Δt = 67 s:
  1.1e-4 at nsub = 12, 7.1e-5 at nsub = 24). A higher-order fast loop is wasted.
- *Lawson/ETD:* the wave error behaves like MIS (independent of the fast loop at the smallest
  stable nsub), but the fast part — and with it the total error — needs a fourth-order loop:
  `_s4` at the smallest stable nsub (its weights |w| ≤ 0.66 allow the regular dτ) reaches the
  exact-propagator error; more substeps or `_y6` bring nothing. Cost 5× the second-order loop.
  With a second-order loop the error decreases only like (Δt/nsub)² and needs nsub ≳ 4× larger.
