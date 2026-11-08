# MethodDesign: order conditions and new split-explicit methods

Ideas and tools to construct new split-explicit (MIS-type) methods for the **frozen** form of the
right-hand side, N(q) = F(q, q), where the first argument is frozen at the stage values and the
second evolves in the fast integration (`MultiRate.TimeStepping.run_frozen`; FV: `src/FV/frozen.jl`,
DG: `src/DGSEM/Euler2D/frozen.jl`). Theory: O. Knoth, J. Wensch, *B-Series and Split-Explicit
Methods*, github.com/OsKnoth/B-Series (`PDF/DocuMISBI.tex`, Fortran program).

Separate environment (Symbolics, Optim, ForwardDiff):

    julia --project=MethodDesign -e 'using Pkg; Pkg.instantiate()'

## Files

- `BiColourMIS.jl` — bicoloured trees and the B-series of methods with ODE stages
  `dZ_i/dλ = h Σ_j a_ij(λ) F(Y_j, Z_i) + Σ_j γ_ij (Y_{j+1} - yⁿ)`, `Z_i(0) = yⁿ + Σ_j α_ij (Y_{j+1} - yⁿ)`
  (the recursion of DocuMISBI with shifted initial values, extended by the γ terms of MIS).
  Generic arithmetic: Float64, Rational, ForwardDiff duals, Symbolics variables.
- `order_conditions.jl` — explicit (symbolic) order conditions of a general s-stage MIS method,
  each marked A (needed also for an additive F = f(y) + g(z)) or M (mixed: only for a
  non-additive F), and the residuals of the MIS tableaus of the package.
- `search.jl` — numerical search for MIS coefficients that satisfy all bicoloured conditions up to
  order p (LBFGS with AD on Σ residual², then Levenberg–Marquardt with minimum-norm steps).
- `design_tools.jl` — packing of (α, β, γ), residual vector, projection `refine` onto the order
  conditions (minimum-norm Levenberg–Marquardt), error constant (norm of the order-(p+1) residuals).
- `stability.jl` — linear stability on the acoustic-advection test of Knoth & Wensch (2014)
  (C-grid, exact fast stages): `amplification`, `adv_limit` (largest stable C_adv per C_sound),
  `stability_penalty`. As a script: the limits of the package tableaus.
- `search_stable.jl` — stability-constrained search: candidates projected onto the order
  conditions, Nelder–Mead on Σ max(0, ρ − 1)² over a (C_sound, C_adv) grid plus 10⁻³·error
  constant²; optional start from a stable package tableau (`… 5 3 4 2.0 MIS54`).
- `app_stability.jl` — application-specific stability: column Fourier blocks of the *real* FV and
  DG frozen splits (J of `rhs_nonlinear!`, L = ∂F/∂z of `frozen_fast!`, coefficients frozen at a
  hydrostatic state with uniform U; whole column with gravity and the vertical discretization),
  exact-fast-stage MIS step in the eigenbasis of L (checked against `LinearAnalysis.step_matrix`).
- `search_app.jl` — the search of `search_stable.jl` with the application penalty
  (`… search_app.jl fv 5 3 MISF53 3 1.6`, `… dg 5 3 MISF53 2 1.2`).

## Can we derive explicit order conditions for MIS from the B-series theory? Yes.

For y' = F(y, y) the exact solution is the B-series over **bicoloured trees**: rooted trees whose
edges carry a colour, black = derivative of F with respect to the first (frozen) argument, white =
with respect to the second (evolving) one; its coefficient is 1/γ(t) for every colouring. A method
with ODE stages has stage series Y_j = B(η_j), Z_i(λ) = B(ζ_i(λ)); inserting them into the stage ODE
gives, for t = [t_b1 … t_bk, t_w1 … t_wl],

    ζ_i(t)(λ) = Σ_j α_ij η_{j+1}(t) + λ Σ_j γ_ij η_{j+1}(t) + ∫_0^λ Σ_j a_ij(μ) Π η_j(t_bm) Π ζ_i(t_wm)(μ) dμ,
    η_{i+1}(t) = ζ_i(t)(1),

and the conditions η_{s+1}(t) = 1/γ(t) for all bicoloured trees with ≤ p vertices. Numbers of trees:

| order | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| all bicoloured trees (non-additive F) | 1 | 2 | 7 | 26 | 107 |
| additive F = f(y) + g(z) (vertices with children of one colour) | 1 | 2 | 6 | 18 | 60 |

The recursion reproduces the orders observed numerically (toy problem with a frozen coefficient,
`test/runtests.jl`) for all MIS tableaus of the package:

| | additive F | non-additive F (frozen coefficient) |
|---|---|---|
| WSRK3, MIS2, MIS3C | 2 | 2 |
| MIS4, MIS4a | 3 | 2 |
| MIS54 | 4 | 2 (fails only [.,o] = F_(1,1)FF at order 3, residual 9.3e-3) |
| MIS_KW3 | 3 | 3 |

Relation to BSeries.jl (Ranocha, Ketcheson; github.com/ranocha/BSeries.jl) and RootedTrees.jl:
they provide B-series of RK, additive RK (bicoloured trees with coloured *vertices*, one colour per
function f, g) and Rosenbrock methods, with composition and substitution. MIS with ODE stages and a
two-argument F(y, z) is not covered: here the colour is on the *edges* (derivative argument), and
the stage series are polynomials in the local time λ. The recursion above could be added there as a
new method type (the trees can be mapped to RootedTrees' bicoloured trees by moving each edge colour
to its child vertex, the root colour being free).

## Ideas for new methods

1. **Frozen order 3 with stability (found: MISF53).** Stability on the Knoth–Wensch test
   (largest stable C_adv for C_sound = 0, 1, 2, 3, 4, 6, 8, 10, 12; exact fast stages):

   | method | frozen / additive order | error const. | C_adv limits |
   |---|---|---|---|
   | MIS4a | 2 / 3 | | 1.95 · 1.70 · 1.65 · 2.05 · 2.95 (C_s = 0, 2, 4, 8, 12) |
   | MIS54 | 2 / 4 | | 2.05 · 2.30 · 2.70 · 2.35 · 3.00 (C_s = 0, 2, 4, 8, 12) |
   | random starts, s = 4 / 5 | 3 / 3 | 0.11–0.23 | stable only for C_sound ≤ 2 |
   | MIS4a projected | 3 / 3 | 0.10 | 1.75 1.60 1.45 1.65 0.00 (C_s = 0, 2, 4, 8, 12) |
   | MIS4a projected + Nelder–Mead | 3 / 3 | 0.10 | 1.90 1.80 1.85 1.80 1.25 1.35 1.70 2.00 0.55 |
   | **MISF53** = MIS54 projected | **3 / 3** | **0.024** | **2.05 2.20 2.45 2.45 2.40 2.20 2.50 3.00 3.00** |

   The frozen-order-3 family is large, but its stable part is small: random starts never reached it.
   The minimum-norm projection of MIS54 onto the 10 bicoloured conditions moves the coefficients
   by only |Δx| ≈ 0.1 and keeps the stability of MIS54 (`TimeStepping.mis_tableau(:MISF53)`).
   The 4-stage family seems to have no member stable at C_sound = 12. Frozen order 3 *and*
   additive order 4 with 5 stages: not found (residual 1.5e-5).
   Gravity wave, frozen split, wave error [K] at Δt = 67/100/130/167/200 s (FV `hevi`):
   MISF53 1.1e-4 9.6e-5 1.4e-4 1.4e-4 1.6e-4, MIS54 1.1e-4 9.6e-5 1.6e-4 1.2e-4 1.6e-4; DG
   (`verlet`, 67/130 s): MISF53 7.2e-5 / 2.4e-4, MIS54 7.9e-5 / 2.2e-4. MISF53 is as good as MIS54
   but not better here: on this test the time error is dominated by other terms, not by the
   mixed tree [.,o] that MIS54 misses. MISF43 (4 stages, earlier, not stability-selected) is
   unstable (amplification 1.035–1.11).
   **The 1D test is not enough: FV and DG need their own methods.** On the column blocks of the
   frozen splits (gravity-wave grids, U = 10/20/40 m/s) the methods that are stable on the 1D test
   grow by 5–25 % per step (FV MIS54 from C_adv = 0.4). Reason: in the frozen split ρ is
   advected in the fast part, while Θ advection is split by the frozen face θ into θ̄ū δρ (fast)
   and ū(δΘ − θ̄δρ) (slow); the 1D test has no such coupling. Methods designed on the blocks
   (`search_app.jl`, ρ − 1 ≤ 1e-3 counted as stable):

   | | frozen order | FV: max ρ − 1, C_adv ≤ 1.6 (U = 10/20/40) | DG: max ρ − 1, C_adv ≤ 1.2 (U = 20/40) |
   |---|---|---|---|
   | MIS54 | 2 | 0.16 / 0.18 / 0.10 | 0.94 / 0.96 |
   | MISF53 | 3 | 0.14 / 0.26 / 0.26 | 0.96 / 0.73 |
   | MIS_FV42 (from MIS4) | 2 | 7e-4 / 7e-3 / 7e-4 | |
   | **MIS_FV53** (from MISF53) | 3 | **−3e-4 / −1e-4 / 6e-4** (2.0: 2e-2 / −3e-4 / 1e-3) | 0.90 / 0.76 |
   | **MIS_DG53** (from MISF53) | 3 | | **1.3e-4 / 1.4e-4** (C_adv 1.6: > 3 for all methods) |

   4 stages with frozen order 3 found no stable FV method (penalty stalls at 0.11). Nonlinear
   checks (`examples/frozen_new_methods_fv.jl`, `…_dg.jl`): FV, long run 36000 s at Δt = 200 s
   (C_adv 1.6): only MIS_FV53 survives (MIS54, MISF53, MIS_FV42 blow up); its 3000 s wave error
   1.2e-4 / 1.8e-4 at 130 / 200 s (MIS54 1.6e-4 / 1.6e-4). DG (Verlet): MIS_DG53 is the only
   method that runs at Δt = 150 s (6.0e-4; MIS54 1.4, MIS_FV53 unstable), at 130 s 4.1e-4
   (MIS54 2.2e-4); all fail at 167 s and above.
2. **Frozen order 4.** 26 conditions at order 4. 6 stages give 51 free coefficients; the search
   reaches residuals 1e-6 … 1e-4 so far, not zero — either more stages or the time-dependent
   weights below are needed.
3. **Time-dependent weights** a_ij(λ) = Σ_p a_ij^p λ^p with Σ_j a_ij^p = 0 for p ≥ 1 (DocuMISBI):
   the frozen argument follows the stage time instead of a fixed average. More freedom without more
   stages (and without more slow evaluations); the fast loop needs the frozen coefficients as a
   polynomial in λ (as `fast_solve_poly!` does for ETD).
4. **Exploit the structure of the split.** With the frozen dp/dΘ the FV fast part is linear in z,
   F_(k,l) = 0 for l ≥ 2: all trees with a vertex of two or more white children drop out. The DG
   split is linear in z except the nodal p(Θ_z). The slow part S(y) has no z-dependence, so white
   children only hang below fast vertices. Counting the remaining trees gives the conditions for the
   actual splits — fewer than for a general F.
5. **Error-optimal and stability-optimal families** as in Bauer & Knoth (MIS54): minimize the
   order-(p+1) residuals subject to stability on the acoustic-advection test problem, now with the
   mixed trees included (`search_stable.jl`, first result MISF53 above; next: 6 stages, frozen
   order 3 with additive order 4, and the real fast loops in the stability test).
6. **Lawson/ETD analogues of the frozen form**: replace the constant weights by φ-function weights
   in the stage-ODE form (the polynomial forcing of `etd.jl`), conditions again from the recursion
   with polynomial a_ij(λ).
