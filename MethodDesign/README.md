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

1. **Frozen-order-3 MIS with 4 stages (found).** `search.jl 4 3` finds a whole family of 4-stage
   methods that satisfy all 10 bicoloured conditions up to order 3 (MIS4/MIS4a satisfy only the 9
   additive ones … and fail [.,o]). Next: select within the family by
   - linear stability of the split-explicit step (`MultiRate.LinearAnalysis.max_amplification`
     on the MISDG/Boussinesq modes at large Courant numbers, with the real fast loops),
   - small order-4 residuals (error constants), positive stage times d_i, moderate coefficients.
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
   mixed trees included.
6. **Lawson/ETD analogues of the frozen form**: replace the constant weights by φ-function weights
   in the stage-ODE form (the polynomial forcing of `etd.jl`), conditions again from the recursion
   with polynomial a_ij(λ).
