"""
    MIS

Construction and analysis of MIS methods with polynomial coupling (docs/mis_polynomial/MISPolynomial.tex,
Definition 3.1): for i = 2, …, s+1, Y₁ = yⁿ,

    Z_i(0) = yⁿ + Σ_j α_ij (Y_j - yⁿ),
    Z_i'(τ) = Σ_j g_ij(τ) (Y_j - yⁿ) + h Σ_j p_ij(τ) F(Y_j, Z_i(τ)),   τ ∈ [0, 1],
    Y_i = Z_i(1),   yⁿ⁺¹ = Y_{s+1},

with polynomials p_ij, g_ij of degree K (K = 0: classical MIS, Knoth & Wensch 2014).

  * `PolyMIS`: the coefficients; from a `TimeStepping.MISTableau` (`PolyMIS(mis_tableau(:MIS54))`)
    or given directly (`etd2rk()` as example of genuinely polynomial coupling).
  * `trees(p; class)`: rooted trees with bicoloured non-root vertices (black: derivative in the frozen
    argument y, white: in the integrated argument z), for the problem classes `:G` (general F(y, z)),
    `:B` (affine in z), `:A` (additive f(y) + g(z)), `:L` (f(y) + L z).
  * `residuals(m, p; class)`, `order(m; class)`: the B-series recursion (Theorem 4.1) in generic
    arithmetic (Rational, Float64, dual numbers), the residuals η_{s+1}(t) - 1/γ(t).
  * Test equation: 1D acoustics with advection, u_t + U u_x + c p_x = 0, p_t + U p_x + c u_x = 0,
    split into the fast acoustic part L and the slow advection S. Fourier/Bloch symbols of several
    spatial discretizations (`fv_symbols`: C-grid; `dg_symbols`: DGSEM with the LMARS (upwind)
    acoustic flux and upwind advection), scaled by the Courant numbers (Δt = 1, mean node spacing 1).
  * `amplification(m, S, L; inner)`: the step matrix for y' = S y + L y with exact inner problems
    (`:exact`) or with substeps (`(:rk4, n)`, `(:fb, n)` forward–backward, `(:verlet, n)`);
    `max_amplification`, `adv_limit`: linear stability over wavenumbers and Courant numbers.
"""
module MIS

using LinearAlgebra
using ..TimeStepping: MISTableau
using ..Euler2D: gll, lagrange_basis

export PolyMIS, etd2rk, commutator_free, cf4, lowstorage_cf, luscher3, ck54_cf, base_method, stage_times, coupling_degree, uniform_fast_speed,
       Tree, trees, tree_count, residuals, order, error_norm, restart_matrices, restart_conditions, latex_conditions,
       fv_symbols, dg_symbols, symbols, amplification, max_amplification, adv_limit, stability_map

include("tableau.jl")
include("trees.jl")
include("bseries.jl")
include("conditions.jl")
include("acoustic1d.jl")
include("stability.jl")

end # module
