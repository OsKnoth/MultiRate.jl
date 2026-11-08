"""
    TimeStepping

Time integrators shared by all spatial discretizations (DGSEM, FV). A spatial model owns the
nonlinear RHS and constructs its frozen fast operator; `MISStepper` holds only integration
policy, so it can be reused with a model implementing this interface:

  * split-explicit MIS / Wicker–Skamarock RK3 (`run_mis`, `mis_step!`) for q' = N(q) with
    the fast part L_n (frozen at qⁿ) and the slow part S = N - L_n,
  * fast integrators for q' = L_n q + f (`fast_solve!`): :imex, :hevi, :verlet, :fb, :wrf,
  * Lawson RK (`LawsonTableau`, :LRK4, :LRK3, :LRK2) with the fast integrators as propagator,
  * ETD RK in stage-ODE form (`ETDTableau`, :EB1, :ETD2, :EB4), the stage forcing a polynomial
    in the local time (`fast_solve_poly!`),
  * EPIRK / EPIRK-W (`EPIRKTableau`, :EPIRKW3a, :EPIRKW3b, :EPIRK4) and exponential
    Adams–Bashforth multistep methods (`EABTableau`, :EAB1 … :EAB4) in the same stage-ODE form,
  * classical RK4 on N (`run_rk4`) as reference.

A discretization provides a model and a fast operator (subtype of `AbstractFastOperator`) and
implements the interface below (methods of the functions declared in `interface.jl`).
"""
module TimeStepping

export AbstractFastOperator,
       rhs_nonlinear, rhs_nonlinear!, fast_operator, fast_apply!, fast_workspace, fast_forcing, fast_mass,
       fast_rhs_thermo!, fast_rhs_momentum!, interface_update!,
       fast_rhs_horizontal!, vertical_apply!, vertical_factor, vertical_solve!,
       fast_rhs_hmomentum!, fast_rhs_hthermo!,
       fast_solve!, fast_solve_poly!, composition, composition_stretch, composition_cost, IMEX_GM, IMEX_DL, IMEX_AE, IMEX_AI, WRF_BETA,
       SplitTableau, MISTableau, LawsonTableau, ETDTableau, EPIRKTableau, EABTableau, EABWork, mis_tableau, lawson_tableau,
       etd_tableau, epirk_tableau, eab_tableau, split_work, reset!, eab_advance!, eab_step_exact, phi_functions, mis_from_paper, mis_kw, nstages, fast_time,
       mis_rk_tableau, rk_order_residuals, rk_order, mis_step_exact, mis_observed_order,
       MISStepper, MISWork, stage_substeps, mis_step!, mis_step, run_mis, run_rk4

using LinearAlgebra: I, norm

include("interface.jl")
include("fast.jl")
include("tableaus.jl")
include("mis.jl")
include("lawson.jl")
include("etd.jl")
include("epirk.jl")
include("analysis.jl")

end # module
