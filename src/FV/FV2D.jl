"""
    FV2D

WRF-style finite-volume discretization (Klemp, Skamarock, Dudhia 2007) of the compressible
Euler equations with gravity in an x–z slice, flux form, potential-temperature variables:

    ρ_t + ∂x U + ∂z W = 0
    U_t + ∂x(U u) + ∂z(W u) + ∂x p = 0
    W_t + ∂x(U w) + ∂z(W w) + ∂z p + g ρ = 0
    Θ_t + ∂x(U θ) + ∂z(W θ) = 0,        p = p00 (R Θ/p00)^γ,  Θ = ρθ

background-free: full p and ρ (the analytic hydrostatic state is balanced to O(Δz²)).
With `FV_HYDRO_REFERENCE[] = true` p' = p - p̄(z), ρ' = ρ - ρ̄(z) of a hydrostatic reference are
used instead (discrete rest state exact).

Grid: Arakawa C with periodic x and a Gal-Chen vertical map
z = zs(x) + η (H-zs(x))/H. The mapped Jacobian J and terrain slope zx enter conservative fluxes
as J Fx horizontally and Fz - zx Fx vertically. The lower wall imposes zero normal flux.
ρ, Θ are at cell centres, U = ρu at x-faces, W = ρw at interior mapped z-faces. On a sloping
wall the physical vertical velocity can be nonzero while the normal flux remains zero.
Advection: 3rd-order upwind-biased fluxes (WRF flux3) on flat grids; mapped terrain tendencies
use the metric-consistent C-grid flux form.

Fast operator (frozen at qⁿ): acoustic-gravity part linearized about qⁿ,
    ρ_t = -∂x U - ∂z W,   Θ_t = -∂x(θ̄ U) - ∂z(θ̄ W),
    U_t = -∂x(a Θ + b),   W_t = -∂z(a Θ + b) - g ρ   (p̄, ρ̄ subtracted with FV_HYDRO_REFERENCE),
    a = γ p/Θ, b = (1 - γ) p at the centres, θ̄ at the faces (averages).
It implements the TimeStepping interface: :imex, :verlet, :fb (no interface variables: the
C-grid split is already Hamiltonian), :hevi and :wrf (vertical part: tridiagonal system for W
per column after eliminating ρ and Θ, as in WRF).

State vector q = [ρ (nx nz); U (nx nz); W (nx (nz-1)); Θ (nx nz)], cell (i, k) at i + (k-1) nx.
"""
module FV2D

using LinearAlgebra
using SparseArrays
using ..Common: Gas, pressure, stratified_reference
import ..Common: project2d, total2d, output_fields, field_section, sound_speed_max, with_sponge
using ..Common: open_vtkhdf, write_vtkhdf!, plot_fields
using ..TimeStepping: AbstractFastOperator, rhs_nonlinear
import ..TimeStepping: rhs_nonlinear!, fast_operator, fast_apply!, fast_rhs_thermo!, fast_rhs_momentum!,
                       fast_rhs_horizontal!, vertical_apply!, vertical_factor, vertical_solve!,
                       fast_rhs_hmomentum!, fast_rhs_hthermo!, hevi_loop!, fast_forcing, fast_mass,
                       fast_workspace, fast_rosenbrock_amf!, frozen_ncoef, frozen_data!, frozen_fast!, frozen_stage_begin!, frozen_vertical_step!

export FV_HYDRO_REFERENCE, FVGrid, CutCellFVGrid, FVModel, FVFast, CutCellFVAMF, with_sponge, project2d, total2d, output_fields, sound_speed_max,
       rhs_nonlinear, rhs_nonlinear!, fast_operator, fast_matrix, stratified_reference,
       iρ, iU, iW, iΘ, field_section, open_vtkhdf, write_vtkhdf!, plot_fields

include("grid.jl")
include("cutcell_grid.jl")
include("model.jl")
include("advection.jl")
include("rhs.jl")
include("fast_operator.jl")
include("vertical.jl")
include("cutcell_solver.jl")
include("cutcell_amf.jl")
include("frozen.jl")

end # module
