"""
    FV3D

Finite volumes on a 3D Arakawa C-grid (periodic in x and y, walls at the bottom and the top),
flat or with Cartesian cut cells for a terrain z_s(x, y), in the frozen form of the right-hand side
for split-explicit MIS (`TimeStepping.run_frozen`): N(q) = F(q, q), with

  fast  thermodynamic part (mass and Θ fluxes through the open face areas, θ frozen at the faces)
        and momentum part (pressure gradient :gradp or in product form :product, gravity),
  slow  momentum advection,

one advection scheme for flat grids, cut cells and terrain following (`advection.jl`: pseudo
third-order upwind with virtual grid sizes from volumes and face areas, shift technique for the
momentum; Jähn et al. 2015),

built from separate blocks (`operator.jl`); the parts of the fast loops (:all, :thermo,
:momentum, :hthermo, :hmomentum, :vertical) are compositions of them. HEVI: tridiagonal column
solve for W (`column.jl`). No background state is subtracted; the sponge damps W only.
Moist air (`moist.jl`, `MoistModel3D`): the 2D model `FV2D.MoistExnerModel` in 3D (ρθ_ρ / ρe / ρs, ρq_v, ρq_c and the
Exner pressure, Fischer–Burmeister relaxation instead of a saturation adjustment), flat or terrain following.
"""
module FV3D

using ..Common: Gas, pressure
using LinearAlgebra, SparseArrays
using ..FV2D: recon3, shiftvel, shift_back
import ..TimeStepping: rhs_nonlinear!, frozen_ncoef, frozen_data!, frozen_fast!, frozen_stage_begin!,
                       frozen_vertical_step!, frozen_vertical_rhs!, frozen_vertical_implicit!, frozen_vertical_solve!,
                       frozen_phase!, frozen_step_end!

export Grid3D, Model3D, project3d, total3d, with_sponge, sponge!, with_nudging, sound_speed_max, theta_prime,
       vertical_velocity, hydrostatic_column, balanced_state, level, FV3D_PRESSURE_FORM, iρ, iU, iV, iW, iΘ, nunknowns,
       MoistModel3D, iqv, iqc, iπ, iqr, moist_exner_state, base_state, thermo_convert, theta_e3d, qc3d, qr3d,
       moist_column_solve!, precipitation_rate, rain_courant, msed!, Smagorinsky, smagorinsky!, eddy_viscosity,
       filter_length

include("grid.jl")
include("lsq.jl")
include("model.jl")
include("sponge.jl")
include("operator.jl")
include("advection.jl")
include("advection_jacobian.jl")
include("krylov.jl")
include("column.jl")
include("smagorinsky.jl")
include("moist.jl")

end
