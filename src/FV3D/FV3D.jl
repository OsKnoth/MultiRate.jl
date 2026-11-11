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
"""
module FV3D

using ..Common: Gas, pressure
using LinearAlgebra, SparseArrays
using ..FV2D: recon3, shiftvel, shift_back
import ..TimeStepping: rhs_nonlinear!, frozen_ncoef, frozen_data!, frozen_fast!, frozen_stage_begin!,
                       frozen_vertical_step!

export Grid3D, Model3D, project3d, total3d, with_sponge, sponge!, sound_speed_max, theta_prime,
       vertical_velocity, hydrostatic_column, balanced_state, level, FV3D_PRESSURE_FORM, iρ, iU, iV, iW, iΘ, nunknowns

include("grid.jl")
include("lsq.jl")
include("model.jl")
include("sponge.jl")
include("operator.jl")
include("advection.jl")
include("advection_jacobian.jl")
include("column.jl")

end
