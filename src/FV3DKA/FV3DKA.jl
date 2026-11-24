"""
    FV3DKA

The moist 3D FV model (`FV3D.MoistModel3D`: ρ, momenta U, V, W, the thermodynamic variable X, ρq_v, ρq_c and the
pressure p (default) or the Exner pressure π; Fischer–Burmeister phase change, no saturation adjustment) written with
KernelAbstractions, so that it runs on the CPU and later on GPUs, with a horizontal (x–y) domain decomposition:

  grid.jl     tile of a rank (nxl × nyl full columns of the global nx × ny × nz grid, flat or terrain following)
              with a halo of `HALO` cells, metric built directly from the terrain function
  decomp.jl   halo exchange: `SerialDecomp` (one rank, periodic copies); MPI tiles in the extension
              `MultiRateMPIExt` (`MPIDecomp`, loaded with `using MPI`)
  model.jl    `MoistKA`: state array Q[I, J, k, v] (v = ρ, U, V, W, X, ρq_v, ρq_c, p), work arrays
  data.jl     frozen coefficients C(y) and the slow part S(y) (face values, momentum advection, shift technique)
  fast.jl     fast parts (:hthermo, :hmomentum, the vertical part, phase change) and the column solve
  mis.jl      MIS step with the component splitting :cs of the small steps (as `TimeStepping.frozen_mis_step!`
              with `fast = :cs`), halo exchanges of p before and of U, V after every momentum step
  convert.jl  conversion to and from the state vector of `FV3D.MoistModel3D` (tests, initial states)

The discretization is the one of `FV3D.MoistModel3D` (tests: equal results up to round-off). The state layout uses
local indices I = H + i (i = 1 … nxl the interior), k = 1 … nz for cells and k = 1 … nz+1 for W (k = 1, nz+1 walls).
"""
module FV3DKA

using KernelAbstractions
using LinearAlgebra, SparseArrays
const KA = KernelAbstractions
using ..Common: Gas, pressure, MoistGas, moist_state, moist_exner, _moist_p, fischer_burmeister, theta_phase_factor,
                pressure_phase_factor, exner_derivatives, theta_e, _Θof, KesslerParams, kessler_rates,
                rain_velocity, liquid_content
using ..FV2D: recon3, shiftvel, shift_back
using ..TimeStepping: MISTableau, fast_substeps, nstages
import ..FV3D

export KAGrid, SerialDecomp, MoistKA, ka_state, global_state, ka_rhs!, ka_mis_step!, ka_run!, ka_step_end!,
       ka_total, ka_project, ka_fields, ka_precipitation_rate, exchange!, mpi_decomp, HALO, VQR, CutGrid, CutKA,
       ka_frozen_data!, ka_vertical!

include("grid.jl")
include("decomp.jl")
include("model.jl")
include("data.jl")
include("fast.jl")
include("mis.jl")
include("convert.jl")
include("cut.jl")

end
