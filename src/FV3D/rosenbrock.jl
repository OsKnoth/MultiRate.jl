# Rosenbrock W-methods for Model3D (TimeStepping/rosenbrock.jl, the comparison to the split-explicit MIS): the
# approximate Jacobian W = J_ac + J_adv at qⁿ,
#   J_ac   the fast part over all faces (horizontal and vertical mass and Θ fluxes with the face θ, pressure gradients
#          of the pressure form, gravity, sponge): with :product the exact Jacobian of the fast part (linear in z),
#   J_adv  first-order upwind advection on all cells: momenta by the shift technique (FV3D_MASK_JFORM :partial a± δM or
#          :conservative also δF u - a u δρ), Θ with the ρ dependence of θ = Θ/ρ (both cells of each face, conservative),
# assembled with `implicit_matrix` and `adv_matrix` of the implicit mask on a copy of the grid in which every fluid cell
# is masked. The stage matrix I - γΔt W is solved exactly (`ros = :exact`, one sparse LU) or approximately factorised
# (`ros = :amf`, (I - γΔt J_ac)(I - γΔt J_adv), two LUs). Both are conservative in ρ and Θ (flux form rows).

"""Rosenbrock data of a Model3D: solve variant, factors of the last `ros_prepare!`, the all-masked model."""
mutable struct RosData
    variant::Symbol
    F1::Any; F2::Any
    mall::Any
end
const _ROS = IdDict{Any,RosData}()

"""
    with_rosenbrock(model; ros = :amf)

The model with the Rosenbrock data (`TimeStepping.run_ros(model, q0, Δt, n; tab)`); `ros = :exact` (one LU of
I - γΔt (J_ac + J_adv)) or `:amf` ((I - γΔt J_ac)(I - γΔt J_adv)).
"""
function with_rosenbrock(m::Model3D; ros::Symbol = :amf)
    ros in (:exact, :amf, :acoustic) || error("ros: :exact, :amf or :acoustic")
    _ROS[m] = RosData(ros, nothing, nothing, nothing)
    return m
end
_rosdata(m::Model3D) = get(_ROS, m) do; error("Model3D: call with_rosenbrock(model) first"); end

"""The grid with every fluid cell masked (all x-, y-faces masked)."""
function _allmasked(g::Grid3D)
    vals = map(fieldnames(Grid3D)) do f
        f === :mask ? g.Vf .> 0 : f === :Mx ? ones(size(g.Mx)) : f === :My ? ones(size(g.My)) : getfield(g, f)
    end
    return Grid3D(vals...)
end

"""J_ac and J_adv at q (coefficients C = frozen data of q)."""
function global_jacobians(m::Model3D, q)
    d = _rosdata(m)
    d.mall === nothing && (d.mall = Model3D(_allmasked(m.grid), m.gas, m.nq, m.ρref, m.Θref, m.damp, m.qtarget, m.cache,
                                            :amf, m.nudge))
    ma = d.mall::Model3D
    C, S = zeros(_ncoef(m.grid)), zeros(m.nq)
    frozen_data!(C, S, m, q)                      # face θ, dp/dΘ, upwind coefficients (all faces) of the model
    s = Column3D(ma.grid)
    for k in 1:ma.grid.nz+1, j in 1:ma.grid.ny, i in 1:ma.grid.nx
        s.θz[i, j, k] = C[cθz(ma.grid, i, j, k)]
    end
    column_coefficients!(pressure_form(), s, ma, C, q)
    Jac = implicit_matrix(ma, s, C, q) - spdiagm(0 => m.damp)
    Jadv = adv_matrix(ma, C)
    return Jac, Jadv
end

function TimeStepping.ros_prepare!(m::Model3D, q, γh)
    d = _rosdata(m)
    Jac, Jadv = global_jacobians(m, q)
    Id = sparse(1.0I, m.nq, m.nq)
    if d.variant === :exact
        d.F1 = lu(Id - γh * (Jac + Jadv)); d.F2 = nothing
    elseif d.variant === :amf
        d.F1 = lu(Id - γh * Jac); d.F2 = lu(Id - γh * Jadv)
    else                                           # :acoustic: W = J_ac only
        d.F1 = lu(Id - γh * Jac); d.F2 = nothing
    end
    return m
end

function TimeStepping.ros_solve!(k, m::Model3D, r)
    d = _rosdata(m)
    k .= d.F1 \ r
    d.F2 === nothing || (k .= d.F2 \ k)
    return k
end
