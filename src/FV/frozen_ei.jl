# Verlet EI (Imex.pdf, "Split-explicit with implicit treatment of part of the fast waves",
# Section 1, first version, EI): hooks of the frozen models for `fast = :ei` of
# `TimeStepping.frozen_fast_solve!`. The vertical part G_v (:vertical of `frozen_fast!`: the W part
# of the thermodynamic fluxes, the vertical pressure gradient and gravity) is scaled row-wise,
# D = ct on the thermodynamic rows and cw on the W rows, and the column system of the stage
# (I - D J_v) Y = R is solved with the solvers of the models (the W system depends on ct cw only).

const FrozenFVModel = Union{FVModel,ExnerModel,MoistExnerModel}

_vgrid(m::FVModel) = m.grid
_vgrid(m::Union{ExnerModel,MoistExnerModel}) = m.base.grid

_vsolve!(y, m::FVModel, ct, cw, R) = _frozen_column_solve!(y, _frozen_column(m), m, ct, R, cw)
_vsolve!(y, m::ExnerModel, ct, cw, R) = exner_column_solve!(y, m, ct, R, cw)
_vsolve!(y, m::MoistExnerModel, ct, cw, R) = moist_column_solve!(y, m, ct, R, cw)

"""out = D G_v(z): the vertical part with the thermodynamic rows scaled by ct, the W rows by cw."""
function frozen_vertical_rhs!(out, m::FrozenFVModel, C, z, ct, cw)
    frozen_fast!(out, m, C, z, :vertical, out)             # F unused for :vertical
    _scale_tw!(out, _vgrid(m), ct, cw)
end

"""z ← z + Y, (I - D J_v) Y = D G_v(z), D = ct (thermodynamic rows), cw (W rows); J_v the column
Jacobian of the stage (exact for the linear vertical parts, approximate for :gradp)."""
function frozen_vertical_implicit!(z, m::FrozenFVModel, C, ct, cw, k, y)
    frozen_vertical_rhs!(k, m, C, z, ct, cw)
    _vsolve!(y, m, ct, cw, k)
    @. z += y
    return z
end

"""y = (I - D J_v)⁻¹ R (:wrf of `frozen_fast_solve!`), D = ct (thermodynamic rows), cw (W rows)."""
frozen_vertical_solve!(y, m::FrozenFVModel, C, ct, cw, R) = _vsolve!(y, m, ct, cw, R)

"""
    frozen_hthermo!(out, m, C, z, F, w)

:hthermo of the implicit step of :ei / :wrf. With FV_METRIC_IMPLICIT the horizontal mass flux is
evaluated with the predictor U + w P_x `mvert`(π) (the metric increment of the implicit step at the
old pressure, docs/verlet_ei Section 5, point (b)): the remaining lag w P_x mvert(Δπ) is O(h Δπ),
and the step keeps its order. EI: w = βh (the weight of the implicit step). WRF: w = h, the predictor
Û = U + h metric(pⁿ) of docs/verlet_ei; implicit and explicit metric term then differ by βh metric(Δp)
per step, which sums to βh metric(p_end - p_start) over a stage, a first-order stage-end defect
(w = βh would make WRF conjugate to EI, but the change of variables then carries the large
hydrostatic metric force (1-β)h z_x ∂η p, and the stage-boundary defect is ~100 times larger).
Otherwise the plain :hthermo part.
"""
function frozen_hthermo!(out, m::Union{ExnerModel,MoistExnerModel}, C, z, F, w)
    g = m.base.grid
    (_metric_implicit(g) && w != 0) || return frozen_fast!(out, m, C, z, :hthermo, F)
    zp = m.column.zp
    zp .= z
    f(i, k) = z[iπ(m, i, k)]
    @inbounds for k in 1:g.nz, i in 1:g.nx
        zp[iU(g, i, k)] += w * m.column.Px[i, k] * mvert(f, g, i, k)
    end
    return frozen_fast!(out, m, C, zp, :hthermo, F)
end
