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
