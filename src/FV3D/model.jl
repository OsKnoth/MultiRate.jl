# Model: grid, gas, reference profile (output only), sponge rates, work arrays

struct Cache3D
    u::Array{Float64,3}           # U / mean ρ at the x-faces (nx, ny, nz)
    v::Array{Float64,3}           # V / mean ρ at the y-faces (nx, ny, nz)
    w::Array{Float64,3}           # W / mean ρ at the z-faces (nx, ny, nz+1), zero at the walls
    θ::Array{Float64,3}           # Θ / ρ at the centres
    pp::Array{Float64,3}          # p(Θ) at the centres (GradP)
    C::Vector{Float64}            # frozen coefficients and slow part of N (rhs_nonlinear! = F(q, q))
    S::Vector{Float64}
    column::Base.RefValue{Any}    # column solver of the HEVI step
    Fx::Array{Float64,3}          # mass fluxes A J U, A J V, A Ω (advection.jl)
    Fy::Array{Float64,3}
    Fz::Array{Float64,3}          # (nx, ny, nz+1)
    sh::NTuple{12,Array{Float64,3}}   # shifted velocities u, v, w (L, R) and their tendencies
    tmp::Vector{Float64}              # work vector (nq)
end

struct Model3D
    grid::Grid3D
    gas::Gas
    nq::Int
    ρref::Array{Float64,3}        # reference at the centres: output only (no background subtracted)
    Θref::Array{Float64,3}
    damp::Vector{Float64}         # sponge rates (W entries only)
    qtarget::Vector{Float64}
    cache::Cache3D
    implicit::Symbol              # HEVI implicit step: :column (vertical), :mask (+ masked horizontal), :amf (+ masked advection)
end

"""
    Model3D(grid, gas, ρref, Θref; implicit = grid.cut ? :amf : :column)

Reference functions ρref(z), Θref(z) are used for output (primed fields) only; the equations use
the full p and ρ, no background state is subtracted. `implicit` selects the implicit step of the
HEVI fast loop: `:column` (vertical part, tridiagonal per column) or `:mask` (vertical part plus
the horizontal fast terms through the faces of the masked cells, sparse LU per stage; the
explicit horizontal half steps then use the unmasked faces only) or `:amf` (as `:mask`, and the
momentum advection of the masked faces in the fast part, simple upwind, implicit by approximate
matrix factorization (I - c J_ac)(I - c J_adv), advection_jacobian.jl). With cut cells `:amf` is
the default: the divisions use the true fluid volumes, so the small cells must be implicit for
the fast step to be limited by the regular Δx.
"""
function Model3D(g::Grid3D, gas::Gas, ρref, Θref; implicit::Symbol = g.cut ? :amf : :column)
    implicit in (:column, :mask, :amf) || error("implicit: :column, :mask or :amf")
    Z = g.Zc                                  # physical heights of the centres
    nq = nunknowns(g)
    sz = (g.nx, g.ny, g.nz)
    cache = Cache3D(zeros(sz), zeros(sz), zeros(g.nx, g.ny, g.nz + 1), zeros(sz), zeros(sz),
                    zeros(_ncoef(g)), zeros(nq), Ref{Any}(nothing), zeros(sz), zeros(sz), zeros(g.nx, g.ny, g.nz + 1),
                    ntuple(_ -> zeros(sz), 12), zeros(nq))
    return Model3D(g, gas, nq, ρref.(Z), Θref.(Z), zeros(nq), zeros(nq), cache, implicit)
end

"""Initialization from `fun(x, y, z) = (ρ, ρu, ρv, ρw, Θ)` at the physical positions of the
unknowns: ρ, Θ at the centres, the momenta at their faces (W at the interior η-faces). Solid cells
keep the values (they are frozen)."""
function project3d(m::Model3D, fun)
    g = m.grid
    q = zeros(m.nq)
    for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        x, y = g.xc[i], g.yc[j]
        c = fun(x, y, g.Zc[i, j, k])
        q[iρ(g, i, j, k)] = c[1]
        q[iΘ(g, i, j, k)] = c[5]
        q[iU(g, i, j, k)] = fun(g.xf[i], y, g.H * (1 - g.Jx[i, j]) + g.zc[k] * g.Jx[i, j])[2] * (g.Ax[i, j, k] > 0)
        q[iV(g, i, j, k)] = fun(x, g.yf[j], g.H * (1 - g.Jy[i, j]) + g.zc[k] * g.Jy[i, j])[3] * (g.Ay[i, j, k] > 0)
        k > 1 && (q[iW(g, i, j, k)] = fun(x, y, g.Zf[i, j, k])[4] * (g.Az[i, j, k] > 0))
    end
    return q
end

"""
    balanced_state(m, Θfun, ρfun; U = 0, V = 0)

Discretely balanced initial state without a background state in the equations: Θ = Θfun(z) at the
centres, ρ per column from the W equation at rest, δη p / J + g (ρ_{k-1} + ρ_k)/2 = 0 (pressure form
of the operator), upwards from ρ_1 = ρfun(z_1), and uniform velocities U, V (momenta U ρ̄, V ρ̄ with
the face means of ρ, as in u = U/ρ̄). On Cartesian grids (flat, cut cells, obstacles) the rest state
is exactly steady; terrain following it is steady up to the horizontal pressure-gradient error.
"""
function balanced_state(m::Model3D, Θfun, ρfun; U = 0.0, V = 0.0)
    g = m.grid
    q = zeros(m.nq)
    for j in 1:g.ny, i in 1:g.nx
        q[iρ(g, i, j, 1)] = ρfun(g.Zc[i, j, 1])
        for k in 1:g.nz
            q[iΘ(g, i, j, k)] = Θfun(g.Zc[i, j, k])
        end
        for k in 2:g.nz
            Θl, Θc = q[iΘ(g, i, j, k - 1)], q[iΘ(g, i, j, k)]
            dp = FV3D_PRESSURE_FORM[] === :product ? dpdΘ_face(m.gas, Θl, Θc) * (Θc - Θl) : pressure(m.gas, Θc) - pressure(m.gas, Θl)
            q[iρ(g, i, j, k)] = -2dp / (m.gas.g * g.Jc[i, j] * g.dz) - q[iρ(g, i, j, k - 1)]
        end
    end
    for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        q[iU(g, i, j, k)] = U * 0.5 * (q[iρ(g, i - 1, j, k)] + q[iρ(g, i, j, k)]) * (g.Ax[i, j, k] > 0)
        q[iV(g, i, j, k)] = V * 0.5 * (q[iρ(g, i, j - 1, k)] + q[iρ(g, i, j, k)]) * (g.Ay[i, j, k] > 0)
    end
    return q
end

"""
    hydrostatic_column(m, Θfun; ρbottom)

Discretely balanced density at the cell levels for Θ_k = Θfun(z_k): the W equation at rest,
δz p + g (ρ_{k-1} + ρ_k)/2 = 0 with the pressure form of the operator, solved upwards from
ρ_1 = ρbottom. A horizontally homogeneous state (ρ_k, 0, 0, 0, Θ_k) is then exactly at rest, also
with cut cells and obstacles (no background state in the equations; the balance is in the
initial state only).
"""
function hydrostatic_column(m::Model3D, Θfun; ρbottom)
    g = m.grid
    ρ = zeros(g.nz); ρ[1] = ρbottom
    for k in 2:g.nz
        Θl, Θc = Θfun(g.zc[k-1]), Θfun(g.zc[k])
        dp = FV3D_PRESSURE_FORM[] === :product ? dpdΘ_face(m.gas, Θl, Θc) * (Θc - Θl) : pressure(m.gas, Θc) - pressure(m.gas, Θl)
        ρ[k] = -2dp / (m.gas.g * g.dz) - ρ[k-1]
    end
    return ρ
end

"""Cell level k of the height z."""
level(g::Grid3D, z) = clamp(floor(Int, z / g.dz) + 1, 1, g.nz)

"""Discrete integral Σ Vf J ΔV q_c of a cell variable (:ρ or :Θ) over the fluid cells."""
function total3d(m::Model3D, q, var::Symbol)
    g = m.grid
    idx = var === :ρ ? iρ : iΘ
    dV = g.dx * g.dy * g.dz
    return sum(q[idx(g, i, j, k)] * g.Jc[i, j] * dV / g.invV[i, j, k] for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx if g.invV[i, j, k] > 0)
end

sound_speed_max(m::Model3D, q) = (g = m.grid;
    maximum(sqrt(m.gas.γ * pressure(m.gas, q[iΘ(g, i, j, k)]) / q[iρ(g, i, j, k)])
            for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx if g.Vf[i, j, k] > 0))

"""θ′ = Θ/ρ - θref at the centres (nx, ny, nz; NaN in solid cells)."""
theta_prime(m::Model3D, q) = (g = m.grid;
    [g.Vf[i, j, k] > 0 ? q[iΘ(g, i, j, k)] / q[iρ(g, i, j, k)] - m.Θref[i, j, k] / m.ρref[i, j, k] : NaN
     for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz])

"""w = W / mean ρ at the interior z-faces (nx, ny, nz-1)."""
vertical_velocity(m::Model3D, q) = (g = m.grid;
    [q[iW(g, i, j, k)] / (0.5 * (q[iρ(g, i, j, k - 1)] + q[iρ(g, i, j, k)])) for i in 1:g.nx, j in 1:g.ny, k in 2:g.nz])
