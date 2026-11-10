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
end

"""
    Model3D(grid, gas, ρref, Θref)

Reference functions ρref(z), Θref(z) are used for output (primed fields) only; the equations use
the full p and ρ, no background state is subtracted.
"""
function Model3D(g::Grid3D, gas::Gas, ρref, Θref)
    Z = [g.zc[k] for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz]
    nq = nunknowns(g)
    sz = (g.nx, g.ny, g.nz)
    cache = Cache3D(zeros(sz), zeros(sz), zeros(g.nx, g.ny, g.nz + 1), zeros(sz), zeros(sz),
                    zeros(_ncoef(g)), zeros(nq), Ref{Any}(nothing))
    return Model3D(g, gas, nq, ρref.(Z), Θref.(Z), zeros(nq), zeros(nq), cache)
end

"""Initialization from `fun(x, y, z) = (ρ, ρu, ρv, ρw, Θ)`: ρ, Θ at the centres, the momenta at
their faces (W at the interior z-faces). Solid cells keep the values (they are frozen)."""
function project3d(m::Model3D, fun)
    g = m.grid
    q = zeros(m.nq)
    for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        x, y, z = g.xc[i], g.yc[j], g.zc[k]
        c = fun(x, y, z)
        q[iρ(g, i, j, k)] = c[1]
        q[iΘ(g, i, j, k)] = c[5]
        q[iU(g, i, j, k)] = fun(g.xf[i], y, z)[2] * (g.Ax[i, j, k] > 0)
        q[iV(g, i, j, k)] = fun(x, g.yf[j], z)[3] * (g.Ay[i, j, k] > 0)
        k > 1 && (q[iW(g, i, j, k)] = fun(x, y, g.zf[k])[4] * (g.Az[i, j, k] > 0))
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

"""Discrete integral Σ Veff ΔV q_c of a cell variable (:ρ or :Θ) over the fluid cells."""
function total3d(m::Model3D, q, var::Symbol)
    g = m.grid
    idx = var === :ρ ? iρ : iΘ
    dV = g.dx * g.dy * g.dz
    return sum(q[idx(g, i, j, k)] * dV / g.invV[i, j, k] for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx if g.invV[i, j, k] > 0)
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
