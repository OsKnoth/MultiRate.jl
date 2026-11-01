# Model, reference state, sponge, initialization, output fields

"""Work arrays of the nonlinear operator (velocities and pressure deviation)."""
struct FVCache
    u::Matrix{Float64}       # at x-faces (nx, nz)
    w::Matrix{Float64}       # at mapped vertical faces (nx, nz+1), wall values from no-normal flow
    θ::Matrix{Float64}       # at centres
    pp::Matrix{Float64}      # p - p̄ at centres
end
FVCache(g::FVGrid) = FVCache(zeros(g.nx, g.nz), zeros(g.nx, g.nz + 1), zeros(g.nx, g.nz), zeros(g.nx, g.nz))

struct FVModel
    grid::FVGrid
    gas::Gas
    nq::Int
    ρref::Matrix{Float64}     # at cell centres (nx, nz)
    Θref::Matrix{Float64}
    pref::Matrix{Float64}
    damp::Vector{Float64}     # Rayleigh damping rate per unknown (sponge)
    qtarget::Vector{Float64}
    cache::FVCache
    fastref::Base.RefValue{Any}
end

"""
    FVModel(grid, gas, ρref, Θref)

Hydrostatic reference given by functions of z (evaluated at the cell centres).
"""
function FVModel(grid::FVGrid, gas::Gas, ρref, Θref)
    R = ρref.(grid.Zc)
    T = Θref.(grid.Zc)
    nq = nunknowns(grid)
    return FVModel(grid, gas, nq, R, T, pressure.(Ref(gas), T), zeros(nq), zeros(nq), FVCache(grid), Ref{Any}(nothing))
end

"""
    with_sponge(model, qtarget; zD, αtop=0.05, xwidth=0.0, αside=αtop)

Rayleigh damping -α (q - qtarget) above zD (sin² profile) and optionally in lateral layers.
"""
function with_sponge(m::FVModel, qtarget::AbstractVector; zD::Real, αtop::Real = 0.05,
                     xwidth::Real = 0.0, αside::Real = αtop)
    g = m.grid
    rate(x, z) = begin
        a = z > zD ? αtop * sin(π / 2 * (z - zD) / (g.H - zD))^2 : 0.0
        if xwidth > 0
            d = max(g.x0 + xwidth - x, x - (g.x0 + g.Lx - xwidth))
            d > 0 && (a = max(a, αside * sin(π / 2 * min(d, xwidth) / xwidth)^2))
        end
        a
    end
    damp = zeros(m.nq)
    for k in 1:g.nz, i in 1:g.nx
        damp[iρ(g, i, k)] = rate(g.xc[i], g.Zc[i, k])
        damp[iΘ(g, i, k)] = rate(g.xc[i], g.Zc[i, k])
        damp[iU(g, i, k)] = rate(g.xf[i], g.Zx[i, k])
        k > 1 && (damp[iW(g, i, k)] = rate(g.xc[i], g.Zf[i, k]))
    end
    return FVModel(g, m.gas, m.nq, m.ρref, m.Θref, m.pref, damp, Vector{Float64}(qtarget), m.cache, m.fastref)
end

"""
    project2d(model, fun)

Initialization from `fun(x, z) = (ρ, m_x, m_z, Θ)`: ρ, Θ at the centres, m_x at the x-faces,
m_z at the interior z-faces.
"""
function project2d(m::FVModel, fun)
    g = m.grid
    q = zeros(m.nq)
    for k in 1:g.nz, i in 1:g.nx
        v = fun(g.xc[i], g.Zc[i, k])
        q[iρ(g, i, k)] = v[1]
        q[iΘ(g, i, k)] = v[4]
        q[iU(g, i, k)] = fun(g.xf[i], g.Zx[i, k])[2]
        k > 1 && (q[iW(g, i, k)] = fun(g.xc[i], g.Zf[i, k])[3])
    end
    return q
end

"""Discrete integral of variable r (1 ρ, 2 ρu, 3 ρw, 4 Θ)."""
function total2d(m::FVModel, q, r)
    g = m.grid
    if r == 1
        return sum(g.Jc[i] * q[iρ(g, i, k)] for i in 1:g.nx, k in 1:g.nz) * g.dx * g.dz
    elseif r == 2
        return sum(g.Jx[i] * q[iU(g, i, k)] for i in 1:g.nx, k in 1:g.nz) * g.dx * g.dz
    elseif r == 3
        return sum(g.Jc[i] * q[iW(g, i, k)] for i in 1:g.nx, k in 2:g.nz) * g.dx * g.dz
    else
        return sum(g.Jc[i] * q[iΘ(g, i, k)] for i in 1:g.nx, k in 1:g.nz) * g.dx * g.dz
    end
end

"""Output fields at the cell centres as name => (nx, nz) array (velocities averaged from the faces)."""
function output_fields(m::FVModel, q)
    g = m.grid
    ρ = [q[iρ(g, i, k)] for i in 1:g.nx, k in 1:g.nz]
    Θ = [q[iΘ(g, i, k)] for i in 1:g.nx, k in 1:g.nz]
    U = [0.5 * (q[iU(g, i, k)] + q[iU(g, i + 1, k)]) for i in 1:g.nx, k in 1:g.nz]
    W = [begin
             Wb = k == 1 && g.terrain ? g.zx_f[i,1] * 0.5 * (q[iU(g,i,1)] + q[iU(g,i+1,1)]) : Wface(g,q,i,k)
             Wt = Wface(g,q,i,k+1)
             0.5 * (Wb + Wt)
         end for i in 1:g.nx, k in 1:g.nz]
    θref = m.Θref ./ m.ρref
    return ["rho" => ρ, "u" => U ./ ρ, "w" => W ./ ρ, "theta" => Θ ./ ρ,
            "rho_prime" => ρ .- m.ρref, "theta_prime" => Θ ./ ρ .- θref,
            "p_prime" => pressure.(Ref(m.gas), Θ) .- m.pref]
end

"""(x, values, zlevel) of an output field at the cell row nearest to height z."""
function field_section(m::FVModel, q, name, z)
    g = m.grid
    F = Dict(output_fields(m, q))[name]
    ks = [argmin(abs.(g.Zc[i, :] .- z)) for i in 1:g.nx]
    values = [F[i, ks[i]] for i in 1:g.nx]
    zlevel = sum(g.Zc[i, ks[i]] for i in 1:g.nx) / g.nx
    return g.xc, values, zlevel
end

sound_speed_max(m::FVModel, q) =
    maximum(sqrt(m.gas.γ * pressure(m.gas, q[iΘ(m.grid, i, k)]) / q[iρ(m.grid, i, k)])
            for i in 1:m.grid.nx, k in 1:m.grid.nz)
