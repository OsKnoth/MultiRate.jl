# Model, reference state, sponge, initialization, output fields

"""Work arrays of the nonlinear operator (velocities and pressure deviation)."""
struct FVCache
    u::Matrix{Float64}       # at x-faces (nx, nz)
    w::Matrix{Float64}       # at mapped vertical faces (nx, nz+1), wall values from no-normal flow
    θ::Matrix{Float64}       # at centres
    pp::Matrix{Float64}      # p - p̄ at centres
    mx::Matrix{Float64}
    mz::Matrix{Float64}
    fx::Matrix{Float64}
    fz::Matrix{Float64}
    fxu::Matrix{Float64}
    fxw::Matrix{Float64}
    fzu::Matrix{Float64}
    fzw::Matrix{Float64}
    mdx::Matrix{Float64}
    mdz::Matrix{Float64}
    cc::Vector{Matrix{Float64}}   # cut-cell third-order advection work arrays, (nx, nz) each
end
FVCache(g) = FVCache(zeros(g.nx,g.nz),zeros(g.nx,g.nz+1),zeros(g.nx,g.nz),zeros(g.nx,g.nz),
                     zeros(g.nx,g.nz),zeros(g.nx,g.nz),zeros(g.nx+1,g.nz),zeros(g.nx,g.nz+1),
                     zeros(g.nx+1,g.nz),zeros(g.nx+1,g.nz),zeros(g.nx,g.nz+1),zeros(g.nx,g.nz+1),
                     zeros(g.nx,g.nz),zeros(g.nx,g.nz),
                     g isa CutCellFVGrid ? [zeros(g.nx,g.nz) for _ in 1:10] : Matrix{Float64}[])

struct FVModel{G}
    grid::G
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

function FVModel(grid::CutCellFVGrid, gas::Gas, ρref, Θref)
    Zref = [cutcell_sample_z(grid,i,k) for i in 1:grid.nx, k in 1:grid.nz]
    R = ρref.(Zref)
    T = Θref.(Zref)
    nq = nunknowns(grid)
    cache = FVCache(grid)
    # background θ̄ for the Θ flux: cells (as the reference state), open x-face midpoints, z-faces
    θbar(z) = Θref(z) / ρref(z)
    g = grid
    push!(cache.cc, T ./ R,
          [θbar((max(g.zf[k], min(g.zs(g.xf[f]), g.zf[k+1])) + g.zf[k+1]) / 2) for f in 1:g.nx+1, k in 1:g.nz],
          [θbar(g.zf[k]) for i in 1:g.nx, k in 1:g.nz+1])
    return FVModel(grid, gas, nq, R, T, pressure.(Ref(gas), T), zeros(nq), zeros(nq), cache, Ref{Any}(nothing))
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

function project2d(m::FVModel{<:CutCellFVGrid}, fun)
    g = m.grid
    q = zeros(m.nq)
    @inbounds for k in 1:g.nz, i in 1:g.nx
        # Use the cell centre when it lies in fluid, otherwise sample halfway through the
        # fluid portion of the cut cell. Fully solid cells retain the reference state so
        # coefficient evaluation remains finite; their open volume and faces are zero.
        zsample = cutcell_sample_z(g,i,k)
        vals = fun(g.xc[i], zsample)
        if g.Vol[i,k] > 0
            q[iρ(g, i, k)] = vals[1]
            q[iΘ(g, i, k)] = vals[4]
        else
            q[iρ(g, i, k)] = m.ρref[i,k]
            q[iΘ(g, i, k)] = m.Θref[i,k]
        end
    end
    @inbounds for k in 1:g.nz, i in 1:g.nx
        if g.FU[i,k] > 0
            z = (max(g.zf[k], g.zs(g.xf[i])) + g.zf[k+1]) / 2
            q[iU(g, i, k)] = fun(g.xf[i], z)[2]
        end
    end
    @inbounds for k in 2:g.nz, i in 1:g.nx
        if g.FW[i,k] > 0
            q[iW(g, i, k)] = fun(g.xc[i], g.zf[k])[3]
        end
    end
    return q
end

"""Discrete integral of variable r (1 ρ, 2 ρu, 3 ρw, 4 Θ)."""
function total2d(m::FVModel, q, r)
    g = m.grid
    if g isa CutCellFVGrid
        if r == 1
            return sum(g.Vol[i,k] * q[iρ(g,i,k)] for i in 1:g.nx, k in 1:g.nz)
        elseif r == 4
            return sum(g.Vol[i,k] * q[iΘ(g,i,k)] for i in 1:g.nx, k in 1:g.nz)
        elseif r == 2
            return sum(g.Vol[i,k] * 0.5 * (q[iU(g,i,k)] + q[iU(g,i+1,k)]) for i in 1:g.nx, k in 1:g.nz)
        else
            return sum(g.Vol[i,k] * 0.5 * (g.FW[i,k] * Wface(g,q,i,k) / max(g.FW[i,k], eps()) +
                                           g.FW[i,k+1] * Wface(g,q,i,k+1) / max(g.FW[i,k+1], eps()))
                       for i in 1:g.nx, k in 1:g.nz)
        end
    end
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

function output_fields(m::FVModel{<:CutCellFVGrid}, q)
    g = m.grid
    ρ = [q[iρ(g,i,k)] for i in 1:g.nx, k in 1:g.nz]
    Θ = [q[iΘ(g,i,k)] for i in 1:g.nx, k in 1:g.nz]
    U = [0.5 * (q[iU(g,i,k)] + q[iU(g,i+1,k)]) for i in 1:g.nx, k in 1:g.nz]
    W = [0.5 * (Wface(g,q,i,k) + Wface(g,q,i,k+1)) for i in 1:g.nx, k in 1:g.nz]
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
