# Rayleigh sponge: damping of the vertical momentum m_z only, (m_z)_t += -α (m_z - m_z,target).
# Kept in one place: `with_sponge` sets the rates α (m_z entries of m.damp), `sponge!` adds the term;
# rhs_nonlinear! and the frozen fast part call it.

"""
    with_sponge(model, qtarget; zD, αtop=0.05, xwidth=0.0, αside=αtop)

Rayleigh damping -α(x,z) (m_z - m_z,target) of the vertical momentum only (ρ, m_x, Θ are not
damped; only the m_z entries of qtarget are used, normally 0), added to N:
top layer α = αtop sin²(π/2 (z - zD)/(H - zD)) for z > zD (Klemp & Lilly 1978 type), and for
xwidth > 0 lateral layers of that width at both ends of the periodic channel,
α = αside sin²(π/2 d/xwidth), d = distance into the layer. Stable for α Δt ≲ 1.
"""
function with_sponge(m::Euler2DModel, qtarget::AbstractVector; zD::Real, αtop::Real = 0.05,
                     xwidth::Real = 0.0, αside::Real = αtop)
    M = m.mesh
    α = zeros(M.np, M.np, M.Ne)
    x0, x1 = M.x0, M.x0 + M.Lx
    for k in eachindex(α)
        z, x = M.Z[k], M.X[k]
        a = z > zD ? αtop * sin(π / 2 * (z - zD) / (M.H - zD))^2 : 0.0
        if xwidth > 0
            d = max(x0 + xwidth - x, x - (x1 - xwidth))
            d > 0 && (a = max(a, αside * sin(π / 2 * min(d, xwidth) / xwidth)^2))
        end
        α[k] = a
    end
    D4 = zeros(M.np, M.np, NV, M.Ne)
    D4[:, :, 3, :] .= α                          # m_z only
    damp = vec(D4)
    return Euler2DModel(M, m.gas, m.nq, m.ρref, m.Θref, m.pref, damp, Vector{Float64}(qtarget), m.vflux, m.gflux, m.clmars,
                        m.cache, m.fastref)
end

"""Add the sponge -α (m_z - m_z,target) to the m_z rows of out (no other rows)."""
function sponge!(out, m::Euler2DModel, q)
    M = m.mesh
    np = M.np
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
        k = node_iq(np, e, 3, i, j)
        out[k] -= m.damp[k] * (q[k] - m.qtarget[k])
    end
    return out
end
