# Rayleigh sponge: damping of the vertical momentum W only, W_t += -α (W - W_target).
# `with_sponge` sets the rates (W entries of m.damp), `sponge!` adds the term.

"""
    with_sponge(model, qtarget; zD, αtop = 0.05)

Rayleigh damping -α (W - W_target) of the vertical momentum only, α = αtop sin²(π/2 (z - zD)/(H - zD))
above zD; only the W entries of qtarget are used (normally 0).
"""
function with_sponge(m::Model3D, qtarget::AbstractVector; zD::Real, αtop::Real = 0.05)
    g = m.grid
    damp = zeros(m.nq)
    for k in 2:g.nz, j in 1:g.ny, i in 1:g.nx
        z = g.zf[k]
        damp[iW(g, i, j, k)] = z > zD ? αtop * sin(π / 2 * (z - zD) / (g.H - zD))^2 : 0.0
    end
    return Model3D(g, m.gas, m.nq, m.ρref, m.Θref, damp, Vector{Float64}(qtarget), m.cache, m.implicit)
end

"""Add the sponge -α (W - W_target) to the W rows of out (no other rows)."""
function sponge!(out, m::Model3D, q)
    g = m.grid
    @inbounds for k in 2:g.nz, j in 1:g.ny, i in 1:g.nx
        n = iW(g, i, j, k)
        out[n] -= m.damp[n] * (q[n] - m.qtarget[n])
    end
    return out
end
