# Rayleigh sponge: damping of the vertical momentum W only, W_t += -α (W - W_target).
# Kept in one place: `with_sponge` sets the rates α (W entries of m.damp), `sponge!` adds the term;
# the operators (rhs_nonlinear!, the cut-cell and terrain operators, the frozen fast part) call it.

"""
    with_sponge(model, qtarget; zD, αtop=0.05, xwidth=0.0, αside=αtop)

Rayleigh damping -α (W - W_target) of the vertical momentum only, above zD (sin² profile) and
optionally in lateral layers; ρ, Θ and U are not damped (only the W entries of qtarget are used,
normally W_target = 0).
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
    for k in 2:g.nz, i in 1:g.nx
        damp[iW(g, i, k)] = rate(g.xc[i], g.Zf[i, k])
    end
    return FVModel(g, m.gas, m.nq, m.ρref, m.Θref, m.pref, m.ρbal, m.pbal, damp, Vector{Float64}(qtarget), m.cache, m.fastref)
end

"""Add the sponge -α (W - W_target) to the W rows of out (no other rows)."""
function sponge!(out, m::FVModel, q)
    g = m.grid
    @inbounds for k in 2:g.nz, i in 1:g.nx
        j = iW(g, i, k)
        out[j] -= m.damp[j] * (q[j] - m.qtarget[j])
    end
    return out
end
