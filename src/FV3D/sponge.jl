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
    return Model3D(g, m.gas, m.nq, m.ρref, m.Θref, damp, Vector{Float64}(qtarget), m.cache, m.implicit, m.nudge)
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

# ------------------------------------------------------------------ nudging zone
# Relaxation of the velocities and of θ towards a prescribed inflow profile in a strip x0 ≤ x ≤ x1 (periodic domain:
# the disturbances leaving downstream are damped there, the target profile re-enters upstream). Slow tendency (in
# S(y), explicit in the large step), in momentum form, ρ unchanged (mass conserved):
#     U_t += -α (U - ρ̄ u_t(z)),  V_t += -α (V - ρ̄ v_t(z)),  W_t += -α W,  Θ_t += -α (Θ - ρ θ_t(z)),
# α(x) = αmax sin²(π (x - x0)/(x1 - x0)) in the strip (zero at its edges), ρ̄ the face means of ρ.

"""Rates at the cell columns, x-face and y-face columns (nx, ny); target profiles u, v, θ at the centres (nz)."""
struct Nudging
    αc::Matrix{Float64}; αx::Matrix{Float64}; αy::Matrix{Float64}
    u::Vector{Float64}; v::Vector{Float64}; θ::Vector{Float64}
end
"""α of the strip [x0, x1] at x (periodic in [0, Lx])."""
function nudging_rate(x, x0, x1, αmax, Lx)
    ξ = mod(x - x0, Lx)
    return ξ <= x1 - x0 ? αmax * sin(π * ξ / (x1 - x0))^2 : 0.0
end

"""
    with_nudging(model; x0, x1, αmax = 0.5, u, v = 0.0, θ)

Nudging zone x0 ≤ x ≤ x1 (rates αmax sin² across the strip, in 1/s) towards the profiles u(z), v(z), θ(z) (functions
or numbers); w towards 0. See above.
"""
function with_nudging(m::Model3D; x0::Real, x1::Real, αmax::Real = 0.5, u, v = 0.0, θ)
    g = m.grid
    f(p) = p isa Number ? (z -> Float64(p)) : p
    a(x) = nudging_rate(x, x0, x1, αmax, g.Lx)
    nd = Nudging([a(g.xc[i]) for i in 1:g.nx, j in 1:g.ny], [a(g.xf[i]) for i in 1:g.nx, j in 1:g.ny],
                 [a(g.xc[i]) for i in 1:g.nx, j in 1:g.ny], f(u).(g.zc), f(v).(g.zc), f(θ).(g.zc))
    return Model3D(g, m.gas, m.nq, m.ρref, m.Θref, m.damp, m.qtarget, m.cache, m.implicit, nd)
end

"""S += the nudging tendencies (nothing without a nudging zone)."""
nudge!(S, m::Model3D, q) = m.nudge === nothing ? S : _nudge!(S, m, m.nudge, q)
function _nudge!(S, m::Model3D, nd::Nudging, q)
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        ρc = q[iρ(g, i, j, k)]
        a = nd.αc[i, j]
        if a > 0 && g.Vf[i, j, k] > 0
            S[iΘ(g, i, j, k)] -= a * (q[iΘ(g, i, j, k)] - ρc * nd.θ[k])
            k >= 2 && g.Az[i, j, k] > 0 && (S[iW(g, i, j, k)] -= a * q[iW(g, i, j, k)])
        end
        a = nd.αx[i, j]
        (a > 0 && g.Ax[i, j, k] > 0) &&
            (S[iU(g, i, j, k)] -= a * (q[iU(g, i, j, k)] - 0.5 * (q[iρ(g, i - 1, j, k)] + ρc) * nd.u[k]))
        a = nd.αy[i, j]
        (a > 0 && g.Ay[i, j, k] > 0) &&
            (S[iV(g, i, j, k)] -= a * (q[iV(g, i, j, k)] - 0.5 * (q[iρ(g, i, j - 1, k)] + ρc) * nd.v[k]))
    end
    return S
end
