# Vertically implicit part: tridiagonal system for W per column (WRF), with the local terrain
# layer thickness J Δη. On terrain the σ-correction -z_x U is in the horizontal operator, so the
# column system does not read U.
#
# (I - c A_v) Y = Ys with Y = (ρ, Θ, W) of one column. ρ_k and Θ_k are explicit in the W of
# the two faces of cell k; inserting them into the W equation of face k (between cells k-1, k)
#   W_k + (c/Δz)(a_k Θ_k - a_{k-1} Θ_{k-1}) + (c g/2)(ρ_{k-1} + ρ_k) = Ws_k
# gives a tridiagonal system for W_2 .. W_nz (W_1 = W_{nz+1} = 0):
#   W_{k-1} [-(c/Δz)² a_{k-1} θ_{k-1} + c² g/(2Δz)]
#   + W_k [1 + (c/Δz)² θ_k (a_k + a_{k-1})]
#   + W_{k+1} [-(c/Δz)² a_k θ_{k+1} - c² g/(2Δz)]
#   = Ws_k - (c/Δz)(a_k Θs_k - a_{k-1} Θs_{k-1}) - (c g/2)(ρs_{k-1} + ρs_k)
# (θ at z-faces). The horizontal momentum rows are the identity.

struct FVVerticalFactor
    c::Float64               # coefficient of I - c A_v
    sub::Matrix{Float64}     # (nz+1, nx): coefficient of W_{k-1} in row k
    cp::Matrix{Float64}      # Thomas: c'
    iden::Matrix{Float64}    # Thomas: 1/denominator
end

function factor_vertical!(F::FVVerticalFactor, s::FVFast, c)
    g = s.grid
    nz = g.nz
    @inbounds for i in 1:g.nx
        dz = g.terrain ? g.Jc[i] * g.dz : g.dz
        r = (c / dz)^2
        gg = c^2 * s.gas.g / (2 * dz)
        for k in 2:nz
            sub = k > 2 ? -r * s.a[i, k-1] * s.θz[i, k-1] + gg : 0.0
            diag = 1 + r * s.θz[i, k] * (s.a[i, k] + s.a[i, k-1])
            sup = k < nz ? -r * s.a[i, k] * s.θz[i, k+1] - gg : 0.0
            den = k == 2 ? diag : diag - sub * F.cp[k-1, i]
            F.sub[k, i] = sub
            F.iden[k, i] = 1 / den
            F.cp[k, i] = sup / den
        end
    end
    return F
end

"""Factors of I - D A_v, D = c on the thermodynamic rows and cw on the W rows: the W system depends
on c cw only (cached by c cw, stored with c = √(c cw))."""
function vertical_factor(s::FVFast, c, cw = c)
    get!(s.lucache, Float64(c * cw)) do
        g = s.grid
        ce = sqrt(c * cw)
        F = FVVerticalFactor(ce, zeros(g.nz + 1, g.nx), zeros(g.nz + 1, g.nx), zeros(g.nz + 1, g.nx))
        factor_vertical!(F, s, ce)
    end
end

"""Y = (I - D A_v)⁻¹ Ys (Y may alias Ys), D = c on the thermodynamic rows and cw on the W rows
(default c = cw = F.c; F must be the factor of c cw)."""
function vertical_solve!(Y, s::FVFast, F::FVVerticalFactor, Ys, c = F.c, cw = c)
    g = s.grid
    nz = g.nz
    grav = s.gas.g
    W = s.colbuf
    @inbounds for i in 1:g.nx
        dz = g.terrain ? g.Jc[i] * g.dz : g.dz
        # forward sweep (dp stored in W)
        for k in 2:nz
            rhs = Ys[iW(g, i, k)] - (cw / dz) * (s.a[i, k] * Ys[iΘ(g, i, k)] - s.a[i, k-1] * Ys[iΘ(g, i, k - 1)]) -
                  (cw * grav / 2) * (Ys[iρ(g, i, k - 1)] + Ys[iρ(g, i, k)])
            W[k] = k == 2 ? rhs * F.iden[k, i] : (rhs - F.sub[k, i] * W[k-1]) * F.iden[k, i]
        end
        for k in nz-1:-1:2
            W[k] -= F.cp[k, i] * W[k+1]
        end
        W[1] = 0.0
        W[nz+1] = 0.0
        # back substitution for ρ and Θ (reads Ys before Y is written)
        for k in 1:nz
            θb = k > 1 ? s.θz[i, k] : 0.0
            θt = k < nz ? s.θz[i, k+1] : 0.0
            ρ = Ys[iρ(g, i, k)] - (c / dz) * (W[k+1] - W[k])
            Θ = Ys[iΘ(g, i, k)] - (c / dz) * (θt * W[k+1] - θb * W[k])
            Y[iρ(g, i, k)] = ρ
            Y[iΘ(g, i, k)] = Θ
            Y[iU(g, i, k)] = Ys[iU(g, i, k)]
        end
        for k in 2:nz
            Y[iW(g, i, k)] = W[k]
        end
    end
    if s.vadv
        # U ← U_s - c ∂_η(W ū)/J_x with the solved W (U is not in the column system)
        @inbounds for i in 1:g.nx, k in 1:nz
            dzx = g.terrain ? g.Jx[i] * g.dz : g.dz
            Y[iU(g, i, k)] -= c * fv_uσ_flux_div(s, Y, i, k) / dzx
        end
    end
    return Y
end

"""FV HEVI fast step: horizontal Verlet half steps around an implicit vertical CN step.

The C-grid horizontal acoustic operator is centred and has imaginary eigenvalues, so the
generic HEVI explicit RK2 horizontal update is not stable. The horizontal subflow is advanced
with the staggered Verlet update; the vertical acoustic-gravity operator is advanced with
Crank–Nicolson using the terrain-aware column solve.
"""
function hevi_loop!(q, s::FVFast, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    F = vertical_factor(s, dt / 2)
    halfdt = dt / 2
    for _ in 1:nsteps
        # Horizontal explicit Verlet half step. `f` includes the MIS slow forcing and the
        # frozen fast forcing; the latter is applied once through the staggered row split.
        fast_rhs_hthermo!(k1, s, q, f)
        @. q += (halfdt / 2) * k1
        fast_rhs_hmomentum!(k2, s, q, f)
        @. q += halfdt * k2
        fast_rhs_hthermo!(k1, s, q, f)
        @. q += (halfdt / 2) * k1

        # Vertical Crank–Nicolson: (I - dt/2 A_v) q⁺ = (I + dt/2 A_v) q⁻.
        vertical_apply!(tmp, s, q)
        @. rhs = q + (dt / 2) * tmp
        vertical_solve!(Y, s, F, rhs)
        q .= Y

        fast_rhs_hthermo!(k1, s, q, f)
        @. q += (halfdt / 2) * k1
        fast_rhs_hmomentum!(k2, s, q, f)
        @. q += halfdt * k2
        fast_rhs_hthermo!(k1, s, q, f)
        @. q += (halfdt / 2) * k1
    end
    return q
end

# ------------------------------------------------------------------ Verlet EI (Imex.pdf, version 1 EI)
"""Vertical part A_v q with the thermodynamic rows scaled by ct and the W rows by cw."""
function ei_vertical!(out, s::FVFast, q, ct, cw)
    s.vadv && error("fast = :ei: vertical advection of U in the fast part (FV_VADV_FAST) is not supported")
    vertical_apply!(out, s, q)
    _scale_tw!(out, s.grid, ct, cw)
end

"""Y = (I - D A_v)⁻¹ R, D = ct on the thermodynamic rows and cw on the W rows."""
ei_vertical_solve!(Y, s::FVFast, ct, cw, R) = vertical_solve!(Y, s, vertical_factor(s, ct, cw), R, ct, cw)

"""out *= ct on the thermodynamic rows, cw on the momentum rows (W; U for the implicit metric term,
FV_METRIC_IMPLICIT; otherwise zero in a vertical part); ct = 0 allowed."""
function _scale_tw!(out, g, ct, cw)
    if ct == 0
        w = [cw * out[iW(g, i, k)] for i in 1:g.nx, k in 2:g.nz]
        u = [cw * out[iU(g, i, k)] for i in 1:g.nx, k in 1:g.nz]
        fill!(out, 0.0)
        @inbounds for k in 2:g.nz, i in 1:g.nx
            out[iW(g, i, k)] = w[i, k-1]
        end
        @inbounds for k in 1:g.nz, i in 1:g.nx
            out[iU(g, i, k)] = u[i, k]
        end
        return out
    end
    @. out *= ct
    f = cw / ct
    @inbounds for k in 2:g.nz, i in 1:g.nx
        out[iW(g, i, k)] *= f
    end
    @inbounds for k in 1:g.nz, i in 1:g.nx
        out[iU(g, i, k)] *= f
    end
    return out
end
