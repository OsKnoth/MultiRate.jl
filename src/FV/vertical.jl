# Vertically implicit part: tridiagonal system for W per column (WRF), with the local terrain
# layer thickness J Δη and the mapped lower-wall flux W - z_x U = 0.
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

@inline function fv_terrain_cell_correction(s, Ys, i, k, c, dz)
    g = s.grid
    Vb = k == 1 ? 0.0 : g.zx_f[i,k] * fv_ux_at_sigma(g, Ys, i, k)
    Vt = k == g.nz ? 0.0 : g.zx_f[i,k+1] * fv_ux_at_sigma(g, Ys, i, k+1)
    θb = k == 1 ? s.θz[i,1] : s.θz[i,k]
    θt = k == g.nz ? s.θz[i,g.nz+1] : s.θz[i,k+1]
    return (c / dz) * (Vt - Vb), (c / dz) * (θt * Vt - θb * Vb)
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

function vertical_factor(s::FVFast, c)
    get!(s.lucache, Float64(c)) do
        g = s.grid
        F = FVVerticalFactor(Float64(c), zeros(g.nz + 1, g.nx), zeros(g.nz + 1, g.nx), zeros(g.nz + 1, g.nx))
        factor_vertical!(F, s, c)
    end
end

"""Y = (I - c A_v)⁻¹ Ys (Y may alias Ys)."""
function vertical_solve!(Y, s::FVFast, F::FVVerticalFactor, Ys)
    g = s.grid
    nz = g.nz
    grav = s.gas.g
    c = F.c
    W = s.colbuf
    @inbounds for i in 1:g.nx
        dz = g.terrain ? g.Jc[i] * g.dz : g.dz
        # forward sweep (dp stored in W)
        for k in 2:nz
            rhs = Ys[iW(g, i, k)] - (c / dz) * (s.a[i, k] * Ys[iΘ(g, i, k)] - s.a[i, k-1] * Ys[iΘ(g, i, k - 1)]) -
                  (c * grav / 2) * (Ys[iρ(g, i, k - 1)] + Ys[iρ(g, i, k)])
            if g.terrain
                ρL, ΘL = fv_terrain_cell_correction(s, Ys, i, k-1, c, dz)
                ρR, ΘR = fv_terrain_cell_correction(s, Ys, i, k, c, dz)
                rhs -= (c / dz) * (s.a[i,k] * ΘR - s.a[i,k-1] * ΘL) + (c * grav / 2) * (ρL + ρR)
            end
            W[k] = k == 2 ? rhs * F.iden[k, i] : (rhs - F.sub[k, i] * W[k-1]) * F.iden[k, i]
        end
        for k in nz-1:-1:2
            W[k] -= F.cp[k, i] * W[k+1]
        end
        W[1] = g.terrain ? g.zx_f[i,1] * fv_ux_at_sigma(g, Ys, i, 1) : 0.0
        W[nz+1] = 0.0
        # back substitution for ρ and Θ (reads Ys before Y is written)
        for k in 1:nz
            θb = k > 1 ? s.θz[i, k] : 0.0
            θt = k < nz ? s.θz[i, k+1] : 0.0
            if g.terrain
                Ub = fv_ux_at_sigma(g, Ys, i, k)
                Ut = fv_ux_at_sigma(g, Ys, i, k+1)
                Vb = W[k] - g.zx_f[i, k] * Ub
                Vt = W[k+1] - g.zx_f[i, k+1] * Ut
                ρ = Ys[iρ(g, i, k)] - (c / dz) * (Vt - Vb)
                Θ = Ys[iΘ(g, i, k)] - (c / dz) * (θt * Vt - θb * Vb)
            else
                ρ = Ys[iρ(g, i, k)] - (c / dz) * (W[k+1] - W[k])
                Θ = Ys[iΘ(g, i, k)] - (c / dz) * (θt * W[k+1] - θb * W[k])
            end
            Y[iρ(g, i, k)] = ρ
            Y[iΘ(g, i, k)] = Θ
            Y[iU(g, i, k)] = Ys[iU(g, i, k)]
        end
        for k in 2:nz
            Y[iW(g, i, k)] = W[k]
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
