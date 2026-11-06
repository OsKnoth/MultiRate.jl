# Nonlinear operator N (flux form, C-grid)

"""θ at the z-face k of column i (3rd order where the stencil fits, 2nd order next to the walls)."""
@inline function face_z(F::AbstractMatrix, i, k, nz, v)
    if k >= 3 && k <= nz - 1
        return interp3(F[i, k-2], F[i, k-1], F[i, k], F[i, k+1], v)
    else
        return interp2(F[i, k-1], F[i, k])
    end
end

"""In-place N(q); allocation-free (uses the model's FVCache)."""
function rhs_nonlinear!(out, m::FVModel, q::AbstractVector)
    m.grid isa CutCellFVGrid && return rhs_cutcell!(out,m,q)
    m.grid.terrain && return rhs_terrain!(out, m, q)
    g = m.grid
    nx = g.nx
    nz = g.nz
    dx = g.dx
    dz = g.dz
    grav = m.gas.g
    c = m.cache
    u = c.u
    w = c.w
    θ = c.θ
    pp = c.pp
    @inbounds for k in 1:nz, i in 1:nx
        ρ = q[iρ(g, i, k)]
        θ[i, k] = q[iΘ(g, i, k)] / ρ
        pp[i, k] = pressure(m.gas, q[iΘ(g, i, k)]) - m.pref[i, k]
        u[i, k] = q[iU(g, i, k)] / (0.5 * (q[iρ(g, i - 1, k)] + ρ))
    end
    @inbounds for i in 1:nx
        w[i, 1] = 0.0
        w[i, nz+1] = 0.0
        for k in 2:nz
            w[i, k] = q[iW(g, i, k)] / (0.5 * (q[iρ(g, i, k - 1)] + q[iρ(g, i, k)]))
        end
    end
    fill!(out, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        # mass and Θ: fluxes through the left x-face (i) and the lower z-face (k) of cell (i, k)
        U = q[iU(g, i, k)]
        θx = interp3(θ[ip(g, i - 2), k], θ[ip(g, i - 1), k], θ[i, k], θ[ip(g, i + 1), k], U)
        out[iρ(g, i, k)] += U / dx
        out[iρ(g, i - 1, k)] -= U / dx
        out[iΘ(g, i, k)] += U * θx / dx
        out[iΘ(g, i - 1, k)] -= U * θx / dx
        if k > 1
            W = q[iW(g, i, k)]
            θz = face_z(θ, i, k, nz, W)
            out[iρ(g, i, k)] += W / dz
            out[iρ(g, i, k - 1)] -= W / dz
            out[iΘ(g, i, k)] += W * θz / dz
            out[iΘ(g, i, k - 1)] -= W * θz / dz
        end
        # U: flux U u at the centre of cell i (between x-faces i and i+1)
        Uc = 0.5 * (q[iU(g, i, k)] + q[iU(g, i + 1, k)])
        uc = interp3(u[ip(g, i - 1), k], u[i, k], u[ip(g, i + 1), k], u[ip(g, i + 2), k], Uc)
        out[iU(g, i + 1, k)] += Uc * uc / dx
        out[iU(g, i, k)] -= Uc * uc / dx
        # U: vertical flux W u at the corner (x-face i, z-face k), k = 2 .. nz
        if k > 1
            Wk = 0.5 * (q[iW(g, i - 1, k)] + q[iW(g, i, k)])
            uz = face_z(u, i, k, nz, Wk)
            out[iU(g, i, k)] += Wk * uz / dz
            out[iU(g, i, k - 1)] -= Wk * uz / dz
            # W: horizontal flux U w at the same corner (between columns i-1 and i, z-face k)
            Uk = 0.5 * (q[iU(g, i, k - 1)] + q[iU(g, i, k)])
            wx = interp3(w[ip(g, i - 2), k], w[ip(g, i - 1), k], w[i, k], w[ip(g, i + 1), k], Uk)
            out[iW(g, i, k)] += Uk * wx / dx
            out[iW(g, i - 1, k)] -= Uk * wx / dx
        end
        # W: vertical flux W w at the centre of cell (i, k), between z-faces k and k+1
        Wc = 0.5 * (Wface(g, q, i, k) + Wface(g, q, i, k + 1))
        if k >= 2 && k <= nz - 1
            wc = interp3(w[i, k-1], w[i, k], w[i, k+1], w[i, k+2], Wc)
        else
            wc = interp2(w[i, k], w[i, k+1])
        end
        k + 1 <= nz && (out[iW(g, i, k + 1)] += Wc * wc / dz)
        k >= 2 && (out[iW(g, i, k)] -= Wc * wc / dz)
        # pressure gradient and gravity (deviations from the reference)
        out[iU(g, i, k)] -= (pp[i, k] - pp[ip(g, i - 1), k]) / dx
        if k > 1
            ρw = 0.5 * (q[iρ(g, i, k - 1)] - m.ρref[i, k-1] + q[iρ(g, i, k)] - m.ρref[i, k])
            out[iW(g, i, k)] -= (pp[i, k] - pp[i, k-1]) / dz + grav * ρw
        end
    end
    @inbounds for j in eachindex(out)
        out[j] -= m.damp[j] * (q[j] - m.qtarget[j])
    end
    return out
end

# Gal-Chen mapped finite-volume tendencies. State values remain physical (unweighted); each
# conservative flux divergence is divided by J = dz/dη. The sigma flux is Fz - z_x Fx.
@inline function fv_u_center(g, q, i, k)
    0.5 * (q[iU(g, i, k)] + q[iU(g, i+1, k)])
end

@inline function fv_mz_face(g, q, i, k)
    k == 1 && return 0.5 * (q[iU(g, i, 1)] + q[iU(g, i+1, 1)]) * g.zx_f[i, 1]
    k == g.nz + 1 && return 0.0
    return q[iW(g, i, k)]
end

@inline function fv_ux_at_sigma(g, q, i, k)
    if k == 1
        return 0.5 * (q[iU(g, i, 1)] + q[iU(g, i+1, 1)])
    elseif k == g.nz + 1
        return 0.5 * (q[iU(g, i, g.nz)] + q[iU(g, i+1, g.nz)])
    end
    return 0.25 * (q[iU(g, i, k-1)] + q[iU(g, i+1, k-1)] + q[iU(g, i, k)] + q[iU(g, i+1, k)])
end

function rhs_terrain!(out, m::FVModel, q)
    g, c = m.grid, m.cache
    nx, nz, dx, dη = g.nx, g.nz, g.dx, g.dz
    @inbounds for k in 1:nz, i in 1:nx
        ρ = q[iρ(g, i, k)]
        c.θ[i, k] = q[iΘ(g, i, k)] / ρ
        c.pp[i, k] = pressure(m.gas, q[iΘ(g, i, k)]) - m.pref[i, k]
        c.u[i, k] = q[iU(g, i, k)] / (0.5 * (q[iρ(g, i-1, k)] + ρ))
    end
    @inbounds for i in 1:nx
        for k in 2:nz
            ρf = 0.5 * (q[iρ(g, i, k-1)] + q[iρ(g, i, k)])
            c.w[i, k] = q[iW(g, i, k)] / ρf
        end
        ρb = q[iρ(g, i, 1)]
        ub = fv_u_center(g, q, i, 1) / ρb
        c.w[i, 1] = g.zx_f[i, 1] * ub
        c.w[i, nz+1] = 0.0
    end
    fill!(out, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        ir = ip(g, i+1)
        J = g.Jc[i]
        U0, U1 = q[iU(g, i, k)], q[iU(g, i+1, k)]
        Fx0, Fx1 = g.Jx[i] * U0, g.Jx[ir] * U1
        for (kf, sg) in ((k, -1.0), (k+1, 1.0))
            mz = fv_mz_face(g, q, i, kf)
            Uc = fv_ux_at_sigma(g, q, i, kf)
            Vσ = mz - g.zx_f[i, kf] * Uc
            out[iρ(g, i, k)] -= sg * Vσ / (J * dη)
            θσ = kf == 1 ? c.θ[i, 1] : kf == nz+1 ? c.θ[i, nz] :
                  interp2(c.θ[i, kf-1], c.θ[i, kf])
            out[iΘ(g, i, k)] -= sg * θσ * Vσ / (J * dη)
        end
        out[iρ(g, i, k)] -= (Fx1 - Fx0) / (J * dx)
        θx0 = interp3(c.θ[ip(g, i-2), k], c.θ[ip(g, i-1), k], c.θ[i, k], c.θ[ip(g, i+1), k], U0)
        θx1 = interp3(c.θ[ip(g, i-1), k], c.θ[i, k], c.θ[ir, k], c.θ[ip(g, i+2), k], U1)
        out[iΘ(g, i, k)] -= (g.Jx[ir] * θx1 * U1 - g.Jx[i] * θx0 * U0) / (J * dx)

        # x-momentum control volume centred on x-face i, including the terrain pressure metric.
        im = ip(g, i-1)
        mxm, mxp = fv_u_center(g, q, im, k), fv_u_center(g, q, i, k)
        um = 0.5 * (c.u[im, k] + c.u[i, k])
        up = 0.5 * (c.u[i, k] + c.u[ir, k])
        pleft = c.pp[im, k]
        pright = c.pp[i, k]
        FxL = g.Jc[im] * (mxm * um + pleft)
        FxR = g.Jc[i] * (mxp * up + pright)
        FσL, FσR = 0.0, 0.0
        for (kf, dest) in ((k, 1), (k+1, 2))
            zxs = g.zx_xf[i, kf]
            pface = 0.5 * ((kf == 1 ? c.pp[im, 1] : kf == nz+1 ? c.pp[im, nz] : 0.5*(c.pp[im,kf-1]+c.pp[im,kf])) +
                           (kf == 1 ? c.pp[i, 1] : kf == nz+1 ? c.pp[i, nz] : 0.5*(c.pp[i,kf-1]+c.pp[i,kf])))
            # σ mass flux at the corner (x-face i, σ-face kf), zero at the ground and the top,
            # times the upwind u at the σ-face: one flux per face, shared by the cells k-1 and k
            Fσ = -zxs * pface
            if 1 < kf <= nz
                Uf = 0.5 * (q[iU(g, i, kf-1)] + q[iU(g, i, kf)])
                Mσ = 0.5 * (q[iW(g, im, kf)] + q[iW(g, i, kf)]) - zxs * Uf
                Fσ += Mσ * face_z(c.u, i, kf, nz, Mσ)
            end
            dest == 1 ? (FσL = Fσ) : (FσR = Fσ)
        end
        out[iU(g, i, k)] = -((FxR - FxL) / dx + (FσR - FσL) / dη) / g.Jx[i]

        # Vertical-momentum control volume centred at sigma-face k.
        if k > 1
            W = q[iW(g, i, k)]
            # horizontal fluxes at x-faces, with u interpolated to the sigma-face
            uL = 0.5 * (c.u[i, k-1] + c.u[i, k])
            uR = 0.5 * (c.u[ir, k-1] + c.u[ir, k])
            mzL = 0.5 * (q[iW(g, i, k)] + q[iW(g, im, k)])
            mzR = 0.5 * (q[iW(g, i, k)] + q[iW(g, ir, k)])
            divx = (g.Jx[ir] * mzR * uR - g.Jx[i] * mzL * uL) / dx
            Fsbot, Fstop = 0.0, 0.0
            for (kc, dest) in ((k-1, 1), (k, 2))
                mb = 0.5 * (fv_mz_face(g, q, i, kc) + fv_mz_face(g, q, i, kc+1))
                wb = 0.5 * (c.w[i, kc] + c.w[i, kc+1])
                ub = 0.5 * (c.u[i, kc] + c.u[ir, kc])
                p = c.pp[i, kc]
                Fs = mb * (wb - g.zx_c[i, kc] * ub) + p
                dest == 1 ? (Fsbot = Fs) : (Fstop = Fs)
            end
            ρpert = 0.5 * (q[iρ(g, i, k-1)] - m.ρref[i, k-1] + q[iρ(g, i, k)] - m.ρref[i, k])
            out[iW(g, i, k)] = -(divx + (Fstop - Fsbot) / dη) / J - m.gas.g * ρpert
        end
    end
    @inbounds for j in eachindex(out)
        out[j] -= m.damp[j] * (q[j] - m.qtarget[j])
    end
    return out
end
