# Frozen coefficients C(y) and the slow part S(y) (FV3D/moist.jl `frozen_data!`): cell quantities, mass fluxes, face
# values (pseudo third-order upwind, advection.jl of FV3D on full cells: volumes J, x-/y-faces J_x, J_y, η-faces 1),
# momentum advection by the shift technique. The state halo must be filled (`exchange!`). Every quantity is computed
# where its stencil lies inside the tile; the interior and its faces are always covered (HALO = 4).

"""Horizontal part -z_x Ū - z_y V̄ of Ω at the η-face k of column (I, J) (zero at the walls and on flat grids)."""
@inline function _Ωh(Q, zxf, zyf, follow, nz, I, J, k)
    (!follow || k == 1 || k == nz + 1) && return 0.0
    Ub = 0.25 * (Q[I, J, k-1, VU] + Q[I+1, J, k-1, VU] + Q[I, J, k, VU] + Q[I+1, J, k, VU])
    Vb = 0.25 * (Q[I, J, k-1, VV] + Q[I, J+1, k-1, VV] + Q[I, J, k, VV] + Q[I, J+1, k, VV])
    return -(zxf[I, J, k] * Ub + zyf[I, J, k] * Vb)
end
@inline _Wface(Q, nz, I, J, k) = (k == 1 || k == nz + 1) ? 0.0 : Q[I, J, k, VW]
"""Open η-face measure (walls 0) and the measure of the shift (walls 1: flat or terrain-following ground, top)."""
@inline _az(nz, k) = (k == 1 || k == nz + 1) ? 0.0 : 1.0

"""Face values at the x-face I (between I-1 and I), the y-face J and the η-face k (2 … nz), upwind by the sign of F."""
@inline _fx(φ, Jc, Jx, I, J, k, F) = !signbit(F) ?
    recon3(φ[I-2, J, k], φ[I-1, J, k], φ[I, J, k], Jc[I-2, J], Jc[I-1, J], Jc[I, J], Jx[I-1, J], Jx[I, J]) :
    recon3(φ[I+1, J, k], φ[I, J, k], φ[I-1, J, k], Jc[I+1, J], Jc[I, J], Jc[I-1, J], Jx[I+1, J], Jx[I, J])
@inline _fy(φ, Jc, Jy, I, J, k, F) = !signbit(F) ?
    recon3(φ[I, J-2, k], φ[I, J-1, k], φ[I, J, k], Jc[I, J-2], Jc[I, J-1], Jc[I, J], Jy[I, J-1], Jy[I, J]) :
    recon3(φ[I, J+1, k], φ[I, J, k], φ[I, J-1, k], Jc[I, J+1], Jc[I, J], Jc[I, J-1], Jy[I, J+1], Jy[I, J])
@inline function _fz(φ, Jc, nz, I, J, k, F)
    v = Jc[I, J]
    if !signbit(F)
        ku = max(k - 2, 1)
        return recon3(φ[I, J, ku], φ[I, J, k-1], φ[I, J, k], v, v, v, 1.0, _az(nz, k), _az(nz, k - 1) > 0)
    else
        ku = min(k + 1, nz)
        return recon3(φ[I, J, ku], φ[I, J, k], φ[I, J, k-1], v, v, v, 1.0, _az(nz, k), _az(nz, k + 1) > 0)
    end
end

# ------------------------------------------------------------------ cells, mass fluxes, shifted velocities
"""Face velocities of each cell: u, v at its left/lower face (slots 1, 2) and at its right/upper face (slots 3, 4),
u = U/ρ̄. Before the exchange from the interior; the exchange brings the outer faces of the halo cells (interior
faces of the neighbouring tiles), after it the remaining faces are computed with the halo densities."""
@kernel function _k_uv!(UV, @Const(Q), nxh, nyh)
    I, J, k = @index(Global, NTuple)
    @inbounds begin
        uf(I) = Q[I, J, k, VU] / (0.5 * (Q[I-1, J, k, VR] + Q[I, J, k, VR]))
        vf(J) = Q[I, J, k, VV] / (0.5 * (Q[I, J-1, k, VR] + Q[I, J, k, VR]))
        I > 1 && (UV[I, J, k, 1] = uf(I))
        J > 1 && (UV[I, J, k, 2] = vf(J))
        I < nxh && (UV[I, J, k, 3] = uf(I + 1))
        J < nyh && (UV[I, J, k, 4] = vf(J + 1))
    end
end

@kernel function _k_cells!(θ, φq, φc, abc, Θh, pc, @Const(Q), mg, pres)
    I, J, k = @index(Global, NTuple)
    @inbounds begin
        ρ = Q[I, J, k, VR]; X = Q[I, J, k, VX]; ρv = Q[I, J, k, VQV]; ρc = Q[I, J, k, VQC]
        θ[I, J, k] = X / ρ; φq[I, J, k] = ρv / ρ; φc[I, J, k] = ρc / ρ
        d = pres ? exner_derivatives(mg, ρ, X, ρv, ρc; f = _moist_p) : exner_derivatives(mg, ρ, X, ρv, ρc; f = moist_exner)
        abc[I, J, k, 1] = d[1]; abc[I, J, k, 2] = d[2]; abc[I, J, k, 3] = d[3]; abc[I, J, k, 4] = d[4]
        πc = Q[I, J, k, VP]
        if pres
            Θh[I, J, k] = 1.0; pc[I, J, k] = πc
        else
            p = mg.p0 * πc^(mg.Cpd / mg.Rd)
            Θh[I, J, k] = mg.Cpd * p / (mg.Rd * πc); pc[I, J, k] = p
        end
    end
end

"""Face w = W / mean ρ for the shift (ground: z_x u + z_y v terrain following, 0 flat; top 0)."""
@kernel function _k_wface!(w, @Const(UV), @Const(Q), zxf, zyf, follow, nz)
    I, J = @index(Global, NTuple)
    @inbounds begin
        w[I, J, 1] = follow ? zxf[I, J, 1] * 0.5 * (UV[I, J, 1, 1] + UV[I, J, 1, 3]) +
                              zyf[I, J, 1] * 0.5 * (UV[I, J, 1, 2] + UV[I, J, 1, 4]) : 0.0
        w[I, J, nz+1] = 0.0
        for k in 2:nz
            w[I, J, k] = Q[I, J, k, VW] / (0.5 * (Q[I, J, k-1, VR] + Q[I, J, k, VR]))
        end
    end
end

"""Mass fluxes J_x U, J_y V (x-, y-faces) and Ω (η-faces of the interior columns)."""
@kernel function _k_fluxes!(Fx, Fy, Fz, @Const(Q), Jx, Jy, zxf, zyf, follow, nz, nxh, nyh)
    I, J, k = @index(Global, NTuple)
    @inbounds begin
        if k <= nz
            Fx[I, J, k] = Jx[I, J] * Q[I, J, k, VU]
            Fy[I, J, k] = Jy[I, J] * Q[I, J, k, VV]
        end
        if k == 1 || k == nz + 1
            Fz[I, J, k] = 0.0
        elseif I < nxh && J < nyh
            Fz[I, J, k] = 1.0 * (_Wface(Q, nz, I, J, k) + _Ωh(Q, zxf, zyf, follow, nz, I, J, k))
        end
    end
end

"""Shifted velocities of the interior and the halo cells (left/right values of u, v, w from the two faces)."""
@kernel function _k_shift!(SH, @Const(UV), @Const(w), Jx, Jy)
    I, J, k = @index(Global, NTuple)
    @inbounds begin
        SH[I, J, k, 1], SH[I, J, k, 2] = shiftvel(Jx[I, J], Jx[I+1, J], UV[I, J, k, 1], UV[I, J, k, 3])
        SH[I, J, k, 3], SH[I, J, k, 4] = shiftvel(Jy[I, J], Jy[I, J+1], UV[I, J, k, 2], UV[I, J, k, 4])
        SH[I, J, k, 5], SH[I, J, k, 6] = shiftvel(1.0, 1.0, w[I, J, k], w[I, J, k+1])
    end
end

# ------------------------------------------------------------------ face values (third order, upwind tile)
"""x-face I (HALO+1 … HALO+nxl+1) of the interior rows: computable if the upwind stencil lies in the tile (two
upwind cells, one downwind): the tile-boundary faces only from the upwind side (the other side by `merge_faces!`)."""
@inline _okx(I, F, nxl) = !signbit(F) ? I >= HALO + 2 : I <= HALO + nxl

@kernel function _k_facevals!(FVx, FVy, @Const(θ), @Const(φq), @Const(φc), @Const(SH), @Const(Fx), @Const(Fy),
                              Jc, Jx, Jy, dx, dy, nxl, nyl)
    i, j, k = @index(Global, NTuple)                  # i ∈ 1 … nxl+1, j ∈ 1 … nyl+1
    I, J = i + HALO, j + HALO
    @inbounds begin
        if j <= nyl
            F = Fx[I, J, k]
            if _okx(I, F, nxl)
                FVx[I, J, k, 1] = _fx(θ, Jc, Jx, I, J, k, F); FVx[I, J, k, 2] = _fx(φq, Jc, Jx, I, J, k, F)
                FVx[I, J, k, 3] = _fx(φc, Jc, Jx, I, J, k, F)
                for s in 1:6
                    FVx[I, J, k, 3+s] = F != 0 ? F * _fx(view(SH, :, :, :, s), Jc, Jx, I, J, k, F) / dx : 0.0
                end
            end
        end
        if i <= nxl
            F = Fy[I, J, k]
            if _okx(J, F, nyl)
                FVy[I, J, k, 1] = _fy(θ, Jc, Jy, I, J, k, F); FVy[I, J, k, 2] = _fy(φq, Jc, Jy, I, J, k, F)
                FVy[I, J, k, 3] = _fy(φc, Jc, Jy, I, J, k, F)
                for s in 1:6
                    FVy[I, J, k, 3+s] = F != 0 ? F * _fy(view(SH, :, :, :, s), Jc, Jy, I, J, k, F) / dy : 0.0
                end
            end
        end
    end
end

"""Coefficients of one face between the cells l (left/below) and r (right/above) into the slots s0 … s0+6."""
@inline function _face!(Cf, I, J, k, s0, θf, qf, cf, l, r, Q, abc, Θh, pc, en)
    χl = θf; χr = θf
    if en                                                   # :energy: χ = e_f + p_c/ρ̄_f
        ρf = 0.5 * (Q[l[1], l[2], l[3], VR] + Q[r[1], r[2], r[3], VR])
        χl = θf + pc[l[1], l[2], l[3]] / ρf; χr = θf + pc[r[1], r[2], r[3]] / ρf
    end
    κ(c, χ) = abc[c[1], c[2], c[3], 1] + abc[c[1], c[2], c[3], 2] * χ + abc[c[1], c[2], c[3], 3] * qf + abc[c[1], c[2], c[3], 4] * cf
    Cf[I, J, k, s0] = χl; Cf[I, J, k, s0+6] = χr; Cf[I, J, k, s0+1] = qf; Cf[I, J, k, s0+5] = cf
    Cf[I, J, k, s0+2] = 0.5 * (Θh[l[1], l[2], l[3]] + Θh[r[1], r[2], r[3]])
    Cf[I, J, k, s0+3] = κ(l, χl); Cf[I, J, k, s0+4] = κ(r, χr)
    return nothing
end

"""Face coefficients of the faces of the interior cells (x-, y-faces from the merged face values, η-faces local)."""
@kernel function _k_coefs!(Cf, @Const(Q), @Const(FVx), @Const(FVy), @Const(θ), @Const(φq), @Const(φc), @Const(abc),
                           @Const(Θh), @Const(pc), @Const(Fz), Jc, nz, nxl, nyl, en)
    i, j, k = @index(Global, NTuple)                  # i ∈ 1 … nxl+1, j ∈ 1 … nyl+1, k ∈ 1 … nz+1
    I, J = i + HALO, j + HALO
    @inbounds begin
        if k <= nz && j <= nyl
            _face!(Cf, I, J, k, sx(0), FVx[I, J, k, 1], FVx[I, J, k, 2], FVx[I, J, k, 3], (I - 1, J, k), (I, J, k), Q, abc, Θh, pc, en)
        end
        if k <= nz && i <= nxl
            _face!(Cf, I, J, k, sy(0), FVy[I, J, k, 1], FVy[I, J, k, 2], FVy[I, J, k, 3], (I, J - 1, k), (I, J, k), Q, abc, Θh, pc, en)
        end
        if 2 <= k <= nz && i <= nxl && j <= nyl
            F = Fz[I, J, k]
            _face!(Cf, I, J, k, sz(0), _fz(θ, Jc, nz, I, J, k, F), _fz(φq, Jc, nz, I, J, k, F), _fz(φc, Jc, nz, I, J, k, F),
                   (I, J, k - 1), (I, J, k), Q, abc, Θh, pc, en)
        end
    end
end

"""Advective tendencies of the six shifted velocities in the interior cells: -(1/J) Σ F φ_f."""
@kernel function _k_advect!(TT, @Const(SH), @Const(FVx), @Const(FVy), @Const(Fz), Jc, dz, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds for s in 1:6
        φ = view(SH, :, :, :, s)
        t = 0.0
        t += FVx[I, J, k, 3+s]
        t += FVy[I, J, k, 3+s]
        k > 1 && (F = Fz[I, J, k]; F != 0 && (t += F * _fz(φ, Jc, nz, I, J, k, F) / dz))
        t -= FVx[I+1, J, k, 3+s]
        t -= FVy[I, J+1, k, 3+s]
        k < nz && (F = Fz[I, J, k+1]; F != 0 && (t -= F * _fz(φ, Jc, nz, I, J, k + 1, F) / dz))
        TT[I, J, k, s] = t * (1.0 / Jc[I, J])
    end
end

"""Face tendencies of U, V, W from the cell tendencies (shift back), on the faces of the interior cells."""
@kernel function _k_shiftback!(S, @Const(TT), Jc, Jx, Jy, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        S[I, J, k, VU] = shift_back(TT[I, J, k, 1], TT[I-1, J, k, 2], Jc[I-1, J], Jc[I, J], Jx[I, J], Jx[I-1, J], Jx[I+1, J])
        S[I, J, k, VV] = shift_back(TT[I, J, k, 3], TT[I, J-1, k, 4], Jc[I, J-1], Jc[I, J], Jy[I, J], Jy[I, J-1], Jy[I, J+1])
        k >= 2 && (S[I, J, k, VW] = shift_back(TT[I, J, k, 5], TT[I, J, k-1, 6], Jc[I, J], Jc[I, J], 1.0, 1.0, 1.0))
    end
end

"""Phase change of cell (I, J, k) of the state Z into out (FV3D/moist.jl `_phase!`)."""
@inline function _phase!(out, Z, mg, pres, τ, I, J, k)
    ρ, X, ρv, ρc = Z[I, J, k, VR], Z[I, J, k, VX], Z[I, J, k, VQV], Z[I, J, k, VQC]
    φ, T = fischer_burmeister(mg, ρ, X, ρv, ρc)
    Sc = -φ / τ
    p = (mg.Rd * (ρ - ρv - ρc) + mg.Rv * ρv) * T
    Qp = pressure_phase_factor(mg, ρ, T, ρv, ρc)
    κS = pres ? Qp : (mg.Rd / mg.Cpd) * (p / mg.p0)^(mg.Rd / mg.Cpd) / p * Qp
    out[I, J, k, VQC] += Sc
    out[I, J, k, VQV] -= Sc
    out[I, J, k, VX] += theta_phase_factor(mg, ρ, X, T, p, ρv, ρc) * Sc
    out[I, J, k, VP] += κS * Sc
    return nothing
end
@kernel function _k_phase!(out, @Const(Z), mg, pres, τ)
    i, j, k = @index(Global, NTuple)
    @inbounds _phase!(out, Z, mg, pres, τ, i + HALO, j + HALO, k)
end

"""
    ka_frozen_data!(Cf, S, m, Q)

C(y) (face coefficients) and S(y) (momentum advection, phase change for micro = :slow) of the state Q. Three
communications: the state with the face velocities u, v (so that the shifted velocities of the halo cells are local),
the tile-boundary face values (from the upwind tile), the cell tendencies of the shifted velocities (shift back).
"""
function ka_frozen_data!(Cf, S, m::MoistKA, Q)
    g, w, dec = m.grid, m.work, m.dec
    nz, nxh, nyh, nxl, nyl = g.nz, g.nxh, g.nyh, g.nxl, g.nyl
    _launch!(m, _k_uv!, (nxh, nyh, nz), w.UV, Q, nxh, nyh)                       # interior faces (halo stale)
    exchange!(dec, (Q, w.UV), g)                                                 # 1: state and u, v
    _launch!(m, _k_uv!, (nxh, nyh, nz), w.UV, Q, nxh, nyh)                       # all other faces with the halo
    _launch!(m, _k_cells!, (nxh, nyh, nz), w.θ, w.φq, w.φc, w.abc, w.Θh, w.pc, Q, m.mgas, m.pres)
    _launch!(m, _k_wface!, (nxh, nyh), w.w, w.UV, Q, g.zxf, g.zyf, g.follow, nz)
    _launch!(m, _k_fluxes!, (nxh, nyh, nz + 1), w.Fx, w.Fy, w.Fz, Q, g.Jx, g.Jy, g.zxf, g.zyf, g.follow, nz, nxh, nyh)
    _launch!(m, _k_shift!, (nxh, nyh, nz), w.SH, w.UV, w.w, g.Jx, g.Jy)
    fill!(w.FVx, 0.0); fill!(w.FVy, 0.0)
    _launch!(m, _k_facevals!, (nxl + 1, nyl + 1, nz), w.FVx, w.FVy, w.θ, w.φq, w.φc, w.SH, w.Fx, w.Fy, g.Jc, g.Jx, g.Jy,
             g.dx, g.dy, nxl, nyl)
    merge_faces!(dec, w.FVx, w.Fx, w.FVy, w.Fy, g)                              # 2: tile-boundary faces
    fill!(Cf, 0.0); fill!(S, 0.0)
    _launch!(m, _k_coefs!, (nxl + 1, nyl + 1, nz + 1), Cf, Q, w.FVx, w.FVy, w.θ, w.φq, w.φc, w.abc, w.Θh, w.pc, w.Fz,
             g.Jc, nz, nxl, nyl, m.mgas.thvar === :energy)
    m.micro == 2 && _launch!(m, _k_phase!, (nxl, nyl, nz), S, Q, m.mgas, m.pres, m.τ)
    _launch!(m, _k_advect!, (nxl, nyl, nz), w.TT, w.SH, w.FVx, w.FVy, w.Fz, g.Jc, g.dz, nz)
    exchange!(dec, w.TT, g, (2, 4))                                              # 3: tendencies of uR, vR
    _launch!(m, _k_shiftback!, (nxl, nyl, nz), S, w.TT, g.Jc, g.Jx, g.Jy, nz)
    return Cf, S
end
