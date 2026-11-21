# Fast part (FV3D/moist.jl) on the interior of the tile: horizontal thermodynamic part with phase change, forcing and
# sponge (:hthermo), horizontal momentum part with forcing (:hmomentum), vertical part (thermodynamic fluxes of W,
# vertical pressure gradient and gravity) scaled by (ct, cw), and the column solve (I - D J_v) Y = R per column.
# Halo needs: :hthermo reads U, V of the faces I+1, J+1; :hmomentum reads p (π) of the cells I-1, J-1 (and their
# columns on terrain).

@inline _ci(i, j) = (i + HALO, j + HALO)

"""Horizontal thermodynamic fluxes (and the horizontal part of Ω) of cell (I, J, k): the values of the ρ, X, ρq_v,
ρq_c and p rows (FV3D/moist.jl `mthermo_h!`)."""
@inline function _thermo_h(Z, Cf, Jc, Jx, Jy, zxf, zyf, follow, dx, dy, dz, nz, I, J, k)
    Fw = Jx[I, J] * Z[I, J, k, VU];   Fe = Jx[I+1, J] * Z[I+1, J, k, VU]
    Fs = Jy[I, J] * Z[I, J, k, VV];   Fn = Jy[I, J+1] * Z[I, J+1, k, VV]
    s = 1.0 / Jc[I, J]
    flux(bl, br) = (Cf[I+1, J, k, sx(bl)] * Fe - Cf[I, J, k, sx(br)] * Fw) / dx +
                   (Cf[I, J+1, k, sy(bl)] * Fn - Cf[I, J, k, sy(br)] * Fs) / dy
    oρ = -(s * ((Fe - Fw) / dx + (Fn - Fs) / dy))
    oX = -(s * flux(0, 6)); oq = -(s * flux(1, 1)); oc = -(s * flux(5, 5)); op = -(s * flux(3, 4))
    if follow
        Hb = _az(nz, k) * _Ωh(Z, zxf, zyf, follow, nz, I, J, k); Ht = _az(nz, k + 1) * _Ωh(Z, zxf, zyf, follow, nz, I, J, k + 1)
        vflux(bb, bt) = (Cf[I, J, k+1, sz(bb)] * Ht - Cf[I, J, k, sz(bt)] * Hb) / dz
        oρ -= s * (Ht - Hb) / dz
        oX -= s * vflux(0, 6); oq -= s * vflux(1, 1); oc -= s * vflux(5, 5); op -= s * vflux(3, 4)
    end
    return oρ, oX, oq, oc, op
end

"""out = :hthermo (thermodynamic rows: horizontal fluxes, phase change (micro = :fast), forcing F; W rows: forcing and
sponge), zero elsewhere (out zeroed by the caller)."""
@kernel function _k_hthermo!(out, @Const(Z), @Const(Cf), @Const(F), Jc, Jx, Jy, zxf, zyf, follow, dx, dy, dz, nz,
                             mg, pres, τ, fastphase, damp)
    i, j, k = @index(Global, NTuple)
    I, J = _ci(i, j)
    @inbounds begin
        oρ, oX, oq, oc, op = _thermo_h(Z, Cf, Jc, Jx, Jy, zxf, zyf, follow, dx, dy, dz, nz, I, J, k)
        out[I, J, k, VR] = oρ; out[I, J, k, VX] = oX; out[I, J, k, VQV] = oq; out[I, J, k, VQC] = oc; out[I, J, k, VP] = op
        fastphase && _phase!(out, Z, mg, pres, τ, I, J, k)
        for v in (VR, VX, VQV, VQC, VP)
            out[I, J, k, v] += F[I, J, k, v]
        end
        if k >= 2
            out[I, J, k, VW] = F[I, J, k, VW]
            out[I, J, k, VW] -= damp[k] * (Z[I, J, k, VW] - 0.0)
        end
    end
end

"""Cell field f at the η-face kf of column (I, J): mean, quadratically extrapolated at the ground and the top."""
@inline function _fσ(Z, v, nz, I, J, kf)
    mf(k) = 0.5 * (Z[I, J, k-1, v] + Z[I, J, k, v])
    kf == 1 && return 3mf(2) - 3mf(3) + mf(4)
    kf == nz + 1 && return 3mf(nz) - 3mf(nz - 1) + mf(nz - 2)
    return mf(kf)
end
@inline function _gradx(Z, Jc, Jx, zxx, follow, dx, dz, nz, I, J, k)
    d = (Jc[I, J] * Z[I, J, k, VP] - Jc[I-1, J] * Z[I-1, J, k, VP]) / dx
    if follow
        cor(kf) = 0.5 * (_fσ(Z, VP, nz, I - 1, J, kf) + _fσ(Z, VP, nz, I, J, kf))
        d -= (zxx[I, J, k+1] * cor(k + 1) - zxx[I, J, k] * cor(k)) / dz
    end
    return d / Jx[I, J]
end
@inline function _grady(Z, Jc, Jy, zyy, follow, dy, dz, nz, I, J, k)
    d = (Jc[I, J] * Z[I, J, k, VP] - Jc[I, J-1] * Z[I, J-1, k, VP]) / dy
    if follow
        cor(kf) = 0.5 * (_fσ(Z, VP, nz, I, J - 1, kf) + _fσ(Z, VP, nz, I, J, kf))
        d -= (zyy[I, J, k+1] * cor(k + 1) - zyy[I, J, k] * cor(k)) / dz
    end
    return d / Jy[I, J]
end

"""out = :hmomentum (U, V rows: -c_pd Θ̂_f ∂π|_z or -∂p|_z, forcing F)."""
@kernel function _k_hmomentum!(out, @Const(Z), @Const(Cf), @Const(F), Jc, Jx, Jy, zxx, zyy, follow, dx, dy, dz, nz)
    i, j, k = @index(Global, NTuple)
    I, J = _ci(i, j)
    @inbounds begin
        out[I, J, k, VU] = -(Cf[I, J, k, sx(2)] * _gradx(Z, Jc, Jx, zxx, follow, dx, dz, nz, I, J, k)) + F[I, J, k, VU]
        out[I, J, k, VV] = -(Cf[I, J, k, sy(2)] * _grady(Z, Jc, Jy, zyy, follow, dy, dz, nz, I, J, k)) + F[I, J, k, VV]
    end
end

"""out = D G_v(Z): vertical thermodynamic fluxes (rows × ct) and the vertical momentum part (W rows × cw)."""
@kernel function _k_vertical!(out, @Const(Z), @Const(Cf), Jc, dz, nz, grav, ct, cw)
    i, j, k = @index(Global, NTuple)
    I, J = _ci(i, j)
    @inbounds begin
        Fb = _az(nz, k) * _Wface(Z, nz, I, J, k); Ft = _az(nz, k + 1) * _Wface(Z, nz, I, J, k + 1)
        s = 1.0 / (Jc[I, J] * dz)
        vflux(bb, bt) = Cf[I, J, k+1, sz(bb)] * Ft - Cf[I, J, k, sz(bt)] * Fb
        out[I, J, k, VR] = -(s * (Ft - Fb)) * ct
        out[I, J, k, VX] = -(s * vflux(0, 6)) * ct
        out[I, J, k, VQV] = -(s * vflux(1, 1)) * ct
        out[I, J, k, VQC] = -(s * vflux(5, 5)) * ct
        out[I, J, k, VP] = -(s * vflux(3, 4)) * ct
        if k >= 2
            out[I, J, k, VW] = -(Cf[I, J, k, sz(2)] * (Z[I, J, k, VP] - Z[I, J, k-1, VP]) / (Jc[I, J] * dz) +
                                 grav * 0.5 * (Z[I, J, k-1, VR] + Z[I, J, k, VR])) * cw
        end
    end
end

# ------------------------------------------------------------------ column solve (FV3D/moist.jl moist_column_solve!)
@kernel function _k_factor!(sub, cpr, iden, @Const(Cf), Jc, dz0, nz, grav, c, cw)
    i, j = @index(Global, NTuple)
    I, J = _ci(i, j)
    @inbounds begin
        dz = Jc[I, J] * dz0
        r = c * cw / dz^2; gg = c * cw * grav / (2dz)
        for k in 2:nz
            P = Cf[I, J, k, sz(2)]
            a = k > 2 ? -r * P * Cf[I, J, k-1, sz(4)] + gg : 0.0
            d = 1 + r * P * (Cf[I, J, k, sz(4)] + Cf[I, J, k, sz(3)])
            b = k < nz ? -r * P * Cf[I, J, k+1, sz(3)] - gg : 0.0
            den = k == 2 ? d : d - a * cpr[k-1, I, J]
            sub[k, I, J] = a; iden[k, I, J] = 1 / den; cpr[k, I, J] = b / den
        end
    end
end

@kernel function _k_csolve!(Y, Wc, @Const(R), @Const(Cf), @Const(sub), @Const(cpr), @Const(iden), Jc, dz0, nz, grav, c, cw)
    i, j = @index(Global, NTuple)
    I, J = _ci(i, j)
    @inbounds begin
        dz = Jc[I, J] * dz0
        for k in 2:nz
            rhs = R[I, J, k, VW] - (cw / dz) * Cf[I, J, k, sz(2)] * (R[I, J, k, VP] - R[I, J, k-1, VP]) -
                  (cw * grav / 2) * (R[I, J, k-1, VR] + R[I, J, k, VR])
            Wc[I, J, k] = k == 2 ? rhs * iden[k, I, J] : (rhs - sub[k, I, J] * Wc[I, J, k-1]) * iden[k, I, J]
        end
        for k in nz-1:-1:2
            Wc[I, J, k] -= cpr[k, I, J] * Wc[I, J, k+1]
        end
        Wc[I, J, 1] = 0.0; Wc[I, J, nz+1] = 0.0
        r = c / dz
        for k in 1:nz
            Wt, Wb = Wc[I, J, k+1], Wc[I, J, k]
            Y[I, J, k, VR] = R[I, J, k, VR] - r * (Wt - Wb)
            Y[I, J, k, VX] = R[I, J, k, VX] - r * (Cf[I, J, k+1, sz(0)] * Wt - Cf[I, J, k, sz(6)] * Wb)
            Y[I, J, k, VQV] = R[I, J, k, VQV] - r * (Cf[I, J, k+1, sz(1)] * Wt - Cf[I, J, k, sz(1)] * Wb)
            Y[I, J, k, VQC] = R[I, J, k, VQC] - r * (Cf[I, J, k+1, sz(5)] * Wt - Cf[I, J, k, sz(5)] * Wb)
            Y[I, J, k, VP] = R[I, J, k, VP] - r * (Cf[I, J, k+1, sz(3)] * Wt - Cf[I, J, k, sz(4)] * Wb)
        end
        for k in 2:nz
            Y[I, J, k, VW] = Wc[I, J, k]
        end
    end
end

"""Column factors of (I - D J_v), D = c (thermodynamic rows), cw (W rows), cached per c cw (cleared per stage)."""
function _factors(m::MoistKA, Cf, c, cw)
    g = m.grid
    get!(m.work.fac, Float64(c * cw)) do
        be = KA.get_backend(m)
        f = ntuple(_ -> KA.zeros(be, Float64, g.nz + 1, g.nxh, g.nyh), 3)
        _launch!(m, _k_factor!, (g.nxl, g.nyl), f..., Cf, g.Jc, g.dz, g.nz, m.grav, Float64(c), Float64(cw))
        f
    end
end
"""Y = (I - D J_v)⁻¹ R on the interior columns (U, V rows and the halo: Y = R)."""
function ka_column_solve!(Y, m::MoistKA, Cf, c, R, cw = c)
    g = m.grid
    sub, cpr, iden = _factors(m, Cf, c, cw)
    Y .= R
    _launch!(m, _k_csolve!, (g.nxl, g.nyl), Y, m.work.Wc, R, Cf, sub, cpr, iden, g.Jc, g.dz, g.nz, m.grav, Float64(c), Float64(cw))
    return Y
end

# ------------------------------------------------------------------ parts
"""out = :hthermo of the state Z (coefficients Cf, forcing F)."""
function ka_hthermo!(out, m::MoistKA, Cf, Z, F)
    g = m.grid
    fill!(out, 0.0)
    _launch!(m, _k_hthermo!, (g.nxl, g.nyl, g.nz), out, Z, Cf, F, g.Jc, g.Jx, g.Jy, g.zxf, g.zyf, g.follow, g.dx, g.dy, g.dz,
             g.nz, m.mgas, m.pres, m.τ, m.micro == 1, m.damp)
    return out
end
"""out = :hmomentum of Z."""
function ka_hmomentum!(out, m::MoistKA, Cf, Z, F)
    g = m.grid
    fill!(out, 0.0)
    _launch!(m, _k_hmomentum!, (g.nxl, g.nyl, g.nz), out, Z, Cf, F, g.Jc, g.Jx, g.Jy, g.zxx, g.zyy, g.follow, g.dx, g.dy, g.dz, g.nz)
    return out
end
"""out = D G_v(Z), D = ct (thermodynamic rows), cw (W rows)."""
function ka_vertical!(out, m::MoistKA, Cf, Z, ct, cw)
    g = m.grid
    fill!(out, 0.0)
    _launch!(m, _k_vertical!, (g.nxl, g.nyl, g.nz), out, Z, Cf, g.Jc, g.dz, g.nz, m.grav, Float64(ct), Float64(cw))
    return out
end
"""out = phase change of Z (micro = :fast); false if there is none."""
function ka_phase!(out, m::MoistKA, Z)
    m.micro == 1 || return false
    g = m.grid
    fill!(out, 0.0)
    _launch!(m, _k_phase!, (g.nxl, g.nyl, g.nz), out, Z, m.mgas, m.pres, m.τ)
    return true
end
"""z ← z + Y, (I - D J_v) Y = D G_v(z)."""
function ka_vertical_implicit!(Z, m::MoistKA, Cf, ct, cw, k, y)
    ka_vertical!(k, m, Cf, Z, ct, cw)
    ka_column_solve!(y, m, Cf, ct, k, cw)
    Z .+= y
    return Z
end

"""
    ka_rhs!(out, m, Q, Cf = ka_zeros(m, NC), S = ka_zeros(m))

N(q) = F(q, q) on the interior (halo of Q filled here): :hthermo + :hmomentum + the vertical part.
"""
function ka_rhs!(out, m::MoistKA, Q, Cf = ka_zeros(m, NC), S = ka_zeros(m), tmp = ka_zeros(m))
    exchange!(m.dec, Q, m.grid)
    ka_frozen_data!(Cf, S, m, Q)
    ka_hthermo!(out, m, Cf, Q, S)
    out .+= ka_hmomentum!(tmp, m, Cf, Q, S)
    out .+= ka_vertical!(tmp, m, Cf, Q, 1.0, 1.0)
    return out
end
