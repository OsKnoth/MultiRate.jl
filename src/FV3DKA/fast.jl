# Fast part (FV3D/moist.jl) on the interior of the tile: horizontal thermodynamic part with phase change, forcing and
# sponge (:hthermo), horizontal momentum part with forcing (:hmomentum), vertical part (thermodynamic fluxes of W,
# vertical pressure gradient and gravity) scaled by (ct, cw), and the column solve (I - D J_v) Y = R per column.
# Halo needs: :hthermo reads U, V of the faces I+1, J+1; :hmomentum reads p (π) of the cells I-1, J-1 (and their
# columns on terrain).
#
# Overlap of computation and communication (mis.jl, as in CGDycore: send, compute what needs no halo, receive, compute
# the rest). The split is by what needs the halo, not by a ring of columns: while p is in flight, :hmomentum on all
# faces except the U faces of the first interior column and the V faces of the first row (they read p of the halo);
# while U, V are in flight, :hthermo on the cells without the last interior column and row (those read U, V of the
# faces I+1, J+1 in the halo) and the phase change. The column solve and the vertical parts need no halo and always
# run once over all columns. Kernels run over the columns of a `Region` and write all rows they own there.

"""Columns of a tile: kind 0 all interior columns, 1 :ws (the first interior column and row, the cells whose west or
south face needs the halo), 2 :en (the last interior column and row: east or north face in the halo), 3 :core (all
without the last column and row)."""
struct Region
    kind::Int
    nxl::Int
    nyl::Int
end
Region(g::KAGrid, kind::Symbol) = Region(kind === :all ? 0 : kind === :ws ? 1 : kind === :en ? 2 : kind === :core ? 3 :
                                         error("region: :all, :ws, :en or :core"), g.nxl, g.nyl)
"""Number of columns of the region."""
ncols(R::Region) = R.kind == 0 ? R.nxl * R.nyl : R.kind == 3 ? (R.nxl - 1) * (R.nyl - 1) : R.nxl + R.nyl - 1
"""Interior column (i, j) (1 … nxl, 1 … nyl) of the r-th column of the region (r runs along i where possible)."""
@inline function _ij(R::Region, r)
    if R.kind == 1                                             # row j = 1, then column i = 1 (j ≥ 2)
        return r <= R.nxl ? (r, 1) : (1, r - R.nxl + 1)
    elseif R.kind == 2                                         # row j = nyl, then column i = nxl (j < nyl)
        return r <= R.nxl ? (r, R.nyl) : (R.nxl, r - R.nxl)
    end
    ni = R.kind == 3 ? R.nxl - 1 : R.nxl
    return ((r - 1) % ni + 1, (r - 1) ÷ ni + 1)
end
@inline function _IJ(R::Region, r)
    i, j = _ij(R, r)
    return (i + HALO, j + HALO)
end

"""Horizontal thermodynamic fluxes (and the horizontal part of Ω) of cell (I, J, k): the values of the ρ, X, ρq_v,
ρq_c and p rows (FV3D/moist.jl `mthermo_h!`)."""
@inline function _thermo_h(Z, Cf, Jc, Jx, Jy, zxf, zyf, follow, dx, dy, dz, nz, I, J, k, rain = false)
    Fw = Jx[I, J] * Z[I, J, k, VU];   Fe = Jx[I+1, J] * Z[I+1, J, k, VU]
    Fs = Jy[I, J] * Z[I, J, k, VV];   Fn = Jy[I, J+1] * Z[I, J+1, k, VV]
    s = 1.0 / Jc[I, J]
    flux(bl, br) = (Cf[I+1, J, k, sx(bl)] * Fe - Cf[I, J, k, sx(br)] * Fw) / dx +
                   (Cf[I, J+1, k, sy(bl)] * Fn - Cf[I, J, k, sy(br)] * Fs) / dy
    oρ = -(s * ((Fe - Fw) / dx + (Fn - Fs) / dy))
    oX = -(s * flux(0, 6)); oq = -(s * flux(1, 1)); oc = -(s * flux(5, 5)); op = -(s * flux(3, 4))
    orr = rain ? -(s * ((Cf[I+1, J, k, SRX] * Fe - Cf[I, J, k, SRX] * Fw) / dx +
                        (Cf[I, J+1, k, SRY] * Fn - Cf[I, J, k, SRY] * Fs) / dy)) : 0.0
    if follow
        Hb = _az(nz, k) * _Ωh(Z, zxf, zyf, follow, nz, I, J, k); Ht = _az(nz, k + 1) * _Ωh(Z, zxf, zyf, follow, nz, I, J, k + 1)
        vflux(bb, bt) = (Cf[I, J, k+1, sz(bb)] * Ht - Cf[I, J, k, sz(bt)] * Hb) / dz
        oρ -= s * (Ht - Hb) / dz
        oX -= s * vflux(0, 6); oq -= s * vflux(1, 1); oc -= s * vflux(5, 5); op -= s * vflux(3, 4)
        rain && (orr -= s * (Cf[I, J, k+1, SRZ] * Ht - Cf[I, J, k, SRZ] * Hb) / dz)
    end
    return oρ, oX, oq, oc, op, orr
end

"""out = :hthermo (thermodynamic rows: horizontal fluxes, phase change (micro = :fast), sedimentation (rain), forcing
F; W rows: forcing, sponge, sedimentation) on the region."""
@kernel function _k_hthermo!(out, @Const(Z), @Const(Cf), @Const(F), Jc, Jx, Jy, zxf, zyf, follow, dx, dy, dz, nz,
                             mg, pres, τ, fastphase, damp, rain, R)
    r, k = @index(Global, NTuple)
    I, J = _IJ(R, r)
    @inbounds begin
        oρ, oX, oq, oc, op, orr = _thermo_h(Z, Cf, Jc, Jx, Jy, zxf, zyf, follow, dx, dy, dz, nz, I, J, k, rain)
        out[I, J, k, VR] = oρ; out[I, J, k, VX] = oX; out[I, J, k, VQV] = oq; out[I, J, k, VQC] = oc; out[I, J, k, VP] = op
        fastphase && _phase!(out, Z, mg, pres, τ, rain, I, J, k)
        for v in (VR, VX, VQV, VQC, VP)
            out[I, J, k, v] += F[I, J, k, v]
        end
        if rain
            dρ, dX, dp = _sed_thermo(Z, Cf, Jc, dz, nz, I, J, k)
            out[I, J, k, VQR] = orr + dρ + F[I, J, k, VQR]
            out[I, J, k, VR] += dρ; out[I, J, k, VX] += dX; out[I, J, k, VP] += dp
        end
        if k >= 2
            out[I, J, k, VW] = F[I, J, k, VW]
            out[I, J, k, VW] -= damp[k] * (Z[I, J, k, VW] - 0.0)
            rain && (out[I, J, k, VW] += _sed_w(Z, Cf, Jc, dz, nz, I, J, k))
        end
    end
end

"""out = phase change of Z on the thermodynamic and W rows of the region (ρ and W rows zero)."""
@kernel function _k_phaseset!(out, @Const(Z), mg, pres, τ, rain, R)
    r, k = @index(Global, NTuple)
    I, J = _IJ(R, r)
    @inbounds begin
        out[I, J, k, VR] = 0.0; out[I, J, k, VX] = 0.0; out[I, J, k, VQV] = 0.0; out[I, J, k, VQC] = 0.0; out[I, J, k, VP] = 0.0
        rain && (out[I, J, k, VQR] = 0.0)
        k >= 2 && (out[I, J, k, VW] = 0.0)
        _phase!(out, Z, mg, pres, τ, rain, I, J, k)
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

"""
:hmomentum (U, V rows: -c_pd Θ̂_f ∂π|_z or -∂p|_z, forcing F) of the faces of the region's cells, times a, added to out
(add) or stored. part 0: all faces, 1: the faces that need no halo (U for i > 1, V for j > 1), 2: the others. out may be
Z (the kernel reads p only).
"""
@kernel function _k_hmomentum!(out, Z, @Const(Cf), @Const(F), Jc, Jx, Jy, zxx, zyy, follow, dx, dy, dz, nz, a, add,
                               part, rain, R)
    r, k = @index(Global, NTuple)
    i, j = _ij(R, r)
    I, J = i + HALO, j + HALO
    @inbounds begin
        su, sv = rain ? _sed_uv(Z, Cf, Jx, Jy, dz, nz, I, J, k) : (0.0, 0.0)    # sedimentation (ρq_r of I-1, J-1)
        if part == 0 || (part == 1) == (i > 1)
            u = a * (-(Cf[I, J, k, sx(2)] * _gradx(Z, Jc, Jx, zxx, follow, dx, dz, nz, I, J, k)) + su + F[I, J, k, VU])
            add ? (out[I, J, k, VU] += u) : (out[I, J, k, VU] = u)
        end
        if part == 0 || (part == 1) == (j > 1)
            v = a * (-(Cf[I, J, k, sy(2)] * _grady(Z, Jc, Jy, zyy, follow, dy, dz, nz, I, J, k)) + sv + F[I, J, k, VV])
            add ? (out[I, J, k, VV] += v) : (out[I, J, k, VV] = v)
        end
    end
end

"""out = D G_v(Z): vertical thermodynamic fluxes (rows × ct) and the vertical momentum part (W rows × cw)."""
@kernel function _k_vertical!(out, @Const(Z), @Const(Cf), Jc, dz, nz, grav, ct, cw, rain, R)
    r, k = @index(Global, NTuple)
    I, J = _IJ(R, r)
    @inbounds begin
        Fb = _az(nz, k) * _Wface(Z, nz, I, J, k); Ft = _az(nz, k + 1) * _Wface(Z, nz, I, J, k + 1)
        s = 1.0 / (Jc[I, J] * dz)
        vflux(bb, bt) = Cf[I, J, k+1, sz(bb)] * Ft - Cf[I, J, k, sz(bt)] * Fb
        out[I, J, k, VR] = -(s * (Ft - Fb)) * ct
        out[I, J, k, VX] = -(s * vflux(0, 6)) * ct
        out[I, J, k, VQV] = -(s * vflux(1, 1)) * ct
        out[I, J, k, VQC] = -(s * vflux(5, 5)) * ct
        out[I, J, k, VP] = -(s * vflux(3, 4)) * ct
        rain && (out[I, J, k, VQR] = -(s * (Cf[I, J, k+1, SRZ] * Ft - Cf[I, J, k, SRZ] * Fb)) * ct)
        if k >= 2
            out[I, J, k, VW] = -(Cf[I, J, k, sz(2)] * (Z[I, J, k, VP] - Z[I, J, k-1, VP]) / (Jc[I, J] * dz) +
                                 grav * 0.5 * (Z[I, J, k-1, VR] + Z[I, J, k, VR])) * cw
        end
    end
end

"""State update on the thermodynamic rows (k ≤ nz) and W (2 ≤ k ≤ nz) of the region: mode 1 Z += a X, mode 2
Z += a X + Y, mode 3 Z += a (X - P + K) + Y. The mode is a type (Val), so only the arrays it uses are read."""
@kernel function _k_update!(Z, @Const(X), @Const(Y), @Const(P), @Const(K), a, md, nz, rain, R)
    r, k = @index(Global, NTuple)
    I, J = _IJ(R, r)
    @inbounds begin
        for v in (VR, VX, VQV, VQC, VP, VW, VQR)
            if (v != VW || k >= 2) && (v != VQR || rain)
                if md isa Val{1}
                    Z[I, J, k, v] += a * X[I, J, k, v]
                elseif md isa Val{2}
                    Z[I, J, k, v] += a * X[I, J, k, v] + Y[I, J, k, v]
                else
                    Z[I, J, k, v] += a * (X[I, J, k, v] - P[I, J, k, v] + K[I, J, k, v]) + Y[I, J, k, v]
                end
            end
        end
    end
end

# ------------------------------------------------------------------ column solve (FV3D/moist.jl moist_column_solve!)
@kernel function _k_factor!(sub, cpr, iden, @Const(Cf), Jc, dz0, nz, grav, c, cw, R)
    r = @index(Global)
    I, J = _IJ(R, r)
    @inbounds begin
        dz = Jc[I, J] * dz0
        rr = c * cw / dz^2; gg = c * cw * grav / (2dz)
        for k in 2:nz
            P = Cf[I, J, k, sz(2)]
            a = k > 2 ? -rr * P * Cf[I, J, k-1, sz(4)] + gg : 0.0
            d = 1 + rr * P * (Cf[I, J, k, sz(4)] + Cf[I, J, k, sz(3)])
            b = k < nz ? -rr * P * Cf[I, J, k+1, sz(3)] - gg : 0.0
            den = k == 2 ? d : d - a * cpr[k-1, I, J]
            sub[k, I, J] = a; iden[k, I, J] = 1 / den; cpr[k, I, J] = b / den
        end
    end
end

"""Y = (I - D J_v)⁻¹ R on the thermodynamic and W rows of the region's columns."""
@kernel function _k_csolve!(Y, Wc, @Const(Rh), @Const(Cf), @Const(sub), @Const(cpr), @Const(iden), Jc, dz0, nz, grav, c, cw,
                            rain, R)
    r = @index(Global)
    I, J = _IJ(R, r)
    @inbounds begin
        dz = Jc[I, J] * dz0
        for k in 2:nz
            rhs = Rh[I, J, k, VW] - (cw / dz) * Cf[I, J, k, sz(2)] * (Rh[I, J, k, VP] - Rh[I, J, k-1, VP]) -
                  (cw * grav / 2) * (Rh[I, J, k-1, VR] + Rh[I, J, k, VR])
            Wc[I, J, k] = k == 2 ? rhs * iden[k, I, J] : (rhs - sub[k, I, J] * Wc[I, J, k-1]) * iden[k, I, J]
        end
        for k in nz-1:-1:2
            Wc[I, J, k] -= cpr[k, I, J] * Wc[I, J, k+1]
        end
        Wc[I, J, 1] = 0.0; Wc[I, J, nz+1] = 0.0
        rr = c / dz
        for k in 1:nz
            Wt, Wb = Wc[I, J, k+1], Wc[I, J, k]
            Y[I, J, k, VR] = Rh[I, J, k, VR] - rr * (Wt - Wb)
            Y[I, J, k, VX] = Rh[I, J, k, VX] - rr * (Cf[I, J, k+1, sz(0)] * Wt - Cf[I, J, k, sz(6)] * Wb)
            Y[I, J, k, VQV] = Rh[I, J, k, VQV] - rr * (Cf[I, J, k+1, sz(1)] * Wt - Cf[I, J, k, sz(1)] * Wb)
            Y[I, J, k, VQC] = Rh[I, J, k, VQC] - rr * (Cf[I, J, k+1, sz(5)] * Wt - Cf[I, J, k, sz(5)] * Wb)
            Y[I, J, k, VP] = Rh[I, J, k, VP] - rr * (Cf[I, J, k+1, sz(3)] * Wt - Cf[I, J, k, sz(4)] * Wb)
            rain && (Y[I, J, k, VQR] = Rh[I, J, k, VQR] - rr * (Cf[I, J, k+1, SRZ] * Wt - Cf[I, J, k, SRZ] * Wb))
        end
        for k in 2:nz
            Y[I, J, k, VW] = Wc[I, J, k]
        end
    end
end

# ------------------------------------------------------------------ launchers (asynchronous: no synchronization)
_rk(m, R) = (ncols(R), m.grid.nz)
"""out = :hthermo of Z on the region."""
function _hthermo!(out, m::MoistKA, Cf, Z, F, R::Region)
    g = m.grid
    _launch!(m, _k_hthermo!, _rk(m, R), out, Z, Cf, F, g.Jc, g.Jx, g.Jy, g.zxf, g.zyf, g.follow, g.dx, g.dy, g.dz,
             g.nz, m.mgas, m.pres, m.τ, m.micro == 1, m.damp, m.rain, R; wg = WG_CELL)
end
"""out (+)= a :hmomentum of Z on the faces `part` (0 all, 1 without halo, 2 with halo) of the region."""
_hmomentum!(out, m::MoistKA, Cf, Z, F, R::Region; a = 1.0, add::Bool = false, part::Int = 0) = (g = m.grid;
    _launch!(m, _k_hmomentum!, _rk(m, R), out, Z, Cf, F, g.Jc, g.Jx, g.Jy, g.zxx, g.zyy, g.follow, g.dx, g.dy, g.dz, g.nz,
             Float64(a), add, part, m.rain, R; wg = WG_CELL))
_vertical!(out, m::MoistKA, Cf, Z, ct, cw, R::Region) = (g = m.grid;
    _launch!(m, _k_vertical!, _rk(m, R), out, Z, Cf, g.Jc, g.dz, g.nz, m.grav, Float64(ct), Float64(cw), m.rain, R; wg = WG_CELL))
_phaseset!(out, m::MoistKA, Z, R::Region) = _launch!(m, _k_phaseset!, _rk(m, R), out, Z, m.mgas, m.pres, m.τ, m.rain, R; wg = WG_CELL)
"""Z += a X (mode 1), a X + Y (2), a (X - P + K) + Y (3) on the thermodynamic and W rows of the region."""
_update!(Z, m::MoistKA, R::Region, a, X, Y = X, P = X, K = X; mode::Int = Y === X ? 1 : P === X ? 2 : 3) =
    _launch!(m, _k_update!, _rk(m, R), Z, X, Y, P, K, Float64(a), Val(mode), m.grid.nz, m.rain, R; wg = WG_CELL)

"""Column factors of (I - D J_v), D = c (thermodynamic rows), cw (W rows): computed when c cw changes (once per stage
for :cs; the work arrays are allocated once)."""
function _factors(m::MoistKA, Cf, c, cw)
    g, w = m.grid, m.work
    key = Float64(c * cw)
    f = get!(w.fac, -1.0) do                                   # the arrays (key -1)
        be = KA.get_backend(m)
        ntuple(_ -> KA.zeros(be, Float64, g.nz + 1, g.nxh, g.nyh), 3)
    end
    if !haskey(w.fac, key)
        for kk in collect(keys(w.fac)); kk != -1.0 && delete!(w.fac, kk); end
        R = Region(g, :all)
        _launch!(m, _k_factor!, (ncols(R),), f..., Cf, g.Jc, g.dz, g.nz, m.grav, Float64(c), Float64(cw), R; wg = WG_COL)
        w.fac[key] = f
    end
    return f
end
"""Y = (I - D J_v)⁻¹ Rh on the thermodynamic and W rows of the region's columns."""
function _csolve!(Y, m::MoistKA, Cf, c, Rh, cw, R::Region)
    g = m.grid
    sub, cpr, iden = _factors(m, Cf, c, cw)
    _launch!(m, _k_csolve!, (ncols(R),), Y, m.work.Wc, Rh, Cf, sub, cpr, iden, g.Jc, g.dz, g.nz, m.grav, Float64(c), Float64(cw),
             m.rain, R; wg = WG_COL)
end

# ------------------------------------------------------------------ parts on all columns (tests, N(q))
"""Y = (I - D J_v)⁻¹ R on the interior columns (U, V rows and the halo: Y = R)."""
function ka_column_solve!(Y, m::MoistKA, Cf, c, R, cw = c)
    Y .= R
    _csolve!(Y, m, Cf, c, R, cw, Region(m.grid, :all))
    return Y
end
"""out = :hthermo of the state Z (coefficients Cf, forcing F); other rows zero."""
ka_hthermo!(out, m::MoistKA, Cf, Z, F) = (fill!(out, 0.0); _hthermo!(out, m, Cf, Z, F, Region(m.grid, :all)); out)
"""out = :hmomentum of Z; other rows zero."""
ka_hmomentum!(out, m::MoistKA, Cf, Z, F) = (fill!(out, 0.0); _hmomentum!(out, m, Cf, Z, F, Region(m.grid, :all)); out)
"""out = D G_v(Z), D = ct (thermodynamic rows), cw (W rows); other rows zero."""
ka_vertical!(out, m::MoistKA, Cf, Z, ct, cw) = (fill!(out, 0.0); _vertical!(out, m, Cf, Z, ct, cw, Region(m.grid, :all)); out)
"""out = phase change of Z (micro = :fast); false if there is none."""
function ka_phase!(out, m::MoistKA, Z)
    m.micro == 1 || return false
    fill!(out, 0.0)
    _phaseset!(out, m, Z, Region(m.grid, :all))
    return true
end

"""
    ka_rhs!(out, m, Q, Cf = ka_zeros(m, m.nc), S = ka_zeros(m))

N(q) = F(q, q) on the interior (halo of Q filled here): :hthermo + :hmomentum + the vertical part.
"""
function ka_rhs!(out, m::MoistKA, Q, Cf = ka_zeros(m, m.nc), S = ka_zeros(m), tmp = ka_zeros(m))
    ka_frozen_data!(Cf, S, m, Q)
    ka_hthermo!(out, m, Cf, Q, S)
    out .+= ka_hmomentum!(tmp, m, Cf, Q, S)
    out .+= ka_vertical!(tmp, m, Cf, Q, 1.0, 1.0)
    KA.synchronize(KA.get_backend(m))
    return out
end
