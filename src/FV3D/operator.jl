# The operator in frozen form, N(q) = F(q, q):
#   fast (evolving z, coefficients from y), per unit fluid volume (s = invV/J, invV = 1/Vf):
#     ρ_t = -s [δx(Ax Jx U) + δy(Ay Jy V) + δη(Az Ω)],      Ω = W - z_x Ū - z_y V̄,
#     Θ_t = -s [δx(Ax Jx θx(y) U) + δy(Ay Jy θy(y) V) + δη(Az θη(y) Ω)],
#     U_t = -(1/Jx)[δx(J p) - δη(z_x p)],  V_t likewise,  W_t = -δη p / J - g ρ̄   (open faces),
#     sponge on W; the W part of Ω is vertical (implicit in HEVI), its Ū, V̄ part horizontal.
#   Cartesian grids (flat, cut cells): J = 1, z_x = z_y = 0; terrain following (Gal-Chen): Ω is the
#   contravariant vertical mass flux, Ω = 0 at the ground and the top.
#   slow (frozen y): momentum advection in flux form.
# θx, θy, θz: upwind face values of θ = Θ/ρ, pseudo third order with virtual grid sizes from the
# volumes and face areas (advection.jl; κ = 1/3 on uniform grids). Pressure (`FV3D_PRESSURE_FORM`): :gradp p(Θ_z) (nothing frozen) or :product
# (dp/dΘ)_f(y) (Θ_r - Θ_l), the face mean of the analytic γ p/Θ frozen (fast part linear in z).
# No background state is subtracted.

"""Pressure form of the 3D FV operator (read in every call): `:gradp` or `:product`."""
const FV3D_PRESSURE_FORM = Ref(:gradp)

struct GradP end
struct Product end
pressure_form() = FV3D_PRESSURE_FORM[] === :product ? Product() :
                  (FV3D_PRESSURE_FORM[] === :gradp ? GradP() : error("FV3D_PRESSURE_FORM: :gradp or :product"))

"""dp/dΘ at a face: mean of the cell values γ p/Θ."""
@inline dpdΘ_face(gas, Θl, Θr) = 0.5 * gas.γ * (pressure(gas, Θl) / Θl + pressure(gas, Θr) / Θr)

# ------------------------------------------------------------------ coefficient vector C
# [θx (N); θy (N); θz (Nz); Px (N); Py (N); Pz (Nz); a⁺x, a⁻x, a⁺y, a⁻y (N each); a⁺z, a⁻z (Nz each)],
# N = nx ny nz, Nz = nx ny (nz+1); a± the upwind coefficients of the masked advection (implicit = :amf)

_ncoef(g::Grid3D) = 8 * ncell(g) + 4 * g.nx * g.ny * (g.nz + 1)
@inline _fz(g::Grid3D, i, j, k) = ip(g, i) + (jp(g, j) - 1) * g.nx + (k - 1) * g.nx * g.ny
@inline _Nz(g::Grid3D) = g.nx * g.ny * (g.nz + 1)
@inline cθx(g, i, j, k) = _c(g, i, j, k)
@inline cθy(g, i, j, k) = ncell(g) + _c(g, i, j, k)
@inline cθz(g, i, j, k) = 2 * ncell(g) + _fz(g, i, j, k)
@inline cPx(g, i, j, k) = 2 * ncell(g) + _Nz(g) + _c(g, i, j, k)
@inline cPy(g, i, j, k) = 3 * ncell(g) + _Nz(g) + _c(g, i, j, k)
@inline cPz(g, i, j, k) = 4 * ncell(g) + _Nz(g) + _fz(g, i, j, k)
@inline _ca0(g) = 4 * ncell(g) + 2 * _Nz(g)
@inline caxp(g, i, j, k) = _ca0(g) + _c(g, i, j, k)
@inline caxm(g, i, j, k) = _ca0(g) + ncell(g) + _c(g, i, j, k)
@inline cayp(g, i, j, k) = _ca0(g) + 2 * ncell(g) + _c(g, i, j, k)
@inline caym(g, i, j, k) = _ca0(g) + 3 * ncell(g) + _c(g, i, j, k)
@inline cazp(g, i, j, k) = _ca0(g) + 4 * ncell(g) + _fz(g, i, j, k)
@inline cazm(g, i, j, k) = _ca0(g) + 4 * ncell(g) + _Nz(g) + _fz(g, i, j, k)

frozen_ncoef(m::Model3D) = _ncoef(m.grid)

# ------------------------------------------------------------------ contravariant vertical flux

"""Horizontal momenta averaged to the η-face k (2 … nz) of column (i, j)."""
@inline Ubar(g, q, i, j, k) = 0.25 * (q[iU(g, i, j, k - 1)] + q[iU(g, i + 1, j, k - 1)] + q[iU(g, i, j, k)] + q[iU(g, i + 1, j, k)])
@inline Vbar(g, q, i, j, k) = 0.25 * (q[iV(g, i, j, k - 1)] + q[iV(g, i, j + 1, k - 1)] + q[iV(g, i, j, k)] + q[iV(g, i, j + 1, k)])
"""Horizontal part -z_x Ū - z_y V̄ of Ω at the η-face k (zero at the ground and the top, and on
Cartesian grids)."""
@inline Ωh(g, q, i, j, k) = (k == 1 || k == g.nz + 1 || !g.follow) ? 0.0 :
    -(g.zxf[ip(g, i), jp(g, j), k] * Ubar(g, q, i, j, k) + g.zyf[ip(g, i), jp(g, j), k] * Vbar(g, q, i, j, k))
"""Ω = W - z_x Ū - z_y V̄ at the η-face k of column (i, j)."""
@inline Ωface(g, q, i, j, k) = Wface(g, q, i, j, k) + Ωh(g, q, i, j, k)

# ------------------------------------------------------------------ C(y) and S(y)

"""C(y) (face θ, face dp/dΘ) and the slow part S(y) (momentum advection)."""
function frozen_data!(C, S, m::Model3D, q)
    g = m.grid
    nx, ny, nz, dx, dy, dz = g.nx, g.ny, g.nz, g.dx, g.dy, g.dz
    c = m.cache
    u, v, w, θ = c.u, c.v, c.w, c.θ
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ρ = q[iρ(g, i, j, k)]
        θ[i, j, k] = q[iΘ(g, i, j, k)] / ρ
        u[i, j, k] = q[iU(g, i, j, k)] / (0.5 * (q[iρ(g, i - 1, j, k)] + ρ))
        v[i, j, k] = q[iV(g, i, j, k)] / (0.5 * (q[iρ(g, i, j - 1, k)] + ρ))
    end
    @inbounds for j in 1:ny, i in 1:nx
        # ground: w = z_x u + z_y v (terrain following; 0 on Cartesian grids)
        w[i, j, 1] = g.follow ? g.zxf[i, j, 1] * 0.5 * (u[i, j, 1] + u[ip(g, i + 1), j, 1]) +
                                g.zyf[i, j, 1] * 0.5 * (v[i, j, 1] + v[i, jp(g, j + 1), 1]) : 0.0
        w[i, j, nz+1] = 0.0
        for k in 2:nz
            w[i, j, k] = q[iW(g, i, j, k)] / (0.5 * (q[iρ(g, i, j, k - 1)] + q[iρ(g, i, j, k)]))
        end
    end
    fill!(C, 0.0)
    fill!(S, 0.0)
    # θ at the open faces: pseudo third-order upwind (advection.jl), by the sign of the mass flux
    mass_fluxes!(m, q)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ax(g, i, j, k) > 0 && (C[cθx(g, i, j, k)] = face_x(g, θ, i, j, k, c.Fx[i, j, k]))
        ay(g, i, j, k) > 0 && (C[cθy(g, i, j, k)] = face_y(g, θ, i, j, k, c.Fy[i, j, k]))
        k > 1 && az(g, i, j, k) > 0 && (C[cθz(g, i, j, k)] = face_z(g, θ, i, j, k, c.Fz[i, j, k]))
    end
    # momentum advection: shift technique (advection.jl)
    momentum_advection!(S, m)
    pressure_coefficients!(pressure_form(), C, m, q)
    # implicit = :amf: the masked momentum advection A1(y) z is fast (advection_jacobian.jl), the
    # slow part keeps S(y) - A1(y) y
    upwind_coefficients!(C, m, q)
    if m.implicit === :amf
        t = m.cache.tmp; fill!(t, 0.0)
        adv_masked!(t, m, C, q)
        S .-= t
    end
    return C, S
end

"""Frozen face derivatives (dp/dΘ)_f(y) for Product; nothing for GradP."""
function pressure_coefficients!(::Product, C, m::Model3D, q)
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        Θc = q[iΘ(g, i, j, k)]
        C[cPx(g, i, j, k)] = dpdΘ_face(m.gas, q[iΘ(g, i - 1, j, k)], Θc)
        C[cPy(g, i, j, k)] = dpdΘ_face(m.gas, q[iΘ(g, i, j - 1, k)], Θc)
        k > 1 && (C[cPz(g, i, j, k)] = dpdΘ_face(m.gas, q[iΘ(g, i, j, k - 1)], Θc))
    end
    return C
end
pressure_coefficients!(::GradP, C, m::Model3D, q) = C

# ------------------------------------------------------------------ pressure differences

"""p(Θ_z) at the centres (GradP); nothing for Product."""
function prepare_pressure!(::GradP, m::Model3D, z)
    g = m.grid
    pp = m.cache.pp
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        pp[i, j, k] = pressure(m.gas, z[iΘ(g, i, j, k)])
    end
    return nothing
end
prepare_pressure!(::Product, m::Model3D, z) = nothing

"""The field whose differences give the pressure gradient: p (GradP) or Θ (Product, times the
frozen face derivative)."""
@inline Πc(::GradP, m, z, i, j, k) = m.cache.pp[ip(m.grid, i), jp(m.grid, j), k]
@inline Πc(::Product, m, z, i, j, k) = z[iΘ(m.grid, i, j, k)]
@inline Pfx(::GradP, m, C, i, j, k) = 1.0
@inline Pfy(::GradP, m, C, i, j, k) = 1.0
@inline Pfz(::GradP, m, C, i, j, k) = 1.0
@inline Pfx(::Product, m, C, i, j, k) = C[cPx(m.grid, i, j, k)]
@inline Pfy(::Product, m, C, i, j, k) = C[cPy(m.grid, i, j, k)]
@inline Pfz(::Product, m, C, i, j, k) = C[cPz(m.grid, i, j, k)]

"""Π at the η-face kf of column (i, j): mean of the two cells; at the ground and the top the face
means are extrapolated quadratically (as `fv_p_sigma` in 2D), so that the metric term -δη(z_x p)
has the truncation error of the interior means."""
@inline function Πσ(pf, m, z, i, j, kf)
    g = m.grid
    mf(k) = 0.5 * (Πc(pf, m, z, i, j, k - 1) + Πc(pf, m, z, i, j, k))
    kf == 1 && return 3mf(2) - 3mf(3) + mf(4)
    kf == g.nz + 1 && return 3mf(g.nz) - 3mf(g.nz - 1) + mf(g.nz - 2)
    return mf(kf)
end

"""-∂p/∂x|_z at the x-face (i - 1/2, j, k): (1/Jx)[δx(J Π) - δη(z_x Π)] times the face factor
(Cartesian: δx Π)."""
@inline function gradx(pf, m, C, z, i, j, k)
    g = m.grid
    im = ip(g, i - 1)
    d = (g.Jc[ip(g, i), jp(g, j)] * Πc(pf, m, z, i, j, k) - g.Jc[im, jp(g, j)] * Πc(pf, m, z, im, j, k)) / g.dx
    if g.follow
        cor(kf) = 0.5 * (Πσ(pf, m, z, im, j, kf) + Πσ(pf, m, z, i, j, kf))
        d -= (g.zxx[ip(g, i), jp(g, j), k+1] * cor(k + 1) - g.zxx[ip(g, i), jp(g, j), k] * cor(k)) / g.dz
    end
    return Pfx(pf, m, C, i, j, k) * d / g.Jx[ip(g, i), jp(g, j)]
end
@inline function grady(pf, m, C, z, i, j, k)
    g = m.grid
    jm = jp(g, j - 1)
    d = (g.Jc[ip(g, i), jp(g, j)] * Πc(pf, m, z, i, j, k) - g.Jc[ip(g, i), jm] * Πc(pf, m, z, i, jm, k)) / g.dy
    if g.follow
        cor(kf) = 0.5 * (Πσ(pf, m, z, i, jm, kf) + Πσ(pf, m, z, i, j, kf))
        d -= (g.zyy[ip(g, i), jp(g, j), k+1] * cor(k + 1) - g.zyy[ip(g, i), jp(g, j), k] * cor(k)) / g.dz
    end
    return Pfy(pf, m, C, i, j, k) * d / g.Jy[ip(g, i), jp(g, j)]
end
"""∂p/∂z at the η-face k (2 … nz): δη Π / J times the face factor."""
@inline gradz(pf, m, C, z, i, j, k) =
    Pfz(pf, m, C, i, j, k) * (Πc(pf, m, z, i, j, k) - Πc(pf, m, z, i, j, k - 1)) / (m.grid.Jc[ip(m.grid, i), jp(m.grid, j)] * m.grid.dz)

# ------------------------------------------------------------------ face selections
# The horizontal blocks act on all faces, on the unmasked faces only (explicit half steps of HEVI
# with implicit = :mask) or on the masked faces only (added to the implicit step).
struct AllFaces end
struct Unmasked end
struct Masked end
@inline fw_x(::AllFaces, g, i, j, k) = 1.0
@inline fw_y(::AllFaces, g, i, j, k) = 1.0
@inline fw_x(::Unmasked, g, i, j, k) = 1.0 - g.Mx[ip(g, i), jp(g, j), k]
@inline fw_y(::Unmasked, g, i, j, k) = 1.0 - g.My[ip(g, i), jp(g, j), k]
@inline fw_x(::Masked, g, i, j, k) = g.Mx[ip(g, i), jp(g, j), k]
@inline fw_y(::Masked, g, i, j, k) = g.My[ip(g, i), jp(g, j), k]
"""Does the selection carry the horizontal part of Ω (terrain following; once, not in Masked)?"""
@inline has_Ωh(::Masked) = false
@inline has_Ωh(_) = true

# ------------------------------------------------------------------ building blocks
# Each block adds its terms to `out` (rows named in the docstring) and nothing else.

"""Thermodynamic part, horizontal (ρ, Θ rows), through the faces of the selection `sel`."""
function thermo_h!(out, m::Model3D, C, z, sel = AllFaces())
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        ie, jn = ip(g, i + 1), jp(g, j + 1)
        Fw = fw_x(sel, g, i, j, k) * g.Ax[i, j, k] * g.Jx[i, j] * z[iU(g, i, j, k)]
        Fe = fw_x(sel, g, i + 1, j, k) * g.Ax[ie, j, k] * g.Jx[ie, j] * z[iU(g, i + 1, j, k)]
        Fs = fw_y(sel, g, i, j, k) * g.Ay[i, j, k] * g.Jy[i, j] * z[iV(g, i, j, k)]
        Fn = fw_y(sel, g, i, j + 1, k) * g.Ay[i, jn, k] * g.Jy[i, jn] * z[iV(g, i, j + 1, k)]
        s = g.invV[i, j, k] / g.Jc[i, j]
        out[iρ(g, i, j, k)] -= s * ((Fe - Fw) / g.dx + (Fn - Fs) / g.dy)
        out[iΘ(g, i, j, k)] -= s * ((C[cθx(g, i + 1, j, k)] * Fe - C[cθx(g, i, j, k)] * Fw) / g.dx +
                                    (C[cθy(g, i, j + 1, k)] * Fn - C[cθy(g, i, j, k)] * Fs) / g.dy)
        if g.follow && has_Ωh(sel)               # horizontal part of the vertical flux Ω
            Hb = g.Az[i, j, k] * Ωh(g, z, i, j, k); Ht = g.Az[i, j, k+1] * Ωh(g, z, i, j, k + 1)
            out[iρ(g, i, j, k)] -= s * (Ht - Hb) / g.dz
            out[iΘ(g, i, j, k)] -= s * (C[cθz(g, i, j, k + 1)] * Ht - C[cθz(g, i, j, k)] * Hb) / g.dz
        end
    end
    return out
end

"""Thermodynamic part, vertical (ρ, Θ rows)."""
function thermo_v!(out, m::Model3D, C, z)
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        Fb = g.Az[i, j, k] * Wface(g, z, i, j, k); Ft = g.Az[i, j, k+1] * Wface(g, z, i, j, k + 1)
        s = g.invV[i, j, k] / g.Jc[i, j]
        out[iρ(g, i, j, k)] -= s * (Ft - Fb) / g.dz
        out[iΘ(g, i, j, k)] -= s * (C[cθz(g, i, j, k + 1)] * Ft - C[cθz(g, i, j, k)] * Fb) / g.dz
    end
    return out
end

"""Momentum part, horizontal (U, V rows): -∂x p|_z, -∂y p|_z on the open faces of the selection `sel`."""
function momentum_h!(out, m::Model3D, C, z, pf, sel = AllFaces())
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        out[iU(g, i, j, k)] -= fw_x(sel, g, i, j, k) * (g.Ax[i, j, k] > 0) * gradx(pf, m, C, z, i, j, k)
        out[iV(g, i, j, k)] -= fw_y(sel, g, i, j, k) * (g.Ay[i, j, k] > 0) * grady(pf, m, C, z, i, j, k)
    end
    return out
end

"""Momentum part, vertical (W rows): -δη p / J - g ρ̄ on the open faces."""
function momentum_v!(out, m::Model3D, C, z, pf)
    g = m.grid
    grav = m.gas.g
    @inbounds for k in 2:g.nz, j in 1:g.ny, i in 1:g.nx
        out[iW(g, i, j, k)] -= (g.Az[i, j, k] > 0) *
            (gradz(pf, m, C, z, i, j, k) + grav * 0.5 * (z[iρ(g, i, j, k - 1)] + z[iρ(g, i, j, k)]))
    end
    return out
end

"""Constant forcing F of the stage on the ρ, Θ / U, V / W rows."""
function forcing_thermo!(out, m::Model3D, F)
    g = m.grid
    @inbounds for n in 1:ncell(g)
        out[n] += F[n]                                            # ρ
        out[nunknowns(g) - ncell(g) + n] += F[nunknowns(g) - ncell(g) + n]   # Θ
    end
    return out
end
function forcing_UV!(out, m::Model3D, F)
    g = m.grid
    @inbounds for n in ncell(g)+1:3*ncell(g)
        out[n] += F[n]
    end
    return out
end
function forcing_W!(out, m::Model3D, F)
    g = m.grid
    @inbounds for n in 3*ncell(g)+1:3*ncell(g)+g.nx*g.ny*(g.nz-1)
        out[n] += F[n]
    end
    return out
end

# ------------------------------------------------------------------ the fast part as compositions
# part        blocks
# :all        thermo (h + v), momentum (h + v), forcing on all rows, sponge
# :thermo     thermo (h + v), forcing ρ, Θ                         (Verlet: thermodynamic step)
# :momentum   momentum (h + v), forcing U, V, W, sponge            (Verlet: momentum step)
# :hthermo    thermo h (explicit faces), forcing ρ, Θ and W, sponge (HEVI: explicit horizontal
# :hmomentum  momentum h (explicit faces), forcing U, V              half steps, with all forcing)
# :vertical   thermo v, momentum v, + thermo h, momentum h on the masked faces (implicit = :mask, :amf)
#             + the masked momentum advection A1 z (implicit = :amf, also in :all and :momentum)
# explicit faces: all (implicit = :column) or the unmasked ones (implicit = :mask, :amf).

_hsel(m::Model3D) = m.implicit in (:mask, :amf) ? Unmasked() : AllFaces()
"""Masked momentum advection A1(C) z of the fast part (implicit = :amf; else nothing)."""
_adv!(out, m, C, z) = (m.implicit === :amf && adv_masked!(out, m, C, z); out)

_fast!(::Val{:all}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z); thermo_v!(out, m, C, z);
    prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf); momentum_v!(out, m, C, z, pf); _adv!(out, m, C, z);
    forcing_thermo!(out, m, F); forcing_UV!(out, m, F); forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:thermo}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z); thermo_v!(out, m, C, z); forcing_thermo!(out, m, F))
_fast!(::Val{:momentum}, out, m, C, z, F, pf) = (prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf);
    momentum_v!(out, m, C, z, pf); _adv!(out, m, C, z); forcing_UV!(out, m, F); forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:hthermo}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z, _hsel(m)); forcing_thermo!(out, m, F);
    forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:hmomentum}, out, m, C, z, F, pf) = (prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf, _hsel(m));
    forcing_UV!(out, m, F))
function _fast!(::Val{:vertical}, out, m, C, z, F, pf)
    thermo_v!(out, m, C, z); prepare_pressure!(pf, m, z); momentum_v!(out, m, C, z, pf)
    if m.implicit in (:mask, :amf)
        thermo_h!(out, m, C, z, Masked()); momentum_h!(out, m, C, z, pf, Masked())
    end
    _adv!(out, m, C, z)
    return out
end
_fast!(::Val{P}, out, m, C, z, F, pf) where {P} = error("frozen_fast!: unknown part $P")

"""Fast part F_f(C, z) + F (and the sponge) on the rows of `part` (table above)."""
function frozen_fast!(out, m::Model3D, C, z, part::Symbol, F)
    fill!(out, 0.0)
    _fast!(Val(part), out, m, C, z, F, pressure_form())
    return out
end

"""N(q) = F(q, q): the full operator from the same building blocks."""
function rhs_nonlinear!(out, m::Model3D, q::AbstractVector)
    c = m.cache
    frozen_data!(c.C, c.S, m, q)
    return frozen_fast!(out, m, c.C, q, :all, c.S)
end
