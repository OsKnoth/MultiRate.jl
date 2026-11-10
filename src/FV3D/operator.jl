# The operator in frozen form, N(q) = F(q, q):
#   fast (evolving z, coefficients from y), per unit fluid volume (invV = 1/Veff):
#     ρ_t = -invV [δx(Ax U) + δy(Ay V) + δz(Az W)],
#     Θ_t = -invV [δx(Ax θx(y) U) + δy(Ay θy(y) V) + δz(Az θz(y) W)],
#     U_t = -δx p,  V_t = -δy p,  W_t = -δz p - g ρ̄   (open faces only),  sponge on W,
#   slow (frozen y): momentum advection in flux form.
# θx, θy, θz: upwind face values of θ = Θ/ρ (third-order κ = 1/3 reconstruction; donor cell next
# to cut cells). Pressure (`FV3D_PRESSURE_FORM`): :gradp p(Θ_z) (nothing frozen) or :product
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
# [θx (N); θy (N); θz (Nz); Px (N); Py (N); Pz (Nz)], N = nx ny nz, Nz = nx ny (nz+1)

_ncoef(g::Grid3D) = 4 * ncell(g) + 2 * g.nx * g.ny * (g.nz + 1)
@inline _fz(g::Grid3D, i, j, k) = ip(g, i) + (jp(g, j) - 1) * g.nx + (k - 1) * g.nx * g.ny
@inline _Nz(g::Grid3D) = g.nx * g.ny * (g.nz + 1)
@inline cθx(g, i, j, k) = _c(g, i, j, k)
@inline cθy(g, i, j, k) = ncell(g) + _c(g, i, j, k)
@inline cθz(g, i, j, k) = 2 * ncell(g) + _fz(g, i, j, k)
@inline cPx(g, i, j, k) = 2 * ncell(g) + _Nz(g) + _c(g, i, j, k)
@inline cPy(g, i, j, k) = 3 * ncell(g) + _Nz(g) + _c(g, i, j, k)
@inline cPz(g, i, j, k) = 4 * ncell(g) + _Nz(g) + _fz(g, i, j, k)

frozen_ncoef(m::Model3D) = _ncoef(m.grid)

# ------------------------------------------------------------------ face values

"""Upwind face value between a1 (left/below) and a0 (right/above): third-order (κ = 1/3,
stencil a2, a1, a0, am) if `full`, donor cell otherwise."""
@inline upwind(a2, a1, a0, am, v, full) = full ? interp3(a2, a1, a0, am, v) : (signbit(v) ? a0 : a1)

@inline fullx(g, i, j, k, a, b) = all(g.full[ip(g, i + o), jp(g, j), k] for o in a:b)
@inline fully(g, i, j, k, a, b) = all(g.full[ip(g, i), jp(g, j + o), k] for o in a:b)
@inline fullz(g, i, j, k, a, b) = all(g.full[ip(g, i), jp(g, j), kk] for kk in max(1, k + a):min(g.nz, k + b))

"""Value at the z-face k (2 … nz) of a centred column array F: third order inside, centred mean
at the faces next to the walls (as the 2D `face_z`), donor cell next to cut cells."""
@inline function face_z3(g, F, i, j, k, v)
    fullz(g, i, j, k, -2, 1) || return signbit(v) ? F[i, j, k] : F[i, j, k-1]
    (k >= 3 && k <= g.nz - 1) && return interp3(F[i, j, k-2], F[i, j, k-1], F[i, j, k], F[i, j, k+1], v)
    return interp2(F[i, j, k-1], F[i, j, k])
end

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
        w[i, j, 1] = 0.0; w[i, j, nz+1] = 0.0
        for k in 2:nz
            w[i, j, k] = q[iW(g, i, j, k)] / (0.5 * (q[iρ(g, i, j, k - 1)] + q[iρ(g, i, j, k)]))
        end
    end
    fill!(C, 0.0)
    fill!(S, 0.0)
    P(a) = a > 0 ? 1 : 0
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        im, jm = ip(g, i - 1), jp(g, j - 1)
        # face values of θ (left x-face, left y-face, lower z-face)
        C[cθx(g, i, j, k)] = upwind(θ[ip(g, i - 2), j, k], θ[im, j, k], θ[i, j, k], θ[ip(g, i + 1), j, k],
                                    q[iU(g, i, j, k)], fullx(g, i, j, k, -2, 1))
        C[cθy(g, i, j, k)] = upwind(θ[i, jp(g, j - 2), k], θ[i, jm, k], θ[i, j, k], θ[i, jp(g, j + 1), k],
                                    q[iV(g, i, j, k)], fully(g, i, j, k, -2, 1))
        k > 1 && (C[cθz(g, i, j, k)] = face_z3(g, θ, i, j, k, q[iW(g, i, j, k)]))
        # ---- momentum advection (fluxes weighted by the open fractions; flat: weights 1)
        # U: x-flux at the centre of cell i, y-flux at the edge (i-1/2, j-1/2), z-flux at (i-1/2, k-1/2)
        Uc = 0.5 * (q[iU(g, i, j, k)] + q[iU(g, i + 1, j, k)])
        uc = upwind(u[im, j, k], u[i, j, k], u[ip(g, i + 1), j, k], u[ip(g, i + 2), j, k], Uc, fullx(g, i, j, k, -1, 1))
        f = g.Vf[i, j, k] * Uc * uc / dx
        S[iU(g, i + 1, j, k)] += f; S[iU(g, i, j, k)] -= f
        Vy = 0.5 * (q[iV(g, i - 1, j, k)] + q[iV(g, i, j, k)])
        uy = upwind(u[i, jp(g, j - 2), k], u[i, jm, k], u[i, j, k], u[i, jp(g, j + 1), k], Vy, fully(g, i, j, k, -2, 1) && fully(g, i - 1, j, k, -2, 1))
        f = 0.5 * (g.Ay[im, j, k] + g.Ay[i, j, k]) * Vy * uy / dy
        S[iU(g, i, j, k)] += f; S[iU(g, i, j - 1, k)] -= f
        # V: x-flux at the edge (i-1/2, j-1/2), y-flux at the centre of cell j, z-flux at (j-1/2, k-1/2)
        Ux = 0.5 * (q[iU(g, i, j - 1, k)] + q[iU(g, i, j, k)])
        vx = upwind(v[ip(g, i - 2), j, k], v[im, j, k], v[i, j, k], v[ip(g, i + 1), j, k], Ux, fullx(g, i, j, k, -2, 1) && fullx(g, i, j - 1, k, -2, 1))
        f = 0.5 * (g.Ax[i, jm, k] + g.Ax[i, j, k]) * Ux * vx / dx
        S[iV(g, i, j, k)] += f; S[iV(g, i - 1, j, k)] -= f
        Vc = 0.5 * (q[iV(g, i, j, k)] + q[iV(g, i, j + 1, k)])
        vc = upwind(v[i, jm, k], v[i, j, k], v[i, jp(g, j + 1), k], v[i, jp(g, j + 2), k], Vc, fully(g, i, j, k, -1, 1))
        f = g.Vf[i, j, k] * Vc * vc / dy
        S[iV(g, i, j + 1, k)] += f; S[iV(g, i, j, k)] -= f
        if k > 1
            Wk = 0.5 * (q[iW(g, i - 1, j, k)] + q[iW(g, i, j, k)])
            uz = face_z3(g, u, i, j, k, Wk)
            f = 0.5 * (g.Az[im, j, k] + g.Az[i, j, k]) * Wk * uz / dz
            S[iU(g, i, j, k)] += f; S[iU(g, i, j, k - 1)] -= f
            Wj = 0.5 * (q[iW(g, i, j - 1, k)] + q[iW(g, i, j, k)])
            vz = face_z3(g, v, i, j, k, Wj)
            f = 0.5 * (g.Az[i, jm, k] + g.Az[i, j, k]) * Wj * vz / dz
            S[iV(g, i, j, k)] += f; S[iV(g, i, j, k - 1)] -= f
            # W: x-flux at (i-1/2, k-1/2), y-flux at (j-1/2, k-1/2)
            Uk = 0.5 * (q[iU(g, i, j, k - 1)] + q[iU(g, i, j, k)])
            wx = upwind(w[ip(g, i - 2), j, k], w[im, j, k], w[i, j, k], w[ip(g, i + 1), j, k], Uk,
                        fullx(g, i, j, k, -2, 1) && fullx(g, i, j, k - 1, -2, 1))
            f = 0.5 * (g.Ax[i, j, k-1] + g.Ax[i, j, k]) * Uk * wx / dx
            S[iW(g, i, j, k)] += f; S[iW(g, i - 1, j, k)] -= f
            Vk = 0.5 * (q[iV(g, i, j, k - 1)] + q[iV(g, i, j, k)])
            wy = upwind(w[i, jp(g, j - 2), k], w[i, jm, k], w[i, j, k], w[i, jp(g, j + 1), k], Vk,
                        fully(g, i, j, k, -2, 1) && fully(g, i, j, k - 1, -2, 1))
            f = 0.5 * (g.Ay[i, j, k-1] + g.Ay[i, j, k]) * Vk * wy / dy
            S[iW(g, i, j, k)] += f; S[iW(g, i, j - 1, k)] -= f
        end
        # W: z-flux at the centre of cell k
        Wc = 0.5 * (Wface(g, q, i, j, k) + Wface(g, q, i, j, k + 1))
        wc = (k >= 2 && k <= nz - 1 && fullz(g, i, j, k, -1, 1)) ?
             interp3(w[i, j, k-1], w[i, j, k], w[i, j, k+1], w[i, j, k+2], Wc) :
             (fullz(g, i, j, k, 0, 0) ? interp2(w[i, j, k], w[i, j, k+1]) : (signbit(Wc) ? w[i, j, k+1] : w[i, j, k]))
        f = g.Vf[i, j, k] * Wc * wc / dz
        k + 1 <= nz && (S[iW(g, i, j, k + 1)] += f)
        k >= 2 && (S[iW(g, i, j, k)] -= f)
    end
    # divide by the fluid fraction of the momentum control volumes; closed faces get no tendency
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        S[iU(g, i, j, k)] *= P(g.Ax[i, j, k]) * 2 / (1 / max(g.invV[ip(g, i - 1), j, k], 1e-300) + 1 / max(g.invV[i, j, k], 1e-300))
        S[iV(g, i, j, k)] *= P(g.Ay[i, j, k]) * 2 / (1 / max(g.invV[i, jp(g, j - 1), k], 1e-300) + 1 / max(g.invV[i, j, k], 1e-300))
        k > 1 && (S[iW(g, i, j, k)] *= P(g.Az[i, j, k]) * 2 / (1 / max(g.invV[i, j, k-1], 1e-300) + 1 / max(g.invV[i, j, k], 1e-300)))
    end
    pressure_coefficients!(pressure_form(), C, m, q)
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

@inline dp_x(::GradP, m, C, z, i, j, k) = m.cache.pp[i, j, k] - m.cache.pp[ip(m.grid, i - 1), j, k]
@inline dp_y(::GradP, m, C, z, i, j, k) = m.cache.pp[i, j, k] - m.cache.pp[i, jp(m.grid, j - 1), k]
@inline dp_z(::GradP, m, C, z, i, j, k) = m.cache.pp[i, j, k] - m.cache.pp[i, j, k-1]
@inline dp_x(::Product, m, C, z, i, j, k) = (g = m.grid; C[cPx(g, i, j, k)] * (z[iΘ(g, i, j, k)] - z[iΘ(g, i - 1, j, k)]))
@inline dp_y(::Product, m, C, z, i, j, k) = (g = m.grid; C[cPy(g, i, j, k)] * (z[iΘ(g, i, j, k)] - z[iΘ(g, i, j - 1, k)]))
@inline dp_z(::Product, m, C, z, i, j, k) = (g = m.grid; C[cPz(g, i, j, k)] * (z[iΘ(g, i, j, k)] - z[iΘ(g, i, j, k - 1)]))

# ------------------------------------------------------------------ building blocks
# Each block adds its terms to `out` (rows named in the docstring) and nothing else.

"""Thermodynamic part, horizontal (ρ, Θ rows)."""
function thermo_h!(out, m::Model3D, C, z)
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        Fw = g.Ax[i, j, k] * z[iU(g, i, j, k)]; Fe = g.Ax[ip(g, i + 1), j, k] * z[iU(g, i + 1, j, k)]
        Fs = g.Ay[i, j, k] * z[iV(g, i, j, k)]; Fn = g.Ay[i, jp(g, j + 1), k] * z[iV(g, i, j + 1, k)]
        s = g.invV[i, j, k]
        out[iρ(g, i, j, k)] -= s * ((Fe - Fw) / g.dx + (Fn - Fs) / g.dy)
        out[iΘ(g, i, j, k)] -= s * ((C[cθx(g, i + 1, j, k)] * Fe - C[cθx(g, i, j, k)] * Fw) / g.dx +
                                    (C[cθy(g, i, j + 1, k)] * Fn - C[cθy(g, i, j, k)] * Fs) / g.dy)
    end
    return out
end

"""Thermodynamic part, vertical (ρ, Θ rows)."""
function thermo_v!(out, m::Model3D, C, z)
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        Fb = g.Az[i, j, k] * Wface(g, z, i, j, k); Ft = g.Az[i, j, k+1] * Wface(g, z, i, j, k + 1)
        s = g.invV[i, j, k]
        out[iρ(g, i, j, k)] -= s * (Ft - Fb) / g.dz
        out[iΘ(g, i, j, k)] -= s * (C[cθz(g, i, j, k + 1)] * Ft - C[cθz(g, i, j, k)] * Fb) / g.dz
    end
    return out
end

"""Momentum part, horizontal (U, V rows): -δx p, -δy p on the open faces."""
function momentum_h!(out, m::Model3D, C, z, pf)
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        out[iU(g, i, j, k)] -= (g.Ax[i, j, k] > 0) * dp_x(pf, m, C, z, i, j, k) / g.dx
        out[iV(g, i, j, k)] -= (g.Ay[i, j, k] > 0) * dp_y(pf, m, C, z, i, j, k) / g.dy
    end
    return out
end

"""Momentum part, vertical (W rows): -δz p - g ρ̄ on the open faces."""
function momentum_v!(out, m::Model3D, C, z, pf)
    g = m.grid
    grav = m.gas.g
    @inbounds for k in 2:g.nz, j in 1:g.ny, i in 1:g.nx
        out[iW(g, i, j, k)] -= (g.Az[i, j, k] > 0) *
            (dp_z(pf, m, C, z, i, j, k) / g.dz + grav * 0.5 * (z[iρ(g, i, j, k - 1)] + z[iρ(g, i, j, k)]))
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
# :hthermo    thermo h, forcing ρ, Θ and W, sponge                 (HEVI: horizontal, with all
# :hmomentum  momentum h, forcing U, V                              forcing; the vertical part
# :vertical   thermo v, momentum v                                  has none)

_fast!(::Val{:all}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z); thermo_v!(out, m, C, z);
    prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf); momentum_v!(out, m, C, z, pf);
    forcing_thermo!(out, m, F); forcing_UV!(out, m, F); forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:thermo}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z); thermo_v!(out, m, C, z); forcing_thermo!(out, m, F))
_fast!(::Val{:momentum}, out, m, C, z, F, pf) = (prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf);
    momentum_v!(out, m, C, z, pf); forcing_UV!(out, m, F); forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:hthermo}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z); forcing_thermo!(out, m, F);
    forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:hmomentum}, out, m, C, z, F, pf) = (prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf); forcing_UV!(out, m, F))
_fast!(::Val{:vertical}, out, m, C, z, F, pf) = (thermo_v!(out, m, C, z); prepare_pressure!(pf, m, z);
    momentum_v!(out, m, C, z, pf))
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
