# Nonlinear operator N (flux form, C-grid)

"""θ at the z-face k of column i (3rd order where the stencil fits, 2nd order next to the walls)."""
@inline function face_z(F::AbstractMatrix, i, k, nz, v)
    if k >= 3 && k <= nz - 1
        return interp3(F[i, k-2], F[i, k-1], F[i, k], F[i, k+1], v)
    else
        return interp2(F[i, k-1], F[i, k])
    end
end

"""
Pressure form of the flat FV operator N (read in every call):
- `:gradp` (default): -(p_r - p_l)/Δ at the faces, p = P(Θ) in the cells;
- `:product`: -(dp/dΘ)_f (Θ_r - Θ_l)/Δ, with (dp/dΘ)_f the mean of the cell values γ p/Θ;
  full p and ρ, no background state subtracted.
The frozen split (`frozen.jl`) follows: `:gradp` keeps P(Θ_z) in the fast part, `:product` freezes
(dp/dΘ)_f, so its fast part is linear in z. Flat grids only.
"""
const FV_PRESSURE_FORM = Ref(:gradp)

"""The :product scheme subtracts no background state: models built with FV_HYDRO_REFERENCE are rejected."""
_no_background(m) = (all(iszero, m.pbal) && all(iszero, m.ρbal)) ||
    error("FV_PRESSURE_FORM = :product: no background subtraction (build the model without FV_HYDRO_REFERENCE)")

"""dp/dΘ at a face: mean of the cell values γ p/Θ."""
@inline dpdΘ_face(gas, Θl, Θr) = 0.5 * gas.γ * (pressure(gas, Θl) / Θl + pressure(gas, Θr) / Θr)

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
    product = FV_PRESSURE_FORM[] === :product
    product || FV_PRESSURE_FORM[] === :gradp || error("FV_PRESSURE_FORM: :gradp or :product")
    product && _no_background(m)
    @inbounds for k in 1:nz, i in 1:nx
        ρ = q[iρ(g, i, k)]
        θ[i, k] = q[iΘ(g, i, k)] / ρ
        pp[i, k] = pressure(m.gas, q[iΘ(g, i, k)]) - m.pbal[i, k]
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
    a = shift_advection!(c.adv, g, q)          # θ face values and momentum advection (shift_advection.jl)
    @inbounds for k in 1:nz, i in 1:nx
        # mass and Θ: fluxes through the left x-face (i) and the lower z-face (k) of cell (i, k)
        U = q[iU(g, i, k)]
        θx = a.θfx[i, k]
        out[iρ(g, i, k)] += U / dx
        out[iρ(g, i - 1, k)] -= U / dx
        out[iΘ(g, i, k)] += U * θx / dx
        out[iΘ(g, i - 1, k)] -= U * θx / dx
        if k > 1
            W = q[iW(g, i, k)]
            θz = a.θfz[i, k]
            out[iρ(g, i, k)] += W / dz
            out[iρ(g, i, k - 1)] -= W / dz
            out[iΘ(g, i, k)] += W * θz / dz
            out[iΘ(g, i, k - 1)] -= W * θz / dz
        end
        # momentum advection
        out[iU(g, i, k)] += a.SU[i, k]
        k > 1 && (out[iW(g, i, k)] += a.SW[i, k])
        # pressure gradient (:gradp or :product) and gravity (deviations from the reference)
        il = ip(g, i - 1)
        if product
            Θc = q[iΘ(g, i, k)]
            Θl = q[iΘ(g, il, k)]
            out[iU(g, i, k)] -= dpdΘ_face(m.gas, Θl, Θc) * (Θc - Θl) / dx
        else
            out[iU(g, i, k)] -= (pp[i, k] - pp[il, k]) / dx
        end
        if k > 1
            ρw = 0.5 * (q[iρ(g, i, k - 1)] - m.ρbal[i, k-1] + q[iρ(g, i, k)] - m.ρbal[i, k])
            if product
                Θc = q[iΘ(g, i, k)]
                Θb = q[iΘ(g, i, k - 1)]
                dpz = dpdΘ_face(m.gas, Θb, Θc) * (Θc - Θb)
            else
                dpz = pp[i, k] - pp[i, k-1]
            end
            out[iW(g, i, k)] -= dpz / dz + grav * ρw
        end
    end
    sponge!(out, m, q)
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

"""
Form of the Θ equation and the horizontal pressure gradient on the Gal-Chen grid:
- `:flux` (default): conservative Θ flux of the full θ (third-order upwind along σ, mean at
  σ-faces), pressure gradient in flux form with the σ correction. Background-free and
  Θ-conservative. Its linearization at rest on terrain has a weak growing mode (U-dominated,
  ∝ N² h): the x-flux carries θ along the sloped σ-surfaces, while the σ correction -z_x U uses
  the four-point average of U, so for grid-scale U the two parts of ∂_xθ|_z = 0 do not cancel.
  Re λ ≈ 5e-6 s⁻¹ on a 160 km channel with a 20 km, 400 m hill (8 × 32 cells), growing with the
  horizontal resolution. Agnesi a100 (2.5 days) is not affected: max|w| = 0.055 m/s.
- `:advective`: Θ_t = θ ρ_t - [ρu (∂_ξθ - (z_x/J) ∂_ηθ) + ρw (1/J) ∂_ηθ] with the same
  cell-centred ρu in both horizontal parts, and -(1/θ_x)[G(θ p') - p' ∂_xθ|_z] for the pressure
  gradient (G: chain-rule gradient adjoint to the σ-corrected mass flux, adjoint to the ρu ∂_xθ|_z
  term). Background-free, matches the fast operator `fv_fast_terrain!`; the mode decays with the
  resolution (1.3e-6, 2.4e-7, 1e-8, 1.3e-9 s⁻¹ for 8, 16, 32, 64 columns). Mass is conservative,
  Θ is not.
- `:background`: as `:advective` but with the prescribed θ̄(z) (θ' reconstruction,
  θ̄ ρ_t - ρw dθ̄/dz); the mode vanishes exactly.
"""
const FV_TERRAIN_FORM = Ref(:flux)

"""p at σ-face kf of column i: mean of the neighbours; at the ground and the top the face means
are extrapolated quadratically (3p̄₃⸝₂ - 3p̄₅⸝₂ + p̄₇⸝₂). The boundary value then has the same
O(Δη²) error as the interior means, which cancels in -∂_η(z_x p)/Δη: second order at k = 1
(a pointwise third-order extrapolation leaves an O(Δη) residual of the full hydrostatic p)."""
@inline fv_p_sigma(p, i, kf, nz) =
    kf == 1 ? 1.5p[i, 1] - p[i, 3] + 0.5p[i, 4] :
    kf == nz+1 ? 1.5p[i, nz] - p[i, nz-2] + 0.5p[i, nz-3] :
    0.5 * (p[i, kf-1] + p[i, kf])

function rhs_terrain!(out, m::FVModel, q)
    g, c = m.grid, m.cache
    nx, nz, dx, dη = g.nx, g.nz, g.dx, g.dz
    form = FV_TERRAIN_FORM[]
    form in (:flux, :advective, :background) || throw(ArgumentError("FV_TERRAIN_FORM = $form"))
    adv = form === :advective
    pert = (form === :background && length(c.cc) >= 3) || adv
    θbc, θbσ = pert && !adv ? (c.cc[1], c.cc[3]) : (c.θ, c.θ)
    # reconstruction variable of the Θ flux: θ' (work array c.mx) or θ
    θr = pert && !adv ? c.mx : c.θ
    @inbounds for k in 1:nz, i in 1:nx
        ρ = q[iρ(g, i, k)]
        c.θ[i, k] = q[iΘ(g, i, k)] / ρ
        c.pp[i, k] = pressure(m.gas, q[iΘ(g, i, k)]) - m.pbal[i, k]
        c.u[i, k] = q[iU(g, i, k)] / (0.5 * (q[iρ(g, i-1, k)] + ρ))
        pert && !adv && (θr[i, k] = c.θ[i, k] - θbc[i, k])
    end
    rc = c.fz             # :advective: ∂_xθ|_z = ∂_ξθ - (z_x/J) ∂_ηθ at the centres
    if adv
        @inbounds for k in 1:nz, i in 1:nx
            ir = ip(g, i+1)
            θx0 = interp3(c.θ[ip(g, i-2), k], c.θ[ip(g, i-1), k], c.θ[i, k], c.θ[ir, k], q[iU(g, i, k)])
            θx1 = interp3(c.θ[ip(g, i-1), k], c.θ[i, k], c.θ[ir, k], c.θ[ip(g, i+2), k], q[iU(g, i+1, k)])
            θb = k == 1 ? 1.5 * c.θ[i, 1] - 0.5 * c.θ[i, 2] : interp2(c.θ[i, k-1], c.θ[i, k])
            θt = k == nz ? 1.5 * c.θ[i, nz] - 0.5 * c.θ[i, nz-1] : interp2(c.θ[i, k], c.θ[i, k+1])
            rc[i, k] = (θx1 - θx0) / dx - g.zx_c[i, k] * (θt - θb) / (g.Jc[i] * dη)
        end
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
    a = shift_advection!(c.adv, g, q)          # θ face values (:flux) and momentum advection
    newθ = form === :flux
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
            θσ = kf == 1 ? θr[i, 1] : kf == nz+1 ? θr[i, nz] :
                  newθ ? a.θfz[i, kf] : interp2(θr[i, kf-1], θr[i, kf])
            out[iΘ(g, i, k)] -= sg * θσ * Vσ / (J * dη)
        end
        out[iρ(g, i, k)] -= (Fx1 - Fx0) / (J * dx)
        θx0 = newθ ? a.θfx[i, k] : interp3(θr[ip(g, i-2), k], θr[ip(g, i-1), k], θr[i, k], θr[ip(g, i+1), k], U0)
        θx1 = newθ ? a.θfx[ir, k] : interp3(θr[ip(g, i-1), k], θr[i, k], θr[ir, k], θr[ip(g, i+2), k], U1)
        out[iΘ(g, i, k)] -= (g.Jx[ir] * θx1 * U1 - g.Jx[i] * θx0 * U0) / (J * dx)
        if adv    # advective form: Θ_t = θ ρ_t - [ρu ∂_ξθ + ρω̃ ∂_ηθ], same cell velocities
            θb = k == 1 ? 1.5 * c.θ[i, 1] - 0.5 * c.θ[i, 2] : interp2(c.θ[i, k-1], c.θ[i, k])
            θt = k == nz ? 1.5 * c.θ[i, nz] - 0.5 * c.θ[i, nz-1] : interp2(c.θ[i, k], c.θ[i, k+1])
            ρuc = 0.5 * (g.Jx[i] * U0 + g.Jx[ir] * U1) / J
            ρwc = 0.5 * (fv_mz_face(g, q, i, k) + fv_mz_face(g, q, i, k+1))
            out[iΘ(g, i, k)] = c.θ[i, k] * out[iρ(g, i, k)] - ρuc * (θx1 - θx0) / dx -
                               (ρwc - g.zx_c[i, k] * ρuc) * (θt - θb) / (J * dη)
        elseif pert   # background: θ̄ ρ_t - ρw dθ̄/dz (out[ρ] holds ρ_t of this cell so far)
            ρw = 0.5 * (fv_mz_face(g, q, i, k) + fv_mz_face(g, q, i, k+1))
            θz = (θbσ[i, k+1] - θbσ[i, k]) / (g.Zf[i, k+1] - g.Zf[i, k])
            out[iΘ(g, i, k)] += θbc[i, k] * out[iρ(g, i, k)] - ρw * θz
        end

        # x-momentum control volume centred on x-face i: terrain pressure metric (the advection
        # is a.SU, shift technique)
        im = ip(g, i-1)
        pleft = pert ? 0.0 : c.pp[im, k]
        pright = pert ? 0.0 : c.pp[i, k]
        FxL = g.Jc[im] * pleft
        FxR = g.Jc[i] * pright
        FσL, FσR = 0.0, 0.0
        for (kf, dest) in ((k, 1), (k+1, 2))
            zxs = g.zx_xf[i, kf]
            # p at the corner; ground and top extrapolated (the full hydrostatic
            # p differs from the cell value by ρgΔz/2, an O(1) error in -∂_η(z_x p) otherwise)
            pface = 0.5 * (fv_p_sigma(c.pp, im, kf, nz) + fv_p_sigma(c.pp, i, kf, nz))
            Fσ = pert ? 0.0 : -zxs * pface
            dest == 1 ? (FσL = Fσ) : (FσR = Fσ)
        end
        out[iU(g, i, k)] = -((FxR - FxL) / dx + (FσR - FσL) / dη) / g.Jx[i] + a.SU[i, k]
        if pert   # -∂p'/∂x|_z in the θ̄-weighted chain-rule form
            θp(ii, kk) = θbc[ii, kk] * c.pp[ii, kk]
            mσ = 0.0
            for ii in (im, i), kf in (k, k+1)
                # ground and top: η-difference extrapolated (second order one-sided gradient)
                δ = kf == 1 ? 2 * (θp(ii, 2) - θp(ii, 1)) - (θp(ii, 3) - θp(ii, 2)) :
                    kf == nz+1 ? 2 * (θp(ii, nz) - θp(ii, nz-1)) - (θp(ii, nz-1) - θp(ii, nz-2)) :
                    θp(ii, kf) - θp(ii, kf-1)
                mσ += g.zx_f[ii, kf] * δ
            end
            # :advective: ∂_x p = (1/θ)[∂_x(θp) - p ∂_xθ], adjoint to the ρu ∂_xθ|_z term of Θ
            pr = adv ? 0.5 * (c.pp[im, k] * rc[im, k] + c.pp[i, k] * rc[i, k]) : 0.0
            out[iU(g, i, k)] -= ((θp(i, k) - θp(im, k)) / dx - mσ / (4 * g.Jx[i] * dη) - pr) /
                                (0.5 * (θbc[im, k] + θbc[i, k]))
        end

        # Vertical-momentum control volume centred at sigma-face k: pressure and gravity (the
        # advection is a.SW, shift technique)
        if k > 1
            ρpert = 0.5 * (q[iρ(g, i, k-1)] - m.ρbal[i, k-1] + q[iρ(g, i, k)] - m.ρbal[i, k])
            out[iW(g, i, k)] = -(c.pp[i, k] - c.pp[i, k-1]) / (J * dη) - m.gas.g * ρpert + a.SW[i, k]
        end
    end
    sponge!(out, m, q)
    return out
end
