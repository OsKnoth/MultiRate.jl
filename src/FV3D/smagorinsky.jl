# Smagorinsky–Lilly subgrid-scale model (Jähn et al. 2015, ASAM v2.7, Sect. 3.1) for MoistModel3D, a slow tendency
# (added to S in `frozen_data!`, once per MIS stage):
#     momentum   ∂t(ρu_a) += ∂_b(ρν (∂_b u_a + ∂_a u_b))                          (τ_ab = −2ν S_ab, eq. 34)
#     scalars    ∂t(ρφ)   += ∂_b(ρν/Pr ∂_b φ),   φ = θ_ρ, q_v, q_c (, q_r)
#     ν = (C_s Δ)² max(0, |S|² (1 − Ri/Pr))^½,  |S|² = 2 S_ab S_ab,  Ri = (g/θ_ρ) ∂_z θ_ρ / |S|²   (eqs. 35–37)
#     Δ = (Δx Δy J Δη)^⅓ f(a1, a2), f = cosh √(4/27 (ln²a1 − ln a1 ln a2 + ln²a2)) (Scotti et al. 1993, eq. 39),
# plus an optional constant viscosity ν0 (C_s = 0, ν0 > 0: constant eddy viscosity, e.g. the density current of
# Straka et al.) and a cap νmax (the term is explicit: ν/Pr Δt/(J Δη)² ≲ 0.25 for the large step Δt).
#
# Discretization (flat or terrain following, conservative in (x, y, η)): cell-centred velocities u_c, v_c, w_c (means of
# the two faces) and scalars; physical gradients at the cells (centred, ∂_x|_z = ∂_x|_η − z_x ∂_z); at a face the normal
# derivative is compact (x, y: `_gradx`, `_grady` with the metric terms of the pressure gradient; η: δη/(J Δη)), the
# tangential ones are the means of the two cells. Fluxes through the x-, y-faces (J F^x, J F^y) and the η-faces
# (F^η = F^z − z_x F^x − z_y F^y), zero at the ground and the top (free slip, no surface fluxes). The cell tendencies of
# the momentum return to the faces with the volume weights (U, V) or the mean (W). The thermodynamic variable X gets the
# change of X(ρθ_ρ, ρ_v, ρ_l) (fixed ρ) along the tendencies of ρθ_ρ, ρ_v, ρ_l (the tendency of ρθ_ρ for thvar =
# :thetarho), so that all thermodynamic variables give the same θ_ρ; π (or p) the frozen ∂π/∂(X, ρ_v, ρ_l) of the tendencies.

"""
    Smagorinsky(; cs = 0.2, Pr = 1/3, stability = true, scotti = true, ν0 = 0.0, νmax = Inf)

Parameters of the Smagorinsky–Lilly subgrid-scale model of `MoistModel3D(…; smagorinsky = Smagorinsky())`:
Smagorinsky constant `cs`, turbulent Prandtl number `Pr` (scalars get ν/Pr), Lilly's stability correction
(`stability`), the filter length of Scotti et al. for anisotropic cells (`scotti`), a constant viscosity `ν0` added
to ν and the cap `νmax` (m²/s) of ν. See the comment at the top of `FV3D/smagorinsky.jl`.
"""
struct Smagorinsky
    cs::Float64
    Pr::Float64
    stability::Bool
    scotti::Bool
    ν0::Float64
    νmax::Float64
end
Smagorinsky(; cs = 0.2, Pr = 1 / 3, stability = true, scotti = true, ν0 = 0.0, νmax = Inf) =
    Smagorinsky(cs, Pr, stability, scotti, ν0, νmax)

"""Work arrays of the Smagorinsky model: cell fields u, v, w, θ_ρ, q_v, q_c, q_r, their gradients, ρν, the tendencies."""
struct SmagWork
    F::Array{Float64,4}            # (7, nx, ny, nz)
    G::Array{Float64,5}            # (3, 7, nx, ny, nz) physical gradients at the cells
    ρν::Array{Float64,3}
    T::Array{Float64,4}            # cell tendencies (7, nx, ny, nz)
    Fx::Array{Float64,3}           # face fluxes J F^x, J F^y, F^η of one field
    Fy::Array{Float64,3}
    Fe::Array{Float64,3}           # (nx, ny, nz+1)
end
SmagWork(g::Grid3D) = SmagWork(zeros(7, g.nx, g.ny, g.nz), zeros(3, 7, g.nx, g.ny, g.nz), zeros(g.nx, g.ny, g.nz),
                               zeros(7, g.nx, g.ny, g.nz), zeros(g.nx, g.ny, g.nz),
                               zeros(g.nx, g.ny, g.nz), zeros(g.nx, g.ny, g.nz + 1))

"""Filter length Δ of column (i, j): (Δx Δy J Δη)^⅓, with the anisotropy factor of Scotti et al. (1993)."""
function filter_length(sp::Smagorinsky, g::Grid3D, i, j)
    h = (g.dx, g.dy, g.Jc[i, j] * g.dz)
    Δ = cbrt(h[1] * h[2] * h[3])
    sp.scotti || return Δ
    s = sort(collect(h)); l1 = log(s[1] / s[3]); l2 = log(s[2] / s[3])
    return Δ * cosh(sqrt(4 / 27 * (l1^2 - l1 * l2 + l2^2)))
end

"""Physical gradient (∂x, ∂y, ∂z) of the cell field n at the centre (i, j, k): centred, one-sided at the walls."""
@inline function _cgrad(F, n, g, i, j, k)
    nz = g.nz; dzi = 1 / (g.Jc[i, j] * g.dz)
    gz = k == 1 ? (F[n, i, j, 2] - F[n, i, j, 1]) * dzi : k == nz ? (F[n, i, j, nz] - F[n, i, j, nz-1]) * dzi :
         0.5 * (F[n, i, j, k+1] - F[n, i, j, k-1]) * dzi
    gx = (F[n, ip(g, i + 1), j, k] - F[n, ip(g, i - 1), j, k]) / (2g.dx)
    gy = (F[n, i, jp(g, j + 1), k] - F[n, i, jp(g, j - 1), k]) / (2g.dy)
    if g.follow
        gx -= 0.5 * (g.zxf[i, j, k] + g.zxf[i, j, k+1]) * gz
        gy -= 0.5 * (g.zyf[i, j, k] + g.zyf[i, j, k+1]) * gz
    end
    return gx, gy, gz
end

"""Physical derivative ∂_b of the cell field n at the face of direction d (1: x-face i, 2: y-face j, 3: η-face k):
compact in the normal direction, the mean of the two cells otherwise."""
@inline function _fgrad(w::SmagWork, g, n, b, d, i, j, k)
    F, G = w.F, w.G
    if d == 1
        b == 1 && return _gradx((ii, jj, kk) -> F[n, ip(g, ii), jp(g, jj), kk], g, i, j, k)
        return 0.5 * (G[b, n, ip(g, i - 1), j, k] + G[b, n, i, j, k])
    elseif d == 2
        b == 2 && return _grady((ii, jj, kk) -> F[n, ip(g, ii), jp(g, jj), kk], g, i, j, k)
        return 0.5 * (G[b, n, i, jp(g, j - 1), k] + G[b, n, i, j, k])
    else
        b == 3 && return (F[n, i, j, k] - F[n, i, j, k-1]) / (g.Jc[i, j] * g.dz)
        return 0.5 * (G[b, n, i, j, k-1] + G[b, n, i, j, k])
    end
end

"""Flux divergence of field n into T[n, …] from the face fluxes flux(d, i, j, k) = physical flux vector component."""
function _sgs_field!(w::SmagWork, g, n, coef, flux)
    nx, ny, nz = g.nx, g.ny, g.nz
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx                    # the faces with ν = 0 are skipped
        K = coef * 0.5 * (w.ρν[ip(g, i - 1), j, k] + w.ρν[i, j, k])
        w.Fx[i, j, k] = K == 0 ? 0.0 : g.Jx[i, j] * K * flux(1, 1, i, j, k)
        K = coef * 0.5 * (w.ρν[i, jp(g, j - 1), k] + w.ρν[i, j, k])
        w.Fy[i, j, k] = (ny == 1 || K == 0) ? 0.0 : g.Jy[i, j] * K * flux(2, 2, i, j, k)
    end
    @inbounds for j in 1:ny, i in 1:nx
        w.Fe[i, j, 1] = 0.0; w.Fe[i, j, nz+1] = 0.0
        for k in 2:nz
            K = coef * 0.5 * (w.ρν[i, j, k-1] + w.ρν[i, j, k])
            K == 0 && (w.Fe[i, j, k] = 0.0; continue)
            fz = flux(3, 3, i, j, k)
            if g.follow
                fz -= g.zxf[i, j, k] * flux(1, 3, i, j, k) + (ny == 1 ? 0.0 : g.zyf[i, j, k] * flux(2, 3, i, j, k))
            end
            w.Fe[i, j, k] = K * fz
        end
    end
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ie, jn = ip(g, i + 1), jp(g, j + 1)
        w.T[n, i, j, k] = ((w.Fx[ie, j, k] - w.Fx[i, j, k]) / g.dx + (w.Fy[i, jn, k] - w.Fy[i, j, k]) / g.dy +
                           (w.Fe[i, j, k+1] - w.Fe[i, j, k]) / g.dz) / g.Jc[i, j]
    end
    return w
end

"""
    smagorinsky!(S, m::MoistModel3D, q)

Adds the subgrid-scale tendencies of the Smagorinsky–Lilly model to the slow part S (`frozen_data!`; needs the face
velocities in the cache and the frozen ∂π derivatives `m.abc` of the state q).
"""
function smagorinsky!(S, m, q)
    sp = m.smag; sp === nothing && return S
    w = m.sw; b = m.base; g = b.grid; c = b.cache; mg = m.mgas; grav = b.gas.g
    nx, ny, nz = g.nx, g.ny, g.nz
    nf = m.rain ? 7 : 6
    mgθ = MoistGas{:thetarho}(mg.Rd, mg.Rv, mg.Cpd, mg.Cpv, mg.Cpl, mg.L0V, mg.T0, mg.p0)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ρ, X, ρv = q[iρ(g, i, j, k)], q[iΘ(g, i, j, k)], q[iqv(m, i, j, k)]
        ρc, ρr = q[iqc(m, i, j, k)], _ρr(m, q, i, j, k); ρl = ρc + ρr
        w.F[1, i, j, k] = 0.5 * (c.u[i, j, k] + c.u[ip(g, i + 1), j, k])
        w.F[2, i, j, k] = 0.5 * (c.v[i, j, k] + c.v[i, jp(g, j + 1), k])
        w.F[3, i, j, k] = 0.5 * (c.w[i, j, k] + c.w[i, j, k+1])
        w.F[4, i, j, k] = mg.thvar === :thetarho ? X / ρ : _Θof(mgθ, ρ, moist_state(mg, ρ, X, ρv, ρl)[1], ρv, ρl) / ρ
        w.F[5, i, j, k] = ρv / ρ; w.F[6, i, j, k] = ρc / ρ; w.F[7, i, j, k] = ρr / ρ
    end
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx, n in 1:nf
        w.G[1, n, i, j, k], w.G[2, n, i, j, k], w.G[3, n, i, j, k] = _cgrad(w.F, n, g, i, j, k)
    end
    # eddy viscosity ρν at the cells
    @inbounds for j in 1:ny, i in 1:nx
        l2 = (sp.cs * filter_length(sp, g, i, j))^2
        for k in 1:nz
            S2 = 0.0                                  # |S|² = 2 S_ab S_ab = Σ (∂_b u_a + ∂_a u_b)²/2
            for a in 1:3, bb in 1:3
                S2 += 0.5 * (w.G[bb, a, i, j, k] + w.G[a, bb, i, j, k])^2
            end
            arg = sp.stability ? S2 - grav * w.G[3, 4, i, j, k] / w.F[4, i, j, k] / sp.Pr : S2
            ν = min(l2 * sqrt(max(arg, 0.0)) + sp.ν0, sp.νmax)
            w.ρν[i, j, k] = q[iρ(g, i, j, k)] * ν
        end
    end
    maximum(w.ρν) == 0 && return S                                   # stable everywhere: no subgrid-scale tendencies
    # scalars θ_ρ, q_v, q_c (, q_r)
    for n in 4:nf
        _sgs_field!(w, g, n, 1 / sp.Pr, (bb, d, i, j, k) -> _fgrad(w, g, n, bb, d, i, j, k))
    end
    # momentum: flux of u_a in direction b: ρν (∂_b u_a + ∂_a u_b)
    for a in 1:3
        _sgs_field!(w, g, a, 1.0, (bb, d, i, j, k) -> _fgrad(w, g, a, bb, d, i, j, k) + _fgrad(w, g, bb, a, d, i, j, k))
    end
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        dv, dc, dr = w.T[5, i, j, k], w.T[6, i, j, k], m.rain ? w.T[7, i, j, k] : 0.0
        dX = w.T[4, i, j, k]
        if mg.thvar !== :thetarho && (dX != 0 || dv != 0 || dc + dr != 0)   # X(ρθ_ρ, ρ_v, ρ_l) at fixed ρ: directional derivative
            ρ, ρv, ρl = q[iρ(g, i, j, k)], q[iqv(m, i, j, k)], _ρl(m, q, i, j, k)
            Θρ = ρ * w.F[4, i, j, k]; ε = 1e-6 / max(abs(dX) / Θρ, (abs(dv) + abs(dc + dr)) / ρ, floatmin())   # relative changes ≤ 1e-6
            Xof(s) = (Θ = Θρ + s * dX; V = ρv + s * dv; L = ρl + s * (dc + dr);
                      _Θof(mg, ρ, moist_state(mgθ, ρ, Θ, V, L)[1], V, L))
            dX = (Xof(ε) - Xof(-ε)) / 2ε
        end
        S[iΘ(g, i, j, k)] += dX; S[iqv(m, i, j, k)] += dv; S[iqc(m, i, j, k)] += dc
        m.rain && (S[iqr(m, i, j, k)] += dr)
        S[iπ(m, i, j, k)] += m.abc[2, i, j, k] * dX + m.abc[3, i, j, k] * dv + m.abc[4, i, j, k] * (dc + dr)
        im, jm = ip(g, i - 1), jp(g, j - 1)
        if g.Ax[i, j, k] > 0
            Jl, Jr = g.Jc[im, j], g.Jc[i, j]
            S[iU(g, i, j, k)] += (Jl * w.T[1, im, j, k] + Jr * w.T[1, i, j, k]) / (Jl + Jr)
        end
        if g.Ay[i, j, k] > 0
            Jl, Jr = g.Jc[i, jm], g.Jc[i, j]
            S[iV(g, i, j, k)] += (Jl * w.T[2, i, jm, k] + Jr * w.T[2, i, j, k]) / (Jl + Jr)
        end
        k >= 2 && g.Az[i, j, k] > 0 && (S[iW(g, i, j, k)] += 0.5 * (w.T[3, i, j, k-1] + w.T[3, i, j, k]))
    end
    return S
end

"""Eddy viscosity ν (m²/s) at the cells of the last `frozen_data!` (zeros without the Smagorinsky model)."""
eddy_viscosity(m, q) = m.smag === nothing ? zeros(size(m.φq)) :
    (g = m.base.grid; [m.sw.ρν[i, j, k] / q[iρ(g, i, j, k)] for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz])
