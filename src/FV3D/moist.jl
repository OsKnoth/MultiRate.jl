# Moist air in 3D with the Exner pressure (or the pressure) as an additional prognostic variable: the 2D model of
# src/FV/moist.jl (formulation (I) of Jähn & Knoth, Bryan & Fritsch 2002 thermodynamics, no saturation adjustment,
# condensation as the explicit Fischer–Burmeister relaxation S_c = -φ_FB(ρ_vs - ρ_v, ρ_c)/τ) on the 3D C-grid,
# flat or terrain following (Gal-Chen), frozen split, vertically implicit column solve per column (i, j).
#
# State [ρ; U; V; W; X; ρq_v; ρq_c; π] (the base state of Model3D followed by the moist variables; X the
# thermodynamic variable of `MoistGas(thvar = ...)`: ρθ_ρ, ρe or ρs). Frozen form per face f and its two cells:
#     ρ, X, ρq_v, ρq_c, π rows: -δ(F), -δ(χ_cell,f F), -δ(qv_f F), -δ(qc_f F), -δ(κ_cell,f F),
#     κ_cell,f = a + b χ_cell,f + c qv_f + d qc_f,  (a, b, c, d) = ∂π/∂(ρ, X, ρq_v, ρq_c) frozen from y,
#     U, V, W rows: -c_pd Θ̂_f ∂π (pvar = :pi, Θ̂ = p/(R_d π)) or -∂p (pvar = :p), - g ρ̄ on W,
#     slow: momentum advection (shift technique, advection.jl), phase change for micro = :slow,
# F the live mass fluxes A J U, A J V, A Ω, χ = θ_f (upwind X/ρ) for :thetarho and :entropy, χ = e_f + p_c/ρ̄_f for
# :energy. The phase change (micro = :fast) is evaluated from the live state in every small step. π is recomputed
# from the equation of state after every MIS step. With ny = 1 the operator is the 2D one (tests).

using ..Common: MoistGas, moist_state, moist_exner, _moist_p, _Θof, theta_e, fischer_burmeister, theta_phase_factor,
                pressure_phase_factor, exner_derivatives

"""Vertical column system of the moist model (per column (i, j); η-faces k = 1 … nz+1)."""
mutable struct MoistColumn3D
    P::Array{Float64,3}            # momentum coefficient c_pd Θ̂ (or 1) at the η-faces
    cB::Array{Float64,3}           # κ of the cell below the face
    cT::Array{Float64,3}           # κ of the cell above the face
    χB::Array{Float64,3}           # χ of the cell below / above the face
    χT::Array{Float64,3}
    qz::Array{Float64,3}           # q_v, q_c at the faces
    cz::Array{Float64,3}
    factors::Dict{Float64,NTuple{3,Array{Float64,3}}}
    W::Vector{Float64}
end
MoistColumn3D(g::Grid3D) = (z = () -> zeros(g.nx, g.ny, g.nz + 1);
    MoistColumn3D(z(), z(), z(), z(), z(), z(), z(), Dict{Float64,NTuple{3,Array{Float64,3}}}(), zeros(g.nz + 1)))

"""
    MoistModel3D(base::Model3D; mgas = MoistGas(), pvar = :pi, micro = :fast, τfac = 0.25, cs = 350.0,
                 wτ = nothing, τ = nothing)

Moist 3D FV model (flat or terrain following; no cut cells, `implicit = :column`), the 3D form of
`FV2D.MoistExnerModel`: state `[ρ; U; V; W; X; ρq_v; ρq_c; π]` (`moist_exner_state`), frozen split for
`TimeStepping.run_frozen` (:verlet, :hevi, :ei, :wrf, :cs, :csfb, :rk4), π (or p) recomputed from the equation of
state after every MIS step. Condensation by the explicit Fischer–Burmeister relaxation, in every fast step
(micro = :fast, default), at the slow stages (:slow) or off (:none). Relaxation time: the given τ, or
τ = min(Δx, Δy, Δz)/wτ (Δz the thinnest physical layer), or by default τ = τfac Δτ_ac with the horizontal acoustic
step Δτ_ac = 0.85·0.92 min(Δx, Δy)/cs of the HEVI fast loop (as in 2D). The base model provides grid, gravity
(`gas.g`), momentum advection and sponge.
"""
struct MoistModel3D
    base::Model3D
    mgas::MoistGas
    pvar::Symbol
    micro::Symbol
    τ::Float64
    nq::Int
    C::Vector{Float64}
    S::Vector{Float64}
    φq::Array{Float64,3}           # q_v, q_c at the centres (work)
    φc::Array{Float64,3}
    abc::Array{Float64,4}          # ∂π/∂(ρ, X, ρq_v, ρq_c) per cell (work)
    Θh::Array{Float64,3}           # momentum coefficient c_pd Θ̂ (or 1) per cell (work)
    pc::Array{Float64,3}           # prognostic pressure per cell (work, :energy)
    column::MoistColumn3D
end
function MoistModel3D(base::Model3D; mgas::MoistGas = MoistGas(), pvar::Symbol = :pi, micro::Symbol = :fast,
                      τfac::Real = 0.25, cs::Real = 350.0, wτ = nothing, τ = nothing)
    pvar in (:pi, :p) || error("pvar: :pi or :p")
    mgas.thvar in (:thetarho, :energy, :entropy) || error("thvar: :thetarho, :energy or :entropy")
    micro in (:slow, :fast, :none) || error("micro: :slow, :fast or :none")
    g = base.grid
    (g.cut && !g.follow) && error("MoistModel3D: flat or terrain-following grids (no cut cells)")
    base.implicit === :column || error("MoistModel3D: implicit = :column")
    N = ncell(g); sz = (g.nx, g.ny, g.nz)
    h = min(g.dx, g.dy, minimum(g.Jc) * g.dz)
    τv = τ !== nothing ? Float64(τ) : wτ !== nothing ? h / wτ : τfac * 0.85 * 0.92 * min(g.dx, g.dy) / cs
    nc = 14N + 7 * g.nx * g.ny * (g.nz + 1)
    return MoistModel3D(base, mgas, pvar, micro, τv, base.nq + 3N, zeros(nc), zeros(base.nq + 3N),
                        zeros(sz), zeros(sz), zeros(4, sz...), zeros(sz), zeros(sz), MoistColumn3D(g))
end

@inline iqv(m::MoistModel3D, i, j, k) = m.base.nq + _c(m.base.grid, i, j, k)
@inline iqc(m::MoistModel3D, i, j, k) = m.base.nq + ncell(m.base.grid) + _c(m.base.grid, i, j, k)
@inline iπ(m::MoistModel3D, i, j, k) = m.base.nq + 2 * ncell(m.base.grid) + _c(m.base.grid, i, j, k)

"""The additional variable of the state: π (pvar = :pi) or p (pvar = :p) of the equation of state."""
moist_pvar(m::MoistModel3D, ρ, X, ρv, ρc) = m.pvar === :p ? _moist_p(m.mgas, ρ, X, ρv, ρc) : moist_exner(m.mgas, ρ, X, ρv, ρc)

"""
    moist_exner_state(m::MoistModel3D, q, Q)

Moist state [q; ρq_v; ρq_c; π] of a base state q (ρ, U, V, W, Θ = ρθ_ρ) and the total water Q (nx, ny, nz) per cell,
split into vapour and condensate of phase equilibrium (initial states only).
"""
function moist_exner_state(m::MoistModel3D, q, Q)
    g = m.base.grid; mg = m.mgas; N = ncell(g)
    x = vcat(q[1:m.base.nq], zeros(3N))
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        ρ, X = q[iρ(g, i, j, k)], q[iΘ(g, i, j, k)]
        ρc = moist_state(mg, ρ, X, Q[i, j, k])[3]
        x[iqv(m, i, j, k)] = Q[i, j, k] - ρc; x[iqc(m, i, j, k)] = ρc
        x[iπ(m, i, j, k)] = moist_pvar(m, ρ, X, Q[i, j, k] - ρc, ρc)
    end
    return x
end
base_state(m::MoistModel3D, q) = q[1:m.base.nq]

"""State of the model `mto` with the same (ρ, momenta, T, ρ_v, ρ_c) as the state x of `mfrom` (another
thermodynamic variable `mgas.thvar`); the pressure variable from the equation of state of `mto`."""
function thermo_convert(mfrom::MoistModel3D, mto::MoistModel3D, x)
    g = mfrom.base.grid
    y = copy(x)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        ρ, V, C = x[iρ(g, i, j, k)], x[iqv(mfrom, i, j, k)], x[iqc(mfrom, i, j, k)]
        T = moist_state(mfrom.mgas, ρ, x[iΘ(g, i, j, k)], V, C)[1]
        y[iΘ(g, i, j, k)] = _Θof(mto.mgas, ρ, T, V, C)
        y[iπ(mto, i, j, k)] = moist_pvar(mto, ρ, y[iΘ(g, i, j, k)], V, C)
    end
    return y
end

"""Discrete integral Σ J ΔV q_c of a cell variable (:ρ, :Θ, :qv, :qc) over the fluid cells."""
function total3d(m::MoistModel3D, q, var::Symbol)
    g = m.base.grid
    idx = var === :qv ? ((i, j, k) -> iqv(m, i, j, k)) : var === :qc ? ((i, j, k) -> iqc(m, i, j, k)) :
          var === :ρ ? ((i, j, k) -> iρ(g, i, j, k)) : ((i, j, k) -> iΘ(g, i, j, k))
    dV = g.dx * g.dy * g.dz
    return sum(q[idx(i, j, k)] * g.Jc[i, j] * dV / g.invV[i, j, k] for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx if g.invV[i, j, k] > 0)
end

"""Equivalent potential temperature at the centres (nx, ny, nz)."""
theta_e3d(m::MoistModel3D, q) = (g = m.base.grid;
    [theta_e(m.mgas, q[iρ(g, i, j, k)], q[iΘ(g, i, j, k)], q[iqv(m, i, j, k)], q[iqc(m, i, j, k)]) for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz])
"""q_c = ρq_c/ρ at the centres (nx, ny, nz)."""
qc3d(m::MoistModel3D, q) = (g = m.base.grid; [q[iqc(m, i, j, k)] / q[iρ(g, i, j, k)] for i in 1:g.nx, j in 1:g.ny, k in 1:g.nz])
vertical_velocity(m::MoistModel3D, q) = vertical_velocity(m.base, q)

# ------------------------------------------------------------------ coefficient vector
# x-faces, y-faces (N each), η-faces (Nz each), slots b = 0 … 6: χ of the cell left/below (0), q_v (1), the momentum
# coefficient c_pd Θ̂ (2), κ left/below (3), κ right/above (4), q_c (5), χ right/above (6).
frozen_ncoef(m::MoistModel3D) = length(m.C)
@inline mcx(C, g, b, i, j, k) = C[b * ncell(g) + _c(g, i, j, k)]
@inline mcy(C, g, b, i, j, k) = C[(7 + b) * ncell(g) + _c(g, i, j, k)]
@inline mcz(C, g, b, i, j, k) = C[14 * ncell(g) + b * _Nz(g) + _fz(g, i, j, k)]
@inline _sx(g, b, i, j, k) = b * ncell(g) + _c(g, i, j, k)
@inline _sy(g, b, i, j, k) = (7 + b) * ncell(g) + _c(g, i, j, k)
@inline _sz(g, b, i, j, k) = 14 * ncell(g) + b * _Nz(g) + _fz(g, i, j, k)

"""Phase-change sources of cell (i, j, k) for the state z (factors at the live state, as in 2D)."""
@inline function _phase!(out, m::MoistModel3D, z, i, j, k)
    g = m.base.grid; mg = m.mgas
    ρ, X, ρv, ρc = z[iρ(g, i, j, k)], z[iΘ(g, i, j, k)], z[iqv(m, i, j, k)], z[iqc(m, i, j, k)]
    φ, T = fischer_burmeister(mg, ρ, X, ρv, ρc)
    Sc = -φ / m.τ
    p = (mg.Rd * (ρ - ρv - ρc) + mg.Rv * ρv) * T
    Qp = pressure_phase_factor(mg, ρ, T, ρv, ρc)
    κS = m.pvar === :p ? Qp : (mg.Rd / mg.Cpd) * (p / mg.p0)^(mg.Rd / mg.Cpd) / p * Qp
    out[iqc(m, i, j, k)] += Sc
    out[iqv(m, i, j, k)] -= Sc
    out[iΘ(g, i, j, k)] += theta_phase_factor(mg, ρ, X, T, p, ρv, ρc) * Sc
    out[iπ(m, i, j, k)] += κS * Sc
    return out
end

"""C(y) and the slow part S(y) (momentum advection; phase change for micro = :slow)."""
function frozen_data!(C, S, m::MoistModel3D, q)
    b = m.base; g = b.grid; mg = m.mgas; c = b.cache
    nx, ny, nz = g.nx, g.ny, g.nz
    f = m.pvar === :p ? _moist_p : moist_exner
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ρ = q[iρ(g, i, j, k)]; X = q[iΘ(g, i, j, k)]; ρv = q[iqv(m, i, j, k)]; ρc = q[iqc(m, i, j, k)]
        c.θ[i, j, k] = X / ρ; m.φq[i, j, k] = ρv / ρ; m.φc[i, j, k] = ρc / ρ
        c.u[i, j, k] = q[iU(g, i, j, k)] / (0.5 * (q[iρ(g, i - 1, j, k)] + ρ))
        c.v[i, j, k] = q[iV(g, i, j, k)] / (0.5 * (q[iρ(g, i, j - 1, k)] + ρ))
        d = exner_derivatives(mg, ρ, X, ρv, ρc; f)
        for l in 1:4; m.abc[l, i, j, k] = d[l]; end
        πc = q[iπ(m, i, j, k)]
        if m.pvar === :p
            m.Θh[i, j, k] = 1.0; m.pc[i, j, k] = πc
        else
            p = mg.p0 * πc^(mg.Cpd / mg.Rd)
            m.Θh[i, j, k] = mg.Cpd * p / (mg.Rd * πc); m.pc[i, j, k] = p
        end
    end
    @inbounds for j in 1:ny, i in 1:nx                       # face w for the shift (as the dry model)
        c.w[i, j, 1] = g.follow ? g.zxf[i, j, 1] * 0.5 * (c.u[i, j, 1] + c.u[ip(g, i + 1), j, 1]) +
                                  g.zyf[i, j, 1] * 0.5 * (c.v[i, j, 1] + c.v[i, jp(g, j + 1), 1]) : 0.0
        c.w[i, j, nz+1] = 0.0
        for k in 2:nz
            c.w[i, j, k] = q[iW(g, i, j, k)] / (0.5 * (q[iρ(g, i, j, k - 1)] + q[iρ(g, i, j, k)]))
        end
    end
    fill!(C, 0.0); fill!(S, 0.0)
    mass_fluxes!(b, q)
    en = mg.thvar === :energy
    function κ(i, j, k, χ, qf, cf)
        ii, jj = ip(g, i), jp(g, j)
        return m.abc[1, ii, jj, k] + m.abc[2, ii, jj, k] * χ + m.abc[3, ii, jj, k] * qf + m.abc[4, ii, jj, k] * cf
    end
    Θh(i, j, k) = m.Θh[ip(g, i), jp(g, j), k]
    pc(i, j, k) = m.pc[ip(g, i), jp(g, j), k]
    ρq(i, j, k) = q[iρ(g, i, j, k)]
    # one face between the cells l (left/below) and r (right/above): slots 0 … 6 at the index s(b)
    function face!(s, F, θf, qf, cf, l, r)
        χl = χr = θf
        if en                                                  # χ = e_f + p_c/ρ̄_f
            ρf = 0.5 * (ρq(l...) + ρq(r...))
            χl = θf + pc(l...) / ρf; χr = θf + pc(r...) / ρf
        end
        C[s(0)] = χl; C[s(6)] = χr; C[s(1)] = qf; C[s(5)] = cf
        C[s(2)] = 0.5 * (Θh(l...) + Θh(r...))
        C[s(3)] = κ(l..., χl, qf, cf); C[s(4)] = κ(r..., χr, qf, cf)
        return nothing
    end
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if ax(g, i, j, k) > 0
            F = c.Fx[i, j, k]
            face!(bb -> _sx(g, bb, i, j, k), F, face_x(g, c.θ, i, j, k, F), face_x(g, m.φq, i, j, k, F),
                  face_x(g, m.φc, i, j, k, F), (i - 1, j, k), (i, j, k))
        end
        if ay(g, i, j, k) > 0
            F = c.Fy[i, j, k]
            face!(bb -> _sy(g, bb, i, j, k), F, face_y(g, c.θ, i, j, k, F), face_y(g, m.φq, i, j, k, F),
                  face_y(g, m.φc, i, j, k, F), (i, j - 1, k), (i, j, k))
        end
        if k > 1 && az(g, i, j, k) > 0
            F = c.Fz[i, j, k]
            face!(bb -> _sz(g, bb, i, j, k), F, face_z(g, c.θ, i, j, k, F), face_z(g, m.φq, i, j, k, F),
                  face_z(g, m.φc, i, j, k, F), (i, j, k - 1), (i, j, k))
        end
        m.micro === :slow && _phase!(S, m, q, i, j, k)
    end
    momentum_advection!(S, b)
    return C, S
end

# ------------------------------------------------------------------ fast part (building blocks)

"""Thermodynamic part, horizontal (ρ, X, ρq_v, ρq_c, π rows): x- and y-fluxes and the horizontal part of Ω."""
function mthermo_h!(out, m::MoistModel3D, C, z)
    g = m.base.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        ie, jn = ip(g, i + 1), jp(g, j + 1)
        Fw = g.Ax[i, j, k] * g.Jx[i, j] * z[iU(g, i, j, k)];   Fe = g.Ax[ie, j, k] * g.Jx[ie, j] * z[iU(g, i + 1, j, k)]
        Fs = g.Ay[i, j, k] * g.Jy[i, j] * z[iV(g, i, j, k)];   Fn = g.Ay[i, jn, k] * g.Jy[i, jn] * z[iV(g, i, j + 1, k)]
        s = g.invV[i, j, k] / g.Jc[i, j]
        flux(bl, br) = (mcx(C, g, bl, i + 1, j, k) * Fe - mcx(C, g, br, i, j, k) * Fw) / g.dx +
                       (mcy(C, g, bl, i, j + 1, k) * Fn - mcy(C, g, br, i, j, k) * Fs) / g.dy
        out[iρ(g, i, j, k)] -= s * ((Fe - Fw) / g.dx + (Fn - Fs) / g.dy)
        out[iΘ(g, i, j, k)] -= s * flux(0, 6)
        out[iqv(m, i, j, k)] -= s * flux(1, 1)
        out[iqc(m, i, j, k)] -= s * flux(5, 5)
        out[iπ(m, i, j, k)] -= s * flux(3, 4)
        if g.follow                                          # horizontal part of the vertical flux Ω
            Hb = g.Az[i, j, k] * Ωh(g, z, i, j, k); Ht = g.Az[i, j, k+1] * Ωh(g, z, i, j, k + 1)
            vflux(bb, bt) = (mcz(C, g, bb, i, j, k + 1) * Ht - mcz(C, g, bt, i, j, k) * Hb) / g.dz
            out[iρ(g, i, j, k)] -= s * (Ht - Hb) / g.dz
            out[iΘ(g, i, j, k)] -= s * vflux(0, 6)
            out[iqv(m, i, j, k)] -= s * vflux(1, 1)
            out[iqc(m, i, j, k)] -= s * vflux(5, 5)
            out[iπ(m, i, j, k)] -= s * vflux(3, 4)
        end
    end
    return out
end

"""Thermodynamic part, vertical (W fluxes)."""
function mthermo_v!(out, m::MoistModel3D, C, z)
    g = m.base.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        Fb = g.Az[i, j, k] * Wface(g, z, i, j, k); Ft = g.Az[i, j, k+1] * Wface(g, z, i, j, k + 1)
        s = g.invV[i, j, k] / (g.Jc[i, j] * g.dz)
        vflux(bb, bt) = mcz(C, g, bb, i, j, k + 1) * Ft - mcz(C, g, bt, i, j, k) * Fb
        out[iρ(g, i, j, k)] -= s * (Ft - Fb)
        out[iΘ(g, i, j, k)] -= s * vflux(0, 6)
        out[iqv(m, i, j, k)] -= s * vflux(1, 1)
        out[iqc(m, i, j, k)] -= s * vflux(5, 5)
        out[iπ(m, i, j, k)] -= s * vflux(3, 4)
    end
    return out
end

"""Cell field f(i, j, k) at the η-face kf: mean of the two cells, quadratically extrapolated face means at the ground
and the top (as `Πσ` of the dry operator)."""
@inline function _fσ(f, g, i, j, kf)
    mf(k) = 0.5 * (f(i, j, k - 1) + f(i, j, k))
    kf == 1 && return 3mf(2) - 3mf(3) + mf(4)
    kf == g.nz + 1 && return 3mf(g.nz) - 3mf(g.nz - 1) + mf(g.nz - 2)
    return mf(kf)
end
"""∂x f|_z at the x-face (i, j, k): (1/Jx)[δx(J f) - δη(z_x f_σ)] (flat: δx f), as `gradx` of the dry operator."""
@inline function _gradx(f, g, i, j, k)
    im = ip(g, i - 1)
    d = (g.Jc[ip(g, i), jp(g, j)] * f(i, j, k) - g.Jc[im, jp(g, j)] * f(im, j, k)) / g.dx
    if g.follow
        cor(kf) = 0.5 * (_fσ(f, g, im, j, kf) + _fσ(f, g, i, j, kf))
        d -= (g.zxx[ip(g, i), jp(g, j), k+1] * cor(k + 1) - g.zxx[ip(g, i), jp(g, j), k] * cor(k)) / g.dz
    end
    return d / g.Jx[ip(g, i), jp(g, j)]
end
@inline function _grady(f, g, i, j, k)
    jm = jp(g, j - 1)
    d = (g.Jc[ip(g, i), jp(g, j)] * f(i, j, k) - g.Jc[ip(g, i), jm] * f(i, jm, k)) / g.dy
    if g.follow
        cor(kf) = 0.5 * (_fσ(f, g, i, jm, kf) + _fσ(f, g, i, j, kf))
        d -= (g.zyy[ip(g, i), jp(g, j), k+1] * cor(k + 1) - g.zyy[ip(g, i), jp(g, j), k] * cor(k)) / g.dz
    end
    return d / g.Jy[ip(g, i), jp(g, j)]
end

"""Momentum part, horizontal (U, V rows): -c_pd Θ̂_f ∂π|_z (or -∂p|_z) on the open faces."""
function mmomentum_h!(out, m::MoistModel3D, C, z)
    g = m.base.grid
    f(i, j, k) = z[iπ(m, i, j, k)]
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        g.Ax[i, j, k] > 0 && (out[iU(g, i, j, k)] -= mcx(C, g, 2, i, j, k) * _gradx(f, g, i, j, k))
        g.Ay[i, j, k] > 0 && (out[iV(g, i, j, k)] -= mcy(C, g, 2, i, j, k) * _grady(f, g, i, j, k))
    end
    return out
end

"""Momentum part, vertical (W rows): -c_pd Θ̂_f δη π / (J Δη) - g ρ̄ on the open faces."""
function mmomentum_v!(out, m::MoistModel3D, C, z)
    g = m.base.grid; grav = m.base.gas.g
    @inbounds for k in 2:g.nz, j in 1:g.ny, i in 1:g.nx
        g.Az[i, j, k] > 0 || continue
        out[iW(g, i, j, k)] -= mcz(C, g, 2, i, j, k) * (z[iπ(m, i, j, k)] - z[iπ(m, i, j, k - 1)]) / (g.Jc[i, j] * g.dz) +
                               grav * 0.5 * (z[iρ(g, i, j, k - 1)] + z[iρ(g, i, j, k)])
    end
    return out
end

"""Constant forcing F on the thermodynamic rows (ρ, X, ρq_v, ρq_c, π)."""
function mforcing_thermo!(out, m::MoistModel3D, F)
    g = m.base.grid; N = ncell(g); nb = m.base.nq
    @inbounds for n in 1:N
        out[n] += F[n]                                   # ρ
        out[nb - N + n] += F[nb - N + n]                 # X
    end
    @inbounds for n in nb+1:nb+3N
        out[n] += F[n]                                   # ρq_v, ρq_c, π
    end
    return out
end

"""Phase change from the fast state (micro = :fast), explicit in every small step."""
function mphase!(out, m::MoistModel3D, z)
    m.micro === :fast || return out
    g = m.base.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        _phase!(out, m, z, i, j, k)
    end
    return out
end

# parts as in 2D: :all, :thermo / :momentum (Verlet), :hthermo / :hmomentum / :vertical (HEVI, EI, WRF, :cs)
_mfast!(::Val{:all}, out, m, C, z, F) = (mthermo_h!(out, m, C, z); mthermo_v!(out, m, C, z); mphase!(out, m, z);
    mmomentum_h!(out, m, C, z); mmomentum_v!(out, m, C, z);
    mforcing_thermo!(out, m, F); forcing_UV!(out, m.base, F); forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_mfast!(::Val{:thermo}, out, m, C, z, F) = (mthermo_h!(out, m, C, z); mthermo_v!(out, m, C, z); mphase!(out, m, z);
    mforcing_thermo!(out, m, F))
_mfast!(::Val{:momentum}, out, m, C, z, F) = (mmomentum_h!(out, m, C, z); mmomentum_v!(out, m, C, z);
    forcing_UV!(out, m.base, F); forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_mfast!(::Val{:hthermo}, out, m, C, z, F) = (mthermo_h!(out, m, C, z); mphase!(out, m, z); mforcing_thermo!(out, m, F);
    forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_mfast!(::Val{:hmomentum}, out, m, C, z, F) = (mmomentum_h!(out, m, C, z); forcing_UV!(out, m.base, F))
_mfast!(::Val{:vertical}, out, m, C, z, F) = (mthermo_v!(out, m, C, z); mmomentum_v!(out, m, C, z))
_mfast!(::Val{P}, out, m, C, z, F) where {P} = error("frozen_fast!: unknown part $P")

function frozen_fast!(out, m::MoistModel3D, C, z, part::Symbol, F)
    fill!(out, 0.0)
    _mfast!(Val(part), out, m, C, z, F)
    return out
end

function rhs_nonlinear!(out, m::MoistModel3D, q::AbstractVector)
    frozen_data!(m.C, m.S, m, q)
    return frozen_fast!(out, m, m.C, q, :all, m.S)
end

"""Phase-change part of :hthermo (`TimeStepping.frozen_phase!`, :cs/:csfb)."""
function frozen_phase!(out, m::MoistModel3D, C, z)
    m.micro === :fast || return false
    fill!(out, 0.0)
    mphase!(out, m, z)
    return true
end

"""π ← π(ρ, X, ρq_v, ρq_c) (or p) of the equation of state after a MIS step (no saturation adjustment)."""
function frozen_step_end!(m::MoistModel3D, q)
    g = m.base.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        q[iπ(m, i, j, k)] = moist_pvar(m, q[iρ(g, i, j, k)], q[iΘ(g, i, j, k)], q[iqv(m, i, j, k)], q[iqc(m, i, j, k)])
    end
    return q
end

# ------------------------------------------------------------------ vertically implicit step
# (I - D J_v) Y = R per column, D = c on the thermodynamic rows and cw on the W rows; eliminating ρ, X, ρq_v, ρq_c, π
# (r = c cw/(J Δη)², gg = c cw g/(2 J Δη), P_k the momentum coefficient, cT, cB the κ of the cells above / below):
#   W_{k-1} [-r P_k cT_{k-1} + gg] + W_k [1 + r P_k (cT_k + cB_k)] + W_{k+1} [-r P_k cB_{k+1} - gg]
#     = R^W_k - (cw/(J Δη)) P_k (R^π_k - R^π_{k-1}) - (cw g/2)(R^ρ_{k-1} + R^ρ_k),
# then the thermodynamic rows from the fluxes (the 2D system of `FV2D.moist_column_solve!` per column (i, j)).

function frozen_stage_begin!(m::MoistModel3D, C, z)
    s = m.column; g = m.base.grid
    @inbounds for k in 1:g.nz+1, j in 1:g.ny, i in 1:g.nx
        s.χB[i, j, k] = mcz(C, g, 0, i, j, k); s.qz[i, j, k] = mcz(C, g, 1, i, j, k); s.P[i, j, k] = mcz(C, g, 2, i, j, k)
        s.cB[i, j, k] = mcz(C, g, 3, i, j, k); s.cT[i, j, k] = mcz(C, g, 4, i, j, k); s.cz[i, j, k] = mcz(C, g, 5, i, j, k)
        s.χT[i, j, k] = mcz(C, g, 6, i, j, k)
    end
    empty!(s.factors)
    return s
end

function _moist_factor(s::MoistColumn3D, m::MoistModel3D, c, cw)
    get!(s.factors, Float64(c * cw)) do
        g = m.base.grid; nz = g.nz
        sub = zeros(nz + 1, g.nx, g.ny); cpr = zeros(nz + 1, g.nx, g.ny); iden = zeros(nz + 1, g.nx, g.ny)
        @inbounds for j in 1:g.ny, i in 1:g.nx
            dz = g.Jc[i, j] * g.dz
            r = c * cw / dz^2; gg = c * cw * m.base.gas.g / (2dz)
            for k in 2:nz
                a = k > 2 ? -r * s.P[i, j, k] * s.cT[i, j, k-1] + gg : 0.0
                d = 1 + r * s.P[i, j, k] * (s.cT[i, j, k] + s.cB[i, j, k])
                b = k < nz ? -r * s.P[i, j, k] * s.cB[i, j, k+1] - gg : 0.0
                den = k == 2 ? d : d - a * cpr[k-1, i, j]
                sub[k, i, j] = a; iden[k, i, j] = 1 / den; cpr[k, i, j] = b / den
            end
        end
        (sub, cpr, iden)
    end
end

"""Y = (I - D J_v)⁻¹ R, D = c on the thermodynamic rows and cw on the W rows (U, V rows: identity)."""
function moist_column_solve!(Y, m::MoistModel3D, c, R, cw = c)
    s = m.column; g = m.base.grid
    nz, grav = g.nz, m.base.gas.g
    sub, cpr, iden = _moist_factor(s, m, c, cw)
    W = s.W
    Y .= R
    @inbounds for j in 1:g.ny, i in 1:g.nx
        dz = g.Jc[i, j] * g.dz
        for k in 2:nz
            rhs = R[iW(g, i, j, k)] - (cw / dz) * s.P[i, j, k] * (R[iπ(m, i, j, k)] - R[iπ(m, i, j, k - 1)]) -
                  (cw * grav / 2) * (R[iρ(g, i, j, k - 1)] + R[iρ(g, i, j, k)])
            W[k] = k == 2 ? rhs * iden[k, i, j] : (rhs - sub[k, i, j] * W[k-1]) * iden[k, i, j]
        end
        for k in nz-1:-1:2
            W[k] -= cpr[k, i, j] * W[k+1]
        end
        W[1] = 0.0; W[nz+1] = 0.0
        for k in 1:nz
            r = c / dz
            Y[iρ(g, i, j, k)] = R[iρ(g, i, j, k)] - r * (W[k+1] - W[k])
            Y[iΘ(g, i, j, k)] = R[iΘ(g, i, j, k)] - r * (s.χB[i, j, k+1] * W[k+1] - s.χT[i, j, k] * W[k])
            Y[iqv(m, i, j, k)] = R[iqv(m, i, j, k)] - r * (s.qz[i, j, k+1] * W[k+1] - s.qz[i, j, k] * W[k])
            Y[iqc(m, i, j, k)] = R[iqc(m, i, j, k)] - r * (s.cz[i, j, k+1] * W[k+1] - s.cz[i, j, k] * W[k])
            Y[iπ(m, i, j, k)] = R[iπ(m, i, j, k)] - r * (s.cB[i, j, k+1] * W[k+1] - s.cT[i, j, k] * W[k])
        end
        for k in 2:nz
            Y[iW(g, i, j, k)] = W[k]
        end
    end
    return Y
end

"""Crank–Nicolson step of the vertical part (:hevi): z ← z + (I - dt/2 J_v)⁻¹ dt G_v(z)."""
function frozen_vertical_step!(z, m::MoistModel3D, C, dt, k, y)
    frozen_fast!(k, m, C, z, :vertical, k)
    @. k *= dt
    moist_column_solve!(y, m, dt / 2, k)
    @. z += y
    return z
end

"""out *= ct on the thermodynamic rows, cw on the W rows (the U, V rows of a vertical part are zero)."""
function _scale_tw!(out, m::MoistModel3D, ct, cw)
    g = m.base.grid; N = ncell(g); nW = g.nx * g.ny * (g.nz - 1)
    @inbounds for n in eachindex(out)
        out[n] *= (3N < n <= 3N + nW) ? cw : (N < n <= 3N ? 0.0 : ct)
    end
    return out
end

"""out = D G_v(z) (:ei, :wrf, :cs): the vertical part with the thermodynamic rows scaled by ct, the W rows by cw."""
function frozen_vertical_rhs!(out, m::MoistModel3D, C, z, ct, cw)
    frozen_fast!(out, m, C, z, :vertical, out)
    return _scale_tw!(out, m, ct, cw)
end
"""z ← z + Y, (I - D J_v) Y = D G_v(z)."""
function frozen_vertical_implicit!(z, m::MoistModel3D, C, ct, cw, k, y)
    frozen_vertical_rhs!(k, m, C, z, ct, cw)
    moist_column_solve!(y, m, ct, k, cw)
    @. z += y
    return z
end
"""y = (I - D J_v)⁻¹ R (:wrf)."""
frozen_vertical_solve!(y, m::MoistModel3D, C, ct, cw, R) = moist_column_solve!(y, m, ct, R, cw)
