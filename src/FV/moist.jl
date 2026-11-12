# Moist air with the Exner pressure as an additional prognostic variable (flat FV grid, frozen split)
#
# Formulation (I) of Jähn & Knoth (poster "Compressible modeling of a cloudy atmosphere using a
# general pressure evolution equation"), Bryan & Fritsch (2002) thermodynamics:
#     ρ_t + ∇·M = 0,   M_t + ∇·(M v) = -∇p - ρ g k,
#     Θ_t + ∇·(θ_ρ M) = Q_θ,  Θ = ρ θ_ρ,
#     V_t + ∇·(q_v M) = -S_c,  V = ρ_v = ρ q_v,     C_t + ∇·(q_c M) = S_c,  C = ρ_c = ρ q_c,
#     p = p0 (R_d Θ / p0)^(1/(1-κ_M)),  κ_M = R_m/c_pm,  R_m = R_d ρ_d + R_v ρ_v,
#     c_pm = c_pd ρ_d + c_pv ρ_v + c_pl ρ_c,  ρ_d = ρ - ρ_v - ρ_c,
# θ_ρ the density potential temperature with the moist κ_M. No saturation adjustment: the
# condensation rate is an explicit relaxation to phase equilibrium with the Fischer–Burmeister function
#     S_c = -φ_FB(ρ_vs(T) - ρ_v, ρ_c) / τ,   φ_FB(a, b) = (a + b - √(a² + b²))/2,
# τ from the grid spacing (default a quarter of the acoustic fast step, the shortest stable τ), and the latent heating
# Q_θ = -Θ P S_c with the prefactor of Bryan & Fritsch (2002) (`theta_phase_factor`, isochoric first law),
# and the source Q_p = [R_m (L - R_v T)/c_vm - R_v T] S_c of the pressure equation (`pressure_phase_factor`). `micro = :fast` (default):
# S_c from the current fast state in every small step; `micro = :slow`: S_c from the stage value y.
# Constants and e_s(T) as in CGDycore.
#
# Exner pressure π = (p/p0)^(R_d/c_pd) (dry exponent): exact product form of the pressure gradient
#     ∇p = c_pd Θ̂ ∇π,   Θ̂ = p/(R_d π),
# and the pressure equation π_t = a ρ_t + b Θ_t + c V_t + d C_t with the partial derivatives
# (a, b, c, d) = ∂π/∂(ρ, Θ, V, C) of the equation of state, frozen from y.
# State [ρ; U; W; Θ; V; C; π]; π is recomputed from the equation of state after every MIS step.
# pvar = :p: the pressure itself instead of π, (a, b, c, d) = ∂p/∂(ρ, Θ, V, C), plain -δp in the
# momentum rows.
# Frozen form (fast part linear in z apart from the :fast phase source), per face f and its two cells:
#     ρ, Θ, V, C, π rows: -δ(F), -δ(θ_f F), -δ(qv_f F), -δ(qc_f F), -δ(κ_cell,f F),
#     κ_cell,f = a + b θ_f + c qv_f + d qc_f,
#     U, W rows: -c_pd Θ̂_f δπ (- g ρ̄),   slow: momentum advection (shift technique).

Base.@kwdef struct MoistGas
    Rd::Float64 = 287.0
    Rv::Float64 = 461.0
    Cpd::Float64 = 1004.0
    Cpv::Float64 = 1885.0
    Cpl::Float64 = 4186.0
    L0V::Float64 = 2.5e6
    T0::Float64 = 273.15
    p0::Float64 = 1.0e5
end

"""Saturation vapour pressure over water (CGDycore `fpws`)."""
@inline psat(mg::MoistGas, T) = 661.2 * exp(17.368 * (T - mg.T0) / (238.88 + T - mg.T0))
"""Latent heat of vaporization L(T) = L0V + (c_pv - c_pl)(T - T0)."""
@inline latent(mg::MoistGas, T) = mg.L0V + (mg.Cpv - mg.Cpl) * (T - mg.T0)

"""Temperature for given (ρ, Θ, Q) and vapour density ρv (CGDycore `fTemp`)."""
@inline function _temp(mg::MoistGas, ρ, Θ, Q, ρv)
    ρd = ρ - Q
    Rm = mg.Rd * ρd + mg.Rv * ρv
    κ = Rm / (mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * (Q - ρv))
    return (mg.Rd * Θ / mg.p0^κ)^(1 / (1 - κ)) / Rm, Rm
end

"""
    moist_state(mg, ρ, Θ, Q) -> (T, ρv, ρc, p)

Phase equilibrium at fixed (ρ, Θ, Q) (total water Q): only for initial states; the model itself has
no saturation adjustment.
"""
function moist_state(mg::MoistGas, ρ, Θ, Q)
    T, Rm = _temp(mg, ρ, Θ, Q, Q)
    psat(mg, T) / (mg.Rv * T) >= Q && return (T, Q, 0.0, Rm * T)       # unsaturated
    f(ρv) = (T = _temp(mg, ρ, Θ, Q, ρv)[1]; ρv - psat(mg, T) / (mg.Rv * T))
    a, b = 0.0, Q; fa, fb = f(a), f(b)                                    # fa < 0 < fb
    for _ in 1:200                                                         # Illinois regula falsi
        c = (a * fb - b * fa) / (fb - fa); fc = f(c)
        if fc * fb < 0
            a, fa = b, fb
        else
            fa /= 2
        end
        b, fb = c, fc
        abs(b - a) <= 1e-15 * Q && break
    end
    ρv = b
    T, Rm = _temp(mg, ρ, Θ, Q, ρv)
    return (T, ρv, Q - ρv, Rm * T)
end

"""(T, ρv, ρc, p) for given vapour and condensate: equation of state without phase equilibrium."""
function moist_state(mg::MoistGas, ρ, Θ, ρv, ρc)
    T, Rm = _temp(mg, ρ, Θ, ρv + ρc, ρv)
    return (T, ρv, ρc, Rm * T)
end

"""Exner pressure (dry exponent) of the equation of state (in phase equilibrium, or for given ρc)."""
moist_exner(mg::MoistGas, ρ, Θ, Q) = (moist_state(mg, ρ, Θ, Q)[4] / mg.p0)^(mg.Rd / mg.Cpd)
moist_exner(mg::MoistGas, ρ, Θ, ρv, ρc) = (moist_state(mg, ρ, Θ, ρv, ρc)[4] / mg.p0)^(mg.Rd / mg.Cpd)
_moist_p(mg, ρ, Θ, ρv, ρc) = moist_state(mg, ρ, Θ, ρv, ρc)[4]

@inline function _theta_e(mg::MoistGas, ρ, Q, T, ρv)
    ρd = ρ - Q; rv = ρv / ρd
    c = mg.Cpd + mg.Cpl * Q / ρd
    return T * (ρd * mg.Rd * T / mg.p0)^(-mg.Rd / c) * exp(latent(mg, T) * rv / (c * T))
end
"""Equivalent potential temperature (poster eq. 17, CGDycore `fThE`), in phase equilibrium (ρ, Θ, Q) or for given (ρ, Θ, ρv, ρc)."""
theta_e(mg::MoistGas, ρ, Θ, Q) = (s = moist_state(mg, ρ, Θ, Q); _theta_e(mg, ρ, Q, s[1], s[2]))
theta_e(mg::MoistGas, ρ, Θ, ρv, ρc) = _theta_e(mg, ρ, ρv + ρc, moist_state(mg, ρ, Θ, ρv, ρc)[1], ρv)

"""Θ = ρθ_ρ of (ρ, T, ρv, ρc)."""
@inline function _Θof(mg::MoistGas, ρ, T, ρv, ρc)
    ρd = ρ - ρv - ρc
    Rm = mg.Rd * ρd + mg.Rv * ρv
    κ = Rm / (mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * ρc)
    return (Rm * T)^(1 - κ) * mg.p0^κ / mg.Rd
end

"""
    fischer_burmeister(mg, ρ, Θ, ρv, ρc) -> (φ, T)

φ_FB(a, b) = (a + b - √(a² + b²))/2 of a = ρ_vs(T) - ρ_v, b = ρ_c: zero in phase equilibrium,
negative when supersaturated, positive when subsaturated with condensate (then φ ≤ ρ_c).
"""
@inline function fischer_burmeister(mg::MoistGas, ρ, Θ, ρv, ρc)
    T = moist_state(mg, ρ, Θ, ρv, ρc)[1]
    a = psat(mg, T) / (mg.Rv * T) - ρv
    return 0.5 * (a + ρc - sqrt(a * a + ρc * ρc)), T
end

"""
    theta_phase_factor(mg, ρ, Θ, T, p, ρv, ρc)

dΘ/dρ_c of the phase change (condensation dρ_c = -dρ_v > 0) at fixed ρ, Bryan & Fritsch (2002),
as CGDycore `SimpleMicrophysicsPot` (there with the evaporation rate F_Ph = -S_c):
    Q_Θ = -Θ P S_c,  P = -L/(c_pm T) - ln(p/p0) (R_m/c_pm) (R_v/R_m + (c_pl - c_pv)/c_pm) + R_v/R_m,
with the density-weighted R_m = R_d ρ_d + R_v ρ_v, c_pm = c_pd ρ_d + c_pv ρ_v + c_pl ρ_c. Through the
equation of state this is the isochoric first law, dT/dρ_c = (L - R_v T)/c_vm, c_vm = c_pm - R_m.
"""
function theta_phase_factor(mg::MoistGas, ρ, Θ, T, p, ρv, ρc)
    ρd = ρ - ρv - ρc
    cpm = mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * ρc; Rm = mg.Rd * ρd + mg.Rv * ρv
    L = latent(mg, T)
    P = -L / (cpm * T) - log(p / mg.p0) * (Rm / cpm) * (mg.Rv / Rm + (mg.Cpl - mg.Cpv) / cpm) + mg.Rv / Rm
    return -Θ * P
end

"""
    pressure_phase_factor(mg, ρ, T, ρv, ρc)

dp/dρ_c of the phase change at fixed ρ (isochoric, energy conserving), the source of the pressure
equation Q_p = [R_m (L - R_v T)/c_vm - R_v T] S_c (p = R_m T: dR_m = -R_v dρ_c, dT = (L - R_v T)/c_vm dρ_c).
Identical to the chain rule (∂p/∂Θ) dΘ/dρ_c + ∂p/∂ρ_c - ∂p/∂ρ_v of the equation of state with
`theta_phase_factor`. Exner pressure: Q_π = (R_d/c_pd) (π/p) Q_p.
"""
function pressure_phase_factor(mg::MoistGas, ρ, T, ρv, ρc)
    ρd = ρ - ρv - ρc
    Rm = mg.Rd * ρd + mg.Rv * ρv; cvm = mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * ρc - Rm
    return Rm * (latent(mg, T) - mg.Rv * T) / cvm - mg.Rv * T
end

"""
    MoistExnerModel(base::FVModel; mgas = MoistGas(), pvar = :pi, micro = :fast, τfac = 0.25, cs = 350.0,
                    wτ = nothing, τ = nothing)

Moist flat FV model with the Exner pressure (pvar = :pi) or the pressure (pvar = :p) as an
additional prognostic variable: state `[ρ; U; W; Θ; V; C; π]` (V = ρq_v, C = ρq_c) (`moist_exner_state`), frozen split for
`TimeStepping.run_frozen` (:verlet, :hevi, :rk4), π recomputed from the equation of state after every
MIS step. Condensation by the explicit Fischer–Burmeister relaxation, evaluated in every fast step
(micro = :fast, default), at the slow stages (:slow), or off (:none). Relaxation time: the given τ, or
τ = min(Δx, Δz)/wτ, or by default on the fast scale τ = τfac Δτ_ac with the acoustic step
Δτ_ac = 0.85·0.92 min(Δx, Δz)/cs of the fast loop (τfac = 0.25, the shortest relaxation that is stable in the Bryan–Fritsch bubble;
at τfac = 0.2 the bubble is already wrong). :slow needs τ ≳ Δt/2 (e.g. wτ = 50 m/s at 250 m). The base model provides grid, momentum advection and sponge; its gas constant g is the gravity.
"""
struct MoistExnerModel{M<:FVModel}
    base::M
    mgas::MoistGas
    pvar::Symbol                   # :pi (Exner pressure) or :p (pressure)
    micro::Symbol                  # :slow, :fast or :none
    τ::Float64                     # relaxation time of the phase change
    nq::Int
    C::Vector{Float64}
    S::Vector{Float64}
    φq::Matrix{Float64}            # q_v at the centres (work)
    φc::Matrix{Float64}            # q_c at the centres (work)
    abc::Array{Float64,3}          # ∂π/∂(ρ, Θ, V, C) per cell (work)
    column::ExnerColumn
    θq::Matrix{Float64}            # qv_f at the z-faces for the column solve
    θc::Matrix{Float64}            # qc_f at the z-faces for the column solve
end
function MoistExnerModel(base::FVModel; mgas::MoistGas = MoistGas(), pvar::Symbol = :pi,
                         micro::Symbol = :fast, τfac::Real = 0.25, cs::Real = 350.0, wτ = nothing, τ = nothing)
    pvar in (:pi, :p) || error("pvar: :pi or :p")
    micro in (:slow, :fast, :none) || error("micro: :slow, :fast or :none")
    _flat_only(base); _no_background(base)
    g = base.grid
    N, Nz = g.nx * g.nz, g.nx * (g.nz + 1)
    z1() = zeros(g.nx, g.nz + 1)
    h = min(g.dx, g.dz)
    τv = τ !== nothing ? Float64(τ) : wτ !== nothing ? h / wτ : τfac * 0.85 * 0.92 * h / cs
    return MoistExnerModel(base, mgas, pvar, micro, τv, base.nq + 3N, zeros(8N + 6Nz), zeros(base.nq + 3N),
                           zeros(g.nx, g.nz), zeros(g.nx, g.nz), zeros(4, g.nx, g.nz),
                           ExnerColumn(z1(), z1(), z1(), z1(), Dict{Float64,NTuple{3,Matrix{Float64}}}(), zeros(g.nz + 1)), z1(), z1())
end

@inline _cell(g, i, k) = ip(g, i) + (k - 1) * g.nx
@inline iV(m::MoistExnerModel, i, k) = m.base.nq + _cell(m.base.grid, i, k)
@inline iC(m::MoistExnerModel, i, k) = m.base.nq + m.base.grid.nx * m.base.grid.nz + _cell(m.base.grid, i, k)
@inline iπ(m::MoistExnerModel, i, k) = m.base.nq + 2 * m.base.grid.nx * m.base.grid.nz + _cell(m.base.grid, i, k)

"""The additional variable of the state: π (pvar = :pi) or p (pvar = :p) of the equation of state."""
moist_pvar(m::MoistExnerModel, ρ, Θ, ρv, ρc) = m.pvar === :p ? _moist_p(m.mgas, ρ, Θ, ρv, ρc) : moist_exner(m.mgas, ρ, Θ, ρv, ρc)
"""Moist state [q; V; C; π] of a base state q (ρ, U, W, Θ = ρθ_ρ) and the total water Q per cell,
split into vapour V and condensate C of phase equilibrium."""
function moist_exner_state(m::MoistExnerModel, q, Q)
    g = m.base.grid; mg = m.mgas
    Cc = [moist_state(mg, q[iρ(g, i, k)], q[iΘ(g, i, k)], Q[i, k])[3] for i in 1:g.nx, k in 1:g.nz]
    Vc = [Q[i, k] - Cc[i, k] for i in 1:g.nx, k in 1:g.nz]
    πc = [moist_pvar(m, q[iρ(g, i, k)], q[iΘ(g, i, k)], Vc[i, k], Cc[i, k]) for i in 1:g.nx, k in 1:g.nz]
    return vcat(q[1:m.base.nq], vec(Vc), vec(Cc), vec(πc))
end
base_state(m::MoistExnerModel, q) = q[1:m.base.nq]

# ------------------------------------------------------------------ coefficient vector
# x-faces (N each): θx, qvx, Θ̂x, κxL, κxR, qcx; z-faces (Nz each): θz, qvz, Θ̂z, κzB, κzT, qcz;
# cells (N each): ∂Θ/∂ρc|θe, ∂π/∂ρc along the phase change
frozen_ncoef(m::MoistExnerModel) = length(m.C)
@inline _mx(g, b, i, k) = b * _eN(g) + _e(g, i, k)
@inline _mz(g, b, i, k) = 6 * _eN(g) + b * _eNz(g) + _e(g, i, k)
@inline _mc(g, b, i, k) = 6 * _eN(g) + 6 * _eNz(g) + b * _eN(g) + _cell(g, i, k)
@inline mθx(C, g, i, k) = C[_mx(g, 0, i, k)];  @inline mqx(C, g, i, k) = C[_mx(g, 1, i, k)]
@inline mPx(C, g, i, k) = C[_mx(g, 2, i, k)];  @inline mκxL(C, g, i, k) = C[_mx(g, 3, i, k)]
@inline mκxR(C, g, i, k) = C[_mx(g, 4, i, k)]; @inline mcx(C, g, i, k) = C[_mx(g, 5, i, k)]
@inline mθz(C, g, i, k) = C[_mz(g, 0, i, k)];  @inline mqz(C, g, i, k) = C[_mz(g, 1, i, k)]
@inline mPz(C, g, i, k) = C[_mz(g, 2, i, k)];  @inline mκzB(C, g, i, k) = C[_mz(g, 3, i, k)]
@inline mκzT(C, g, i, k) = C[_mz(g, 4, i, k)]; @inline mcz(C, g, i, k) = C[_mz(g, 5, i, k)]
@inline mθS(C, g, i, k) = C[_mc(g, 0, i, k)];  @inline mκS(C, g, i, k) = C[_mc(g, 1, i, k)]

"""∂π/∂(ρ, Θ, V, C) (or ∂p, `f = _moist_p`) of the equation of state (central differences)."""
function exner_derivatives(mg::MoistGas, ρ, Θ, ρv, ρc; f = moist_exner)
    h = 1e-6
    dρ, dΘ, dQ = h * ρ, h * Θ, h * ρ
    return ((f(mg, ρ + dρ, Θ, ρv, ρc) - f(mg, ρ - dρ, Θ, ρv, ρc)) / 2dρ,
            (f(mg, ρ, Θ + dΘ, ρv, ρc) - f(mg, ρ, Θ - dΘ, ρv, ρc)) / 2dΘ,
            (f(mg, ρ, Θ, ρv + dQ, ρc) - f(mg, ρ, Θ, ρv - dQ, ρc)) / 2dQ,
            (f(mg, ρ, Θ, ρv, ρc + dQ) - f(mg, ρ, Θ, ρv, ρc - dQ)) / 2dQ)
end

"""Phase-change sources of cell (i, k) for the state z: S_c to C, -S_c to V, (∂Θ/∂ρc) S_c to Θ, κ_S S_c to π."""
@inline function _phase!(out, m::MoistExnerModel, C, z, i, k)
    g = m.base.grid
    Sc = -fischer_burmeister(m.mgas, z[iρ(g, i, k)], z[iΘ(g, i, k)], z[iV(m, i, k)], z[iC(m, i, k)])[1] / m.τ
    out[iC(m, i, k)] += Sc
    out[iV(m, i, k)] -= Sc
    out[iΘ(g, i, k)] += mθS(C, g, i, k) * Sc
    out[iπ(m, i, k)] += mκS(C, g, i, k) * Sc
    return out
end

"""C(y) and the slow part S(y) (momentum advection; phase change for micro = :slow)."""
function frozen_data!(C, S, m::MoistExnerModel, q)
    b = m.base; g = b.grid; mg = m.mgas
    nx, nz = g.nx, g.nz
    a = shift_advection!(b.cache.adv, g, q)
    f = m.pvar === :p ? _moist_p : moist_exner
    fill!(C, 0.0); fill!(S, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        ρ = q[iρ(g, i, k)]; Θ = q[iΘ(g, i, k)]; ρv = q[iV(m, i, k)]; ρc = q[iC(m, i, k)]
        m.φq[i, k] = ρv / ρ; m.φc[i, k] = ρc / ρ
        d = exner_derivatives(mg, ρ, Θ, ρv, ρc; f)
        for j in 1:4; m.abc[j, i, k] = d[j]; end
        T, _, _, p = moist_state(mg, ρ, Θ, ρv, ρc)
        C[_mc(g, 0, i, k)] = theta_phase_factor(mg, ρ, Θ, T, p, ρv, ρc)          # Q_Θ / S_c (Bryan–Fritsch)
        Qp = pressure_phase_factor(mg, ρ, T, ρv, ρc)                            # Q_p / S_c
        C[_mc(g, 1, i, k)] = m.pvar === :p ? Qp : (mg.Rd / mg.Cpd) * (p / mg.p0)^(mg.Rd / mg.Cpd) / p * Qp
    end
    # momentum coefficient: c_pd Θ̂, Θ̂ = p/(R_d π) with p = p0 π^(c_pd/R_d) (pvar = :pi), or 1 (pvar = :p)
    Θh(i, k) = m.pvar === :p ? 1.0 : (p = mg.p0 * q[iπ(m, i, k)]^(mg.Cpd / mg.Rd); mg.Cpd * p / (mg.Rd * q[iπ(m, i, k)]))
    κ(i, k, θf, qf, cf) = (ii = ip(g, i); m.abc[1, ii, k] + m.abc[2, ii, k] * θf + m.abc[3, ii, k] * qf + m.abc[4, ii, k] * cf)
    @inbounds for k in 1:nz, i in 1:nx
        θx = a.θfx[i, k]; F = a.Fx[i, k]
        qx = aface_x(g, m.φq, i, k, F); cx = aface_x(g, m.φc, i, k, F)
        C[_mx(g, 0, i, k)] = θx; C[_mx(g, 1, i, k)] = qx; C[_mx(g, 5, i, k)] = cx
        C[_mx(g, 2, i, k)] = 0.5 * (Θh(i - 1, k) + Θh(i, k))
        C[_mx(g, 3, i, k)] = κ(i - 1, k, θx, qx, cx); C[_mx(g, 4, i, k)] = κ(i, k, θx, qx, cx)
        S[iU(g, i, k)] = a.SU[i, k]
        if k > 1
            θz = a.θfz[i, k]; Fz = a.Fz[i, k]
            qz = aface_z(g, m.φq, i, k, Fz); cz = aface_z(g, m.φc, i, k, Fz)
            C[_mz(g, 0, i, k)] = θz; C[_mz(g, 1, i, k)] = qz; C[_mz(g, 5, i, k)] = cz
            C[_mz(g, 2, i, k)] = 0.5 * (Θh(i, k - 1) + Θh(i, k))
            C[_mz(g, 3, i, k)] = κ(i, k - 1, θz, qz, cz); C[_mz(g, 4, i, k)] = κ(i, k, θz, qz, cz)
            S[iW(g, i, k)] = a.SW[i, k]
        end
        m.micro === :slow && _phase!(S, m, C, q, i, k)
    end
    return C, S
end

# ------------------------------------------------------------------ fast part
function mthermo_h!(out, m::MoistExnerModel, C, z)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        Ul, Ur = z[iU(g, i, k)], z[iU(g, i + 1, k)]
        out[iρ(g, i, k)] -= (Ur - Ul) / g.dx
        out[iΘ(g, i, k)] -= (mθx(C, g, i + 1, k) * Ur - mθx(C, g, i, k) * Ul) / g.dx
        out[iV(m, i, k)] -= (mqx(C, g, i + 1, k) * Ur - mqx(C, g, i, k) * Ul) / g.dx
        out[iC(m, i, k)] -= (mcx(C, g, i + 1, k) * Ur - mcx(C, g, i, k) * Ul) / g.dx
        out[iπ(m, i, k)] -= (mκxL(C, g, i + 1, k) * Ur - mκxR(C, g, i, k) * Ul) / g.dx
    end
    return out
end
function mthermo_v!(out, m::MoistExnerModel, C, z)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        Wb, Wt = Wface(g, z, i, k), Wface(g, z, i, k + 1)
        out[iρ(g, i, k)] -= (Wt - Wb) / g.dz
        out[iΘ(g, i, k)] -= (mθz(C, g, i, k + 1) * Wt - mθz(C, g, i, k) * Wb) / g.dz
        out[iV(m, i, k)] -= (mqz(C, g, i, k + 1) * Wt - mqz(C, g, i, k) * Wb) / g.dz
        out[iC(m, i, k)] -= (mcz(C, g, i, k + 1) * Wt - mcz(C, g, i, k) * Wb) / g.dz
        out[iπ(m, i, k)] -= (mκzB(C, g, i, k + 1) * Wt - mκzT(C, g, i, k) * Wb) / g.dz
    end
    return out
end
function mmomentum_h!(out, m::MoistExnerModel, C, z)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        out[iU(g, i, k)] -= mPx(C, g, i, k) * (z[iπ(m, i, k)] - z[iπ(m, i - 1, k)]) / g.dx
    end
    return out
end
function mmomentum_v!(out, m::MoistExnerModel, C, z)
    g = m.base.grid; grav = m.base.gas.g
    @inbounds for k in 2:g.nz, i in 1:g.nx
        out[iW(g, i, k)] -= mPz(C, g, i, k) * (z[iπ(m, i, k)] - z[iπ(m, i, k - 1)]) / g.dz +
                            grav * 0.5 * (z[iρ(g, i, k - 1)] + z[iρ(g, i, k)])
    end
    return out
end
function mforcing_thermo!(out, m::MoistExnerModel, F)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        for j in (iρ(g, i, k), iΘ(g, i, k), iV(m, i, k), iC(m, i, k), iπ(m, i, k))
            out[j] += F[j]
        end
    end
    return out
end
"""Phase change from the fast state (micro = :fast), explicit in every small step."""
function mphase!(out, m::MoistExnerModel, C, z)
    m.micro === :fast || return out
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        _phase!(out, m, C, z, i, k)
    end
    return out
end

_mfast!(::Val{:all}, out, m, C, z, F) = (mthermo_h!(out, m, C, z); mthermo_v!(out, m, C, z); mphase!(out, m, C, z);
    mmomentum_h!(out, m, C, z); mmomentum_v!(out, m, C, z);
    mforcing_thermo!(out, m, F); forcing_U!(out, m.base, F); forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_mfast!(::Val{:thermo}, out, m, C, z, F) = (mthermo_h!(out, m, C, z); mthermo_v!(out, m, C, z); mphase!(out, m, C, z); mforcing_thermo!(out, m, F))
_mfast!(::Val{:momentum}, out, m, C, z, F) = (mmomentum_h!(out, m, C, z); mmomentum_v!(out, m, C, z);
    forcing_U!(out, m.base, F); forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_mfast!(::Val{:hthermo}, out, m, C, z, F) = (mthermo_h!(out, m, C, z); mphase!(out, m, C, z); mforcing_thermo!(out, m, F);
    forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_mfast!(::Val{:hmomentum}, out, m, C, z, F) = (mmomentum_h!(out, m, C, z); forcing_U!(out, m.base, F))
_mfast!(::Val{:vertical}, out, m, C, z, F) = (mthermo_v!(out, m, C, z); mmomentum_v!(out, m, C, z))
_mfast!(::Val{P}, out, m, C, z, F) where {P} = error("frozen_fast!: unknown part $P")

function frozen_fast!(out, m::MoistExnerModel, C, z, part::Symbol, F)
    fill!(out, 0.0)
    _mfast!(Val(part), out, m, C, z, F)
    return out
end

function rhs_nonlinear!(out, m::MoistExnerModel, q::AbstractVector)
    frozen_data!(m.C, m.S, m, q)
    return frozen_fast!(out, m, m.C, q, :all, m.S)
end

"""π ← π(ρ, Θ, V, C) of the equation of state after a MIS step (no saturation adjustment)."""
function frozen_step_end!(m::MoistExnerModel, q)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        q[iπ(m, i, k)] = moist_pvar(m, q[iρ(g, i, k)], q[iΘ(g, i, k)], q[iV(m, i, k)], q[iC(m, i, k)])
    end
    return q
end

# ------------------------------------------------------------------ HEVI column solve (as exner.jl, κ in place of c θ)
function frozen_stage_begin!(m::MoistExnerModel, C, z)
    s = m.column; g = m.base.grid
    @inbounds for k in 1:g.nz+1, i in 1:g.nx
        s.P[i, k] = mPz(C, g, i, k); s.cB[i, k] = mκzB(C, g, i, k); s.cT[i, k] = mκzT(C, g, i, k)
        s.θz[i, k] = mθz(C, g, i, k); m.θq[i, k] = mqz(C, g, i, k); m.θc[i, k] = mcz(C, g, i, k)
    end
    empty!(s.factors)
    return s
end

function moist_column_solve!(Y, m::MoistExnerModel, c, R)
    s = m.column; g = m.base.grid
    nz, dz, grav = g.nz, g.dz, m.base.gas.g
    sub, cpr, iden = _exner_factor(s, m, c)
    W = s.W
    @inbounds for i in 1:g.nx
        for k in 2:nz
            rhs = R[iW(g, i, k)] - (c / dz) * s.P[i, k] * (R[iπ(m, i, k)] - R[iπ(m, i, k - 1)]) -
                  (c * grav / 2) * (R[iρ(g, i, k - 1)] + R[iρ(g, i, k)])
            W[k] = k == 2 ? rhs * iden[k, i] : (rhs - sub[k, i] * W[k-1]) * iden[k, i]
        end
        for k in nz-1:-1:2
            W[k] -= cpr[k, i] * W[k+1]
        end
        W[1] = 0.0; W[nz+1] = 0.0
        for k in 1:nz
            Y[iρ(g, i, k)] = R[iρ(g, i, k)] - (c / dz) * (W[k+1] - W[k])
            Y[iΘ(g, i, k)] = R[iΘ(g, i, k)] - (c / dz) * (s.θz[i, k+1] * W[k+1] - s.θz[i, k] * W[k])
            Y[iV(m, i, k)] = R[iV(m, i, k)] - (c / dz) * (m.θq[i, k+1] * W[k+1] - m.θq[i, k] * W[k])
            Y[iC(m, i, k)] = R[iC(m, i, k)] - (c / dz) * (m.θc[i, k+1] * W[k+1] - m.θc[i, k] * W[k])
            Y[iπ(m, i, k)] = R[iπ(m, i, k)] - (c / dz) * (s.cB[i, k+1] * W[k+1] - s.cT[i, k] * W[k])
            Y[iU(g, i, k)] = R[iU(g, i, k)]
        end
        for k in 2:nz
            Y[iW(g, i, k)] = W[k]
        end
    end
    return Y
end

function frozen_vertical_step!(z, m::MoistExnerModel, C, dt, k, y)
    frozen_fast!(k, m, C, z, :vertical, k)
    @. k *= dt
    moist_column_solve!(y, m, dt / 2, k)
    @. z += y
    return z
end
