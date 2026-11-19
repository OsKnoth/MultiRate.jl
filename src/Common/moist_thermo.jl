# Thermodynamics of moist air (Bryan & Fritsch 2002, constants and e_s(T) as in CGDycore), shared by the FV and the
# DG moist models: MoistGas with the thermodynamic variable X (ρθ_ρ, ρe or ρs), equation of state, phase
# equilibrium (initial states only), Fischer–Burmeister function of the condensation relaxation and the phase factors
# of the X and pressure equations. Formulation: src/FV/moist.jl.

Base.@kwdef struct MoistGas
    Rd::Float64 = 287.0
    Rv::Float64 = 461.0
    Cpd::Float64 = 1004.0
    Cpv::Float64 = 1885.0
    Cpl::Float64 = 4186.0
    L0V::Float64 = 2.5e6
    T0::Float64 = 273.15
    p0::Float64 = 1.0e5
    thvar::Symbol = :thetarho      # thermodynamic variable X: :thetarho (ρθ_ρ), :energy (internal energy ρe) or :entropy (ρs)
end
"""Energy constant of the vapour: e_v = c_vv T + e0v, so that L(T) = h_v - h_l = e0v + (c_pv - c_pl) T."""
@inline _e0v(mg::MoistGas) = mg.L0V - (mg.Cpv - mg.Cpl) * mg.T0

"""Saturation vapour pressure over water (CGDycore `fpws`)."""
@inline psat(mg::MoistGas, T) = 661.2 * exp(17.368 * (T - mg.T0) / (238.88 + T - mg.T0))

"""x ln(x/r) (0 for x ≤ 0)."""
@inline _xlog(x, r) = x > 0 ? x * log(x / r) : 0.0
"""Density part B of the moist entropy, ρs = c_vm ln(T/T0) + B (thvar = :entropy)."""
@inline function _sB(mg::MoistGas, ρd, ρv)
    ρd0 = mg.p0 / (mg.Rd * mg.T0); ρv0 = psat(mg, mg.T0) / (mg.Rv * mg.T0)
    return -mg.Rd * _xlog(ρd, ρd0) - mg.Rv * _xlog(ρv, ρv0) + mg.L0V / mg.T0 * ρv
end
"""Latent heat of vaporization L(T) = L0V + (c_pv - c_pl)(T - T0)."""
@inline latent(mg::MoistGas, T) = mg.L0V + (mg.Cpv - mg.Cpl) * (T - mg.T0)

"""Temperature for given (ρ, Θ, Q) and vapour density ρv (CGDycore `fTemp`)."""
@inline function _temp(mg::MoistGas, ρ, Θ, Q, ρv)
    ρd = ρ - Q
    Rm = mg.Rd * ρd + mg.Rv * ρv
    if mg.thvar === :energy                # ρe = c_vm T + e0v ρ_v, c_vm = c_vd ρ_d + c_vv ρ_v + c_pl ρ_c
        cvm = (mg.Cpd - mg.Rd) * ρd + (mg.Cpv - mg.Rv) * ρv + mg.Cpl * (Q - ρv)
        return (Θ - _e0v(mg) * ρv) / cvm, Rm
    end
    if mg.thvar === :entropy               # ρs = c_vm ln(T/T0) + B(ρ_d, ρ_v)
        cvm = (mg.Cpd - mg.Rd) * ρd + (mg.Cpv - mg.Rv) * ρv + mg.Cpl * (Q - ρv)
        return mg.T0 * exp((Θ - _sB(mg, ρd, ρv)) / cvm), Rm
    end
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

"""Thermodynamic variable of (ρ, T, ρv, ρc): Θ = ρθ_ρ (thvar = :thetarho), the internal energy ρe (:energy)
or the moist entropy ρs (:entropy)."""
@inline function _Θof(mg::MoistGas, ρ, T, ρv, ρc)
    ρd = ρ - ρv - ρc
    Rm = mg.Rd * ρd + mg.Rv * ρv
    mg.thvar === :energy && return ((mg.Cpd - mg.Rd) * ρd + (mg.Cpv - mg.Rv) * ρv + mg.Cpl * ρc) * T + _e0v(mg) * ρv
    mg.thvar === :entropy && return ((mg.Cpd - mg.Rd) * ρd + (mg.Cpv - mg.Rv) * ρv + mg.Cpl * ρc) * log(T / mg.T0) + _sB(mg, ρd, ρv)
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
:energy: 0. :entropy: dρs/dρ_c = c_vm/T dT/dρ_c + ∂ρs/∂ρ_c - ∂ρs/∂ρ_v (at fixed T)
    = (L - R_v T)/T + (c_pl - c_vv) ln(T/T0) + R_v (ln(ρ_v/ρ_v0) + 1) - L0V/T0
(zero in phase equilibrium at T0: no entropy production at equilibrium).
"""
function theta_phase_factor(mg::MoistGas, ρ, Θ, T, p, ρv, ρc)
    mg.thvar === :energy && return 0.0     # the internal energy does not change in an isochoric phase change
    ρd = ρ - ρv - ρc
    if mg.thvar === :entropy
        ρv0 = psat(mg, mg.T0) / (mg.Rv * mg.T0)
        return (latent(mg, T) - mg.Rv * T) / T + (mg.Cpl - mg.Cpv + mg.Rv) * log(T / mg.T0) +
               mg.Rv * (log(max(ρv, 1e-12 * ρv0) / ρv0) + 1) - mg.L0V / mg.T0
    end
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

"""∂π/∂(ρ, Θ, V, C) (or ∂p, `f = _moist_p`) of the equation of state (central differences)."""
function exner_derivatives(mg::MoistGas, ρ, Θ, ρv, ρc; f = moist_exner)
    h = 1e-6
    dρ, dΘ, dQ = h * ρ, mg.thvar === :entropy ? h * mg.Cpd * ρ : h * Θ, h * ρ   # ρs may be near zero
    return ((f(mg, ρ + dρ, Θ, ρv, ρc) - f(mg, ρ - dρ, Θ, ρv, ρc)) / 2dρ,
            (f(mg, ρ, Θ + dΘ, ρv, ρc) - f(mg, ρ, Θ - dΘ, ρv, ρc)) / 2dΘ,
            (f(mg, ρ, Θ, ρv + dQ, ρc) - f(mg, ρ, Θ, ρv - dQ, ρc)) / 2dQ,
            (f(mg, ρ, Θ, ρv, ρc + dQ) - f(mg, ρ, Θ, ρv, ρc - dQ)) / 2dQ)
end

