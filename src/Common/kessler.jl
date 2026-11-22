# Warm-rain Kessler microphysics after DCMIP 2016 (kessler.f90, Klemp & Wilhelmson 1978, KW below) for the moist
# models with prognostic ρq_v, ρq_c and rain ρq_r. As decided for MultiRate, vapour ↔ cloud water stays the
# Fischer–Burmeister relaxation (no saturation adjustment); Kessler adds the rain processes
#     autoconversion + accretion   ρ_c → ρ_r,   rate ρ_d [k1 max(r_c − a, 0) + k2 r_c r_r^0.875]      (KW 2.13)
#     rain evaporation             ρ_r → ρ_v,   rate ρ_d E,                                            (KW 2.14)
#         E = (1.6 + 124.9 (ρ̄ r_r)^0.2046)(ρ̄ r_r)^0.525 / (5.4e5 + 2.55e6/(p_mb r_vs)) · max(r_vs − r_v, 0)/(ρ̄ r_vs)
#     sedimentation with the terminal velocity  V_r = 36.34 (ρ̄ r_r)^0.1364 √(ρ_d,s/ρ_d)                (KW 2.15)
# with the mixing ratios r_x = ρ_x/ρ_d of DCMIP, ρ̄ = 0.001 ρ_d (g cm⁻³), p_mb the pressure in hPa and ρ_d,s the dry
# density at the bottom of the column. r_vs = ρ_vs/ρ_d with ρ_vs = e_s(T)/(R_v T) of `psat` (the saturation of the
# Fischer–Burmeister relaxation, not the Teten formula of kessler.f90).

"""
    KesslerParams(; k1 = 1e-3, a = 1e-3, k2 = 2.2, cv = 36.34, τr = 10.0)

Constants of the warm-rain Kessler scheme (DCMIP 2016): autoconversion rate `k1` (s⁻¹) and threshold `a` (kg/kg),
accretion constant `k2`, terminal velocity constant `cv` (m/s). `τr` (s) bounds the explicit rates for positivity:
conversion ≤ ρ_c/τr, evaporation ≤ min(ρ_r, max(ρ_vs − ρ_v − ρ_c, 0))/τr (DCMIP limits per time step; choose τr ≳ the
time step of the slow stages).
"""
struct KesslerParams
    k1::Float64
    a::Float64
    k2::Float64
    cv::Float64
    τr::Float64
end
KesslerParams(; k1 = 1e-3, a = 1e-3, k2 = 2.2, cv = 36.34, τr = 10.0) = KesslerParams(k1, a, k2, cv, τr)

"""Terminal velocity V_r (m/s, positive downwards) of rain, KW 2.15; ρds the dry density at the bottom of the column."""
@inline function rain_velocity(kp::KesslerParams, ρd, ρr, ρds)
    ρr > 0 || return 0.0
    return kp.cv * (1e-3 * ρr)^0.1364 * sqrt(ρds / ρd)            # ρ̄ r_r = 0.001 ρ_d r_r = 0.001 ρ_r
end

"""
    kessler_rates(kp, mg, ρ, T, p, ρv, ρc, ρr) -> (P, E)

Conversion rate P (ρ_c → ρ_r, autoconversion + accretion) and rain evaporation rate E (ρ_r → ρ_v), both in
kg m⁻³ s⁻¹ and limited by τr (see `KesslerParams`).
"""
@inline function kessler_rates(kp::KesslerParams, mg::MoistGas, ρ, T, p, ρv, ρc, ρr)
    ρd = ρ - ρv - ρc - ρr
    rc = max(ρc, 0.0) / ρd; rr = max(ρr, 0.0) / ρd
    P = ρd * (kp.k1 * max(rc - kp.a, 0.0) + kp.k2 * rc * rr^0.875)
    P = min(P, max(ρc, 0.0) / kp.τr)
    ρvs = psat(mg, T) / (mg.Rv * T)
    E = 0.0
    if ρr > 0 && ρv < ρvs
        rvs = ρvs / ρd; rv = ρv / ρd; ρrb = 1e-3 * ρr               # ρ̄ r_r (g cm⁻³)
        E = ρd * (1.6 + 124.9 * ρrb^0.2046) * ρrb^0.525 / (5.4e5 + 2.55e6 / (1e-2 * p * rvs)) *
            (rvs - rv) / (1e-3 * ρd * rvs)
        E = min(E, min(ρr, max(ρvs - ρv - ρc, 0.0)) / kp.τr)
    end
    return P, E
end

"""
    liquid_content(mg, ρ, X, T, p, ρv, ρl)

∂X/∂ρ_l at fixed T, ρ_d and ρ_v: the amount of the thermodynamic variable X carried by a unit mass of liquid water
at the temperature T (falling rain leaves the air temperature unchanged): c_pl T (ρe), c_pl ln(T/T0) (ρs),
X ln(p/p0) R_m c_pl/c_pm² (ρθ_ρ).
"""
@inline function liquid_content(mg::MoistGas, ρ, X, T, p, ρv, ρl)
    mg.thvar === :energy && return mg.Cpl * T
    mg.thvar === :entropy && return mg.Cpl * log(T / mg.T0)
    ρd = ρ - ρv - ρl
    Rm = mg.Rd * ρd + mg.Rv * ρv; cpm = mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * ρl
    return X * log(p / mg.p0) * Rm * mg.Cpl / cpm^2
end

"""
    fischer_burmeister(mg, ρ, Θ, ρv, ρc, ρr) -> (φ, T)

Fischer–Burmeister function of the cloud water with rain present: T from the total liquid ρ_c + ρ_r, φ_FB of
a = ρ_vs(T) − ρ_v and b = ρ_c (rain does not take part in the relaxation).
"""
@inline function fischer_burmeister(mg::MoistGas, ρ, Θ, ρv, ρc, ρr)
    T = moist_state(mg, ρ, Θ, ρv, ρc + ρr)[1]
    a = psat(mg, T) / (mg.Rv * T) - ρv
    return 0.5 * (a + ρc - sqrt(a * a + ρc * ρc)), T
end
