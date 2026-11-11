# Fast integrators for q' = L q + f (L frozen, f constant during the fast steps)

# IMEX tableau (explicit part: 2-stage RK2; implicit part: SDIRK, γ = 1 - 1/√2)
const IMEX_GM = 1 - 1 / sqrt(2)
const IMEX_DL = 1 - 1 / (2IMEX_GM)
const IMEX_AE = [0.0 0.0 0.0; IMEX_GM 0.0 0.0; IMEX_DL (1 - IMEX_DL) 0.0]
const IMEX_AI = [0.0 0.0 0.0; 0.0 IMEX_GM 0.0; 0.0 (1 - IMEX_GM) IMEX_GM]

"""Off-centring of the WRF vertically implicit acoustic step, β = (1 + epssm)/2 with epssm = 0.1."""
const WRF_BETA = 0.55

"""WRF divergence damping (smdiv): horizontal pressure gradient from p + γd (p - p_old)."""
const WRF_SMDIV = 0.1

"""
    fast_solve!(qout, s, q0, F_slow, dt, nsteps; method=:imex)

Integrate the fast system q' = L q + f_b/M + F_slow over nsteps steps of size dt, starting from
q0 (qout may alias q0). Methods:
  :imex    explicit 2-stage RK2 of the IMEX tableau (needs `fast_apply!`)
  :hevi    IMEX: horizontal explicit, vertical implicit (horizontal/vertical split); FV
           overrides it with the Strang Verlet/Crank–Nicolson composition below
  :hevi_verlet  Strang: horizontal Störmer–Verlet half step, vertical Crank–Nicolson,
           horizontal half step (needs `fast_rhs_hthermo!`, `fast_rhs_hmomentum!`)
  :verlet  Störmer–Verlet: (ρ, Θ) half step, momentum full step, (ρ, Θ) half step
  :fb      forward-backward: momentum explicit, interface implicit, then (ρ, Θ)
  :wrf     WRF acoustic step: horizontal momentum forward, then (ρ, Θ, vertical momentum)
           vertically implicit, off-centred Crank–Nicolson (β = WRF_BETA), divergence damping
           by the pressure extrapolation of WRF (γd = WRF_SMDIV)
  :rosenbrock_amf model-specific Rosenbrock-W fast integration with approximate matrix factors
  :hevi_amf model-specific HEVI with approximate matrix factors in the implicit step
  :<base>_y4, :<base>_s4, :<base>_y6  symmetric compositions of order 4/4/6 of a symmetric
           second-order loop (:verlet, FV :hevi, :hevi_verlet), see `composition`
The standard explicit and HEVI loops are allocation-free for operators with a cached
`fast_workspace` (7 vectors); model-specific Rosenbrock-W implementations may build additional
factors when their fast operator is frozen.
"""
function fast_solve!(qout, s::AbstractFastOperator, q0, F_slow, dt, nsteps; method::Symbol = :imex)
    f, q, Y, k1, k2, tmp, rhs = fast_workspace(s)
    fb = fast_forcing(s)
    if F_slow === nothing
        @. f = fb
    else
        M = fast_mass(s)
        @. f = fb + M * F_slow
    end
    base, ws = composition(method)
    if base == :rosenbrock_amf
        length(ws) == 1 || error("no composition of :rosenbrock_amf")
        fast_rosenbrock_amf!(qout, s, q0, f, dt, nsteps)
        return qout
    elseif base == :hevi_amf
        length(ws) == 1 || error("no composition of :hevi_amf")
        fast_hevi_amf!(qout, s, q0, f, dt, nsteps)
        return qout
    end
    q .= q0
    if length(ws) == 1
        _fast_loop!(base, q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    else
        for _ in 1:nsteps, w in ws
            _fast_loop!(base, q, s, f, w * dt, 1, Y, k1, k2, tmp, rhs)
        end
    end
    qout .= q
    return qout
end

"""
    fast_solve_poly!(qout, s, q0, G, T, nsteps; method)

Integrate q' = L q + f_b/M + Σ_k (t/T)^{k-1} G_k over [0, T] in nsteps substeps (forcing of an
ETD stage). Each base step of the loop (also inside a composition) uses the forcing at its
midpoint in time, which keeps the symmetric loops symmetric in (q, t): the compositions
`_s4`, `_y4`, `_y6` stay of order 4 and 6. `G` is a vector of vectors (unscaled, like F_slow).
"""
function fast_solve_poly!(qout, s::AbstractFastOperator, q0, G, T, nsteps; method::Symbol = :imex)
    f, q, Y, k1, k2, tmp, rhs = fast_workspace(s)
    fb = fast_forcing(s)
    M = fast_mass(s)
    base, ws = composition(method)
    base in (:rosenbrock_amf, :hevi_amf) && error("fast_solve_poly!: :$base has no time-dependent forcing")
    K = length(G)
    q .= q0
    dt = T / nsteps
    t = 0.0
    for _ in 1:nsteps, w in ws
        τ = (t + w * dt / 2) / T
        f .= G[K]
        for k in K-1:-1:1                                # Horner
            @. f = τ * f + G[k]
        end
        @. f = fb + M * f
        _fast_loop!(base, q, s, f, w * dt, 1, Y, k1, k2, tmp, rhs)
        t += w * dt
    end
    qout .= q
    return qout
end

# Symmetric compositions (Yoshida 1990, Suzuki 1990) of a symmetric second-order fast loop
const YOSHIDA4 = (w = 1 / (2 - cbrt(2)); (w, 1 - 2w, w))
const SUZUKI4 = (p = 1 / (4 - cbrt(4)); (p, p, 1 - 4p, p, p))
const YOSHIDA6 = (w1 = -1.17767998417887; w2 = 0.235573213359357; w3 = 0.784513610477560;
                  w0 = 1 - 2(w1 + w2 + w3); (w3, w2, w1, w0, w1, w2, w3))

"""
    composition(method) -> (base, weights)

`:<base>_y4` (Yoshida triple jump, weights 1.35, -1.70, 1.35), `:<base>_s4` (Suzuki, 5 stages,
|w| ≤ 0.66) and `:<base>_y6` (Yoshida 6th order, 7 stages, |w| ≤ 1.32) compose the symmetric
second-order loop `base` (:verlet, :hevi for FV, :hevi_verlet) to order 4 or 6: one fast substep
of size dτ is the product of base steps of size w_k dτ. The base loop must be stable for
max|w_k| dτ (see `composition_stretch`). Other methods: (method, (1.0,)).
"""
function composition(method::Symbol)
    c = get(COMPOSITIONS, method, nothing)
    return c === nothing ? (method, NO_COMPOSITION) : c
end

const NO_COMPOSITION = [1.0]
const COMPOSITIONS = Dict{Symbol,Tuple{Symbol,Vector{Float64}}}(
    Symbol(base, suf) => (base, collect(ws))
    for base in (:verlet, :hevi, :hevi_verlet, :hevi_relax, :imex, :fb)
    for (suf, ws) in (("_y4", YOSHIDA4), ("_s4", SUZUKI4), ("_y6", YOSHIDA6)))

"""max|w_k| of the composition: the base loop sees substeps up to this factor times dτ."""
composition_stretch(method::Symbol) = maximum(abs, composition(method)[2])

"""Base fast steps per substep of a composition (cost factor)."""
composition_cost(method::Symbol) = length(composition(method)[2])

function _fast_loop!(method, q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    if method == :imex
        imex_loop!(q, s, f, dt, nsteps, Y, k1, k2)
    elseif method == :hevi
        hevi_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    elseif method == :hevi_verlet
        hevi_verlet_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    elseif method == :hevi_relax
        hevi_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs; relax_horizontal = true)
    elseif method == :verlet
        verlet_loop!(q, s, f, dt, nsteps, k1, k2)
    elseif method == :fb
        fb_loop!(q, s, f, dt, nsteps, k1, k2)
    elseif method == :wrf
        wrf_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs)
    else
        error("unknown fast method $method (available: :imex, :hevi, :hevi_verlet, :hevi_relax, :verlet, :fb, :wrf, :rosenbrock_amf, :hevi_amf, and _y4/_s4/_y6 compositions of the symmetric ones)")
    end
    return q
end

function imex_loop!(q, s, f, dt, nsteps, Y, k1, k2)
    a21 = IMEX_AE[2, 1]
    a31 = IMEX_AE[3, 1]
    a32 = IMEX_AE[3, 2]
    for _ in 1:nsteps
        fast_apply!(k1, s, q, f)
        @. Y = q + (dt * a21) * k1
        fast_apply!(k2, s, Y, f)
        @. q = q + dt * (a31 * k1 + a32 * k2)
    end
    return q
end

"""HEVI with the IMEX tableau: horizontal part explicit, vertical part implicit."""
function hevi_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs; relax_horizontal = false)
    F = vertical_factor(s, dt * IMEX_GM)
    a21 = IMEX_AE[2, 1]
    a31 = IMEX_AE[3, 1]
    a32 = IMEX_AE[3, 2]
    i32 = IMEX_AI[3, 2]
    for _ in 1:nsteps
        if relax_horizontal
            fast_rhs_horizontal_relaxed!(k1, s, q, f, dt)
        else
            fast_rhs_horizontal!(k1, s, q, f)
        end
        @. rhs = q + (dt * a21) * k1
        vertical_solve!(Y, s, F, rhs)                    # implicit stage 2
        if relax_horizontal
            fast_rhs_horizontal_relaxed!(k2, s, Y, f, dt)
        else
            fast_rhs_horizontal!(k2, s, Y, f)
        end
        vertical_apply!(tmp, s, Y)
        @. rhs = q + dt * (a31 * k1 + a32 * k2 + i32 * tmp)
        vertical_solve!(q, s, F, rhs)                    # stage 3 = new state
    end
    return q
end

"""
Horizontally explicit Störmer–Verlet, vertically Crank–Nicolson (Strang composition). The
explicit RK2 stages of `hevi` are unstable on the imaginary axis, so a centred horizontal
acoustic operator only survives there through upwind dissipation; Verlet is stable on the
imaginary axis up to |λ dt| = 2.
"""
function hevi_verlet_loop!(q, s, f, dt, nsteps, Y, k1, k2, tmp, rhs; forcing::Symbol = HEVI_VERLET_FORCING[])
    F = vertical_factor(s, dt / 2)
    halfdt = dt / 2
    # forcing = :vertical adds f in the implicit vertical step, so stiff vertical modes relax to
    # the forced equilibrium -A_v⁻¹ f; :horizontal adds it in the explicit horizontal half steps.
    fh = forcing == :horizontal ? f : zero(f)
    fv = forcing == :vertical ? 1.0 : 0.0
    for _ in 1:nsteps
        fast_rhs_hthermo!(k1, s, q, fh);   @. q += (halfdt / 2) * k1
        fast_rhs_hmomentum!(k2, s, q, fh); @. q += halfdt * k2
        fast_rhs_hthermo!(k1, s, q, fh);   @. q += (halfdt / 2) * k1
        vertical_apply!(tmp, s, q)                     # (I - dt/2 A_v) q⁺ = (I + dt/2 A_v) q⁻ + dt f
        @. rhs = q + (dt / 2) * tmp + (fv * dt) * f
        vertical_solve!(Y, s, F, rhs)
        q .= Y
        fast_rhs_hthermo!(k1, s, q, fh);   @. q += (halfdt / 2) * k1
        fast_rhs_hmomentum!(k2, s, q, fh); @. q += halfdt * k2
        fast_rhs_hthermo!(k1, s, q, fh);   @. q += (halfdt / 2) * k1
    end
    return q
end

"""Placement of the constant fast forcing in `:hevi_verlet` (:vertical or :horizontal)."""
const HEVI_VERLET_FORCING = Ref(:horizontal)

"""Störmer–Verlet (Xu, van der Vegt, Bokhove 2008, eq. 4.6 with f = 0)."""
function verlet_loop!(q, s, f, dt, nsteps, kT, km)
    for _ in 1:nsteps
        interface_update!(s, q)
        fast_rhs_thermo!(kT, s, q, f)
        @. q += (dt / 2) * kT
        fast_rhs_momentum!(km, s, q, f)
        @. q += dt * km
        interface_update!(s, q)
        fast_rhs_thermo!(kT, s, q, f)
        @. q += (dt / 2) * kT
    end
    return q
end

"""Forward-backward Euler: momentum explicit, interface variables implicit (relaxed), then (ρ, Θ)."""
function fb_loop!(q, s, f, dt, nsteps, kT, km)
    interface_update!(s, q)
    for _ in 1:nsteps
        fast_rhs_momentum!(km, s, q, f)
        @. q += dt * km
        interface_update!(s, q; relax = true, dt = dt)
        fast_rhs_thermo!(kT, s, q, f)
        @. q += dt * kT
    end
    return q
end

"""
WRF acoustic step (Klemp, Skamarock, Dudhia 2007): horizontal momentum forward with the old
thermodynamic state extrapolated by the divergence damping, qⁿ + γd (qⁿ - qⁿ⁻¹), then (ρ, Θ,
vertical momentum) with the new horizontal fluxes,
vertically implicit with off-centring β:
    (I - β dt A_v) Yⁿ⁺¹ = Yⁿ + dt [(1 - β) A_v Yⁿ + H(qⁿ with new horizontal momentum) + f]
"""
function wrf_loop!(q, s, f, dt, nsteps, qold, kh, kt, tmp, rhs)
    β = WRF_BETA
    γd = WRF_SMDIV
    F = vertical_factor(s, β * dt)
    qold .= q
    for _ in 1:nsteps
        @. tmp = q + γd * (q - qold)                     # divergence damping (pressure extrapolation)
        qold .= q
        fast_rhs_hmomentum!(kh, s, tmp, f)
        @. q += dt * kh                                  # horizontal momentum forward
        fast_rhs_hthermo!(kt, s, q, f)                   # horizontal divergence with the new momentum
        vertical_apply!(tmp, s, q)
        @. rhs = q + dt * (kt + (1 - β) * tmp)
        vertical_solve!(q, s, F, rhs)
    end
    return q
end
