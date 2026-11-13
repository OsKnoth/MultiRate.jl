# MIS with a frozen part of the right-hand side (no linearized fast operator)
#
# The right-hand side is written with two arguments, N(q) = F(q, q), the first one frozen at the
# stage values, the second one evolving in the fast integration (Knoth & Wensch, "B-series and
# split-explicit methods", github.com/OsKnoth/B-Series):
#     Z_i' = Σ_j a_ij F(Y_j, Z_i),   Z_i(0) = qⁿ + Σ_j α_ij (Y_j - qⁿ).
# The model splits F(y, z) = F_f(C(y), z) + S(y): the slow part S depends on y only, the fast
# part F_f is nonlinear in z with coefficients C(y) frozen from y and linear in C, so that
# Σ_j w_j F(Y_j, Z) = F_f(Σ_j w_j C(Y_j), Z) + Σ_j w_j S(Y_j).
#
# Model interface:
#   frozen_ncoef(model)                          length of the coefficient vector C
#   frozen_data!(C, S, model, y)                 C(y) and S(y)
#   frozen_fast!(out, model, C, z, part, F)      out = F_f(C, z) + F on the rows of part
#                                                (:thermo, :momentum or :all), zero elsewhere
# with N(q) = F_f(C(q), q) + S(q) exactly.

#   frozen_stage_begin!(model, C, z)             (:hevi) prepare the vertical solver of the stage
#   frozen_vertical_step!(z, model, C, dt, k, y) (:hevi) z ← z + (I - dt/2 J_v)⁻¹ dt G_v(z)
#   frozen_vertical_rhs!(out, model, C, z, ct, cw)      (:ei) out = D G_v(z)
#   frozen_vertical_implicit!(z, model, C, ct, cw, k, y) (:ei) z ← z + (I - D J_v)⁻¹ D G_v(z)
#       D = ct on the thermodynamic rows, cw on the W rows
#   frozen_vertical_solve!(y, model, C, ct, cw, R)       (:wrf) y = (I - D J_v)⁻¹ R
# with the parts :hthermo, :hmomentum (horizontal, with F) and :vertical (no F) of frozen_fast!.

function frozen_ncoef end
function frozen_data! end
function frozen_fast! end
frozen_stage_begin!(model, C, z) = nothing
function frozen_vertical_step! end
function frozen_vertical_rhs! end
function frozen_vertical_implicit! end
function frozen_vertical_solve! end
"""out = :hthermo of the implicit step, the horizontal mass flux with the predictor U + w (metric
increment at the old pressure) for models with an implicit metric term; default: plain :hthermo."""
frozen_hthermo!(out, model, C, z, F, w) = frozen_fast!(out, model, C, z, :hthermo, F)
#   frozen_step_end!(model, q)                   optional: after every MIS step (e.g. recompute a
#                                                redundant variable from the others)
frozen_step_end!(model, q) = nothing

"""Work arrays of the frozen MIS step: per stage C(Y_j), S(Y_j), ΔY_j = Y_{j+1} - qⁿ; combined
coefficients and forcing of the current stage; stage and fast-loop vectors."""
struct FrozenWork
    C::Vector{Vector{Float64}}
    S::Vector{Vector{Float64}}
    ΔY::Vector{Vector{Float64}}
    c::Vector{Float64}
    F::Vector{Float64}
    Y::Vector{Float64}
    Z::Vector{Float64}
    k::Vector{Float64}
end
FrozenWork(nq::Int, nc::Int, ns::Int) =
    FrozenWork([zeros(nc) for _ in 1:ns], [zeros(nq) for _ in 1:ns], [zeros(nq) for _ in 1:ns],
               zeros(nc), zeros(nq), zeros(nq), zeros(nq), zeros(nq))
FrozenWork(model, nq::Int, tab::SplitTableau) = FrozenWork(nq, frozen_ncoef(model), max(2, nstages(tab), _ncache(tab)))
_ncache(t::SplitTableau) = 0
_ncache(t::LawsonTableau) = length(t.nodes)

"""
    frozen_fast_solve!(Z, model, C, F, dt, nsteps; method = :verlet)

Integrate Z' = F_f(C, Z) + F over nsteps steps dt (C, F constant). :verlet Störmer–Verlet
(thermodynamic half step, momentum step, thermodynamic half step, as `SplitExplicit.m`),
:fb forward-backward (momentum, then thermodynamics), :rk4 classical RK4 on F_f + F,
:hevi Strang: horizontal Verlet half step, linearly implicit vertical Crank–Nicolson step
(`frozen_vertical_step!`), horizontal half step; :ei Verlet EI (Imex.pdf version 1 EI, see `ei_loop!`):
thermodynamic half step explicit in the full divergence, U explicit, W trapezoidal in the vertical
pressure gradient and gravity (off-centring `EI_GAMMA`), thermodynamic half step, one column solve
(`frozen_vertical_rhs!`, `frozen_vertical_implicit!`); for :gradp linearly implicit;
:wrf WRF acoustic step (Klemp, Skamarock & Dudhia 2007; forward-backward, vertically implicit):
U forward (with `smdiv` at the extrapolated thermodynamic state), then (ρ, Θ, W) with the new
horizontal fluxes, off-centred Crank–Nicolson in the vertical part (β = (1 + offcenter)/2, linearly
implicit for :gradp), one column solve (`frozen_vertical_rhs!`, `frozen_vertical_solve!`);
compositions `:verlet_s4`, `_y4`, `_y6` (see `composition`).

`smdiv = β > 0`: divergence damping by forward extrapolation of the thermodynamic variables in the
momentum part (as WRF, `WRF_SMDIV`): the momentum rows are evaluated at Z̃ = Z + β (Z - Z_prev),
Z_prev the state of the previous momentum evaluation of this fast loop (Z_prev = Z at the start of
the stage). The momentum rows depend only on the thermodynamic rows (pressure variable, ρ for
gravity; and W for the sponge in `:momentum`), so this extrapolates p, Θ, π. :hevi: the explicit
horizontal momentum steps; :verlet, :fb: the momentum step; :wrf, :ei: the momentum kick (for :ei
the extrapolation starts at the first t½). Needs `tmp` (4 work vectors for :ei).

`offcenter = ε ≥ 0` (default 0): off-centring of :wrf and :ei, β = (1 + ε)/2 (WRF's epssm). :wrf: the
vertical Crank–Nicolson step with β; :ei: all rows of E and I with (1-β)h and βh (`ei_weights`).
With the same (smdiv, offcenter) both are conjugate, EI = E⁻¹ WRF E (docs/verlet_ei). For ε = 0 :ei
uses `EI_GAMMA[]` (Imex version III, W rows only).
"""
function frozen_fast_solve!(Z, model, C, F, dt, nsteps, k; method::Symbol = :verlet, tmp = nothing, smdiv::Float64 = 0.0, offcenter::Float64 = 0.0)
    base, ws = composition(method)
    damp = smdiv > 0
    damp && tmp === nothing && error("smdiv > 0 needs work vectors (tmp)")
    zl, zt = damp ? (tmp[2], tmp[3]) : (Z, Z)
    damp && (zl .= Z)
    """Momentum part at the extrapolated state (smdiv) or at Z."""
    function momentum!(part)
        if damp
            @. zt = Z + smdiv * (Z - zl); zl .= Z
            frozen_fast!(k, model, C, zt, part, F)
        else
            frozen_fast!(k, model, C, Z, part, F)
        end
    end
    offcenter < 0 && error("offcenter ≥ 0")
    base == :ei && length(ws) == 1 && EI_MERGE[] &&
        return _ei_merged!(Z, model, C, F, dt, nsteps, k, tmp, damp ? tmp[2] : nothing, offcenter, momentum!)
    first = true
    for _ in 1:nsteps, wt in ws
        h = wt * dt
        if base == :verlet
            frozen_fast!(k, model, C, Z, :thermo, F); @. Z += (h / 2) * k
            momentum!(:momentum); @. Z += h * k
            frozen_fast!(k, model, C, Z, :thermo, F); @. Z += (h / 2) * k
        elseif base == :fb
            momentum!(:momentum); @. Z += h * k
            frozen_fast!(k, model, C, Z, :thermo, F); @. Z += h * k
        elseif base == :hevi
            y = tmp === nothing ? error(":hevi needs a work vector") : tmp[1]
            for half in 1:2
                frozen_fast!(k, model, C, Z, :hthermo, F); @. Z += (h / 4) * k
                momentum!(:hmomentum); @. Z += (h / 2) * k
                frozen_fast!(k, model, C, Z, :hthermo, F); @. Z += (h / 4) * k
                half == 1 && frozen_vertical_step!(Z, model, C, h, k, y)
            end
        elseif base == :ei
            y = tmp === nothing ? error(":ei needs a work vector") : tmp[1]
            et, ew, it, iw = ei_weights(h, offcenter)
            frozen_fast!(k, model, C, Z, :hthermo, F)                 # E: H_t (A_h U + f_t, f_W, sponge)
            frozen_vertical_rhs!(y, model, C, Z, et, ew)              #    and G_v, weights (et, ew)
            @. Z += et * k + y
            damp && first && (tmp[2] .= Z)                            # extrapolation starts at t½
            first = false
            momentum!(:hmomentum); @. Z += h * k                      # K: U⁺ from t½
            frozen_hthermo!(k, model, C, Z, F, iw); @. Z += it * k    # I: implicit step, weights (it, iw)
            frozen_vertical_implicit!(Z, model, C, it, iw, k, y)      #    t⁺, W⁺
        elseif base == :wrf
            y = tmp === nothing ? error(":wrf needs a work vector") : tmp[1]
            β = (1 + offcenter) / 2
            momentum!(:hmomentum); @. Z += h * k                      # U⁺ (extrapolated t if smdiv)
            frozen_hthermo!(k, model, C, Z, F, h)                     # A_h U⁺ + f_t, f_W, sponge (metric
                                                                      # predictor h: Û = U + h metric(pⁿ))
            frozen_vertical_rhs!(y, model, C, Z, 1.0, 1.0)            # G_v(Z)
            @. k = h * (k + y)
            frozen_vertical_solve!(y, model, C, β * h, β * h, k)      # (I - βh J_v) Δ = h (H + G_v)
            @. Z += y
        elseif base == :rk4
            tmp === nothing && error(":rk4 needs work vectors")
            damp && error(":rk4 has no divergence damping (smdiv)")
            y0, y, acc = tmp
            y0 .= Z
            frozen_fast!(k, model, C, Z, :all, F); @. acc = k; @. y = y0 + (h / 2) * k
            frozen_fast!(k, model, C, y, :all, F); @. acc += 2k; @. y = y0 + (h / 2) * k
            frozen_fast!(k, model, C, y, :all, F); @. acc += 2k; @. y = y0 + h * k
            frozen_fast!(k, model, C, y, :all, F); @. acc += k
            @. Z = y0 + (h / 6) * acc
        else
            error("frozen fast integrator $method: :verlet, :fb, :hevi, :ei, :wrf, :rk4 (or a composition of :verlet, :hevi, :ei)")
        end
    end
    return Z
end

"""
    ei_weights(h, ε) -> (et, ew, it, iw)

Weights of the Verlet EI step EI = I_(it, iw) ∘ K_h ∘ E_(et, ew): E adds et·H_t + D G_v with D = (et
on the thermodynamic rows, ew on W), I solves z⁺ = z + it·H_t + D G_v(z⁺) with D = (it, iw).
ε = 0: Imex version 1 EI / version III, (h/2, (1-γ)h, h/2, γh) with γ = `EI_GAMMA[]` (W rows only);
ε > 0: off-centring as WRF, β = (1+ε)/2 on all rows, ((1-β)h, (1-β)h, βh, βh): then E and I commute
and EI(ε) = E⁻¹ WRF(ε) E (needs `EI_GAMMA[] = 1/2`).
"""
function ei_weights(h, ε)
    if ε == 0
        γ = EI_GAMMA[]
        return (h / 2, (1 - γ) * h, h / 2, γ * h)
    end
    EI_GAMMA[] == 0.5 || error(":ei: offcenter > 0 and EI_GAMMA ≠ 1/2 exclude each other")
    β = (1 + ε) / 2
    return ((1 - β) * h, (1 - β) * h, β * h, β * h)
end

"""
Verlet EI with the half steps of consecutive steps combined (`EI_MERGE[]`). With the weights of
`ei_weights`, the implicit step I of step j adds it·H(z⁺) = z⁺ - z_s (z_s the state before I, linear
parts and exact column solve), and the explicit step E of step j+1 adds et·H(z⁺) on the
thermodynamic rows and et·(H_t)_W + ew·B_v(t⁺) on the W rows, so between two steps
    z ← z⁺ + (et/it)(z⁺ - z_s) + (ew - et·iw/it) B_v(t⁺)        (W rows only, if ≠ 0)
(2z⁺ - z_s for ε = 0, γ = 1/2; the correction only for ε = 0, γ ≠ 1/2; the sponge of the W rows is
taken at the state before I). Per step one horizontal momentum, one horizontal thermodynamic and one
vertical evaluation and one column solve; the first E and the last I as in `:ei`. Divergence
damping (`smdiv`, `zl` the previous state of the extrapolation): starts at t½ after the first E.
"""
function _ei_merged!(Z, model, C, F, h, nsteps, k, tmp, zl, ε, momentum!)
    tmp === nothing && error(":ei needs work vectors")
    y = tmp[1]
    zs = zl !== nothing ? (length(tmp) >= 4 ? tmp[4] : error(":ei with smdiv needs 4 work vectors")) : tmp[2]
    et, ew, it, iw = ei_weights(h, ε)
    corr = ew - et * iw / it
    frozen_fast!(k, model, C, Z, :hthermo, F)                 # first E
    frozen_vertical_rhs!(y, model, C, Z, et, ew)
    @. Z += et * k + y
    zl !== nothing && (zl .= Z)
    for j in 1:nsteps
        momentum!(:hmomentum); @. Z += h * k                  # K: U⁺ from t½
        zs .= Z
        frozen_hthermo!(k, model, C, Z, F, iw); @. Z += it * k
        frozen_vertical_implicit!(Z, model, C, it, iw, k, y)  # I: t⁺, W⁺
        if j < nsteps                                         # I of step j and E of step j+1 combined
            if abs(corr) > 1e-14 * h
                frozen_vertical_rhs!(y, model, C, Z, 0.0, corr)   # at t⁺, before the update
                @. Z += (et / it) * (Z - zs) + y
            else
                @. Z += (et / it) * (Z - zs)
            end
        end
    end
    return Z
end

"""
    frozen_mis_step!(qnew, model, q, Δt, tab::MISTableau, w::FrozenWork; nsub = 12, fast = :verlet)

One MIS step with a frozen part of the right-hand side: stage i (d_i = Σ_j β_ij) integrates
    Z' = F_f(Σ_j w_ij C(Y_j), Z) + Σ_j w_ij S(Y_j) + Σ_j γ_ij/(d_i Δt) (Y_{j+1} - qⁿ),  w_ij = β_ij/d_i,
over d_i Δt in max(1, round(d_i nsub)) steps from qⁿ + Σ_j α_ij (Y_{j+1} - qⁿ). For an additive
F(y, z) = S(y) + G(z) this is the MIS step of `mis_step!` with the fast part G.
"""
function frozen_mis_step!(qnew, model, q, Δt, tab::MISTableau, w::FrozenWork; nsub::Int = 12,
                          fast::Symbol = :verlet, tmp = nothing, smdiv::Float64 = 0.0, offcenter::Float64 = 0.0)
    for i in 1:nstages(tab)
        if i == 1
            w.Y .= q
        else
            @. w.Y = q + w.ΔY[i-1]
        end
        frozen_data!(w.C[i], w.S[i], model, w.Y)
        d = tab.d[i]
        w.c .= 0
        w.F .= 0
        w.Z .= q
        for j in 1:i
            b = tab.β[i, j] / d
            if b != 0
                @. w.c += b * w.C[j]
                @. w.F += b * w.S[j]
            end
            if j < i
                a, g = tab.α[i, j], tab.γ[i, j] / (d * Δt)
                ΔYj = w.ΔY[j]
                a != 0 && (@. w.Z += a * ΔYj)
                g != 0 && (@. w.F += g * ΔYj)
            end
        end
        n = fast_substeps(d, nsub)
        frozen_stage_begin!(model, w.c, w.Z)
        frozen_fast_solve!(w.Z, model, w.c, w.F, d * Δt / n, n, w.k; method = fast, tmp, smdiv, offcenter)
        @. w.ΔY[i] = w.Z - q
    end
    qnew .= w.Z
    return qnew
end

# ------------------------------------------------------------------ Lawson and ETD with the fast part frozen at qⁿ
#
# As the linear Lawson/ETD steps (lawson.jl, etd.jl) with L replaced by the frozen fast part at the
# start of the step, F_f(C(qⁿ), ·): the fast flow is Z' = F_f(C_n, Z) (nonlinear in Z, e.g. the
# Fischer–Burmeister source, micro = :fast), and the slow forcing of stage j is the remainder
#     S̃(Y_j) = N(Y_j) - F_f(C_n, Y_j) = F_f(C(Y_j), Y_j) + S(Y_j) - F_f(C_n, Y_j).

"""S̃(Y) = N(Y) - F_f(C_n, Y) into out (C, S, k work vectors)."""
function _frozen_remainder!(out, model, Cn, Y, C, S, k)
    frozen_data!(C, S, model, Y)
    frozen_fast!(out, model, C, Y, :all, S)                 # N(Y)
    fill!(S, 0.0)
    frozen_fast!(k, model, Cn, Y, :all, S)                  # F_f(C_n, Y)
    out .-= k
    return out
end

"""Substeps over the fraction c of Δt: ⌈c nsub⌉, so the fast step never exceeds Δt/nsub (an explicit
stiff source such as the Fischer–Burmeister relaxation at its stability limit needs this)."""
_nfast(c, nsub) = max(1, ceil(Int, c * nsub - 1e-10))

"""Fast flow of the frozen part over τ (⌈nsub τ/Δt⌉ substeps) with the constant forcing F."""
function _frozen_flow!(X, model, Cn, F, τ, Δt, nsub, w, fast, tmp, smdiv, offcenter)
    n = _nfast(τ / Δt, nsub)
    return frozen_fast_solve!(X, model, Cn, F, τ / n, n, w.k; method = fast, tmp, smdiv, offcenter)
end

"""
    frozen_mis_step!(qnew, model, q, Δt, tab::LawsonTableau, w::FrozenWork; nsub, fast, tmp, smdiv, offcenter)

Lawson step (`LawsonTableau`) with the fast part frozen at qⁿ: φ_τ is the frozen fast loop with
coefficients C(qⁿ) and no forcing, the slow terms are S̃(Y_j) = N(Y_j) - F_f(C(qⁿ), Y_j).
φ_{cΔt}(qⁿ) is computed once per node (cached in w.ΔY).
"""
function frozen_mis_step!(qnew, model, q, Δt, tab::LawsonTableau, w::FrozenWork; nsub::Int = 12,
                          fast::Symbol = :verlet, tmp = nothing, smdiv::Float64 = 0.0, offcenter::Float64 = 0.0)
    s, nodes = nstages(tab), tab.nodes
    Cn = w.C[1]
    frozen_data!(Cn, w.F, model, q)
    frozen_stage_begin!(model, Cn, q)
    zeroF = w.F; fill!(zeroF, 0.0)                          # no forcing in φ_τ; w.F is also the S(Y_j)
    Cs, Ss = w.C[2], w.F                                    # work of the remainder, which leaves it zero
    cache = w.ΔY
    cache[1] .= q
    ncached = 1
    function cached(kk)
        while ncached < kk
            cache[ncached+1] .= cache[ncached]
            _frozen_flow!(cache[ncached+1], model, Cn, zeroF, (nodes[ncached+1] - nodes[ncached]) * Δt, Δt, nsub, w, fast, tmp, smdiv, offcenter)
            ncached += 1
        end
        return cache[kk]
    end
    function stage!(X, wts, c)
        c0 = lawson_start(tab, wts, c)
        X .= cached(findfirst(==(c0), nodes))
        cur = c0
        for ν in nodes
            ν < c0 && continue
            ν > c && break
            if ν > cur
                _frozen_flow!(X, model, Cn, zeroF, (ν - cur) * Δt, Δt, nsub, w, fast, tmp, smdiv, offcenter)
                cur = ν
            end
            for j in 1:s
                (wts[j] != 0 && tab.c[j] == ν) && (X .+= (Δt * wts[j]) .* w.S[j])
            end
        end
        cur < c && _frozen_flow!(X, model, Cn, zeroF, (c - cur) * Δt, Δt, nsub, w, fast, tmp, smdiv, offcenter)
        return X
    end
    for i in 1:s
        i == 1 ? (w.Y .= q) : stage!(w.Y, view(tab.A, i, :), tab.c[i])
        _frozen_remainder!(w.S[i], model, Cn, w.Y, Cs, Ss, w.k)
    end
    stage!(w.Z, tab.b, 1.0)
    qnew .= w.Z
    return qnew
end

"""
    frozen_mis_step!(qnew, model, q, Δt, tab::ETDTableau, w::FrozenWork; nsub, fast, tmp, smdiv, offcenter)

Exponential RK step (`ETDTableau`, stage-ODE form) with the fast part frozen at qⁿ: stage i
integrates Z' = F_f(C(qⁿ), Z) + Σ_j Σ_k P[i,j,k] τ^{k-1} S̃(Y_j) from qⁿ over c_i Δt in
⌈c_i nsub⌉ substeps; the polynomial forcing is evaluated at the substep midpoint.
"""
function frozen_mis_step!(qnew, model, q, Δt, tab::ETDTableau, w::FrozenWork; nsub::Int = 12,
                          fast::Symbol = :verlet, tmp = nothing, smdiv::Float64 = 0.0, offcenter::Float64 = 0.0)
    s, K = nstages(tab), size(tab.P, 3)
    Cn = w.C[1]
    frozen_data!(Cn, w.F, model, q)
    frozen_stage_begin!(model, Cn, q)
    Cs = w.C[2]
    G = view(w.ΔY, 1:K)
    w.Y .= q
    for i in 1:s
        _frozen_remainder!(w.S[i], model, Cn, w.Y, Cs, w.F, w.k)
        for kk in 1:K
            Gk = G[kk]; Gk .= 0
            for j in 1:i
                p = tab.P[i, j, kk]
                p != 0 && (Gk .+= p .* w.S[j])
            end
        end
        T = tab.c[i] * Δt
        n = _nfast(tab.c[i], nsub)
        dt = T / n
        w.Y .= q
        for m in 1:n
            τ = (m - 0.5) / n
            w.F .= G[K]
            for kk in K-1:-1:1
                @. w.F = τ * w.F + G[kk]
            end
            frozen_fast_solve!(w.Y, model, Cn, w.F, dt, 1, w.k; method = fast, tmp, smdiv, offcenter)
        end
    end
    qnew .= w.Y
    return qnew
end

"""
    run_frozen(model, q0, Δt, nsteps; tab = mis_tableau(:MIS4a), nsub = 12, fast = :verlet,
               smdiv = 0.0, offcenter = 0.0, monitor = nothing, every = 1)

Split-explicit integration with a frozen part of the right-hand side (`frozen_mis_step!`): MIS,
or Lawson / ETD tableaus with the fast part frozen at the start of the step;
`smdiv`: divergence damping by extrapolation of the thermodynamic variables in the fast loop,
`offcenter`: off-centring ε of :ei and :wrf (`frozen_fast_solve!`).
"""
function run_frozen(model, q0, Δt, nsteps; tab::SplitTableau = mis_tableau(:MIS4a), nsub::Int = 12,
                    fast::Symbol = :verlet, smdiv::Real = 0.0, offcenter::Real = 0.0, monitor = nothing, every::Int = 1)
    q = Vector{Float64}(q0)
    w = FrozenWork(model, length(q), tab)
    tmp = (composition(fast)[1] in (:rk4, :hevi, :ei, :wrf) || smdiv > 0) ? ntuple(_ -> zeros(length(q)), 4) : nothing
    for n in 1:nsteps
        frozen_mis_step!(q, model, q, Δt, tab, w; nsub, fast, tmp, smdiv = Float64(smdiv), offcenter = Float64(offcenter))
        frozen_step_end!(model, q)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end
