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
# with the parts :hthermo, :hmomentum (horizontal, with F) and :vertical (no F) of frozen_fast!.

function frozen_ncoef end
function frozen_data! end
function frozen_fast! end
frozen_stage_begin!(model, C, z) = nothing
function frozen_vertical_step! end
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
(`frozen_vertical_step!`), horizontal half step;
compositions `:verlet_s4`, `_y4`, `_y6` (see `composition`).

`smdiv = β > 0`: divergence damping by forward extrapolation of the thermodynamic variables in the
momentum part (as WRF, `WRF_SMDIV`): the momentum rows are evaluated at Z̃ = Z + β (Z - Z_prev),
Z_prev the state of the previous momentum evaluation of this fast loop (Z_prev = Z at the start of
the stage). The momentum rows depend only on the thermodynamic rows (pressure variable, ρ for
gravity; and W for the sponge in `:momentum`), so this extrapolates p, Θ, π. :hevi: the explicit
horizontal momentum steps; :verlet, :fb: the momentum step. Needs `tmp` (3 work vectors).
"""
function frozen_fast_solve!(Z, model, C, F, dt, nsteps, k; method::Symbol = :verlet, tmp = nothing, smdiv::Float64 = 0.0)
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
            error("frozen fast integrator $method: :verlet, :fb, :rk4 (or a composition of :verlet)")
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
                          fast::Symbol = :verlet, tmp = nothing, smdiv::Float64 = 0.0)
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
        n = max(1, round(Int, d * nsub))
        frozen_stage_begin!(model, w.c, w.Z)
        frozen_fast_solve!(w.Z, model, w.c, w.F, d * Δt / n, n, w.k; method = fast, tmp, smdiv)
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
function _frozen_flow!(X, model, Cn, F, τ, Δt, nsub, w, fast, tmp, smdiv)
    n = _nfast(τ / Δt, nsub)
    return frozen_fast_solve!(X, model, Cn, F, τ / n, n, w.k; method = fast, tmp, smdiv)
end

"""
    frozen_mis_step!(qnew, model, q, Δt, tab::LawsonTableau, w::FrozenWork; nsub, fast, tmp, smdiv)

Lawson step (`LawsonTableau`) with the fast part frozen at qⁿ: φ_τ is the frozen fast loop with
coefficients C(qⁿ) and no forcing, the slow terms are S̃(Y_j) = N(Y_j) - F_f(C(qⁿ), Y_j).
φ_{cΔt}(qⁿ) is computed once per node (cached in w.ΔY).
"""
function frozen_mis_step!(qnew, model, q, Δt, tab::LawsonTableau, w::FrozenWork; nsub::Int = 12,
                          fast::Symbol = :verlet, tmp = nothing, smdiv::Float64 = 0.0)
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
            _frozen_flow!(cache[ncached+1], model, Cn, zeroF, (nodes[ncached+1] - nodes[ncached]) * Δt, Δt, nsub, w, fast, tmp, smdiv)
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
                _frozen_flow!(X, model, Cn, zeroF, (ν - cur) * Δt, Δt, nsub, w, fast, tmp, smdiv)
                cur = ν
            end
            for j in 1:s
                (wts[j] != 0 && tab.c[j] == ν) && (X .+= (Δt * wts[j]) .* w.S[j])
            end
        end
        cur < c && _frozen_flow!(X, model, Cn, zeroF, (c - cur) * Δt, Δt, nsub, w, fast, tmp, smdiv)
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
    frozen_mis_step!(qnew, model, q, Δt, tab::ETDTableau, w::FrozenWork; nsub, fast, tmp, smdiv)

Exponential RK step (`ETDTableau`, stage-ODE form) with the fast part frozen at qⁿ: stage i
integrates Z' = F_f(C(qⁿ), Z) + Σ_j Σ_k P[i,j,k] τ^{k-1} S̃(Y_j) from qⁿ over c_i Δt in
⌈c_i nsub⌉ substeps; the polynomial forcing is evaluated at the substep midpoint.
"""
function frozen_mis_step!(qnew, model, q, Δt, tab::ETDTableau, w::FrozenWork; nsub::Int = 12,
                          fast::Symbol = :verlet, tmp = nothing, smdiv::Float64 = 0.0)
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
            frozen_fast_solve!(w.Y, model, Cn, w.F, dt, 1, w.k; method = fast, tmp, smdiv)
        end
    end
    qnew .= w.Y
    return qnew
end

"""
    run_frozen(model, q0, Δt, nsteps; tab = mis_tableau(:MIS4a), nsub = 12, fast = :verlet,
               smdiv = 0.0, monitor = nothing, every = 1)

Split-explicit integration with a frozen part of the right-hand side (`frozen_mis_step!`): MIS,
or Lawson / ETD tableaus with the fast part frozen at the start of the step;
`smdiv`: divergence damping by extrapolation of the thermodynamic variables in the fast loop
(`frozen_fast_solve!`).
"""
function run_frozen(model, q0, Δt, nsteps; tab::SplitTableau = mis_tableau(:MIS4a), nsub::Int = 12,
                    fast::Symbol = :verlet, smdiv::Real = 0.0, monitor = nothing, every::Int = 1)
    q = Vector{Float64}(q0)
    w = FrozenWork(model, length(q), tab)
    tmp = (composition(fast)[1] in (:rk4, :hevi) || smdiv > 0) ? (zeros(length(q)), zeros(length(q)), zeros(length(q))) : nothing
    for n in 1:nsteps
        frozen_mis_step!(q, model, q, Δt, tab, w; nsub, fast, tmp, smdiv = Float64(smdiv))
        frozen_step_end!(model, q)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end
