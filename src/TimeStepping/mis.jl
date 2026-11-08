# MIS time stepper (split-explicit), RK4 reference

"""Preallocated work vectors of the MIS step (ΔY_i = Y_{i+1} - qⁿ, S_i = S(Y_i))."""
struct MISWork
    ΔY::Vector{Vector{Float64}}
    S::Vector{Vector{Float64}}
    Y::Vector{Float64}
    Z::Vector{Float64}
    Fs::Vector{Float64}
    L::Vector{Float64}
end
MISWork(nq::Int, ns::Int) = MISWork([zeros(nq) for _ in 1:ns], [zeros(nq) for _ in 1:ns],
                                    zeros(nq), zeros(nq), zeros(nq), zeros(nq))

"""Configuration for split-explicit MIS, independent of the spatial model being integrated.

`fast` selects the fast-system integrator (`:hevi` uses explicit horizontal and implicit
vertical tendencies). `fast_options` are passed to the model's `fast_operator` constructor.
The model-specific fast operator and work arrays are allocated by `run_mis`.
"""
struct MISStepper{K<:NamedTuple}
    tableau::SplitTableau
    nsub::Int
    fast::Symbol
    refreeze::Bool
    refreeze_every::Int
    fast_options::K
    function MISStepper(tableau::SplitTableau, nsub::Integer, fast::Symbol, refreeze::Bool,
                        refreeze_every::Integer, fast_options::NamedTuple)
        nsub > 0 || throw(ArgumentError("nsub must be positive"))
        refreeze_every > 0 || throw(ArgumentError("refreeze_every must be positive"))
        new{typeof(fast_options)}(tableau, Int(nsub), fast, refreeze, Int(refreeze_every), fast_options)
    end
end

MISStepper(; tab::SplitTableau = mis_tableau(:MIS4a), nsub::Integer = 12, fast::Symbol = :imex,
           refreeze::Bool = true, refreeze_every::Integer = 1, fast_options::NamedTuple = NamedTuple()) =
    MISStepper(tab, nsub, fast, refreeze, refreeze_every, fast_options)

"""Fast substeps per MIS stage: n_i = max(1, round(d_i nsub)); for WSRK3 nsub/3, nsub/2, nsub (WRF)."""
stage_substeps(tab::MISTableau, nsub::Int) = [max(1, round(Int, d * nsub)) for d in tab.d]

"""
    mis_step!(qnew, model, q, Δt, tab, sF, work; nsub=12, fast=:imex)

One large step with the frozen fast operator sF; stage i integrates the fast system over d_i Δt
with n_i = max(1, round(d_i nsub)) substeps. qnew may alias q. Allocation-free for models with
in-place `rhs_nonlinear!` and operators with a cached fast workspace.
"""
function mis_step!(qnew, model, q, Δt, tab::MISTableau, sF, w::MISWork; nsub::Int = 12, fast::Symbol = :imex)
    for i in 1:nstages(tab)
        if i == 1
            w.Y .= q
        else
            @. w.Y = q + w.ΔY[i-1]
        end
        S = w.S[i]
        rhs_nonlinear!(S, model, w.Y)
        fast_apply!(w.L, sF, w.Y, fast_forcing(sF))       # L_n(Y_i)
        S .-= w.L                                          # S(Y_i) = N(Y_i) - L_n(Y_i)
        d = tab.d[i]
        c = tab.β[i, i] / d
        w.Z .= q
        @. w.Fs = c * S
        for j in 1:i-1
            a, g, b = tab.α[i, j], tab.γ[i, j] / (d * Δt), tab.β[i, j] / d
            ΔYj, Sj = w.ΔY[j], w.S[j]
            @. w.Z += a * ΔYj
            @. w.Fs += g * ΔYj + b * Sj
        end
        n = max(1, round(Int, d * nsub))
        fast_solve!(w.Z, sF, w.Z, w.Fs, d * Δt / n, n; method = fast)
        @. w.ΔY[i] = w.Z - q
    end
    qnew .= w.Z
    return qnew
end

"""
    mis_step(model, q, Δt, tab; nsub=12, fast=:imex, sF=nothing, kwargs...)

Allocating convenience version of `mis_step!` (fast operator frozen at q unless sF is given;
kwargs are passed to `fast_operator`).
"""
function mis_step(model, q::AbstractVector, Δt::Real, tab::SplitTableau; nsub::Int = 12, fast::Symbol = :imex,
                  sF = nothing, kwargs...)
    sF === nothing && (sF = fast_operator(model, q; fast = composition(fast)[1], kwargs...))
    return mis_step!(similar(q), model, q, Δt, tab, sF, MISWork(length(q), nstages(tab)); nsub = nsub, fast = fast)
end

"""
    run_mis(model, q0, Δt, nsteps, stepper::MISStepper; monitor=nothing, every=1)
    run_mis(model, q0, Δt, nsteps; tab=mis_tableau(:MIS4a), nsub=12, fast=:imex,
            refreeze=true, refreeze_every=1, monitor=nothing, every=1, kwargs...)

Split-explicit integration with `nsub` fast substeps per large step (ratio fast/slow; WRF uses
6 or 12 with RK3; keep it ≤ 24). Prefer a `MISStepper` when reusing the integration policy;
the keyword form remains as a convenience wrapper. The fast operator is re-frozen at the current
state every `refreeze_every` large steps; `fast_options` (e.g. pfac, eps, tau_scale) are passed to
`fast_operator`.
"""
function run_mis(model, q0, Δt, nsteps, stepper::MISStepper; monitor = nothing, every::Int = 1)
    nsub, fast = stepper.nsub, stepper.fast
    nsub > 24 && @warn "nsub = $nsub fast steps per slow step (ratio > 24)"
    q = Vector{Float64}(q0)
    sF = fast_operator(model, q; fast = composition(fast)[1], stepper.fast_options...)
    w = MISWork(length(q), nstages(stepper.tableau))
    for n in 1:nsteps
        stepper.refreeze && n > 1 && (n - 1) % stepper.refreeze_every == 0 &&
            (sF = fast_operator(model, q; fast = composition(fast)[1], stepper.fast_options...))
        mis_step!(q, model, q, Δt, stepper.tableau, sF, w; nsub = nsub, fast = fast)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end

function run_mis(model, q0, Δt, nsteps; tab::SplitTableau = mis_tableau(:MIS4a), nsub::Int = 12,
                 fast::Symbol = :imex, refreeze::Bool = true, refreeze_every::Int = 1,
                 monitor = nothing, every::Int = 1, kwargs...)
    stepper = MISStepper(; tab, nsub, fast, refreeze, refreeze_every, fast_options = (; kwargs...))
    return run_mis(model, q0, Δt, nsteps, stepper; monitor, every)
end

"""Classical RK4 on the full nonlinear operator (reference solution)."""
function run_rk4(model, q0, dt, nsteps)
    q = Vector{Float64}(q0)
    k1, k2, k3, k4, y = (zeros(length(q)) for _ in 1:5)
    for _ in 1:nsteps
        rhs_nonlinear!(k1, model, q)
        @. y = q + (dt / 2) * k1
        rhs_nonlinear!(k2, model, y)
        @. y = q + (dt / 2) * k2
        rhs_nonlinear!(k3, model, y)
        @. y = q + dt * k3
        rhs_nonlinear!(k4, model, y)
        @. q += (dt / 6) * (k1 + 2k2 + 2k3 + k4)
    end
    return q
end
