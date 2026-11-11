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
FrozenWork(model, nq::Int, tab::MISTableau) = FrozenWork(nq, frozen_ncoef(model), nstages(tab))

"""
    frozen_fast_solve!(Z, model, C, F, dt, nsteps; method = :verlet)

Integrate Z' = F_f(C, Z) + F over nsteps steps dt (C, F constant). :verlet Störmer–Verlet
(thermodynamic half step, momentum step, thermodynamic half step, as `SplitExplicit.m`),
:fb forward-backward (momentum, then thermodynamics), :rk4 classical RK4 on F_f + F,
:hevi Strang: horizontal Verlet half step, linearly implicit vertical Crank–Nicolson step
(`frozen_vertical_step!`), horizontal half step;
compositions `:verlet_s4`, `_y4`, `_y6` (see `composition`).
"""
function frozen_fast_solve!(Z, model, C, F, dt, nsteps, k; method::Symbol = :verlet, tmp = nothing)
    base, ws = composition(method)
    for _ in 1:nsteps, wt in ws
        h = wt * dt
        if base == :verlet
            frozen_fast!(k, model, C, Z, :thermo, F); @. Z += (h / 2) * k
            frozen_fast!(k, model, C, Z, :momentum, F); @. Z += h * k
            frozen_fast!(k, model, C, Z, :thermo, F); @. Z += (h / 2) * k
        elseif base == :fb
            frozen_fast!(k, model, C, Z, :momentum, F); @. Z += h * k
            frozen_fast!(k, model, C, Z, :thermo, F); @. Z += h * k
        elseif base == :hevi
            y = tmp === nothing ? error(":hevi needs a work vector") : tmp[1]
            for half in 1:2
                frozen_fast!(k, model, C, Z, :hthermo, F); @. Z += (h / 4) * k
                frozen_fast!(k, model, C, Z, :hmomentum, F); @. Z += (h / 2) * k
                frozen_fast!(k, model, C, Z, :hthermo, F); @. Z += (h / 4) * k
                half == 1 && frozen_vertical_step!(Z, model, C, h, k, y)
            end
        elseif base == :rk4
            tmp === nothing && error(":rk4 needs work vectors")
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
                          fast::Symbol = :verlet, tmp = nothing)
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
        frozen_fast_solve!(w.Z, model, w.c, w.F, d * Δt / n, n, w.k; method = fast, tmp)
        @. w.ΔY[i] = w.Z - q
    end
    qnew .= w.Z
    return qnew
end

"""
    run_frozen(model, q0, Δt, nsteps; tab = mis_tableau(:MIS4a), nsub = 12, fast = :verlet,
               monitor = nothing, every = 1)

Split-explicit MIS integration with a frozen part of the right-hand side (`frozen_mis_step!`).
"""
function run_frozen(model, q0, Δt, nsteps; tab::MISTableau = mis_tableau(:MIS4a), nsub::Int = 12,
                    fast::Symbol = :verlet, monitor = nothing, every::Int = 1)
    q = Vector{Float64}(q0)
    w = FrozenWork(model, length(q), tab)
    tmp = composition(fast)[1] in (:rk4, :hevi) ? (zeros(length(q)), zeros(length(q)), zeros(length(q))) : nothing
    for n in 1:nsteps
        frozen_mis_step!(q, model, q, Δt, tab, w; nsub, fast, tmp)
        frozen_step_end!(model, q)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end
