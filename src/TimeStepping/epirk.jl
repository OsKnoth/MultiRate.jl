# EPIRK(-W) methods (Tokman) and exponential Adams–Bashforth methods in stage-ODE form,
# with the fast integrators of the MIS step in place of the Krylov φ-function evaluations

# ------------------------------------------------------------------ EPIRK

"""
    EPIRKTableau(name, A, g, p; order)

Unpartitioned EPIRK method (Tokman 2011; EPIRK-W: Narayanamurthi, Tranquilli, Sandu, Tokman,
J. Sci. Comput. 2019, eqs. (10)-(12)) with the frozen fast operator L_n as matrix,
    Y_i = qⁿ + a_{i1} ψ_1(g_{i1} h L) h N(qⁿ) + Σ_{j=2}^{i} a_{ij} ψ_j(g_{ij} h L) h Δ^{(j-1)} r(qⁿ),
    ψ_j = Σ_{k ≤ j} p_{jk} φ_k,   r(y) = N(y) - N(qⁿ) - L(y - qⁿ) = S(y) - S(qⁿ),
Δ^{(j)} the forward differences over qⁿ = Y_0, Y_1, …; the last row of A (= b) gives qⁿ⁺¹.
The classical EPIRK order needs L = ∂N/∂q; EPIRK-W methods keep their order for any L.

Stage-ODE form: for each distinct g > 0 in a row the terms are one fast integration from qⁿ
over g h, W' = L W - L qⁿ + Σ_k τ^{k-1} G_k (W = qⁿ + z, z' = L z + forcing), with
G_k = Σ_j a_{ij} p_{jk} V_j / ((k-1)! g) (V_1 = N(qⁿ), V_j = Δ^{(j-1)} r); terms with g = 0 are
explicit, h a_{ij} ψ_j(0) V_j.
"""
struct EPIRKTableau <: SplitTableau
    name::Symbol
    A::Matrix{Float64}       # a_ij, last row b
    g::Matrix{Float64}
    p::Matrix{Float64}       # p[j, k]: ψ_j = Σ_k p[j, k] φ_k
    order::Int
    scales::Vector{Vector{Float64}}   # distinct g > 0 of the active terms of each row
    function EPIRKTableau(name::Symbol, A::AbstractMatrix, g::AbstractMatrix, p::AbstractMatrix; order::Int)
        s = size(A, 1)
        size(A) == size(g) == size(p) == (s, s) || throw(ArgumentError("A, g, p must be s × s"))
        sc = [sort!(unique([Float64(g[i, j]) for j in 1:i if A[i, j] != 0 && g[i, j] > 0])) for i in 1:s]
        new(name, Matrix{Float64}(A), Matrix{Float64}(g), Matrix{Float64}(p), order, sc)
    end
end
nstages(t::EPIRKTableau) = size(t.A, 1)

"""Distinct g > 0 of the active terms of row i (one fast integration each)."""
epirk_scales(t::EPIRKTableau, i) = t.scales[i]
fast_time(t::EPIRKTableau) = sum(sum(epirk_scales(t, i); init = 0.0) for i in 1:nstages(t))

"""EPIRK tableaus (coefficients of Narayanamurthi et al. 2019, Figs. 1-3):
:EPIRKW3a, :EPIRKW3b EPIRK-W of order 3 (any L); :EPIRK4 the classical order-4 method of
epirkk4 (Corollary 4.6, order 4 only for L = Jacobian)."""
function epirk_tableau(name::Symbol)
    if name == :EPIRKW3a
        A = [1/2 0 0; 0 1 0; 3/4 1/2 1]
        g = [2/3 0 0; 0 0 0; 1 3/5 0]
        p = [4/3 0 0; 1 2 0; 0 0 3/4]
        return EPIRKTableau(name, A, g, p; order = 3)
    elseif name == :EPIRKW3b
        a11, a21, a22 = 0.22824182961171620396, 0.45648365922343240794, 0.33161664063356950085
        gg = 0.34706341174296320958
        A = [a11 0 0; a21 a22 0; 1 2.0931591383832578214 1.2623969257900804404]
        g = [0 0 0; gg gg gg; 1 1 1]
        p = [1 0 0; 0 2.0931604100438501004 0; 1 1 1]
        return EPIRKTableau(name, A, g, p; order = 3)
    elseif name == :EPIRK4
        r = 692665874901013 / 799821658665135
        A = [r 0 0; r 3/4 0; 1/r 352/729 64/729]
        g = [3/4 0 0; 3/4 0 0; 1 9/16 9/16]
        p = [r 0 0; 1 1 0; 1 1 0]
        return EPIRKTableau(name, A, g, p; order = 4)
    end
    error("unknown EPIRK tableau $name (:EPIRKW3a, :EPIRKW3b, :EPIRK4)")
end

split_work(t::EPIRKTableau, nq::Int) = MISWork(nq, 2nstages(t) + 1)

"""
    mis_step!(qnew, model, q, Δt, tab::EPIRKTableau, sF, work; nsub = 12, fast = :imex)

One EPIRK step in stage-ODE form (see `EPIRKTableau`); a fast integration over g h uses
max(1, round(g nsub)) substeps. Work (`split_work`): S[1] = N(qⁿ), S[j+1] = r(Y_j),
ΔY[1:s] = V_j, ΔY[s+1:2s] = G_k, ΔY[2s+1] = S(qⁿ), Fs = L qⁿ.
"""
function mis_step!(qnew, model, q, Δt, tab::EPIRKTableau, sF, w::MISWork; nsub::Int = 12, fast::Symbol = :imex)
    s = nstages(tab)
    V = view(w.ΔY, 1:s)
    G = view(w.ΔY, s+1:2s)
    Sn = w.ΔY[2s+1]
    fb = fast_forcing(sF)
    rhs_nonlinear!(w.S[1], model, q)
    fast_apply!(w.Fs, sF, q, fb)                           # L qⁿ (with the constant forcing)
    @. Sn = w.S[1] - w.Fs
    V[1] .= w.S[1]
    w.Y .= q
    for i in 1:s
        if i >= 2                                          # r(Y_{i-1}) and V_i = Δ^{(i-1)} r(qⁿ)
            ri = w.S[i]
            rhs_nonlinear!(ri, model, w.Y)
            fast_apply!(w.L, sF, w.Y, fb)
            @. ri = ri - w.L - Sn
            V[i] .= 0
            for m in 1:i-1
                c = (-1)^(i - 1 - m) * binomial(i - 1, m)
                @. V[i] += c * w.S[m+1]
            end
        end
        w.Y .= q
        for j in 1:i                                       # explicit terms (g = 0)
            a = tab.A[i, j]
            (a == 0 || tab.g[i, j] > 0) && continue
            ψ0 = sum(tab.p[j, k] / factorial(k) for k in 1:s)
            @. w.Y += (Δt * a * ψ0) * V[j]
        end
        for gs in epirk_scales(tab, i)
            for k in 1:s
                Gk = G[k]
                Gk .= 0
                for j in 1:i
                    a = tab.A[i, j]
                    (a != 0 && tab.g[i, j] == gs && tab.p[j, k] != 0) || continue
                    @. Gk += (a * tab.p[j, k] / (factorial(k - 1) * gs)) * V[j]
                end
            end
            G[1] .-= w.Fs
            fast_solve_poly!(w.Z, sF, q, G, gs * Δt, max(1, round(Int, gs * nsub)); method = fast)
            @. w.Y += w.Z - q
        end
    end
    qnew .= w.Y
    return qnew
end

"""φ_1 … φ_K of a small matrix Z (exponential of the augmented block matrix)."""
function phi_functions(Z::AbstractMatrix, K::Int)
    n = size(Z, 1)
    M = zeros(eltype(Z), (K + 1) * n, (K + 1) * n)
    M[1:n, 1:n] = Z
    for k in 1:K
        M[(k-1)*n+1:k*n, k*n+1:(k+1)*n] = I(n)
    end
    E = exp(M)
    return [E[1:n, k*n+1:(k+1)*n] for k in 1:K]
end

"""EPIRK step for q' = L q + G(q) with exact φ-functions of L, directly from the EPIRK formula
(independent of the stage-ODE form of the production step)."""
function mis_step_exact(L::AbstractMatrix, G, q::AbstractVector, Δt::Real, tab::EPIRKTableau)
    s = nstages(tab)
    Gn = G(q)
    fn = L * q + Gn
    r = Vector{Vector{Float64}}(undef, s)           # r[m] = r(Y_m), m = 1 … s-1
    V = Vector{Vector{Float64}}(undef, s)
    V[1] = fn
    Y = copy(q)
    for i in 1:s
        if i >= 2
            r[i-1] = G(Y) - Gn
            V[i] = sum((-1)^(i - 1 - m) * binomial(i - 1, m) * r[m] for m in 1:i-1)
        end
        Y = copy(q)
        for j in 1:i
            a = tab.A[i, j]
            a == 0 && continue
            φ = phi_functions(tab.g[i, j] * Δt * L, s)
            ψ = sum(tab.p[j, k] * φ[k] for k in 1:s)
            Y += Δt * a * (ψ * V[j])
        end
    end
    return Y
end

# ------------------------------------------------------------------ exponential Adams–Bashforth

"""
    EABTableau(name, k; startup = :EB4)

Exponential Adams–Bashforth method with k steps (Hochbruck & Ostermann, Acta Numerica 2010,
Sect. 2.5), in the spirit of the EPI multistep methods of Tokman and Gaudreault but for an
arbitrary linear part: one fast integration per step,
    Y' = L Y + p(τ),   Y(0) = qⁿ,   τ = t/Δt ∈ [0, 1],
with p the polynomial through S(q^{n-m}) at τ = -m, m = 0 … k-1, all S with the current
frozen L (S(q^{n-m}) = N(q^{n-m}) - L_n q^{n-m}). Q[κ, m+1] are the monomial coefficients
(p(τ) = Σ_κ τ^{κ-1} Σ_m Q[κ, m+1] S(q^{n-m})). The first k-1 steps use the one-step method
`startup`.
"""
struct EABTableau <: SplitTableau
    name::Symbol
    k::Int
    Q::Matrix{Float64}
    startup::SplitTableau
end
function EABTableau(name::Symbol, k::Int; startup = :EB4)
    Q = zeros(k, k)
    for m in 0:k-1                                   # Lagrange basis through τ = 0, -1, …, -(k-1)
        c = [1.0]
        for l in 0:k-1
            l == m && continue
            c = ([c; 0.0] .* l .+ [0.0; c]) ./ (l - m)   # c(τ) (τ + l) / (l - m)
        end
        Q[:, m+1] = c
    end
    return EABTableau(name, k, Q, startup isa Symbol ? mis_tableau(startup) : startup)
end
nstages(t::EABTableau) = 1
fast_time(t::EABTableau) = 1.0

"""Exponential Adams–Bashforth tableaus :EAB1 (= exponential Euler), :EAB2, :EAB3, :EAB4."""
eab_tableau(name::Symbol; startup = :EB4) =
    name in (:EAB1, :EAB2, :EAB3, :EAB4) ? EABTableau(name, parse(Int, string(name)[end]); startup) :
    error("unknown EAB tableau $name (:EAB1 … :EAB4)")

"""Work of the EAB step: history q^{n-m}, N(q^{n-m}) (m = 0 … k-1, newest first), forcing
coefficients, and the work of the start-up method; `nhist` counts the valid history entries."""
mutable struct EABWork
    y::Vector{Vector{Float64}}
    N::Vector{Vector{Float64}}
    G::Vector{Vector{Float64}}
    S::Vector{Float64}
    L::Vector{Float64}
    base::MISWork
    nhist::Int
end
split_work(t::EABTableau, nq::Int) =
    EABWork([zeros(nq) for _ in 1:t.k], [zeros(nq) for _ in 1:t.k], [zeros(nq) for _ in 1:t.k],
            zeros(nq), zeros(nq), split_work(t.startup, nq), 0)

"""Forget the history (the next steps are start-up steps again)."""
reset!(w::EABWork) = (w.nhist = 0; w)

_rotate!(v) = (last = v[end]; for m in length(v):-1:2; v[m] = v[m-1]; end; v[1] = last; v)

"""
    mis_step!(qnew, model, q, Δt, tab::EABTableau, sF, work::EABWork; nsub = 12, fast = :imex)

One exponential Adams–Bashforth step (see `EABTableau`). Multistep: successive calls must pass
the previous result as q with the same Δt (`reset!(work)` to start again); the first k-1 calls
are steps of `tab.startup`.
"""
function mis_step!(qnew, model, q, Δt, tab::EABTableau, sF, w::EABWork; nsub::Int = 12, fast::Symbol = :imex)
    _rotate!(w.y); _rotate!(w.N)
    w.y[1] .= q
    rhs_nonlinear!(w.N[1], model, q)
    w.nhist = min(w.nhist + 1, tab.k)
    if w.nhist < tab.k
        return mis_step!(qnew, model, q, Δt, tab.startup, sF, w.base; nsub, fast)
    end
    return eab_advance!(qnew, q, Δt, tab, sF, w; nsub, fast)
end

"""EAB step from the stored history w.y, w.N (q = w.y[1]) with the frozen fast operator sF."""
function eab_advance!(qnew, q, Δt, tab::EABTableau, sF, w::EABWork; nsub::Int = 12, fast::Symbol = :imex)
    fb = fast_forcing(sF)
    for κ in 1:tab.k
        w.G[κ] .= 0
    end
    for m in 1:tab.k
        fast_apply!(w.L, sF, w.y[m], fb)
        @. w.S = w.N[m] - w.L                              # S(q^{n-m+1}) with the current L
        for κ in 1:tab.k
            c = tab.Q[κ, m]
            c != 0 && (@. w.G[κ] += c * w.S)
        end
    end
    fast_solve_poly!(qnew, sF, q, w.G, Δt, nsub; method = fast)
    return qnew
end

"""Flow of Y' = L Y + Σ_κ (t/T)^{κ-1} F_κ over T from q (augmented exponential)."""
function _poly_flow(L::AbstractMatrix, q::AbstractVector, F::AbstractVector, T::Real)
    n, K = length(q), length(F)
    M = zeros(n + K, n + K)
    M[1:n, 1:n] = T * L
    for κ in 1:K
        M[1:n, n+κ] = T * F[κ]
        κ > 1 && (M[n+κ, n+κ-1] = κ - 1)
    end
    return (exp(M) * vcat(q, 1.0, zeros(K - 1)))[1:n]
end

"""EAB step for q' = L q + G(q) with the exact flow; ys = (qⁿ, q^{n-1}, …, q^{n-k+1})."""
function eab_step_exact(L::AbstractMatrix, G, ys::AbstractVector, Δt::Real, tab::EABTableau)
    S = [G(y) for y in ys]
    F = [sum(tab.Q[κ, m] * S[m] for m in 1:tab.k) for κ in 1:tab.k]
    return _poly_flow(L, ys[1], F, Δt)
end
