# Order analysis of MIS tableaus

"""
    mis_rk_tableau(tab) -> (A, b, c)

Explicit RK method to which the MIS method reduces for L = 0: the stage increments satisfy
ΔY = (α + γ) ΔY + Δt β S, so ΔY = Δt (I - α - γ)⁻¹ β S. Stage 1 is qⁿ, stage i+1 is qⁿ + ΔY_i.
"""
function mis_rk_tableau(tab::MISTableau)
    s = nstages(tab)
    B = (I - tab.α - tab.γ) \ tab.β
    A = zeros(s, s)
    for i in 2:s
        A[i, :] = B[i-1, :]
    end
    b = B[s, :]
    return A, b, A * ones(s)
end

"""
    rk_order_residuals(A, b)

Residuals of the classical order conditions up to order 4, as a vector of (order, residual).
"""
function rk_order_residuals(A::AbstractMatrix, b::AbstractVector)
    c = A * ones(length(b))
    Ac = A * c
    return [(1, sum(b) - 1),
            (2, b' * c - 1 / 2),
            (3, b' * c .^ 2 - 1 / 3), (3, b' * Ac - 1 / 6),
            (4, b' * c .^ 3 - 1 / 4), (4, b' * (c .* Ac) - 1 / 8),
            (4, b' * (A * c .^ 2) - 1 / 12), (4, b' * (A * Ac) - 1 / 24)]
end

"""Highest order p ≤ 4 for which all classical RK conditions hold to `tol`."""
function rk_order(A::AbstractMatrix, b::AbstractVector; tol = 1e-10)
    r = rk_order_residuals(A, b)
    p = 0
    for k in 1:4
        all(abs(x) <= tol for (o, x) in r if o == k) || break
        p = k
    end
    return p
end

"""
    mis_step_exact(L, G, q, Δt, tab)

One MIS step for q' = L q + G(q) with constant matrix L as fast part and the fast stages solved
exactly: stage i integrates Z' = L Z + F_i over d_i Δt, Z(d_i Δt) = e^{TL} Z(0) + T φ₁(TL) F_i,
taken from the exponential of the augmented matrix [TL T·I; 0 0].
"""
function mis_step_exact(L::AbstractMatrix, G, q::AbstractVector, Δt::Real, tab::MISTableau)
    n, s = length(q), nstages(tab)
    ΔY = [zeros(eltype(q), n) for _ in 1:s]
    S = [zeros(eltype(q), n) for _ in 1:s]
    Z = similar(q)
    for i in 1:s
        S[i] = G(i == 1 ? q : q + ΔY[i-1])
        d = tab.d[i]
        Z0 = copy(q)
        F = tab.β[i, i] / d * S[i]
        for j in 1:i-1
            Z0 += tab.α[i, j] * ΔY[j]
            F += tab.γ[i, j] / (d * Δt) * ΔY[j] + tab.β[i, j] / d * S[j]
        end
        T = d * Δt
        X = exp([T*L T*I; zeros(n, 2n)])
        Z = X[1:n, 1:n] * Z0 + X[1:n, n+1:2n] * F
        ΔY[i] = Z - q
    end
    return Z
end

"""Lawson step for q' = L q + G(q) with the exact propagator e^{τL} (same node ordering and
caching as the production `mis_step!` for a `LawsonTableau`)."""
function mis_step_exact(L::AbstractMatrix, G, q::AbstractVector, Δt::Real, tab::LawsonTableau)
    s, nodes = nstages(tab), tab.nodes
    E = Dict{Float64,Matrix{Float64}}()
    prop(x, τ) = get!(() -> exp(τ * Δt * L), E, τ) * x
    cache = [copy(q)]
    cached(k) = (while length(cache) < k; push!(cache, prop(cache[end], nodes[length(cache)+1] - nodes[length(cache)])); end; cache[k])
    S = Vector{Vector{Float64}}(undef, s)
    function stage(wts, c)
        c0 = lawson_start(tab, wts, c)
        k0 = findfirst(==(c0), nodes)
        X = copy(cached(k0)); cur = c0
        for k in k0:length(nodes)
            ν = nodes[k]; ν > c && break
            ν > cur && (X = prop(X, ν - cur); cur = ν)
            for j in 1:s
                (wts[j] != 0 && tab.c[j] == ν) && (X += Δt * wts[j] * S[j])
            end
        end
        cur < c && (X = prop(X, c - cur))
        return X
    end
    for i in 1:s
        S[i] = G(i == 1 ? q : stage(tab.A[i, :], tab.c[i]))
    end
    return stage(tab.b, 1.0)
end

mis_rk_tableau(tab::LawsonTableau) = (tab.A, tab.b, tab.c)

"""ETD step for q' = L q + G(q) with exact fast stages: augmented exponential with the
monomials u_k = τ^{k-1} of the local time (u_k' = (k-1) u_{k-1})."""
function mis_step_exact(L::AbstractMatrix, G, q::AbstractVector, Δt::Real, tab::ETDTableau)
    s, K, n = nstages(tab), size(tab.P, 3), length(q)
    S = Vector{Vector{Float64}}(undef, s)
    Y = copy(q)
    for i in 1:s
        S[i] = G(Y)
        T = tab.c[i] * Δt
        M = zeros(n + K, n + K)
        M[1:n, 1:n] = T * L
        for k in 1:K
            M[1:n, n+k] = T * sum(tab.P[i, j, k] * S[j] for j in 1:i)
            k > 1 && (M[n+k, n+k-1] = k - 1)
        end
        Y = (exp(M) * vcat(q, 1.0, zeros(K - 1)))[1:n]
    end
    return Y
end

"""RK tableau of the ETD method for L = 0 (φ_k(0) = 1/k!)."""
function mis_rk_tableau(tab::ETDTableau)
    s, K = nstages(tab), size(tab.a, 3)
    B = [sum(tab.a[i, j, k] / factorial(k) for k in 1:K) for i in 1:s, j in 1:s]
    A = zeros(s, s)
    for i in 2:s
        A[i, :] = B[i-1, :]
    end
    return A, B[s, :], A * ones(s)
end

# Non-commuting test problem: fast rotation, nonlinear slow part
const ORDER_TEST_L = [0.0 4.0; -4.0 0.0]
order_test_G(q) = [0.5sin(q[1]) + q[2]^2, -0.3q[1] * q[2] + cos(q[1])]

function _rk4_reference(L, G, q, T, n)
    f(y) = L * y + G(y)
    h = T / n
    for _ in 1:n
        k1 = f(q); k2 = f(q + h / 2 * k1); k3 = f(q + h / 2 * k2); k4 = f(q + h * k3)
        q = q + h / 6 * (k1 + 2k2 + 2k3 + k4)
    end
    return q
end

"""
    mis_observed_order(tab; L, G, q0, T=1, n=(10, 20, 40)) -> (errors, rates)

Errors at t = T of MIS with exact fast stages for q' = L q + G(q) (default: rotation with
frequency 4 and a nonlinear slow part, so L and G do not commute) against a fine RK4 solution,
and the observed rates between successive step counts `n`. Includes the MIS coupling
conditions, which `rk_order` does not see.
"""
function mis_observed_order(tab::SplitTableau; L = ORDER_TEST_L, G = order_test_G, q0 = [1.0, 0.5],
                            T = 1.0, n = (10, 20, 40))
    qref = _rk4_reference(L, G, q0, T, 20000)
    err = map(n) do m
        q = copy(q0)
        for _ in 1:m
            q = mis_step_exact(L, G, q, T / m, tab)
        end
        norm(q - qref)
    end
    rates = [log2(err[k] / err[k+1]) * log(2) / log(n[k+1] / n[k]) for k in 1:length(n)-1]
    return collect(err), rates
end

"""Observed order of the exponential Adams–Bashforth method with exact flows; the k-1 start-up
values are steps of `tab.startup` with exact fast stages."""
function mis_observed_order(tab::EABTableau; L = ORDER_TEST_L, G = order_test_G, q0 = [1.0, 0.5],
                            T = 1.0, n = (10, 20, 40))
    qref = _rk4_reference(L, G, q0, T, 20000)
    err = map(n) do m
        h = T / m
        ys = [copy(q0)]
        for _ in 1:min(tab.k - 1, m)
            pushfirst!(ys, mis_step_exact(L, G, ys[1], h, tab.startup))
        end
        for _ in length(ys):m
            pushfirst!(ys, eab_step_exact(L, G, ys[1:tab.k], h, tab))
        end
        norm(ys[1] - qref)
    end
    rates = [log2(err[k] / err[k+1]) * log(2) / log(n[k+1] / n[k]) for k in 1:length(n)-1]
    return collect(err), rates
end
