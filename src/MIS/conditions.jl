# Explicit forms of the order conditions:
#   * restart class (g = 0, Z_i(0) = Y_{k(i)}): Butcher-like matrix form with the weighted moment matrices
#     W⁽ᵐ⁾_ij = ∫₀¹ p_ij(σ) z_i(σ)ᵐ dσ, z_i(σ) = c_{k(i)} + Q_i(σ) (mis_theory.tex, Proposition "matrix form"),
#   * any method: LaTeX table of the Runge–Kutta-like form Σ_i b̂_i d_i(t) = 1/γ(t) (Theorem 5.1) for all trees of a class.

"""Start stage k(i) of each stage of a restart-class method (`nothing` if the method is not of restart type)."""
function restart_starts(m::PolyMIS; tol = 1e-14)
    n = size(m.α, 1)
    all(all(abs(x) <= tol for x in m.g[i, j]) for i in 2:n for j in 1:i-1) || return nothing
    k = ones(Int, n)
    for i in 2:n
        nz = findall(x -> abs(x) > tol, m.α[i, 1:i-1])
        isempty(nz) && continue
        (length(nz) == 1 && abs(m.α[i, nz[1]] - 1) <= tol) || return nothing
        k[i] = nz[1]
    end
    return k
end

"""
    restart_matrices(m) -> (; c, Λ, bh, W, A, b)

Quantities of the matrix form for a restart-class method: stage times `c`, chain matrix Λ = (I - α)⁻¹, weights
b̂ = Λ[s+1, :], moment matrices `W[m+1]` = W⁽ᵐ⁾ (m = 0, 1, 2), the base method 𝔄 = Λ W⁽⁰⁾ and its weights b.
"""
function restart_matrices(m::PolyMIS)
    k = restart_starts(m)
    k === nothing && throw(ArgumentError("not a restart-class method (g = 0, rows of α unit vectors)"))
    T = eltype(m.α); n = size(m.α, 1)
    c = stage_times(m)
    Λ = inv(Matrix{T}(I, n, n) - m.α)
    W = [zeros(T, n, n) for _ in 0:2]
    for i in 2:n
        Q = reduce(padd, (pantider(m.p[i, j]) for j in 1:i-1))
        z = padd(T[c[k[i]]], Q)
        zm = T[one(T)]
        for mm in 0:2
            for j in 1:i-1; W[mm+1][i, j] = pint1(pmul(m.p[i, j], zm)); end
            zm = pmul(zm, z)
        end
    end
    A = Λ * W[1]
    return (; c, Λ, bh = Λ[n, :], W, A, b = A[n, :])
end

"""
    restart_conditions(m) -> Vector of (name, residual)

Residuals of the order conditions up to order four of a restart-class method in matrix form: the eight classical
conditions of the base method 𝔄 (b, A, c of the Runge–Kutta method, stages 1 … s) and the coupling conditions
b̂ᵀW⁽¹⁾c = 1/3 (order 3) and b̂ᵀW⁽¹⁾c² = 1/4, b̂ᵀW⁽²⁾c = 1/4, b̂ᵀW⁽¹⁾𝔄c = 1/8, b̂ᵀW⁽⁰⁾ΛW⁽¹⁾c = 1/12 (order 4).
"""
function restart_conditions(m::PolyMIS)
    q = restart_matrices(m)
    c, bh, W, A, Λ = q.c, q.bh, q.W, q.A, q.Λ
    b = q.b; Ac = A * c
    return [("b'1 = 1", sum(b) - 1), ("b'c = 1/2", b' * c - 1 // 2),
            ("b'c^2 = 1/3", b' * c .^ 2 - 1 // 3), ("b'Ac = 1/6", b' * Ac - 1 // 6),
            ("b'c^3 = 1/4", b' * c .^ 3 - 1 // 4), ("b'(c.Ac) = 1/8", b' * (c .* Ac) - 1 // 8),
            ("b'Ac^2 = 1/12", b' * (A * c .^ 2) - 1 // 12), ("b'AAc = 1/24", b' * (A * Ac) - 1 // 24),
            ("bh'W1 c = 1/3", bh' * W[2] * c - 1 // 3),
            ("bh'W1 c^2 = 1/4", bh' * W[2] * c .^ 2 - 1 // 4), ("bh'W2 c = 1/4", bh' * W[3] * c - 1 // 4),
            ("bh'W1 Ac = 1/8", bh' * W[2] * Ac - 1 // 8), ("bh'W0 Λ W1 c = 1/12", bh' * W[1] * Λ * W[2] * c - 1 // 12)]
end

# ---------------------------------------------------------------------------------------------------------------
# LaTeX table of the conditions in Runge–Kutta-like form

"""forest code of tree k (root drawn black, children coloured)."""
function forest(ts::Vector{Tree}, k::Int)
    sub(w, k) = "[,$(w ? "w" : "b")" * join(" " * sub(w2, c) for (w2, c) in ts[k].children) * "]"
    return "\\btree{[,b" * join(" " * sub(w, c) for (w, c) in ts[k].children) * "]}"
end

function powstr(base, e)
    e == 1 && return base
    return occursin(r"[(]", base) ? "\\big($base\\big)^{$e}" : "$(base)^{$e}"
end

"""d_i(t) of Theorem 5.1 as a LaTeX string; subtrees of more than one vertex are drawn inline."""
function dstring(ts::Vector{Tree}, k::Int)
    t = ts[k]
    isempty(t.children) && return "Q_i(1)"
    blacks = Dict{Int,Int}(); whites = Dict{Int,Int}()
    for (w, c) in t.children; d = w ? whites : blacks; d[c] = get(d, c, 0) + 1; end
    η(c) = ts[c].n == 1 ? "c_j" : "\\eta_j($(forest(ts, c)))"
    ζ(c) = ts[c].n == 1 ? "z_i(\\sigma)" : "\\zeta_i(\\sigma;$(forest(ts, c)))"
    bpart = join((powstr(η(c), e) for (c, e) in sort(collect(blacks))), "\\,")
    wpart = join((powstr(ζ(c), e) for (c, e) in sort(collect(whites))), "\\,")
    isempty(whites) && return "\\sum_j P_{ij}(1)\\,$bpart"
    isempty(blacks) && return "\\int_0^1 q_i(\\sigma)\\,$wpart\\dd\\sigma"
    return "\\sum_j $bpart\\int_0^1 p_{ij}(\\sigma)\\,$wpart\\dd\\sigma"
end

"""
    latex_conditions(p; class = :G) -> String

LaTeX longtable of the order conditions Σ_i b̂_i d_i(t) = 1/γ(t) for all trees of order ≤ p of the class (`:G`, `:B`,
`:A`, `:L`), in the notation of mis_theory.tex (needs its \\btree macro and \\dd). Column "R": `F` for the purely
fast trees, which hold automatically in the restart class (Theorem 5.3).
"""
function latex_conditions(p::Int; class::Symbol = :G)
    ts = trees(p; class)
    allwhite(k) = all(w && allwhite(c) for (w, c) in ts[k].children)
    io = IOBuffer()
    println(io, "{\\renewcommand{\\arraystretch}{1.5}\n\\begin{longtable}{rclll}\n\\toprule\nNo. & \$t\$ & \$d_i(t)\$ & \$1/\\gamma(t)\$ & R\\\\\\midrule\n\\endhead\n\\bottomrule\n\\endfoot")
    no = 0
    for k in eachindex(ts)
        no += 1
        fl = ts[k].n > 1 && allwhite(k) ? "F" : ""
        println(io, "$no & $(forest(ts, k)) & \$$(dstring(ts, k))\$ & \$\\frac1{$(ts[k].γ)}\$ & $fl\\\\[3pt]")
        k < length(ts) && ts[k+1].n > ts[k].n && println(io, "\\midrule")
    end
    println(io, "\\end{longtable}}")
    return String(take!(io))
end

# ---------------------------------------------------------------------------------------------------------------
# Number of independent conditions per order: rank of the Jacobian of the residuals with respect to the coefficients
# at a random method (complex-step derivatives, exact to round-off since the residuals are polynomials).

"""Random method with free coefficients `x`: restart class (g = 0, chain `k`) or general α, g; coupling degree K."""
function random_family(x, s, K; restart::Bool, k = nothing)
    T = eltype(x); n = s + 1; c = 0
    α = zeros(T, n, n); p = fill(T[], n, n); g = fill(T[], n, n)
    for i in 2:n
        restart && k[i] > 1 && (α[i, k[i]] = one(T))
        for j in 1:i-1
            restart || (α[i, j] = x[c+=1]; g[i, j] = x[c+1:c+K+1]; c += K + 1)
            p[i, j] = x[c+1:c+K+1]; c += K + 1
            restart && (g[i, j] = [zero(T)])
        end
    end
    return PolyMIS(α, p, g)
end
family_size(s, K; restart::Bool) = (restart ? K + 1 : 2K + 3) * s * (s + 1) ÷ 2

function family_jacobian(x, s, K, k, restart, p, class, ts)
    r = residuals(random_family(x, s, K; restart, k), p; class, ts)
    J = zeros(length(ts), length(x))
    for l in eachindex(x)
        xc = complex.(x); xc[l] += 1e-30im
        J[:, l] = imag.(residuals(random_family(xc, s, K; restart, k), p; class, ts)) ./ 1e-30
    end
    return r, J
end

"""Minimum-norm Gauss–Newton iteration onto the set where the residuals of all trees of order ≤ q vanish."""
function project_conditions(x, s, K, k, restart, q, class; it = 80, tol = 1e-11)
    ts = trees(q; class)
    res(x) = norm(residuals(random_family(x, s, K; restart, k), q; class, ts))
    for _ in 1:it
        r, J = family_jacobian(x, s, K, k, restart, q, class, ts)
        nr = norm(r); nr < tol && return x, true
        dx = pinv(J) * r; λ = 1.0                                          # backtracking on ‖r‖
        while λ > 1e-4 && !(res(x - λ * dx) < nr); λ /= 2; end
        x = x - λ * dx
        all(isfinite, x) || return x, false
    end
    return x, norm(residuals(random_family(x, s, K; restart, k), q; class, ts)) < tol
end

"""
    independent_conditions(p; class = :G, restart = true, s = 8, K = 2, samples = 4) -> (; total, base, coupling)

Numbers of independent order conditions of order q = 1 … p for the problem class, in the sense relevant for the
construction of methods: the codimension that the conditions of order q add to the set of methods of order q - 1. For
each q, random s-stage methods of coupling degree K (restart class with random chain, or general α, g_ij) are driven to
order q - 1 by a minimum-norm Gauss–Newton iteration, and the rank increment of the Jacobian of the residuals (complex
step, exact to round-off) is taken there; the maximum over `samples` methods (generic rank). `base` counts the black
trees (conditions of the base method), `coupling` the remainder. Samples whose iteration fails are skipped.
"""
function independent_conditions(p::Int; class::Symbol = :G, restart::Bool = true, s::Int = 8, K::Int = 2,
                                samples::Int = 4, rtol = 1e-8)
    ts = trees(p; class)
    black(j) = all(!w && black(c) for (w, c) in ts[j].children)
    N = family_size(s, K; restart)
    total = zeros(Int, p); base = zeros(Int, p)
    for q in 1:p, seed in 1:samples
        u(l) = (hash((seed, q, l)) % UInt(1 << 30)) / (1 << 30)          # deterministic pseudo-random in [0, 1)
        k = [1; [1 + floor(Int, u(-i) * (i - 1)) for i in 2:s+1]]
        x = [u(l) - 0.5 for l in 1:N]
        if q > 1
            x, ok = project_conditions(x, s, K, k, restart, q - 1, class)
            ok || continue
        end
        _, J = family_jacobian(x, s, K, k, restart, q, class, ts[1:count(t -> t.n <= q, ts)])
        rk(sel) = isempty(sel) ? 0 : rank(J[sel, :]; rtol)
        lo = [j for j in eachindex(ts) if ts[j].n < q]; hi = [j for j in eachindex(ts) if ts[j].n <= q]
        total[q] = max(total[q], rk(hi) - rk(lo))
        base[q] = max(base[q], rk(filter(black, hi)) - rk(filter(black, lo)))
    end
    return (; total, base, coupling = total .- base)
end
