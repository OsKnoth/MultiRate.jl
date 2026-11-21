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
