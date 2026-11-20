# B-series of MIS methods with polynomial coupling (Theorem 4.1 of the note): for t = [t₁ … t_m],
#     ζ_i(τ; t) = Σ_j α_ij η_j(t) + ∫₀^τ [ Σ_j g_ij η_j(t) + Σ_j p_ij Π_{black} η_j(t_k) Π_{white} ζ_i(·; t_k) ],
#     η_i(t) = ζ_i(1; t),  η₁(t) = 0,
# order p ⇔ η_{s+1}(t) = 1/γ(t) for all trees with ≤ p vertices.

"""
    residuals(m::PolyMIS, p; class = :G, ts = trees(p; class))

Residuals η_{s+1}(t) - 1/γ(t) for the trees `ts` (generic arithmetic: exact with a Rational method).
"""
function residuals(m::PolyMIS, p::Int; class::Symbol = :G, ts::Vector{Tree} = trees(p; class))
    T = eltype(m.α)
    n = size(m.α, 1)
    η = Vector{Vector{T}}(undef, length(ts))          # η[t][i]
    ζ = Vector{Vector{Vector{T}}}(undef, length(ts))  # ζ[t][i] polynomial in τ
    res = Vector{T}(undef, length(ts))
    for (k, t) in enumerate(ts)
        ηk = zeros(T, n); ζk = [T[] for _ in 1:n]
        for i in 2:n
            φ = T[]
            for j in 1:i-1
                φ = padd(φ, pscale(m.g[i, j], ηk[j]))
                r = T[one(T)]
                for (w, c) in t.children
                    r = w ? pmul(r, ζ[c][i]) : pscale(r, η[c][j])
                end
                φ = padd(φ, pmul(m.p[i, j], r))
            end
            z0 = sum(m.α[i, j] * ηk[j] for j in 1:i-1)
            ζk[i] = padd(T[z0], pantider(φ))
            ηk[i] = peval(ζk[i], one(T))
        end
        η[k] = ηk; ζ[k] = ζk
        res[k] = ηk[n] - one(T) / t.γ
    end
    return res
end

"""Order of the method for the problem class (largest p with all residuals ≤ tol, up to pmax)."""
function order(m::PolyMIS; class::Symbol = :G, pmax::Int = 5, tol = 1e-10)
    ts = trees(pmax; class)
    r = residuals(m, pmax; class, ts)
    for p in 1:pmax
        any(abs(r[k]) > tol for k in eachindex(ts) if ts[k].n == p) && return p - 1
    end
    return pmax
end

"""Euclidean norm of the residuals of order p (error constant of an order p-1 method)."""
function error_norm(m::PolyMIS, p::Int; class::Symbol = :G)
    ts = trees(p; class)
    r = residuals(m, p; class, ts)
    return sqrt(sum(abs2(r[k]) for k in eachindex(ts) if ts[k].n == p))
end
