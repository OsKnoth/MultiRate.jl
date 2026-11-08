# Shared helpers of the method searches: packing of the MIS coefficients, order residuals,
# projection onto the order conditions (Levenberg–Marquardt with minimum-norm steps).

include(joinpath(@__DIR__, "BiColourMIS.jl"))
using .BiColourMIS, LinearAlgebra, ForwardDiff

lower_pairs(s) = [(i, j) for i in 1:s for j in 1:i-1]      # α, γ
lowerdiag_pairs(s) = [(i, j) for i in 1:s for j in 1:i]    # β
nvars(s) = 2length(lower_pairs(s)) + length(lowerdiag_pairs(s))

"""x ↦ (α, β, γ) of an s-stage MIS method."""
function unpack(x::AbstractVector{T}, s) where {T}
    α = zeros(T, s, s); γ = zeros(T, s, s); β = zeros(T, s, s)
    k = 0
    for (i, j) in lower_pairs(s); α[i, j] = x[k+=1]; end
    for (i, j) in lower_pairs(s); γ[i, j] = x[k+=1]; end
    for (i, j) in lowerdiag_pairs(s); β[i, j] = x[k+=1]; end
    return α, β, γ
end

"""(α, β, γ) ↦ x."""
pack(α, β, γ) = (s = size(α, 1); vcat([α[i, j] for (i, j) in lower_pairs(s)], [γ[i, j] for (i, j) in lower_pairs(s)],
                                       [β[i, j] for (i, j) in lowerdiag_pairs(s)]))

"""Residuals of all bicoloured trees ≤ p (additive_only: additive trees) and the stage-time
constraints d_i = Σ_j β_ij ≥ dmin."""
function resvec(x, s, p; additive_only = false, dmin = 0.1)
    α, β, γ = unpack(x, s)
    r = [v for (_, v) in residuals(mis_method("x", α, β, γ), p; additive_only)]
    d = vec(sum(β; dims = 2))
    return vcat(r, [3 * max(zero(di), dmin - di) for di in d])
end

"""Project x onto the order conditions (Levenberg–Marquardt, minimum-norm steps). Returns (x, max|r|)."""
function refine(x, s, p; iters = 100, additive_only = false)
    f(y) = resvec(y, s, p; additive_only)
    μ = 1e-6
    r = f(x)
    for _ in 1:iters
        maximum(abs, r) < 1e-15 && break
        J = ForwardDiff.jacobian(f, x)
        xn = x - J' * ((J * J' + μ * I) \ r)
        rn = all(isfinite, xn) ? f(xn) : fill(Inf, length(r))
        if all(isfinite, rn) && sum(abs2, rn) < sum(abs2, r)
            x, r, μ = xn, rn, max(μ / 10, 1e-16)
        else
            μ *= 10
            μ > 1e6 && break
        end
    end
    return x, maximum(abs, r)
end

"""Norm of the residuals of order p+1 (error constant)."""
function error_constant(x, s, p)
    α, β, γ = unpack(x, s)
    r = residuals(mis_method("x", α, β, γ), p + 1)
    return sqrt(sum(abs2(v) for (t, v) in r if order(t) == p + 1))
end
