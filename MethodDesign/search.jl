# Search for MIS methods that keep their order with a frozen (non-additive) F(y, z)
#
#     julia --project=MethodDesign MethodDesign/search.jl [s] [p] [nstart]
#
# Unknowns: α, γ (strictly lower), β (lower incl. diagonal) of an s-stage MIS method. Objective:
# Σ residual² of all bicoloured trees with ≤ p vertices (BiColourMIS), plus penalties that keep the
# stage times d_i = Σ_j β_ij ≥ 0.1 (forward fast integration) and the coefficients moderate.
# Gradient by forward-mode AD through the B-series recursion. Prints the best methods found with
# their residual norms and stage times; save a result with `TS.MISTableau(name, α, β, γ)`.

include(joinpath(@__DIR__, "BiColourMIS.jl"))
using .BiColourMIS, Optim, Printf, Random, LinearAlgebra, ADTypes, ForwardDiff

s = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
p = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
nstart = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20

const IA = [(i, j) for i in 1:s for j in 1:i-1]          # α, γ
const IB = [(i, j) for i in 1:s for j in 1:i]            # β
nvar = 2length(IA) + length(IB)

function unpack(x::AbstractVector{T}) where {T}
    α = zeros(T, s, s); γ = zeros(T, s, s); β = zeros(T, s, s)
    k = 0
    for (i, j) in IA; α[i, j] = x[k+=1]; end
    for (i, j) in IA; γ[i, j] = x[k+=1]; end
    for (i, j) in IB; β[i, j] = x[k+=1]; end
    return α, β, γ
end

function objective(x)
    α, β, γ = unpack(x)
    r = residuals(mis_method("x", α, β, γ), p)
    f = sum(abs2(v) for (_, v) in r)
    d = sum(β; dims = 2)
    f += 10 * sum(max(0, 0.1 - di)^2 for di in d)        # stage times d_i ≥ 0.1
    f += 1e-6 * sum(abs2, x)                              # moderate coefficients
    return f
end

"""Residual vector (all trees ≤ p, stage-time penalties) for the Gauss–Newton refinement."""
function resvec(x)
    α, β, γ = unpack(x)
    r = [v for (_, v) in residuals(mis_method("x", α, β, γ), p)]
    d = vec(sum(β; dims = 2))
    return vcat(r, [3 * max(zero(di), 0.1 - di) for di in d])
end

"""Levenberg–Marquardt with minimum-norm steps (underdetermined system), from the optimizer's result."""
function refine(x; iters = 100)
    μ = 1e-6
    r = resvec(x)
    for _ in 1:iters
        maximum(abs, r) < 1e-15 && break
        J = ForwardDiff.jacobian(resvec, x)
        xn = x - J' * ((J * J' + μ * I) \ r)
        rn = all(isfinite, xn) ? resvec(xn) : fill(Inf, length(r))
        if all(isfinite, rn) && sum(abs2, rn) < sum(abs2, r)
            x, r, μ = xn, rn, max(μ / 10, 1e-16)
        else
            μ *= 10
            μ > 1e6 && break
        end
    end
    return x
end

Random.seed!(1)
results = []
for k in 1:nstart
    x0 = 0.5 .* randn(nvar)
    res = optimize(objective, x0, LBFGS(), Optim.Options(iterations = 5000, g_tol = 1e-14); autodiff = AutoForwardDiff())
    x = refine(Optim.minimizer(res))
    α, β, γ = unpack(x)
    r = residuals(mis_method("x", α, β, γ), p)
    rmax = maximum(abs(v) for (_, v) in r)
    push!(results, (rmax, x))
    @printf("start %2d: max residual %.1e, d = %s, max|coef| %.2f\n", k, rmax,
            join([@sprintf("%.3f", v) for v in vec(sum(β; dims = 2))], " "), maximum(abs, x))
end
sort!(results; by = first)
best = results[1][2]
α, β, γ = unpack(best)
println("\nbest: max residual $(results[1][1])")
println("α = ", round.(α; digits = 12)); println("β = ", round.(β; digits = 12)); println("γ = ", round.(γ; digits = 12))
m = mis_method("best", α, β, γ)
@printf("order (all bicoloured trees): %d, additive trees: %d\n", order_of(m, p + 1; tol = 1e-8),
        order_of(m, p + 1; additive_only = true, tol = 1e-8))
