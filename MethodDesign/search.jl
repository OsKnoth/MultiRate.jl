# Search for MIS methods that keep their order with a frozen (non-additive) F(y, z)
#
#     julia --project=MethodDesign MethodDesign/search.jl [s] [p] [nstart]
#
# Unknowns: α, γ (strictly lower), β (lower incl. diagonal) of an s-stage MIS method. Objective:
# Σ residual² of all bicoloured trees with ≤ p vertices (BiColourMIS), plus penalties that keep the
# stage times d_i = Σ_j β_ij ≥ 0.1 (forward fast integration) and the coefficients moderate.
# Gradient by forward-mode AD through the B-series recursion. Prints the best methods found with
# their residual norms and stage times; save a result with `TS.MISTableau(name, α, β, γ)`.

include(joinpath(@__DIR__, "design_tools.jl"))
using Optim, Printf, Random, ADTypes

s = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
p = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
nstart = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 20
nvar = nvars(s)

function objective(x)
    r = resvec(x, s, p)
    return sum(abs2, r) + 1e-6 * sum(abs2, x)             # moderate coefficients
end

Random.seed!(1)
results = []
for k in 1:nstart
    x0 = 0.5 .* randn(nvar)
    res = optimize(objective, x0, LBFGS(), Optim.Options(iterations = 5000, g_tol = 1e-14); autodiff = AutoForwardDiff())
    x, _ = refine(Optim.minimizer(res), s, p)
    α, β, γ = unpack(x, s)
    r = residuals(mis_method("x", α, β, γ), p)
    rmax = maximum(abs(v) for (_, v) in r)
    push!(results, (rmax, x))
    @printf("start %2d: max residual %.1e, d = %s, max|coef| %.2f\n", k, rmax,
            join([@sprintf("%.3f", v) for v in vec(sum(β; dims = 2))], " "), maximum(abs, x))
end
sort!(results; by = first)
best = results[1][2]
α, β, γ = unpack(best, s)
println("\nbest: max residual $(results[1][1])")
println("α = ", round.(α; digits = 12)); println("β = ", round.(β; digits = 12)); println("γ = ", round.(γ; digits = 12))
m = mis_method("best", α, β, γ)
@printf("order (all bicoloured trees): %d, additive trees: %d\n", order_of(m, p + 1; tol = 1e-8),
        order_of(m, p + 1; additive_only = true, tol = 1e-8))
