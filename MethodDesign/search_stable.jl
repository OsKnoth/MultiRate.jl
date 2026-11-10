# Search for MIS methods of frozen order p with stability constraints as in Knoth & Wensch (2014)
#
#     julia --project=MethodDesign MethodDesign/search_stable.jl [s] [p] [nstart] [camax] [from]
#
# from (optional): name of a package tableau with s stages (e.g. MIS54, MIS4a). The first start is
# its minimum-norm projection onto the frozen-order-p conditions, the others perturb it (σ = 0.05).
# Random starts rarely reach the stable region; starting from a stable method is what found MISF53
# (MIS54 projected, |Δx| ≈ 0.1, C_adv limit ≥ 2.05 for C_sound ≤ 12). Without `from`: random starts.
#
# Every candidate x is projected onto all bicoloured order conditions up to order p (`refine`); the
# objective of the projected method is
#     Σ_{C_sound, C_adv} max(0, ρ(C_sound, C_adv) - 1)²  +  w · ‖order-(p+1) residuals‖²,
# ρ the maximum amplification of the Knoth–Wensch acoustic-advection test (C-grid, exact fast
# stages) over C_sound ∈ {0, 1, 2, 3, 4, 6, 8, 10, 12} and 8 values C_adv ∈ (0, camax] (a coarser
# grid misses instabilities between its points). Nelder–Mead (the amplification is
# not smooth). Starts: random points projected onto the order manifold. The best methods are
# checked with the stability limits per C_sound and printed with their coefficients.
# The projected start itself is kept as a candidate (Nelder–Mead may not improve it).

include(joinpath(@__DIR__, "design_tools.jl"))
include(joinpath(@__DIR__, "stability.jl"))
using Optim, Random
const CLASS = Symbol(get(ENV, "CLASS", "general"))   # tree class of the order conditions (BiColourMIS.in_class)

s = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
p = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
nstart = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
camax = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1.8
const CS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0)
const CA = Tuple(range(camax / 8, camax; length = 8))
const WERR = 1e-3
from = length(ARGS) >= 5 ? Symbol(ARGS[5]) : nothing

"""(α, β, γ) of a package tableau in the compact s×s form (drops a leading zero row/column)."""
function compact(tab)
    n = size(tab.α, 1)
    iszero(tab.β[1, 1]) && return tab.α[2:n, 1:n-1], tab.β[2:n, 1:n-1], tab.γ[2:n, 1:n-1]
    return tab.α, tab.β, tab.γ
end
xfrom = from === nothing ? nothing : pack(compact(TS.mis_tableau(from))...)

function tableau(x)
    α, β, γ = unpack(x, s)
    return TS.MISTableau(:search, α, β, γ)
end

function objective(x)
    xp, r = refine(x, s, p; iters = 30, class = CLASS)
    r > 1e-10 && return 1e3 + r
    return stability_penalty(tableau(xp); Cs = CS, Ca = CA, nmodes = 16) + WERR * error_constant(xp, s, p; class = CLASS)^2
end

Random.seed!(2)
best = []
for k in 1:nstart
    xs = xfrom === nothing ? 0.5 .* randn(nvars(s)) : xfrom .+ (k == 1 ? 0.0 : 0.05) .* randn(nvars(s))
    x0, r0 = refine(xs, s, p; class = CLASS)
    r0 > 1e-10 && continue
    f0 = objective(x0)
    @printf("start %d: penalty %.3e\n", k, f0)
    res = optimize(objective, x0, NelderMead(), Optim.Options(iterations = 3000, f_tol = 1e-12))
    xb, _ = refine(Optim.minimizer(res), s, p; class = CLASS)
    f = objective(xb)
    f > f0 && ((xb, f) = (x0, f0))
    lim = [adv_limit(tableau(xb), cs; nmodes = 24) for cs in CS]
    @printf("   -> objective %.3e, error constant %.3e, C_adv limits %s\n", f, error_constant(xb, s, p; class = CLASS),
            join([@sprintf("%.2f", l) for l in lim], " "))
    push!(best, (minimum(lim), f, xb))
end
sort!(best; by = b -> -b[1])
xb = best[1][3]
α, β, γ = unpack(xb, s)
m = mis_method("best", α, β, γ)
@printf("\nbest: min C_adv limit %.2f, frozen order %d, additive order %d, error constant %.3e\n", best[1][1],
        order_of(m, p + 1; class = CLASS, tol = 1e-9), order_of(m, p + 1; additive_only = true, tol = 1e-9), error_constant(xb, s, p; class = CLASS))
println("α = ", repr(α)); println("β = ", repr(β)); println("γ = ", repr(γ))
println("d = ", vec(sum(β; dims = 2)))
