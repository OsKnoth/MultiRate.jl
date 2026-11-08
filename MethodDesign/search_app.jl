# Search for MIS methods of frozen order p that are stable for one application (FV or DG), using
# the column Fourier blocks of `app_stability.jl` instead of the 1D Knoth–Wensch test.
#
#     julia --project=MethodDesign MethodDesign/search_app.jl app s p [from] [nstart] [camax]
#
# app ∈ fv, dg; from: package tableau to start from (projected onto the frozen-order-p conditions;
# further starts perturb it, σ = 0.05). Objective of a projected candidate:
#     Σ_{U, θ, C_adv ≤ camax} max(0, ρ - 1)²  +  10⁻³ ‖order-(p+1) residuals‖²
# on a reduced design model (fewer modes and winds; DG with 2 vertical elements). The best method is
# checked on the full model (`app_modes(app)`) and printed with its coefficients.

include(joinpath(@__DIR__, "design_tools.jl"))
include(joinpath(@__DIR__, "app_stability.jl"))
using Optim, Random

app = Symbol(ARGS[1])
s = parse(Int, ARGS[2]); p = parse(Int, ARGS[3])
from = length(ARGS) >= 4 && ARGS[4] != "-" ? Symbol(ARGS[4]) : nothing
nstart = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 3
camax = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : 2.0
const CA = range(camax / 6, camax; length = 6)

Mdesign = app === :fv ? app_modes(:fv; Us = (10.0, 20.0, 40.0), nθ = 8) :
                        app_modes(:dg; Us = (20.0, 40.0), nθ = 6, NEZ = 2)
@printf("design model %s: block %d, %d winds, %d modes\n", app, length(Mdesign.Λ[1]), length(Mdesign.Us), length(Mdesign.θs))

function objective(x)
    xp, r = refine(x, s, p; iters = 30)
    r > 1e-10 && return 1e3 + r
    return app_penalty(Mdesign, unpack(xp, s); Ca = CA) + 1e-3 * error_constant(xp, s, p)^2
end

Random.seed!(3)
xfrom = from === nothing ? nothing : pack(compact_tableau(TSA.mis_tableau(from))...)
best = []
for k in 1:nstart
    xs = xfrom === nothing ? 0.5 .* randn(nvars(s)) : xfrom .+ (k == 1 ? 0.0 : 0.05) .* randn(nvars(s))
    x0, r0 = refine(xs, s, p)
    r0 > 1e-10 && continue
    f0 = objective(x0)
    res = optimize(objective, x0, NelderMead(), Optim.Options(iterations = 800))
    xb, _ = refine(Optim.minimizer(res), s, p)
    f = objective(xb)
    f > f0 && ((xb, f) = (x0, f0))
    @printf("start %d: penalty %.3e -> %.3e, error constant %.3e\n", k, f0, f, error_constant(xb, s, p)); flush(stdout)
    push!(best, (f, xb))
end
sort!(best; by = first)
xb = best[1][2]
α, β, γ = unpack(xb, s)
m = mis_method("best", α, β, γ)
M = app_modes(app)
@printf("\nbest: objective %.3e, frozen order %d, additive order %d, error constant %.3e\n", best[1][1],
        order_of(m, p + 1; tol = 1e-9), order_of(m, p + 1; additive_only = true, tol = 1e-9), error_constant(xb, s, p))
@printf("full %s model, largest stable C_adv for U = %s m/s: %s\n", app, join(Int.(M.Us), "/"),
        join([@sprintf("%.1f", l) for l in app_limits(M, (α, β, γ))], " "))
println("α = ", repr(α)); println("β = ", repr(β)); println("γ = ", repr(γ))
println("d = ", vec(sum(β; dims = 2)))
