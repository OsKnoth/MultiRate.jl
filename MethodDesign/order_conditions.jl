# Explicit order conditions of MIS with ODE stages for y' = F(y, y) (frozen first argument)
#
#     julia --project=MethodDesign MethodDesign/order_conditions.jl [s] [p]
#
# Symbolic residuals η_{s+1}(t) - 1/γ(t) of a general s-stage MIS method (α, γ strictly lower,
# β lower triangular incl. diagonal, ClimaTimeSteppers convention) for all bicoloured trees with
# ≤ p vertices, marked additive (A: also needed for F = f(y) + g(z)) or mixed (M: only for a
# non-additive F, e.g. a frozen coefficient in the fast part). Then the numerical residuals of the
# MIS tableaus of `MultiRate.TimeStepping`. Default s = 3, p = 3.

include(joinpath(@__DIR__, "BiColourMIS.jl"))
using .BiColourMIS, Symbolics, Printf, MultiRate
const TS = MultiRate.TimeStepping

s = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
p = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3

α = [j < i ? Symbolics.variable(:α, i, j) : Num(0) for i in 1:s, j in 1:s]
γ = [j < i ? Symbolics.variable(:γ, i, j) : Num(0) for i in 1:s, j in 1:s]
β = [j <= i ? Symbolics.variable(:β, i, j) : Num(0) for i in 1:s, j in 1:s]
m = mis_method("MIS$s", α, β, γ)
println("General $s-stage MIS, order conditions up to order $p (residual = 0):\n")
for (t, r) in residuals(m, p)
    @printf("%-2d %s %-12s %s = 0\n", order(t), additive(t) ? "A" : "M", string(t), string(Symbolics.expand(r)))
end

println("\nTableaus: max |residual| per order (additive trees / mixed trees)")
for name in (:WSRK3, :MIS2, :MIS3C, :MIS4, :MIS4a, :MIS54, :MIS_KW3)
    tb = TS.mis_tableau(name)
    r = residuals(mis_method(name, tb.α, tb.β, tb.γ), 4)
    cols = map(1:4) do n
        a = maximum((abs(x) for (t, x) in r if order(t) == n && additive(t)); init = 0.0)
        b = maximum((abs(x) for (t, x) in r if order(t) == n && !additive(t)); init = 0.0)
        @sprintf("%d: %.1e / %.1e", n, a, b)
    end
    @printf("%-8s %s\n", name, join(cols, "   "))
end
