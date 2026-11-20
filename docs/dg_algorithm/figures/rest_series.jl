# rest over the hill: max|m| over time, favoured setting; writes the series to ARGS[1]
# run twice: with the current code and with commit a149165 (julia --project=<repo> rest_series.jl rest_new.bin / rest_old.bin)
using MultiRate, Printf, LinearAlgebra, Serialization
const E2 = MultiRate.Euler2D; const TS = MultiRate.TimeStepping; const CM = MultiRate.Common
L, H = 80e3, 10e3
hill(x) = 400 / (1 + ((x - L / 2) / 5e3)^2)
Tiso = 250.0; gas = CM.Gas()
M = E2.Mesh2D(8, 4, 3, L, H; zs = hill)
ρi(z) = gas.p00 / (gas.R * Tiso) * exp(-gas.g * z / (gas.R * Tiso))
Θi(z) = gas.p00 / gas.R * (gas.R * Tiso * ρi(z) / gas.p00)^(1 / gas.γ)
base = E2.Euler2DModel(M, gas, ρi, Θi)
qi = CM.project2d(base, (x, z) -> (ρi(z), 0.0, 0.0, Θi(z)))
mom(q) = maximum(abs, q[E2.node_iq(M.np, e, r, i, j)] for r in (2, 3), e in 1:M.Ne, j in 1:M.np, i in 1:M.np)
res = Dict{String,Any}()
for (lab, vs, vsol) in (("contravariant, HDG", :contravariant_momentum, :hdg), ("full, LU", :full, :lu))
    m = E2.DGPressureModel(base; lmars = :momentum, vsplit = vs, vsolver = vsol, hpredict = false)
    q = E2.pressure_state(m, qi); t = [0.0]; e = [mom(q)]
    for k in 1:60
        q = TS.run_frozen(m, q, 30.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 24, fast = :cs, offcenter = 0.1, smdiv = 0.1)
        push!(t, 60k); push!(e, mom(q))
    end
    @printf("%s: max|m| at 1 h %.2e\n", lab, e[end]); res[lab] = (t, e)
end
serialize(ARGS[1], res)
