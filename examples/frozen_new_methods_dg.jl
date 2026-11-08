# New frozen-split MIS methods (MethodDesign/) on the DG gravity wave (P = 4, 30 × 4 elements,
# explicit Verlet fast loop): wave error at T = 3000 s against RK4 for Δt = 67 … 200 s.
#     julia --project=examples examples/frozen_new_methods_dg.jl

include(joinpath(@__DIR__, "euler2d_common.jl"))
L, H, a, U, T = 300e3, 10e3, 5e3, 20.0, 3000.0
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
model = build_model(:dg; L, H, P = 4, NEX = 30, NEZ = 4, ρr, Θr, gas)
q0 = project2d(model, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / H) / (1 + ((x - L/3) / a)^2)); (ρ, ρ * U, 0.0, Θr(z))))
c = sound_speed_max(model, q0)
dτ = fast_step(model, q0, :verlet)
nref = ceil(Int, T / (0.5dτ))
t = @elapsed qref = TS.run_rk4(model, q0, T / nref, nref)
xs, yr, _ = field_section(model, qref, "theta_prime", H / 2)
θs(q) = field_section(model, q, "theta_prime", H / 2)[2]
wave(y) = y .- sum(y) / length(y)
err(q) = all(isfinite, q) ? maximum(abs.(wave(θs(q)) .- wave(yr))) : Inf
@printf("DG: verlet substep %.2f s, RK4 reference %d steps (%.0f s), max θ' %.2e\n", dτ, nref, t, maximum(yr))
for name in (:MIS54, :MISF53, :MIS_FV53, :MIS_DG53), (kind, fast) in ((:frozen, :verlet),)
    es = String[]
    for n in (45, 23, 20, 18, 15)
        Δt = T / n
        m = ceil(Int, 1.01Δt / fast_step(model, q0, fast))
        tt = @elapsed q = try
            kind == :frozen ? TS.run_frozen(model, q0, Δt, n; tab = TS.mis_tableau(name), nsub = m, fast) :
                              TS.run_mis(model, q0, Δt, n; tab = TS.mis_tableau(name), nsub = m, fast)
        catch e; e isa DomainError || rethrow(); fill(NaN, length(q0)) end
        push!(es, @sprintf("%.1e (nsub %d, %.0f s)", err(q), m, tt))
    end
    @printf("%-6s %-10s %-7s Δt 67/130/150/167/200: %s\n", name, kind, fast, join(es, "  "))
end
