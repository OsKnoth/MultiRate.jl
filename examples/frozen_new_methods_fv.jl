# New frozen-split MIS methods (MethodDesign/) on the FV gravity wave: wave error at T = 3000 s
# against RK4 and a long run (36000 s, Δt = 200 s, 180 steps) that shows weak linear instabilities.
#     julia --project=examples examples/frozen_new_methods_fv.jl


include(joinpath(@__DIR__, "euler2d_common.jl"))

L, H, a, U, T = 300e3, 10e3, 5e3, 20.0, 3000.0
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
model = build_model(:fv; L, H, P = 4, NEX = 30, NEZ = 4, ρr, Θr, gas)
xc = L / 3
q0 = project2d(model, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / H) / (1 + ((x - xc) / a)^2));
                                 (ρ, ρ * U, 0.0, Θr(z))))
g = model.grid
c = sound_speed_max(model, q0)
nref = ceil(Int, T / (0.4 * min(g.dx, g.dz) / (c + U)))
qref = TS.run_rk4(model, q0, T / nref, nref)
xs, yr, _ = field_section(model, qref, "theta_prime", H / 2)
θs(q) = field_section(model, q, "theta_prime", H / 2)[2]
wave(y) = y .- sum(y) / length(y)
err(q) = all(isfinite, q) ? maximum(abs.(wave(θs(q)) .- wave(yr))) : Inf
dτ = Dict(:verlet => 0.9 * g.dz / c, :hevi => fast_step(model, q0, :hevi))


steps = (45, 30, 23, 18, 15)
for name in (:MIS54, :MISF53, :MIS_FV42, :MIS_FV53)
    tab = TS.mis_tableau(name); es = Float64[]
    for n in steps
        Δt = T / n; m = ceil(Int, 1.01 * Δt / dτ[:hevi]); set_vadv!(U, Δt, H / 16)
        q = TS.run_frozen(model, q0, Δt, n; tab, nsub = m, fast = :hevi)
        push!(es, err(q))
    end
    @printf("%-9s frozen hevi wave error at Δt = 67/100/130/167/200: %s
", name, join([@sprintf("%.1e", e) for e in es], " "))
    flush(stdout)
end
# long run: 36000 s at Δt = 200 s (180 steps)
for name in (:MIS54, :MISF53, :MIS_FV42, :MIS_FV53)
    Δt = 200.0; n = 180; m = ceil(Int, 1.01 * Δt / dτ[:hevi]); set_vadv!(U, Δt, H / 16)
    q = try TS.run_frozen(model, q0, Δt, n; tab = TS.mis_tableau(name), nsub = m, fast = :hevi) catch e; fill(NaN, length(q0)) end
    w = all(isfinite, q) ? maximum(abs, field_section(model, q, "w", H / 2)[2]) : NaN
    @printf("%-9s long run 36000 s, Δt = 200 s: max|w| at z = 5 km %.3e m/s, max|θ′| %.3e K
", name, w,
            all(isfinite, q) ? maximum(abs, θs(q)) : NaN)
    flush(stdout)
end
