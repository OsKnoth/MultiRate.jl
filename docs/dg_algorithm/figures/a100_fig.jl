using Serialization
# DG a100 Agnesi hill (hydrostatic), DGPressureModel with MIS54 + S2; comparison with the linear hydrostatic solution
include(joinpath(@__DIR__, "..", "..", "..", "examples", "euler2d_common.jl"))
a, U, NBV, h0 = 100e3, 10.0, 0.01, parse(Float64, get(ENV, "H0", "100"))
L, H, NEX, NEZ, P, ZD = 40a, 15600.0, 50, 16, 4, 9000.0
T = 43200.0; Δt = 300.0
gas = Gas(); ρr, Θr, _ = E2.stratified_reference(gas; θ0 = 300.0, N = NBV)
xm = L / 2; zs(x) = h0 / (1 + ((x - xm) / a)^2)
base0 = build_model(:dg; L, H, P, NEX, NEZ, ρr, Θr, gas, zs)
qb = project2d(base0, (x, z) -> (ρr(z), ρr(z) * U, 0.0, Θr(z)))
α = 5U * min(1.0, U / (NBV * a)) / (H - ZD)
base = with_sponge(base0, qb; zD = ZD, αtop = α)
M = base.mesh
c = sound_speed_max(base, qb)
dτh = 0.85 * 0.63 * (M.Lx / M.Nex) / ((2M.p + 1) * c); nsub = ceil(Int, Δt / dτh)
@printf("a100: %s, Δt = %.0f s, nsub %d (h = %.1f s), U Δt/Δz_node = %.1f, T = %.0f h\n", grid_info(base), Δt, nsub, Δt / nsub,
        U * Δt / (M.H / M.Nez / M.p), T / 3600)
l = NBV / U
wlin(x, z) = (X = x - xm; U * h0 * a * (-a * X * 2 * cos(l * z) / (X^2 + a^2)^2 - sin(l * z) * (a^2 - X^2) / (X^2 + a^2)^2))
for (lab, kw) in (("favoured: contravariant_momentum, hdg, hpredict=false", (vsplit = :contravariant_momentum, vsolver = :hdg, hpredict = false)),)
    m = E2.DGPressureModel(base; lmars = :momentum, kw...)
    q = E2.pressure_state(m, qb)
    nchunk = round(Int, T / Δt); t = 0.0; ok = true
    el = @elapsed while t < T - 1
        try
            q = TS.run_frozen(m, q, Δt, nchunk; tab = TS.mis_tableau(:MIS54), nsub, fast = :cs, offcenter = 0.1, smdiv = 0.1)
        catch e
            e isa DomainError || rethrow(); ok = false
        end
        t += nchunk * Δt
        w = E2.field(base, q[1:base.nq], 3) ./ E2.field(base, q[1:base.nq], 1)
        (ok && all(isfinite, w)) || (println("  $lab: failed at t = $(t/3600) h"); ok = false; break)
        sel = [(abs(M.X[k] - xm) < 15a && M.Z[k] < 8e3) for k in eachindex(M.X)]
        el = maximum(abs, (w .- wlin.(M.X, M.Z))[sel])
        @printf("  %-50s t = %5.1f h: max|w| %.3e m/s (linear %.3e), max|w - w_lin| (|x-xm|<15a, z<8 km) %.3e\n", lab, t / 3600,
                maximum(abs, w[sel]), maximum(abs, wlin.(M.X, M.Z)[sel]), el)
        flush(stdout)
    end
    @printf("  %s: %.0f s\n", lab, el)
    w = E2.field(base, q[1:base.nq], 3) ./ E2.field(base, q[1:base.nq], 1)
    serialize(joinpath(@__DIR__, "a100_fig.bin"), (X = vec(M.X), Z = vec(M.Z), w = vec(w), wl = vec(wlin.(M.X, M.Z)), xm = xm, a = a))
end
