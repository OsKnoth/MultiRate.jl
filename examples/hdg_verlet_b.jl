# HDG fast operator (`HDGFast2D`: face momentum M̂ = M̂*, LMARS face pressure with the momentum jump as the
# only self-coupling D_b): explicit Störmer–Verlet (:verlet) against Störmer–Verlet with D_b integrated
# exactly in the half steps (Bokhove's partition, :verlet_b; docs/frozen_dg, Section sec:hdgb). DG gravity
# wave (300 km × 10 km, 30 × 4 elements of degree 4, U = 20 m/s, N = 0.01/s), flat or over an Agnesi hill,
# MIS of the refrozen split (`run_mis`), error of θ (minus its mean) at T = 3000 s against RK4.
#
#     [TERRAIN=flat|hill] [METHODS=MIS4a] [FASTS=verlet,verlet_b] [DTS=10,20] [CFLS=0.535,1.0,1.2]
#     julia --project=examples examples/hdg_verlet_b.jl
#
# CFL as in `fast_step`: dτ = CFL h/((2p+1) c), h = min(element width, element height). Stability of one fast
# step (flat and hill): :verlet ≤ 0.63, :verlet_b ≤ 1.4.

include(joinpath(@__DIR__, "euler2d_common.jl"))

L, H, a, U, T = 300e3, 10e3, 5e3, 20.0, 3000.0
const TERRAIN = get(ENV, "TERRAIN", "flat")
const METHODS = Symbol.(split(get(ENV, "METHODS", "MIS4a"), ","))
const FASTS = Symbol.(split(get(ENV, "FASTS", "verlet,verlet_b"), ","))
const DTS = parse.(Float64, split(get(ENV, "DTS", "10,20"), ","))
const CFLS = parse.(Float64, split(get(ENV, "CFLS", "0.535,1.0,1.2"), ","))
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
hill = TERRAIN == "hill"
zsf(x) = hill ? 400.0 / (1 + ((x - L / 2) / 10e3)^2) : 0.0
dzs(x) = hill ? -400.0 * 2 * (x - L / 2) / 10e3^2 / (1 + ((x - L / 2) / 10e3)^2)^2 : 0.0
base = build_model(:dg; L, H, P = 4, NEX = 30, NEZ = 4, ρr, Θr, gas, zs = zsf)
q0 = project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / H) / (1 + ((x - L / 3) / a)^2));
                                 (ρ, ρ * U, ρ * U * dzs(x) * (H - z) / (H - zsf(x)), Θr(z))))
M = base.mesh; c = sound_speed_max(base, q0)
h = min(M.Lx / M.Nex, minimum(M.Z[:, end, :] .- M.Z[:, 1, :]))
dτ(cfl) = cfl * h / ((2M.p + 1) * c)
nref = ceil(Int, T / (0.5 * dτ(0.535)))
tr = @elapsed qref = TS.run_rk4(base, q0, T / nref, nref)
θf(q) = (y = vec(E2.field(base, q, 4) ./ E2.field(base, q, 1)); y .- sum(y) / length(y))
yr = θf(qref)
err(q) = all(isfinite, q) ? maximum(abs.(θf(q) .- yr)) : Inf
@printf("DG %s, %s; RK4 %d steps (%.0f s), max|θ - mean| %.2f K\n", TERRAIN, grid_info(base), nref, tr, maximum(abs, yr))
for meth in METHODS, Δt in DTS, fast in FASTS, cfl in CFLS
    fast === :verlet && cfl > 0.63 && continue
    n = round(Int, T / Δt)
    nsub = ceil(Int, Δt / dτ(cfl))
    q = similar(q0)
    t = @elapsed try
        q = TS.run_mis(base, q0, Δt, n; tab = TS.mis_tableau(meth), nsub, fast)
    catch e
        e isa DomainError || rethrow(); fill!(q, NaN)
    end
    e = err(q)
    @printf("%-6s Δt = %4.1f s %-8s CFL %.3f nsub %3d: %s (%.0f s)\n", meth, Δt, fast, cfl, nsub,
            isfinite(e) ? @sprintf("error θ %.2e K", e) : "failed", t)
end
