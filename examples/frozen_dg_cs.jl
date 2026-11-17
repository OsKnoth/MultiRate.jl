# DG (P = 4) with the pressure as an additional prognostic variable and a column-implicit vertical part:
# the small steps :cs / :csfb of docs/moist_verlet on the DG gravity wave (300 km × 10 km, 30 × 4 elements,
# node spacing ≈ 2.5 km × 625 m, U = 20 m/s, N = 0.01/s), flat or over an Agnesi hill (h0 = 400 m,
# a = 10 km; initial flow along the terrain-following coordinate surfaces). Error of θ (minus its mean) at
# T = 3000 s against RK4 of the base DG model; explicit Verlet of the base model for comparison.
#
#     [TERRAIN=flat|hill] [METHODS=MIS54,WSRK3,EB4] [FASTS=verlet,cs,csfb] [OFFCENTER=ε] [SMDIV=γ]
#     [DTS=100,50] [NSUBFAC=1,2] julia --project=examples examples/frozen_dg_cs.jl
#
# Small steps: Verlet at the vertical limit (`fast_step(:verlet)`, the vertical node spacing), :cs/:csfb at the
# horizontal limit 0.85·0.63 Δx_e/((2p+1) c) (Δx_e the element width), divided by NSUBFAC.

include(joinpath(@__DIR__, "euler2d_common.jl"))

L, H, a, U, T = 300e3, 10e3, 5e3, 20.0, 3000.0
const TERRAIN = get(ENV, "TERRAIN", "flat")
const METHODS = Symbol.(split(get(ENV, "METHODS", "MIS54,WSRK3,EB4"), ","))
const FASTS = Symbol.(split(get(ENV, "FASTS", "verlet,cs,csfb"), ","))
const OFFC = parse(Float64, get(ENV, "OFFCENTER", "0.1"))
const SMDIV = parse(Float64, get(ENV, "SMDIV", "0.1"))
const DTS = parse.(Float64, split(get(ENV, "DTS", "100,50"), ","))
const NSUBFAC = parse.(Float64, split(get(ENV, "NSUBFAC", "1"), ","))
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
hill = TERRAIN == "hill"
zsf(x) = hill ? 400.0 / (1 + ((x - L / 2) / 10e3)^2) : 0.0
dzs(x) = hill ? -400.0 * 2 * (x - L / 2) / 10e3^2 / (1 + ((x - L / 2) / 10e3)^2)^2 : 0.0
base = build_model(:dg; L, H, P = 4, NEX = 30, NEZ = 4, ρr, Θr, gas, zs = zsf)
m = E2.DGPressureModel(base)
qb = project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / H) / (1 + ((x - L / 3) / a)^2));
                                 (ρ, ρ * U, ρ * U * dzs(x) * (H - z) / (H - zsf(x)), Θr(z))))
q0 = E2.pressure_state(m, qb)
M = base.mesh; c = sound_speed_max(base, qb)
dτv = fast_step(base, qb, :verlet)
dτh = 0.85 * 0.63 * (M.Lx / M.Nex) / ((2M.p + 1) * c)
nref = ceil(Int, T / (0.5dτv))
tr = @elapsed qref = TS.run_rk4(base, qb, T / nref, nref)
θf(q) = (y = vec(E2.field(base, q, 4) ./ E2.field(base, q, 1)); y .- sum(y) / length(y))
yr = θf(qref)
err(q) = all(isfinite, q) ? maximum(abs.(θf(q[1:base.nq]) .- yr)) : Inf
@printf("DG %s, %s; Verlet step %.2f s (vertical limit), horizontal limit %.2f s; RK4 %d steps (%.0f s), max|θ - mean| %.2f K\n",
        TERRAIN, grid_info(base), dτv, dτh, nref, tr, maximum(abs, yr))
@printf("ε = %.2f, γ_d = %.2f for :cs/:csfb\n", OFFC, SMDIV)
for meth in METHODS, Δt in DTS, fast in FASTS, f in (fast === :verlet ? (1.0,) : NSUBFAC)
    n = round(Int, T / Δt)
    model, x0 = fast === :verlet ? (base, qb) : (m, q0)
    nsub = ceil(Int, f * Δt / (fast === :verlet ? dτv : dτh))
    ε, γ = fast === :verlet ? (0.0, 0.0) : (OFFC, SMDIV)
    q = similar(x0)
    t = @elapsed try
        q = TS.run_frozen(model, x0, Δt, n; tab = TS.mis_tableau(meth), nsub, fast, offcenter = ε, smdiv = γ)
    catch e
        e isa DomainError || rethrow(); fill!(q, NaN)
    end
    e = err(q)
    @printf("%-6s Δt = %5.1f s %-6s nsub %4d: %s (%.0f s)\n", meth, Δt, fast, nsub, isfinite(e) ? @sprintf("error θ %.2e K", e) : "failed", t)
end
