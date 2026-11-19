# DG (P = 4) with the pressure as an additional prognostic variable and a column-implicit vertical part:
# the small steps :cs / :csfb of docs/moist_verlet on the DG gravity wave (300 km × 10 km, 30 × 4 elements,
# node spacing ≈ 2.5 km × 625 m, U = 20 m/s, N = 0.01/s), flat or over an Agnesi hill (h0 = 400 m,
# a = 10 km; initial flow along the terrain-following coordinate surfaces). Error of θ (minus its mean) at
# T = 3000 s against RK4 of the base DG model; explicit Verlet of the base model for comparison.
#
#     [TERRAIN=flat|hill] [METHODS=MIS54,WSRK3,EB4] [FASTS=verlet,cs,csfb,csb] [OFFCENTER=ε] [SMDIV=γ]
#     [DTS=100,50] [NSUBFAC=1,2] [LMARS=velocity|momentum] [VSPLIT=full|contravariant] [VSOLVER=lu|hdg]
#     julia --project=examples examples/frozen_dg_cs.jl
#
# LMARS=velocity|momentum: the form of LMARS (default velocity = base model); the reference is RK4 of the model with
# the same form. VSPLIT=full|contravariant|contravariant_volume|contravariant_momentum (comma separated list): the vertical
# (implicit) part, all η-terms (full), only those without m_x (contravariant: the column contains ρ, Θ, p, m_z, as the FV
# trick), or with the contravariant momenta (Ũ, W̃) = (Jξ·m, Jη·m) in place of (m_x, m_z) in the fast solver
# (contravariant_momentum: the column contains ρ, Θ, p, W̃). VSOLVER=hdg (with LMARS=momentum): the column system by static
# condensation (interface momentum only in the solve), otherwise sparse LU. HPREDICT=0: D_M^x explicit (S2 of docs/fast_2d).
#
# Small steps: Verlet at the vertical limit (`fast_step(:verlet)`, the vertical node spacing), :cs/:csfb at the
# horizontal limit 0.85·0.63 Δx_e/((2p+1) c) (Δx_e the element width), divided by NSUBFAC. :csb is CS with
# Bokhove's partition (the ξ-face dissipation integrated exactly in the split flows, docs/frozen_dg
# Section sec:implicitD); it is stable up to about twice this step (NSUBFAC = 0.5).

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
const LMARS = Symbol(get(ENV, "LMARS", "velocity"))
const VSPLITS = Symbol.(split(get(ENV, "VSPLIT", "full"), ","))
const VSOLVER = Symbol(get(ENV, "VSOLVER", "lu"))
const HPREDICT = get(ENV, "HPREDICT", "1") == "1"
m = E2.DGPressureModel(base; lmars = LMARS)                    # reference model (RK4)
qb = project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / H) / (1 + ((x - L / 3) / a)^2));
                                 (ρ, ρ * U, ρ * U * dzs(x) * (H - z) / (H - zsf(x)), Θr(z))))
q0 = E2.pressure_state(m, qb)
M = base.mesh; c = sound_speed_max(base, qb)
dτv = fast_step(base, qb, :verlet)
dτh = 0.85 * 0.63 * (M.Lx / M.Nex) / ((2M.p + 1) * c)
nref = ceil(Int, T / (0.5dτv))
# reference: RK4 of the model with the same LMARS (velocity form: the base model)
tr = @elapsed qref = LMARS === :velocity ? TS.run_rk4(base, qb, T / nref, nref) : TS.run_rk4(m, q0, T / nref, nref)[1:base.nq]
θf(q) = (y = vec(E2.field(base, q, 4) ./ E2.field(base, q, 1)); y .- sum(y) / length(y))
yr = θf(qref)
err(q) = all(isfinite, q) ? maximum(abs.(θf(q[1:base.nq]) .- yr)) : Inf
@printf("DG %s, %s; Verlet step %.2f s (vertical limit), horizontal limit %.2f s; RK4 %d steps (%.0f s), max|θ - mean| %.2f K\n",
        TERRAIN, grid_info(base), dτv, dτh, nref, tr, maximum(abs, yr))
@printf("ε = %.2f, γ_d = %.2f for :cs/:csfb, LMARS in the %s form, column solve %s, midpoint predictor %s\n", OFFC, SMDIV, LMARS, VSOLVER, HPREDICT)
for meth in METHODS, Δt in DTS, fast in FASTS, vs in (fast === :verlet ? (:full,) : VSPLITS), f in (fast === :verlet ? (1.0,) : NSUBFAC)
    n = round(Int, T / Δt)
    model, x0 = fast === :verlet ? (base, qb) :
                (E2.DGPressureModel(base; lmars = LMARS, vsplit = vs, vsolver = VSOLVER, hpredict = HPREDICT), q0)
    nsub = ceil(Int, f * Δt / (fast === :verlet ? dτv : dτh))
    ε, γ = fast === :verlet ? (0.0, 0.0) : (OFFC, SMDIV)
    q = similar(x0)
    t = @elapsed try
        q = TS.run_frozen(model, x0, Δt, n; tab = TS.mis_tableau(meth), nsub, fast, offcenter = ε, smdiv = γ)
    catch e
        e isa DomainError || rethrow(); fill!(q, NaN)
    end
    e = err(q)
    @printf("%-6s Δt = %5.1f s %-6s %-22s nsub %4d: %s (%.0f s)\n", meth, Δt, fast, fast === :verlet ? "" : vs, nsub,
            isfinite(e) ? @sprintf("error θ %.2e K", e) : "failed", t)
end
