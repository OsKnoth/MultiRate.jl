# Moist bubble (Bryan & Fritsch 2002) with the DGSEM: frozen MIS with the pressure as an additional prognostic variable,
# moist air with ρθ_ρ, ρq_v, ρq_c (`E2.DGPressureModel(base; moist = true)`): no saturation adjustment, condensation as the
# explicit Fischer–Burmeister relaxation (τ on the fast scale) with the latent heating in the ρθ_ρ and the p equation;
# p recomputed from the equation of state after every MIS step. Setup as the FV example frozen_moist_bubble.jl:
# saturated neutral atmosphere θ_e = 320 K, r_t = 0.02, 20 km × 10 km, θ_ρ perturbed by 2 K cos²(π r/(2 r_c)), r_c = 2 km,
# at (10 km, 2 km) at fixed pressure and r_t. Reference: RK4 of the same operator.
#
#     [P=4] [NEX=20] [NEZ=10] [FAST=verlet|cs] [TAUFAC=0.5] [DTS=1,2] [VSOLVER=lu|hdg]
#     julia --project=examples examples/dg_moist_bubble.jl [T]
#
# FAST: the small step, explicit Störmer–Verlet S1 (default; the grid is isotropic) or S2 (:cs, z implicit, at the
# horizontal limit). TAUFAC: τ = TAUFAC × 0.85·0.63 Δx_e/((2p+1) c_s). The RK4 reference is cached in examples/dg_moist_bubble_ref_*.bin.
include(joinpath(@__DIR__, "euler2d_common.jl"))
using Serialization

T = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1000.0
P = parse(Int, get(ENV, "P", "4")); NEX = parse(Int, get(ENV, "NEX", "20")); NEZ = parse(Int, get(ENV, "NEZ", "10"))
const FAST = Symbol(get(ENV, "FAST", "verlet"))
const TAUFAC = parse(Float64, get(ENV, "TAUFAC", "0.5"))
const DTS = parse.(Float64, split(get(ENV, "DTS", "1,2"), ","))
const VSOLVER = Symbol(get(ENV, "VSOLVER", "lu"))
LX, H = 20e3, 10e3
mg = FV.MoistGas()
gas = Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0, g = 9.81)
ρr, Θr, _ = stratified_reference(gas; θ0 = 300.0, N = 0.0)          # output reference only
base = build_model(:dg; L = LX, H, P, NEX, NEZ, ρr, Θr, gas)
M = base.mesh; np = M.np; N = np^2 * M.Ne
m = E2.DGPressureModel(base; lmars = :momentum, moist = true, mgas = mg, τfac = TAUFAC, vsolver = VSOLVER)
const θe0, rt0 = 320.0, 0.02

"""Saturated state with θ_e = 320 K, r_t = 0.02 at pressure p: (ρ, Θ = ρθ_ρ, Q = ρq_t)."""
function sat_state(p)
    c = mg.Cpd + rt0 * mg.Cpl
    function θe(T)
        e = FV.psat(mg, T); pd = p - e; rv = mg.Rd / mg.Rv * e / pd
        T * (pd / mg.p0)^(-mg.Rd / c) * exp(FV.latent(mg, T) * rv / (c * T))
    end
    a, b = 200.0, 330.0
    for _ in 1:200
        T = (a + b) / 2
        θe(T) > θe0 ? (b = T) : (a = T)
    end
    T = (a + b) / 2
    e = FV.psat(mg, T); pd = p - e
    ρd = pd / (mg.Rd * T); ρv = e / (mg.Rv * T); Q = ρd * rt0; ρ = ρd + Q
    Rm = mg.Rd * ρd + mg.Rv * ρv; κ = Rm / (mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * (Q - ρv))
    return ρ, p^(1 - κ) * mg.p0^κ / mg.Rd, Q
end

"""Perturbation of θ_ρ at fixed pressure and r_t, saturated (CGDycore `PerturbMoistProfile`)."""
function perturb(x, z, ρ, Θ, Q, p)
    xc, zc, rc, Δθ = LX / 2, 2e3, 2e3, 2.0
    r = sqrt((x - xc)^2 + (z - zc)^2)
    r < rc || return ρ, Θ, Q
    κd = mg.Rd / mg.Cpd
    T, ρv, ρc, _ = FV.moist_state(mg, ρ, Θ, Q)
    ρd = ρ - Q
    κM = (mg.Rd * ρd + mg.Rv * ρv) / (mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * ρc)
    θdens = Θ / ρ * (p / mg.p0)^(κM - κd)
    θdens_new = θdens * (1 + Δθ * cospi(0.5 * r / rc)^2 / 300)
    rt = Q / ρd; rv = ρv / ρd
    θl = θdens_new * (1 + rt) / (1 + mg.Rv / mg.Rd * rv)
    ρd_new, rvs = ρd, rv
    for _ in 1:200
        Tl = θl * (p / mg.p0)^κd
        pvs = FV.psat(mg, Tl)
        ρd_new = (p - pvs) / (mg.Rd * Tl)
        rvs = pvs / (mg.Rv * ρd_new * Tl)
        θn = θdens_new * (1 + rt) / (1 + mg.Rv / mg.Rd * rvs)
        abs(θn - θl) <= θl * 1e-12 && (θl = θn; break)
        θl = θn
    end
    ρv = rvs * ρd_new; ρc = (rt - rvs) * ρd_new
    ρ = ρd_new * (1 + rt)
    κM = (mg.Rd * ρd_new + mg.Rv * ρv) / (mg.Cpd * ρd_new + mg.Cpv * ρv + mg.Cpl * ρc)
    return ρ, ρ * θdens_new * (p / mg.p0)^(κd - κM), ρd_new * rt
end

# hydrostatic pressure p(z) of the saturated column (RK4 in steps of 2 m), linear interpolation to the nodes
const DZP = 2.0
pz = let p = mg.p0, ps = [p]
    for _ in 1:round(Int, H / DZP)
        f(p) = -gas.g * sat_state(p)[1]
        k1 = f(p); k2 = f(p + DZP / 2 * k1); k3 = f(p + DZP / 2 * k2); k4 = f(p + DZP * k3)
        p += DZP * (k1 + 2k2 + 2k3 + k4) / 6; push!(ps, p)
    end
    ps
end
pofz(z) = (s = clamp(z / DZP, 0, length(pz) - 1 - 1e-12); k = floor(Int, s); pz[k+1] + (s - k) * (pz[k+2] - pz[k+1]))

function initial(perturbed)
    q = zeros(base.nq); Qn = Dict{Tuple{Float64,Float64},Float64}()
    for e in 1:M.Ne, j in 1:np, i in 1:np
        x, z = M.X[i, j, e], M.Z[i, j, e]; p = pofz(z)
        ρ, Θ, Q = sat_state(p)
        perturbed && ((ρ, Θ, Q) = perturb(x, z, ρ, Θ, Q, p))
        q[E2.node_iq(np, e, 1, i, j)] = ρ; q[E2.node_iq(np, e, 4, i, j)] = Θ; Qn[(x, z)] = Q
    end
    return E2.moist_dg_state(m, q, (x, z) -> Qn[(x, z)])
end
rest = initial(false); x0 = initial(true)
Nr = TS.rhs_nonlinear(m, rest)
@printf("DG moist bubble, %s, τ = %.3f s, fast %s, column solve %s; rest-state residual max|N_mz| = %.2e kg/(m² s²)\n",
        grid_info(base), m.τ, FAST, VSOLVER, maximum(abs, Nr[E2.node_iq(np, e, 3, i, j)] for e in 1:M.Ne, j in 1:np, i in 1:np))

pn(e, i, j) = ((e - 1) * np + (j - 1)) * np + i
θe_field(x) = [FV.theta_e(mg, x[E2.node_iq(np, e, 1, i, j)], x[E2.node_iq(np, e, 4, i, j)], x[base.nq + N + pn(e, i, j)],
                          x[base.nq + 2N + pn(e, i, j)]) - θe0 for i in 1:np, j in 1:np, e in 1:M.Ne]
w_field(x) = [x[E2.node_iq(np, e, 3, i, j)] / x[E2.node_iq(np, e, 1, i, j)] for i in 1:np, j in 1:np, e in 1:M.Ne]
qc_field(x) = [x[base.nq + 2N + pn(e, i, j)] / x[E2.node_iq(np, e, 1, i, j)] for i in 1:np, j in 1:np, e in 1:M.Ne]
wts = [M.J[i, j, e] * M.wg[i] * M.wg[j] for i in 1:np, j in 1:np, e in 1:M.Ne]
tot(x, r) = r == :water ? sum(vec(wts) .* (x[base.nq+N+1:base.nq+2N] .+ x[base.nq+2N+1:base.nq+3N])) :
            sum(wts[i, j, e] * x[E2.node_iq(np, e, r, i, j)] for i in 1:np, j in 1:np, e in 1:M.Ne)

c = 1.05 * sound_speed_max(base, x0[1:base.nq])
dτv = 0.85 * 0.52 * min(M.Lx / M.Nex, M.H / M.Nez) / ((2M.p + 1) * c)     # explicit Verlet limit (P = 4 table)
dτh = 0.85 * 0.63 * (M.Lx / M.Nex) / ((2M.p + 1) * c)                     # :cs, horizontal limit
dτ = FAST === :cs ? dτh : dτv
nref = ceil(Int, T / min(0.5dτv, 0.5m.τ))
reffile = joinpath(@__DIR__, @sprintf("dg_moist_bubble_ref_P%d_%dx%d_T%g_tf%g.bin", P, NEX, NEZ, T, TAUFAC))   # RK4 cached
tr = @elapsed xr = isfile(reffile) ? deserialize(reffile) : (y = TS.run_rk4(m, x0, T / nref, nref); serialize(reffile, y); y)
θer = θe_field(xr)
@printf("RK4 reference (%d steps, %.0f s): θ_e′ ∈ [%.3f, %.3f] K, max|w| = %.2f m/s, max q_c = %.2e\n",
        nref, tr, extrema(θer)..., maximum(abs, w_field(xr)), maximum(qc_field(xr)))
res = Pair{String,Vector{Float64}}[]
for Δt in DTS
    n = round(Int, T / Δt); nsub = ceil(Int, Δt / dτ)
    x = similar(x0)
    t = @elapsed try
        x = TS.run_frozen(m, copy(x0), Δt, n; tab = TS.mis_tableau(:MIS54), nsub, fast = FAST,
                          offcenter = FAST === :cs ? 0.1 : 0.0, smdiv = 0.0)
    catch e
        e isa DomainError || rethrow(); fill!(x, NaN)
    end
    if !all(isfinite, x)
        @printf("MIS54 Δt = %4.1f s nsub %3d: failed\n", Δt, nsub); continue
    end
    θe = θe_field(x)
    @printf("MIS54 Δt = %4.1f s nsub %3d (small step %.3f s): θ_e′ ∈ [%.3f, %.3f] K, max|w| %.2f, error θ_e′ %.2e K, error w %.2e m/s, Δmass %.1e, Δ(ρq_t) %.1e (%.0f s)\n",
            Δt, nsub, Δt / nsub, extrema(θe)..., maximum(abs, w_field(x)), maximum(abs, θe - θer), maximum(abs, w_field(x) - w_field(xr)),
            tot(x, 1) / tot(x0, 1) - 1, tot(x, :water) / tot(x0, :water) - 1, t)
    push!(res, @sprintf("MIS54 Δt = %g s, %s", Δt, FAST) => x)
end

# θ_e′, w, q_c′ at T (DG nodes on element faces are plotted twice); q_c′: minus the mean over the nodes of the same height
zkey = round.(vec(M.Z); digits = 3)
function qcp(x)
    q = 1e3 .* vec(qc_field(x)); mean = Dict{Float64,Tuple{Float64,Int}}()
    for (k, v) in zip(zkey, q); s, n = get(mean, k, (0.0, 0)); mean[k] = (s + v, n + 1); end
    return [v - mean[k][1] / mean[k][2] for (k, v) in zip(zkey, q)]
end
qcmax = max(1e-3, maximum(abs, qcp(xr)))
rows = vcat(res[1:min(1, end)], ["RK4" => xr])
fig = Figure(size = (1500, 420 * length(rows)))
X, Z = vec(M.X) ./ 1e3, vec(M.Z) ./ 1e3
for (r, (lab, x)) in enumerate(rows)
    for (col, (name, f, cr, cm)) in enumerate((("θ_e′ [K]", θe_field(x), (-0.5, 2.5), :thermal), ("w [m/s]", w_field(x), (-12, 12), :balance),
                                               ("q_c′ [g/kg] (minus horizontal mean)", qcp(x), (-1, 1) .* qcmax, :balance)))
        ax = Axis(fig[r, 2col-1]; title = "$lab: $name", xlabel = "x [km]", ylabel = "z [km]", aspect = DataAspect(), limits = (4, 16, 0, 10))
        hm = tricontourf!(ax, X, Z, vec(f); levels = range(cr..., length = 21), colormap = cm, extendlow = :auto, extendhigh = :auto)
        Colorbar(fig[r, 2col], hm)
    end
end
fn = joinpath(@__DIR__, "dg_moist_bubble_$(FAST).png")
save(fn, fig)
println("wrote ", fn)
