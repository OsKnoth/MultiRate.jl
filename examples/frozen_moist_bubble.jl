# Moist bubble (Bryan & Fritsch 2002), flat FV grid: frozen MIS with the Exner pressure as an
# additional prognostic variable and ρθ_ρ, ρq_v, ρq_c as the thermodynamic variables (`FV.MoistExnerModel`:
# no saturation adjustment, condensation as an explicit Fischer–Burmeister relaxation with τ on the fast scale
# and the latent heating Q_θ in the ρθ_ρ equation; π recomputed after every MIS step).
#
#     [PVAR=pi|p] [MICRO=fast|slow] [TAUFAC=f] [WTAU=w] julia --project=examples examples/frozen_moist_bubble.jl [T] [nx]
#
# Saturated, neutral atmosphere θ_e = 320 K, r_t = 0.02, 20 km × 10 km, nx × nx/2 cells (default
# 100 × 50, Δx = 200 m). The background is in discrete hydrostatic balance with the pressure form
# of the model (no background state in the equations); the perturbation of θ_ρ by
# 2 K cos²(π r/(2 r_c)), r_c = 2 km, at (10 km, 2 km) at fixed pressure and r_t, saturated (as
# CGDycore `PerturbMoistProfile`). MIS54, HEVI; reference RK4 of the same operator. T = 1000 s.
# PVAR = p: the pressure itself as the additional variable (∂p/∂(ρ, Θ, V, C), plain δp).
# MICRO = fast (default): phase change in every small step, τ = half the acoustic step; slow: at the
# slow stages (then set WTAU, τ = min(Δx, Δz)/WTAU ≳ Δt/2).
# Output: table, frozen_moist_bubble_<PVAR>_<MICRO>.png (θ_e′, w, q_c at T).

include(joinpath(@__DIR__, "euler2d_common.jl"))
using LinearAlgebra: norm

T = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1000.0
nx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100
nz = nx ÷ 2
LX, H = 20e3, 10e3
mg = FV.MoistGas()
gas = Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0, g = 9.81)
grid = FV.FVGrid(nx, nz, LX, H)
FV.FV_PRESSURE_FORM[] = :gradp
ρr, Θr, _ = stratified_reference(gas; θ0 = 300.0, N = 0.0)        # output reference only
base = FV.FVModel(grid, gas, ρr, Θr)
const PVAR = Symbol(get(ENV, "PVAR", "pi"))
const MICRO = Symbol(get(ENV, "MICRO", "fast"))
m = FV.MoistExnerModel(base; mgas = mg, pvar = PVAR, micro = MICRO, τfac = parse(Float64, get(ENV, "TAUFAC", "0.5")),
                        wτ = haskey(ENV, "WTAU") ? parse(Float64, ENV["WTAU"]) : nothing)
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
πof(p) = (p / mg.p0)^(mg.Rd / mg.Cpd)

"""Discretely balanced saturated column: p_1 from dp/dz = -g ρ (RK2 from p = p0 at z = 0), then
c_pd Θ̂_f (π_k - π_{k-1}) + g Δz ρ̄ = 0 upwards (secant in p_k)."""
function balanced_column()
    grav = gas.g
    p = mg.p0; z = 0.0; nsteps = 50; h = grid.zc[1] / nsteps
    for _ in 1:nsteps
        k1 = -grav * sat_state(p)[1]; k2 = -grav * sat_state(p + h * k1)[1]
        p += h * (k1 + k2) / 2
    end
    ps = zeros(nz); ps[1] = p
    for k in 2:nz
        pl = ps[k-1]; ρl = sat_state(pl)[1]; Θhl = pl / (mg.Rd * πof(pl))
        R(pk) = (PVAR === :p ? pk - pl : mg.Cpd * 0.5 * (Θhl + pk / (mg.Rd * πof(pk))) * (πof(pk) - πof(pl))) +
                grav * grid.dz * 0.5 * (ρl + sat_state(pk)[1])
        x0 = pl - grav * ρl * grid.dz; x1 = x0 * (1 - 1e-4); f0, f1 = R(x0), R(x1)
        for _ in 1:50
            (f1 == 0 || f1 == f0) && break
            x0, x1, f0 = x1, x1 - f1 * (x1 - x0) / (f1 - f0), f1
            abs(x1 - x0) <= 1e-13 * abs(x1) && break
            f1 = R(x1)
        end
        ps[k] = x1
    end
    return ps
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

ps = balanced_column()
cols = [sat_state(ps[k]) for k in 1:nz]
qb = zeros(base.nq); Qb = zeros(nx, nz); q0 = zeros(base.nq); Q0 = zeros(nx, nz)
for k in 1:nz, i in 1:nx
    ρ, Θ, Q = cols[k]
    qb[FV.iρ(grid, i, k)] = ρ; qb[FV.iΘ(grid, i, k)] = Θ; Qb[i, k] = Q
    ρp, Θp, Qp = perturb(grid.xc[i], grid.zc[k], ρ, Θ, Q, ps[k])
    q0[FV.iρ(grid, i, k)] = ρp; q0[FV.iΘ(grid, i, k)] = Θp; Q0[i, k] = Qp
end
rest = FV.moist_exner_state(m, qb, Qb)
x0 = FV.moist_exner_state(m, q0, Q0)
Nr = TS.rhs_nonlinear(m, rest)
@printf("moist bubble: %d × %d cells (Δx = %.0f m); rest-state residual max|N_W| = %.2e; surface p = %.1f Pa, ρ_c at z_1 = %.2e kg/m³\n",
        nx, nz, grid.dx, maximum(abs, Nr[FV.iW(grid, i, k)] for i in 1:nx, k in 2:nz), ps[1],
        FV.moist_state(mg, cols[1]...)[3])

θe_field(x) = [FV.theta_e(mg, x[FV.iρ(grid, i, k)], x[FV.iΘ(grid, i, k)], x[FV.iV(m, i, k)], x[FV.iC(m, i, k)]) - θe0 for i in 1:nx, k in 1:nz]
qc_field(x) = [x[FV.iC(m, i, k)] / x[FV.iρ(grid, i, k)] for i in 1:nx, k in 1:nz]
fb_field(x) = [abs(FV.fischer_burmeister(mg, x[FV.iρ(grid, i, k)], x[FV.iΘ(grid, i, k)], x[FV.iV(m, i, k)], x[FV.iC(m, i, k)])[1]) / x[FV.iρ(grid, i, k)] for i in 1:nx, k in 1:nz]
w_field(x) = [x[FV.iW(grid, i, k)] / (0.5 * (x[FV.iρ(grid, i, k - 1)] + x[FV.iρ(grid, i, k)])) for i in 1:nx, k in 2:nz]
tot(x, f) = sum(x[f(i, k)] for i in 1:nx, k in 1:nz)

c = 1.05 * FV.sound_speed_max(base, q0)
nref = ceil(Int, T / min(0.4 * grid.dx / c, 0.5 * m.τ))           # RK4: also resolve τ
tr = @elapsed xr = TS.run_rk4(m, x0, T / nref, nref)
θer = θe_field(xr)
@printf("RK4 reference (%d steps, %.0f s): θ_e′ ∈ [%.3f, %.3f] K, max|w| = %.2f m/s, max q_c = %.2e (τ = %.1f s, micro = %s)\n",
        nref, tr, extrema(θer)..., maximum(abs, w_field(xr)), maximum(qc_field(xr)), m.τ, MICRO)

tab = TS.mis_tableau(:MIS54)
dτ = 0.85 * 0.92 * grid.dx / c
res5 = nothing
for Δt in (2.0, 5.0, 10.0)
    n = round(Int, T / Δt); nsub = ceil(Int, Δt / dτ)
    x = copy(x0)
    w = TS.FrozenWork(m, length(x), tab); tmp = (zeros(length(x)), zeros(length(x)), zeros(length(x)))
    drift = 0.0
    t = @elapsed try
        for _ in 1:n
            TS.frozen_mis_step!(x, m, x, Δt, tab, w; nsub, fast = :hevi, tmp)
            drift = max(drift, maximum(abs(x[FV.iπ(m, i, k)] / FV.moist_pvar(m, x[FV.iρ(grid, i, k)], x[FV.iΘ(grid, i, k)], x[FV.iV(m, i, k)], x[FV.iC(m, i, k)]) - 1)
                                       for i in 1:nx, k in 1:nz))
            TS.frozen_step_end!(m, x)
        end
    catch e
        x .= NaN
    end
    if !all(isfinite, x)
        @printf("Δt = %4.1f s nsub %2d: failed\n", Δt, nsub); continue
    end
    θe = θe_field(x)
    @printf("Δt = %4.1f s nsub %2d: θ_e′ ∈ [%.3f, %.3f] K, max|w| %.2f, error θ_e′ max %.2e K, Δmass %.1e, ΔΘ %.1e, Δ(ρq_t) %.1e, max rel. |π - π_eos| %.1e, max |φ_FB|/ρ %.1e, %.0f s\n",
            Δt, nsub, extrema(θe)..., maximum(abs, w_field(x)), maximum(abs, θe - θer),
            tot(x, (i, k) -> FV.iρ(grid, i, k)) / tot(x0, (i, k) -> FV.iρ(grid, i, k)) - 1,
            tot(x, (i, k) -> FV.iΘ(grid, i, k)) / tot(x0, (i, k) -> FV.iΘ(grid, i, k)) - 1,
            (tot(x, (i, k) -> FV.iV(m, i, k)) + tot(x, (i, k) -> FV.iC(m, i, k))) / (tot(x0, (i, k) -> FV.iV(m, i, k)) + tot(x0, (i, k) -> FV.iC(m, i, k))) - 1, drift, maximum(fb_field(x)), t)
    Δt == 5.0 && (global res5 = x)
    flush(stdout)
end

if res5 !== nothing
    fig = Figure(size = (1500, 800))
    xk, zk = grid.xc ./ 1e3, grid.zc ./ 1e3
    for (r, (lab, x)) in enumerate((("frozen MIS54, Δt = 5 s", res5), ("RK4 reference", xr)))
        for (col, (name, f, cr, cm)) in enumerate((("θ_e′ [K]", θe_field(x), (-0.5, 2.5), :thermal),
                                                   ("w [m/s]", w_field(x), (-12, 12), :balance),
                                                   ("q_c [kg/kg]", qc_field(x), (0, 0.02), :blues)))
            ax = Axis(fig[r, 2col-1]; title = "$lab: $name", xlabel = "x [km]", ylabel = "z [km]", aspect = DataAspect(), limits = (4, 16, 0, 10))
            hm = heatmap!(ax, xk, col == 2 ? grid.zf[2:nz] ./ 1e3 : zk, f; colormap = cm, colorrange = cr)
            Colorbar(fig[r, 2col], hm)
        end
    end
    save("frozen_moist_bubble_$(PVAR)_$(MICRO).png", fig)
    println("wrote frozen_moist_bubble_$(PVAR)_$(MICRO).png")
end
