# Moist bubble in 3D (Bryan & Fritsch 2002, spherical perturbation), flat FV3D grid: frozen MIS with the moist 3D
# model `FV3D.MoistModel3D` (the 2D model of examples/frozen_moist_bubble.jl in 3D: ρθ_ρ, ρq_v, ρq_c and the Exner
# pressure as prognostic variables, no saturation adjustment, condensation as an explicit Fischer–Burmeister
# relaxation with τ on the fast scale, π recomputed after every MIS step).
#
#     [PVAR=p|pi] [THVAR=thetarho|energy|entropy] [FAST=verlet|hevi|cs|ei] [MICRO=fast|slow] [TAUFAC=f] [WTAU=w]
#         [METHODS=MIS54,WSRK3] [DTS=5] [DTPLOT=5] [REF=1] julia -t auto --project=examples examples/fv3d_moist_bubble.jl [T] [nx]
#
# Saturated, neutral atmosphere θ_e = 320 K, r_t = 0.02, 20 km × 20 km × 10 km, nx × nx × nx/2 cells (default
# nx = 40, Δx = 500 m), in discrete hydrostatic balance with the pressure form of the model; perturbation of θ_ρ by
# 2 K cos²(π r/(2 r_c)), r_c = 2 km, at (10 km, 10 km, 2 km) at fixed pressure and r_t, saturated (as in 2D).
# FAST: small-step integrator (explicit Störmer–Verlet by default: the grid is isotropic). TAUFAC: τ = TAUFAC Δτ_ac,
# default 0.5 (Verlet, :cs, :ei: one phase-change evaluation with weight c needs c/τ ≲ 1) and 0.25 for :hevi. REF = 1: RK4 reference
# of the same operator (expensive: the step also resolves τ). Output: table, fv3d_moist_bubble[_<suffix>].png
# (θ_e′, w, q_c in the x–z plane through the centre and θ_e′ in the x–y plane at the height of max w, at T).

using MultiRate, Printf, CairoMakie
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const CM = MultiRate.Common

T = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1000.0
nx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 40
nz = nx ÷ 2
L, H = 20e3, 10e3
mg = CM.MoistGas()
gas = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0, g = 9.81)
grid = F3.Grid3D(nx, nx, nz, L, L, H)
ρr, Θr, _ = CM.stratified_reference(gas; θ0 = 300.0, N = 0.0)        # output reference only
base = F3.Model3D(grid, gas, ρr, Θr)
const PVAR = Symbol(get(ENV, "PVAR", "p"))
const MICRO = Symbol(get(ENV, "MICRO", "fast"))
const THVAR = Symbol(get(ENV, "THVAR", "thetarho"))
const FAST = Symbol(get(ENV, "FAST", "verlet"))
mθ = F3.MoistModel3D(base; mgas = mg, pvar = PVAR, micro = MICRO)
m = F3.MoistModel3D(base; mgas = CM.MoistGas(thvar = THVAR), pvar = PVAR, micro = MICRO,
                    τfac = parse(Float64, get(ENV, "TAUFAC", FAST === :hevi ? "0.25" : "0.5")),
                    wτ = haskey(ENV, "WTAU") ? parse(Float64, ENV["WTAU"]) : nothing)
const θe0, rt0 = 320.0, 0.02
πof(p) = (p / mg.p0)^(mg.Rd / mg.Cpd)

"""Saturated state with θ_e = 320 K, r_t = 0.02 at pressure p: (ρ, Θ = ρθ_ρ, Q = ρq_t)."""
function sat_state(p)
    c = mg.Cpd + rt0 * mg.Cpl
    function θe(T)
        e = CM.psat(mg, T); pd = p - e; rv = mg.Rd / mg.Rv * e / pd
        T * (pd / mg.p0)^(-mg.Rd / c) * exp(CM.latent(mg, T) * rv / (c * T))
    end
    a, b = 200.0, 330.0
    for _ in 1:200
        Tm = (a + b) / 2
        θe(Tm) > θe0 ? (b = Tm) : (a = Tm)
    end
    Tm = (a + b) / 2
    e = CM.psat(mg, Tm); pd = p - e
    ρd = pd / (mg.Rd * Tm); ρv = e / (mg.Rv * Tm); Q = ρd * rt0; ρ = ρd + Q
    Rm = mg.Rd * ρd + mg.Rv * ρv; κ = Rm / (mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * (Q - ρv))
    return ρ, p^(1 - κ) * mg.p0^κ / mg.Rd, Q
end

"""Discretely balanced saturated column: p_1 from dp/dz = -g ρ, then c_pd Θ̂_f (π_k - π_{k-1}) + g Δz ρ̄ = 0
(pvar = :pi) or p_k - p_{k-1} + g Δz ρ̄ = 0 (pvar = :p) upwards (secant in p_k)."""
function balanced_column()
    grav = gas.g
    p = mg.p0; nsteps = 50; h = grid.zc[1] / nsteps
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

"""Perturbation of θ_ρ at fixed pressure and r_t, saturated (spherical, r_c = 2 km at (L/2, L/2, 2 km))."""
function perturb(x, y, z, ρ, Θ, Q, p)
    xc, yc, zc, rc, Δθ = L / 2, L / 2, 2e3, 2e3, 2.0
    r = sqrt((x - xc)^2 + (y - yc)^2 + (z - zc)^2)
    r < rc || return ρ, Θ, Q
    κd = mg.Rd / mg.Cpd
    _, ρv, ρc, _ = CM.moist_state(mg, ρ, Θ, Q)
    ρd = ρ - Q
    κM = (mg.Rd * ρd + mg.Rv * ρv) / (mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * ρc)
    θdens_new = Θ / ρ * (p / mg.p0)^(κM - κd) * (1 + Δθ * cospi(0.5 * r / rc)^2 / 300)
    rt = Q / ρd; rv = ρv / ρd
    θl = θdens_new * (1 + rt) / (1 + mg.Rv / mg.Rd * rv)
    ρd_new, rvs = ρd, rv
    for _ in 1:200
        Tl = θl * (p / mg.p0)^κd
        pvs = CM.psat(mg, Tl)
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
qb = zeros(base.nq); Qb = zeros(nx, nx, nz); q0 = zeros(base.nq); Q0 = zeros(nx, nx, nz)
for k in 1:nz, j in 1:nx, i in 1:nx
    ρ, Θ, Q = cols[k]
    qb[F3.iρ(grid, i, j, k)] = ρ; qb[F3.iΘ(grid, i, j, k)] = Θ; Qb[i, j, k] = Q
    ρp, Θp, Qp = perturb(grid.xc[i], grid.yc[j], grid.zc[k], ρ, Θ, Q, ps[k])
    q0[F3.iρ(grid, i, j, k)] = ρp; q0[F3.iΘ(grid, i, j, k)] = Θp; Q0[i, j, k] = Qp
end
rest = F3.thermo_convert(mθ, m, F3.moist_exner_state(mθ, qb, Qb))
x0 = F3.thermo_convert(mθ, m, F3.moist_exner_state(mθ, q0, Q0))
Nr = TS.rhs_nonlinear(m, rest)
@printf("3D moist bubble: %d × %d × %d cells (Δx = %.0f m); rest-state residual max|N_W| = %.2e; surface p = %.1f Pa; τ = %.2f s\n",
        nx, nx, nz, grid.dx, maximum(abs, Nr[F3.iW(grid, i, j, k)] for i in 1:nx, j in 1:nx, k in 2:nz), ps[1], m.τ)

θe_field(x) = F3.theta_e3d(m, x) .- θe0
w_field(x) = F3.vertical_velocity(m, x)
tot(x, v) = F3.total3d(m, x, v)

c = 1.05 * maximum(sqrt(mg.Cpd / (mg.Cpd - mg.Rd) * CM.moist_state(mg, cols[k]...)[4] / cols[k][1]) for k in 1:nz)
# small step: 0.9 of the 3D acoustic limit of the explicit forward–backward step, c Δτ √(1/Δx² + 1/Δy² + 1/Δz²) ≤ 1
dτ = 0.9 / (c * sqrt(1 / grid.dx^2 + 1 / grid.dy^2 + 1 / grid.dz^2))
xr = nothing
if get(ENV, "REF", "0") == "1"
    nref = ceil(Int, T / min(0.4 * grid.dx / c, 0.5 * m.τ))
    tr = @elapsed global xr = TS.run_rk4(m, x0, T / nref, nref)
    @printf("RK4 reference (%d steps, %.0f s): θ_e′ ∈ [%.3f, %.3f] K, max|w| = %.2f m/s\n", nref, tr, extrema(θe_field(xr))..., maximum(abs, w_field(xr)))
end

const METHODS = Symbol.(split(get(ENV, "METHODS", "MIS54"), ","))
const DTS = parse.(Float64, split(get(ENV, "DTS", "5"), ","))
const DTPLOT = parse(Float64, get(ENV, "DTPLOT", "5"))
@printf("thermodynamic variable %s, pressure variable %s, fast integrator %s, micro %s\n", THVAR, PVAR, FAST, MICRO)
res = Pair{String,Vector{Float64}}[]
for meth in METHODS, Δt in DTS
    tab = TS.mis_tableau(meth)
    n = round(Int, T / Δt); nsub = ceil(Int, Δt / dτ)
    x = copy(x0)
    w = TS.FrozenWork(m, length(x), tab); tmp = ntuple(_ -> zeros(length(x)), 6)
    t = @elapsed try
        for _ in 1:n
            TS.frozen_mis_step!(x, m, x, Δt, tab, w; nsub, fast = FAST, tmp)
            TS.frozen_step_end!(m, x)
        end
    catch e
        e isa InterruptException && rethrow()
        x .= NaN
    end
    if !all(isfinite, x)
        @printf("%-6s Δt = %4.1f s nsub %2d: failed\n", meth, Δt, nsub); continue
    end
    θe = θe_field(x)
    @printf("%-6s Δt = %4.1f s nsub %2d: θ_e′ ∈ [%.3f, %.3f] K, max|w| %.2f m/s, max q_c %.2e%s, Δmass %.1e, Δ(ρq_t) %.1e, %.0f s\n",
            meth, Δt, nsub, extrema(θe)..., maximum(abs, w_field(x)), maximum(F3.qc3d(m, x)),
            xr === nothing ? "" : @sprintf(", error θ_e′ max %.2e K", maximum(abs, θe - θe_field(xr))),
            tot(x, :ρ) / tot(x0, :ρ) - 1, (tot(x, :qv) + tot(x, :qc)) / (tot(x0, :qv) + tot(x0, :qc)) - 1, t)
    Δt == DTPLOT && push!(res, "$meth, Δt = $(round(Int, Δt)) s" => x)
    flush(stdout)
end

if !isempty(res)
    rows = xr === nothing ? res : vcat(res, ["RK4 reference" => xr])
    jc = nx ÷ 2 + 1                                       # x–z plane next to the centre y = L/2
    fig = Figure(size = (1800, 380 * length(rows)))
    xk, zk = grid.xc ./ 1e3, grid.zc ./ 1e3
    for (r, (lab, x)) in enumerate(rows)
        θe = θe_field(x); wf = w_field(x)
        qc = 1e3 .* F3.qc3d(m, x); qc .-= sum(qc; dims = (1, 2)) ./ nx^2         # minus the horizontal mean
        kw = argmax(abs.(wf))[3] + 1                      # level of max |w| (centres next to the face)
        panels = (("θ_e′ [K], y = $(round(grid.yc[jc] / 1e3, digits = 2)) km", xk, zk, θe[:, jc, :], (-0.5, 2.5), :thermal),
                  ("w [m/s]", xk, grid.zf[2:nz] ./ 1e3, wf[:, jc, :], (-12, 12), :balance),
                  ("q_c′ [g/kg] (minus horizontal mean)", xk, zk, qc[:, jc, :], (-1, 1) .* max(1e-3, maximum(abs, qc)), :balance),
                  ("θ_e′ [K], z = $(round(grid.zc[kw] / 1e3, digits = 2)) km", xk, grid.yc ./ 1e3, θe[:, :, kw], (-0.5, 2.5), :thermal))
        for (col, (name, a, b, f, cr, cm)) in enumerate(panels)
            ax = Axis(fig[r, 2col-1]; title = "$lab: $name", xlabel = "x [km]", ylabel = col == 4 ? "y [km]" : "z [km]",
                      aspect = DataAspect(), limits = col == 4 ? (4, 16, 4, 16) : (4, 16, 0, 10))
            hm = heatmap!(ax, a, b, f; colormap = cm, colorrange = cr)
            Colorbar(fig[r, 2col], hm)
        end
    end
    sfx = join(filter(!isempty, [PVAR === :p ? "" : string(PVAR), THVAR === :thetarho ? "" : string(THVAR),
                                 FAST === :verlet ? "" : string(FAST), MICRO === :fast ? "" : string(MICRO)]), "_")
    fn = get(ENV, "OUT", "fv3d_moist_bubble$(isempty(sfx) ? "" : "_" * sfx).png")
    save(fn, fig)
    println("wrote ", fn)
end
