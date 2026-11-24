# Moist flow over a mountain with orographic precipitation, the test case of Jähn et al. (2015), ASAM v2.7, GMD 8,
# 317–340, Section 4.6 (after Kunz and Wassermann 2011), in the 3D FV model `FV3D.MoistModel3D` (terrain following,
# periodic): ρθ_ρ (or ρe, ρs), ρq_v, ρq_c, ρq_r and p as prognostic variables, condensation by the Fischer–Burmeister
# relaxation, warm-rain Kessler microphysics (DCMIP 2016, rain = true; the paper uses a two-moment warm scheme),
# sedimentation in the fast part, Smagorinsky–Lilly subgrid-scale model (as in the paper: C_s = 0.2, Pr = 1/3, Lilly's
# stability correction, Scotti filter length; explicit in the slow part, ν capped by ν/Pr Δt/(J Δη)² ≤ 0.25).
# HILL3D=0: the 2D ridge h(x) in ny rows; HILL3D=1: the bell mountain h(x, y) of eq. 84 in a domain LX × LY (km).
#
#     [RHS=0.95|0.5] [RAIN=1|0] [SMAG=1|0] [HILL3D=0|1] [LX=553] [LY=LX] [NY=1] [DT=30] [FAST=cs|ei|hevi] [PVAR=p|pi]
#         [THVAR=thetarho|energy|entropy] [METHOD=MIS54] [TAUFAC=0.5] [EVERY=3600 (s, output)]
#         julia --project=examples examples/fv3d_moist_hill.jl [T] [nx]
#
# Section 4.6: domain 553 km × 553 km (nx = 200, Δx = 2.765 km), 70 layers of Δz = 200 m (top 14 km), mountain
# h(x, y) = H/(((x - xm)² + (y - ym)²)/a² + 1)^1.5 (eq. 84; the ridge without y), H = 1 km, a = 11 km, hydrostatic background with T_s = 283.15 K,
# U = 10 m/s, dry stability N_d = 0.011 1/s, relative humidity RH(z) = RH_s (1/2 - arctan((z - z_m)/500 m)/π),
# z_m = 5 km (eq. 85: constant below z_m, decreasing above), RH_s = 95 % (RH95) or 50 % (RH50), Rayleigh damping
# above 11 km. Here: periodic (the paper: inflow/outflow), free slip, no surface fluxes, the flow
# starts impulsively; every column in discrete hydrostatic balance with its layer thickness J Δη. The paper gives no
# integration time; default T = 6 h.
# Output: table (every hour: max|w|, max q_c, q_r, surface precipitation, mass and water budget),
# fv3d_moist_hill[3d]_rh<RH_s %>.png: w, q_c and q_r (cf. Figs. 18, 19 of the paper) in the x–z plane through the
# summit and the accumulated surface precipitation (3D: map in the x–y plane).

using MultiRate, Printf, CairoMakie
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const CM = MultiRate.Common

T = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 6 * 3600.0
const HILL3D = get(ENV, "HILL3D", "0") == "1"
const LX = 1e3 * parse(Float64, get(ENV, "LX", "553"))
nx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : round(Int, LX / 2765)
const LY = 1e3 * parse(Float64, get(ENV, "LY", string(LX / 1e3)))
const ny = HILL3D ? round(Int, LY / (LX / nx)) : parse(Int, get(ENV, "NY", "1"))
const nz = 70
const HTOP, ZD = 14e3, 11e3
const SMAG = get(ENV, "SMAG", "1") == "1"
const Hm, a, U0, Nd, Ts, zm = 1e3, 11e3, 10.0, 0.011, 283.15, 5e3
const RHS = parse(Float64, get(ENV, "RHS", "0.95"))
const RAIN = get(ENV, "RAIN", "1") == "1"
const DT = parse(Float64, get(ENV, "DT", "30"))
const FAST = Symbol(get(ENV, "FAST", "cs"))
const PVAR = Symbol(get(ENV, "PVAR", "p"))
const THVAR = Symbol(get(ENV, "THVAR", "thetarho"))
const METHOD = Symbol(get(ENV, "METHOD", "MIS54"))
const OFFC = parse(Float64, get(ENV, "OFFCENTER", FAST in (:ei, :cs) ? "0.1" : "0"))
const SMDIV = parse(Float64, get(ENV, "SMDIV", FAST === :cs ? "0.1" : "0"))
xm = LX / 2
dx = LX / nx
ly = HILL3D ? LY : ny * dx
ym = ly / 2

mg = CM.MoistGas(thvar = THVAR)
gas = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0, g = 9.81)
ridge(x, y) = Hm / (((x - xm)^2 + (HILL3D ? (y - ym)^2 : 0.0)) / a^2 + 1)^1.5
grid = F3.Grid3D(nx, ny, nz, LX, ly, HTOP; zs = ridge, terrain = :follow)
Zc(i, j, k) = HTOP * (1 - grid.Jc[i, j]) + grid.zc[k] * grid.Jc[i, j]      # physical height of the cell centre
θd(z) = Ts * exp(Nd^2 * z / gas.g)                                         # dry θ, p_s = p0 at z = 0
RH(z) = RHS * (0.5 - atan((z - zm) / 500) / π)

"""(ρ, X, ρ_v) of subsaturated air at pressure p and height z (dry θ and RH of the profiles)."""
function air(p, z)
    Tz = θd(z) * (p / mg.p0)^(mg.Rd / mg.Cpd)
    e = min(RH(z), 1.0) * CM.psat(mg, Tz)
    ρv = e / (mg.Rv * Tz); ρd = (p - e) / (mg.Rd * Tz); ρ = ρd + ρv
    return ρ, CM._Θof(mg, ρ, Tz, ρv, 0.0), ρv
end
πof(p) = (p / mg.p0)^(mg.Rd / mg.Cpd)

"""Discretely balanced column (i, j): p at the lowest centre from dp/dz = -g ρ (RK2 from p0 at z = 0), then the W row
of the model upwards, δp + g J Δη ρ̄ = 0 (pvar = :p) or c_pd Θ̂_f δπ + g J Δη ρ̄ = 0 (secant)."""
function balanced_column(i, j)
    grav = gas.g; z1 = Zc(i, j, 1)
    p = mg.p0; n = 100; h = z1 / n; z = 0.0
    for _ in 1:n
        k1 = -grav * air(p, z)[1]; k2 = -grav * air(p + h * k1, z + h)[1]
        p += h * (k1 + k2) / 2; z += h
    end
    ps = zeros(nz); ps[1] = p
    dzi = grid.Jc[i, j] * grid.dz
    for k in 2:nz
        pl = ps[k-1]; ρl = air(pl, Zc(i, j, k - 1))[1]; Θhl = pl / (mg.Rd * πof(pl))
        R(pk) = (PVAR === :p ? pk - pl : mg.Cpd * 0.5 * (Θhl + pk / (mg.Rd * πof(pk))) * (πof(pk) - πof(pl))) +
                grav * dzi * 0.5 * (ρl + air(pk, Zc(i, j, k))[1])
        x0 = pl - grav * ρl * dzi; x1 = x0 * (1 - 1e-4); f0, f1 = R(x0), R(x1)
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

ρref(z) = air(mg.p0 * exp(-gas.g * z / (mg.Rd * Ts)), z)[1]               # output reference only
Θref(z) = air(mg.p0 * exp(-gas.g * z / (mg.Rd * Ts)), z)[2]
base0 = F3.Model3D(grid, gas, ρref, Θref)
cgz = U0 * min(1.0, U0 / (Nd * a))
base = F3.with_sponge(base0, zeros(F3.nunknowns(grid)); zD = ZD, αtop = 5cgz / (HTOP - ZD))
smag = SMAG ? F3.Smagorinsky(νmax = 0.25 * (1 / 3) * (minimum(grid.Jc) * grid.dz)^2 / DT) : nothing
m = F3.MoistModel3D(base; mgas = mg, pvar = PVAR, micro = :fast, rain = RAIN, smagorinsky = smag,
                    τfac = parse(Float64, get(ENV, "TAUFAC", FAST === :hevi ? "0.25" : FAST === :ei ? "1" : "0.5")))
qb = zeros(base.nq); Q = zeros(nx, ny, nz)
for j in 1:ny, i in 1:nx
    ps = balanced_column(i, j)
    for k in 1:nz
        ρ, X, ρv = air(ps[k], Zc(i, j, k))
        qb[F3.iρ(grid, i, j, k)] = ρ; qb[F3.iΘ(grid, i, j, k)] = X; Q[i, j, k] = ρv
    end
end
rest = F3.moist_exner_state(m, qb, Q)
Nr = TS.rhs_nonlinear(m, rest)
x0 = copy(rest)
for k in 1:nz, j in 1:ny, i in 1:nx
    x0[F3.iU(grid, i, j, k)] = U0 * 0.5 * (qb[F3.iρ(grid, i - 1, j, k)] + qb[F3.iρ(grid, mod1(i, nx), j, k)])
end
@printf("moist %s (Jähn et al. 2015, Sect. 4.6): %d × %d × %d cells, Δx = %.0f m, Δy = %.0f m, Δη = %.0f m, RH_s = %.0f %%, rain %s, Smagorinsky %s, %s, τ = %.2f s\n",
        HILL3D ? "mountain" : "ridge", nx, ny, nz, grid.dx, grid.dy, grid.dz, 100RHS, RAIN, SMAG ? @sprintf("on (ν ≤ %.0f m²/s)", smag.νmax) : "off", PVAR, m.τ)
@printf("rest state on terrain: max|N_W| = %.2e, max|N_U| = %.2e (horizontal pressure-gradient error)\n",
        maximum(abs, Nr[F3.iW(grid, i, j, k)] for i in 1:nx, j in 1:ny, k in 2:nz),
        maximum(abs, Nr[F3.iU(grid, i, j, k)] for i in 1:nx, j in 1:ny, k in 1:nz))

c = 350.0
dτ = 0.85 * 0.92 * (ny > 1 ? 1 / sqrt(1 / grid.dx^2 + 1 / grid.dy^2) : grid.dx) / c   # 2D acoustic limit in x–y
nsub = ceil(Int, DT / dτ)
tab = TS.mis_tableau(METHOD)
n = round(Int, T / DT); every = max(1, round(Int, parse(Float64, get(ENV, "EVERY", "3600")) / DT))
@printf("%s, Δt = %.0f s, nsub = %d (h = %.2f s), %d steps, fast integrator %s (offcenter %.2f, smdiv %.2f), thermodynamic variable %s\n",
        METHOD, DT, nsub, DT / nsub, n, FAST, OFFC, SMDIV, THVAR)
tot(x, v) = F3.total3d(m, x, v)
water(x) = tot(x, :qv) + tot(x, :qc) + (RAIN ? tot(x, :qr) : 0.0)
x = copy(x0)
w = TS.FrozenWork(m, length(x), tab); tmp = ntuple(_ -> zeros(length(x)), 6)
precip = zeros(nx, ny)                                    # accumulated surface precipitation (kg/m² = mm)
prate() = RAIN ? F3.precipitation_rate(m, x) : zeros(nx, ny)
mass0, water0 = tot(x0, :ρ), water(x0)
dA = grid.dx * grid.dy
t0 = time()
for s in 1:n
    precip .+= 0.5DT .* prate()
    TS.frozen_mis_step!(x, m, x, DT, tab, w; nsub, fast = FAST, tmp, smdiv = SMDIV, offcenter = OFFC)
    TS.frozen_step_end!(m, x)
    precip .+= 0.5DT .* prate()
    all(isfinite, x) || (println("unstable at t = $(s * DT) s"); exit(1))
    if s % every == 0 || s == n
        dM = tot(x, :ρ) - mass0
        νt = SMAG ? (TS.frozen_data!(m.C, m.S, m, x); F3.eddy_viscosity(m, x)) : zeros(1)
        @printf("t = %5.2f h: max|w| %.2f m/s, max q_c %.2e, max q_r %.2e, max ν %.0f m²/s (%.1f %% of the cells with ν > 0), Courant V_r h/Δz %.2f, precipitation mean %.3f mm, max %.2f mm, Δmass %.1e, (Δwater + P)/water %.1e, %.0f s\n",
                s * DT / 3600, maximum(abs, F3.vertical_velocity(m, x)), maximum(F3.qc3d(m, x)),
                RAIN ? maximum(F3.qr3d(m, x)) : 0.0, maximum(νt), 100 * count(>(0), νt) / length(νt), RAIN ? F3.rain_courant(m, x, DT / nsub) : 0.0,
                sum(precip) / (nx * ny), maximum(precip), dM / mass0,
                (water(x) - water0 + sum(precip) * dA) / water0, time() - t0)
        flush(stdout)
    end
end

# x–z plane through the summit (row j = ny/2 + 1 in 3D): w at the faces, q_c, q_r at the centres (physical heights), precipitation
j = HILL3D ? ny ÷ 2 + 1 : 1
X = [grid.xc[i] / 1e3 for i in 1:nx, k in 1:nz]
Zk = [Zc(i, j, k) / 1e3 for i in 1:nx, k in 1:nz]
wf = F3.vertical_velocity(m, x)[:, j, :]                  # η-faces 2 … nz
Xw = [grid.xc[i] / 1e3 for i in 1:nx, k in 2:nz]
Zw = [(HTOP * (1 - grid.Jc[i, j]) + grid.zf[k] * grid.Jc[i, j]) / 1e3 for i in 1:nx, k in 2:nz]
qc = 1e3 .* F3.qc3d(m, x)[:, j, :]; qr = 1e3 .* F3.qr3d(m, x)[:, j, :]
xk, Lk = xm / 1e3, LX / 1e3
w1 = (max(0.0, xk - 77), min(Lk, xk + 83)); w2 = (max(0.0, xk - 37), min(Lk, xk + 63))     # 200–360, 240–340 km for 553 km
sel = findall(xx -> w1[1] <= xx <= w1[2], X[:, 1])
fig = Figure(size = (1400, 1300))
ax1 = Axis(fig[1, 1]; title = "RH$(round(Int, 100RHS)), t = $(round(T / 3600, digits = 2)) h: w [m/s]", xlabel = "x [km]",
           ylabel = "z [km]", limits = (w1..., 0, 14))
s1 = surface!(ax1, Xw[sel, :], Zw[sel, :], zeros(length(sel), nz - 1); color = wf[sel, :], colormap = :balance,
              colorrange = (-1, 1) .* max(1e-3, maximum(abs, wf[sel, :])), shading = NoShading)
lines!(ax1, X[sel, 1], [ridge(grid.xc[i], grid.yc[j]) / 1e3 for i in sel]; color = :black)
Colorbar(fig[1, 2], s1)
sel2 = findall(xx -> w2[1] <= xx <= w2[2], X[:, 1])
for (r, (name, f, cm)) in enumerate((("q_c [g/kg]", qc, :blues), ("q_r [g/kg]", qr, :heat)))
    ax = Axis(fig[r+1, 1]; title = name, xlabel = "x [km]", ylabel = "z [km]", limits = (w2..., 0, 7))
    sf = surface!(ax, X[sel2, :], Zk[sel2, :], zeros(length(sel2), nz); color = f[sel2, :], colormap = cm,
                  colorrange = (0, max(1e-3, maximum(f))), shading = NoShading)
    lines!(ax, X[sel2, 1], [ridge(grid.xc[i], grid.yc[j]) / 1e3 for i in sel2]; color = :black)
    Colorbar(fig[r+1, 2], sf)
end
if HILL3D
    ax3 = Axis(fig[4, 1]; title = "accumulated surface precipitation [mm], terrain contours 100 m", xlabel = "x [km]", ylabel = "y [km]",
               aspect = DataAspect())
    hm = heatmap!(ax3, grid.xc ./ 1e3, grid.yc ./ 1e3, precip; colormap = :blues, colorrange = (0, max(1e-3, maximum(precip))))
    contour!(ax3, grid.xc ./ 1e3, grid.yc ./ 1e3, [ridge(xx, yy) for xx in grid.xc, yy in grid.yc]; levels = 100:100:900, color = :black)
    Colorbar(fig[4, 2], hm)
else
    ax3 = Axis(fig[4, 1]; title = "accumulated surface precipitation [mm]", xlabel = "x [km]", limits = (w1, nothing))
    lines!(ax3, grid.xc ./ 1e3, precip[:, j])
end
fname = get(ENV, "OUT", "fv3d_moist_hill$(HILL3D ? "3d" : "")_rh$(round(Int, 100RHS))$(RAIN ? "" : "_norain")$(SMAG ? "" : "_nosmag").png")
save(fname, fig)
println("wrote $fname")
