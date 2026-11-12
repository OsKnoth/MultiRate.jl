# Moist flow over a bell-shaped hill on the terrain-following (Gal-Chen) FV grid: frozen MIS with
# the Exner pressure (or p) as an additional prognostic variable and ρθ_ρ, ρq_v, ρq_c
# (`FV.MoistExnerModel`), condensation by the explicit Fischer–Burmeister relaxation with the
# Bryan–Fritsch latent heating ("simple microphysics", no rain). Hydrostatic mountain wave
# (Gallus & Klemp 2000, a = 10 km case): h(x) = h0/(1 + (x - xm)²/a²), U = 10 m/s, N = 0.01 1/s
# (dry θ), subsaturated air with relative humidity RH0 below ZRH, decreasing to 20 % 2 km higher.
# Air lifted by the wave saturates: a cloud forms where the displacement is upward.
#
#     [PVAR=pi|p] [RH0=0.95] [H0=400] [DT=30] [METHOD=MIS54] [TAUFAC=0.25] \
#         julia --project=examples examples/frozen_moist_hill.jl [T] [nx]
#
# RH0 = 0: dry run. Periodic channel 40 a, model top 12 km, Rayleigh damping of W above 8 km,
# nx × 48 cells (default nx = 200, Δx = 2 km, Δη = 250 m). Each column is in discrete hydrostatic
# balance with its own layer thickness J Δη; the flow starts impulsively. T default 21.6 a/U.
# Output: table (every hour), frozen_moist_hill_<PVAR>.png (w, q_c, RH at T).

include(joinpath(@__DIR__, "euler2d_common.jl"))

const a, U0, NBV, θ0 = 10e3, 10.0, 0.01, 288.0
const h0 = parse(Float64, get(ENV, "H0", "400"))
const RH0 = parse(Float64, get(ENV, "RH0", "0.95"))
const ZRH = 7e3
const LX, HTOP, ZD = 40a, 12e3, 8e3
T = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 21.6a / U0
nx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
nz = 48
const PVAR = Symbol(get(ENV, "PVAR", "pi"))
const DT = parse(Float64, get(ENV, "DT", "30"))
const METHOD = Symbol(get(ENV, "METHOD", "MIS54"))
xm = LX / 2

mg = FV.MoistGas()
gas = Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0, g = 9.81)
grid = FV.FVGrid(nx, nz, LX, HTOP; zs = x -> h0 / (1 + ((x - xm) / a)^2))
θd(z) = θ0 * exp(NBV^2 * z / gas.g)
RH(z) = RH0 * (z <= ZRH ? 1.0 : max(0.2 / max(RH0, 1e-12), 1 - (z - ZRH) / 2e3 * (1 - 0.2 / max(RH0, 1e-12))))

"""(ρ, Θ = ρθ_ρ, Q = ρ_v) of subsaturated air at pressure p and height z (RH of the profile)."""
function air(p, z)
    T = θd(z) * (p / mg.p0)^(mg.Rd / mg.Cpd)
    e = RH0 > 0 ? min(RH(z), 1.0) * FV.psat(mg, T) : 0.0
    ρv = e / (mg.Rv * T); ρd = (p - e) / (mg.Rd * T); ρ = ρd + ρv
    return ρ, FV._Θof(mg, ρ, T, ρv, 0.0), ρv
end
πof(p) = (p / mg.p0)^(mg.Rd / mg.Cpd)

"""Discretely balanced column i: p at the lowest centre from dp/dz = -g ρ (RK2 from p0 at z = 0),
then the W row of the model, c_pd Θ̂_f (π_k - π_{k-1}) + g J Δη ρ̄ = 0 (or δp + g J Δη ρ̄ = 0), upwards."""
function balanced_column(i)
    grav = gas.g; z1 = grid.Zc[i, 1]
    p = mg.p0; n = 100; h = z1 / n; z = 0.0
    for _ in 1:n
        k1 = -grav * air(p, z)[1]; k2 = -grav * air(p + h * k1, z + h)[1]
        p += h * (k1 + k2) / 2; z += h
    end
    ps = zeros(nz); ps[1] = p
    dzi = grid.Jc[i] * grid.dz
    for k in 2:nz
        pl = ps[k-1]; ρl = air(pl, grid.Zc[i, k-1])[1]; Θhl = pl / (mg.Rd * πof(pl))
        R(pk) = (PVAR === :p ? pk - pl : mg.Cpd * 0.5 * (Θhl + pk / (mg.Rd * πof(pk))) * (πof(pk) - πof(pl))) +
                grav * dzi * 0.5 * (ρl + air(pk, grid.Zc[i, k])[1])
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

ρr(z) = air(mg.p0 * exp(-gas.g * z / (mg.Rd * θ0)), z)[1]              # output reference only
Θr(z) = air(mg.p0 * exp(-gas.g * z / (mg.Rd * θ0)), z)[2]
base0 = FV.FVModel(grid, gas, ρr, Θr)
qb = zeros(base0.nq); Q = zeros(nx, nz)
for i in 1:nx
    ps = balanced_column(i)
    for k in 1:nz
        ρ, Θ, ρv = air(ps[k], grid.Zc[i, k])
        qb[FV.iρ(grid, i, k)] = ρ; qb[FV.iΘ(grid, i, k)] = Θ; Q[i, k] = ρv
    end
end
cgz = U0 * min(1.0, U0 / (NBV * a))
base = FV.with_sponge(base0, zeros(base0.nq); zD = ZD, αtop = 5cgz / (HTOP - ZD))
m = FV.MoistExnerModel(base; mgas = mg, pvar = PVAR, micro = :fast, τfac = parse(Float64, get(ENV, "TAUFAC", "0.25")))
rest = FV.moist_exner_state(m, qb, Q)
Nr = TS.rhs_nonlinear(m, rest)
@printf("moist hill: %d × %d cells, Δx = %.0f m, Δη = %.0f m, h0 = %.0f m, RH0 = %.2f, %s, τ = %.2f s\n",
        nx, nz, grid.dx, grid.dz, h0, RH0, PVAR, m.τ)
@printf("rest state on terrain: max|N_W| = %.2e, max|N_U| = %.2e (horizontal pressure-gradient error)\n",
        maximum(abs, Nr[FV.iW(grid, i, k)] for i in 1:nx, k in 2:nz), maximum(abs, Nr[FV.iU(grid, i, k)] for i in 1:nx, k in 1:nz))
x0 = copy(rest)
for k in 1:nz, i in 1:nx
    x0[FV.iU(grid, i, k)] = U0 * 0.5 * (qb[FV.iρ(grid, i - 1, k)] + qb[FV.iρ(grid, i, k)])
end

wphys(x, i, k) = x[FV.iW(grid, i, k)] / (0.5 * (x[FV.iρ(grid, i, k - 1)] + x[FV.iρ(grid, i, k)]))
w_c(x) = [0.5 * ((k > 1 ? wphys(x, i, k) : grid.zx_f[i, 1] * U0) + (k < nz ? wphys(x, i, k + 1) : 0.0)) for i in 1:nx, k in 1:nz]
qc_c(x) = [x[FV.iC(m, i, k)] / x[FV.iρ(grid, i, k)] for i in 1:nx, k in 1:nz]
function rh_c(x)
    [begin
        ρ, Θ, V, C = x[FV.iρ(grid, i, k)], x[FV.iΘ(grid, i, k)], x[FV.iV(m, i, k)], x[FV.iC(m, i, k)]
        T = FV.moist_state(mg, ρ, Θ, V, C)[1]
        V * mg.Rv * T / FV.psat(mg, T)
     end for i in 1:nx, k in 1:nz]
end
tot(x, f) = sum(grid.Jc[i] * x[f(i, k)] for i in 1:nx, k in 1:nz)

c = 1.05 * FV.sound_speed_max(base, qb)
dτ = 0.85 * 0.92 * grid.dx / c
nsub = ceil(Int, DT / dτ)
tab = TS.mis_tableau(METHOD)
n = round(Int, T / DT); every = max(1, round(Int, 3600 / DT))
@printf("%s, Δt = %.0f s, nsub = %d (Δτ = %.2f s), %d steps\n", METHOD, DT, nsub, DT / nsub, n)
x = copy(x0)
w = TS.FrozenWork(m, length(x), tab); tmp = (zeros(length(x)), zeros(length(x)), zeros(length(x)))
mass0, water0 = tot(x0, (i, k) -> FV.iρ(grid, i, k)), tot(x0, (i, k) -> FV.iV(m, i, k)) + tot(x0, (i, k) -> FV.iC(m, i, k))
t0 = time()
for s in 1:n
    TS.frozen_mis_step!(x, m, x, DT, tab, w; nsub, fast = :hevi, tmp)
    TS.frozen_step_end!(m, x)
    all(isfinite, x) || (println("unstable at t = $(s * DT) s"); exit(1))
    if s % every == 0 || s == n
        wc, qc = w_c(x), qc_c(x)
        cloudy = qc .> 1e-5
        @printf("t = %6.0f s: max|w| %.3f m/s, max q_c %.2e, cloudy cells %4d (w > 0 in %3.0f %%), Δmass %.1e, Δwater %.1e, %.0f s\n",
                s * DT, maximum(abs, wc), maximum(qc), count(cloudy), 100 * count(cloudy .& (wc .> 0)) / max(1, count(cloudy)),
                tot(x, (i, k) -> FV.iρ(grid, i, k)) / mass0 - 1,
                water0 > 0 ? (tot(x, (i, k) -> FV.iV(m, i, k)) + tot(x, (i, k) -> FV.iC(m, i, k))) / water0 - 1 : 0.0, time() - t0)
        flush(stdout)
    end
end

# cloud base / top and position relative to the hill
qc = qc_c(x); wc = w_c(x)
if any(qc .> 1e-5)
    idx = findall(qc .> 1e-5)
    xs = [grid.xc[I[1]] for I in idx]; zs_ = [grid.Zc[I] for I in idx]
    @printf("cloud (q_c > 1e-5): x - xm ∈ [%.1f, %.1f] km, z ∈ [%.2f, %.2f] km, max q_c %.2e at x - xm = %.1f km, z = %.2f km\n",
            (extrema(xs) .- xm) ./ 1e3..., extrema(zs_) ./ 1e3..., maximum(qc), (grid.xc[argmax(qc)[1]] - xm) / 1e3, grid.Zc[argmax(qc)] / 1e3)
end

X = [grid.xc[i] / 1e3 for i in 1:nx, k in 1:nz] .- xm / 1e3
Z = grid.Zc ./ 1e3
sel = findall(xx -> abs(xx) <= 60, X[:, 1])
fig = Figure(size = (1500, 450))
for (col, (name, f, cr, cm)) in enumerate((("w [m/s]", wc, (-1, 1) .* max(1e-3, maximum(abs, wc[sel, :])), :balance),
                                           ("q_c [g/kg]", 1e3 .* qc, (0, max(1e-3, 1e3 * maximum(qc))), :blues),
                                           ("RH", rh_c(x), (0.5, 1.05), :viridis)))
    ax = Axis(fig[1, 2col-1]; title = "$(RH0 > 0 ? "moist" : "dry") hill, t = $(round(Int, T)) s: $name",
              xlabel = "x - xm [km]", ylabel = "z [km]", limits = (-60, 60, 0, 9))
    sf = surface!(ax, X[sel, :], Z[sel, :], zeros(length(sel), nz); color = f[sel, :], colormap = cm, colorrange = cr, shading = NoShading)
    lines!(ax, X[sel, 1], grid.Zf[sel, 1] ./ 1e3; color = :black)
    Colorbar(fig[1, 2col], sf)
end
fname = "frozen_moist_hill_$(PVAR)$(RH0 > 0 ? "" : "_dry").png"
save(fname, fig)
println("wrote $fname")
