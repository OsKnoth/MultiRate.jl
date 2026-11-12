# Dry hydrostatic mountain wave over the Agnesi hill (Gallus & Klemp 2000, a = 10 km) on the
# terrain-following (Gal-Chen) FV grid: comparison of split-explicit time integrators, fast part
# HEVI (horizontal Störmer–Verlet half steps around a (linearly implicit) Crank–Nicolson vertical step).
# FORM (env):
#   exner    frozen split of the extended system with the Exner pressure as an additional variable (`FV.ExnerModel`)
#   gradp    frozen split of the FV operator, full pressure p(Θ_z) in the fast part (FV_PRESSURE_FORM = :gradp)
#   product  frozen split, frozen product form (dp/dΘ)_f(y) δΘ (FV_PRESSURE_FORM = :product)
#   linear   classical split N = L_n + (N - L_n): linear fast operator frozen at qⁿ (`fast_operator`), same
#            operator N as gradp (shares its reference)
#
#     FORM=exner MODE=ref julia --project=examples examples/frozen_agnesi.jl          # RK4 reference (stored)
#     [METHODS=MIS2,MIS3C,MIS4a,MIS54,WSRK3,ETD3,EB4,LRK4] [DTS=10,20,40,60,90,120] [NSUBFAC=1] \
#         julia --project=examples examples/frozen_agnesi.jl                 # scan, table lines "RES …"
#
# h(x) = h0/(1 + (x - xm)²/a²), h0 = 400 m, a = 10 km, U = 10 m/s, N = 0.01 1/s, θ0 = 300 K; periodic
# channel 40 a = 400 km, top 15.6 km, Rayleigh damping of W above 9 km (rate 5 c_gz/(H - z_D));
# Δx = 2 km (200 columns), Δη = 200 m (78 layers); each column in discrete hydrostatic balance of the
# discretization of FORM; impulsive start; T = 10.8 a/U = 10800 s (env T). nsub = NSUBFAC ⌈Δt/Δτ⌉ with the
# horizontal acoustic step Δτ = 0.85·0.92 Δx/c. Reference: RK4 of the same semi-discrete operator.
# Output: lines "RES form method Δt nsub status max|w| relL2(w) max|Δw| relL2(θ') fast-steps time" (errors
# NaN without reference) and the final states agnesi_x_<form>_<method>_<Δt>_<nsub>.jls in OUTDIR (SAVE=0: none).

include(joinpath(@__DIR__, "euler2d_common.jl"))
using Serialization

const a, h0, U0, NBV, θ0 = 10e3, 400.0, 10.0, 0.01, 300.0
const LX, HTOP, ZD = 40a, 15.6e3, 9e3
const nx, nz = 200, 78
const TEND = parse(Float64, get(ENV, "T", string(10.8a / U0)))
const FORM = Symbol(get(ENV, "FORM", "exner"))
FORM in (:exner, :gradp, :product, :linear) || error("FORM: exner, gradp, product or linear")
FV.FV_PRESSURE_FORM[] = FORM === :product ? :product : :gradp
FV.FV_TERRAIN_FORM[] = :flux
const OUTDIR = get(ENV, "OUTDIR", ".")
xm = LX / 2

gas = Gas()
cpd = gas.R * gas.γ / (gas.γ - 1); κ = gas.R / cpd
grid = FV.FVGrid(nx, nz, LX, HTOP; zs = x -> h0 / (1 + ((x - xm) / a)^2))
θb(z) = θ0 * exp(NBV^2 * z / gas.g)
ρof(p, z) = p / (gas.R * θb(z) * (p / gas.p00)^κ)
πof(p) = (p / gas.p00)^κ

"""Column i in discrete hydrostatic balance of the W row of FORM: p at the lowest centre from
dp/dz = -g ρ (RK2 from p00 at z = 0), then upwards cp Θ_f (π_k - π_{k-1}) (exner),
(γp/Θ)_f (Θ_k - Θ_{k-1}) (product) or p_k - p_{k-1} (gradp, linear) + g J Δη ρ̄ = 0."""
function balanced_column(i)
    z1 = grid.Zc[i, 1]; p = gas.p00; n = 100; h = z1 / n; z = 0.0
    for _ in 1:n
        k1 = -gas.g * ρof(p, z); k2 = -gas.g * ρof(p + h * k1, z + h)
        p += h * (k1 + k2) / 2; z += h
    end
    ps = zeros(nz); ps[1] = p; dzi = grid.Jc[i] * grid.dz
    for k in 2:nz
        pl = ps[k-1]; zl = grid.Zc[i, k-1]; zk = grid.Zc[i, k]
        ρl = ρof(pl, zl); Θl = ρl * θb(zl)
        Θk(pk) = ρof(pk, zk) * θb(zk)
        dP(pk) = FORM === :exner ? cpd * 0.5 * (Θl + Θk(pk)) * (πof(pk) - πof(pl)) :
                 FORM === :product ? 0.5 * gas.γ * (pl / Θl + pk / Θk(pk)) * (Θk(pk) - Θl) : pk - pl
        R(pk) = dP(pk) + gas.g * dzi * 0.5 * (ρl + ρof(pk, zk))
        x0 = pl - gas.g * ρl * dzi; x1 = x0 * (1 - 1e-4); f0, f1 = R(x0), R(x1)
        for _ in 1:60
            (f1 == 0 || f1 == f0) && break
            x0, x1, f0 = x1, x1 - f1 * (x1 - x0) / (f1 - f0), f1
            abs(x1 - x0) <= 1e-14 * abs(x1) && break
            f1 = R(x1)
        end
        ps[k] = x1
    end
    return ps
end

ρr(z) = ρof(gas.p00 * exp(-gas.g * z / (gas.R * θ0)), z)          # output reference only
Θr(z) = ρr(z) * θb(z)
base0 = FV.FVModel(grid, gas, ρr, Θr)
qb = zeros(base0.nq)
for i in 1:nx
    ps = balanced_column(i)
    for k in 1:nz
        ρ = ρof(ps[k], grid.Zc[i, k])
        qb[FV.iρ(grid, i, k)] = ρ; qb[FV.iΘ(grid, i, k)] = ρ * θb(grid.Zc[i, k])
    end
end
cgz = U0 * min(1.0, U0 / (NBV * a))
base = FV.with_sponge(base0, zeros(base0.nq); zD = ZD, αtop = 5cgz / (HTOP - ZD))
m = FORM === :exner ? FV.ExnerModel(base; pvar = :pi) : base
rest = FORM === :exner ? FV.exner_state(m, qb) : copy(qb)
x0 = copy(rest)
for k in 1:nz, i in 1:nx
    x0[FV.iU(grid, i, k)] = U0 * 0.5 * (qb[FV.iρ(grid, i - 1, k)] + qb[FV.iρ(grid, i, k)])
end

wphys(x, i, k) = x[FV.iW(grid, i, k)] / (0.5 * (x[FV.iρ(grid, i, k - 1)] + x[FV.iρ(grid, i, k)]))
"""Physical w at the cell centres (ground: w = z_x u)."""
w_c(x) = [0.5 * ((k > 1 ? wphys(x, i, k) : grid.zx_f[i, 1] * U0) + (k < nz ? wphys(x, i, k + 1) : 0.0)) for i in 1:nx, k in 1:nz]
θp(x) = [x[FV.iΘ(grid, i, k)] / x[FV.iρ(grid, i, k)] - θb(grid.Zc[i, k]) for i in 1:nx, k in 1:nz]
# comparison region: |x - xm| ≤ 3a = 30 km, z ≤ 9 km (below the sponge), where the steady wave dominates
# (farther away the transients of the impulsive start dominate)
const REG = [abs(grid.xc[i] - xm) <= 3a && grid.Zc[i, k] <= ZD for i in 1:nx, k in 1:nz]
relL2(f, g) = sqrt(sum(abs2, (f .- g)[REG]) / sum(abs2, g[REG]))

c = 1.05 * FV.sound_speed_max(base, qb)
dτ = 0.85 * 0.92 * grid.dx / c
reffile = joinpath(OUTDIR, "agnesi_ref_$(FORM === :linear ? :gradp : FORM)_T$(round(Int, TEND)).jls")

if get(ENV, "MODE", "scan") == "ref"
    Nr = TS.rhs_nonlinear(m, rest)
    @printf("rest state: max|N_W| = %.2e, max|N_U| = %.2e\n", maximum(abs, Nr[FV.iW(grid, i, k)] for i in 1:nx, k in 2:nz),
            maximum(abs, Nr[FV.iU(grid, i, k)] for i in 1:nx, k in 1:nz))
    dzmin = minimum(grid.Jc) * grid.dz
    nref = ceil(Int, TEND / (0.4 * dzmin / (c + U0)))
    t = @elapsed xr = TS.run_rk4(m, x0, TEND / nref, nref)
    serialize(reffile, xr)
    @printf("RK4 reference: %d steps (Δt = %.3f s), %.0f s, max|w| = %.4f m/s -> %s\n", nref, TEND / nref, t, maximum(abs, w_c(xr)), reffile)
    exit()
end

haveref = isfile(reffile)
xr = haveref ? deserialize(reffile) : x0
wr, θr_ = w_c(xr), θp(xr)
wmax_r = haveref ? maximum(abs, wr) : 0.4          # without reference: errors NaN, instability bound 20·0.4 m/s
methods = Symbol.(split(get(ENV, "METHODS", "MIS2,MIS3C,MIS4a,MIS54,WSRK3,ETD3,EB4,LRK4"), ","))
dts = parse.(Float64, split(get(ENV, "DTS", "10,20,40,60,90,120"), ","))
facs = parse.(Int, split(get(ENV, "NSUBFAC", "1"), ","))
@printf("Agnesi FORM = %s, a = 10 km, %d × %d, Δx = %.0f m, Δη = %.0f m, T = %.0f s, Δτ_ac = %.2f s, reference max|w| = %.4f m/s\n",
        FORM, nx, nz, grid.dx, grid.dz, TEND, dτ, wmax_r)
for meth in methods, Δt in dts, fac in facs
    tab = TS.mis_tableau(meth)
    nsub = fac * ceil(Int, Δt / dτ)
    n = round(Int, TEND / Δt)
    x = copy(x0)
    if FORM === :linear
        w = TS.split_work(tab, length(x))
    else
        w = TS.FrozenWork(m, length(x), tab); tmp = (zeros(length(x)), zeros(length(x)), zeros(length(x)))
    end
    ok = true
    t = @elapsed try
        for s in 1:n
            if FORM === :linear      # linear fast operator frozen at qⁿ, refrozen every step
                sF = TS.fast_operator(m, x; fast = :hevi)
                TS.mis_step!(x, m, x, Δt, tab, sF, w; nsub, fast = :hevi)
            else
                TS.frozen_mis_step!(x, m, x, Δt, tab, w; nsub, fast = :hevi, tmp)
                TS.frozen_step_end!(m, x)
            end
            if s % 20 == 0 && (!all(isfinite, x) || maximum(abs, w_c(x)) > 20wmax_r)
                ok = false; break
            end
        end
    catch
        ok = false
    end
    ok = ok && all(isfinite, x)
    fast = round(Int, n * TS.fast_time(tab) * nsub)
    if ok
        wx = w_c(x); ok = maximum(abs, wx) <= 20wmax_r
    end
    if ok
        e = haveref ? (relL2(wx, wr), maximum(abs, (wx .- wr)[REG]), relL2(θp(x), θr_)) : (NaN, NaN, NaN)
        @printf("RES %-7s %-6s %6.1f %3d stable   %.4f %.3e %.3e %.3e %7d %6.0f\n", FORM, meth, Δt, nsub, maximum(abs, wx), e..., fast, t)
        get(ENV, "SAVE", "1") == "1" && serialize(joinpath(OUTDIR, "agnesi_x_$(FORM)_$(meth)_$(round(Int, Δt))_$(nsub).jls"), x)
    else
        @printf("RES %-7s %-6s %6.1f %3d unstable %.4f %.3e %.3e %.3e %7d %6.0f\n", FORM, meth, Δt, nsub, NaN, NaN, NaN, NaN, fast, t)
    end
    flush(stdout)
end
