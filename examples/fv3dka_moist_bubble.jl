# 3D moist bubble (as examples/fv3d_moist_bubble.jl) with the KernelAbstractions model FV3DKA: x–y tiles, serial or
# MPI, :cs small steps, pressure p as the additional variable. Every rank builds only its tile (no global state).
#
#     julia --project=examples examples/fv3dka_moist_bubble.jl [T] [nx]                       (one rank)
#     julia --project=examples -e 'using MPI; run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project=examples examples/fv3dka_moist_bubble.jl 1000 40 2 2`)'
#
# Arguments: T (s, default 1000), nx (default 40; nx × nx × nx/2 cells on 20 km × 20 km × 10 km), px py (tiles,
# default 1 1). Environment: METHOD (MIS54), DT (5 s), TAUFAC (0.5), THVAR (thetarho). Output: diagnostics on rank 0,
# fv3dka_moist_bubble.png (θ_e′, w, q_c′ in the x–z plane next to the centre) at T.

const PX = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
const PY = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
const USE_MPI = PX * PY > 1
USE_MPI && (using MPI; MPI.Init())
using MultiRate, Printf, CairoMakie
const TS = MultiRate.TimeStepping
const K = MultiRate.FV3DKA
const CM = MultiRate.Common

T = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1000.0
nx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 40
nz = nx ÷ 2
L, H = 20e3, 10e3
dec = USE_MPI ? K.mpi_decomp(PX, PY) : K.SerialDecomp()
rank = USE_MPI ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
cx, cy = K.decomp_coords(dec)
mg = CM.MoistGas()
grav = 9.81
grid = K.KAGrid(nx, nx, nz, L, L, H; px = PX, py = PY, cx, cy)
m = K.MoistKA(grid, dec; mgas = CM.MoistGas(thvar = Symbol(get(ENV, "THVAR", "thetarho"))), pvar = :p,
              τfac = parse(Float64, get(ENV, "TAUFAC", "0.5")), grav)
const θe0, rt0 = 320.0, 0.02

"""Saturated state with θ_e = 320 K, r_t = 0.02 at pressure p: (ρ, Θ = ρθ_ρ, Q = ρq_t)."""
function sat_state(p)
    c = mg.Cpd + rt0 * mg.Cpl
    function θe(Tm)
        e = CM.psat(mg, Tm); pd = p - e; rv = mg.Rd / mg.Rv * e / pd
        Tm * (pd / mg.p0)^(-mg.Rd / c) * exp(CM.latent(mg, Tm) * rv / (c * Tm))
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

"""Discretely balanced saturated column of the pressure form: p_k - p_{k-1} + g Δz ρ̄ = 0 (secant in p_k)."""
function balanced_column()
    p = mg.p0; nsteps = 50; h = 0.5grid.dz / nsteps
    for _ in 1:nsteps
        k1 = -grav * sat_state(p)[1]; k2 = -grav * sat_state(p + h * k1)[1]
        p += h * (k1 + k2) / 2
    end
    ps = zeros(nz); ps[1] = p
    for k in 2:nz
        pl = ps[k-1]; ρl = sat_state(pl)[1]
        R(pk) = pk - pl + grav * grid.dz * 0.5 * (ρl + sat_state(pk)[1])
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
    r = sqrt((x - L / 2)^2 + (y - L / 2)^2 + (z - 2e3)^2)
    r < 2e3 || return ρ, Θ, Q
    κd = mg.Rd / mg.Cpd
    _, ρv, ρc, _ = CM.moist_state(mg, ρ, Θ, Q)
    ρd = ρ - Q
    κM = (mg.Rd * ρd + mg.Rv * ρv) / (mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * ρc)
    θdens_new = Θ / ρ * (p / mg.p0)^(κM - κd) * (1 + 2.0 * cospi(0.5 * r / 2e3)^2 / 300)
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
level(z) = clamp(floor(Int, z / grid.dz) + 1, 1, nz)
function initial(x, y, z)
    k = level(z)
    ρ, Θ, Q = perturb(x, y, z, cols[k]..., ps[k])
    return (ρ, 0.0, 0.0, 0.0, Θ, Q)
end
q = K.ka_project(m, initial)
c = 1.05 * maximum(sqrt(mg.Cpd / (mg.Cpd - mg.Rd) * CM.moist_state(mg, cols[k]...)[4] / cols[k][1]) for k in 1:nz)
dτ = 0.9 / (c * sqrt(1 / grid.dx^2 + 1 / grid.dy^2 + 1 / grid.dz^2))
Δt = parse(Float64, get(ENV, "DT", "5"))
tab = TS.mis_tableau(Symbol(get(ENV, "METHOD", "MIS54")))
nsub = ceil(Int, Δt / dτ)
mass0 = K.ka_total(m, q, K.VR); water0 = K.ka_total(m, q, K.VQV) + K.ka_total(m, q, K.VQC)
rank == 0 && @printf("3D moist bubble (FV3DKA): %d × %d × %d cells (Δx = %.0f m), %d × %d tiles, %s Δt = %.1f s, nsub %d, τ = %.2f s, backend %s\n",
                     nx, nx, nz, grid.dx, PX, PY, tab.name, Δt, nsub, m.τ, typeof(K.KA.get_backend(m)))
function report(n, q)
    f = K.ka_fields(m, q)
    wmax = K.allreduce_max(dec, maximum(abs, f.w)); θmax = K.allreduce_max(dec, maximum(f.θe)) - θe0
    mass = K.ka_total(m, q, K.VR) / mass0 - 1
    water = (K.ka_total(m, q, K.VQV) + K.ka_total(m, q, K.VQC)) / water0 - 1
    rank == 0 && @printf("t = %6.0f s: max|w| %6.2f m/s, max θ_e′ %.3f K, Δmass %.1e, Δwater %.1e\n", n * Δt, wmax, θmax, mass, water)
end
nsteps = round(Int, T / Δt)
t = @elapsed K.ka_run!(m, q, Δt, nsteps; tab, nsub, monitor = report, every = max(1, round(Int, 200 / Δt)))
rank == 0 && @printf("%d steps in %.0f s (%.2f s per step)\n", nsteps, t, t / nsteps)

# x–z section next to the centre (j = nx/2 + 1), collected on all ranks
jc = nx ÷ 2 + 1
f = K.ka_fields(m, q)
sec(a, nk) = (s = zeros(nx, nk);
              grid.j0 < jc <= grid.j0 + grid.nyl && (s[grid.i0+1:grid.i0+grid.nxl, :] .= a[:, jc-grid.j0, :]);
              K.allreduce_sum(dec, s))
θs, ws, qs = sec(f.θe, nz) .- θe0, sec(f.w, nz - 1), 1e3 .* sec(f.qc, nz)
if rank == 0
    qs .-= sum(qs; dims = 1) ./ nx
    fig = Figure(size = (1500, 400))
    xk, zk = (grid.dx / 1e3) .* ((1:nx) .- 0.5), (grid.dz / 1e3) .* ((1:nz) .- 0.5)
    for (col, (name, z, a, cr, cm)) in enumerate((("θ_e′ [K]", zk, θs, (-0.5, 2.5), :thermal),
                                                  ("w [m/s]", grid.zf[2:nz] ./ 1e3, ws, (-12, 12), :balance),
                                                  ("q_c′ [g/kg] (minus horizontal mean)", zk, qs, (-1, 1) .* max(1e-3, maximum(abs, qs)), :balance)))
        ax = Axis(fig[1, 2col-1]; title = "$(tab.name), Δt = $(Δt) s, t = $(round(Int, T)) s: $name", xlabel = "x [km]",
                  ylabel = "z [km]", aspect = DataAspect(), limits = (4, 16, 0, 10))
        hm = heatmap!(ax, xk, z, a; colormap = cm, colorrange = cr)
        Colorbar(fig[1, 2col], hm)
    end
    fn = get(ENV, "OUT", "fv3dka_moist_bubble.png")
    save(fn, fig)
    println("wrote ", fn)
end
USE_MPI && MPI.Finalize()
