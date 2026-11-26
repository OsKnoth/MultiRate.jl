# Flow through an array of simple buildings: the cut-cell model FV3DKA.CutKA (dry, implicit = :amf, HEVI with GMRES)
# on the geometry of MultiRate.Buildings (ASAM level set → cut-cell fractions).
#
# Periodic domain 240 m × 192 m (the array in 0 … 192 m, a nudging strip 192 … 240 m towards 5 m/s and θ_ref(z), slow
# force, rates up to BUILD_ALPHA = 0.2 1/s), Δx = Δy = 4 m; levels 4 m up to 32 m, stretched (× 1.1) to 300 m, sponge above 200 m.
# 4 × 4 buildings with 16 m × 16 m footprints on a 48 m lattice, heights 16 m and 24 m alternating, every second one
# rotated by 30° (cut cells in general position). Stable stratification (N = 0.01 1/s), initial wind 5 m/s in x
# (divergence-free projection), kept up by the nudging strip, Koren limiter.
#
#     julia -t 8 --project=examples examples/building_array.jl [T = 120] [Δt = 0.4] [nsub = 40]
#     BUILDINGS_GPU=1 julia --project=gpuenv examples/building_array.jl 300     (CUDA; the state is saved, plot locally:)
#     julia --project=examples examples/building_array.jl plot examples/out/building_array_state.bin

using MultiRate, Printf, LinearAlgebra, SparseArrays, Serialization, KernelAbstractions
const ONGPU = get(ENV, "BUILDINGS_GPU", "0") == "1"
ONGPU && @eval using CUDA
const PLOTONLY = length(ARGS) >= 1 && ARGS[1] == "plot"
ONGPU || @eval using CairoMakie
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const K = MultiRate.FV3DKA
const CM = MultiRate.Common
const BD = MultiRate.Buildings
include(joinpath(@__DIR__, "..", "test", "cutka_projection.jl"))

T = length(ARGS) >= 1 && !PLOTONLY ? parse(Float64, ARGS[1]) : 120.0
Δt = length(ARGS) >= 2 && !PLOTONLY ? parse(Float64, ARGS[2]) : 0.4
nsub = length(ARGS) >= 3 && !PLOTONLY ? parse(Int, ARGS[3]) : 20
nout = length(ARGS) >= 4 && !PLOTONLY ? parse(Int, ARGS[4]) : 10                  # outputs during the run
tag = length(ARGS) >= 5 ? ARGS[5] : ""
out = joinpath(@__DIR__, "out"); mkpath(out)

# ------------------------------------------------------------------ geometry
L, Δx, a = 192.0, 4.0, 16.0
Lx = L + parse(Float64, get(ENV, "BUILD_NUDGE", "48"))               # nudging strip L … Lx downstream of the array (0: none)
αn = parse(Float64, get(ENV, "BUILD_ALPHA", "0.2"))                  # its largest rate [1/s]
"""Top face of a building: square of side a centred at (xc, yc), rotated by α, at height h."""
function block(n, xc, yc, h, α)
    c, s = cos(α), sin(α)
    ring = [(xc + c * u - s * v, yc + s * u + c * v, h) for (u, v) in ((-a / 2, -a / 2), (a / 2, -a / 2), (a / 2, a / 2), (-a / 2, a / 2))]
    return BD.BuildingPolygon(n, [ring])
end
polys = BD.BuildingPolygon[]
for i in 0:3, j in 0:3
    rot = get(ENV, "BUILD_ROT", "alt")                                # none | alt | all: rotated buildings
    α = rot == "all" || (rot == "alt" && isodd(i)) ? π / 6 : 0.0
    push!(polys, block(length(polys) + 1, 24.0 + 48i, 24.0 + 48j, isodd(i + j) ? 24.0 : 16.0, α))
end
const NOBUILD = get(ENV, "BUILD_NONE", "0") == "1"                  # no buildings (flat ground, noise in the start)
zf = get(ENV, "BUILD_LEVELS", "stretched") == "uniform" ? collect(0.0:4.0:300.0) : BD.stretched_levels(300.0; dz0 = 4.0, z1 = 32.0, r = 1.1)
limiter = Symbol(get(ENV, "BUILD_LIM", "koren"))
if NOBUILD
    t = 0.0
    g3 = F3.Grid3D(round(Int, Lx / Δx), 48, length(zf) - 1, Lx, L, zf[end]; zf, limiter)
else
    φ = BD.building_levelset(polys; origin = (0.0, 0.0), L = (Lx, L), z0 = 0.0)
    t = @elapsed cg = BD.cut_geometry(φ; Δx, zf, nsub = (8, 8), δsub = 0.5)
    g3 = BD.fv3d_grid(cg; limiter)
end
ncut = count(0 .< g3.Vf .< 1)
@printf("grid %d × %d × %d (dz %.0f … %.1f m), %d cut cells, %d masked, geometry %.1f s, building volume %.0f m³ (exact %.0f)\n",
        g3.nx, g3.ny, g3.nz, minimum(g3.dzc), maximum(g3.dzc), ncut, count(g3.mask), t,
        sum((1 - g3.Vf[i, j, k]) * Δx^2 * g3.dzc[k] for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz), 8 * a^2 * 16 + 8 * a^2 * 24)
@printf("rotation %s, limiter %s, jform %s, levels %s, N %s, buildings %s, nudging strip %.0f m (α %.2f), β %s, smdiv %s, c Δt/Δx %.1f, U Δt/Δx %.2f\n", get(ENV, "BUILD_ROT", "alt"), limiter,
        get(ENV, "BUILD_JFORM", "partial"), get(ENV, "BUILD_LEVELS", "stretched"), get(ENV, "BUILD_N", "0.01"), !NOBUILD, Lx - L, αn, get(ENV, "BUILD_BETA", "0.5"), get(ENV, "BUILD_SMDIV", "0.0"), 347Δt / Δx, 5Δt / Δx)
flush(stdout)

# ------------------------------------------------------------------ model and initial state
gas = CM.Gas()
ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = parse(Float64, get(ENV, "BUILD_N", "0.01")))
F3.FV3D_PRESSURE_FORM[] = :product
F3.FV3D_MASK_JFORM[] = Symbol(get(ENV, "BUILD_JFORM", "partial"))
F3.FV3D_HEVI_BETA[] = parse(Float64, get(ENV, "BUILD_BETA", "0.5"))  # off-centring of the implicit HEVI step
m3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 200.0, αtop = 0.05)
nud = Lx > L ? (x0 = L, x1 = Lx, αmax = αn, u = 5.0, v = 0.0, θ = θr) : nothing    # slow force (PALM-4U-like nudging)
nud === nothing || (m3 = F3.with_nudging(m3; nud...))
ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))
noise(x, y, z) = NOBUILD && z < 40 ? 0.05 * sin(2π * x / 24 + 1.3) * sin(2π * y / 32) * sin(π * z / 40) + 0.03 * sin(2π * (x + y) / 12) : 0.0
q0 = project_divfree(m3, F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k], (5 + 5noise(x, y, z)) * ρa[k],
                                                                              ρa[k] * noise(y, x, z), 0.0, Θr(g3.zc[k])))))
if PLOTONLY
    q, nsteps, Δt = deserialize(ARGS[2])
else
m = K.CutKA(K.CutGrid(g3; backend = ONGPU ? CUDABackend() : KernelAbstractions.CPU()); gas, sponge = (200.0, 0.05), nudge = nud,
            jform = F3.FV3D_MASK_JFORM[])
Q = K.ka_state(m, g3, q0)
tab = TS.mis_tableau(:MIS54)
w = K.CutMISWork(m, TS.nstages(tab))

# ------------------------------------------------------------------ run
nsteps = round(Int, T / Δt)
umean(q) = sum(q[F3.iU(g3, i, j, k)] for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz) /
           sum(q[F3.iρ(g3, i, j, k)] * (g3.Vf[i, j, k] > 0) for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz)
tw = @elapsed for n in 1:nsteps
    K.ka_mis_step!(m, Q, Δt, tab, w; nsub, tol = 1e-8, smdiv = parse(Float64, get(ENV, "BUILD_SMDIV", "0.0")))
    if n % max(1, nsteps ÷ nout) == 0 || n == nsteps
        q = K.global_state(m, g3, Q)
        wv = F3.vertical_velocity(m3, q)
        uf = [g3.Ax[i, j, k] > 0 ? q[F3.iU(g3, i, j, k)] / (0.5 * (q[F3.iρ(g3, i, j, k)] + q[F3.iρ(g3, F3.ip(g3, i - 1), j, k)])) : 0.0
              for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz]
        wm, iw = findmax(abs, wv)
        um, iu = findmax(abs, uf)
        @printf("t = %6.1f s: max|w| %.3f m/s at %s (Vf %.2f), max|u| %.3f at %s (Ax %.2f, Vf %.2f|%.2f, mask %d), mean u %.3f m/s, GMRES its %.1f, Δρ %.1e ΔΘ %.1e\n",
                n * Δt, wm, Tuple(iw) .+ (0, 0, 1), g3.Vf[iw[1], iw[2], iw[3]], um, Tuple(iu), g3.Ax[iu],
                g3.Vf[F3.ip(g3, iu[1] - 1), iu[2], iu[3]], g3.Vf[iu], g3.mask[iu] || g3.mask[F3.ip(g3, iu[1] - 1), iu[2], iu[3]],
                umean(q), sum(m.its) / length(m.its), F3.total3d(m3, q, :ρ) / F3.total3d(m3, q0, :ρ) - 1,
                F3.total3d(m3, q, :Θ) / F3.total3d(m3, q0, :Θ) - 1)
        empty!(m.its); flush(stdout)
        any(isnan, q) && error("NaN at step $n")
    end
end
@printf("%d MIS steps (Δt %.2f s, nsub %d) in %.0f s: %.2f s per step\n", nsteps, Δt, nsub, tw, tw / nsteps)
q = K.global_state(m, g3, Q)
serialize(joinpath(out, "building_array_state$tag.bin"), (q, nsteps, Δt))
println("state: ", joinpath(out, "building_array_state$tag.bin"))
end

# ------------------------------------------------------------------ plots: u at z = 10 m, w in the vertical section y = 24 m
ONGPU && exit()
uc = [g3.Vf[i, j, k] > 0 ? 0.5 * (q[F3.iU(g3, i, j, k)] + q[F3.iU(g3, F3.ip(g3, i + 1), j, k)]) / q[F3.iρ(g3, i, j, k)] : NaN
      for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz]
wc = [g3.Vf[i, j, k] > 0 ? 0.5 * (F3.Wface(g3, q, i, j, k) + F3.Wface(g3, q, i, j, k + 1)) / q[F3.iρ(g3, i, j, k)] : NaN
      for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz]
k10 = F3.level(g3, 10.0); j24 = round(Int, 24.0 / Δx) + 1
fig = Figure(size = (1100, 480))
ax1 = Axis(fig[1, 1]; title = @sprintf("u at z = %.0f m, t = %.0f s", g3.zc[k10], nsteps * Δt), xlabel = "x [m]", ylabel = "y [m]", aspect = DataAspect())
hm = heatmap!(ax1, g3.xc, g3.yc, uc[:, :, k10]; colormap = :viridis, nan_color = :gray30)
Colorbar(fig[1, 2], hm; label = "u [m/s]")
kt = F3.level(g3, 60.0)
ax2 = Axis(fig[1, 3]; title = @sprintf("w at y = %.0f m", g3.yc[j24]), xlabel = "x [m]", ylabel = "z [m]")
lim = maximum(abs, filter(!isnan, wc[:, j24, 1:kt]))
hw = heatmap!(ax2, g3.xc, g3.zc[1:kt], wc[:, j24, 1:kt]; colormap = :balance, colorrange = (-lim, lim), nan_color = :gray30)
Colorbar(fig[1, 4], hw; label = "w [m/s]")
save(joinpath(out, "building_array.png"), fig)
println("plot: ", joinpath(out, "building_array.png"))
