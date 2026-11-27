# Flow along the Eisenbahnstraße in Leipzig (LoD2 CityGML of the city, EPSG:25833): the cut-cell model FV3DKA.CutKA
# on the geometry of MultiRate.Buildings (ASAM level set → cut-cell fractions), Rosenbrock Ros2AMF with the acoustic
# multigrid (or MIS, as in examples/building_array.jl).
#
# Domain 2304 m × 512 m × 300 m around the street (x from the west end, house numbers 1 … 189), Δx = Δy = 4 m
# (576 × 128 columns, divisible by 2⁶ for the multigrid), levels 4 m up to 32 m, stretched (× 1.1) to 300 m, sponge
# above 200 m. Periodic lateral boundaries; the buildings inside [8 m, Lx - LEIP_NUDGE - 8 m] × [8 m, Ly - 8 m], a
# building-free nudging strip at the east end (slow force towards u = LEIP_U, θ_ref(z)). Terrain flattened: every
# building stands on z = 0 with its height above its own ground (lowest vertex), the ground of the city rises by
# about 11 m along the street. Westerly wind LEIP_U = 5 m/s along the street, stable stratification N = 0.01 1/s.
#
#     julia -t 16 --project=examples examples/leipzig_eisenbahnstrasse.jl geometry      (writes out/leipzig_grid.bin)
#     (cut cells with Vf < LEIP_VMIN = 0.1 removed, faces with A < LEIP_AMIN = 0.1 closed: real roofs and walls
#     give cut cells down to Vf ~ 1e-6 and slits between buildings with A ~ 1e-3)
#     BUILDINGS_GPU=1 julia --project=gpuenv examples/leipzig_eisenbahnstrasse.jl [T = 600] [Δt = 0.5] [nout = 12]
#     LEIP_PX=2 LEIP_PY=2 BUILDINGS_GPU=1 srun -n 4 gpu_wrapper.sh -n 4 -e "julia --project=gpuenv examples/leipzig_eisenbahnstrasse.jl …"
#         (px × py GPUs, one per MPI rank, GPU-aware MPI; the tiles must be even for the multigrid)
#     LEIP_ADPREC = none | gs | jacobi, LEIP_SWEEPS = 2: the preconditioner of the advection factor (CutRos adprec)
#     julia --project=examples examples/leipzig_eisenbahnstrasse.jl plot examples/out/leipzig_state.bin
#
# LEIP_DIR: the directory of the CityGML files (default ~/CityLeipzig).

using MultiRate, Printf, LinearAlgebra, SparseArrays, Serialization, KernelAbstractions
const ONGPU = get(ENV, "BUILDINGS_GPU", "0") == "1"
ONGPU && @eval using CUDA
const PX, PY = parse(Int, get(ENV, "LEIP_PX", "1")), parse(Int, get(ENV, "LEIP_PY", "1"))
const PAR = PX * PY > 1
PAR && @eval (using MPI; MPI.Init())
const MODE = length(ARGS) >= 1 && ARGS[1] in ("geometry", "plot") ? ARGS[1] : "run"
ONGPU || PAR || MODE == "geometry" || @eval using CairoMakie
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const K = MultiRate.FV3DKA
const CM = MultiRate.Common
const BD = MultiRate.Buildings
include(joinpath(@__DIR__, "..", "test", "cutka_projection.jl"))
out = joinpath(@__DIR__, "out"); mkpath(out)
gridfile = joinpath(out, "leipzig_grid.bin")

# ------------------------------------------------------------------ geometry
const X0, Y0 = 318533.0, 5691170.0                                  # south-west corner (EPSG:25833)
const Lx, Ly, Δx = 2304.0, 512.0, 4.0
const LN = parse(Float64, get(ENV, "LEIP_NUDGE", "96"))             # nudging strip Lx - LN … Lx
zf = BD.stretched_levels(300.0; dz0 = 4.0, z1 = 32.0, r = 1.1)

"""The buildings inside the domain (all of their vertices within the margins), each on z = 0."""
function street_buildings(dir)
    files = filter(f -> any(occursin(s, f) for s in ("_01_", "_10_", "_11_", "_20_", "_21_", "_22_", "_23_", "_30_")),
                   readdir(dir; join = true))
    t = @elapsed polys = BD.read_citygml(files)
    byb = Dict{Int,Vector{BD.BuildingPolygon}}()
    for P in polys
        push!(get!(byb, P.building, BD.BuildingPolygon[]), P)
    end
    keep = BD.BuildingPolygon[]
    nb = 0; hs = Float64[]
    for (b, Ps) in byb
        pts = [p for P in Ps for r in P.rings for p in r]
        xmin, xmax = extrema(p[1] - X0 for p in pts); ymin, ymax = extrema(p[2] - Y0 for p in pts)
        (xmin >= 8 && xmax <= Lx - LN - 8 && ymin >= 8 && ymax <= Ly - 8) || continue
        zg, zt = extrema(p[3] for p in pts)
        nb += 1; push!(hs, zt - zg)
        for P in Ps
            push!(keep, BD.BuildingPolygon(nb, [[(p[1], p[2], p[3] - zg) for p in r] for r in P.rings]))
        end
    end
    @printf("CityGML: %d files, %d polygons in %.1f s; %d buildings in the domain, heights %.1f … %.1f m (median %.1f)\n",
            length(files), length(polys), t, nb, minimum(hs), maximum(hs), sort(hs)[length(hs) ÷ 2 + 1])
    return keep
end

if MODE == "geometry"
    polys = street_buildings(expanduser(get(ENV, "LEIP_DIR", "~/CityLeipzig")))
    t = @elapsed φ = BD.building_levelset(polys; origin = (X0, Y0), L = (Lx, Ly), z0 = 0.0)
    @printf("level set: %d solids, highest roof %.1f m, %.1f s\n", length(φ.solids), φ.hmax, t); flush(stdout)
    t = @elapsed cg = BD.cut_geometry(φ; Δx, zf, nsub = (8, 8), δsub = 0.5)
    vmin = parse(Float64, get(ENV, "LEIP_VMIN", "0.1"))
    amin = parse(Float64, get(ENV, "LEIP_AMIN", "0.1"))
    nr, vr, nf = BD.remove_small_cells!(cg; vmin, amin)
    @printf("small cells (Vf < %.2f) removed: %d, %.1e of the fluid volume; small faces (A < %.2f) closed: %d\n", vmin, nr, vr, amin, nf)
    g3 = BD.fv3d_grid(cg; limiter = :koren)
    @printf("grid %d × %d × %d (dz %.0f … %.1f m), %d cut cells, %d solid, %d masked, geometry %.0f s\n", g3.nx, g3.ny, g3.nz,
            minimum(g3.dzc), maximum(g3.dzc), count(0 .< g3.Vf .< 1), count(g3.Vf .== 0), count(g3.mask), t)
    serialize(gridfile, g3)
    println("grid: ", gridfile)
    exit()
end

g3 = deserialize(gridfile)
const RANK = PAR ? MPI.Comm_rank(MPI.COMM_WORLD) : 0
macro root(ex); :(RANK == 0 && $(esc(ex))); end
U0 = parse(Float64, get(ENV, "LEIP_U", "5.0"))
T = MODE == "run" && length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 600.0
Δt = MODE == "run" && length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
nout = MODE == "run" && length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 12
scheme = get(ENV, "LEIP_SCHEME", "ros")                              # ros (Ros2AMF + MG) | mis (ConvexMIS42, masked AMF)
adprec = Symbol(get(ENV, "LEIP_ADPREC", "none")); sweeps = parse(Int, get(ENV, "LEIP_SWEEPS", "2"))
rtol = parse(Float64, get(ENV, "LEIP_TOL", "1e-3"))                 # relative tolerance (drop) of the stage solves
enorm = get(ENV, "LEIP_NORM", "energy") == "energy"                 # energy: residuals in Pa (ρ c², Θ γR, m c)
atolp = parse(Float64, get(ENV, "LEIP_ATOL", "1e-4"))                  # absolute RMS residual / dynamic pressure ρU²/2
tag = get(ENV, "LEIP_TAG", "")
@root @printf("grid %d × %d × %d, %d cut cells, %d masked; U %.1f m/s, Δt %.2f s, c Δt/Δx %.1f, U Δt/Δx %.2f, scheme %s, %d × %d %s, advection preconditioner %s (%d sweeps), tol %.0e, norm %s, atol %.0e ρU²/2\n",
        g3.nx, g3.ny, g3.nz, count(0 .< g3.Vf .< 1), count(g3.mask), U0, Δt, 347Δt / Δx, U0 * Δt / Δx, scheme, PX, PY,
        ONGPU ? "GPUs" : "CPU tiles", adprec, sweeps, rtol, enorm ? "energy" : "euclid", atolp)
flush(stdout)

# ------------------------------------------------------------------ model and initial state
gas = CM.Gas()
ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
F3.FV3D_PRESSURE_FORM[] = :product
F3.FV3D_MASK_JFORM[] = :partial
F3.FV3D_HEVI_BETA[] = 0.5
m3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 200.0, αtop = 0.05)
nud = (x0 = Lx - LN, x1 = Lx, αmax = 0.2, u = U0, v = 0.0, θ = θr)
m3 = F3.with_nudging(m3; nud...)
ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))

if MODE == "plot"
    q, nsteps, Δt = deserialize(ARGS[2])
else
# the divergence-free start (PCG on the host, cached next to the grid: computed once by rank 0)
q0file = joinpath(out, "leipzig_q0_$(U0).bin")
if RANK == 0 && !(isfile(q0file) && mtime(q0file) > mtime(gridfile))
    t = @elapsed q0 = project_divfree(m3, F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k], U0 * ρa[k], 0.0, 0.0, Θr(g3.zc[k])))))
    serialize(q0file, q0)
    @printf("divergence-free start: %.0f s\n", t); flush(stdout)
end
PAR && MPI.Barrier(MPI.COMM_WORLD)
q0 = deserialize(q0file)
backend = ONGPU ? CUDABackend() : KernelAbstractions.CPU()
if PAR
    dec = K.mpi_decomp(PX, PY); cx, cy = K.decomp_coords(dec)
    cgrid = K.CutGrid(g3; px = PX, py = PY, cx, cy, backend)
else
    cgrid = K.CutGrid(g3; backend)
end
kw = (gas, sponge = (200.0, 0.05), nudge = nud, jform = :partial, vertical = scheme == "ros" ? :implicit : :explicit,
      prec = scheme == "ros" ? :columns : :mask,
      mask = scheme == "ros" ? (;) : (acoustic = :cg, advection = :gs, sweeps = 1, outer = :none, momenta = :krylov))
m = PAR ? K.CutKA(cgrid, dec; kw...) : K.CutKA(cgrid; kw...)
Q = K.ka_state(m, g3, q0)
tabr = TS.ros_tableau(:Ros2AMF)
wr = scheme == "ros" ? K.CutRos(m, tabr.s; aprec = :mg, adprec, sweeps, aggcols = parse(Int, get(ENV, "LEIP_AGG", "4096"))) : nothing
scheme == "ros" && enorm && K.ros_energy_norm!(wr)
atol = atolp * 0.5 * ρa[1] * U0^2
gstate(Q) = PAR ? K.allreduce_sum(dec, K.global_state(m, g3, Q)) : K.global_state(m, g3, Q)
tab = TS.mis_tableau(:ConvexMIS42)
w = scheme == "ros" ? nothing : K.CutMISWork(m, TS.nstages(tab))
nsub = parse(Int, get(ENV, "LEIP_NSUB", "20"))

# ------------------------------------------------------------------ run
nsteps = round(Int, T / Δt); t0 = time()
tw = @elapsed for n in 1:nsteps
    if scheme == "ros"
        K.ka_ros_step!(m, Q, Δt, tabr, wr; tol = rtol, atol, ros = :amf)
    else
        K.ka_mis_step!(m, Q, Δt, tab, w; nsub, tol = 1e-8)
    end
    if n % max(1, nsteps ÷ nout) == 0 || n == nsteps
        q = gstate(Q)
        if RANK == 0
        wv = F3.vertical_velocity(m3, q)
        wm, iw = findmax(abs, wv)
        k10 = F3.level(g3, 10.0)
        u10 = [q[F3.iU(g3, i, j, k10)] / q[F3.iρ(g3, i, j, k10)] for i in 1:g3.nx, j in 1:g3.ny if g3.Vf[i, j, k10] > 0]
        @printf("t = %6.1f s: max|w| %.2f m/s at %s, u(10 m) mean %.2f max %.2f m/s, its %.1f, Δρ %.1e ΔΘ %.1e, %.1f s\n",
                n * Δt, wm, Tuple(iw), sum(u10) / length(u10), maximum(u10),
                scheme == "ros" ? sum(wr.its) / max(1, length(wr.its)) : sum(m.its) / max(1, length(m.its)),
                F3.total3d(m3, q, :ρ) / F3.total3d(m3, q0, :ρ) - 1, F3.total3d(m3, q, :Θ) / F3.total3d(m3, q0, :Θ) - 1, time() - t0)
        if scheme == "ros" && !isempty(wr.its)
            ia, id = wr.its[1:2:end], wr.its[2:2:end]
            @printf("              GMRES its per solve: acoustic %.1f (%d … %d), advection %.1f (%d … %d); wall s: setup %.1f, N(Y) %.1f, acoustic %.1f, advection %.1f\n",
                    sum(ia) / length(ia), extrema(ia)..., sum(id) / length(id), extrema(id)..., wr.tsec...)
        end
        any(isnan, q) && error("NaN at step $n")
        serialize(joinpath(out, "leipzig_state$tag.bin"), (q, n, Δt))
        flush(stdout)
        end
        scheme == "ros" ? empty!(wr.its) : empty!(m.its)
    end
end
if scheme == "ros"
    tot = sum(wr.tsec)
    @root @printf("%d steps (Δt %.2f s) in %.0f s: %.2f s per step, %.2f s per simulated second; setup %.0f %%, N(Y) %.0f %%, acoustic %.0f %%, advection %.0f %% of the solver time\n",
                  nsteps, Δt, tw, tw / nsteps, tw / (nsteps * Δt), (100 .* wr.tsec ./ tot)...)
else
    @root @printf("%d steps (Δt %.2f s) in %.0f s: %.2f s per step, %.2f s per simulated second\n", nsteps, Δt, tw, tw / nsteps, tw / (nsteps * Δt))
end
@root println("state: ", joinpath(out, "leipzig_state$tag.bin"))
PAR && MPI.Finalize()
end

# ------------------------------------------------------------------ plots: wind speed at z = 10 m, w in a section along the street
(ONGPU || PAR) && exit()
uc = [g3.Vf[i, j, k] > 0 ? 0.5 * (q[F3.iU(g3, i, j, k)] + q[F3.iU(g3, F3.ip(g3, i + 1), j, k)]) / q[F3.iρ(g3, i, j, k)] : NaN
      for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz]
vc = [g3.Vf[i, j, k] > 0 ? 0.5 * (q[F3.iV(g3, i, j, k)] + q[F3.iV(g3, i, F3.jp(g3, j + 1), k)]) / q[F3.iρ(g3, i, j, k)] : NaN
      for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz]
wc = [g3.Vf[i, j, k] > 0 ? 0.5 * (F3.Wface(g3, q, i, j, k) + F3.Wface(g3, q, i, j, k + 1)) / q[F3.iρ(g3, i, j, k)] : NaN
      for i in 1:g3.nx, j in 1:g3.ny, k in 1:g3.nz]
k10 = F3.level(g3, 10.0)
sp = @. sqrt(uc[:, :, k10]^2 + vc[:, :, k10]^2)
# the street: the row of columns with the most fluid at 10 m in the middle third
jmid = argmax([count(!isnan, sp[:, j]) for j in 1:g3.ny] .* [abs(j - g3.ny / 2) < g3.ny / 6 for j in 1:g3.ny])
fig = Figure(size = (1600, 900))
ax1 = Axis(fig[1, 1]; title = @sprintf("wind speed at z = %.0f m, t = %.0f s", g3.zc[k10], nsteps * Δt), xlabel = "x [m]", ylabel = "y [m]",
           aspect = DataAspect())
hm = heatmap!(ax1, g3.xc, g3.yc, sp; colormap = :viridis, nan_color = :gray30)
hlines!(ax1, [g3.yc[jmid]]; color = :white, linestyle = :dash)
Colorbar(fig[1, 2], hm; label = "|u| [m/s]")
kt = F3.level(g3, 80.0)
ax2 = Axis(fig[2, 1]; title = @sprintf("w at y = %.0f m", g3.yc[jmid]), xlabel = "x [m]", ylabel = "z [m]")
lim = maximum(abs, filter(!isnan, wc[:, jmid, 1:kt]))
hw = heatmap!(ax2, g3.xc, g3.zc[1:kt], wc[:, jmid, 1:kt]; colormap = :balance, colorrange = (-lim, lim), nan_color = :gray30)
Colorbar(fig[2, 2], hw; label = "w [m/s]")
rowsize!(fig.layout, 2, Relative(0.3))
save(joinpath(out, "leipzig_eisenbahnstrasse.png"), fig)
println("plot: ", joinpath(out, "leipzig_eisenbahnstrasse.png"))
