# MPI agglomeration of the acoustic multigrid (CutMG aggcols): px × py ranks against the serial run, cut-cell buildings
# case, Ros2AMF + multigrid; aggcols = 0 (no agglomeration), 64 (after two tile levels), 4096 (at once).
#
#     julia --project=examples -e 'using MPI; run(`$(mpiexec()) -n 4 julia --project=examples test/cutka_mgagg_mpi.jl 2 2`)'
using MPI; MPI.Init()
using MultiRate, Printf, LinearAlgebra, SparseArrays
const TS = MultiRate.TimeStepping; const F3 = MultiRate.FV3D; const K = MultiRate.FV3DKA; const CM = MultiRate.Common; const BD = MultiRate.Buildings
include(joinpath(@__DIR__, "cutka_projection.jl"))
px, py = parse(Int, ARGS[1]), parse(Int, ARGS[2])
rank = MPI.Comm_rank(MPI.COMM_WORLD)
gas = CM.Gas(); ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
F3.FV3D_PRESSURE_FORM[] = :product; F3.FV3D_MASK_JFORM[] = :partial
function block(n, xc, yc, h, α; a = 16.0)
    c, s = cos(α), sin(α)
    BD.BuildingPolygon(n, [[(xc + c * u - s * v, yc + s * u + c * v, h) for (u, v) in ((-a/2, -a/2), (a/2, -a/2), (a/2, a/2), (-a/2, a/2))]])
end
polys = [block(1, 30.0, 32.0, 16.0, 0.0), block(2, 80.0, 30.0, 24.0, π / 6), block(3, 100.0, 70.0, 12.0, 0.3)]
zf = BD.stretched_levels(120.0; dz0 = 4.0, z1 = 32.0, r = 1.1)
φ = BD.building_levelset(polys; origin = (0.0, 0.0), L = (128.0, 128.0), z0 = 0.0)
cg = BD.cut_geometry(φ; Δx = 4.0, zf, nsub = (8, 8), δsub = 0.5)
BD.remove_small_cells!(cg; vmin = 0.1, amin = 0.1)
g3 = BD.fv3d_grid(cg; limiter = :koren)
m3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 80.0, αtop = 0.05)
ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))
q0 = project_divfree(m3, F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k], 5ρa[k], 1ρa[k], 0.0, Θr(g3.zc[k])))))
tab = TS.ros_tableau(:Ros2AMF)
dec = K.mpi_decomp(px, py); cx, cy = K.decomp_coords(dec)
function run(dec, grid, aggcols)
    m = K.CutKA(grid, dec; gas, sponge = (80.0, 0.05), jform = :partial)
    Q = K.ka_state(m, g3, q0)
    w = K.CutRos(m, tab.s; aprec = :mg, aggcols)
    for _ in 1:3; K.ka_ros_step!(m, Q, 1.0, tab, w; tol = 1e-10, ros = :amf); end
    q = dec isa K.SerialDecomp ? K.global_state(m, g3, Q) : K.allreduce_sum(dec, K.global_state(m, g3, Q))
    return q, w.its[1:2:end], m
end
qs, its_s, _ = run(K.SerialDecomp(), K.CutGrid(g3), 0)
for aggcols in (0, 64, 4096)
    q, its, m = run(dec, K.CutGrid(g3; px, py, cx, cy), aggcols)
    rank == 0 && @printf("%d × %d ranks, aggcols %4d: acoustic its %s (serial %s), max diff to serial %.1e\n", px, py, aggcols,
                         string(its), string(its_s), maximum(abs, q - qs) / maximum(abs, qs - q0))
    aggcols > 0 && (its != its_s || maximum(abs, q - qs) > 1e-12 * maximum(abs, qs - q0)) && error("agglomeration differs from serial")
end
MPI.Finalize()
