# Timing of FV3DKA.CutKA on px × py GPUs (one per MPI rank, GPU-aware MPI): the 1500 m bell hill on an nx × nx × 40 grid
# (Δx = 1 km), projected start, MIS54, Δt = 30 s, nsub = 14, HEVI with GMRES and the column preconditioner.
#
#     srun -n 4 gpu_wrapper.sh -n 4 -e "julia --project=gpuenv test/cutka_mgpu_bench.jl 2 2 128 10"

using MPI
MPI.Init()
using CUDA, MultiRate, Printf, LinearAlgebra, SparseArrays
using KernelAbstractions
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const K = MultiRate.FV3DKA
const CM = MultiRate.Common
include(joinpath(@__DIR__, "cutka_projection.jl"))

px, py, nx, nsteps = (parse(Int, a) for a in ARGS)
dec = K.mpi_decomp(px, py)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
cx, cy = K.decomp_coords(dec)
gas = CM.Gas()
ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
F3.FV3D_PRESSURE_FORM[] = :product
tab = TS.mis_tableau(:MIS54)
L, H = nx * 1e3, 10e3
bell(x, y) = 1500.0 / (1 + ((x - L / 2)^2 + (y - L / 2)^2) / 4e3^2)^1.5
gb = F3.Grid3D(nx, nx, 40, L, L, H; zs = bell)
mb = F3.with_sponge(F3.Model3D(gb, gas, ρr, Θr), zeros(F3.nunknowns(gb)); zD = 7e3, αtop = 0.02)
ρb = F3.hydrostatic_column(mb, Θr; ρbottom = ρr(gb.zc[1]))
qa = project_divfree(mb, F3.project3d(mb, (x, y, z) -> (k = F3.level(gb, z); (ρb[k], 10ρb[k], 0.0, 0.0, Θr(gb.zc[k])))))
for jf in (:partial, :conservative)
    m = K.CutKA(K.CutGrid(gb; px, py, cx, cy, backend = CUDABackend()), dec; gas, sponge = (7e3, 0.02), jform = jf)
    Q = K.ka_state(m, gb, qa)
    w = K.CutMISWork(m, TS.nstages(tab))
    K.ka_mis_step!(m, Q, 30.0, tab, w; nsub = 14, tol = 1e-8)
    CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD); empty!(m.its)
    t = @elapsed (for _ in 2:nsteps; K.ka_mis_step!(m, Q, 30.0, tab, w; nsub = 14, tol = 1e-8); end; CUDA.synchronize(); MPI.Barrier(MPI.COMM_WORLD))
    q = K.allreduce_sum(dec, K.global_state(m, gb, Q))
    rank == 0 && @printf("%d × %d GPUs, %d×%d×40, %-12s: %.2f s per MIS step, GMRES its %.1f, max|w| %.3f, Δρ %.1e ΔΘ %.1e\n",
                         px, py, nx, nx, jf, t / (nsteps - 1), sum(m.its) / length(m.its), maximum(abs, F3.vertical_velocity(mb, q)),
                         F3.total3d(mb, q, :ρ) / F3.total3d(mb, qa, :ρ) - 1, F3.total3d(mb, q, :Θ) / F3.total3d(mb, qa, :Θ) - 1)
    flush(stdout)
end
MPI.Finalize()
