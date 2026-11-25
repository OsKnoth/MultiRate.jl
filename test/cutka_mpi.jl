# MPI test of FV3DKA.CutKA (dry Cartesian cut cells, implicit = :amf, HEVI with GMRES) on x–y tiles: every rank builds
# the CPU model FV3D.Model3D on the whole (small) grid as reference and compares N(q) and three MIS54 steps of its tile,
# for both preconditioners (:columns, :amf) and both masked advection Jacobians.
#
#     julia --project=examples -e 'using MPI; run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project=examples test/cutka_mpi.jl 2 2`)'
#
# Arguments px py (default 2 2); the number of ranks must be px py. CUTKA_GPU=1: CUDA backend (one GPU per rank,
# GPU-aware MPI).

using MPI
MPI.Init()
const ONGPU = get(ENV, "CUTKA_GPU", "0") == "1"
ONGPU && @eval using CUDA
using MultiRate, Printf
using KernelAbstractions
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const K = MultiRate.FV3DKA
const CM = MultiRate.Common

px = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
py = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
dec = K.mpi_decomp(px, py)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
cx, cy = K.decomp_coords(dec)

gas = CM.Gas()
ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
F3.FV3D_PRESSURE_FORM[] = :product
hill(x, y) = 1500.0 * exp(-((x - 10e3)^2 + (y - 8e3)^2) / 3e3^2)
g3 = F3.Grid3D(12, 8, 6, 20e3, 16e3, 6e3; zs = hill)
tab = TS.mis_tableau(:MIS54)
fails = 0
for jf in (:partial, :conservative), prec in (:columns, :amf)
    F3.FV3D_MASK_JFORM[] = jf; F3.FV3D_MASK_SOLVER[] = :gmres; F3.FV3D_MASK_TOL[] = 1e-12; F3.FV3D_MASK_TILES[] = (2, 2)
    m3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.02)
    ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))
    q0 = F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k] * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)), 10ρa[k],
                                                              -3ρa[k], 0.0, Θr(g3.zc[k]))))
    be = ONGPU ? CUDABackend() : CPU()
    m = K.CutKA(K.CutGrid(g3; px, py, cx, cy, backend = be), dec; gas, sponge = (4e3, 0.02), jform = jf, prec)
    Q = K.ka_state(m, g3, q0)
    N3 = TS.rhs_nonlinear(m3, q0)
    Nk = K.allreduce_sum(dec, K.global_state(m, g3, K.ka_rhs!(K.ka_zeros(m), m, Q)))
    eN = maximum(abs, Nk - N3) / maximum(abs, N3)
    q3 = TS.run_frozen(m3, q0, 20.0, 3; tab, nsub = 4, fast = :hevi)
    Q = K.ka_state(m, g3, q0)
    K.ka_run!(m, Q, 20.0, 3; tab, nsub = 4, tol = 1e-12)
    qk = K.allreduce_sum(dec, K.global_state(m, g3, Q))
    eR = maximum(abs, qk - q3) / maximum(abs, q3 - q0)
    dρ = F3.total3d(m3, qk, :ρ) / F3.total3d(m3, q0, :ρ) - 1
    dΘ = F3.total3d(m3, qk, :Θ) / F3.total3d(m3, q0, :Θ) - 1
    its = sum(m.its) / length(m.its)
    ok = eN < 1e-12 && eR < 1e-9 && abs(dρ) < 1e-13 && abs(dΘ) < 1e-13
    global fails += !ok
    rank == 0 && @printf("%s %d × %d tiles, %-12s %-8s: N rel %.1e, 3 MIS54 steps rel %.1e, Δρ %.1e ΔΘ %.1e, GMRES its %.1f  %s\n",
                         ONGPU ? "GPU" : "CPU", px, py, jf, prec, eN, eR, dρ, dΘ, its, ok ? "ok" : "FAILED")
end
MPI.Barrier(MPI.COMM_WORLD)
rank == 0 && println(fails == 0 ? "all passed" : "$fails failed")
MPI.Finalize()
