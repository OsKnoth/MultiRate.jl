# FV3DKA.CutKA on one GPU (CUDA): (1) the small bell hill on the GPU against the CPU (KernelAbstractions CPU backend)
# and FV3D (N(q), G_v(v), three MIS54 steps); (2) timing on the 1500 m bell hill 64×64×40 (projected start, MIS54,
# Δt = 30 s, nsub = 14, HEVI with GMRES and the column preconditioner).
#
#     julia --project=gpuenv test/cutka_gpu.jl [nsteps] [nx]

using CUDA, MultiRate, Printf, LinearAlgebra, SparseArrays
using KernelAbstractions
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const K = MultiRate.FV3DKA
const CM = MultiRate.Common
gpu = CUDABackend()
println(CUDA.name(CUDA.device())); flush(stdout)

gas = CM.Gas()
ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
F3.FV3D_PRESSURE_FORM[] = :product
tab = TS.mis_tableau(:MIS54)

# ------------------------------------------------------------------ (1) small hill: GPU = CPU = FV3D
hill(x, y) = 1500.0 * exp(-((x - 10e3)^2 + (y - 8e3)^2) / 3e3^2)
g3 = F3.Grid3D(12, 8, 6, 20e3, 16e3, 6e3; zs = hill)
for jf in (:partial, :conservative)
    F3.FV3D_MASK_JFORM[] = jf; F3.FV3D_MASK_SOLVER[] = :gmres; F3.FV3D_MASK_TOL[] = 1e-12; F3.FV3D_MASK_TILES[] = (2, 2)
    m3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.02)
    ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))
    q0 = F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k] * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)), 10ρa[k],
                                                              -3ρa[k], 0.0, Θr(g3.zc[k]))))
    mg = K.CutKA(K.CutGrid(g3; backend = gpu); gas, sponge = (4e3, 0.02), jform = jf)
    mc = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), jform = jf)
    N3 = TS.rhs_nonlinear(m3, q0)
    Ng = K.global_state(mg, g3, K.ka_rhs!(K.ka_zeros(mg), mg, K.ka_state(mg, g3, q0)))
    Nc = K.global_state(mc, g3, K.ka_rhs!(K.ka_zeros(mc), mc, K.ka_state(mc, g3, q0)))
    q3 = TS.run_frozen(m3, q0, 20.0, 3; tab, nsub = 4, fast = :hevi)
    Qg = K.ka_state(mg, g3, q0); K.ka_run!(mg, Qg, 20.0, 3; tab, nsub = 4, tol = 1e-12)
    Qc = K.ka_state(mc, g3, q0); K.ka_run!(mc, Qc, 20.0, 3; tab, nsub = 4, tol = 1e-12)
    qg, qc = K.global_state(mg, g3, Qg), K.global_state(mc, g3, Qc)
    @printf("%-12s N: |GPU - FV3D| %.1e |GPU - CPU| %.1e; 3 MIS54 steps: |GPU - FV3D| %.1e |GPU - CPU| %.1e; GMRES its %.1f / %.1f\n",
            jf, maximum(abs, Ng - N3) / maximum(abs, N3), maximum(abs, Ng - Nc) / maximum(abs, N3),
            maximum(abs, qg - q3) / maximum(abs, q3 - q0), maximum(abs, qg - qc) / maximum(abs, q3 - q0),
            sum(mg.its) / length(mg.its), sum(mc.its) / length(mc.its))
    flush(stdout)
end

# ------------------------------------------------------------------ (2) timing on the 1500 m bell hill
include(joinpath(@__DIR__, "cutka_projection.jl"))

nsteps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
nx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 64
L, H = nx * 1e3, 10e3
bell(x, y) = 1500.0 / (1 + ((x - L / 2)^2 + (y - L / 2)^2) / 4e3^2)^1.5
gb = F3.Grid3D(nx, nx, 40, L, L, H; zs = bell)
F3.FV3D_MASK_JFORM[] = :partial
mb = F3.with_sponge(F3.Model3D(gb, gas, ρr, Θr), zeros(F3.nunknowns(gb)); zD = 7e3, αtop = 0.02)
ρb = F3.hydrostatic_column(mb, Θr; ρbottom = ρr(gb.zc[1]))
qa = project_divfree(mb, F3.project3d(mb, (x, y, z) -> (k = F3.level(gb, z); (ρb[k], 10ρb[k], 0.0, 0.0, Θr(gb.zc[k])))))
for jf in (:partial, :conservative)
    m = K.CutKA(K.CutGrid(gb; backend = gpu); gas, sponge = (7e3, 0.02), jform = jf)
    Q = K.ka_state(m, gb, qa)
    w = K.CutMISWork(m, TS.nstages(tab))
    K.ka_mis_step!(m, Q, 30.0, tab, w; nsub = 14, tol = 1e-8)              # compile
    CUDA.synchronize(); empty!(m.its)
    t = @elapsed (for _ in 2:nsteps; K.ka_mis_step!(m, Q, 30.0, tab, w; nsub = 14, tol = 1e-8); end; CUDA.synchronize())
    q = K.global_state(m, gb, Q)
    @printf("bell hill %d×%d×40, %-12s: %.2f s per MIS step (Δt 30 s) on the GPU, GMRES its %.1f, max|w| %.3f after %d steps, Δρ %.1e ΔΘ %.1e\n",
            nx, nx, jf, t / (nsteps - 1), sum(m.its) / length(m.its), maximum(abs, F3.vertical_velocity(mb, q)), nsteps,
            F3.total3d(mb, q, :ρ) / F3.total3d(mb, qa, :ρ) - 1, F3.total3d(mb, q, :Θ) / F3.total3d(mb, qa, :Θ) - 1)
    flush(stdout)
end
