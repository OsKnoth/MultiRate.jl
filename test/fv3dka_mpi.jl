# MPI test of FV3DKA (the moist 3D model with KernelAbstractions and x–y tiles): every rank builds the CPU model
# FV3D.MoistModel3D on the whole (small) grid as reference and compares N(q) and two MIS54 steps (:cs) of its tile.
#
#     julia --project=examples -e 'using MPI; run(`$(mpiexec()) -n 4 $(Base.julia_cmd()) --project=examples test/fv3dka_mpi.jl 2 2`)'
#
# Arguments px py (default 2 2); the number of ranks must be px py.

using MPI
MPI.Init()
using MultiRate, Printf
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const K = MultiRate.FV3DKA
const CM = MultiRate.Common

px = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
py = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2
dec = K.mpi_decomp(px, py)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
cx, cy = K.decomp_coords(dec)

mgθ = CM.MoistGas()
gas = CM.Gas(R = mgθ.Rd, γ = mgθ.Cpd / (mgθ.Cpd - mgθ.Rd), p00 = mgθ.p0)
ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
hill(x, y) = 1200.0 * exp(-((x - 10e3)^2 + (y - 7e3)^2) / 4e3^2)
nx, ny, nz, L, Ly, H = 12, 8, 6, 20e3, 16e3, 6e3
fails = 0
for terr in (false, true), (thv, pvar, rain) in ((:thetarho, :p, false), (:energy, :pi, false), (:thetarho, :p, true))
    g3 = terr ? F3.Grid3D(nx, ny, nz, L, Ly, H; zs = hill, terrain = :follow) : F3.Grid3D(nx, ny, nz, L, Ly, H)
    b3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.05)
    mg = CM.MoistGas(thvar = thv)
    mθ = F3.MoistModel3D(b3; mgas = mgθ, pvar, rain); m3 = F3.MoistModel3D(b3; mgas = mg, pvar, rain)
    fq(x, y, z) = (ρ = ρr(z) * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)); (ρ, 8ρ + sin(y / 3e3), 5ρ + cos(x / 4e3), 0.2ρ * sin(π * z / H) * cos((x + 2y) / 5e3), Θr(z) * (1 + 1e-3 * cos((x - y) / 3e3))))
    q = F3.project3d(b3, fq)
    x = F3.thermo_convert(mθ, m3, F3.moist_exner_state(mθ, q, [0.013 * q[F3.iρ(g3, i, j, k)] for i in 1:nx, j in 1:ny, k in 1:nz]))
    for k in 1:nz, j in 1:ny, i in 1:nx
        c = x[F3.iqc(m3, i, j, k)]; x[F3.iqv(m3, i, j, k)] += 0.3c; x[F3.iqc(m3, i, j, k)] -= 0.3c
        rain && (x[F3.iqr(m3, i, j, k)] = 2e-3 * x[F3.iρ(g3, i, j, k)] * (1 + 0.5 * sin(i + 2j + k)))   # warm rain
    end
    TS.frozen_step_end!(m3, x)
    gk = K.KAGrid(nx, ny, nz, L, Ly, H; zs = terr ? hill : nothing, px, py, cx, cy)
    mk = K.MoistKA(gk, dec; mgas = mg, pvar, sponge = (4e3, 0.05), rain)
    Q = K.ka_state(mk, m3, x)
    N3 = TS.rhs_nonlinear(m3, x)
    out = K.ka_zeros(mk); K.ka_rhs!(out, mk, Q)
    Nk = K.allreduce_sum(dec, K.global_state(mk, m3, out))
    eN = maximum(abs, Nk - N3) / maximum(abs, N3)
    y3 = TS.run_frozen(m3, x, 3.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6, fast = :cs)
    K.ka_run!(mk, Q, 3.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6)
    yk = K.allreduce_sum(dec, K.global_state(mk, m3, Q))
    eR = maximum(abs, yk - y3) / maximum(abs, y3 - x)
    mass = K.ka_total(mk, Q, K.VR) / F3.total3d(m3, y3, :ρ) - 1     # (rain leaves through the ground)
    ok = eN < 1e-9 && eR < 1e-8 && abs(mass) < 1e-13
    global fails += !ok
    rank == 0 && @printf("%d × %d tiles, terrain=%-5s %-8s %-2s%s: N rel %.1e, 2 MIS54 steps rel %.1e, Δmass %.1e  %s\n",
                         px, py, terr, thv, pvar, rain ? " rain" : "", eN, eR, mass, ok ? "ok" : "FAILED")
end
MPI.Barrier(MPI.COMM_WORLD)
rank == 0 && println(fails == 0 ? "all passed" : "$fails failed")
MPI.Finalize()
exit(fails == 0 ? 0 : 1)
