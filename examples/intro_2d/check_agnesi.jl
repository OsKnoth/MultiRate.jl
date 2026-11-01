# Short terrain-following Agnesi smoke checks: DGSEM/HDG/HEVI and FV/HEVI.
using HDGEulerGravity
using LinearAlgebra

const TS = HDGEulerGravity.TimeStepping
const E2 = HDGEulerGravity.Euler2D
const FV = HDGEulerGravity.FV2D
const CM = HDGEulerGravity.Common

gas = CM.Gas()
ρr, Θr, _ = E2.stratified_reference(gas; θ0 = 300.0, N = 0.01)
a, L, H, xm = 10e3, 400e3, 10e3, 200e3
zs(x) = 100.0 / (1 + ((x - xm) / a)^2)
mesh = E2.Mesh2D(8, 3, 2, L, H; zs = zs)
model = E2.Euler2DModel(mesh, gas, ρr, Θr)
q0 = E2.project2d(model, (x, z) -> (ρr(z), 10ρr(z), 0.0, Θr(z)))
stepper = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = 4, fast = :hevi, refreeze_every = 2)
q1 = TS.run_mis(model, q0, 0.02, 2, stepper)
all(isfinite, q1) || error("DGSEM/HDG/HEVI Agnesi state is not finite")

println("Agnesi hill MIS/HEVI check passed for the DGSEM/HDG terrain grid")

fvgrid = FV.FVGrid(16, 6, L, H; zs = zs)
fvmodel = FV.FVModel(fvgrid, gas, ρr, Θr)
for i in 1:fvgrid.nx
    ir = mod1(i + 1, fvgrid.nx)
    metric_x = (fvgrid.Jx[ir] - fvgrid.Jx[i]) / fvgrid.dx
    metric_η = (fvgrid.zx_f[i,end] - fvgrid.zx_f[i,1]) / fvgrid.H
    abs(metric_x - metric_η) < 1e-12 || error("FV terrain metric identity failed in column $i")
end
qfv_rest = FV.project2d(fvmodel, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
rrest = FV.rhs_nonlinear(fvmodel, qfv_rest)
maximum(abs, rrest) < 1e-12 || error("FV terrain reference state is not well balanced")
qfv = FV.project2d(fvmodel, (x, z) -> (ρr(z), 10ρr(z), 0.0, Θr(z)))
rfv = FV.rhs_nonlinear(fvmodel, qfv)
abs(FV.total2d(fvmodel, rfv, 1)) < 1e-12 * FV.total2d(fvmodel, qfv, 1) || error("FV terrain mass is not conservative")
abs(FV.total2d(fvmodel, rfv, 4)) < 1e-12 * FV.total2d(fvmodel, qfv, 4) || error("FV terrain Θ is not conservative")
sv = FV.fast_operator(fvmodel, qfv; fast = :wrf, reuse = false)
Av = zeros(fvmodel.nq, fvmodel.nq)
e = zeros(fvmodel.nq); col = similar(e)
for j in eachindex(e)
    e[j] = 1.0
    FV.vertical_apply!(col, sv, e)
    Av[:, j] .= col
    e[j] = 0.0
end
c = 0.01
Ys = copy(qfv); Y = similar(Ys)
FV.vertical_solve!(Y, sv, FV.vertical_factor(sv, c), Ys)
norm(Y - (I - c * Av) \ Ys) < 1e-11 * norm(Y) || error("FV terrain WRF column solve disagrees with its vertical operator")
fvstepper = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = 12, fast = :hevi)
qfv1 = TS.run_mis(fvmodel, qfv, 0.02, 2, fvstepper)
all(isfinite, qfv1) || error("FV/WRF Agnesi state is not finite")
println("Agnesi hill MIS/HEVI check passed for the terrain-following FV grid")
