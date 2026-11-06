# Short terrain-following and cut-cell Agnesi smoke checks, including DGSEM/HDG and FV/HEVI.
using MultiRate
using LinearAlgebra

const TS = MultiRate.TimeStepping
const E2 = MultiRate.Euler2D
const FV = MultiRate.FV2D
const CM = MultiRate.Common

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
FV.FV_HYDRO_REFERENCE[] = true
fvref = FV.FVModel(fvgrid, gas, ρr, Θr)
FV.FV_HYDRO_REFERENCE[] = false
maximum(abs, FV.rhs_nonlinear(fvref, qfv_rest)) < 1e-12 || error("FV terrain reference state is not well balanced")
# background-free: the analytic hydrostatic state is balanced to second order (small, not zero)
maximum(abs, FV.rhs_nonlinear(fvmodel, qfv_rest)) < 0.05 || error("FV terrain background-free rest state imbalance too large")
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

gcut = FV.CutCellFVGrid(40, 8, L, H; zs = zs)
mcut = FV.FVModel(gcut, gas, ρr, Θr)
FV.FV_HYDRO_REFERENCE[] = true
mcutref = FV.FVModel(gcut, gas, ρr, Θr)
qcut_ref = FV.project2d(mcutref, (x,z) -> (ρr(z), 0.0, 0.0, Θr(z)))
FV.FV_HYDRO_REFERENCE[] = false
maximum(abs, FV.rhs_nonlinear(mcutref, qcut_ref)) < 1e-12 || error("cut-cell FV reference rest state is not balanced")
qcut_rest = FV.project2d(mcut, (x,z) -> (ρr(z), 0.0, 0.0, Θr(z)))
maximum(abs, FV.rhs_nonlinear(mcut, qcut_rest)) < 0.05 || error("cut-cell FV background-free rest state imbalance too large")
qcut = FV.project2d(mcut, (x,z) -> (ρr(z), 10ρr(z), 0.0, Θr(z)))
rcut = FV.rhs_nonlinear(mcut, qcut)
abs(FV.total2d(mcut, rcut, 1)) < 1e-12 * FV.total2d(mcut, qcut, 1) || error("cut-cell mass tendency is not conservative")
abs(FV.total2d(mcut, rcut, 4)) < 1e-12 * FV.total2d(mcut, qcut, 4) || error("cut-cell Θ tendency is not conservative")
scut = FV.fast_operator(mcut, qcut; fast = :hevi, reuse = false)
Acut = zeros(mcut.nq, mcut.nq)
ec = zeros(mcut.nq); colc = similar(ec)
for j in eachindex(ec)
    ec[j] = 1.0
    FV.vertical_apply!(colc, scut, ec)
    Acut[:,j] .= colc
    ec[j] = 0.0
end
ccut = 0.01
Yscut = copy(qcut); Ycut = similar(Yscut)
FV.vertical_solve!(Ycut, scut, FV.vertical_factor(scut, ccut), Yscut)
norm(Ycut - (I - ccut*Acut) \ Yscut) < 1e-9 * norm(Ycut) || error("cut-cell HEVI column solve disagrees with its vertical operator")
cut_nsub = 8
cut_hmin = minimum(2gcut.Vol[i,k] / (gcut.FU[i,k] + gcut.FU[i+1,k] + gcut.FW[i,k] + gcut.FW[i,k+1])
                    for i in 1:gcut.nx, k in 1:gcut.nz if gcut.Vol[i,k] > 0)
cut_dt_hevi = min(0.02, cut_nsub * 0.2 * 0.85 * 0.92 * cut_hmin / FV.sound_speed_max(mcut,qcut))
cut_hevi = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = cut_nsub, fast = :hevi)
qcut_hevi = TS.run_mis(mcut, qcut, cut_dt_hevi, 2, cut_hevi)
all(isfinite, qcut_hevi) || error("cut-cell FV/HEVI Agnesi state is not finite at its small-cell step")
cut_amf = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = cut_nsub, fast = :rosenbrock_amf)
qcut_amf = TS.run_mis(mcut, qcut, 0.02, 2, cut_amf)
all(isfinite, qcut_amf) || error("cut-cell FV/Rosenbrock-W AMF Agnesi state is not finite")
abs(FV.total2d(mcut,qcut_amf,1)-FV.total2d(mcut,qcut,1)) < 1e-10*FV.total2d(mcut,qcut,1) ||
    error("cut-cell FV/Rosenbrock-W AMF lost mass")
println("Agnesi cut-cell geometry, conservation, column solve, small-step HEVI and large-step AMF MIS checks passed")
