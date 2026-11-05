# Short warm-bubble smoke check for the DGSEM/HDG and finite-volume MIS paths.
using MultiRate

const TS = MultiRate.TimeStepping
const E2 = MultiRate.Euler2D
const FV = MultiRate.FV2D
const CM = MultiRate.Common

gas = CM.Gas()
ρr, Θr, θr = E2.stratified_reference(gas; θ0 = 300.0, N = 0.01)

function bubble_state(model, project)
    project(model, (x, z) -> begin
        θ = θr(z) + 2.0 * exp(-((x - 5e3) / 2e3)^2 - ((z - 2e3) / 2e3)^2)
        ρ = Θr(z) / θ
        (ρ, 0.0, 0.0, Θr(z))
    end)
end

dg = E2.Euler2DModel(E2.Mesh2D(5, 3, 2, 10e3, 6e3), gas, ρr, Θr)
q_dg = bubble_state(dg, E2.project2d)
dg_stepper = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = 4, fast = :hevi)
q_dg_end = TS.run_mis(dg, q_dg, 0.05, 2, dg_stepper)
all(isfinite, q_dg_end) || error("DGSEM/HDG/HEVI bubble state is not finite")
dg_relaxed = E2.Euler2DModel(E2.Mesh2D(5, 3, 2, 10e3, 6e3), gas, ρr, Θr)
dg_relaxed_stepper = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = 4, fast = :hevi_relax,
                                   fast_options = (; eps_if = 0.0))
q_dg_relaxed0 = TS.run_mis(dg_relaxed, q_dg, 0.05, 2, dg_relaxed_stepper)
maximum(abs, q_dg_relaxed0 .- q_dg_end) < 1e-12 || error(":hevi_relax with eps_if=0 must match :hevi")
dg_relax_positive = E2.Euler2DModel(E2.Mesh2D(5, 3, 2, 10e3, 6e3), gas, ρr, Θr)
dg_relax_positive_stepper = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = 4, fast = :hevi_relax,
                                          fast_options = (; eps_if = 0.1))
q_dg_relaxed = TS.run_mis(dg_relax_positive, q_dg, 0.05, 2, dg_relax_positive_stepper)
all(isfinite, q_dg_relaxed) || error("DGSEM/HDG/relaxed HEVI bubble state is not finite")
maximum(abs, q_dg_relaxed .- q_dg_end) > 0 || error("positive EPS_IF should change the DG HEVI bubble result")

fv = FV.FVModel(FV.FVGrid(10, 6, 10e3, 6e3), gas, ρr, Θr)
q_fv = bubble_state(fv, FV.project2d)
fv_stepper = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = 12, fast = :hevi)
q_fv_end = TS.run_mis(fv, q_fv, 0.05, 2, fv_stepper)
all(isfinite, q_fv_end) || error("FV/HEVI bubble state is not finite")

zs_cut(x) = 500.0 / (1 + ((x - 5e3) / 1e3)^2)
fvc = FV.FVModel(FV.CutCellFVGrid(10, 6, 10e3, 6e3; zs = zs_cut), gas, ρr, Θr)
q_fvc = bubble_state(fvc, FV.project2d)
fvc_mass0 = FV.total2d(fvc, q_fvc, 1)
fvc_stepper = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = 8, fast = :hevi)
q_fvc_end = TS.run_mis(fvc, q_fvc, 0.05, 2, fvc_stepper)
all(isfinite, q_fvc_end) || error("cut-cell FV/HEVI bubble state is not finite")
abs(FV.total2d(fvc, q_fvc_end, 1) - fvc_mass0) < 1e-11 * abs(fvc_mass0) ||
    error("cut-cell FV/HEVI bubble mass is not conservative")
fvc_amf = FV.FVModel(FV.CutCellFVGrid(10, 6, 10e3, 6e3; zs = zs_cut), gas, ρr, Θr)
q_fvc_amf = bubble_state(fvc_amf, FV.project2d)
fvc_amf_mass0 = FV.total2d(fvc_amf, q_fvc_amf, 1)
fvc_amf_stepper = TS.MISStepper(; tab = TS.mis_tableau(:MIS4), nsub = 8, fast = :rosenbrock_amf)
q_fvc_amf_end = TS.run_mis(fvc_amf, q_fvc_amf, 0.05, 2, fvc_amf_stepper)
all(isfinite, q_fvc_amf_end) || error("cut-cell FV/Rosenbrock-W AMF bubble state is not finite")
abs(FV.total2d(fvc_amf, q_fvc_amf_end, 1) - fvc_amf_mass0) < 1e-11 * abs(fvc_amf_mass0) ||
    error("cut-cell FV/Rosenbrock-W AMF bubble mass is not conservative")

println("warm-bubble MIS checks passed for DGSEM/HDG/HEVI, relaxed HEVI, terrain FV, and cut-cell FV HEVI/AMF")
