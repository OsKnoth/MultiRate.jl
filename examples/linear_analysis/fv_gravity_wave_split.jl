# Linearized production FV on the long-channel gravity wave (euler2d_gravity_wave_long.jl):
# which terms of the slow part S = N - L break MIS4 at the default step Δt = 540 s.
#
#     julia --project=examples examples/linear_analysis/fv_gravity_wave_split.jl
#
# The flat FV model is invariant under shifts in x, so for every horizontal Fourier mode the
# Jacobian J of N (finite differences of rhs_nonlinear!) and the frozen fast operator L reduce to
# 63 × 63 column blocks. The MIS step of the production code on each block (step_matrix) is
# compared with the exact linear solution exp(T J) for the initial condition of the driver.
# Prints the θ′ maxima of the two packets at z = 4688 m, the error and the largest |eigenvalue|
# of the step (modes j ≤ 60 of 300). Takes a few minutes.

using MultiRate, LinearAlgebra, Printf
const LA = MultiRate.LinearAnalysis
const TS = MultiRate.TimeStepping
const FV = MultiRate.FV2D
const CM = MultiRate.Common

gas = CM.Gas()
U, H, Lx, a, T = 20.0, 10e3, 6000e3, 100e3, 60000.0
ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
g = FV.FVGrid(300, 16, Lx, H)
m = FV.FVModel(g, gas, ρr, Θr)
nz = g.nz
qb = CM.project2d(m, (x, z) -> (ρr(z), ρr(z) * U, 0.0, Θr(z)))
xc = Lx / 3
q0 = CM.project2d(m, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / H) / (1 + ((x - xc) / a)^2));
                                (ρ, ρ * U, 0.0, Θr(z))))
idx = vcat([[FV.iρ(g, i, k) for i in 1:g.nx] for k in 1:nz], [[FV.iU(g, i, k) for i in 1:g.nx] for k in 1:nz],
           [[FV.iW(g, i, k) for i in 1:g.nx] for k in 2:nz], [[FV.iΘ(g, i, k) for i in 1:g.nx] for k in 1:nz])
nd = length(idx)
thermo = vcat(trues(nz), falses(nz), falses(nz - 1), trues(nz))
hmom = vcat(falses(nz), trues(nz), falses(nz - 1), falses(nz))
Urows = Diagonal(vcat(zeros(nz), ones(nz), zeros(nz - 1), zeros(nz)))
Q0 = LA.fourier_coefficients(q0 .- qb, idx)
kθ = 8                                                     # z = 4688 m
θprime(δ) = [(δ[FV.iΘ(g, i, kθ)] - qb[FV.iΘ(g, i, kθ)] / qb[FV.iρ(g, i, kθ)] * δ[FV.iρ(g, i, kθ)]) /
             qb[FV.iρ(g, i, kθ)] for i in 1:g.nx]
function packets(th)
    l = g.xc .< mod(xc + U * T, Lx)
    return maximum(th[l]), maximum(th[.!l])
end

function run(variant, fast, nsteps, nsub; jmax = 60)
    FV.FV_VADV_FAST[] = variant == :vadv
    Δt = T / nsteps
    Qn = zeros(ComplexF64, nd, g.nx); Qe = zeros(ComplexF64, nd, g.nx)
    ρmax = 0.0
    for j in 0:jmax
        _, J, L, Lv = LA.linearized_mode_model(m, qb, idx, 2π * j / g.nx; thermo, hmom, Δt = Δt)
        variant == :Urows && (L = L + Urows * (J - L))      # all of S's horizontal-momentum rows in L
        R = LA.step_matrix(LA.mode_model(J, L; Lv, thermo, hmom), 1.0, TS.mis_tableau(:MIS4); nsub, fast)
        ρmax = max(ρmax, LA.spectral_radius(R))
        q = R^nsteps * vcat(real(Q0[:, j+1]), imag(Q0[:, j+1]))
        Qn[:, j+1] = q[1:nd] + im * q[nd+1:end]
        Qe[:, j+1] = exp(nsteps * J) * Q0[:, j+1]
        if j > 0
            Qn[:, g.nx-j+1] = conj(Qn[:, j+1]); Qe[:, g.nx-j+1] = conj(Qe[:, j+1])
        end
    end
    FV.FV_VADV_FAST[] = false
    thn = θprime(LA.fourier_synthesis(Qn, idx, length(qb)))
    the = θprime(LA.fourier_synthesis(Qe, idx, length(qb)))
    return packets(thn), packets(the), maximum(abs.(thn - the)), ρmax
end

@printf("%-28s %-12s %8s  %-22s %-22s %9s %8s\n", "L", "fast", "Δt [s]", "MIS4 packets [K]", "exact packets [K]",
        "max err", "max ρ")
for (variant, label) in ((:default, "as in the driver (no vadv)"), (:vadv, "vertical u-transport (vadv)"),
                         (:Urows, "all S momentum rows in L")),
    (nsteps, nsub) in ((111, 12), (443, 3)), fast in (:exact, :hevi_verlet)
    pn, pe, err, ρ = run(variant, fast, nsteps, nsub)
    @printf("%-28s %-12s %8.1f  %.2e / %.2e    %.2e / %.2e    %9.2e %8.4f\n", label, fast, T / nsteps,
            pn..., pe..., err, ρ)
end
