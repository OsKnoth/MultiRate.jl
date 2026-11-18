# Extended formulation only (docs/fast_2d, Section "Extended formulation", scheme E1): the Skamarock & Klemp (1994)
# nonhydrostatic gravity wave (300 km × 10 km, Δx = Δz = 1 km: DG degree 5 on 60 × 2 elements, U = 20 m/s, N = 0.01/s,
# Δθ0 = 0.01 K, a = 5 km, T = 3000 s) with MIS54 and the explicit Störmer–Verlet small step.
#
#     [DT=12] [NSUBFAC=0.6] [KAPPA=0.5] [REF=1] julia --project=examples examples/sk94_extended.jl
#
# NSUBFAC: small step as a fraction of the explicit Verlet limit of the standard formulation (E1 needs ≤ 0.6 for κ = 1/2).
# REF=1: also the RK4 reference of the standard formulation (LMARS in ρu), ≈ 4 min, and the error of θ.
#
# Where the extended formulation is in the code (src/DGSEM/Euler2D/frozen_p.jl):
#   DGPressureModel(base; interface = true, kappa)  the model; state [ρ, m_x, m_z, Θ | p | M̂], one M̂ per interior face node
#                                                   (contravariant, M̂ = ñ·m̂), index m.mhat[face pair]
#   _lmars_momentum!     face coefficients of LMARS in ρu (C₁ = C₂ = 1/2, C₃ = 1/(2c), C₇ = C₈ = c/2)
#   _dgp_terms!          all terms of the linear fast operator as emit(row, col, coef); the branch `elseif m.interface`
#                        holds the face terms of the extended formulation: ρ, Θ (and p) rows with M̂, momentum rows with the
#                        interface pressure of their side (no pressure), and the M̂ row M̂_t = κ σ̄ s (P̂_L - P̂_R)
#   _dgp_isthermo        the groups: thermodynamic rows T = (ρ, Θ, p); all others (m_x, m_z, M̂) are the momentum group M
#   frozen_fast!         the parts :thermo (T rows) and :momentum (M rows) used by the Verlet small step
#   interface_reset!     M̂ ← s m*(q), the LMARS value (start of the run and after every MIS step, via frozen_step_end!)
# The small step (src/TimeStepping/frozen.jl, frozen_fast_solve!, method :verlet) is E1:
#   T½ = T⁰ + h/2 (A_T M⁰ + f_T);   M¹ = M⁰ + h (A_M T½ + D_M M⁰ + f_M);   T¹ = T½ + h/2 (A_T M¹ + f_T)
include(joinpath(@__DIR__, "euler2d_common.jl"))

L, H, a, U, T, Δθ0 = 300e3, 10e3, 5e3, 20.0, 3000.0, 0.01
Δt = parse(Float64, get(ENV, "DT", "12"))
fac = parse(Float64, get(ENV, "NSUBFAC", "0.6"))
κ = parse(Float64, get(ENV, "KAPPA", "0.5"))
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
base = build_model(:dg; L, H, P = 5, NEX = 60, NEZ = 2, ρr, Θr, gas)
qb = project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + Δθ0 * sin(π * z / H) / (1 + ((x - L / 3) / a)^2)); (ρ, ρ * U, 0.0, Θr(z))))
M = base.mesh

# the extended model and its initial state (p = p(Θ), M̂ = s m* of the initial state)
me = E2.DGPressureModel(base; interface = true, kappa = κ)
q0 = E2.pressure_state(me, qb)
@printf("unknowns: %d element values (ρ, m_x, m_z, Θ), %d pressures, %d interface momenta\n",
        base.nq, M.np^2 * M.Ne, me.nq - base.nq - M.np^2 * M.Ne)

# small step: fraction of the explicit Verlet limit (CFL table of the DG examples, degree 5)
c = sound_speed_max(base, qb)
dτ0 = 0.85 * 0.52 * min(M.Lx / M.Nex, M.H / M.Nez) / ((2M.p + 1) * c)
nsub = ceil(Int, Δt / (fac * dτ0))
n = round(Int, T / Δt)
t = @elapsed q = TS.run_frozen(me, q0, Δt, n; tab = TS.mis_tableau(:MIS54), nsub, fast = :verlet)
θf(x) = E2.field(base, x[1:base.nq], 4) ./ E2.field(base, x[1:base.nq], 1)
θp = θf(q) .- θr.(M.Z)
@printf("E1, κ = %.2f, MIS54 Δt = %.0f s, small step %.2f s (%.2f × limit, nsub %d): max θ' %.3e K, min θ' %.3e K (%.0f s)\n",
        κ, Δt, Δt / nsub, Δt / nsub / dτ0, nsub, maximum(θp), minimum(θp), t)

if get(ENV, "REF", "0") == "1"
    ms = E2.DGPressureModel(base; lmars = :momentum)          # standard formulation, same LMARS
    nref = ceil(Int, T / (0.5dτ0))
    qref = TS.run_rk4(ms, E2.pressure_state(ms, qb), T / nref, nref)
    @printf("error θ against RK4 of the standard formulation: %.3e K (max θ' of the reference %.3e K)\n",
            maximum(abs.(θf(q) .- θf(qref))), maximum(θf(qref) .- θr.(M.Z)))
end

# θ' at T (contour interval 5e-4 K, negative dashed); DG nodes on element faces are averaged
d = Dict{Tuple{Float64,Float64},Tuple{Float64,Int}}()
for (x, z, v) in zip(vec(M.X) ./ 1e3, vec(M.Z) ./ 1e3, vec(θp))
    k = (round(x; digits = 6), round(z; digits = 6)); s, m = get(d, k, (0.0, 0)); d[k] = (s + v, m + 1)
end
ks = collect(keys(d)); xs, zs, vs = first.(ks), last.(ks), [d[k][1] / d[k][2] for k in ks]
fig = Figure(; size = (1100, 320))
ax = Axis(fig[1, 1]; title = @sprintf("θ' [K] at t = 3000 s, extended formulation E1, κ = %.2f", κ), xlabel = "x [km]", ylabel = "z [km]")
contour!(ax, xs, zs, vs; levels = collect(5e-4:5e-4:5e-3), color = :black)
contour!(ax, xs, zs, vs; levels = collect(-5e-3:5e-4:-5e-4), color = :black, linestyle = :dash)
save(joinpath(@__DIR__, "sk94_extended.png"), fig)
println("wrote examples/sk94_extended.png")
