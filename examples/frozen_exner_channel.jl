# Gravity-wave channel (Skamarock & Klemp 1994), flat FV grid: frozen MIS with the Exner pressure
# as an additional prognostic variable (`FV.ExnerModel`), with the pressure itself (pvar = :p) and
# the product form (dp/dΘ) ∇Θ.
#
#     julia --project=examples examples/frozen_exner_channel.jl
#
# 300 km × 10 km, 120 × 16 cells, N = 0.01, U = 20 m/s, θ′ = 0.01 K sin(πz/H)/(1 + ((x - L/3)/a)²),
# a = 5 km, T = 3000 s. Wave error of θ′ at z = H/2 (mean removed) against RK4 of the same operator;
# discretely balanced initial column per pressure form (no background state); long run 36000 s at
# Δt = 200 s. MIS54, HEVI.

include(joinpath(@__DIR__, "euler2d_common.jl"))

L, H, a, U, T = 300e3, 10e3, 5e3, 20.0, 3000.0
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
grid = FV.FVGrid(120, 16, L, H)
xc = L / 3
"""Discretely balanced ρ at the levels for Θ_k = Θr(z_k) (W equation at rest with the active
pressure form, upwards from ρ_1 = ρr(z_1)); no background state in the equations."""
function balanced_ρ(b)
    g = b.grid; cp = gas.R * gas.γ / (gas.γ - 1)
    Θk = Θr.(g.zc); ρ = zeros(g.nz); ρ[1] = ρr(g.zc[1])
    for k in 2:g.nz
        dp = FV.FV_PRESSURE_FORM[] === :exner ? cp * 0.5 * (Θk[k-1] + Θk[k]) * (FV.exner(gas, Θk[k]) - FV.exner(gas, Θk[k-1])) :
             FV.FV_PRESSURE_FORM[] === :gradp ? pressure(gas, Θk[k]) - pressure(gas, Θk[k-1]) :
                                                FV.dpdΘ_face(gas, Θk[k-1], Θk[k]) * (Θk[k] - Θk[k-1])
        ρ[k] = -2dp / (gas.g * g.dz) - ρ[k-1]
    end
    return ρ
end
"""Balanced column with the θ′ perturbation at constant pressure (Θ unchanged, ρ = Θ/θ)."""
function initial(b; amp = 0.01)
    ρb = balanced_ρ(b); g = b.grid
    lev(z) = clamp(floor(Int, z / g.dz) + 1, 1, g.nz)
    project2d(b, (x, z) -> (k = lev(z); Θ = Θr(g.zc[k]); θ = Θ / ρb[k] + amp * sin(π * z / H) / (1 + ((x - xc) / a)^2);
                            ρ = Θ / θ; (ρ, ρ * U, 0.0, Θ)))
end
θs(b, q) = field_section(b, q, "theta_prime", H / 2)[2]
wave(y) = y .- sum(y) / length(y)
tab = TS.mis_tableau(:MIS54)

function run(form, b, q0, Δt, n, nsub)
    if form in (:exner, :p)
        m = FV.ExnerModel(b; pvar = form === :p ? :p : :pi)
        return FV.base_state(m, TS.run_frozen(m, FV.exner_state(m, q0), Δt, n; tab, nsub, fast = :hevi))
    end
    return TS.run_frozen(b, q0, Δt, n; tab, nsub, fast = :hevi)
end

for form in (:product, :exner, :p)
    FV.FV_PRESSURE_FORM[] = form === :p ? :gradp : form
    b = FV.FVModel(grid, gas, ρr, Θr)
    q0 = initial(b)
    rest = initial(b; amp = 0.0); rest[[FV.iU(grid, i, k) for i in 1:grid.nx, k in 1:grid.nz]] .= 0
    Nr = TS.rhs_nonlinear(b, rest)
    c = sound_speed_max(b, q0)
    nref = ceil(Int, T / (0.4 * min(grid.dx, grid.dz) / (c + U)))
    qr = TS.run_rk4(b, q0, T / nref, nref)
    yr = wave(θs(b, qr))
    @printf("\n%s: rest-state residual max|N_W| = %.2e, wave amplitude (RK4) %.3e K\n", form,
            maximum(abs, Nr[FV.iW(grid, i, k)] for i in 1:grid.nx, k in 2:grid.nz), maximum(abs, yr))
    dτ = fast_step(b, q0, :hevi)
    for Δt in (20.0, 50.0, 100.0, 150.0, 200.0)
        n = round(Int, T / Δt); nsub = ceil(Int, 1.01 * Δt / dτ); set_vadv!(U, Δt, grid.dz)
        q = try run(form, b, q0, Δt, n, nsub) catch; fill(NaN, b.nq) end
        e = all(isfinite, q) ? maximum(abs.(wave(θs(b, q)) .- yr)) : Inf
        @printf("  Δt = %5.1f s nsub %2d: wave error %.2e K (%.1f %%)\n", Δt, nsub, e, 100e / maximum(abs, yr))
        flush(stdout)
    end
    # long run: 36000 s at Δt = 200 s
    Δt = 200.0; n = 180; nsub = ceil(Int, 1.01 * Δt / dτ); set_vadv!(U, Δt, grid.dz)
    q = try run(form, b, q0, Δt, n, nsub) catch; fill(NaN, b.nq) end
    if all(isfinite, q)
        F = Dict(output_fields(b, q))
        @printf("  long run 36000 s, Δt = 200 s: max|w| %.3e m/s, max|θ′| at H/2 %.3e K\n", maximum(abs, F["w"]), maximum(abs, θs(b, q)))
    else
        println("  long run 36000 s, Δt = 200 s: unstable")
    end
end
FV.FV_PRESSURE_FORM[] = :gradp
