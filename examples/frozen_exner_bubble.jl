# Dry warm bubble, flat FV grid: frozen MIS with the Exner pressure as an additional prognostic
# variable (`FV.ExnerModel`, pressure gradient cp Θ ∇π, π recomputed from Θ after every MIS step)
# against the frozen split with the product form (dp/dΘ) ∇Θ.
#
#     julia --project=examples examples/frozen_exner_bubble.jl [T] [nx]
#
# Isentropic atmosphere θ = 300 K, 20 km × 10 km, Gaussian bubble Δθ = 2 K (radius 2 km) at
# z = 2 km and constant pressure, nx × nx/2 cells (default 160 × 80, Δx = Δz = 125 m), T = 1000 s.
# MIS54, HEVI fast loop. Reference per pressure form: RK4 of the unsplit operator.
# Output: table on stdout, frozen_exner_bubble.png (θ′ at T).

include(joinpath(@__DIR__, "euler2d_common.jl"))
using LinearAlgebra: norm

T = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1000.0
nx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 160
nz = nx ÷ 2
LX, H, Δθ, r0, zb = 20e3, 10e3, 2.0, 2e3, 2e3
gas = Gas()
ρr, Θr, θr = E2.stratified_reference(gas; θ0 = 300.0, N = 0.0)
grid = FV.FVGrid(nx, nz, LX, H)
initial(b) = project2d(b, (x, z) -> begin
    θ = θr(z) + Δθ * exp(-((x - LX / 2) / r0)^2 - ((z - zb) / r0)^2)
    (Θr(z) / θ, 0.0, 0.0, Θr(z))
end)
fields(b, q) = Dict(output_fields(b, q))
Θsum(b, q) = sum(q[FV.iΘ(b.grid, i, k)] for i in 1:nx, k in 1:nz)

"""Frozen MIS run of the Exner model; returns the base state and the largest |π - π(Θ)| before the
recomputation at the end of a step."""
function run_exner(m, q0, Δt, n; tab, nsub)
    q = FV.exner_state(m, q0)
    w = TS.FrozenWork(m, length(q), tab)
    tmp = (zeros(length(q)), zeros(length(q)), zeros(length(q)))
    g = m.base.grid; drift = 0.0
    for _ in 1:n
        TS.frozen_mis_step!(q, m, q, Δt, tab, w; nsub, fast = :hevi, tmp)
        drift = max(drift, maximum(abs(q[FV.iπ(m, i, k)] - FV.exner(gas, q[FV.iΘ(g, i, k)])) for i in 1:nx, k in 1:nz))
        TS.frozen_step_end!(m, q)
    end
    return FV.base_state(m, q), drift
end

results = Dict{Symbol,Any}()
tab = TS.mis_tableau(:MIS54)
for form in (:product, :exner)
    FV.FV_PRESSURE_FORM[] = form
    b = FV.FVModel(grid, gas, ρr, Θr)
    q0 = initial(b)
    c = sound_speed_max(b, q0)
    rest = project2d(b, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
    @printf("\n%s: %d × %d cells, Δx = %.0f m; rest state max|N_W| = %.2e\n", form, nx, nz, grid.dx, maximum(abs, TS.rhs_nonlinear(b, rest)))
    nref = ceil(Int, T / (0.4 * grid.dx / c))
    tr = @elapsed qr = TS.run_rk4(b, q0, T / nref, nref)
    Fr = fields(b, qr)
    @printf("  RK4 reference (%d steps, %.0f s): θ′ ∈ [%.3f, %.3f] K, max|w| = %.2f m/s\n", nref, tr, extrema(Fr["theta_prime"])..., maximum(abs, Fr["w"]))
    dτ = 0.85 * 0.92 * grid.dx / c
    for Δt in (2.0, 5.0, 10.0, 15.0)
        n = round(Int, T / Δt); nsub = ceil(Int, Δt / dτ)
        local q, drift = nothing, NaN
        t = @elapsed try
            if form === :exner
                q, drift = run_exner(FV.ExnerModel(b), q0, Δt, n; tab, nsub)
            else
                q = TS.run_frozen(b, q0, Δt, n; tab, nsub, fast = :hevi)
            end
        catch e
            q = nothing
        end
        if q === nothing || !all(isfinite, q)
            @printf("  Δt = %4.1f s nsub %2d: failed\n", Δt, nsub); continue
        end
        F = fields(b, q)
        @printf("  Δt = %4.1f s nsub %2d: θ′ max %.3f K, max|w| %.2f, error θ′ max %.2e K (rel. state %.2e), Δmass %.1e, ΔΘ %.1e%s, %.0f s\n",
                Δt, nsub, maximum(F["theta_prime"]), maximum(abs, F["w"]), maximum(abs, F["theta_prime"] - Fr["theta_prime"]),
                norm(q - qr) / norm(qr - q0), total2d(b, q, 1) / total2d(b, q0, 1) - 1, Θsum(b, q) / Θsum(b, q0) - 1,
                form === :exner ? @sprintf(", max|π - π(Θ)| before recompute %.1e", drift) : "", t)
        Δt == 5.0 && (results[form] = (F["theta_prime"], Fr["theta_prime"]))
        flush(stdout)
    end
end
FV.FV_PRESSURE_FORM[] = :gradp

if length(results) == 2
    fig = Figure(size = (1500, 820))
    x, z = grid.xc ./ 1e3, grid.zc ./ 1e3
    for (r, form) in enumerate((:product, :exner))
        θ, θref = results[form]
        for (col, (name, fld, cr, cm)) in enumerate((("$form frozen MIS54, Δt = 5 s: θ′", θ, (-0.2, 2.0), :thermal),
                                                     ("$form RK4 reference: θ′", θref, (-0.2, 2.0), :thermal),
                                                     ("$form: MIS − RK4", θ .- θref, (-0.2, 0.2), :balance)))
            ax = Axis(fig[r, 2col-1]; title = name, xlabel = "x [km]", ylabel = "z [km]", aspect = DataAspect(), limits = (4, 16, 0, 10))
            hm = heatmap!(ax, x, z, fld; colormap = cm, colorrange = cr)
            Colorbar(fig[r, 2col], hm)
        end
    end
    save("frozen_exner_bubble.png", fig)
    println("\nwrote frozen_exner_bubble.png")
end
