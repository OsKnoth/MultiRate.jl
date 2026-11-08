# MIS with a frozen part of the right-hand side against MIS with a linearized fast operator
#
#     julia --project=examples examples/frozen_gravity_wave.jl
#
# Skamarock & Klemp gravity wave in the 300 km channel (flat FV, 120 × 16 cells, U = 20 m/s,
# T = 3000 s), reference explicit RK4 on N. Frozen split (`TimeStepping.run_frozen`):
# N(q) = F(q, q), fast part with the nonlinear p(Θ) and the face θ frozen at the stage values,
# slow part momentum advection; with FV_FROZEN_PRESSURE = :derivative dp/dΘ is frozen at the faces
# as well (fast part linear).
# Linearized split (`run_mis`): fast operator L frozen at qⁿ,
# S = N - L. Fast loops :verlet (explicit) and :hevi, nsub at the substep limit. Wave error of θ′
# at z = 5 km (horizontal mean removed) against the slow step.
# Output: frozen_gravity_wave.png, frozen_gravity_wave_sections.png, table on stdout.

include(joinpath(@__DIR__, "euler2d_common.jl"))

L, H, a, U, T = 300e3, 10e3, 5e3, 20.0, 3000.0
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
model = build_model(:fv; L, H, P = 4, NEX = 30, NEZ = 4, ρr, Θr, gas)
xc = L / 3
q0 = project2d(model, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / H) / (1 + ((x - xc) / a)^2));
                                 (ρ, ρ * U, 0.0, Θr(z))))
g = model.grid
c = sound_speed_max(model, q0)
nref = ceil(Int, T / (0.4 * min(g.dx, g.dz) / (c + U)))
qref = TS.run_rk4(model, q0, T / nref, nref)
xs, yr, _ = field_section(model, qref, "theta_prime", H / 2)
θs(q) = field_section(model, q, "theta_prime", H / 2)[2]
wave(y) = y .- sum(y) / length(y)
err(q) = all(isfinite, q) ? maximum(abs.(wave(θs(q)) .- wave(yr))) : Inf
dτ = Dict(:verlet => 0.9 * g.dz / c, :hevi => fast_step(model, q0, :hevi))

steps = (45, 30, 23, 18, 15)                              # Δt = 67 … 200 s
tabs = (:WSRK3, :MIS4, :MIS4a, :MIS54, :MIS_KW3)
colors = Dict(:WSRK3 => :gray40, :MIS4 => :darkorange, :MIS4a => :purple, :MIS54 => :seagreen, :MIS_KW3 => :royalblue)
# (split, fast loop, pressure of the frozen fast part, line style)
runs = ((:frozen, :hevi, :nonlinear, :solid), (:frozen, :verlet, :nonlinear, :dot),
        (:frozen, :hevi, :derivative, :dashdot), (:linearized, :hevi, :nonlinear, :dash))
sections = Dict{String,Vector{Float64}}()
fig = Figure(size = (1100, 520))
ax = Axis(fig[1, 1]; xlabel = "U Δt/Δx", ylabel = "wave error max |θ′ − θ′_RK4| at z = 5 km [K]", yscale = log10,
          title = "FV gravity wave: frozen RHS (solid :hevi, dotted :verlet, dash-dot :hevi with dp/dΘ frozen) vs linearized L (dashed)")
for name in tabs, (kind, fast, pmode, sty) in runs
    tab = TS.mis_tableau(name)
    MultiRate.FV2D.FV_FROZEN_PRESSURE[] = pmode
    es = Float64[]
    for n in steps
        Δt = T / n
        m = ceil(Int, 1.01 * Δt / dτ[fast])
        set_vadv!(U, Δt, H / 16)
        q = try
            kind == :frozen ? TS.run_frozen(model, q0, Δt, n; tab, nsub = m, fast) :
                              TS.run_mis(model, q0, Δt, n; tab, nsub = m, fast)
        catch e
            e isa DomainError || rethrow()
            fill(NaN, length(q0))
        end
        push!(es, err(q))
        n == 15 && fast == :hevi && pmode == :nonlinear && (sections["$name $kind"] = θs(q))
    end
    @printf("%-8s %-10s %-7s %-10s wave error at Δt = %s: %s\n", name, kind, fast, pmode, join([@sprintf("%.0f", T / n) for n in steps], "/"),
            join([@sprintf("%.1e", e) for e in es], " "))
    cr = [U * T / n / g.dx for n in steps]
    ok = isfinite.(es) .& (es .< 1e-2)
    any(ok) && scatterlines!(ax, cr[ok], es[ok]; color = colors[name], linestyle = sty, label = "$name $kind $fast" * (pmode == :derivative ? " dp/dΘ" : ""), markersize = 6)
end
MultiRate.FV2D.FV_FROZEN_PRESSURE[] = :nonlinear
ylims!(ax, 3e-5, 1e-2)
Legend(fig[1, 2], ax; framevisible = false, labelsize = 10)
save("frozen_gravity_wave.png", fig)

fig2 = Figure(size = (1100, 420))
ax2 = Axis(fig2[1, 1]; xlabel = "x [km]", ylabel = "θ′ [K]", title = "θ′ at z = 5 km, Δt = 200 s, :hevi at the substep limit")
lines!(ax2, xs ./ 1e3, yr; color = :black, linewidth = 3, label = "RK4 reference")
for (k, sty) in (("MIS54 frozen", (:seagreen, :solid)), ("MIS54 linearized", (:seagreen, :dash)),
                 ("MIS4 frozen", (:darkorange, :solid)), ("MIS_KW3 frozen", (:royalblue, :solid)))
    haskey(sections, k) && all(isfinite, sections[k]) && lines!(ax2, xs ./ 1e3, sections[k]; color = sty[1], linestyle = sty[2], label = k)
end
axislegend(ax2; position = :lt)
save("frozen_gravity_wave_sections.png", fig2)
println("wrote frozen_gravity_wave.png, frozen_gravity_wave_sections.png")
