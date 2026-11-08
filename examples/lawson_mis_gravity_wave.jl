# MIS, Lawson and ETD (Krogstad EB4) on the nonlinear FV gravity wave at large slow steps
#
#     julia --project=examples examples/lawson_mis_gravity_wave.jl
#
# Skamarock & Klemp gravity wave in the 300 km channel (FV, 120 × 16 cells, U = 20 m/s, T = 3000 s),
# reference: explicit RK4 on N with a small step. θ′ error at z = 5 km against the number of
# fast substeps per slow step for the slow steps Δt = 67 s (the default, nsub = 12 at the acoustic
# limit), 133 s and 200 s (U Δt/Δx = 0.53, 1.07, 1.6; LRK4 is limited to 1.75). Fast loops: FV
# :hevi (horizontal Verlet / vertical Crank–Nicolson, Strang) and its compositions of order 4
# (Suzuki, :hevi_s4) and 6 (Yoshida, :hevi_y6). Output: lawson_mis_gravity_wave.png (error vs fast
# cost), lawson_mis_gravity_wave_sections.png (θ′ at z = 5 km, Δt = 200 s), table on stdout.

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
wave(y) = y .- sum(y) / length(y)              # θ′ without its horizontal mean
# The unbalanced initial state (θ′ at constant pressure) excites a vertical acoustic oscillation
# of the horizontal mean; RK4, Lawson and ETD keep it, MIS damps it. The total error measures it,
# the wave error (horizontal mean removed) does not.
dτmax = fast_step(model, q0, :hevi)                       # stable :hevi substep (acoustic limit)
@printf("RK4 reference: %d steps; :hevi substep limit %.2f s; max θ′ = %.2e K\n", nref, dτmax, maximum(yr))

fasts = (:hevi, :hevi_s4, :hevi_y6)
tabs = (:MIS4, :MIS54, :LRK4, :EB4)
colors = Dict(:MIS4 => :darkorange, :MIS54 => :seagreen, :LRK4 => :royalblue, :EB4 => :crimson)
styles = Dict(:hevi => :dash, :hevi_s4 => :solid, :hevi_y6 => :dot)
fast_cost(tab, nsub, fast) = TS.composition_cost(fast) *
    (tab isa TS.MISTableau ? sum(TS.stage_substeps(tab, nsub)) : TS.fast_time(tab) * nsub)

steps = (45, 23, 15)                                       # Δt = 66.7, 130.4, 200 s
nsubs = (12, 16, 24, 36, 48, 64, 96)
sections = Dict{String,Vector{Float64}}()
fig = Figure(size = (1600, 500))
axs = Axis[]
for (col, n) in enumerate(steps)
    Δt = T / n
    set_vadv!(U, Δt, H / 16)
    ax = Axis(fig[1, col];  xscale = log10, yscale = log10, xlabel = "base fast steps per slow step",
              ylabel = col == 1 ? "wave error max |θ′ − θ′_RK4| at z = 5 km [K]" : "",
              title = @sprintf("Δt = %.0f s, U Δt/Δx = %.2f", Δt, U * Δt / g.dx))
    @printf("\nΔt = %.1f s (%d steps)\n", Δt, n)
    for name in tabs, fast in fasts
        tab = TS.mis_tableau(name)
        es = Float64[]; ets = Float64[]; cs = Float64[]
        for m in nsubs
            Δt / m > 1.01 * dτmax / max(1, TS.composition_stretch(fast)) && continue   # substep beyond the limit
            q = try
                TS.run_mis(model, q0, Δt, n; tab, nsub = m, fast, refreeze_every = 1)
            catch err
                err isa DomainError || rethrow()
                fill(NaN, length(q0))                      # blew up (negative pressure)
            end
            e = all(isfinite, q) ? maximum(abs.(wave(θs(q)) .- wave(yr))) : Inf
            et = all(isfinite, q) ? maximum(abs.(θs(q) .- yr)) : Inf
            push!(es, e); push!(ets, et); push!(cs, fast_cost(tab, m, fast))
            n == 15 && m == 48 && (sections["$name $fast"] = θs(q))
        end
        ok = isfinite.(es) .& (es .< 1e-2)
        any(ok) && scatterlines!(ax, cs[ok], es[ok]; color = colors[name], linestyle = styles[fast],
                                 label = "$name $fast", markersize = 6)
        @printf("  %-6s %-8s wave/total error (base fast steps): %s\n", name, fast, join([@sprintf("%.1e/%.1e (%.0f)", e, et, cc) for (e, et, cc) in zip(es, ets, cs)], " "))
    end
    ylims!(ax, 1e-6, 1e-2)
    push!(axs, ax)
end
Legend(fig[1, 4], axs[1]; framevisible = false, labelsize = 11, unique = true)
Label(fig[0, :], "FV gravity wave (300 km), horizontal mean of θ′ removed: :hevi (dashed), Suzuki order 4 (solid), Yoshida order 6 (dotted); missing: unstable")
save("lawson_mis_gravity_wave.png", fig)

fig2 = Figure(size = (1100, 420))
ax2 = Axis(fig2[1, 1]; xlabel = "x [km]", ylabel = "θ′ [K]", title = "θ′ at z = 5 km, Δt = 200 s, nsub = 48")
lines!(ax2, xs ./ 1e3, yr; color = :black, linewidth = 3, label = "RK4 reference")
for (k, sty) in (("MIS4 hevi", (:darkorange, :solid)), ("MIS54 hevi", (:seagreen, :solid)),
                 ("LRK4 hevi", (:royalblue, :dash)), ("LRK4 hevi_s4", (:royalblue, :solid)),
                 ("EB4 hevi", (:crimson, :dash)), ("EB4 hevi_s4", (:crimson, :solid)))
    haskey(sections, k) && lines!(ax2, xs ./ 1e3, sections[k]; color = sty[1], linestyle = sty[2], label = k)
end
axislegend(ax2; position = :lt)
save("lawson_mis_gravity_wave_sections.png", fig2)
println("\nwrote lawson_mis_gravity_wave.png, lawson_mis_gravity_wave_sections.png")
