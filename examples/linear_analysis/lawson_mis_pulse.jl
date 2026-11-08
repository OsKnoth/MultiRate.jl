# MIS, Lawson and ETD at large slow steps: error against the number of fast substeps (linear pulse)
#
#     julia --project=examples examples/linear_analysis/lawson_mis_pulse.jl
#
# 1D pressure pulse of the MISDG talk (staggered 4th-order acoustics, 5th-order upwind advection,
# c = 350 m/s, U = 50 m/s, 400 cells over 1000 km, T = 10⁴ s), relative L2 error at T against the
# exact semi-discrete solution. Large slow steps: h = 50 s (C_sound = 7, C_adv = 1) and h = 66.7 s
# (C_sound = 9.3, C_adv = 1.33). Fast loops: Störmer–Verlet and its compositions of order 4
# (Suzuki, `_s4`) and 6 (Yoshida, `_y6`); `exact` = exact fast stages / propagators.
# Output: lawson_mis_pulse.png (current directory) and a table.

using MultiRate, CairoMakie, Printf
const LA = MultiRate.LinearAnalysis
const TS = MultiRate.TimeStepping

T = 1e4
nsubs = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
tabs = (:MIS4a, :MIS54, :LRK4, :EB4)
fasts = (:verlet, :verlet_s4, :verlet_y6)
colors = Dict(:MIS4a => :darkorange, :MIS54 => :seagreen, :LRK4 => :royalblue, :EB4 => :crimson)
styles = Dict(:verlet => :dash, :verlet_s4 => :solid, :verlet_y6 => :dot)

"""Base fast steps per slow step (MIS: Σ round(d_i nsub); Lawson: its propagations; ETD: Σ c_i nsub)."""
function fast_cost(tab, nsub, fast)
    c = TS.composition_cost(fast)
    tab isa TS.MISTableau && return c * sum(TS.stage_substeps(tab, nsub))
    return c * TS.fast_time(tab) * nsub
end

fig = Figure(size = (1450, 520))
axs = Axis[]
for (col, n) in enumerate((200, 150))
    h = T / n
    ax = Axis(fig[1, col];  xscale = log10, yscale = log10, xlabel = "base fast steps per slow step",
              ylabel = "relative L2 error at T",
              title = @sprintf("h = %.1f s: C_sound = %.1f, C_adv = %.2f", h, 350h / 2500, 50h / 2500))
    @printf("\nh = %.1f s (C_sound = %.1f, C_adv = %.2f)\n", h, 350h / 2500, 50h / 2500)
    for name in tabs
        tab = TS.mis_tableau(name)
        eex = LA.pulse_error(tab, h; fast = :exact)
        hlines!(ax, [eex]; color = colors[name], linewidth = 0.8)
        @printf("%-6s exact fast stages: %.2e\n", name, eex)
        for fast in fasts
            e = [LA.pulse_error(tab, h; fast, nsub = m) for m in nsubs]
            cost = [fast_cost(tab, m, fast) for m in nsubs]
            ok = isfinite.(e) .& (e .< 1)
            any(ok) && scatterlines!(ax, cost[ok], e[ok]; color = colors[name], linestyle = styles[fast],
                                     label = "$name $fast", markersize = 6)
            @printf("  %-10s nsub %s\n", fast, join([@sprintf("%2d:%.1e", m, x) for (m, x) in zip(nsubs, e)], " "))
        end
    end
    ylims!(ax, 1e-7, 1)
    push!(axs, ax)
end
Legend(fig[1, 3], axs[1]; framevisible = false, labelsize = 11, unique = true)
Label(fig[0, :], "1D pulse: MIS, Lawson (LRK4) and ETD (EB4) with Verlet (dashed), Suzuki order 4 (solid), Yoshida order 6 (dotted); thin lines: exact fast stages")
save("lawson_mis_pulse.png", fig)
println("\nwrote lawson_mis_pulse.png")
