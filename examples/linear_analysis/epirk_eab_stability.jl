# Stability of EPIRK(-W) and exponential Adams–Bashforth against MIS, Lawson and ETD
#
#     julia --project=examples examples/linear_analysis/epirk_eab_stability.jl
#
# Maximum amplification over the Fourier modes on the MISDG grid (4th-order staggered acoustics,
# 5th-order upwind advection) with exact fast stages, against the advective Courant number C_adv:
# left without acoustics (c = 0), right with c/U = 7 (C_sound = 7 C_adv, as in the pulse test).
# Output: epirk_eab_stability.png (current directory) and a table.

using MultiRate, CairoMakie, Printf
const LA = MultiRate.LinearAnalysis
const TS = MultiRate.TimeStepping

tabs = (:MIS54, :LRK4, :EB4, :EPIRKW3a, :EPIRKW3b, :EPIRK4, :EAB2, :EAB3, :EAB4)
colors = Dict(:MIS54 => :seagreen, :LRK4 => :royalblue, :EB4 => :crimson, :EPIRKW3a => :purple,
              :EPIRKW3b => :magenta, :EPIRK4 => :gray40, :EAB2 => :goldenrod, :EAB3 => :darkorange,
              :EAB4 => :saddlebrown)
Cas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0]
amp(tab, cs, ca) = LA.max_amplification(θ -> LA.staggered_mode(θ; C_sound = max(cs, 1e-9), C_adv = ca), tab;
                                        fast = :exact, nmodes = 40)

fig = Figure(size = (1300, 500))
axs = Axis[]
for (col, ratio) in enumerate((0.0, 7.0))
    ax = Axis(fig[1, col]; xscale = log10, yscale = log10, xlabel = "C_adv = U h/Δx",
              ylabel = "max amplification − 1 (floor 1e-8)",
              title = ratio == 0 ? "advection only (c = 0)" : "with acoustics, C_sound = 7 C_adv")
    @printf("\nc/U = %.0f, C_adv = %s\n", ratio, join(Cas, " "))
    for name in tabs
        tab = TS.mis_tableau(name)
        ρ = [amp(tab, ratio * ca, ca) for ca in Cas]
        scatterlines!(ax, Cas, max.(ρ .- 1, 1e-8); color = colors[name], label = string(name), markersize = 6)
        @printf("  %-9s %s\n", name, join([@sprintf("%.4f", r) for r in ρ], " "))
    end
    ylims!(ax, 5e-9, 10)
    push!(axs, ax)
end
Legend(fig[1, 3], axs[1]; framevisible = false, unique = true)
Label(fig[0, :], "MISDG grid, exact fast stages: MIS54, Lawson (LRK4), ETD (EB4), EPIRK-W (order 3), classical EPIRK4 with L ≠ Jacobian, exponential Adams–Bashforth")
save("epirk_eab_stability.png", fig)
println("\nwrote epirk_eab_stability.png")
