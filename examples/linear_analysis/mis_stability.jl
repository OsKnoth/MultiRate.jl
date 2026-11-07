# Linear stability and convergence of the MIS tableaus (after the MISDG experiments)
#
#     julia --project=examples examples/linear_analysis/mis_stability.jl
#
# 1) Knoth–Wensch (2014, eq. 10) stability regions over (C_sound, C_adv), exact fast stages
# 2) MISDG grid (4th-order staggered acoustics, 5th-order upwind advection, C_adv = C_sound/7):
#    max_k ρ - 1 per step for the fast loops of the models with nsub = 12
# 3) 1D pressure pulse: error against the exact semi-discrete solution vs the slow step h
# Output: mis_stability_kw.png, mis_pulse_convergence.png (current directory), tables on stdout.

using MultiRate, CairoMakie, Printf
const LA = MultiRate.LinearAnalysis
const TS = MultiRate.TimeStepping

tabs = (:WSRK3, :MIS_KW3, :MIS4a, :MIS54)

# 1) stability regions
cs = 0.0:0.2:12.0
ca = 0.0:0.05:2.2
fig = Figure(size = (1400, 380))
for (k, name) in enumerate(tabs)
    tab = TS.mis_tableau(name)
    S = [LA.max_amplification(θ -> LA.kw_mode(θ; C_sound = x, C_adv = y), tab; fast = :exact) <= 1 + 1e-10
         for x in cs, y in ca]
    ax = Axis(fig[1, k]; title = String(name), xlabel = "C_sound = c Δt/Δx", ylabel = k == 1 ? "C_adv = U Δt/Δx" : "")
    heatmap!(ax, cs, ca, Float64.(S); colormap = [:mistyrose, :steelblue], colorrange = (0, 1))
    @printf("%-8s stable fraction of the (C_sound, C_adv) box: %.2f\n", name, sum(S) / length(S))
end
Label(fig[0, :], "Knoth–Wensch linear acoustics: stable (blue), exact fast stages, 40 Fourier modes")
save("mis_stability_kw.png", fig)

# 2) MISDG grid with the fast loops of the models
css = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
println("\nMISDG grid, C_adv = C_sound/7, nsub = 12: max_k ρ - 1 per step")
@printf("%-8s %-7s %s\n", "", "C_sound", join([@sprintf("%9.0f", c) for c in css]))
for name in (:WSRK3, :MIS4, :MIS4a, :MIS54), fast in (:exact, :verlet, :fb, :wrf)
    tab = TS.mis_tableau(name)
    r = [LA.max_amplification(θ -> LA.staggered_mode(θ; C_sound = c, C_adv = c / 7), tab; nmodes = 80,
                              fast = fast, nsub = 12) - 1 for c in css]
    @printf("%-8s %-7s %s\n", name, fast, join([@sprintf("%9.1e", x) for x in r]))
end

# 3) pulse convergence (c = 350 m/s, U = 50 m/s, 400 cells over 1000 km, T = 10⁴ s)
T = 3.5e6 / 350
ns = (100, 200, 400, 800, 1600)
fig = Figure(size = (650, 480))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "slow step h [s]", ylabel = "relative L2 error at T",
          title = "1D pressure pulse, exact fast stages")
println("\npulse: relative L2 error at T (exact fast stages)")
for name in (:WSRK3, :MIS4a, :MIS54)
    tab = TS.mis_tableau(name)
    e = [LA.pulse_error(tab, T / n; fast = :exact) for n in ns]
    ok = isfinite.(e) .& (e .< 1)
    scatterlines!(ax, [T / n for n in ns][ok], e[ok]; label = String(name))
    @printf("%-8s %s\n", name, join([@sprintf("%9.2e", x) for x in e], " "))
end
axislegend(ax; position = :rb)
save("mis_pulse_convergence.png", fig)
println("\nwrote mis_stability_kw.png, mis_pulse_convergence.png")
