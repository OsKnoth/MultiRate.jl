# Stability regions of the MIS tableaus on the acoustic-advection test of Knoth & Wensch (2014):
# C-grid, centred 2nd-order acoustics (fast), 3rd-order upwind advection (slow), exact fast part.
# Stable region (max_θ ρ(θ) ≤ 1 + 1e-8, blue) in C_sound = cΔt/Δx ∈ [0, 24], C_adv = UΔt/Δx ∈ [0, 4]; region of the
# base Runge–Kutta method applied to the whole operator (no fast integration, orange); line U/c = 1/6 (dashed) with
# the largest C_sound up to which the method is stable on it.
# The regions are also cached in stability_fv/regions.bin for docs/mis_polynomial/mis_tableaus.tex.
#
#     julia --project=examples examples/mis_stability_fv.jl [names...]
#
# One figure per method in docs/mis_polynomial/stability_fv/<name>.png and an overview all.png.

using MultiRate, CairoMakie, Printf
const M = MultiRate.MIS
const TS = MultiRate.TimeStepping

names = isempty(ARGS) ? [:WSRK3, :MIS2, :MIS3C, :MIS4, :MIS4a, :MIS_KW3, :MIS54, :MISF53, :MIS_FV42, :MIS_FV53, :MIS_DG53,
                         :TVDMISB, :SSPMIS43, :ConvexMIS52, :ConvexMIS52r20, :ConvexMIS42] : Symbol.(ARGS)
out = joinpath(@__DIR__, "..", "docs", "mis_polynomial", "stability_fv"); mkpath(out)
sym = M.symbols(:fv; acoustic = 2, adv = 3)
Cs = range(0, 24, length = 193); Ca = range(0, 4, length = 81); θs = range(0, π, length = 40)
const MACH = 1 / 6                                   # line C_adv = C_sound / 6, i.e. U/c = 1/6

function region(name; base = false)
    m = M.PolyMIS(TS.mis_tableau(name))
    ρ = zeros(length(Cs), length(Ca))
    Threads.@threads for i in eachindex(Cs)
        for j in eachindex(Ca)
            ρ[i, j] = M.max_amplification(m, sym; Cadv = Ca[j], Csound = Cs[i], θs, base)
        end
    end
    return ρ
end

"""Largest C_sound such that the method is stable at all points C_adv = MACH C_sound' of the line, C_sound' ≤ C_sound."""
function line_limit(name; dc = 1 / 16)
    m = M.PolyMIS(TS.mis_tableau(name)); c = 0.0
    while c + dc <= Cs[end] + 1e-12
        M.max_amplification(m, sym; Cadv = MACH * (c + dc), Csound = c + dc, θs) > 1 + 1e-8 && return c
        c += dc
    end
    return c
end

function panel!(ax, ρ, ρrk)
    st = [x <= 1 + 1e-8 ? 1.0 : NaN for x in ρ]
    heatmap!(ax, Cs, Ca, st; colormap = [:steelblue, :steelblue], nan_color = :white)
    contour!(ax, Cs, Ca, ρ; levels = [1 + 1e-8], color = :black, linewidth = 1)
    strk = [x <= 1 + 1e-8 ? 1.0 : NaN for x in ρrk]
    heatmap!(ax, Cs, Ca, strk; colormap = [(:darkorange, 0.75), (:darkorange, 0.75)], nan_color = :transparent)
    contour!(ax, Cs, Ca, ρrk; levels = [1 + 1e-8], color = :darkorange3, linewidth = 1)
    lines!(ax, [0, Cs[end]], [0, MACH * Cs[end]]; color = :black, linestyle = :dash, linewidth = 1)
    limits!(ax, 0, Cs[end], 0, Ca[end])
end

res = Dict{Symbol,Matrix{Float64}}(); resrk = Dict{Symbol,Matrix{Float64}}(); line = Dict{Symbol,Float64}()
for n in names
    ρ = region(n); res[n] = ρ; resrk[n] = region(n; base = true); line[n] = line_limit(n)
    t = TS.mis_tableau(n); m = M.PolyMIS(t)
    title = @sprintf("%s  (s = %d, Σd = %.2f, order %d / additive %d)", n, length(t.d), sum(t.d), M.order(m; tol = 1e-5),
                     M.order(m; class = :A, tol = 1e-5))
    fig = Figure(size = (720, 330))
    ax = Axis(fig[1, 1]; title, xlabel = "C_sound = cΔt/Δx", ylabel = "C_adv = UΔt/Δx")
    panel!(ax, ρ, resrk[n])
    axislegend(ax, [PolyElement(color = :steelblue), PolyElement(color = (:darkorange, 0.75)),
                    LineElement(color = :black, linestyle = :dash)],
               ["MIS (exact fast part)", "base RK, all explicit", "U/c = 1/6"]; position = :lt, labelsize = 10,
               framevisible = false, backgroundcolor = (:white, 0.7))
    save(joinpath(out, "$n.png"), fig; px_per_unit = 2)
    @printf("%-15s stable fraction %.2f, U/c = 1/6 stable up to C_sound = %.3f\n", n, count(<=(1 + 1e-8), ρ) / length(ρ),
            line[n])
end
nc = 4; nr = cld(length(names), nc)
fig = Figure(size = (360nc, 220nr))
for (k, n) in enumerate(names)
    ax = Axis(fig[cld(k, nc), mod1(k, nc)]; title = string(n), titlesize = 13,
              xlabel = cld(k, nc) == nr ? "C_sound" : "", ylabel = mod1(k, nc) == 1 ? "C_adv" : "")
    panel!(ax, res[n], resrk[n])
end
save(joinpath(out, "all.png"), fig; px_per_unit = 2)
using Serialization; serialize(joinpath(out, "regions.bin"), (Cs = Cs, Ca = Ca, res = res, resrk = resrk, line = line, mach = MACH))
println("wrote ", out)
