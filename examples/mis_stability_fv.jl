# Stability regions of the MIS tableaus on the acoustic-advection test of Knoth & Wensch (2014):
# C-grid, centred 2nd-order acoustics (fast), 3rd-order upwind advection (slow), exact fast part.
# Stable region (max_θ ρ(θ) ≤ 1 + 1e-8, blue) in C_sound = cΔt/Δx ∈ [0, 12], C_adv = UΔt/Δx ∈ [0, 3].
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
Cs = range(0, 12, length = 97); Ca = range(0, 3, length = 61); θs = range(0, π, length = 40)

function region(name)
    m = M.PolyMIS(TS.mis_tableau(name))
    ρ = zeros(length(Cs), length(Ca))
    Threads.@threads for i in eachindex(Cs)
        for j in eachindex(Ca)
            ρ[i, j] = M.max_amplification(m, sym; Cadv = Ca[j], Csound = Cs[i], θs)
        end
    end
    return ρ
end

function panel!(ax, ρ)
    st = [x <= 1 + 1e-8 ? 1.0 : NaN for x in ρ]
    heatmap!(ax, Cs, Ca, st; colormap = [:steelblue, :steelblue], nan_color = :white)
    contour!(ax, Cs, Ca, ρ; levels = [1 + 1e-8], color = :black, linewidth = 1)
    limits!(ax, 0, 12, 0, 3)
end

res = Dict{Symbol,Matrix{Float64}}()
for n in names
    ρ = region(n); res[n] = ρ
    t = TS.mis_tableau(n); m = M.PolyMIS(t)
    title = @sprintf("%s  (s = %d, Σd = %.2f, order %d / additive %d)", n, length(t.d), sum(t.d), M.order(m; tol = 1e-5),
                     M.order(m; class = :A, tol = 1e-5))
    fig = Figure(size = (640, 300))
    ax = Axis(fig[1, 1]; title, xlabel = "C_sound = cΔt/Δx", ylabel = "C_adv = UΔt/Δx")
    panel!(ax, ρ)
    save(joinpath(out, "$n.png"), fig; px_per_unit = 2)
    @printf("%-15s stable fraction %.2f\n", n, count(<=(1 + 1e-8), ρ) / length(ρ))
end
nc = 4; nr = cld(length(names), nc)
fig = Figure(size = (360nc, 220nr))
for (k, n) in enumerate(names)
    ax = Axis(fig[cld(k, nc), mod1(k, nc)]; title = string(n), titlesize = 13,
              xlabel = cld(k, nc) == nr ? "C_sound" : "", ylabel = mod1(k, nc) == 1 ? "C_adv" : "")
    panel!(ax, res[n])
end
save(joinpath(out, "all.png"), fig; px_per_unit = 2)
using Serialization; serialize(joinpath(out, "regions.bin"), (Cs = Cs, Ca = Ca, res = res))
println("wrote ", out)
