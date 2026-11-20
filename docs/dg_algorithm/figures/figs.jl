using CairoMakie, Serialization
S = @__DIR__; D = joinpath(@__DIR__, "..")
if isfile(joinpath(S, "rest_old.bin"))
    new, old = deserialize(joinpath(S, "rest_new.bin")), deserialize(joinpath(S, "rest_old.bin"))
    fig = Figure(size = (700, 380))
    ax = Axis(fig[1, 1]; xlabel = "t [min]", ylabel = "max |ρu|, |ρw|  [kg m⁻² s⁻¹]", yscale = log10,
              title = "Isothermal rest over a 400 m hill, MIS54 + S2 (:cs), Δt = 30 s")
    for (lab, c) in (("contravariant, HDG", :royalblue), ("full, LU", :darkorange))
        t, e = old[lab]; lines!(ax, t ./ 60, max.(e, 1e-16); color = c, linestyle = :dash, label = "previous: $lab")
        t, e = new[lab]; lines!(ax, t ./ 60, max.(e, 1e-16); color = c, label = "well balanced: $lab")
    end
    axislegend(ax; position = (0.98, 0.7)); save(joinpath(D, "rest_drift.png"), fig; px_per_unit = 2)
end
if isfile(joinpath(S, "a100_fig.bin"))
    r = deserialize(joinpath(S, "a100_fig.bin"))
    X, Z = (r.X .- r.xm) ./ 1e3, r.Z ./ 1e3; sel = (abs.(X) .< 150) .& (Z .< 9)
    fig = Figure(size = (1300, 380)); lv = range(-1.2e-2, 1.2e-2, length = 25)
    for (c, (tt, f)) in enumerate((("DG, MIS54 + S2, t = 12 h", r.w), ("linear hydrostatic solution", r.wl), ("difference", r.w .- r.wl)))
        ax = Axis(fig[1, c]; title = tt, xlabel = "x − x₀ [km]", ylabel = c == 1 ? "z [km]" : "")
        hm = tricontourf!(ax, X[sel], Z[sel], f[sel]; levels = lv, colormap = :balance, extendlow = :auto, extendhigh = :auto)
        c == 3 && Colorbar(fig[1, 4], hm; label = "w [m/s]")
    end
    save(joinpath(D, "a100_w.png"), fig; px_per_unit = 2)
end
