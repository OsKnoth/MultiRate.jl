# Postprocessing of examples/frozen_agnesi.jl: errors of all saved runs against the RK4 reference
# of their FORM, tables (LaTeX) and figures for docs/mis_agnesi.
#
#     julia --project=examples examples/frozen_agnesi_post.jl <OUTDIR with agnesi_*.jls> <figure dir>

include(joinpath(@__DIR__, "euler2d_common.jl"))
using Serialization

const DIR, FIG = ARGS[1], ARGS[2]
const a, h0, U0, NBV, θ0 = 10e3, 400.0, 10.0, 0.01, 300.0
const LX, HTOP, ZD = 40a, 15.6e3, 9e3
const nx, nz = 200, 78
xm = LX / 2
gas = Gas(); κ = gas.R * (gas.γ - 1) / (gas.R * gas.γ)
grid = FV.FVGrid(nx, nz, LX, HTOP; zs = x -> h0 / (1 + ((x - xm) / a)^2))
θb(z) = θ0 * exp(NBV^2 * z / gas.g)
ρref(z) = (p = gas.p00 * exp(-gas.g * z / (gas.R * θ0)); p / (gas.R * θb(z) * (p / gas.p00)^((gas.γ - 1) / gas.γ)))
wphys(x, i, k) = x[FV.iW(grid, i, k)] / (0.5 * (x[FV.iρ(grid, i, k - 1)] + x[FV.iρ(grid, i, k)]))
w_c(x) = [0.5 * ((k > 1 ? wphys(x, i, k) : grid.zx_f[i, 1] * U0) + (k < nz ? wphys(x, i, k + 1) : 0.0)) for i in 1:nx, k in 1:nz]
θp(x) = [x[FV.iΘ(grid, i, k)] / x[FV.iρ(grid, i, k)] - θb(grid.Zc[i, k]) for i in 1:nx, k in 1:nz]
const REG = [abs(grid.xc[i] - xm) <= 3a && grid.Zc[i, k] <= ZD for i in 1:nx, k in 1:nz]
relL2(f, g) = sqrt(sum(abs2, (f .- g)[REG]) / sum(abs2, g[REG]))

"""Linear steady hydrostatic/nonhydrostatic w over the Agnesi hill (Fourier integral, radiation
condition, amplitude factor √(ρ(0)/ρ(z)); as examples/euler2d_hills.jl)."""
function w_linear(x, z; nk = 20000)
    l = NBV / U0; kmax = 40 / a + 2l
    k = range(0, kmax; length = nk); dk = step(k); s = 0.0im
    for kk in k
        E = kk < l ? exp(im * sqrt(l^2 - kk^2) * z) : exp(-sqrt(kk^2 - l^2) * z)
        s += im * kk * U0 * h0 * a * exp(-kk * a) * exp(im * kk * (x - xm)) * E
    end
    return real(s) * dk * sqrt(ρref(0.0) / ρref(z))
end

T = 10800
refs = Dict{Symbol,Any}()
for f in (:exner, :gradp, :product)
    fn = joinpath(DIR, "agnesi_ref_$(f)_T$(T).jls")
    isfile(fn) && (refs[f] = deserialize(fn))
end
refs[:linear] = get(refs, :gradp, nothing)
rw = Dict(f => (x === nothing ? nothing : w_c(x)) for (f, x) in refs)
rθ = Dict(f => (x === nothing ? nothing : θp(x)) for (f, x) in refs)

# ------------------------------------------------------------------ all runs
struct Run; form::Symbol; meth::Symbol; dt::Float64; nsub::Int; wmax::Float64; ew::Float64; eθ::Float64; end
runs = Run[]
for fn in readdir(DIR)
    mt = match(r"^agnesi_x_(\w+?)_(\w+)_(\d+)_(\d+)\.jls$", fn)
    mt === nothing && continue
    f, me = Symbol(mt[1]), Symbol(mt[2])
    rw[f] === nothing && continue
    x = deserialize(joinpath(DIR, fn)); w = w_c(x)
    push!(runs, Run(f, me, parse(Float64, mt[3]), parse(Int, mt[4]), maximum(abs, w), relL2(w, rw[f]), relL2(θp(x), rθ[f])))
end
# unstable runs from the logs
unstable = Set{Tuple{Symbol,Symbol,Float64,Int}}()
for fn in readdir(joinpath(DIR, "logs")), ln in eachline(joinpath(DIR, "logs", fn))
    sp = split(ln)
    length(sp) >= 6 && sp[1] == "RES" && sp[6] == "unstable" && push!(unstable, (Symbol(sp[2]), Symbol(sp[3]), parse(Float64, sp[4]), parse(Int, sp[5])))
end
@printf("%d stable runs, %d unstable\n", length(runs), length(unstable))

forms = [:exner, :gradp, :product, :linear]
formlabel = Dict(:exner => "frozen, extended (π)", :gradp => "frozen, full pressure", :product => "frozen, product form",
                 :linear => "linearized (L + (N − L))")
texlabel = Dict(:exner => "frozen, extended (\\(\\pi\\))", :gradp => "frozen, full pressure", :product => "frozen, product form",
                :linear => "linearized (\\(L+(N-L)\\))")
methods = [:MIS2, :MIS3C, :MIS4a, :MIS54, :WSRK3, :ETD3, :EB4, :LRK4]
mlabel = Dict(:MIS2 => "MIS2", :MIS3C => "MIS3C", :MIS4a => "MIS4a", :MIS54 => "MIS54", :WSRK3 => "RK3 (WS)",
              :ETD3 => "ETD3", :EB4 => "ETD4", :LRK4 => "Lawson4")
nsub1(dt) = ceil(Int, dt / 4.30)            # the standard nsub of the scan (Δτ_ac = 4.30 s, printed by the runs)
basic(r) = r.nsub == nsub1(r.dt)
# frozen: cool hues, linearized: warm (validated categorical slots)
fsty = Dict(:exner => ("#2a78d6", :circle), :gradp => ("#1baf7a", :utriangle), :product => ("#6250d6", :rect),
            :linear => ("#eb6834", :diamond))

# ------------------------------------------------------------------ tables
dts = sort(unique([r.dt for r in runs if basic(r)] ∪ [u[3] for u in unstable]))
# primary measure: θ' (the w field carries the undamped vertical acoustic modes of the impulsive start,
# which no split method reproduces in phase); w as a secondary table
const THθ = 0.1                             # "accurate": relative L2 error of θ' below 10 %
err(r, met) = met === :θ ? r.eθ : r.ew
for (met, fn) in ((:θ, "tab_error_theta.tex"), (:w, "tab_error_w.tex"))
    open(joinpath(FIG, fn), "w") do io
        for f in forms
            println(io, "\\multicolumn{$(length(dts)+1)}{l}{\\emph{$(texlabel[f])}}\\\\")
            for me in methods
                cells = map(dts) do dt
                    r = findfirst(r -> r.form == f && r.meth == me && r.dt == dt && basic(r), runs)
                    r !== nothing ? @sprintf("%.3f", err(runs[r], met)) : ((f, me, dt, nsub1(dt)) in unstable ? "u" : "--")
                end
                println(io, mlabel[me], " & ", join(cells, " & "), "\\\\")
            end
            f === :linear || println(io, "\\midrule")
        end
    end
end
open(joinpath(FIG, "tab_dtmax.tex"), "w") do io
    for me in methods
        cells = map(forms) do f
            st = [r.dt for r in runs if r.form == f && r.meth == me && basic(r) && r.eθ < THθ]
            isempty(st) ? "--" : @sprintf("%.0f", maximum(st))
        end
        println(io, mlabel[me], " & ", join(cells, " & "), "\\\\")
    end
end
open(joinpath(FIG, "tab_nsub.tex"), "w") do io
    for f in (:exner, :gradp)
        println(io, "\\multicolumn{5}{l}{\\emph{$(texlabel[f])}}\\\\")
        for me in methods
            cells = map((1, 2, 4, 8)) do fac
                r = findfirst(r -> r.form == f && r.meth == me && r.dt == 60 && r.nsub == fac * nsub1(60), runs)
                r !== nothing ? @sprintf("%.4f", runs[r].eθ) : ((f, me, 60.0, fac * nsub1(60)) in unstable ? "u" : "--")
            end
            println(io, mlabel[me], " & ", join(cells, " & "), "\\\\")
        end
        f === :exner && println(io, "\\midrule")
    end
end

# ------------------------------------------------------------------ figures
set_theme!(fontsize = 14)
gridkw = (xgridcolor = (:black, 0.08), ygridcolor = (:black, 0.08))
X = [grid.xc[i] / 1e3 - xm / 1e3 for i in 1:nx, k in 1:nz]; Z = grid.Zc ./ 1e3
sel = findall(x -> abs(x) <= 60, X[:, 1])

# (1) reference wave and linear theory at z ≈ 3 km
if rw[:exner] !== nothing
    fig = Figure(size = (1300, 430), backgroundcolor = :white)
    ax = Axis(fig[1, 1]; title = "w [m/s], RK4 reference (frozen extended system), t = $(T) s", xlabel = "x − x_m [km]",
              ylabel = "z [km]", limits = (-60, 60, 0, 12))
    W = rw[:exner]; cr = maximum(abs, W[sel, :])
    sf = surface!(ax, X[sel, :], Z[sel, :], zeros(length(sel), nz); color = W[sel, :], colormap = :balance, colorrange = (-cr, cr), shading = NoShading)
    lines!(ax, X[sel, 1], grid.Zf[sel, 1] ./ 1e3; color = :black)
    Colorbar(fig[1, 2], sf)
    ax2 = Axis(fig[1, 3]; title = "w at z ≈ 3 km", xlabel = "x − x_m [km]", ylabel = "w [m/s]", limits = ((-40, 40), nothing), gridkw...)
    k3 = [argmin(abs.(grid.Zc[i, :] .- 3e3)) for i in 1:nx]
    xs = [X[i, 1] for i in sel]
    lines!(ax2, xs, [w_linear(grid.xc[i], grid.Zc[i, k3[i]]) for i in sel]; color = :gray40, linewidth = 2, linestyle = :dash, label = "linear theory")
    for f in forms
        rw[f] === nothing && continue
        f === :linear && continue
        col, mk = fsty[f]
        scatterlines!(ax2, xs, [rw[f][i, k3[i]] for i in sel]; color = col, marker = mk, markersize = 7, linewidth = 2, label = "RK4, " * formlabel[f])
    end
    axislegend(ax2; position = :lb, framevisible = false, labelsize = 11)
    save(joinpath(FIG, "agnesi_reference.png"), fig; px_per_unit = 2)
end

# (2) error against Δt: one panel per method, one line per form
fig = Figure(size = (1300, 640), backgroundcolor = :white)
for (j, me) in enumerate(methods)
    ax = Axis(fig[(j - 1) ÷ 4 + 1, (j - 1) % 4 + 1]; title = mlabel[me], xscale = log10, yscale = log10,
              xlabel = j > 4 ? "Δt [s]" : "", ylabel = (j - 1) % 4 == 0 ? "rel. L2 error of θ'" : "",
              xticks = [10, 20, 40, 60, 120, 240], limits = ((8, 300), (5e-4, 3)), gridkw...)
    for f in forms
        rs = sort([r for r in runs if r.form == f && r.meth == me && basic(r)]; by = r -> r.dt)
        isempty(rs) && continue
        col, mk = fsty[f]
        scatterlines!(ax, [r.dt for r in rs], [max(r.eθ, 6e-4) for r in rs]; color = col, marker = mk, markersize = 9, linewidth = 2,
                      strokecolor = :white, strokewidth = 1, label = formlabel[f])
        us = [u[3] for u in unstable if u[1] == f && u[2] == me && u[4] == nsub1(u[3])]
        isempty(us) || scatter!(ax, us, fill(2.0, length(us)); color = col, marker = :xcross, markersize = 11)
    end
end
Legend(fig[3, 1:4], [MarkerElement(color = fsty[f][1], marker = fsty[f][2], markersize = 10) for f in forms],
       [formlabel[f] for f in forms]; orientation = :horizontal, framevisible = false, labelsize = 13)
Label(fig[0, 1:4], "Agnesi, a = 10 km, HEVI, t = $(T) s: error of θ' against RK4 of the same operator (×: unstable)"; fontsize = 15, font = :bold)
save(joinpath(FIG, "agnesi_error_dt.png"), fig; px_per_unit = 2)

# (3) error against the number of small steps (Δt = 60 s), extended system and full pressure
# the 8 categorical slots in their validated order, with distinct markers
msty = Dict(:MIS2 => ("#2a78d6", :circle), :MIS3C => ("#eb6834", :utriangle), :MIS4a => ("#1baf7a", :rect), :MIS54 => ("#eda100", :diamond),
            :WSRK3 => ("#e87ba4", :xcross), :ETD3 => ("#008300", :dtriangle), :EB4 => ("#6250d6", :cross), :LRK4 => ("#e34948", :star5))
fig = Figure(size = (1300, 460), backgroundcolor = :white)
for (j, f) in enumerate((:exner, :gradp))
    local ax = Axis(fig[1, j]; title = "$(formlabel[f]), Δt = 60 s", xscale = log2, yscale = log10,
              xlabel = "n_sub (Δτ = 60 s/n_sub)", ylabel = j == 1 ? "rel. L2 error of θ'" : "", xticks = [14, 28, 56, 112],
              limits = ((12, 130), (1e-3, 0.3)), yticks = ([0.001, 0.003, 0.01, 0.03, 0.1], ["0.001", "0.003", "0.01", "0.03", "0.1"]), gridkw...)
    for me in methods
        rs = sort([r for r in runs if r.form == f && r.meth == me && r.dt == 60]; by = r -> r.nsub)
        col, mk = msty[me]
        isempty(rs) || scatterlines!(ax, [r.nsub for r in rs], [max(r.eθ, 1e-3) for r in rs]; color = col, marker = mk,
                                     markersize = 10, linewidth = 2, strokecolor = :white, strokewidth = 1)
        us = [u[4] for u in unstable if u[1] == f && u[2] == me && u[3] == 60.0]
        isempty(us) || scatter!(ax, us, fill(0.25, length(us)); color = col, marker = :xcross, markersize = 11)
    end
end
Legend(fig[1, 3], [MarkerElement(color = msty[me][1], marker = msty[me][2], markersize = 10) for me in methods],
       [mlabel[me] for me in methods]; framevisible = false, labelsize = 12)
Label(fig[0, 1:2], "Error against the number of small steps (×: unstable)"; fontsize = 15, font = :bold)
save(joinpath(FIG, "agnesi_error_nsub.png"), fig; px_per_unit = 2)

# (4) θ' difference fields at a large time step (full pressure)
dtl = 120.0
cand = [me for me in (:MIS2, :MIS54, :WSRK3, :EB4) if any(r -> r.form == :gradp && r.meth == me && r.dt == dtl && basic(r), runs)]
if rθ[:gradp] !== nothing && !isempty(cand)
    fig = Figure(size = (1300, 300 * cld(length(cand), 2)), backgroundcolor = :white)
    for (j, me) in enumerate(cand)
        r = runs[findfirst(r -> r.form == :gradp && r.meth == me && r.dt == dtl && basic(r), runs)]
        x = deserialize(joinpath(DIR, "agnesi_x_gradp_$(me)_$(round(Int, dtl))_$(r.nsub).jls"))
        d = θp(x) .- rθ[:gradp]
        local ax = Axis(fig[(j - 1) ÷ 2 + 1, 2((j - 1) % 2) + 1]; title = @sprintf("%s, Δt = %.0f s: θ' − θ'_RK4 [K] (error %.3f)", mlabel[me], dtl, r.eθ),
                  xlabel = "x − x_m [km]", ylabel = "z [km]", limits = (-60, 60, 0, 12))
        local cr = max(1e-4, maximum(abs, d[sel, :]))
        local sf = surface!(ax, X[sel, :], Z[sel, :], zeros(length(sel), nz); color = d[sel, :], colormap = :balance, colorrange = (-cr, cr), shading = NoShading)
        Colorbar(fig[(j - 1) ÷ 2 + 1, 2((j - 1) % 2) + 2], sf)
    end
    save(joinpath(FIG, "agnesi_diff_large_dt.png"), fig; px_per_unit = 2)
end

# (5) the acoustic content of w: MIS2 (Δt = 10 s) − RK4 in the whole channel, w and θ'
fn = joinpath(DIR, "agnesi_x_gradp_MIS2_10_$(nsub1(10)).jls")
if rθ[:gradp] !== nothing && isfile(fn)
    x = deserialize(fn)
    fig = Figure(size = (1300, 560), backgroundcolor = :white)
    for (j, (d, lab)) in enumerate(((w_c(x) .- rw[:gradp], "w − w_RK4 [m/s]"), (θp(x) .- rθ[:gradp], "θ' − θ'_RK4 [K]")))
        local ax = Axis(fig[j, 1]; title = "MIS2, Δt = 10 s, full pressure: $lab, whole channel (relative error in the box: " *
                  @sprintf("%.3f)", j == 1 ? relL2(w_c(x), rw[:gradp]) : relL2(θp(x), rθ[:gradp])),
                  xlabel = j == 2 ? "x − x_m [km]" : "", ylabel = "z [km]", limits = (-200, 200, 0, 15.6))
        local cr = maximum(abs, d)
        local sf = surface!(ax, (grid.xc .- xm) ./ 1e3 .* ones(1, nz), grid.Zc ./ 1e3, zeros(nx, nz); color = d, colormap = :balance,
                      colorrange = (-cr, cr), shading = NoShading)
        lines!(ax, [-30, 30, 30, -30, -30], [0, 0, 9, 9, 0]; color = :black, linewidth = 1.5, linestyle = :dash)
        Colorbar(fig[j, 2], sf)
    end
    save(joinpath(FIG, "agnesi_acoustic.png"), fig; px_per_unit = 2)
end
println("done")
