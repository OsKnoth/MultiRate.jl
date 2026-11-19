# Hydrostatic gravity wave of Skamarock & Klemp (1994, Mon. Wea. Rev. 122, 2623, Fig. 3): channel 6000 km × 10 km,
# U = 20 m/s, N = 0.01/s, θ' = Δθ0 sin(πz/H)/(1 + ((x - L/3)/a)²), Δθ0 = 0.01 K, a = 100 km, T = 60000 s,
# Δx = 20 km, Δz = 1 km (DG degree 5 on 60 × 2 elements of 100 km × 5 km). Standard formulation (DGSEM, LMARS in ρu),
# MIS54 with scheme S2 of docs/fast_2d (Störmer–Verlet, z implicit, :cs) at the horizontal limit. The column system
# is solved as in hybridised DG (vsolver = :hdg): the interface momentum M̂ = s m_f of the η-faces is introduced
# only in the solve, the elements are eliminated locally and a tridiagonal system in M̂ remains per node column;
# compared with the sparse LU of the column system (vsolver = :lu).
#
#     [DT=200] [NSUBFAC=1] [OFFCENTER=0.1] [REF=1] julia --project=examples examples/sk94_long_channel.jl
#
# NSUBFAC: small step as a fraction of the explicit horizontal limit. REF=1: RK4 reference of the same model
# (≈ 2·10⁵ steps at the vertical limit, ≈ 75 min), saved to examples/sk94_long_ref.bin and reused afterwards.
include(joinpath(@__DIR__, "euler2d_common.jl"))
using Serialization

L, H, a, U, T, Δθ0 = 6000e3, 10e3, 100e3, 20.0, 60000.0, 0.01
Δt = parse(Float64, get(ENV, "DT", "200"))
fac = parse(Float64, get(ENV, "NSUBFAC", "1"))
ε = parse(Float64, get(ENV, "OFFCENTER", "0.1"))
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
base = build_model(:dg; L, H, P = 5, NEX = 60, NEZ = 2, ρr, Θr, gas)
qb = project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + Δθ0 * sin(π * z / H) / (1 + ((x - L / 3) / a)^2)); (ρ, ρ * U, 0.0, Θr(z))))
M = base.mesh
c = sound_speed_max(base, qb)
dτh = 0.85 * 0.63 * (M.Lx / M.Nex) / ((2M.p + 1) * c)    # explicit horizontal limit (degree 5)
dτv = 0.85 * 0.52 * (M.H / M.Nez) / ((2M.p + 1) * c)     # explicit vertical limit
nsub = ceil(Int, Δt / (fac * dτh)); n = round(Int, T / Δt)
θf(q) = E2.field(base, q[1:base.nq], 4) ./ E2.field(base, q[1:base.nq], 1)
@printf("SK94 hydrostatic wave, %s; horizontal limit %.2f s, vertical limit %.2f s\n", grid_info(base), dτh, dτv)

res = Dict{Tuple{Symbol,Bool},Vector{Float64}}()
for (vs, hp) in ((:lu, true), (:hdg, true), (:hdg, false))
    m = E2.DGPressureModel(base; lmars = :momentum, vsolver = vs, hpredict = hp)
    t = @elapsed q = TS.run_frozen(m, E2.pressure_state(m, qb), Δt, n; tab = TS.mis_tableau(:MIS54), nsub, fast = :cs,
                                  offcenter = ε, smdiv = 0.0)
    res[(vs, hp)] = q
    θp = θf(q) .- θr.(M.Z)
    @printf("S2, column solve %-4s, %s: MIS54 Δt = %.0f s, nsub %d (small step %.1f s): max θ' %.3e, min θ' %.3e K (%.0f s)\n",
            vs, hp ? "midpoint predictor of D_M^x" : "D_M^x explicit         ", Δt, nsub, Δt / nsub, maximum(θp), minimum(θp), t)
end
@printf("max |θ(hdg) - θ(lu)| = %.2e K\n", maximum(abs, θf(res[(:hdg, true)]) .- θf(res[(:lu, true)])))

reffile = joinpath(@__DIR__, "sk94_long_ref.bin")
ms = E2.DGPressureModel(base; lmars = :momentum)
nref = ceil(Int, T / (0.5dτv))
if get(ENV, "REF", "0") == "1"
    tr = @elapsed qref = TS.run_rk4(ms, E2.pressure_state(ms, qb), T / nref, nref)
    serialize(reffile, qref)
    @printf("RK4 reference: %d steps (%.0f s), saved to %s\n", nref, tr, reffile)
end
qref = isfile(reffile) ? deserialize(reffile) : nothing
if qref !== nothing
    for hp in (true, false)
        @printf("error θ against RK4 (%s): %.3e K\n", hp ? "midpoint predictor" : "D_M^x explicit", maximum(abs, θf(res[(:hdg, hp)]) .- θf(qref)))
    end
end

# θ' at T (Fig. 3 of the paper, contour interval 5e-4 K, negative dashed) and the difference to RK4;
# DG nodes on element faces exist twice and are averaged
function unique_points(X, Z, V)
    d = Dict{Tuple{Float64,Float64},Tuple{Float64,Int}}()
    for (x, z, v) in zip(X, Z, V)
        k = (round(x; digits = 6), round(z; digits = 6)); s, k2 = get(d, k, (0.0, 0)); d[k] = (s + v, k2 + 1)
    end
    ks = collect(keys(d))
    return first.(ks), last.(ks), [d[k][1] / d[k][2] for k in ks]
end
X = vec(M.X) ./ 1e3; Z = vec(M.Z) ./ 1e3
q = res[(:hdg, true)]
panels = Any[(@sprintf("θ' [K] at t = %.0f s, S2 (HDG column solve), MIS54 Δt = %.0f s, nsub = %d", T, Δt, nsub), vec(θf(q) .- θr.(M.Z)), false)]
qref !== nothing && push!(panels, ("θ' [K], RK4", vec(θf(qref) .- θr.(M.Z)), false), ("θ(S2) - θ(RK4) [K]", vec(θf(q) .- θf(qref)), true))
fig = Figure(; size = (1300, 300 * length(panels)))
for (r, (lab, v, diff)) in enumerate(panels)
    x, z, u = unique_points(X, Z, v)
    lv = diff ? (d = maximum(abs, u) / 10; collect(-10d:d:10d)) : vcat(collect(-5e-3:5e-4:-5e-4), collect(5e-4:5e-4:5e-3))
    ax = Axis(fig[r, 1]; title = diff ? @sprintf("%s, max %.2e, contour interval %.1e", lab, maximum(abs, u), maximum(abs, u) / 10) : lab,
              xlabel = "x [km]", ylabel = "z [km]")
    contour!(ax, x, z, u; levels = filter(l -> l > 0, lv), color = :black)
    contour!(ax, x, z, u; levels = filter(l -> l < 0, lv), color = :black, linestyle = :dash)
    xlims!(ax, 0, L / 1e3)
end
save(joinpath(@__DIR__, "sk94_long_channel.png"), fig)
println("wrote examples/sk94_long_channel.png")
