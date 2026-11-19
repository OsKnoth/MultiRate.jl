# Nonhydrostatic gravity wave of Skamarock & Klemp (1994, Mon. Wea. Rev. 122, 2623): channel 300 km × 10 km, U = 20 m/s,
# N = 0.01/s, θ' = Δθ0 sin(πz/H)/(1 + ((x - L/3)/a)²), Δθ0 = 0.01 K, a = 5 km, T = 3000 s, grid aspect ratio 1
# (Δx = Δz = 1 km: DG degree 5 on 60 × 2 elements of 5 km). MIS54 with the explicit Störmer–Verlet small step
# (docs/fast_2d, scheme S1; DGSEM, LMARS in ρu). Error of θ against RK4; θ' at T as in Fig. 1 of the paper.
#
#     [DTS=12,24] [NSUBS=..] julia --project=examples examples/sk94_gravity_wave.jl
include(joinpath(@__DIR__, "euler2d_common.jl"))

L, H, a, U, T, Δθ0 = 300e3, 10e3, 5e3, 20.0, 3000.0, 0.01
const DTS = parse.(Float64, split(get(ENV, "DTS", "12,24"), ","))
const FACS = parse.(Float64, split(get(ENV, "FACS", "1,0.8,0.6,0.5"), ","))   # small step / explicit limit (P = 5)
gas = Gas()
ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
base = build_model(:dg; L, H, P = 5, NEX = 60, NEZ = 2, ρr, Θr, gas)
qb = project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + Δθ0 * sin(π * z / H) / (1 + ((x - L / 3) / a)^2)); (ρ, ρ * U, 0.0, Θr(z))))
M = base.mesh
ms = E2.DGPressureModel(base; lmars = :momentum)            # standard
c = sound_speed_max(base, qb)
dτ0 = 0.85 * 0.52 * min(M.Lx / M.Nex, M.H / M.Nez) / ((2M.p + 1) * c)   # explicit Verlet limit (CFL table, P = 5)
nref = ceil(Int, T / (0.5dτ0))
tr = @elapsed qref = TS.run_rk4(ms, E2.pressure_state(ms, qb), T / nref, nref)
θf(q) = E2.field(base, q[1:base.nq], 4) ./ E2.field(base, q[1:base.nq], 1)
θref = θf(qref)
@printf("SK94 gravity wave, %s; explicit small step %.2f s; RK4 %d steps (%.0f s); max θ' %.2e K\n",
        grid_info(base), dτ0, nref, tr, maximum(θref .- θr.(M.Z)))
res = Dict{Tuple{Symbol,Float64,Float64},Any}()
for Δt in DTS, (lab, m) in ((:S1, ms),), f in FACS
    n = round(Int, T / Δt); nsub = ceil(Int, Δt / (f * dτ0))
    q = similar(qref)
    t = @elapsed try
        q = TS.run_frozen(m, E2.pressure_state(m, qb), Δt, n; tab = TS.mis_tableau(:MIS54), nsub, fast = :verlet)
    catch e
        e isa DomainError || rethrow(); fill!(q, NaN)
    end
    e = all(isfinite, q) ? maximum(abs.(θf(q) .- θref)) : Inf
    res[(lab, Δt, f)] = q
    @printf("MIS54 Δt = %4.1f s  %s  small step %.2f s (%.1f × limit) nsub %3d: %s (%.0f s)\n", Δt, lab, Δt / nsub, Δt / nsub / dτ0, nsub,
            isfinite(e) ? @sprintf("error θ %.2e K", e) : "failed", t)
end

# θ' at T (Fig. 1 of the paper): RK4 and S1 at Δt = DTS[1] with the first stable small step. DG nodes on element faces
# exist twice: average them for the contour plot.
function unique_points(X, Z, V)
    d = Dict{Tuple{Float64,Float64},Tuple{Float64,Int}}()
    for (x, z, v) in zip(X, Z, V)
        k = (round(x; digits = 6), round(z; digits = 6)); s, n = get(d, k, (0.0, 0)); d[k] = (s + v, n + 1)
    end
    ks = collect(keys(d))
    return first.(ks), last.(ks), [d[k][1] / d[k][2] for k in ks]
end
firststable(lab) = (for f in FACS; q = res[(lab, DTS[1], f)]; all(isfinite, q) && return (f, q); end; (NaN, fill(NaN, 1)))
fS, qS = firststable(:S1)
X = vec(M.X) ./ 1e3; Z = vec(M.Z) ./ 1e3
lev = vcat(collect(-5e-3:5e-4:-5e-4), collect(5e-4:5e-4:5e-3))
fig = Figure(; size = (1100, 760))
panels = (("θ' [K], RK4", qref, false), (@sprintf("θ' [K], S1, Δt = %g s, small step %.1f × limit", DTS[1], fS), qS, false),
          ("θ(S1) - θ(RK4) [K]", qS, true))
for (r, (lab, q, diff)) in enumerate(panels)
    all(isfinite, q) || continue
    v = diff ? vec(θf(q) .- θref) : vec(θf(q) .- θr.(M.Z))
    x, z, u = unique_points(X, Z, v)
    # differences: contour interval relative to their own maximum (10 levels), so that the small S1 error is visible
    lv = diff ? (d = maximum(abs, u) / 10; d > 0 ? collect(-10d:d:10d) : [0.0]) : lev
    ax = Axis(fig[r, 1]; title = (diff ? @sprintf("%s at t = 3000 s, max %.2e, contour interval %.1e", lab, maximum(abs, u), maximum(abs, u) / 10) :
                                         lab * " at t = 3000 s"), xlabel = "x [km]", ylabel = "z [km]")
    contour!(ax, x, z, u; levels = filter(l -> l > 0, lv), color = :black)
    contour!(ax, x, z, u; levels = filter(l -> l < 0, lv), color = :black, linestyle = :dash)
end
save(joinpath(@__DIR__, "sk94_gravity_wave.png"), fig)
println("wrote examples/sk94_gravity_wave.png")
