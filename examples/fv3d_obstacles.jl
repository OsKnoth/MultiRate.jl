# 3D finite volumes with cut cells (MultiRate.FV3D): stratified flow over obstacles
#
#     julia --project=examples examples/fv3d_obstacles.jl [T]
#
# U = 10 m/s, N = 0.01 1/s, 64 km × 64 km × 10 km, 32 × 32 × 20 cells (Δx = 2 km, Δz = 500 m),
# obstacles: a bell-shaped hill (h = 400 m, a = 4 km, cut cells), a cube 8 km × 8 km × 1 km
# aligned with the grid (only full and solid cells) and the same cube shifted by 0.3 Δx and
# 1.1 km high (cut cells). Initial state: discretely balanced column (`hydrostatic_column`, no
# background state in the equations) with uniform U. Frozen MIS54, HEVI fast loop, product-form
# pressure, sponge on W above 7 km. Output: fv3d_obstacles.png (w at z = 1.75 km and in the
# x–z plane through the centre), table on stdout.

using MultiRate, Printf, CairoMakie
const TS = MultiRate.TimeStepping
const F3 = MultiRate.FV3D
const CM = MultiRate.Common

T = isempty(ARGS) ? 3600.0 : parse(Float64, ARGS[1])
gas = CM.Gas()
ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
L, H, U, nx, nz = 64e3, 10e3, 10.0, 32, 20
F3.FV3D_PRESSURE_FORM[] = :product
hill(x, y) = 400.0 / (1 + ((x - L / 2)^2 + (y - L / 2)^2) / 4e3^2)^1.5
cases = (("bell hill (cut cells)", (; zs = hill)),
         ("cube aligned with the grid", (; box = (28e3, 36e3, 28e3, 36e3, 1000.0))),
         ("cube shifted (cut cells)", (; box = (28.6e3, 36.6e3, 28.6e3, 36.6e3, 1100.0))))
Δt = 30.0
fig = Figure(size = (1200, 760))
for (n, (name, geo)) in enumerate(cases)
    g = F3.Grid3D(nx, nx, nz, L, L, H; geo...)
    m = F3.with_sponge(F3.Model3D(g, gas, ρr, Θr), zeros(F3.nunknowns(g)); zD = 7e3, αtop = 0.02)
    ρb = F3.hydrostatic_column(m, Θr; ρbottom = ρr(g.zc[1]))
    q0 = F3.project3d(m, (x, y, z) -> (k = F3.level(g, z); (ρb[k], U * ρb[k], 0.0, 0.0, Θr(g.zc[k]))))
    c = F3.sound_speed_max(m, q0)
    nsub = ceil(Int, Δt / (0.8 * g.dx / (c + U)))
    t = @elapsed q = TS.run_frozen(m, q0, Δt, round(Int, T / Δt); tab = TS.mis_tableau(:MIS54), nsub, fast = :hevi)
    w = F3.vertical_velocity(m, q)                       # faces k = 2 … nz, z = zf[k]
    zw = g.zf[2:nz]
    kw = findfirst(>=(1.75e3), zw)
    @printf("%-28s cut %4d solid %3d | max|w| %.3f m/s (above 2 km %.3f) | Δmass %.1e ΔΘ %.1e | %.0f s\n", name,
            count(0 .< g.Vf .< 1), count(g.Vf .== 0), maximum(abs, w), maximum(abs, w[:, :, zw .>= 2e3]),
            F3.total3d(m, q, :ρ) / F3.total3d(m, q0, :ρ) - 1, F3.total3d(m, q, :Θ) / F3.total3d(m, q0, :Θ) - 1, t)
    ax1 = Axis(fig[1, n]; title = "$name\nw at z = $(round(Int, zw[kw])) m", xlabel = "x [km]", ylabel = "y [km]", aspect = 1)
    hm = heatmap!(ax1, g.xc ./ 1e3, g.yc ./ 1e3, w[:, :, kw]; colormap = :balance, colorrange = (-1, 1))
    jc = nx ÷ 2
    ax2 = Axis(fig[2, n]; title = "w, x–z plane y = $(round(g.yc[jc] / 1e3; digits = 1)) km", xlabel = "x [km]", ylabel = "z [km]")
    heatmap!(ax2, g.xc ./ 1e3, zw ./ 1e3, w[:, jc, :]; colormap = :balance, colorrange = (-1, 1))
    n == 3 && Colorbar(fig[1:2, 4], hm; label = "w [m/s]")
end
F3.FV3D_PRESSURE_FORM[] = :gradp
save("fv3d_obstacles.png", fig)
println("wrote fv3d_obstacles.png")
