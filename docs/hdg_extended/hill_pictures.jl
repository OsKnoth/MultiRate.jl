# Pictures for docs/hdg_extended: DG gravity wave over the Agnesi hill (300 km × 10 km, 30 × 4 elements of degree 4,
# U = 20 m/s, h0 = 400 m, a = 10 km), T = 3000 s, LMARS in the momentum form. RK4 reference, MIS54 with :cs
# (LMARS) and with the interface momentum (VSPLIT=full|contravariant: vertical split of both) (κ = 1/2, diagonal of D_Y implicit in the horizontal momentum step),
# Δt = 100 s, 59 small steps.
#     julia --project=examples docs/hdg_extended/hill_pictures.jl
include(joinpath(@__DIR__, "..", "..", "examples", "euler2d_common.jl"))

L, H, a, U, T = 300e3, 10e3, 5e3, 20.0, 3000.0
gas = Gas(); ρr, Θr, θr = stratified_reference(gas; θ0 = 300.0, N = 0.01)
zsf(x) = 400.0 / (1 + ((x - L / 2) / 10e3)^2)
dzs(x) = -400.0 * 2 * (x - L / 2) / 10e3^2 / (1 + ((x - L / 2) / 10e3)^2)^2
base = build_model(:dg; L, H, P = 4, NEX = 30, NEZ = 4, ρr, Θr, gas, zs = zsf)
qb = project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / H) / (1 + ((x - L / 3) / a)^2));
                                 (ρ, ρ * U, ρ * U * dzs(x) * (H - z) / (H - zsf(x)), Θr(z))))
M = base.mesh; np = M.np; nqb = base.nq
const VSPLIT = Symbol(get(ENV, "VSPLIT", "full"))          # full | contravariant
mm = E2.DGPressureModel(base; lmars = :momentum)                        # reference model (RK4)
mc = E2.DGPressureModel(base; lmars = :momentum, vsplit = VSPLIT)      # :cs without interface momentum
mi = E2.DGPressureModel(base; interface = true, vsplit = VSPLIT)
c = sound_speed_max(base, qb)
dτh = 0.85 * 0.63 * (M.Lx / M.Nex) / ((2M.p + 1) * c)
Δt = 100.0; n = round(Int, T / Δt); nsub = ceil(Int, Δt / dτh)
dτv = fast_step(base, qb, :verlet); nref = ceil(Int, T / (0.5dτv))
qref = TS.run_rk4(mm, E2.pressure_state(mm, qb), T / nref, nref)
run(m) = TS.run_frozen(m, E2.pressure_state(m, qb), Δt, n; tab = TS.mis_tableau(:MIS54), nsub, fast = :cs, offcenter = 0.1, smdiv = 0.1)
qc = run(mc); qi = run(mi)

θ(q) = E2.field(base, q[1:nqb], 4) ./ E2.field(base, q[1:nqb], 1)
w(q) = E2.field(base, q[1:nqb], 3) ./ E2.field(base, q[1:nqb], 1)
θp(q) = (t = θ(q); t .- θr.(M.Z))
X = vec(M.X) ./ 1e3; Z = vec(M.Z) ./ 1e3
function panel!(fig, pos, vals, title; sym = true)
    ax = Axis(fig[pos...]; title, xlabel = "x [km]", ylabel = "z [km]")
    v = vec(vals); a = maximum(abs, v); a = a > 0 ? a : 1.0
    lv = sym ? range(-a, a; length = 21) : range(extrema(v)...; length = 21)
    p = tricontourf!(ax, X, Z, v; levels = lv, colormap = :balance)
    Colorbar(fig[pos[1], pos[2]+1], p)
    lines!(ax, range(0, L; length = 400) ./ 1e3, zsf.(range(0, L; length = 400)) ./ 1e3; color = :black)
    return ax
end
err(q) = maximum(abs, θ(q) .- θ(qref))
fig = Figure(; size = (1400, 1050))
for (r, (lab, q)) in enumerate((("RK4 (LMARS in ρu)", qref), (@sprintf(":cs (%s), n_sub = %d", VSPLIT, nsub), qc),
                                 (@sprintf("interface momentum, κ = 1/2, n_sub = %d", nsub), qi)))
    panel!(fig, (r, 1), θp(q), "θ' [K], " * lab)
    panel!(fig, (r, 3), w(q), "w [m/s], " * lab)
end
save(joinpath(@__DIR__, "hill_solutions_$(VSPLIT).png"), fig)

fig = Figure(; size = (1400, 720))
panel!(fig, (1, 1), θ(qc) .- θ(qref), @sprintf("θ(:cs) - θ(RK4) [K], max %.2e", err(qc)))
panel!(fig, (1, 3), w(qc) .- w(qref), "w(:cs) - w(RK4) [m/s]")
panel!(fig, (2, 1), θ(qi) .- θ(qref), @sprintf("θ(interface) - θ(RK4) [K], max %.2e", err(qi)))
panel!(fig, (2, 3), w(qi) .- w(qref), "w(interface) - w(RK4) [m/s]")
save(joinpath(@__DIR__, "hill_differences_$(VSPLIT).png"), fig)

# contravariant vertical interface flux M̂ = ñ·m̂ on the η-faces between the first and the second element layer:
# one fast loop (Δt, n_sub small steps, :cs) from the final state of the interface run with its stage coefficients,
# M̂ against its LMARS value s m_f(z) (same frozen coefficients) and ñ·m of the elements below and above the face
Cs = zeros(TS.frozen_ncoef(mi)); Ss = zeros(mi.nq); TS.frozen_data!(Cs, Ss, mi, qi)
Zf = copy(qi); TS.frozen_stage_begin!(mi, Cs, Zf)
TS.frozen_fast_solve!(Zf, mi, Cs, Ss, Δt / nsub, nsub, zeros(mi.nq); method = :cs, tmp = ntuple(_ -> zeros(mi.nq), 6),
                      offcenter = 0.1, smdiv = 0.1)
xs, Mh, mb, ma, mstar = Float64[], Float64[], Float64[], Float64[], Float64[]
E2._foreach_face_dir(M) do fi, dir, kind, iL, jL, eL, iR, jR, eR, ñx, ñz, we
    (dir == 2 && kind === :interior && jL == np && eL <= M.Nex) || return nothing
    o = E2._fc(M, fi); sf = hypot(ñx, ñz)
    nb = ñx * Zf[E2.node_iq(np, eL, 2, iL, jL)] + ñz * Zf[E2.node_iq(np, eL, 3, iL, jL)]
    na = ñx * Zf[E2.node_iq(np, eR, 2, iR, jR)] + ñz * Zf[E2.node_iq(np, eR, 3, iR, jR)]
    push!(xs, M.X[iL, jL, eL] / 1e3); push!(Mh, Zf[mi.mhat[fi]]); push!(mb, nb); push!(ma, na)
    push!(mstar, Cs[o+1] * nb + Cs[o+2] * na - sf * Cs[o+3] * (Zf[E2.ip_dg(mi, eR, iR, jR)] - Zf[E2.ip_dg(mi, eL, iL, jL)]))
    nothing
end
p = sortperm(xs)
fig = Figure(; size = (1200, 700))
ax = Axis(fig[1, 1]; title = "contravariant vertical mass flux ñ·m at the first interior η-face (z ≈ 2.5 km), after one fast loop of Δt",
          xlabel = "x [km]", ylabel = "ñ·m (contravariant)")
lines!(ax, xs[p], Mh[p]; label = "M̂ (prognostic)", linewidth = 2)
lines!(ax, xs[p], mstar[p]; label = "s m_f(q) (LMARS value)", linestyle = :dash)
scatter!(ax, xs[p], mb[p]; markersize = 4, label = "ñ·m below")
scatter!(ax, xs[p], ma[p]; markersize = 4, label = "ñ·m above")
axislegend(ax; position = :lt)
ax2 = Axis(fig[2, 1]; xlabel = "x [km]", ylabel = "M̂ - s m_f(q)")
lines!(ax2, xs[p], Mh[p] .- mstar[p])
save(joinpath(@__DIR__, "interface_flux_$(VSPLIT).png"), fig)
@printf("error θ: :cs %.2e K, interface %.2e K; max |M̂ - s m_f| %.2e (max |M̂| %.2e)\n", err(qc), err(qi),
        maximum(abs, Mh .- mstar), maximum(abs, Mh))
