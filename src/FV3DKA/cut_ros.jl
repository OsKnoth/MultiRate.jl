# Rosenbrock W-methods for the cut-cell model CutKA (TimeStepping/rosenbrock.jl, FV3D/rosenbrock.jl): W = J_ac + J_adv
# at qⁿ, matrix-free, applied by `ka_vertical!` on the twin model of the all-masked grid (every fluid cell masked:
# the horizontal acoustics through all faces, the upwind Θ transport and momentum advection on all cells) with the
# coefficients C(qⁿ) of the model. Stage solves:
#   ros = :amf   (I - γΔt J_ac) x = r (GMRES, column preconditioner), then (I - γΔt J_adv) k = x (GMRES, none);
#   ros = :full  (I - γΔt (J_ac + J_adv)) k = r (GMRES, column preconditioner);
# after each solve ρ and Θ from the fluxes of the solution (conserved for any tolerance). The preconditioner of the
# acoustic factor is the column solve (vertical acoustics exact, horizontal acoustics not): with implicit horizontal
# acoustics its iterations grow with c Δt/Δx; the slot for a geometric multigrid (horizontal coarsening with volume-
# averaged cut-cell fractions, the column solve as line smoother, as ASAM JacAccGrav) is `prec` of the twin model.

"""Rosenbrock data of a CutKA model: the all-masked twin model, frozen coefficients at qⁿ, stage and GMRES work."""
struct CutRos{M,A}
    ma::M
    Cf::A; S::A
    Cf2::A; S2::A; tmp::A
    k::Vector{A}; Y::A; f::A; r::A; x::A
    gm::CutGMRES
    its::Vector{Int}                               # GMRES iterations per solve (acoustic, advection alternating for :amf)
end
"""
    CutRos(m::CutKA, s; restart = 30, aprec = :columns, ν = 2, νc = 20)

Rosenbrock work for s stages; the twin model on the all-masked grid (same tile, decomposition, sponge). `aprec`: the
preconditioner of the acoustic factor, `:columns` (column solve) or `:mg` (Schur + geometric multigrid, cut_mg.jl,
ν zebra sweeps per level, νc on the coarsest).
"""
function CutRos(m::CutKA, s::Int; restart::Int = 30, aprec::Symbol = :columns, ν::Int = 2, νc::Int = 20)
    g = m.grid
    ga = CutGrid(FV3D._allmasked(g.g3); px = g.px, py = g.py, cx = g.i0 ÷ g.nxl, cy = g.j0 ÷ g.nyl, backend = KA.get_backend(g))
    ma = CutKA(ga, m.dec; gas = m.gas, jform = m.complete ? :conservative : :partial, prec = :columns)
    copyto!(ma.damp, m.damp)
    aprec === :mg && (ma.mgp[] = CutMG(ma; ν, νc))
    z() = ka_zeros(m); zc() = ka_zeros(m, NCC)
    return CutRos(ma, zc(), z(), zc(), z(), z(), [z() for _ in 1:s], z(), z(), z(), z(), CutGMRES(ma, restart), Int[])
end

"""y = (I - c J_op)⁻¹ r (GMRES on the twin model, initial guess the preconditioner), ρ and Θ of y from the fluxes."""
function _ros_solve!(y, w::CutRos, c, r, op::Symbol; tol)
    ma, gm = w.ma, w.gm
    g = ma.grid
    _pack!(gm.b, ma, r)
    _cut_prec!(gm.x, ma, w.Cf, c, gm.b, gm; op)
    it, _ = _cut_gmres!(gm.x, ma, w.Cf, c, gm.b, gm; tol, op)
    push!(w.its, it)
    fill!(y, 0.0); _unpack!(y, ma, gm.x)
    ka_vertical!(gm.Y, ma, w.Cf, y; halo = true, parts = op === :full ? :all : op, sponge = op !== :advection)
    _launch!(ma, _kc_conserve!, (g.nxl, g.nyl, g.nz), y, r, gm.Y, c)
    return y
end

"""
    ka_ros_step!(m::CutKA, Q, Δt, tab::RosTableau, w::CutRos; tol = 1e-8, ros = :amf)

One Rosenbrock step of the cut-cell model (TimeStepping `ros_step!`), in place: W at qⁿ, the stages
rᵢ = Δt N(Yᵢ) + Σ βᵢⱼ kⱼ, kᵢ = (I - γΔt W)⁻¹ rᵢ (AMF or full), Yᵢ₊₁ = qⁿ + Σ αᵢⱼ kⱼ.
"""
function ka_ros_step!(m::CutKA, Q, Δt, tab::RosTableau, w::CutRos; tol::Real = 1e-8, ros::Symbol = :amf)
    ros in (:amf, :full) || error("ros: :amf or :full")
    γh = tab.γ * Δt
    ka_frozen_data!(w.Cf, w.S, m, Q)                       # coefficients of W at qⁿ (exchanges the halo of Q)
    w.ma.mgp[] === nothing || mg_setup!(w.ma.mgp[], w.ma, w.Cf, γh)
    w.Y .= Q
    for i in 1:tab.s
        ka_rhs!(w.f, m, w.Y, w.Cf2, w.S2, w.tmp)
        @. w.r = Δt * w.f
        for j in 1:i-1
            b = tab.β[i, j]
            b != 0 && (@. w.r += b * w.k[j])
        end
        if ros === :amf
            _ros_solve!(w.x, w, γh, w.r, :acoustic; tol = Float64(tol))
            _ros_solve!(w.k[i], w, γh, w.x, :advection; tol = Float64(tol))
        else
            _ros_solve!(w.k[i], w, γh, w.r, :full; tol = Float64(tol))
        end
        w.Y .= Q
        for j in 1:i
            a = tab.α[i, j]
            a != 0 && (@. w.Y += a * w.k[j])
        end
    end
    Q .= w.Y
    return Q
end

"""nsteps Rosenbrock steps of the cut-cell model (in place); `monitor(n, Q)` every `every` steps."""
function ka_ros_run!(m::CutKA, Q, Δt, nsteps; tab::RosTableau, tol::Real = 1e-8, ros::Symbol = :amf, monitor = nothing,
                     every::Int = 1, aprec::Symbol = :columns)
    w = CutRos(m, tab.s; aprec)
    for n in 1:nsteps
        ka_ros_step!(m, Q, Δt, tab, w; tol, ros)
        monitor !== nothing && n % every == 0 && monitor(n, Q)
    end
    return Q, w
end
