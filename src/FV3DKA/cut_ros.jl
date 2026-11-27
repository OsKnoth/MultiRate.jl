# Rosenbrock W-methods for the cut-cell model CutKA (TimeStepping/rosenbrock.jl, FV3D/rosenbrock.jl): W = J_ac + J_adv
# at qⁿ, matrix-free, applied by `ka_vertical!` on the twin model of the all-masked grid (every fluid cell masked:
# the horizontal acoustics through all faces, the upwind Θ transport and momentum advection on all cells) with the
# coefficients C(qⁿ) of the model. Stage solves:
#   ros = :amf   (I - γΔt J_ac) x = r (GMRES, column preconditioner or multigrid), then (I - γΔt J_adv) k = x
#                (GMRES, no preconditioner or `adprec`: the advection factor of the masked AMF on the all-masked twin,
#                red–black Gauss–Seidel on the upwind stencil T for Θ and the shifted momenta, cut_mask.jl);
#   ros = :full  (I - γΔt (J_ac + J_adv)) k = r (GMRES, column preconditioner);
# after each solve ρ and Θ from the fluxes of the solution (conserved for any tolerance). The preconditioner of the
# acoustic factor is the column solve (vertical acoustics exact, horizontal acoustics not): with implicit horizontal
# acoustics its iterations grow with c Δt/Δx; the slot for a geometric multigrid (horizontal coarsening with volume-
# averaged cut-cell fractions, the column solve as line smoother, as ASAM JacAccGrav) is `prec` of the twin model.

"""Rosenbrock data of a CutKA model: the all-masked twin model, frozen coefficients at qⁿ, stage and GMRES work."""
mutable struct CutRos{M,A}
    ma::M
    Cf::A; S::A
    Cf2::A; S2::A; tmp::A
    k::Vector{A}; Y::A; f::A; r::A; x::A
    gm::CutGMRES
    its::Vector{Int}                               # GMRES iterations per solve (acoustic, advection alternating for :amf)
    tsec::Vector{Float64}                          # wall time: setup (W, multigrid, stencils), N(Y), acoustic, advection
end
"""
    CutRos(m::CutKA, s; restart = 30, aprec = :columns, ν = 2, νc = 20, aggcols = 4096, adprec = :none, sweeps = 2,
           admom = :shift)

Rosenbrock work for s stages; the twin model on the all-masked grid (same tile, decomposition, sponge). `aprec`: the
preconditioner of the acoustic factor, `:columns` (column solve) or `:mg` (Schur + geometric multigrid, cut_mg.jl,
ν zebra sweeps per level, νc on the coarsest; with MPI the levels of at most `aggcols` global columns agglomerated on
every rank, 0: never). `adprec`: the preconditioner of the advection factor, `:none`, `:gs`
(`sweeps` red–black Gauss–Seidel sweeps on T) or `:jacobi` (`sweeps` damped Jacobi sweeps); `admom`: the momenta in it
(`:shift`: T on the shifted cell momenta, back to the faces; `:identity`).
"""
function CutRos(m::CutKA, s::Int; restart::Int = 30, aprec::Symbol = :columns, ν::Int = 2, νc::Int = 20,
                aggcols::Int = 4096, adprec::Symbol = :none, sweeps::Int = 2, admom::Symbol = :shift)
    adprec in (:none, :gs, :jacobi) || error("adprec: :none, :gs or :jacobi")
    g = m.grid
    ga = CutGrid(FV3D._allmasked(g.g3); px = g.px, py = g.py, cx = g.i0 ÷ g.nxl, cy = g.j0 ÷ g.nyl, backend = KA.get_backend(g))
    ma = CutKA(ga, m.dec; gas = m.gas, jform = m.complete ? :conservative : :partial, prec = :columns)
    copyto!(ma.damp, m.damp)
    aprec === :mg && (ma.mgp[] = CutMG(ma; ν, νc, aggcols))
    adprec === :none || (ma.mprec[] = MaskAMF(ma; acoustic = :skip, advection = adprec, momenta = admom, sweeps, maxit = sweeps))
    z() = ka_zeros(m); zc() = ka_zeros(m, NCC)
    return CutRos(ma, zc(), z(), zc(), z(), z(), [z() for _ in 1:s], z(), z(), z(), z(), CutGMRES(ma, restart), Int[], zeros(4))
end

"""
    ros_energy_norm!(w::CutRos; θ0 = 300.0)

The stage solves of w in the acoustic energy norm (`energy_norm!`): residuals in pascal; `ka_ros_step!(…; atol)`
then is an RMS pressure residual (e.g. 1e-4 ρU²/2: the low-Mach pressure variations converge whatever the background
pressure), `tol` the relative drop.
"""
ros_energy_norm!(w::CutRos; θ0::Real = 300.0) = (w.gm = energy_norm!(w.gm, w.ma; θ0); w)

"""y = (I - c J_op)⁻¹ r (GMRES on the twin model, initial guess the preconditioner), ρ and Θ of y from the fluxes;
`atol`: absolute RMS residual (2-norm / √(number of unknowns))."""
function _ros_solve!(y, w::CutRos, c, r, op::Symbol; tol, atol = 0.0)
    ma, gm = w.ma, w.gm
    g = ma.grid
    _pack!(gm.b, ma, r)
    _cut_prec!(gm.x, ma, w.Cf, c, gm.b, gm; op)
    na = g.nx * g.ny * g.nz * NVC
    it, _ = _cut_gmres!(gm.x, ma, w.Cf, c, gm.b, gm; tol, atol = atol * sqrt(na), op)
    push!(w.its, it)
    fill!(y, 0.0); _unpack!(y, ma, gm.x)
    ka_vertical!(gm.Y, ma, w.Cf, y; halo = true, parts = op === :full ? :all : op, sponge = op !== :advection)
    _launch!(ma, _kc_conserve!, (g.nxl, g.nyl, g.nz), y, r, gm.Y, c)
    return y
end

"""
    ka_ros_step!(m::CutKA, Q, Δt, tab::RosTableau, w::CutRos; tol = 1e-8, atol = 0.0, ros = :amf)

One Rosenbrock step of the cut-cell model (TimeStepping `ros_step!`), in place: W at qⁿ, the stages
rᵢ = Δt N(Yᵢ) + Σ βᵢⱼ kⱼ, kᵢ = (I - γΔt W)⁻¹ rᵢ (AMF or full), Yᵢ₊₁ = qⁿ + Σ αᵢⱼ kⱼ.
"""
function ka_ros_step!(m::CutKA, Q, Δt, tab::RosTableau, w::CutRos; tol::Real = 1e-8, atol::Real = 0.0, ros::Symbol = :amf)
    ros in (:amf, :full) || error("ros: :amf or :full")
    γh = tab.γ * Δt
    be = KA.get_backend(m)
    t = time()
    ka_frozen_data!(w.Cf, w.S, m, Q)                       # coefficients of W at qⁿ (exchanges the halo of Q)
    w.ma.mgp[] === nothing || mg_setup!(w.ma.mgp[], w.ma, w.Cf, γh)
    w.ma.mprec[] === nothing || (mask_setup!(w.ma.mprec[], w.ma, w.Cf, γh); empty!(w.ma.mprec[].its))
    KA.synchronize(be); w.tsec[1] += time() - t
    w.Y .= Q
    for i in 1:tab.s
        t = time()
        ka_rhs!(w.f, m, w.Y, w.Cf2, w.S2, w.tmp)
        @. w.r = Δt * w.f
        for j in 1:i-1
            b = tab.β[i, j]
            b != 0 && (@. w.r += b * w.k[j])
        end
        if ros === :amf
            KA.synchronize(be); t1 = time(); w.tsec[2] += t1 - t
            _ros_solve!(w.x, w, γh, w.r, :acoustic; tol = Float64(tol), atol = Float64(atol))
            KA.synchronize(be); t2 = time(); w.tsec[3] += t2 - t1
            _ros_solve!(w.k[i], w, γh, w.x, :advection; tol = Float64(tol), atol = Float64(atol))
            KA.synchronize(be); w.tsec[4] += time() - t2
        else
            _ros_solve!(w.k[i], w, γh, w.r, :full; tol = Float64(tol), atol = Float64(atol))
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
                     every::Int = 1, aprec::Symbol = :columns, adprec::Symbol = :none)
    w = CutRos(m, tab.s; aprec, adprec)
    for n in 1:nsteps
        ka_ros_step!(m, Q, Δt, tab, w; tol, ros)
        monitor !== nothing && n % every == 0 && monitor(n, Q)
    end
    return Q, w
end
