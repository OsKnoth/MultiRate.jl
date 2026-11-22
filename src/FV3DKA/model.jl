# Model: grid tile, decomposition, gas, options, work arrays. State Q[I, J, k, v] (nxh, nyh, nz+1, NV): cell
# variables at k = 1 … nz, W at the η-faces k = 2 … nz (k = 1 and nz+1: walls, zero).

const VR, VU, VV, VW, VX, VQV, VQC, VP = 1, 2, 3, 4, 5, 6, 7, 8
const NV = 8
"""Coefficient slots of the faces (x: 1 … 7, y: 8 … 14, η: 15 … 21), slot b = 0 … 6 as in FV3D/moist.jl: χ of the
cell left/below (0), q_v (1), momentum coefficient c_pd Θ̂ or 1 (2), κ left/below (3), κ right/above (4), q_c (5),
χ right/above (6)."""
const NC = 21
@inline sx(b) = 1 + b
@inline sy(b) = 8 + b
@inline sz(b) = 15 + b

struct KAWork{A3,A4}
    θ::A3; φq::A3; φc::A3; Θh::A3; pc::A3                    # cells (nxh, nyh, nz)
    UV::A4                                                   # face velocities (…, nz, 4): u, v at the left/lower
                                                             # face, u, v at the right/upper face of each cell
    w::A3                                                    # η-faces (nxh, nyh, nz+1)
    abc::A4                                                  # ∂π/∂(ρ, X, ρq_v, ρq_c) (nxh, nyh, nz, 4)
    Fx::A3; Fy::A3; Fz::A3                                   # mass fluxes (Fz: nz+1)
    SH::A4                                                   # shifted velocities uL, uR, vL, vR, wL, wR (…, nz, 6)
    TT::A4                                                   # their advective tendencies (…, nz, 6)
    FVx::A4                                                  # x-face values θ, q_v, q_c and fluxes F φ_f/Δx of
    FVy::A4                                                  # the six shifted velocities (…, nz, 9); y-faces
    Wc::A3                                                   # W of the column solve (nxh, nyh, nz+1)
    fac::Dict{Float64,NTuple{3,A3}}                          # column factors (nz+1, nxh, nyh) per c cw
end

"""
    MoistKA(grid::KAGrid, dec = SerialDecomp(); mgas = MoistGas(), pvar = :p, micro = :fast, τfac = 0.25,
            cs = 350.0, wτ = nothing, τ = nothing, grav = 9.81, sponge = nothing)

The moist model of `FV3D.MoistModel3D` on the tile `grid` of the decomposition `dec` (pressure p as the additional
variable by default, or the Exner pressure, pvar = :pi). τ as in MoistModel3D (τfac Δτ_ac, or min(Δx, Δy, Δz_min)/wτ,
or given). `sponge = (zD, αtop)`: Rayleigh damping of W above zD (as `FV3D.with_sponge`), or a vector of the rates
at the η-faces k = 1 … nz+1.
"""
struct MoistKA{TV,G<:KAGrid,D<:AbstractDecomp,A1,W<:KAWork}
    grid::G
    dec::D
    mgas::MoistGas{TV}
    pres::Bool                 # pvar = :p
    micro::Int                 # 0 :none, 1 :fast, 2 :slow
    τ::Float64
    grav::Float64
    damp::A1                   # sponge rates at the η-faces (nz+1)
    work::W
end
function MoistKA(g::KAGrid, dec::AbstractDecomp = SerialDecomp(); mgas::MoistGas = MoistGas(), pvar::Symbol = :p,
                 micro::Symbol = :fast, τfac::Real = 0.25, cs::Real = 350.0, wτ = nothing, τ = nothing,
                 grav::Real = 9.81, sponge = nothing)
    pvar in (:pi, :p) || error("pvar: :pi or :p")
    micro in (:none, :fast, :slow) || error("micro: :none, :fast or :slow")
    be = KA.get_backend(g)
    Jmin = allreduce_min(dec, minimum(Array(g.Jc)[HALO+1:HALO+g.nxl, HALO+1:HALO+g.nyl]))
    h = min(g.dx, g.dy, Jmin * g.dz)
    τv = τ !== nothing ? Float64(τ) : wτ !== nothing ? h / wτ : τfac * 0.85 * 0.92 * min(g.dx, g.dy) / cs
    damp = zeros(g.nz + 1)
    if sponge isa Tuple
        zD, αtop = sponge
        for k in 2:g.nz
            z = g.zf[k]
            damp[k] = z > zD ? αtop * sin(π / 2 * (z - zD) / (g.H - zD))^2 : 0.0
        end
    elseif sponge !== nothing
        damp .= sponge
    end
    z3(nk) = KA.zeros(be, Float64, g.nxh, g.nyh, nk)
    z4(nv) = KA.zeros(be, Float64, g.nxh, g.nyh, g.nz, nv)
    nz = g.nz
    work = KAWork(z3(nz), z3(nz), z3(nz), z3(nz), z3(nz), z4(4), z3(nz + 1), KA.zeros(be, Float64, g.nxh, g.nyh, nz, 4),
                  z3(nz), z3(nz), z3(nz + 1), z4(6), z4(6), z4(9), z4(9), z3(nz + 1),
                  Dict{Float64,NTuple{3,typeof(z3(nz))}}())
    dd = KA.allocate(be, Float64, nz + 1); copyto!(dd, damp)
    init_exchange!(dec, g)                                   # communication buffers, once
    return MoistKA(g, dec, mgas, pvar === :p, micro === :none ? 0 : micro === :fast ? 1 : 2, τv, Float64(grav), dd, work)
end

KA.get_backend(m::MoistKA) = KA.get_backend(m.grid)
"""A zero state (or coefficient array with `nv = NC`) on the device of the model."""
ka_zeros(m::MoistKA, nv::Int = NV) = KA.zeros(KA.get_backend(m), Float64, m.grid.nxh, m.grid.nyh, m.grid.nz + 1, nv)

"""The additional variable of the equation of state: p (pres) or π."""
@inline ka_pvar(mg, pres, ρ, X, ρv, ρc) = pres ? _moist_p(mg, ρ, X, ρv, ρc) : moist_exner(mg, ρ, X, ρv, ρc)

"""Launch a kernel on the backend of the model (asynchronous: kernels of one task run in order on the device; the
host synchronizes before it reads device data). `wg`: the workgroup size (default: chosen by KernelAbstractions)."""
function _launch!(m, kernel!, ndrange, args...; wg = nothing)
    be = KA.get_backend(m)
    (wg === nothing ? kernel!(be) : kernel!(be, wg))(args...; ndrange)
    return nothing
end
"""Workgroups of the fast loop: one thread per column (column solve) in groups of 64, so that the few thousand columns
of a tile spread over all multiprocessors; one thread per (column, level) in groups of 256 along the columns."""
const WG_COL = 64
const WG_CELL = (256, 1)
