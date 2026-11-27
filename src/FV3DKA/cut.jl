# Dry model with Cartesian cut cells (FV3D.Model3D, implicit = :amf, FV3D_PRESSURE_FORM = :product, recon = :ash) with
# KernelAbstractions on a tile of the horizontal decomposition (halo HALO = 1, as the moist model). State Q[I, J, k, v],
# v = ρ, U, V, W, Θ (`CR … CT`), W at the η-faces k = 2 … nz (k = 1, nz + 1: walls, zero). Coefficients Cf[I, J, k, s]
# (`NCC` slots): face θ (third order, first-order upwind on the faces of masked cells), face dp/dΘ, upwind coefficients
# a± of the masked advection and the cell velocities of the shifted momenta (complete Jacobian, :conservative).
#
# The geometry of the tile (fluid fractions, open face fractions, mask) is a slice of the global FV3D.Grid3D, so that the
# operator equals FV3D up to round-off. Communication as in the moist model: the state halo, the tile-boundary face
# values from the upwind tile (`merge_faces!`), the cell tendencies of the shifted velocities; the masked advection of
# the fast part (A1 z) also exchanges the shifted momenta and their tendencies.

const CR, CU, CV, CW, CT = 1, 2, 3, 4, 5
const NVC = 5
const KθX, KθY, KθZ, KPX, KPY, KPZ = 1, 2, 3, 4, 5, 6
const KAXP, KAXM, KAYP, KAYM, KAZP, KAZM = 7, 8, 9, 10, 11, 12
"""Slot of the cell velocity M_side/ρ of the shifted momentum (direction d, side 1 = L, 2 = R)."""
@inline kvel(d, side) = 12 + 2 * (d - 1) + side
const NCC = 18

# ------------------------------------------------------------------ tile geometry

"""
    CutGrid(g3::FV3D.Grid3D; px = 1, py = 1, cx = 0, cy = 0, backend = CPU())

Tile (cx, cy) of a px × py decomposition of the Cartesian cut-cell grid g3 (periodic in x, y): the fractions of g3 on
the tile with halo (arrays (nxh+1, nyh+1, nz+1) on the device: the outer faces of the right and upper halo cells
included).
"""
struct CutGrid{M3,M1} <: AbstractTileGrid
    nx::Int; ny::Int; nz::Int
    nxl::Int; nyl::Int
    i0::Int; j0::Int
    nxh::Int; nyh::Int
    dx::Float64; dy::Float64; dz::Float64; H::Float64
    Vf::M3; iV::M3; Ax::M3; Ay::M3; Az::M3
    msk::M3                                  # 1 in the masked cells
    Mx::M3; My::M3                           # 1 on the x-, y-faces of masked cells
    gr::M3                                   # gr[I, J, 1]: ground area of the shift (1 full, 0 below a cut cell)
    dzc::M1; dzf::M1; wlo::M1                # levels (FV3D.Grid3D): cell thicknesses, centre distances, face weights (nz+1)
    lim::Int                                 # limiter of the face values (FV3D.Grid3D lim, recon3)
    zf::Vector{Float64}
    px::Int; py::Int
    g3::FV3D.Grid3D                          # the global grid (host; tile models of the AMF preconditioner)
end

function CutGrid(g3::FV3D.Grid3D; px::Int = 1, py::Int = 1, cx::Int = 0, cy::Int = 0, backend = CPU())
    (g3.follow || g3.recon !== :ash) && error("CutGrid: Cartesian cut cells with recon = :ash")
    nx, ny, nz = g3.nx, g3.ny, g3.nz
    (nx % px == 0 && ny % py == 0) || error("CutGrid: nx, ny must be divisible by px, py")
    nxl, nyl = nx ÷ px, ny ÷ py
    (nxl >= 3 && nyl >= 3) || error("CutGrid: tiles need at least 3 cells per direction")
    i0, j0 = cx * nxl, cy * nyl
    nxh, nyh = nxl + 2HALO, nyl + 2HALO
    gi(I) = mod1(i0 + I - HALO, nx); gj(J) = mod1(j0 + J - HALO, ny)
    slab(A, nk) = [k <= nk ? Float64(A[gi(I), gj(J), k]) : 0.0 for I in 1:nxh+1, J in 1:nyh+1, k in 1:nz+1]
    gr = [k == 1 && g3.full[gi(I), gj(J), 1] ? 1.0 : 0.0 for I in 1:nxh+1, J in 1:nyh+1, k in 1:nz+1]
    dev(a) = (b = KA.allocate(backend, Float64, size(a)...); copyto!(b, a); b)
    return CutGrid(nx, ny, nz, nxl, nyl, i0, j0, nxh, nyh, g3.dx, g3.dy, g3.dz, g3.H,
                   dev(slab(g3.Vf, nz)), dev(slab(g3.invV, nz)), dev(slab(g3.Ax, nz)), dev(slab(g3.Ay, nz)),
                   dev(slab(g3.Az, nz + 1)), dev(slab(g3.mask, nz)), dev(slab(g3.Mx, nz)), dev(slab(g3.My, nz)), dev(gr),
                   dev(vcat(g3.dzc, g3.dzc[nz])), dev(g3.dzf), dev(g3.wlo), g3.lim, collect(g3.zf), px, py, g3)
end
KA.get_backend(g::CutGrid) = KA.get_backend(g.Vf)
"""The geometry arrays as a named tuple (kernel argument)."""
_geo(g::CutGrid) = (Vf = g.Vf, iV = g.iV, Ax = g.Ax, Ay = g.Ay, Az = g.Az, msk = g.msk, Mx = g.Mx, My = g.My, gr = g.gr,
                     dzc = g.dzc, dzf = g.dzf, wlo = g.wlo, lim = g.lim)
"""Density at the η-face k of the state Z (FV3D `ρface`: mass of the face control volume)."""
@inline _ρf(G, Z, I, J, k) = G.wlo[k] * Z[I, J, k-1, CR] + (1 - G.wlo[k]) * Z[I, J, k, CR]

@inline _azs(G, nz, I, J, k) = k == 1 ? G.gr[I, J, 1] : (k == nz + 1 ? 1.0 : G.Az[I, J, k])
"""Faces of masked cells (rows of A1): left x-face, left y-face, lower η-face (2 … nz) of cell (I, J, k)."""
@inline _rowx(G, I, J, k) = G.Mx[I, J, k] > 0 && G.Ax[I, J, k] > 0
@inline _rowy(G, I, J, k) = G.My[I, J, k] > 0 && G.Ay[I, J, k] > 0
@inline _rowz(G, nz, I, J, k) = 2 <= k <= nz && G.Az[I, J, k] > 0 && (G.msk[I, J, k-1] > 0 || G.msk[I, J, k] > 0)
"""Faces of masked cells for the masked θ transport (any face between a masked and another cell)."""
@inline _mfx(G, I, J, k) = G.msk[I-1, J, k] > 0 || G.msk[I, J, k] > 0
@inline _mfy(G, I, J, k) = G.msk[I, J-1, k] > 0 || G.msk[I, J, k] > 0
@inline _mfz(G, I, J, k) = G.msk[I, J, k-1] > 0 || G.msk[I, J, k] > 0
"""Implicit weight of the z-face k (2 … nz) of column (I, J): 1 for every face with `vimp` (vertical = :implicit,
HEVI), else 1 only on the faces of masked cells (vertical = :explicit: outside the mask the vertical acoustics and
gravity are explicit, in the Verlet sub-steps with weight 1 - w)."""
@inline _wzI(G, nz, I, J, k, vimp) = (2 <= k <= nz && (vimp || G.msk[I, J, k-1] > 0 || G.msk[I, J, k] > 0)) ? 1.0 : 0.0

"""
Columns of the tile on which a kernel runs (overlap of communication and computation): `:all` the interior, `:core`
the columns 2 … nxl-1 × 2 … nyl-1 (their ±1 stencils need no halo), `:ring` the rest (first and last row and column),
run after the receive. Tiles narrower than 3 columns: core empty, ring = all.
"""
struct CutRegion
    kind::Int                                      # 0 all, 1 core, 2 ring
    nxl::Int
    nyl::Int
end
function CutRegion(g::AbstractTileGrid, kind::Symbol)
    small = min(g.nxl, g.nyl) < 3
    k = kind === :all ? 0 : kind === :core ? (small ? 3 : 1) : kind === :ring ? (small ? 0 : 2) :
        error("region: :all, :core or :ring")
    return CutRegion(k, g.nxl, g.nyl)
end
"""Number of columns of the region (kind 3: empty core of a narrow tile)."""
ncols(R::CutRegion) = R.kind == 0 ? R.nxl * R.nyl : R.kind == 1 ? (R.nxl - 2) * (R.nyl - 2) :
                      R.kind == 2 ? 2 * (R.nxl + R.nyl) - 4 : 0
"""Tile indices (I, J) (halo included) of the r-th column of the region (along i where possible)."""
@inline function _cIJ(R::CutRegion, r)
    if R.kind == 1
        ni = R.nxl - 2
        return ((r - 1) % ni + 2 + HALO, (r - 1) ÷ ni + 2 + HALO)
    elseif R.kind == 2
        r <= R.nxl && return (r + HALO, 1 + HALO)                                  # first row
        r <= 2 * R.nxl && return (r - R.nxl + HALO, R.nyl + HALO)                  # last row
        q = r - 2 * R.nxl                                                          # columns, rows 2 … nyl-1
        return q <= R.nyl - 2 ? (1 + HALO, q + 1 + HALO) : (R.nxl + HALO, q - (R.nyl - 2) + 1 + HALO)
    end
    return ((r - 1) % R.nxl + 1 + HALO, (r - 1) ÷ R.nxl + 1 + HALO)
end
"""Launch a cell kernel (index (column, level)) over the columns of the region and nk levels (nothing if empty)."""
function _launch_r!(m, kernel!, R::CutRegion, nk, args...)
    n = ncols(R)
    n == 0 && return nothing
    return _launch!(m, kernel!, (n, nk), args..., R)
end

# ------------------------------------------------------------------ model

struct CutWork{A3,A4}
    θ::A3; w::A3; Fx::A3; Fy::A3; Fz::A3           # cells / faces (nxh, nyh, nz+1)
    UV::A4                                         # u, v at the left/lower and right/upper face of each cell (4)
    SH::A4; TT::A4                                 # shifted velocities and their tendencies (6)
    FVx::A4; FVy::A4                               # x-, y-face values: θ, F φ_f/Δx of the six shifted velocities (7)
    M::A4; T::A4                                   # masked advection A1 z: shifted momenta, tendencies (6)
    Wc::A3; cpw::A3; idw::A3                       # column solve: W, eliminated upper diagonal, inverse pivots
end

"""
    CutKA(grid::CutGrid, dec = SerialDecomp(); gas = Gas(), sponge = nothing, jform = :partial, prec = :columns)

The dry cut-cell model of `FV3D.Model3D(…; implicit = :amf)` with FV3D_PRESSURE_FORM = :product on the tile `grid`.
`sponge = (zD, αtop)` damps W above zD (as `FV3D.with_sponge`, target 0); `jform` the masked advection Jacobian
(:partial or :conservative, FV3D_MASK_JFORM); `prec` the preconditioner of the implicit solve: `:columns` (default) the
column solve (no matrix, no communication; with the matrix-free product J v = G_v(v) the implicit solve uses no
assembled matrix at all), `:amf` the approximate matrix factorization of the tile (with one halo column as overlap in
decomposed directions, restricted additive Schwarz; host assembly and sparse LU per stage: fewer iterations, but on
the bell hill 42 s against 24 s per step). `nudge = (; x0, x1, αmax, u, v, θ)`: nudging zone x0 ≤ x ≤ x1 towards the
profiles u(z), v(z), θ(z) (as `FV3D.with_nudging`). `prec = :mask` (with vertical = :explicit): the
masked AMF preconditioner with switchable inner solvers, `mask = (acoustic = :cg | :jacobi | :bicgstab | :hostlu,
advection = :bicgstab | :jacobi | :hostlu, tol, maxit)` (cut_mask.jl). `vertical`: `:implicit` (HEVI: the vertical acoustics and
gravity implicit in every column; for Δz ≪ Δx, hills) or `:explicit` (implicit only on the faces of the masked cells,
outside the mask all three directions explicit in the Verlet sub-steps, gravity explicit everywhere; for Δz ≈ Δx,
buildings).
"""
struct CutKA{G<:CutGrid,D<:AbstractDecomp,A1,W<:CutWork}
    grid::G
    dec::D
    gas::Gas
    damp::A1
    zdamp::A1                                      # zeros (no sponge in the implicit operator)
    complete::Bool
    work::W
    its::Vector{Int}                               # Krylov iterations per implicit solve
    mv::Vector{Int}                                # matrix-vector products per implicit solve
    prec::Symbol                                   # preconditioner: :columns (default) or :amf (tile AMF)
    amf::Any                                       # CutAMF (prec = :amf)
    mgp::Base.RefValue{Any}                        # acoustic multigrid (Rosenbrock twin model, cut_mg.jl) or nothing
    vimp::Bool                                     # vertical = :implicit (HEVI: all z-faces implicit) or :explicit (mask only)
    mprec::Base.RefValue{Any}                      # prec = :mask: the masked AMF preconditioner (cut_mask.jl)
    nudge::Any                                     # nudging zone: rates (cells, x-, y-faces) and profiles on the device
end
function CutKA(g::CutGrid, dec::AbstractDecomp = SerialDecomp(); gas::Gas = Gas(), sponge = nothing, jform::Symbol = :partial,
               prec::Symbol = :columns, nudge = nothing, vertical::Symbol = :implicit, mask = (;))
    jform in (:partial, :conservative) || error("jform: :partial or :conservative")
    prec === :mask && vertical !== :explicit && error("prec = :mask needs vertical = :explicit (gravity explicit)")
    vertical in (:implicit, :explicit) || error("vertical: :implicit (HEVI) or :explicit (implicit only in the mask)")
    prec in (:amf, :columns, :mask) || error("prec: :amf, :columns or :mask")
    be = KA.get_backend(g)
    damp = zeros(g.nz + 1)
    if sponge !== nothing
        zD, αtop = sponge
        for k in 2:g.nz
            z = g.zf[k]
            damp[k] = z > zD ? αtop * sin(π / 2 * (z - zD) / (g.H - zD))^2 : 0.0
        end
    end
    z3() = KA.zeros(be, Float64, g.nxh, g.nyh, g.nz + 1)
    z4(n) = KA.zeros(be, Float64, g.nxh, g.nyh, g.nz + 1, n)
    work = CutWork(z3(), z3(), z3(), z3(), z3(), z4(4), z4(6), z4(6), z4(7), z4(7), z4(6), z4(6), z3(), z3(), z3())
    dd = KA.allocate(be, Float64, g.nz + 1); copyto!(dd, damp)
    init_exchange!(dec, g, max(NVC + 4, NCC))
    nd = nothing
    if nudge !== nothing
        g3 = g.g3
        f(p) = p isa Number ? (z -> Float64(p)) : p
        a(x) = FV3D.nudging_rate(x, nudge.x0, nudge.x1, get(nudge, :αmax, 0.5), g3.Lx)
        xc(I) = (g.i0 + I - HALO - 0.5) * g.dx; xf(I) = (g.i0 + I - HALO - 1) * g.dx
        dev(A) = (b = KA.allocate(be, Float64, size(A)...); copyto!(b, A); b)
        prof(p) = dev([k <= g.nz ? f(p)(g3.zc[k]) : 0.0 for k in 1:g.nz+1])
        nd = (αc = dev([a(xc(I)) for I in 1:g.nxh+1, J in 1:g.nyh+1]), αx = dev([a(xf(I)) for I in 1:g.nxh+1, J in 1:g.nyh+1]),
              u = prof(nudge.u), v = prof(get(nudge, :v, 0.0)), θ = prof(nudge.θ))
    end
    mk = CutKA(g, dec, gas, dd, KA.zeros(be, Float64, g.nz + 1), jform === :conservative, work, Int[], Int[], prec, prec === :amf ? CutAMF(g, gas) : nothing, Ref{Any}(nothing),
                 vertical === :implicit, Ref{Any}(nothing), nd)
    prec === :mask && (mk.mprec[] = MaskAMF(mk; mask...))
    return mk
end
KA.get_backend(m::CutKA) = KA.get_backend(m.grid)
"""A zero state (or a coefficient array, nv = NCC) on the device of the model."""
ka_zeros(m::CutKA, nv::Int = NVC) = KA.zeros(KA.get_backend(m), Float64, m.grid.nxh, m.grid.nyh, m.grid.nz + 1, nv)

# ------------------------------------------------------------------ face values (FV3D/advection.jl, recon = :ash)

@inline _cfx(φ, G, I, J, k, F) = !signbit(F) ?
    recon3(φ[I-2, J, k], φ[I-1, J, k], φ[I, J, k], G.Vf[I-2, J, k], G.Vf[I-1, J, k], G.Vf[I, J, k], G.Ax[I-1, J, k], G.Ax[I, J, k],
           G.Ax[I-1, J, k] > 0, 1.0, 1.0, 1.0, G.lim) :
    recon3(φ[I+1, J, k], φ[I, J, k], φ[I-1, J, k], G.Vf[I+1, J, k], G.Vf[I, J, k], G.Vf[I-1, J, k], G.Ax[I+1, J, k], G.Ax[I, J, k],
           G.Ax[I+1, J, k] > 0, 1.0, 1.0, 1.0, G.lim)
@inline _cfy(φ, G, I, J, k, F) = !signbit(F) ?
    recon3(φ[I, J-2, k], φ[I, J-1, k], φ[I, J, k], G.Vf[I, J-2, k], G.Vf[I, J-1, k], G.Vf[I, J, k], G.Ay[I, J-1, k], G.Ay[I, J, k],
           G.Ay[I, J-1, k] > 0, 1.0, 1.0, 1.0, G.lim) :
    recon3(φ[I, J+1, k], φ[I, J, k], φ[I, J-1, k], G.Vf[I, J+1, k], G.Vf[I, J, k], G.Vf[I, J-1, k], G.Ay[I, J+1, k], G.Ay[I, J, k],
           G.Ay[I, J+1, k] > 0, 1.0, 1.0, 1.0, G.lim)
@inline function _cfz(φ, G, nz, I, J, k, F)
    if !signbit(F)
        ku = max(k - 2, 1)
        return recon3(φ[I, J, ku], φ[I, J, k-1], φ[I, J, k], G.Vf[I, J, ku], G.Vf[I, J, k-1], G.Vf[I, J, k],
                      _azs(G, nz, I, J, k - 1), G.Az[I, J, k], G.Az[I, J, k-1] > 0, G.dzc[ku], G.dzc[k-1], G.dzc[k], G.lim)
    else
        ku = min(k + 1, nz)
        return recon3(φ[I, J, ku], φ[I, J, k], φ[I, J, k-1], G.Vf[I, J, ku], G.Vf[I, J, k], G.Vf[I, J, k-1],
                      _azs(G, nz, I, J, k + 1), G.Az[I, J, k], G.Az[I, J, k+1] > 0, G.dzc[ku], G.dzc[k], G.dzc[k-1], G.lim)
    end
end

# ------------------------------------------------------------------ C(y), S(y)

@kernel function _kc_uv!(UV, @Const(Q), nxh, nyh)
    I, J, k = @index(Global, NTuple)
    @inbounds begin
        uf(I) = Q[I, J, k, CU] / (0.5 * (Q[I-1, J, k, CR] + Q[I, J, k, CR]))
        vf(J) = Q[I, J, k, CV] / (0.5 * (Q[I, J-1, k, CR] + Q[I, J, k, CR]))
        I > 1 && (UV[I, J, k, 1] = uf(I))
        J > 1 && (UV[I, J, k, 2] = vf(J))
        I < nxh && (UV[I, J, k, 3] = uf(I + 1))
        J < nyh && (UV[I, J, k, 4] = vf(J + 1))
    end
end

"""θ = Θ/ρ of the cells, w = W / mean ρ, mass fluxes A U, A V, A W of all faces of the tile."""
@kernel function _kc_cells!(θ, w, Fx, Fy, Fz, @Const(Q), G, nz)
    I, J, k = @index(Global, NTuple)
    @inbounds begin
        if k <= nz
            θ[I, J, k] = Q[I, J, k, CT] / Q[I, J, k, CR]
            Fx[I, J, k] = G.Ax[I, J, k] * Q[I, J, k, CU]
            Fy[I, J, k] = G.Ay[I, J, k] * Q[I, J, k, CV]
        end
        if k == 1 || k == nz + 1
            w[I, J, k] = 0.0; Fz[I, J, k] = 0.0
        else
            w[I, J, k] = Q[I, J, k, CW] / _ρf(G, Q, I, J, k)
            Fz[I, J, k] = G.Az[I, J, k] * Q[I, J, k, CW]
        end
    end
end

"""Shifted velocities (left/right values of u, v, w from the two faces, cut-cell areas) of the fluid cells."""
@kernel function _kc_shift!(SH, @Const(UV), @Const(w), G, nz)
    I, J, k = @index(Global, NTuple)
    @inbounds if G.Vf[I, J, k] > 0
        SH[I, J, k, 1], SH[I, J, k, 2] = shiftvel(G.Ax[I, J, k], G.Ax[I+1, J, k], UV[I, J, k, 1], UV[I, J, k, 3])
        SH[I, J, k, 3], SH[I, J, k, 4] = shiftvel(G.Ay[I, J, k], G.Ay[I, J+1, k], UV[I, J, k, 2], UV[I, J, k, 4])
        SH[I, J, k, 5], SH[I, J, k, 6] = shiftvel(_azs(G, nz, I, J, k), _azs(G, nz, I, J, k + 1), w[I, J, k], w[I, J, k+1])
    else
        for s in 1:6; SH[I, J, k, s] = 0.0; end
    end
end

"""x- and y-face values (θ; F φ_f/Δx of the shifted velocities) of the interior rows, upwind tile only."""
@kernel function _kc_facevals!(FVx, FVy, @Const(θ), @Const(SH), @Const(Fx), @Const(Fy), G, dx, dy, nxl, nyl)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        if j <= nyl
            F = Fx[I, J, k]
            if _okx(I, F, nxl)
                G.Ax[I, J, k] > 0 && (FVx[I, J, k, 1] = _cfx(θ, G, I, J, k, F))
                for s in 1:6
                    FVx[I, J, k, 1+s] = F != 0 ? F * _cfx(view(SH, :, :, :, s), G, I, J, k, F) / dx : 0.0
                end
            end
        end
        if i <= nxl
            F = Fy[I, J, k]
            if _okx(J, F, nyl)
                G.Ay[I, J, k] > 0 && (FVy[I, J, k, 1] = _cfy(θ, G, I, J, k, F))
                for s in 1:6
                    FVy[I, J, k, 1+s] = F != 0 ? F * _cfy(view(SH, :, :, :, s), G, I, J, k, F) / dy : 0.0
                end
            end
        end
    end
end

"""Face coefficients of the faces of the interior cells (θ, dp/dΘ, a±) and the cell velocities of the interior cells."""
@kernel function _kc_coefs!(Cf, @Const(Q), @Const(FVx), @Const(FVy), @Const(θ), @Const(Fx), @Const(Fy), @Const(Fz), G, gas,
                            nz, nxl, nyl)
    i, j, k = @index(Global, NTuple)                  # i ∈ 1 … nxl+1, j ∈ 1 … nyl+1, k ∈ 1 … nz+1
    I, J = i + HALO, j + HALO
    @inbounds begin
        if k <= nz && j <= nyl                        # x-face I
            F = Fx[I, J, k]
            if G.Ax[I, J, k] > 0
                Cf[I, J, k, KθX] = _mfx(G, I, J, k) ? (F >= 0 ? θ[I-1, J, k] : θ[I, J, k]) : FVx[I, J, k, 1]
            end
            Cf[I, J, k, KPX] = FV3D.dpdΘ_face(gas, Q[I-1, J, k, CT], Q[I, J, k, CT])
            Cf[I, J, k, KAXP] = max(F, 0.0) / Q[I-1, J, k, CR]; Cf[I, J, k, KAXM] = min(F, 0.0) / Q[I, J, k, CR]
        end
        if k <= nz && i <= nxl                        # y-face J
            F = Fy[I, J, k]
            if G.Ay[I, J, k] > 0
                Cf[I, J, k, KθY] = _mfy(G, I, J, k) ? (F >= 0 ? θ[I, J-1, k] : θ[I, J, k]) : FVy[I, J, k, 1]
            end
            Cf[I, J, k, KPY] = FV3D.dpdΘ_face(gas, Q[I, J-1, k, CT], Q[I, J, k, CT])
            Cf[I, J, k, KAYP] = max(F, 0.0) / Q[I, J-1, k, CR]; Cf[I, J, k, KAYM] = min(F, 0.0) / Q[I, J, k, CR]
        end
        if 2 <= k <= nz && i <= nxl && j <= nyl       # η-face k
            F = Fz[I, J, k]
            if G.Az[I, J, k] > 0
                Cf[I, J, k, KθZ] = _mfz(G, I, J, k) ? (F >= 0 ? θ[I, J, k-1] : θ[I, J, k]) : _cfz(θ, G, nz, I, J, k, F)
            end
            Cf[I, J, k, KPZ] = FV3D.dpdΘ_face(gas, Q[I, J, k-1, CT], Q[I, J, k, CT])
            Cf[I, J, k, KAZP] = max(F, 0.0) / Q[I, J, k-1, CR]; Cf[I, J, k, KAZM] = min(F, 0.0) / Q[I, J, k, CR]
        end
        if k <= nz && i <= nxl && j <= nyl && G.Vf[I, J, k] > 0      # cell velocities of the shifted momenta
            ρc = Q[I, J, k, CR]
            for d in 1:3
                (lL, lR), (rL, rR) = FV3D.shiftcoef(_alo(G, nz, d, I, J, k), _ahi(G, nz, d, I, J, k))
                vL, vR = _mlo(Q, nz, d, I, J, k), _mhi(Q, nz, d, I, J, k)
                Cf[I, J, k, kvel(d, 1)] = (lL * vL + lR * vR) / ρc
                Cf[I, J, k, kvel(d, 2)] = (rL * vL + rR * vR) / ρc
            end
        end
    end
end

# shift geometry per component d = 1 (U), 2 (V), 3 (W): faces of cell (I, J, k), momenta of z at these faces
@inline _alo(G, nz, d, I, J, k) = d == 1 ? G.Ax[I, J, k] : d == 2 ? G.Ay[I, J, k] : _azs(G, nz, I, J, k)
@inline _ahi(G, nz, d, I, J, k) = d == 1 ? G.Ax[I+1, J, k] : d == 2 ? G.Ay[I, J+1, k] : _azs(G, nz, I, J, k + 1)
@inline _mlo(Z, nz, d, I, J, k) = d == 1 ? Z[I, J, k, CU] : d == 2 ? Z[I, J, k, CV] : (k == 1 ? 0.0 : Z[I, J, k, CW])
@inline _mhi(Z, nz, d, I, J, k) = d == 1 ? Z[I+1, J, k, CU] : d == 2 ? Z[I, J+1, k, CV] : (k == nz ? 0.0 : Z[I, J, k+1, CW])

"""Advective tendencies of the six shifted velocities in the interior cells: -(1/V_f) Σ F φ_f."""
@kernel function _kc_advect!(TT, @Const(SH), @Const(FVx), @Const(FVy), @Const(Fz), G, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds for s in 1:6
        φ = view(SH, :, :, :, s)
        t = FVx[I, J, k, 1+s] + FVy[I, J, k, 1+s] - FVx[I+1, J, k, 1+s] - FVy[I, J+1, k, 1+s]
        k > 1 && (F = Fz[I, J, k]; F != 0 && (t += F * _cfz(φ, G, nz, I, J, k, F) / G.dzc[k]))
        k < nz && (F = Fz[I, J, k+1]; F != 0 && (t -= F * _cfz(φ, G, nz, I, J, k + 1, F) / G.dzc[k]))
        TT[I, J, k, s] = t * G.iV[I, J, k]
    end
end

"""Face tendencies of U, V, W (shift back) on the faces of the interior cells; zero on the faces of masked cells (the
first-order upwind advection A1 z of the fast part replaces it there)."""
@kernel function _kc_shiftback!(S, @Const(TT), G, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        a = G.Ax[I, J, k]
        S[I, J, k, CU] = (a > 0 && !_rowx(G, I, J, k)) ?
            shift_back(TT[I, J, k, 1], TT[I-1, J, k, 2], G.Vf[I-1, J, k], G.Vf[I, J, k], a, G.Ax[I-1, J, k], G.Ax[I+1, J, k]) : 0.0
        a = G.Ay[I, J, k]
        S[I, J, k, CV] = (a > 0 && !_rowy(G, I, J, k)) ?
            shift_back(TT[I, J, k, 3], TT[I, J-1, k, 4], G.Vf[I, J-1, k], G.Vf[I, J, k], a, G.Ay[I, J-1, k], G.Ay[I, J+1, k]) : 0.0
        if k >= 2
            a = G.Az[I, J, k]
            S[I, J, k, CW] = (a > 0 && !_rowz(G, nz, I, J, k)) ?
                shift_back(TT[I, J, k, 5], TT[I, J, k-1, 6], G.Vf[I, J, k-1] * G.dzc[k-1], G.Vf[I, J, k] * G.dzc[k], a, _azs(G, nz, I, J, k - 1),
                           _azs(G, nz, I, J, k + 1)) : 0.0
        end
    end
end

"""
    ka_frozen_data!(Cf, S, m::CutKA, Q)

C(y) and S(y) of the state Q (FV3D `frozen_data!` with implicit = :amf). Communications: the state with the face
velocities, the tile-boundary face values, the coefficients of the cell velocities, the tendencies of the shifted
velocities.
"""
function ka_frozen_data!(Cf, S, m::CutKA, Q)
    g, w, dec = m.grid, m.work, m.dec
    G = _geo(g)
    nz, nxh, nyh, nxl, nyl = g.nz, g.nxh, g.nyh, g.nxl, g.nyl
    _launch!(m, _kc_uv!, (nxh, nyh, nz), w.UV, Q, nxh, nyh)
    exchange!(dec, (Q, w.UV), g)
    _launch!(m, _kc_uv!, (nxh, nyh, nz), w.UV, Q, nxh, nyh)
    _launch!(m, _kc_cells!, (nxh, nyh, nz + 1), w.θ, w.w, w.Fx, w.Fy, w.Fz, Q, G, nz)
    _launch!(m, _kc_shift!, (nxh, nyh, nz), w.SH, w.UV, w.w, G, nz)
    fill!(w.FVx, 0.0); fill!(w.FVy, 0.0)
    _launch!(m, _kc_facevals!, (nxl + 1, nyl + 1, nz), w.FVx, w.FVy, w.θ, w.SH, w.Fx, w.Fy, G, g.dx, g.dy, nxl, nyl)
    merge_faces!(dec, w.FVx, w.Fx, w.FVy, w.Fy, g)
    fill!(Cf, 0.0); fill!(S, 0.0)
    _launch!(m, _kc_coefs!, (nxl + 1, nyl + 1, nz + 1), Cf, Q, w.FVx, w.FVy, w.θ, w.Fx, w.Fy, w.Fz, G, m.gas, nz, nxl, nyl)
    exchange!(dec, Cf, g, Tuple(kvel(d, s) for d in 1:3 for s in 1:2))
    _launch!(m, _kc_advect!, (nxl, nyl, nz), w.TT, w.SH, w.FVx, w.FVy, w.Fz, G, nz)
    exchange!(dec, w.TT, g, (2, 4))
    _launch!(m, _kc_shiftback!, (nxl, nyl, nz), S, w.TT, G, nz)
    m.nudge === nothing || _ka_nudge!(S, m, m.nudge, Q, G)
    return Cf, S
end

"""S += nudging tendencies on the interior (FV3D `nudge!`): U, V, W towards ρ̄ u_t, ρ̄ v_t, 0, Θ towards ρ θ_t."""
@kernel function _kc_nudge!(S, @Const(Q), αc, αx, ut, vt, θt, G, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        ρc = Q[I, J, k, CR]
        a = αc[I, J]
        if a > 0 && G.Vf[I, J, k] > 0
            S[I, J, k, CT] -= a * (Q[I, J, k, CT] - ρc * θt[k])
            k >= 2 && G.Az[I, J, k] > 0 && (S[I, J, k, CW] -= a * Q[I, J, k, CW])
        end
        a = αx[I, J]
        (a > 0 && G.Ax[I, J, k] > 0) && (S[I, J, k, CU] -= a * (Q[I, J, k, CU] - 0.5 * (Q[I-1, J, k, CR] + ρc) * ut[k]))
        a = αc[I, J]
        (a > 0 && G.Ay[I, J, k] > 0) && (S[I, J, k, CV] -= a * (Q[I, J, k, CV] - 0.5 * (Q[I, J-1, k, CR] + ρc) * vt[k]))
    end
end
_ka_nudge!(S, m, nd, Q, G) = (g = m.grid; _launch!(m, _kc_nudge!, (g.nxl, g.nyl, g.nz), S, Q, nd.αc, nd.αx, nd.u, nd.v, nd.θ, G, g.nz))

# ------------------------------------------------------------------ fast part

"""ρ and Θ rows of cell (I, J, k): horizontal fluxes through the unmasked (sel = 1) or masked (sel = 2) faces."""
@inline function _cthermo_h(Z, Cf, G, dx, dy, I, J, k, sel)
    wx(I, J) = sel == 1 ? 1.0 - G.Mx[I, J, k] : G.Mx[I, J, k]
    wy(I, J) = sel == 1 ? 1.0 - G.My[I, J, k] : G.My[I, J, k]
    Fw = wx(I, J) * G.Ax[I, J, k] * Z[I, J, k, CU];       Fe = wx(I + 1, J) * G.Ax[I+1, J, k] * Z[I+1, J, k, CU]
    Fs = wy(I, J) * G.Ay[I, J, k] * Z[I, J, k, CV];       Fn = wy(I, J + 1) * G.Ay[I, J+1, k] * Z[I, J+1, k, CV]
    s = G.iV[I, J, k]
    oρ = -(s * ((Fe - Fw) / dx + (Fn - Fs) / dy))
    oΘ = -(s * ((Cf[I+1, J, k, KθX] * Fe - Cf[I, J, k, KθX] * Fw) / dx + (Cf[I, J+1, k, KθY] * Fn - Cf[I, J, k, KθY] * Fs) / dy))
    return oρ, oΘ
end
"""U, V rows of the faces I, J: -Pf δΠ/Δx through the unmasked (sel = 1) or masked (sel = 2) faces."""
@inline function _cmomentum_h(Z, Cf, G, dx, dy, I, J, k, sel)
    wx = sel == 1 ? 1.0 - G.Mx[I, J, k] : G.Mx[I, J, k]
    wy = sel == 1 ? 1.0 - G.My[I, J, k] : G.My[I, J, k]
    oU = -(wx * (G.Ax[I, J, k] > 0) * Cf[I, J, k, KPX] * (Z[I, J, k, CT] - Z[I-1, J, k, CT]) / dx)
    oV = -(wy * (G.Ay[I, J, k] > 0) * Cf[I, J, k, KPY] * (Z[I, J, k, CT] - Z[I, J-1, k, CT]) / dy)
    return oU, oV
end

"""out = :hthermo (ρ, Θ: fluxes through the unmasked faces, forcing; W: forcing, sponge) on the interior."""
@kernel function _kc_hthermo!(out, @Const(Z), @Const(Cf), @Const(F), G, damp, dx, dy, nz, vimp, RG)
    ic, k = @index(Global, NTuple)
    I, J = _cIJ(RG, ic)
    @inbounds begin
        oρ, oΘ = _cthermo_h(Z, Cf, G, dx, dy, I, J, k, 1)
        if !vimp                                       # vertical fluxes through the unmasked z-faces (explicit)
            Fb = k >= 2 ? (1 - _wzI(G, nz, I, J, k, false)) * G.Az[I, J, k] * Z[I, J, k, CW] : 0.0
            Ft = k < nz ? (1 - _wzI(G, nz, I, J, k + 1, false)) * G.Az[I, J, k+1] * Z[I, J, k+1, CW] : 0.0
            s = G.iV[I, J, k]; dz = G.dzc[k]
            oρ -= s * (Ft - Fb) / dz
            oΘ -= s * ((k < nz ? Cf[I, J, k+1, KθZ] : 0.0) * Ft - (k >= 2 ? Cf[I, J, k, KθZ] : 0.0) * Fb) / dz
        end
        out[I, J, k, CR] = oρ + F[I, J, k, CR]; out[I, J, k, CT] = oΘ + F[I, J, k, CT]
        out[I, J, k, CU] = 0.0; out[I, J, k, CV] = 0.0
        out[I, J, k, CW] = k >= 2 ? F[I, J, k, CW] - damp[k] * Z[I, J, k, CW] : 0.0
    end
end
"""out = :hmomentum (U, V: pressure gradient through the unmasked faces, forcing) on the interior."""
@kernel function _kc_hmomentum!(out, @Const(Z), @Const(Cf), @Const(F), G, dx, dy, nz, grav, vimp, RG)
    ic, k = @index(Global, NTuple)
    I, J = _cIJ(RG, ic)
    @inbounds begin
        oU, oV = _cmomentum_h(Z, Cf, G, dx, dy, I, J, k, 1)
        out[I, J, k, CU] = oU + F[I, J, k, CU]; out[I, J, k, CV] = oV + F[I, J, k, CV]
        out[I, J, k, CR] = 0.0; out[I, J, k, CT] = 0.0
        # vertical = :explicit: vertical pressure gradient on the unmasked z-faces, gravity on all z-faces
        out[I, J, k, CW] = (!vimp && k >= 2 && G.Az[I, J, k] > 0) ?
            -((1 - _wzI(G, nz, I, J, k, false)) * Cf[I, J, k, KPZ] * (Z[I, J, k, CT] - Z[I, J, k-1, CT]) / G.dzf[k] +
              grav * _ρf(G, Z, I, J, k)) : 0.0
    end
end

"""Implicit part without the masked momentum advection on the interior: vertical fluxes, vertical pressure gradient and
gravity, the fluxes and pressure gradients through the masked x-, y-faces and the masked θ transport."""
@kernel function _kc_vertical!(out, @Const(Z), @Const(Cf), G, dx, dy, nz, grav, ac, tr, damp, vimp, RG)
    ic, k = @index(Global, NTuple)
    I, J = _cIJ(RG, ic)
    @inbounds begin
        s = G.iV[I, J, k]
        dz = G.dzc[k]
        oρ = 0.0; oΘ = 0.0
        if ac                                          # acoustics: vertical, masked horizontal faces
            Fb = k >= 2 ? _wzI(G, nz, I, J, k, vimp) * G.Az[I, J, k] * Z[I, J, k, CW] : 0.0
            Ft = k < nz ? _wzI(G, nz, I, J, k + 1, vimp) * G.Az[I, J, k+1] * Z[I, J, k+1, CW] : 0.0
            oρ = -(s * (Ft - Fb) / dz)
            oΘ = -(s * ((k < nz ? Cf[I, J, k+1, KθZ] : 0.0) * Ft - (k >= 2 ? Cf[I, J, k, KθZ] : 0.0) * Fb) / dz)
            hρ, hΘ = _cthermo_h(Z, Cf, G, dx, dy, I, J, k, 2)
            oρ += hρ; oΘ += hΘ
        end
        # masked θ transport: f = a⁺ Θ_l + a⁻ Θ_r - θ_f (a⁺ ρ_l + a⁻ ρ_r) through the faces of masked cells
        fθ(ap, am, θf, l, r) = ap * Z[l..., CT] + am * Z[r..., CT] - θf * (ap * Z[l..., CR] + am * Z[r..., CR])
        if tr && _mfx(G, I, J, k)
            oΘ += s * fθ(Cf[I, J, k, KAXP], Cf[I, J, k, KAXM], Cf[I, J, k, KθX], (I - 1, J, k), (I, J, k)) / dx
        end
        if tr && _mfx(G, I + 1, J, k)
            oΘ -= s * fθ(Cf[I+1, J, k, KAXP], Cf[I+1, J, k, KAXM], Cf[I+1, J, k, KθX], (I, J, k), (I + 1, J, k)) / dx
        end
        if tr && _mfy(G, I, J, k)
            oΘ += s * fθ(Cf[I, J, k, KAYP], Cf[I, J, k, KAYM], Cf[I, J, k, KθY], (I, J - 1, k), (I, J, k)) / dy
        end
        if tr && _mfy(G, I, J + 1, k)
            oΘ -= s * fθ(Cf[I, J+1, k, KAYP], Cf[I, J+1, k, KAYM], Cf[I, J+1, k, KθY], (I, J, k), (I, J + 1, k)) / dy
        end
        if tr && k >= 2 && _mfz(G, I, J, k)
            oΘ += s * fθ(Cf[I, J, k, KAZP], Cf[I, J, k, KAZM], Cf[I, J, k, KθZ], (I, J, k - 1), (I, J, k)) / dz
        end
        if tr && k < nz && _mfz(G, I, J, k + 1)
            oΘ -= s * fθ(Cf[I, J, k+1, KAZP], Cf[I, J, k+1, KAZM], Cf[I, J, k+1, KθZ], (I, J, k), (I, J, k + 1)) / dz
        end
        out[I, J, k, CR] = oρ; out[I, J, k, CT] = oΘ
        oU, oV = ac ? _cmomentum_h(Z, Cf, G, dx, dy, I, J, k, 2) : (0.0, 0.0)
        out[I, J, k, CU] = oU; out[I, J, k, CV] = oV
        out[I, J, k, CW] = (ac && k >= 2 && G.Az[I, J, k] > 0) ?
            -(_wzI(G, nz, I, J, k, vimp) * (Cf[I, J, k, KPZ] * (Z[I, J, k, CT] - Z[I, J, k-1, CT]) / G.dzf[k] +
              (vimp ? grav * _ρf(G, Z, I, J, k) : 0.0)) + damp[k] * Z[I, J, k, CW]) : 0.0
    end
end

"""Shifted momenta M_side of z (direction d = 1 … 3, sides L, R: slots 2(d-1) + side) of the interior fluid cells."""
@kernel function _kc_mshift!(M, @Const(Z), G, nz, RG)
    ic, k = @index(Global, NTuple)
    I, J = _cIJ(RG, ic)
    @inbounds for d in 1:3
        if G.Vf[I, J, k] > 0
            (lL, lR), (rL, rR) = FV3D.shiftcoef(_alo(G, nz, d, I, J, k), _ahi(G, nz, d, I, J, k))
            vL, vR = _mlo(Z, nz, d, I, J, k), _mhi(Z, nz, d, I, J, k)
            M[I, J, k, 2d-1] = lL * vL + lR * vR; M[I, J, k, 2d] = rL * vL + rR * vR
        else
            M[I, J, k, 2d-1] = 0.0; M[I, J, k, 2d] = 0.0
        end
    end
end

"""Upwind transport of the shifted momenta in the interior cells (FV3D `upwind_transport!`) and, with `complete`, the
rest of the Jacobian (FV3D `_mom_complete!`)."""
@kernel function _kc_mtransport!(T, @Const(M), @Const(Z), @Const(Cf), G, dx, dy, nz, complete, RG)
    ic, k = @index(Global, NTuple)
    I, J = _cIJ(RG, ic)
    @inbounds begin
        dz = G.dzc[k]
        s = G.iV[I, J, k]
        for sl in 1:6
            d, side = (sl + 1) ÷ 2, 2 - sl % 2
            t = 0.0
            # faces: x lower (I), x upper (I+1), y lower, y upper, η lower (k), η upper (k+1); flux f, T += ±f/h
            fl(ap, am, l, r) = ap * M[l..., sl] + am * M[r..., sl]
            t += fl(Cf[I, J, k, KAXP], Cf[I, J, k, KAXM], (I - 1, J, k), (I, J, k)) / dx
            t -= fl(Cf[I+1, J, k, KAXP], Cf[I+1, J, k, KAXM], (I, J, k), (I + 1, J, k)) / dx
            t += fl(Cf[I, J, k, KAYP], Cf[I, J, k, KAYM], (I, J - 1, k), (I, J, k)) / dy
            t -= fl(Cf[I, J+1, k, KAYP], Cf[I, J+1, k, KAYM], (I, J, k), (I, J + 1, k)) / dy
            k > 1 && (t += fl(Cf[I, J, k, KAZP], Cf[I, J, k, KAZM], (I, J, k - 1), (I, J, k)) / dz)
            k < nz && (t -= fl(Cf[I, J, k+1, KAZP], Cf[I, J, k+1, KAZM], (I, J, k), (I, J, k + 1)) / dz)
            if complete
                ks = kvel(d, side)
                fc(ap, am, a, mom, l, r) = (vl = Cf[l..., ks]; vr = Cf[r..., ks];
                    a * mom * (ap > 0 ? vl : vr) - ap * vl * Z[l..., CR] - am * vr * Z[r..., CR])
                t += fc(Cf[I, J, k, KAXP], Cf[I, J, k, KAXM], G.Ax[I, J, k], Z[I, J, k, CU], (I - 1, J, k), (I, J, k)) / dx
                t -= fc(Cf[I+1, J, k, KAXP], Cf[I+1, J, k, KAXM], G.Ax[I+1, J, k], Z[I+1, J, k, CU], (I, J, k), (I + 1, J, k)) / dx
                t += fc(Cf[I, J, k, KAYP], Cf[I, J, k, KAYM], G.Ay[I, J, k], Z[I, J, k, CV], (I, J - 1, k), (I, J, k)) / dy
                t -= fc(Cf[I, J+1, k, KAYP], Cf[I, J+1, k, KAYM], G.Ay[I, J+1, k], Z[I, J+1, k, CV], (I, J, k), (I, J + 1, k)) / dy
                k > 1 && (t += fc(Cf[I, J, k, KAZP], Cf[I, J, k, KAZM], G.Az[I, J, k], Z[I, J, k, CW], (I, J, k - 1), (I, J, k)) / dz)
                k < nz && (t -= fc(Cf[I, J, k+1, KAZP], Cf[I, J, k+1, KAZM], G.Az[I, J, k+1], Z[I, J, k+1, CW], (I, J, k),
                                   (I, J, k + 1)) / dz)
            end
            T[I, J, k, sl] = t * s
        end
    end
end

"""out += A1 z on the momentum rows of the faces of masked cells (shift back of the tendencies of the two cells)."""
@kernel function _kc_mback!(out, @Const(T), G, nz, RG)
    ic, k = @index(Global, NTuple)
    I, J = _cIJ(RG, ic)
    @inbounds begin
        if _rowx(G, I, J, k)
            a = G.Ax[I, J, k]
            out[I, J, k, CU] += shift_back(T[I, J, k, 1], T[I-1, J, k, 2], G.Vf[I-1, J, k], G.Vf[I, J, k], a,
                                           G.Ax[I-1, J, k], G.Ax[I+1, J, k])
        end
        if _rowy(G, I, J, k)
            a = G.Ay[I, J, k]
            out[I, J, k, CV] += shift_back(T[I, J, k, 3], T[I, J-1, k, 4], G.Vf[I, J-1, k], G.Vf[I, J, k], a,
                                           G.Ay[I, J-1, k], G.Ay[I, J+1, k])
        end
        if _rowz(G, nz, I, J, k)
            a = G.Az[I, J, k]
            out[I, J, k, CW] += shift_back(T[I, J, k, 5], T[I, J, k-1, 6], G.Vf[I, J, k-1] * G.dzc[k-1], G.Vf[I, J, k] * G.dzc[k], a,
                                           _azs(G, nz, I, J, k - 1), _azs(G, nz, I, J, k + 1))
        end
    end
end

"""
    ka_vertical!(out, m::CutKA, Cf, Z; halo = false, parts = :all, sponge = false)

out = G_v(Z), the implicit part of the HEVI step (FV3D `:vertical` with implicit = :amf) on the interior. Linear in Z
(pressure form :product): J v = G_v(v). `halo = false`: the halo of Z must be filled; `halo = true`: only the interior
of Z, its side halo is exchanged here.

Communication overlapped with computation (CGDycore): every exchange is non-blocking (sides only: all stencils are
±1 in one direction), and while it is in flight the kernel that needs it runs on the core columns; after the receive
the same kernel on the ring of first and last rows and columns:
    [Z in flight]  vertical part, shifted momenta M (core)      ─ receive ─ ring ─ send M
    [M in flight]  transport T of M (core)                       ─ receive ─ ring ─ send T (slots 2, 4)
    [T in flight]  shift back A1 z (core)                        ─ receive ─ ring.
"""
function ka_vertical!(out, m::CutKA, Cf, Z; halo::Bool = false, parts::Symbol = :all, sponge::Bool = false)
    g, w, dec = m.grid, m.work, m.dec
    G = _geo(g)
    nz = g.nz
    ac, tr = parts !== :advection, parts !== :acoustic   # :acoustic, :advection (Rosenbrock AMF factors), :all
    damp = sponge ? m.damp : m.zdamp
    core, ring = CutRegion(g, :core), CutRegion(g, :ring)
    hz = halo ? start_halo!(dec, Z, g, (CR, CU, CV, CW, CT)) : nothing
    for R in (core, ring)
        R === ring && finish_halo!(dec, hz)
        _launch_r!(m, _kc_vertical!, R, nz, out, Z, Cf, G, g.dx, g.dy, nz, m.gas.g, ac, tr, damp, m.vimp)
        tr && _launch_r!(m, _kc_mshift!, R, nz, w.M, Z, G, nz)
    end
    tr || return out
    hm = start_halo!(dec, w.M, g, (1, 2, 3, 4, 5, 6))
    for R in (core, ring)
        R === ring && finish_halo!(dec, hm)
        _launch_r!(m, _kc_mtransport!, R, nz, w.T, w.M, Z, Cf, G, g.dx, g.dy, nz, m.complete)
    end
    ht = start_halo!(dec, w.T, g, (2, 4))
    for R in (core, ring)
        R === ring && finish_halo!(dec, ht)
        _launch_r!(m, _kc_mback!, R, nz, out, w.T, G, nz)
    end
    return out
end
"""out = :hthermo (k: ρ, Θ, W) / :hmomentum (U, V) on the columns of the region R (default: the interior)."""
ka_hthermo!(out, m::CutKA, Cf, Z, F, R::CutRegion = CutRegion(m.grid, :all)) =
    (g = m.grid; _launch_r!(m, _kc_hthermo!, R, g.nz, out, Z, Cf, F, _geo(g), m.damp, g.dx, g.dy, g.nz, m.vimp); out)
ka_hmomentum!(out, m::CutKA, Cf, Z, F, R::CutRegion = CutRegion(m.grid, :all)) =
    (g = m.grid; _launch_r!(m, _kc_hmomentum!, R, g.nz, out, Z, Cf, F, _geo(g), g.dx, g.dy, g.nz, m.gas.g, m.vimp); out)

"""
    ka_rhs!(out, m::CutKA, Q, Cf, S, tmp)

N(Q) = F_f(C(Q), Q) + S(Q) on the interior (FV3D `rhs_nonlinear!`), as :hthermo + :hmomentum + :vertical.
"""
function ka_rhs!(out, m::CutKA, Q, Cf = ka_zeros(m, NCC), S = ka_zeros(m), tmp = ka_zeros(m))
    ka_frozen_data!(Cf, S, m, Q)                                  # exchanges the halo of Q
    ka_hthermo!(out, m, Cf, Q, S)
    ka_hmomentum!(tmp, m, Cf, Q, S); out .+= tmp
    ka_vertical!(tmp, m, Cf, Q); out .+= tmp
    return out
end

# ------------------------------------------------------------------ conversion

"""Tile state Q (device of m, halo included) of the state vector x of FV3D.Model3D on the same grid."""
function ka_state(m::CutKA, g3::FV3D.Grid3D, x::AbstractVector)
    g = m.grid
    Q = zeros(g.nxh, g.nyh, g.nz + 1, NVC)
    for k in 1:g.nz, J in 1:g.nyh, I in 1:g.nxh
        i, j = mod1(g.i0 + I - HALO, g.nx), mod1(g.j0 + J - HALO, g.ny)
        Q[I, J, k, CR] = x[FV3D.iρ(g3, i, j, k)]; Q[I, J, k, CU] = x[FV3D.iU(g3, i, j, k)]
        Q[I, J, k, CV] = x[FV3D.iV(g3, i, j, k)]; Q[I, J, k, CT] = x[FV3D.iΘ(g3, i, j, k)]
        k >= 2 && (Q[I, J, k, CW] = x[FV3D.iW(g3, i, j, k)])
    end
    D = ka_zeros(m); copyto!(D, Q)
    return D
end
"""State vector of FV3D.Model3D with the interior of the tile Q (other tiles: `fill`)."""
function global_state(m::CutKA, g3::FV3D.Grid3D, Q; fill::Real = 0.0)
    g = m.grid
    KA.synchronize(KA.get_backend(m))
    A = Array(Q)
    x = Base.fill(Float64(fill), FV3D.nunknowns(g3))
    for k in 1:g.nz, j in 1:g.nyl, i in 1:g.nxl
        I, J = i + HALO, j + HALO; gi, gj = g.i0 + i, g.j0 + j
        x[FV3D.iρ(g3, gi, gj, k)] = A[I, J, k, CR]; x[FV3D.iU(g3, gi, gj, k)] = A[I, J, k, CU]
        x[FV3D.iV(g3, gi, gj, k)] = A[I, J, k, CV]; x[FV3D.iΘ(g3, gi, gj, k)] = A[I, J, k, CT]
        k >= 2 && (x[FV3D.iW(g3, gi, gj, k)] = A[I, J, k, CW])
    end
    return x
end

# ------------------------------------------------------------------ implicit step (HEVI): GMRES, column preconditioner

"""
Column solve Y = (I - c J_v)⁻¹ R per interior column (FV3D `column_solve!` with the frozen face dp/dΘ, pressure form
:product): the tridiagonal W system, then ρ, Θ from the fluxes; U, V rows identity. Preconditioner of the implicit
system (block Jacobi over the columns, no communication).
"""
@kernel function _kc_csolve!(Y, @Const(R), @Const(Cf), G, Wc, cpw, idw, c, nz, grav, vimp)
    i, j = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        for k in 2:nz
            A = G.Az[I, J, k] * _wzI(G, nz, I, J, k, vimp)
            if A == 0
                a, d, b = 0.0, 1.0, 0.0
                rhs = R[I, J, k, CW]
            else
                P = Cf[I, J, k, KPZ]
                gv = vimp ? grav : 0.0                         # vertical = :explicit: gravity explicit
                r = c / G.dzf[k]; cgl = c * gv * G.wlo[k]; cgu = c * gv * (1 - G.wlo[k])
                sk, sm = c * G.iV[I, J, k] / G.dzc[k], c * G.iV[I, J, k-1] / G.dzc[k-1]
                Am, Ap = G.Az[I, J, k-1] * _wzI(G, nz, I, J, k - 1, vimp), G.Az[I, J, k+1] * _wzI(G, nz, I, J, k + 1, vimp)
                a = k > 2 ? -r * P * sm * Am * Cf[I, J, k-1, KθZ] + cgl * sm * Am : 0.0
                d = 1 + r * A * Cf[I, J, k, KθZ] * (P * sk + P * sm) + A * (cgu * sk - cgl * sm)
                b = k < nz ? -r * P * sk * Ap * Cf[I, J, k+1, KθZ] - cgu * sk * Ap : 0.0
                rhs = R[I, J, k, CW] - r * P * (R[I, J, k, CT] - R[I, J, k-1, CT]) - (cgl * R[I, J, k-1, CR] + cgu * R[I, J, k, CR])
            end
            den = k == 2 ? d : d - a * cpw[I, J, k-1]
            idw[I, J, k] = 1 / den; cpw[I, J, k] = b / den
            Wc[I, J, k] = k == 2 ? rhs / den : (rhs - a * Wc[I, J, k-1]) / den
        end
        for k in nz-1:-1:2
            Wc[I, J, k] -= cpw[I, J, k] * Wc[I, J, k+1]
        end
        Wc[I, J, 1] = 0.0; Wc[I, J, nz+1] = 0.0
        for k in 1:nz
            sk = c * G.iV[I, J, k] / G.dzc[k]
            Fb = G.Az[I, J, k] * _wzI(G, nz, I, J, k, vimp) * Wc[I, J, k]
            Ft = G.Az[I, J, k+1] * _wzI(G, nz, I, J, k + 1, vimp) * Wc[I, J, k+1]
            Y[I, J, k, CR] = R[I, J, k, CR] - sk * (Ft - Fb)
            Y[I, J, k, CT] = R[I, J, k, CT] - sk * ((k < nz ? Cf[I, J, k+1, KθZ] : 0.0) * Ft - (k > 1 ? Cf[I, J, k, KθZ] : 0.0) * Fb)
            Y[I, J, k, CU] = R[I, J, k, CU]; Y[I, J, k, CV] = R[I, J, k, CV]
            Y[I, J, k, CW] = k >= 2 ? Wc[I, J, k] : 0.0
        end
    end
end
ka_column_solve!(Y, m::CutKA, Cf, c, R) = (g = m.grid; w = m.work;
    _launch!(m, _kc_csolve!, (g.nxl, g.nyl), Y, R, Cf, _geo(g), w.Wc, w.cpw, w.idw, Float64(c), g.nz, m.gas.g, m.vimp); Y)

# ------------------------------------------------------------------ AMF preconditioner of the tile (host)

"""
Tile model of the AMF preconditioner: FV3D.Model3D on the subgrid of the tile (decomposed directions: the tile with its
halo column on each side as overlap; others: the whole periodic direction), its coefficient vector and work vectors.
The couplings across the edges of the subgrid (its own periodicity) are removed from the matrices.
"""
struct CutAMF
    mt::FV3D.Model3D
    ox::Bool; oy::Bool                 # decomposed (overlap) in x, y
    C::Vector{Float64}
    rt::Vector{Float64}; zt::Vector{Float64}; z0::Vector{Float64}
end
function CutAMF(g::CutGrid, gas::Gas)
    ox, oy = g.px > 1, g.py > 1
    nxt = ox ? g.nxh : g.nx; nyt = oy ? g.nyh : g.ny
    gt = FV3D.subgrid(g.g3, ox ? g.i0 - HALO : 0, oy ? g.j0 - HALO : 0, nxt, nyt)
    mt = FV3D.Model3D(gt, gas, z -> 1.0, z -> 300.0; implicit = :amf)
    return CutAMF(mt, ox, oy, zeros(FV3D._ncoef(gt)), zeros(mt.nq), zeros(mt.nq), zeros(mt.nq))
end
"""Tile array index (I, J) of the subgrid cell (it, jt)."""
@inline _tI(P::CutAMF, it) = P.ox ? it : it + HALO
@inline _tJ(P::CutAMF, jt) = P.oy ? jt : jt + HALO

"""Remove the entries of A that couple unknowns across an edge of the subgrid (column distance > 3)."""
function _cut_edges(A, gt, ox, oy)
    (ox || oy) || return A
    I, J, V = findnz(A)
    keep = [(!ox || abs(FV3D.homecol(gt, r)[1] - FV3D.homecol(gt, c)[1]) <= 3) &&
            (!oy || abs(FV3D.homecol(gt, r)[2] - FV3D.homecol(gt, c)[2]) <= 3) for (r, c) in zip(I, J)]
    return sparse(I[keep], J[keep], V[keep], size(A)...)
end

"""
    ka_prec_stage!(m, Cf)

Preconditioner of the stage with the (combined) coefficients Cf: for prec = :amf the matrices J_ac, J_adv of the tile
model (FV3D `frozen_stage_begin!`); the factors follow per c in the first solve.
"""
function ka_prec_stage!(m::CutKA, Cf)
    m.prec === :amf || return nothing
    exchange!(m.dec, Cf, m.grid)
    return _amf_stage!(m.amf::CutAMF, Array(Cf), m.complete)
end
function _amf_stage!(P::CutAMF, A::Array{Float64,4}, complete::Bool)
    mt = P.mt; gt = mt.grid
    C = P.C; fill!(C, 0.0)
    for k in 1:gt.nz+1, jt in 1:gt.ny, it in 1:gt.nx
        I, J = _tI(P, it), _tJ(P, jt)
        if k <= gt.nz
            C[FV3D.cθx(gt, it, jt, k)] = A[I, J, k, KθX]; C[FV3D.cθy(gt, it, jt, k)] = A[I, J, k, KθY]
            C[FV3D.cPx(gt, it, jt, k)] = A[I, J, k, KPX]; C[FV3D.cPy(gt, it, jt, k)] = A[I, J, k, KPY]
            C[FV3D.caxp(gt, it, jt, k)] = A[I, J, k, KAXP]; C[FV3D.caxm(gt, it, jt, k)] = A[I, J, k, KAXM]
            C[FV3D.cayp(gt, it, jt, k)] = A[I, J, k, KAYP]; C[FV3D.caym(gt, it, jt, k)] = A[I, J, k, KAYM]
            for d in 1:3, sd in 1:2
                C[FV3D.cvel(gt, d, sd, it, jt, k)] = A[I, J, k, kvel(d, sd)]
            end
        end
        C[FV3D.cθz(gt, it, jt, k)] = A[I, J, k, KθZ]; C[FV3D.cPz(gt, it, jt, k)] = A[I, J, k, KPZ]
        C[FV3D.cazp(gt, it, jt, k)] = A[I, J, k, KAZP]; C[FV3D.cazm(gt, it, jt, k)] = A[I, J, k, KAZM]
    end
    pf, jf = FV3D.FV3D_PRESSURE_FORM[], FV3D.FV3D_MASK_JFORM[]
    try
        FV3D.FV3D_PRESSURE_FORM[] = :product; FV3D.FV3D_MASK_JFORM[] = complete ? :conservative : :partial
        s = FV3D.frozen_stage_begin!(mt, C, P.z0)
        s.J = _cut_edges(s.J, gt, P.ox, P.oy); s.Jadv = _cut_edges(s.Jadv, gt, P.ox, P.oy)
        s.ridx = unique(s.Jadv.rowval)
    finally
        FV3D.FV3D_PRESSURE_FORM[] = pf; FV3D.FV3D_MASK_JFORM[] = jf
    end
    return nothing
end

"""Y = AMF⁻¹ X on the interior (restricted additive Schwarz over the tiles: X on the tile with its halo, the result on
the interior)."""
function _amf_solve!(Y, m::CutKA, c, X)
    P = m.amf::CutAMF
    (P.ox || P.oy) && exchange!(m.dec, X, m.grid)
    B = _amf_host!(P, m.grid.nxl, m.grid.nyl, Float64(c), Array(X))
    copyto!(Y, B)
    return Y
end
function _amf_host!(P::CutAMF, nxl::Int, nyl::Int, c::Float64, A::Array{Float64,4})
    mt = P.mt; gt = mt.grid
    rt = P.rt
    for k in 1:gt.nz, jt in 1:gt.ny, it in 1:gt.nx
        I, J = _tI(P, it), _tJ(P, jt)
        rt[FV3D.iρ(gt, it, jt, k)] = A[I, J, k, CR]; rt[FV3D.iU(gt, it, jt, k)] = A[I, J, k, CU]
        rt[FV3D.iV(gt, it, jt, k)] = A[I, J, k, CV]; rt[FV3D.iΘ(gt, it, jt, k)] = A[I, J, k, CT]
        k >= 2 && (rt[FV3D.iW(gt, it, jt, k)] = A[I, J, k, CW])
    end
    FV3D.implicit_solve!(P.zt, mt, c, c, rt; solver = :amf)
    B = zeros(size(A))
    for k in 1:gt.nz, j in 1:nyl, i in 1:nxl
        I, J = i + HALO, j + HALO
        it, jt = P.ox ? I : i, P.oy ? J : j
        B[I, J, k, CR] = P.zt[FV3D.iρ(gt, it, jt, k)]; B[I, J, k, CU] = P.zt[FV3D.iU(gt, it, jt, k)]
        B[I, J, k, CV] = P.zt[FV3D.iV(gt, it, jt, k)]; B[I, J, k, CT] = P.zt[FV3D.iΘ(gt, it, jt, k)]
        k >= 2 && (B[I, J, k, CW] = P.zt[FV3D.iW(gt, it, jt, k)])
    end
    return B
end

"""Interior of a state array as a vector (view of the packed block (nxl, nyl, nz+1, NVC))."""
_interior(m::CutKA, Q) = (g = m.grid; view(Q, HALO+1:HALO+g.nxl, HALO+1:HALO+g.nyl, :, :))
_pack!(v, m::CutKA, Q) = (g = m.grid; reshape(v, g.nxl, g.nyl, g.nz + 1, NVC) .= _interior(m, Q); v)
_unpack!(Q, m::CutKA, v) = (g = m.grid; _interior(m, Q) .= reshape(v, g.nxl, g.nyl, g.nz + 1, NVC); Q)

"""Work of the implicit step: packed vectors, Krylov basis, two state arrays."""
struct CutGMRES{V,Mx}
    V::Mx; Z::Mx
    H::Matrix{Float64}; cs::Vector{Float64}; sn::Vector{Float64}; e::Vector{Float64}
    r::V; w::V; x::V; b::V
    rh::V; p::V; v::V; s::V; t::V; ph::V; sh::V    # BiCGStab
    X::Any; Y::Any                     # state-shaped work arrays
end
function CutGMRES(m::CutKA, restart::Int = 30)
    g = m.grid
    n = g.nxl * g.nyl * (g.nz + 1) * NVC
    be = KA.get_backend(m)
    vz() = KA.zeros(be, Float64, n)
    return CutGMRES(KA.zeros(be, Float64, n, restart + 1), KA.zeros(be, Float64, n, restart), zeros(restart + 1, restart),
                    zeros(restart), zeros(restart), zeros(restart + 1), vz(), vz(), vz(), vz(),
                    vz(), vz(), vz(), vz(), vz(), vz(), vz(), ka_zeros(m), ka_zeros(m))
end

_dot(m::CutKA, a, b) = allreduce_sum(m.dec, dot(a, b))
"""w -= V h with h = Vᵀ w (classical Gram–Schmidt: one gemv and one allreduce of the vector h for all basis vectors,
instead of one reduction per vector); returns h (host). Applied twice (CGS2): as stable as modified Gram–Schmidt."""
function _project!(w, m::CutKA, V)
    h = allreduce_sum(m.dec, Array(V' * w))
    hd = similar(w, length(h)); copyto!(hd, h)
    mul!(w, V, hd, -1.0, 1.0)
    return h
end
_norm(m::CutKA, a) = sqrt(_dot(m, a, a))

"""out = (I - c J) x (packed; halo exchange): J v = G_v(v) (op = :vertical, HEVI), or the Rosenbrock factors on the
all-masked model `m` (op = :acoustic: global acoustics with the sponge; :advection: upwind transport; :full: both)."""
function _cut_apply!(out, m::CutKA, Cf, c, x, W::CutGMRES; op::Symbol = :vertical)
    _unpack!(W.X, m, x)
    if op === :vertical
        ka_vertical!(W.Y, m, Cf, W.X; halo = true)            # side halo of X overlapped
    else
        ka_vertical!(W.Y, m, Cf, W.X; halo = true, parts = op === :full ? :all : op, sponge = op !== :advection)
    end
    _pack!(out, m, W.Y)
    @. out = x - c * out
    return out
end
"""z = M⁻¹ r (packed): column solve (prec = :columns) or tile AMF (:amf)."""
function _cut_prec!(z, m::CutKA, Cf, c, r, W::CutGMRES; op::Symbol = :vertical)
    op === :advection && return copyto!(z, r)                 # upwind transport: no preconditioner
    _unpack!(W.X, m, r)
    if op === :vertical && m.prec === :mask                   # masked AMF: Θ Schur + upwind transport (cut_mask.jl)
        mask_precondition!(W.Y, m.mprec[], m, Cf, c, W.X)
        return _pack!(z, m, W.Y)
    end
    if op === :acoustic && m.mgp[] !== nothing                # acoustic multigrid (Schur, V-cycle)
        mg = m.mgp[]
        mg_precondition!(W.Y, mg, m, Cf, c, W.X, mg.X, mg.Y)
        return _pack!(z, m, W.Y)
    end
    if m.prec === :amf && op === :vertical
        _amf_solve!(W.Y, m, c, W.X)
    else
        ka_column_solve!(W.Y, m, Cf, c, W.X)
    end
    return _pack!(z, m, W.Y)
end

"""Right-preconditioned restarted GMRES for (I - c J) x = b, x the initial guess; returns (iterations, matrix-vector
products)."""
function _cut_gmres!(x, m::CutKA, Cf, c, b, W::CutGMRES; tol = 1e-8, maxit = 300, op::Symbol = :vertical)
    mr = size(W.H, 2)
    V, Z, H, cs, sn, e, r, w = W.V, W.Z, W.H, W.cs, W.sn, W.e, W.r, W.w
    nb = _norm(m, b); nb == 0 && (fill!(x, 0.0); return 0, 0)
    it = 0; nmv = 0
    while true
        _cut_apply!(r, m, Cf, c, x, W; op); @. r = b - r; nmv += 1
        β = _norm(m, r)
        (β <= tol * nb || it >= maxit) && return it, nmv
        @views V[:, 1] .= r ./ β
        fill!(e, 0.0); e[1] = β
        j = 0
        for jj in 1:mr
            j = jj; it += 1
            @views _cut_prec!(Z[:, jj], m, Cf, c, V[:, jj], W; op)
            @views _cut_apply!(w, m, Cf, c, Z[:, jj], W; op); nmv += 1
            @views Vj = V[:, 1:jj]
            @views H[1:jj, jj] .= _project!(w, m, Vj) .+ _project!(w, m, Vj)     # CGS2
            H[jj+1, jj] = _norm(m, w)
            @views V[:, jj+1] .= w ./ H[jj+1, jj]
            for i in 1:jj-1
                t = cs[i] * H[i, jj] + sn[i] * H[i+1, jj]
                H[i+1, jj] = -sn[i] * H[i, jj] + cs[i] * H[i+1, jj]; H[i, jj] = t
            end
            ρ = hypot(H[jj, jj], H[jj+1, jj])
            cs[jj] = H[jj, jj] / ρ; sn[jj] = H[jj+1, jj] / ρ
            H[jj, jj] = ρ; H[jj+1, jj] = 0.0
            e[jj+1] = -sn[jj] * e[jj]; e[jj] *= cs[jj]
            (abs(e[jj+1]) <= tol * nb || it >= maxit) && break
        end
        y = UpperTriangular(H[1:j, 1:j]) \ e[1:j]
        yd = similar(x, j); copyto!(yd, y)
        @views mul!(x, Z[:, 1:j], yd, 1.0, 1.0)
    end
end

"""
Right-preconditioned BiCGStab (van der Vorst 1992) for (I - c J) x = b, x the initial guess; returns (iterations,
matrix-vector products). Two products and two preconditioner solves per iteration, eight vectors.
"""
function _cut_bicgstab!(x, m::CutKA, Cf, c, b, W::CutGMRES; tol = 1e-8, maxit = 300)
    r, rh, p, v, s, t, ph, sh = W.r, W.rh, W.p, W.v, W.s, W.t, W.ph, W.sh
    nb = _norm(m, b); nb == 0 && (fill!(x, 0.0); return 0, 0)
    _cut_apply!(r, m, Cf, c, x, W); @. r = b - r; nmv = 1
    _norm(m, r) <= tol * nb && return 0, nmv
    rh .= r; fill!(p, 0.0); fill!(v, 0.0)
    ρ = α = ω = 1.0
    for it in 1:maxit
        ρn = _dot(m, rh, r)
        β = (ρn / ρ) * (α / ω); ρ = ρn
        @. p = r + β * (p - ω * v)
        _cut_prec!(ph, m, Cf, c, p, W)
        _cut_apply!(v, m, Cf, c, ph, W); nmv += 1
        α = ρ / _dot(m, rh, v)
        @. s = r - α * v
        if _norm(m, s) <= tol * nb
            @. x += α * ph
            return it, nmv
        end
        _cut_prec!(sh, m, Cf, c, s, W)
        _cut_apply!(t, m, Cf, c, sh, W); nmv += 1
        ω = _dot(m, t, s) / _dot(m, t, t)
        @. x += α * ph + ω * sh
        @. r = s - ω * t
        _norm(m, r) <= tol * nb && return it, nmv
    end
    return maxit, nmv
end

"""Interior update Z += a k (the halo is exchanged before every use)."""
_axpy!(Z, m::CutKA, a, k) = (_interior(m, Z) .+= a .* _interior(m, k); Z)

@kernel function _kc_conserve!(Y, @Const(K), @Const(T), c)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        Y[I, J, k, CR] = K[I, J, k, CR] + c * T[I, J, k, CR]
        Y[I, J, k, CT] = K[I, J, k, CT] + c * T[I, J, k, CT]
    end
end

"""
    ka_vertical_step!(Z, m, Cf, h, k, y, W; tol, krylov = :gmres)

Linearly implicit Crank–Nicolson step z ← z + (I - h/2 J)⁻¹ h G_v(z) (FV3D `frozen_vertical_step!` with
FV3D_MASK_SOLVER = :gmres): GMRES (`krylov = :gmres`) or BiCGStab (`:bicgstab`) on all unknowns of the tile, column
preconditioner, initial guess the column solve; ρ and Θ afterwards from the fluxes of the solution (conserved for any
tolerance).
"""
function ka_vertical_step!(Z, m::CutKA, Cf, h, k, y, W::CutGMRES; tol = 1e-8, krylov::Symbol = :gmres)
    g = m.grid
    c = FV3D.FV3D_HEVI_BETA[] * h                      # β = 1/2 Crank–Nicolson, > 1/2 off-centred (FV3D_HEVI_BETA)
    ka_vertical!(k, m, Cf, Z; halo = true); _interior(m, k) .*= h
    _pack!(W.b, m, k)
    _cut_prec!(W.x, m, Cf, c, W.b, W)
    krylov in (:gmres, :bicgstab) || error("krylov: :gmres or :bicgstab")
    it, nmv = (m.prec === :mask && m.mprec[].method.outer === :none) ? (0, 0) :      # AMF as the solve (ASAM)
              krylov === :gmres ? _cut_gmres!(W.x, m, Cf, c, W.b, W; tol) : _cut_bicgstab!(W.x, m, Cf, c, W.b, W; tol)
    push!(m.its, it); push!(m.mv, nmv)
    _unpack!(y, m, W.x)
    ka_vertical!(W.Y, m, Cf, y; halo = true)
    _launch!(m, _kc_conserve!, (g.nxl, g.nyl, g.nz), y, k, W.Y, c)
    return _axpy!(Z, m, 1.0, y)
end

"""Work arrays of the MIS step of the cut-cell model."""
struct CutMISWork{A}
    C::Vector{A}; S::Vector{A}; ΔY::Vector{A}
    c::A; F::A; Y::A; Z::A; k::A; y::A
    zl::A; zt::A                                   # divergence damping: state at the last momentum step, extrapolated state
    gm::CutGMRES
end
CutMISWork(m::CutKA, s::Int; restart::Int = 30) =
    CutMISWork([ka_zeros(m, NCC) for _ in 1:s], [ka_zeros(m) for _ in 1:s], [ka_zeros(m) for _ in 1:s], ka_zeros(m, NCC),
               ka_zeros(m), ka_zeros(m), ka_zeros(m), ka_zeros(m), ka_zeros(m), ka_zeros(m), ka_zeros(m), CutGMRES(m, restart))

"""HEVI small steps (TimeStepping `frozen_fast_solve!` with method = :hevi): Strang splitting, horizontal Verlet half
steps on the unmasked faces, the implicit Crank–Nicolson step in the middle (β = FV3D_HEVI_BETA). `smdiv = γd > 0`:
divergence damping as WRF (TimeStepping `smdiv`): the horizontal pressure gradient at Θ̃ = Θ + γd (Θ - Θ_prev), Θ_prev
at the previous momentum step of the stage."""
function _cut_hevi_loop!(Z, m::CutKA, Cf, F, h, N, w::CutMISWork; tol, krylov = :gmres, smdiv::Float64 = 0.0)
    g, dec, k = m.grid, m.dec, w.k
    core, ring = CutRegion(g, :core), CutRegion(g, :ring)
    ka_prec_stage!(m, Cf)
    m.prec === :mask && mask_setup!(m.mprec[], m, Cf, FV3D.FV3D_HEVI_BETA[] * h)
    damp = smdiv > 0
    damp && (w.zl .= Z)
    # one Verlet sub-step: the halo of `vars` (the variables the kernel reads from the neighbours) of the input X in
    # flight while the kernel runs on the core, the ring after the receive, then Z += a k on the interior
    function sub!(kernel!, X, vars, a)
        hz = start_halo!(dec, X, g, vars)
        kernel!(k, m, Cf, X, F, core)
        finish_halo!(dec, hz)
        kernel!(k, m, Cf, X, F, ring)
        return _axpy!(Z, m, a, k)
    end
    for _ in 1:N, half in 1:2
        sub!(ka_hthermo!, Z, (CU, CV), h / 4)                 # fluxes: U, V of the east / north neighbours
        if damp                                               # pressure gradient at the extrapolated Θ
            @views @. w.zt[:, :, :, CT] = Z[:, :, :, CT] + smdiv * (Z[:, :, :, CT] - w.zl[:, :, :, CT])
            @views w.zl[:, :, :, CT] .= Z[:, :, :, CT]
            sub!(ka_hmomentum!, w.zt, (CT,), h / 2)
        else
            sub!(ka_hmomentum!, Z, (CT,), h / 2)              # pressure gradient: Θ of the west / south neighbours
        end
        sub!(ka_hthermo!, Z, (CU, CV), h / 4)
        half == 1 && ka_vertical_step!(Z, m, Cf, h, k, w.y, w.gm; tol, krylov)
    end
    return Z
end

"""
    ka_mis_step!(m::CutKA, q, Δt, tab, w::CutMISWork; nsub = 12, tol = 1e-8, krylov = :gmres)

One MIS step (TimeStepping `frozen_mis_step!` with fast = :hevi) of the cut-cell model, in place.
"""
function ka_mis_step!(m::CutKA, q, Δt, tab::MISTableau, w::CutMISWork; nsub::Int = 12, tol::Real = 1e-8,
                      krylov::Symbol = :gmres, smdiv::Real = 0.0)
    for i in 1:nstages(tab)
        i == 1 ? (w.Y .= q) : (@. w.Y = q + w.ΔY[i-1])
        ka_frozen_data!(w.C[i], w.S[i], m, w.Y)
        d = tab.d[i]
        w.c .= 0; w.F .= 0; w.Z .= q
        for j in 1:i
            b = tab.β[i, j] / d
            if b != 0
                @. w.c += b * w.C[j]
                @. w.F += b * w.S[j]
            end
            if j < i
                a, gm = tab.α[i, j], tab.γ[i, j] / (d * Δt)
                ΔYj = w.ΔY[j]
                a != 0 && (@. w.Z += a * ΔYj)
                gm != 0 && (@. w.F += gm * ΔYj)
            end
        end
        n = fast_substeps(d, nsub)
        _cut_hevi_loop!(w.Z, m, w.c, w.F, d * Δt / n, n, w; tol = Float64(tol), krylov, smdiv = Float64(smdiv))
        @. w.ΔY[i] = w.Z - q
    end
    q .= w.Z
    return q
end

"""nsteps MIS steps of the cut-cell model (in place); `monitor(n, q)` every `every` steps."""
function ka_run!(m::CutKA, q, Δt, nsteps; tab::MISTableau, nsub::Int = 12, tol::Real = 1e-8, krylov::Symbol = :gmres,
                 smdiv::Real = 0.0, monitor = nothing, every::Int = 1)
    w = CutMISWork(m, nstages(tab))
    for n in 1:nsteps
        ka_mis_step!(m, q, Δt, tab, w; nsub, tol, krylov, smdiv)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end
