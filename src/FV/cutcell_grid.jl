# Cartesian cut-cell geometry for the staggered FV discretization.

"""Geometry for a Cartesian C-grid cut by a terrain profile `zs(x)`.

`Vol[i,k]` is the fluid area in a cell, `FU` the open vertical-face length, and `FW` the open
horizontal-face length. The representation follows the MATLAB `CutCellGeometry` layout while
keeping the terrain-following `FVGrid` unchanged.
"""
struct CutCellFVGrid
    nx::Int
    nz::Int
    x0::Float64
    z0::Float64
    Lx::Float64
    H::Float64
    dx::Float64
    dz::Float64
    xc::Vector{Float64}
    zc::Vector{Float64}
    xf::Vector{Float64}
    zf::Vector{Float64}
    Zc::Matrix{Float64}
    Zf::Matrix{Float64}
    Zx::Matrix{Float64}
    Zv::Matrix{Float64}
    Vol::Matrix{Float64}       # fluid cell areas, (nx, nz)
    FU::Matrix{Float64}        # open vertical-face lengths, (nx+1, nz)
    FW::Matrix{Float64}        # open horizontal-face lengths, (nx, nz+1)
    alpha::Matrix{Float64}     # Vol / (dx dz)
    alphaU::Matrix{Float64}    # FU / dz
    alphaW::Matrix{Float64}    # FW / dx
    MaskVol::BitMatrix         # AMF implicit cells (heuristic, see `mask`)
    MaskU::BitMatrix
    MaskW::BitMatrix
    polygons::Matrix{Vector{NTuple{2,Float64}}}
    terrain::Bool
    zs::Any
end

ncell(g::CutCellFVGrid) = g.nx * g.nz
nunknowns(g::CutCellFVGrid) = 3 * ncell(g) + g.nx * (g.nz - 1)
@inline ip(g::CutCellFVGrid, i) = mod1(i, g.nx)
@inline iρ(g::CutCellFVGrid, i, k) = ip(g, i) + (k - 1) * g.nx
@inline iU(g::CutCellFVGrid, i, k) = ncell(g) + ip(g, i) + (k - 1) * g.nx
@inline iW(g::CutCellFVGrid, i, k) = 2 * ncell(g) + ip(g, i) + (k - 2) * g.nx
@inline iΘ(g::CutCellFVGrid, i, k) = 2 * ncell(g) + g.nx * (g.nz - 1) + ip(g, i) + (k - 1) * g.nx
@inline Wface(g::CutCellFVGrid, q, i, k) = (k == 1 || k == g.nz + 1) ? 0.0 : q[iW(g, i, k)]
@inline function cutcell_sample_z(g::CutCellFVGrid, i, k)
    g.Vol[i,k] == 0 && return g.zc[k]
    lo = clamp(g.zs(g.xc[i]), g.zf[k], g.zf[k+1])
    return clamp(max(g.zc[k], (lo + g.zf[k+1]) / 2), lo, g.zf[k+1])
end

const CUT_GAUSS_X = (-0.9602898564975363, -0.7966664774136267, -0.5255324099163290,
                     -0.1834346424956498, 0.1834346424956498, 0.5255324099163290,
                     0.7966664774136267, 0.9602898564975363)
const CUT_GAUSS_W = (0.1012285362903763, 0.2223810344533745, 0.3137066458778873,
                     0.3626837833783620, 0.3626837833783620, 0.3137066458778873,
                     0.2223810344533745, 0.1012285362903763)

@inline function cut_area(zs, xl, xr, zl, zr)
    mid, half = (xl + xr) / 2, (xr - xl) / 2
    a = 0.0
    @inbounds for j in eachindex(CUT_GAUSS_X)
        x = mid + half * CUT_GAUSS_X[j]
        a += CUT_GAUSS_W[j] * clamp(zr - max(zl, zs(x)), 0.0, zr - zl)
    end
    return half * a
end

function cut_open_xlength(zs, xl, xr, z)
    # Integrate the fluid indicator by locating terrain crossings on a small subdivision.
    # Crossing positions are bisected to near machine precision; this handles both shoulders
    # of a single-hill profile without smearing open-face lengths over the mesh spacing.
    n = 32
    dx = (xr - xl) / n
    length_open = 0.0
    xa = xl
    fa = z - zs(xa)
    @inbounds for j in 1:n
        xb = j == n ? xr : xl + j * dx
        fb = z - zs(xb)
        if (fa >= 0) == (fb >= 0)
            fa >= 0 && (length_open += xb - xa)
        else
            lo, hi, flo = xa, xb, fa
            for _ in 1:48
                xm = (lo + hi) / 2
                fm = z - zs(xm)
                if (fm >= 0) == (flo >= 0)
                    lo, flo = xm, fm
                else
                    hi = xm
                end
            end
            root = (lo + hi) / 2
            length_open += fa >= 0 ? root - xa : xb - root
        end
        xa, fa = xb, fb
    end
    return length_open
end

function cut_polygon(zs, xl, xr, zl, zr)
    # Polygon metadata is a linear-in-x approximation to the terrain inside this cell, as in
    # the MATLAB display geometry. Conservative areas and face apertures are integrated above.
    hL, hR = zs(xl), zs(xr)
    h(x) = hL + (hR - hL) * ((x - xl) / (xr - xl))
    pts = [(xl, zl), (xr, zl), (xr, zr), (xl, zr)]
    out = NTuple{2,Float64}[]
    for j in eachindex(pts)
        p, q = pts[j], pts[mod1(j + 1, length(pts))]
        fp, fq = p[2] - h(p[1]), q[2] - h(q[1])
        pin, qin = fp >= 0, fq >= 0
        if pin && qin
            push!(out, q)
        elseif pin && !qin
            t = fp / (fp - fq)
            push!(out, (p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
        elseif !pin && qin
            t = fp / (fp - fq)
            push!(out, (p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
            push!(out, q)
        end
    end
    return out
end

"""    CutCellFVGrid(nx, nz, Lx, H; x0=0, z0=0, zs=x->0, mask=:diag)

Construct Cartesian cut-cell areas and apertures for fluid above `z = zs(x)`. Horizontal x is
periodic in the FV equations; `zs` should therefore be periodic over `Lx` (or negligible at the
domain endpoints). Geometry is computed once when the grid is built. `mask` selects the
heuristic AMF implicit region (`:diag`, `:cut`, `:cut4`, `:cut8`; `:all` for diagnostics).
"""
function CutCellFVGrid(nx::Int, nz::Int, Lx::Real, H::Real; x0::Real = 0.0, z0::Real = 0.0,
                       zs = x -> 0.0, mask::Symbol = :diag)
    nx > 1 && nz > 1 || throw(ArgumentError("CutCellFVGrid needs nx,nz > 1"))
    Lx > 0 && H > z0 || throw(ArgumentError("invalid cut-cell domain dimensions"))
    dx, dz = Float64(Lx) / nx, (Float64(H) - Float64(z0)) / nz
    xc = [x0 + (i - 0.5) * dx for i in 1:nx]
    zc = [z0 + (k - 0.5) * dz for k in 1:nz]
    xf = [x0 + (i - 1) * dx for i in 1:nx+1]
    zf = [z0 + (k - 1) * dz for k in 1:nz+1]
    heights = Float64.(zs.(range(Float64(x0), Float64(x0)+Float64(Lx); length=max(4nx,64))))
    all(isfinite, heights) && minimum(heights) >= z0 && maximum(heights) < H ||
        throw(ArgumentError("cut-cell terrain must be finite and satisfy z0 ≤ zs(x) < H"))
    Zc = repeat(reshape(zc, 1, :), nx, 1)
    Zf = repeat(reshape(zf, 1, :), nx, 1)
    Zx = copy(Zc)
    Zv = repeat(reshape(zf, 1, :), nx+1, 1)
    Vol = [cut_area(zs, xf[i], xf[i+1], zf[k], zf[k+1]) for i in 1:nx, k in 1:nz]
    FU = [clamp(zf[k+1] - max(zf[k], zs(xf[i])), 0.0, dz) for i in 1:nx+1, k in 1:nz]
    FW = [cut_open_xlength(zs, xf[i], xf[i+1], zf[k]) for i in 1:nx, k in 1:nz+1]
    alpha = clamp.(Vol ./ (dx * dz), 0.0, 1.0)
    alphaU = clamp.(FU ./ dz, 0.0, 1.0)
    alphaW = clamp.(FW ./ dx, 0.0, 1.0)
    # Heuristic implicit region for the AMF solver. Fluid cell (i,k) is masked when a non-full
    # cell (cut or solid, as in MATLAB `CutCellGeometry`: Vol < dx dz) lies at one of the
    # listed offsets (di,dk); (0,0) includes the cell itself. As in MATLAB, k±1 is clamped to
    # the domain (bottom-row cells look at their horizontal neighbours); x is periodic here.
    cut = alpha .< 1 - 64eps(Float64)
    offsets = mask == :diag  ? ((-1,-1), (-1,1), (1,-1), (1,1)) :
              mask == :cut   ? ((0,0),) :
              mask == :cut4  ? ((0,0), (-1,0), (1,0), (0,-1), (0,1)) :
              mask == :cut8  ? Tuple((di,dk) for di in -1:1 for dk in -1:1) :
              mask == :all   ? () :
              throw(ArgumentError("unknown cut-cell mask $mask (use :diag, :cut, :cut4, :cut8, :all)"))
    MaskVol = falses(nx, nz)
    for k in 1:nz, i in 1:nx
        alpha[i,k] > 0 || continue
        mask == :all && (MaskVol[i,k] = true; continue)   # diagnostic: everything implicit
        for (di, dk) in offsets
            kk = clamp(k + dk, 1, nz)
            ii = mod1(i + di, nx)
            cut[ii,kk] && (MaskVol[i,k] = true)
        end
    end
    MaskU = falses(nx+1, nz)
    MaskW = falses(nx, nz+1)
    for k in 1:nz, i in 1:nx
        if MaskVol[i,k]
            MaskU[i,k] = true
            MaskU[i+1,k] = true
            MaskW[i,k] = true
            MaskW[i,k+1] = true
        end
    end
    # x-periodic duplicate apertures/masks share one geometric face.
    FU[end,:] .= FU[1,:]
    alphaU[end,:] .= alphaU[1,:]
    MaskU[end,:] .= MaskU[1,:]
    polygons = [cut_polygon(zs, xf[i], xf[i+1], zf[k], zf[k+1]) for i in 1:nx, k in 1:nz]
    return CutCellFVGrid(nx, nz, Float64(x0), Float64(z0), Float64(Lx), Float64(H), dx, dz,
                         xc, zc, xf, zf, Zc, Zf, Zx, Zv, Vol, FU, FW, alpha, alphaU, alphaW,
                         MaskVol, MaskU, MaskW, polygons, any(x -> !iszero(x), zs.(xc)), zs)
end
