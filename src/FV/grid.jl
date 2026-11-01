# Grid: Arakawa C-grid in a Gal-Chen terrain-following vertical coordinate, periodic in x

"""
    FVGrid(nx, nz, Lx, H; x0=0)

Uniform C-grid with nx × nz computational cells on [x0, x0 + Lx] × [0, H]. The physical
height is `z(x,η) = zs(x) + η (H-zs(x))/H`; `zs` defaults to a flat lower boundary. `zc` and
`zf` are computational vertical levels, while `Zc` and `Zf` are their physical heights at cell
centres and vertical faces. The vertical metric is `J = ∂z/∂η`; `zx_c` and `zx_f` store
`∂z/∂x|η` at cell and vertical-face locations.
"""
struct FVGrid
    nx::Int
    nz::Int
    x0::Float64
    Lx::Float64
    H::Float64
    dx::Float64
    dz::Float64
    xc::Vector{Float64}
    zc::Vector{Float64}
    xf::Vector{Float64}
    zf::Vector{Float64}
    Zc::Matrix{Float64}           # physical cell-centre heights (nx, nz)
    Zf::Matrix{Float64}           # physical vertical-face heights (nx, nz+1)
    Zx::Matrix{Float64}           # physical x-face heights at cell-centre η (nx, nz)
    Zv::Matrix{Float64}           # physical mesh-vertex heights (nx+1, nz+1)
    Jc::Vector{Float64}           # ∂z/∂η at cell-centre x locations
    Jx::Vector{Float64}           # ∂z/∂η at x-face locations
    zx_c::Matrix{Float64}         # ∂z/∂x|η at cell centres
    zx_f::Matrix{Float64}         # ∂z/∂x|η at vertical faces
    zx_xf::Matrix{Float64}        # ∂z/∂x|η at x-face / vertical-face intersections
    terrain::Bool
end

function FVGrid(nx::Int, nz::Int, Lx::Real, H::Real; x0::Real = 0.0, zs = x -> 0.0, zs_x = nothing)
    nx > 1 && nz > 1 || throw(ArgumentError("FVGrid needs nx,nz > 1"))
    Lx > 0 && H > 0 || throw(ArgumentError("FVGrid dimensions must be positive"))
    dx = Lx / nx
    dz = H / nz
    xc = [x0 + (i - 0.5) * dx for i in 1:nx]
    zc = [(k - 0.5) * dz for k in 1:nz]
    xf = [x0 + (i - 1) * dx for i in 1:nx]
    zf = [(k - 1) * dz for k in 1:nz+1]
    zsc = Float64.(zs.(xc))
    zsf = Float64.(zs.(xf))
    maximum(zsc) < H && minimum(zsc) >= 0 || throw(ArgumentError("terrain must satisfy 0 ≤ zs(x) < H"))
    maximum(zsf) < H && minimum(zsf) >= 0 || throw(ArgumentError("terrain must satisfy 0 ≤ zs(x) < H"))
    slope(x) = zs_x === nothing ? (zs(x + dx / 2) - zs(x - dx / 2)) / dx : zs_x(x)
    dzsdx_c = Float64.(slope.(xc))
    dzsdx_f = zs_x === nothing ?
        [(zsc[i] - zsc[mod1(i - 1, nx)]) / dx for i in 1:nx] : Float64.(zs_x.(xf))
    Jc = 1 .- zsc ./ H
    Jx = 1 .- zsf ./ H
    Zc = [zsc[i] + zc[k] * Jc[i] for i in 1:nx, k in 1:nz]
    Zf = [zsc[i] + zf[k] * Jc[i] for i in 1:nx, k in 1:nz+1]
    Zx = [zsf[i] + zc[k] * Jx[i] for i in 1:nx, k in 1:nz]
    xv = [x0 + (i - 1) * dx for i in 1:nx+1]
    zsv = Float64.(zs.(xv))
    Zv = [zsv[i] + zf[k] * (1 - zsv[i] / H) for i in 1:nx+1, k in 1:nz+1]
    zx_c = [dzsdx_c[i] * (1 - zc[k] / H) for i in 1:nx, k in 1:nz]
    zx_f = [dzsdx_c[i] * (1 - zf[k] / H) for i in 1:nx, k in 1:nz+1]
    zx_xf = [dzsdx_f[i] * (1 - zf[k] / H) for i in 1:nx, k in 1:nz+1]
    terrain = any(x -> !iszero(x), zsc) || any(x -> !iszero(x), zsf)
    return FVGrid(nx, nz, Float64(x0), Float64(Lx), Float64(H), dx, dz, xc, zc, xf, zf,
                  Zc, Zf, Zx, Zv, Jc, Jx, zx_c, zx_f, zx_xf, terrain)
end

ncell(g::FVGrid) = g.nx * g.nz
nunknowns(g::FVGrid) = 3 * ncell(g) + g.nx * (g.nz - 1)

# indices into the state vector (i periodic in x; W only at interior faces k = 2 .. nz)
@inline ip(g::FVGrid, i) = mod1(i, g.nx)
@inline iρ(g::FVGrid, i, k) = ip(g, i) + (k - 1) * g.nx
@inline iU(g::FVGrid, i, k) = ncell(g) + ip(g, i) + (k - 1) * g.nx
@inline iW(g::FVGrid, i, k) = 2 * ncell(g) + ip(g, i) + (k - 2) * g.nx
@inline iΘ(g::FVGrid, i, k) = 2 * ncell(g) + g.nx * (g.nz - 1) + ip(g, i) + (k - 1) * g.nx

"""W at z-face k of column i, zero at the walls."""
@inline Wface(g::FVGrid, q, i, k) = (k == 1 || k == g.nz + 1) ? 0.0 : q[iW(g, i, k)]
