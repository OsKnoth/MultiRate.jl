# Tile of a rank: nxl × nyl full columns of the global periodic nx × ny × nz grid, with HALO cells on each side in x
# and y. Local index I = HALO + i of the global index i0 + i (periodic). The metric is built from the terrain
# function with the formulas of FV3D.Grid3D (terrain = :follow), so that the operator is the same up to round-off.

"""Halo width: one cell. The third-order face values of the tile-boundary faces are computed by the upwind tile and
exchanged (`merge_faces!`); the shift technique of the momentum advection exchanges the shifted velocities and their
cell tendencies (data.jl)."""
const HALO = 1

"""
    KAGrid(nx, ny, nz, Lx, Ly, H; zs = nothing, px = 1, py = 1, cx = 0, cy = 0, backend = CPU())

Tile (cx, cy) (0-based) of a px × py decomposition of the periodic C-grid with nx × ny × nz cells on
[0, Lx] × [0, Ly] × [0, H], flat or terrain following (Gal-Chen) with the terrain `zs(x, y)`. nx, ny must be
divisible by px, py. Metric arrays on the device of `backend` (with halo).
"""
struct KAGrid{M2,M3}
    nx::Int; ny::Int; nz::Int                  # global
    nxl::Int; nyl::Int                         # interior of the tile
    i0::Int; j0::Int                           # global index of the first interior cell minus 1
    nxh::Int; nyh::Int                         # with halo
    Lx::Float64; Ly::Float64; H::Float64
    dx::Float64; dy::Float64; dz::Float64
    follow::Bool
    Jc::M2; Jx::M2; Jy::M2                     # (nxh+1, nyh+1)
    zxf::M3; zyf::M3; zxx::M3; zyy::M3         # (nxh, nyh, nz+1)
    zf::Vector{Float64}                        # η of the faces (host)
end

function KAGrid(nx, ny, nz, Lx, Ly, H; zs = nothing, px::Int = 1, py::Int = 1, cx::Int = 0, cy::Int = 0,
                backend = CPU())
    (nx % px == 0 && ny % py == 0) || error("KAGrid: nx, ny must be divisible by px, py")
    nxl, nyl = nx ÷ px, ny ÷ py
    (nxl >= 3 && nyl >= 3) || error("KAGrid: tiles need at least 3 cells per direction")
    i0, j0 = cx * nxl, cy * nyl
    nxh, nyh = nxl + 2HALO, nyl + 2HALO
    dx, dy, dz = Lx / nx, Ly / ny, H / nz
    follow = zs !== nothing
    h(x, y) = follow ? Float64(zs(x, y)) : 0.0
    xc(i) = (i - 0.5) * dx; yc(j) = (j - 0.5) * dy; xf(i) = (i - 1) * dx; yf(j) = (j - 1) * dy
    zf = [(k - 1) * dz for k in 1:nz+1]
    gi(I) = mod1(i0 + I - HALO, nx); gj(J) = mod1(j0 + J - HALO, ny)
    zsc(i, j) = h(xc(i), yc(j))
    # 2D metric with one more column and row (the outer faces of the right and upper halo cells)
    Jc = [1 - zsc(gi(I), gj(J)) / H for I in 1:nxh+1, J in 1:nyh+1]
    Jx = [1 - h(xf(gi(I)), yc(gj(J))) / H for I in 1:nxh+1, J in 1:nyh+1]
    Jy = [1 - h(xc(gi(I)), yf(gj(J))) / H for I in 1:nxh+1, J in 1:nyh+1]
    sx(i, j) = (h(xc(i) + dx / 2, yc(j)) - h(xc(i) - dx / 2, yc(j))) / dx
    sy(i, j) = (h(xc(i), yc(j) + dy / 2) - h(xc(i), yc(j) - dy / 2)) / dy
    sxx(i, j) = (zsc(i, j) - zsc(mod1(i - 1, nx), j)) / dx
    syy(i, j) = (zsc(i, j) - zsc(i, mod1(j - 1, ny))) / dy
    fac = [1 - zf[k] / H for k in 1:nz+1]
    zxf = [sx(gi(I), gj(J)) * fac[k] for I in 1:nxh, J in 1:nyh, k in 1:nz+1]
    zyf = [sy(gi(I), gj(J)) * fac[k] for I in 1:nxh, J in 1:nyh, k in 1:nz+1]
    zxx = [sxx(gi(I), gj(J)) * fac[k] for I in 1:nxh, J in 1:nyh, k in 1:nz+1]
    zyy = [syy(gi(I), gj(J)) * fac[k] for I in 1:nxh, J in 1:nyh, k in 1:nz+1]
    (maximum(Jc) <= 1 + 1e-15 && minimum(Jc) > 0) || error("terrain must satisfy 0 ≤ zs < H")
    dev(a) = (b = KA.allocate(backend, Float64, size(a)...); copyto!(b, a); b)
    return KAGrid(nx, ny, nz, nxl, nyl, i0, j0, nxh, nyh, Float64(Lx), Float64(Ly), Float64(H), dx, dy, dz, follow,
                  dev(Jc), dev(Jx), dev(Jy), dev(zxf), dev(zyf), dev(zxx), dev(zyy), zf)
end

KA.get_backend(g::KAGrid) = KA.get_backend(g.Jc)
