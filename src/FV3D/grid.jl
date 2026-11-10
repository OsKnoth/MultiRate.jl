# Grid: Cartesian C-grid in 3D, periodic in x and y, walls at z = 0 and z = H, optionally cut by a
# terrain z_s(x, y) (cut cells). The flat grid is the special case with all fractions equal to 1.

"""
    Grid3D(nx, ny, nz, Lx, Ly, H; zs = nothing, box = nothing, αmin = 0.2)

Uniform C-grid with nx × ny × nz cells on [0, Lx] × [0, Ly] × [0, H]. With a terrain function
`zs(x, y)` the cells are cut (Cartesian cut cells): `Vf[i,j,k]` is the fluid volume fraction of a
cell, `Ax[i,j,k]`, `Ay[i,j,k]` the open fractions of its left x- and y-face, `Az[i,j,k]` of its
lower z-face (k = 1 … nz+1, walls at k = 1 and nz+1 closed). The fractions are computed with
nq-point Gauss quadrature in the horizontal directions and exactly in z (the fluid part of a
column of a cell is `clamp(z_top - max(zs, z_bot), 0, Δz)`). Small cut cells use the effective
volume `Veff = max(Vf, αmin)` in the divisions (mass and Θ are conserved with these weights);
fully solid cells (Vf = 0) are frozen. `box = (x0, x1, y0, y1, h)`: a rectangular obstacle
[x0, x1] × [y0, y1] × [0, h] on the ground, with exact fractions (aligned with the grid faces:
only full and solid cells; shifted: cut cells).
"""
struct Grid3D
    nx::Int
    ny::Int
    nz::Int
    Lx::Float64
    Ly::Float64
    H::Float64
    dx::Float64
    dy::Float64
    dz::Float64
    xc::Vector{Float64}
    yc::Vector{Float64}
    zc::Vector{Float64}
    xf::Vector{Float64}           # x of the left x-faces (nx)
    yf::Vector{Float64}           # y of the left y-faces (ny)
    zf::Vector{Float64}           # z of the z-faces (nz+1)
    Vf::Array{Float64,3}          # fluid volume fractions (nx, ny, nz)
    invV::Array{Float64,3}        # 1/Veff, 0 in solid cells
    Ax::Array{Float64,3}          # open fraction of the left x-face (nx, ny, nz)
    Ay::Array{Float64,3}          # open fraction of the left y-face (nx, ny, nz)
    Az::Array{Float64,3}          # open fraction of the lower z-face (nx, ny, nz+1)
    full::BitArray{3}             # Vf == 1 (third-order stencils allowed)
    cut::Bool
    zs::Any
    αmin::Float64
end

ncell(g::Grid3D) = g.nx * g.ny * g.nz
nunknowns(g::Grid3D) = 4 * ncell(g) + g.nx * g.ny * (g.nz - 1)
@inline ip(g::Grid3D, i) = mod1(i, g.nx)
@inline jp(g::Grid3D, j) = mod1(j, g.ny)
@inline _c(g::Grid3D, i, j, k) = ip(g, i) + (jp(g, j) - 1) * g.nx + (k - 1) * g.nx * g.ny
@inline iρ(g::Grid3D, i, j, k) = _c(g, i, j, k)
@inline iU(g::Grid3D, i, j, k) = ncell(g) + _c(g, i, j, k)
@inline iV(g::Grid3D, i, j, k) = 2 * ncell(g) + _c(g, i, j, k)
@inline iW(g::Grid3D, i, j, k) = 3 * ncell(g) + ip(g, i) + (jp(g, j) - 1) * g.nx + (k - 2) * g.nx * g.ny
@inline iΘ(g::Grid3D, i, j, k) = 3 * ncell(g) + g.nx * g.ny * (g.nz - 1) + _c(g, i, j, k)
"""W at the z-face k of column (i, j); zero at the walls k = 1, nz+1."""
@inline Wface(g::Grid3D, q, i, j, k) = (k == 1 || k == g.nz + 1) ? 0.0 : q[iW(g, i, j, k)]

const _GX = (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526)
const _GW = (0.3478548451374538, 0.6521451548625462, 0.6521451548625462, 0.3478548451374538)

"""Fluid fraction of the z-interval [zb, zt] above the terrain height s."""
@inline _wet(s, zb, zt) = clamp((zt - max(s, zb)) / (zt - zb), 0.0, 1.0)

"""Length of the overlap of the intervals [a0, a1] and [b0, b1]."""
@inline _ov(a0, a1, b0, b1) = max(0.0, min(a1, b1) - max(a0, b0))

function Grid3D(nx, ny, nz, Lx, Ly, H; zs = nothing, box = nothing, αmin = 0.2)
    zs !== nothing && box !== nothing && error("Grid3D: give zs or box, not both")
    dx, dy, dz = Lx / nx, Ly / ny, H / nz
    xc = [(i - 0.5) * dx for i in 1:nx]; yc = [(j - 0.5) * dy for j in 1:ny]; zc = [(k - 0.5) * dz for k in 1:nz]
    xf = [(i - 1) * dx for i in 1:nx]; yf = [(j - 1) * dy for j in 1:ny]; zf = [(k - 1) * dz for k in 1:nz+1]
    Vf = ones(nx, ny, nz); Ax = ones(nx, ny, nz); Ay = ones(nx, ny, nz); Az = ones(nx, ny, nz + 1)
    if zs !== nothing
        # quadrature points of a cell footprint and of the faces
        for k in 1:nz, j in 1:ny, i in 1:nx
            zb, zt = zf[k], zf[k+1]
            Vf[i, j, k] = sum(_GW[a] * _GW[b] * _wet(zs(xc[i] + _GX[a] * dx / 2, yc[j] + _GX[b] * dy / 2), zb, zt)
                              for a in 1:4, b in 1:4) / 4
            Ax[i, j, k] = sum(_GW[b] * _wet(zs(xf[i], yc[j] + _GX[b] * dy / 2), zb, zt) for b in 1:4) / 2
            Ay[i, j, k] = sum(_GW[a] * _wet(zs(xc[i] + _GX[a] * dx / 2, yf[j]), zb, zt) for a in 1:4) / 2
        end
        for k in 1:nz+1, j in 1:ny, i in 1:nx
            Az[i, j, k] = sum(_GW[a] * _GW[b] * (zs(xc[i] + _GX[a] * dx / 2, yc[j] + _GX[b] * dy / 2) < zf[k])
                              for a in 1:4, b in 1:4) / 4
        end
    end
    if box !== nothing
        x0, x1, y0, y1, h = box
        for k in 1:nz, j in 1:ny, i in 1:nx
            ox = _ov(xf[i], xf[i] + dx, x0, x1) / dx; oy = _ov(yf[j], yf[j] + dy, y0, y1) / dy
            oz = _ov(zf[k], zf[k+1], 0.0, h) / dz
            Vf[i, j, k] = 1 - ox * oy * oz
            Ax[i, j, k] = 1 - (x0 <= xf[i] <= x1) * oy * oz      # the box surface itself is closed
            Ay[i, j, k] = 1 - (y0 <= yf[j] <= y1) * ox * oz
        end
        for k in 2:nz, j in 1:ny, i in 1:nx
            Az[i, j, k] = 1 - (zf[k] <= h) * _ov(xf[i], xf[i] + dx, x0, x1) / dx * _ov(yf[j], yf[j] + dy, y0, y1) / dy
        end
    end
    if zs !== nothing || box !== nothing
        # quadrature rounding: snap to 0 and 1
        snap(a) = a > 1 - 1e-12 ? 1.0 : (a < 1e-12 ? 0.0 : a)
        Vf .= snap.(Vf); Ax .= snap.(Ax); Ay .= snap.(Ay); Az .= snap.(Az)
        # a face is open only between two fluid cells
        for k in 1:nz, j in 1:ny, i in 1:nx
            (Vf[i, j, k] == 0 || Vf[mod1(i - 1, nx), j, k] == 0) && (Ax[i, j, k] = 0.0)
            (Vf[i, j, k] == 0 || Vf[i, mod1(j - 1, ny), k] == 0) && (Ay[i, j, k] = 0.0)
            k > 1 && (Vf[i, j, k] == 0 || Vf[i, j, k-1] == 0) && (Az[i, j, k] = 0.0)
        end
    end
    Az[:, :, 1] .= 0.0; Az[:, :, nz+1] .= 0.0                   # walls
    invV = [v > 0 ? 1 / max(v, αmin) : 0.0 for v in Vf]
    return Grid3D(nx, ny, nz, Lx, Ly, H, dx, dy, dz, xc, yc, zc, xf, yf, zf, Vf, invV, Ax, Ay, Az,
                  Vf .== 1, zs !== nothing || box !== nothing, zs === nothing ? box : zs, Float64(αmin))
end
