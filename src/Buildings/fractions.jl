# Cut-cell fractions of the simulation grid from the level set: sub-cells, Freudenthal tetrahedra, exact fluid
# fractions of the linear interpolant.

"""
    stretched_levels(H; dz0 = 4.0, z1 = 0.0, r = 1.08, dzmax = Inf) -> zf

z-faces 0 = zf[1] < … < zf[end] = H: uniform dz0 up to (at least) z1, above growing by the factor r per level up to
dzmax; the levels above z1 are scaled so that the last face is H.
"""
function stretched_levels(H::Real; dz0::Real = 4.0, z1::Real = 0.0, r::Real = 1.08, dzmax::Real = Inf)
    zf = [0.0]
    while zf[end] < z1 - 1e-9 && zf[end] + dz0 < H
        push!(zf, zf[end] + dz0)
    end
    zu = zf[end]
    dz = dz0
    while zf[end] < H - 1e-9
        dz = min(dz * r, dzmax)
        push!(zf, zf[end] + dz)
    end
    length(zf) > 2 && zf[end] - zf[end-1] < 0.5 * (zf[end-1] - zf[end-2]) && zf[end-1] > zu && deleteat!(zf, length(zf) - 1)
    s = (H - zu) / (zf[end] - zu)                  # the stretched part scaled to end at H
    for k in eachindex(zf)
        zf[k] > zu && (zf[k] = zu + (zf[k] - zu) * s)
    end
    zf[end] = H
    return zf
end

"""Fluid (φ > 0) fraction of a tetrahedron with the linear function of the vertex values (exact; φ = 0 counts as solid)."""
function tet_fraction(p1, p2, p3, p4)
    a, b, c, d = _sort4(p1, p2, p3, p4)                  # a ≥ b ≥ c ≥ d
    d > 0 && return 1.0
    a <= 0 && return 0.0
    if b <= 0                                            # one fluid vertex
        return a^3 / ((a - b) * (a - c) * (a - d))
    elseif c <= 0                                        # two
        return (a^2 * b^2 - a * b * (a + b) * (c + d) + c * d * (a^2 + a * b + b^2)) / ((a - c) * (a - d) * (b - c) * (b - d))
    else                                                 # three
        return 1.0 - (-d)^3 / ((a - d) * (b - d) * (c - d))
    end
end
@inline function _sort4(a, b, c, d)
    a < b && ((a, b) = (b, a)); c < d && ((c, d) = (d, c))
    a < c && ((a, c) = (c, a)); b < d && ((b, d) = (d, b))
    b < c && ((b, c) = (c, b))
    return a, b, c, d
end
"""Fluid fraction of a triangle with the linear function of the vertex values (exact)."""
function tri_fraction(p1, p2, p3)
    a, b, c = p1, p2, p3
    a < b && ((a, b) = (b, a)); b < c && ((b, c) = (c, b)); a < b && ((a, b) = (b, a))
    c > 0 && return 1.0
    a <= 0 && return 0.0
    b <= 0 && return a^2 / ((a - b) * (a - c))
    return 1.0 - c^2 / ((a - c) * (b - c))
end

"""Fluid fraction of a cube with the corner values f(ix, iy, iz), ix, iy, iz ∈ {0, 1}: mean of the six Freudenthal
tetrahedra (paths 000 → 111 along the axes); their faces split every face of the cube along its diagonal from the
lowest to the highest corner, as `square_fraction`."""
@inline function cube_fraction(f)
    v000, v111 = f(0, 0, 0), f(1, 1, 1)
    v100, v010, v001 = f(1, 0, 0), f(0, 1, 0), f(0, 0, 1)
    v110, v101, v011 = f(1, 1, 0), f(1, 0, 1), f(0, 1, 1)
    return (tet_fraction(v000, v100, v110, v111) + tet_fraction(v000, v100, v101, v111) +
            tet_fraction(v000, v010, v110, v111) + tet_fraction(v000, v010, v011, v111) +
            tet_fraction(v000, v001, v101, v111) + tet_fraction(v000, v001, v011, v111)) / 6
end
"""Fluid fraction of a square with the corner values f00, f10, f01, f11: two triangles split along 00–11."""
@inline square_fraction(f00, f10, f01, f11) = (tri_fraction(f00, f10, f11) + tri_fraction(f00, f01, f11)) / 2

"""
Cut-cell geometry of a simulation grid: nx × ny columns of width Δx, Δy, z-faces zf (nz + 1), fluid volume fractions
Vf (nx, ny, nz), open fractions of the left x-faces Ax, of the left y-faces Ay (nx, ny, nz) and of the lower z-faces Az
(nx, ny, nz + 1; walls at k = 1, nz + 1 closed), in the local coordinates of the level set (GIS origin `origin`).
"""
struct CutGeometry
    origin::NTuple{3,Float64}
    nx::Int; ny::Int; nz::Int
    Δx::Float64; Δy::Float64
    zf::Vector{Float64}
    Vf::Array{Float64,3}
    Ax::Array{Float64,3}
    Ay::Array{Float64,3}
    Az::Array{Float64,3}
end

"""
    cut_geometry(φ; Δx = 4.0, Δy = Δx, zf, nsub = (4, 4), δsub = 1.0)

The fractions of the grid with columns Δx × Δy (Lx, Ly of φ multiples of them) and the z-faces zf from the level set
φ (`BuildingLevelSet` or its cache `LevelSet`): cells into which no solid reaches (bounding boxes, the tree) are
fluid; the others are divided into nsub[1] × nsub[2] × ⌈Δz/δsub⌉ sub-cells, φ at their corners, the fluid fractions
of the Freudenthal tetrahedra of every sub-cell and of the triangles of the faces (exact for the linear interpolant
on the tetrahedra); fractions below 1e-12 set to 0, above 1 - 1e-12 to 1. The band of φ must exceed the diagonal of
the sub-cells.
"""
function cut_geometry(ls; Δx::Real = 4.0, Δy::Real = Δx, zf::AbstractVector, nsub = (4, 4), δsub::Real = 1.0)
    nx, ny = round(Int, ls.Lx / Δx), round(Int, ls.Ly / Δy)
    (abs(nx * Δx - ls.Lx) < 1e-6 && abs(ny * Δy - ls.Ly) < 1e-6) || error("cut_geometry: Lx, Ly not multiples of Δx, Δy")
    nz = length(zf) - 1
    sx, sy = nsub
    ls.band >= sqrt((Δx / sx)^2 + (Δy / sy)^2 + δsub^2) || error("cut_geometry: band of φ smaller than the sub-cell diagonal")
    sz = [max(1, ceil(Int, (zf[k+1] - zf[k]) / δsub - 1e-9)) for k in 1:nz]
    Vf = ones(nx, ny, nz); Ax = ones(nx, ny, nz); Ay = ones(nx, ny, nz); Az = ones(nx, ny, nz + 1)
    @threads for j in 1:ny
        Φ = Array{Float64}(undef, sx + 1, sy + 1, maximum(sz) + 1)
        for i in 1:nx, k in 1:nz
            zb, zt = zf[k], zf[k+1]
            x0, y0 = (i - 1) * Δx, (j - 1) * Δy
            solid_near(ls, x0, x0 + Δx, y0, y0 + Δy, zb, zt) || continue      # all fluid
            nzs = sz[k]
            for c in 0:nzs, b in 0:sy, a in 0:sx
                Φ[a+1, b+1, c+1] = ls(x0 + a * Δx / sx, y0 + b * Δy / sy, zb + c * (zt - zb) / nzs)
            end
            lo, hi = extrema(view(Φ, :, :, 1:nzs+1))
            if lo > 0                                          # all fluid: the fractions stay 1
                continue
            elseif hi <= 0                                     # all solid
                Vf[i, j, k] = 0.0; Ax[i, j, k] = 0.0; Ay[i, j, k] = 0.0; Az[i, j, k] = 0.0
                continue
            end
            v = 0.0
            for c in 1:nzs, b in 1:sy, a in 1:sx
                v += cube_fraction((p, q, r) -> Φ[a+p, b+q, c+r])
            end
            Vf[i, j, k] = v / (sx * sy * nzs)
            ax = 0.0
            for c in 1:nzs, b in 1:sy
                ax += square_fraction(Φ[1, b, c], Φ[1, b+1, c], Φ[1, b, c+1], Φ[1, b+1, c+1])
            end
            Ax[i, j, k] = ax / (sy * nzs)
            ay = 0.0
            for c in 1:nzs, a in 1:sx
                ay += square_fraction(Φ[a, 1, c], Φ[a+1, 1, c], Φ[a, 1, c+1], Φ[a+1, 1, c+1])
            end
            Ay[i, j, k] = ay / (sx * nzs)
            az = 0.0
            for b in 1:sy, a in 1:sx
                az += square_fraction(Φ[a, b, 1], Φ[a+1, b, 1], Φ[a, b+1, 1], Φ[a+1, b+1, 1])
            end
            Az[i, j, k] = az / (sx * sy)
        end
    end
    snap(a) = a > 1 - 1e-12 ? 1.0 : (a < 1e-12 ? 0.0 : a)
    Vf .= snap.(Vf); Ax .= snap.(Ax); Ay .= snap.(Ay); Az .= snap.(Az)
    Az[:, :, 1] .= 0.0; Az[:, :, nz+1] .= 0.0
    return CutGeometry(ls.origin, nx, ny, nz, Float64(Δx), Float64(Δy), collect(Float64, zf), Vf, Ax, Ay, Az)
end

"""
    fv3d_grid(cg::CutGeometry; kw...)

FV3D.Grid3D with the fractions and the (stretched) levels zf of cg (keywords: mask_width, limiter).
"""
fv3d_grid(cg::CutGeometry; kw...) =
    FV3D.Grid3D(cg.nx, cg.ny, cg.nz, cg.nx * cg.Δx, cg.ny * cg.Δy, cg.zf[end];
                fractions = (Vf = cg.Vf, Ax = cg.Ax, Ay = cg.Ay, Az = cg.Az), zf = cg.zf, kw...)
