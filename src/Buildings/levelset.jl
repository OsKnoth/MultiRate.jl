# Optional cache of the level set: φ sampled on a fine uniform auxiliary grid (trilinear interpolation).

"""
φ at the nodes (x, y, z) = (i δx, j δy, k δz), i = 0 … nx, j = 0 … ny, k = 0 … nz (`φ[i+1, j+1, k+1]`, Float32) of the
domain [0, Lx] × [0, Ly] × [0, nz δz]; above the top node φ = +band. `origin` (x0, y0, z0): the GIS coordinates of the
local origin. Callable: φ(x, y, z) by trilinear interpolation.
"""
struct LevelSet
    origin::NTuple{3,Float64}
    Lx::Float64; Ly::Float64
    δx::Float64; δy::Float64; δz::Float64
    nx::Int; ny::Int; nz::Int
    band::Float64
    φ::Array{Float32,3}
end

"""
    levelset(φ::BuildingLevelSet; δ = 1.0, δz = δ) -> LevelSet

φ at the nodes of the auxiliary grid with spacing δ (horizontal, adjusted so that Lx, Ly are multiples) and δz
(vertical) up to the highest roof plus the band (the band must exceed the diagonal of an auxiliary cell).
"""
function levelset(φ::BuildingLevelSet; δ::Real = 1.0, δz::Real = δ)
    nx, ny = max(1, round(Int, φ.Lx / δ)), max(1, round(Int, φ.Ly / δ))
    δx, δy = φ.Lx / nx, φ.Ly / ny
    φ.band >= sqrt(δx^2 + δy^2 + δz^2) || error("levelset: band $(φ.band) smaller than the diagonal of an auxiliary cell")
    nz = ceil(Int, (φ.hmax + φ.band) / δz) + 1
    A = Array{Float32}(undef, nx + 1, ny + 1, nz + 1)
    @threads for j in 0:ny
        for k in 0:nz, i in 0:nx
            A[i+1, j+1, k+1] = Float32(φ(i * δx, j * δy, k * δz))
        end
    end
    return LevelSet(φ.origin, φ.Lx, φ.Ly, δx, δy, δz, nx, ny, nz, φ.band, A)
end

function (ls::LevelSet)(x, y, z)
    z >= ls.nz * ls.δz && return ls.band
    sx = clamp(x / ls.δx, 0.0, Float64(ls.nx)); sy = clamp(y / ls.δy, 0.0, Float64(ls.ny)); sz = clamp(z / ls.δz, 0.0, Float64(ls.nz))
    i = min(floor(Int, sx), ls.nx - 1); j = min(floor(Int, sy), ls.ny - 1); k = min(floor(Int, sz), ls.nz - 1)
    a, b, c = sx - i, sy - j, sz - k
    φ = ls.φ
    @inbounds begin
        v00 = (1 - a) * φ[i+1, j+1, k+1] + a * φ[i+2, j+1, k+1]
        v10 = (1 - a) * φ[i+1, j+2, k+1] + a * φ[i+2, j+2, k+1]
        v01 = (1 - a) * φ[i+1, j+1, k+2] + a * φ[i+2, j+1, k+2]
        v11 = (1 - a) * φ[i+1, j+2, k+2] + a * φ[i+2, j+2, k+2]
    end
    return (1 - c) * ((1 - b) * v00 + b * v10) + c * ((1 - b) * v01 + b * v11)
end
"""Whether a solid may reach into the box (false: all fluid there)."""
solid_near(ls::LevelSet, x0, x1, y0, y1, z0, z1) = z0 < ls.nz * ls.δz
