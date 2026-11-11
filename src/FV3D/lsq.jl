# Least-squares quadratic reconstruction near cut cells (recon = :lsq)
#
# Berger & Giuliani (2021, JCP 109820, sect. 3.3, eqs. 28–30): in a cell C of the mask (cut cells
# and their neighbours) the field is the quadratic
#     φ(x) = φ̄_C + Σ_l a_l (p_l(x) - p̄_l,C),   p = (x, y, z, x², y², z², xy, xz, yz),
# conservative in C (p̄_l,C the mean of p_l over the fluid part of C), fitted in the least-squares
# sense to the means of the fluid neighbours n (3 × 3 × 3; cut cells: 5 × 5 in x, y and k-1 … k+2),
#     Σ_l a_l (p̄_l,n - p̄_l,C) = φ̄_n - φ̄_C,
# with the means over the true fluid parts (centroids and second moments of the cut cells). The
# face value of the upwind cell C is the mean of φ over the open part of the face:
#     φ_f = φ̄_C + Σ_n w_n (φ̄_n - φ̄_C),   w = (p̂_f - p̄_C)ᵀ A⁺,
# A⁺ the pseudo-inverse (minimum-norm, so degenerate directions such as ny = 1 drop out). On a
# uniform 1D grid this is the κ = 1/3 reconstruction (the quadratic through three cell means).
# Coordinates are scaled by (Δx, Δy, Δz) and relative to the centre of C.

using LinearAlgebra: pinv

const _NB = 9     # p = (x, y, z, xx, yy, zz, xy, xz, yz)

"""Lowest fluid height of the column at (x, y): terrain z_s or the box height inside its footprint."""
_zlow(zs, box, x, y) = zs !== nothing ? Float64(zs(x, y)) :
    (box !== nothing && box[1] <= x <= box[2] && box[3] <= y <= box[4] ? Float64(box[5]) : -Inf)

"""∫ ζ^p dζ over [a, b] (scaled vertical coordinate)."""
@inline _zint(p, a, b) = (b^(p + 1) - a^(p + 1)) / (p + 1)

"""Means of (1, p) over the fluid part of cell (i, j, k), local scaled coordinates (zero vector if solid)."""
function _cell_moments(g::Grid3D, zs, box, i, j, k)
    m = zeros(_NB + 1)
    for a in 1:4, b in 1:4
        ξ, υ = _GX[a] / 2, _GX[b] / 2
        w = _GW[a] * _GW[b] / 4
        zl = _zlow(zs, box, g.xc[i] + ξ * g.dx, g.yc[j] + υ * g.dy)
        lo = clamp((zl - g.zc[k]) / g.dz, -0.5, 0.5); hi = 0.5
        hi > lo || continue
        I0, I1, I2 = _zint(0, lo, hi), _zint(1, lo, hi), _zint(2, lo, hi)
        m[1] += w * I0
        m[2] += w * ξ * I0; m[3] += w * υ * I0; m[4] += w * I1
        m[5] += w * ξ^2 * I0; m[6] += w * υ^2 * I0; m[7] += w * I2
        m[8] += w * ξ * υ * I0; m[9] += w * ξ * I1; m[10] += w * υ * I1
    end
    m[1] > 0 && (m[2:end] ./= m[1])
    return m
end

"""Means of p over the open part of a face of cell (i, j, k): d = 1, 2, 3 (x, y, z), s = -1 (lower), +1 (upper)."""
function _face_moments(g::Grid3D, zs, box, i, j, k, d, s)
    m = zeros(_NB + 1)
    for a in 1:4
        if d == 3
            for b in 1:4
                ξ, υ = _GX[a] / 2, _GX[b] / 2; ζ = s / 2
                zl = _zlow(zs, box, g.xc[i] + ξ * g.dx, g.yc[j] + υ * g.dy)
                zl < g.zc[k] + ζ * g.dz || continue
                w = _GW[a] * _GW[b] / 4
                m[1] += w
                m[2:end] .+= w .* (ξ, υ, ζ, ξ^2, υ^2, ζ^2, ξ * υ, ξ * ζ, υ * ζ)
            end
        else
            t = _GX[a] / 2; w = _GW[a] / 2
            ξ, υ = d == 1 ? (s / 2, t) : (t, s / 2)
            zl = _zlow(zs, box, g.xc[i] + ξ * g.dx, g.yc[j] + υ * g.dy)
            lo = clamp((zl - g.zc[k]) / g.dz, -0.5, 0.5); hi = 0.5
            hi > lo || continue
            I0, I1, I2 = _zint(0, lo, hi), _zint(1, lo, hi), _zint(2, lo, hi)
            m[1] += w * I0
            m[2] += w * ξ * I0; m[3] += w * υ * I0; m[4] += w * I1
            m[5] += w * ξ^2 * I0; m[6] += w * υ^2 * I0; m[7] += w * I2
            m[8] += w * ξ * υ * I0; m[9] += w * ξ * I1; m[10] += w * υ * I1
        end
    end
    m[1] > 0 && (m[2:end] ./= m[1])
    return m[2:end]
end

"""Means of p of a cell with local means m (around its own centre), seen from a centre shifted by (a, b, c) cells."""
function _shift(m, a, b, c)
    x, y, z = m[2], m[3], m[4]
    return [x + a, y + b, z + c,
            m[5] + 2a * x + a^2, m[6] + 2b * y + b^2, m[7] + 2c * z + c^2,
            m[8] + a * y + b * x + a * b, m[9] + a * z + c * x + a * c, m[10] + b * z + c * y + b * c]
end

"""
LSQ data of the mask cells (`lsqid[i, j, k]` > 0: index into the vector). Needs the terrain `zs` or
the `box` to integrate the moments.
"""
function lsq_reconstruction!(g::Grid3D, zs, box)
    nx, ny, nz = g.nx, g.ny, g.nz
    mom = Array{Vector{Float64}}(undef, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        mom[i, j, k] = _cell_moments(g, zs, box, i, j, k)
    end
    id = g.lsqid; cells = g.lsq
    fill!(id, 0); empty!(cells)
    for k in 1:nz, j in 1:ny, i in 1:nx
        (g.mask[i, j, k] && g.Vf[i, j, k] > 0) || continue
        cut = g.Vf[i, j, k] < 1
        r = cut ? 2 : 1
        nb = NTuple{3,Int}[]; rows = Vector{Float64}[]
        pC = mom[i, j, k][2:end]
        for c in (cut ? (-1:2) : (-1:1)), b in -r:r, a in -r:r
            (a, b, c) == (0, 0, 0) && continue
            kk = k + c
            1 <= kk <= nz || continue
            ii, jj = mod1(i + a, nx), mod1(j + b, ny)
            g.Vf[ii, jj, kk] > 0 || continue
            push!(nb, (ii, jj, kk)); push!(rows, _shift(mom[ii, jj, kk], a, b, c) .- pC)
        end
        A = permutedims(reduce(hcat, rows))
        P = pinv(A; rtol = 1e-10)
        w = ntuple(6) do f
            d, s = (f + 1) ÷ 2, isodd(f) ? -1 : 1
            vec(permutedims(_face_moments(g, zs, box, i, j, k, d, s) .- pC) * P)
        end
        push!(cells, LSQCell(nb, w)); id[i, j, k] = length(cells)
    end
    return g
end

"""Face value of the upwind cell (i, j, k) through its face f (1 … 6) by its LSQ polynomial."""
@inline function lsq_face(g, φ, i, j, k, f)
    L = g.lsq[g.lsqid[i, j, k]]
    φC = φ[i, j, k]
    v = φC
    w = L.w[f]
    @inbounds for n in eachindex(L.nb)
        a, b, c = L.nb[n]
        v += w[n] * (φ[a, b, c] - φC)
    end
    return v
end

# ------------------------------------------------------------------ recon = :lin
# Linear correction of the pseudo third-order reconstruction (advection.jl) near cut cells:
#     φ_f = ash(φ̄) + ∇φ_C · (x̂_f - ash(x̄)),
# ash the 1D reconstruction of the upwind cell C (recon3 with the geometry of the face), x̄ the
# fluid centroids of its stencil cells, x̂_f the centroid of the open part of the face, ∇φ_C the
# linear least-squares gradient over the fluid centroids of the neighbours (Berger & Giuliani,
# eq. 29). Exact for linear fields; zero correction where the centroids are the cell centres
# (full cells, full faces), so the scheme is unchanged away from the cut cells.

"""ash stencil of face f (1 … 6) of cell (i, j, k) as upwind cell: (offset of U, offset of D,
a_UC, a_CD, upstream exists) — as in face_x, face_y, face_z."""
function _ash_stencil(g::Grid3D, i, j, k, f)
    if f == 2      # x upper face, flow +x
        return (-1, 0, 0), (1, 0, 0), ax(g, i, j, k), ax(g, i + 1, j, k), ax(g, i, j, k) > 0
    elseif f == 1  # x lower face, flow -x
        return (1, 0, 0), (-1, 0, 0), ax(g, i + 1, j, k), ax(g, i, j, k), ax(g, i + 1, j, k) > 0
    elseif f == 4
        return (0, -1, 0), (0, 1, 0), ay(g, i, j, k), ay(g, i, j + 1, k), ay(g, i, j, k) > 0
    elseif f == 3
        return (0, 1, 0), (0, -1, 0), ay(g, i, j + 1, k), ay(g, i, j, k), ay(g, i, j + 1, k) > 0
    elseif f == 6  # z upper face (k+1), flow up
        return (0, 0, k > 1 ? -1 : 0), (0, 0, 1), azs(g, i, j, k), az(g, i, j, k + 1), az(g, i, j, k) > 0
    else           # z lower face (k), flow down
        return (0, 0, k < g.nz ? 1 : 0), (0, 0, -1), azs(g, i, j, k + 1), az(g, i, j, k), az(g, i, j, k + 1) > 0
    end
end

"""Correction weights of the mask cells (recon = :lin), stored as LSQCell: φ_f = ash + Σ w (φ_n - φ_C)."""
function lin_reconstruction!(g::Grid3D, zs, box)
    nx, ny, nz = g.nx, g.ny, g.nz
    mom = Array{Vector{Float64}}(undef, nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        mom[i, j, k] = _cell_moments(g, zs, box, i, j, k)
    end
    id = g.lsqid; cells = g.lsq
    fill!(id, 0); empty!(cells)
    for k in 1:nz, j in 1:ny, i in 1:nx
        (g.mask[i, j, k] && g.Vf[i, j, k] > 0) || continue
        cut = g.Vf[i, j, k] < 1
        r = cut ? 2 : 1
        xC = mom[i, j, k][2:4]
        nb = NTuple{3,Int}[]; rows = Vector{Float64}[]
        for c in (cut ? (-1:2) : (-1:1)), b in -r:r, a in -r:r
            (a, b, c) == (0, 0, 0) && continue
            kk = k + c
            1 <= kk <= nz || continue
            ii, jj = mod1(i + a, nx), mod1(j + b, ny)
            g.Vf[ii, jj, kk] > 0 || continue
            push!(nb, (ii, jj, kk)); push!(rows, _shift(mom[ii, jj, kk], a, b, c)[1:3] .- xC)
        end
        G = pinv(permutedims(reduce(hcat, rows)); rtol = 1e-10)       # 3 × nb: ∇φ = G (φ_n - φ_C)
        w = ntuple(6) do f
            oU, oD, aUC, aCD, up = _ash_stencil(g, i, j, k, f)
            aCD > 0 || return zeros(length(nb))
            cU = (mod1(i + oU[1], nx), mod1(j + oU[2], ny), k + oU[3])
            cD = (mod1(i + oD[1], nx), mod1(j + oD[2], ny), k + oD[3])
            1 <= cD[3] <= nz || return zeros(length(nb))
            xU = _shift(mom[cU...], oU...)[1:3]; xD = _shift(mom[cD...], oD...)[1:3]
            xash = [recon3(xU[l], xC[l], xD[l], vol(g, cU...), vol(g, i, j, k), vol(g, cD...), aUC, aCD, up) for l in 1:3]
            d, s = (f + 1) ÷ 2, isodd(f) ? -1 : 1
            xf = _face_moments(g, zs, box, i, j, k, d, s)[1:3]
            vec(permutedims(xf .- xash) * G)
        end
        push!(cells, LSQCell(nb, w)); id[i, j, k] = length(cells)
    end
    return g
end

"""Σ_n w_n (φ_n - φ_C) of the face f of cell (i, j, k): the :lin correction (or the :lsq face value - φ_C)."""
@inline function lsq_corr(g, φ, i, j, k, f)
    L = g.lsq[g.lsqid[i, j, k]]
    φC = φ[i, j, k]; v = 0.0; w = L.w[f]
    @inbounds for n in eachindex(L.nb)
        a, b, c = L.nb[n]
        v += w[n] * (φ[a, b, c] - φC)
    end
    return v
end
