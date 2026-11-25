# Top surfaces of the buildings (planes over polygons in x, y), the height field h(x, y), uniform bins.

"""A top surface: z = a + b x + c y over the polygon `rings` (x, y), bounding box (xmin, xmax, ymin, ymax)."""
struct Roof
    a::Float64; b::Float64; c::Float64
    rings::Vector{Vector{V2}}
    bb::NTuple{4,Float64}
    building::Int
end
@inline roofz(R::Roof, x, y) = R.a + R.b * x + R.c * y

"""Newell normal (length 2 × area) and the vertex mean of a ring."""
function _newell(r::Vector{V3})
    nx = ny = nz = 0.0
    n = length(r)
    for i in 1:n
        a, b = r[i], r[mod1(i + 1, n)]
        nx += (a[2] - b[2]) * (a[3] + b[3]); ny += (a[3] - b[3]) * (a[1] + b[1]); nz += (a[1] - b[1]) * (a[2] + b[2])
    end
    return (nx, ny, nz), (sum(p[1] for p in r) / n, sum(p[2] for p in r) / n, sum(p[3] for p in r) / n)
end
"""Even–odd point-in-polygon test (all rings)."""
function _inside(rings::Vector{Vector{V2}}, s, t)
    c = false
    @inbounds for r in rings
        n = length(r)
        for i in 1:n
            a, b = r[i], r[mod1(i + 1, n)]
            if (a[2] > t) != (b[2] > t)
                x = a[1] + (t - a[2]) / (b[2] - a[2]) * (b[1] - a[1])
                x > s && (c = !c)
            end
        end
    end
    return c
end
"""Uniform bins over [0, Lx] × [0, Ly] with the objects whose bounding boxes overlap them."""
struct Bins
    s::Float64; nx::Int; ny::Int
    lists::Matrix{Vector{Int32}}
end
function _bins(Lx, Ly, s, bbs)
    nx, ny = max(1, ceil(Int, Lx / s)), max(1, ceil(Int, Ly / s))
    lists = [Int32[] for _ in 1:nx, _ in 1:ny]
    for (n, bb) in enumerate(bbs)
        (bb[2] < 0 || bb[1] > Lx || bb[4] < 0 || bb[3] > Ly) && continue
        for j in clamp(floor(Int, bb[3] / s) + 1, 1, ny):clamp(floor(Int, bb[4] / s) + 1, 1, ny),
            i in clamp(floor(Int, bb[1] / s) + 1, 1, nx):clamp(floor(Int, bb[2] / s) + 1, 1, nx)
            push!(lists[i, j], n)
        end
    end
    return Bins(s, nx, ny, lists)
end
@inline binlist(B::Bins, x, y) = B.lists[clamp(floor(Int, x / B.s) + 1, 1, B.nx), clamp(floor(Int, y / B.s) + 1, 1, B.ny)]
function _height(roofs, bins, x, y; skip = 0, h0 = 0.0)
    h = h0
    for n in binlist(bins, x, y)
        n == skip && continue
        R = roofs[n]
        (R.bb[1] <= x <= R.bb[2] && R.bb[3] <= y <= R.bb[4]) || continue
        _inside(R.rings, x, y) && (h = max(h, roofz(R, x, y)))
    end
    return h
end

"""A point inside the polygon (holes excluded): the area centroid if inside, otherwise a point just inside the
middle of an edge of the exterior ring."""
function _interior_point(R::Roof)
    r = R.rings[1]; n = length(r)
    A = cx = cy = 0.0
    for i in 1:n
        a, b = r[i], r[mod1(i + 1, n)]
        w = a[1] * b[2] - b[1] * a[2]
        A += w; cx += (a[1] + b[1]) * w; cy += (a[2] + b[2]) * w
    end
    A != 0 && (p = (cx / (3A), cy / (3A)); _inside(R.rings, p...)) && return p
    for i in 1:n, ε in (1e-2, 1e-1)
        a, b = r[i], r[mod1(i + 1, n)]
        le = hypot(b[1] - a[1], b[2] - a[2])
        le > 10ε || continue
        for s in (1, -1)
            p = ((a[1] + b[1]) / 2 + s * ε * (b[2] - a[2]) / le, (a[2] + b[2]) / 2 - s * ε * (b[1] - a[1]) / le)
            _inside(R.rings, p...) && return p
        end
    end
    return nothing
end

