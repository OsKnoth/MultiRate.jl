# Decomposition of a simple polygon (x, y) into convex polygons: ear clipping and Hertel–Mehlhorn (merge the
# triangles across a diagonal as long as the union stays convex; at most four times the minimal number of pieces).

const V2 = NTuple{2,Float64}
const V3 = NTuple{3,Float64}

"""Signed area (positive: counter-clockwise)."""
function _area(r::AbstractVector{V2})
    n = length(r); A = 0.0
    for i in 1:n
        a, b = r[i], r[mod1(i + 1, n)]
        A += a[1] * b[2] - b[1] * a[2]
    end
    return A / 2
end
"""(b - a) × (c - b): > 0 for a left turn at b."""
@inline _turn(a::V2, b::V2, c::V2) = (b[1] - a[1]) * (c[2] - b[2]) - (b[2] - a[2]) * (c[1] - b[1])

"""Ring without repeated vertices and without vertices closer than `tol` to the line of their neighbours."""
function _clean(r::AbstractVector{V2}; tol = 1e-6)
    out = collect(r)
    changed = true
    while changed && length(out) >= 3
        changed = false
        n = length(out)
        for i in 1:n
            a, b, c = out[mod1(i - 1, n)], out[i], out[mod1(i + 1, n)]
            lac = hypot(c[1] - a[1], c[2] - a[2])
            if hypot(b[1] - a[1], b[2] - a[2]) < tol || abs(_turn(a, b, c)) <= tol * lac
                deleteat!(out, i); changed = true
                break
            end
        end
    end
    return out
end

"""p in the closed triangle abc (counter-clockwise)."""
@inline _intri(p::V2, a::V2, b::V2, c::V2) = _turn(a, b, p) >= 0 && _turn(b, c, p) >= 0 && _turn(c, a, p) >= 0

"""Ear clipping of a simple counter-clockwise polygon: triangles as vertex index triples (nothing if it fails)."""
function _triangulate(r::AbstractVector{V2})
    idx = collect(1:length(r))
    tris = NTuple{3,Int}[]
    while length(idx) > 3
        m = length(idx)
        found = false
        for k in 1:m
            ia, ib, ic = idx[mod1(k - 1, m)], idx[k], idx[mod1(k + 1, m)]
            a, b, c = r[ia], r[ib], r[ic]
            _turn(a, b, c) > 0 || continue                       # reflex or flat vertex: no ear
            ear = true
            for j in idx
                (j == ia || j == ib || j == ic) && continue
                r[j] == a || r[j] == b || r[j] == c || !_intri(r[j], a, b, c) || (ear = false; break)
            end
            ear || continue
            push!(tris, (ia, ib, ic)); deleteat!(idx, k)
            found = true
            break
        end
        found || return nothing
    end
    push!(tris, (idx[1], idx[2], idx[3]))
    return tris
end

"""The union of the convex polygons A and B (index lists, counter-clockwise) across a common edge, if convex."""
function _merge(A::Vector{Int}, B::Vector{Int}, r)
    na, nb = length(A), length(B)
    for i in 1:na
        u, v = A[i], A[mod1(i + 1, na)]
        for j in 1:nb
            (B[j] == v && B[mod1(j + 1, nb)] == u) || continue
            M = [A[mod1(i + 1 + s, na)] for s in 0:na-1]          # v … u
            append!(M, B[mod1(j + 1 + s, nb)] for s in 1:nb-2)     # after u … before v
            n = length(M)
            all(_turn(r[M[mod1(t - 1, n)]], r[M[t]], r[M[mod1(t + 1, n)]]) >= -1e-12 for t in 1:n) && return M
            return nothing
        end
    end
    return nothing
end

"""
    convex_pieces(ring) -> Vector{Vector{V2}} or nothing

Convex counter-clockwise polygons whose union is the simple polygon `ring` (either orientation): ear clipping, then
Hertel–Mehlhorn merging. `nothing` if the polygon is degenerate or not simple.
"""
function convex_pieces(ring::AbstractVector{V2})
    r = _clean(ring)
    length(r) >= 3 || return nothing
    _area(r) < 0 && (r = reverse(r))
    tris = _triangulate(r)
    tris === nothing && return nothing
    P = [collect(t) for t in tris]
    while true
        hit = false
        for p in 1:length(P), q in p+1:length(P)
            M = _merge(P[p], P[q], r)
            M === nothing && continue
            P[p] = M; deleteat!(P, q); hit = true
            break
        end
        hit || break
    end
    return [[r[i] for i in Q] for Q in P]
end
