# The level set of the buildings as in ASAM (SRC/GRID: Haus_Mod `Dist`, Tree_Mod, Function_Mod `DistanceHaus`):
# every top surface of a building (roof face of LoD2, top face of an LoD1 solid) over its polygon in (x, y) is a prism
# down to the ground, decomposed into convex prisms; a convex prism is the intersection of half-spaces with outward
# unit normals n_f, its level set the largest signed plane distance max_f (n_f · p - d_f); the buildings are the
# union, φ = min over the prisms, and with the ground min(φ, z). Holes of a top surface (courtyards) are removed:
# max(φ, -φ_hole) with the vertical prisms of the hole. The prisms are sorted into a binary tree of boxes (x and y
# halved alternately); a prism lies in the smallest box that contains its bounding box, a point visits the boxes from
# its leaf up to the root.
#
# φ is negative in the buildings, zero exactly on the polyhedral surface, positive in the air; it is the signed
# distance inside a convex prism and in front of its faces, and a lower bound of the distance at its edges; values
# are capped at `band` (prisms farther than band are skipped).

"""Convex prism: the half-spaces a x + b y + c z ≤ d (unit normals (a, b, c)); φ = max (a x + b y + c z - d)."""
struct Convex
    planes::Vector{NTuple{4,Float64}}
end
@inline function _phi(C::Convex, x, y, z)
    v = -Inf
    @inbounds for (a, b, c, d) in C.planes
        v = max(v, a * x + b * y + c * z - d)
    end
    return v
end

"""The prism below a top surface z = a + b x + c y: union of the convex parts minus the vertical prisms of the holes;
bounding box (xmin, xmax, ymin, ymax, zmin, zmax), building number."""
struct Solid
    parts::Vector{Convex}
    holes::Vector{Convex}
    bb::NTuple{6,Float64}
    building::Int
end
@inline function _phi(S::Solid, x, y, z)
    v = Inf
    for C in S.parts
        v = min(v, _phi(C, x, y, z))
    end
    if !isempty(S.holes)
        h = Inf
        for C in S.holes
            h = min(h, _phi(C, x, y, z))
        end
        v = max(v, -h)
    end
    return v
end

"""Planes of the vertical walls of a convex counter-clockwise polygon (outward normals)."""
function _walls(q::Vector{V2})
    pl = NTuple{4,Float64}[]
    n = length(q)
    for i in 1:n
        P, Q = q[i], q[mod1(i + 1, n)]
        ex, ey = Q[1] - P[1], Q[2] - P[2]
        le = hypot(ex, ey)
        nx, ny = ey / le, -ex / le
        push!(pl, (nx, ny, 0.0, nx * P[1] + ny * P[2]))
    end
    return pl
end

"""Binary tree of boxes (ASAM Tree_Mod): node boxes, children (0: leaf), parents (0: root), lists of solids."""
struct BoxTree
    box::Vector{NTuple{4,Float64}}
    child::Vector{NTuple{2,Int}}
    parent::Vector{Int}
    lists::Vector{Vector{Int32}}
end
function BoxTree(box::NTuple{4,Float64}, depth::Int)
    T = BoxTree([box], [(0, 0)], [0], [Int32[]])
    function split!(n, d, dir)
        d == 0 && return
        x0, x1, y0, y1 = T.box[n]
        b1, b2 = dir == 1 ? ((x0, (x0 + x1) / 2, y0, y1), ((x0 + x1) / 2, x1, y0, y1)) :
                            ((x0, x1, y0, (y0 + y1) / 2), (x0, x1, (y0 + y1) / 2, y1))
        for b in (b1, b2)
            push!(T.box, b); push!(T.child, (0, 0)); push!(T.parent, n); push!(T.lists, Int32[])
        end
        c = length(T.box)
        T.child[n] = (c - 1, c)
        split!(c - 1, d - 1, 3 - dir); split!(c, d - 1, 3 - dir)
    end
    split!(1, depth, 1)
    return T
end
@inline _contains(b, x0, x1, y0, y1) = b[1] <= x0 && x1 <= b[2] && b[3] <= y0 && y1 <= b[4]
"""Insert solid n with the (padded) bounding box into the smallest box containing it (ASAM InsertHaus)."""
function _insert!(T::BoxTree, n, x0, x1, y0, y1)
    node = 1
    _contains(T.box[1], x0, x1, y0, y1) || return push!(T.lists[1], n)
    while true
        c1, c2 = T.child[node]
        if c1 != 0 && _contains(T.box[c1], x0, x1, y0, y1)
            node = c1
        elseif c2 != 0 && _contains(T.box[c2], x0, x1, y0, y1)
            node = c2
        else
            return push!(T.lists[node], n)
        end
    end
end
"""The leaf containing (x, y) (ASAM FindBoxPoint)."""
function _leaf(T::BoxTree, x, y)
    node = 1
    while true
        c1, c2 = T.child[node]
        c1 == 0 && return node
        b = T.box[c1]
        node = (b[1] <= x <= b[2] && b[3] <= y <= b[4]) ? c1 : c2
    end
end

"""
The level set of the buildings: the solids, their tree, the cap `band`, the domain [0, Lx] × [0, Ly], the GIS origin
(x0, y0, z0), the highest roof; `ground`: min(φ, z) (the ground z = 0 as solid below). Callable: φ(x, y, z).
"""
struct BuildingLevelSet
    solids::Vector{Solid}
    tree::BoxTree
    band::Float64
    Lx::Float64; Ly::Float64
    origin::V3
    hmax::Float64
    ground::Bool
    roofs::Vector{Roof}
    rbins::Bins
end
function (φ::BuildingLevelSet)(x, y, z)
    v = φ.band
    B = φ.band
    T = φ.tree
    node = _leaf(T, x, y)
    @inbounds while node != 0
        for n in T.lists[node]
            S = φ.solids[n]
            b = S.bb
            (b[1] - B <= x <= b[2] + B && b[3] - B <= y <= b[4] + B && z <= b[6] + B) || continue
            v = min(v, _phi(S, x, y, z))
        end
        node = T.parent[node]
    end
    φ.ground && (v = min(v, z))
    return v
end
"""h(x, y): the height of the buildings (local coordinates; 0 outside)."""
height_at(φ::BuildingLevelSet, x, y) = _height(φ.roofs, φ.rbins, x, y)

"""Whether a solid may reach into the box [x0, x1] × [y0, y1] × [z0, z1] (bounding boxes; false: φ > 0 there except
on the ground)."""
function solid_near(φ::BuildingLevelSet, x0, x1, y0, y1, z0, z1)
    z0 > φ.hmax && return false
    T = φ.tree
    stack = [1]
    while !isempty(stack)
        n = pop!(stack)
        for m in T.lists[n]
            s = φ.solids[m].bb
            (s[1] <= x1 && x0 <= s[2] && s[3] <= y1 && y0 <= s[4] && z0 <= s[6]) && return true
        end
        c1, c2 = T.child[n]
        c1 == 0 && continue
        for c in (c1, c2)
            cb = T.box[c]
            # a solid of a child box lies inside it (padded by band at insertion)
            (cb[1] <= x1 && x0 <= cb[2] && cb[3] <= y1 && y0 <= cb[4]) && push!(stack, c)
        end
    end
    return false
end

"""
    building_levelset(polys; origin = :auto, L = :auto, margin = 20.0, z0 = :auto, band = 4.0, ground = true,
                      leaf = 32.0, vtol = 1e-3, tol = 0.05)

The level set of the buildings (ASAM) in the domain [0, Lx] × [0, Ly] with the corner `origin = (x0, y0)` (GIS
coordinates) and `L = (Lx, Ly)`; `:auto`: the bounding box of the polygons with `margin`, rounded up to multiples of
4 m. Heights relative to `z0` (`:auto`: the lowest vertex). Top surfaces: the polygons with |n_z| ≥ vtol |n| that do
not lie more than `tol` below another one at an interior point (ground faces and intermediate faces dropped); each
one becomes the prism below it, decomposed into convex prisms (`convex_pieces`), holes removed. Tree leaves of about
`leaf` metres. `band`: cap of φ (must exceed the diagonal of the sub-cells of `cut_geometry` or the auxiliary cells
of `levelset`).
"""
function building_levelset(polys::Vector{BuildingPolygon}; origin = :auto, L = :auto, margin::Real = 20.0, z0 = :auto,
                           band::Real = 4.0, ground::Bool = true, leaf::Real = 32.0, vtol::Real = 1e-3, tol::Real = 0.05)
    isempty(polys) && error("building_levelset: no polygons")
    allp = [p for P in polys for r in P.rings for p in r]
    xmin, xmax = extrema(p[1] for p in allp); ymin, ymax = extrema(p[2] for p in allp)
    x0, y0 = origin === :auto ? (xmin - margin, ymin - margin) : Float64.(origin)
    Lx, Ly = L === :auto ? (4 * ceil((xmax - xmin + 2margin) / 4), 4 * ceil((ymax - ymin + 2margin) / 4)) : Float64.(L)
    zr = z0 === :auto ? minimum(p[3] for p in allp) : Float64(z0)
    loc(p) = (p[1] - x0, p[2] - y0, p[3] - zr)
    # top-surface candidates and the filter
    cand = Roof[]
    for P in polys
        r3 = [loc.(r) for r in P.rings]
        nn, c = _newell(r3[1])
        nl = sqrt(nn[1]^2 + nn[2]^2 + nn[3]^2)
        (nl > 0 && abs(nn[3]) >= vtol * nl) || continue
        b, cc = -nn[1] / nn[3], -nn[2] / nn[3]
        r2 = [[(p[1], p[2]) for p in r] for r in r3]
        ps = r2[1]
        bb = (minimum(p[1] for p in ps), maximum(p[1] for p in ps), minimum(p[2] for p in ps), maximum(p[2] for p in ps))
        (bb[2] < 0 || bb[1] > Lx || bb[4] < 0 || bb[3] > Ly) && continue
        push!(cand, Roof(c[3] - b * c[1] - cc * c[2], b, cc, r2, bb, P.building))
    end
    cb = _bins(Lx, Ly, 16.0, [R.bb for R in cand])
    keep = trues(length(cand))
    for (n, R) in enumerate(cand)
        p = _interior_point(R)
        p === nothing && continue
        _height(cand, cb, p[1], p[2]; skip = n, h0 = -Inf) > roofz(R, p...) + tol && (keep[n] = false)
    end
    roofs = cand[keep]
    # solids: convex prisms below the top surfaces
    solids = Solid[]
    nfail = 0
    hmax = 0.0
    for R in roofs
        s = sqrt(1 + R.b^2 + R.c^2)
        roofplane = (-R.b / s, -R.c / s, 1 / s, R.a / s)
        outer = convex_pieces(R.rings[1])
        outer === nothing && (nfail += 1; continue)
        parts = [Convex(push!(_walls(q), roofplane)) for q in outer]
        holes = Convex[]
        for hr in R.rings[2:end]
            hp = convex_pieces(hr)
            hp === nothing && (nfail += 1; continue)
            append!(holes, Convex(_walls(q)) for q in hp)
        end
        zt = maximum(roofz(R, p...) for p in R.rings[1])
        hmax = max(hmax, zt)
        push!(solids, Solid(parts, holes, (R.bb..., 0.0, zt), R.building))
    end
    nfail > 0 && @warn "building_levelset: $nfail polygons could not be decomposed (not simple) and were skipped"
    # tree over the domain, padded by band
    bx = (min(0.0, minimum((S.bb[1] for S in solids); init = 0.0)) - band, max(Lx, maximum((S.bb[2] for S in solids); init = Lx)) + band,
          min(0.0, minimum((S.bb[3] for S in solids); init = 0.0)) - band, max(Ly, maximum((S.bb[4] for S in solids); init = Ly)) + band)
    depth = max(0, 2 * ceil(Int, log2(max(bx[2] - bx[1], bx[4] - bx[3]) / leaf)))
    T = BoxTree(bx, depth)
    for (n, S) in enumerate(solids)
        _insert!(T, n, S.bb[1] - band, S.bb[2] + band, S.bb[3] - band, S.bb[4] + band)
    end
    any(R -> R.bb[1] < 0 || R.bb[2] > Lx || R.bb[3] < 0 || R.bb[4] > Ly, roofs) &&
        @warn "building_levelset: buildings cross the domain boundary (periodic domain)"
    return BuildingLevelSet(solids, T, Float64(band), Lx, Ly, (x0, y0, zr), hmax, ground, roofs, _bins(Lx, Ly, 16.0, [R.bb for R in roofs]))
end
