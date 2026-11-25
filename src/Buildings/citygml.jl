# CityGML reader (CityGML 1.0 / 2.0 / 3.0, GML 3 `posList` / `pos`, GML 2 `coordinates`): the polygons of the
# Building elements (with their BuildingParts). No XML library: the files are scanned with regular expressions for
# the elements that carry geometry, which also works for large files (only the text of one Building is processed at
# a time). Namespace prefixes are arbitrary.

"""A planar polygon of building `building`: `rings[1]` the exterior ring, further rings holes; vertices (x, y, z)."""
struct BuildingPolygon
    building::Int
    rings::Vector{Vector{NTuple{3,Float64}}}
end

const _RX_BOPEN = r"<(?:[\w.-]+:)?Building[\s>]"
const _RX_BCLOSE = r"</(?:[\w.-]+:)?Building>"
const _RX_LOD = r"<(?:[\w.-]+:)?lod([0-4])(Solid|MultiSurface|CompositeSurface|Geometry|MultiSolid)[\s>].*?</(?:[\w.-]+:)?lod\1\2>"s
const _RX_POLYGON = r"<(?:[\w.-]+:)?Polygon[\s>].*?</(?:[\w.-]+:)?Polygon>"s
const _RX_RING = r"<(?:[\w.-]+:)?(exterior|interior|outerBoundaryIs|innerBoundaryIs)[\s>].*?</(?:[\w.-]+:)?\1>"s
const _RX_POSLIST = r"<(?:[\w.-]+:)?posList([^>]*)>(.*?)</(?:[\w.-]+:)?posList>"s
const _RX_POS = r"<(?:[\w.-]+:)?pos(?:\s[^>]*)?>(.*?)</(?:[\w.-]+:)?pos>"s
const _RX_COORDS = r"<(?:[\w.-]+:)?coordinates[^>]*>(.*?)</(?:[\w.-]+:)?coordinates>"s
const _RX_SRSDIM = r"srsDimension\s*=\s*\"(\d)\""

"""The vertices of one ring (closing vertex removed)."""
function _ring(txt::AbstractString)
    vals = Float64[]
    dim = 3
    m = match(_RX_POSLIST, txt)
    if m !== nothing
        d = match(_RX_SRSDIM, m.captures[1])
        d !== nothing && (dim = parse(Int, d.captures[1]))
        append!(vals, parse.(Float64, split(strip(m.captures[2]))))
    else
        ps = collect(eachmatch(_RX_POS, txt))
        if !isempty(ps)
            for p in ps
                v = parse.(Float64, split(strip(p.captures[1])))
                dim = length(v)
                append!(vals, v)
            end
        else
            c = match(_RX_COORDS, txt)
            c === nothing && return NTuple{3,Float64}[]
            for t in split(strip(c.captures[1]))
                v = parse.(Float64, split(t, ','))
                dim = length(v)
                append!(vals, v)
            end
        end
    end
    dim == 3 || error("CityGML: vertices with $dim coordinates (3 needed: x y z)")
    n = length(vals) ÷ 3
    r = [(vals[3i-2], vals[3i-1], vals[3i]) for i in 1:n]
    length(r) > 1 && r[end] == r[1] && pop!(r)
    return r
end

"""The polygons of a text (a Building or one of its LoD geometries)."""
function _polygons!(out, txt::AbstractString, b::Int)
    for pm in eachmatch(_RX_POLYGON, txt)
        rings = Vector{NTuple{3,Float64}}[]
        ext = nothing
        for rm in eachmatch(_RX_RING, pm.match)
            r = _ring(rm.match)
            length(r) >= 3 || continue
            if rm.captures[1] in ("exterior", "outerBoundaryIs")
                ext = r
            else
                push!(rings, r)
            end
        end
        ext === nothing && continue
        push!(out, BuildingPolygon(b, pushfirst!(rings, ext)))
    end
    return out
end

"""
    read_citygml(files; lod = :highest) -> Vector{BuildingPolygon}

The polygons of all buildings of the CityGML files (a file name or a vector of names). `lod = :highest`: per building
the geometry of its highest level of detail (LoD2 roofs if present, otherwise the LoD1 solid); `lod = 1` or `2`: only
that level (buildings without it are skipped). Buildings without lodN elements: all their polygons.
"""
function read_citygml(files; lod = :highest)
    files isa AbstractString && (files = [files])
    out = BuildingPolygon[]
    nb = 0
    for f in files
        txt = read(f, String)
        pos = 1
        while true
            o = findnext(_RX_BOPEN, txt, pos)
            o === nothing && break
            c = findnext(_RX_BCLOSE, txt, last(o))
            c === nothing && error("CityGML $f: Building without end tag")
            btxt = SubString(txt, first(o), last(c))
            pos = last(c) + 1
            nb += 1
            blocks = [(parse(Int, m.captures[1]), m.match) for m in eachmatch(_RX_LOD, btxt)]
            if isempty(blocks)
                lod === :highest && _polygons!(out, btxt, nb)
                continue
            end
            want = lod === :highest ? maximum(first, blocks) : Int(lod)
            for (l, t) in blocks
                l == want && _polygons!(out, t, nb)
            end
        end
    end
    return out
end
