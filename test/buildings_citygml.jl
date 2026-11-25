# Synthetic CityGML (2.0) for the tests of MultiRate.Buildings: an LoD1 block (lod1Solid, flat roof) and an LoD2 house
# with a gable roof (RoofSurface, WallSurface, GroundSurface in lod2MultiSurface), UTM-like coordinates, ground 100 m.

_pl(ps) = join((join(string.(p), " ") for p in vcat(ps, [ps[1]])), " ")
_poly(ps) = """<gml:Polygon><gml:exterior><gml:LinearRing><gml:posList srsDimension="3">$(_pl(ps))</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon>"""
_ring(tag, ps) = """<gml:$tag><gml:LinearRing><gml:posList srsDimension="3">$(_pl(ps))</gml:posList></gml:LinearRing></gml:$tag>"""
_polyh(ps, hs...) = "<gml:Polygon>" * _ring("exterior", ps) * join(_ring("interior", h) for h in hs) * "</gml:Polygon>"
_surf(kind, ps) = """<bldg:boundedBy><bldg:$kind><bldg:lod2MultiSurface><gml:MultiSurface><gml:surfaceMember>$(_poly(ps))</gml:surfaceMember></gml:MultiSurface></bldg:lod2MultiSurface></bldg:$kind></bldg:boundedBy>"""

"""Box [x0, x1] × [y0, y1] × [z0, z1] as the six faces."""
function _boxfaces(x0, x1, y0, y1, z0, z1)
    return [[(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)], [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
            [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)], [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)]]
end

"""CityGML text; X0, Y0: offset of the coordinates (GIS), G: ground height. Exact volumes: block 20.4 × 13.7 × 15.5,
house 12 × 10 × 6 + 12 × 10 × 3 / 2 = 900, courtyard block (20 × 20 - 8 × 8) × 10 = 3360."""
function synthetic_citygml(; X0 = 400000.0, Y0 = 5650000.0, G = 100.0)
    bx = _boxfaces(X0 + 30.3, X0 + 50.7, Y0 + 32.2, Y0 + 45.9, G, G + 15.5)
    lod1 = """<bldg:lod1Solid><gml:Solid><gml:exterior><gml:CompositeSurface>""" *
           join("<gml:surfaceMember>" * _poly(f) * "</gml:surfaceMember>" for f in bx) *
           """</gml:CompositeSurface></gml:exterior></gml:Solid></bldg:lod1Solid>"""
    xa, xb, ya, yb, ym = X0 + 70.5, X0 + 82.5, Y0 + 40.25, Y0 + 50.25, Y0 + 45.25
    e, r = G + 6.0, G + 9.0
    house = _surf("GroundSurface", [(xa, ya, G), (xa, yb, G), (xb, yb, G), (xb, ya, G)]) *
            _surf("RoofSurface", [(xa, ya, e), (xb, ya, e), (xb, ym, r), (xa, ym, r)]) *
            _surf("RoofSurface", [(xa, ym, r), (xb, ym, r), (xb, yb, e), (xa, yb, e)]) *
            _surf("WallSurface", [(xa, ya, G), (xb, ya, G), (xb, ya, e), (xa, ya, e)]) *
            _surf("WallSurface", [(xb, yb, G), (xa, yb, G), (xa, yb, e), (xb, yb, e)]) *
            _surf("WallSurface", [(xa, yb, G), (xa, ya, G), (xa, ya, e), (xa, ym, r), (xa, yb, e)]) *
            _surf("WallSurface", [(xb, ya, G), (xb, yb, G), (xb, yb, e), (xb, ym, r), (xb, ya, e)])
    sq(a, b, c, d, z) = [(a, c, z), (b, c, z), (b, d, z), (a, d, z)]
    qa, qb, qc, qd = X0 + 88.5, X0 + 108.5, Y0 + 10.5, Y0 + 30.5
    ha, hb, hc, hd = X0 + 94.5, X0 + 102.5, Y0 + 16.5, Y0 + 24.5
    court = """<bldg:lod1Solid><gml:Solid><gml:exterior><gml:CompositeSurface>""" *
            "<gml:surfaceMember>" * _polyh(sq(qa, qb, qc, qd, G + 10.0), reverse(sq(ha, hb, hc, hd, G + 10.0))) * "</gml:surfaceMember>" *
            "<gml:surfaceMember>" * _polyh(reverse(sq(qa, qb, qc, qd, G)), sq(ha, hb, hc, hd, G)) * "</gml:surfaceMember>" *
            """</gml:CompositeSurface></gml:exterior></gml:Solid></bldg:lod1Solid>"""
    return """<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel xmlns:core="http://www.opengis.net/citygml/2.0" xmlns:bldg="http://www.opengis.net/citygml/building/2.0" xmlns:gml="http://www.opengis.net/gml">
<core:cityObjectMember><bldg:Building gml:id="block">$lod1</bldg:Building></core:cityObjectMember>
<core:cityObjectMember><bldg:Building gml:id="house">$house</bldg:Building></core:cityObjectMember>
<core:cityObjectMember><bldg:Building gml:id="court">$court</bldg:Building></core:cityObjectMember>
</core:CityModel>
"""
end
