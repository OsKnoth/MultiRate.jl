"""
    Buildings

Cut-cell geometry of buildings from GIS data (CityGML LoD1 / LoD2) via the level set of ASAM (SRC/GRID):

 1. `read_citygml(files)`: the polygons of the buildings (exterior ring and holes, 3D vertices in the GIS coordinates,
    metres, e.g. ETRS89 / UTM with heights above sea level).
 2. `building_levelset(polys; origin, L)`: the level set function φ(x, y, z) of the buildings in a domain
    [0, Lx] × [0, Ly] (local coordinates, ground z = 0). Every top surface (LoD2 roof face, top face of an LoD1 solid;
    faces below another face such as ground faces are dropped) is the prism below it, decomposed into convex prisms
    (`convex_pieces`: ear clipping, Hertel–Mehlhorn); a convex prism is the intersection of half-spaces, its level
    set the largest signed distance to its planes, max_f n_f · (p - p_f); the buildings are the union,
    φ = min(min over prisms, z), courtyards are removed by max(φ, -φ_hole). The prisms are sorted into a binary tree
    of boxes (ASAM Tree_Mod). φ < 0 in the buildings, φ = 0 exactly on their polyhedral surface.
 3. `cut_geometry(φ; Δx, zf)`: the fractions of the simulation grid (horizontal resolution Δx = Δy, e.g. 4 m,
    vertical z-faces zf, e.g. stretched by `stretched_levels`): every cell divided into sub-cells, φ at their corners,
    the fluid volume of the six tetrahedra of the Freudenthal (Kuhn) decomposition of every sub-cell exact for the
    linear interpolant, the face fractions from the triangles of the same decomposition on the faces, so that volume
    and face fractions describe one closed piecewise planar surface. Exact wherever one plane is active in a
    tetrahedron; errors only in the tetrahedra at the edges of the buildings (second order in the sub-cell size).

Optional: `levelset(φ; δ, δz)` samples φ on a fine uniform auxiliary grid (a `LevelSet`, callable as well, saved and
loaded as HDF5 by `save_levelset`, `load_levelset`); `cut_geometry` accepts it in place of φ. `fv3d_grid(cg)` makes
the FV3D.Grid3D of a geometry (stretched levels included); `save_geometry` writes the fractions (extension MultiRateHDF5Ext,
`using HDF5`).
"""
module Buildings

using LinearAlgebra, Printf
using Base.Threads: @threads
import ..FV3D

export read_citygml, BuildingPolygon, building_levelset, BuildingLevelSet, convex_pieces, height_at, levelset, LevelSet,
       stretched_levels, cut_geometry, CutGeometry, remove_small_cells!, fv3d_grid, save_levelset, load_levelset, save_geometry

include("citygml.jl")
include("polygon.jl")
include("roofs.jl")
include("solid.jl")
include("levelset.jl")
include("fractions.jl")

"""Save a level set (`save_levelset(file, ls)`) or a cut geometry (`save_geometry(file, cg)`) as HDF5; load a level
set (`load_levelset(file)`). Defined by the extension MultiRateHDF5Ext (`using HDF5`)."""
function save_levelset end
function load_levelset end
function save_geometry end

end
