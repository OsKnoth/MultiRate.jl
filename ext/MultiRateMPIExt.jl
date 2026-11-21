# MPI tiles of FV3DKA: px × py ranks in a periodic Cartesian communicator, halo exchange (one message per neighbour and
# direction, x first, then y over the whole x range so that the corners are filled) and the merge of the tile-boundary
# faces of the third-order face values (the upwind tile's value). Data are staged through host buffers (GPU-aware
# MPI can replace this later).
module MultiRateMPIExt

using MultiRate, MPI
using KernelAbstractions
const KA = KernelAbstractions
import MultiRate.FV3DKA: exchange!, merge_faces!, allreduce_sum, allreduce_max, allreduce_min, mpi_decomp,
                         decomp_dims, decomp_coords
using MultiRate.FV3DKA: AbstractDecomp, HALO, KAGrid, _as4

struct MPIDecomp <: AbstractDecomp
    comm::MPI.Comm
    px::Int; py::Int
    cx::Int; cy::Int
    west::Int; east::Int; south::Int; north::Int
end

function mpi_decomp(px::Int, py::Int; comm = MPI.COMM_WORLD)
    MPI.Initialized() || MPI.Init()
    MPI.Comm_size(comm) == px * py || error("mpi_decomp: $(MPI.Comm_size(comm)) ranks for $px × $py tiles")
    cart = MPI.Cart_create(comm, [px, py]; periodic = [true, true], reorder = false)
    cx, cy = MPI.Cart_coords(cart)
    west, east = MPI.Cart_shift(cart, 0, 1)
    south, north = MPI.Cart_shift(cart, 1, 1)
    return MPIDecomp(cart, px, py, cx, cy, west, east, south, north)
end
decomp_dims(d::MPIDecomp) = (d.px, d.py)
decomp_coords(d::MPIDecomp) = (d.cx, d.cy)
allreduce_sum(d::MPIDecomp, x) = MPI.Allreduce(x, +, d.comm)
allreduce_max(d::MPIDecomp, x) = MPI.Allreduce(x, max, d.comm)
allreduce_min(d::MPIDecomp, x) = MPI.Allreduce(x, min, d.comm)

"""Host copy of the slab A[r1, r2, :, v] for all (array, v) of `items`, concatenated."""
_pack(items, r1, r2) = reduce(vcat, (vec(Array(view(_as4(A), r1, r2, :, v))) for (A, v) in items))
"""Write the concatenated host data into the slabs A[r1, r2, :, v]."""
function _unpack!(items, r1, r2, data)
    o = 0
    for (A, v) in items
        V = view(_as4(A), r1, r2, :, v)
        n = length(V)
        tmp = similar(_as4(A), size(V)...)
        copyto!(tmp, reshape(data[o+1:o+n], size(V)))
        V .= tmp
        o += n
    end
    return nothing
end

"""One phase of the halo exchange: send the interior slabs to (lo, hi), receive the halo slabs."""
function _phase!(d, items, sendlo, sendhi, recvlo, recvhi, other, lo, hi)
    s_lo = _pack(items, sendlo, other); s_hi = _pack(items, sendhi, other)
    r_lo = similar(s_lo); r_hi = similar(s_hi)
    MPI.Sendrecv!(s_hi, r_lo, d.comm; dest = hi, sendtag = 1, source = lo, recvtag = 1)
    MPI.Sendrecv!(s_lo, r_hi, d.comm; dest = lo, sendtag = 2, source = hi, recvtag = 2)
    _unpack!(items, recvlo, other, r_lo); _unpack!(items, recvhi, other, r_hi)
    return nothing
end

function _exchange_items!(d::MPIDecomp, items, g::KAGrid)
    H, nxl, nyl = HALO, g.nxl, g.nyl
    # x: send the first / last H interior columns to the west / east, receive the west / east halo
    _phase!(d, items, H+1:2H, nxl+1:nxl+H, 1:H, H+nxl+1:2H+nxl, 1:g.nyh, d.west, d.east)
    # y: rows, over the whole x range (halo columns included: corners)
    s_lo = reduce(vcat, (vec(Array(view(_as4(A), :, H+1:2H, :, v))) for (A, v) in items))
    s_hi = reduce(vcat, (vec(Array(view(_as4(A), :, nyl+1:nyl+H, :, v))) for (A, v) in items))
    r_lo = similar(s_lo); r_hi = similar(s_hi)
    MPI.Sendrecv!(s_hi, r_lo, d.comm; dest = d.north, sendtag = 3, source = d.south, recvtag = 3)
    MPI.Sendrecv!(s_lo, r_hi, d.comm; dest = d.south, sendtag = 4, source = d.north, recvtag = 4)
    _unpack!(items, :, 1:H, r_lo)
    _unpack!(items, :, H+nyl+1:2H+nyl, r_hi)
    return nothing
end
# _unpack! with the ranges in (r1, r2) order: for y the first range is the full x range
_unpack!(items, ::Colon, r2::UnitRange, data) = _unpack!(items, 1:size(_as4(first(items)[1]), 1), r2, data)

function exchange!(d::MPIDecomp, A, g::KAGrid, vars::Tuple = Tuple(1:size(_as4(A), 4)))
    _exchange_items!(d, [(A, v) for v in vars], g)
    return A
end
function exchange!(d::MPIDecomp, As::Tuple, g::KAGrid)
    _exchange_items!(d, [(A, v) for A in As for v in 1:size(_as4(A), 4)], g)
    return As
end

"""Tile-boundary faces: the value of the upwind tile on both tiles (x-faces by the sign of Fx, then y-faces)."""
function merge_faces!(d::MPIDecomp, Ax, Fx, Ay, Fy, g::KAGrid)
    H, nxl, nyl = HALO, g.nxl, g.nyl
    for (A, F, dir) in ((Ax, Fx, 1), (Ay, Fy, 2))
        n = dir == 1 ? nxl : nyl
        l, r = H + 1, H + n + 1
        lo, hi = dir == 1 ? (d.west, d.east) : (d.south, d.north)
        sl(B, f) = dir == 1 ? view(B, f, :, :, :) : view(B, :, f, :, :)
        sF(f) = dir == 1 ? view(F, f, :, :) : view(F, :, f, :)
        Al, Ar = Array(sl(A, l)), Array(sl(A, r))                 # my left / right boundary face values
        Fl, Fr = Array(sF(l)), Array(sF(r))
        fromlo = similar(Ar); fromhi = similar(Al)
        MPI.Sendrecv!(Ar, fromlo, d.comm; dest = hi, sendtag = 5 + dir, source = lo, recvtag = 5 + dir)   # lo's right face = my left
        MPI.Sendrecv!(Al, fromhi, d.comm; dest = lo, sendtag = 7 + dir, source = hi, recvtag = 7 + dir)   # hi's left face = my right
        nl = copy(Al); nr = copy(Ar)
        for c in CartesianIndices(Fl)
            up = !signbit(Fl[c])                                   # flow from lo: lo's value
            for s in axes(Al, 3)
                up && (nl[c, s] = fromlo[c, s])
            end
        end
        for c in CartesianIndices(Fr)
            down = signbit(Fr[c])                                  # flow from hi: hi's value
            for s in axes(Ar, 3)
                down && (nr[c, s] = fromhi[c, s])
            end
        end
        tl = similar(A, size(nl)...); copyto!(tl, nl); sl(A, l) .= tl
        tr = similar(A, size(nr)...); copyto!(tr, nr); sl(A, r) .= tr
    end
    return nothing
end

end
