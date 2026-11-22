# Halo exchange of the tiles (halo width HALO = 1). Two kinds of communication:
#   exchange!      cell halo: the halo cells get the values of the neighbouring tiles (periodic);
#   merge_faces!   tile-boundary faces of the third-order face values: a boundary face is computed by the tile on
#                  its upwind side (only there the stencil (two upwind cells, one downwind) is complete), the value is
#                  sent to the other tile ("one halo cell and two communications").
# `SerialDecomp`: one rank, periodic images. MPI tiles: `mpi_decomp(px, py; comm)` of the extension MultiRateMPIExt
# (`using MPI`), the same interface.

abstract type AbstractDecomp end

"""One rank: halo = periodic image (x first, then y over the whole x range, so the corners are filled)."""
struct SerialDecomp <: AbstractDecomp end
decomp_dims(::SerialDecomp) = (1, 1)
decomp_coords(::SerialDecomp) = (0, 0)
"""Sum / maximum / minimum over all ranks (serial: the value itself)."""
allreduce_sum(::SerialDecomp, x) = x
allreduce_max(::SerialDecomp, x) = x
allreduce_min(::SerialDecomp, x) = x

"""
    mpi_decomp(px, py; comm = MPI.COMM_WORLD)

px × py decomposition of the horizontal plane into MPI ranks (periodic Cartesian communicator). Defined by the
extension MultiRateMPIExt: load MPI first (`using MPI`).
"""
function mpi_decomp end

"""
    init_exchange!(dec, g)

Allocate the communication buffers of the decomposition for the tile grid g once (as CGDycore `InitExchangeData3D`):
per side one send and one receive buffer on the device of the grid, large enough for every exchange of the model
(`exchange_length`); all exchanges use views of them. Serial: nothing to do.
"""
init_exchange!(dec::SerialDecomp, g, nfields::Int = 12) = dec
"""Values per side of the largest message: the halo of `nfields` arrays of nz+1 levels, the state (NV = 8 variables,
9 with rain) and the four face velocities in one exchange (`ka_frozen_data!`)."""
exchange_length(g, nfields::Int = 12) = HALO * max(g.nxh, g.nyh) * (g.nz + 1) * nfields

"""4D view of a 3D array (one variable)."""
_as4(A::AbstractArray{T,4}) where {T} = A
_as4(A::AbstractArray{T,3}) where {T} = reshape(A, size(A)..., 1)

@kernel function _k_perx!(A, nxl, vars)
    I, J, K, n = @index(Global, NTuple)
    v = vars[n]
    @inbounds A[I, J, K, v] = A[I + nxl, J, K, v]
    @inbounds A[HALO + nxl + I, J, K, v] = A[HALO + I, J, K, v]
end
@kernel function _k_pery!(A, nyl, vars)
    I, J, K, n = @index(Global, NTuple)
    v = vars[n]
    @inbounds A[I, J, K, v] = A[I, J + nyl, K, v]
    @inbounds A[I, HALO + nyl + J, K, v] = A[I, HALO + J, K, v]
end

"""
    exchange!(dec, A, g, vars = all)

Fill the halo cells of the variables `vars` (indices into the last dimension of a 4D array A; a 3D array is one
variable).
"""
function exchange!(::SerialDecomp, A, g::KAGrid, vars::Tuple = Tuple(1:size(_as4(A), 4)))
    B = _as4(A)
    be = KA.get_backend(B)
    nk = size(B, 3)
    _k_perx!(be)(B, g.nxl, vars; ndrange = (HALO, g.nyh, nk, length(vars)))
    _k_pery!(be)(B, g.nyl, vars; ndrange = (g.nxh, HALO, nk, length(vars)))
    KA.synchronize(be)
    return A
end

"""
    h = start_halo!(dec, A, g, vars); finish_halo!(dec, h)

Non-blocking exchange of the halo of the variables `vars` of A between the side neighbours (no corners: the fast
loop needs the halo cells next to the faces only). Between start and finish the interior can be computed; the cells
that are sent (first and last interior column and row) must be final at the start. Serial: periodic copies at the
start, finish does nothing.
"""
start_halo!(dec::SerialDecomp, A, g::KAGrid, vars::Tuple) = (exchange!(dec, A, g, vars); nothing)
finish_halo!(::SerialDecomp, h) = nothing

"""Exchange of several arrays in one communication (MPI: one message per neighbour)."""
exchange!(dec::SerialDecomp, As::Tuple, g::KAGrid) = (foreach(A -> exchange!(dec, A, g), As); As)

# Face arrays are stored at the left (lower) face of each cell: index I is the face between the cells I-1 and I.
# The tile-boundary faces are I = HALO+1 (left) and I = HALO+nxl+1 (right; the left boundary face of the right
# neighbour). Serial: they are the same physical face; the upwind tile has the value.
@kernel function _k_mergex!(A, @Const(F), nxl)
    J, k, n = @index(Global, NTuple)
    @inbounds begin
        l, r = HALO + 1, HALO + nxl + 1
        vl, vr = A[l, J, k, n], A[r, J, k, n]
        up = !signbit(F[r, J, k])                 # F at both copies of the face is the same
        A[l, J, k, n] = up ? vr : vl
        A[r, J, k, n] = up ? vr : vl
    end
end
@kernel function _k_mergey!(A, @Const(F), nyl)
    I, k, n = @index(Global, NTuple)
    @inbounds begin
        l, r = HALO + 1, HALO + nyl + 1
        vl, vr = A[I, l, k, n], A[I, r, k, n]
        up = !signbit(F[I, r, k])
        A[I, l, k, n] = up ? vr : vl
        A[I, r, k, n] = up ? vr : vl
    end
end

"""
    merge_faces!(dec, Ax, Fx, Ay, Fy, g)

Tile-boundary faces of the face arrays Ax (x-faces, 4D: the last dimension enumerates the arrays) and Ay (y-faces):
the value of the tile upwind of the face (by the sign of the mass flux Fx, Fy) on both tiles.
"""
function merge_faces!(::SerialDecomp, Ax, Fx, Ay, Fy, g::KAGrid)
    be = KA.get_backend(Ax)
    nk = size(Ax, 3)
    _k_mergex!(be)(Ax, Fx, g.nxl; ndrange = (g.nyh, nk, size(Ax, 4)))
    _k_mergey!(be)(Ay, Fy, g.nyl; ndrange = (g.nxh, nk, size(Ay, 4)))
    KA.synchronize(be)
    return nothing
end
