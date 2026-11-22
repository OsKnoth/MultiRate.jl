# MPI tiles of FV3DKA: px × py ranks in a periodic Cartesian communicator, halo exchange (x first, then y over the
# whole x range so that the corners are filled), the merge of the tile-boundary faces of the third-order face values
# (the upwind tile's value) and the non-blocking side exchange of the fast loop.
#
# Buffers as in CGDycore (ExchangeStruct, InitExchangeData3D): per side (west, east, south, north) one send and one
# receive buffer on the device of the arrays, allocated once by `init_exchange!` with the length of the largest
# message, and preallocated requests; every exchange packs into views of them. At most one exchange is in flight at a
# time, so all share the buffers. On a GPU, MPI must be GPU-aware (CUDA-aware OpenMPI/UCX): there is no staging
# through host memory.
module MultiRateMPIExt

using MultiRate, MPI
using KernelAbstractions
const KA = KernelAbstractions
import MultiRate.FV3DKA: exchange!, merge_faces!, allreduce_sum, allreduce_max, allreduce_min, mpi_decomp,
                         decomp_dims, decomp_coords, start_halo!, finish_halo!, init_exchange!
using MultiRate.FV3DKA: AbstractDecomp, HALO, KAGrid, _as4, exchange_length

const W_, E_, S_, N_ = 1, 2, 3, 4
const OPP = (E_, W_, N_, S_)

"""Send / receive buffers per side and the requests (receives 1 … 4, sends 5 … 8)."""
struct ExchangeBuffers{B}
    send::Vector{B}
    recv::Vector{B}
    reqs::MPI.UnsafeMultiRequest
end

struct MPIDecomp <: AbstractDecomp
    comm::MPI.Comm
    px::Int; py::Int
    cx::Int; cy::Int
    west::Int; east::Int; south::Int; north::Int
    buf::Base.RefValue{Any}
end
_nb(d::MPIDecomp) = (d.west, d.east, d.south, d.north)

function mpi_decomp(px::Int, py::Int; comm = MPI.COMM_WORLD)
    MPI.Initialized() || MPI.Init()
    MPI.Comm_size(comm) == px * py || error("mpi_decomp: $(MPI.Comm_size(comm)) ranks for $px × $py tiles")
    cart = MPI.Cart_create(comm, [px, py]; periodic = [true, true], reorder = false)
    cx, cy = MPI.Cart_coords(cart)
    west, east = MPI.Cart_shift(cart, 0, 1)
    south, north = MPI.Cart_shift(cart, 1, 1)
    return MPIDecomp(cart, px, py, cx, cy, west, east, south, north, Ref{Any}(nothing))
end
decomp_dims(d::MPIDecomp) = (d.px, d.py)
decomp_coords(d::MPIDecomp) = (d.cx, d.cy)
allreduce_sum(d::MPIDecomp, x) = MPI.Allreduce(x, +, d.comm)
allreduce_max(d::MPIDecomp, x) = MPI.Allreduce(x, max, d.comm)
allreduce_min(d::MPIDecomp, x) = MPI.Allreduce(x, min, d.comm)

"""The buffers for the tile grid g, once (a later call with a larger grid enlarges them)."""
function init_exchange!(d::MPIDecomp, g::KAGrid)
    L = exchange_length(g)
    b = d.buf[]
    (b !== nothing && length(b.send[1]) >= L) && return d
    be = KA.get_backend(g)
    d.buf[] = ExchangeBuffers([KA.zeros(be, Float64, L) for _ in 1:4], [KA.zeros(be, Float64, L) for _ in 1:4],
                              MPI.UnsafeMultiRequest(8))
    return d
end
function _buffers(d::MPIDecomp)
    b = d.buf[]
    b === nothing && error("MPIDecomp: no buffers (init_exchange!(dec, grid), done by MoistKA)")
    return b
end

"""The values o+1 … o+prod(dims) of the buffer B as an array of size dims."""
function _slab(B, o, dims)
    n = prod(dims)
    o + n <= length(B) || error("exchange: message of $(o + n) values, buffers of $(length(B)) (exchange_length)")
    return reshape(view(B, o+1:o+n), dims)
end

"""Post the receives and sends of the sides with the first n[s] values of the buffers (wait: MPI.Waitall(b.reqs))."""
function _sendrecv!(d::MPIDecomp, b, n, sides)
    nb = _nb(d)
    for s in sides
        MPI.Irecv!(view(b.recv[s], 1:n[s]), d.comm, b.reqs[s]; source = nb[s], tag = 10 + s)
    end
    for s in sides                       # the slab sent to side s arrives there as the halo of side OPP[s]
        MPI.Isend(view(b.send[s], 1:n[s]), d.comm, b.reqs[4+s]; dest = nb[s], tag = 10 + OPP[s])
    end
    return b
end
_sync(x) = KA.synchronize(KA.get_backend(x))

# ------------------------------------------------------------------ halo exchange (blocking, HALO deep, with corners)
"""Pack the slabs A[r1, r2, :, v] of all (array, v) of `items` into the buffer B; the number of values."""
function _pack!(B, items, r1, r2)
    o = 0
    for (A, v) in items
        V = view(_as4(A), r1, r2, :, v)
        _slab(B, o, size(V)) .= V
        o += length(V)
    end
    return o
end
function _unpack!(B, items, r1, r2)
    o = 0
    for (A, v) in items
        V = view(_as4(A), r1, r2, :, v)
        V .= _slab(B, o, size(V))
        o += length(V)
    end
    return nothing
end

function _exchange_items!(d::MPIDecomp, items, g::KAGrid)
    H, nxl, nyl = HALO, g.nxl, g.nyl
    b = _buffers(d)
    A1 = _as4(first(items)[1])
    n = zeros(Int, 4)
    # x: the first / last H interior columns to the west / east, receive the west / east halo
    n[W_] = _pack!(b.send[W_], items, H+1:2H, 1:g.nyh)
    n[E_] = _pack!(b.send[E_], items, nxl+1:nxl+H, 1:g.nyh)
    _sync(A1)                                                      # packed before MPI reads the buffers
    _sendrecv!(d, b, n, (W_, E_)); MPI.Waitall(b.reqs)
    _unpack!(b.recv[W_], items, 1:H, 1:g.nyh); _unpack!(b.recv[E_], items, H+nxl+1:2H+nxl, 1:g.nyh)
    # y: rows over the whole x range (halo columns included: corners)
    n[S_] = _pack!(b.send[S_], items, 1:g.nxh, H+1:2H)
    n[N_] = _pack!(b.send[N_], items, 1:g.nxh, nyl+1:nyl+H)
    _sync(A1)
    _sendrecv!(d, b, n, (S_, N_)); MPI.Waitall(b.reqs)
    _unpack!(b.recv[S_], items, 1:g.nxh, 1:H); _unpack!(b.recv[N_], items, 1:g.nxh, H+nyl+1:2H+nyl)
    _sync(A1)
    return nothing
end

function exchange!(d::MPIDecomp, A, g::KAGrid, vars::Tuple = Tuple(1:size(_as4(A), 4)))
    _exchange_items!(d, [(A, v) for v in vars], g)
    return A
end
function exchange!(d::MPIDecomp, As::Tuple, g::KAGrid)
    _exchange_items!(d, [(A, v) for A in As for v in 1:size(_as4(A), 4)], g)
    return As
end

# ------------------------------------------------------------------ tile-boundary faces
"""Tile-boundary faces: the value of the upwind tile on both tiles (x-faces by the sign of Fx, then y-faces)."""
function merge_faces!(d::MPIDecomp, Ax, Fx, Ay, Fy, g::KAGrid)
    H, nxl, nyl = HALO, g.nxl, g.nyl
    b = _buffers(d)
    n = zeros(Int, 4)
    _sync(Ax)
    for (A, F, dir) in ((Ax, Fx, 1), (Ay, Fy, 2))
        nn = dir == 1 ? nxl : nyl
        l, r = H + 1, H + nn + 1
        lo, hi = dir == 1 ? (W_, E_) : (S_, N_)
        sl(B, f) = dir == 1 ? view(B, f, :, :, :) : view(B, :, f, :, :)
        sF(f) = dir == 1 ? view(F, f, :, :) : view(F, :, f, :)
        dims = size(sl(A, l))
        Al, Ar = _slab(b.send[lo], 0, dims), _slab(b.send[hi], 0, dims)   # my left / right boundary face values
        Al .= sl(A, l); Ar .= sl(A, r)
        n[lo] = n[hi] = prod(dims)
        _sync(A)
        _sendrecv!(d, b, n, (lo, hi)); MPI.Waitall(b.reqs)
        fromlo, fromhi = _slab(b.recv[lo], 0, dims), _slab(b.recv[hi], 0, dims)
        sl(A, l) .= ifelse.(signbit.(sF(l)), Al, fromlo)            # flow from lo: lo's value
        sl(A, r) .= ifelse.(signbit.(sF(r)), fromhi, Ar)            # flow from hi: hi's value
        _sync(A)                                                   # buffers free for the next direction
    end
    return nothing
end

# ------------------------------------------------------------------ non-blocking side exchange (fast loop)
# Slabs of the four sides (interior rows / columns only, no corners): x-sides (nyl, nk, nv), y-sides (nxl, nk, nv),
# packed on the device into the side buffers, Isend/Irecv; finish: Waitall, unpack.

@kernel function _k_packx!(buf, @Const(A), I0, vars)
    j, k, n = @index(Global, NTuple)
    @inbounds buf[j, k, n] = A[I0, HALO + j, k, vars[n]]
end
@kernel function _k_unpackx!(A, @Const(buf), I0, vars)
    j, k, n = @index(Global, NTuple)
    @inbounds A[I0, HALO + j, k, vars[n]] = buf[j, k, n]
end
@kernel function _k_packy!(buf, @Const(A), J0, vars)
    i, k, n = @index(Global, NTuple)
    @inbounds buf[i, k, n] = A[HALO + i, J0, k, vars[n]]
end
@kernel function _k_unpacky!(A, @Const(buf), J0, vars)
    i, k, n = @index(Global, NTuple)
    @inbounds A[HALO + i, J0, k, vars[n]] = buf[i, k, n]
end

struct HaloHandle{A,V}
    A::A
    vars::V
    g::KAGrid
end
"""The side slabs (x-sides (nyl, nk, nv), y-sides (nxl, nk, nv)) of the buffers bufs."""
_sides(bufs, g, nk, nv) = ntuple(s -> _slab(bufs[s], 0, (s <= 2 ? g.nyl : g.nxl, nk, nv)), 4)

function start_halo!(d::MPIDecomp, A, g::KAGrid, vars::Tuple)
    H, nxl, nyl = HALO, g.nxl, g.nyl
    nk, nv = size(A, 3), length(vars)
    be = KA.get_backend(A)
    b = _buffers(d)
    sb = _sides(b.send, g, nk, nv)
    # sides: west (first column), east (last column), south (first row), north (last row)
    _k_packx!(be)(sb[W_], A, H + 1, vars; ndrange = (nyl, nk, nv))
    _k_packx!(be)(sb[E_], A, H + nxl, vars; ndrange = (nyl, nk, nv))
    _k_packy!(be)(sb[S_], A, H + 1, vars; ndrange = (nxl, nk, nv))
    _k_packy!(be)(sb[N_], A, H + nyl, vars; ndrange = (nxl, nk, nv))
    KA.synchronize(be)                                        # packed (and the last unpack done) before MPI
    _sendrecv!(d, b, map(length, sb), (W_, E_, S_, N_))
    return HaloHandle(A, vars, g)
end

function finish_halo!(d::MPIDecomp, h::HaloHandle)
    b = _buffers(d)
    MPI.Waitall(b.reqs)
    g = h.g; H, nxl, nyl = HALO, g.nxl, g.nyl
    nk, nv = size(h.A, 3), length(h.vars)
    be = KA.get_backend(h.A)
    rb = _sides(b.recv, g, nk, nv)
    _k_unpackx!(be)(h.A, rb[W_], 1, h.vars; ndrange = (nyl, nk, nv))              # from the west: west halo
    _k_unpackx!(be)(h.A, rb[E_], H + nxl + 1, h.vars; ndrange = (nyl, nk, nv))    # from the east: east halo
    _k_unpacky!(be)(h.A, rb[S_], 1, h.vars; ndrange = (nxl, nk, nv))
    _k_unpacky!(be)(h.A, rb[N_], H + nyl + 1, h.vars; ndrange = (nxl, nk, nv))
    return nothing
end
finish_halo!(::MPIDecomp, ::Nothing) = nothing

end
