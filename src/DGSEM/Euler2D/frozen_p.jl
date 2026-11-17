# DG with the pressure as an additional prognostic variable; horizontal/vertical split of the frozen
# operator and an implicit vertical part for the small steps :cs / :csfb / :ei / :wrf / :hevi
#
# State q̃ = [q; p] (q = (ρ, m_x, m_z, Θ) of `Euler2DModel`, p nodal after the base unknowns). In
# continuous time p = p(Θ); p has its own equation p_t = P'(Θ) Θ_t, P'(Θ) = γ p/Θ, and is reset to
# p(Θ) after every MIS step (`frozen_step_end!`), as π/p in the FV model (src/FV/exner.jl).
#
# Frozen form (coefficients C(y) of `frozen_data!` of the base model, plus P'(y) per node): the fast
# part of `frozen.jl` with the nodal p(Θ_z) replaced by the prognostic p of z. It is then linear in
# z = (ρ, m, Θ, p):
#   volume pair (a, b), direction ξ or η, contravariant m̃ = ax m_x + az m_z (averaged metric):
#     ρ_a  -= (D_ab/J_a) (C1 m̃_a + C2 m̃_b),   Θ_a -= (D_ab/J_a) (C3 m̃_a + C4 m̃_b),
#     m_a  -= (D_ab/J_a) (ax, az) [(p_a + p_b) + C5 ½(ρ_a + ρ_b) g (z_b - z_a)]
#   faces: LMARS with frozen ρ_up, θ_up, ρ_m c_m (base `frozen.jl`), walls by the mirror state;
#   p rows: P'_a times the Θ row of node a (volume and faces).
# N(q̃) = F_f(C(q̃), q̃) + S(q̃) equals the base N(q) where p = p(Θ).
#
# Split: horizontal = ξ-volume pairs and ξ-faces (explicit), vertical = η-volume pairs and η-faces
# (top/bottom element faces and walls; implicit). On terrain the η part acts on both momenta
# (metric term z_x ∂η p and gravity in the η pairs): the column system contains (ρ, m_x, m_z, Θ, p)
# of a node column (ex, i) and is solved as one sparse LU per stage and weight pair (UMFPACK).
# Parts of `frozen_fast!`: :hthermo = horizontal thermodynamic rows + their forcing + sponge,
# :hmomentum = horizontal momentum rows + their forcing (so that b⁺ contains the forcing before the
# horizontal mass flux of a⁺ is evaluated), :vertical = η part (no forcing); :thermo, :momentum, :all
# as in the base model.

using SparseArrays: sparse, SparseMatrixCSC, spzeros
using LinearAlgebra: lu, lu!, ldiv!, mul!, Diagonal, I

"""Stage data: the coefficients C of the stage (`frozen_stage_begin!`), the operator assembled as sparse
matrices J_h (ξ) and J_v (η) (each fast evaluation with these C is then a matrix-vector product), and
the LU factors of I - D J_v."""
mutable struct DGVertical
    C::Vector{Float64}                                  # coefficients of the current stage
    Jh::SparseMatrixCSC{Float64,Int}
    Jv::SparseMatrixCSC{Float64,Int}
    valid::Bool
    hmap::Vector{Int}                                   # emission -> nzval position of J_h (pattern fixed)
    vmap::Vector{Int}
    factors::Dict{Tuple{Float64,Float64},Any}           # LU of I - D J_v per (ct, cw), this stage
    lus::Dict{Tuple{Float64,Float64},Any}               # LU objects kept across stages (symbolic reuse)
    zp::Vector{Float64}                                 # work: predicted state
    zv::Vector{Float64}                                 # work
end

"""
    DGPressureModel(base::Euler2DModel; pvar = :p)

DG model with the pressure p as an additional prognostic variable (state `[q; p]`, `pressure_state`),
frozen split with horizontal (:hthermo, :hmomentum) and vertical (:vertical) parts and a column-implicit
vertical step, so that the small steps :cs, :csfb, :ei, :wrf and :hevi of `TimeStepping.frozen_fast_solve!`
apply. p is reset to p(Θ) after every MIS step.
"""
struct DGPressureModel
    base::Euler2DModel
    pvar::Symbol
    nq::Int
    C::Vector{Float64}
    S::Vector{Float64}
    vert::DGVertical
end
function DGPressureModel(base::Euler2DModel; pvar::Symbol = :p)
    pvar === :p || error("DGPressureModel: pvar = :p (π not yet)")
    M = base.mesh
    nq = base.nq + M.np^2 * M.Ne
    nc = frozen_ncoef(base) + M.np^2 * M.Ne
    return DGPressureModel(base, pvar, nq, zeros(nc), zeros(nq),
                           DGVertical(zeros(nc), spzeros(nq, nq), spzeros(nq, nq), false, Int[], Int[],
                                      Dict{Tuple{Float64,Float64},Any}(), Dict{Tuple{Float64,Float64},Any}(), zeros(nq), zeros(nq)))
end

@inline _pnode(np, e, i, j) = ((e - 1) * np + (j - 1)) * np + i
"""Index of the pressure of node (i, j) of element e in the extended state."""
@inline ip_dg(m::DGPressureModel, e, i, j) = m.base.nq + _pnode(m.base.mesh.np, e, i, j)

"""Extended state [q; p(Θ)] of a base state q."""
function pressure_state(m::DGPressureModel, q)
    M = m.base.mesh; np = M.np
    x = vcat(q[1:m.base.nq], zeros(np^2 * M.Ne))
    frozen_step_end!(m, x)
    return x
end
"""Base state [ρ; m; Θ] of an extended state."""
base_state(m::DGPressureModel, q) = q[1:m.base.nq]
sound_speed_max(m::DGPressureModel, q) = sound_speed_max(m.base, q)
project2d(m::DGPressureModel, fun) = pressure_state(m, project2d(m.base, fun))
output_fields(m::DGPressureModel, q) = output_fields(m.base, base_state(m, q))
field_section(m::DGPressureModel, q, name, z) = field_section(m.base, base_state(m, q), name, z)
total2d(m::DGPressureModel, q, r) = total2d(m.base, q, r)

frozen_ncoef(m::DGPressureModel) = length(m.C)

"""C(y) and S(y): those of the base model, plus P'(Θ) = γ p/Θ per node."""
function frozen_data!(C, S, m::DGPressureModel, q)
    b = m.base; M = b.mesh; np = M.np
    frozen_data!(C, S, b, q)                   # writes the base coefficients; S = 0 on the p rows
    nb = frozen_ncoef(b)
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
        Θ = q[node_iq(np, e, 4, i, j)]
        C[nb + _pnode(np, e, i, j)] = b.gas.γ * pressure(b.gas, Θ) / Θ
    end
    return C, S
end

# --------------------------------------------------------------- the linear frozen operator
struct _DGApply{V,Z}
    out::V
    z::Z
end
@inline (a::_DGApply)(r, c, v) = (@inbounds a.out[r] += v * a.z[c]; nothing)
struct _DGCollect
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
end
@inline (a::_DGCollect)(r, c, v) = (push!(a.I, r); push!(a.J, c); push!(a.V, v); nothing)
mutable struct _DGFill
    nz::Vector{Float64}
    map::Vector{Int}
    n::Int
end
@inline (a::_DGFill)(r, c, v) = (@inbounds a.nz[a.map[a.n += 1]] += v; nothing)

"""Face node pairs as `_foreach_face` (same order, so the face coefficients match), with the
direction (1: ξ-face, 2: η-face or wall)."""
@inline function _foreach_face_dir(f, M::Mesh2D)
    np, w = M.np, M.wg
    fi = 0
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            f(fi += 1, 1, :interior, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[np])
        end
        for i in 1:np
            if ez < M.Nez
                f(fi += 1, 2, :interior, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[np])
            else
                f(fi += 1, 2, :wall, i, np, e, i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[np])
            end
            ez == 1 && f(fi += 1, 2, :wall, i, 1, e, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[1])
        end
    end
    return fi
end

"""
    _dgp_terms!(emit, m, C, dirs)

All terms of the linear fast operator in the directions `dirs` (1 = ξ, 2 = η; volume pairs and
faces) as emit(row, column, coefficient): out[row] += coefficient z[column].
"""
function _dgp_terms!(emit::E, m::DGPressureModel, C, dirs) where {E}
    b = m.base; M = b.mesh
    np, Ne, D, g = M.np, M.Ne, M.D, b.gas.g
    nb = frozen_ncoef(b); nqb = b.nq
    iq(e, r, i, j) = node_iq(np, e, r, i, j)
    ipr(e, i, j) = nqb + _pnode(np, e, i, j)
    CP(e, i, j) = C[nb + _pnode(np, e, i, j)]
    # Θ row and the p row (P' times the Θ row) of node (e, i, j)
    @inline emitΘ(e, i, j, col, v) = (emit(iq(e, 4, i, j), col, v); emit(ipr(e, i, j), col, CP(e, i, j) * v))
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        Jinv = 1 / M.J[i, j, e]
        ρa, mxa, mza, pa = iq(e, 1, i, j), iq(e, 2, i, j), iq(e, 3, i, j), ipr(e, i, j)
        for dir in dirs, k in 1:np
            ib, jb = dir == 1 ? (k, j) : (i, k)
            ax = dir == 1 ? 0.5 * (M.Jξx[i, j, e] + M.Jξx[k, j, e]) : 0.5 * (M.Jηx[i, j, e] + M.Jηx[i, k, e])
            az = dir == 1 ? 0.5 * (M.Jξz[i, j, e] + M.Jξz[k, j, e]) : 0.5 * (M.Jηz[i, j, e] + M.Jηz[i, k, e])
            o = _vc(M, dir, i, j, k, e)
            f = -Jinv * D[dir == 1 ? i : j, k]
            f == 0 && continue
            ρb, mxb, mzb, pb = iq(e, 1, ib, jb), iq(e, 2, ib, jb), iq(e, 3, ib, jb), ipr(e, ib, jb)
            emit(ρa, mxa, f * C[o+1] * ax); emit(ρa, mza, f * C[o+1] * az)
            emit(ρa, mxb, f * C[o+2] * ax); emit(ρa, mzb, f * C[o+2] * az)
            emitΘ(e, i, j, mxa, f * C[o+3] * ax); emitΘ(e, i, j, mza, f * C[o+3] * az)
            emitΘ(e, i, j, mxb, f * C[o+4] * ax); emitΘ(e, i, j, mzb, f * C[o+4] * az)
            G = f * C[o+5] * 0.5 * g * (M.Z[ib, jb, e] - M.Z[i, j, e])
            for (row, a) in ((mxa, ax), (mza, az))
                emit(row, pa, f * a); emit(row, pb, f * a)
                emit(row, ρa, G * a); emit(row, ρb, G * a)
            end
        end
    end
    _foreach_face_dir(M) do fi, dir, kind, iL, jL, eL, iR, jR, eR, ñx, ñz, we
        dir in dirs || return nothing
        s = hypot(ñx, ñz); n1, n2 = ñx / s, ñz / s
        o = _fc(M, fi)
        ρL, mxL, mzL, pL = iq(eL, 1, iL, jL), iq(eL, 2, iL, jL), iq(eL, 3, iL, jL), ipr(eL, iL, jL)
        σL = 1 / (M.J[iL, jL, eL] * we)
        mn(row, mx, mz, v) = (emit(row, mx, v * n1); emit(row, mz, v * n2))
        mnΘ(e, i, j, mx, mz, v) = (emitΘ(e, i, j, mx, v * n1); emitΘ(e, i, j, mz, v * n2))
        if kind === :wall
            c = σL * s
            mn(ρL, mxL, mzL, c)
            mnΘ(eL, iL, jL, mxL, mzL, c * C[o+2])
            mn(mxL, mxL, mzL, -c * n1 * C[o+1])
            mn(mzL, mxL, mzL, -c * n2 * C[o+1])
        else
            ρR, mxR, mzR, pR = iq(eR, 1, iR, jR), iq(eR, 2, iR, jR), iq(eR, 3, iR, jR), ipr(eR, iR, jR)
            σR = 1 / (M.J[iR, jR, eR] * we)
            cL, cR = σL * s, σR * s
            # left: -cL (m_f - m_nL), -cL (f_Θ - θ_L m_nL), -cL n (p_s - p_L)
            mn(ρL, mxL, mzL, -cL * (C[o+1] - 1)); mn(ρL, mxR, mzR, -cL * C[o+2])
            emit(ρL, pR, cL * C[o+3]); emit(ρL, pL, -cL * C[o+3])
            mnΘ(eL, iL, jL, mxL, mzL, -cL * (C[o+4] - C[o+9])); mnΘ(eL, iL, jL, mxR, mzR, -cL * C[o+5])
            emitΘ(eL, iL, jL, pR, cL * C[o+6]); emitΘ(eL, iL, jL, pL, -cL * C[o+6])
            for (row, nn) in ((mxL, n1), (mzL, n2))
                emit(row, pR, -cL * nn * 0.5); emit(row, pL, cL * nn * 0.5)
                mn(row, mxR, mzR, cL * nn * C[o+8]); mn(row, mxL, mzL, -cL * nn * C[o+7])
            end
            # right: +cR (m_f - m_nR), +cR (f_Θ - θ_R m_nR), +cR n (p_s - p_R)
            mn(ρR, mxL, mzL, cR * C[o+1]); mn(ρR, mxR, mzR, cR * (C[o+2] - 1))
            emit(ρR, pR, -cR * C[o+3]); emit(ρR, pL, cR * C[o+3])
            mnΘ(eR, iR, jR, mxL, mzL, cR * C[o+4]); mnΘ(eR, iR, jR, mxR, mzR, cR * (C[o+5] - C[o+10]))
            emitΘ(eR, iR, jR, pR, -cR * C[o+6]); emitΘ(eR, iR, jR, pL, cR * C[o+6])
            for (row, nn) in ((mxR, n1), (mzR, n2))
                emit(row, pL, cR * nn * 0.5); emit(row, pR, -cR * nn * 0.5)
                mn(row, mxR, mzR, -cR * nn * C[o+8]); mn(row, mxL, mzL, cR * nn * C[o+7])
            end
        end
        nothing
    end
    return nothing
end

@inline _dgp_isthermo(m::DGPressureModel, k) = k > m.base.nq || mod1(cld(k, m.base.mesh.np^2), NV) in (1, 4)

"""Fast part F_f(C, z) + F on the rows of `part`: :all, :thermo (ρ, Θ, p), :momentum (m_x, m_z),
:hthermo (horizontal thermodynamic rows and their forcing, sponge), :hmomentum (horizontal momentum
rows and their forcing), :vertical (η part, all rows, no forcing)."""
function frozen_fast!(out, m::DGPressureModel, C, z, part::Symbol, F)
    dirs = part in (:all, :thermo, :momentum) ? (1, 2) : part in (:hthermo, :hmomentum) ? (1,) :
           part === :vertical ? (2,) : error("frozen_fast!: unknown part $part")
    v = m.vert
    if v.valid && C === v.C                        # stage matrices
        if dirs == (1, 2)
            mul!(out, v.Jh, z); mul!(out, v.Jv, z, 1.0, 1.0)
        else
            mul!(out, dirs == (1,) ? v.Jh : v.Jv, z)
        end
    else
        fill!(out, 0.0)
        _dgp_terms!(_DGApply(out, z), m, C, dirs)
    end
    part === :vertical && return out
    keepth = part in (:all, :thermo, :hthermo)
    keepmo = part in (:all, :momentum, :hmomentum)
    Fth = part in (:all, :thermo, :hthermo)
    Fmo = part in (:all, :momentum, :hmomentum)      # momentum forcing with the momentum step (in b⁺ before :hthermo)
    @inbounds for k in eachindex(out)
        th = _dgp_isthermo(m, k)
        (th ? keepth : keepmo) || (out[k] = 0.0)
        (th ? Fth : Fmo) && (out[k] += F[k])
    end
    part in (:all, :momentum, :hthermo) && sponge!(out, m.base, z)
    return out
end

"""N(q̃) = F(q̃, q̃) of the extended system."""
function rhs_nonlinear!(out, m::DGPressureModel, q::AbstractVector)
    frozen_data!(m.C, m.S, m, q)
    return frozen_fast!(out, m, m.C, q, :all, m.S)
end

"""p ← p(Θ) after a MIS step."""
function frozen_step_end!(m::DGPressureModel, q)
    b = m.base; M = b.mesh; np = M.np
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
        q[ip_dg(m, e, i, j)] = pressure(b.gas, q[node_iq(np, e, 4, i, j)])
    end
    return q
end

# --------------------------------------------------------------- vertically implicit part
"""Sparse matrix of the operator terms in the directions dirs for the coefficients C. The first call
builds the pattern and the map emission -> nzval position; later calls only refill the values of A."""
function _dgp_matrix!(A, map, m::DGPressureModel, C, dirs)
    if isempty(map)
        col = _DGCollect(Int[], Int[], Float64[])
        _dgp_terms!(col, m, C, dirs)
        A = sparse(col.I, col.J, col.V, m.nq, m.nq)
        resize!(map, length(col.I))
        @inbounds for t in eachindex(col.I)
            r, c = col.I[t], col.J[t]
            rng = A.colptr[c]:A.colptr[c+1]-1
            map[t] = first(rng) - 1 + searchsortedfirst(view(A.rowval, rng), r)
        end
        return A
    end
    fill!(A.nzval, 0.0)
    f = _DGFill(A.nzval, map, 0)
    _dgp_terms!(f, m, C, dirs)
    f.n == length(map) || error("DG operator pattern changed")
    return A
end

"""Assemble J_h and J_v of the stage coefficients C (C must not change during the stage)."""
function frozen_stage_begin!(m::DGPressureModel, C, z)
    v = m.vert
    v.C = C
    v.Jh = _dgp_matrix!(v.Jh, v.hmap, m, C, (1,))
    v.Jv = _dgp_matrix!(v.Jv, v.vmap, m, C, (2,))
    v.valid = true
    empty!(v.factors)
    return nothing
end

"""Scale the thermodynamic rows by ct and the momentum rows by cw."""
function _dgp_scale!(out, m::DGPressureModel, ct, cw)
    @inbounds for k in eachindex(out)
        out[k] *= _dgp_isthermo(m, k) ? ct : cw
    end
    return out
end

"""LU of I - D J_v (D = ct on the thermodynamic rows, cw on the momentum rows) for the stage
coefficients; J_v from the same kernel as the operator. Block diagonal over the node columns."""
function _dgp_factor(m::DGPressureModel, ct, cw)
    v = m.vert
    v.valid || error("DGPressureModel: frozen_stage_begin! must precede the vertical solve")
    key = (Float64(ct), Float64(cw))
    get!(v.factors, key) do
        d = [_dgp_isthermo(m, r) ? ct : cw for r in 1:m.nq]
        A = I - Diagonal(d) * v.Jv
        v.lus[key] = haskey(v.lus, key) ? lu!(v.lus[key], A) : lu(A)    # same pattern: symbolic reuse
    end
end

"""out = D G_v(z), D = ct (thermodynamic rows), cw (momentum rows)."""
function frozen_vertical_rhs!(out, m::DGPressureModel, C, z, ct, cw)
    frozen_fast!(out, m, C, z, :vertical, out)
    return _dgp_scale!(out, m, ct, cw)
end

"""z ← z + Y, (I - D J_v) Y = D G_v(z): exact implicit step of the linear vertical part."""
function frozen_vertical_implicit!(z, m::DGPressureModel, C, ct, cw, k, y)
    frozen_vertical_rhs!(k, m, C, z, ct, cw)
    ldiv!(y, _dgp_factor(m, ct, cw), k)
    @. z += y
    return z
end

# --------------------------------------------------------------- predictors of the :cs / :csfb step
# On terrain the η part acts on m_x (metric term z_x ∂η p, gravity, wall LMARS), and the ξ part depends on
# m_x (horizontal divergence; LMARS dissipation of the ξ-faces). Evaluated with the momenta before the
# column solve, both lag by βh L₁a⁺ and the small step drops to first order (docs/frozen_dg). Hence
#   :hthermo of a⁺ (`frozen_hthermo!`, w = βh): at b̂ = b + w (J_v z)_b (the implicit vertical force at the
#       current state; the remaining lag w J_v(z⁺ - z) = O(h²) keeps the order, as the FV metric predictor),
#   :hmomentum of b⁺ (`frozen_hmomentum!`): at the midpoint predictor b + h/2 (J_h z)_b (the ξ-face
#       dissipation b←b is then a second-order midpoint step instead of explicit Euler).
# Forcing and sponge are evaluated at z.

frozen_hthermo_reeval(::DGPressureModel) = true       # LMARS: the ξ-face mass flux depends on [p]

"""Momentum rows of y ← z + c (src)_b (thermodynamic rows of z unchanged)."""
function _dgp_predict!(y, m::DGPressureModel, z, c, src)
    @inbounds for k in eachindex(y)
        y[k] = _dgp_isthermo(m, k) ? z[k] : z[k] + c * src[k]
    end
    return y
end

function frozen_hthermo!(out, m::DGPressureModel, C, z, F, w)
    w == 0 && return frozen_fast!(out, m, C, z, :hthermo, F)
    v = m.vert
    frozen_fast!(v.zv, m, C, z, :vertical, F)
    _dgp_predict!(v.zp, m, z, w, v.zv)
    frozen_fast!(out, m, C, v.zp, :hthermo, F)
    b = m.base
    @inbounds for k in eachindex(b.damp)                 # sponge at z, not at the predictor
        b.damp[k] != 0 && (out[k] += b.damp[k] * (v.zp[k] - z[k]))
    end
    return out
end

function frozen_hmomentum!(out, m::DGPressureModel, C, z, F, h)
    v = m.vert
    frozen_fast!(v.zv, m, C, z, :hmomentum, F)
    _dgp_predict!(v.zp, m, z, h / 2, v.zv)
    return frozen_fast!(out, m, C, v.zp, :hmomentum, F)
end

"""y = (I - D J_v)⁻¹ R."""
frozen_vertical_solve!(y, m::DGPressureModel, C, ct, cw, R) = ldiv!(y, _dgp_factor(m, ct, cw), R)

"""Crank–Nicolson step of the vertical part: z ← z + (I - dt/2 J_v)⁻¹ dt G_v(z)."""
function frozen_vertical_step!(z, m::DGPressureModel, C, dt, k, y)
    frozen_vertical_rhs!(k, m, C, z, dt, dt)
    ldiv!(y, _dgp_factor(m, dt / 2, dt / 2), k)
    @. z += y
    return z
end
