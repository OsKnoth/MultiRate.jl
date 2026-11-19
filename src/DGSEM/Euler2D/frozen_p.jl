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
# of a node column (ex, i) and is solved as one sparse LU per stage and weight pair (UMFPACK), or, with
# vsolver = :hdg, by static condensation as in hybridised DG (end of this file).
# Parts of `frozen_fast!`: :hthermo = horizontal thermodynamic rows + their forcing + sponge,
# :hmomentum = horizontal momentum rows + their forcing (so that b⁺ contains the forcing before the
# horizontal mass flux of a⁺ is evaluated), :vertical = η part (no forcing); :thermo, :momentum, :all
# as in the base model.

using SparseArrays: sparse, SparseMatrixCSC, spzeros, findnz, nzrange, rowvals, nonzeros
using LinearAlgebra: lu, lu!, ldiv!, mul!, Diagonal, I, diag

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
    zs::Vector{Float64}                                 # work
    groups::Vector{Vector{Int}}                         # diagonal blocks of D₀ (ξ-face dissipation, a←a and b←b of J_h)
    jpos::Vector{Matrix{Int}}                           # positions of the block entries in J_h.nzval (0: none)
    epos::Vector{Matrix{Int}}                           # positions of the blocks in Epat.nzval
    diagpos::Vector{Int}                                # identity entries of Epat outside the blocks
    Epat::SparseMatrixCSC{Float64,Int}                  # pattern of e^{cD₀} and c φ₁(cD₀)
    flows::Dict{Float64,Tuple{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Float64,Int}}}   # c -> (e^{cD₀}, c φ₁(cD₀))
end

"""
    DGPressureModel(base::Euler2DModel; pvar = :p, lmars = :velocity, vsplit = :full, vsolver = :lu, hpredict = true)

DG model with the pressure p as an additional prognostic variable (state `[q; p]`, `pressure_state`),
frozen split with horizontal (:hthermo, :hmomentum) and vertical (:vertical) parts and a column-implicit
vertical step, so that the small steps :cs, :csfb, :csb, :ei, :wrf and :hevi of `TimeStepping.frozen_fast_solve!`
apply. p is reset to p(Θ) after every MIS step.

`lmars`: the LMARS of the fast face terms, `:velocity` (the base model: w* = {u_n} - [p]/(2ρ̄c), m_f = ρ_up w*,
p* = {p} - ρ̄c [u_n]/2 with frozen densities) or `:momentum` (in ρu: m_f = {m_n} - [p]/(2c), p* = {p} - c [m_n]/2,
c the mean face sound speed; θ_up from the upwind side of the base). The slow part is that of the base model.

`vsplit`: what of the η-terms is vertical (implicit). `:full`: all η volume pairs, η-faces and walls, acting on
both momenta (on terrain the column reads m_x). `:contravariant` (the FV trick, src/FV/vertical.jl): the η-terms are
split by the components of the vertical normal ñ_η = (Jη_x, Jη_z) of the contravariant vertical flux ñ_η·m; every
term with m_x (the Jη_x m_x part of the η-mass flux, the metric force Jη_x ∂_η p and the η-face terms on m_x) goes
to the horizontal, explicit part, so that the column contains only ρ, Θ, p, m_z.
`:contravariant_volume`: only the η volume pairs are split this way; the η-faces and walls stay vertical (the column
reads m_x at the face nodes). Identical to `:full` on a flat grid.

`vsolver`: the column solve of I - D J_v. `:lu` one sparse LU (UMFPACK); `:hdg` (needs `lmars = :momentum`) static
condensation as in hybridised DG: the interface momentum M̂ = s m_f of the η-faces is introduced only in the solve,
the elements are eliminated locally and a tridiagonal system in M̂ remains per node column (`_hdg_factor`); the
result is that of the LU.

`hpredict`: the horizontal momentum step of :cs with the midpoint predictor of the ξ-face self-coupling (true) or
with D_M^x explicit (false, docs/fast_2d scheme S2).
"""
struct DGPressureModel
    base::Euler2DModel
    pvar::Symbol
    nq::Int
    C::Vector{Float64}
    S::Vector{Float64}
    vert::DGVertical
    hybrid::Bool                                        # internal: face terms written with M̂ (vsolver = :hdg only)
    lmars::Symbol
    vsplit::Symbol
    mhat::Vector{Int}                                   # hybrid: face pair index fi -> index of M̂ (0: wall)
    vsolver::Symbol                                     # column solve: :lu (sparse LU) or :hdg (static condensation)
    hpredict::Bool                                      # midpoint predictor of the horizontal momentum step
    aux::Union{Nothing,DGPressureModel}                 # :hdg: the same operator written with M̂ (only its J_v is used)
    cols::Vector{Tuple{Vector{Vector{Int}},Vector{Int}}}  # :hdg: per node column the element unknowns and the η-face M̂ (aux)
end
function DGPressureModel(base::Euler2DModel; pvar::Symbol = :p, lmars::Symbol = :velocity, vsplit::Symbol = :full,
                         vsolver::Symbol = :lu, hpredict::Bool = true)
    vsolver in (:lu, :hdg) || error("DGPressureModel: vsolver = :lu or :hdg")
    vsolver === :hdg && lmars !== :momentum && error("DGPressureModel: vsolver = :hdg needs lmars = :momentum")
    vsplit in (:full, :contravariant, :contravariant_volume) ||
        error("DGPressureModel: vsplit = :full, :contravariant or :contravariant_volume")
    lmars in (:velocity, :momentum) || error("DGPressureModel: lmars = :velocity or :momentum")
    pvar === :p || error("DGPressureModel: pvar = :p (π not yet)")
    aux = vsolver === :hdg ? _dgp_build(base, pvar, :momentum, vsplit, :lu, hpredict, true, nothing, Tuple{Vector{Vector{Int}},Vector{Int}}[]) : nothing
    cols = vsolver === :hdg ? _hdg_columns(base, aux) : Tuple{Vector{Vector{Int}},Vector{Int}}[]
    return _dgp_build(base, pvar, lmars, vsplit, vsolver, hpredict, false, aux, cols)
end
function _dgp_build(base, pvar, lmars, vsplit, vsolver, hpredict, hybrid, aux, cols)
    M = base.mesh
    nqp = base.nq + M.np^2 * M.Ne
    mhat = Int[]; nM = 0
    if hybrid
        _foreach_face_dir(M) do fi, dir, kind, args...
            push!(mhat, kind === :wall ? 0 : nqp + (nM += 1)); nothing
        end
    end
    nq = nqp + nM
    nc = frozen_ncoef(base) + M.np^2 * M.Ne
    return DGPressureModel(base, pvar, nq, zeros(nc), zeros(nq),
                           DGVertical(zeros(nc), spzeros(nq, nq), spzeros(nq, nq), false, Int[], Int[],
                                      Dict{Tuple{Float64,Float64},Any}(), Dict{Tuple{Float64,Float64},Any}(), zeros(nq), zeros(nq),
                                      zeros(nq), Vector{Int}[], Matrix{Int}[], Matrix{Int}[], Int[], spzeros(nq, nq),
                                      Dict{Float64,Tuple{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Float64,Int}}}()),
                           hybrid, lmars, vsplit, mhat, vsolver, hpredict, aux, cols)
end

"""Node columns (ex, i) for the condensed column solve: the element unknowns (ρ, m_x, m_z, Θ, p at the np nodes of the
η-line i) of the elements ez = 1..Nez, and the M̂ index (in the hybrid model `mi`) of the η-faces between them."""
function _hdg_columns(base::Euler2DModel, mi)
    M = base.mesh; np = M.np
    fM = Dict{Tuple{Int,Int},Int}()
    _foreach_face_dir(M) do fi, dir, kind, iL, jL, eL, args...
        dir == 2 && kind === :interior && (fM[(eL, iL)] = mi.mhat[fi]); nothing
    end
    cols = Tuple{Vector{Vector{Int}},Vector{Int}}[]
    for ex in 1:M.Nex, i in 1:np
        es = [elem(M, ex, ez) for ez in 1:M.Nez]
        idx = [vcat([node_iq(np, e, r, i, j) for r in 1:4 for j in 1:np], [base.nq + _pnode(np, e, i, j) for j in 1:np]) for e in es]
        push!(cols, (idx, [fM[(es[ez], i)] for ez in 1:M.Nez-1]))
    end
    return cols
end

@inline _pnode(np, e, i, j) = ((e - 1) * np + (j - 1)) * np + i
"""Index of the pressure of node (i, j) of element e in the extended state."""
@inline ip_dg(m::DGPressureModel, e, i, j) = m.base.nq + _pnode(m.base.mesh.np, e, i, j)

"""Extended state [q; p(Θ)] of a base state q."""
function pressure_state(m::DGPressureModel, q)
    M = m.base.mesh; np = M.np
    x = vcat(q[1:m.base.nq], zeros(m.nq - m.base.nq))
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
    m.lmars === :momentum && _lmars_momentum!(C, m, q)
    return C, S
end

"""Face coefficients of LMARS in ρu: m_f = {m_n} - [p]/(2c), f_Θ = θ_up m_f, p* = {p} - c [m_n]/2 (interior faces;
the wall p + c m_n is the same in both forms), c = mean of the sound speeds of the two sides, θ_up from the base."""
function _lmars_momentum!(C, m::DGPressureModel, q)
    b = m.base; M = b.mesh; np = M.np; γ = b.gas.γ
    _foreach_face_dir(M) do fi, dir, kind, iL, jL, eL, iR, jR, eR, ñx, ñz, we
        kind === :wall && return nothing
        o = _fc(M, fi)
        sp(e, i, j) = sqrt(γ * pressure(b.gas, q[node_iq(np, e, 4, i, j)]) / q[node_iq(np, e, 1, i, j)])
        cm = b.clmars > 0 ? b.clmars : 0.5 * (sp(eL, iL, jL) + sp(eR, iR, jR))
        θup = C[o+4] / C[o+1]
        C[o+1] = 0.5; C[o+2] = 0.5; C[o+3] = 1 / (2cm)
        C[o+4] = θup * C[o+1]; C[o+5] = θup * C[o+2]; C[o+6] = θup * C[o+3]
        C[o+7] = 0.5cm; C[o+8] = 0.5cm
        nothing
    end
    return C
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
function _dgp_terms!(emit::E, m::DGPressureModel, C, dirs; vol::Bool = true, fac::Bool = true) where {E}
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
        vol || break
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
        (fac && dir in dirs) || return nothing
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
        elseif m.hybrid
            # the LMARS face terms written with the interface momentum M̂ (static condensation, vsolver = :hdg):
            # thermodynamic rows with M̂, momentum rows with the interface pressure of their side,
            # P̂ - p = M̂/(2C₃) - 2C₇ m_nL (L) resp. M̂/(2C₃) - 2C₈ m_nR (R); M̂ row: the constraint s m_f(z) - M̂
            # (P̂_L = P̂_R). With M̂ = s m_f(z) the element rows equal those of the LMARS operator.
            ρR, mxR, mzR, pR = iq(eR, 1, iR, jR), iq(eR, 2, iR, jR), iq(eR, 3, iR, jR), ipr(eR, iR, jR)
            σR = 1 / (M.J[iR, jR, eR] * we)
            cL, cR = σL * s, σR * s
            # M̂ is stored contravariant, M̂ = ñ·m̂ = s m̂_n: the face terms use M̂/s
            kM = m.mhat[fi]
            θup = C[o+4] / C[o+1]
            iZ = 1 / (2 * C[o+3])
            is = 1 / s
            emit(ρL, kM, -cL * is); mn(ρL, mxL, mzL, cL)
            emitΘ(eL, iL, jL, kM, -cL * θup * is); mnΘ(eL, iL, jL, mxL, mzL, cL * C[o+9])
            emit(ρR, kM, cR * is); mn(ρR, mxR, mzR, -cR)
            emitΘ(eR, iR, jR, kM, cR * θup * is); mnΘ(eR, iR, jR, mxR, mzR, -cR * C[o+10])
            for (row, nn) in ((mxL, n1), (mzL, n2))
                emit(row, kM, cL * nn * iZ * is); mn(row, mxL, mzL, -cL * nn * 2 * C[o+7])
            end
            for (row, nn) in ((mxR, n1), (mzR, n2))
                emit(row, kM, cR * nn * iZ * is); mn(row, mxR, mzR, -cR * nn * 2 * C[o+8])
            end
            # constraint row: s m_f - M̂, s m_f = C₁ ñ·m_L + C₂ ñ·m_R - s C₃ [p]
            mn(kM, mxL, mzL, s * C[o+1]); mn(kM, mxR, mzR, s * C[o+2])
            emit(kM, pR, -s * C[o+3]); emit(kM, pL, s * C[o+3]); emit(kM, kM, -1.0)
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

"""Emitter that passes on only the terms with (`mx = true`) or without (`mx = false`) the horizontal momentum m_x
in row or column (the split `vsplit = :contravariant`)."""
struct _DGFilt{E}
    e::E
    nqb::Int
    np2::Int
    mx::Bool
end
@inline _dgp_ismx(f::_DGFilt, k) = k <= f.nqb && mod1(cld(k, f.np2), NV) == 2
@inline (f::_DGFilt)(r, c, v) = ((_dgp_ismx(f, r) || _dgp_ismx(f, c)) == f.mx && f.e(r, c, v); nothing)

"""Terms of the horizontal (`:h`), vertical (`:v`) or whole (`:all`) fast operator, as emit(row, col, coef)."""
function _dgp_part_terms!(emit, m::DGPressureModel, C, which::Symbol)
    which === :all && return _dgp_terms!(emit, m, C, (1, 2))
    if m.vsplit === :full
        return _dgp_terms!(emit, m, C, which === :h ? (1,) : (2,))
    end
    nqb, np2 = m.base.nq, m.base.mesh.np^2
    allfaces = m.vsplit === :contravariant_volume     # only the η volume pairs split, η-faces and walls vertical
    if which === :h
        _dgp_terms!(emit, m, C, (1,))
        _dgp_terms!(_DGFilt(emit, nqb, np2, true), m, C, (2,); fac = !allfaces)
    else
        _dgp_terms!(_DGFilt(emit, nqb, np2, false), m, C, (2,); fac = !allfaces)
        allfaces && _dgp_terms!(emit, m, C, (2,); vol = false)
    end
    return nothing
end

@inline _dgp_isthermo(m::DGPressureModel, k) = k > m.base.nq ? k <= m.base.nq + m.base.mesh.np^2 * m.base.mesh.Ne :
                                                 mod1(cld(k, m.base.mesh.np^2), NV) in (1, 4)

"""Fast part F_f(C, z) + F on the rows of `part`: :all, :thermo (ρ, Θ, p), :momentum (m_x, m_z),
:hthermo (horizontal thermodynamic rows and their forcing, sponge), :hmomentum (horizontal momentum
rows and their forcing), :vertical (η part, all rows, no forcing)."""
function frozen_fast!(out, m::DGPressureModel, C, z, part::Symbol, F)
    which = part in (:all, :thermo, :momentum) ? :all : part in (:hthermo, :hmomentum) ? :h :
            part === :vertical ? :v : error("frozen_fast!: unknown part $part")
    dirs = which === :all ? (1, 2) : which === :h ? (1,) : (2,)
    v = m.vert
    if v.valid && C === v.C                        # stage matrices
        if dirs == (1, 2)
            mul!(out, v.Jh, z); mul!(out, v.Jv, z, 1.0, 1.0)
        else
            mul!(out, dirs == (1,) ? v.Jh : v.Jv, z)
        end
    else
        fill!(out, 0.0)
        _dgp_part_terms!(_DGApply(out, z), m, C, which)
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
function _dgp_matrix!(A, map, m::DGPressureModel, C, which::Symbol)
    if isempty(map)
        col = _DGCollect(Int[], Int[], Float64[])
        _dgp_part_terms!(col, m, C, which)
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
    _dgp_part_terms!(f, m, C, which)
    f.n == length(map) || error("DG operator pattern changed")
    return A
end

"""Assemble J_h and J_v of the stage coefficients C (C must not change during the stage)."""
function frozen_stage_begin!(m::DGPressureModel, C, z)
    v = m.vert
    v.C = C
    v.Jh = _dgp_matrix!(v.Jh, v.hmap, m, C, :h)
    v.Jv = _dgp_matrix!(v.Jv, v.vmap, m, C, :v)
    if m.vsolver === :hdg                              # η part written with M̂, same C
        a = m.aux.vert
        a.Jv = _dgp_matrix!(a.Jv, a.vmap, m.aux, C, :v)
    end
    v.valid = true
    empty!(v.factors)
    _dgp_dissipation!(m)                               # blocks of D₀ (once)
    empty!(v.flows)
    return nothing
end

# --------------------------------------------------------------- Bokhove's partition (:csb)
# J_h = [D_a0 -K₀; L₀ D_b0] (a = ρ, Θ, p; b = m_x, m_z). D₀ = diag(D_a0, D_b0) is the LMARS dissipation of the
# ξ-faces (D_a0: the [p] term of m*, D_b0: the [m_n] term of p*) and couples only the two nodes of a ξ-face:
# block diagonal with blocks of at most 6 unknowns. Xu, van der Vegt, Bokhove (J. Sci. Comput. 35 (2008), eq. 4.6)
# integrate the local self-coupling of a block row exactly inside the split flows; here
#   A_c: a' = -K₀ b, b' = D_b0 b:   b ← e^{cD₀} b,  a ← a + (J_h Φ_c b)_a,
#   B_c: a' = D_a0 a, b' = L₀ a:    a ← e^{cD₀} a,  b ← b + (J_h Φ_c a)_b,     Φ_c = c φ₁(cD₀),
# exact for these flows (block triangular), Φ_c b the mean momentum over the substep (Bokhove's U_int).
# docs/frozen_dg, Section sec:implicitD.

"""Diagonal blocks of D₀ (the a←a and b←b entries of J_h), computed once (the pattern of J_h is fixed): the
unknowns of each block, the positions of its entries in J_h.nzval (0: none), and the fixed pattern of
e^{cD₀} and c φ₁(cD₀) (the blocks plus the identity elsewhere) with the positions of the blocks in it."""
function _dgp_dissipation!(m::DGPressureModel)
    v = m.vert
    isempty(v.groups) || return nothing
    n = m.nq
    Is, Js, _ = findnz(v.Jh)
    keep = [_dgp_isthermo(m, Is[t]) == _dgp_isthermo(m, Js[t]) for t in eachindex(Is)]
    parent = collect(1:n)
    root(x) = (while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end; x)
    act = falses(n)
    for t in eachindex(Is)
        keep[t] || continue
        r, c = Is[t], Js[t]
        act[r] = act[c] = true
        a, b = root(r), root(c)
        a != b && (parent[a] = b)
    end
    g = Dict{Int,Vector{Int}}()
    for k in 1:n
        act[k] && push!(get!(g, root(k), Int[]), k)
    end
    v.groups = sort!(collect(values(g)); by = first)
    # pattern of e^{cD₀}, c φ₁(cD₀)
    IE, JE = Int[], Int[]
    for idx in v.groups, j in idx, i in idx
        push!(IE, i); push!(JE, j)
    end
    for k in 1:n
        act[k] || (push!(IE, k); push!(JE, k))
    end
    v.Epat = sparse(IE, JE, zeros(length(IE)), n, n)
    pos(A, i, j) = (r = A.colptr[j]:A.colptr[j+1]-1; t = searchsortedfirst(view(A.rowval, r), i);
                    t <= length(r) && A.rowval[r[t]] == i ? r[t] : 0)
    v.jpos = [[keep_pos(v.Jh, i, j, m) for i in idx, j in idx] for idx in v.groups]
    v.epos = [[pos(v.Epat, i, j) for i in idx, j in idx] for idx in v.groups]
    v.diagpos = [pos(v.Epat, k, k) for k in 1:n if !act[k]]
    return nothing
end
"""Position of entry (i, j) of J_h in its nzval if it belongs to D₀ (rows and column both a or both b), else 0."""
function keep_pos(A, i, j, m)
    _dgp_isthermo(m, i) == _dgp_isthermo(m, j) || return 0
    r = A.colptr[j]:A.colptr[j+1]-1
    t = searchsortedfirst(view(A.rowval, r), i)
    return t <= length(r) && A.rowval[r[t]] == i ? r[t] : 0
end

const _INVFACT = [Float64(1 / factorial(big(j))) for j in 1:16]      # 1/j!
"""E = e^X and P = φ₁(X) = (e^X - I)/X of a small dense X: Taylor series of φ₁ after scaling to
‖X‖₁ ≤ 1/2, then squaring with φ₁(2X) = ½ φ₁(X)(e^X + I), e^{2X} = (e^X)²."""
function _phi1exp(X::Matrix{Float64})
    s = size(X, 1)
    nrm = maximum(sum(abs, X; dims = 1))
    q = nrm > 0.5 ? ceil(Int, log2(nrm / 0.5)) : 0
    Y = X ./ 2.0^q
    Id = Matrix(1.0I, s, s)
    K = 14
    P = Id .* _INVFACT[K+1]
    for k in K-1:-1:0
        P = Y * P .+ Id .* _INVFACT[k+1]
    end
    E = Id .+ Y * P
    for _ in 1:q
        P = 0.5 .* (P * (E .+ Id))
        E = E * E
    end
    return E, P
end

"""(e^{cD₀}, c φ₁(cD₀)) of the stage, blockwise from the entries of J_h."""
function _dgp_flowmats(m::DGPressureModel, c)
    v = m.vert
    get!(v.flows, Float64(c)) do
        E = copy(v.Epat); Φ = copy(v.Epat)
        nzE, nzΦ, nzJ = E.nzval, Φ.nzval, v.Jh.nzval
        @inbounds for k in v.diagpos
            nzE[k] = 1.0; nzΦ[k] = c
        end
        @inbounds for (g, idx) in enumerate(v.groups)
            jp, ep = v.jpos[g], v.epos[g]
            X = [jp[i, j] == 0 ? 0.0 : c * nzJ[jp[i, j]] for i in eachindex(idx), j in eachindex(idx)]
            Eb, Pb = _phi1exp(X)
            for j in eachindex(idx), i in eachindex(idx)
                nzE[ep[i, j]] = Eb[i, j]; nzΦ[ep[i, j]] = c * Pb[i, j]
            end
        end
        (E, Φ)
    end
end

"""
    frozen_hflow!(z, m, C, F, flow, c, zsrc, w)

Exact flow of length c of Bokhove's partition (horizontal part):
  flow = :a (A_c): a ← a + (J_h Φ_c b̂)_a + c (F_a + sponge(z)), b ← (e^{cD₀} z)_b, with b̂ the momenta of
      zsrc + w (J_v zsrc) (w = βh: the predictor of the implicit vertical force, as in `frozen_hthermo!`);
  flow = :b (B_c): b ← b + (J_h Φ_c ã)_b + c F_b, a ← (e^{cD₀} z)_a, ã the thermodynamic rows of zsrc
      (the extrapolated state of the divergence damping).
"""
function frozen_hflow!(z, m::DGPressureModel, C, F, flow::Symbol, c, zsrc, w = 0.0)
    v = m.vert
    v.valid && C === v.C || error("frozen_hflow!: frozen_stage_begin! with these coefficients first")
    E, Φ = _dgp_flowmats(m, c)
    tha = flow === :a ? false : flow === :b ? true : error("frozen_hflow!: flow :a or :b")
    # source: the b rows (flow A) or the a rows (flow B) of zsrc
    if tha
        @inbounds for k in eachindex(z); v.zs[k] = _dgp_isthermo(m, k) ? zsrc[k] : 0.0; end
    else
        if w != 0
            mul!(v.zv, v.Jv, zsrc)
            @inbounds for k in eachindex(z); v.zs[k] = _dgp_isthermo(m, k) ? 0.0 : zsrc[k] + w * v.zv[k]; end
        else
            @inbounds for k in eachindex(z); v.zs[k] = _dgp_isthermo(m, k) ? 0.0 : zsrc[k]; end
        end
    end
    mul!(v.zp, Φ, v.zs)
    mul!(v.zv, v.Jh, v.zp)                       # coupling increment (a rows for A, b rows for B)
    mul!(v.zp, E, z)                             # e^{cD₀} z
    if !tha
        fill!(v.zs, 0.0); sponge!(v.zs, m.base, z)
    end
    @inbounds for k in eachindex(z)
        if _dgp_isthermo(m, k)
            z[k] = tha ? v.zp[k] : z[k] + v.zv[k] + c * F[k]
        else
            z[k] = tha ? z[k] + v.zv[k] + c * F[k] : v.zp[k] + c * v.zs[k]
        end
    end
    return z
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
    m.vsolver === :hdg && return get!(() -> _hdg_factor(m, ct, cw), v.factors, key)
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

frozen_hthermo_reeval(m::DGPressureModel) = true   # LMARS: the ξ-face mass flux depends on [p]

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
    m.hpredict || return frozen_fast!(out, m, C, z, :hmomentum, F)    # D_M^x M⁰ explicit (docs/fast_2d, S2)
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

# --------------------------------------------------------------- condensed column solve (vsolver = :hdg)
# The column system (I - Δ J_v) Z = R of the standard formulation (LMARS in ρu), solved as in hybridised DG: the
# interface momentum M̂ = s m_f(Z) of the η-faces is introduced as an auxiliary unknown with the exact constraint.
# With it the element rows only contain the unknowns of their element and the M̂ of its two faces (J_v of the
# hybrid model `m.aux`, element rows: J_ee, J_eM), and m_f(Z) = G Z (the constraint rows of the hybrid model). Per node column and element K
#   E_K Z_K - Δ_K J_eM,K M̂ = R_K,  E_K = I - Δ_K J_ee,K,
#   Z_K = y_K + W_K M̂,  y_K = E_K⁻¹ R_K,  W_K = E_K⁻¹ Δ_K J_eM,K,
#   M̂ = G Z:  (I - Σ_K G_K W_K) M̂ = Σ_K G_K y_K   (tridiagonal over the Nez - 1 η-faces of the column).
# Since the element rows with M̂ = s m_f equal the LMARS operator, the result is that of the sparse LU.
const _TriLU = LinearAlgebra.LU{Float64,LinearAlgebra.Tridiagonal{Float64,Vector{Float64}},Vector{Int}}
struct HDGColumnFactor
    cols::Vector{Tuple{Vector{Vector{Int}},Vector{Int}}}
    E::Vector{Vector{LinearAlgebra.LU{Float64,Matrix{Float64},Vector{Int}}}}
    W::Vector{Vector{Matrix{Float64}}}
    G::Vector{Vector{Matrix{Float64}}}
    S::Vector{_TriLU}                                   # LU of the tridiagonal interface system
    yk::Vector{Float64}
    r::Vector{Float64}
end

function _hdg_factor(m::DGPressureModel, ct, cw)
    Jx = m.aux.vert.Jv
    nqp = m.base.nq + m.base.mesh.np^2 * m.base.mesh.Ne
    # owners: element unknown k -> (column, element, local index), η-face M̂ -> (column, face)
    oc = zeros(Int, nqp); oK = zeros(Int, nqp); ot = zeros(Int, nqp)
    fc = zeros(Int, size(Jx, 1)); ff = zeros(Int, size(Jx, 1))
    for (c, (idx, F)) in enumerate(m.cols)
        for (K, id) in enumerate(idx), (t, k) in enumerate(id); oc[k] = c; oK[k] = K; ot[k] = t; end
        for (f, k) in enumerate(F); fc[k] = c; ff[k] = f; end
    end
    A = [[zeros(length(id), length(id)) for id in idx] for (idx, F) in m.cols]       # J_ee
    B = [[zeros(length(id), length(F)) for id in idx] for (idx, F) in m.cols]        # J_eM
    G = [[zeros(length(F), length(id)) for id in idx] for (idx, F) in m.cols]        # M̂ rows, element columns
    dF = [zeros(length(F)) for (idx, F) in m.cols]                                    # minus the M̂ diagonal
    rv, nz = rowvals(Jx), nonzeros(Jx)
    @inbounds for j in 1:size(Jx, 2), q in nzrange(Jx, j)
        r, v = rv[q], nz[q]
        v == 0 && continue
        if r <= nqp
            c, K, t = oc[r], oK[r], ot[r]
            if j <= nqp
                (oc[j] == c && oK[j] == K) || error("vsolver = :hdg: element rows couple to another element")
                A[c][K][t, ot[j]] += v
            else
                fc[j] == c || error("vsolver = :hdg: element row couples to a face of another column")
                B[c][K][t, ff[j]] += v
            end
        elseif fc[r] > 0
            c, f = fc[r], ff[r]
            if j <= nqp
                oc[j] == c || error("vsolver = :hdg: constraint couples to another column")
                G[c][oK[j]][f, ot[j]] += v
            elseif j == r
                dF[c][f] = -v
            else
                error("vsolver = :hdg: constraint couples two faces")
            end
        end
    end
    Es = Vector{LinearAlgebra.LU{Float64,Matrix{Float64},Vector{Int}}}[]; Ws = Vector{Matrix{Float64}}[]; Ss = _TriLU[]
    nmax = 0
    for (c, (idx, F)) in enumerate(m.cols)
        nf = length(F)
        S = Matrix{Float64}(I, max(nf, 1), max(nf, 1))
        EK = LinearAlgebra.LU{Float64,Matrix{Float64},Vector{Int}}[]; WK = Matrix{Float64}[]
        for (K, id) in enumerate(idx)
            Δ = [_dgp_isthermo(m, r) ? ct : cw for r in id]
            Ak = A[c][K]; Ak .*= -Δ; for t in axes(Ak, 1); Ak[t, t] += 1; end
            E = lu!(Ak)
            W = B[c][K]; W .*= Δ; ldiv!(E, W)
            G[c][K] ./= dF[c]
            nf > 0 && (S .-= G[c][K] * W)
            push!(EK, E); push!(WK, W)
            nmax += length(id)
        end
        push!(Es, EK); push!(Ws, WK); push!(Ss, lu!(LinearAlgebra.Tridiagonal(S)))
    end
    nyk = maximum(cl -> sum(length, cl[1]), m.cols)
    return HDGColumnFactor(m.cols, Es, Ws, G, Ss, zeros(nyk), zeros(max(1, maximum(cl -> length(cl[2]), m.cols))))
end

function LinearAlgebra.ldiv!(y::AbstractVector, F::HDGColumnFactor, R::AbstractVector)
    @inbounds for (c, (idx, Fc)) in enumerate(F.cols)
        nf = length(Fc); r = view(F.r, 1:nf); fill!(r, 0.0)
        o = 0
        for (K, id) in enumerate(idx)
            yK = view(F.yk, o+1:o+length(id))
            for (t, k) in enumerate(id); yK[t] = R[k]; end
            ldiv!(F.E[c][K], yK)
            nf > 0 && mul!(r, F.G[c][K], yK, 1.0, 1.0)
            o += length(id)
        end
        nf > 0 && ldiv!(F.S[c], r)
        o = 0
        for (K, id) in enumerate(idx)
            yK = view(F.yk, o+1:o+length(id))
            nf > 0 && mul!(yK, F.W[c][K], r, 1.0, 1.0)
            for (t, k) in enumerate(id); y[k] = yK[t]; end
            o += length(id)
        end
    end
    return y
end
