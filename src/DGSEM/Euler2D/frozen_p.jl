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

using SparseArrays: sparse, SparseMatrixCSC, spzeros, findnz, nzrange, rowvals, nonzeros, nnz
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
    Jh0::SparseMatrixCSC{Float64,Int}                   # vsplit = :contravariant_momentum: the ξ/η operators as assembled
    Jv0::SparseMatrixCSC{Float64,Int}
    P::SparseMatrixCSC{Float64,Int}                     # nodal transform (m_x, m_z) -> (Ũ, W̃) = (Jξ·m, Jη·m), identity elsewhere
    Pinv::SparseMatrixCSC{Float64,Int}
    mfactors::Dict{Float64,Any}                         # LU of I - w J_v^{bb} (momentum block) per w, this stage
    cmap::SparseMatrixCSC{Float64,Int}                  # :contravariant_momentum: J_v0.nzval -> J_v.nzval (fixed linear map)
    hA::SparseMatrixCSC{Float64,Int}                    #   J_h.nzval = hA J_h0.nzval + hB J_v0.nzval
    hB::SparseMatrixCSC{Float64,Int}
    mmat::SparseMatrixCSC{Float64,Int}                  # I - w J_v^{bb} (pattern fixed), positions of J_v^{bb} in J_v.nzval
    mpos::Vector{Int}
    mlus::Dict{Float64,Any}                             # its LU objects across stages (symbolic reuse)
    isth::BitVector                                     # thermodynamic row (ρ, Θ, p, ρq_v, ρq_c)?
    Jht::SparseMatrixCSC{Float64,Int}                   # J_h restricted to the thermodynamic rows, J_h and J_v to the momentum rows
    Jhm::SparseMatrixCSC{Float64,Int}                   # (same size, the other rows empty; refilled per stage from the positions)
    Jvm::SparseMatrixCSC{Float64,Int}
    pht::Vector{Int}
    phm::Vector{Int}
    pvm::Vector{Int}
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
reads m_x at the face nodes). `:contravariant_momentum`: in the fast solver the momenta are replaced node by node by the
contravariant ones, (Ũ, W̃) = (Jξ·m, Jη·m) = T m; the vertical part is the η part of the transformed operator restricted
to the thermodynamic rows and columns and W̃ (all terms with Ũ are horizontal), J_v = P⁻¹ Π(P J_v P⁻¹) P,
J_h = J_h + J_v,old - J_v (P = T on the momenta); the column contains ρ, Θ, p and W̃. All are identical to `:full` on a
flat grid.

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
    moist::Bool                                         # moist air: state [q; p; ρq_v; ρq_c], Θ = ρθ_ρ
    mgas::MoistGas
    micro::Symbol                                       # phase change: :fast (every small step), :slow (stages), :none
    τ::Float64                                          # relaxation time of the phase change
    nqp::Int                                            # unknowns without M̂
end
function DGPressureModel(base::Euler2DModel; pvar::Symbol = :p, lmars::Symbol = :velocity, vsplit::Symbol = :full,
                         vsolver::Symbol = :lu, hpredict::Bool = true, moist::Bool = false, mgas::MoistGas = MoistGas(),
                         micro::Symbol = :fast, τfac::Real = 0.5, cs::Real = 350.0, τ = nothing)
    vsolver in (:lu, :hdg) || error("DGPressureModel: vsolver = :lu or :hdg")
    vsolver === :hdg && lmars !== :momentum && error("DGPressureModel: vsolver = :hdg needs lmars = :momentum")
    vsplit in (:full, :contravariant, :contravariant_volume, :contravariant_momentum) ||
        error("DGPressureModel: vsplit = :full, :contravariant, :contravariant_volume or :contravariant_momentum")
    lmars in (:velocity, :momentum) || error("DGPressureModel: lmars = :velocity or :momentum")
    pvar === :p || error("DGPressureModel: pvar = :p (π not yet)")
    micro in (:fast, :slow, :none) || error("DGPressureModel: micro = :fast, :slow or :none")
    moist && mgas.thvar !== :thetarho && error("DGPressureModel: moist with mgas.thvar = :thetarho (Θ = ρθ_ρ) only")
    M = base.mesh
    # relaxation time: given, or τfac times the explicit horizontal acoustic step 0.85·0.63 Δx_e/((2p+1) c_s)
    τv = τ !== nothing ? Float64(τ) : τfac * 0.85 * 0.63 * (M.Lx / M.Nex) / ((2M.p + 1) * cs)
    ph = (moist, mgas, micro, τv)
    aux = vsolver === :hdg ? _dgp_build(base, pvar, :momentum, vsplit, :lu, hpredict, true, nothing, Tuple{Vector{Vector{Int}},Vector{Int}}[], ph) : nothing
    cols = vsolver === :hdg ? _hdg_columns(base, aux) : Tuple{Vector{Vector{Int}},Vector{Int}}[]
    return _dgp_build(base, pvar, lmars, vsplit, vsolver, hpredict, false, aux, cols, ph)
end
function _dgp_build(base, pvar, lmars, vsplit, vsolver, hpredict, hybrid, aux, cols, ph)
    moist = ph[1]
    M = base.mesh
    N = M.np^2 * M.Ne
    nqp = base.nq + (moist ? 3 : 1) * N
    mhat = Int[]; nM = 0
    if hybrid
        _foreach_face_dir(M) do fi, dir, kind, args...
            push!(mhat, kind === :wall ? 0 : nqp + (nM += 1)); nothing
        end
    end
    nq = nqp + nM
    # coefficients: base, then per node ∂p/∂Θ; moist also ∂p/∂ρ, ∂p/∂ρ_v, ∂p/∂ρ_c, q_v, q_c and per face the upwind side
    nc = frozen_ncoef(base) + (moist ? 6N + _nface(M) : N)
    return DGPressureModel(base, pvar, nq, zeros(nc), zeros(nq),
                           DGVertical(zeros(nc), spzeros(nq, nq), spzeros(nq, nq), false, Int[], Int[],
                                      Dict{Tuple{Float64,Float64},Any}(), Dict{Tuple{Float64,Float64},Any}(), zeros(nq), zeros(nq),
                                      zeros(nq), Vector{Int}[], Matrix{Int}[], Matrix{Int}[], Int[], spzeros(nq, nq),
                                      Dict{Float64,Tuple{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Float64,Int}}}(),
                                      spzeros(nq, nq), spzeros(nq, nq), _dgp_contra(base, nq, vsplit)...,
                                      Dict{Float64,Any}(), spzeros(0, 0), spzeros(0, 0), spzeros(0, 0), spzeros(0, 0), Int[],
                                      Dict{Float64,Any}(), _dgp_isth(base, nq, nqp), spzeros(0, 0), spzeros(0, 0), spzeros(0, 0),
                                      Int[], Int[], Int[]),
                           hybrid, lmars, vsplit, mhat, vsolver, hpredict, aux, cols, ph..., nqp)
end

"""(P, P⁻¹) of vsplit = :contravariant_momentum: per node (m_x, m_z) -> (Ũ, W̃) = (Jξ·m, Jη·m) on the momentum rows
(Ũ at the m_x index, W̃ at the m_z index), identity on all other unknowns (empty matrices for the other splits)."""
function _dgp_contra(base, nq, vsplit)
    vsplit === :contravariant_momentum || return spzeros(nq, nq), spzeros(nq, nq)
    M = base.mesh; np = M.np
    I_ = collect(1:nq); J_ = collect(1:nq); P_ = ones(nq); Q_ = ones(nq)
    Ii, Jj, Pv, Qv = Int[], Int[], Float64[], Float64[]
    keep = trues(nq)
    for e in 1:M.Ne, j in 1:np, i in 1:np
        kx, kz = node_iq(np, e, 2, i, j), node_iq(np, e, 3, i, j)
        keep[kx] = keep[kz] = false
        T = [M.Jξx[i, j, e] M.Jξz[i, j, e]; M.Jηx[i, j, e] M.Jηz[i, j, e]]; Ti = inv(T)
        for (a, ka) in ((1, kx), (2, kz)), (b, kb) in ((1, kx), (2, kz))
            push!(Ii, ka); push!(Jj, kb); push!(Pv, T[a, b]); push!(Qv, Ti[a, b])
        end
    end
    P = sparse(vcat(I_[keep], Ii), vcat(J_[keep], Jj), vcat(P_[keep], Pv), nq, nq)
    Pinv = sparse(vcat(I_[keep], Ii), vcat(J_[keep], Jj), vcat(Q_[keep], Qv), nq, nq)
    return P, Pinv
end

"""Position of entry (i, j) in A.nzval (0: not in the pattern)."""
@inline function _nzpos(A::SparseMatrixCSC, i, j)
    r = A.colptr[j]:A.colptr[j+1]-1
    t = searchsortedfirst(view(A.rowval, r), i)
    return t <= length(r) && A.rowval[r[t]] == i ? r[t] : 0
end

"""Pattern (values zero) of the matrix with the entries (I, J), and the map of the triplets (I[k], J[k], T[k], V[k]) as a
sparse matrix: out.nzval = map * in.nzval with out_{I[k] J[k]} += V[k] in_{T[k]}."""
function _dgp_linmap(I_, J_, T_, V_, n, nin)
    A = sparse(I_, J_, ones(length(I_)), n, n); fill!(A.nzval, 0.0)
    return A, sparse([_nzpos(A, I_[k], J_[k]) for k in eachindex(I_)], T_, V_, nnz(A), nin)
end

"""Vertical operator of vsplit = :contravariant_momentum: P⁻¹ Π(P J P⁻¹) P, Π dropping every entry with row or column Ũ.
P is node-local (2 × 2 on the momenta), so with the pattern of J fixed this is a fixed linear map of J.nzval, built at
the first stage (v.cmap; for the main model also J_h = J_h0 + J_v0 - J_v as hA, hB); later stages only multiply."""
function _dgp_contra_vertical!(m::DGPressureModel, v::DGVertical, Jv0, Jh0 = nothing)
    if size(v.cmap, 2) != nnz(Jv0)
        nqb, np2, n = m.base.nq, m.base.mesh.np^2, size(Jv0, 1)
        isU(k) = k <= nqb && mod1(cld(k, np2), NV) == 2
        P, Pi = v.P, v.Pinv; PT = sparse(transpose(P)); PiT = sparse(transpose(Pi))
        rP, vP, rPi, vPi, rPT, vPT, rPiT, vPiT = rowvals(P), nonzeros(P), rowvals(Pi), nonzeros(Pi), rowvals(PT), nonzeros(PT),
                                                 rowvals(PiT), nonzeros(PiT)
        I_, J_, T_, V_ = Int[], Int[], Int[], Float64[]
        rv = rowvals(Jv0)
        for j in 1:n, t in nzrange(Jv0, j)
            i = rv[t]
            for q1 in nzrange(P, i), q2 in nzrange(PiT, j)          # (P J P⁻¹)_{rs} ∋ P_ri J_ij P⁻¹_js
                r, s = rP[q1], rPiT[q2]
                (isU(r) || isU(s)) && continue
                c = vP[q1] * vPiT[q2]
                for q3 in nzrange(Pi, r), q4 in nzrange(PT, s)      # (P⁻¹ Jt P)_{r2 s2} ∋ P⁻¹_{r2 r} Jt_rs P_{s s2}
                    push!(I_, rPi[q3]); push!(J_, rPT[q4]); push!(T_, t); push!(V_, vPi[q3] * c * vPT[q4])
                end
            end
        end
        v.Jv, v.cmap = _dgp_linmap(I_, J_, T_, V_, n, nnz(Jv0))
        if Jh0 !== nothing                                          # J_h = J_h0 + J_v0 - J_v on the union pattern
            Ih, Jh_, _ = findnz(Jh0); I0, J0, _ = findnz(Jv0); I1, J1, _ = findnz(v.Jv)
            v.Jh = sparse(vcat(Ih, I0, I1), vcat(Jh_, J0, J1), ones(length(Ih) + length(I0) + length(I1)), n, n)
            fill!(v.Jh.nzval, 0.0)
            emb(Is, Js) = sparse([_nzpos(v.Jh, Is[k], Js[k]) for k in eachindex(Is)], 1:length(Is), 1.0, nnz(v.Jh), length(Is))
            v.hA = emb(Ih, Jh_)
            v.hB = emb(I0, J0) - emb(I1, J1) * v.cmap
        end
    end
    mul!(v.Jv.nzval, v.cmap, Jv0.nzval)
    if Jh0 !== nothing
        mul!(v.Jh.nzval, v.hA, Jh0.nzval); mul!(v.Jh.nzval, v.hB, Jv0.nzval, 1.0, 1.0)
    end
    return v.Jv
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
        N = np^2 * M.Ne; nt = (mi.nqp - base.nq) ÷ N          # p (and ρq_v, ρq_c)
        idx = [vcat([node_iq(np, e, r, i, j) for r in 1:4 for j in 1:np],
                    [base.nq + (t - 1) * N + _pnode(np, e, i, j) for t in 1:nt for j in 1:np]) for e in es]
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
    nb = frozen_ncoef(b); N = np^2 * M.Ne
    if !m.moist
        frozen_data!(C, S, b, q)               # writes the base coefficients; S = 0 on the p rows
        @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
            Θ = q[node_iq(np, e, 4, i, j)]
            C[nb + _pnode(np, e, i, j)] = b.gas.γ * pressure(b.gas, Θ) / Θ
        end
        m.lmars === :momentum && _lmars_momentum!(C, m, q)
        return C, S
    end
    # moist: LMARS coefficients with the pressure of the moist equation of state, ∂p/∂(Θ, ρ, ρq_v, ρq_c), q_v, q_c
    mg = m.mgas
    pf(e, i, j) = _dgp_peos(m, q, e, i, j)
    frozen_data!(C, S, b, q; pfun = pf)
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
        pn = _pnode(np, e, i, j)
        ρ, Θ = q[node_iq(np, e, 1, i, j)], q[node_iq(np, e, 4, i, j)]
        V, Cc = q[b.nq + N + pn], q[b.nq + 2N + pn]
        a, bb, c, d = exner_derivatives(mg, ρ, Θ, V, Cc; f = _moist_p)
        C[nb + pn] = bb; C[nb + N + pn] = a; C[nb + 2N + pn] = c; C[nb + 3N + pn] = d
        C[nb + 4N + pn] = V / ρ; C[nb + 5N + pn] = Cc / ρ
    end
    # upwind side of every face (that of the θ_ρ of the base coefficients, C₄ = θ_up C₁)
    _foreach_face_dir(M) do fi, dir, kind, iL, jL, eL, iR, jR, eR, ñx, ñz, we
        o = _fc(M, fi)
        if kind === :wall
            C[nb + 6N + fi] = 1.0
        else
            θup = C[o+4] / C[o+1]
            C[nb + 6N + fi] = abs(θup - C[o+9]) <= abs(θup - C[o+10]) ? 1.0 : 0.0
        end
        nothing
    end
    m.lmars === :momentum && _lmars_momentum!(C, m, q)
    m.micro === :slow && _dgp_phase!(S, m, q)
    return C, S
end

"""Pressure of the equation of state at node (e, i, j) of q (moist: p(ρ, Θ, ρq_v, ρq_c), dry: p(Θ))."""
@inline function _dgp_peos(m::DGPressureModel, q, e, i, j)
    b = m.base; np = b.mesh.np
    Θ = q[node_iq(np, e, 4, i, j)]
    m.moist || return pressure(b.gas, Θ)
    N = np^2 * b.mesh.Ne; pn = _pnode(np, e, i, j)
    return _moist_p(m.mgas, q[node_iq(np, e, 1, i, j)], Θ, q[b.nq + N + pn], q[b.nq + 2N + pn])
end

"""Phase change (moist): S_c = -φ_FB(ρ_vs - ρ_v, ρ_c)/τ at every node of z, explicit: S_c to ρq_c, -S_c to ρq_v, the
latent heating (∂Θ/∂ρ_c) S_c to Θ = ρθ_ρ and (∂p/∂ρ_c) S_c to p (Bryan & Fritsch; factors at the live state)."""
function _dgp_phase!(out, m::DGPressureModel, z)
    b = m.base; np = b.mesh.np; mg = m.mgas
    N = np^2 * b.mesh.Ne
    @inbounds for e in 1:b.mesh.Ne, j in 1:np, i in 1:np
        pn = _pnode(np, e, i, j)
        kρ, kΘ, kV, kC = node_iq(np, e, 1, i, j), node_iq(np, e, 4, i, j), b.nq + N + pn, b.nq + 2N + pn
        ρ, Θ, ρv, ρc = z[kρ], z[kΘ], z[kV], z[kC]
        φ, T = fischer_burmeister(mg, ρ, Θ, ρv, ρc)
        Sc = -φ / m.τ
        p = (mg.Rd * (ρ - ρv - ρc) + mg.Rv * ρv) * T
        out[kC] += Sc
        out[kV] -= Sc
        out[kΘ] += theta_phase_factor(mg, ρ, Θ, T, p, ρv, ρc) * Sc
        out[b.nq + pn] += pressure_phase_factor(mg, ρ, T, ρv, ρc) * Sc
    end
    return out
end

"""Moist state [q; p; ρq_v; ρq_c] of a base state q (ρ, m_x, m_z, Θ = ρθ_ρ) and the total water density Q(x, z), split
into vapour and condensate of phase equilibrium (initial states only; the model has no saturation adjustment)."""
function moist_dg_state(m::DGPressureModel, q, Q)
    m.moist || error("moist_dg_state: model with moist = true")
    b = m.base; M = b.mesh; np = M.np; N = np^2 * M.Ne
    x = vcat(q[1:b.nq], zeros(m.nq - b.nq))
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
        pn = _pnode(np, e, i, j)
        T, ρv, ρc, _ = moist_state(m.mgas, q[node_iq(np, e, 1, i, j)], q[node_iq(np, e, 4, i, j)], Q(M.X[i, j, e], M.Z[i, j, e]))
        x[b.nq + N + pn] = ρv; x[b.nq + 2N + pn] = ρc
    end
    frozen_step_end!(m, x)
    return x
end

"""Face coefficients of LMARS in ρu: m_f = {m_n} - [p]/(2c), f_Θ = θ_up m_f, p* = {p} - c [m_n]/2 (interior faces;
the wall p + c m_n is the same in both forms), c = mean of the sound speeds of the two sides, θ_up from the base."""
function _lmars_momentum!(C, m::DGPressureModel, q)
    b = m.base; M = b.mesh; np = M.np; γ = b.gas.γ
    _foreach_face_dir(M) do fi, dir, kind, iL, jL, eL, iR, jR, eR, ñx, ñz, we
        kind === :wall && return nothing
        o = _fc(M, fi)
        sp(e, i, j) = sqrt(γ * _dgp_peos(m, q, e, i, j) / q[node_iq(np, e, 1, i, j)])
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

# p row of the frozen pressure equation p_t = a ρ_t + b Θ_t + c (ρq_v)_t + d (ρq_c)_t: every emission to a ρ, Θ, ρq_v
# or ρq_c row is passed on to the p row of the node with its coefficient (dry: b Θ_t only).
struct _DGPRow{E}
    emit::E
    C::Vector{Float64}
    nqb::Int
    np2::Int
    N::Int
    nb::Int
    moist::Bool
end
@inline function (w::_DGPRow)(r, c, v)
    w.emit(r, c, v)
    if r <= w.nqb
        var = mod1(cld(r, w.np2), NV)
        (var == 4 || (var == 1 && w.moist)) || return nothing
        pn = (cld(r, NV * w.np2) - 1) * w.np2 + mod1(r, w.np2)
        @inbounds w.emit(w.nqb + pn, c, (var == 4 ? w.C[w.nb + pn] : w.C[w.nb + w.N + pn]) * v)
    elseif w.moist && r > w.nqb + w.N && r <= w.nqb + 3w.N
        t = cld(r - w.nqb, w.N)                                   # 2: ρq_v, 3: ρq_c
        pn = r - w.nqb - (t - 1) * w.N
        @inbounds w.emit(w.nqb + pn, c, w.C[w.nb + t * w.N + pn] * v)
    end
    return nothing
end

"""
    _dgp_terms!(emit, m, C, dirs)

All terms of the linear fast operator in the directions `dirs` (1 = ξ, 2 = η; volume pairs and
faces) as emit(row, column, coefficient): out[row] += coefficient z[column].
"""
function _dgp_terms!(emit0::E, m::DGPressureModel, C, dirs; vol::Bool = true, fac::Bool = true) where {E}
    b = m.base; M = b.mesh
    np, Ne, D, g = M.np, M.Ne, M.D, b.gas.g
    nb = frozen_ncoef(b); nqb = b.nq; N = np^2 * Ne
    emit = _DGPRow(emit0, C, nqb, np^2, N, nb, m.moist)
    iq(e, r, i, j) = node_iq(np, e, r, i, j)
    ipr(e, i, j) = nqb + _pnode(np, e, i, j)
    iV(e, i, j) = nqb + N + _pnode(np, e, i, j)
    iC(e, i, j) = nqb + 2N + _pnode(np, e, i, j)
    qv(e, i, j) = C[nb + 4N + _pnode(np, e, i, j)]
    qc(e, i, j) = C[nb + 5N + _pnode(np, e, i, j)]
    # Θ row (the p row follows through `emit`)
    @inline emitΘ(e, i, j, col, v) = emit(iq(e, 4, i, j), col, v)
    # tracer rows ρq_v, ρq_c of node (e, i, j) with the mixing ratios (qa, qb): the mass flux term v times q
    @inline emitQ(e, i, j, col, v, a, c) = (emit(iV(e, i, j), col, a * v); emit(iC(e, i, j), col, c * v))
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
            if m.moist                                   # ρq: the mass flux with the pair mean of q
                qa, qcm = 0.5 * (qv(e, i, j) + qv(e, ib, jb)), 0.5 * (qc(e, i, j) + qc(e, ib, jb))
                emitQ(e, i, j, mxa, f * C[o+1] * ax, qa, qcm); emitQ(e, i, j, mza, f * C[o+1] * az, qa, qcm)
                emitQ(e, i, j, mxb, f * C[o+2] * ax, qa, qcm); emitQ(e, i, j, mzb, f * C[o+2] * az, qa, qcm)
            end
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
        mnQ(e, i, j, mx, mz, v, a, c) = (emitQ(e, i, j, mx, v * n1, a, c); emitQ(e, i, j, mz, v * n2, a, c))
        if m.moist && kind !== :wall
            up = C[nb + 6N + fi] > 0.5                   # upwind side of the face (left)
            qvu = up ? qv(eL, iL, jL) : qv(eR, iR, jR); qcu = up ? qc(eL, iL, jL) : qc(eR, iR, jR)
        else
            qvu = qcu = 0.0
        end
        if kind === :wall
            c = σL * s
            mn(ρL, mxL, mzL, c)
            m.moist && mnQ(eL, iL, jL, mxL, mzL, c, qv(eL, iL, jL), qc(eL, iL, jL))
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
            if m.moist
                emitQ(eL, iL, jL, kM, -cL * is, qvu, qcu); mnQ(eL, iL, jL, mxL, mzL, cL, qv(eL, iL, jL), qc(eL, iL, jL))
                emitQ(eR, iR, jR, kM, cR * is, qvu, qcu); mnQ(eR, iR, jR, mxR, mzR, -cR, qv(eR, iR, jR), qc(eR, iR, jR))
            end
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
            if m.moist                                   # -cL (q_up m_f - q_L m_nL), +cR (q_up m_f - q_R m_nR)
                for (e_, i_, j_, sg) in ((eL, iL, jL, -cL), (eR, iR, jR, cR))
                    emitQ(e_, i_, j_, mxL, sg * C[o+1] * n1, qvu, qcu); emitQ(e_, i_, j_, mzL, sg * C[o+1] * n2, qvu, qcu)
                    emitQ(e_, i_, j_, mxR, sg * C[o+2] * n1, qvu, qcu); emitQ(e_, i_, j_, mzR, sg * C[o+2] * n2, qvu, qcu)
                    emitQ(e_, i_, j_, pR, -sg * C[o+3], qvu, qcu); emitQ(e_, i_, j_, pL, sg * C[o+3], qvu, qcu)
                end
                mnQ(eL, iL, jL, mxL, mzL, cL, qv(eL, iL, jL), qc(eL, iL, jL))
                mnQ(eR, iR, jR, mxR, mzR, -cR, qv(eR, iR, jR), qc(eR, iR, jR))
            end
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
    if m.vsplit === :full || m.vsplit === :contravariant_momentum      # (the latter is split in frozen_stage_begin!)
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

@inline _dgp_isthermo(m::DGPressureModel, k) = @inbounds m.vert.isth[k]
"""Thermodynamic rows of the state (ρ, Θ of the base, p, ρq_v, ρq_c; not m_x, m_z, M̂)."""
_dgp_isth(base, nq, nqp) = BitVector([k > base.nq ? k <= nqp : mod1(cld(k, base.mesh.np^2), NV) in (1, 4) for k in 1:nq])

"""Rows of A selected by `rows` (same size, other rows empty) and the positions of its entries in A.nzval."""
function _dgp_rows(A::SparseMatrixCSC, rows::BitVector)
    Is, Js, _ = findnz(A)
    sel = [t for t in eachindex(Is) if rows[Is[t]]]
    B = sparse(Is[sel], Js[sel], ones(length(sel)), size(A)...)
    nnz(B) == length(sel) || error("_dgp_rows: pattern")
    B.nzval .= A.nzval[sel]
    return B, sel
end
"""Refill the row blocks J_h^T, J_h^M, J_v^M of the stage (patterns once)."""
function _dgp_rowblocks!(v::DGVertical)
    if length(v.pht) + length(v.phm) != nnz(v.Jh) || size(v.Jvm, 1) != size(v.Jv, 1)
        v.Jht, v.pht = _dgp_rows(v.Jh, v.isth)
        v.Jhm, v.phm = _dgp_rows(v.Jh, .!v.isth)
        v.Jvm, v.pvm = _dgp_rows(v.Jv, .!v.isth)
    else
        nh, nv = v.Jh.nzval, v.Jv.nzval
        @inbounds for (k, t) in enumerate(v.pht); v.Jht.nzval[k] = nh[t]; end
        @inbounds for (k, t) in enumerate(v.phm); v.Jhm.nzval[k] = nh[t]; end
        @inbounds for (k, t) in enumerate(v.pvm); v.Jvm.nzval[k] = nv[t]; end
    end
    return nothing
end

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
        elseif dirs == (1,)                         # only the rows that are kept
            mul!(out, part === :hthermo ? v.Jht : part === :hmomentum ? v.Jhm : v.Jh, z)
        else
            mul!(out, v.Jv, z)
        end
    else
        fill!(out, 0.0)
        _dgp_part_terms!(_DGApply(out, z), m, C, which)
    end
    part === :vertical && return out
    keepth = part in (:all, :thermo, :hthermo)
    m.moist && m.micro === :fast && keepth && _dgp_phase!(out, m, z)     # explicit phase change (before the row mask)
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

"""p ← p of the equation of state after a MIS step (moist: p(ρ, Θ, ρq_v, ρq_c), no saturation adjustment)."""
function frozen_step_end!(m::DGPressureModel, q)
    b = m.base; M = b.mesh; np = M.np
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
        q[ip_dg(m, e, i, j)] = _dgp_peos(m, q, e, i, j)
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
    if m.vsplit === :contravariant_momentum
        v.Jh0 = _dgp_matrix!(v.Jh0, v.hmap, m, C, :h)
        v.Jv0 = _dgp_matrix!(v.Jv0, v.vmap, m, C, :v)
        _dgp_contra_vertical!(m, v, v.Jv0, v.Jh0)                 # J_v, J_h
    else
        v.Jh = _dgp_matrix!(v.Jh, v.hmap, m, C, :h)
        v.Jv = _dgp_matrix!(v.Jv, v.vmap, m, C, :v)
    end
    if m.vsolver === :hdg                              # η part written with M̂, same C
        a = m.aux.vert
        if m.vsplit === :contravariant_momentum
            a.Jv0 = _dgp_matrix!(a.Jv0, a.vmap, m.aux, C, :v)
            _dgp_contra_vertical!(m, a, a.Jv0)
        else
            a.Jv = _dgp_matrix!(a.Jv, a.vmap, m.aux, C, :v)
        end
    end
    _dgp_rowblocks!(v)
    v.valid = true
    empty!(v.factors); empty!(v.mfactors)
    _dgp_dissipation!(m)                              # blocks of D₀ (once)
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
        v.lus[key] = haskey(v.lus, key) ? (try lu!(v.lus[key], A) catch; lu(A) end) : lu(A)   # same pattern: symbolic reuse
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
#   :hthermo of a⁺ (`frozen_hthermo!`, w = βh): at b̂ = b + w (I - w J_v^{bb})⁻¹ (J_v z)_b, the momenta of the column
#       solve with the thermodynamic rows frozen at the current state (J_v^{bb} the momentum block, the vertical
#       LMARS term [m_n] of p*, local to the face node pairs). The remaining lag O(h²) keeps the order, as the FV
#       metric predictor. Well balanced: at a discrete rest state (J z + F = 0, b = 0) the momenta after the
#       horizontal momentum step are b = -w (J_v z)_b|_{b=0}, so b̂ = 0 and the column solve returns the rest state
#       (the explicit predictor b + w (J_v z)_b leaves w² J_v^{bb} (J_v z)_b, and the step drifts on terrain),
#   :hmomentum of b⁺ (`frozen_hmomentum!`, hpredict): at the midpoint predictor b + h/2 (J z + F)_b with the whole
#       force (the ξ-face dissipation b←b is then a second-order midpoint step instead of explicit Euler; zero at rest).
# Both act on bⁿ: the explicit vertical force (1-β)h L₁a is added to b after the horizontal momentum step
# (`frozen_hmomentum_selfcoupled`), since the ξ-face p* (and with :contravariant_momentum the η-face terms on Ũ)
# depend on b. With these the :cs/:csfb step keeps a discrete rest state (N = 0, e.g. the isothermal atmosphere with
# the gravity mean :waruszewski or the constant-θ one with :artiano) on terrain to round-off.
# Forcing and sponge are evaluated at z.

frozen_hthermo_reeval(m::DGPressureModel) = true   # LMARS: the ξ-face mass flux depends on [p]
frozen_hmomentum_selfcoupled(m::DGPressureModel) = true   # LMARS: the ξ-face p* depends on [m_n]

"""Momentum rows of y ← z + c (src)_b (thermodynamic rows of z unchanged)."""
function _dgp_predict!(y, m::DGPressureModel, z, c, src)
    @inbounds for k in eachindex(y)
        y[k] = _dgp_isthermo(m, k) ? z[k] : z[k] + c * src[k]
    end
    return y
end

"""LU of I - w J_v^{bb}: the momentum block of J_v (momentum rows and columns; on the η-faces and walls the LMARS
term [m_n] of p*, so block diagonal over the face node pairs), identity on the thermodynamic rows."""
function _dgp_mfactor(m::DGPressureModel, w)
    v = m.vert
    get!(v.mfactors, Float64(w)) do
        if size(v.mmat, 1) != m.nq                      # pattern of I - w J_bb and the positions of J_bb in J_v.nzval (once)
            Is, Js, _ = findnz(v.Jv)
            keep = [!_dgp_isthermo(m, Is[t]) && !_dgp_isthermo(m, Js[t]) for t in eachindex(Is)]
            v.mmat = sparse(vcat(Is[keep], 1:m.nq), vcat(Js[keep], 1:m.nq), ones(count(keep) + m.nq), m.nq, m.nq)
            v.mpos = zeros(Int, nnz(v.mmat))
            rv = rowvals(v.mmat)
            for j in 1:m.nq, q in nzrange(v.mmat, j)
                i = rv[q]
                (_dgp_isthermo(m, i) || _dgp_isthermo(m, j)) || (v.mpos[q] = _nzpos(v.Jv, i, j))
            end
        end
        A = v.mmat; nz = nonzeros(A); rv = rowvals(A); jv = nonzeros(v.Jv)
        @inbounds for j in 1:m.nq, q in nzrange(A, j)
            nz[q] = (rv[q] == j ? 1.0 : 0.0) - (v.mpos[q] > 0 ? w * jv[v.mpos[q]] : 0.0)
        end
        haskey(v.mlus, w) ? (try lu!(v.mlus[w], A) catch; lu(A) end) : (v.mlus[w] = lu(A))
    end
end

"""out = (J_v z)_b: the vertical part on the momentum rows, zero on the thermodynamic rows."""
function _dgp_vmom!(out, m::DGPressureModel, C, z)
    v = m.vert
    if v.valid && C === v.C
        mul!(out, v.Jvm, z)
    else
        frozen_fast!(out, m, C, z, :vertical, out)
        @inbounds for k in eachindex(out); v.isth[k] && (out[k] = 0.0); end
    end
    return out
end
frozen_momentum_rows!(out, m::DGPressureModel, y) = (@inbounds for k in eachindex(out); out[k] = m.vert.isth[k] ? 0.0 : y[k]; end; out)

function frozen_hthermo!(out, m::DGPressureModel, C, z, F, w)
    w == 0 && return frozen_fast!(out, m, C, z, :hthermo, F)
    v = m.vert
    _dgp_vmom!(v.zs, m, C, z)                          # (J_v z)_b, zero on the thermodynamic rows
    ldiv!(v.zv, _dgp_mfactor(m, w), v.zs)              # (I - w J_bb)⁻¹ (J_v z)_b: well balanced (see above)
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
    _dgp_vmom!(v.zs, m, C, z)                          # with the vertical force: b⁰ + h/2 (J z + F)_b (= 0 at rest)
    v.zv .+= v.zs
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
#
# Storage: element K of a column touches only its two η-faces f = K-1 (bottom, local 1) and f = K (top, local 2), so
# W_K is n × 2 and G_K is 2 × n, and E_K⁻¹ is kept as a dense inverse (n = 5np dry, 7np moist). All blocks of all
# columns lie in contiguous arrays, and the solve is plain loops (no small BLAS calls): per element one n × n
# product and two n × 2 products, per column one tridiagonal solve (LU with pivoting).
struct HDGColumnFactor
    n::Int                                              # unknowns per element block
    nK::Int                                             # elements per column (Nez)
    idx::Matrix{Int}                                    # (n, nK·ncol): global unknown of each block row
    Einv::Array{Float64,3}                              # (n, n, nK·ncol)
    W::Array{Float64,3}                                 # (n, 2, nK·ncol)
    G::Array{Float64,3}                                 # (2, n, nK·ncol), rows divided by the M̂ diagonal
    S::Vector{_TriLU}                                   # LU of the tridiagonal interface system per column
    yk::Vector{Float64}                                 # work: one column
    r::Vector{Float64}
end

function _hdg_factor(m::DGPressureModel, ct, cw)
    Jx = m.aux.vert.Jv
    nqp = m.nqp
    ncol = length(m.cols); nK = length(m.cols[1][1]); n = length(m.cols[1][1][1])
    idx = zeros(Int, n, nK * ncol)
    oc = zeros(Int, nqp); ob = zeros(Int, nqp); ot = zeros(Int, nqp)     # element unknown -> (column, block, local)
    fc = zeros(Int, size(Jx, 1)); ff = zeros(Int, size(Jx, 1))             # M̂ -> (column, face)
    for (c, (ids, F)) in enumerate(m.cols)
        length(ids) == nK || error("vsolver = :hdg: columns of different length")
        for (K, id) in enumerate(ids)
            length(id) == n || error("vsolver = :hdg: element blocks of different size")
            bk = (c - 1) * nK + K
            for (t, k) in enumerate(id); oc[k] = c; ob[k] = bk; ot[k] = t; idx[t, bk] = k; end
        end
        for (f, k) in enumerate(F); fc[k] = c; ff[k] = f; end
    end
    A = zeros(n, n, nK * ncol)                         # J_ee
    W = zeros(n, 2, nK * ncol)                         # J_eM (then W_K)
    G = zeros(2, n, nK * ncol)                         # constraint rows of the two faces of the element
    dF = zeros(max(nK - 1, 1), ncol)                   # minus the M̂ diagonal
    rv, nz = rowvals(Jx), nonzeros(Jx)
    @inbounds for j in 1:size(Jx, 2), q in nzrange(Jx, j)
        r, v = rv[q], nz[q]
        v == 0 && continue
        if r <= nqp
            c, bk, t = oc[r], ob[r], ot[r]
            K = bk - (c - 1) * nK
            if j <= nqp
                ob[j] == bk || error("vsolver = :hdg: element rows couple to another element")
                A[t, ot[j], bk] += v
            else
                fc[j] == c || error("vsolver = :hdg: element row couples to a face of another column")
                l = ff[j] - K + 2                      # f = K-1 -> 1, f = K -> 2
                1 <= l <= 2 || error("vsolver = :hdg: element row couples to a face not its own")
                W[t, l, bk] += v
            end
        elseif fc[r] > 0
            c, f = fc[r], ff[r]
            if j <= nqp
                oc[j] == c || error("vsolver = :hdg: constraint couples to another column")
                bk = ob[j]; K = bk - (c - 1) * nK
                l = f - K + 2
                1 <= l <= 2 || error("vsolver = :hdg: constraint couples to an element not adjacent")
                G[l, ot[j], bk] += v
            elseif j == r
                dF[f, c] = -v
            else
                error("vsolver = :hdg: constraint couples two faces")
            end
        end
    end
    nf = nK - 1
    dl = zeros(max(nf - 1, 1), ncol); d = ones(max(nf, 1), ncol); du = zeros(max(nf - 1, 1), ncol)
    Δ = zeros(n)
    Einv = zeros(n, n, nK * ncol)
    for c in 1:ncol, K in 1:nK
        bk = (c - 1) * nK + K
        for t in 1:n; Δ[t] = _dgp_isthermo(m, idx[t, bk]) ? ct : cw; end
        Ak = view(A, :, :, bk)
        Ak .*= -Δ
        for t in 1:n; Ak[t, t] += 1; end
        Ei = inv(lu!(Matrix(Ak)))                      # E_K⁻¹ (small, dense)
        Einv[:, :, bk] .= Ei
        Wk = view(W, :, :, bk)
        Wk .= Ei * (Δ .* Wk)                           # W_K = E_K⁻¹ Δ_K J_eM
        Gk = view(G, :, :, bk)
        K > 1 && (Gk[1, :] ./= dF[K-1, c])
        K < nK && (Gk[2, :] ./= dF[K, c])
        # S = I - Σ_K G_K W_K on the faces (K-1, K)
        GW = Gk * Wk
        if K > 1
            d[K-1, c] -= GW[1, 1]
            K < nK && (du[K-1, c] -= GW[1, 2])
        end
        if K < nK
            d[K, c] -= GW[2, 2]
            K > 1 && (dl[K-1, c] -= GW[2, 1])
        end
    end
    S = nf == 0 ? _TriLU[] : [lu!(LinearAlgebra.Tridiagonal(dl[1:nf-1, c], d[1:nf, c], du[1:nf-1, c])) for c in 1:ncol]
    return HDGColumnFactor(n, nK, idx, Einv, W, G, S, zeros(n * nK), zeros(max(nf, 1)))
end

function LinearAlgebra.ldiv!(y::AbstractVector, F::HDGColumnFactor, R::AbstractVector)
    n, nK = F.n, F.nK
    nf = nK - 1
    ncol = size(F.idx, 2) ÷ nK
    yk, r = F.yk, F.r
    @inbounds for c in 1:ncol
        for f in 1:nf; r[f] = 0.0; end
        for K in 1:nK
            bk = (c - 1) * nK + K; o = (K - 1) * n
            for t in 1:n; yk[o+t] = 0.0; end
            for s in 1:n                               # y_K = E_K⁻¹ R_K
                Rs = R[F.idx[s, bk]]
                @simd for t in 1:n
                    yk[o+t] += F.Einv[t, s, bk] * Rs
                end
            end
            g1 = g2 = 0.0                              # G_K y_K
            @simd for t in 1:n
                g1 += F.G[1, t, bk] * yk[o+t]; g2 += F.G[2, t, bk] * yk[o+t]
            end
            K > 1 && (r[K-1] += g1)
            K < nK && (r[K] += g2)
        end
        nf > 0 && ldiv!(F.S[c], view(r, 1:nf))        # S M̂ = r
        for K in 1:nK                                  # Z_K = y_K + W_K M̂
            bk = (c - 1) * nK + K; o = (K - 1) * n
            mb = K > 1 ? r[K-1] : 0.0; mt = K < nK ? r[K] : 0.0
            @simd for t in 1:n
                y[F.idx[t, bk]] = yk[o+t] + F.W[t, 1, bk] * mb + F.W[t, 2, bk] * mt
            end
        end
    end
    return y
end
