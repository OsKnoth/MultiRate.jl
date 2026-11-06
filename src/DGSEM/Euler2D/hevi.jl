# HEVI: column solver for the implicit vertical part

# ---------------------------------------------------------------------------
# HEVI: column solver for the implicit vertical part (cf. CGDycore.jl, DGSEM/JacHDGJNeu.jl)
#
# The vertical part reads the momentum only through U = Ja^η·m and changes it only along Ja^η.
# With the face momentum M̂ (LMARS, tau_scale = 1) as unknown, all vertical SAT terms are
# functions of U_b - M̂ (ñ = outward Ja^η, κ = c/|ñ|, w = horizontal weight):
#   ρ: w (U_b - M̂),  Θ: θ̄ w (U_b - M̂),  m: -w κ ñ (U_b - M̂)    [p_b - P̂ = -κ (U_b - M̂)]
#   face: κ (U_L + U_R - 2 M̂) = a (Θ_R - Θ_L);  walls: M̂ = 0.
# Per node column (ex, i): (I - dt γ M⁻¹A_v) Y = Ys is solved by element-local Schur
# elimination of (ρ, Θ) (dense p+1 LU for U per element column), a tridiagonal system for M̂ at the Nez-1
# interior faces (Thomas), and back substitution; m ← m_s + Ja^η (U - U_s)/|Ja^η|².
# ---------------------------------------------------------------------------
mutable struct HeviFactor
    A::Array{Float64,3}        # (3np, 3np, nblk) local block coefficients
    S::Array{Float64,3}        # (np, np, nblk) local Schur LU for U
    piv::Matrix{Int}           # (np, nblk)
    zb::Matrix{Float64}        # A⁻¹ B_bot (3np, nblk)
    zt::Matrix{Float64}        # A⁻¹ B_top
    cp::Matrix{Float64}        # Thomas: c' (Nez-1, ncol)
    iden::Matrix{Float64}      # Thomas: 1/denominator
    sub::Matrix{Float64}       # sub-diagonal
    y::Matrix{Float64}         # work: A⁻¹ r per block (3np, nblk)
    dp::Vector{Float64}
    Mh::Vector{Float64}  # work (Nez-1)
    c::Float64           # dt γ of the factor
    ys::Vector{Float64}  # momentum back substitution with DG_VADV_FAST (copy of Ys, M⁻¹A_v Y)
    av::Vector{Float64}
end

@inline hevi_ncol(M::Mesh2D) = M.Nex * M.np
@inline hevi_blk(M::Mesh2D, c, ez) = (c - 1) * M.Nez + ez

"""Face coefficients of the column face between element (ex, ez) and (ex, ez+1) at node i:
returns (κ, a) with κ = c/|Ja^η|."""
@inline function face_coef(s, M, ex, ez, i)
    e = elem(M, ex, ez)
    np = M.np
    a, θ = s.a[i, np, e], s.θ[i, np, e]
    return sqrt(a * θ) / hypot(M.Jηx[i, np, e], M.Jηz[i, np, e]), a
end

"""Assemble and factor the element-column blocks and the tridiagonal face systems of I - c A_v."""
function factor_hevi!(F::HeviFactor, s::HDGFast2D, c)
    abs(s.tau - 1) < 1e-14 || error("the HEVI column solver needs tau_scale = 1 (LMARS)")
    M = s.mesh
    np, D, w, Nez = M.np, M.D, M.wg, M.Nez
    g = s.gas.g
    dtg = c
    n3 = 3np
    for ex in 1:M.Nex, i in 1:np
        c = (ex - 1) * np + i
        for ez in 1:Nez
            e = elem(M, ex, ez)
            b = hevi_blk(M, c, ez)
            A = view(F.A, :, :, b)
            fill!(A, 0.0)
            Bb = view(F.zb, :, b)
            Bt = view(F.zt, :, b)
            fill!(Bb, 0.0)
            fill!(Bt, 0.0)
            @inbounds for j in 1:np
                Jn = M.J[i, j, e]
                W = w[i] * w[j]
                θ, a, φ = s.θ[i, j, e], s.a[i, j, e], s.φ[i, j, e]
                Ja2 = M.Jηx[i, j, e]^2 + M.Jηz[i, j, e]^2
                dηθ = 0.0
                for k in 1:np
                    dηθ += D[j, k] * s.θ[i, k, e]
                end
                ρj, Θj, Uj = j, np + j, 2np + j
                A[ρj, ρj] = 1.0
                A[Θj, Θj] = 1.0
                A[Uj, Uj] = 1.0
                cρ = dtg / Jn
                cu = dtg * Ja2 / Jn
                if s.split
                    # linearized split form: ρ, Θ ← Σ_k D_jk M_jk (U_j/ρ_j + U_k/ρ_k);
                    # U ← |Ja|² Σ_k D_jk (a_k Θ_k + g (z_k - z_j)(G1_jk ρ_j + G2_jk ρ_k))
                    for k in 1:np
                        A[ρj, 2np + k] += cρ * D[j, k] * s.Mv1[j, k, i, e] / s.ρ0[i, k, e]
                        A[Θj, 2np + k] += cρ * D[j, k] * s.Mv4[j, k, i, e] / s.ρ0[i, k, e]
                        A[ρj, Uj] += cρ * D[j, k] * s.Mv1[j, k, i, e] / s.ρ0[i, j, e]
                        A[Θj, Uj] += cρ * D[j, k] * s.Mv4[j, k, i, e] / s.ρ0[i, j, e]
                        A[Uj, np + k] += cu * D[j, k] * s.a[i, k, e]
                        gdz = g * (M.Z[i, k, e] - M.Z[i, j, e])
                        A[Uj, ρj] += cu * D[j, k] * gdz * s.Gv1[j, k, i, e]
                        A[Uj, k] += cu * D[j, k] * gdz * s.Gv2[j, k, i, e]
                        if s.vadv     # U ← Ja_j·v̄_jk Σ_k D_jk M_jk (U_j/ρ_j + U_k/ρ_k)
                            vj = M.Jηx[i, j, e] * 0.5 * (s.u0[i, j, e] + s.u0[i, k, e]) +
                                 M.Jηz[i, j, e] * 0.5 * (s.w0[i, j, e] + s.w0[i, k, e])
                            A[Uj, 2np + k] += cρ * D[j, k] * s.Mv1[j, k, i, e] * vj / s.ρ0[i, k, e]
                            A[Uj, Uj] += cρ * D[j, k] * s.Mv1[j, k, i, e] * vj / s.ρ0[i, j, e]
                        end
                    end
                else
                    for k in 1:np
                        A[ρj, 2np + k] += cρ * D[j, k]
                        A[Θj, 2np + k] += cρ * θ * D[j, k]
                        A[Uj, np + k] += cu * D[j, k] * s.φa[i, k, e] / φ
                    end
                    A[Θj, Uj] += cρ * dηθ
                    A[Uj, Θj] -= cu * (a * dηθ / θ + s.gZη[i, j, e] / θ)
                    A[Uj, ρj] += cu * s.gZη[i, j, e]
                    if s.vadv     # U ← Ja_j·Σ_k D_jk U_k v̄_k
                        for k in 1:np
                            vk = M.Jηx[i, j, e] * s.u0[i, k, e] + M.Jηz[i, j, e] * s.w0[i, k, e]
                            A[Uj, 2np + k] += cρ * D[j, k] * vk
                        end
                    end
                end
                # SAT at the top (j = np) and bottom (j = 1) node
                if j == np || j == 1
                    κ = sqrt(a * θ) / sqrt(Ja2)
                    sw = j == np ? 1.0 : -1.0                   # ρ, Θ: ±w (U_b - M̂) with sign of ñ
                    fρ = cρ * w[i] / W                          # dt γ w / (J W)
                    fu = cu * w[i] * κ / W                      # dt γ |Ja|² w κ / (J W)
                    A[ρj, Uj] -= fρ * sw                        # top: -(dtγ/JW) w U_np, bottom: +(dtγ/JW) w U_1
                    A[Θj, Uj] -= fρ * θ * sw
                    A[Uj, Uj] += fu
                    B = j == np ? Bt : Bb
                    B[ρj] += fρ * sw
                    B[Θj] += fρ * θ * sw
                    B[Uj] -= fu
                    if s.vadv                                   # advective momentum SAT: v̄_b × mass SAT
                        vb = M.Jηx[i, j, e] * s.u0[i, j, e] + M.Jηz[i, j, e] * s.w0[i, j, e]
                        A[Uj, Uj] -= fρ * sw * vb
                        B[Uj] += fρ * sw * vb
                    end
                    if (j == 1 && ez == 1) || (j == np && ez == Nez)
                        B[ρj] = 0.0
                        B[Θj] = 0.0
                        B[Uj] = 0.0  # wall: M̂ = 0
                    end
                end
            end
            # Locally condense (ρ, Θ), whose diagonal blocks are identities, leaving the
            # Schur system for U: S = A_UU - A_Uρ A_ρU - A_UΘ A_ΘU.
            S = view(F.S, :, :, b)
            @inbounds for j in 1:np, k in 1:np
                v = A[2np+j, 2np+k]
                for l in 1:np
                    v -= A[2np+j, l] * A[l, 2np+k]
                    v -= A[2np+j, np+l] * A[np+l, 2np+k]
                end
                S[j, k] = v
            end
            # Factor only the reduced U block. Keep the original local block in `A` long
            # enough to transform face and stage right-hand sides through the condensation.
            _, ipiv, info = LAPACK.getrf!(S)
            info == 0 || error("singular HEVI local Schur block")
            F.piv[:, b] .= ipiv
            for rhs in (Bb, Bt)
                @inbounds for j in 1:np
                    v = rhs[2np+j]
                    for l in 1:np
                        v -= A[2np+j, l] * rhs[l]
                        v -= A[2np+j, np+l] * rhs[np+l]
                    end
                    rhs[2np+j] = v
                end
                LAPACK.getrs!('N', S, view(F.piv, :, b), view(rhs, 2np+1:3np))
                @inbounds for j in 1:np
                    for l in 1:np
                        rhs[j] -= A[j, 2np+l] * rhs[2np+l]
                        rhs[np+j] -= A[np+j, 2np+l] * rhs[2np+l]
                    end
                end
            end
        end
        # tridiagonal system for M̂_f, f = 1..Nez-1 (face between ez = f and f+1):
        #   κ (U_top(f) + U_bot(f+1)) + a (Θ_top(f) - Θ_bot(f+1)) - 2κ M̂_f = 0
        for f in 1:Nez-1
            κ, a = face_coef(s, M, ex, f, i)
            bL = hevi_blk(M, c, f)
            bR = hevi_blk(M, c, f + 1)
            # x_e = y_e - zb_e M̂_{f-1} - zt_e M̂_f for e = f;  x_{e+1} = y - zb M̂_f - zt M̂_{f+1}
            FtL(z) = κ * z[2np+np] + a * z[np+np]        # coefficient vector of element f (top node)
            FbR(z) = κ * z[2np+1] - a * z[np+1]          # element f+1 (bottom node)
            diag = -2κ - FtL(view(F.zt, :, bL)) - FbR(view(F.zb, :, bR))
            sub = f > 1 ? -FtL(view(F.zb, :, bL)) : 0.0
            sup = f < Nez - 1 ? -FbR(view(F.zt, :, bR)) : 0.0
            den = f == 1 ? diag : diag - sub * F.cp[f-1, c]
            F.iden[f, c] = 1 / den
            F.cp[f, c] = sup / den
            F.sub[f, c] = sub
        end
    end
    return F
end

function hevi_factor(s::HDGFast2D, c)
    get!(s.lucache, Float64(c)) do
        M = s.mesh
        n3 = 3M.np
        nblk = hevi_ncol(M) * M.Nez
        nf = max(M.Nez - 1, 1)
        F = HeviFactor(zeros(n3, n3, nblk), zeros(M.np, M.np, nblk), zeros(Int, M.np, nblk), zeros(n3, nblk), zeros(n3, nblk),
                       zeros(nf, hevi_ncol(M)), zeros(nf, hevi_ncol(M)), zeros(nf, hevi_ncol(M)),
                       zeros(n3, nblk), zeros(nf), zeros(nf), Float64(c), zeros(s.nq), zeros(s.nq))
        factor_hevi!(F, s, c)
    end
end

"""Solve (I - c M⁻¹ A_v) Y = Ys column by column (Y may alias Ys)."""
function hevi_solve!(Y, s::HDGFast2D, F::HeviFactor, Ys)
    M = s.mesh
    np, Nez = M.np, M.Nez
    s.vadv && copyto!(F.ys, Ys)
    @inbounds for ex in 1:M.Nex, i in 1:np
        c = (ex - 1) * np + i
        for ez in 1:Nez
            e = elem(M, ex, ez)
            b = hevi_blk(M, c, ez)
            for j in 1:np
                F.y[j, b] = Ys[node_iq(np, e, 1, i, j)]
                F.y[np+j, b] = Ys[node_iq(np, e, 4, i, j)]
                F.y[2np+j, b] = M.Jηx[i, j, e] * Ys[node_iq(np, e, 2, i, j)] + M.Jηz[i, j, e] * Ys[node_iq(np, e, 3, i, j)]
            end
            A = view(F.A, :, :, b)
            S = view(F.S, :, :, b)
            @inbounds for j in 1:np
                v = F.y[2np+j, b]
                for k in 1:np
                    v -= A[2np+j, k] * F.y[k, b]
                    v -= A[2np+j, np+k] * F.y[np+k, b]
                end
                F.y[2np+j, b] = v
            end
            LAPACK.getrs!('N', S, view(F.piv, :, b), view(F.y, 2np+1:3np, b))
            @inbounds for j in 1:np
                for k in 1:np
                    F.y[j, b] -= A[j, 2np+k] * F.y[2np+k, b]
                    F.y[np+j, b] -= A[np+j, 2np+k] * F.y[2np+k, b]
                end
            end
        end
        # Thomas for M̂ (right-hand side: -F_f(y))
        for f in 1:Nez-1
            κ, a = face_coef(s, M, ex, f, i)
            bL = hevi_blk(M, c, f)
            bR = hevi_blk(M, c, f + 1)
            r = -(κ * F.y[3np, bL] + a * F.y[2np, bL]) - (κ * F.y[2np+1, bR] - a * F.y[np+1, bR])
            F.dp[f] = (f == 1 ? r : r - F.sub[f, c] * F.dp[f-1]) * F.iden[f, c]
        end
        for f in Nez-1:-1:1
            F.Mh[f] = f == Nez - 1 ? F.dp[f] : F.dp[f] - F.cp[f, c] * F.Mh[f+1]
        end
        # back substitution and update
        for ez in 1:Nez
            e = elem(M, ex, ez)
            b = hevi_blk(M, c, ez)
            Mb = ez > 1 ? F.Mh[ez-1] : 0.0
            Mt = ez < Nez ? F.Mh[ez] : 0.0
            for j in 1:np
                ρ = F.y[j, b] - F.zb[j, b] * Mb - F.zt[j, b] * Mt
                Θ = F.y[np+j, b] - F.zb[np+j, b] * Mb - F.zt[np+j, b] * Mt
                U = F.y[2np+j, b] - F.zb[2np+j, b] * Mb - F.zt[2np+j, b] * Mt
                jx, jz = M.Jηx[i, j, e], M.Jηz[i, j, e]
                mx, mz = Ys[node_iq(np, e, 2, i, j)], Ys[node_iq(np, e, 3, i, j)]
                σ = (U - (jx * mx + jz * mz)) / (jx^2 + jz^2)
                Y[node_iq(np, e, 1, i, j)] = ρ
                Y[node_iq(np, e, 4, i, j)] = Θ
                Y[node_iq(np, e, 2, i, j)] = mx + jx * σ
                Y[node_iq(np, e, 3, i, j)] = mz + jz * σ
            end
        end
    end
    if s.vadv
        # the momentum change is no longer along Ja^η: m = m_s + c (M⁻¹A_v Y)_m, where A_v reads
        # the momentum only through the solved U
        vertical_apply!(F.av, s, Y)
        @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np, r in 2:3
            k = node_iq(np, e, r, i, j)
            Y[k] = F.ys[k] + F.c * F.av[k]
        end
    end
    return Y
end

# interface of TimeStepping (:hevi): horizontal part explicit, vertical part with the column solver
fast_rhs_horizontal!(out, s::HDGFast2D, q, f) = fast_kernel!(out, s, q, f, Val(true), Val(false))
vertical_factor(s::HDGFast2D, c) = hevi_factor(s, c)
vertical_solve!(Y, s::HDGFast2D, F::HeviFactor, Ys) = hevi_solve!(Y, s, F, Ys)

# interface of TimeStepping (:hevi_verlet): Strang composition of a horizontal Störmer–Verlet half
# step (ρ, Θ | momentum | ρ, Θ), vertical Crank–Nicolson and a second horizontal half step. The
# horizontal operator is split by rows; each group carries its share of the forcing, so f enters
# exactly once per substep. Dissipative LMARS self-coupling within a group is explicit.
"""Zero the rows `keep = false` of a DG state vector (rows 1 ρ, 2 mx, 3 mz, 4 Θ)."""
function dg_zero_rows!(out, M::Mesh2D, keep::NTuple{4,Bool})
    n2 = M.np^2
    @inbounds for e in 1:M.Ne, r in 1:NV
        keep[r] && continue
        o = ((e - 1) * NV + (r - 1)) * n2
        for k in 1:n2
            out[o + k] = 0.0
        end
    end
    return out
end

function fast_rhs_hthermo!(out, s::HDGFast2D, q, f)
    fast_kernel!(out, s, q, f, Val(true), Val(false))
    return dg_zero_rows!(out, s.mesh, (true, false, false, true))
end

function fast_rhs_hmomentum!(out, s::HDGFast2D, q, f)
    fast_kernel!(out, s, q, f, Val(true), Val(false))
    return dg_zero_rows!(out, s.mesh, (false, true, true, false))
end

"""out = M⁻¹ A_v q (vertical part, no forcing), matrix-free."""
function vertical_apply!(out, s::HDGFast2D, q)
    fast_kernel!(out, s, q, ZeroVec(), Val(false), Val(true))
end

"""Lazy zero vector (forcing argument of the vertical kernel)."""
struct ZeroVec <: AbstractVector{Float64} end
Base.size(::ZeroVec) = (typemax(Int),)
Base.getindex(::ZeroVec, ::Int) = 0.0
