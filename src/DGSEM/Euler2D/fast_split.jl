# Thermodynamic/momentum split with interface momentum; Verlet and forward-backward

# ---------------------------------------------------------------------------
# Thermodynamic / momentum split of the fast operator with the interface variable
# M̂ = ñ·m̂ (contravariant momentum at the face nodes):
#   interior  M̂ = ñ·{m} - |ñ| a [Θ]/(2c)     (LMARS mass flux, [Θ] = Θ_R - Θ_L)
#   wall      M̂ = 0
#   ρ_b += w (ñ·m_b - M̂)/(J W),  Θ_b += θ̄ w (ñ·m_b - M̂)/(J W)     (ñ outward)
#   m_b += w ñ (a Θ_b - P̂)/(J W),  P̂ = {aΘ} - c [m_n]/2,  wall P̂ = a Θ_b + c n̂·m_b
# thermo + momentum with M̂ = M̂(q) reproduce fast_apply! exactly. The dissipative parts scale
# with tau_scale; tau_scale = 0 gives the central (alternating, θ = 1/2) interface of Xu, van der
# Vegt, Bokhove (J. Sci. Comput. 35, 2008): the operator is then energy conserving and Verlet
# (their eq. 4.6 with f = 0) is exactly symplectic.
# ---------------------------------------------------------------------------
"""Interface momentum M̂ (LMARS) from the state q; with relax = true it is relaxed towards that
value by one implicit Euler step, M̂ ← (ε M̂ + dt M̂*)/(ε + dt) (ε = s.eps_if)."""
function interface_momentum!(s::HDGFast2D, q; relax::Bool = false, dt = 0.0)
    M = s.mesh
    np = M.np
    r = relax && s.eps_if > 0 ? s.eps_if / (s.eps_if + dt) : 0.0
    @inbounds for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            v = mhat_face(s, q, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e])
            s.Mhx[j, e] = r * s.Mhx[j, e] + (1 - r) * v
        end
        for i in 1:np
            ftop = ez * M.Nex + ex
            v = ez < M.Nez ? mhat_face(s, q, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e]) : 0.0
            s.Mhz[i, ftop] = r * s.Mhz[i, ftop] + (1 - r) * v
            ez == 1 && (s.Mhz[i, ex] = 0.0)
        end
    end
    return s
end

"""Horizontal HEVI tendency with an implicitly relaxed horizontal face mass flux."""
function fast_rhs_horizontal_relaxed!(out, s::HDGFast2D, q, f, dt)
    M = s.mesh
    np, w = M.np, M.wg
    # Start from the ordinary horizontal operator. Only its mass/thermodynamic face flux
    # changes under the existing interface-momentum relaxation; pressure SATs stay LMARS.
    fast_kernel!(out, s, q, f, Val(true), Val(false))
    r = s.eps_if > 0 ? s.eps_if / (s.eps_if + dt) : 0.0
    @inbounds for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            nx, nz = M.Jξx[np, j, e], M.Jξz[np, j, e]
            target = mhat_face(s, q, np, j, e, 1, j, eR, nx, nz)
            old = s.Mhx[j, e]
            relaxed = r * old + (1 - r) * target
            s.Mhx[j, e] = relaxed
            δ = target - relaxed
            kL = node_iq(np, e, 1, np, j)
            kΘL = node_iq(np, e, 4, np, j)
            kR = node_iq(np, eR, 1, 1, j)
            kΘR = node_iq(np, eR, 4, 1, j)
            vL = w[j] * δ / s.Mdiag[kL]
            vR = w[j] * δ / s.Mdiag[kR]
            out[kL] += vL
            out[kΘL] += s.θ[np, j, e] * vL
            out[kR] -= vR
            out[kΘR] -= s.θ[1, j, eR] * vR
        end
    end
    return out
end

@inline function mhat_face(s, q, iL, jL, eL, iR, jR, eR, nx, nz)
    np = s.mesh.np
    @inbounds begin
        a, θ = s.a[iL, jL, eL], s.θ[iL, jL, eL]
        c = sqrt(a * θ)
        mL = nx * q[node_iq(np, eL, 2, iL, jL)] + nz * q[node_iq(np, eL, 3, iL, jL)]
        mR = nx * q[node_iq(np, eR, 2, iR, jR)] + nz * q[node_iq(np, eR, 3, iR, jR)]
        return 0.5 * (mL + mR) - s.tau * hypot(nx, nz) * a / (2c) * (q[node_iq(np, eR, 4, iR, jR)] - q[node_iq(np, eL, 4, iL, jL)])
    end
end

@inline function thermo_side!(out, s, q, i, j, e, nx, nz, we, Mh)
    np = s.mesh.np
    @inbounds begin
        d = we * (nx * q[node_iq(np, e, 2, i, j)] + nz * q[node_iq(np, e, 3, i, j)] - Mh)
        out[node_iq(np, e, 1, i, j)] += d
        out[node_iq(np, e, 4, i, j)] += s.θ[i, j, e] * d
    end
    return nothing
end

"""
    fast_rhs_thermo!(out, s, q, f)

ρ and Θ tendencies of the fast operator (rows 1 and 4 of out; the momentum rows are set to
zero), using the interface momentum stored in s (`interface_momentum!`). f: forcing (length nq).
"""
function fast_rhs_thermo!(out, s::HDGFast2D, q, f)
    M = s.mesh
    np, D, w = M.np, M.D, M.wg
    Fξ, Fη = s.Fξ, s.Fη
    @inbounds for e in 1:M.Ne
        for j in 1:np, i in 1:np
            mx = q[node_iq(np, e, 2, i, j)]
            mz = q[node_iq(np, e, 3, i, j)]
            Fξ[i, j] = M.Jξx[i, j, e] * mx + M.Jξz[i, j, e] * mz
            Fη[i, j] = M.Jηx[i, j, e] * mx + M.Jηz[i, j, e] * mz
        end
        for j in 1:np, i in 1:np
            div = 0.0
            for k in 1:np
                div += D[i, k] * Fξ[k, j] + D[j, k] * Fη[i, k]
            end
            mx = q[node_iq(np, e, 2, i, j)]
            mz = q[node_iq(np, e, 3, i, j)]
            W = w[i] * w[j]
            out[node_iq(np, e, 1, i, j)] = -W * div
            out[node_iq(np, e, 4, i, j)] = -W * (s.θ[i, j, e] * div + mx * (s.gξx[i, j, e] + s.gηx[i, j, e]) +
                                                 mz * (s.gξz[i, j, e] + s.gηz[i, j, e]))
            out[node_iq(np, e, 2, i, j)] = 0.0
            out[node_iq(np, e, 3, i, j)] = 0.0
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            nx, nz = M.Jξx[np, j, e], M.Jξz[np, j, e]
            Mh = s.Mhx[j, e]
            thermo_side!(out, s, q, np, j, e, nx, nz, w[j], Mh)
            thermo_side!(out, s, q, 1, j, eR, -nx, -nz, w[j], -Mh)
        end
        for i in 1:np
            nx, nz = M.Jηx[i, np, e], M.Jηz[i, np, e]
            Mh = s.Mhz[i, ez * M.Nex + ex]
            thermo_side!(out, s, q, i, np, e, nx, nz, w[i], Mh)
            ez < M.Nez && thermo_side!(out, s, q, i, 1, elem(M, ex, ez + 1), -nx, -nz, w[i], -Mh)
            ez == 1 && thermo_side!(out, s, q, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[i], -s.Mhz[i, ex])
        end
    end
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np, r in (1, 4)
        k = node_iq(np, e, r, i, j)
        out[k] = (out[k] + f[k]) * s.Minv[k]
    end
    return out
end

@inline function momentum_face!(out, s, q, iL, jL, eL, iR, jR, eR, nx, nz, we)
    np = s.mesh.np
    @inbounds begin
        a, θ = s.a[iL, jL, eL], s.θ[iL, jL, eL]
        c = sqrt(a * θ)
        h = hypot(nx, nz)
        n1, n2 = nx / h, nz / h
        pL = a * q[node_iq(np, eL, 4, iL, jL)]
        pR = a * q[node_iq(np, eR, 4, iR, jR)]
        mnL = n1 * q[node_iq(np, eL, 2, iL, jL)] + n2 * q[node_iq(np, eL, 3, iL, jL)]
        mnR = n1 * q[node_iq(np, eR, 2, iR, jR)] + n2 * q[node_iq(np, eR, 3, iR, jR)]
        Ph = 0.5 * (pL + pR) - 0.5 * s.tau * c * (mnR - mnL)
        out[node_iq(np, eL, 2, iL, jL)] += we * nx * (pL - Ph)
        out[node_iq(np, eL, 3, iL, jL)] += we * nz * (pL - Ph)
        out[node_iq(np, eR, 2, iR, jR)] -= we * nx * (pR - Ph)
        out[node_iq(np, eR, 3, iR, jR)] -= we * nz * (pR - Ph)
    end
    return nothing
end

@inline function momentum_wall!(out, s, q, i, j, e, nx, nz, we)
    np = s.mesh.np
    @inbounds begin
        a, θ = s.a[i, j, e], s.θ[i, j, e]
        c = sqrt(a * θ)
        h = hypot(nx, nz)
        mn = (nx * q[node_iq(np, e, 2, i, j)] + nz * q[node_iq(np, e, 3, i, j)]) / h
        d = -s.tau * c * mn                                          # p_b - P̂
        out[node_iq(np, e, 2, i, j)] += we * nx * d
        out[node_iq(np, e, 3, i, j)] += we * nz * d
    end
    return nothing
end

"""
    fast_rhs_momentum!(out, s, q, f)

Momentum tendencies of the fast operator (rows 2 and 3 of out; the thermodynamic rows are set to
zero): φ-weighted pressure gradient, buoyancy, b-gradient (in f) and the LMARS interface pressure.
"""
function fast_rhs_momentum!(out, s::HDGFast2D, q, f)
    M = s.mesh
    np, D, w, g = M.np, M.D, M.wg, s.gas.g
    ψ = s.ψ
    @inbounds for e in 1:M.Ne
        for j in 1:np, i in 1:np
            ψ[i, j] = s.φa[i, j, e] * q[node_iq(np, e, 4, i, j)]
        end
        for j in 1:np, i in 1:np
            dξ = dη = 0.0
            for k in 1:np
                dξ += D[i, k] * ψ[k, j]
                dη += D[j, k] * ψ[i, k]
            end
            ρ = q[node_iq(np, e, 1, i, j)]
            Θ = q[node_iq(np, e, 4, i, j)]
            θ = s.θ[i, j, e]
            aΘθ = s.a[i, j, e] * Θ / θ
            iφ = 1 / s.φ[i, j, e]
            W = w[i] * w[j]
            out[node_iq(np, e, 2, i, j)] = -W * ((M.Jξx[i, j, e] * dξ + M.Jηx[i, j, e] * dη) * iφ -
                                                 aΘθ * (s.gξx[i, j, e] + s.gηx[i, j, e]))
            out[node_iq(np, e, 3, i, j)] = -W * ((M.Jξz[i, j, e] * dξ + M.Jηz[i, j, e] * dη) * iφ -
                                                 aΘθ * (s.gξz[i, j, e] + s.gηz[i, j, e])) +
                                           W * M.J[i, j, e] * g * (Θ / θ - ρ)
            out[node_iq(np, e, 1, i, j)] = 0.0
            out[node_iq(np, e, 4, i, j)] = 0.0
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            momentum_face!(out, s, q, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[j])
        end
        for i in 1:np
            if ez < M.Nez
                momentum_face!(out, s, q, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
            else
                momentum_wall!(out, s, q, i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
            end
            ez == 1 && momentum_wall!(out, s, q, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[i])
        end
    end
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np, r in (2, 3)
        k = node_iq(np, e, r, i, j)
        out[k] = (out[k] + f[k]) * s.Minv[k]
    end
    return out
end

"""Interface variable of the thermodynamic/momentum split: the face momentum M̂."""
interface_update!(s::HDGFast2D, q; relax::Bool = false, dt = 0.0) = interface_momentum!(s, q; relax = relax, dt = dt)

tendency(s::HDGFast2D, q; F_slow = nothing) =
    fast_apply!(zeros(s.nq), s, q, F_slow === nothing ? s.fb : s.fb .+ s.Mdiag .* F_slow)
initial_state(s::HDGFast2D, q) = Vector{Float64}(q)

"""Dense matrix of q ↦ (K + G H) q ./ M (testing and spectra; small meshes only)."""
function fast_matrix(s::HDGFast2D)
    A = zeros(s.nq, s.nq)
    e = zeros(s.nq)
    z = zeros(s.nq)
    col = zeros(s.nq)
    for k in 1:s.nq
        e[k] = 1.0
        fast_kernel!(col, s, e, z, Val(true), Val(true))
        A[:, k] .= col
        e[k] = 0.0
    end
    return A
end

# ---------------------------------------------------------------------------
# Störmer–Verlet with Bokhove's partition (fast = :verlet_b). In the HDG form the thermodynamic rows see only the
# face momentum M̂ = ñ·{m} - τ|ñ| a [Θ]/(2c), the momentum rows the face pressure P̂ = {aΘ} - τ c [m_n]/2. Their
# jump parts are the self-couplings D_a (a←a) and D_b (b←b), both local to a face (corner nodes link the faces of
# up to four elements: blocks of at most 8 unknowns). As the Coriolis term in Xu, van der Vegt, Bokhove
# (J. Sci. Comput. 35 (2008), eq. 4.6) they are integrated exactly inside the split flows
#   A_c: a' = K̂ b + f_a, b' = D_b b:   b ← e^{cD_b} b,  a ← a + K̂ ∫b + c f_a,  ∫b = cφ₁(cD_b) b,
#   B_c: a' = D_a a, b' = L a + f_b:   a ← e^{cD_a} a,  b ← b + L ∫a + c f_b,  ∫a = cφ₁(cD_a) a,
# (K̂: M̂ with the central {m} only, L: P̂ with the central {aΘ} only), step A_{h/2} ∘ B_h ∘ A_{h/2}; consecutive
# A_{h/2} combine to A_h. The forcing stays with K̂ and L (f_b with L a keeps the hydrostatic balance within B).
# docs/frozen_dg, Section sec:hdgb.
# ---------------------------------------------------------------------------
using SparseArrays: sparse, SparseMatrixCSC, nzrange, rowvals, nonzeros
using LinearAlgebra: mul!

"""Sparse D_b (momentum rows, scaled by M⁻¹): the b←b part of the LMARS face pressure at faces and walls."""
function momentum_damping(s::HDGFast2D)
    M = s.mesh
    np, w = M.np, M.wg
    I_, J_, V_ = Int[], Int[], Float64[]
    add(r, c, v) = (push!(I_, r); push!(J_, c); push!(V_, v * s.Minv[r]))
    function face(iL, jL, eL, iR, jR, eR, nx, nz, we)
        a, θ = s.a[iL, jL, eL], s.θ[iL, jL, eL]
        c = sqrt(a * θ); h = hypot(nx, nz); n1, n2 = nx / h, nz / h
        κ = 0.5 * s.tau * c * we                 # out_L += we n (½τc)(m_nR - m_nL), out_R -= the same
        for (r, nr) in ((node_iq(np, eL, 2, iL, jL), nx), (node_iq(np, eL, 3, iL, jL), nz))
            add(r, node_iq(np, eR, 2, iR, jR), κ * nr * n1); add(r, node_iq(np, eR, 3, iR, jR), κ * nr * n2)
            add(r, node_iq(np, eL, 2, iL, jL), -κ * nr * n1); add(r, node_iq(np, eL, 3, iL, jL), -κ * nr * n2)
        end
        for (r, nr) in ((node_iq(np, eR, 2, iR, jR), nx), (node_iq(np, eR, 3, iR, jR), nz))
            add(r, node_iq(np, eR, 2, iR, jR), -κ * nr * n1); add(r, node_iq(np, eR, 3, iR, jR), -κ * nr * n2)
            add(r, node_iq(np, eL, 2, iL, jL), κ * nr * n1); add(r, node_iq(np, eL, 3, iL, jL), κ * nr * n2)
        end
    end
    function wall(i, j, e, nx, nz, we)
        a, θ = s.a[i, j, e], s.θ[i, j, e]
        c = sqrt(a * θ); h = hypot(nx, nz)
        κ = s.tau * c * we / h                   # out += we n (-τ c m_n), m_n = (nx m_x + nz m_z)/h
        for (r, nr) in ((node_iq(np, e, 2, i, j), nx), (node_iq(np, e, 3, i, j), nz))
            add(r, node_iq(np, e, 2, i, j), -κ * nr * nx); add(r, node_iq(np, e, 3, i, j), -κ * nr * nz)
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            face(np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[j])
        end
        for i in 1:np
            if ez < M.Nez
                face(i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
            else
                wall(i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
            end
            ez == 1 && wall(i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[i])
        end
    end
    return sparse(I_, J_, V_, s.nq, s.nq)
end

"""Block structure of a sparse D with a block-diagonal pattern (connected components): the unknowns of each
block, the positions of its entries in D.nzval (0: none), and the pattern of e^{cD}, c φ₁(cD) (blocks plus
the diagonal elsewhere) with the positions of the blocks and of the remaining diagonal in it."""
struct BlockPattern
    groups::Vector{Vector{Int}}
    dpos::Vector{Matrix{Int}}
    epos::Vector{Matrix{Int}}
    diagpos::Vector{Int}
    pattern::SparseMatrixCSC{Float64,Int}
end
function BlockPattern(D::SparseMatrixCSC)
    n = size(D, 1)
    parent = collect(1:n)
    root(x) = (while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end; x)
    rv, act = rowvals(D), falses(n)
    for j in 1:n, k in nzrange(D, j)
        r = rv[k]; act[r] = act[j] = true
        a, b = root(r), root(j); a != b && (parent[a] = b)
    end
    g = Dict{Int,Vector{Int}}()
    for k in 1:n
        act[k] && push!(get!(g, root(k), Int[]), k)
    end
    groups = sort!(collect(values(g)); by = first)
    IE, JE = Int[], Int[]
    for idx in groups, j in idx, i in idx
        push!(IE, i); push!(JE, j)
    end
    for k in 1:n
        act[k] || (push!(IE, k); push!(JE, k))
    end
    pat = sparse(IE, JE, zeros(length(IE)), n, n)
    pos(A, i, j) = (r = nzrange(A, j); t = searchsortedfirst(view(rowvals(A), r), i);
                    t <= length(r) && rowvals(A)[r[t]] == i ? r[t] : 0)
    dpos = [[pos(D, i, j) for i in idx, j in idx] for idx in groups]
    epos = [[pos(pat, i, j) for i in idx, j in idx] for idx in groups]
    diagpos = [pos(pat, k, k) for k in 1:n if !act[k]]
    return BlockPattern(groups, dpos, epos, diagpos, pat)
end

"""φ₂(x) = (e^x - 1 - x)/x², φ₃(x) = (e^x - 1 - x - x²/2)/x³."""
_phi2(x) = abs(x) < 1e-3 ? 0.5 + x / 6 + x^2 / 24 + x^3 / 120 : (expm1(x) - x) / x^2
_phi3(x) = abs(x) < 1e-2 ? 1 / 6 + x / 24 + x^2 / 120 + x^3 / 720 : (expm1(x) - x - x^2 / 2) / x^3

"""(e^{cD}, c φ₁(cD), c² φ₂(cD)) blockwise on the fixed pattern of `bp` (D with the pattern of bp's
construction). Rank-one blocks (X² = λX, λ = tr X: a face node pair) in closed form, e^X = I + φ₁(λ)X,
φ₁(X) = I + φ₂(λ)X, φ₂(X) = I/2 + φ₃(λ)X; the others (blocks linked at element corners) from the exponential of
[X I 0; 0 0 I; 0 0 0]."""
function block_exp(bp::BlockPattern, D::SparseMatrixCSC, c)
    E = copy(bp.pattern); Φ = copy(bp.pattern); Ψ = copy(bp.pattern)
    nzE, nzΦ, nzΨ, nzD = nonzeros(E), nonzeros(Φ), nonzeros(Ψ), nonzeros(D)
    @inbounds for k in bp.diagpos
        nzE[k] = 1.0; nzΦ[k] = c; nzΨ[k] = c^2 / 2
    end
    for (g, idx) in enumerate(bp.groups)
        dp, ep = bp.dpos[g], bp.epos[g]
        m = length(idx)
        X = [dp[i, j] == 0 ? 0.0 : c * nzD[dp[i, j]] for i in 1:m, j in 1:m]
        λ = sum(X[i, i] for i in 1:m)
        if maximum(abs, X * X .- λ .* X) <= 1e-12 * max(maximum(abs, X)^2, eps())
            a1, a2, a3 = expm1(λ) / (λ == 0 ? 1.0 : λ), _phi2(λ), _phi3(λ)
            λ == 0 && (a1 = 1.0)
            @inbounds for j in 1:m, i in 1:m
                δ = i == j ? 1.0 : 0.0
                nzE[ep[i, j]] = δ + a1 * X[i, j]
                nzΦ[ep[i, j]] = c * (δ + a2 * X[i, j])
                nzΨ[ep[i, j]] = c^2 * (δ / 2 + a3 * X[i, j])
            end
        else
            Id = Matrix(1.0I, m, m); Z = zeros(m, m)
            Y = exp([X Id Z; Z Z Id; Z Z Z])                     # [e^X φ₁(X) φ₂(X); …]
            @inbounds for j in 1:m, i in 1:m
                nzE[ep[i, j]] = Y[i, j]; nzΦ[ep[i, j]] = c * Y[i, m+j]; nzΨ[ep[i, j]] = c^2 * Y[i, 2m+j]
            end
        end
    end
    return E, Φ, Ψ
end
block_exp(D::SparseMatrixCSC, c) = block_exp(BlockPattern(D), D, c)

"""Sparse D_a (thermodynamic rows, scaled by M⁻¹): the a←a part of the thermodynamic tendency, from the
pressure (Θ) jump of the LMARS face momentum M̂* = ñ·{m} - τ|ñ| a [Θ]/(2c) at the interior faces."""
function thermo_damping(s::HDGFast2D)
    M = s.mesh
    np, w = M.np, M.wg
    I_, J_, V_ = Int[], Int[], Float64[]
    add(r, c, v) = (push!(I_, r); push!(J_, c); push!(V_, v * s.Minv[r]))
    function face(iL, jL, eL, iR, jR, eR, nx, nz, we)
        a, θ = s.a[iL, jL, eL], s.θ[iL, jL, eL]
        κ = we * s.tau * hypot(nx, nz) * a / (2 * sqrt(a * θ))     # d_L = κ (Θ_R - Θ_L) = -d_R
        ΘL, ΘR = node_iq(np, eL, 4, iL, jL), node_iq(np, eR, 4, iR, jR)
        for (i, j, e, sg) in ((iL, jL, eL, 1.0), (iR, jR, eR, -1.0))
            for (r, f) in ((node_iq(np, e, 1, i, j), 1.0), (node_iq(np, e, 4, i, j), s.θ[i, j, e]))
                add(r, ΘR, sg * κ * f); add(r, ΘL, -sg * κ * f)
            end
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            face(np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[j])
        end
        for i in 1:np
            ez < M.Nez && face(i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[i])
        end
    end
    return sparse(I_, J_, V_, s.nq, s.nq)
end


"""
Störmer–Verlet with Bokhove's partition: A_{h/2} ∘ B_h ∘ A_{h/2},
  A_c: b ← e^{cD_b} b,  a ← a + c (K̂ b̄ + f_a),  b̄ = φ₁(cD_b) b   (K̂: M̂ with the central {m} only),
  B_c: a ← e^{cD_a} a,  b ← b + c (L ā + f_b),  ā = φ₁(cD_a) a,
both flows exact; D = D_a + D_b is block diagonal (face node pairs, linked at the element corners).
"""
function bokhove_verlet_loop!(q, s::HDGFast2D, f, dt, nsteps, k1, k2)
    D = get!(s.bflows, -1.0) do
        thermo_damping(s) + momentum_damping(s)
    end::SparseMatrixCSC{Float64,Int}
    s.bblocks === nothing && (s.bblocks = BlockPattern(D))
    bp = s.bblocks::BlockPattern
    SM = SparseMatrixCSC{Float64,Int}
    np2 = s.mesh.np^2
    getM(c) = (c, get!(() -> block_exp(bp, D, c), s.bflows, Float64(c))::Tuple{SM,SM,SM}...)
    zb = get!(() -> zeros(s.nq), s.bflows, -2.0)::Vector{Float64}
    z0 = get!(() -> zeros(s.nq), s.bflows, -3.0)::Vector{Float64}
    fs = get!(() -> zeros(s.nq), s.bflows, -4.0)::Vector{Float64}
    @. fs = f * s.Minv                                         # forcing in state units
    return _bokhove_verlet!(q, s, fs, dt, nsteps, k1, k2, getM(dt / 2), getM(dt), zb, z0)
end

@inline _hdg_ismom(np2, k) = mod1(cld(k, np2), NV) in (2, 3)

"""Exact flow A_c: a' = K̂ b + f_a, b' = D_b b: b ← e^{cD_b} b, a ← a + K̂ ∫b + c f_a, ∫b = cφ₁ b."""
function _hdg_aflow!(q, s::HDGFast2D, fs, (c, E, Φ, Ψ), zb, z0, k1, k2)
    np2 = s.mesh.np^2
    mul!(zb, Φ, q)
    @inbounds for k in eachindex(zb); _hdg_ismom(np2, k) || (zb[k] = 0.0); end      # (0, ∫b)
    interface_momentum!(s, zb)                                                       # M̂ = ñ·{∫b}
    fast_rhs_thermo!(k1, s, zb, z0)                                                  # K̂ ∫b (zero on b)
    mul!(k2, E, q)
    @inbounds for k in eachindex(q)
        q[k] = _hdg_ismom(np2, k) ? k2[k] : q[k] + k1[k] + c * fs[k]
    end
    return q
end

"""Exact flow B_c: a' = D_a a, b' = L a + f_b: a ← e^{cD_a} a, b ← b + L ∫a + c f_b, ∫a = cφ₁ a."""
function _hdg_bflow!(q, s::HDGFast2D, fs, (c, E, Φ, Ψ), zb, z0, k1, k2)
    np2 = s.mesh.np^2
    mul!(zb, Φ, q)
    @inbounds for k in eachindex(zb); _hdg_ismom(np2, k) && (zb[k] = 0.0); end      # (∫a, 0)
    fast_rhs_momentum!(k1, s, zb, z0)                                                # L ∫a (zero on a)
    mul!(k2, E, q)
    @inbounds for k in eachindex(q)
        q[k] = _hdg_ismom(np2, k) ? q[k] + k1[k] + c * fs[k] : k2[k]
    end
    return q
end

# A_{h/2} ∘ B_h ∘ A_{h/2} per step; A is an exact flow, so the A_{h/2} of consecutive steps combine to A_h.
function _bokhove_verlet!(q, s::HDGFast2D, fs, dt, nsteps, k1, k2, MH, M1, zb, z0)
    _hdg_aflow!(q, s, fs, MH, zb, z0, k1, k2)
    for n in 1:nsteps
        _hdg_bflow!(q, s, fs, M1, zb, z0, k1, k2)
        _hdg_aflow!(q, s, fs, n < nsteps ? M1 : MH, zb, z0, k1, k2)
    end
    return q
end
