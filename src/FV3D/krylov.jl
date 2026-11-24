# Iterative solve of the coupled masked system (I - c (J_ac + J_adv)) x = b of implicit = :amf
# (FV3D_MASK_SOLVER = :gmres): right-preconditioned restarted GMRES with a restricted additive Schwarz (RAS)
# preconditioner. Subdomains: tiles of columns (FV3D_MASK_TILES) extended by FV3D_MASK_OVERLAP columns in x and y, the
# correction written back on the owned unknowns only. Local solver (FV3D_MASK_LOCAL): :amf (default) the approximate
# factorization (I - c J_ac)_S (I - c J_adv)_S of the subdomain, two sparse LU (on the GPU: column solves and a small
# masked block), or :lu the sparse LU of the coupled subdomain matrix. Tiles of one column and
# overlap 0: block Jacobi over the columns. Each subdomain is independent (one per MPI tile / batched on the GPU).
#
# Hill 1500 m, 64×64×40, min V_f 3e-4, :conservative Jacobian, 5e5 unknowns, iterations for tol 1e-6 / 1e-10:
# RAS with LU 3 / 5 for 4×4 … 32×32 tiles, RAS 8×8 with AMF 5 / 7 (setup and solve ~15 % faster than LU), global AMF
# 5 / 7, block Jacobi over the columns 11 / 18.

"""Home column (i, j) of unknown n: ρ, Θ of the cells and W of the faces of the column, U, V of its left faces."""
@inline function homecol(g::Grid3D, n)
    N = ncell(g); nW = g.nx * g.ny * (g.nz - 1)
    c = n <= 3N ? (n - 1) % N : (n <= 3N + nW ? n - 3N - 1 : n - 3N - nW - 1)
    return (c % g.nx + 1, (c ÷ g.nx) % g.ny + 1)
end

struct RAS
    sets::Vector{Vector{Int}}          # local unknowns (positions in the index vector) of the extended subdomains
    own::Vector{Vector{Int}}           # owned local unknowns
    pos::Vector{Vector{Int}}           # positions of the owned unknowns in `sets`
    facs::Vector{Any}                  # per subdomain: tuple of sparse LU factors, applied in order
end

"""Subdomains of the unknowns `id` (sorted global indices): tiles (tx, ty) of columns, overlap `ov` columns."""
function ras_sets(g::Grid3D, id, tx, ty, ov)
    tile(i, j) = ((i - 1) * tx ÷ g.nx + 1, (j - 1) * ty ÷ g.ny + 1)
    tid(t) = t[1] + (t[2] - 1) * tx
    sets = [Int[] for _ in 1:tx*ty]; own = [Int[] for _ in 1:tx*ty]
    ts = Int[]
    for (l, n) in enumerate(id)
        i, j = homecol(g, n)
        push!(own[tid(tile(i, j))], l)
        empty!(ts)
        for d2 in -ov:ov, d1 in -ov:ov
            t = tid(tile(ip(g, i + d1), jp(g, j + d2)))
            t in ts || push!(ts, t)
        end
        for t in ts; push!(sets[t], l); end
    end
    keep = findall(!isempty, own)
    return sets[keep], own[keep]
end

"""RAS with the local solver (A_n)_S⁻¹ ⋯ (A_1)_S⁻¹ for the matrices As = (A_1, …, A_n): (A,) the exact local LU,
(I - c J_ac, I - c J_adv) the approximate factorization."""
function RAS(As::Tuple, sets, own)
    pos = [[searchsortedfirst(S, o) for o in O] for (S, O) in zip(sets, own)]
    facs = Vector{Any}(undef, length(sets))
    Threads.@threads for t in eachindex(sets)
        S = sets[t]
        facs[t] = map(A -> lu(A[S, S]), As)
    end
    return RAS(sets, own, pos, facs)
end

"""z = M⁻¹ r (restricted additive Schwarz)."""
function apply!(z, P::RAS, r)
    Threads.@threads for t in eachindex(P.sets)
        y = r[P.sets[t]]
        for F in P.facs[t]
            y = F \ y
        end
        @inbounds for (o, p) in zip(P.own[t], P.pos[t])
            z[o] = y[p]
        end
    end
    return z
end

"""Workspace of the restarted GMRES."""
struct GMRESWork
    V::Matrix{Float64}; Z::Matrix{Float64}; H::Matrix{Float64}
    cs::Vector{Float64}; sn::Vector{Float64}; e::Vector{Float64}; r::Vector{Float64}; w::Vector{Float64}
end
GMRESWork(n, m) = GMRESWork(zeros(n, m + 1), zeros(n, m), zeros(m + 1, m), zeros(m), zeros(m), zeros(m + 1), zeros(n), zeros(n))

"""
Right-preconditioned restarted GMRES for A x = b, x the initial guess on entry. Stops at |b - A x| ≤ tol |b|.
Returns the number of iterations.
"""
function gmres!(x, A, b, P, W::GMRESWork; tol = 1e-8, maxit = 200)
    m = size(W.H, 2)
    V, Z, H, cs, sn, e, r, w = W.V, W.Z, W.H, W.cs, W.sn, W.e, W.r, W.w
    nb = norm(b); nb == 0 && (fill!(x, 0.0); return 0)
    it = 0
    while true
        mul!(r, A, x); @. r = b - r
        β = norm(r)
        (β <= tol * nb || it >= maxit) && return it
        @views V[:, 1] .= r ./ β
        fill!(e, 0.0); e[1] = β
        j = 0
        for jj in 1:m
            j = jj; it += 1
            @views apply!(Z[:, jj], P, V[:, jj])
            @views mul!(w, A, Z[:, jj])
            for i in 1:jj                                   # modified Gram–Schmidt
                @views H[i, jj] = dot(V[:, i], w)
                @views w .-= H[i, jj] .* V[:, i]
            end
            H[jj+1, jj] = norm(w)
            @views V[:, jj+1] .= w ./ H[jj+1, jj]
            for i in 1:jj-1                                 # Givens rotations
                t = cs[i] * H[i, jj] + sn[i] * H[i+1, jj]
                H[i+1, jj] = -sn[i] * H[i, jj] + cs[i] * H[i+1, jj]; H[i, jj] = t
            end
            ρ = hypot(H[jj, jj], H[jj+1, jj])
            cs[jj] = H[jj, jj] / ρ; sn[jj] = H[jj+1, jj] / ρ
            H[jj, jj] = ρ; H[jj+1, jj] = 0.0
            e[jj+1] = -sn[jj] * e[jj]; e[jj] *= cs[jj]
            (abs(e[jj+1]) <= tol * nb || it >= maxit) && break
        end
        y = UpperTriangular(view(H, 1:j, 1:j)) \ e[1:j]
        @views mul!(x, Z[:, 1:j], y, 1.0, 1.0)
    end
end
