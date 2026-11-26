# Divergence-free projection of the momenta of FV3D.Model3D (projected start of the hill and building runs), for
# uniform and stretched levels.
#
# The discrete divergence of the model times the cell volume (per Δx Δy),
#     D̃ U = Δz_k (Δ(A_x U)/Δx + Δ(A_y V)/Δy) + Δ(A_z W),
# the correction U_f = U0_f - K_f (D̃ᵀ λ)_f with K_f = ρ_f / (A_x Δz_k) on x-faces (y alike) and ρ_f / (A_z Δz̃_k) on
# z-faces, so that the correction is ρ_f times the gradient of λ; λ from the symmetric system D̃ K D̃ᵀ λ = D̃ U0.

"""Divergence-free projection of the momenta (as in the hill experiments of FV3D)."""
function project_divfree(m, q)
    g = m.grid; nx, ny, nz = g.nx, g.ny, g.nz
    cid = zeros(Int, nx, ny, nz); nc = 0
    for k in 1:nz, j in 1:ny, i in 1:nx; g.Vf[i, j, k] > 0 && (cid[i, j, k] = (nc += 1)); end
    faces = Int[]; Ir = Int[]; Jc = Int[]; Vv = Float64[]; w = Float64[]
    ρ(i, j, k) = q[F3.iρ(g, i, j, k)]
    function face!(n, d, l, r, ρf, kf)
        d > 0 || return
        push!(faces, n); f = length(faces)
        cid[l...] > 0 && (push!(Ir, cid[l...]); push!(Jc, f); push!(Vv, -d))
        cid[r...] > 0 && (push!(Ir, cid[r...]); push!(Jc, f); push!(Vv, d))
        push!(w, ρf * kf)
    end
    for k in 1:nz, j in 1:ny, i in 1:nx
        im, jm = F3.ip(g, i - 1), F3.jp(g, j - 1)
        a = F3.ax(g, i, j, k)
        face!(F3.iU(g, i, j, k), a * g.dzc[k] / g.dx, (im, j, k), (i, j, k), 0.5 * (ρ(im, j, k) + ρ(i, j, k)), a > 0 ? 1 / (a * g.dzc[k]) : 0.0)
        a = F3.ay(g, i, j, k)
        face!(F3.iV(g, i, j, k), a * g.dzc[k] / g.dy, (i, jm, k), (i, j, k), 0.5 * (ρ(i, jm, k) + ρ(i, j, k)), a > 0 ? 1 / (a * g.dzc[k]) : 0.0)
        if k > 1
            a = F3.az(g, i, j, k)
            face!(F3.iW(g, i, j, k), a, (i, j, k - 1), (i, j, k), F3.ρface(g, q, i, j, k), a > 0 ? 1 / (a * g.dzf[k]) : 0.0)
        end
    end
    D = sparse(Ir, Jc, Vv, nc, length(faces))
    U0 = q[faces]
    Lp = D * Diagonal(w) * D'; rhs = D * U0
    rhs .-= sum(rhs) / nc                              # consistent with the constant null space (periodic, closed)
    λ = _pcg_columns(Lp, rhs, cid)
    qp = copy(q); qp[faces] = U0 - w .* (D' * λ)
    return qp
end

"""PCG for the singular, symmetric Poisson matrix Lp (constant null space, consistent right-hand side), preconditioned
by the vertical lines (tridiagonal part of Lp within each column, factorised without fill): memory O(n), unlike the
sparse LU (126 GB for 256×256×40)."""
function _pcg_columns(Lp, b, cid; tol = 1e-10, maxit = 5000)
    nx, ny, nz = size(cid)
    col = zeros(Int, size(Lp, 1))
    for k in 1:nz, j in 1:ny, i in 1:nx; cid[i, j, k] > 0 && (col[cid[i, j, k]] = i + nx * (j - 1)); end
    I, J, V = findnz(Lp)
    keep = col[I] .== col[J]
    F = lu(sparse(I[keep], J[keep], V[keep], size(Lp)...))
    x = zeros(length(b)); r = copy(b); z = F \ r; p = copy(z); rz = dot(r, z); nb = norm(b)
    for it in 1:maxit
        Ap = Lp * p; α = rz / dot(p, Ap)
        x .+= α .* p; r .-= α .* Ap
        norm(r) <= tol * nb && break
        z = F \ r; rzn = dot(r, z); p .= z .+ (rzn / rz) .* p; rz = rzn
    end
    return x
end
