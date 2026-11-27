# Geometric multigrid preconditioner for the acoustic factor A = I - c J_ac of the Rosenbrock step (rosenbrock.jl).
# Prototype on assembled matrices (CPU), to choose the method for the GPU (FV3DKA).
#
# Reduction: in J_ac the momentum rows depend on ρ and Θ only (pressure gradient, gravity) and the momentum block is
# diagonal (sponge), the ρ, Θ rows on the momenta only (fluxes). With t = (ρ, Θ), m = (U, V, W):
#     A = [I  A_tm; A_mt  D_m],  D_m diagonal,   S = I - A_tm D_m⁻¹ A_mt   (Schur complement, 2 unknowns per cell),
#     S x_t = r_t - A_tm D_m⁻¹ r_m,   x_m = D_m⁻¹ (r_m - A_mt x_t)   (exact).
# Θ couples horizontally through the pressure Laplacian (face weights A θ_f P_f), ρ and Θ vertically through the
# fluxes and gravity.
#
# Multigrid on S (Weber et al. 2015, ASAM JacAccGrav): horizontal 2 × 2 coarsening (the vertical kept: Δz ≪ Δx near
# the ground), restriction R the sum over the fine cells (finite-volume integral), prolongation P = Rᵀ (injection),
# Galerkin coarse operators R S P (equivalent to summing the face areas on the coarse faces), the coarsest level by LU.
# Smoother: the column blocks of S (vertical lines, exact vertical acoustics and gravity per column) — damped block
# Jacobi (`:jacobi`) or zebra (`:zebra`, columns in red–black order). One V-cycle per application.

"""One level: S, the LU of its column blocks, the colour of the unknowns (zebra), the restriction to the next level."""
struct MGLevel
    S::SparseMatrixCSC{Float64,Int}
    B::Any                                   # LU of the column-block-diagonal part of S
    red::BitVector                           # unknowns of the red columns
    R::SparseMatrixCSC{Float64,Int}          # restriction to the next coarser level (0 × 0 on the coarsest)
    nx::Int; ny::Int
end

"""Acoustic multigrid preconditioner (GMRES `apply!`): Schur reduction, levels, smoother."""
struct AcousticMG
    t::Vector{Int}; m::Vector{Int}           # ρ, Θ and momentum unknowns of the model
    Atm::SparseMatrixCSC{Float64,Int}; Amt::SparseMatrixCSC{Float64,Int}; dinv::Vector{Float64}
    levels::Vector{MGLevel}
    coarse::Any                              # LU of the coarsest S
    smoother::Symbol; ν::Int; ω::Float64
    work::Vector{Vector{Float64}}
end

"""Column (i + nx (j - 1)) of the reduced unknown n (ρ of the cells, then Θ) on an nx × ny × nz level."""
@inline _rcol(n, nx, ny, nz) = (c = (n - 1) % (nx * ny * nz); c % (nx * ny) + 1)

function _mglevel(S, nx, ny, nz, coarser::Bool)
    N2 = size(S, 1)
    col = [_rcol(n, nx, ny, nz) for n in 1:N2]
    I, J, V = findnz(S)
    keep = col[I] .== col[J]
    B = lu(sparse(I[keep], J[keep], V[keep], N2, N2))
    red = BitVector([(c = col[n]; i = (c - 1) % nx; j = (c - 1) ÷ nx; iseven(i + j)) for n in 1:N2])
    R = spzeros(0, 0)
    if coarser
        nxc, nyc = nx ÷ 2, ny ÷ 2
        rows = Int[]; cols = Int[]
        for v in 0:1, k in 1:nz, j in 1:ny, i in 1:nx
            push!(cols, v * nx * ny * nz + i + (j - 1) * nx + (k - 1) * nx * ny)
            push!(rows, v * nxc * nyc * nz + (i + 1) ÷ 2 + ((j + 1) ÷ 2 - 1) * nxc + (k - 1) * nxc * nyc)
        end
        R = sparse(rows, cols, ones(length(rows)), 2nxc * nyc * nz, N2)
    end
    return MGLevel(S, B, red, R, nx, ny)
end

"""
    AcousticMG(A, g; levels = :max, smoother = :zebra, ν = 2, ω = 0.8, coarsest = 2)

The multigrid preconditioner of A = I - c J_ac (sparse, unknowns of Model3D on the grid g): Schur reduction to (ρ, Θ),
horizontal coarsening while nx, ny are even and at least 2 `coarsest` (`levels` limits the number of coarsenings;
0: the column preconditioner = one block-Jacobi sweep with ω = 1).
"""
function AcousticMG(A::SparseMatrixCSC, g::Grid3D; levels = :max, smoother::Symbol = :zebra, ν::Int = 2, ω::Real = 0.8,
                    coarsest::Int = 2)
    smoother in (:jacobi, :zebra) || error("smoother: :jacobi or :zebra")
    N = ncell(g)
    t = vcat(1:N, (nunknowns(g) - N + 1):nunknowns(g))
    m = collect(N+1:nunknowns(g)-N)
    Dm = Vector(diag(A[m, m]))
    norm(A[m, m] - spdiagm(0 => Dm), Inf) < 1e-12 * norm(Dm, Inf) || error("AcousticMG: momentum block not diagonal")
    Att = A[t, t]
    norm(Att - I, Inf) < 1e-12 || error("AcousticMG: ρ, Θ block not the identity")
    Atm, Amt = A[t, m], A[m, t]
    dinv = 1 ./ Dm
    S = sparse(I(2N)) - Atm * spdiagm(0 => dinv) * Amt
    lv = MGLevel[]
    nx, ny, nz = g.nx, g.ny, g.nz
    nl = levels === :max ? typemax(Int) : Int(levels)
    while true
        more = length(lv) < nl && iseven(nx) && iseven(ny) && nx ÷ 2 >= coarsest && ny ÷ 2 >= coarsest
        L = _mglevel(S, nx, ny, nz, more)
        push!(lv, L)
        more || break
        S = L.R * S * sparse(L.R')
        nx, ny = nx ÷ 2, ny ÷ 2
    end
    coarse = length(lv) > 1 ? lu(lv[end].S) : nothing
    work = [zeros(size(L.S, 1)) for L in lv for _ in 1:3]
    return AcousticMG(t, m, Atm, Amt, dinv, lv, coarse, smoother, ν, Float64(ω), work)
end

"""ν smoothing sweeps on S x = b (x updated)."""
function _smooth!(x, L::MGLevel, b, M::AcousticMG, r, d)
    for _ in 1:M.ν
        if M.smoother === :jacobi
            r .= b .- L.S * x; d .= L.B \ r
            @. x += M.ω * d
        else
            for colour in (true, false)                          # red columns, then black with the new residual
                r .= b .- L.S * x; d .= L.B \ r
                @inbounds for n in eachindex(x)
                    L.red[n] == colour && (x[n] += d[n])
                end
            end
        end
    end
    return x
end

"""One V-cycle on level l for S_l x = b (x = 0 on entry)."""
function _vcycle!(x, M::AcousticMG, l::Int, b)
    L = M.levels[l]
    r, d = M.work[3l-2], M.work[3l-1]
    if l == length(M.levels) && M.coarse !== nothing
        x .= M.coarse \ b
        return x
    end
    if length(M.levels) == 1                                     # no coarsening: one block-Jacobi sweep (ω = 1)
        x .= L.B \ b
        return x
    end
    fill!(x, 0.0)
    _smooth!(x, L, b, M, r, d)
    r .= b .- L.S * x
    bc = L.R * r
    xc = M.work[3(l+1)]
    _vcycle!(xc, M, l + 1, bc)
    x .+= L.R' * xc
    _smooth!(x, L, b, M, r, d)
    return x
end

"""z = M⁻¹ r: Schur reduction, one V-cycle on S, momenta by back substitution (GMRES preconditioner)."""
function apply!(z, M::AcousticMG, r)
    rt = r[M.t] .- M.Atm * (M.dinv .* r[M.m])
    xt = M.work[3]
    _vcycle!(xt, M, 1, rt)
    z[M.t] .= xt
    z[M.m] .= M.dinv .* (r[M.m] .- M.Amt * xt)
    return z
end
