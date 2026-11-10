# Vertically implicit step of the frozen HEVI loop: per column (i, j) a tridiagonal system for W.
#
# Vertical part J_v (open fractions A_k = Az of the z-face k, s_k = c invV_k / Δz, r = c/Δz):
#   ρ_k = Rρ_k - s_k (A_{k+1} W_{k+1} - A_k W_k),   Θ_k = RΘ_k - s_k (A_{k+1} θ_{k+1} W_{k+1} - A_k θ_k W_k),
#   W_k = RW_k - r (α_k Θ_k - β_k Θ_{k-1}) - (c g/2)(ρ_{k-1} + ρ_k),
# θ_k the frozen face θ, α_k Θ_k - β_k Θ_{k-1} the face pressure difference (divided by J; terrain
# following: s_k = c invV_k / (J Δz)). Eliminating ρ, Θ:
#   W_{k-1} [-r β_k s_{k-1} A_{k-1} θ_{k-1} + (c g/2) s_{k-1} A_{k-1}]
#   + W_k [1 + r A_k θ_k (α_k s_k + β_k s_{k-1}) + (c g/2) A_k (s_k - s_{k-1})]
#   + W_{k+1} [-r α_k s_k A_{k+1} θ_{k+1} - (c g/2) s_k A_{k+1}]
#   = RW_k - r (α_k RΘ_k - β_k RΘ_{k-1}) - (c g/2)(Rρ_{k-1} + Rρ_k).
# Full cells (A = 1, invV = 1) give the 2D column system of src/FV/frozen.jl. Closed faces: W_k = RW_k.
# :product α_k = β_k = (dp/dΘ)_k frozen (exact, the vertical part is linear: Crank–Nicolson);
# :gradp α_k = a_k, β_k = a_{k-1}, a = γ p/Θ at the start of the stage (approximate Jacobian).

mutable struct Column3D
    α::Array{Float64,3}
    β::Array{Float64,3}
    θz::Array{Float64,3}
    factors::Dict{Float64,NTuple{3,Array{Float64,3}}}
    W::Vector{Float64}
    J::SparseMatrixCSC{Float64,Int}                 # implicit = :mask: matrix of the implicit part
    lu::Dict{Float64,Any}                           # its LU factors of I - c J per c
end
Column3D(g::Grid3D) = Column3D(zeros(g.nx, g.ny, g.nz + 1), zeros(g.nx, g.ny, g.nz + 1), zeros(g.nx, g.ny, g.nz + 1),
                               Dict{Float64,NTuple{3,Array{Float64,3}}}(), zeros(g.nz + 1),
                               spzeros(0, 0), Dict{Float64,Any}())

function _column(m::Model3D)
    s = m.cache.column[]
    s isa Column3D || (s = Column3D(m.grid); m.cache.column[] = s)
    return s::Column3D
end

column_coefficients!(::Product, s, m::Model3D, C, z) = (g = m.grid;
    for k in 2:g.nz, j in 1:g.ny, i in 1:g.nx
        s.α[i, j, k] = s.β[i, j, k] = C[cPz(g, i, j, k)] / g.Jc[i, j]
    end; s)
function column_coefficients!(::GradP, s, m::Model3D, C, z)
    g = m.grid
    a(i, j, k) = (Θ = z[iΘ(g, i, j, k)]; m.gas.γ * pressure(m.gas, Θ) / Θ)
    for k in 2:g.nz, j in 1:g.ny, i in 1:g.nx
        s.α[i, j, k] = a(i, j, k) / g.Jc[i, j]; s.β[i, j, k] = a(i, j, k - 1) / g.Jc[i, j]
    end
    return s
end

function frozen_stage_begin!(m::Model3D, C, z)
    s = _column(m)
    g = m.grid
    for k in 1:g.nz+1, j in 1:g.ny, i in 1:g.nx
        s.θz[i, j, k] = C[cθz(g, i, j, k)]
    end
    column_coefficients!(pressure_form(), s, m, C, z)
    empty!(s.factors)
    if m.implicit === :mask
        s.J = implicit_matrix(m, s, C, z)
        empty!(s.lu)
    end
    return s
end

"""Face pressure coefficients (α, β) of α Θ_r - β Θ_l across the x- or y-face: Product the frozen
face derivative, GradP γ p/Θ of the two cells (approximate Jacobian)."""
@inline hcoef(::Product, m, C, z, i, j, k, dir) = (P = dir == 1 ? C[cPx(m.grid, i, j, k)] : C[cPy(m.grid, i, j, k)]; (P, P))
@inline function hcoef(::GradP, m, C, z, i, j, k, dir)
    g = m.grid
    a(ii, jj) = (Θ = z[iΘ(g, ii, jj, k)]; m.gas.γ * pressure(m.gas, Θ) / Θ)
    return dir == 1 ? (a(i, j), a(i - 1, j)) : (a(i, j), a(i, j - 1))
end

"""
Matrix J of the implicit part of HEVI with implicit = :mask: the vertical part (columns) and the
horizontal fast terms through the masked faces (mass, Θ fluxes and pressure differences), as in
`_fast!(Val(:vertical), …)`. Linear in z for :product (exact), approximate Jacobian for :gradp.
"""
function implicit_matrix(m::Model3D, s::Column3D, C, z)
    g = m.grid
    pf = pressure_form()
    I = Int[]; Jc = Int[]; V = Float64[]
    add!(r, c, v) = (v != 0 && (push!(I, r); push!(Jc, c); push!(V, v)); nothing)
    grav = m.gas.g
    for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        sv = g.invV[i, j, k] / g.Jc[i, j]
        # vertical fluxes through the lower (k) and upper (k+1) face of cell (i, j, k)
        if k >= 2
            A = g.Az[i, j, k]
            add!(iρ(g, i, j, k), iW(g, i, j, k), sv * A / g.dz)
            add!(iΘ(g, i, j, k), iW(g, i, j, k), sv * A * s.θz[i, j, k] / g.dz)
        end
        if k + 1 <= g.nz
            A = g.Az[i, j, k+1]
            add!(iρ(g, i, j, k), iW(g, i, j, k + 1), -sv * A / g.dz)
            add!(iΘ(g, i, j, k), iW(g, i, j, k + 1), -sv * A * s.θz[i, j, k+1] / g.dz)
        end
        # W row of the face k: pressure difference and gravity
        if k >= 2 && g.Az[i, j, k] > 0
            r = iW(g, i, j, k)
            add!(r, iΘ(g, i, j, k), -s.α[i, j, k] / g.dz); add!(r, iΘ(g, i, j, k - 1), s.β[i, j, k] / g.dz)
            add!(r, iρ(g, i, j, k - 1), -grav / 2); add!(r, iρ(g, i, j, k), -grav / 2)
        end
        # masked x- and y-faces (left faces of cell (i, j, k)): flux between the two cells, pressure
        for (dir, M, A, h, iu, θc, l) in ((1, g.Mx, g.Ax, g.dx, iU(g, i, j, k), C[cθx(g, i, j, k)], (ip(g, i - 1), j)),
                                          (2, g.My, g.Ay, g.dy, iV(g, i, j, k), C[cθy(g, i, j, k)], (i, jp(g, j - 1))))
            (M[i, j, k] > 0 && A[i, j, k] > 0) || continue
            a = A[i, j, k]
            add!(iρ(g, i, j, k), iu, sv * a / h);                 add!(iρ(g, l[1], l[2], k), iu, -g.invV[l[1], l[2], k] * a / h)
            add!(iΘ(g, i, j, k), iu, sv * a * θc / h);            add!(iΘ(g, l[1], l[2], k), iu, -g.invV[l[1], l[2], k] * a * θc / h)
            α, β = hcoef(pf, m, C, z, i, j, k, dir)
            add!(iu, iΘ(g, i, j, k), -α / h); add!(iu, iΘ(g, l[1], l[2], k), β / h)
        end
    end
    return sparse(I, Jc, V, m.nq, m.nq)
end

"""LU factors (Thomas) of the column systems for c, cached per stage."""
function _factor(s::Column3D, m::Model3D, c)
    get!(s.factors, Float64(c)) do
        g = m.grid
        nz = g.nz
        r = c / g.dz; cg = c * m.gas.g / 2
        sub = zeros(nz + 1, g.nx, g.ny); cp = zeros(nz + 1, g.nx, g.ny); iden = zeros(nz + 1, g.nx, g.ny)
        @inbounds for j in 1:g.ny, i in 1:g.nx, k in 2:nz
            A = g.Az[i, j, k]
            if A == 0
                a, d, b = 0.0, 1.0, 0.0
            else
                sk, sm = c * g.invV[i, j, k] / (g.Jc[i, j] * g.dz), c * g.invV[i, j, k-1] / (g.Jc[i, j] * g.dz)
                Am, Ap = g.Az[i, j, k-1], g.Az[i, j, k+1]
                a = k > 2 ? -r * s.β[i, j, k] * sm * Am * s.θz[i, j, k-1] + cg * sm * Am : 0.0
                d = 1 + r * A * s.θz[i, j, k] * (s.α[i, j, k] * sk + s.β[i, j, k] * sm) + cg * A * (sk - sm)
                b = k < nz ? -r * s.α[i, j, k] * sk * Ap * s.θz[i, j, k+1] - cg * sk * Ap : 0.0
            end
            den = k == 2 ? d : d - a * cp[k-1, i, j]
            sub[k, i, j] = a; iden[k, i, j] = 1 / den; cp[k, i, j] = b / den
        end
        (sub, cp, iden)
    end
end

"""Y = (I - c J_v)⁻¹ R (U, V rows: identity)."""
function column_solve!(Y, s::Column3D, m::Model3D, c, R)
    g = m.grid
    nz, dz = g.nz, g.dz
    r = c / dz; cg = c * m.gas.g / 2
    sub, cp, iden = _factor(s, m, c)
    W = s.W
    Y .= R
    @inbounds for j in 1:g.ny, i in 1:g.nx
        for k in 2:nz
            rhs = g.Az[i, j, k] == 0 ? R[iW(g, i, j, k)] :
                  R[iW(g, i, j, k)] - r * (s.α[i, j, k] * R[iΘ(g, i, j, k)] - s.β[i, j, k] * R[iΘ(g, i, j, k - 1)]) -
                  cg * (R[iρ(g, i, j, k - 1)] + R[iρ(g, i, j, k)])
            W[k] = k == 2 ? rhs * iden[k, i, j] : (rhs - sub[k, i, j] * W[k-1]) * iden[k, i, j]
        end
        for k in nz-1:-1:2
            W[k] -= cp[k, i, j] * W[k+1]
        end
        W[1] = 0.0; W[nz+1] = 0.0
        for k in 1:nz
            sk = c * g.invV[i, j, k] / (g.Jc[i, j] * dz)
            Fb = g.Az[i, j, k] * W[k]; Ft = g.Az[i, j, k+1] * W[k+1]
            Y[iρ(g, i, j, k)] = R[iρ(g, i, j, k)] - sk * (Ft - Fb)
            Y[iΘ(g, i, j, k)] = R[iΘ(g, i, j, k)] - sk * (s.θz[i, j, k+1] * Ft - s.θz[i, j, k] * Fb)
        end
        for k in 2:nz
            Y[iW(g, i, j, k)] = W[k]
        end
    end
    return Y
end

"""Linearly implicit Crank–Nicolson step of the implicit part: z ← z + (I - dt/2 J)⁻¹ dt G(z),
G = `frozen_fast!(…, :vertical, …)`; J the column systems (:column) or the sparse matrix (:mask)."""
function frozen_vertical_step!(z, m::Model3D, C, dt, k, y)
    s = _column(m)
    frozen_fast!(k, m, C, z, :vertical, k)
    @. k *= dt
    if m.implicit === :mask
        F = get!(() -> lu(I - (dt / 2) * s.J), s.lu, Float64(dt / 2))
        ldiv!(y, F, k)
    else
        column_solve!(y, s, m, dt / 2, k)
    end
    @. z += y
    return z
end
