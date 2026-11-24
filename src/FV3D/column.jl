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
    factors::Dict{NTuple{3,Float64},NTuple{3,Array{Float64,3}}}   # Thomas factors per (ct, cw, wm)
    W::Vector{Float64}
    J::SparseMatrixCSC{Float64,Int}                 # implicit = :mask, :amf: matrix of the implicit acoustics
    Jadv::SparseMatrixCSC{Float64,Int}              # implicit = :amf: masked advection (simple upwind)
    lu::Dict{Any,Any}                               # factors of the restricted systems per (ct, cw, wm), weights, work
    idx::Vector{Int}                                # unknowns of the columns at masked faces (+ masked U, V)
    ridx::Vector{Int}                               # rows of Jadv (masked faces)
    its::Vector{Int}                                # FV3D_MASK_SOLVER = :gmres: iterations per solve
end
Column3D(g::Grid3D) = Column3D(zeros(g.nx, g.ny, g.nz + 1), zeros(g.nx, g.ny, g.nz + 1), zeros(g.nx, g.ny, g.nz + 1),
                               Dict{NTuple{3,Float64},NTuple{3,Array{Float64,3}}}(), zeros(g.nz + 1),
                               spzeros(0, 0), spzeros(0, 0), Dict{Any,Any}(), Int[], Int[], Int[])

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
    if m.implicit in (:mask, :amf)
        s.J = implicit_matrix(m, s, C, z)
        isempty(s.idx) && (s.idx = masked_unknowns(m))
        if m.implicit === :amf
            s.Jadv = adv_matrix(m, C)
            s.ridx = unique(s.Jadv.rowval)
        end
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

"""
Unknowns coupled by the horizontal part of the implicit matrix: ρ, Θ, W of every column with a
cell next to a masked x- or y-face, and the masked U, V. The remaining columns are independent
tridiagonal systems (`column_solve!`); only this block needs the sparse LU.
"""
function masked_unknowns(m::Model3D)
    g = m.grid
    col = falses(g.nx, g.ny)
    uv = Int[]
    for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        if g.Mx[i, j, k] > 0 && g.Ax[i, j, k] > 0
            col[i, j] = col[ip(g, i - 1), j] = true; push!(uv, iU(g, i, j, k))
        end
        if g.My[i, j, k] > 0 && g.Ay[i, j, k] > 0
            col[i, j] = col[i, jp(g, j - 1)] = true; push!(uv, iV(g, i, j, k))
        end
    end
    idx = Int[]
    for j in 1:g.ny, i in 1:g.nx
        col[i, j] || continue
        for k in 1:g.nz
            push!(idx, iρ(g, i, j, k), iΘ(g, i, j, k))
            k > 1 && push!(idx, iW(g, i, j, k))
        end
    end
    return sort!(vcat(idx, uv))
end

"""
Face weights of the implicit part: a flux through a face has one weight for both cells (conservative). wt for the
ρ, Θ fluxes and wf for the momentum rows of the face: (ct, cw), or wm on the faces of masked cells (`_row`) if wm is
not NaN (:cs with FV3D_CS_MASK_EULER).
"""
@inline _wt(g, ct, wm, d, i, j, k) = (!isnan(wm) && _row(g, d, i, j, k)) ? wm : ct
@inline _wf(g, cw, wm, d, i, j, k) = (!isnan(wm) && _row(g, d, i, j, k)) ? wm : cw

"""LU factors (Thomas) of the column systems for the weights (ct, cw, wm), cached per stage."""
function _factor(s::Column3D, m::Model3D, ct, cw, wm = NaN)
    get!(s.factors, (Float64(ct), Float64(cw), Float64(wm))) do
        g = m.grid
        nz = g.nz
        sub = zeros(nz + 1, g.nx, g.ny); cp = zeros(nz + 1, g.nx, g.ny); iden = zeros(nz + 1, g.nx, g.ny)
        @inbounds for j in 1:g.ny, i in 1:g.nx, k in 2:nz
            A = g.Az[i, j, k]
            if A == 0
                a, d, b = 0.0, 1.0, 0.0
            else
                cwk = _wf(g, cw, wm, 3, i, j, k)
                r = cwk / g.dz; cg = cwk * m.gas.g / 2
                vk, vm = g.invV[i, j, k] / (g.Jc[i, j] * g.dz), g.invV[i, j, k-1] / (g.Jc[i, j] * g.dz)
                t0 = _wt(g, ct, wm, 3, i, j, k)                         # flux weight of the face k
                tm = k > 2 ? _wt(g, ct, wm, 3, i, j, k - 1) : 0.0       # … of the faces k - 1, k + 1
                tp = k < nz ? _wt(g, ct, wm, 3, i, j, k + 1) : 0.0
                Am, Ap = g.Az[i, j, k-1], g.Az[i, j, k+1]
                a = k > 2 ? (-r * s.β[i, j, k] * s.θz[i, j, k-1] + cg) * tm * vm * Am : 0.0
                d = 1 + t0 * (r * A * s.θz[i, j, k] * (s.α[i, j, k] * vk + s.β[i, j, k] * vm) + cg * A * (vk - vm))
                b = k < nz ? (-r * s.α[i, j, k] * s.θz[i, j, k+1] - cg) * tp * vk * Ap : 0.0
            end
            den = k == 2 ? d : d - a * cp[k-1, i, j]
            sub[k, i, j] = a; iden[k, i, j] = 1 / den; cp[k, i, j] = b / den
        end
        (sub, cp, iden)
    end
end

"""Y = (I - D J_v)⁻¹ R, D the face weights (ct, cw, wm) (`_wt`, `_wf`; U, V rows: identity)."""
column_solve!(Y, s::Column3D, m::Model3D, c, R) = column_solve!(Y, s, m, c, c, R)
function column_solve!(Y, s::Column3D, m::Model3D, ct, cw, R, wm = NaN)
    g = m.grid
    nz, dz = g.nz, g.dz
    sub, cp, iden = _factor(s, m, ct, cw, wm)
    W = s.W
    Y .= R
    @inbounds for j in 1:g.ny, i in 1:g.nx
        for k in 2:nz
            cwk = _wf(g, cw, wm, 3, i, j, k)
            r = cwk / dz; cg = cwk * m.gas.g / 2
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
            vk = g.invV[i, j, k] / (g.Jc[i, j] * dz)
            Fb = (k > 1 ? _wt(g, ct, wm, 3, i, j, k) : 0.0) * g.Az[i, j, k] * W[k]
            Ft = (k < nz ? _wt(g, ct, wm, 3, i, j, k + 1) : 0.0) * g.Az[i, j, k+1] * W[k+1]
            Y[iρ(g, i, j, k)] = R[iρ(g, i, j, k)] - vk * (Ft - Fb)
            Y[iΘ(g, i, j, k)] = R[iΘ(g, i, j, k)] - vk * (s.θz[i, j, k+1] * Ft - s.θz[i, j, k] * Fb)
        end
        for k in 2:nz
            Y[iW(g, i, j, k)] = W[k]
        end
    end
    return Y
end

"""(kind, i, j, k) of unknown n: kind 0 ρ, 1 U, 2 V, 3 W, 4 Θ."""
@inline function _unknown(g::Grid3D, n)
    N = ncell(g); nW = g.nx * g.ny * (g.nz - 1)
    kind, c = n <= N ? (0, n - 1) : n <= 2N ? (1, n - N - 1) : n <= 3N ? (2, n - 2N - 1) :
              n <= 3N + nW ? (3, n - 3N - 1) : (4, n - 3N - nW - 1)
    kind == 3 && return (3, c % g.nx + 1, (c ÷ g.nx) % g.ny + 1, c ÷ (g.nx * g.ny) + 2)
    return (kind, c % g.nx + 1, (c ÷ g.nx) % g.ny + 1, c ÷ (g.nx * g.ny) + 1)
end

"""
Weight vectors of D G_v for (ct, cw, wm): dr on the momentum rows (wf of the face; 1 on ρ, Θ), dc on the momentum
columns of the ρ, Θ fluxes (wt of the face; 1 on ρ, Θ), da on the rows of the masked advection (ρ, Θ: wm or ct;
momentum: dr). D G_v(z) = thermo(dc .* z) + dr .* momentum(z) + da .* adv(z), D J = Dr J_ac Dc + Da J_adv.
"""
function _weights(m::Model3D, ct, cw, wm)
    g = m.grid
    n = m.nq
    dr = ones(n); dc = ones(n); da = ones(n)
    @inbounds for l in 1:n
        kind, i, j, k = _unknown(g, l)
        if kind in (0, 4)
            da[l] = isnan(wm) ? ct : wm
        else
            dr[l] = da[l] = _wf(g, cw, wm, kind, i, j, k); dc[l] = _wt(g, ct, wm, kind, i, j, k)
        end
    end
    return dr, dc, da
end
_weights(s::Column3D, m::Model3D, key) = get!(() -> _weights(m, key...), s.lu, (:w, key...))

"""
    implicit_solve!(y, m::Model3D, ct, cw, k, wm = NaN)

y = P⁻¹ k with P ≈ I - D J of the stage, D the face weights (`_weights`; ct on the ρ, Θ fluxes and cw on the momentum
rows, wm on the faces of masked cells if not NaN): J the column systems (:column), the sparse matrix (:mask),
P = (I - D J)(I - D J_adv) (:amf, approximate matrix factorization), or with FV3D_MASK_SOLVER = :lu / :gmres the
coupled P = I - D (J + J_adv) on the masked unknowns.
"""
function implicit_solve!(y, m::Model3D, ct, cw, k, wm = NaN; solver::Symbol = FV3D_MASK_SOLVER[])
    s = _column(m)
    solver = m.implicit === :amf ? solver : :amf
    solver in (:amf, :lu, :gmres) || error("FV3D_MASK_SOLVER: :amf, :lu or :gmres")
    key = (Float64(ct), Float64(cw), Float64(wm))
    column_solve!(y, s, m, ct, cw, k, wm)             # the independent columns (and a first guess)
    m.implicit in (:mask, :amf) || return y
    isempty(s.idx) && isempty(s.ridx) && return y
    if solver !== :amf
        # I - D (J_ac + J_adv) on the masked unknowns ∪ the masked advection rows (no AMF)
        F = get!(s.lu, key) do
            g = m.grid
            dr, dc, da = _weights(s, m, key)
            DJa = Diagonal(dr) * s.J * Diagonal(dc); DJd = Diagonal(da) * s.Jadv
            DJ = DJa + DJd
            id2 = sort!(union(s.idx, s.ridx))
            out = setdiff(1:m.nq, id2)
            Id = sparse(1.0I, length(id2), length(id2))
            A = Id - DJ[id2, id2]
            if solver === :lu
                (lu(A), id2, DJ[id2, out], out, nothing, nothing, nothing, nothing)
            else
                tx, ty = FV3D_MASK_TILES[]
                sets, own = ras_sets(g, id2, min(tx, g.nx), min(ty, g.ny), FV3D_MASK_OVERLAP[])
                loc = FV3D_MASK_LOCAL[]
                loc in (:amf, :lu) || error("FV3D_MASK_LOCAL: :amf or :lu")
                As = loc === :lu ? (A,) : (Id - DJa[id2, id2], Id - DJd[id2, id2])
                N = ncell(g); nW = g.nx * g.ny * (g.nz - 1)
                rc = [n for n in id2 if n <= N || n > 3N + nW]           # ρ, Θ rows (flux form)
                (RAS(As, sets, own), id2, DJ[id2, out], out, A, GMRESWork(length(id2), 30), rc, DJ[rc, :])
            end
        end
        P, id2, DJo, out, A, W, rc, DJr = F
        rhs = k[id2] .+ DJo * y[out]
        if solver === :lu
            y[id2] = P \ rhs
        else
            x = y[id2]                                 # initial guess: the column solve
            push!(s.its, gmres!(x, A, rhs, P, W; tol = FV3D_MASK_TOL[]))
            y[id2] = x
            # ρ, Θ from the fluxes of the iterate: conservative to roundoff whatever the GMRES residual
            y[rc] = k[rc] .+ DJr * y
        end
    else
        F = get!(s.lu, key) do
            dr, dc, da = _weights(s, m, key)
            Fac = isempty(s.idx) ? nothing : lu(I - (Diagonal(dr) * s.J * Diagonal(dc))[s.idx, s.idx])
            amf = m.implicit === :amf && !isempty(s.ridx)
            DJd = amf ? Diagonal(da) * s.Jadv : nothing
            Arow = amf ? DJd[s.ridx, :] : nothing; Arr = amf ? DJd[s.ridx, s.ridx] : nothing
            (Fac, amf ? lu(I - Arr) : nothing, Arow, Arr)
        end
        # acoustic factor: independent columns tridiagonal, the masked block sparse
        isempty(s.idx) || (y[s.idx] = F[1] \ k[s.idx])
        # advection factor (rows r = masked faces): x_r = (I - D A_rr)⁻¹ (y_r + D A_ru y_u), x_u = y_u
        if F[2] !== nothing
            yr = y[s.ridx]
            b = yr .+ (F[3] * y .- F[4] * yr)
            y[s.ridx] = F[2] \ b
        end
    end
    return y
end

"""Linearly implicit Crank–Nicolson step of the implicit part (:hevi): z ← z + P⁻¹ dt G(z),
G = `frozen_fast!(…, :vertical, …)`, P ≈ I - β dt J (`implicit_solve!` with ct = cw = β dt, β = FV3D_HEVI_BETA)."""
function frozen_vertical_step!(z, m::Model3D, C, dt, k, y)
    frozen_fast!(k, m, C, z, :vertical, k)
    @. k *= dt
    c = FV3D_HEVI_BETA[] * dt                          # β = 1/2: Crank–Nicolson; > 1/2 off-centred (experimental)
    implicit_solve!(y, m, c, c, k)
    @. z += y
    return z
end

"""Faces of the masked cells in the :cs hooks linearly implicit Euler (FV3D_CS_MASK_EULER and a masked model)?"""
_cs_euler(m::Model3D) = FV3D_CS_MASK_EULER[] && m.implicit in (:mask, :amf)

"""out = D G_v(z) with the face weights (ct, cw, wm) (`_weights`): ρ, Θ fluxes of the weighted momenta, weighted
momentum rows and masked advection."""
function _vertical_weighted!(out, m::Model3D, C, z, ct, cw, wm)
    s = _column(m)
    dr, dc, da = _weights(s, m, (Float64(ct), Float64(cw), Float64(wm)))
    c = m.cache
    zs = get!(() -> similar(z), s.lu, :zs)::Vector{Float64}
    t = get!(() -> similar(z), s.lu, :t)::Vector{Float64}
    pf = pressure_form()
    fill!(out, 0.0)
    @. zs = dc * z                                     # momenta weighted by the face weights of the fluxes
    thermo_v!(out, m, C, zs)
    m.implicit in (:mask, :amf) && thermo_h!(out, m, C, zs, Masked())
    fill!(t, 0.0)
    prepare_pressure!(pf, m, z); momentum_v!(t, m, C, z, pf)
    m.implicit in (:mask, :amf) && momentum_h!(t, m, C, z, pf, Masked())
    @. out += dr * t
    fill!(t, 0.0)
    _adv!(t, m, C, z); _advθ!(t, m, C, z)
    @. out += da * t
    return out
end

"""
out = D G_v(z) (:cs, :ei, :wrf): the implicit part with the ρ, Θ fluxes weighted by ct, the momentum rows by cw. With
FV3D_CS_MASK_EULER (masked models) the faces of the masked cells have the weight 0: there the implicit part is
integrated by `frozen_vertical_implicit!` alone (linearly implicit Euler).
"""
frozen_vertical_rhs!(out, m::Model3D, C, z, ct, cw) = _vertical_weighted!(out, m, C, z, ct, cw, _cs_euler(m) ? 0.0 : NaN)
"""
z ← z + Y, P Y = D G_v(z) (`implicit_solve!`). With FV3D_CS_MASK_EULER (masked models) the faces of the masked cells
have the weight 2ct, the full small step h of the merged :cs loop (ct = h/2): linearly implicit Euler there.
"""
function frozen_vertical_implicit!(z, m::Model3D, C, ct, cw, k, y)
    wm = _cs_euler(m) ? 2ct : NaN
    _vertical_weighted!(k, m, C, z, ct, cw, wm)
    implicit_solve!(y, m, ct, cw, k, wm)
    @. z += y
    return z
end
"""y = P⁻¹ R (:wrf)."""
frozen_vertical_solve!(y, m::Model3D, C, ct, cw, R) = implicit_solve!(y, m, ct, cw, R)
