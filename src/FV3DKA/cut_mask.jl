# Masked implicit small step of CutKA with switchable linear algebra (prec = :mask, vertical = :explicit).
#
# The implicit part of the small step (`ka_vertical_step!`) acts only on the faces of the masked cells, horizontal and
# vertical together (vertical = :explicit: outside the mask everything is explicit, gravity explicit everywhere). Its
# matrix A = I - c (J_ac + J_adv) is preconditioned by the AMF product (I - c J_ac)(I - c J_adv), two sparse systems:
#
#   acoustic   without gravity the momentum rows depend on Θ only: S_Θ = I - c² J_Θm J_mΘ (Θ-only Schur complement,
#              ASAM BLOCKMULTI solveM), r̃_Θ = r_Θ + c J_Θm r_m, then m = r_m + c J_mΘ Θ, ρ = r_ρ + c J_ρm m;
#   advection  the masked upwind transport T = I - c A_up(a±), the same matrix for Θ (ρ term on the right-hand side,
#              ρ unchanged) and for the six shifted momenta (shift technique; back to the faces with the volume
#              weights: an approximate inverse of the face system, corrected by the outer GMRES).
#
# Both are 7-point cell stencils (`StencilSys`, the identity outside the mask), solved by an inner method chosen per
# system: :jacobi, :cg (S_Θ only: symmetric after scaling the rows by the cell volumes), :bicgstab, :hostlu (assembled
# on the host, sparse LU: the exact reference; tile-local with MPI). J_Θm, J_mΘ, J_ρm are applied matrix-free by
# `ka_vertical!(…; parts = :acoustic)`; the outer GMRES and the flux recomputation of ρ, Θ are unchanged.

const STC, STW, STE, STS, STN, STB, STT = 1, 2, 3, 4, 5, 6, 7

"""7-point stencil system on the cells of the tile: coefficients St (nxh, nyh, nz, 7), row scaling D (CG), work fields
for nv right-hand sides."""
mutable struct StencilSys{A,B}
    St::A; D::B
    nv::Int
    r::A; z::A; p::A; q::A; s::A; t::A; rh::A; v::A
    host::Any                                       # :hostlu: the LU of the assembled matrix
end
function StencilSys(m::CutKA, nv::Int)
    g = m.grid; be = KA.get_backend(m)
    z(n) = KA.zeros(be, Float64, g.nxh, g.nyh, g.nz, n)
    return StencilSys(z(7), KA.zeros(be, Float64, g.nxh, g.nyh, g.nz), nv, z(nv), z(nv), z(nv), z(nv), z(nv), z(nv), z(nv), z(nv),
                      nothing)
end

@kernel function _st_apply!(y, @Const(St), @Const(x), nz)
    i, j, k, v = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        s = St[I, J, k, STC] * x[I, J, k, v] + St[I, J, k, STW] * x[I-1, J, k, v] + St[I, J, k, STE] * x[I+1, J, k, v] +
            St[I, J, k, STS] * x[I, J-1, k, v] + St[I, J, k, STN] * x[I, J+1, k, v]
        k > 1 && (s += St[I, J, k, STB] * x[I, J, k-1, v])
        k < nz && (s += St[I, J, k, STT] * x[I, J, k+1, v])
        y[I, J, k, v] = s
    end
end
"""Red–black Gauss–Seidel half sweep (colour by the parity of the global i + j + k)."""
@kernel function _st_gs!(x, @Const(St), @Const(b), nz, i0, j0, colour)
    i, j, k, v = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds if iseven(i0 + i + j0 + j + k) == colour
        s = b[I, J, k, v] - (St[I, J, k, STW] * x[I-1, J, k, v] + St[I, J, k, STE] * x[I+1, J, k, v] +
                             St[I, J, k, STS] * x[I, J-1, k, v] + St[I, J, k, STN] * x[I, J+1, k, v])
        k > 1 && (s -= St[I, J, k, STB] * x[I, J, k-1, v])
        k < nz && (s -= St[I, J, k, STT] * x[I, J, k+1, v])
        x[I, J, k, v] = s / St[I, J, k, STC]
    end
end
"""y = S x (interior; the halo of x exchanged)."""
function st_apply!(y, sys::StencilSys, m::CutKA, x)
    g = m.grid
    exchange!(m.dec, x, g)
    _launch!(m, _st_apply!, (g.nxl, g.nyl, g.nz, size(x, 4)), y, sys.St, x, g.nz)
    return y
end
_int(m::CutKA, A) = (g = m.grid; view(A, HALO+1:HALO+g.nxl, HALO+1:HALO+g.nyl, :, :))
_sdot(m::CutKA, a, b) = allreduce_sum(m.dec, dot(_int(m, a), _int(m, b)))
_diagv(sys::StencilSys, m::CutKA) = view(sys.St, :, :, :, STC:STC)

"""x ← S⁻¹ b approximately with the method `method` (initial guess x); returns the iterations."""
function st_solve!(x, sys::StencilSys, m::CutKA, b, method::Symbol; tol = 1e-6, maxit = 100, ω = 0.8)
    if method === :jacobi
        dg = _diagv(sys, m)
        for it in 1:maxit
            st_apply!(sys.r, sys, m, x)
            @. sys.r = b - sys.r
            @. x += ω * sys.r / dg
        end
        return maxit
    elseif method === :cg
        return _st_cg!(x, sys, m, b; tol, maxit)
    elseif method === :bicgstab
        return _st_bicgstab!(x, sys, m, b; tol, maxit)
    elseif method === :hostlu
        return _st_hostlu!(x, sys, m, b)
    elseif method === :gs                           # red–black Gauss–Seidel, maxit sweeps
        g = m.grid
        for _ in 1:maxit, colour in (true, false)
            exchange!(m.dec, x, g)
            _launch!(m, _st_gs!, (g.nxl, g.nyl, g.nz, size(x, 4)), x, sys.St, b, g.nz, g.i0, g.j0, colour)
        end
        return maxit
    elseif method === :none                         # identity (diagnostics: the factor left out)
        x .= b
        return 0
    end
    error("stencil solver: :jacobi, :gs, :cg, :bicgstab, :hostlu or :none")
end

"""PCG on the row-scaled system D S x = D b (D the cell volumes, symmetric), Jacobi preconditioner."""
function _st_cg!(x, sys::StencilSys, m::CutKA, b; tol, maxit)
    r, z, p, q = sys.r, sys.z, sys.p, sys.q
    D = reshape(sys.D, size(sys.D)..., 1)
    dg = _diagv(sys, m)
    st_apply!(q, sys, m, x)
    @. r = D * (b - q)
    nb = sqrt(_sdot(m, D .* b, D .* b)); nb == 0 && (fill!(x, 0.0); return 0)
    @. z = r / (D * dg)
    p .= z
    rz = _sdot(m, r, z)
    for it in 1:maxit
        st_apply!(q, sys, m, p); @. q *= D
        α = rz / _sdot(m, p, q)
        @. x += α * p; @. r -= α * q
        sqrt(_sdot(m, r, r)) <= tol * nb && return it
        @. z = r / (D * dg)
        rzn = _sdot(m, r, z)
        @. p = z + (rzn / rz) * p
        rz = rzn
    end
    return maxit
end

"""Right-preconditioned (Jacobi) BiCGStab on S x = b (all fields as one vector)."""
function _st_bicgstab!(x, sys::StencilSys, m::CutKA, b; tol, maxit)
    r, rh, p, v, s, t, ph = sys.r, sys.rh, sys.p, sys.v, sys.s, sys.t, sys.z
    st_apply!(r, sys, m, x); @. r = b - r
    nb = sqrt(_sdot(m, b, b)); nb == 0 && (fill!(x, 0.0); return 0)
    sqrt(_sdot(m, r, r)) <= tol * nb && return 0
    rh .= r; fill!(p, 0.0); fill!(v, 0.0)
    dg = _diagv(sys, m)
    ρ = α = ω = 1.0
    for it in 1:maxit
        ρn = _sdot(m, rh, r)
        β = (ρn / ρ) * (α / ω); ρ = ρn
        @. p = r + β * (p - ω * v)
        @. ph = p / dg
        st_apply!(v, sys, m, ph)
        α = ρ / _sdot(m, rh, v)
        @. s = r - α * v
        if sqrt(_sdot(m, s, s)) <= tol * nb
            @. x += α * ph
            return it
        end
        @. sys.q = s / dg
        st_apply!(t, sys, m, sys.q)
        ω = _sdot(m, t, s) / _sdot(m, t, t)
        @. x += α * ph + ω * sys.q
        @. r = s - ω * t
        sqrt(_sdot(m, r, r)) <= tol * nb && return it
    end
    return maxit
end

"""Exact solve with the LU of the assembled stencil (host; tile-local, the couplings across tile edges dropped unless
the tile is periodic in that direction)."""
function _st_hostlu!(x, sys::StencilSys, m::CutKA, b)
    g = m.grid
    nxl, nyl, nz = g.nxl, g.nyl, g.nz
    if sys.host === nothing
        St = Array(sys.St)
        id(i, j, k) = i + (j - 1) * nxl + (k - 1) * nxl * nyl
        px, py = g.px == 1, g.py == 1
        I = Int[]; J = Int[]; V = Float64[]
        for k in 1:nz, j in 1:nyl, i in 1:nxl
            r = id(i, j, k); IH, JH = i + HALO, j + HALO
            push!(I, r); push!(J, r); push!(V, St[IH, JH, k, STC])
            for (sl, di, dj, dk) in ((STW, -1, 0, 0), (STE, 1, 0, 0), (STS, 0, -1, 0), (STN, 0, 1, 0), (STB, 0, 0, -1), (STT, 0, 0, 1))
                a = St[IH, JH, k, sl]; a == 0 && continue
                ii, jj, kk = i + di, j + dj, k + dk
                (1 <= kk <= nz) || continue
                if !(1 <= ii <= nxl); px || continue; ii = mod1(ii, nxl); end
                if !(1 <= jj <= nyl); py || continue; jj = mod1(jj, nyl); end
                push!(I, r); push!(J, id(ii, jj, kk)); push!(V, a)
            end
        end
        sys.host = lu(sparse(I, J, V, nxl * nyl * nz, nxl * nyl * nz))
    end
    bh = Array(_int(m, b)); xh = similar(bh)
    for v in 1:size(bh, 4)
        xh[:, :, :, v] .= reshape(sys.host \ vec(bh[:, :, :, v]), nxl, nyl, nz)
    end
    copyto!(_int(m, x), xh)
    return 1
end

# ------------------------------------------------------------------ the two stencils

"""S_Θ = I - c² J_Θm J_mΘ over the masked faces (horizontal: Mx, My; vertical: faces of masked cells), no gravity;
D the cell volumes (CG row scaling)."""
@kernel function _ms_acoustic!(St, D, @Const(Cf), G, c, dx, dy, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        for s in 1:7; St[I, J, k, s] = 0.0; end
        St[I, J, k, STC] = 1.0
        iV = G.iV[I, J, k]
        D[I, J, k] = iV > 0 ? G.Vf[I, J, k] * G.dzc[k] * dx * dy : 1.0
        if iV > 0
            c2 = c * c
            for (a, M, P, θf, slot, h) in ((G.Ax[I, J, k], G.Mx[I, J, k], Cf[I, J, k, KPX], Cf[I, J, k, KθX], STW, dx),
                                           (G.Ax[I+1, J, k], G.Mx[I+1, J, k], Cf[I+1, J, k, KPX], Cf[I+1, J, k, KθX], STE, dx),
                                           (G.Ay[I, J, k], G.My[I, J, k], Cf[I, J, k, KPY], Cf[I, J, k, KθY], STS, dy),
                                           (G.Ay[I, J+1, k], G.My[I, J+1, k], Cf[I, J+1, k, KPY], Cf[I, J+1, k, KθY], STN, dy))
                if a > 0 && M > 0
                    w = c2 * iV * a * P * θf / h^2
                    St[I, J, k, STC] += w; St[I, J, k, slot] -= w
                end
            end
            if k >= 2 && G.Az[I, J, k] > 0 && _wzI(G, nz, I, J, k, false) > 0
                w = c2 * iV * G.Az[I, J, k] * Cf[I, J, k, KPZ] * Cf[I, J, k, KθZ] / (G.dzc[k] * G.dzf[k])
                St[I, J, k, STC] += w; St[I, J, k, STB] -= w
            end
            if k < nz && G.Az[I, J, k+1] > 0 && _wzI(G, nz, I, J, k + 1, false) > 0
                w = c2 * iV * G.Az[I, J, k+1] * Cf[I, J, k+1, KPZ] * Cf[I, J, k+1, KθZ] / (G.dzc[k] * G.dzf[k+1])
                St[I, J, k, STC] += w; St[I, J, k, STT] -= w
            end
        end
    end
end

"""T = I - c A_up over the faces of masked cells (upwind coefficients a± of C, as the masked Θ transport)."""
@kernel function _ms_transport!(St, @Const(Cf), G, c, dx, dy, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        for s in 1:7; St[I, J, k, s] = 0.0; end
        St[I, J, k, STC] = 1.0
        s = c * G.iV[I, J, k]
        if s > 0
            if _mfx(G, I, J, k)                        # lower x-face: + (a⁺ φ_W + a⁻ φ_c)/Δx
                St[I, J, k, STW] -= s * Cf[I, J, k, KAXP] / dx; St[I, J, k, STC] -= s * Cf[I, J, k, KAXM] / dx
            end
            if _mfx(G, I + 1, J, k)                    # upper x-face: - (a⁺ φ_c + a⁻ φ_E)/Δx
                St[I, J, k, STC] += s * Cf[I+1, J, k, KAXP] / dx; St[I, J, k, STE] += s * Cf[I+1, J, k, KAXM] / dx
            end
            if _mfy(G, I, J, k)
                St[I, J, k, STS] -= s * Cf[I, J, k, KAYP] / dy; St[I, J, k, STC] -= s * Cf[I, J, k, KAYM] / dy
            end
            if _mfy(G, I, J + 1, k)
                St[I, J, k, STC] += s * Cf[I, J+1, k, KAYP] / dy; St[I, J, k, STN] += s * Cf[I, J+1, k, KAYM] / dy
            end
            dz = G.dzc[k]
            if k >= 2 && _mfz(G, I, J, k)
                St[I, J, k, STB] -= s * Cf[I, J, k, KAZP] / dz; St[I, J, k, STC] -= s * Cf[I, J, k, KAZM] / dz
            end
            if k < nz && _mfz(G, I, J, k + 1)
                St[I, J, k, STC] += s * Cf[I, J, k+1, KAZP] / dz; St[I, J, k, STT] += s * Cf[I, J, k+1, KAZM] / dz
            end
        end
    end
end

# ------------------------------------------------------------------ the AMF preconditioner

"""Masked AMF preconditioner: the two stencil systems, their solvers, work fields."""
struct MaskAMF{SA,SB,A,F}
    ac::SA; adv::SB
    method::NamedTuple
    X::A; Y::A                                      # state-shaped work
    θ::F; b1::F; f7::F; x7::F                      # cell fields: Θ, its right-hand side (1), Θ + six shifted momenta (7)
    its::Vector{NTuple{2,Int}}
    K::Vector{A}                                    # momentum BiCGStab work (state-shaped)
end
"""
    MaskAMF(m; acoustic = :cg, advection = :bicgstab, momenta = :identity, tol = 1e-3, maxit = 50, sweeps = 2,
            outer = :gmres)

The preconditioner `prec = :mask` of the masked small step (vertical = :explicit): inner solvers of the acoustic (S_Θ)
and the advection (T) system (:skip leaves a factor out), the momenta of the advection factor (`:krylov` BiCGStab on
the masked momentum advection, matrix-free; `:shift` T on the shifted cell momenta, back to the faces (approximate);
`:identity`), the relative tolerance and iteration limit of the inner solvers, `sweeps` of `advection = :gs` (red–black
Gauss–Seidel), `outer`: `:gmres` (the AMF product preconditions GMRES on the masked system) or `:none` (the AMF product
is the solve, as ASAM IntRos: acoustics globally, then advection globally; AMF splitting error c² J_ac J_adv).
"""
function MaskAMF(m::CutKA; acoustic::Symbol = :cg, advection::Symbol = :bicgstab, momenta::Symbol = :identity,
                 tol::Real = 1e-3, maxit::Int = 50, sweeps::Int = 2, outer::Symbol = :gmres)
    momenta in (:shift, :identity, :krylov) || error("momenta: :shift, :identity or :krylov")
    outer in (:gmres, :none) || error("outer: :gmres (AMF as preconditioner) or :none (AMF as the solve, ASAM)")
    outer === :none && momenta === :identity &&
        error("outer = :none needs momenta = :krylov or :shift (else the masked momentum advection is lost)")
    g = m.grid; be = KA.get_backend(m)
    f(n) = KA.zeros(be, Float64, g.nxh, g.nyh, g.nz, n)
    return MaskAMF(StencilSys(m, 1), StencilSys(m, 7),
                   (acoustic = acoustic, advection = advection, momenta = momenta, tol = Float64(tol), maxit = maxit,
                    sweeps = sweeps, outer = outer),
                   ka_zeros(m), ka_zeros(m), f(1), f(1), f(7), f(7), NTuple{2,Int}[], [ka_zeros(m) for _ in 1:7])
end

"""The stencils for c = βh with the coefficients Cf (once per stage)."""
function mask_setup!(P::MaskAMF, m::CutKA, Cf, c)
    g = m.grid
    _launch!(m, _ms_acoustic!, (g.nxl, g.nyl, g.nz), P.ac.St, P.ac.D, Cf, _geo(g), Float64(c), g.dx, g.dy, g.nz)
    _launch!(m, _ms_transport!, (g.nxl, g.nyl, g.nz), P.adv.St, Cf, _geo(g), Float64(c), g.dx, g.dy, g.nz)
    P.ac.host = nothing; P.adv.host = nothing
    return P
end

@kernel function _ms_pick!(F, @Const(Z), var, slot)
    i, j, k = @index(Global, NTuple)
    @inbounds F[i+HALO, j+HALO, k, slot] = Z[i+HALO, j+HALO, k, var]
end
@kernel function _ms_zero_except!(X, @Const(R), keep1, keep2, keep3)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds for v in 1:NVC
        X[I, J, k, v] = (v == keep1 || v == keep2 || v == keep3) ? R[I, J, k, v] : 0.0
    end
end
"""Back substitution of the acoustic factor: Z = (ρ_r + c Y_ρ (with Y = J (0, 0, m)), Θ, m) is done in two passes."""
@kernel function _ms_axpy!(Z, @Const(R), @Const(Y), c, vars)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds for v in vars
        Z[I, J, k, v] = R[I, J, k, v] + c * Y[I, J, k, v]
    end
end
"""Shift back of the solved shifted momenta to the masked momentum rows (the volume weights of `shift_back`)."""
@kernel function _ms_back!(Z, @Const(M), G, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        _rowx(G, I, J, k) && (Z[I, J, k, CU] = shift_back(M[I, J, k, 1], M[I-1, J, k, 2], G.Vf[I-1, J, k], G.Vf[I, J, k],
                                                          G.Ax[I, J, k], G.Ax[I-1, J, k], G.Ax[I+1, J, k]))
        _rowy(G, I, J, k) && (Z[I, J, k, CV] = shift_back(M[I, J, k, 3], M[I, J-1, k, 4], G.Vf[I, J-1, k], G.Vf[I, J, k],
                                                          G.Ay[I, J, k], G.Ay[I, J-1, k], G.Ay[I, J+1, k]))
        _rowz(G, nz, I, J, k) && (Z[I, J, k, CW] = shift_back(M[I, J, k, 5], M[I, J, k-1, 6], G.Vf[I, J, k-1] * G.dzc[k-1],
                                                              G.Vf[I, J, k] * G.dzc[k], G.Az[I, J, k],
                                                              _azs(G, nz, I, J, k - 1), _azs(G, nz, I, J, k + 1)))
    end
end
@kernel function _ms_shift7!(F, @Const(M))
    i, j, k = @index(Global, NTuple)
    @inbounds for s in 1:6
        F[i+HALO, j+HALO, k, s+1] = M[i+HALO, j+HALO, k, s]
    end
end

"""
Z = P⁻¹ R (state arrays): acoustic factor (Θ Schur, back substitution of the momenta and ρ), then the advection
factor (Θ and the shifted momenta with T, shifted back on the masked rows).
"""
function mask_precondition!(Z, P::MaskAMF, m::CutKA, Cf, c, R)
    g, w = m.grid, m.work
    nz = g.nz
    X, Y = P.X, P.Y
    nd = (g.nxl, g.nyl, nz + 1)
    tol, maxit = P.method.tol, P.method.maxit
    if P.method.acoustic === :skip                           # advection factor only (diagnostics)
        Z .= R; ia = 0
        @goto advection
    end
    # acoustic: r̃_Θ = r_Θ + c J_Θm r_m
    _launch!(m, _ms_zero_except!, nd, X, R, CU, CV, CW)
    ka_vertical!(Y, m, Cf, X; halo = true, parts = :acoustic)
    _launch!(m, _ms_axpy!, nd, Z, R, Y, Float64(c), (CT,))                      # Z_Θ = r̃_Θ
    fill!(P.θ, 0.0)
    _launch!(m, _ms_pick!, (g.nxl, g.nyl, nz), P.b1, Z, CT, 1)
    ia = st_solve!(P.θ, P.ac, m, P.b1, P.method.acoustic; tol, maxit)
    # m = r_m + c J_mΘ Θ, ρ = r_ρ + c J_ρm m
    fill!(X, 0.0); _launch!(m, _ms_unpick!, (g.nxl, g.nyl, nz), X, P.θ, CT, 1)
    ka_vertical!(Y, m, Cf, X; halo = true, parts = :acoustic)
    _launch!(m, _ms_axpy!, nd, Z, R, Y, Float64(c), (CU, CV, CW))
    _launch!(m, _ms_unpick!, (g.nxl, g.nyl, nz), Z, P.θ, CT, 1)
    _launch!(m, _ms_zero_except!, nd, X, Z, CU, CV, CW)
    ka_vertical!(Y, m, Cf, X; halo = true, parts = :acoustic)
    _launch!(m, _ms_axpy!, nd, Z, R, Y, Float64(c), (CR,))
    if P.method.advection === :skip                          # acoustic factor only
        push!(P.its, (ia, 0))
        return Z
    end
    @label advection
    # advection: Θ with the ρ term on the right-hand side (ρ unchanged), shifted momenta, the same T
    _launch!(m, _ms_zero_except!, nd, X, Z, CR, CR, CR)
    ka_vertical!(Y, m, Cf, X; halo = true, parts = :advection)                    # Y_Θ = -(θ_f-weighted transport of ρ)
    fill!(P.f7, 0.0)
    _launch!(m, _ms_axpy!, nd, X, Z, Y, Float64(c), (CT,))
    _launch!(m, _ms_pick!, (g.nxl, g.nyl, nz), P.f7, X, CT, 1)
    mom = P.method.momenta
    if mom === :shift                                        # shifted momenta with T, back to the masked rows
        exchange!(m.dec, Z, g)
        _launch_r!(m, _kc_mshift!, CutRegion(g, :all), nz, w.M, Z, _geo(g), nz)
        _launch!(m, _ms_shift7!, (g.nxl, g.nyl, nz), P.f7, w.M)
    end
    P.x7 .= P.f7
    id = st_solve!(P.x7, P.adv, m, P.f7, P.method.advection; tol, maxit = P.method.advection === :gs ? P.method.sweeps : maxit)
    _launch!(m, _ms_unpick!, (g.nxl, g.nyl, nz), Z, P.x7, CT, 1)
    if mom === :shift
        _launch!(m, _ms_unshift7!, (g.nxl, g.nyl, nz), w.M, P.x7)
        exchange!(m.dec, w.M, g)
        _launch!(m, _ms_back!, (g.nxl, g.nyl, nz), Z, w.M, _geo(g), nz)
    elseif mom === :krylov                                   # (I - c J_adv,mm) m = r_m, matrix-free BiCGStab
        id += _mom_bicgstab!(Z, P, m, Cf, c; tol, maxit)
    end
    push!(P.its, (ia, id))
    return Z
end
@kernel function _ms_unpick!(Z, @Const(F), var, slot)
    i, j, k = @index(Global, NTuple)
    @inbounds Z[i+HALO, j+HALO, k, var] = F[i+HALO, j+HALO, k, slot]
end
@kernel function _ms_unshift7!(M, @Const(F))
    i, j, k = @index(Global, NTuple)
    @inbounds for s in 1:6
        M[i+HALO, j+HALO, k, s] = F[i+HALO, j+HALO, k, s+1]
    end
end

@kernel function _ms_momonly!(X, @Const(Z))
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        X[I, J, k, CR] = 0.0; X[I, J, k, CT] = 0.0
        X[I, J, k, CU] = Z[I, J, k, CU]; X[I, J, k, CV] = Z[I, J, k, CV]; X[I, J, k, CW] = Z[I, J, k, CW]
    end
end
_mdot(m::CutKA, a, b) = (g = m.grid; allreduce_sum(m.dec,
    dot(view(a, HALO+1:HALO+g.nxl, HALO+1:HALO+g.nyl, :, CU:CW), view(b, HALO+1:HALO+g.nxl, HALO+1:HALO+g.nyl, :, CU:CW))))
"""y = (I - c J_adv) x on the momentum rows (x with ρ = Θ = 0; the masked momentum advection, matrix-free)."""
function _mom_apply!(y, m::CutKA, Cf, c, x)
    ka_vertical!(y, m, Cf, x; halo = true, parts = :advection)
    @. y = x - c * y
    g = m.grid
    _launch!(m, _ms_momonly!, (g.nxl, g.nyl, g.nz + 1), y, copy(y))
    return y
end
"""The momenta of Z ← (I - c J_adv,mm)⁻¹ (momenta of Z) by BiCGStab (no preconditioner); returns the iterations."""
function _mom_bicgstab!(Z, P::MaskAMF, m::CutKA, Cf, c; tol, maxit)
    g = m.grid
    b, x, r, rh, p, v, t = P.K
    _launch!(m, _ms_momonly!, (g.nxl, g.nyl, g.nz + 1), b, Z)
    x .= b
    _mom_apply!(r, m, Cf, c, x); @. r = b - r
    nb = sqrt(_mdot(m, b, b)); nb == 0 && return 0
    rh .= r; fill!(p, 0.0); fill!(v, 0.0)
    ρ = α = ω = 1.0
    its = 0
    for it in 1:maxit
        its = it
        sqrt(_mdot(m, r, r)) <= tol * nb && (its = it - 1; break)
        ρn = _mdot(m, rh, r)
        β = (ρn / ρ) * (α / ω); ρ = ρn
        @. p = r + β * (p - ω * v)
        _mom_apply!(v, m, Cf, c, p)
        α = ρ / _mdot(m, rh, v)
        @. r = r - α * v                              # s
        @. x += α * p
        sqrt(_mdot(m, r, r)) <= tol * nb && break
        _mom_apply!(t, m, Cf, c, r)
        ω = _mdot(m, t, r) / _mdot(m, t, t)
        @. x += ω * r
        @. r = r - ω * t
    end
    _launch!(m, _ms_setmom!, (g.nxl, g.nyl, g.nz + 1), Z, x)
    return its
end
@kernel function _ms_setmom!(Z, @Const(X))
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        Z[I, J, k, CU] = X[I, J, k, CU]; Z[I, J, k, CV] = X[I, J, k, CV]; Z[I, J, k, CW] = X[I, J, k, CW]
    end
end
