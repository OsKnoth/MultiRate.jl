# Geometric multigrid preconditioner for the acoustic factor A = I - c J_ac of the Rosenbrock step of CutKA
# (cut_ros.jl), the GPU version of FV3D/multigrid.jl.
#
# Schur reduction (exact): A = [I A_tm; A_mt D_m], D_m = 1 (U, V), 1 + c α_sponge (W), S = I - c² J_tm D_m⁻¹ J_mt on
# t = (ρ, Θ) per cell; preconditioner: r̃_t = r_t + c J_tm D_m⁻¹ r_m, x_t ≈ S⁻¹ r̃_t (one V-cycle), x_m =
# D_m⁻¹ (r_m + c J_mt x_t). J_tm and J_mt are applied by `ka_vertical!(…; parts = :acoustic)` of the all-masked twin.
#
# S as a stencil per cell, rows ρ and Θ (slots 1–10 the ρ row, 11–20 the Θ row): coefficients on Θ of the cell, of its
# west, east, south, north, lower and upper neighbour (slots 1–7) and on ρ of the cell, lower and upper neighbour
# (8–10). Galerkin coarsening R S P with horizontal 2 × 2 cells, R the sum, P = Rᵀ (injection) keeps the stencil:
# couplings to a neighbour inside the same coarse cell go to the diagonal. Smoother: zebra (red–black columns by the
# global column index), the block-tridiagonal (2 × 2) column systems of S solved exactly per column (vertical lines).
# The coarsest level: `νc` zebra sweeps.

const MGS = 20                                      # stencil slots per cell
const SΘC, SΘW, SΘE, SΘS, SΘN, SΘB, SΘT, SρC, SρB, SρT = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10

"""Tile of a multigrid level (an AbstractTileGrid for the halo exchange)."""
struct MGTile <: AbstractTileGrid
    nxl::Int; nyl::Int; nxh::Int; nyh::Int; nz::Int
    i0::Int; j0::Int                                # global offset of the tile at this level (colours)
end
MGTile(nxl, nyl, nz, i0, j0) = MGTile(nxl, nyl, nxl + 2HALO, nyl + 2HALO, nz, i0, j0)

"""One level: tile, stencil, unknowns x, right-hand side b, residual r (2 fields ρ, Θ), column-solve work."""
struct MGLevelKA{A}
    t::MGTile
    St::A; x::A; b::A; r::A; C::A; g::A
end

"""Multigrid of the all-masked twin model `ma`: levels, sweeps ν (pre and post), coarsest sweeps νc."""
struct CutMG{L,A}
    levels::Vector{L}
    ν::Int; νc::Int
    X::A; Y::A                                      # state-shaped work (Schur reduction)
end

"""
    CutMG(ma::CutKA; ν = 2, νc = 20, coarsest = 2)

Levels of horizontal 2 × 2 coarsening of the tile of `ma` while the tile sizes are even and at least `coarsest`.
"""
function CutMG(ma::CutKA; ν::Int = 2, νc::Int = 20, coarsest::Int = 2)
    g = ma.grid
    be = KA.get_backend(g)
    nxl, nyl, nz, i0, j0 = g.nxl, g.nyl, g.nz, g.i0, g.j0
    lv = []
    while true
        t = MGTile(nxl, nyl, nz, i0, j0)
        z(n) = KA.zeros(be, Float64, t.nxh, t.nyh, nz, n)
        push!(lv, MGLevelKA(t, z(MGS), z(2), z(2), z(2), z(4), z(2)))
        (iseven(nxl) && iseven(nyl) && nxl ÷ 2 >= coarsest && nyl ÷ 2 >= coarsest) || break
        nxl, nyl, i0, j0 = nxl ÷ 2, nyl ÷ 2, i0 ÷ 2, j0 ÷ 2
    end
    return CutMG([l for l in lv], ν, νc, ka_zeros(ma), ka_zeros(ma))
end

# ------------------------------------------------------------------ stencils

"""Fine stencil of S = I - c² J_tm D⁻¹ J_mt (all faces of the all-masked twin; coefficients of the model)."""
@kernel function _mg_fine!(St, @Const(Cf), G, damp, c, grav, dx, dy, nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        for s in 1:MGS; St[I, J, k, s] = 0.0; end
        St[I, J, k, SρC] = 1.0; St[I, J, k, 10 + SΘC] = 1.0
        iV = G.iV[I, J, k]
        if iV > 0
            c2 = c * c
            # horizontal faces: ± c² iV a P / h² (Θ_c - Θ_n), the Θ row with the face θ
            for (a, P, θf, slot, h) in ((G.Ax[I, J, k], Cf[I, J, k, KPX], Cf[I, J, k, KθX], SΘW, dx),
                                        (G.Ax[I+1, J, k], Cf[I+1, J, k, KPX], Cf[I+1, J, k, KθX], SΘE, dx),
                                        (G.Ay[I, J, k], Cf[I, J, k, KPY], Cf[I, J, k, KθY], SΘS, dy),
                                        (G.Ay[I, J+1, k], Cf[I, J+1, k, KPY], Cf[I, J+1, k, KθY], SΘN, dy))
                if a > 0
                    w = c2 * iV * a * P / h^2
                    St[I, J, k, SΘC] += w; St[I, J, k, slot] -= w
                    St[I, J, k, 10 + SΘC] += θf * w; St[I, J, k, 10 + slot] -= θf * w
                end
            end
            # lower face k (the cell above it): + c² iV A/(Δz_k D) [P (Θ_k - Θ_{k-1})/Δz̃ + g (w ρ_{k-1} + (1-w) ρ_k)]
            if k >= 2 && G.Az[I, J, k] > 0
                β = c2 * iV * G.Az[I, J, k] / (G.dzc[k] * (1 + c * damp[k]))
                P, θf, wl = Cf[I, J, k, KPZ], Cf[I, J, k, KθZ], G.wlo[k]
                for (row, f) in ((0, 1.0), (10, θf))
                    St[I, J, k, row + SΘC] += f * β * P / G.dzf[k]; St[I, J, k, row + SΘB] -= f * β * P / G.dzf[k]
                    St[I, J, k, row + SρB] += f * β * grav * wl; St[I, J, k, row + SρC] += f * β * grav * (1 - wl)
                end
            end
            # upper face k + 1 (the cell below it): - c² iV A/(Δz_k D) [P (Θ_{k+1} - Θ_k)/Δz̃ + g (w ρ_k + (1-w) ρ_{k+1})]
            if k < nz && G.Az[I, J, k+1] > 0
                β = c2 * iV * G.Az[I, J, k+1] / (G.dzc[k] * (1 + c * damp[k+1]))
                P, θf, wl = Cf[I, J, k+1, KPZ], Cf[I, J, k+1, KθZ], G.wlo[k+1]
                for (row, f) in ((0, 1.0), (10, θf))
                    St[I, J, k, row + SΘT] -= f * β * P / G.dzf[k+1]; St[I, J, k, row + SΘC] += f * β * P / G.dzf[k+1]
                    St[I, J, k, row + SρC] -= f * β * grav * wl; St[I, J, k, row + SρT] -= f * β * grav * (1 - wl)
                end
            end
        end
    end
end

"""Galerkin coarse stencil: the sum of the rows of the 2 × 2 fine cells, couplings inside the coarse cell to the
diagonal."""
@kernel function _mg_coarsen!(Sc, @Const(Sf))
    ic, jc, k = @index(Global, NTuple)
    Ic, Jc = ic + HALO, jc + HALO
    @inbounds begin
        for s in 1:MGS; Sc[Ic, Jc, k, s] = 0.0; end
        for b in 0:1, a in 0:1
            I, J = HALO + 2ic - 1 + a, HALO + 2jc - 1 + b
            for row in (0, 10)
                Sc[Ic, Jc, k, row + SΘC] += Sf[I, J, k, row + SΘC]
                Sc[Ic, Jc, k, row + SΘB] += Sf[I, J, k, row + SΘB]; Sc[Ic, Jc, k, row + SΘT] += Sf[I, J, k, row + SΘT]
                Sc[Ic, Jc, k, row + SρC] += Sf[I, J, k, row + SρC]
                Sc[Ic, Jc, k, row + SρB] += Sf[I, J, k, row + SρB]; Sc[Ic, Jc, k, row + SρT] += Sf[I, J, k, row + SρT]
                # west / east neighbour: inside the coarse cell for a = 1 / a = 0
                Sc[Ic, Jc, k, row + (a == 1 ? SΘC : SΘW)] += Sf[I, J, k, row + SΘW]
                Sc[Ic, Jc, k, row + (a == 0 ? SΘC : SΘE)] += Sf[I, J, k, row + SΘE]
                Sc[Ic, Jc, k, row + (b == 1 ? SΘC : SΘS)] += Sf[I, J, k, row + SΘS]
                Sc[Ic, Jc, k, row + (b == 0 ? SΘC : SΘN)] += Sf[I, J, k, row + SΘN]
            end
        end
    end
end

"""(S x)_row of cell (I, J, k), row 0 (ρ) or 10 (Θ)."""
@inline function _mg_row(St, x, I, J, k, nz, row)
    s = St[I, J, k, row + SΘC] * x[I, J, k, 2] + St[I, J, k, row + SρC] * x[I, J, k, 1] +
        St[I, J, k, row + SΘW] * x[I-1, J, k, 2] + St[I, J, k, row + SΘE] * x[I+1, J, k, 2] +
        St[I, J, k, row + SΘS] * x[I, J-1, k, 2] + St[I, J, k, row + SΘN] * x[I, J+1, k, 2]
    k > 1 && (s += St[I, J, k, row + SΘB] * x[I, J, k-1, 2] + St[I, J, k, row + SρB] * x[I, J, k-1, 1])
    k < nz && (s += St[I, J, k, row + SΘT] * x[I, J, k+1, 2] + St[I, J, k, row + SρT] * x[I, J, k+1, 1])
    return s
end
"""r = b - S x on the interior (the Θ halo of x filled)."""
@kernel function _mg_residual!(r, @Const(St), @Const(x), @Const(b), nz)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        r[I, J, k, 1] = b[I, J, k, 1] - _mg_row(St, x, I, J, k, nz, 0)
        r[I, J, k, 2] = b[I, J, k, 2] - _mg_row(St, x, I, J, k, nz, 10)
    end
end

"""Zebra half-sweep: the columns of one colour solved exactly (block Thomas, 2 × 2 blocks) with the horizontal
neighbours at their current values (Θ halo filled)."""
@kernel function _mg_zebra!(x, @Const(St), @Const(b), C, g, nz, i0, j0, colour)
    i, j = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds if iseven(i0 + i + j0 + j) == colour
        # forward: D'_k = D_k - L_k C_{k-1}, C_k = D'_k⁻¹ U_k, g_k = D'_k⁻¹ (f_k - L_k g_{k-1})
        for k in 1:nz
            f1 = b[I, J, k, 1] - (St[I, J, k, SΘW] * x[I-1, J, k, 2] + St[I, J, k, SΘE] * x[I+1, J, k, 2] +
                                  St[I, J, k, SΘS] * x[I, J-1, k, 2] + St[I, J, k, SΘN] * x[I, J+1, k, 2])
            f2 = b[I, J, k, 2] - (St[I, J, k, 10+SΘW] * x[I-1, J, k, 2] + St[I, J, k, 10+SΘE] * x[I+1, J, k, 2] +
                                  St[I, J, k, 10+SΘS] * x[I, J-1, k, 2] + St[I, J, k, 10+SΘN] * x[I, J+1, k, 2])
            # D = [d11 d12; d21 d22] on (ρ, Θ), L, U the couplings to k - 1, k + 1
            d11, d12 = St[I, J, k, SρC], St[I, J, k, SΘC]
            d21, d22 = St[I, J, k, 10+SρC], St[I, J, k, 10+SΘC]
            if k > 1
                l11, l12 = St[I, J, k, SρB], St[I, J, k, SΘB]
                l21, l22 = St[I, J, k, 10+SρB], St[I, J, k, 10+SΘB]
                c11, c12, c21, c22 = C[I, J, k-1, 1], C[I, J, k-1, 2], C[I, J, k-1, 3], C[I, J, k-1, 4]
                d11 -= l11 * c11 + l12 * c21; d12 -= l11 * c12 + l12 * c22
                d21 -= l21 * c11 + l22 * c21; d22 -= l21 * c12 + l22 * c22
                f1 -= l11 * g[I, J, k-1, 1] + l12 * g[I, J, k-1, 2]
                f2 -= l21 * g[I, J, k-1, 1] + l22 * g[I, J, k-1, 2]
            end
            det = d11 * d22 - d12 * d21
            e11, e12, e21, e22 = d22 / det, -d12 / det, -d21 / det, d11 / det
            if k < nz
                u11, u12 = St[I, J, k, SρT], St[I, J, k, SΘT]
                u21, u22 = St[I, J, k, 10+SρT], St[I, J, k, 10+SΘT]
                C[I, J, k, 1] = e11 * u11 + e12 * u21; C[I, J, k, 2] = e11 * u12 + e12 * u22
                C[I, J, k, 3] = e21 * u11 + e22 * u21; C[I, J, k, 4] = e21 * u12 + e22 * u22
            end
            g[I, J, k, 1] = e11 * f1 + e12 * f2; g[I, J, k, 2] = e21 * f1 + e22 * f2
        end
        x[I, J, nz, 1] = g[I, J, nz, 1]; x[I, J, nz, 2] = g[I, J, nz, 2]
        for k in nz-1:-1:1
            x[I, J, k, 1] = g[I, J, k, 1] - (C[I, J, k, 1] * x[I, J, k+1, 1] + C[I, J, k, 2] * x[I, J, k+1, 2])
            x[I, J, k, 2] = g[I, J, k, 2] - (C[I, J, k, 3] * x[I, J, k+1, 1] + C[I, J, k, 4] * x[I, J, k+1, 2])
        end
    end
end

"""b_c = R r (sum over the 2 × 2 fine cells)."""
@kernel function _mg_restrict!(bc, @Const(r))
    ic, jc, k, v = @index(Global, NTuple)
    @inbounds bc[ic+HALO, jc+HALO, k, v] = r[HALO+2ic-1, HALO+2jc-1, k, v] + r[HALO+2ic, HALO+2jc-1, k, v] +
                                            r[HALO+2ic-1, HALO+2jc, k, v] + r[HALO+2ic, HALO+2jc, k, v]
end
"""x += P x_c (injection)."""
@kernel function _mg_prolong!(x, @Const(xc))
    i, j, k, v = @index(Global, NTuple)
    @inbounds x[i+HALO, j+HALO, k, v] += xc[(i + 1) ÷ 2 + HALO, (j + 1) ÷ 2 + HALO, k, v]
end

# ------------------------------------------------------------------ driver

_mgl!(ma, k!, nd, args...) = _launch!(ma, k!, nd, args...)

"""Stencils of all levels for A = I - c J_ac with the coefficients Cf (once per Rosenbrock step)."""
function mg_setup!(mg::CutMG, ma::CutKA, Cf, c)
    g = ma.grid
    L1 = mg.levels[1]
    _mgl!(ma, _mg_fine!, (g.nxl, g.nyl, g.nz), L1.St, Cf, _geo(g), ma.damp, Float64(c), ma.gas.g, g.dx, g.dy, g.nz)
    for l in 2:length(mg.levels)
        t = mg.levels[l].t
        _mgl!(ma, _mg_coarsen!, (t.nxl, t.nyl, t.nz), mg.levels[l].St, mg.levels[l-1].St)
    end
    return mg
end

"""ν zebra sweeps on level L (x updated)."""
function _mg_smooth!(mg::CutMG, ma::CutKA, L, ν)
    t = L.t
    for _ in 1:ν, colour in (true, false)
        exchange!(ma.dec, L.x, t, (2,))
        _mgl!(ma, _mg_zebra!, (t.nxl, t.nyl), L.x, L.St, L.b, L.C, L.g, t.nz, t.i0, t.j0, colour)
    end
    return L
end

"""One V-cycle on level l for S_l x = b_l (x = 0 on entry)."""
function mg_vcycle!(mg::CutMG, ma::CutKA, l::Int = 1)
    L = mg.levels[l]
    t = L.t
    fill!(L.x, 0.0)
    if l == length(mg.levels)
        return _mg_smooth!(mg, ma, L, mg.νc)
    end
    _mg_smooth!(mg, ma, L, mg.ν)
    exchange!(ma.dec, L.x, t, (2,))
    _mgl!(ma, _mg_residual!, (t.nxl, t.nyl, t.nz), L.r, L.St, L.x, L.b, t.nz)
    Lc = mg.levels[l+1]
    _mgl!(ma, _mg_restrict!, (Lc.t.nxl, Lc.t.nyl, t.nz, 2), Lc.b, L.r)
    mg_vcycle!(mg, ma, l + 1)
    _mgl!(ma, _mg_prolong!, (t.nxl, t.nyl, t.nz, 2), L.x, Lc.x)
    _mg_smooth!(mg, ma, L, mg.ν)
    return L
end

"""
z = M⁻¹ r for A = I - c J_ac (state arrays, interior): Schur reduction with the twin `ma`, one V-cycle on S, momenta
by back substitution. X, Y: state-shaped work arrays.
"""
function mg_precondition!(Z, mg::CutMG, ma::CutKA, Cf, c, R, X, Y)
    g = ma.grid
    nz = g.nz
    L1 = mg.levels[1]
    _mgl!(ma, _mg_dinv!, (g.nxl, g.nyl, nz + 1), X, R, ma.damp, Float64(c))              # X = (0, D⁻¹ r_m)
    ka_vertical!(Y, ma, Cf, X; halo = true, parts = :acoustic, sponge = true)           # J_tm D⁻¹ r_m
    _mgl!(ma, _mg_reduce!, (g.nxl, g.nyl, nz), L1.b, R, Y, Float64(c))                   # b = r_t + c J_tm D⁻¹ r_m
    mg_vcycle!(mg, ma, 1)
    _mgl!(ma, _mg_tofield!, (g.nxl, g.nyl, nz + 1), X, L1.x)                            # X = (x_t, 0)
    ka_vertical!(Y, ma, Cf, X; halo = true, parts = :acoustic, sponge = true)           # J_mt x_t
    _mgl!(ma, _mg_backsub!, (g.nxl, g.nyl, nz + 1), Z, R, Y, L1.x, ma.damp, Float64(c))
    return Z
end

"""X = (0, D⁻¹ r_m): ρ, Θ zero, momenta scaled by 1/(1 + c α) (W)."""
@kernel function _mg_dinv!(X, @Const(R), damp, c)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        X[I, J, k, CR] = 0.0; X[I, J, k, CT] = 0.0
        X[I, J, k, CU] = R[I, J, k, CU]; X[I, J, k, CV] = R[I, J, k, CV]
        X[I, J, k, CW] = R[I, J, k, CW] / (1 + c * damp[k])
    end
end
@kernel function _mg_reduce!(b, @Const(R), @Const(Y), c)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        b[I, J, k, 1] = R[I, J, k, CR] + c * Y[I, J, k, CR]
        b[I, J, k, 2] = R[I, J, k, CT] + c * Y[I, J, k, CT]
    end
end
@kernel function _mg_tofield!(X, @Const(x))
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        nz = size(x, 3)
        X[I, J, k, CR] = k <= nz ? x[I, J, k, 1] : 0.0; X[I, J, k, CT] = k <= nz ? x[I, J, k, 2] : 0.0
        X[I, J, k, CU] = 0.0; X[I, J, k, CV] = 0.0; X[I, J, k, CW] = 0.0
    end
end
@kernel function _mg_backsub!(Z, @Const(R), @Const(Y), @Const(x), damp, c)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds begin
        nz = size(x, 3)
        Z[I, J, k, CR] = k <= nz ? x[I, J, k, 1] : 0.0; Z[I, J, k, CT] = k <= nz ? x[I, J, k, 2] : 0.0
        if k <= nz
            Z[I, J, k, CU] = R[I, J, k, CU] + c * Y[I, J, k, CU]
            Z[I, J, k, CV] = R[I, J, k, CV] + c * Y[I, J, k, CV]
            Z[I, J, k, CW] = (R[I, J, k, CW] + c * Y[I, J, k, CW]) / (1 + c * damp[k])
        else
            Z[I, J, k, CU] = R[I, J, k, CU]; Z[I, J, k, CV] = R[I, J, k, CV]; Z[I, J, k, CW] = R[I, J, k, CW]
        end
    end
end
