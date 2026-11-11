# Masked momentum advection in the fast part (implicit = :amf): simple upwind, shift technique.
#
# In the frozen form the transport of ρ and Θ is already fast (mass flux of z times the frozen
# face θ, implicit through the masked faces). With implicit = :amf the momentum advection of the
# masked faces moves into the fast part as a first-order operator linear in z,
#     A1(y) z:  face momenta of z → two cell values per direction (shift, `shiftcoef`), upwind
#               transport with the frozen face coefficients a⁺ = max(F, 0)/ρ_left,
#               a⁻ = min(F, 0)/ρ_right (F the mass flux of y), back to the faces with the volume
#               weights of `shift_back`; rows: the masked x-, y- and z-faces,
# and the slow part keeps S(y) - A1(y) y on these rows, so F(q, q) = N(q) is exact. a⁺, a⁻ are
# coefficients of C (linear in C, as the MIS combination of the stages requires). The implicit
# HEVI step factors I - c (J_ac + J_adv) ≈ (I - c J_ac)(I - c J_adv) (approximate matrix
# factorization, J_ac the matrix of implicit = :mask, J_adv = A1).

"""Coefficients of the shift (`shiftvel`) as a linear map: L = cLL vL + cLR vR, R = cRL vL + cRR vR."""
@inline shiftcoef(aL, aR) = aL >= aR ? ((1.0, 0.0), aL > 0 ? ((aL - aR) / aL, aR / aL) : (0.0, 1.0)) :
                                       ((aL / aR, (aR - aL) / aR), (0.0, 1.0))

"""Is the z-face k (2 … nz) of column (i, j) a face of a masked cell (and open)?"""
@inline maskz(g, i, j, k) = g.Az[i, j, k] > 0 && (g.mask[i, j, k-1] || g.mask[i, j, k])

"""Upwind coefficients a⁺, a⁻ of all faces into C (mass fluxes in the cache, ρ of y)."""
function upwind_coefficients!(C, m::Model3D, q)
    g, c = m.grid, m.cache
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        ρc = q[iρ(g, i, j, k)]
        F = c.Fx[i, j, k]; C[caxp(g, i, j, k)] = max(F, 0) / q[iρ(g, i - 1, j, k)]; C[caxm(g, i, j, k)] = min(F, 0) / ρc
        F = c.Fy[i, j, k]; C[cayp(g, i, j, k)] = max(F, 0) / q[iρ(g, i, j - 1, k)]; C[caym(g, i, j, k)] = min(F, 0) / ρc
        if k > 1
            F = c.Fz[i, j, k]; C[cazp(g, i, j, k)] = max(F, 0) / q[iρ(g, i, j, k - 1)]; C[cazm(g, i, j, k)] = min(F, 0) / ρc
        end
    end
    return C
end

# geometry of the shift per component d = 1 (U, x), 2 (V, y), 3 (W, z): faces of cell (i, j, k)
@inline _alo(g, d, i, j, k) = d == 1 ? ax(g, i, j, k) : d == 2 ? ay(g, i, j, k) : azs(g, i, j, k)
@inline _ahi(g, d, i, j, k) = d == 1 ? ax(g, i + 1, j, k) : d == 2 ? ay(g, i, j + 1, k) : azs(g, i, j, k + 1)
"""Momentum unknown index of the lower / upper face of cell (i, j, k) in direction d (0: wall)."""
@inline _ilo(g, d, i, j, k) = d == 1 ? iU(g, i, j, k) : d == 2 ? iV(g, i, j, k) : (k == 1 ? 0 : iW(g, i, j, k))
@inline _ihi(g, d, i, j, k) = d == 1 ? iU(g, i + 1, j, k) : d == 2 ? iV(g, i, j + 1, k) : (k == g.nz ? 0 : iW(g, i, j, k + 1))
"""Is the face (lower face of cell (i, j, k) in direction d) a row of A1?"""
@inline _row(g, d, i, j, k) = d == 1 ? (g.Mx[i, j, k] > 0 && g.Ax[i, j, k] > 0) :
                              d == 2 ? (g.My[i, j, k] > 0 && g.Ay[i, j, k] > 0) : (k > 1 && maskz(g, i, j, k))

"""
    adv_masked!(out, m, C, z)

out += A1(C) z on the rows of the masked faces (matrix-free; `adv_matrix` is the same operator
assembled).
"""
function adv_masked!(out, m::Model3D, C, z)
    g, c = m.grid, m.cache
    ML, MR, T1, T2 = c.sh[1], c.sh[2], c.sh[3], c.sh[4]
    for d in 1:3
        @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
            if vol(g, i, j, k) > 0
                (lL, lR), (rL, rR) = shiftcoef(_alo(g, d, i, j, k), _ahi(g, d, i, j, k))
                il, ih = _ilo(g, d, i, j, k), _ihi(g, d, i, j, k)
                vL = il == 0 ? 0.0 : z[il]; vR = ih == 0 ? 0.0 : z[ih]
                ML[i, j, k] = lL * vL + lR * vR; MR[i, j, k] = rL * vL + rR * vR
            else
                ML[i, j, k] = MR[i, j, k] = 0.0
            end
        end
        upwind_transport!(T1, ML, m, C); upwind_transport!(T2, MR, m, C)
        @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
            _row(g, d, i, j, k) || continue
            lo = d == 1 ? (ip(g, i - 1), j, k) : d == 2 ? (i, jp(g, j - 1), k) : (i, j, k - 1)
            a = _alo(g, d, i, j, k)
            out[_ilo(g, d, i, j, k)] += shift_back(T1[i, j, k], T2[lo...], vol(g, lo...), vol(g, i, j, k), a,
                                                   _alo(g, d, lo...), _ahi(g, d, i, j, k))
        end
    end
    return out
end

"""T = -(invV/J) div(a⁺ M_left + a⁻ M_right) of the cell field M (upwind, coefficients of C)."""
function upwind_transport!(T, M, m::Model3D, C)
    g = m.grid
    fill!(T, 0.0)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        im, jm = ip(g, i - 1), jp(g, j - 1)
        f = (C[caxp(g, i, j, k)] * M[im, j, k] + C[caxm(g, i, j, k)] * M[i, j, k]) / g.dx
        T[im, j, k] -= f; T[i, j, k] += f
        f = (C[cayp(g, i, j, k)] * M[i, jm, k] + C[caym(g, i, j, k)] * M[i, j, k]) / g.dy
        T[i, jm, k] -= f; T[i, j, k] += f
        if k > 1
            f = (C[cazp(g, i, j, k)] * M[i, j, k-1] + C[cazm(g, i, j, k)] * M[i, j, k]) / g.dz
            T[i, j, k-1] -= f; T[i, j, k] += f
        end
    end
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        T[i, j, k] *= g.invV[i, j, k] / g.Jc[i, j]
    end
    return T
end

"""The matrix of A1(C) (rows: masked faces), assembled from the same pieces as `adv_masked!`."""
function adv_matrix(m::Model3D, C)
    g = m.grid
    I = Int[]; J = Int[]; V = Float64[]
    # T_side(cell) = Σ_cells' w · M_side(cell'), M_side(cell') = Σ shift coefficients · face momenta
    function addM!(r, w, d, side, cc)
        vol(g, cc...) > 0 || return
        (lL, lR), (rL, rR) = shiftcoef(_alo(g, d, cc...), _ahi(g, d, cc...))
        cL, cR = side == 1 ? (lL, lR) : (rL, rR)
        il, ih = _ilo(g, d, cc...), _ihi(g, d, cc...)
        il != 0 && cL != 0 && (push!(I, r); push!(J, il); push!(V, w * cL))
        ih != 0 && cR != 0 && (push!(I, r); push!(J, ih); push!(V, w * cR))
    end
    function addT!(r, w, d, side, i, j, k)
        s = w * g.invV[i, j, k] / g.Jc[i, j]
        s == 0 && return
        # faces of cell (i, j, k): lower x (left neighbour), upper x, … ; flux = a⁺ M_left + a⁻ M_right
        for (cp, cm, h, lo, hi) in ((caxp, caxm, g.dx, (ip(g, i - 1), j, k), (ip(g, i + 1), j, k)),
                                    (cayp, caym, g.dy, (i, jp(g, j - 1), k), (i, jp(g, j + 1), k)))
            # lower face (index of this cell): +flux / h
            addM!(r, s * C[cp(g, i, j, k)] / h, d, side, lo); addM!(r, s * C[cm(g, i, j, k)] / h, d, side, (i, j, k))
            # upper face (index of the upper neighbour): -flux / h
            up = cp === caxp ? (ip(g, i + 1), j, k) : (i, jp(g, j + 1), k)
            addM!(r, -s * C[cp(g, up...)] / h, d, side, (i, j, k)); addM!(r, -s * C[cm(g, up...)] / h, d, side, hi)
        end
        if k > 1
            addM!(r, s * C[cazp(g, i, j, k)] / g.dz, d, side, (i, j, k - 1)); addM!(r, s * C[cazm(g, i, j, k)] / g.dz, d, side, (i, j, k))
        end
        if k < g.nz
            addM!(r, -s * C[cazp(g, i, j, k + 1)] / g.dz, d, side, (i, j, k)); addM!(r, -s * C[cazm(g, i, j, k + 1)] / g.dz, d, side, (i, j, k + 1))
        end
    end
    for d in 1:3, k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        _row(g, d, i, j, k) || continue
        lo = d == 1 ? (ip(g, i - 1), j, k) : d == 2 ? (i, jp(g, j - 1), k) : (i, j, k - 1)
        a = _alo(g, d, i, j, k)
        VR = vol(g, i, j, k) * a / (a + _ahi(g, d, i, j, k)); VL = vol(g, lo...) * a / (a + _alo(g, d, lo...))
        r = _ilo(g, d, i, j, k)
        addT!(r, VR / (VL + VR), d, 1, i, j, k)         # T_LC of the upper cell
        addT!(r, VL / (VL + VR), d, 2, lo...)           # T_RC of the lower cell
    end
    return sparse(I, J, V, m.nq, m.nq)
end
