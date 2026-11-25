# Masked momentum advection in the fast part (implicit = :amf): simple upwind, shift technique.
#
# In the frozen form the transport of ρ and Θ is already fast (mass flux of z times the frozen
# face θ, implicit through the masked faces). With implicit = :amf the momentum advection of the
# masked faces moves into the fast part as a first-order operator linear in z,
#     A1(y) z:  face momenta of z → two cell values per direction (shift, `shiftcoef`), upwind
#               transport with the frozen face coefficients a⁺ = max(F, 0)/ρ_left,
#               a⁻ = min(F, 0)/ρ_right (F the mass flux of y), back to the faces with the volume
#               weights of `shift_back`; rows: the masked x-, y- and z-faces,
# and the slow part keeps no momentum advection on these rows (first-order upwind in the mask, below). a⁺, a⁻ are
# coefficients of C (linear in C, as the MIS combination of the stages requires). The implicit
# HEVI step factors I - c (J_ac + J_adv) ≈ (I - c J_ac)(I - c J_adv) (approximate matrix
# factorization, J_ac the matrix of implicit = :mask, J_adv = A1).
#
# The mask uses first-order upwind fluxes only: θ on the faces of masked cells is the upwind θ (C), the masked momentum
# rows have no high-order remainder in the slow part (A1 z alone). The faces of the masked cells get the upwind
# transport Aθ(y) z = -(invV/J) div(a⁺ Θ_left + a⁻ Θ_right) of Θ(z) together with the ρ dependence of θ = Θ/ρ,
# -(invV/J) div(θ_f (a⁺ ρ_left + a⁻ ρ_right)) subtracted, on both cells of each face (`masked_face_transport!`, Θ conserved): with θ_f δF(z) of the acoustic part this is the
# exact Jacobian of the upwind Θ flux; without the ρ term the stiff part of the Jacobian is incomplete and the step is
# unstable in small cut cells (Courant numbers ≫ 1). The slow part keeps -A(y) y, so F(q, q) = N(q).
#
# FV3D_MASK_JFORM (operator.jl) selects the momentum rows:
#   :partial       a± δM (default; with the approximate factorization stable on the 1500 m bell hill, Δt = 30 s)
#   :conservative  also δF u_up - a± u δρ (`_mom_complete!`): the exact Jacobian of the masked upwind advection. Needs
#                  the coupled solve of I - c (J_ac + J_adv) (FV3D_MASK_SOLVER = :lu or :gmres); with the factorization the ρ and
#                  cross-direction columns lag and the run becomes unstable.
#   :advective     the simplified Jacobian of Jebens, Knoth, Weiner (JCP 2011): upwind transport of ρ, Θ and the momenta,
#                  no ρ columns in the Θ and momentum rows. In the frozen MIS step the slow part then holds +div F(y) on
#                  the masked ρ rows explicitly; unstable on the bell hill within 300 s.

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
        # cell velocities of the shifted momenta (complete Jacobian of the masked momentum flux)
        if vol(g, i, j, k) > 0
            for d in 1:3
                (lL, lR), (rL, rR) = shiftcoef(_alo(g, d, i, j, k), _ahi(g, d, i, j, k))
                il, ih = _ilo(g, d, i, j, k), _ihi(g, d, i, j, k)
                vL = il == 0 ? 0.0 : q[il]; vR = ih == 0 ? 0.0 : q[ih]
                C[cvel(g, d, 1, i, j, k)] = (lL * vL + lR * vR) / ρc
                C[cvel(g, d, 2, i, j, k)] = (rL * vL + rR * vR) / ρc
            end
        end
    end
    return C
end

"""
    _mom_complete!(T, m, C, z, d, side)

T += -(invV/J) div(δF(z) v_up - a⁺ v_left ρ_left(z) - a⁻ v_right ρ_right(z)): the rest of the Jacobian of the upwind
flux F⁺ M_left/ρ_left + F⁻ M_right/ρ_right of the shifted momentum (direction d, side) besides a± δM (v = M/ρ of y in C,
δF the mass flux of z).
"""
function _mom_complete!(T, m::Model3D, C, z, d, side)
    g = m.grid
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        im, jm = ip(g, i - 1), jp(g, j - 1)
        vc = C[cvel(g, d, side, i, j, k)]; ρc = z[iρ(g, i, j, k)]
        ap, am = C[caxp(g, i, j, k)], C[caxm(g, i, j, k)]
        vl = C[cvel(g, d, side, im, j, k)]
        f = (ax(g, i, j, k) * z[iU(g, i, j, k)] * (ap > 0 ? vl : vc) - ap * vl * z[iρ(g, im, j, k)] - am * vc * ρc) / g.dx
        T[im, j, k] -= f * g.invV[im, j, k] / g.Jc[im, j]; T[i, j, k] += f * g.invV[i, j, k] / g.Jc[i, j]
        ap, am = C[cayp(g, i, j, k)], C[caym(g, i, j, k)]
        vl = C[cvel(g, d, side, i, jm, k)]
        f = (ay(g, i, j, k) * z[iV(g, i, j, k)] * (ap > 0 ? vl : vc) - ap * vl * z[iρ(g, i, jm, k)] - am * vc * ρc) / g.dy
        T[i, jm, k] -= f * g.invV[i, jm, k] / g.Jc[i, jm]; T[i, j, k] += f * g.invV[i, j, k] / g.Jc[i, j]
        if k > 1
            ap, am = C[cazp(g, i, j, k)], C[cazm(g, i, j, k)]
            vl = C[cvel(g, d, side, i, j, k - 1)]
            f = az(g, i, j, k) * z[iW(g, i, j, k)] * (ap > 0 ? vl : vc) - ap * vl * z[iρ(g, i, j, k - 1)] - am * vc * ρc
            T[i, j, k-1] -= f * g.invV[i, j, k-1] / (g.Jc[i, j] * g.dzc[k-1]); T[i, j, k] += f * g.invV[i, j, k] / (g.Jc[i, j] * g.dzc[k])
        end
    end
    return T
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
        if FV3D_MASK_JFORM[] === :conservative
            _mom_complete!(T1, m, C, z, d, 1); _mom_complete!(T2, m, C, z, d, 2)
        end
        @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
            _row(g, d, i, j, k) || continue
            lo = d == 1 ? (ip(g, i - 1), j, k) : d == 2 ? (i, jp(g, j - 1), k) : (i, j, k - 1)
            a = _alo(g, d, i, j, k)
            out[_ilo(g, d, i, j, k)] += shift_back(T1[i, j, k], T2[lo...], vold(g, d, lo...), vold(g, d, i, j, k), a,
                                                   _alo(g, d, lo...), _ahi(g, d, i, j, k))
        end
    end
    return out
end

"""Is the face between the cells L and R a face of a masked cell?"""
@inline maskface(g, L, R) = g.mask[L...] || g.mask[R...]

"""
Upwind transport through the faces of the masked cells, conservative (both cells of each face):
T = -(invV/J) div f, f = a⁺ Φ_left + a⁻ Φ_right - θ_f (a⁺ ρ_left + a⁻ ρ_right) on the faces of masked cells (0 elsewhere),
Φ = z[ivar] (iΘ or iρ); the ρ term only with `θterm` (the ρ dependence of θ = Θ/ρ).
"""
function masked_face_transport!(T, m::Model3D, C, z, ivar, θterm)
    g = m.grid
    fill!(T, 0.0)
    @inline function face!(L, R, cp, cm, cθ, f, hL, hR)
        maskface(g, L, R) || return
        ap, am = C[cp(g, f...)], C[cm(g, f...)]
        fl = ap * z[ivar(g, L...)] + am * z[ivar(g, R...)]
        θterm && (fl -= C[cθ(g, f...)] * (ap * z[iρ(g, L...)] + am * z[iρ(g, R...)]))
        T[L...] -= fl * g.invV[L...] / (g.Jc[L[1], L[2]] * hL); T[R...] += fl * g.invV[R...] / (g.Jc[R[1], R[2]] * hR)
    end
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        c0 = (i, j, k)
        face!((ip(g, i - 1), j, k), c0, caxp, caxm, cθx, c0, g.dx, g.dx)
        face!((i, jp(g, j - 1), k), c0, cayp, caym, cθy, c0, g.dy, g.dy)
        k > 1 && face!((i, j, k - 1), c0, cazp, cazm, cθz, c0, g.dzc[k-1], g.dzc[k])
    end
    return T
end

"""
out += upwind transport of Θ(z) through the faces of the masked cells with the ρ dependence of θ = Θ/ρ subtracted
(`masked_face_transport!`, both cells of each face: Θ conserved); :advective without the ρ term.
"""
function adv_theta_masked!(out, m::Model3D, C, z)
    g = m.grid; T = m.cache.sh[3]
    masked_face_transport!(T, m, C, z, iΘ, FV3D_MASK_JFORM[] !== :advective)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        out[iΘ(g, i, j, k)] += T[i, j, k]
    end
    return out
end

"""FV3D_MASK_JFORM = :advective: out += upwind transport of ρ(z) through the faces of the masked cells."""
function adv_rho_masked!(out, m::Model3D, C, z)
    g = m.grid; T = m.cache.sh[4]
    masked_face_transport!(T, m, C, z, iρ, false)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        out[iρ(g, i, j, k)] += T[i, j, k]
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
            f = C[cazp(g, i, j, k)] * M[i, j, k-1] + C[cazm(g, i, j, k)] * M[i, j, k]
            T[i, j, k-1] -= f / g.dzc[k-1]; T[i, j, k] += f / g.dzc[k]
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
            addM!(r, s * C[cazp(g, i, j, k)] / g.dzc[k], d, side, (i, j, k - 1)); addM!(r, s * C[cazm(g, i, j, k)] / g.dzc[k], d, side, (i, j, k))
        end
        if k < g.nz
            addM!(r, -s * C[cazp(g, i, j, k + 1)] / g.dzc[k], d, side, (i, j, k)); addM!(r, -s * C[cazm(g, i, j, k + 1)] / g.dzc[k], d, side, (i, j, k + 1))
        end
        FV3D_MASK_JFORM[] === :conservative || return
        # complete Jacobian (_mom_complete!): face (index f, mass-flux column cF, measure A, lower cell L, upper cell H)
        vel(cc) = C[cvel(g, d, side, cc...)]
        function face!(w, cp, cm, cF, A, L, H, f)
            ap, am = C[cp(g, f...)], C[cm(g, f...)]
            vu = ap > 0 ? vel(L) : vel(H)
            A * vu != 0 && (push!(I, r); push!(J, cF); push!(V, w * A * vu))
            ap != 0 && (push!(I, r); push!(J, iρ(g, L...)); push!(V, -w * ap * vel(L)))
            am != 0 && (push!(I, r); push!(J, iρ(g, H...)); push!(V, -w * am * vel(H)))
        end
        c0 = (i, j, k); xl = (ip(g, i - 1), j, k); xh = (ip(g, i + 1), j, k); yl = (i, jp(g, j - 1), k); yh = (i, jp(g, j + 1), k)
        face!(s / g.dx, caxp, caxm, iU(g, c0...), ax(g, c0...), xl, c0, c0)
        face!(-s / g.dx, caxp, caxm, iU(g, xh...), ax(g, xh...), c0, xh, xh)
        face!(s / g.dy, cayp, caym, iV(g, c0...), ay(g, c0...), yl, c0, c0)
        face!(-s / g.dy, cayp, caym, iV(g, yh...), ay(g, yh...), c0, yh, yh)
        k > 1 && face!(s / g.dzc[k], cazp, cazm, iW(g, c0...), az(g, c0...), (i, j, k - 1), c0, c0)
        k < g.nz && face!(-s / g.dzc[k], cazp, cazm, iW(g, i, j, k + 1), az(g, i, j, k + 1), c0, (i, j, k + 1), (i, j, k + 1))
    end
    # Θ (and for :advective ρ) rows: upwind transport through the faces of the masked cells (masked_face_transport!)
    function sface!(L, R, cp, cm, cθ, f, hL, hR, ivar, θterm)
        maskface(g, L, R) || return
        ap, am = C[cp(g, f...)], C[cm(g, f...)]
        θf = C[cθ(g, f...)]
        for (cc, sg, h) in ((L, -1, hL), (R, 1, hR))
            w = sg * g.invV[cc...] / (g.Jc[cc[1], cc[2]] * h)
            w == 0 && continue
            r = ivar(g, cc...)
            ap != 0 && (push!(I, r); push!(J, ivar(g, L...)); push!(V, w * ap))
            am != 0 && (push!(I, r); push!(J, ivar(g, R...)); push!(V, w * am))
            if θterm
                ap != 0 && (push!(I, r); push!(J, iρ(g, L...)); push!(V, -w * θf * ap))
                am != 0 && (push!(I, r); push!(J, iρ(g, R...)); push!(V, -w * θf * am))
            end
        end
    end
    for (ivar, θterm) in (FV3D_MASK_JFORM[] === :advective ? ((iΘ, false), (iρ, false)) : ((iΘ, true),))
        for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
            c0 = (i, j, k)
            sface!((ip(g, i - 1), j, k), c0, caxp, caxm, cθx, c0, g.dx, g.dx, ivar, θterm)
            sface!((i, jp(g, j - 1), k), c0, cayp, caym, cθy, c0, g.dy, g.dy, ivar, θterm)
            k > 1 && sface!((i, j, k - 1), c0, cazp, cazm, cθz, c0, g.dzc[k-1], g.dzc[k], ivar, θterm)
        end
    end
    for d in 1:3, k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        _row(g, d, i, j, k) || continue
        lo = d == 1 ? (ip(g, i - 1), j, k) : d == 2 ? (i, jp(g, j - 1), k) : (i, j, k - 1)
        a = _alo(g, d, i, j, k)
        VR = vold(g, d, i, j, k) * a / (a + _ahi(g, d, i, j, k)); VL = vold(g, d, lo...) * a / (a + _alo(g, d, lo...))
        r = _ilo(g, d, i, j, k)
        addT!(r, VR / (VL + VR), d, 1, i, j, k)         # T_LC of the upper cell
        addT!(r, VL / (VL + VR), d, 2, lo...)           # T_RC of the lower cell
    end
    return sparse(I, J, V, m.nq, m.nq)
end
