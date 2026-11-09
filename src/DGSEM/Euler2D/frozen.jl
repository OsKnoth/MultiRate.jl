# Frozen-part split of the DGSEM operator for `TimeStepping.run_frozen`
#
# N(q) = F(q, q), first argument y frozen at the stage values, second z evolving in the fast loop.
# Volume (flux differencing, pair a = (i, j), b = (k, j) or (i, k), contravariant metric (ax, az)):
#   N:     f_ρ = A_ab v_n, f_Θ = B_ab v_n, v_n = ax ū + az w̄ (ū, w̄ two-point means of the velocity),
#          f_m = f_ρ (ū, w̄) + {p} (ax, az), gravity M_g(ρa, ρb) g (z_b - z_a) (ax, az)
#   fast:  f_ρ = (A/ρa)(y) ½ m̃_a + (A/ρb)(y) ½ m̃_b with m̃ = ax m_x + az m_z of z, f_Θ the same with B,
#          {p} of the nodal p(Θ_z), gravity R_ab(y) ½(ρa + ρb)_z g Δz, R = M_g / arithmetic mean
#   slow:  f_ρ(y) (ū, w̄)(y)  (momentum advection)
# Faces (LMARS along the unit normal n, upwind side from y):
#   N:     m_f = ρ_up w_s, w_s = ½(u_nL + u_nR) - (p_R - p_L)/(2 ρ_m c_m), p_s = {p} - ½ ρ_m c_m [u_n],
#          F = (m_f, m_f u_up + p_s n, m_f θ_up), minus the inner flux (m·ñ, m u_ñ + p ñ, θ m·ñ)
#   fast:  m_f = (ρ_up/2ρ_L) m_nL + (ρ_up/2ρ_R) m_nR - ρ_up/(2ρ_m c_m) (p_R - p_L), θ_up m_f,
#          p_s with ρ_m c_m/(2ρ) m_n, the inner m·ñ, θ(y) m·ñ, p ñ (all frozen coefficients from y)
#   slow:  m_f(y) u_up(y) and the inner m u_ñ(y)
# Walls (mirror state): p_s = p + c m_n (c the LMARS sound speed of y). Reference residual and
# sponge are in the fast part. F(q, q) = N(q) exactly; the fast part is linear in (ρ, m, Θ) of z
# except for the nodal p(Θ_z).

const _NVC = 5          # coefficients per volume pair: A/ρa, A/ρb, B/ρa, B/ρb, R
const _NFC = 10         # coefficients per face node pair

_nface(M::Mesh2D) = M.Ne * 2 * M.np + M.Nex * M.np

frozen_ncoef(m::Euler2DModel) = (M = m.mesh; 2 * _NVC * M.np^3 * M.Ne + _NFC * _nface(M))

@inline _vc(M, dir, i, j, k, e) = ((((e - 1) * 2 + (dir - 1)) * M.np + (k - 1)) * M.np^2 + (j - 1) * M.np + i - 1) * _NVC
@inline _fc(M, f) = 2 * _NVC * M.np^3 * M.Ne + (f - 1) * _NFC

"""LMARS pieces along the unit normal: (w_s, p_s, ρ_m c_m, upwind left?)."""
@inline function _lmars_parts(γ, cfix, ρL, unL, pL, ρR, unR, pR)
    ρm = 0.5 * (ρL + ρR)
    cm = cfix > 0 ? cfix : 0.5 * (sqrt(γ * pL / ρL) + sqrt(γ * pR / ρR))
    ws = 0.5 * (unL + unR) - (pR - pL) / (2 * ρm * cm)
    ps = 0.5 * (pL + pR) - 0.5 * ρm * cm * (unR - unL)
    return ws, ps, ρm * cm, ws >= 0
end

"""Visit the face node pairs in a fixed order: f(fidx, kind, iL, jL, eL, iR, jR, eR, ñx, ñz, w),
kind = :interior or :wall (then the R arguments repeat L)."""
@inline function _foreach_face(f, M::Mesh2D)
    np, w = M.np, M.wg
    fi = 0
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        eR = elem(M, mod1(ex + 1, M.Nex), ez)
        for j in 1:np
            f(fi += 1, :interior, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[np])
        end
        for i in 1:np
            if ez < M.Nez
                f(fi += 1, :interior, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[np])
            else
                f(fi += 1, :wall, i, np, e, i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[np])
            end
            ez == 1 && f(fi += 1, :wall, i, 1, e, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[1])
        end
    end
    return fi
end

"""C(y) and the slow part S(y) (momentum advection) of the DG operator."""
function frozen_data!(C, S, m::Euler2DModel, q)
    M = m.mesh
    np, Ne, D, γ, g = M.np, M.Ne, M.D, m.gas.γ, m.gas.g
    U, W, P = m.cache.U, m.cache.W, m.cache.P
    vf, gf = m.vflux, m.gflux
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        ρ = q[node_iq(np, e, 1, i, j)]
        U[i, j, e] = q[node_iq(np, e, 2, i, j)] / ρ
        W[i, j, e] = q[node_iq(np, e, 3, i, j)] / ρ
        P[i, j, e] = pressure(m.gas, q[node_iq(np, e, 4, i, j)])
    end
    fill!(S, 0.0)
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        ρa = q[node_iq(np, e, 1, i, j)]
        Θa = q[node_iq(np, e, 4, i, j)]
        sx = sz = 0.0
        for dir in 1:2, k in 1:np
            ib, jb = dir == 1 ? (k, j) : (i, k)
            ρb = q[node_iq(np, e, 1, ib, jb)]
            Θb = q[node_iq(np, e, 4, ib, jb)]
            ax = dir == 1 ? 0.5 * (M.Jξx[i, j, e] + M.Jξx[k, j, e]) : 0.5 * (M.Jηx[i, j, e] + M.Jηx[i, k, e])
            az = dir == 1 ? 0.5 * (M.Jξz[i, j, e] + M.Jξz[k, j, e]) : 0.5 * (M.Jηz[i, j, e] + M.Jηz[i, k, e])
            A, B = mass_theta_flux(vf, ρa, Θa, ρb, Θb, 1.0, γ)
            o = _vc(M, dir, i, j, k, e)
            C[o+1] = A / ρa; C[o+2] = A / ρb; C[o+3] = B / ρa; C[o+4] = B / ρb
            C[o+5] = gravity_mean(gf, ρa, ρb, γ) / (0.5 * (ρa + ρb))
            um = 0.5 * (U[i, j, e] + U[ib, jb, e])
            wm = 0.5 * (W[i, j, e] + W[ib, jb, e])
            f1 = A * (ax * um + az * wm)
            d = 2 * D[dir == 1 ? i : j, k]
            sx += d * f1 * um
            sz += d * f1 * wm
        end
        Jinv = 1 / M.J[i, j, e]
        S[node_iq(np, e, 2, i, j)] = -sx * Jinv
        S[node_iq(np, e, 3, i, j)] = -sz * Jinv
    end
    _foreach_face(M) do fi, kind, iL, jL, eL, iR, jR, eR, ñx, ñz, we
        s = hypot(ñx, ñz); n1, n2 = ñx / s, ñz / s
        o = _fc(M, fi)
        ρL = q[node_iq(np, eL, 1, iL, jL)]; θL = q[node_iq(np, eL, 4, iL, jL)] / ρL
        uL, wL, pL = U[iL, jL, eL], W[iL, jL, eL], P[iL, jL, eL]
        σL = 1 / (M.J[iL, jL, eL] * we)
        if kind === :wall
            c = m.clmars > 0 ? m.clmars : sqrt(γ * pL / ρL)
            C[o+1] = c; C[o+2] = θL
            # slow: inner flux m u_ñ (the wall LMARS mass flux vanishes)
            unñ = uL * ñx + wL * ñz
            S[node_iq(np, eL, 2, iL, jL)] += σL * q[node_iq(np, eL, 2, iL, jL)] * unñ
            S[node_iq(np, eL, 3, iL, jL)] += σL * q[node_iq(np, eL, 3, iL, jL)] * unñ
        else
            ρR = q[node_iq(np, eR, 1, iR, jR)]; θR = q[node_iq(np, eR, 4, iR, jR)] / ρR
            uR, wR, pR = U[iR, jR, eR], W[iR, jR, eR], P[iR, jR, eR]
            unL = uL * n1 + wL * n2; unR = uR * n1 + wR * n2
            ws, ps, ρc, left = _lmars_parts(γ, m.clmars, ρL, unL, pL, ρR, unR, pR)
            ρup = left ? ρL : ρR
            θup = left ? θL : θR
            C[o+1] = ρup / (2ρL); C[o+2] = ρup / (2ρR); C[o+3] = ρup / (2ρc)
            C[o+4] = θup * C[o+1]; C[o+5] = θup * C[o+2]; C[o+6] = θup * C[o+3]
            C[o+7] = ρc / (2ρL); C[o+8] = ρc / (2ρR)
            C[o+9] = θL; C[o+10] = θR
            mf = ws * ρup
            uup, wup = left ? (uL, wL) : (uR, wR)
            σR = 1 / (M.J[iR, jR, eR] * we)
            # slow: s m_f u_up minus the inner m u_ñ, on both sides (side R with -ñ, -F)
            S[node_iq(np, eL, 2, iL, jL)] -= σL * (s * mf * uup - q[node_iq(np, eL, 2, iL, jL)] * (uL * ñx + wL * ñz))
            S[node_iq(np, eL, 3, iL, jL)] -= σL * (s * mf * wup - q[node_iq(np, eL, 3, iL, jL)] * (uL * ñx + wL * ñz))
            S[node_iq(np, eR, 2, iR, jR)] -= σR * (-s * mf * uup + q[node_iq(np, eR, 2, iR, jR)] * (uR * ñx + wR * ñz))
            S[node_iq(np, eR, 3, iR, jR)] -= σR * (-s * mf * wup + q[node_iq(np, eR, 3, iR, jR)] * (uR * ñx + wR * ñz))
        end
        nothing
    end
    return C, S
end

"""Fast part F_f(C, z) + F (with reference residual and sponge) on the rows of part
(:thermo = ρ, Θ; :momentum = m_x, m_z; :all)."""
function frozen_fast!(out, m::Euler2DModel, C, z, part::Symbol, F)
    M = m.mesh
    np, Ne, D, g = M.np, M.Ne, M.D, m.gas.g
    P = m.cache.P
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        P[i, j, e] = pressure(m.gas, z[node_iq(np, e, 4, i, j)])
    end
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        s1 = s2 = s3 = s4 = gx = gz = 0.0
        ka = (node_iq(np, e, 1, i, j), node_iq(np, e, 2, i, j), node_iq(np, e, 3, i, j))
        ρa, mxa, mza, pa, za = z[ka[1]], z[ka[2]], z[ka[3]], P[i, j, e], M.Z[i, j, e]
        for dir in 1:2, k in 1:np
            ib, jb = dir == 1 ? (k, j) : (i, k)
            ax = dir == 1 ? 0.5 * (M.Jξx[i, j, e] + M.Jξx[k, j, e]) : 0.5 * (M.Jηx[i, j, e] + M.Jηx[i, k, e])
            az = dir == 1 ? 0.5 * (M.Jξz[i, j, e] + M.Jξz[k, j, e]) : 0.5 * (M.Jηz[i, j, e] + M.Jηz[i, k, e])
            ρb = z[node_iq(np, e, 1, ib, jb)]
            m̃a = ax * mxa + az * mza
            m̃b = ax * z[node_iq(np, e, 2, ib, jb)] + az * z[node_iq(np, e, 3, ib, jb)]
            o = _vc(M, dir, i, j, k, e)
            f1 = 0.5 * (C[o+1] * m̃a + C[o+2] * m̃b)
            f4 = 0.5 * (C[o+3] * m̃a + C[o+4] * m̃b)
            pm = 0.5 * (pa + P[ib, jb, e])
            G = C[o+5] * 0.5 * (ρa + ρb) * g * (M.Z[ib, jb, e] - za)
            dd = D[dir == 1 ? i : j, k]
            s1 += 2dd * f1
            s4 += 2dd * f4
            s2 += 2dd * ax * pm
            s3 += 2dd * az * pm
            gx += dd * ax * G
            gz += dd * az * G
        end
        Jinv = 1 / M.J[i, j, e]
        out[ka[1]] = -s1 * Jinv
        out[ka[2]] = -(s2 + gx) * Jinv
        out[ka[3]] = -(s3 + gz) * Jinv
        out[node_iq(np, e, 4, i, j)] = -s4 * Jinv
    end
    _foreach_face(M) do fi, kind, iL, jL, eL, iR, jR, eR, ñx, ñz, we
        s = hypot(ñx, ñz); n1, n2 = ñx / s, ñz / s
        o = _fc(M, fi)
        k1, k2, k3, k4 = node_iq(np, eL, 1, iL, jL), node_iq(np, eL, 2, iL, jL), node_iq(np, eL, 3, iL, jL), node_iq(np, eL, 4, iL, jL)
        mnL = z[k2] * n1 + z[k3] * n2
        pL = P[iL, jL, eL]
        σL = 1 / (M.J[iL, jL, eL] * we)
        if kind === :wall
            ps = pL + C[o+1] * mnL
            out[k1] -= σL * (-s * mnL)
            out[k4] -= σL * (-s * C[o+2] * mnL)
            out[k2] -= σL * s * n1 * (ps - pL)
            out[k3] -= σL * s * n2 * (ps - pL)
        else
            r1, r2, r3, r4 = node_iq(np, eR, 1, iR, jR), node_iq(np, eR, 2, iR, jR), node_iq(np, eR, 3, iR, jR), node_iq(np, eR, 4, iR, jR)
            mnR = z[r2] * n1 + z[r3] * n2
            pR = P[iR, jR, eR]
            mf = C[o+1] * mnL + C[o+2] * mnR - C[o+3] * (pR - pL)
            fΘ = C[o+4] * mnL + C[o+5] * mnR - C[o+6] * (pR - pL)
            ps = 0.5 * (pL + pR) - (C[o+8] * mnR - C[o+7] * mnL)
            σR = 1 / (M.J[iR, jR, eR] * we)
            out[k1] -= σL * s * (mf - mnL)
            out[k4] -= σL * s * (fΘ - C[o+9] * mnL)
            out[k2] -= σL * s * n1 * (ps - pL)
            out[k3] -= σL * s * n2 * (ps - pL)
            out[r1] += σR * s * (mf - mnR)
            out[r4] += σR * s * (fΘ - C[o+10] * mnR)
            out[r2] += σR * s * n1 * (ps - pR)
            out[r3] += σR * s * n2 * (ps - pR)
        end
        nothing
    end
    th = part !== :momentum
    mo = part !== :thermo
    @inbounds for e in 1:Ne, r in 1:NV, j in 1:np, i in 1:np
        k = node_iq(np, e, r, i, j)
        if (r == 1 || r == 4) ? th : mo
            out[k] += F[k] - m.nres[k]
        else
            out[k] = 0.0
        end
    end
    mo && sponge!(out, m, z)
    return out
end
