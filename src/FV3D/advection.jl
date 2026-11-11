# Advection: one scheme for flat grids, cut cells and terrain following.
#
# Face values: pseudo third-order upwind reconstruction (MATLAB `AdvectionCutThird`; Jähn et al.
# 2015, GMD 8, 317, eqs. 6 and 11–13) with virtual grid sizes from the volumes and face areas,
#     h_U = V_U / a_UC,   h_C = V_C / (½ (a_UC + a_CD)),   h_D = V_D / a_CD,
#     φ_f = φ_C + α1 (φ_D - φ_C) + α2 (φ_C - φ_U),
#     α1 = h_C (h_U + h_C) / ((h_C + h_D)(h_U + h_C + h_D)),  α2 = h_C h_D / ((h_U + h_C)(h_U + h_C + h_D)),
# (C upwind cell, U its upstream and D its downstream neighbour, a_UC, a_CD the faces of C towards
# U and D). V = Vf J and a = A J (x, y) or A (η) include the terrain-following metric. Uniform
# grid: α1 = 1/3, α2 = 1/6 (κ = 1/3). A closed upstream face or a wall gives h_U = ∞: the linear
# interpolation φ_C + h_C/(h_C + h_D) (φ_D - φ_C) (the centred mean on uniform grids); h_C takes
# the geometric area of a wall face (top, flat and terrain-following ground 1, below a cut cell 0). As in the
# MATLAB code, neighbours smaller than C are blended towards φ_C with their volume fraction.
#
# Momentum: shift technique (Hicken et al. 2005; Jähn et al. 2015, eqs. 14–15). Each cell gets a
# left and a right value of each velocity component from its two faces in that direction,
#     u_LC = u_L                          if a_L ≥ a_R,   else (a_L u_L + (a_R - a_L) u_R)/a_R,
# (u_RC likewise), the six cell fields are advected as cell-centred mixing ratios with the mass
# fluxes of the scalars, and the cell tendencies return to the face with the volume weights
#     T_f = (V_R T_LC(R) + V_L T_RC(L)) / (V_L + V_R),
#     V_R = V(R) a_f / (a_f + a_R'),  V_L = V(L) a_f / (a_f + a_L'),
# a_R', a_L' the opposite faces of the cells R and L. The shifted quantities are the face
# velocities u = U/ρ̄ (a uniform flow is preserved exactly). The wall faces enter the shift with
# their geometric area: flat ground and the top with w = 0, terrain-following ground with
# w = z_x u + z_y v, the ground below a cut cell with area 0.

# recon3, shiftvel, shift_back: shared with FV2D (src/FV/shift_advection.jl)

"""Geometric volume Vf J of cell (i, j, k) (zero in solid cells)."""
@inline vol(g, i, j, k) = g.Vf[ip(g, i), jp(g, j), k] * g.Jc[ip(g, i), jp(g, j)]
"""Open face measures: left x-face, left y-face (A J), lower η-face (A; walls 0)."""
@inline ax(g, i, j, k) = g.Ax[ip(g, i), jp(g, j), k] * g.Jx[ip(g, i), jp(g, j)]
@inline ay(g, i, j, k) = g.Ay[ip(g, i), jp(g, j), k] * g.Jy[ip(g, i), jp(g, j)]
@inline az(g, i, j, k) = g.Az[ip(g, i), jp(g, j), k]
"""η-face measure of the shift: the open area inside, the geometric area at the walls (top 1;
ground 1 on flat and terrain-following grids, 0 below a cut cell)."""
@inline azs(g, i, j, k) = k == 1 ? ((g.follow || g.full[ip(g, i), jp(g, j), 1]) ? 1.0 : 0.0) :
                          k == g.nz + 1 ? 1.0 : g.Az[ip(g, i), jp(g, j), k]

"""Face value at the x-face i (between i-1 and i) of the cell field φ, upwind by the sign of F."""
@inline function face_x(g, φ, i, j, k, F)
    if !signbit(F)
        return recon3(φ[ip(g, i - 2), j, k], φ[ip(g, i - 1), j, k], φ[ip(g, i), j, k],
                      vol(g, i - 2, j, k), vol(g, i - 1, j, k), vol(g, i, j, k), ax(g, i - 1, j, k), ax(g, i, j, k))
    else
        return recon3(φ[ip(g, i + 1), j, k], φ[ip(g, i), j, k], φ[ip(g, i - 1), j, k],
                      vol(g, i + 1, j, k), vol(g, i, j, k), vol(g, i - 1, j, k), ax(g, i + 1, j, k), ax(g, i, j, k))
    end
end
@inline function face_y(g, φ, i, j, k, F)
    if !signbit(F)
        return recon3(φ[i, jp(g, j - 2), k], φ[i, jp(g, j - 1), k], φ[i, jp(g, j), k],
                      vol(g, i, j - 2, k), vol(g, i, j - 1, k), vol(g, i, j, k), ay(g, i, j - 1, k), ay(g, i, j, k))
    else
        return recon3(φ[i, jp(g, j + 1), k], φ[i, jp(g, j), k], φ[i, jp(g, j - 1), k],
                      vol(g, i, j + 1, k), vol(g, i, j, k), vol(g, i, j - 1, k), ay(g, i, j + 1, k), ay(g, i, j, k))
    end
end
"""Face value at the η-face k (2 … nz); beyond the walls the upstream face is closed."""
@inline function face_z(g, φ, i, j, k, F)
    if !signbit(F)
        ku = max(k - 2, 1)
        return recon3(φ[i, j, ku], φ[i, j, k-1], φ[i, j, k], vol(g, i, j, ku), vol(g, i, j, k - 1), vol(g, i, j, k),
                      azs(g, i, j, k - 1), az(g, i, j, k), az(g, i, j, k - 1) > 0)
    else
        ku = min(k + 1, g.nz)
        return recon3(φ[i, j, ku], φ[i, j, k], φ[i, j, k-1], vol(g, i, j, ku), vol(g, i, j, k), vol(g, i, j, k - 1),
                      azs(g, i, j, k + 1), az(g, i, j, k), az(g, i, j, k + 1) > 0)
    end
end

"""Mass fluxes A J U, A J V (x-, y-faces) and A Ω (η-faces) into the cache."""
function mass_fluxes!(m::Model3D, q)
    g, c = m.grid, m.cache
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        c.Fx[i, j, k] = ax(g, i, j, k) * q[iU(g, i, j, k)]
        c.Fy[i, j, k] = ay(g, i, j, k) * q[iV(g, i, j, k)]
    end
    @inbounds for k in 1:g.nz+1, j in 1:g.ny, i in 1:g.nx
        c.Fz[i, j, k] = (k == 1 || k == g.nz + 1) ? 0.0 : az(g, i, j, k) * Ωface(g, q, i, j, k)
    end
    return nothing
end

"""Advective tendency T = -(1/(Vf J)) ∮ F φ_f of the cell field φ (mass fluxes from the cache)."""
function advect_cells!(T, φ, m::Model3D)
    g, c = m.grid, m.cache
    fill!(T, 0.0)
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        F = c.Fx[i, j, k]
        if F != 0
            f = F * face_x(g, φ, i, j, k, F) / g.dx
            T[ip(g, i - 1), j, k] -= f; T[i, j, k] += f
        end
        F = c.Fy[i, j, k]
        if F != 0
            f = F * face_y(g, φ, i, j, k, F) / g.dy
            T[i, jp(g, j - 1), k] -= f; T[i, j, k] += f
        end
        F = c.Fz[i, j, k]
        if F != 0
            f = F * face_z(g, φ, i, j, k, F) / g.dz
            T[i, j, k-1] -= f; T[i, j, k] += f
        end
    end
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        T[i, j, k] *= g.invV[i, j, k] / g.Jc[i, j]
    end
    return T
end

"""Momentum advection by the shift technique into the U, V, W rows of S (face velocities u, v, w
in the cache)."""
function momentum_advection!(S, m::Model3D)
    g, c = m.grid, m.cache
    uL, uR, vL, vR, wL, wR, TuL, TuR, TvL, TvR, TwL, TwR = c.sh
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        ie, jn = ip(g, i + 1), jp(g, j + 1)
        uL[i, j, k], uR[i, j, k] = vol(g, i, j, k) > 0 ? shiftvel(ax(g, i, j, k), ax(g, ie, j, k), c.u[i, j, k], c.u[ie, j, k]) : (0.0, 0.0)
        vL[i, j, k], vR[i, j, k] = vol(g, i, j, k) > 0 ? shiftvel(ay(g, i, j, k), ay(g, i, jn, k), c.v[i, j, k], c.v[i, jn, k]) : (0.0, 0.0)
        wL[i, j, k], wR[i, j, k] = vol(g, i, j, k) > 0 ? shiftvel(azs(g, i, j, k), azs(g, i, j, k + 1), c.w[i, j, k], c.w[i, j, k+1]) : (0.0, 0.0)
    end
    for (T, φ) in ((TuL, uL), (TuR, uR), (TvL, vL), (TvR, vR), (TwL, wL), (TwR, wR))
        advect_cells!(T, φ, m)
    end
    @inbounds for k in 1:g.nz, j in 1:g.ny, i in 1:g.nx
        im, jm = ip(g, i - 1), jp(g, j - 1)
        a = ax(g, i, j, k)
        S[iU(g, i, j, k)] = a > 0 ? shift_back(TuL[i, j, k], TuR[im, j, k], vol(g, im, j, k), vol(g, i, j, k),
                                               a, ax(g, im, j, k), ax(g, i + 1, j, k)) : 0.0
        a = ay(g, i, j, k)
        S[iV(g, i, j, k)] = a > 0 ? shift_back(TvL[i, j, k], TvR[i, jm, k], vol(g, i, jm, k), vol(g, i, j, k),
                                               a, ay(g, i, jm, k), ay(g, i, j + 1, k)) : 0.0
        if k > 1
            a = az(g, i, j, k)
            S[iW(g, i, j, k)] = a > 0 ? shift_back(TwL[i, j, k], TwR[i, j, k-1], vol(g, i, j, k - 1), vol(g, i, j, k),
                                                   a, azs(g, i, j, k - 1), azs(g, i, j, k + 1)) : 0.0
        end
    end
    return S
end
