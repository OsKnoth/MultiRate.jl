# Advection: one scheme for flat grids, terrain following (FVGrid) and cut cells (CutCellFVGrid),
# the same as in 3D (src/FV3D/advection.jl).
#
# Face values: pseudo third-order upwind reconstruction (MATLAB `AdvectionCutThird`; Jähn et al.
# 2015, GMD 8, 317, eqs. 6 and 11–13) with virtual grid sizes from the volumes and face areas,
#     h_U = V_U / a_UC,   h_C = V_C / (½ (a_UC + a_CD)),   h_D = V_D / a_CD,
#     φ_f = φ_C + α1 (φ_D - φ_C) + α2 (φ_C - φ_U),
#     α1 = h_C (h_U + h_C) / ((h_C + h_D)(h_U + h_C + h_D)),  α2 = h_C h_D / ((h_U + h_C)(h_U + h_C + h_D)),
# (C upwind cell, U its upstream and D its downstream neighbour, a_UC, a_CD the faces of C towards
# U and D). Terrain following: V = J, a = Jx (x), 1 (η); cut cells: V = Vol/(Δx Δz), a = FU/Δz,
# FW/Δx. Uniform grid: α1 = 1/3, α2 = 1/6 (κ = 1/3). A closed upstream face or a wall gives
# h_U = ∞: the linear interpolation φ_C + h_C/(h_C + h_D) (φ_D - φ_C) (the centred mean on uniform
# grids); h_C takes the geometric area of a wall face (top, flat and terrain-following ground 1,
# the ground below a cut cell 0). As in the MATLAB code, neighbours smaller than C are blended towards φ_C with their
# volume fraction.
#
# Momentum: shift technique (Hicken et al. 2005; Jähn et al. 2015, eqs. 14–15). Each cell gets a
# left and a right value of each velocity component from its two faces in that direction,
#     u_LC = u_L                          if a_L ≥ a_R,   else (a_L u_L + (a_R - a_L) u_R)/a_R,
# (u_RC likewise), the cell fields are advected as cell-centred mixing ratios with the mass fluxes
# of the scalars, and the cell tendencies return to the face with the volume weights
#     T_f = (V_R T_LC(R) + V_L T_RC(L)) / (V_L + V_R),
#     V_R = V(R) a_f / (a_f + a_R'),  V_L = V(L) a_f / (a_f + a_L'),
# a_R', a_L' the opposite faces of the cells R and L. The shifted quantities are the face
# velocities u = U/ρ̄ (a uniform flow is preserved exactly). The wall faces enter the shift with
# their geometric area: flat ground and the top with w = 0, terrain-following ground with
# w = z_x u, the ground below a cut cell with area 0.

"""Pseudo third-order face value in the upwind cell C (upstream U, downstream D): volumes v,
geometric faces a_UC, a_CD of C (a_UC of a wall is its geometric area); `up = false` (wall or
closed face, no upstream neighbour): linear interpolation."""
@inline function recon3(φU, φC, φD, vU, vC, vD, aUC, aCD, up::Bool = aUC > 0)
    vD < vC && (φD = ((vC - vD) * φC + vD * φD) / vC)
    hC = vC / (0.5 * (aUC + aCD)); hD = vD / aCD
    up || return φC + hC / (hC + hD) * (φD - φC)
    vU < vC && (φU = ((vC - vU) * φC + vU * φU) / vC)
    hU = vU / aUC
    s = hU + hC + hD
    return φC + hC * (hU + hC) / ((hC + hD) * s) * (φD - φC) + hC * hD / ((hU + hC) * s) * (φC - φU)
end

"""Left/right cell values of a face field (MATLAB `ShiftFaceCellCut`)."""
@inline shiftvel(aL, aR, vL, vR) = aL >= aR ? (vL, aL > 0 ? (aR * vR + (aL - aR) * vL) / aL : vR) :
                                              ((aL * vL + (aR - aL) * vR) / aR, vR)

"""Volume-weighted face tendency from T_LC of the right cell and T_RC of the left cell (MATLAB
`ShiftCellFaceCut`): volumes vL, vR, the face a and the opposite faces aL', aR'."""
@inline shift_back(TLC_R, TRC_L, vL, vR, a, aLo, aRo) =
    (VR = vR * a / (a + aRo); VL = vL * a / (a + aLo); (VR * TLC_R + VL * TRC_L) / (VL + VR))

# ------------------------------------------------------------------ geometry of the two grids
"""Geometric volume (fraction of Δx Δη) of cell (i, k)."""
@inline avol(g::FVGrid, i, k) = g.Jc[ip(g, i)]
@inline avol(g::CutCellFVGrid, i, k) = g.alpha[ip(g, i), k]
"""Open measure of the left x-face of cell (i, k)."""
@inline aax(g::FVGrid, i, k) = g.Jx[ip(g, i)]
@inline aax(g::CutCellFVGrid, i, k) = g.alphaU[ip(g, i), k]
"""Open measure of the lower η-face k of column i (walls 0)."""
@inline aaz(g::FVGrid, i, k) = (k == 1 || k == g.nz + 1) ? 0.0 : 1.0
@inline aaz(g::CutCellFVGrid, i, k) = (k == 1 || k == g.nz + 1) ? 0.0 : g.alphaW[ip(g, i), k]
"""η-face measure of the shift (geometric at the walls: top 1, ground 1 or 0 below a cut cell)."""
@inline aazs(g::FVGrid, i, k) = 1.0
@inline aazs(g::CutCellFVGrid, i, k) = k == 1 ? (g.alpha[ip(g, i), 1] == 1 ? 1.0 : 0.0) :
                                       k == g.nz + 1 ? 1.0 : g.alphaW[ip(g, i), k]
"""1/(V J) of the flux divergence."""
@inline ainv(g::FVGrid, i, k) = 1 / g.Jc[ip(g, i)]
@inline ainv(g::CutCellFVGrid, i, k) = g.alpha[ip(g, i), k] > 0 ? 1 / g.alpha[ip(g, i), k] : 0.0
"""Mass flux through the η-face k (2 … nz): Ω = W - z_x Ū (terrain following) or FW/Δx W."""
@inline aFz(g::FVGrid, q, i, k) = q[iW(g, i, k)] - (g.terrain ? g.zx_f[ip(g, i), k] *
    0.25 * (q[iU(g, i, k-1)] + q[iU(g, i+1, k-1)] + q[iU(g, i, k)] + q[iU(g, i+1, k)]) : 0.0)
@inline aFz(g::CutCellFVGrid, q, i, k) = g.alphaW[ip(g, i), k] * q[iW(g, i, k)]
"""Physical w at the ground: z_x ū (terrain following, mean of the face velocities uF), 0 on
Cartesian grids."""
@inline wground(g::FVGrid, uF, i) = g.terrain ? g.zx_f[ip(g, i), 1] * 0.5 * (uF[ip(g, i), 1] + uF[ip(g, i + 1), 1]) : 0.0
@inline wground(g::CutCellFVGrid, uF, i) = 0.0

"""Work arrays of the advection (nx, nz) or (nx, nz+1)."""
struct AdvCache
    Fx::Matrix{Float64}; Fz::Matrix{Float64}
    θ::Matrix{Float64}; θfx::Matrix{Float64}; θfz::Matrix{Float64}
    uF::Matrix{Float64}; wF::Matrix{Float64}
    SU::Matrix{Float64}; SW::Matrix{Float64}
    sh::NTuple{8,Matrix{Float64}}     # u, w (L, R) and their tendencies
end
AdvCache(g) = (n = (g.nx, g.nz); n1 = (g.nx, g.nz + 1);
    AdvCache(zeros(n), zeros(n1), zeros(n), zeros(n), zeros(n1), zeros(n), zeros(n1), zeros(n), zeros(n1),
             ntuple(_ -> zeros(n), 8)))

@inline function aface_x(g, φ, i, k, F)
    !signbit(F) ? recon3(φ[ip(g, i - 2), k], φ[ip(g, i - 1), k], φ[ip(g, i), k], avol(g, i - 2, k), avol(g, i - 1, k),
                         avol(g, i, k), aax(g, i - 1, k), aax(g, i, k)) :
                  recon3(φ[ip(g, i + 1), k], φ[ip(g, i), k], φ[ip(g, i - 1), k], avol(g, i + 1, k), avol(g, i, k),
                         avol(g, i - 1, k), aax(g, i + 1, k), aax(g, i, k))
end
@inline function aface_z(g, φ, i, k, F)
    if !signbit(F)
        ku = max(k - 2, 1)
        return recon3(φ[i, ku], φ[i, k-1], φ[i, k], avol(g, i, ku), avol(g, i, k - 1), avol(g, i, k),
                      aazs(g, i, k - 1), aaz(g, i, k), aaz(g, i, k - 1) > 0)
    else
        ku = min(k + 1, g.nz)
        return recon3(φ[i, ku], φ[i, k], φ[i, k-1], avol(g, i, ku), avol(g, i, k), avol(g, i, k - 1),
                      aazs(g, i, k + 1), aaz(g, i, k), aaz(g, i, k + 1) > 0)
    end
end

"""Advective tendency T = -(1/(V J)) ∮ F φ_f of the cell field φ (mass fluxes in a)."""
function aadvect_cells!(T, φ, g, a::AdvCache)
    fill!(T, 0.0)
    @inbounds for k in 1:g.nz, i in 1:g.nx
        F = a.Fx[i, k]
        if F != 0
            f = F * aface_x(g, φ, i, k, F) / g.dx
            T[ip(g, i - 1), k] -= f; T[i, k] += f
        end
        F = a.Fz[i, k]
        if F != 0
            f = F * aface_z(g, φ, i, k, F) / g.dz
            T[i, k-1] -= f; T[i, k] += f
        end
    end
    @inbounds for k in 1:g.nz, i in 1:g.nx
        T[i, k] *= ainv(g, i, k)
    end
    return T
end

"""
    shift_advection!(a, g, q)

Advection of the state q: the θ face values `a.θfx` (x-face i of cell i), `a.θfz` (η-face k = 2 … nz)
upwind by the mass fluxes `a.Fx`, `a.Fz`, and the momentum tendencies `a.SU`, `a.SW` (per unit
volume, U and W rows; zero on closed faces) by the shift technique.
"""
function shift_advection!(a::AdvCache, g, q)
    nx, nz = g.nx, g.nz
    uL, uR, wL, wR, TuL, TuR, TwL, TwR = a.sh
    @inbounds for k in 1:nz, i in 1:nx
        ρ = q[iρ(g, i, k)]
        a.θ[i, k] = avol(g, i, k) > 0 ? q[iΘ(g, i, k)] / ρ : 0.0
        a.Fx[i, k] = aax(g, i, k) * q[iU(g, i, k)]
        ρf = 0.5 * (q[iρ(g, i - 1, k)] + ρ)
        a.uF[i, k] = aax(g, i, k) > 0 ? q[iU(g, i, k)] / ρf : 0.0
    end
    @inbounds for i in 1:nx
        a.Fz[i, 1] = 0.0; a.Fz[i, nz+1] = 0.0
        a.wF[i, 1] = wground(g, a.uF, i); a.wF[i, nz+1] = 0.0
        for k in 2:nz
            a.Fz[i, k] = aaz(g, i, k) > 0 ? aFz(g, q, i, k) : 0.0
            a.wF[i, k] = aaz(g, i, k) > 0 ? q[iW(g, i, k)] / (0.5 * (q[iρ(g, i, k - 1)] + q[iρ(g, i, k)])) : 0.0
        end
    end
    @inbounds for k in 1:nz, i in 1:nx
        a.θfx[i, k] = aax(g, i, k) > 0 ? aface_x(g, a.θ, i, k, a.Fx[i, k]) : 0.0
        k > 1 && (a.θfz[i, k] = aaz(g, i, k) > 0 ? aface_z(g, a.θ, i, k, a.Fz[i, k]) : 0.0)
        if avol(g, i, k) > 0
            uL[i, k], uR[i, k] = shiftvel(aax(g, i, k), aax(g, i + 1, k), a.uF[i, k], a.uF[ip(g, i + 1), k])
            wL[i, k], wR[i, k] = shiftvel(aazs(g, i, k), aazs(g, i, k + 1), a.wF[i, k], a.wF[i, k+1])
        else
            uL[i, k] = uR[i, k] = wL[i, k] = wR[i, k] = 0.0
        end
    end
    for (T, φ) in ((TuL, uL), (TuR, uR), (TwL, wL), (TwR, wR))
        aadvect_cells!(T, φ, g, a)
    end
    fill!(a.SW, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        im = ip(g, i - 1)
        f = aax(g, i, k)
        a.SU[i, k] = f > 0 ? shift_back(TuL[i, k], TuR[im, k], avol(g, im, k), avol(g, i, k), f, aax(g, im, k), aax(g, i + 1, k)) : 0.0
        if k > 1
            f = aaz(g, i, k)
            a.SW[i, k] = f > 0 ? shift_back(TwL[i, k], TwR[i, k-1], avol(g, i, k - 1), avol(g, i, k), f,
                                            aazs(g, i, k - 1), aazs(g, i, k + 1)) : 0.0
        end
    end
    return a
end
