# Fast operator (acoustic-gravity part linearized about the frozen state), C-grid

mutable struct FVFast <: AbstractFastOperator
    grid::FVGrid
    gas::Gas
    nq::Int
    a::Matrix{Float64}             # γ p/Θ at the centres (nx, nz)
    b::Matrix{Float64}             # (1 - γ) p at the centres
    θx::Matrix{Float64}            # θ̄ at the x-faces (nx, nz)
    θz::Matrix{Float64}            # θ̄ at the z-faces (nx, nz+1), walls unused
    fb::Vector{Float64}            # constant forcing of the fast operator
    Mdiag::Vector{Float64}         # mass (ones: tendencies per unit volume)
    work::Vector{Vector{Float64}}  # 7 work vectors for the fast integrators
    lucache::Dict{Float64,Any}     # c -> tridiagonal factors of I - c A_v
    cmax::Float64
    colbuf::Vector{Float64}        # column work (nz + 1)
    vadv::Bool                     # FV_VADV_FAST at the last re-freeze
    uσ::Matrix{Float64}            # frozen u at the (x-face, z-face) corners (nx, nz+1)
    θc::Matrix{Float64}            # θ̄ at the centres (terrain operator)
end

"""
Include in the vertical fast part the transport of x-momentum by the fast vertical mass flux,
U_t ∋ -∂_η(W' ū)/(J_x), with ū frozen at the corners. Left in the explicit slow part, this term
couples the large step to the vertical acoustics with Δt ū/Δz (Agnesi a100: Δt ū/Δz ≈ 22).
The column solve updates U from the solved W afterwards (exact: the column system does not
read U). Off by default; switch it on only when the vertical Courant number of the slow step
U Δt/Δz is large (the cut-cell `:rosenbrock_amf` has advection in its L anyway).
"""
const FV_VADV_FAST = Ref(false)

function FVFast(m::FVModel)
    g = m.grid
    nq = m.nq
    return FVFast(g, m.gas, nq, zeros(g.nx, g.nz), zeros(g.nx, g.nz), zeros(g.nx, g.nz), zeros(g.nx, g.nz + 1),
                  zeros(nq), ones(nq), [zeros(nq) for _ in 1:7], Dict{Float64,Any}(), 0.0, zeros(g.nz + 1),
                  false, zeros(g.nx, g.nz + 1), zeros(g.nx, g.nz))
end

"""
    fast_operator(model, q; pfac=γ, fast=:imex, reuse=true)

Fast operator frozen at q: a = pfac p/Θ, b = (1 - pfac) p, θ̄ = Θ/ρ averaged to the faces.
With reuse = true the operator stored in the model is re-frozen in place.
"""
function fast_operator(m::FVModel, q; pfac = m.gas.γ, fast::Symbol = :imex, reuse::Bool = true, kwargs...)
    m.grid isa CutCellFVGrid && return cutcell_fast_operator(m,q;pfac,fast,reuse,kwargs...)
    s = reuse && m.fastref[] isa FVFast ? m.fastref[]::FVFast : FVFast(m)
    refreeze!(s, m, q, pfac)
    reuse && (m.fastref[] = s)
    return s
end

function refreeze!(s::FVFast, m::FVModel, q, pfac)
    g = s.grid
    nx = g.nx
    nz = g.nz
    s.cmax = 0.0
    @inbounds for k in 1:nz, i in 1:nx
        Θ = q[iΘ(g, i, k)]
        p = pressure(m.gas, Θ)
        s.a[i, k] = pfac * p / Θ
        s.b[i, k] = (1 - pfac) * p
        s.cmax = max(s.cmax, sqrt(m.gas.γ * p / q[iρ(g, i, k)]))
    end
    @inbounds for k in 1:nz, i in 1:nx
        θl = q[iΘ(g, i - 1, k)] / q[iρ(g, i - 1, k)]
        θr = q[iΘ(g, i, k)] / q[iρ(g, i, k)]
        s.θc[i, k] = θr
        s.θx[i, k] = 0.5 * (θl + θr)
        k == 1 && (s.θz[i, 1] = θr)
        if k > 1
            θb = q[iΘ(g, i, k - 1)] / q[iρ(g, i, k - 1)]
            s.θz[i, k] = 0.5 * (θb + θr)
        end
        k == nz && (s.θz[i, nz+1] = θr)
    end
    s.vadv = FV_VADV_FAST[]
    @inbounds for i in 1:nx
        ux(k) = q[iU(g, i, k)] / (0.5 * (q[iρ(g, i - 1, k)] + q[iρ(g, i, k)]))
        s.uσ[i, 1] = 0.0
        s.uσ[i, nz+1] = 0.0
        for k in 2:nz
            s.uσ[i, k] = 0.5 * (ux(k - 1) + ux(k))
        end
    end
    fill!(s.fb, 0.0)
    grav = m.gas.g
    @inbounds for k in 1:nz, i in 1:nx
        if g.terrain
            im = ip(g, i - 1)
            bbot = k == 1 ? 0.5 * (s.b[im, 1] + s.b[i, 1]) :
                   0.25 * (s.b[im,k-1] + s.b[im,k] + s.b[i,k-1] + s.b[i,k])
            btop = k == nz ? 0.5 * (s.b[im, nz] + s.b[i, nz]) :
                   0.25 * (s.b[im,k] + s.b[im,k+1] + s.b[i,k] + s.b[i,k+1])
            s.fb[iU(g, i, k)] = -(g.Jc[i] * s.b[i, k] - g.Jc[im] * s.b[im, k]) / (g.Jx[i] * g.dx) +
                (g.zx_xf[i,k+1] * btop - g.zx_xf[i,k] * bbot) / (g.Jx[i] * g.dz)
        else
            s.fb[iU(g, i, k)] = -(s.b[i, k] - s.b[ip(g, i - 1), k]) / g.dx
        end
        if k > 1
            if g.terrain
                s.fb[iW(g, i, k)] = (-(s.b[i, k] - s.b[i, k-1]) + (m.pbal[i, k] - m.pbal[i, k-1])) /
                    (g.Jc[i] * g.dz) + grav * 0.5 * (m.ρbal[i, k-1] + m.ρbal[i, k])
            else
                s.fb[iW(g, i, k)] = -(s.b[i, k] - s.b[i, k-1]) / g.dz + (m.pbal[i, k] - m.pbal[i, k-1]) / g.dz +
                                    grav * 0.5 * (m.ρbal[i, k-1] + m.ρbal[i, k])
            end
        end
    end
    for (_, F) in s.lucache
        factor_vertical!(F, s, F.c)
    end
    return s
end

"""
    fv_fast!(out, s, q, f, hor, ver, rows, frows)

Fast tendency: horizontal (hor) and/or vertical (ver) part of the rows selected by
rows = (ρ, U, W, Θ), plus the forcing f on the rows selected by frows; all other rows are zero.
"""
function fv_fast!(out, s::FVFast, q, f, hor::Bool, ver::Bool, rows::NTuple{4,Bool}, frows::NTuple{4,Bool})
    s.grid.terrain && return fv_fast_terrain!(out, s, q, f, hor, ver, rows, frows)
    g = s.grid
    nx = g.nx
    nz = g.nz
    dx = g.dx
    dz = g.dz
    grav = s.gas.g
    rρ, rU, rW, rΘ = rows
    fill!(out, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        if hor
            Ul = q[iU(g, i, k)]
            Ur = q[iU(g, i + 1, k)]
            rρ && (out[iρ(g, i, k)] -= (Ur - Ul) / dx)
            rΘ && (out[iΘ(g, i, k)] -= (s.θx[ip(g, i + 1), k] * Ur - s.θx[i, k] * Ul) / dx)
            rU && (out[iU(g, i, k)] -= (s.a[i, k] * q[iΘ(g, i, k)] - s.a[ip(g, i - 1), k] * q[iΘ(g, i - 1, k)]) / dx)
        end
        if ver
            Wb = Wface(g, q, i, k)
            Wt = Wface(g, q, i, k + 1)
            rρ && (out[iρ(g, i, k)] -= (Wt - Wb) / dz)
            if rΘ
                θt = k < nz ? s.θz[i, k+1] : 0.0
                θb = k > 1 ? s.θz[i, k] : 0.0
                out[iΘ(g, i, k)] -= (θt * Wt - θb * Wb) / dz
            end
            if rW && k > 1
                out[iW(g, i, k)] -= (s.a[i, k] * q[iΘ(g, i, k)] - s.a[i, k-1] * q[iΘ(g, i, k - 1)]) / dz +
                                    grav * 0.5 * (q[iρ(g, i, k - 1)] + q[iρ(g, i, k)])
            end
            rU && s.vadv && (out[iU(g, i, k)] -= fv_uσ_flux_div(s, q, i, k) / dz)
        end
    end
    fρ, fU, fW, fΘ = frows
    @inbounds for k in 1:nz, i in 1:nx
        fρ && (out[iρ(g, i, k)] += f[iρ(g, i, k)])
        fU && (out[iU(g, i, k)] += f[iU(g, i, k)])
        fΘ && (out[iΘ(g, i, k)] += f[iΘ(g, i, k)])
        fW && k > 1 && (out[iW(g, i, k)] += f[iW(g, i, k)])
    end
    return out
end

"""Difference of the corner fluxes ½(W_{i-1} + W_i) ū at the z-faces k+1 and k of the x-face i
(walls: zero)."""
@inline function fv_uσ_flux_div(s, q, i, k)
    g = s.grid
    im = ip(g, i - 1)
    Fb = k > 1 ? 0.5 * (q[iW(g, im, k)] + q[iW(g, i, k)]) * s.uσ[i, k] : 0.0
    Ft = k < g.nz ? 0.5 * (q[iW(g, im, k+1)] + q[iW(g, i, k+1)]) * s.uσ[i, k+1] : 0.0
    return Ft - Fb
end

const ALL = (true, true, true, true)
const NONE = (false, false, false, false)
const THERMO = (true, false, false, true)
const MOMENTUM = (false, true, true, false)
const HMOM = (false, true, false, false)
const NOT_HMOM = (true, false, true, true)

"""
Ground σ-face in the chain-rule pressure gradient of the terrain fast operator: with `true` the
η-difference is extrapolated as in N (`:advective`/`:flux` with fv_p_sigma), otherwise the face is
left out (half of z_x ∂_η p at k = 1; exactly adjoint to the mass divergence, but N - L then holds
an acoustic term that is explicit in MIS: 2Δx noise along the ground for steep hills, a1).
"""
const FV_FAST_GROUND_GRADIENT = Ref(true)

"""
Terrain (Gal-Chen) fast operator in an energy-consistent form.
- Horizontal part: mass divergence D(U) = δ_x(J_x U)/(J Δx) - δ_η(z_x Ũ)/(J Δη) (σ-correction
  on the interior σ-faces; zero at the ground, where W = z_x U); Θ' ← -θ̄ D(U), so horizontal
  momentum does not advect the background θ̄ (it varies along σ-surfaces, but θ̄ = θ̄(z)).
  The x-momentum gets the exact adjoint (weights J a cells, J_x θ̄_x x-faces):
  U ← -δ_x(θ̄P)/(θ̄_x Δx) + chain-rule term (z_x/J) δ_η(θ̄P)/θ̄_x averaged to the x-face.
- Vertical part: the flat one with J Δη (Θ' ← -δ_η(θ̄_z W) carries w' dθ̄/dz).
`N` keeps the conservative form; the difference is slow (W-method).
"""
function fv_fast_terrain!(out, s::FVFast, q, f, hor, ver, rows, frows)
    g = s.grid
    nx, nz, dx, dη = g.nx, g.nz, g.dx, g.dz
    grav = s.gas.g
    rρ, rU, rW, rΘ = rows
    fill!(out, 0.0)
    P(i, k) = s.a[ip(g, i), k] * q[iΘ(g, i, k)]
    # 4-point average of U around the σ-face kf of cell column i
    Ũ(i, kf) = 0.25 * (q[iU(g, i, kf-1)] + q[iU(g, i+1, kf-1)] + q[iU(g, i, kf)] + q[iU(g, i+1, kf)])
    @inbounds for k in 1:nz, i in 1:nx
        im, ir = ip(g, i-1), ip(g, i+1)
        J = g.Jc[i]
        Ul, Ur = q[iU(g, i, k)], q[iU(g, i+1, k)]
        θl, θr = s.θx[i,k], s.θx[ir,k]
        if hor
            Sb = k > 1 ? g.zx_f[i,k] * Ũ(i, k) : 0.0          # z_x U at the σ-faces (interior)
            St = k < nz ? g.zx_f[i,k+1] * Ũ(i, k+1) : 0.0
            rρ && (out[iρ(g,i,k)] -= (g.Jx[ir]*Ur - g.Jx[i]*Ul) / (J*dx) - (St - Sb) / (J*dη))
            rΘ && (out[iΘ(g,i,k)] -= s.θc[i,k] * ((g.Jx[ir]*Ur - g.Jx[i]*Ul) / (J*dx) - (St - Sb) / (J*dη)))
            if rU
                m = 0.0
                θP(ii, kk) = s.θc[ii,kk] * P(ii, kk)
                for ii in (im, i), kf in (k, k+1)
                    if 1 < kf <= nz
                        m += g.zx_f[ii,kf] * (θP(ii, kf) - θP(ii, kf-1))
                    elseif FV_FAST_GROUND_GRADIENT[] && kf == 1   # ground face: extrapolated difference as in N
                        m += g.zx_f[ii,1] * (2 * (θP(ii, 2) - θP(ii, 1)) - (θP(ii, 3) - θP(ii, 2)))
                    end
                end
                out[iU(g,i,k)] -= ((s.θc[i,k]*P(i, k) - s.θc[im,k]*P(im, k)) / dx - m / (4 * g.Jx[i] * dη)) / s.θx[i,k]
            end
        end
        if ver
            Wb = k == 1 ? 0.0 : q[iW(g,i,k)]
            Wt = k == nz ? 0.0 : q[iW(g,i,k+1)]
            rρ && (out[iρ(g,i,k)] -= (Wt-Wb)/(J*dη))
            rΘ && (out[iΘ(g,i,k)] -= (s.θz[i,k+1]*Wt - s.θz[i,k]*Wb)/(J*dη))
            if rW && k > 1
                out[iW(g,i,k)] -= (P(i,k) - P(i,k-1))/(J*dη) + grav*0.5*(q[iρ(g,i,k-1)] + q[iρ(g,i,k)])
            end
            rU && s.vadv && (out[iU(g,i,k)] -= fv_uσ_flux_div(s, q, i, k) / (g.Jx[i]*dη))
        end
    end
    fρ, fU, fW, fΘ = frows
    @inbounds for k in 1:nz, i in 1:nx
        fρ && (out[iρ(g,i,k)] += f[iρ(g,i,k)])
        fU && (out[iU(g,i,k)] += f[iU(g,i,k)])
        fΘ && (out[iΘ(g,i,k)] += f[iΘ(g,i,k)])
        fW && k > 1 && (out[iW(g,i,k)] += f[iW(g,i,k)])
    end
    return out
end

# TimeStepping interface
fast_apply!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, true, ALL, ALL)
fast_rhs_thermo!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, true, THERMO, THERMO)
fast_rhs_momentum!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, true, MOMENTUM, MOMENTUM)
fast_rhs_horizontal!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, false, ALL, ALL)
vertical_apply!(out, s::FVFast, q) = fv_fast!(out, s, q, s.fb, false, true, ALL, NONE)
fast_rhs_hmomentum!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, false, HMOM, HMOM)
fast_rhs_hthermo!(out, s::FVFast, q, f) = fv_fast!(out, s, q, f, true, false, THERMO, NOT_HMOM)

"""Dense matrix of the linear part of the fast operator (tests and spectra; small grids only)."""
function fast_matrix(s::FVFast)
    A = zeros(s.nq, s.nq)
    e = zeros(s.nq)
    col = zeros(s.nq)
    z = zeros(s.nq)
    for k in 1:s.nq
        e[k] = 1.0
        fv_fast!(col, s, e, z, true, true, ALL, NONE)
        A[:, k] .= col
        e[k] = 0.0
    end
    return A
end
