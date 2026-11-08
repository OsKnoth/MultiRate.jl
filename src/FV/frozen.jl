# Frozen-part split of the flat FV right-hand side for `TimeStepping.run_frozen`
#
# N(q) = F(q, q) with
#   fast (evolving z, coefficients from y):
#     ρ_t = -∂x U - ∂z W,   Θ_t = -∂x(θx(y) U) - ∂z(θz(y) W),
#     U_t = -∂x p'(Θ),      W_t = -∂z p'(Θ) - g ρ',     sponge -damp (z - qtarget) on all rows,
#   slow (frozen y): momentum advection ∂x(U u) + ∂z(W u), ∂x(U w) + ∂z(W w).
# θx, θz are the upwinded face values of θ = Θ/ρ of the full operator (3rd order, upwind direction
# from U, W of y). The pressure in the fast part (`FV_FROZEN_PRESSURE`):
#   :nonlinear   p(Θ_z), the full equation of state (nothing frozen)
#   :derivative  ∇p = (dp/dΘ)(y) ∇Θ_z, dp/dΘ frozen at the faces: the discrete derivative
#                (p_r - p_l)/(Θ_r - Θ_l) of the two neighbouring cells (γ p/Θ at the mean Θ if
#                they are nearly equal), so that it reproduces the pressure difference of N.
# In both cases F(q, q) = N(q); with :derivative the fast part is linear in z. (Freezing p/Θ,
# p = (p/Θ)(y) Θ_z, was tried and removed: it puts (γ-1)/γ of the acoustic stiffness into the frozen
# argument and is stable only for c Δt/Δz ≲ 1, see docs/FINDINGS.md §11.)

"""Pressure in the fast part of the frozen FV split: `:nonlinear` (default) or `:derivative`
(dp/dΘ frozen at the faces). Read in every call."""
const FV_FROZEN_PRESSURE = Ref(:nonlinear)

_flat_only(m::FVModel) = (m.grid isa CutCellFVGrid || m.grid.terrain) &&
    error("frozen split: flat FV grids only (no terrain, no cut cells)")

"""Coefficient vector C = [θx (nx nz); θz (nx (nz+1)); P1 (nx nz); P2 (nx (nz+1))]: P1, P2 = dp/dΘ
at the x- and z-faces (:derivative, otherwise unused)."""
frozen_ncoef(m::FVModel) = (_flat_only(m); 2 * (m.grid.nx * m.grid.nz + m.grid.nx * (m.grid.nz + 1)))

@inline _θx(C, g, i, k) = C[i+(k-1)*g.nx]
@inline _θz(C, g, i, k) = C[g.nx*g.nz+i+(k-1)*g.nx]
@inline _off3(g) = g.nx * g.nz + g.nx * (g.nz + 1)
@inline _P1(C, g, i, k) = C[_off3(g)+i+(k-1)*g.nx]
@inline _P2(C, g, i, k) = C[_off3(g)+g.nx*g.nz+i+(k-1)*g.nx]

"""dp/dΘ between two cells: secant of the equation of state (exact pressure difference)."""
@inline function _dpdΘ(gas, Θl, Θr)
    d = Θr - Θl
    abs(d) > 1e-7 * abs(Θr) && return (pressure(gas, Θr) - pressure(gas, Θl)) / d
    Θm = 0.5 * (Θl + Θr)
    return gas.γ * pressure(gas, Θm) / Θm
end

"""C(y) and the slow part S(y) (momentum advection) of the flat FV operator."""
function frozen_data!(C, S, m::FVModel, q)
    _flat_only(m)
    g = m.grid
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    c = m.cache
    u, w, θ = c.u, c.w, c.θ
    mode = FV_FROZEN_PRESSURE[]
    @inbounds for k in 1:nz, i in 1:nx
        ρ = q[iρ(g, i, k)]
        θ[i, k] = q[iΘ(g, i, k)] / ρ
        u[i, k] = q[iU(g, i, k)] / (0.5 * (q[iρ(g, i - 1, k)] + ρ))
    end
    @inbounds for i in 1:nx
        w[i, 1] = 0.0
        w[i, nz+1] = 0.0
        for k in 2:nz
            w[i, k] = q[iW(g, i, k)] / (0.5 * (q[iρ(g, i, k - 1)] + q[iρ(g, i, k)]))
        end
    end
    fill!(C, 0.0)
    fill!(S, 0.0)
    o3 = _off3(g)
    @inbounds for k in 1:nz, i in 1:nx
        U = q[iU(g, i, k)]
        C[i+(k-1)*nx] = interp3(θ[ip(g, i - 2), k], θ[ip(g, i - 1), k], θ[i, k], θ[ip(g, i + 1), k], U)
        k > 1 && (C[nx*nz+i+(k-1)*nx] = face_z(θ, i, k, nz, q[iW(g, i, k)]))
        Θc = q[iΘ(g, i, k)]
        if mode === :derivative
            C[o3+i+(k-1)*nx] = _dpdΘ(m.gas, q[iΘ(g, i - 1, k)], Θc)
            k > 1 && (C[o3+nx*nz+i+(k-1)*nx] = _dpdΘ(m.gas, q[iΘ(g, i, k - 1)], Θc))
        end
        # momentum advection, as in rhs_nonlinear!
        Uc = 0.5 * (q[iU(g, i, k)] + q[iU(g, i + 1, k)])
        uc = interp3(u[ip(g, i - 1), k], u[i, k], u[ip(g, i + 1), k], u[ip(g, i + 2), k], Uc)
        S[iU(g, i + 1, k)] += Uc * uc / dx
        S[iU(g, i, k)] -= Uc * uc / dx
        if k > 1
            Wk = 0.5 * (q[iW(g, i - 1, k)] + q[iW(g, i, k)])
            uz = face_z(u, i, k, nz, Wk)
            S[iU(g, i, k)] += Wk * uz / dz
            S[iU(g, i, k - 1)] -= Wk * uz / dz
            Uk = 0.5 * (q[iU(g, i, k - 1)] + q[iU(g, i, k)])
            wx = interp3(w[ip(g, i - 2), k], w[ip(g, i - 1), k], w[i, k], w[ip(g, i + 1), k], Uk)
            S[iW(g, i, k)] += Uk * wx / dx
            S[iW(g, i - 1, k)] -= Uk * wx / dx
        end
        Wc = 0.5 * (Wface(g, q, i, k) + Wface(g, q, i, k + 1))
        if k >= 2 && k <= nz - 1
            wc = interp3(w[i, k-1], w[i, k], w[i, k+1], w[i, k+2], Wc)
        else
            wc = interp2(w[i, k], w[i, k+1])
        end
        k + 1 <= nz && (S[iW(g, i, k + 1)] += Wc * wc / dz)
        k >= 2 && (S[iW(g, i, k)] -= Wc * wc / dz)
    end
    return C, S
end

"""Parts of the fast tendency: (horizontal terms, vertical terms, rows ρ/Θ, U, W, forcing and
sponge on ρ/Θ, U, W). :hthermo/:hmomentum/:vertical are the pieces of the HEVI loop (forcing once,
in the horizontal parts, as in `hevi_loop!`)."""
@inline function _frozen_part(part::Symbol)
    part === :all && return (true, true, true, true, true, true, true, true)
    part === :thermo && return (true, true, true, false, false, true, false, false)
    part === :momentum && return (true, true, false, true, true, false, true, true)
    part === :hthermo && return (true, false, true, false, false, true, false, true)
    part === :hmomentum && return (true, false, false, true, false, false, true, false)
    part === :vertical && return (false, true, true, false, true, false, false, false)
    error("frozen_fast!: unknown part $part")
end

"""Pressure difference p'_{right/top} - p'_{left/bottom} across an x-face (dir = 1) or z-face
(dir = 2) of the fast part, for the selected pressure form (pp holds p' at the centres for
:nonlinear)."""
@inline function _dp(m, C, z, pp, mode, i, k, dir)
    g = m.grid
    if mode === :derivative
        if dir == 1
            il = ip(g, i - 1)
            return _P1(C, g, i, k) * (z[iΘ(g, i, k)] - z[iΘ(g, il, k)]) - (m.pbal[i, k] - m.pbal[il, k])
        else
            return _P2(C, g, i, k) * (z[iΘ(g, i, k)] - z[iΘ(g, i, k - 1)]) - (m.pbal[i, k] - m.pbal[i, k-1])
        end
    end
    return dir == 1 ? pp[i, k] - pp[ip(g, i - 1), k] : pp[i, k] - pp[i, k-1]
end

"""Fast part F_f(C, z) + F (and the sponge) on the rows of part (see `_frozen_part`)."""
function frozen_fast!(out, m::FVModel, C, z, part::Symbol, F)
    g = m.grid
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    grav = m.gas.g
    hor, ver, rT, rU, rW, fT, fU, fW = _frozen_part(part)
    mode = FV_FROZEN_PRESSURE[]
    pp = m.cache.pp
    fill!(out, 0.0)
    if (rU || rW) && mode !== :derivative
        @inbounds for k in 1:nz, i in 1:nx
            Θ = z[iΘ(g, i, k)]
            pp[i, k] = pressure(m.gas, Θ) - m.pbal[i, k]
        end
    end
    @inbounds for k in 1:nz, i in 1:nx
        if rT
            Ul, Ur = z[iU(g, i, k)], z[iU(g, i + 1, k)]
            Wb, Wt = Wface(g, z, i, k), Wface(g, z, i, k + 1)
            fρ = 0.0; fΘ = 0.0
            if hor
                fρ -= (Ur - Ul) / dx
                fΘ -= (_θx(C, g, ip(g, i + 1), k) * Ur - _θx(C, g, i, k) * Ul) / dx
            end
            if ver
                fρ -= (Wt - Wb) / dz
                fΘ -= (_θz(C, g, i, k + 1) * Wt - _θz(C, g, i, k) * Wb) / dz
            end
            out[iρ(g, i, k)] = fρ
            out[iΘ(g, i, k)] = fΘ
        end
        rU && hor && (out[iU(g, i, k)] = -_dp(m, C, z, pp, mode, i, k, 1) / dx)
        if rW && ver && k > 1
            ρw = 0.5 * (z[iρ(g, i, k - 1)] - m.ρbal[i, k-1] + z[iρ(g, i, k)] - m.ρbal[i, k])
            out[iW(g, i, k)] = -_dp(m, C, z, pp, mode, i, k, 2) / dz - grav * ρw
        end
    end
    @inbounds for k in 1:nz, i in 1:nx
        if fT
            for j in (iρ(g, i, k), iΘ(g, i, k))
                out[j] += F[j] - m.damp[j] * (z[j] - m.qtarget[j])
            end
        end
        if fU
            j = iU(g, i, k)
            out[j] += F[j] - m.damp[j] * (z[j] - m.qtarget[j])
        end
        if fW && k > 1
            j = iW(g, i, k)
            out[j] += F[j] - m.damp[j] * (z[j] - m.qtarget[j])
        end
    end
    return out
end

# ------------------------------------------------------------------ vertically implicit step

"""Column solver of the frozen HEVI loop for (I - c J_v) Y = R, J_v the vertical part with the
pressure difference α_k Θ_k - β_k Θ_{k-1} at the z-face k (WRF elimination of ρ and Θ):
    W_{k-1} [-r β_k θ_{k-1} + c² g/(2Δz)] + W_k [1 + r θ_k (α_k + β_k)]
    + W_{k+1} [-r α_k θ_{k+1} - c² g/(2Δz)] = R^W_k - (c/Δz)(α_k R^Θ_k - β_k R^Θ_{k-1}) - (c g/2)(R^ρ_{k-1} + R^ρ_k),
r = (c/Δz)². :nonlinear α_k = a_k, β_k = a_{k-1} with a = γ p/Θ at the start of the stage
(approximate Jacobian); :derivative α_k = β_k = (dp/dΘ)_k (exact: the vertical part is then linear
and the step is Crank–Nicolson)."""
mutable struct FrozenColumn
    α::Matrix{Float64}
    β::Matrix{Float64}
    θz::Matrix{Float64}
    factors::Dict{Float64,NTuple{3,Matrix{Float64}}}
    W::Vector{Float64}
end
FrozenColumn(g) = FrozenColumn(zeros(g.nx, g.nz + 1), zeros(g.nx, g.nz + 1), zeros(g.nx, g.nz + 1),
                               Dict{Float64,NTuple{3,Matrix{Float64}}}(), zeros(g.nz + 1))

function _frozen_column(m::FVModel)
    s = m.cache.frozen[]
    s isa FrozenColumn || (s = FrozenColumn(m.grid); m.cache.frozen[] = s)
    return s::FrozenColumn
end

function frozen_stage_begin!(m::FVModel, C, z)
    s = _frozen_column(m)
    g = m.grid
    mode = FV_FROZEN_PRESSURE[]
    @inbounds for k in 1:g.nz+1, i in 1:g.nx
        s.θz[i, k] = _θz(C, g, i, k)
    end
    @inbounds for k in 2:g.nz, i in 1:g.nx
        if mode === :derivative
            s.α[i, k] = s.β[i, k] = _P2(C, g, i, k)
        else
            a(kk) = (Θ = z[iΘ(g, i, kk)]; m.gas.γ * pressure(m.gas, Θ) / Θ)
            s.α[i, k] = a(k); s.β[i, k] = a(k - 1)
        end
    end
    empty!(s.factors)
    return s
end

function _frozen_factor(s::FrozenColumn, m::FVModel, c)
    get!(s.factors, Float64(c)) do
        g = m.grid
        nz, dz = g.nz, g.dz
        r = (c / dz)^2
        gg = c^2 * m.gas.g / (2dz)
        sub = zeros(nz + 1, g.nx); cp = zeros(nz + 1, g.nx); iden = zeros(nz + 1, g.nx)
        @inbounds for i in 1:g.nx, k in 2:nz
            a = k > 2 ? -r * s.β[i, k] * s.θz[i, k-1] + gg : 0.0
            d = 1 + r * s.θz[i, k] * (s.α[i, k] + s.β[i, k])
            b = k < nz ? -r * s.α[i, k] * s.θz[i, k+1] - gg : 0.0
            den = k == 2 ? d : d - a * cp[k-1, i]
            sub[k, i] = a; iden[k, i] = 1 / den; cp[k, i] = b / den
        end
        (sub, cp, iden)
    end
end

"""Y = (I - c J_v)⁻¹ R for the frozen column (U rows: identity)."""
function _frozen_column_solve!(Y, s::FrozenColumn, m::FVModel, c, R)
    g = m.grid
    nz, dz, grav = g.nz, g.dz, m.gas.g
    sub, cp, iden = _frozen_factor(s, m, c)
    W = s.W
    @inbounds for i in 1:g.nx
        for k in 2:nz
            rhs = R[iW(g, i, k)] - (c / dz) * (s.α[i, k] * R[iΘ(g, i, k)] - s.β[i, k] * R[iΘ(g, i, k - 1)]) -
                  (c * grav / 2) * (R[iρ(g, i, k - 1)] + R[iρ(g, i, k)])
            W[k] = k == 2 ? rhs * iden[k, i] : (rhs - sub[k, i] * W[k-1]) * iden[k, i]
        end
        for k in nz-1:-1:2
            W[k] -= cp[k, i] * W[k+1]
        end
        W[1] = 0.0; W[nz+1] = 0.0
        for k in 1:nz
            θb = k > 1 ? s.θz[i, k] : 0.0
            θt = k < nz ? s.θz[i, k+1] : 0.0
            Y[iρ(g, i, k)] = R[iρ(g, i, k)] - (c / dz) * (W[k+1] - W[k])
            Y[iΘ(g, i, k)] = R[iΘ(g, i, k)] - (c / dz) * (θt * W[k+1] - θb * W[k])
            Y[iU(g, i, k)] = R[iU(g, i, k)]
        end
        for k in 2:nz
            Y[iW(g, i, k)] = W[k]
        end
    end
    return Y
end

"""Linearly implicit Crank–Nicolson step of the vertical part G_v:
z ← z + (I - dt/2 J_v)⁻¹ dt G_v(z) (= Crank–Nicolson for :derivative)."""
function frozen_vertical_step!(z, m::FVModel, C, dt, k, y)
    s = _frozen_column(m)
    frozen_fast!(k, m, C, z, :vertical, k)        # F unused for :vertical
    @. k *= dt
    _frozen_column_solve!(y, s, m, dt / 2, k)
    @. z += y
    return z
end
