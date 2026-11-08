# Frozen-part split of the flat FV right-hand side for `TimeStepping.run_frozen`
#
# N(q) = F(q, q) with
#   fast (evolving z, coefficients from y):
#     ρ_t = -∂x U - ∂z W,   Θ_t = -∂x(θx(y) U) - ∂z(θz(y) W),
#     U_t = -∂x p'(Θ),      W_t = -∂z p'(Θ) - g ρ',      p(Θ) the full nonlinear equation of state,
#     sponge -damp (z - qtarget) on all rows,
#   slow (frozen y): momentum advection ∂x(U u) + ∂z(W u), ∂x(U w) + ∂z(W w).
# θx, θz are the upwinded face values of θ = Θ/ρ of the full operator (3rd order, upwind direction
# from U, W of y), so F(q, q) = N(q) exactly; nothing is linearized.

"""Coefficient vector C = [θ at the x-faces (nx nz); θ at the z-faces (nx (nz+1))]."""
frozen_ncoef(m::FVModel) = (_flat_only(m); m.grid.nx * m.grid.nz + m.grid.nx * (m.grid.nz + 1))

_flat_only(m::FVModel) = (m.grid isa CutCellFVGrid || m.grid.terrain) &&
    error("frozen split: flat FV grids only (no terrain, no cut cells)")

@inline _θx(C, g, i, k) = C[i+(k-1)*g.nx]
@inline _θz(C, g, i, k) = C[g.nx*g.nz+i+(k-1)*g.nx]

"""C(y) (upwinded face θ) and the slow part S(y) (momentum advection) of the flat FV operator."""
function frozen_data!(C, S, m::FVModel, q)
    _flat_only(m)
    g = m.grid
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    c = m.cache
    u, w, θ = c.u, c.w, c.θ
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
    fill!(S, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        U = q[iU(g, i, k)]
        C[i+(k-1)*nx] = interp3(θ[ip(g, i - 2), k], θ[ip(g, i - 1), k], θ[i, k], θ[ip(g, i + 1), k], U)
        C[nx*nz+i+(k-1)*nx] = k > 1 ? face_z(θ, i, k, nz, q[iW(g, i, k)]) : 0.0
        k == nz && (C[nx*nz+i+nz*nx] = 0.0)
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

"""Fast part F_f(C, z) + F (and the sponge) on the rows of part (see `_frozen_part`)."""
function frozen_fast!(out, m::FVModel, C, z, part::Symbol, F)
    g = m.grid
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    grav = m.gas.g
    hor, ver, rT, rU, rW, fT, fU, fW = _frozen_part(part)
    pp = m.cache.pp
    fill!(out, 0.0)
    if rU || rW
        @inbounds for k in 1:nz, i in 1:nx
            pp[i, k] = pressure(m.gas, z[iΘ(g, i, k)]) - m.pbal[i, k]
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
        rU && hor && (out[iU(g, i, k)] = -(pp[i, k] - pp[ip(g, i - 1), k]) / dx)
        if rW && ver && k > 1
            ρw = 0.5 * (z[iρ(g, i, k - 1)] - m.ρbal[i, k-1] + z[iρ(g, i, k)] - m.ρbal[i, k])
            out[iW(g, i, k)] = -(pp[i, k] - pp[i, k-1]) / dz - grav * ρw
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

"""Column solver of the frozen HEVI loop: an `FVFast` whose θ̄_z is the frozen face θ of the
stage and whose a = γ p/Θ is taken at the start of the stage (approximate Jacobian of the
vertical part; the explicit vertical tendency keeps the nonlinear p(Θ))."""
function _frozen_column(m::FVModel)
    s = m.fastref[]
    s isa FVFast || (s = FVFast(m); m.fastref[] = s)
    return s::FVFast
end

function frozen_stage_begin!(m::FVModel, C, z)
    s = _frozen_column(m)
    g = s.grid
    s.vadv = false
    @inbounds for k in 1:g.nz, i in 1:g.nx
        Θ = z[iΘ(g, i, k)]
        s.a[i, k] = m.gas.γ * pressure(m.gas, Θ) / Θ
    end
    @inbounds for k in 1:g.nz+1, i in 1:g.nx
        s.θz[i, k] = _θz(C, g, i, k)
    end
    empty!(s.lucache)
    return s
end

"""Linearly implicit Crank–Nicolson step of the vertical part G_v:
z ← z + (I - dt/2 J_v)⁻¹ dt G_v(z) (= CN for linear G_v)."""
function frozen_vertical_step!(z, m::FVModel, C, dt, k, y)
    s = _frozen_column(m)
    frozen_fast!(k, m, C, z, :vertical, k)        # F unused for :vertical
    @. k *= dt
    vertical_solve!(y, s, vertical_factor(s, dt / 2), k)
    @. z += y
    return z
end
