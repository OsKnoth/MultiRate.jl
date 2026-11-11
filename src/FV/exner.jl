# Exner pressure as an additional prognostic variable (flat FV grid, frozen split)
#
# π = (p/p00)^(R/cp) = (R Θ/p00)^(γ-1), Θ = ρθ. The pressure gradient is the product
#     ∇p = cp Θ ∇π                                   (exact identity, cp = γ R/(γ-1)),
# and π has its own evolution equation next to the conservative Θ equation,
#     π_t = (dπ/dΘ) Θ_t = -c ∇·(θ M),   c = (γ-1) π/Θ,   M = ρ v.
# Extended state q̃ = [q; π] (q = [ρ; U; W; Θ] of the FV model, π at the cell centres). The
# extended system keeps π = π(Θ) exactly in continuous time; the time discretization does not, so π
# is recomputed from Θ after every MIS step (`frozen_step_end!`), not at the internal stages.
#
# Frozen form N(q̃) = F(q̃, q̃), fast part linear in the evolving z, coefficients from y:
#     ρ_t = -∂x U - ∂z W,
#     Θ_t = -∂x(θx(y) U) - ∂z(θz(y) W),
#     π_t = -c(y) [∂x(θx(y) U) + ∂z(θz(y) W)],
#     U_t = -cp Θx(y) ∂x π,     W_t = -cp Θz(y) ∂z π - g ρ̄,     sponge on W,
#   slow (y): momentum advection (shift technique, shift_advection.jl).
# θx, θz: upwind face values of θ; Θx, Θz: face means of Θ; the products c θ of a cell with its
# faces are coefficients of their own (the MIS stage combination is linear in C). No background
# state: for an isentropic atmosphere π is linear in z and the analytic state is exactly balanced,
# cp Θz δz π + g ρ̄ = 0.
# The base model's N has the same pressure gradient with `FV_PRESSURE_FORM[] = :exner` (π(Θ)).

"""cp of the gas."""
cp_gas(gas::Gas) = gas.R * gas.γ / (gas.γ - 1)
"""Exner pressure π(Θ) = (R Θ/p00)^(γ-1)."""
exner(gas::Gas, Θ) = (gas.R * Θ / gas.p00)^(gas.γ - 1)

mutable struct ExnerColumn
    P::Matrix{Float64}        # cp Θz at the z-faces (nx, nz+1)
    cB::Matrix{Float64}       # c θz of the cell below the face
    cT::Matrix{Float64}       # c θz of the cell above the face
    θz::Matrix{Float64}
    factors::Dict{Float64,NTuple{3,Matrix{Float64}}}
    W::Vector{Float64}
end

"""
    ExnerModel(base::FVModel)

Flat FV model with the Exner pressure as an additional prognostic variable: state
`[q; π]` (`exner_state`, `base_state`), frozen split for `TimeStepping.run_frozen` (fast loops
:verlet, :hevi, :rk4), π recomputed from Θ after every MIS step.
"""
struct ExnerModel{M<:FVModel}
    base::M
    nq::Int
    C::Vector{Float64}
    S::Vector{Float64}
    column::ExnerColumn
end
function ExnerModel(base::FVModel)
    _flat_only(base); _no_background(base)
    g = base.grid
    nq = base.nq + g.nx * g.nz
    z1() = zeros(g.nx, g.nz + 1)
    m = ExnerModel(base, nq, zeros(4 * (g.nx * g.nz + g.nx * (g.nz + 1))), zeros(nq),
                   ExnerColumn(z1(), z1(), z1(), z1(), Dict{Float64,NTuple{3,Matrix{Float64}}}(), zeros(g.nz + 1)))
    return m
end

@inline iπ(m::ExnerModel, i, k) = m.base.nq + ip(m.base.grid, i) + (k - 1) * m.base.grid.nx
"""Extended state [q; π(Θ)] of a base state q."""
exner_state(m::ExnerModel, q) = (g = m.base.grid;
    vcat(q[1:m.base.nq], [exner(m.base.gas, q[iΘ(g, i, k)]) for k in 1:g.nz for i in 1:g.nx]))
"""Base state [ρ; U; W; Θ] of an extended state."""
base_state(m::ExnerModel, q) = q[1:m.base.nq]

sound_speed_max(m::ExnerModel, q) = sound_speed_max(m.base, q)

# ------------------------------------------------------------------ coefficient vector
# C = [θx (N); θz (Nz); Θx (N); Θz (Nz); cxL (N); cxR (N); czB (Nz); czT (Nz)], N = nx nz, Nz = nx (nz+1);
# face (i, k): left x-face of cell i, lower z-face of cell k. cxL/cxR: c θx of the cell left/right
# of the x-face, czB/czT: c θz of the cell below/above the z-face.
frozen_ncoef(m::ExnerModel) = length(m.C)
@inline _e(g, i, k) = ip(g, i) + (k - 1) * g.nx
@inline _eN(g) = g.nx * g.nz
@inline _eNz(g) = g.nx * (g.nz + 1)
@inline eθx(C, g, i, k) = C[_e(g, i, k)]
@inline eθz(C, g, i, k) = C[_eN(g)+_e(g, i, k)]
@inline eΘx(C, g, i, k) = C[_eN(g)+_eNz(g)+_e(g, i, k)]
@inline eΘz(C, g, i, k) = C[2*_eN(g)+_eNz(g)+_e(g, i, k)]
@inline ecxL(C, g, i, k) = C[2*_eN(g)+2*_eNz(g)+_e(g, i, k)]
@inline ecxR(C, g, i, k) = C[3*_eN(g)+2*_eNz(g)+_e(g, i, k)]
@inline eczB(C, g, i, k) = C[4*_eN(g)+2*_eNz(g)+_e(g, i, k)]
@inline eczT(C, g, i, k) = C[4*_eN(g)+3*_eNz(g)+_e(g, i, k)]

"""C(y) and the slow part S(y) (momentum advection)."""
function frozen_data!(C, S, m::ExnerModel, q)
    b = m.base; g = b.grid
    nx, nz = g.nx, g.nz
    N, Nz = _eN(g), _eNz(g)
    a = shift_advection!(b.cache.adv, g, q)
    κ = b.gas.γ - 1
    c(i, k) = κ * q[iπ(m, i, k)] / q[iΘ(g, i, k)]
    fill!(C, 0.0); fill!(S, 0.0)
    @inbounds for k in 1:nz, i in 1:nx
        e = _e(g, i, k)
        θx = a.θfx[i, k]
        C[e] = θx
        C[N+Nz+e] = 0.5 * (q[iΘ(g, i - 1, k)] + q[iΘ(g, i, k)])
        C[2N+2Nz+e] = c(i - 1, k) * θx
        C[3N+2Nz+e] = c(i, k) * θx
        S[iU(g, i, k)] = a.SU[i, k]
        if k > 1
            θz = a.θfz[i, k]
            C[N+e] = θz
            C[2N+Nz+e] = 0.5 * (q[iΘ(g, i, k - 1)] + q[iΘ(g, i, k)])
            C[4N+2Nz+e] = c(i, k - 1) * θz
            C[4N+3Nz+e] = c(i, k) * θz
            S[iW(g, i, k)] = a.SW[i, k]
        end
    end
    return C, S
end

# ------------------------------------------------------------------ building blocks of the fast part

"""Thermodynamic part, horizontal (ρ, Θ, π rows)."""
function ethermo_h!(out, m::ExnerModel, C, z)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        Ul, Ur = z[iU(g, i, k)], z[iU(g, i + 1, k)]
        out[iρ(g, i, k)] -= (Ur - Ul) / g.dx
        out[iΘ(g, i, k)] -= (eθx(C, g, i + 1, k) * Ur - eθx(C, g, i, k) * Ul) / g.dx
        out[iπ(m, i, k)] -= (ecxL(C, g, i + 1, k) * Ur - ecxR(C, g, i, k) * Ul) / g.dx
    end
    return out
end

"""Thermodynamic part, vertical (ρ, Θ, π rows)."""
function ethermo_v!(out, m::ExnerModel, C, z)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        Wb, Wt = Wface(g, z, i, k), Wface(g, z, i, k + 1)
        out[iρ(g, i, k)] -= (Wt - Wb) / g.dz
        out[iΘ(g, i, k)] -= (eθz(C, g, i, k + 1) * Wt - eθz(C, g, i, k) * Wb) / g.dz
        out[iπ(m, i, k)] -= (eczB(C, g, i, k + 1) * Wt - eczT(C, g, i, k) * Wb) / g.dz
    end
    return out
end

"""Momentum part, horizontal (U rows): -cp Θx ∂x π."""
function emomentum_h!(out, m::ExnerModel, C, z)
    g = m.base.grid; cp = cp_gas(m.base.gas)
    @inbounds for k in 1:g.nz, i in 1:g.nx
        out[iU(g, i, k)] -= cp * eΘx(C, g, i, k) * (z[iπ(m, i, k)] - z[iπ(m, i - 1, k)]) / g.dx
    end
    return out
end

"""Momentum part, vertical (W rows): -cp Θz ∂z π - g ρ̄."""
function emomentum_v!(out, m::ExnerModel, C, z)
    g = m.base.grid; cp = cp_gas(m.base.gas); grav = m.base.gas.g
    @inbounds for k in 2:g.nz, i in 1:g.nx
        out[iW(g, i, k)] -= cp * eΘz(C, g, i, k) * (z[iπ(m, i, k)] - z[iπ(m, i, k - 1)]) / g.dz +
                            grav * 0.5 * (z[iρ(g, i, k - 1)] + z[iρ(g, i, k)])
    end
    return out
end

function eforcing_thermo!(out, m::ExnerModel, F)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        out[iρ(g, i, k)] += F[iρ(g, i, k)]
        out[iΘ(g, i, k)] += F[iΘ(g, i, k)]
        out[iπ(m, i, k)] += F[iπ(m, i, k)]
    end
    return out
end

# parts as in frozen.jl (table there); the thermodynamic rows include π
_efast!(::Val{:all}, out, m, C, z, F) = (ethermo_h!(out, m, C, z); ethermo_v!(out, m, C, z);
    emomentum_h!(out, m, C, z); emomentum_v!(out, m, C, z);
    eforcing_thermo!(out, m, F); forcing_U!(out, m.base, F); forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_efast!(::Val{:thermo}, out, m, C, z, F) = (ethermo_h!(out, m, C, z); ethermo_v!(out, m, C, z); eforcing_thermo!(out, m, F))
_efast!(::Val{:momentum}, out, m, C, z, F) = (emomentum_h!(out, m, C, z); emomentum_v!(out, m, C, z);
    forcing_U!(out, m.base, F); forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_efast!(::Val{:hthermo}, out, m, C, z, F) = (ethermo_h!(out, m, C, z); eforcing_thermo!(out, m, F);
    forcing_W!(out, m.base, F); sponge!(out, m.base, z))
_efast!(::Val{:hmomentum}, out, m, C, z, F) = (emomentum_h!(out, m, C, z); forcing_U!(out, m.base, F))
_efast!(::Val{:vertical}, out, m, C, z, F) = (ethermo_v!(out, m, C, z); emomentum_v!(out, m, C, z))
_efast!(::Val{P}, out, m, C, z, F) where {P} = error("frozen_fast!: unknown part $P")

"""Fast part F_f(C, z) + F (and the sponge) on the rows of `part`."""
function frozen_fast!(out, m::ExnerModel, C, z, part::Symbol, F)
    fill!(out, 0.0)
    _efast!(Val(part), out, m, C, z, F)
    return out
end

"""N(q̃) = F(q̃, q̃) of the extended system."""
function rhs_nonlinear!(out, m::ExnerModel, q::AbstractVector)
    frozen_data!(m.C, m.S, m, q)
    return frozen_fast!(out, m, m.C, q, :all, m.S)
end

"""π ← π(Θ) after a MIS step."""
function frozen_step_end!(m::ExnerModel, q)
    g = m.base.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        q[iπ(m, i, k)] = exner(m.base.gas, q[iΘ(g, i, k)])
    end
    return q
end

# ------------------------------------------------------------------ vertically implicit step (HEVI)
# (I - c J_v) Y = R per column, eliminating ρ, Θ, π (r = (c/Δz)², gg = c² g/(2Δz), P_k = cp Θz_k):
#   W_{k-1} [-r P_k cT_{k-1} + gg] + W_k [1 + r P_k (cT_k + cB_k)] + W_{k+1} [-r P_k cB_{k+1} - gg]
#     = R^W_k - (c/Δz) P_k (R^π_k - R^π_{k-1}) - (c g/2)(R^ρ_{k-1} + R^ρ_k),
# then ρ, Θ, π from the fluxes. The vertical part is linear in z: the step is Crank–Nicolson.

function frozen_stage_begin!(m::ExnerModel, C, z)
    s = m.column; g = m.base.grid; cp = cp_gas(m.base.gas)
    @inbounds for k in 1:g.nz+1, i in 1:g.nx
        s.P[i, k] = cp * eΘz(C, g, i, k); s.cB[i, k] = eczB(C, g, i, k); s.cT[i, k] = eczT(C, g, i, k)
        s.θz[i, k] = eθz(C, g, i, k)
    end
    empty!(s.factors)
    return s
end

function _exner_factor(s::ExnerColumn, m::ExnerModel, c)
    get!(s.factors, Float64(c)) do
        g = m.base.grid
        nz, dz = g.nz, g.dz
        r = (c / dz)^2
        gg = c^2 * m.base.gas.g / (2dz)
        sub = zeros(nz + 1, g.nx); cpr = zeros(nz + 1, g.nx); iden = zeros(nz + 1, g.nx)
        @inbounds for i in 1:g.nx, k in 2:nz
            a = k > 2 ? -r * s.P[i, k] * s.cT[i, k-1] + gg : 0.0
            d = 1 + r * s.P[i, k] * (s.cT[i, k] + s.cB[i, k])
            b = k < nz ? -r * s.P[i, k] * s.cB[i, k+1] - gg : 0.0
            den = k == 2 ? d : d - a * cpr[k-1, i]
            sub[k, i] = a; iden[k, i] = 1 / den; cpr[k, i] = b / den
        end
        (sub, cpr, iden)
    end
end

"""Y = (I - c J_v)⁻¹ R (U rows: identity)."""
function exner_column_solve!(Y, m::ExnerModel, c, R)
    s = m.column; g = m.base.grid
    nz, dz, grav = g.nz, g.dz, m.base.gas.g
    sub, cpr, iden = _exner_factor(s, m, c)
    W = s.W
    @inbounds for i in 1:g.nx
        for k in 2:nz
            rhs = R[iW(g, i, k)] - (c / dz) * s.P[i, k] * (R[iπ(m, i, k)] - R[iπ(m, i, k - 1)]) -
                  (c * grav / 2) * (R[iρ(g, i, k - 1)] + R[iρ(g, i, k)])
            W[k] = k == 2 ? rhs * iden[k, i] : (rhs - sub[k, i] * W[k-1]) * iden[k, i]
        end
        for k in nz-1:-1:2
            W[k] -= cpr[k, i] * W[k+1]
        end
        W[1] = 0.0; W[nz+1] = 0.0
        for k in 1:nz
            Y[iρ(g, i, k)] = R[iρ(g, i, k)] - (c / dz) * (W[k+1] - W[k])
            Y[iΘ(g, i, k)] = R[iΘ(g, i, k)] - (c / dz) * (s.θz[i, k+1] * W[k+1] - s.θz[i, k] * W[k])
            Y[iπ(m, i, k)] = R[iπ(m, i, k)] - (c / dz) * (s.cB[i, k+1] * W[k+1] - s.cT[i, k] * W[k])
            Y[iU(g, i, k)] = R[iU(g, i, k)]
        end
        for k in 2:nz
            Y[iW(g, i, k)] = W[k]
        end
    end
    return Y
end

"""Crank–Nicolson step of the vertical part: z ← z + (I - dt/2 J_v)⁻¹ dt G_v(z)."""
function frozen_vertical_step!(z, m::ExnerModel, C, dt, k, y)
    frozen_fast!(k, m, C, z, :vertical, k)
    @. k *= dt
    exner_column_solve!(y, m, dt / 2, k)
    @. z += y
    return z
end
