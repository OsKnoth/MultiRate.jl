# Frozen-part split of the flat FV right-hand side for `TimeStepping.run_frozen`
#
# N(q) = F(q, q) with
#   fast (evolving z, coefficients from y):
#     ρ_t = -∂x U - ∂z W,   Θ_t = -∂x(θx(y) U) - ∂z(θz(y) W),
#     U_t = -∂x p(Θ),       W_t = -∂z p(Θ) - g ρ,       sponge on W (`sponge!`, sponge.jl),
#   slow (frozen y): momentum advection ∂x(U u) + ∂z(W u), ∂x(U w) + ∂z(W w) (shift technique).
# θx, θz are the upwinded face values of θ = Θ/ρ of the full operator (pseudo third order,
# shift_advection.jl; upwind direction from U, W of y). The pressure follows the form of N (`FV_PRESSURE_FORM`, rhs.jl):
#   :gradp    p(Θ_z), the full equation of state (nothing frozen)
#   :product  ∇p = (dp/dΘ)_f(y) ∇Θ_z, the face derivative (mean of the cell values γ p/Θ) frozen
# No background state is subtracted (full p and ρ; models built with FV_HYDRO_REFERENCE are
# rejected). In both cases F(q, q) = N(q); with :product the fast part is linear in z. (Freezing p/Θ,
# p = (p/Θ)(y) Θ_z, was tried and removed: it puts (γ-1)/γ of the acoustic stiffness into the frozen
# argument and is stable only for c Δt/Δz ≲ 1, see docs/FINDINGS.md §11. The earlier secant
# (p_r - p_l)/(Θ_r - Θ_l) of the :gradp scheme is replaced by the :product scheme, where the
# derivative is defined also for equal neighbours.)

_flat_only(m::FVModel) = (m.grid isa CutCellFVGrid || m.grid.terrain) &&
    error("frozen split: flat FV grids only (no terrain, no cut cells)")

"""Coefficient vector C = [θx (nx nz); θz (nx (nz+1)); P1 (nx nz); P2 (nx (nz+1))]: P1, P2 = dp/dΘ
at the x- and z-faces (:product, otherwise unused)."""
frozen_ncoef(m::FVModel) = (_flat_only(m); 2 * (m.grid.nx * m.grid.nz + m.grid.nx * (m.grid.nz + 1)))

@inline _θx(C, g, i, k) = C[i+(k-1)*g.nx]
@inline _θz(C, g, i, k) = C[g.nx*g.nz+i+(k-1)*g.nx]
@inline _off3(g) = g.nx * g.nz + g.nx * (g.nz + 1)
@inline _P1(C, g, i, k) = C[_off3(g)+i+(k-1)*g.nx]
@inline _P2(C, g, i, k) = C[_off3(g)+g.nx*g.nz+i+(k-1)*g.nx]

"""Pressure form of the fast part, chosen once per call (`FV_PRESSURE_FORM`)."""
struct GradP end      # p(Θ_z) at the centres, differences across the faces
struct Product end    # (dp/dΘ)_f(y) (Θ_r - Θ_l), the face derivative frozen in C
pressure_form() = FV_PRESSURE_FORM[] === :product ? Product() :
                  FV_PRESSURE_FORM[] === :exner ? error("frozen split with :exner: use ExnerModel (exner.jl)") : GradP()


"""C(y) and the slow part S(y) (momentum advection) of the flat FV operator."""
function frozen_data!(C, S, m::FVModel, q)
    _flat_only(m)
    g = m.grid
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    c = m.cache
    u, w, θ = c.u, c.w, c.θ
    _no_background(m)
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
    a = shift_advection!(c.adv, g, q)          # as in rhs_nonlinear! (shift_advection.jl)
    @inbounds for k in 1:nz, i in 1:nx
        C[i+(k-1)*nx] = a.θfx[i, k]
        k > 1 && (C[nx*nz+i+(k-1)*nx] = a.θfz[i, k])
        S[iU(g, i, k)] = a.SU[i, k]
        k > 1 && (S[iW(g, i, k)] = a.SW[i, k])
    end
    pressure_coefficients!(pressure_form(), C, m, q)
    return C, S
end

"""Frozen pressure coefficients: Product the face derivatives (dp/dΘ)_f(y) (P1, P2 of C); GradP none."""
function pressure_coefficients!(::Product, C, m::FVModel, q)
    g = m.grid
    nx, nz = g.nx, g.nz
    o3 = _off3(g)
    @inbounds for k in 1:nz, i in 1:nx
        Θc = q[iΘ(g, i, k)]
        C[o3+i+(k-1)*nx] = dpdΘ_face(m.gas, q[iΘ(g, i - 1, k)], Θc)
        k > 1 && (C[o3+nx*nz+i+(k-1)*nx] = dpdΘ_face(m.gas, q[iΘ(g, i, k - 1)], Θc))
    end
    return C
end
pressure_coefficients!(::GradP, C, m::FVModel, q) = C

# ------------------------------------------------------------------ pressure form

"""Prepare the pressure of the fast state z (GradP: p(Θ_z) at the centres in m.cache.pp)."""
function prepare_pressure!(::GradP, m::FVModel, z)
    g = m.grid
    pp = m.cache.pp
    @inbounds for k in 1:g.nz, i in 1:g.nx
        pp[i, k] = pressure(m.gas, z[iΘ(g, i, k)])
    end
    return nothing
end
prepare_pressure!(::Product, m::FVModel, z) = nothing

"""Pressure difference right - left across the x-face (i - 1/2, k) and top - bottom across the
z-face (i, k - 1/2)."""
@inline dp_x(::GradP, m, C, z, i, k) = m.cache.pp[i, k] - m.cache.pp[ip(m.grid, i - 1), k]
@inline dp_z(::GradP, m, C, z, i, k) = m.cache.pp[i, k] - m.cache.pp[i, k-1]
@inline dp_x(::Product, m, C, z, i, k) = (g = m.grid; _P1(C, g, i, k) * (z[iΘ(g, i, k)] - z[iΘ(g, ip(g, i - 1), k)]))
@inline dp_z(::Product, m, C, z, i, k) = (g = m.grid; _P2(C, g, i, k) * (z[iΘ(g, i, k)] - z[iΘ(g, i, k - 1)]))

# ------------------------------------------------------------------ building blocks of the fast part
# Each block adds its terms to `out` (rows named in the comment) and nothing else. The momentum
# blocks need `prepare_pressure!` (once per call) before them.

"""Thermodynamic part, horizontal (ρ, Θ rows): -∂x U, -∂x(θx(y) U)."""
function thermo_h!(out, m::FVModel, C, z)
    g = m.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        Ul, Ur = z[iU(g, i, k)], z[iU(g, i + 1, k)]
        out[iρ(g, i, k)] -= (Ur - Ul) / g.dx
        out[iΘ(g, i, k)] -= (_θx(C, g, ip(g, i + 1), k) * Ur - _θx(C, g, i, k) * Ul) / g.dx
    end
    return out
end

"""Thermodynamic part, vertical (ρ, Θ rows): -∂z W, -∂z(θz(y) W)."""
function thermo_v!(out, m::FVModel, C, z)
    g = m.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        Wb, Wt = Wface(g, z, i, k), Wface(g, z, i, k + 1)
        out[iρ(g, i, k)] -= (Wt - Wb) / g.dz
        out[iΘ(g, i, k)] -= (_θz(C, g, i, k + 1) * Wt - _θz(C, g, i, k) * Wb) / g.dz
    end
    return out
end

"""Momentum part, horizontal (U rows): -∂x p."""
function momentum_h!(out, m::FVModel, C, z, pf)
    g = m.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        out[iU(g, i, k)] -= dp_x(pf, m, C, z, i, k) / g.dx
    end
    return out
end

"""Momentum part, vertical (W rows): -∂z p - g ρ (ρ the mean of the two cells)."""
function momentum_v!(out, m::FVModel, C, z, pf)
    g = m.grid
    grav = m.gas.g
    @inbounds for k in 2:g.nz, i in 1:g.nx
        out[iW(g, i, k)] -= dp_z(pf, m, C, z, i, k) / g.dz + grav * 0.5 * (z[iρ(g, i, k - 1)] + z[iρ(g, i, k)])
    end
    return out
end

"""Constant forcing F of the stage (slow part and MIS coupling terms) on the ρ, Θ / U / W rows."""
function forcing_thermo!(out, m::FVModel, F)
    g = m.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        out[iρ(g, i, k)] += F[iρ(g, i, k)]
        out[iΘ(g, i, k)] += F[iΘ(g, i, k)]
    end
    return out
end
function forcing_U!(out, m::FVModel, F)
    g = m.grid
    @inbounds for k in 1:g.nz, i in 1:g.nx
        out[iU(g, i, k)] += F[iU(g, i, k)]
    end
    return out
end
function forcing_W!(out, m::FVModel, F)
    g = m.grid
    @inbounds for k in 2:g.nz, i in 1:g.nx
        out[iW(g, i, k)] += F[iW(g, i, k)]
    end
    return out
end

# ------------------------------------------------------------------ the fast part as compositions
# part        blocks
# :all        thermo (h + v), momentum (h + v), forcing on all rows, sponge
# :thermo     thermo (h + v), forcing ρ, Θ                         (Verlet: thermodynamic step)
# :momentum   momentum (h + v), forcing U, W, sponge               (Verlet: momentum step)
# :hthermo    thermo h, forcing ρ, Θ and W, sponge                 (HEVI: horizontal, with all
# :hmomentum  momentum h, forcing U                                 forcing; the vertical part
# :vertical   thermo v, momentum v                                  has none)

_fast!(::Val{:all}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z); thermo_v!(out, m, C, z);
    prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf); momentum_v!(out, m, C, z, pf);
    forcing_thermo!(out, m, F); forcing_U!(out, m, F); forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:thermo}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z); thermo_v!(out, m, C, z); forcing_thermo!(out, m, F))
_fast!(::Val{:momentum}, out, m, C, z, F, pf) = (prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf); momentum_v!(out, m, C, z, pf);
    forcing_U!(out, m, F); forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:hthermo}, out, m, C, z, F, pf) = (thermo_h!(out, m, C, z); forcing_thermo!(out, m, F);
    forcing_W!(out, m, F); sponge!(out, m, z))
_fast!(::Val{:hmomentum}, out, m, C, z, F, pf) = (prepare_pressure!(pf, m, z); momentum_h!(out, m, C, z, pf); forcing_U!(out, m, F))
_fast!(::Val{:vertical}, out, m, C, z, F, pf) = (thermo_v!(out, m, C, z); prepare_pressure!(pf, m, z);
    momentum_v!(out, m, C, z, pf))
_fast!(::Val{P}, out, m, C, z, F, pf) where {P} = error("frozen_fast!: unknown part $P")

"""Fast part F_f(C, z) + F (and the sponge) on the rows of `part` (table above)."""
function frozen_fast!(out, m::FVModel, C, z, part::Symbol, F)
    fill!(out, 0.0)
    _fast!(Val(part), out, m, C, z, F, pressure_form())
    return out
end

# ------------------------------------------------------------------ vertically implicit step

"""Column solver of the frozen HEVI loop for (I - c J_v) Y = R, J_v the vertical part with the
pressure difference α_k Θ_k - β_k Θ_{k-1} at the z-face k (WRF elimination of ρ and Θ):
    W_{k-1} [-r β_k θ_{k-1} + c² g/(2Δz)] + W_k [1 + r θ_k (α_k + β_k)]
    + W_{k+1} [-r α_k θ_{k+1} - c² g/(2Δz)] = R^W_k - (c/Δz)(α_k R^Θ_k - β_k R^Θ_{k-1}) - (c g/2)(R^ρ_{k-1} + R^ρ_k),
r = (c/Δz)². :gradp α_k = a_k, β_k = a_{k-1} with a = γ p/Θ at the start of the stage
(approximate Jacobian); :product α_k = β_k = (dp/dΘ)_k (exact: the vertical part is then linear
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
    @inbounds for k in 1:g.nz+1, i in 1:g.nx
        s.θz[i, k] = _θz(C, g, i, k)
    end
    column_coefficients!(pressure_form(), s, m, C, z)
    empty!(s.factors)
    return s
end

"""Coefficients α_k, β_k of the face pressure difference α_k Θ_k - β_k Θ_{k-1} in the column solve:
Product the frozen face derivative (exact), GradP γ p/Θ of the cells at the stage start."""
function column_coefficients!(::Product, s, m::FVModel, C, z)
    g = m.grid
    @inbounds for k in 2:g.nz, i in 1:g.nx
        s.α[i, k] = s.β[i, k] = _P2(C, g, i, k)
    end
    return s
end
function column_coefficients!(::GradP, s, m::FVModel, C, z)
    g = m.grid
    a(i, k) = (Θ = z[iΘ(g, i, k)]; m.gas.γ * pressure(m.gas, Θ) / Θ)
    @inbounds for k in 2:g.nz, i in 1:g.nx
        s.α[i, k] = a(i, k); s.β[i, k] = a(i, k - 1)
    end
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
z ← z + (I - dt/2 J_v)⁻¹ dt G_v(z) (= Crank–Nicolson for :product)."""
function frozen_vertical_step!(z, m::FVModel, C, dt, k, y)
    s = _frozen_column(m)
    frozen_fast!(k, m, C, z, :vertical, k)        # F unused for :vertical
    @. k *= dt
    _frozen_column_solve!(y, s, m, dt / 2, k)
    @. z += y
    return z
end
