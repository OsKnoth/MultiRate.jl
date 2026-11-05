# Fast operator (matrix-free, traces eliminated), re-freezing

mutable struct HDGFast2D <: AbstractFastOperator
    mesh::Mesh2D
    gas::Gas
    nq::Int
    nt::Int
    eps::Float64
    tau::Float64
    Mdiag::Vector{Float64}
    Minv::Vector{Float64}
    fb::Vector{Float64}
    # frozen, continuous coefficients at the nodes (np × np × Ne)
    θ::Array{Float64,3}
    a::Array{Float64,3}
    φ::Array{Float64,3}
    φa::Array{Float64,3}
    b::Array{Float64,3}
    gξx::Array{Float64,3}
    gξz::Array{Float64,3}
    gηx::Array{Float64,3}
    gηz::Array{Float64,3}  # Ja ∂θ̄
    cmax::Float64
    # buoyancy split: g J e_z = Ja^ξ D_ξ(gZ) + Ja^η D_η(gZ) (exact with collocation metrics)
    gZξ::Array{Float64,3}
    gZη::Array{Float64,3}
    # HEVI: column solver (element-local elimination + tridiagonal system for the face momentum)
    hevi::Bool
    lucache::Dict{Float64,Any}      # substep size dt -> HeviFactor
    # interface variable: contravariant momentum ñ·m̂ at the face nodes (Verlet / forward-backward)
    Mhx::Matrix{Float64}            # ξ-faces: (np, Nex Nez), face to the right of element e
    Mhz::Matrix{Float64}            # η-faces: (np, Nex (Nez+1)), index (kz-1) Nex + ex
    eps_if::Float64                 # relaxation time of the interface variable (:fb)
    # work arrays
    work::Vector{Vector{Float64}}
    Fξ::Matrix{Float64}
    Fη::Matrix{Float64}
    ψ::Matrix{Float64}
    Iint::Matrix{Float64}
    dssbuf::Vector{Float64}
    # split-consistent linearization of N (see `DG_SPLIT_FAST`): frozen density and the
    # two-point means of the split form along η-lines [j,k,i,e] and ξ-lines [i,k,j,e]
    split::Bool
    ρ0::Array{Float64,3}
    Mv1::Array{Float64,4}; Mv4::Array{Float64,4}; Gv1::Array{Float64,4}; Gv2::Array{Float64,4}
    Mh1::Array{Float64,4}; Mh4::Array{Float64,4}; Gh1::Array{Float64,4}; Gh2::Array{Float64,4}
    splitok::Bool                   # fast solver uses fast_kernel!/column solver only (:hevi, :hevi_verlet, :hevi_relax)
    vadv::Bool                      # vertical momentum transport by the fast velocity (DG_VADV_FAST)
    u0::Array{Float64,3}
    w0::Array{Float64,3}
end

"""
Include in the vertical fast part the transport of momentum by the fast vertical mass flux,
m_t ∋ -∂_η(ρ̄ w'·v̄) (linearized split form, frozen velocity v̄; SAT accordingly). Without it
this term sits in the explicit slow part and couples the slow step to the vertical acoustics
with Δt·|v̄|·∂_z (DG Agnesi a100 instability). Needs `DG_SPLIT_FAST`; both apply to the
fast solvers :hevi, :hevi_verlet and :hevi_relax only.
"""
const DG_VADV_FAST = Ref(true)

"""
Build the fast operator as the linearization of the split-form N (two-point means of mass,
Θ and gravity, frozen at qⁿ) instead of pointwise products. The two differ in the highest
polynomial mode of each element, which leaves a purely imaginary mode ≈ 0.02 (p+1) s⁻¹ in the
explicitly treated slow part N - L. The vertical part uses the nodal metric Ja^η_j, so the
column solver still sees the momentum only through U = Ja^η·m.
"""
const DG_SPLIT_FAST = Ref(true)

function HDGFast2D(m::Euler2DModel; tau_scale = 1.0)
    M = m.mesh
    np, Ne, nq = M.np, M.Ne, m.nq
    arr() = zeros(np, np, Ne)
    Mdiag = zeros(nq)
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np, r in 1:NV
        Mdiag[node_iq(np, e, r, i, j)] = M.J[i, j, e] * M.wg[i] * M.wg[j]
    end
    xq, wq = gauss_legendre(np)
    Iint = zeros(np, np)                               # Iint[j, k] = ∫_{-1}^{η_j} ℓ_k
    for j in 1:np
        V, _ = lagrange_basis(M.xg, -1.0 .+ 0.5 * (M.xg[j] + 1) .* (xq .+ 1))
        Iint[j, :] = 0.5 * (M.xg[j] + 1) .* (V' * wq)
    end
    gZξ = arr()
    gZη = arr()
    g = m.gas.g
    for e in 1:Ne
        gZξ[:, :, e] = g .* (M.D * M.Z[:, :, e])
        gZη[:, :, e] = g .* (M.Z[:, :, e] * M.D')
    end
    return HDGFast2D(M, m.gas, nq, 0, 0.0, Float64(tau_scale), Mdiag, 1 ./ Mdiag, zeros(nq),
                     arr(), arr(), arr(), arr(), arr(), arr(), arr(), arr(), arr(), 0.0,
                     gZξ, gZη, false, Dict{Float64,Any}(),
                     zeros(np, Ne), zeros(np, M.Nex * (M.Nez + 1)), 0.0,
                     [zeros(nq) for _ in 1:7], zeros(np, np), zeros(np, np), zeros(np, np), Iint,
                     zeros(length(M.mult)),
                     false, arr(), (zeros(np, np, np, Ne) for _ in 1:8)..., false, false, arr(), arr())
end

"""
    fast_operator(model, q; pfac=γ, eps=0.0, tau_scale=1.0, fast=:imex, reuse=true)

Fast operator with coefficients frozen at the full 2D state q (matrix-free). `eps_if` is the
relaxation time for the face momentum used by `:fb`. With reuse = true the operator stored in
the model is re-frozen in place (no reallocation); fast = :hevi also (re)builds the vertical
implicit operator.
"""
function fast_operator(m::Euler2DModel, q; pfac = m.gas.γ, eps = 0.0, eps_if = 0.0, tau_scale = 1.0, fast::Symbol = :imex,
                       reuse::Bool = true)
    eps == 0 || error("the 2D fast operator eliminates the traces (eps = 0)")
    eps_if >= 0 || throw(ArgumentError("eps_if must be nonnegative"))
    s = reuse && m.fastref[] isa HDGFast2D ? m.fastref[]::HDGFast2D : HDGFast2D(m; tau_scale = tau_scale)
    s.tau = Float64(tau_scale)
    s.eps_if = Float64(eps_if)
    s.hevi |= fast in (:hevi, :hevi_relax, :hevi_verlet)
    # fb, verlet and imex integrate their own pointwise kernels; N - L must use the same L
    s.splitok = fast in (:hevi, :hevi_verlet, :hevi_relax)
    refreeze!(s, m, q, pfac)
    reuse && (m.fastref[] = s)
    return s
end

"""Re-freeze the coefficients of s at the state q (in place)."""
function refreeze!(s::HDGFast2D, m::Euler2DModel, q, pfac)
    M = m.mesh
    np, Ne, D, w, g = M.np, M.Ne, M.D, M.wg, m.gas.g
    θ, a, b, φ = s.θ, s.a, s.b, s.φ
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        ρ = q[node_iq(np, e, 1, i, j)]
        Θ = q[node_iq(np, e, 4, i, j)]
        p = pressure(m.gas, Θ)
        θ[i, j, e] = Θ / ρ
        a[i, j, e] = pfac * p / Θ
        b[i, j, e] = (1 - pfac) * p
    end
    dss!(θ, M, s.dssbuf)
    dss!(a, M, s.dssbuf)
    dss!(b, M, s.dssbuf)
    # χ = ∫ g/c² dz along the η-lines from χ_s = g z_s / c²(z_s); φ = θ̄ exp(χ)
    @inbounds for ex in 1:M.Nex, i in 1:np
        e1 = elem(M, ex, 1)
        base = g * M.Z[i, 1, e1] / (a[i, 1, e1] * θ[i, 1, e1])
        for ez in 1:M.Nez
            e = elem(M, ex, ez)
            for j in 1:np
                I = base
                for k in 1:np
                    zη = 0.0
                    for l in 1:np
                        zη += D[k, l] * M.Z[i, l, e]
                    end
                    I += s.Iint[j, k] * g / (a[i, k, e] * θ[i, k, e]) * zη
                end
                φ[i, j, e] = I
            end
            base = φ[i, np, e]
        end
    end
    s.cmax = 0.0
    @inbounds for k in eachindex(φ)
        φ[k] = θ[k] * exp(φ[k])
        s.φa[k] = φ[k] * a[k]
        s.cmax = max(s.cmax, sqrt(a[k] * θ[k]))
    end
    fill!(s.fb, 0.0)
    @inbounds for e in 1:Ne, j in 1:np, i in 1:np
        dξθ = dηθ = dξb = dηb = 0.0
        for k in 1:np
            dξθ += D[i, k] * θ[k, j, e]
            dηθ += D[j, k] * θ[i, k, e]
            dξb += D[i, k] * b[k, j, e]
            dηb += D[j, k] * b[i, k, e]
        end
        s.gξx[i, j, e] = M.Jξx[i, j, e] * dξθ
        s.gξz[i, j, e] = M.Jξz[i, j, e] * dξθ
        s.gηx[i, j, e] = M.Jηx[i, j, e] * dηθ
        s.gηz[i, j, e] = M.Jηz[i, j, e] * dηθ
        W = w[i] * w[j]
        s.fb[node_iq(np, e, 2, i, j)] = -W * (M.Jξx[i, j, e] * dξb + M.Jηx[i, j, e] * dηb)
        s.fb[node_iq(np, e, 3, i, j)] = -W * (M.Jξz[i, j, e] * dξb + M.Jηz[i, j, e] * dηb)
    end
    s.split = DG_SPLIT_FAST[] && s.splitok
    s.split && refreeze_split!(s, m, q)
    s.vadv = s.split && DG_VADV_FAST[]
    for (c, F) in s.lucache
        factor_hevi!(F, s, c)
    end
    return s
end

"""Two-point means of the split form at the frozen state (unit normal velocity) and the
derivatives of the gravity density mean."""
function refreeze_split!(s::HDGFast2D, m, q)
    M = m.mesh
    np, γ, vf, gf = M.np, m.gas.γ, m.vflux, m.gflux
    gd(ρa, ρb) = begin
        h = 1e-6 * ρa
        (gravity_mean(gf, ρa + h, ρb, γ) - gravity_mean(gf, ρa - h, ρb, γ)) / 2h
    end
    @inbounds for e in 1:M.Ne, j in 1:np, i in 1:np
        s.ρ0[i, j, e] = q[node_iq(np, e, 1, i, j)]
        s.u0[i, j, e] = q[node_iq(np, e, 2, i, j)] / s.ρ0[i, j, e]
        s.w0[i, j, e] = q[node_iq(np, e, 3, i, j)] / s.ρ0[i, j, e]
    end
    @inbounds for e in 1:M.Ne, i in 1:np, j in 1:np, k in 1:np
        ρa, Θa = q[node_iq(np, e, 1, i, j)], q[node_iq(np, e, 4, i, j)]
        ρb, Θb = q[node_iq(np, e, 1, i, k)], q[node_iq(np, e, 4, i, k)]       # η-line pair (j, k)
        s.Mv1[j, k, i, e], s.Mv4[j, k, i, e] = mass_theta_flux(vf, ρa, Θa, ρb, Θb, 1.0, γ)
        s.Gv1[j, k, i, e], s.Gv2[j, k, i, e] = gd(ρa, ρb), gd(ρb, ρa)
        ρa, Θa = q[node_iq(np, e, 1, j, i)], q[node_iq(np, e, 4, j, i)]
        ρb, Θb = q[node_iq(np, e, 1, k, i)], q[node_iq(np, e, 4, k, i)]       # ξ-line pair (j, k) at row i
        s.Mh1[j, k, i, e], s.Mh4[j, k, i, e] = mass_theta_flux(vf, ρa, Θa, ρb, Θb, 1.0, γ)
        s.Gh1[j, k, i, e], s.Gh2[j, k, i, e] = gd(ρa, ρb), gd(ρb, ρa)
    end
    return s
end

"""(A_ñ d, T_ñ d) of the SAT at a face node: ñ = scaled outward normal, d = (d1, d2, d3, d4)."""
@inline function sat_parts(a, θ, nx, nz, d2, d3, d4, tau)
    c = sqrt(a * θ)
    s = hypot(nx, nz)
    n1, n2 = nx / s, nz / s
    mn = nx * d2 + nz * d3
    dn = n1 * d2 + n2 * d3
    st = tau * s
    return (mn, a * nx * d4, a * nz * d4, θ * mn), (st * a / c * d4, st * c * n1 * dn, st * c * n2 * dn, st * c * d4)
end

@inline function add4!(out, np, e, i, j, f, v1, v2, v3, v4)
    @inbounds begin
        out[node_iq(np, e, 1, i, j)] += f * v1
        out[node_iq(np, e, 2, i, j)] += f * v2
        out[node_iq(np, e, 3, i, j)] += f * v3
        out[node_iq(np, e, 4, i, j)] += f * v4
    end
    return nothing
end

"""SAT contribution of an interior face node pair L (outward ñ) / R, traces = averages."""
@inline function sat_face!(out, s, q, iL, jL, eL, iR, jR, eR, nx, nz, we, adv::Bool = false)
    np = s.mesh.np
    @inbounds begin
        d2 = 0.5 * (q[node_iq(np, eL, 2, iL, jL)] - q[node_iq(np, eR, 2, iR, jR)])
        d3 = 0.5 * (q[node_iq(np, eL, 3, iL, jL)] - q[node_iq(np, eR, 3, iR, jR)])
        d4 = 0.5 * (q[node_iq(np, eL, 4, iL, jL)] - q[node_iq(np, eR, 4, iR, jR)])
        Ad, Td = sat_parts(s.a[iL, jL, eL], s.θ[iL, jL, eL], nx, nz, d2, d3, d4, s.tau)
        # L: (A_ñ - T) d;  R: (A_{-ñ} - T)(-d) = (A_ñ + T) d
        add4!(out, np, eL, iL, jL, we, Ad[1] - Td[1], Ad[2] - Td[2], Ad[3] - Td[3], Ad[4] - Td[4])
        add4!(out, np, eR, iR, jR, we, Ad[1] + Td[1], Ad[2] + Td[2], Ad[3] + Td[3], Ad[4] + Td[4])
        if adv
            cL, cR = Ad[1] - Td[1], Ad[1] + Td[1]
            add4!(out, np, eL, iL, jL, we, 0.0, cL * s.u0[iL, jL, eL], cL * s.w0[iL, jL, eL], 0.0)
            add4!(out, np, eR, iR, jR, we, 0.0, cR * s.u0[iR, jR, eR], cR * s.w0[iR, jR, eR], 0.0)
        end
    end
    return nothing
end

"""SAT at a wall node: q - q̂ = (0, n̂ (n̂·m), 0)."""
@inline function sat_wall!(out, s, q, i, j, e, nx, nz, we, adv::Bool = false)
    np = s.mesh.np
    @inbounds begin
        h = hypot(nx, nz)
        n1, n2 = nx / h, nz / h
        mn = n1 * q[node_iq(np, e, 2, i, j)] + n2 * q[node_iq(np, e, 3, i, j)]
        Ad, Td = sat_parts(s.a[i, j, e], s.θ[i, j, e], nx, nz, n1 * mn, n2 * mn, 0.0, s.tau)
        add4!(out, np, e, i, j, we, Ad[1] - Td[1], Ad[2] - Td[2], Ad[3] - Td[3], Ad[4] - Td[4])
        if adv
            cL = Ad[1] - Td[1]
            add4!(out, np, e, i, j, we, 0.0, cL * s.u0[i, j, e], cL * s.w0[i, j, e], 0.0)
        end
    end
    return nothing
end

"""
out = (K_part q + G_part H q + f) ./ M, matrix-free; HP: horizontal (ξ) part, VP: vertical (η)
part including buoyancy. Allocation-free.
"""
function fast_kernel!(out, s::HDGFast2D, q, f, ::Val{HP}, ::Val{VP}) where {HP,VP}
    M = s.mesh
    np, D, w, g = M.np, M.D, M.wg, s.gas.g
    Fξ, Fη, ψ = s.Fξ, s.Fη, s.ψ
    fill!(out, 0.0)
    @inbounds for e in 1:M.Ne
        for j in 1:np, i in 1:np
            mx = q[node_iq(np, e, 2, i, j)]
            mz = q[node_iq(np, e, 3, i, j)]
            Fξ[i, j] = M.Jξx[i, j, e] * mx + M.Jξz[i, j, e] * mz
            Fη[i, j] = M.Jηx[i, j, e] * mx + M.Jηz[i, j, e] * mz
            ψ[i, j] = s.φa[i, j, e] * q[node_iq(np, e, 4, i, j)]
        end
        for j in 1:np, i in 1:np
            ρ = q[node_iq(np, e, 1, i, j)]
            mx = q[node_iq(np, e, 2, i, j)]
            mz = q[node_iq(np, e, 3, i, j)]
            Θ = q[node_iq(np, e, 4, i, j)]
            θ = s.θ[i, j, e]
            aΘθ = s.a[i, j, e] * Θ / θ
            iφ = 1 / s.φ[i, j, e]
            r1 = r2 = r3 = r4 = 0.0
            if HP && s.split
                d1 = d4 = pg = gg = 0.0
                for k in 1:np
                    ax = 0.5 * (M.Jξx[i, j, e] + M.Jξx[k, j, e])
                    az = 0.5 * (M.Jξz[i, j, e] + M.Jξz[k, j, e])
                    vn = ax * (mx / s.ρ0[i, j, e] + q[node_iq(np, e, 2, k, j)] / s.ρ0[k, j, e]) +
                         az * (mz / s.ρ0[i, j, e] + q[node_iq(np, e, 3, k, j)] / s.ρ0[k, j, e])
                    d1 += D[i, k] * s.Mh1[i, k, j, e] * vn
                    d4 += D[i, k] * s.Mh4[i, k, j, e] * vn
                    dp = s.a[i, j, e] * Θ + s.a[k, j, e] * q[node_iq(np, e, 4, k, j)]
                    gr = g * (M.Z[k, j, e] - M.Z[i, j, e]) *
                         (s.Gh1[i, k, j, e] * ρ + s.Gh2[i, k, j, e] * q[node_iq(np, e, 1, k, j)])
                    r2 -= D[i, k] * ax * (dp + gr)
                    r3 -= D[i, k] * az * (dp + gr)
                end
                r1 -= d1
                r4 -= d4
            elseif HP
                div = dψ = 0.0
                for k in 1:np
                    div += D[i, k] * Fξ[k, j]
                    dψ += D[i, k] * ψ[k, j]
                end
                gx, gz = s.gξx[i, j, e], s.gξz[i, j, e]
                r1 -= div
                r4 -= θ * div + mx * gx + mz * gz
                r2 -= M.Jξx[i, j, e] * dψ * iφ - aΘθ * gx
                r3 -= M.Jξz[i, j, e] * dψ * iφ - aΘθ * gz
                bξ = (Θ / θ - ρ) * s.gZξ[i, j, e]                # buoyancy, ξ part
                r2 += bξ * M.Jξx[i, j, e]
                r3 += bξ * M.Jξz[i, j, e]
            end
            if VP && s.split
                d1 = d4 = pg = 0.0
                Uj = Fη[i, j] / s.ρ0[i, j, e]
                for k in 1:np
                    vn = Uj + Fη[i, k] / s.ρ0[i, k, e]
                    t1 = D[j, k] * s.Mv1[j, k, i, e] * vn
                    d1 += t1
                    if s.vadv
                        r2 -= t1 * 0.5 * (s.u0[i, j, e] + s.u0[i, k, e])
                        r3 -= t1 * 0.5 * (s.w0[i, j, e] + s.w0[i, k, e])
                    end
                    d4 += D[j, k] * s.Mv4[j, k, i, e] * vn
                    pg += D[j, k] * (s.a[i, k, e] * q[node_iq(np, e, 4, i, k)] +
                                     g * (M.Z[i, k, e] - M.Z[i, j, e]) *
                                     (s.Gv1[j, k, i, e] * ρ + s.Gv2[j, k, i, e] * q[node_iq(np, e, 1, i, k)]))
                end
                r1 -= d1
                r4 -= d4
                r2 -= M.Jηx[i, j, e] * pg
                r3 -= M.Jηz[i, j, e] * pg
            elseif VP
                div = dψ = 0.0
                for k in 1:np
                    div += D[j, k] * Fη[i, k]
                    dψ += D[j, k] * ψ[i, k]
                end
                gx, gz = s.gηx[i, j, e], s.gηz[i, j, e]
                r1 -= div
                r4 -= θ * div + mx * gx + mz * gz
                r2 -= M.Jηx[i, j, e] * dψ * iφ - aΘθ * gx
                r3 -= M.Jηz[i, j, e] * dψ * iφ - aΘθ * gz
                bη = (Θ / θ - ρ) * s.gZη[i, j, e]                # buoyancy, η part
                r2 += bη * M.Jηx[i, j, e]
                r3 += bη * M.Jηz[i, j, e]
            end
            W = w[i] * w[j]
            out[node_iq(np, e, 1, i, j)] = W * r1
            out[node_iq(np, e, 2, i, j)] = W * r2
            out[node_iq(np, e, 3, i, j)] = W * r3
            out[node_iq(np, e, 4, i, j)] = W * r4
        end
    end
    for ez in 1:M.Nez, ex in 1:M.Nex
        e = elem(M, ex, ez)
        if HP
            eR = elem(M, mod1(ex + 1, M.Nex), ez)
            for j in 1:np
                sat_face!(out, s, q, np, j, e, 1, j, eR, M.Jξx[np, j, e], M.Jξz[np, j, e], w[j])
            end
        end
        if VP
            for i in 1:np
                if ez < M.Nez
                    sat_face!(out, s, q, i, np, e, i, 1, elem(M, ex, ez + 1), M.Jηx[i, np, e], M.Jηz[i, np, e], w[i], s.vadv)
                else
                    sat_wall!(out, s, q, i, np, e, M.Jηx[i, np, e], M.Jηz[i, np, e], w[i], s.vadv)
                end
                ez == 1 && sat_wall!(out, s, q, i, 1, e, -M.Jηx[i, 1, e], -M.Jηz[i, 1, e], w[i], s.vadv)
            end
        end
    end
    @inbounds for k in eachindex(out)
        out[k] = (out[k] + f[k]) * s.Minv[k]
    end
    return out
end

fast_apply!(out, s::HDGFast2D, q, f) = fast_kernel!(out, s, q, f, Val(true), Val(true))
