# Application-specific linear stability of MIS methods for the frozen split of the FV and the DG
# model (instead of the 1D acoustic-advection test of Knoth & Wensch, `stability.jl`).
#
# For a linear problem the frozen split with exact fast stages is MIS with L = ∂F/∂z and
# N - L = ∂F/∂y (F(q, q) = N(q)). Linearized about a horizontally homogeneous hydrostatic state with
# uniform wind U, both operators are invariant under shifts of the columns (FV cells, DG element
# columns): per horizontal Fourier mode θ one block over the whole column, including gravity and the
# real vertical discretization (`MultiRate.LinearAnalysis.fourier_block`). J = Jacobian of
# `rhs_nonlinear!` (central differences), L = Jacobian of `frozen_fast!` with the coefficients
# frozen at the background (`frozen_data!`).
#
# The step matrix of an exact-fast-stage MIS method is formed in the eigenbasis of L (computed once
# per (U, θ)): stage i, dZ/dλ = Δt d_i L Z + Δt Σ_j β_ij S Y_j + Σ_j γ_ij (Y_{j+1} - yⁿ),
# Z(0) = yⁿ + Σ_j α_ij (Y_{j+1} - yⁿ), Y_{i+1} = Z(1) = e^{A} Z(0) + φ₁(A) f, A = Δt d_i L.
# Parameters: U and the advective Courant number C_adv = U Δt / Δx (Δx = cell width for FV, element
# width / p for DG, i.e. the node spacing); C_sound = c Δt / Δx follows (c/U ≈ 17 at U = 20 m/s).

using MultiRate, LinearAlgebra, Printf
const LA = MultiRate.LinearAnalysis
const TSA = MultiRate.TimeStepping
const FVM = MultiRate.FV2D
const E2M = MultiRate.Euler2D
const CMM = MultiRate.Common

"""Blocks of one application: per (U, θ) the eigenvalues Λ of L and S = V⁻¹ (J - L) V; dx the
horizontal spacing, c the sound speed."""
struct AppModes
    name::Symbol
    dx::Float64
    c::Float64
    Us::Vector{Float64}
    θs::Vector{Float64}
    Λ::Matrix{Vector{ComplexF64}}       # [iU, iθ]
    S::Matrix{Matrix{ComplexF64}}
    condV::Float64
end

_jac(f!, q0, n) = begin
    h = 1e-6 * max(1.0, maximum(abs, q0)); rp = zeros(n); rm = zeros(n)
    (out, v) -> (f!(rp, q0 .+ h .* v); f!(rm, q0 .- h .* v); @. out = (rp - rm) / (2h))
end

"""(model, idx) of the application: :fv (C-grid cells, nx columns) or :dg (P, Nex element columns)."""
function app_model(app; H = 10e3, nz = 16, nx = 24, dx = 2.5e3, P = 4, NEZ = 4, NEX = 24, DX = 10e3)
    gas = CMM.Gas()
    ρr, Θr, θr = CMM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    if app === :fv
        m = FVM.FVModel(FVM.FVGrid(nx, nz, nx * dx, H), gas, ρr, Θr)
        g = m.grid
        idx = vcat([[FVM.iρ(g, i, k) for i in 1:nx] for k in 1:nz], [[FVM.iU(g, i, k) for i in 1:nx] for k in 1:nz],
                   [[FVM.iW(g, i, k) for i in 1:nx] for k in 2:nz], [[FVM.iΘ(g, i, k) for i in 1:nx] for k in 1:nz])
        return m, idx, dx, (ρr, Θr)
    else
        m = E2M.Euler2DModel(E2M.Mesh2D(NEX, NEZ, P, NEX * DX, H), gas, ρr, Θr)
        me = m.mesh; np = me.np
        LI = LinearIndices((np, np, 4, me.Ne))
        idx = [[LI[a, b, v, E2M.elem(me, ex, ez)] for ex in 1:NEX] for a in 1:np, b in 1:np, v in 1:4, ez in 1:NEZ][:]
        return m, idx, DX / P, (ρr, Θr)
    end
end

"""Fourier blocks (J, L) [1/s] of the frozen split of `app` for the wind U and the mode θ."""
function app_blocks(m, idx, U, θ, refs)
    ρr, Θr = refs
    qb = CMM.project2d(m, (x, z) -> (ρr(z), U * ρr(z), 0.0, Θr(z)))
    n = length(qb)
    C = zeros(TSA.frozen_ncoef(m)); Sb = zeros(n)
    TSA.frozen_data!(C, Sb, m, qb)
    zero_f = zeros(n)
    Jf! = _jac((out, q) -> TSA.rhs_nonlinear!(out, m, q), qb, n)
    Lf! = _jac((out, z) -> TSA.frozen_fast!(out, m, C, z, :all, zero_f), qb, n)
    return LA.fourier_block(Jf!, idx, θ, n), LA.fourier_block(Lf!, idx, θ, n)
end

"""Build the modes of :fv or :dg for the winds Us and nθ modes θ ∈ (0, π] (the blocks for 2π - θ
are complex conjugate: same spectral radius). kwargs go to `app_model`."""
function app_modes(app; Us = (10.0, 20.0, 40.0), nθ = 12, kwargs...)
    m, idx, dx, refs = app_model(app; kwargs...)
    nc = length(idx[1])
    θs = [2π * j / nc for j in 1:min(nθ, nc ÷ 2)]
    Λ = Matrix{Vector{ComplexF64}}(undef, length(Us), length(θs)); S = Matrix{Matrix{ComplexF64}}(undef, length(Us), length(θs))
    cmax = 0.0
    for (iu, U) in enumerate(Us), (it, θ) in enumerate(θs)
        J, L = app_blocks(m, idx, U, θ, refs)
        E = eigen(L)
        cmax = max(cmax, cond(E.vectors))
        Λ[iu, it] = E.values
        S[iu, it] = E.vectors \ ((J - L) * E.vectors)
    end
    c = sqrt(CMM.Gas().γ * 1e5 / 1.16)        # nominal sound speed (≈ 347 m/s) for C_sound
    return AppModes(app, dx, c, collect(Float64, Us), θs, Λ, S, cmax)
end

φ1(z) = abs(z) < 1e-6 ? 1 + z / 2 + z^2 / 6 : (exp(z) - 1) / z

"""Spectral radius of the exact-fast-stage MIS step (compact α, β, γ) for the block (Λ, S), Δt."""
function step_radius(Λ, S, Δt, α, β, γ)
    n = length(Λ); s = size(α, 1)
    Y = Vector{Matrix{ComplexF64}}(undef, s + 1)
    Y[1] = Matrix{ComplexF64}(I, n, n)
    SY = Vector{Matrix{ComplexF64}}(undef, s)
    for i in 1:s
        SY[i] = S * Y[i]
        d = sum(β[i, 1:i])
        Z0 = copy(Y[1]); f = zeros(ComplexF64, n, n)
        for j in 1:i
            iszero(β[i, j]) || (f .+= (Δt * β[i, j]) .* SY[j])
            j < i || continue
            iszero(α[i, j]) || (Z0 .+= α[i, j] .* (Y[j+1] .- Y[1]))
            iszero(γ[i, j]) || (f .+= γ[i, j] .* (Y[j+1] .- Y[1]))
        end
        A = Δt * d .* Λ
        Y[i+1] = exp.(A) .* Z0 .+ φ1.(A) .* f
    end
    return maximum(abs, eigvals(Y[s+1]))
end

"""(α, β, γ) of a package tableau in the compact s × s form."""
function compact_tableau(tab)
    n = size(tab.α, 1)
    iszero(tab.β[1, 1]) && return tab.α[2:n, 1:n-1], tab.β[2:n, 1:n-1], tab.γ[2:n, 1:n-1]
    return Matrix(tab.α), Matrix(tab.β), Matrix(tab.γ)
end

"""Maximum amplification over θ for the wind Us[iu] and C_adv."""
app_amplification(M::AppModes, abc, iu, ca) =
    maximum(step_radius(M.Λ[iu, it], M.S[iu, it], ca * M.dx / M.Us[iu], abc...) for it in eachindex(M.θs))

"""Largest stable C_adv (resolution dc, up to camax) per wind."""
function app_limits(M::AppModes, abc; camax = 3.0, dc = 0.1, tol = 1e-8)
    map(eachindex(M.Us)) do iu
        ca = dc
        while ca <= camax + 1e-12
            app_amplification(M, abc, iu, ca) > 1 + tol && return ca - dc
            ca += dc
        end
        camax
    end
end

"""Σ max(0, ρ - 1)² over the winds, C_adv ∈ Ca and the modes."""
app_penalty(M::AppModes, abc; Ca = range(0.25, 2.0; length = 8)) =
    sum(max(0.0, step_radius(M.Λ[iu, it], M.S[iu, it], ca * M.dx / M.Us[iu], abc...) - 1)^2
        for iu in eachindex(M.Us), it in eachindex(M.θs), ca in Ca)

if abspath(PROGRAM_FILE) == @__FILE__
    names = (:WSRK3, :MIS4, :MIS4a, :MIS54, :MISF53, :MIS_KW3)
    for app in (:fv, :dg)
        t = @elapsed M = app_modes(app)
        @printf("\n%s: Δx = %.0f m, U = %s m/s, %d modes, block size %d, cond(V) ≤ %.1e (%.0f s)\n", app, M.dx,
                join(Int.(M.Us), "/"), length(M.θs), length(M.Λ[1]), M.condV, t)
        @printf("max Re λ(J) check skipped; largest stable C_adv (C_sound = c/U · C_adv ≈ %s · C_adv):\n",
                join([@sprintf("%.0f", M.c / U) for U in M.Us], "/"))
        for name in names
            abc = compact_tableau(TSA.mis_tableau(name))
            @printf("  %-8s %s\n", name, join([@sprintf("%.1f", l) for l in app_limits(M, abc)], "  "))
            flush(stdout)
        end
    end
end
