# Linear stability of MIS methods with polynomial coupling for y' = S y + L y (class :L, the
# Fourier/Bloch symbols of the test equation): step matrix with exact or approximate inner problems.
# Stage i: Z' = q_i(τ) L Z + r_i(τ), r_i = Σ_j g_ij (Y_j - I) + Σ_j p_ij S Y_j (h = 1).

"""Forcing coefficients R_k (r_i(τ) = Σ_k R_k τᵏ) of stage i, given the stage matrices Y."""
function stage_forcing(m::PolyMIS, i, S, Y)
    K = maximum(max(length(m.p[i, j]), length(m.g[i, j])) for j in 1:i-1)
    n = size(S, 1)
    R = [zeros(ComplexF64, n, n) for _ in 1:K]
    for j in 1:i-1
        SY = S * Y[j]
        for (k, a) in enumerate(m.p[i, j]); R[k] .+= a .* SY; end
        for (k, a) in enumerate(m.g[i, j]); R[k] .+= a .* (Y[j] - I); end
    end
    return R
end

"""∫_{τa}^{τb} Σ_k R_k τᵏ dτ."""
forcing_integral(R, τa, τb) = sum(R[k] .* ((τb^k - τa^k) / k) for k in eachindex(R))
forcing_value(R, τ) = sum(R[k] .* τ^(k - 1) for k in eachindex(R))

"""Inner problem of one stage over τ ∈ [0, 1]."""
function inner_solve(Z0, q, L, R, inner, thermo)
    n = size(L, 1)
    if inner === :exact
        all(abs(c) <= 1e-13 for c in q[2:end]) ||
            throw(ArgumentError("inner = :exact needs uniform fast speed (constant q_i)"))
        K = length(R)
        # augmented autonomous system: Z' = q L Z + u₀, u_k' = u_{k+1}, u_k(0) = k! R_k
        N = n * (K + 1)
        M = zeros(ComplexF64, N, N)
        M[1:n, 1:n] .= q[1] .* L
        for k in 1:K
            M[(k-1)*n+1:k*n, k*n+1:(k+1)*n] .= I(n)
        end
        X0 = vcat(Z0, (factorial(k - 1) .* R[k] for k in 1:K)...)
        return (exp(M)*X0)[1:n, :]
    end
    kind, ns = inner
    Δ = 1 / ns
    Z = copy(Z0)
    qv(τ) = peval(q, τ)
    if kind === :rk4
        f(τ, Z) = qv(τ) .* (L * Z) .+ forcing_value(R, τ)
        for k in 0:ns-1
            τ = k * Δ
            k1 = f(τ, Z); k2 = f(τ + Δ / 2, Z .+ Δ / 2 .* k1)
            k3 = f(τ + Δ / 2, Z .+ Δ / 2 .* k2); k4 = f(τ + Δ, Z .+ Δ .* k3)
            Z = Z .+ Δ / 6 .* (k1 .+ 2k2 .+ 2k3 .+ k4)
        end
        return Z
    end
    T, U = thermo, .!thermo
    LTT, LTU, LUT, LUU = L[T, T], L[T, U], L[U, T], L[U, U]
    for k in 0:ns-1
        τ = k * Δ
        w = Δ * qv(τ + Δ / 2)                       # fast time of the substep
        ZT, ZU = Z[T, :], Z[U, :]
        if kind === :fb                             # thermodynamic rows first, momentum with the new ones
            RI = forcing_integral(R, τ, τ + Δ)
            ZT = ZT .+ w .* (LTT * ZT .+ LTU * ZU) .+ RI[T, :]
            ZU = ZU .+ w .* (LUT * ZT .+ LUU * ZU) .+ RI[U, :]
        elseif kind === :verlet                     # half thermodynamic, full momentum, half thermodynamic
            R1 = forcing_integral(R, τ, τ + Δ / 2); R2 = forcing_integral(R, τ + Δ / 2, τ + Δ)
            ZT = ZT .+ (w / 2) .* (LTT * ZT .+ LTU * ZU) .+ R1[T, :]
            ZU = ZU .+ w .* (LUT * ZT .+ LUU * ZU) .+ (R1 .+ R2)[U, :]
            ZT = ZT .+ (w / 2) .* (LTT * ZT .+ LTU * ZU) .+ R2[T, :]
        else
            throw(ArgumentError("inner = :exact, (:rk4, n), (:fb, n) or (:verlet, n)"))
        end
        Z = similar(Z); Z[T, :] = ZT; Z[U, :] = ZU
    end
    return Z
end

"""
    amplification(m::PolyMIS, S, L; inner = :exact, thermo = first half of the unknowns)

Step matrix of the MIS method for y' = S y + L y (h = 1): exact inner problems or `ns` substeps of
`(:rk4, ns)`, `(:fb, ns)` (forward–backward, thermodynamic rows first) or `(:verlet, ns)`.
"""
function amplification(m::PolyMIS, S::AbstractMatrix, L::AbstractMatrix; inner = :exact,
                       thermo = (n = size(L, 1); [k <= n ÷ 2 for k in 1:n]))
    n = size(S, 1)
    s1 = size(m.α, 1)
    Y = Vector{Matrix{ComplexF64}}(undef, s1)
    Y[1] = Matrix{ComplexF64}(I, n, n)
    for i in 2:s1
        Z0 = Matrix{ComplexF64}(I, n, n) .+ sum(m.α[i, j] .* (Y[j] - I) for j in 1:i-1)
        R = stage_forcing(m, i, S, Y)
        Y[i] = inner_solve(Z0, fast_speed(m, i), L, R, inner, BitVector(thermo))
    end
    return Y[s1]
end

"""
    max_amplification(m, sym; Cadv, Csound, θs = range(0, π, length = 48), inner = :exact)

max over θ of the spectral radius of the step matrix for S = C_adv S₁(θ), L = C_sound L₁(θ), with
`sym = symbols(:fv; …)` or `symbols(:dg; …)`.
"""
function max_amplification(m::PolyMIS, sym; Cadv, Csound, θs = range(0, π, length = 48), inner = :exact)
    ρ = 0.0
    for θ in θs
        S1, L1 = sym(θ)
        ρ = max(ρ, maximum(abs, eigvals(amplification(m, Cadv .* S1, Csound .* L1; inner))))
    end
    return ρ
end

"""
    adv_limit(m, sym; Csound, dc = 0.05, cmax = 4.0, tol = 1e-8, kw...)

Largest C_adv (on the grid dc, 2dc, …) such that the method is stable (ρ ≤ 1 + tol) for all
C_adv' ≤ C_adv at the given C_sound.
"""
function adv_limit(m::PolyMIS, sym; Csound, dc = 0.05, cmax = 4.0, tol = 1e-8, kw...)
    c = 0.0
    while c + dc <= cmax + 1e-12
        max_amplification(m, sym; Cadv = c + dc, Csound, kw...) > 1 + tol && return c
        c += dc
    end
    return c
end

"""ρ - 1 on the grid Cadvs × Csounds."""
stability_map(m::PolyMIS, sym; Cadvs, Csounds, kw...) =
    [max_amplification(m, sym; Cadv = ca, Csound = cs, kw...) - 1 for ca in Cadvs, cs in Csounds]
