# Fast step for the DGSEM-LMARS system with the interface momentum M̂ as unknown (docs/hdg_extended), 1D periodic
# acoustics, DGSEM degree P, c = 1, element width 1, a = p (thermodynamic), b = m. Groups X = a, Y = (b, M̂):
#   a' = K Y,   Y' = G a + D_Y Y,
#   K Y:  a_k' = -(Dv b)_k - lw (M̂_ν - b_ν)                     (face nodes; ν outward)
#   b_k' = -(Dv a)_k + lw c ν (M̂_ν - b_ν),   M̂' = κ (a_L - a_R + c (b_L + b_R) - 2c M̂),   ε = 1/(2cκ).
# Steps: Störmer–Verlet in the groups (Bokhove, without the Coriolis term): a half step, Y full step, a half step;
# in the Y step D_Y explicit (plain Verlet) or trapezoidal (Verwer); or symplectic Euler (Y step, then a).
# Prints the stability limit h c/Δx_e (spectral radius of the step map ≤ 1 + 1e-6).
#     P=4 julia docs/hdg_extended/fast_groups_1d.jl
using LinearAlgebra, Printf
function lgl(P)
    x = -cos.(π * (0:P) / P)
    for _ in 1:100
        L0 = ones(P + 1); L1 = copy(x)
        for k in 2:P
            L2 = ((2k - 1) .* x .* L1 .- (k - 1) .* L0) ./ k; L0, L1 = L1, L2
        end
        dx = (x .* L1 .- L0) ./ ((P + 1) .* L1); x .-= dx
        maximum(abs, dx) < 1e-15 && break
    end
    L0 = ones(P + 1); L1 = copy(x)
    for k in 2:P
        L2 = ((2k - 1) .* x .* L1 .- (k - 1) .* L0) ./ k; L0, L1 = L1, L2
    end
    w = 2 ./ (P * (P + 1) .* L1 .^ 2)
    D = [i != j ? L1[i] / (L1[j] * (x[i] - x[j])) : (i == 1 ? -P * (P + 1) / 4 : i == P + 1 ? P * (P + 1) / 4 : 0.0) for i in 1:P+1, j in 1:P+1]
    return x, w, D
end
P = parse(Int, get(ENV, "P", "4")); Ne = 16; c = 1.0
x, w, Dm = lgl(P); np = P + 1; n = np * Ne; J = 0.5
idx(e, i) = (mod1(e, Ne) - 1) * np + i
Dv = zeros(n, n)
for e in 1:Ne, i in 1:np, j in 1:np
    Dv[idx(e, i), idx(e, j)] = Dm[i, j] / J
end
lw = 1 / (J * w[1])
function operators(κ)
    Ka = -copy(Dv); KM = zeros(n, Ne)             # a' = Ka b + KM M̂
    Lb = -copy(Dv); Db = zeros(n, n); QM = zeros(n, Ne)   # b' = Lb a + Db b + QM M̂
    Ha = zeros(Ne, n); Hb = zeros(Ne, n); HM = zeros(Ne, Ne)   # M̂' = Ha a + Hb b + HM M̂
    for f in 1:Ne
        kL, kR = idx(f, np), idx(f + 1, 1)
        # L: ν = +1, M̂_ν = M̂, b_ν = b_L;  R: ν = -1, M̂_ν = -M̂, b_ν = -b_R
        KM[kL, f] += -lw; Ka[kL, kL] += lw                 # -lw (M̂ - b_L)
        KM[kR, f] += lw;  Ka[kR, kR] += -lw                # +lw (M̂ - b_R)
        QM[kL, f] += lw * c; Db[kL, kL] += -lw * c         # +lw c (M̂ - b_L)
        QM[kR, f] += lw * c; Db[kR, kR] += -lw * c         # +lw c (M̂ - b_R)
        Ha[f, kL] += κ; Ha[f, kR] += -κ
        Hb[f, kL] += κ * c; Hb[f, kR] += κ * c
        HM[f, f] = -2c * κ
    end
    K = [Ka KM]                                        # a' = K Y
    G = [Lb; Ha]                                       # Y' = G a + DY Y
    DY = [Db QM; Hb HM]
    return K, G, DY
end
ρ(M) = maximum(abs, eigvals(M))
function stepmap(h, κ, kind)
    K, G, DY = operators(κ)
    nY = size(DY, 1); IY = Matrix(1.0I, nY, nY); Ia = Matrix(1.0I, n, n)
    if kind == :trap                                    # (I - h/2 D) Y⁺ = (I + h/2 D) Y + h (G a½)
        E = (IY - h / 2 * DY) \ (IY + h / 2 * DY); Φ = (IY - h / 2 * DY) \ (h * IY)
    else                                                # explicit: Y⁺ = Y + h (G a½ + D Y)
        E = IY + h * DY; Φ = h * IY
    end
    A = [Ia h/2*K; zeros(nY, n) IY]                     # a half step
    B = [Ia zeros(n, nY); Φ*G E]                        # Y step with a frozen at a½
    kind == :seuler && return [Ia h*K; zeros(nY, n) IY] * B
    return A * B * A
end
function limit(κ, kind)
    lo, hi = 0.0, 1.0
    for _ in 1:40
        mid = (lo + hi) / 2
        ρ(stepmap(mid, κ, kind)) <= 1 + 1e-6 ? (lo = mid) : (hi = mid)
    end
    lo
end
# generator check: eigenvalues of the continuous extended system (κ finite) must not grow
for ε in (1e-3, 1e-2, 0.1)
    K, G, DY = operators(1 / (2c * ε))
    @printf("ε = %.3f: max Re λ of the extended system %+.2e\n", ε, maximum(real, eigvals([zeros(n, n) K; G DY])))
end
@printf("P = %d, limit h c/Δx_e:\n", P)
for ε in (1e-4, 1e-3, 1e-2, 0.05, 0.1)
    κ = 1 / (2c * ε)
    @printf("ε = %.4f: Verlet, D_Y explicit %.3f   Verlet, D_Y trapezoidal %.3f   symplectic Euler, D_Y explicit %.3f\n",
            ε, limit(κ, :expl), limit(κ, :trap), limit(κ, :seuler))
end
# model error of the extended system against DGSEM-LMARS (ε = 0, M̂ eliminated): smooth wave, wavelength 4 elements,
# relative difference of a at T = 4 (two periods)
xn = vcat([(e - 1) .+ (x .+ 1) ./ 2 for e in 1:Ne]...)
a0 = sin.(2π .* xn ./ 4); b0 = copy(a0)                    # right-going wave
function lmars_gen()
    K, G, DY = operators(1.0)                               # κ irrelevant: eliminate M̂ = M̂*(a, b)
    Ka, KM = K[:, 1:n], K[:, n+1:end]
    Lb, Db, QM = G[1:n, :], DY[1:n, 1:n], DY[1:n, n+1:end]
    Ha, Hb, HM = G[n+1:end, :], DY[n+1:end, 1:n], DY[n+1:end, n+1:end]
    Ma, Mb = -(HM \ Ha), -(HM \ Hb)                          # M̂* = Ma a + Mb b
    [KM*Ma (Ka + KM * Mb); (Lb + QM * Ma) (Db + QM * Mb)]
end
T = 4.0
z0 = exp(T * lmars_gen()) * [a0; b0]
for ε in (1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5)
    K, G, DY = operators(1 / (2c * ε))
    # consistent initial M̂ = M̂*(a0, b0)
    Ha, Hb, HM = G[n+1:end, :], DY[n+1:end, 1:n], DY[n+1:end, n+1:end]
    M0 = -(HM \ (Ha * a0 + Hb * b0))
    z = exp(T * [zeros(n, n) K; G DY]) * [a0; b0; M0]
    @printf("ε = %.3f: model error |a - a_LMARS|/|a_LMARS| = %.2e, limits explicit %.3f, trapezoidal %.3f\n",
            ε, norm(z[1:n] .- z0[1:n]) / norm(z0[1:n]), limit(1 / (2c * ε), :expl), limit(1 / (2c * ε), :trap))
end
# order of the grouped Verlet (D_Y trapezoidal) against the exact flow of the extended system, T = 1
for ε in (0.05, 0.2)
    κ = 1 / (2c * ε)
    K, G, DY = operators(κ)
    Ha, Hb, HM = G[n+1:end, :], DY[n+1:end, 1:n], DY[n+1:end, n+1:end]
    zi = [a0; b0; -(HM \ (Ha * a0 + Hb * b0))]
    zr = exp(1.0 * [zeros(n, n) K; G DY]) * zi             # exact flow (reference)
    e = [norm(stepmap(1.0 / N, κ, :trap)^N * zi .- zr) / norm(zr) for N in (20, 40, 80)]
    @printf("ε = %.3f: grouped Verlet (trapezoidal), N = 20/40/80: %.2e %.2e %.2e (ratios %.1f %.1f)\n", ε, e..., e[1] / e[2], e[2] / e[3])
end
