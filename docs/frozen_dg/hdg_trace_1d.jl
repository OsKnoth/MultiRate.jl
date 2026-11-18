# 1D periodic acoustics, DGSEM degree P, c = 1, element width 1, in the HDG form of HDGFast2D:
#   a' = -Dvol b - s lw (M̂ - b_self),          M̂ interface momentum (one per face),
#   b' = -Dvol a - s lw (P̂ - a_self),          P̂ = {a} - [b]/2   (D_b: dissipation in the b row only),
#   M̂* = {b} - [a]/2 (LMARS value); M̂ = M̂* (ε = 0) or relaxed, M̂ ← (ε M̂ + dt M̂*)/(ε + dt).
# Stability limit h/Δx_e of Verlet- and symplectic-Euler-type steps, with D_b explicit (as in the code) or
# integrated exactly (Bokhove: b' = D_b b inside the flow, local per face).
#     P=4 julia docs/frozen_dg/hdg_trace_1d.jl
using LinearAlgebra, Printf
function lgl(P)
    x = -cos.(π * (0:P) / P)
    for _ in 1:100
        L0 = ones(P + 1); L1 = copy(x)
        for k in 2:P
            L2 = ((2k - 1) .* x .* L1 .- (k - 1) .* L0) ./ k; L0, L1 = L1, L2
        end
        # L1 = P_P, L0 = P_{P-1}
        dx = (x .* L1 .- L0) ./ ((P + 1) .* L1)
        x .-= dx
        maximum(abs, dx) < 1e-15 && break
    end
    L0 = ones(P + 1); L1 = copy(x)
    for k in 2:P
        L2 = ((2k - 1) .* x .* L1 .- (k - 1) .* L0) ./ k; L0, L1 = L1, L2
    end
    w = 2 ./ (P * (P + 1) .* L1 .^ 2)
    D = zeros(P + 1, P + 1)
    for i in 1:P+1, j in 1:P+1
        D[i, j] = i != j ? L1[i] / (L1[j] * (x[i] - x[j])) : (i == 1 ? -P * (P + 1) / 4 : i == P + 1 ? P * (P + 1) / 4 : 0.0)
    end
    return x, w, D
end
P = parse(Int, get(ENV, "P", "4")); Ne = 16
x, w, D = lgl(P); np = P + 1; n = np * Ne; J = 0.5
idx(e, i) = (mod1(e, Ne) - 1) * np + i
Dv = zeros(n, n)
for e in 1:Ne, i in 1:np, j in 1:np
    Dv[idx(e, i), idx(e, j)] = D[i, j] / J
end
# face f = right face of element f: L = idx(f, np), R = idx(f+1, 1)
Lift = zeros(n, Ne)            # a' += -s lw (M̂_f - b_self): the M̂ part
Self = zeros(n, n)             # the + s lw b_self part (same structure for a in the b row)
Avg = zeros(Ne, n); Jmp = zeros(Ne, n)
for f in 1:Ne
    kL, kR = idx(f, np), idx(f + 1, 1)
    for (k, s) in ((kL, 1.0), (kR, -1.0))
        lw = 1 / (J * w[k == kL ? np : 1])
        Lift[k, f] -= s * lw; Self[k, k] += s * lw
    end
    Avg[f, kL] = Avg[f, kR] = 0.5; Jmp[f, kR] = 1.0; Jmp[f, kL] = -1.0
end
Z = zeros(n, n); Zf = zeros(n, Ne); Id = Matrix(1.0I, n, n)
Ms_b, Ms_a = Avg, -0.5 .* Jmp                       # M̂* = Ms_b b + Ms_a a
Ka = -Dv + Self                                     # a' = Ka b + Lift M̂
Lb = -Dv + Self + Lift * Avg                        # b' = Lb a + Db b   (central P̂ part)
Db = Lift * (-0.5 .* Jmp)                           # D_b (only in the b row)
N = 2n + Ne
ia, ib, im = 1:n, n+1:2n, 2n+1:N
function op(blocks)                                 # assemble an N×N map from row/col blocks
    M = Matrix(1.0I, N, N)
    for (r, c, B) in blocks; M[r, c] .= B; end
    M
end
ρ(M) = maximum(abs, eigvals(M))
# elementary maps
Mhat(dt, ε) = (r = ε / (ε + dt); op([(im, ia, (1 - r) * Ms_a), (im, ib, (1 - r) * Ms_b), (im, im, r * Matrix(1.0I, Ne, Ne))]))
ahalf(c) = op([(ia, ib, c * Ka), (ia, im, c * Lift)])                      # a += c (Ka b + Lift M̂)
bexpl(c) = op([(ib, ia, c * Lb), (ib, ib, Id + c * Db)])                    # b += c (Lb a + Db b)
function bexact(c)                                                          # b ← e^{cDb} b + c φ1(cDb) Lb a
    X = exp([c*Db c*Matrix(1.0I, n, n); zeros(n, 2n)])
    op([(ib, ib, X[1:n, 1:n]), (ib, ia, X[1:n, n+1:2n] * Lb)])
end
function aflow(c)                                                           # Bokhove A: a' = Ka b + Lift M̂*(a, b), b' = Db b
    X = exp([c*Db c*Matrix(1.0I, n, n); zeros(n, 2n)])                      # (M̂ = M̂* evaluated with the mean b)
    E, Φ = X[1:n, 1:n], X[1:n, n+1:2n]
    op([(ia, ib, (Ka + Lift * Ms_b) * Φ), (ia, ia, Id + c * Lift * Ms_a), (ib, ib, E), (im, ia, Ms_a), (im, ib, Ms_b), (im, im, zeros(Ne, Ne))])
end
bcentral(c) = op([(ib, ia, c * Lb)])
# steps (ε relative to h)
verlet(h, κ)   = ahalf(h/2) * Mhat(h/2, κ*h) * bexpl(h) * ahalf(h/2) * Mhat(h/2, κ*h)
verletB(h, κ)  = aflow(h/2) * bcentral(h) * aflow(h/2)
seuler(h, κ)   = ahalf(h) * Mhat(h, κ*h) * bexpl(h)
seulerB(h, κ)  = ahalf(h) * Mhat(h, κ*h) * bexact(h)
function limit(S, κ)
    lo, hi = 0.0, 1.0
    for _ in 1:40
        mid = (lo + hi) / 2
        ρ(S(mid, κ)) <= 1 + 1e-6 ? (lo = mid) : (hi = mid)
    end
    lo
end
@printf("P = %d, limit h c/Δx_e (ε = κ h):\n", P)
for κ in (0.0, 0.25, 0.5, 1.0, 2.0)
    @printf("κ = %.1f:  Verlet %.3f   symplectic Euler %.3f   symplectic Euler + exact D_b %.3f\n",
            κ, limit(verlet, κ), limit(seuler, κ), limit(seulerB, κ))
end
@printf("Verlet with Bokhove's flows (M̂ = M̂*, D_b exact): %.3f\n", limit(verletB, 0.0))
# (tolerance 1e-6: with relaxation the eigenvalue 1 of the constant mode is defective, roundoff ~1e-8)
