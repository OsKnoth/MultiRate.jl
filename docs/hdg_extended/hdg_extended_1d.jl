# Continuous-time stability of the extended (augmented) HDG systems, 1D periodic acoustics, DGSEM degree P,
# c = 1, element width 1 (docs/hdg_extended). Form I: ε M̂' = M̂*(a, b) - M̂ (full face momentum relaxed, as
# HDGFast2D with eps_if > 0); form II: M̂ = {b} + Ŝ, ε Ŝ' = -[a]/2 - Ŝ (only the dissipative part relaxed);
# form II': form II plus a face pressure deviation T̂, ε T̂' = -[b]/2 - T̂. Prints max Re λ of the generator.
#     P=4 julia docs/hdg_extended/hdg_extended_1d.jl
# Operators as in docs/frozen_dg/hdg_trace_1d.jl:
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

I_f = Matrix(1.0I, Ne, Ne)
# form I: U = (a, b, M̂)
formI(ε) = [Z Ka Lift; Lb Db Zf; Ms_a ./ ε Ms_b ./ ε -I_f ./ ε]
# form II: U = (a, b, Ŝ), M̂ = Avg b + Ŝ
formII(ε) = [Z (Ka + Lift * Avg) Lift; Lb Db Zf; Ms_a ./ ε zeros(Ne, n) -I_f ./ ε]
# form II': U = (a, b, Ŝ, T̂), P̂ = {a} + T̂ (b row: Lb a + Lift T̂ instead of Lb a + Db b)
formIIp(ε) = [Z (Ka + Lift * Avg) Lift Zf; Lb Z Zf Lift; Ms_a ./ ε zeros(Ne, n) -I_f ./ ε zeros(Ne, Ne);
              zeros(Ne, n) (-0.5 .* Jmp) ./ ε zeros(Ne, Ne) -I_f ./ ε]
maxre(A) = maximum(real, eigvals(A))
@printf("P = %d, element width 1, c = 1: max Re λ of the generator\n", P)
for ε in (1e-3, 1e-2, 0.05, 0.2)
    @printf("ε = %.3f:  form I %+.3e   form II %+.3e   form II' %+.3e\n", ε, maxre(formI(ε)), maxre(formII(ε)), maxre(formIIp(ε)))
end
