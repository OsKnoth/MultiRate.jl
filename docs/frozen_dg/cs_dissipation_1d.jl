# 1D periodic linear acoustics a' = -c u_x, b' = -c a_x, DGSEM LGL degree P, element width 1, c = 1.
# Split: a' = -K b + Da a,  b' = L a + Db b  (K, L central flux, Da, Db LMARS/upwind penalties).
# (docs/frozen_dg, Section sec:implicitD.) Run: P=4 julia docs/frozen_dg/cs_dissipation_1d.jl
# Stability limit h c/Δx_e of: explicit Verlet (D explicit), Verlet with D = 0,
# Strang C_{h/2}(D) ∘ Verlet_h(K,L) ∘ C_{h/2}(D) (Cayley), and the general Störmer-Verlet (D implicit).
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
function ops(P, Ne; τ = 1.0)
    x, w, D = lgl(P); np = P + 1; n = np * Ne
    J = 0.5  # element width 1
    G = zeros(n, n)          # strong-form derivative with central flux: d/dx + lift({u}-u_L) terms
    Pen = zeros(n, n)        # jump penalty (upwind part), acts on the same variable
    idx(e, i) = (mod1(e, Ne) - 1) * np + i
    for e in 1:Ne
        for i in 1:np, j in 1:np
            G[idx(e, i), idx(e, j)] += D[i, j] / J
        end
        # right face of e: L = (e, np), R = (e+1, 1); n = +1
        for (k, s, kL, kR) in ((idx(e, np), 1.0, idx(e, np), idx(e + 1, 1)), (idx(e + 1, 1), -1.0, idx(e, np), idx(e + 1, 1)))
            lw = 1 / (J * w[k == kL ? np : 1])
            # central: flux* = {u}; surface term s*(u* - u_self) with sign for d/dx
            G[k, kL] += s * lw * 0.5; G[k, kR] += s * lw * 0.5; G[k, k] -= s * lw
            # upwind penalty: -(c/2)[[v]] in the flux of the other variable becomes a self term: -lw/2 * (v_self - v_other)
            other = k == kL ? kR : kL
            Pen[k, k] -= τ * lw * 0.5; Pen[k, other] += τ * lw * 0.5
        end
    end
    return G, Pen
end
P = parse(Int, get(ENV, "P", "4")); Ne = 16
G, Pen = ops(P, Ne)
n = size(G, 1); Z = zeros(n, n); Id = Matrix(1.0I, n, n)
K = G; L = -G; Da = Pen; Db = Pen
ρ(M) = maximum(abs, eigvals(M))
function Mexpl(h)   # explicit Verlet, D with the explicit half steps (code)
    A1 = [Id + h / 2 * Da -h / 2 * K; Z Id]
    B = [Id Z; h * L Id + h * Db]
    A2 = A1
    A2 * B * A1
end
Mfree(h) = [Id -h/2*K; Z Id] * [Id Z; h*L Id] * [Id -h/2*K; Z Id]
cay(h, D) = (Id - h / 2 * D) \ (Id + h / 2 * D)
function Mstrang(h)
    C = [cay(h / 2, Da) Z; Z cay(h / 2, Db)]
    C * Mfree(h) * C
end
function Mgsv(h)    # general Störmer-Verlet: a½ implicit in Da, b trapezoidal in Db, a explicit
    A1 = [inv(Id - h / 2 * Da) inv(Id - h / 2 * Da) * (-h / 2 * K); Z Id]
    B = [Id Z; (Id - h / 2 * Db) \ (h * L) (Id - h / 2 * Db) \ (Id + h / 2 * Db)]
    A2 = [Id + h / 2 * Da -h / 2 * K; Z Id]
    A2 * B * A1
end
function limit(M)
    lo, hi = 0.0, 2.0
    for _ in 1:50
        mid = (lo + hi) / 2
        ρ(M(mid)) <= 1 + 1e-9 ? (lo = mid) : (hi = mid)
    end
    lo
end
@printf("P = %d: max|λ| central %.2f, upwind %.2f (element width 1, c = 1)\n", P, ρ([Z K; -L Z]), ρ([Da -K; L Db]))
@printf("RK-free limits h c/Δx_e:  explicit Verlet with D %.3f   Verlet D=0 %.3f   Strang Cayley(D) %.3f   general SV %.3f\n",
        limit(Mexpl), limit(Mfree), limit(Mstrang), limit(Mgsv))
@printf("damping of Strang at 0.9·limit: %.3f   general SV: %.3f\n", ρ(Mstrang(0.9limit(Mfree))), ρ(Mgsv(0.9limit(Mfree))))
function Mcode(h)   # as the code: b row midpoint predictor for Db (correction 2)
    A1 = [Id + h / 2 * Da -h / 2 * K; Z Id]
    Bm = Id + h * Db + h^2 / 2 * Db^2              # b⁺ = b + h(L a½ + Db(b + h/2(L a½ + Db b)))
    B = [Id Z; h * (Id + h / 2 * Db) * L Bm]
    A1 * B * A1
end
@printf("code (midpoint predictor in the b row): %.3f\n", limit(Mcode))
# derivative with a one-sided face value: flux value = wL*u_L + wR*u_R
function Gflux(P, Ne, wL)
    x, w, D = lgl(P); np = P + 1; n = np * Ne; J = 0.5
    G = zeros(n, n); idx(e, i) = (mod1(e, Ne) - 1) * np + i
    for e in 1:Ne
        for i in 1:np, j in 1:np; G[idx(e, i), idx(e, j)] += D[i, j] / J; end
        kL, kR = idx(e, np), idx(e + 1, 1)
        for (k, s) in ((kL, 1.0), (kR, -1.0))
            lw = 1 / (J * w[k == kL ? np : 1])
            G[k, kL] += s * lw * wL; G[k, kR] += s * lw * (1 - wL); G[k, k] -= s * lw
        end
    end
    G
end
Ga, Gb = Gflux(P, Ne, 0.0), Gflux(P, Ne, 1.0)   # alternating: a-row uses b_R, b-row uses a_L
Malt(h) = [Id -h/2*Ga; Z Id] * [Id Z; -h*Gb Id] * [Id -h/2*Ga; Z Id]
# Bokhove partition: half steps of the flow (a' = -K b, b' = Db b), full step of (a' = Da a, b' = L a), exact
Mbok(h) = exp(h / 2 * [Z -K; Z Db]) * exp(h * [Da Z; L Z]) * exp(h / 2 * [Z -K; Z Db])
# same with Cayley (trapezoidal) sub-flows: local in practice
cayM(h, A) = (I - h / 2 * A) \ (I + h / 2 * A)
MbokC(h) = cayM(h / 2, [Z -K; Z Db]) * cayM(h, [Da Z; L Z]) * cayM(h / 2, [Z -K; Z Db])
@printf("P=%d alternating flux, D=0: %.3f   Bokhove partition exact: %.3f   Bokhove partition trapezoidal: %.3f\n",
        P, limit(Malt), limit(Mbok), limit(MbokC))
