# MIS methods with polynomial coupling: coefficients, base method, stage times

"""
    PolyMIS(α, p, g)

s-stage MIS method with polynomial coupling. Stages are numbered as in the paper, 1 … s+1 (stage 1
is yⁿ, stage s+1 is yⁿ⁺¹). `α` is (s+1)×(s+1) strictly lower triangular; `p[i, j]`, `g[i, j]`
(j < i) are the coefficient vectors of p_ij(τ) = Σ_k p[i, j][k+1] τᵏ and g_ij(τ). Coefficients of
any number type (Rational for exact order conditions, dual numbers for gradients).
"""
struct PolyMIS{T}
    α::Matrix{T}
    p::Matrix{Vector{T}}
    g::Matrix{Vector{T}}
end
nstages(m::PolyMIS) = size(m.α, 1) - 1
coupling_degree(m::PolyMIS) = maximum(length(v) for v in vcat(vec(m.p), vec(m.g))) - 1

function PolyMIS(α::AbstractMatrix, p::AbstractMatrix, g::AbstractMatrix)
    T = promote_type(eltype(α), mapreduce(eltype, promote_type, p), mapreduce(eltype, promote_type, g))
    n = size(α, 1)
    size(α) == size(p) == size(g) == (n, n) || throw(ArgumentError("α, p, g must be (s+1)×(s+1)"))
    P = [Vector{T}(j < i ? p[i, j] : T[]) for i in 1:n, j in 1:n]
    G = [Vector{T}(j < i ? g[i, j] : T[]) for i in 1:n, j in 1:n]
    return PolyMIS{T}(Matrix{T}([j < i ? α[i, j] : zero(T) for i in 1:n, j in 1:n]), P, G)
end

"""Classical MIS (constant coupling) from a `MISTableau` (α, β, γ in the ClimaTimeSteppers /
Knoth–Wensch convention: row i is paper stage i+1, α, γ column j multiplies Y_{j+1} - yⁿ, β column
j multiplies the slow tendency of Y_j)."""
function PolyMIS(t::MISTableau; T::Type = Float64)
    s = length(t.d)
    α = zeros(T, s + 1, s + 1)
    p = fill(T[], s + 1, s + 1)
    g = fill(T[], s + 1, s + 1)
    for i in 2:s+1, j in 1:i-1
        p[i, j] = [T(t.β[i-1, j])]
        g[i, j] = [j == 1 ? zero(T) : T(t.γ[i-1, j-1])]
        α[i, j] = j == 1 ? zero(T) : T(t.α[i-1, j-1])
    end
    return PolyMIS(α, p, g)
end

"""ETD2RK (Cox & Matthews) as MIS method: p₂₁ = 1, p₃₁ = 1 - τ, p₃₂ = τ (slow tendency linearly
interpolated in time between the two stages)."""
function etd2rk(T::Type = Rational{Int})
    o, z = one(T), zero(T)
    α = zeros(T, 3, 3)
    p = fill(T[], 3, 3); g = fill(T[], 3, 3)
    p[2, 1] = [o]; p[3, 1] = [o, -o]; p[3, 2] = [z, o]
    g[2, 1] = [z]; g[3, 1] = [z]; g[3, 2] = [z]
    return PolyMIS(α, p, g)
end

# polynomials as coefficient vectors (constant term first)
pint1(a) = isempty(a) ? zero(eltype(a)) : sum(a[k] / k for k in eachindex(a))       # ∫₀¹
function pantider(a)                                                                # ∫₀^τ
    T = eltype(a)
    r = Vector{T}(undef, length(a) + 1); r[1] = zero(T)
    for k in eachindex(a); r[k+1] = a[k] / k; end
    return r
end
function padd(a, b)
    T = promote_type(eltype(a), eltype(b))
    r = zeros(T, max(length(a), length(b)))
    for k in eachindex(a); r[k] += a[k]; end
    for k in eachindex(b); r[k] += b[k]; end
    return r
end
function pmul(a, b)
    T = promote_type(eltype(a), eltype(b))
    (isempty(a) || isempty(b)) && return T[]
    r = zeros(T, length(a) + length(b) - 1)
    for i in eachindex(a), j in eachindex(b); r[i+j-1] += a[i] * b[j]; end
    return r
end
pscale(a, c) = a .* c
peval(a, τ) = isempty(a) ? zero(eltype(a)) : evalpoly(τ, a)

"""Fast speed q_i(τ) = Σ_j p_ij(τ) of stage i (coefficient vector)."""
fast_speed(m::PolyMIS, i) = reduce(padd, (m.p[i, j] for j in 1:i-1); init = eltype(m.α)[])

"""true if every q_i is constant (uniform fast speed, eq. (3.4))."""
uniform_fast_speed(m::PolyMIS; tol = 1e-13) =
    all(all(abs(c) <= tol for c in fast_speed(m, i)[2:end]) for i in 2:nstages(m)+1)

"""Stage times c₁ … c_{s+1} (method applied to y' = 1)."""
function stage_times(m::PolyMIS)
    n = size(m.α, 1); T = eltype(m.α)
    c = zeros(T, n)
    for i in 2:n
        c[i] = sum((m.α[i, j] + pint1(m.g[i, j])) * c[j] + pint1(m.p[i, j]) for j in 1:i-1)
    end
    return c
end

"""Base method (A, b) of eq. (3.6): the explicit RK method the scheme reduces to if F does not
depend on its second argument."""
function base_method(m::PolyMIS)
    n = size(m.α, 1); T = eltype(m.α)
    Ab = [j < i ? m.α[i, j] + pint1(m.g[i, j]) : zero(T) for i in 1:n, j in 1:n]
    Pb = [j < i ? pint1(m.p[i, j]) : zero(T) for i in 1:n, j in 1:n]
    M = (I - Ab) \ Pb
    return M[1:n-1, 1:n-1], M[n, 1:n-1]
end

"""
    commutator_free(stages; T = Rational{Int})

Commutator-free Lie group method (Celledoni, Marthinsen & Owren 2003) as MIS method. `stages[i]` is the list of
coefficient vectors β⁽¹⁾, …, β⁽ᴶ⁾ of the true stage i+1 (the last entry is the output y_{n+1}):
    Y_{i+1} = exp(h Σ_j β⁽ᴶ⁾_j F_j) ∘ … ∘ exp(h Σ_j β⁽¹⁾_j F_j) yⁿ,   F_j = F(Y_j, ·) the frozen vector field.
Each exponential is the flow of a frozen field over unit time, i.e. one MIS stage with constant coupling that starts
from the previous one (restart class, g = 0); the intermediate stages are not used as frozen arguments.
"""
function commutator_free(stages::AbstractVector; T::Type = Rational{Int})
    ntrue = length(stages) + 1                       # true stages incl. Y₁ = yⁿ
    nmis = 1 + sum(length, stages)
    α = zeros(T, nmis, nmis); p = fill(T[], nmis, nmis); g = fill(T[], nmis, nmis)
    truemis = [1]                                    # MIS index of true stage k
    cur = 1
    for (i, exps) in enumerate(stages)
        prev = 1                                     # every true stage starts from yⁿ
        for β in exps
            cur += 1
            for j in 1:cur-1; p[cur, j] = [zero(T)]; g[cur, j] = [zero(T)]; end
            prev > 1 && (α[cur, prev] = one(T))
            for (k, b) in enumerate(β); p[cur, truemis[k]] = [T(b)]; end
            prev = cur
        end
        push!(truemis, cur)
    end
    return PolyMIS(α, p, g)
end

"""CF4 of Celledoni, Marthinsen & Owren (2003): four stages, five exponentials, order 4 as a Lie group method."""
cf4(T::Type = Rational{Int}) = commutator_free([
    [[1//2]],
    [[0, 1//2]],
    [[1//2], [-1//2, 0, 1]],                                   # Y₄ = exp(F₃ - ½F₁) exp(½F₁) yⁿ
    [[3//12, 2//12, 2//12, -1//12], [-1//12, 2//12, 2//12, 3//12]]]; T)

"""
    lowstorage_cf(A, B; T = Float64)

Low-storage commutator-free method of Bazavov (BIT 2022) from a Williamson 2N-storage Runge–Kutta method
ΔY_i = A_i ΔY_{i-1} + h F(Y_{i-1}), Y_i = Y_{i-1} + B_i ΔY_i: Z_i = A_i Z_{i-1} + h F(Y_{i-1}), Y_i = exp(B_i Z_i) Y_{i-1}.
In MIS form a telescopic method with constant coupling p_{i+1,j} = B_i Π_{k=j+1}^{i} A_k.
"""
function lowstorage_cf(A::AbstractVector, B::AbstractVector; T::Type = Float64)
    s = length(B)
    α = zeros(T, s + 1, s + 1); p = fill(T[], s + 1, s + 1); g = fill(T[], s + 1, s + 1)
    for i in 1:s
        i > 1 && (α[i+1, i] = one(T))
        for j in 1:i
            p[i+1, j] = [T(B[i] * prod(A[k] for k in j+1:i; init = one(T)))]
            g[i+1, j] = [zero(T)]
        end
    end
    return PolyMIS(α, p, g)
end

"""Lüscher's RK3 for the Wilson flow (JHEP 2010): the 2N-storage RK3 of Williamson, A = (0, -17/32, -32/27),
B = (1/4, 8/9, 3/4), in the format of `lowstorage_cf`."""
luscher3(T::Type = Rational{Int}) = lowstorage_cf(T[0, -17//32, -32//27], T[1//4, 8//9, 3//4]; T)

"""Carpenter & Kennedy (1994) 2N-storage RK(5,4), solution 3, in the format of `lowstorage_cf`."""
ck54_cf() = lowstorage_cf(
    [0.0, -567301805773 / 1357537059087, -2404267990393 / 2016746695238, -3550918686646 / 2091501179385,
     -1275806237668 / 842570457699],
    [1432997174477 / 9575080441755, 5161836677717 / 13612068292357, 1720146321549 / 2090206949498,
     3134564353537 / 4481467310338, 2277821191437 / 14882151754819])
