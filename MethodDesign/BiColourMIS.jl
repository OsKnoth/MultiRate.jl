"""
    BiColourMIS

Order conditions of split-explicit methods with ODE stages for y' = F(y, y), after Knoth & Wensch,
"B-series and split-explicit methods" (github.com/OsKnoth/B-Series, PDF/DocuMISBI.tex):

    Y_1 = yⁿ,
    Z_i(0) = yⁿ + Σ_{j<i} α_ij (Y_{j+1} - yⁿ),
    dZ_i/dλ = h Σ_{j≤i} a_ij(λ) F(Y_j, Z_i) + Σ_{j<i} γ_ij (Y_{j+1} - yⁿ),   λ ∈ [0, 1],
    Y_{i+1} = Z_i(1),   i = 1 … s,   yⁿ⁺¹ = Y_{s+1},

a_ij(λ) = Σ_p a[i, j, p+1] λ^p. MIS (Knoth & Wensch 2014, ClimaTimeSteppers convention of
`MultiRate.TimeStepping.MISTableau`): a_ij = β_ij constant (the stage runs over d_i Δt with
weights β_ij/d_i, d_i = Σ_j β_ij). The first argument of F is frozen at the stage values, the
second one evolves (`TimeStepping.run_frozen`).

Bicoloured trees: rooted trees whose edges are coloured: an edge to a black child is a derivative
of F with respect to the first (frozen) argument, to a white child with respect to the second
(evolving) one. With B-series Y_j = B(η_j), Z_i(λ) = B(ζ_i(λ)) the stage ODE gives the recursion
of DocuMISBI (algorithm with shifted initial values), for t = [t_b1 … t_bk, t_w1 … t_wl]:

    ζ_i(t)(λ) = Σ_j α_ij η_{j+1}(t) + λ Σ_j γ_ij η_{j+1}(t)
                + ∫_0^λ Σ_j a_ij(μ) Π_m η_j(t_bm) Π_m ζ_i(t_wm)(μ) dμ,
    η_{i+1}(t) = ζ_i(t)(1),   η_1(t) = 0,

and the order conditions η_{s+1}(t) = 1/γ(t) (γ the density of the uncoloured tree) for all
bicoloured trees with at most p vertices. For an additive F(y, z) = f(y) + g(z) only trees whose
vertices have children of one colour enter (`additive`).

The arithmetic is generic: Float64, Rational{BigInt} (exact residuals) or any number type with
+, *, / (e.g. symbolic variables, see `README.md`).
"""
module BiColourMIS

export BTree, trees, order, density, additive, linear_z, linear_y, in_class, notation, Method, mis_method, eta, residuals, order_of

# ------------------------------------------------------------------ bicoloured trees

"""Rooted tree with coloured edges: children (white::Bool, subtree), sorted canonically."""
struct BTree
    children::Vector{Tuple{Bool,BTree}}
    key::String
end
function BTree(children::Vector{Tuple{Bool,BTree}})
    ch = sort(children; by = c -> (c[1], c[2].key))
    return BTree(ch, "{" * join([(w ? "w" : "b") * c.key for (w, c) in ch], ",") * "}")
end
const LEAF = BTree(Tuple{Bool,BTree}[])
Base.:(==)(a::BTree, b::BTree) = a.key == b.key
Base.hash(t::BTree, h::UInt) = hash(t.key, h)

order(t::BTree) = 1 + sum((order(c) for (_, c) in t.children); init = 0)
density(t::BTree) = order(t) * prod((density(c) for (_, c) in t.children); init = 1)
"""Each vertex has children of one colour only (the trees of an additive F = f(y) + g(z))."""
additive(t::BTree) = (length(unique(w for (w, _) in t.children)) <= 1) && all(additive(c) for (_, c) in t.children)
"""Each vertex has at most one white child: F affine in the evolving argument, F(y, z) = S(y) + A(y) z
(F_(k,l) = 0 for l ≥ 2), as the frozen FV and DG splits with the product-form pressure."""
linear_z(t::BTree) = count(w for (w, _) in t.children) <= 1 && all(linear_z(c) for (_, c) in t.children)
"""Each vertex has at most one black child: F affine in the frozen argument."""
linear_y(t::BTree) = count(!w for (w, _) in t.children) <= 1 && all(linear_y(c) for (_, c) in t.children)
"""Tree classes: :general (any F), :additive (F = f(y) + g(z)), :linz (F = S(y) + A(y) z),
:linz_additive (F = f(y) + A z), :liny (affine in y), :bilinear (affine in both)."""
in_class(t::BTree, class::Symbol) =
    class === :general ? true : class === :additive ? additive(t) : class === :linz ? linear_z(t) :
    class === :linz_additive ? (linear_z(t) && additive(t)) : class === :liny ? linear_y(t) :
    class === :bilinear ? (linear_z(t) && linear_y(t)) : error("unknown tree class $class")

"""Notation of DocuMISBI: black leaf `.`, white leaf `o`, black subtree `[…]`, white subtree `(…)`;
the root as `[…]` (or `.`)."""
function notation(t::BTree, white::Bool = false)
    isempty(t.children) && return white ? "o" : "."
    inner = join([notation(c, w) for (w, c) in t.children], ",")
    return white ? "(" * inner * ")" : "[" * inner * "]"
end
Base.show(io::IO, t::BTree) = print(io, notation(t))

const TREES = Dict{Int,Vector{BTree}}(1 => [LEAF])

"""All bicoloured trees with n vertices."""
function trees(n::Int)
    haskey(TREES, n) && return TREES[n]
    types = [(w, c) for k in 1:n-1 for c in trees(k) for w in (false, true)]
    out = BTree[]
    function rec(rem, start, acc)
        rem == 0 && (push!(out, BTree(copy(acc))); return)
        for idx in start:length(types)
            w, c = types[idx]
            order(c) <= rem || continue
            push!(acc, (w, c))
            rec(rem - order(c), idx, acc)
            pop!(acc)
        end
    end
    rec(n - 1, 1, Tuple{Bool,BTree}[])
    TREES[n] = unique(out)
    return TREES[n]
end

# ------------------------------------------------------------------ methods

"""Method with s ODE stages: α, γ (s × s, strictly lower), a (s × s × P, a[i, j, p+1] the
coefficient of λ^p, j ≤ i)."""
struct Method{T}
    name::String
    α::Matrix{T}
    γ::Matrix{T}
    a::Array{T,3}
end
nstages(m::Method) = size(m.α, 1)

"""MIS in the ClimaTimeSteppers convention (row i = stage i, α/γ column j ↔ Y_{j+1}, β column j ↔ Y_j)."""
mis_method(name, α, β, γ) = Method(string(name), Matrix(α), Matrix(γ), reshape(Matrix(β), size(β)..., 1))

# polynomials in λ: coefficient vectors, p[1] constant term
_mul(p, q) = (r = fill(zero(promote_type(eltype(p), eltype(q))), length(p) + length(q) - 1);
              for i in eachindex(p), j in eachindex(q); r[i+j-1] += p[i] * q[j]; end; r)
_add(p, q) = (n = max(length(p), length(q)); T = promote_type(eltype(p), eltype(q));
              [(i <= length(p) ? p[i] : zero(T)) + (i <= length(q) ? q[i] : zero(T)) for i in 1:n])
_int(p) = vcat(zero(eltype(p)), [p[k] / k for k in eachindex(p)])          # ∫_0^λ
_eval1(p) = sum(p)                                                            # p(1)
_nz(x) = !isequal(x, 0)                  # structural nonzero (also for symbolic coefficients)

"""
    eta(m, t) -> η_{s+1}(t)

B-series coefficient of the step for the tree t (memoized per method in `cache`).
"""
function eta(m::Method{T}, t::BTree, cache = Dict{Tuple{Int,String},Any}()) where {T}
    return _eta(m, nstages(m) + 1, t, cache)
end

function _eta(m::Method{T}, i::Int, t::BTree, cache) where {T}
    i == 1 && return zero(T)
    return _eval1(_zeta(m, i - 1, t, cache))
end

function _zeta(m::Method{T}, i::Int, t::BTree, cache) where {T}
    k = (i, t.key)
    haskey(cache, k) && return cache[k]
    c0 = zero(T)
    c1 = zero(T)
    for j in 1:i-1
        if _nz(m.α[i, j]) || _nz(m.γ[i, j])
            e = _eta(m, j + 1, t, cache)
            c0 += m.α[i, j] * e
            c1 += m.γ[i, j] * e
        end
    end
    integrand = [zero(T)]
    for j in 1:i
        aj = m.a[i, j, :]
        any(_nz, aj) || continue
        p = Vector{T}(aj)
        for (w, c) in t.children
            p = w ? _mul(p, _zeta(m, i, c, cache)) : p .* _eta(m, j, c, cache)
        end
        integrand = _add(integrand, p)
    end
    z = _add([c0, c1], _int(integrand))
    cache[k] = z
    return z
end

"""Residuals η_{s+1}(t) - 1/γ(t) for all trees with ≤ p vertices of the class (`in_class`;
additive_only = true is class = :additive), as a vector of (tree, residual)."""
function residuals(m::Method{T}, p::Int; additive_only::Bool = false, class::Symbol = additive_only ? :additive : :general) where {T}
    cache = Dict{Tuple{Int,String},Any}()
    out = Tuple{BTree,Any}[]
    for n in 1:p, t in trees(n)
        in_class(t, class) || continue
        push!(out, (t, eta(m, t, cache) - one(T) / density(t)))
    end
    return out
end

"""Largest p ≤ pmax with all residuals of order ≤ p below tol."""
function order_of(m::Method, pmax::Int = 5; additive_only::Bool = false, class::Symbol = additive_only ? :additive : :general, tol = 1e-10)
    r = residuals(m, pmax; class)
    p = 0
    for n in 1:pmax
        all(abs(Float64(x)) < tol for (t, x) in r if order(t) == n) || break
        p = n
    end
    return p
end

end # module
