# Rooted trees with bicoloured non-root vertices (the colour of a vertex = the argument of the F at
# its parent that is differentiated: black = frozen y, white = integrated z; the root colour plays
# no role). A tree is the multiset of its coloured subtrees; trees are stored in a list ordered by
# the number of vertices, a subtree is referred to by its index in that list.

"""Tree: `children` = (white, index) pairs (sorted), `n` vertices, density `γ`."""
struct Tree
    children::Vector{Tuple{Bool,Int}}
    n::Int
    γ::Int
end

"""Admissible child multisets per vertex for the problem classes: `:G` general F(y, z), `:B` affine
in z (≤ 1 white child), `:BA` affine plus additive f(y) + B(y) z + g(z) (≤ 1 white child or only white children),
`:A` additive f(y) + g(z) (children of one colour), `:L` f(y) + L z (only black children or exactly one white child)."""
function admissible(ch, class::Symbol)
    nw = count(first, ch)
    class == :G && return true
    class == :B && return nw <= 1
    class == :BA && return nw <= 1 || nw == length(ch)
    class == :A && return nw == 0 || nw == length(ch)
    class == :L && return nw == 0 || (nw == 1 && length(ch) == 1)
    throw(ArgumentError("class must be :G, :B, :BA, :A or :L"))
end

"""All trees with at most `p` vertices of the class (`:G`, `:B`, `:BA`, `:A`, `:L`), ordered by size."""
function trees(p::Int; class::Symbol = :G)
    ts = Tree[Tree(Tuple{Bool,Int}[], 1, 1)]
    for n in 2:p
        labels = [(w, k) for k in eachindex(ts) for w in (false, true)]     # coloured subtrees
        new = Tree[]
        function rec(ch, start, rest)
            if rest == 0
                admissible(ch, class) || return
                push!(new, Tree(copy(ch), n, n * prod(ts[k].γ for (_, k) in ch)))
                return
            end
            for l in start:length(labels)
                (w, k) = labels[l]
                ts[k].n <= rest || continue
                push!(ch, (w, k)); rec(ch, l, rest - ts[k].n); pop!(ch)
            end
        end
        rec(Tuple{Bool,Int}[], 1, n - 1)
        append!(ts, new)
    end
    return ts
end

"""Number of trees per order 1 … p (Table 1 of the note: :G 1 2 7 26 107, :B 1 2 6 20 73,
:A 1 2 6 18 60, :L 1 2 5 13 37)."""
tree_count(p::Int; class::Symbol = :G) = [count(t -> t.n == k, trees(p; class)) for k in 1:p]

"""Bracket notation of tree k of the list: b = black, w = white child, e.g. •[b[w]]."""
function treestring(ts::Vector{Tree}, k::Int)
    str(k) = isempty(ts[k].children) ? "" :
        "[" * join((w ? "w" : "b") * str(c) for (w, c) in ts[k].children) * "]"
    return "•" * str(k)
end
