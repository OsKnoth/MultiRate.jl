# Lawson (integrating-factor) RK step with the fast integrator of the MIS step as propagator

"""
    mis_step!(qnew, model, q, Δt, tab::LawsonTableau, sF, work; nsub = 12, fast = :imex)

One Lawson step (see `LawsonTableau`). The propagator e^{τL} (with the constant forcing of L) is
the fast integrator `fast` of the MIS step with no slow forcing, over max(1, round(nsub τ/Δt))
substeps, so the fast step is ≈ Δt/nsub as in MIS. φ_{cΔt}(qⁿ) is computed once per node and
shared by the stages. Uses the MIS work arrays: S holds S(Y_j), ΔY the propagated qⁿ.
"""
function mis_step!(qnew, model, q, Δt, tab::LawsonTableau, sF, w::MISWork; nsub::Int = 12, fast::Symbol = :imex)
    s = nstages(tab)
    nodes = tab.nodes
    cache = w.ΔY                                   # cache[k] = φ_{nodes[k] Δt}(qⁿ), k ≤ ncached
    cache[1] .= q
    ncached = 1
    function cached(k)
        while ncached < k
            _propagate!(cache[ncached+1], sF, cache[ncached], (nodes[ncached+1] - nodes[ncached]) * Δt, Δt, nsub, fast)
            ncached += 1
        end
        return cache[k]
    end
    function stage!(X, wts, c)
        c0 = lawson_start(tab, wts, c)
        k0 = findfirst(==(c0), nodes)
        X .= cached(k0)
        cur = c0
        for k in k0:length(nodes)
            ν = nodes[k]
            ν > c && break
            if ν > cur
                _propagate!(X, sF, X, (ν - cur) * Δt, Δt, nsub, fast)
                cur = ν
            end
            for j in 1:s
                wj = wts[j]
                (wj != 0 && tab.c[j] == ν) && (X .+= (Δt * wj) .* w.S[j])
            end
        end
        cur < c && _propagate!(X, sF, X, (c - cur) * Δt, Δt, nsub, fast)
        return X
    end
    for i in 1:s
        i == 1 ? (w.Y .= q) : stage!(w.Y, view(tab.A, i, :), tab.c[i])
        Si = w.S[i]
        rhs_nonlinear!(Si, model, w.Y)
        fast_apply!(w.L, sF, w.Y, fast_forcing(sF))       # L_n(Y_i)
        Si .-= w.L
    end
    stage!(w.Z, tab.b, 1.0)
    qnew .= w.Z
    return qnew
end

function _propagate!(out, sF, x, τ, Δt, nsub, fast)
    n = fast_substeps(τ / Δt, nsub)
    return fast_solve!(out, sF, x, nothing, τ / n, n; method = fast)
end
