# Exponential RK (ETD) step in stage-ODE form with the fast integrators of the MIS step

"""
    mis_step!(qnew, model, q, Δt, tab::ETDTableau, sF, work; nsub = 12, fast = :imex)

One ETD step (see `ETDTableau`): stage i starts from qⁿ and integrates the fast system over
c_i Δt with n_i = max(1, round(c_i nsub)) substeps and the polynomial slow forcing
Σ_j Σ_k P[i,j,k] τ^{k-1} S(Y_j) (`fast_solve_poly!`). Uses the MIS work arrays: S holds S(Y_j),
ΔY[1:K] the forcing coefficients G_k.
"""
function mis_step!(qnew, model, q, Δt, tab::ETDTableau, sF, w::MISWork; nsub::Int = 12, fast::Symbol = :imex)
    s, K = nstages(tab), size(tab.P, 3)
    G = view(w.ΔY, 1:K)
    w.Y .= q
    for i in 1:s
        Si = w.S[i]
        rhs_nonlinear!(Si, model, w.Y)
        fast_apply!(w.L, sF, w.Y, fast_forcing(sF))
        Si .-= w.L                                        # S(Y_i) = N(Y_i) - L_n(Y_i)
        for k in 1:K
            Gk = G[k]
            Gk .= 0
            for j in 1:i
                p = tab.P[i, j, k]
                p != 0 && (Gk .+= p .* w.S[j])
            end
        end
        T = tab.c[i] * Δt
        n = fast_substeps(tab.c[i], nsub)
        fast_solve_poly!(w.Y, sF, q, G, T, n; method = fast)    # Y_{i+1}
    end
    qnew .= w.Y
    return qnew
end
