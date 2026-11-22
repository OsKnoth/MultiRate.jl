# MIS step with the component splitting :cs of the small steps (TimeStepping.frozen_mis_step! with fast = :cs and
# `_cs_loop!(…; halfsteps = true)`, CS_MERGE): per stage i the coefficients and the forcing are the combinations
# c = Σ_j β_ij/d_i C(Y_j), F = Σ_j β_ij/d_i S(Y_j) + Σ_j γ_ij/(d_i Δt) ΔY_j, the start value qⁿ + Σ_j α_ij ΔY_j, and
# n = ⌈d_i nsub⌉ small steps of length h = d_i Δt/n:
#     a½ = a + h/2 (-K b + f_a(a))                                                     (first half step)
#     for n: b⁺ = b + h L₀a½ + h f_b,  a⁺, b⁺ ← column solve (h/2 on a, βh on b)          (explicit part E at (a½, b⁺))
#            a½ ← a⁺ + h/2 (E - P(a½) + P(a⁺)) + vertical part (h/2, (1-β)h)            (n < N)
# Communications per small step: p (π) before the horizontal momentum step, U and V after it, both non-blocking and
# overlapped with the computation that needs no halo (`overlap = true`).

"""Work arrays of the MIS step: per stage C(Y_j), S(Y_j), ΔY_j; combined coefficients and forcing; vectors of the
fast loop."""
struct KAMISWork{A}
    C::Vector{A}; S::Vector{A}; ΔY::Vector{A}
    c::A; F::A; Y::A; Z::A; k::A; E::A; P::A; y::A
end
function KAMISWork(m::MoistKA, s::Int)
    z() = ka_zeros(m); zc() = ka_zeros(m, m.nc)
    return KAMISWork([zc() for _ in 1:s], [z() for _ in 1:s], [z() for _ in 1:s], zc(), z(), z(), z(), z(), z(), z(), z())
end

"""
Small steps of :cs (forward–backward form, docs/moist_verlet) over N steps of length h; Z halo filled.

`overlap = true`: communication overlaps computation as in CGDycore (non-blocking send and receive, posted as soon as
the data is final). Per small step
    [p in flight]  U, V += h :hmomentum on the faces without halo          ─ receive p ─ the faces with halo ─ send U, V
    [U, V in flight] E = :hthermo on the core, P(a½)                       ─ receive U, V ─ E on the last column and row
    column part on all columns (update, column solve, next half step)      ─ send p.
`overlap = false`: the same kernels, the exchanges wait at once.
"""
function _ka_cs_loop!(Z, m::MoistKA, Cf, F, h, N, w::KAMISWork, ε; overlap::Bool = true)
    k, E, P, y = w.k, w.E, w.P, w.y
    β = (1 + ε) / 2
    g, dec = m.grid, m.dec
    fast = m.micro == 1
    all_, ws, en, core = Region(g, :all), Region(g, :ws), Region(g, :en), Region(g, :core)
    pvars = m.rain ? (VP, VQR) : (VP,)                 # :hmomentum reads p (and ρq_r: sedimentation) of the halo
    send!(vars) = (h_ = start_halo!(dec, Z, g, vars); overlap ? h_ : (finish_halo!(dec, h_); nothing))
    # first half step a½ = a + h/2 (-K b + f_a(a)) with the vertical part (h/2 on a, (1-β)h on b)
    _hthermo!(k, m, Cf, Z, F, all_)
    _vertical!(y, m, Cf, Z, h / 2, (1 - β) * h, all_)
    _update!(Z, m, all_, h / 2, k, y; mode = 2)
    hp = send!(pvars)
    for n in 1:N
        # U⁺ = U + h L₀ã½ + h f_U
        _hmomentum!(Z, m, Cf, Z, F, all_; a = h, add = true, part = 1)
        finish_halo!(dec, hp)                                         # p of the neighbours
        _hmomentum!(Z, m, Cf, Z, F, ws; a = h, add = true, part = 2)
        huv = send!((VU, VV))
        # E at (a½, b⁺), P(a½)
        _hthermo!(E, m, Cf, Z, F, core)
        fast && _phaseset!(P, m, Z, all_)
        finish_halo!(dec, huv)                                        # U, V of the neighbours
        _hthermo!(E, m, Cf, Z, F, en)
        # (a⁺, b⁺): column solve
        _update!(Z, m, all_, h / 2, E; mode = 1)
        _vertical!(k, m, Cf, Z, h / 2, β * h, all_)
        _csolve!(y, m, Cf, h / 2, k, β * h, all_)
        _update!(Z, m, all_, 1.0, y; mode = 1)
        if n < N                                                      # half step after the solve
            _vertical!(y, m, Cf, Z, h / 2, (1 - β) * h, all_)
            if fast
                _phaseset!(k, m, Z, all_)                             # P(a⁺)
                _update!(Z, m, all_, h / 2, E, y, P, k; mode = 3)
            else
                _update!(Z, m, all_, h / 2, E, y; mode = 2)
            end
            hp = send!(pvars)
        end
    end
    return Z
end

"""
    ka_mis_step!(m, q, Δt, tab::MISTableau, w::KAMISWork; nsub = 12, offcenter = 0.0)

One MIS step q ← q(t + Δt) with the :cs small steps (in place; π/p recomputed at the end, `ka_step_end!`).
"""
function ka_mis_step!(m::MoistKA, q, Δt, tab::MISTableau, w::KAMISWork; nsub::Int = 12, offcenter::Real = 0.0,
                      overlap::Bool = true)
    g, dec = m.grid, m.dec
    for i in 1:nstages(tab)
        i == 1 ? (w.Y .= q) : (@. w.Y = q + w.ΔY[i-1])
        ka_frozen_data!(w.C[i], w.S[i], m, w.Y)                           # exchanges the halo of Y
        d = tab.d[i]
        w.c .= 0; w.F .= 0; w.Z .= q
        for j in 1:i
            b = tab.β[i, j] / d
            if b != 0
                @. w.c += b * w.C[j]
                @. w.F += b * w.S[j]
            end
            if j < i
                a, gm = tab.α[i, j], tab.γ[i, j] / (d * Δt)
                ΔYj = w.ΔY[j]
                a != 0 && (@. w.Z += a * ΔYj)
                gm != 0 && (@. w.F += gm * ΔYj)
            end
        end
        n = fast_substeps(d, nsub)
        for kk in collect(keys(m.work.fac)); kk != -1.0 && delete!(m.work.fac, kk); end   # new coefficients
        exchange!(dec, w.Z, g)
        _ka_cs_loop!(w.Z, m, w.c, w.F, d * Δt / n, n, w, Float64(offcenter); overlap)
        @. w.ΔY[i] = w.Z - q
    end
    q .= w.Z
    return ka_step_end!(m, q)
end

@kernel function _k_step_end!(Q, mg, pres, rain)
    i, j, k = @index(Global, NTuple)
    I, J = i + HALO, j + HALO
    @inbounds Q[I, J, k, VP] = ka_pvar(mg, pres, Q[I, J, k, VR], Q[I, J, k, VX], Q[I, J, k, VQV],
                                       Q[I, J, k, VQC] + _qr(Q, rain, I, J, k))
end
"""p (π) ← p(ρ, X, ρq_v, ρq_c + ρq_r) of the equation of state on the interior (no saturation adjustment)."""
function ka_step_end!(m::MoistKA, q)
    g = m.grid
    _launch!(m, _k_step_end!, (g.nxl, g.nyl, g.nz), q, m.mgas, m.pres, m.rain)
    return q
end

"""
    ka_run!(m, q, Δt, nsteps; tab, nsub = 12, offcenter = 0.0, monitor = nothing, every = 1)

nsteps MIS steps with the :cs small steps (in place). `monitor(n, q)` every `every` steps.
"""
function ka_run!(m::MoistKA, q, Δt, nsteps; tab::MISTableau, nsub::Int = 12, offcenter::Real = 0.0,
                 monitor = nothing, every::Int = 1, overlap::Bool = true)
    w = KAMISWork(m, nstages(tab))
    for n in 1:nsteps
        ka_mis_step!(m, q, Δt, tab, w; nsub, offcenter, overlap)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end
