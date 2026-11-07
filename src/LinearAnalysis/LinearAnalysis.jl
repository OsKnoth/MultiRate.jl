"""
    LinearAnalysis

Linear stability and accuracy analysis of the split-explicit MIS methods, after the MISDG
experiments (acoustics with advection on a staggered grid, Knoth & Wensch 2014).

A linear problem q' = N q with fast part L is a `LinearSplitModel`. It implements the
TimeStepping interface, so `step_matrix` runs the production `mis_step!` and fast loops
(:verlet, :fb, :wrf, :hevi, :hevi_verlet, :imex) on unit vectors; `fast = :exact` uses exact
fast stages instead. A Fourier mode with complex symbols is turned into a real model of twice
the size (`mode_model`), whose eigenvalues are those of the complex symbol and their conjugates.

Test problems (Δx = 1, Δt = 1, so the speeds are Courant numbers):
  * `kw_mode`: Knoth–Wensch eq. (10), u_t + U u_x = -c Π_x, Π_t + U Π_x = -c u_x, C-grid with
    2nd-order acoustics and 3rd-order upwind advection,
  * `staggered_mode`: MISDG grid, 4th-order staggered acoustics, 5th-order upwind flux advection,
  * `pulse_error`: 1D pressure pulse on the MISDG grid, error against the exact semi-discrete
    solution, mode by mode.
"""
module LinearAnalysis

using LinearAlgebra
using ..TimeStepping
import ..TimeStepping: rhs_nonlinear!, fast_operator, fast_apply!, fast_rhs_thermo!, fast_rhs_momentum!,
                       fast_rhs_horizontal!, vertical_apply!, vertical_factor, vertical_solve!,
                       fast_rhs_hmomentum!, fast_rhs_hthermo!

export LinearSplitModel, mode_model, step_matrix, spectral_radius, max_amplification,
       kw_mode, staggered_mode, staggered_symbols, pulse_error,
       boussinesq_symbols, boussinesq_mode, gravity_wave_section,
       fourier_block, linearized_mode_model, fourier_coefficients, fourier_synthesis,
       linearized_step, arnoldi_eigs

"""
    LinearSplitModel(N, L; Lv = 0, thermo, hmom = .!thermo)

q' = N q with fast part L. `Lv` is the vertical part of L (implicit in :hevi, :wrf), `thermo`
marks the rows of the thermodynamic kick (:verlet, :fb; the others are momentum), `hmom` the
horizontal-momentum rows (:wrf, :hevi_verlet).
"""
struct LinearSplitModel
    N::Matrix{Float64}
    L::Matrix{Float64}
    Lv::Matrix{Float64}
    thermo::BitVector
    hmom::BitVector
end
function LinearSplitModel(N::AbstractMatrix, L::AbstractMatrix; Lv = zeros(size(L)),
                          thermo = falses(size(L, 1)), hmom = .!BitVector(thermo))
    return LinearSplitModel(Matrix{Float64}(N), Matrix{Float64}(L), Matrix{Float64}(Lv),
                            BitVector(thermo), BitVector(hmom))
end

realify(Z::AbstractMatrix) = [real(Z) -imag(Z); imag(Z) real(Z)]

"""Real model of one Fourier mode with complex symbols N, L (and Lv); masks per complex row."""
mode_model(N, L; Lv = zero(L), thermo, hmom = .!BitVector(thermo)) =
    LinearSplitModel(realify(N), realify(L); Lv = realify(Lv), thermo = vcat(thermo, thermo),
                     hmom = vcat(hmom, hmom))

struct LinearFastOp <: AbstractFastOperator
    L::Matrix{Float64}
    Lh::Matrix{Float64}
    Lv::Matrix{Float64}
    thermo::BitVector
    hmom::BitVector
    fb::Vector{Float64}
    Mdiag::Vector{Float64}
    work::NTuple{7,Vector{Float64}}
end

rhs_nonlinear!(out, m::LinearSplitModel, q) = mul!(out, m.N, q)
function fast_operator(m::LinearSplitModel, q; kwargs...)
    n = size(m.L, 1)
    return LinearFastOp(m.L, m.L - m.Lv, m.Lv, m.thermo, m.hmom, zeros(n), ones(n), ntuple(_ -> zeros(n), 7))
end

function _rows!(out, A, q, f, rows)
    mul!(out, A, q)
    @. out = ifelse(rows, out + f, 0.0)
    return out
end
fast_apply!(out, s::LinearFastOp, q, f) = (mul!(out, s.L, q); out .+= f)
fast_rhs_thermo!(out, s::LinearFastOp, q, f) = _rows!(out, s.L, q, f, s.thermo)
fast_rhs_momentum!(out, s::LinearFastOp, q, f) = _rows!(out, s.L, q, f, .!s.thermo)
fast_rhs_horizontal!(out, s::LinearFastOp, q, f) = (mul!(out, s.Lh, q); out .+= f)
fast_rhs_hmomentum!(out, s::LinearFastOp, q, f) = _rows!(out, s.Lh, q, f, s.hmom)
fast_rhs_hthermo!(out, s::LinearFastOp, q, f) = _rows!(out, s.Lh, q, f, .!s.hmom)
vertical_apply!(out, s::LinearFastOp, q) = mul!(out, s.Lv, q)
vertical_factor(s::LinearFastOp, c) = lu(I - c * s.Lv)
vertical_solve!(Y, s::LinearFastOp, F, Ys) = (Y .= F \ Ys)

"""
    step_matrix(m, Δt, tab; nsub = 12, fast = :verlet)

Matrix of one MIS step for the linear model m (stage i: round(d_i nsub) fast substeps, as in
`mis_step!`). `fast = :exact` solves the fast stages exactly.
"""
function step_matrix(m::LinearSplitModel, Δt, tab::MISTableau; nsub::Int = 12, fast::Symbol = :verlet)
    n = size(m.N, 1)
    R = zeros(n, n)
    e = zeros(n)
    if fast == :exact
        S = m.N - m.L
        G = q -> S * q
        for j in 1:n
            e .= 0; e[j] = 1
            R[:, j] = mis_step_exact(m.L, G, e, Δt, tab)
        end
    else
        sF = fast_operator(m, e)
        w = MISWork(n, nstages(tab))
        out = zeros(n)
        for j in 1:n
            e .= 0; e[j] = 1
            mis_step!(out, m, e, Δt, tab, sF, w; nsub = nsub, fast = fast)
            R[:, j] = out
        end
    end
    return R
end

spectral_radius(R) = maximum(abs, eigvals(R))

"""
    max_amplification(modefun, tab; nmodes = 40, Δt = 1, kwargs...)

max over the Fourier modes θ_j = 2π j / nmodes (j = 1 … nmodes-1) of the spectral radius of
the step matrix of `modefun(θ)`; kwargs go to `step_matrix`.
"""
max_amplification(modefun, tab::MISTableau; nmodes::Int = 40, Δt = 1.0, kwargs...) =
    maximum(spectral_radius(step_matrix(modefun(2π * j / nmodes), Δt, tab; kwargs...)) for j in 1:nmodes-1)

# ------------------------------------------------------------------ test problems

"""
    kw_mode(θ; C_sound, C_adv)

Knoth–Wensch (2014, eq. 10) on a C-grid: Π' = a u, u' = b Π (2nd-order central acoustics),
advection -U ∂x by 3rd-order upwinding, Δx = Δt = 1. State (Π, u); Π is the thermodynamic row.
"""
function kw_mode(θ; C_sound, C_adv)
    z = cis(θ)
    a = -C_sound * (z - 1)
    b = -C_sound * (1 - 1 / z)
    ν = -C_adv / 6 * (2z + 3 - 6 / z + 1 / z^2)
    L = ComplexF64[0 a; b 0]
    return mode_model(L + ν * I, L; thermo = [true, false])
end

"""Symbols (a, b, ν) of the MISDG grid: p' = a u + ν p, u' = b p + ν u, 4th-order staggered
acoustics, 5th-order upwind-biased flux-form advection (Δx = 1, u scaled by ρc)."""
function staggered_symbols(θ; C_sound, C_adv)
    z = cis(θ)
    D1 = 9 / 8 * (z - 1) - 1 / 24 * (z^2 - 1 / z)
    D2 = 9 / 8 * (1 - 1 / z) - 1 / 24 * (z - 1 / z^2)
    r = (2 / z^2 - 13 / z + 47 + 27z - 3z^2) / 60
    return -C_sound * D1, -C_sound * D2, -C_adv * r * (1 - 1 / z)
end

"""Fourier mode of the MISDG grid (c = C_sound, U = C_adv, Δx = Δt = 1), state (p, u)."""
function staggered_mode(θ; C_sound, C_adv)
    a, b, ν = staggered_symbols(θ; C_sound, C_adv)
    L = ComplexF64[0 a; b 0]
    return mode_model(L + ν * I, L; thermo = [true, false])
end

"""
    pulse_error(tab, h; nx = 400, Lx = 1e6, c = 350, U = 50, T = 3.5 Lx/c, kwargs...)

1D pressure pulse of the MISDG talk (Gaussian of width Lx/16 at Lx/4, u = 0; at T both acoustic
pulses are back at the start) with slow step h: relative L2 error of p and u at T against the
exact semi-discrete solution, computed per Fourier mode. kwargs go to `step_matrix`
(`nsub`, `fast`).
"""
function pulse_error(tab::MISTableau, h; nx::Int = 400, Lx = 1e6, c = 350.0, U = 50.0,
                     T = 3.5Lx / c, kwargs...)
    nsteps = round(Int, T / h)
    abs(nsteps * h - T) < 1e-9T || throw(ArgumentError("T/h must be an integer"))
    dx = Lx / nx
    x = [(j - 0.5) * dx for j in 1:nx]
    p0 = @. exp(-((x - Lx / 4) / (Lx / 16))^2)
    e2, n2 = 0.0, 0.0
    for k in 0:nx-1
        θ = 2π * k / nx
        p̂ = sum(p0[j] * cis(-θ * (j - 1)) for j in 1:nx)
        # Courant numbers for Δx = 1: time unit h
        a, b, ν = staggered_symbols(θ; C_sound = c * h / dx, C_adv = U * h / dx)
        Lk = ComplexF64[0 a; b 0]
        qex = exp((Lk + ν * I) * (T / h)) * ComplexF64[p̂, 0]
        R = step_matrix(mode_model(Lk + ν * I, Lk; thermo = [true, false]), 1.0, tab; kwargs...)
        qr = R^nsteps * [real(p̂), 0, imag(p̂), 0]
        qn = ComplexF64[qr[1] + im * qr[3], qr[2] + im * qr[4]]
        e2 += sum(abs2, qn - qex)
        n2 += sum(abs2, qex)
    end
    return sqrt(e2 / n2)
end

# ------------------------------------------------------------------ acoustic-gravity waves

"""
    boussinesq_symbols(θx, n; Δx, Δz, H, c, N, U, Δt) -> (A, Lfast, Lv)

Linear compressible Boussinesq equations with uniform wind U and stability N,
    u_t + U u_x = -p_x,   w_t + U w_x = -p_z + b,   b_t + U b_x = -N² w,   p_t + U p_x = -c² (u_x + w_z),
on the grids of the FV model: horizontal C-grid (u at x-faces), vertical Lorenz grid (b and p at
centres, w at faces, b averaged to the faces and w to the centres), rigid lid at 0 and H. The
vertical mode n is represented exactly (p, u ∝ cos(m z), w, b ∝ sin(m z), m = nπ/H); θx is the
horizontal phase per cell. Advection: 3rd-order upwind. State (u, w, b, p), rates scaled by Δt.
Returns the full matrix A, the fast part (acoustics and buoyancy) and its vertical part Lv.
"""
function boussinesq_symbols(θx, n::Int; Δx, Δz, H, c, N, U, Δt)
    z = cis(θx)
    m = n * π / H
    sz = 2 / Δz * sin(m * Δz / 2)                  # |vertical difference| of the mode
    az = cos(m * Δz / 2)                           # vertical averaging (Lorenz grid)
    dcf = (z - 1) / Δx                             # x: centre → face
    dfc = (1 - 1 / z) / Δx                         # x: face → centre
    ν = -U / (6Δx) * (2z + 3 - 6 / z + 1 / z^2)    # 3rd-order upwind advection
    Lh = ComplexF64[0 0 0 -dcf; 0 0 0 0; 0 0 0 0; -c^2*dfc 0 0 0]
    Lv = ComplexF64[0 0 0 0; 0 0 az sz; 0 -N^2*az 0 0; 0 -c^2*sz 0 0]
    L = Lh + Lv
    return Δt * (L + ν * I), Δt * L, Δt * Lv
end

"""Fourier/vertical mode of `boussinesq_symbols` as a model for `step_matrix` (Δt = 1):
thermodynamic rows (b, p) for :verlet, horizontal momentum u for :hevi_verlet and :wrf."""
function boussinesq_mode(θx, n::Int; kwargs...)
    A, L, Lv = boussinesq_symbols(θx, n; kwargs...)
    return mode_model(A, L; Lv = Lv, thermo = [false, false, true, true], hmom = [true, false, false, false])
end

"""
    gravity_wave_section(tab, Δt; nsub = 12, fast = :hevi_verlet, nx = 300, Lx = 6000e3,
                         H = 10e3, nz = 16, a = 100e3, c = 347, N = 0.01, U = 20, T = 60000)

The long-channel gravity wave of `examples/euler2d_gravity_wave_long.jl` for the linear model of
`boussinesq_symbols`: initial b ∝ sin(πz/H)/(1 + (x - Lx/3)²/a²), u = w = p = 0 (perturbations).
Returns (x, b_numerical, b_exact) at the cell-centre level nearest H/2 and time T, with b
normalized to an initial maximum of 1 (multiply by 0.01 K for θ′).
"""
function gravity_wave_section(tab::MISTableau, Δt; nsub::Int = 12, fast::Symbol = :hevi_verlet, nx::Int = 300,
                              Lx = 6000e3, H = 10e3, nz::Int = 16, a = 100e3, c = 347.0, N = 0.01, U = 20.0,
                              T = 60000.0)
    nsteps = round(Int, T / Δt)
    abs(nsteps * Δt - T) < 1e-9T || throw(ArgumentError("T/Δt must be an integer"))
    Δx, Δz = Lx / nx, H / nz
    x = [(i - 0.5) * Δx for i in 1:nx]
    b0 = @. 1 / (1 + ((x - Lx / 3) / a)^2)
    k = nz ÷ 2
    zk = (k - 0.5) * Δz
    bn = zeros(ComplexF64, nx); be = zeros(ComplexF64, nx)
    for j in 0:nx-1
        θ = 2π * j / nx
        b̂ = sum(b0[i] * cis(-θ * (i - 1)) for i in 1:nx)
        A, L, Lv = boussinesq_symbols(θ, 1; Δx, Δz, H, c, N, U, Δt)
        q0 = ComplexF64[0, 0, b̂, 0]
        be[j+1] = (exp(A * nsteps) * q0)[3]
        R = step_matrix(mode_model(A, L; Lv = Lv, thermo = [false, false, true, true],
                                   hmom = [true, false, false, false]), 1.0, tab; nsub, fast)
        qr = R^nsteps * vcat(real(q0), imag(q0))
        bn[j+1] = qr[3] + im * qr[7]
    end
    back(f) = [real(sum(f[j+1] * cis(2π * j * (i - 1) / nx) for j in 0:nx-1)) / nx for i in 1:nx] .* sin(π * zk / H)
    return x, back(bn), back(be)
end

# ------------------------------------------------------------------ linearized production models

"""
    fourier_block(apply!, idx, θ, n) -> B

Block of a real linear operator `apply!(out, v)` (vectors of length n) for the horizontal
Fourier mode θ (phase per column). `idx[d][i]` is the global index of column unknown d in column
i; the operator must be invariant under shifts of the columns (flat, horizontally homogeneous
state, periodic in x). Then A Σ_i e^{iθ(i-1)} e_{idx[d][i]} = Σ_{d'} B[d', d] Σ_i e^{iθ(i-1)} e_{idx[d'][i]}.
"""
function fourier_block(apply!, idx::AbstractVector, θ, n::Int)
    nd, nx = length(idx), length(idx[1])
    B = zeros(ComplexF64, nd, nd)
    v = zeros(n); outr = zeros(n); outi = zeros(n)
    ph = [cis(-θ * (i - 1)) for i in 1:nx]
    for d in 1:nd
        v .= 0; for i in 1:nx; v[idx[d][i]] = cos(θ * (i - 1)); end
        apply!(outr, v)
        v .= 0; for i in 1:nx; v[idx[d][i]] = sin(θ * (i - 1)); end
        apply!(outi, v)
        for d2 in 1:nd
            B[d2, d] = sum((outr[idx[d2][i]] + im * outi[idx[d2][i]]) * ph[i] for i in 1:nx) / nx
        end
    end
    return B
end

"""
    linearized_mode_model(model, qb, idx, θ; thermo, hmom, ε = 1e-6, fast = :hevi, Δt = 1, fast_options...)

Fourier mode θ of a production model linearized about the steady, horizontally homogeneous
state qb: N → Jacobian of `rhs_nonlinear!` (central differences with step ε relative to the
size of qb), L and its vertical part from the fast operator frozen at qb. `thermo`/`hmom` mark
the column unknowns of the thermodynamic kick and of the horizontal momentum. All rates are
multiplied by Δt (for `step_matrix(…, 1.0, tab)`). Returns (mode model, J, L, Lv) with the
complex blocks.
"""
function linearized_mode_model(model, qb::AbstractVector, idx::AbstractVector, θ; thermo, hmom,
                               ε = 1e-6, fast::Symbol = :hevi, Δt = 1.0, fast_options...)
    n = length(qb)
    sF = fast_operator(model, qb; fast = fast, fast_options...)
    h = ε * max(1.0, maximum(abs, qb))
    rp = zeros(n); rm = zeros(n); z = zeros(n)
    Japply!(out, v) = (rhs_nonlinear!(rp, model, qb .+ h .* v); rhs_nonlinear!(rm, model, qb .- h .* v);
                       @. out = (rp - rm) / (2h))
    Lapply!(out, v) = fast_apply!(out, sF, v, z)
    Lvapply!(out, v) = vertical_apply!(out, sF, v)
    J = Δt * fourier_block(Japply!, idx, θ, n)
    L = Δt * fourier_block(Lapply!, idx, θ, n)
    Lv = Δt * fourier_block(Lvapply!, idx, θ, n)
    return mode_model(J, L; Lv = Lv, thermo = thermo, hmom = hmom), J, L, Lv
end

"""Fourier coefficients δ̂[d, j+1] = Σ_i δ[idx[d][i]] e^{-iθ_j (i-1)}, θ_j = 2π j / nx."""
function fourier_coefficients(δ::AbstractVector, idx::AbstractVector)
    nd, nx = length(idx), length(idx[1])
    return [sum(δ[idx[d][i]] * cis(-2π * j * (i - 1) / nx) for i in 1:nx) for d in 1:nd, j in 0:nx-1]
end

"""Inverse of `fourier_coefficients` into a state vector of length n (zeros elsewhere)."""
function fourier_synthesis(Q::AbstractMatrix, idx::AbstractVector, n::Int)
    nd, nx = size(Q)
    δ = zeros(n)
    for d in 1:nd, i in 1:nx
        δ[idx[d][i]] = real(sum(Q[d, j+1] * cis(2π * j * (i - 1) / nx) for j in 0:nx-1)) / nx
    end
    return δ
end

# ------------------------------------------------------------------ matrix-free analysis

"""
    linearized_step(model, qb, Δt, tab; nsub = 12, fast = :hevi, sF = nothing, ε = 1e-6, fast_options...)
        -> apply!(out, v)

Action of the linearized production MIS step about qb, v ↦ (Φ(qb + h v) - Φ(qb - h v)) / 2h,
with h = ε max(1, max|qb|) / max|v|. Φ is `mis_step!` with the fast operator frozen at qb (or sF), as in
a run between two re-freezes. Needs no shift invariance (terrain, cut cells, DG).
"""
function linearized_step(model, qb::AbstractVector, Δt, tab::MISTableau; nsub::Int = 12, fast::Symbol = :hevi,
                         sF = nothing, ε = 1e-6, fast_options...)
    n = length(qb)
    sF === nothing && (sF = fast_operator(model, qb; fast = fast, fast_options...))
    w = MISWork(n, nstages(tab))
    qp = zeros(n); qm = zeros(n)
    s = max(1.0, maximum(abs, qb))
    function apply!(out, v)
        h = ε * s / max(maximum(abs, v), floatmin())
        @. qp = qb + h * v; @. qm = qb - h * v
        mis_step!(qp, model, qp, Δt, tab, sF, w; nsub = nsub, fast = fast)
        mis_step!(qm, model, qm, Δt, tab, sF, w; nsub = nsub, fast = fast)
        @. out = (qp - qm) / (2h)
        return out
    end
    return apply!
end

"""
    arnoldi_eigs(apply!, n; nev = 6, krylovdim = 40, maxrestarts = 30, tol = 1e-6, v0 = randn(n),
                 weights = ones(n)) -> (λ, V, resid)

Eigenvalues of largest modulus of a real linear map `apply!(out, v)` on Rⁿ (matrix-free).
Arnoldi with explicit restarts: the new start vector is the sum of the real and imaginary parts
of the nev wanted Ritz vectors, scaled by their residuals. `weights` defines the inner product
(scale the unknowns to comparable size, e.g. 1/max|qb| per field). Returns λ sorted by
decreasing modulus, the Ritz vectors (columns, complex) and their relative residuals.
"""
function arnoldi_eigs(apply!, n::Int; nev::Int = 6, krylovdim::Int = 40, maxrestarts::Int = 30, tol = 1e-6,
                      v0 = randn(n), weights = ones(n), verbose::Bool = false)
    m = krylovdim
    sw = sqrt.(weights)
    V = zeros(n, m + 1); Hm = zeros(m + 1, m)
    x = zeros(n); y = zeros(n)
    start = v0 .* sw
    λ = ComplexF64[]; X = zeros(ComplexF64, n, 0); res = Float64[]
    for it in 1:maxrestarts
        V[:, 1] = start / norm(start); Hm .= 0
        k = m
        for j in 1:m                                 # Arnoldi in the weighted inner product
            x .= view(V, :, j) ./ sw
            apply!(y, x)
            u = y .* sw
            for _ in 1:2, i in 1:j                   # classical Gram-Schmidt, twice
                h = dot(view(V, :, i), u)
                Hm[i, j] += h
                u .-= h .* view(V, :, i)
            end
            Hm[j+1, j] = norm(u)
            if Hm[j+1, j] < 1e-14 * norm(view(Hm, 1:j, j))
                k = j; break
            end
            V[:, j+1] = u / Hm[j+1, j]
        end
        F = eigen(Hm[1:k, 1:k])
        p = sortperm(abs.(F.values); rev = true)
        nw = min(nev, k)
        λ = F.values[p[1:nw]]
        Y = F.vectors[:, p[1:nw]]
        res = [abs(Hm[k+1, k] * Y[k, i]) / max(abs(λ[i]), 1e-300) for i in 1:nw]
        X = (V[:, 1:k] * Y) ./ sw
        verbose && println("  restart $it: |λ| = ", round.(abs.(λ); digits = 6), "  max resid = ", maximum(res))
        maximum(res) < tol && break
        start = V[:, 1:k] * real.(sum(Y[:, i] * (1 + res[i] / tol) for i in 1:nw)) .+
                V[:, 1:k] * imag.(sum(Y[:, i] * (1 + res[i] / tol) for i in 1:nw))
    end
    return λ, X, res
end

end # module
