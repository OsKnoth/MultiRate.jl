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
       kw_mode, staggered_mode, staggered_symbols, pulse_error

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

end # module
