# 1D acoustics with advection, the test equation of the method design:
#     p_t + U p_x + c u_x = 0,   u_t + U u_x + c p_x = 0,
# fast part L = acoustics (c), slow part S = advection (U). Periodic, Fourier/Bloch symbols per
# wavenumber θ ∈ [0, π] (per cell resp. per element), scaled to U = c = 1, Δt = 1 and mean node
# spacing 1, so that multiplying by C_adv = UΔt/Δx and C_sound = cΔt/Δx gives the step operators.
# Unknowns ordered thermodynamic variable first: (p, u) resp. (p₁ … p_np, u₁ … u_np).

"""
    fv_symbols(θ; acoustic = 2, adv = 3) -> (S, L)

C-grid finite volumes (p in the cells, u on the faces): centred staggered acoustics of order
`acoustic` (2 or 4), upwind-biased flux-form advection of order `adv` (3 or 5) of both variables
(Knoth & Wensch 2014, eq. (10); the MISDG grid is acoustic = 4, adv = 5).
"""
function fv_symbols(θ; acoustic::Int = 2, adv::Int = 3)
    z = cis(θ)
    if acoustic == 2
        a, b = -(z - 1), -(1 - 1 / z)
    elseif acoustic == 4
        a = -(9 / 8 * (z - 1) - 1 / 24 * (z^2 - 1 / z))
        b = -(9 / 8 * (1 - 1 / z) - 1 / 24 * (z - 1 / z^2))
    else
        throw(ArgumentError("acoustic = 2 or 4"))
    end
    ν = adv == 3 ? -(2z + 3 - 6 / z + 1 / z^2) / 6 :
        adv == 5 ? -(2 / z^2 - 13 / z + 47 + 27z - 3z^2) / 60 * (1 - 1 / z) :
        throw(ArgumentError("adv = 3 or 5"))
    return ComplexF64[ν 0; 0 ν], ComplexF64[0 a; b 0]
end

"""Bloch block of the DGSEM operator for q_t + A q_x = 0 (2 variables, np GLL nodes, collocation,
strong form) with the numerical flux f* = A⁺ q⁻ + A⁻ q⁺, element length Δx, element wavenumber θ."""
function dg_block(A, Ap, Am, D, w, θ, Δx)
    np = length(w)
    z = cis(θ)
    J = 2 / Δx
    M = -J .* kron(ComplexF64.(A), ComplexF64.(D))
    idx(v, l) = (v - 1) * np + l
    for v in 1:2, u in 1:2
        # right face of the element: f*_R - A q_np, f*_R = A⁺ q_np + A⁻ q₁(next element)
        M[idx(v, np), idx(u, np)] -= J / w[np] * (Ap[v, u] - A[v, u])
        M[idx(v, np), idx(u, 1)] -= J / w[np] * Am[v, u] * z
        # left face: -(f*_L - A q₁), f*_L = A⁺ q_np(previous element) + A⁻ q₁
        M[idx(v, 1), idx(u, np)] += J / w[1] * Ap[v, u] / z
        M[idx(v, 1), idx(u, 1)] += J / w[1] * (Am[v, u] - A[v, u])
    end
    return M
end

"""
    dg_symbols(θ; P = 3, flux = :lmars, adv = :upwind) -> (S, L)

DGSEM of degree `P` on GLL nodes, element length P+1 (mean node spacing 1), θ the element
wavenumber. Acoustic flux `:lmars` (for the linear acoustics LMARS is the upwind flux:
p* = {p} - c/2 [u], u* = {u} - [p]/(2c)) or `:central`; advection `:upwind` or `:central`.
"""
function dg_symbols(θ; P::Int = 3, flux::Symbol = :lmars, adv::Symbol = :upwind)
    x, w = gll(P)
    _, D = lagrange_basis(x, x)
    Δx = P + 1.0
    Af = [0.0 1.0; 1.0 0.0]
    δf = flux == :lmars ? 1.0 : flux == :central ? 0.0 : throw(ArgumentError("flux = :lmars or :central"))
    L = dg_block(Af, (Af + δf * I) / 2, (Af - δf * I) / 2, D, w, θ, Δx)
    As = Matrix(1.0I, 2, 2)
    δs = adv == :upwind ? 1.0 : adv == :central ? 0.0 : throw(ArgumentError("adv = :upwind or :central"))
    S = dg_block(As, (As + δs * I) / 2, (As - δs * I) / 2, D, w, θ, Δx)
    return S, L
end

"""
    symbols(disc; kw...) -> θ -> (S, L)

`disc = :fv` (keywords of `fv_symbols`) or `:dg` (keywords of `dg_symbols`).
"""
function symbols(disc::Symbol; kw...)
    disc == :fv && return θ -> fv_symbols(θ; kw...)
    disc == :dg && return θ -> dg_symbols(θ; kw...)
    throw(ArgumentError("disc = :fv or :dg"))
end
