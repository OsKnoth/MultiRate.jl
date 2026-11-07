"""
fast_fb.jl -- exponential-type multirate integrators for 1D linear acoustics with advection
on a periodic staggered grid.  ALL exponentials exp(hL) and ALL phi-functions phi_k(hL) are
computed by one high-order forward-backward (leapfrog-type) fast integrator with small steps.

    p_t + U p_x + rho c^2 u_x = 0        p at cell centers,  u at cell faces
    u_t + U u_x + p_x / rho   = 0

  L : acoustic part (fast), 4th order staggered differences     (A: p-kick, B: u-kick)
  N : advection (slow),     5th order upwind-biased flux form

Fast integrator.  State (p,u,t).  Split the vector field of  z' = L z + s(t)  into
    F_A :  p' = -rho c^2 D u + s_p(t),   t' = 1      (u frozen)    p-kick of size alpha
    F_B :  u' = -D p / rho   + s_u(t),   t' = 0      (p,t frozen)  u-kick of size beta
Both flows are exact.  The symmetric composition
    A(a_0) B(b_1) A(a_1) ... B(b_n) A(a_n)
(Strang, or Yoshida triple/septuple jump) gives order 2, 4, 6.  Adjacent A-kicks of
consecutive substeps are merged, so order 2 is plain leapfrog: one A and one B per substep.

Sources are polynomials in time,  s(t) = sum_k G_k t^(k-1)/(k-1)!, hence
    z(h) = exp(hL) z(0) + sum_k h^k phi_k(hL) G_k ,
i.e. exp and any combination of phi-functions come out of ONE fast pass.
"""
module FastFB

using LinearAlgebra: I

export Model, Coeffs, LawsonWS, ETDWS, RK3WS, fast!, adv!, lawson_rk4_step!, etdrk4_step!, rk3_split_step!,
       MISCoeffs, MISWS, mis_step!, mis_nsteps, mis_amplification, mis_amplification_mode, fb_stage_map,
       dft_matrix, exact_solution, exact_phi, phi_scalar, rk3_amplification

# ------------------------------------------------------------------ model
struct Model
    nx::Int
    dx::Float64
    Lx::Float64
    c::Float64
    U::Float64
    rho::Float64
    xc::Vector{Float64}          # p points (cell centers)
end

function Model(; nx=400, Lx=1.0e6, c=350.0, U=50.0, rho=1.0)
    dx = Lx / nx
    return Model(nx, dx, Lx, c, U, rho, [(j - 0.5) * dx for j in 1:nx])
end

@inline wrap(i, n) = i < 1 ? i + n : (i > n ? i - n : i)

# ------------------------------------------------------------------ composition coefficients
struct Coeffs
    a::Vector{Float64}           # a_0..a_n  (length n+1)
    b::Vector{Float64}           # b_1..b_n  (length n)
end

"Coefficients of the symmetric composition of Strang steps of the given order (2, 4 or 6)."
function Coeffs(order::Int)
    local w::Vector{Float64}
    if order == 2
        w = [1.0]
    elseif order == 4
        g1 = 1 / (2 - 2^(1/3))
        w = [g1, 1 - 2*g1, g1]
    elseif order == 6                       # Yoshida (1990), solution A
        w1 = -1.17767998417887
        w2 =  0.235573213359357
        w3 =  0.784513610477560
        w0 = 1 - 2*(w1 + w2 + w3)
        w = [w3, w2, w1, w0, w1, w2, w3]
    else
        error("order must be 2, 4 or 6")
    end
    n = length(w)
    a = zeros(n + 1)
    a[1] = w[1] / 2
    a[n+1] = w[n] / 2
    for i in 1:n-1
        a[i+1] = (w[i] + w[i+1]) / 2
    end
    return Coeffs(a, copy(w))
end

# ------------------------------------------------------------------ exact kicks
# p-kick:  p += alpha*(-rho c^2 D_x u) + sum_k Gp[k] * ((t+alpha)^k - t^k)/k!      (t advances by alpha)
function kickA!(p, u, m::Model, alpha, t, Gp)
    n = m.nx
    a = -alpha * m.rho * m.c^2 / m.dx
    @inbounds for j in 1:n
        d = 9/8 * (u[wrap(j+1, n)] - u[j]) - 1/24 * (u[wrap(j+2, n)] - u[wrap(j-1, n)])
        p[j] += a * d
    end
    for k in 1:length(Gp)
        w = ((t + alpha)^k - t^k) / factorial(k)
        gk = Gp[k]
        @. p += w * gk
    end
    return nothing
end

# u-kick:  u += beta*( -D_x p / rho + sum_k Gu[k] t^(k-1)/(k-1)! )                 (t frozen)
function kickB!(u, p, m::Model, beta, t, Gu)
    n = m.nx
    a = -beta / (m.rho * m.dx)
    @inbounds for j in 1:n
        d = 9/8 * (p[j] - p[wrap(j-1, n)]) - 1/24 * (p[wrap(j+1, n)] - p[wrap(j-2, n)])
        u[j] += a * d
    end
    for k in 1:length(Gu)
        w = beta * t^(k-1) / factorial(k-1)
        gk = Gu[k]
        @. u += w * gk
    end
    return nothing
end

# ------------------------------------------------------------------ fast integrator
"""
    fast!(p, u, m, h, nsub, cf, Gp, Gu)

In place:  (p,u) <- exp(hL)(p,u) + sum_k h^k phi_k(hL) G_k,  with `nsub` substeps of size h/nsub
of the order-`cf` composition.  `Gp`, `Gu` are vectors of arrays (G_1, G_2, ...) for the p and
u component; pass empty vectors for a pure exponential.

Stability: every kick has size <= max|b| * tau, so need  c*tau*max|b|/dx <= 6/7
(max|b| = 1, 1.70, 1.32 for order 2, 4, 6; sufficient, not necessary).
"""
function fast!(p, u, m::Model, h, nsub::Int, cf::Coeffs, Gp, Gu)
    tau = h / nsub
    n = length(cf.b)
    t = 0.0
    pend = cf.a[1] * tau
    for s in 1:nsub
        if s > 1
            pend += cf.a[1] * tau
        end
        for i in 1:n
            kickA!(p, u, m, pend, t, Gp)
            t += pend
            kickB!(u, p, m, cf.b[i] * tau, t, Gu)
            pend = cf.a[i+1] * tau
        end
    end
    kickA!(p, u, m, pend, t, Gp)
    return nothing
end

const NOSRC = Vector{Vector{Float64}}()

"Pure exponential: (p,u) <- exp(hL)(p,u)."
fast_exp!(p, u, m::Model, h, nsub, cf::Coeffs) = fast!(p, u, m, h, nsub, cf, NOSRC, NOSRC)

# ------------------------------------------------------------------ slow operator N
"out <- N f : 5th order upwind-biased flux form advection (U > 0); `fh` is work space."
function adv!(out, f, m::Model, fh)
    n = m.nx
    cU = -m.U / m.dx
    @inbounds for j in 1:n
        fh[j] = (2*f[wrap(j-2, n)] - 13*f[wrap(j-1, n)] + 47*f[j] +
                 27*f[wrap(j+1, n)] - 3*f[wrap(j+2, n)]) / 60       # flux at j+1/2
    end
    @inbounds for j in 1:n
        out[j] = cU * (fh[j] - fh[wrap(j-1, n)])
    end
    return nothing
end

# ------------------------------------------------------------------ Lawson-RK4
Base.@kwdef struct LawsonWS
    n::Int
    ap::Vector{Float64} = zeros(n)
    au::Vector{Float64} = zeros(n)      # k1, later k4
    wp::Vector{Float64} = zeros(n)
    wu::Vector{Float64} = zeros(n)      # w = E y
    y2p::Vector{Float64} = zeros(n)
    y2u::Vector{Float64} = zeros(n)     # Y2
    bp::Vector{Float64} = zeros(n)
    bu::Vector{Float64} = zeros(n)      # k2
    cp::Vector{Float64} = zeros(n)
    cu::Vector{Float64} = zeros(n)      # k3
    tp::Vector{Float64} = zeros(n)
    tu::Vector{Float64} = zeros(n)      # temporaries
    fh::Vector{Float64} = zeros(n)      # flux work array
end

"""
    lawson_rk4_step!(p, u, m, h, nsub, cf, ws)

One Lawson-RK4 step: 4 evaluations of N and 4 applications of E = exp(hL/2), each E being
`nsub` substeps of the fast integrator over h/2.  exp(hL) = E^2 is never formed.
"""
function lawson_rk4_step!(p, u, m::Model, h, nsub::Int, cf::Coeffs, ws::LawsonWS)
    fh = ws.fh
    E!(a, b) = fast_exp!(a, b, m, h / 2, nsub, cf)
    adv!(ws.ap, p, m, fh); adv!(ws.au, u, m, fh)                       # k1
    copyto!(ws.wp, p); copyto!(ws.wu, u); E!(ws.wp, ws.wu)             # w = E y
    @. ws.y2p = p + h/2 * ws.ap
    @. ws.y2u = u + h/2 * ws.au
    E!(ws.y2p, ws.y2u)                                                 # Y2 = E(y + h/2 k1)
    adv!(ws.bp, ws.y2p, m, fh); adv!(ws.bu, ws.y2u, m, fh)             # k2
    @. ws.tp = ws.wp + h/2 * ws.bp                                     # Y3 = w + h/2 k2
    @. ws.tu = ws.wu + h/2 * ws.bu
    adv!(ws.cp, ws.tp, m, fh); adv!(ws.cu, ws.tu, m, fh)               # k3
    @. ws.tp = ws.wp + h * ws.cp
    @. ws.tu = ws.wu + h * ws.cu
    E!(ws.tp, ws.tu)                                                   # Y4 = E(w + h k3)
    adv!(ws.ap, ws.tp, m, fh); adv!(ws.au, ws.tu, m, fh)               # k4 (reuses k1 storage)
    @. ws.tp = ws.wp + (ws.y2p - ws.wp) / 3 + h/3 * (ws.bp + ws.cp)
    @. ws.tu = ws.wu + (ws.y2u - ws.wu) / 3 + h/3 * (ws.bu + ws.cu)
    E!(ws.tp, ws.tu)                                                   # E[w + (Y2-w)/3 + h/3 (k2+k3)]
    @. p = ws.tp + h/6 * ws.ap
    @. u = ws.tu + h/6 * ws.au
    return nothing
end

# ------------------------------------------------------------------ ETDRK4 (Cox-Matthews)
Base.@kwdef struct ETDWS
    n::Int
    g1p::Vector{Float64} = zeros(n)
    g1u::Vector{Float64} = zeros(n)
    g2p::Vector{Float64} = zeros(n)
    g2u::Vector{Float64} = zeros(n)
    g3p::Vector{Float64} = zeros(n)
    g3u::Vector{Float64} = zeros(n)
    g4p::Vector{Float64} = zeros(n)
    g4u::Vector{Float64} = zeros(n)
    ap::Vector{Float64} = zeros(n)
    au::Vector{Float64} = zeros(n)
    bp::Vector{Float64} = zeros(n)
    bu::Vector{Float64} = zeros(n)
    cp::Vector{Float64} = zeros(n)
    cu::Vector{Float64} = zeros(n)
    sp::Vector{Float64} = zeros(n)
    su::Vector{Float64} = zeros(n)
    G2p::Vector{Float64} = zeros(n)
    G2u::Vector{Float64} = zeros(n)
    G3p::Vector{Float64} = zeros(n)
    G3u::Vector{Float64} = zeros(n)
    fh::Vector{Float64} = zeros(n)
end

"""
    etdrk4_step!(p, u, m, h, nsub, cf, ws)

Cox-Matthews ETDRK4.  Every exp(.L) and phi_k(.L) is evaluated by `fast!` through a polynomial
source (see the module header).  With g_i = N(stage i):
  a = E y + h/2 phi1(hL/2) g1,   b = E y + h/2 phi1(hL/2) g2,   c = E a + h/2 phi1(hL/2)(2 g3 - g1)
  y+ = e^{hL} y + h phi1 G1 + h^2 phi2 G2 + h^3 phi3 G3  with
       G1 = g1,  G2 = (-3 g1 + 2 (g2+g3) - g4)/h,  G3 = 4 (g1 - g2 - g3 + g4)/h^2,
which is the usual  h[(phi1-3phi2+4phi3) g1 + 2(phi2-2phi3)(g2+g3) + (-phi2+4phi3) g4].
`nsub` is the number of substeps per h/2 (the final pass over h uses 2*nsub).
"""
function etdrk4_step!(p, u, m::Model, h, nsub::Int, cf::Coeffs, ws::ETDWS)
    fh = ws.fh
    adv!(ws.g1p, p, m, fh); adv!(ws.g1u, u, m, fh)                      # g1 = N(y)
    copyto!(ws.ap, p); copyto!(ws.au, u)
    fast!(ws.ap, ws.au, m, h/2, nsub, cf, [ws.g1p], [ws.g1u])           # a
    adv!(ws.g2p, ws.ap, m, fh); adv!(ws.g2u, ws.au, m, fh)              # g2 = N(a)
    copyto!(ws.bp, p); copyto!(ws.bu, u)
    fast!(ws.bp, ws.bu, m, h/2, nsub, cf, [ws.g2p], [ws.g2u])           # b
    adv!(ws.g3p, ws.bp, m, fh); adv!(ws.g3u, ws.bu, m, fh)              # g3 = N(b)
    @. ws.sp = 2 * ws.g3p - ws.g1p
    @. ws.su = 2 * ws.g3u - ws.g1u
    copyto!(ws.cp, ws.ap); copyto!(ws.cu, ws.au)
    fast!(ws.cp, ws.cu, m, h/2, nsub, cf, [ws.sp], [ws.su])             # c = E a + h/2 phi1 (2 g3 - g1)
    adv!(ws.g4p, ws.cp, m, fh); adv!(ws.g4u, ws.cu, m, fh)              # g4 = N(c)
    @. ws.G2p = (-3 * ws.g1p + 2 * (ws.g2p + ws.g3p) - ws.g4p) / h
    @. ws.G2u = (-3 * ws.g1u + 2 * (ws.g2u + ws.g3u) - ws.g4u) / h
    @. ws.G3p = 4 * (ws.g1p - ws.g2p - ws.g3p + ws.g4p) / h^2
    @. ws.G3u = 4 * (ws.g1u - ws.g2u - ws.g3u + ws.g4u) / h^2
    fast!(p, u, m, h, 2 * nsub, cf, [ws.g1p, ws.G2p, ws.G3p], [ws.g1u, ws.G2u, ws.G3u])
    return nothing
end

# ------------------------------------------------------------------ split-explicit RK3 (WRF)
Base.@kwdef struct RK3WS
    n::Int
    sp::Vector{Float64} = zeros(n)
    su::Vector{Float64} = zeros(n)      # stage value q*, q**
    np::Vector{Float64} = zeros(n)
    nu::Vector{Float64} = zeros(n)      # frozen slow tendency N(q*)
    pe::Vector{Float64} = zeros(n)      # extrapolated pressure
    po::Vector{Float64} = zeros(n)      # pressure of the previous acoustic step
    fh::Vector{Float64} = zeros(n)
end

"""
Forward-backward acoustic substeps as in WRF (u first, then p with the new u), slow tendency
(Np, Nu) held constant:  `ns` steps of size `dts`.  The u-kick uses the extrapolated pressure
    p~ = p^tau + beta (p^tau - p^(tau-dts))        (Klemp, Skamarock, Dudhia 2007; WRF smdiv = beta)
which acts as divergence damping.  At the start of a stage p^(tau-dts) = p^tau.  `pe`, `po` are work space.
"""
function fb_wrf!(p, u, m::Model, dts, ns::Int, Np, Nu, beta, pe, po)
    n = m.nx
    au = -dts / (m.rho * m.dx)
    ap = -dts * m.rho * m.c^2 / m.dx
    copyto!(po, p)
    for _ in 1:ns
        @. pe = p + beta * (p - po)
        copyto!(po, p)
        @inbounds for j in 1:n
            d = 9/8 * (pe[j] - pe[wrap(j-1, n)]) - 1/24 * (pe[wrap(j+1, n)] - pe[wrap(j-2, n)])
            u[j] += au * d + dts * Nu[j]
        end
        @inbounds for j in 1:n
            d = 9/8 * (u[wrap(j+1, n)] - u[j]) - 1/24 * (u[wrap(j+2, n)] - u[wrap(j-1, n)])
            p[j] += ap * d + dts * Np[j]
        end
    end
    return nothing
end

"""
    rk3_split_step!(p, u, m, h, nsub, ws; first = :wrf, beta = 0.0)

Wicker-Skamarock (2002) split-explicit RK3 as in WRF:
  q*   = q^n + FB over h/3 with N(q^n)
  q**  = q^n + FB over h/2 with N(q*)
  q^+  = q^n + FB over h   with N(q**)
`nsub` (even) acoustic steps of size tau = h/nsub per h; stage 2 uses nsub/2 of them.  Stage 1:
`first = :wrf` takes ONE step of size h/3 (WRF), `first = :uniform` takes nsub/3 steps of size tau
(nsub divisible by 6).  `beta`: pressure extrapolation in the acoustic steps (see `fb_wrf!`; WRF uses 0.1).
"""
function rk3_split_step!(p, u, m::Model, h, nsub::Int, ws::RK3WS; first::Symbol = :wrf, beta = 0.0)
    (iseven(nsub) && nsub >= 2) || error("nsub must be even and >= 2")
    fh = ws.fh
    tau = h / nsub
    if first === :wrf
        n1, dt1 = 1, h / 3
    elseif first === :uniform
        nsub % 6 == 0 || error("first = :uniform needs nsub divisible by 6")
        n1, dt1 = nsub ÷ 3, tau
    else
        error("first must be :wrf or :uniform")
    end
    adv!(ws.np, p, m, fh); adv!(ws.nu, u, m, fh)                        # N(q^n)
    copyto!(ws.sp, p); copyto!(ws.su, u)
    fb_wrf!(ws.sp, ws.su, m, dt1, n1, ws.np, ws.nu, beta, ws.pe, ws.po)   # q*
    adv!(ws.np, ws.sp, m, fh); adv!(ws.nu, ws.su, m, fh)                # N(q*)
    copyto!(ws.sp, p); copyto!(ws.su, u)
    fb_wrf!(ws.sp, ws.su, m, tau, nsub ÷ 2, ws.np, ws.nu, beta, ws.pe, ws.po)   # q**
    adv!(ws.np, ws.sp, m, fh); adv!(ws.nu, ws.su, m, fh)                # N(q**)
    fb_wrf!(p, u, m, tau, nsub, ws.np, ws.nu, beta, ws.pe, ws.po)   # q^{n+1}
    return nothing
end

# ------------------------------------------------------------------ MIS (multirate infinitesimal step)
"""
MIS method (Wensch, Knoth, Galant 2009; Knoth, Wensch 2014), s stages, strictly lower triangular
(s+1)x(s+1) matrices alpha, beta, gamma:
    Z_i(0)  = y_n + sum_j alpha_ij (Y_j - y_n)
    Z_i'(t) = 1/h sum_j gamma_ij (Y_j - y_n) + sum_j beta_ij N(Y_j) + d_i L Z_i(t),   t in [0,h]
    Y_i     = Z_i(h),    d_i = sum_j beta_ij,    y_{n+1} = Y_{s+1}.
With t = d_i * time the stage is  z' = L z + F_i/d_i  over a time d_i h  (F_i = forcing above), i.e.
    Y_i = exp(d_i h L) Z_i(0) + h phi_1(d_i h L) F_i .
"""
struct MISCoeffs
    name::String
    alpha::Matrix{Float64}
    beta::Matrix{Float64}
    gamma::Matrix{Float64}
    d::Vector{Float64}
end

function MISCoeffs(name, alpha, beta, gamma)
    return MISCoeffs(name, alpha, beta, gamma, vec(sum(beta, dims = 2)))
end

"MIS-KW form (alpha = subdiagonal ones, gamma = 0) of an explicit RK method with matrix A and weights b."
function mis_kw(name, A, b)
    s = length(b)
    Ae = zeros(s + 1, s + 1); Ae[1:s, 1:s] = A; Ae[s+1, 1:s] = b
    beta = zeros(s + 1, s + 1); alpha = zeros(s + 1, s + 1)
    for i in 2:s+1
        beta[i, :] = Ae[i, :] - Ae[i-1, :]
        alpha[i, i-1] = 1
    end
    return MISCoeffs(name, alpha, beta, zeros(s + 1, s + 1))
end

"""
    MISCoeffs(name::Symbol)

  :rk3ws  Wicker-Skamarock RK3 (stages h/3, h/2, h from y_n), alpha = gamma = 0      (order 2)
  :kw3    MIS-KW form of the Knoth-Wolke RK3 (c = 0, 1/3, 3/4), recursive stages      (order 3)
  :mis54  5 stages, order 4: Bauer, Knoth, J. Comput. Appl. Math. 387 (2021), Table 2
"""
function MISCoeffs(name::Symbol)
    if name === :rk3ws
        beta = zeros(4, 4); beta[2, 1] = 1/3; beta[3, 2] = 1/2; beta[4, 3] = 1
        return MISCoeffs("RK3 (WS)", zeros(4, 4), beta, zeros(4, 4))
    elseif name === :kw3
        return mis_kw("MIS-KW3", [0 0 0; 1/3 0 0; -3/16 15/16 0], [1/6, 3/10, 8/15])
    elseif name === :mis54
        al = zeros(6, 6); ga = zeros(6, 6); be = zeros(6, 6)
        al[2,1] = -0.056843003311023; ga[2,1] =  0.168489083931286; be[2,1] =  0.219579314792533
        al[3,1] =  0.071035715986068; ga[3,1] = -0.025097850341834; be[3,1] = -0.032864918414060
        al[3,2] =  0.050143439731979; ga[3,2] =  0.025515704040468; be[3,2] =  0.634699918767414
        al[4,1] =  0.021491523917140; ga[4,1] =  0.106139356407192; be[4,1] = -0.241761887431829
        al[4,2] =  0.287530720188756; ga[4,2] =  0.264445452990869; be[4,2] = -0.120631540663984
        al[4,3] =  0.239030810792355; ga[4,3] =  0.402246482358727; be[4,3] =  0.374686620841487
        al[5,1] =  0.027558616966568; ga[5,1] = -0.031464053194458; be[5,1] = -0.058474324094343
        al[5,2] =  0.382675659910308; ga[5,2] = -0.068258296801680; be[5,2] =  0.351217252190521
        al[5,3] =  0.177185696263246; ga[5,3] =  0.027558616966568; be[5,3] =  0.309657030167295
        al[5,4] = -0.314894383613333; ga[5,4] =  0.015830368641068; be[5,4] =  0.168604799122988
        al[6,1] =  0.065158401284120; ga[6,1] =  0.150547662349659; be[6,1] = -0.056205055946158
        al[6,2] =  0.079591607322196; ga[6,2] =  0.088610905686011; be[6,2] = -0.068390330952311
        al[6,3] =  0.459806401597571; ga[6,3] =  0.067880982803316; be[6,3] = -0.086209210260269
        al[6,4] =  0.086725275506356; ga[6,4] = -0.297416190393485; be[6,4] =  0.034904705602768
        al[6,5] =  0.439945196292364; ga[6,5] =  0.148246909195494; be[6,5] =  0.448964988009822
        return MISCoeffs("MIS54", al, be, ga)
    else
        error("unknown MIS method $name")
    end
end

struct MISWS
    Yp::Vector{Vector{Float64}}
    Yu::Vector{Vector{Float64}}
    Np::Vector{Vector{Float64}}
    Nu::Vector{Vector{Float64}}
    Fp::Vector{Float64}
    Fu::Vector{Float64}
    pe::Vector{Float64}
    po::Vector{Float64}
    fh::Vector{Float64}
end
MISWS(n::Int, s::Int) = MISWS([zeros(n) for _ in 1:s+1], [zeros(n) for _ in 1:s+1],
                              [zeros(n) for _ in 1:s], [zeros(n) for _ in 1:s],
                              zeros(n), zeros(n), zeros(n), zeros(n), zeros(n))
MISWS(n::Int, mc::MISCoeffs) = MISWS(n, length(mc.d) - 1)

"Number of fast substeps of MIS stage i (time d_i h) for a target substep `tau`."
mis_nsteps(mc::MISCoeffs, h, tau) = [max(1, ceil(Int, mc.d[i] * h / tau - 1e-9)) for i in 2:length(mc.d)]

"""
    mis_step!(p, u, m, h, mc, ws; tau, fast = Coeffs(4), beta = 0.0)

One MIS step.  Stage i integrates  z' = L z + F_i/d_i  over d_i h  with ceil(d_i h/tau) substeps:
`fast::Coeffs` uses the composition integrator `fast!` (order 2/4/6), `fast = :fb` the WRF
forward-backward scheme `fb_wrf!` (with pressure extrapolation `beta`).
"""
function mis_step!(p, u, m::Model, h, mc::MISCoeffs, ws::MISWS; tau, fast = Coeffs(4), beta = 0.0)
    s = length(mc.d) - 1
    al, be, ga = mc.alpha, mc.beta, mc.gamma
    ns = mis_nsteps(mc, h, tau)
    copyto!(ws.Yp[1], p); copyto!(ws.Yu[1], u)
    adv!(ws.Np[1], p, m, ws.fh); adv!(ws.Nu[1], u, m, ws.fh)
    for i in 2:s+1
        Zp = ws.Yp[i]; Zu = ws.Yu[i]
        copyto!(Zp, p); copyto!(Zu, u)
        fill!(ws.Fp, 0.0); fill!(ws.Fu, 0.0)
        for j in 1:i-1
            aij = al[i, j]; gij = ga[i, j] / h; bij = be[i, j]
            Yp = ws.Yp[j]; Yu = ws.Yu[j]; Np = ws.Np[j]; Nu = ws.Nu[j]
            if aij != 0
                @. Zp += aij * (Yp - p); @. Zu += aij * (Yu - u)
            end
            if gij != 0
                @. ws.Fp += gij * (Yp - p); @. ws.Fu += gij * (Yu - u)
            end
            if bij != 0
                @. ws.Fp += bij * Np; @. ws.Fu += bij * Nu
            end
        end
        d = mc.d[i]
        ws.Fp ./= d; ws.Fu ./= d
        if fast === :fb
            fb_wrf!(Zp, Zu, m, d * h / ns[i-1], ns[i-1], ws.Fp, ws.Fu, beta, ws.pe, ws.po)
        else
            fast!(Zp, Zu, m, d * h, ns[i-1], fast, [ws.Fp], [ws.Fu])
        end
        if i <= s
            adv!(ws.Np[i], Zp, m, ws.fh); adv!(ws.Nu[i], Zu, m, ws.fh)
        end
    end
    copyto!(p, ws.Yp[s+1]); copyto!(u, ws.Yu[s+1])
    return nothing
end

"""
    mis_amplification_mode(a, b, nu, h, mc; nfb = nothing, beta = 0.0)

2x2 amplification matrix of one MIS step for one Fourier mode of  p' = a u + nu p,  u' = b p + nu u
(L = [0 a; b 0], N = nu I).  Fast stages exact (`nfb = nothing`) or `nfb[i]` forward-backward steps.
"""
function mis_amplification_mode(a, b, nu, h, mc::MISCoeffs; nfb = nothing, beta = 0.0)
    s = length(mc.d) - 1
    Id = Matrix{ComplexF64}(I, 2, 2)
    L = ComplexF64[0 a; b 0]
    Y = Vector{Matrix{ComplexF64}}(undef, s + 1)
    Y[1] = Id
    for i in 2:s+1
        Z0 = copy(Id); F = zeros(ComplexF64, 2, 2)
        for j in 1:i-1
            Z0 += mc.alpha[i, j] * (Y[j] - Id)
            F  += mc.gamma[i, j] / h * (Y[j] - Id) + mc.beta[i, j] * nu * Y[j]
        end
        d = mc.d[i]
        if nfb === nothing
            Bm = zeros(ComplexF64, 4, 4)                  # exp([dhL dhI; 0 0]) = [E  dh phi1(dhL); 0 I]
            Bm[1:2, 1:2] = d * h * L; Bm[1:2, 3:4] = d * h * Id
            X = exp(Bm)
            Y[i] = X[1:2, 1:2] * Z0 + X[1:2, 3:4] * (F / d)
        else
            P, W = fb_stage_map(a, b, d * h / nfb[i-1], nfb[i-1], beta)
            Y[i] = P * Z0 + W * (F / d)
        end
    end
    return Y[s+1]
end

"Amplification matrices of `mis_step!` (with `fast = :fb`, or exact fast stages if `exact`) for all modes of m."
function mis_amplification(m::Model, h, mc::MISCoeffs; tau = Inf, exact = false, beta = 0.0)
    a, b, _, ns = symbols(m)
    nfb = exact ? nothing : mis_nsteps(mc, h, tau)
    return [mis_amplification_mode(a[k], b[k], ns[k], h, mc; nfb = nfb, beta = beta) for k in 1:m.nx]
end

# ------------------------------------------------------------------ exact references (testing only)
# L and N are circulant on the periodic grid and commute, so the semi-discrete problem is solved
# exactly in Fourier space.  A dense DFT matrix is used to avoid external packages.

dft_matrix(n) = [cis(-2pi * mod((j - 1) * (k - 1), n) / n) for j in 1:n, k in 1:n]   # exact integer phase reduction

function symbols(m::Model)
    n = m.nx
    z  = [cis(2pi * k / n) for k in 0:n-1]
    D1 = @. (9/8 * (z - 1) - 1/24 * (z^2 - 1/z)) / m.dx
    D2 = @. (9/8 * (1 - 1/z) - 1/24 * (z - 1/z^2)) / m.dx
    a  = @. -m.rho * m.c^2 * D1
    b  = @. -D2 / m.rho
    om = @. sqrt(max(-(m.c^2) * real(D1 * D2), 0.0))
    r  = @. (2/z^2 - 13/z + 47 + 27*z - 3*z^2) / 60
    ns = @. -(m.U / m.dx) * r * (1 - 1/z)
    return a, b, om, ns
end

"""
    fb_stage_map(a, b, dts, nst, beta)

One Fourier mode of `fb_wrf!`: `nst` forward-backward steps of size `dts` for  p' = a u + g_p,
u' = b p + g_u  with constant g.  Returns (P, W) with  (p,u)_end = P (p,u)_0 + W g.
Pressure extrapolation makes the step a 3-level scheme: state (p, u, p_old), p_old = p at the start.
"""
function fb_stage_map(a, b, dts, nst::Int, beta)
    M = [1 + dts^2*a*b*(1+beta)  dts*a  -dts^2*a*b*beta;
         dts*b*(1+beta)          1      -dts*b*beta;
         1                       0       0]
    K = ComplexF64[dts dts^2*a; 0 dts; 0 0]
    E = ComplexF64[1 0; 0 1; 1 0]
    P = Matrix{ComplexF64}(I, 3, 3); W = zeros(ComplexF64, 3, 2)
    for _ in 1:nst
        W = M * W + K; P = M * P
    end
    return (P * E)[1:2, :], W[1:2, :]
end

"""
    rk3_amplification(m, h, nsub; beta = 0.0, first = :wrf)

Amplification matrices (2x2 per Fourier mode, acting on (p^, u^)) of one `rk3_split_step!`.
"""
function rk3_amplification(m::Model, h, nsub::Int; beta = 0.0, first::Symbol = :wrf)
    a, b, _, ns = symbols(m)
    tau = h / nsub
    n1, dt1 = first === :wrf ? (1, h / 3) : (nsub ÷ 3, tau)
    stage(ak, bk, dts, nst) = fb_stage_map(ak, bk, dts, nst, beta)
    R = Vector{Matrix{ComplexF64}}(undef, m.nx)
    for k in 1:m.nx
        P1, W1 = stage(a[k], b[k], dt1, n1)
        P2, W2 = stage(a[k], b[k], tau, nsub ÷ 2)
        P3, W3 = stage(a[k], b[k], tau, nsub)
        qs  = P1 + ns[k] * W1
        qss = P2 + ns[k] * W2 * qs
        R[k] = P3 + ns[k] * W3 * qss
    end
    return R
end

"Exact solution of the semi-discrete problem (L + N) at time t."
function exact_solution(m::Model, p0, u0, t; F=dft_matrix(m.nx))
    n = m.nx
    a, b, om, ns = symbols(m)
    C = cos.(om .* t)
    S = [t * sinc(w * t / pi) for w in om]
    g = exp.(t .* ns)
    ph = F * p0; uh = F * u0
    p = real.(F' * (g .* (C .* ph .+ a .* S .* uh))) ./ n
    u = real.(F' * (g .* (b .* S .* ph .+ C .* uh))) ./ n
    return p, u
end

"phi_k(z) for complex z (series for |z| < 3, recursion otherwise)."
function phi_scalar(k::Int, z::ComplexF64)
    if abs(z) < 3
        acc = 0.0 + 0.0im
        term = ComplexF64(1 / factorial(k))
        for j in 0:59
            acc += term
            term *= z / (j + k + 1)
        end
        return acc
    else
        f = (exp(z) - 1) / z
        for j in 1:k-1
            f = (f - 1 / factorial(j)) / z
        end
        return f
    end
end

"phi_k(hL) g for the acoustic operator L alone (no advection); used to test `fast!`."
function exact_phi(m::Model, k::Int, gp, gu, h; F=dft_matrix(m.nx))
    n = m.nx
    a, b, om, _ = symbols(m)
    th = om .* h
    alpha = [real((phi_scalar(k, 1im * t) + phi_scalar(k, -1im * t)) / 2) for t in th]
    beta  = [t == 0 ? 0.0 : real((phi_scalar(k, 1im * t) - phi_scalar(k, -1im * t)) / (2im * t)) for t in th]
    ph = F * gp; uh = F * gu
    p = real.(F' * (alpha .* ph .+ beta .* h .* a .* uh)) ./ n
    u = real.(F' * (alpha .* uh .+ beta .* h .* b .* ph)) ./ n
    return p, u
end

end # module
