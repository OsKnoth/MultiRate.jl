# run_experiments.jl -- 1D pressure pulse, convergence and work-precision data for the slides
#   (Lawson-RK4, ETDRK4 with the high-order fast integrator; split-explicit RK3 as in WRF for comparison)
# usage:  julia run_experiments.jl      (writes data_*.csv next to this file)
include("fast_fb.jl")
using .FastFB
using Printf

m  = Model()                               # L = 1000 km, 400 cells, c = 350 m/s, U = 50 m/s
T  = 3.5 * m.Lx / m.c                      # (c+U)T = 4L, (c-U)T = 3L: both pulses return
x0 = 0.25 * m.Lx
p0 = [exp(-((x - x0) / (m.Lx / 16))^2) for x in m.xc]
u0 = zeros(m.nx)
F  = dft_matrix(m.nx)
pex, _ = exact_solution(m, p0, u0, T; F = F)

# ---------------------------------------------------------------- helpers
function run_lawson(order, n, nsub; snapshots = Int[])
    cf = Coeffs(order); ws = LawsonWS(n = m.nx)
    p = copy(p0); u = copy(u0); h = T / n
    snaps = Dict{Int,Vector{Float64}}()
    for k in 1:n
        lawson_rk4_step!(p, u, m, h, nsub, cf, ws)
        k in snapshots && (snaps[k] = copy(p))
    end
    return p, snaps
end

function run_etdrk4(order, n, nsub)
    cf = Coeffs(order); ws = ETDWS(n = m.nx)
    p = copy(p0); u = copy(u0); h = T / n
    for k in 1:n
        etdrk4_step!(p, u, m, h, nsub, cf, ws)
    end
    return p
end

# split-explicit RK3 (WRF), `nsub` forward-backward steps per h, `beta` = pressure extrapolation
function run_rk3(n, nsub; beta = 0.0)
    ws = RK3WS(n = m.nx)
    p = copy(p0); u = copy(u0); h = T / n
    for k in 1:n
        rk3_split_step!(p, u, m, h, nsub, ws; beta = beta)
    end
    return p
end

maxerr(p) = maximum(abs.(p .- pex))
stab(e) = (isfinite(e) && e < 1) ? e : NaN          # NaN = unstable

# ---------------------------------------------------------------- 1) pressure snapshots
# Lawson-RK4, fast order 4, h = 50 s (c h/dx = 7, U h/dx = 1), 8 substeps per h/2
n = 200
pT, snaps = run_lawson(4, n, 8; snapshots = [n ÷ 2, n])
open("data_pressure_1d.csv", "w") do io
    println(io, "x_km,p0,p_half,p_T")
    for j in 1:m.nx
        @printf(io, "%.6f,%.8e,%.8e,%.8e\n", m.xc[j] / 1e3, p0[j], snaps[n÷2][j], snaps[n][j])
    end
end
@printf("pressure: max|p(T)-p0| = %.2e, max p(T/2) = %.4f\n", maximum(abs.(snaps[n] .- p0)), maximum(snaps[n÷2]))

# ---------------------------------------------------------------- 2) convergence in h, FIXED acoustic substep
# tau = 1.5625 s is fixed by stability (c tau/dx = 0.22): substeps per h/2 = 16*200/n = 16, 8, 4, 2, 1
# split-explicit RK3: the same tau, i.e. 2*nsub forward-backward steps per h (stage 1: one step h/3, as in WRF)
open("data_conv_1d.csv", "w") do io
    println(io, "h,lawson_o2,lawson_o4,etdrk4_o4,rk3_fb,rk3_fb_pe")
    for n in (200, 400, 800, 1600, 3200)
        nsub = 16 * 200 ÷ n
        e2 = maxerr(run_lawson(2, n, nsub)[1])
        e4 = maxerr(run_lawson(4, n, nsub)[1])
        ee = maxerr(run_etdrk4(4, n, nsub))
        r0 = stab(maxerr(run_rk3(n, 2 * nsub)))
        r1 = stab(maxerr(run_rk3(n, 2 * nsub; beta = 0.1)))
        @printf(io, "%.6f,%.6e,%.6e,%.6e,%.6e,%.6e\n", T / n, e2, e4, ee, r0, r1)
        @printf("h = %7.3f   Lawson-RK4 (fast o2) %.2e   (fast o4) %.2e   ETDRK4 (fast o4) %.2e   RK3 split FB %.2e   FB+p-extrap. %.2e\n",
                T / n, e2, e4, ee, r0, r1)
    end
end

# ---------------------------------------------------------------- 3) work-precision: h = 50 s (n = 200) and h = 12.5 s (n = 800)
for (n, nsubs) in ((200, (5, 6, 8, 12, 16, 24, 32, 48)), (800, (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)))
    for order in (2, 4, 6)
        stages = length(Coeffs(order).b)
        open("data_work_n$(n)_o$(order).csv", "w") do io
            println(io, "cost,nsub,err")
            for nsub in nsubs
                e = maxerr(run_lawson(order, n, nsub)[1])
                (isfinite(e) && e < 1) || continue          # skip unstable (substep too large)
                @printf(io, "%d,%d,%.6e\n", 4 * nsub * stages, nsub, e)
            end
        end
    end
end
# split-explicit RK3 (WRF): cost = forward-backward steps per h = 1 + nsub/2 + nsub; NaN = unstable
for (n, nsubs) in ((200, (6, 12, 18, 24, 36, 48, 72, 96)), (800, (2, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96)))
    open("data_work_n$(n)_rk3.csv", "w") do io
        println(io, "cost,nsub,err_fb,err_fb_pe")
        for nsub in nsubs
            r0 = stab(maxerr(run_rk3(n, nsub)))
            r1 = stab(maxerr(run_rk3(n, nsub; beta = 0.1)))
            @printf(io, "%d,%d,%.6e,%.6e\n", 1 + nsub ÷ 2 + nsub, nsub, r0, r1)
        end
    end
end
# split-explicit RK3: order of the slow coupling, fast part well resolved (tau = 0.195 s)
open("data_conv_rk3.csv", "w") do io
    println(io, "h,nsub,rk3_fb,rk3_fb_pe")
    for n in (800, 1600, 3200, 6400, 12800)
        nsub = 64 * 800 ÷ n
        r0 = stab(maxerr(run_rk3(n, nsub)))
        r1 = stab(maxerr(run_rk3(n, nsub; beta = 0.1)))
        @printf(io, "%.6f,%d,%.6e,%.6e\n", T / n, nsub, r0, r1)
    end
end

# split-explicit RK3: max spectral radius over all Fourier modes, acoustic substep tau <= 1.5625 s.
# Stage 1 as in WRF (one step h/3) or uniform (nsub/3 steps tau, nsub divisible by 6).
using LinearAlgebra: eigvals
specrad(R) = maximum(r -> maximum(abs.(eigvals(r))), R)
open("data_rk3_stab.csv", "w") do io
    println(io, "h,rho_fb,rho_fb_pe,rho_fb_uni,rho_fb_pe_uni")
    for h in 2.0:0.25:80.0
        nw = 2 * ceil(Int, h / (2 * 1.5625))
        nu = 6 * ceil(Int, h / (6 * 1.5625))
        @printf(io, "%.2f,%.6f,%.6f,%.6f,%.6f\n", h,
                specrad(rk3_amplification(m, h, nw)), specrad(rk3_amplification(m, h, nw; beta = 0.1)),
                specrad(rk3_amplification(m, h, nu; first = :uniform)),
                specrad(rk3_amplification(m, h, nu; first = :uniform, beta = 0.1)))
    end
end
println("done")

# ================================================================ MIS methods
# MIS (Wensch-Knoth-Galant 2009, Knoth-Wensch 2014): RK3 (WS) in MIS form, MIS-KW3, MIS54 (Bauer-Knoth 2021)
function run_mis(name, n; tau, fast = Coeffs(4), beta = 0.0)
    mc = MISCoeffs(name); ws = MISWS(m.nx, mc)
    p = copy(p0); u = copy(u0); h = T / n
    for k in 1:n
        mis_step!(p, u, m, h, mc, ws; tau = tau, fast = fast, beta = beta)
    end
    return p
end

# convergence in h: fast part ~exact (composition order 6, tau = 0.5 s), and with tau = 1.5625 s:
# symmetric leapfrog (fast order 2) and the WRF forward-backward scheme
open("data_conv_mis.csv", "w") do io
    println(io, "h,rk3ws_ex,kw3_ex,mis54_ex,kw3_lf,mis54_lf,kw3_fb,mis54_fb")
    for n in (100, 200, 400, 800, 1600, 3200)
        e = Float64[]
        for nm in (:rk3ws, :kw3, :mis54)
            push!(e, stab(maxerr(run_mis(nm, n; tau = 0.5, fast = Coeffs(6)))))
        end
        for nm in (:kw3, :mis54)
            push!(e, stab(maxerr(run_mis(nm, n; tau = 1.5625, fast = Coeffs(2)))))
        end
        for nm in (:kw3, :mis54)
            push!(e, stab(maxerr(run_mis(nm, n; tau = 1.5625, fast = :fb))))
        end
        @printf(io, "%.6f,%s\n", T / n, join([@sprintf("%.6e", x) for x in e], ","))
        @printf("h = %7.3f   MIS exact fast: RK3 %.2e  KW3 %.2e  MIS54 %.2e   leapfrog: KW3 %.2e  MIS54 %.2e   FB: KW3 %.2e  MIS54 %.2e\n",
                T / n, e...)
    end
end

# Stability regions of the linear acoustics test of Knoth-Wensch (2014, eq. 10):
#   u_t + U u_x = -c_s Pi_x,  Pi_t + U Pi_x = -c_s u_x,
# staggered 2nd order acoustics, 3rd order upwind advection, exact fast integration, 40 Fourier modes.
function kw_specrad(mc, ca, cs; N = 40)
    r = 0.0
    for j in 1:N-1
        z = cis(2pi * j / N)
        a  = -cs * (z - 1)                                # Pi' = a u
        b  = -cs * (1 - 1/z)                              # u'  = b Pi
        nu = -ca / 6 * (2z + 3 - 6/z + 1/z^2)             # 3rd order upwind
        r = max(r, maximum(abs.(eigvals(mis_amplification_mode(a, b, nu, 1.0, mc)))))
    end
    return r
end
open("data_kw_stab.csv", "w") do io
    println(io, "cs,ca,rk3ws,kw3,mis54")
    mcs = [MISCoeffs(nm) for nm in (:rk3ws, :kw3, :mis54)]
    for cs in 0.0:0.2:12.0, ca in 0.0:0.05:2.2
        st = [kw_specrad(mc, ca, cs) <= 1 + 1e-10 ? 1 : 0 for mc in mcs]
        @printf(io, "%.2f,%.3f,%d,%d,%d\n", cs, ca, st...)
    end
end
println("done (MIS)")
