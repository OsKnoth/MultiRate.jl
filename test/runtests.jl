# Unit tests (fast).
#
#     julia --project -e 'using Pkg; Pkg.test()'

using MultiRate
using LinearAlgebra, Random, Test, SparseArrays

const E2 = MultiRate.Euler2D
const FV = MultiRate.FV2D
const TS = MultiRate.TimeStepping
const CM = MultiRate.Common
include(joinpath(@__DIR__, "..", "MethodDesign", "BiColourMIS.jl"))

rate(e1, e2) = log2(e1 / e2)

"""Bytes allocated by f() after a warm-up call (measured inside a function, not at global scope)."""
allocs(f) = (f(); @allocated f())

"""Finite-difference Jacobian of q ↦ f(q) at q0, and its largest real eigenvalue part."""
function jacobian_max_re(f, q0)
    f0 = f(q0)
    J = zeros(length(f0), length(q0))
    for k in eachindex(q0)
        h = 1e-7 * max(abs(q0[k]), 1.0); qp = copy(q0); qp[k] += h
        J[:, k] = (f(qp) .- f0) ./ h
    end
    return maximum(real.(eigvals(J)))
end

"""Toy problem for the frozen split: fast oscillator with a frozen coefficient c(y) (or c = 1,
additive), slow nonlinear part; N(q) = F_f(c(q), q) + S(q)."""
struct FrozenToy; additive::Bool; end
TS.frozen_ncoef(::FrozenToy) = 1
function TS.frozen_data!(C, S, m::FrozenToy, y)
    C[1] = m.additive ? 1.0 : 1 + 0.5sin(y[1])
    S[1] = 0.5sin(y[1]) + y[2]^2; S[2] = -0.3y[1] * y[2] + cos(y[1])
    return C, S
end
function TS.frozen_fast!(out, m::FrozenToy, C, z, part, F)
    out .= 0
    part !== :momentum && (out[1] = 4C[1] * z[2] + F[1])
    part !== :thermo && (out[2] = -4z[1] - 0.5z[1]^3 + F[2])
    return out
end

@testset "MultiRate" begin

@testset "Euler2D: x–z slice with terrain" begin
    gas = CM.Gas()
    ρr, Θr, θr = E2.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    # isothermal atmosphere (T = 250 K): exactly at rest with the logarithmic gravity mean
    # (:waruszewski, default) and no background state subtracted (Artiano et al.)
    Tiso = 250.0
    ρi(z) = 1e5 / (gas.R * Tiso) * exp(-gas.g * z / (gas.R * Tiso))
    Θi(z) = gas.p00 / gas.R * (gas.R * Tiso * ρi(z) / gas.p00)^(1 / gas.γ)
    for zs in (x -> 0.0, x -> 400.0 * exp(-((x - 10e3) / 2e3)^2))
        M = E2.Mesh2D(5, 3, 3, 20e3, 10e3; zs = zs)
        @test maximum(abs(sum(M.D[i, k] * M.Jξx[k, j, e] + M.D[j, k] * M.Jηx[i, k, e] for k in 1:M.np))
                      for i in 1:M.np, j in 1:M.np, e in 1:M.Ne) < 1e-9                    # metric identities
        m = E2.Euler2DModel(M, gas, ρr, Θr)
        qi = E2.project2d(m, (x, z) -> (ρi(z), 0.0, 0.0, Θi(z)))
        @test maximum(abs.(E2.rhs_nonlinear(m, qi))) < 1e-12                               # rest state (isothermal)
        qr = E2.project2d(m, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
        q = E2.project2d(m, (x, z) -> (ρr(z) * (1 - 0.01 * exp(-((x - 8e3)^2 + (z - 4e3)^2) / 2e3^2)),
                                       10.0 * ρr(z), 0.0, Θr(z)))
        Nq = E2.rhs_nonlinear(m, q)
        @test abs(E2.total2d(m, Nq, 1)) < 1e-12 * E2.total2d(m, q, 1)                       # conservation
        @test abs(E2.total2d(m, Nq, 4)) < 1e-12 * E2.total2d(m, q, 4)
        s = E2.fast_operator(m, qr; fast = :hevi)                                            # energy stable
        @test maximum(real.(eigvals(E2.fast_matrix(s)))) < 1e-12
        sv = E2.fast_operator(m, q; fast = :hevi, reuse = false)                              # HEVI column solver
        Av = zeros(m.nq, m.nq); ev = zeros(m.nq); col = zeros(m.nq)                          # = dense solve
        for k in 1:m.nq
            ev[k] = 1.0; E2.vertical_apply!(col, sv, ev); Av[:, k] .= col; ev[k] = 0.0
        end
        Ys = q .* (1 .+ 0.1 .* sin.(1:m.nq)); Y = similar(Ys)
        E2.hevi_solve!(Y, sv, E2.hevi_factor(sv, 3.0 * TS.IMEX_GM), Ys)   # factor of I - c A_v, c = dt γ
        @test norm(Y .- (I - 3.0 * TS.IMEX_GM * Av) \ Ys) < 1e-12 * norm(Y)
        s = E2.fast_operator(m, q)                                                           # bounded growth
        @test maximum(real.(eigvals(E2.fast_matrix(s)))) < 1e-3
    end
    # linear stability of N: at rest over terrain and with mean flow on a flat, very flat-element mesh
    M = E2.Mesh2D(4, 2, 3, 20e3, 10e3; zs = x -> 2000.0 / (1 + ((x - 10e3) / 3e3)^2))
    m = E2.Euler2DModel(M, gas, ρr, Θr)
    @test jacobian_max_re(q -> E2.rhs_nonlinear(m, q), E2.project2d(m, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))) < 1e-10
    M = E2.Mesh2D(4, 2, 3, 320e3, 10e3)
    m = E2.Euler2DModel(M, gas, ρr, Θr)
    @test jacobian_max_re(q -> E2.rhs_nonlinear(m, q), E2.project2d(m, (x, z) -> (ρr(z), 20ρr(z), 0.0, Θr(z)))) < 1e-7

    # MIS with explicit and HEVI fast part against RK4 on N (gravity-wave setup, 100 s)
    M = E2.Mesh2D(10, 2, 3, 100e3, 10e3)
    m = E2.Euler2DModel(M, gas, ρr, Θr)
    q0 = E2.project2d(m, (x, z) -> begin
        ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / 10e3) / (1 + ((x - 50e3) / 5e3)^2))
        (ρ, 20.0 * ρ, 0.0, Θr(z))
    end)
    c = E2.sound_speed_max(m, q0)
    dtr = 0.25 * 5e3 / ((c + 20) * 7)
    nref = ceil(Int, 100 / dtr)
    qref = TS.run_rk4(m, q0, 100 / nref, nref)
    dev = norm(qref .- q0)
    for (fast, Δts) in ((:imex, (4.0, 2.0)), (:hevi, (10.0, 5.0)))                   # nsub = 12: dτ refined with Δt
        e = [norm(TS.run_mis(m, q0, Δt, round(Int, 100 / Δt); tab = TS.mis_tableau(:MIS4), nsub = 12,
                             fast = fast) .- qref) for Δt in Δts]
        @test e[2] < 0.05 * dev
        @test rate(e...) > 1.7
    end
    qb = E2.project2d(m, (x, z) -> (ρi(z), 0.0, 0.0, Θi(z)))                                  # isothermal rest
    q = TS.run_mis(m, qb, 10.0, 3; tab = TS.mis_tableau(:MIS4), nsub = 12, fast = :hevi)
    @test maximum(abs.(q .- qb) ./ max.(abs.(qb), 1.0)) < 1e-10                              # well balanced

    # thermodynamic/momentum split with the interface momentum, Verlet and forward-backward
    let m = m, q0 = q0
        s2 = E2.fast_operator(m, q0; reuse = false)
        kf = zeros(m.nq); kT = zeros(m.nq); km = zeros(m.nq)
        TS.fast_apply!(kf, s2, q0, s2.fb)
        E2.interface_momentum!(s2, q0); E2.fast_rhs_thermo!(kT, s2, q0, s2.fb); E2.fast_rhs_momentum!(km, s2, q0, s2.fb)
        @test maximum(abs.(kT .+ km .- kf)) < 1e-12 * maximum(abs.(kf))
        Ms = E2.Mesh2D(4, 2, 3, 20e3, 10e3; zs = x -> 1000.0 / (1 + ((x - 10e3) / 3e3)^2))
        ms = E2.Euler2DModel(Ms, gas, ρr, Θr)
        qr = E2.project2d(ms, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
        ss = E2.fast_operator(ms, qr; reuse = false)
        hmin = minimum(Ms.Z[:, end, :] .- Ms.Z[:, 1, :])
        for (method, cfl) in ((:verlet, 0.7), (:fb, 0.5))          # one-step amplification ≤ 1
            dt = cfl * hmin / (7 * ss.cmax)
            A = zeros(ms.nq, ms.nq); e = zeros(ms.nq); y = zeros(ms.nq); F = -ss.fb ./ ss.Mdiag
            for k in 1:ms.nq
                e[k] = 1.0; TS.fast_solve!(y, ss, e, F, dt, 1; method = method); A[:, k] .= y; e[k] = 0.0
            end
            @test maximum(abs.(eigvals(A))) < 1 + 1e-10
        end
        # central interface (tau_scale = 0, Xu–van der Vegt–Bokhove): energy conserving operator,
        # Verlet step with all eigenvalues on the unit circle
        sc = E2.fast_operator(ms, qr; reuse = false, tau_scale = 0.0)
        λc = eigvals(E2.fast_matrix(sc))
        @test maximum(abs.(real.(λc))) < 1e-10 * maximum(abs.(λc))
        dt = 0.5 * hmin / (7 * sc.cmax)
        A = zeros(ms.nq, ms.nq); e = zeros(ms.nq); y = zeros(ms.nq); F = -sc.fb ./ sc.Mdiag
        for k in 1:ms.nq
            e[k] = 1.0; TS.fast_solve!(y, sc, e, F, dt, 1; method = :verlet); A[:, k] .= y; e[k] = 0.0
        end
        μ = abs.(eigvals(A))
        @test maximum(abs.(μ .- 1)) < 1e-8
        qq = copy(q0)
        @test allocs(() -> TS.fast_solve!(qq, s2, q0, nothing, 0.05, 5; method = :verlet)) == 0
        @test allocs(() -> TS.fast_solve!(qq, s2, q0, nothing, 0.05, 5; method = :fb)) == 0
        sh2 = E2.fast_operator(m, q0; fast = :hevi, reuse = false)
        @test allocs(() -> TS.fast_solve!(qq, sh2, q0, nothing, 0.5, 5; method = :hevi)) < 50_000
    end
    e = [norm(TS.run_mis(m, q0, Δt, round(Int, 100 / Δt); tab = TS.mis_tableau(:MIS4), nsub = 12,
                         fast = :verlet) .- qref) for Δt in (8.0, 4.0)]
    @test e[2] < 0.05 * dev
    @test rate(e...) > 1.7
    q = TS.run_mis(m, qb, 8.0, 3; tab = TS.mis_tableau(:MIS4), nsub = 12, fast = :verlet)
    @test maximum(abs.(q .- qb) ./ max.(abs.(qb), 1.0)) < 1e-10

    # allocation-free kernels (in a let block: the testset reassigns m, s, q, which would be boxed)
    let m = m, q0 = q0
        out = zeros(m.nq)
        @test allocs(() -> TS.rhs_nonlinear!(out, m, q0)) == 0
        sh = E2.fast_operator(m, q0; fast = :hevi)
        @test allocs(() -> TS.fast_apply!(out, sh, q0, sh.fb)) == 0
        qq = copy(q0)
        @test allocs(() -> TS.fast_solve!(qq, sh, q0, nothing, 0.1, 5; method = :imex)) == 0
        tab = TS.mis_tableau(:MIS4); w = TS.MISWork(m.nq, 4)
        @test allocs(() -> TS.mis_step!(qq, m, q0, 4.0, tab, sh, w; nsub = 12, fast = :imex)) == 0
        se = E2.fast_operator(m, q0; reuse = false)               # explicit only: no LU factors
        @test allocs(() -> E2.refreeze!(se, m, q0, 1.4)) == 0     # coefficients re-frozen in place
    end
end

@testset "FV2D: WRF-style C-grid" begin
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas)
    g = FV.FVGrid(8, 6, 40e3, 10e3)
    m = FV.FVModel(g, gas, ρr, Θr)
    qr = CM.project2d(m, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
    withref(f) = (FV.FV_HYDRO_REFERENCE[] = true; try f() finally FV.FV_HYDRO_REFERENCE[] = false end)
    mref = withref(() -> FV.FVModel(g, gas, ρr, Θr))
    @test maximum(abs.(TS.rhs_nonlinear(mref, qr))) == 0                                # rest state exact
    # background-free (default): the analytic hydrostatic state is balanced to second order
    res = map((6, 12)) do nz
        gz = FV.FVGrid(8, nz, 40e3, 10e3); mz = FV.FVModel(gz, gas, ρr, Θr)
        maximum(abs.(TS.rhs_nonlinear(mz, CM.project2d(mz, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z))))))
    end
    @test rate(res...) > 1.9
    q = CM.project2d(m, (x, z) -> (ρr(z) * (1 - 0.01 * exp(-((x - 15e3)^2 + (z - 4e3)^2) / 3e3^2)), 10ρr(z),
                                   0.3ρr(z) * sin(2π * x / 40e3), Θr(z)))
    Nq = TS.rhs_nonlinear(m, q)
    @test abs(CM.total2d(m, Nq, 1)) < 1e-12 * CM.total2d(m, q, 1)                        # conservation
    @test abs(CM.total2d(m, Nq, 4)) < 1e-12 * CM.total2d(m, q, 4)
    s = FV.fast_operator(m, q; reuse = false)
    kf, kT, km, kh, kv, kH, kt = (zeros(m.nq) for _ in 1:7)
    TS.fast_apply!(kf, s, q, s.fb)
    TS.fast_rhs_thermo!(kT, s, q, s.fb)
    TS.fast_rhs_momentum!(km, s, q, s.fb)
    TS.fast_rhs_horizontal!(kh, s, q, s.fb)
    TS.vertical_apply!(kv, s, q)
    TS.fast_rhs_hmomentum!(kH, s, q, s.fb)
    TS.fast_rhs_hthermo!(kt, s, q, s.fb)
    @test maximum(abs.(kT .+ km .- kf)) < 1e-14 * maximum(abs.(kf))                    # splits
    @test maximum(abs.(kh .+ kv .- kf)) < 1e-14 * maximum(abs.(kf))
    @test maximum(abs.(kH .+ kt .- kh)) < 1e-14 * maximum(abs.(kh))
    Av = zeros(m.nq, m.nq)
    e = zeros(m.nq)
    col = zeros(m.nq)
    for k in 1:m.nq
        e[k] = 1.0
        TS.vertical_apply!(col, s, e)
        Av[:, k] .= col
        e[k] = 0.0
    end
    Ys = q .* (1 .+ 0.1 .* sin.(1:m.nq))
    Y = similar(Ys)
    TS.vertical_solve!(Y, s, TS.vertical_factor(s, 2.0), Ys)
    @test norm(Y .- (I - 2.0 * Av) \ Ys) < 1e-12 * norm(Y)                                # tridiagonal solver
    sr = FV.fast_operator(m, qr; reuse = false)
    @test maximum(abs.(real.(eigvals(FV.fast_matrix(sr))))) < 1e-12                     # energy conserving at rest

    # gravity-wave setup: MIS against RK4 on N
    gw = FV.FVGrid(40, 10, 100e3, 10e3)
    # with the reference: the dz = 1 km imbalance of the background-free form would dominate
    mw = withref(() -> FV.FVModel(gw, gas, ρr, Θr))
    q0 = CM.project2d(mw, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / 10e3) / (1 + ((x - 50e3) / 5e3)^2));
                                     (ρ, 20ρ, 0.0, Θr(z))))
    c = CM.sound_speed_max(mw, q0)
    nref = ceil(Int, 200 / (0.3 * gw.dz / (c + 20)))
    qref = TS.run_rk4(mw, q0, 200 / nref, nref)
    dev = norm(qref .- q0)
    # :wrf is first order in dτ (off-centring β = 0.55), so with a fixed ratio nsub the order is 1
    for (tab, nsub, fast, lim, order) in ((:WSRK3, 6, :wrf, gw.dx, 1.0), (:MIS4, 12, :verlet, gw.dz, 2.0),
                                           (:MIS54, 12, :verlet, gw.dz, 2.0), (:LRK4, 12, :verlet, gw.dz, 2.0),
                                           (:EB4, 12, :verlet, gw.dz, 2.0), (:EPIRKW3b, 4, :verlet, gw.dz, 2.0),
                                           (:EAB3, 4, :verlet, gw.dz, 2.0))
        Δt1 = nsub * 0.8 * lim / c
        errs = [norm(TS.run_mis(mw, q0, Δt, round(Int, 200 / Δt); tab = TS.mis_tableau(tab), nsub = nsub,
                                fast = fast) .- qref) for Δt in (200 / ceil(200 / Δt1), 100 / ceil(200 / Δt1))]
        @test errs[2] < 0.05 * dev
        @test rate(errs...) > order - 0.3
    end
    qb = CM.project2d(mw, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
    qw = TS.run_mis(mw, qb, 20.0, 5; tab = TS.mis_tableau(:WSRK3), nsub = 6, fast = :wrf)
    @test maximum(abs.(qw .- qb) ./ max.(abs.(qb), 1.0)) < 1e-12                       # well balanced

    let mw = mw, q0 = q0                                                                # allocation-free
        out = zeros(mw.nq)
        @test allocs(() -> TS.rhs_nonlinear!(out, mw, q0)) == 0
        sw = FV.fast_operator(mw, q0)
        qq = copy(q0)
        @test allocs(() -> TS.fast_solve!(qq, sw, q0, nothing, 1.0, 5; method = :wrf)) == 0
        @test allocs(() -> TS.fast_solve!(qq, sw, q0, nothing, 1.0, 5; method = :verlet)) == 0
        w = TS.MISWork(mw.nq, 3)
        tab = TS.mis_tableau(:WSRK3)
        @test allocs(() -> TS.mis_step!(qq, mw, q0, 20.0, tab, sw, w; nsub = 6, fast = :wrf)) == 0
    end
    # cut cells, HEVI with the mask (cut cells and neighbours): the implicit matrix is the full
    # linear fast operator minus its explicit part; a run with the regular Δx fast step is stable
    gc = FV.CutCellFVGrid(32, 20, 64e3, 10e3; zs = x -> 400.0 / (1 + ((x - 32e3) / 4e3)^2))
    mc = FV.FVModel(gc, gas, ρr, Θr)
    qc = CM.project2d(mc, (x, z) -> (ρr(z), 10ρr(z), 0.0, Θr(z)))
    sc = FV.fast_operator(mc, qc; fast = :hevi, reuse = false)
    @test any(sc.maskU) && count(sc.maskU) < count(gc.FU .> 0)
    v = 1e-3 .* sin.(1:mc.nq); z0 = zeros(mc.nq)
    full = FV.cutcell_fast!(zeros(mc.nq), sc, v, z0, true, true, FV.CUT_ALL, FV.CUT_NONE)
    expl = FV.cutcell_fast!(zeros(mc.nq), sc, v, z0, true, false, FV.CUT_ALL, FV.CUT_NONE, true)
    @test maximum(abs, FV.cutcell_implicit_matrix(sc) * v - (full - expl)) < 1e-12 * maximum(abs, full)
    dτ = 0.85 * 0.92 * gc.dx / FV.sound_speed_max(mc, qc)
    qe = TS.run_mis(mc, qc, 30.0, 20; tab = TS.mis_tableau(:MIS4), nsub = ceil(Int, 30.0 / dτ), fast = :hevi, refreeze_every = 10)
    @test all(isfinite, qe) && abs(CM.total2d(mc, qe, 1) / CM.total2d(mc, qc, 1) - 1) < 1e-12
    # the two Jacobians (acoustics: columns + masked faces; advection: cut-cell mask), HEVI and
    # Rosenbrock-W with P = (I - c A_ac)(I - c A_adv)
    sa = FV.fast_operator(mc, qc; fast = :hevi_amf, reuse = false)
    @test !isempty(sa.Aadv.nzval) && length(sa.Aadv.nzval) < length(sc.Mdiag)   # advection in the cut-cell mask only
    for fast in (:hevi_amf, :rosenbrock_amf)
        qa = TS.run_mis(mc, qc, 30.0, 20; tab = TS.mis_tableau(:MIS4), nsub = ceil(Int, 30.0 / dτ), fast, refreeze_every = 10)
        @test all(isfinite, qa) && abs(CM.total2d(mc, qa, 1) / CM.total2d(mc, qc, 1) - 1) < 1e-12
        @test maximum(abs, qa - qe) < 0.1 * maximum(abs, qe - qc)    # another splitting: masked advection in L
    end
end

@testset "TimeStepping: MIS tableaus and order" begin
    # MIS-KW form reproduces the RK method for L = 0
    A3, b3 = [0 0 0; 1/3 0 0; -3/16 15/16 0], [1/6, 3/10, 8/15]
    A, b, c = TS.mis_rk_tableau(TS.mis_kw(:kw, A3, b3))
    @test A ≈ A3 && b ≈ b3
    @test TS.mis_tableau(:MIS_KW3).d ≈ [1/3, 5/12, 1/4]
    @test sum(TS.mis_tableau(:MIS54).d) ≈ 1.878 atol = 1e-3
    # classical RK order (L = 0) and observed order with exact fast stages and non-commuting L, G
    # (MIS3C: RK order 3, but only 2 with L ≠ 0: a coupling condition fails)
    for (name, prk, pmis) in ((:WSRK3, 2, 2), (:MIS2, 2, 2), (:MIS3C, 3, 2), (:MIS4, 3, 3), (:MIS4a, 3, 3),
                              (:MIS_KW3, 3, 3), (:MIS54, 4, 4))
        tab = TS.mis_tableau(name)
        @test TS.rk_order(TS.mis_rk_tableau(tab)[1:2]...) == prk
        _, r = TS.mis_observed_order(tab; n = (20, 40, 80))
        @test all(abs.(r .- pmis) .< 0.15)
    end
end

@testset "TimeStepping: Lawson RK with the fast integrators as propagator" begin
    LA = MultiRate.LinearAnalysis
    @test TS.fast_time(TS.mis_tableau(:LRK4)) ≈ 2.5
    @test TS.fast_time(TS.mis_tableau(:MIS4)) ≈ sum(TS.mis_tableau(:MIS4).d)
    @test_throws ArgumentError TS.LawsonTableau(:ssp3, [0 0 0; 1 0 0; 1/4 1/4 0], [1/6, 1/6, 2/3])  # c₃ < c₂
    for (name, p) in ((:LRK2, 2), (:LRK3, 3), (:LRK4, 4))
        _, r = TS.mis_observed_order(TS.mis_tableau(name); n = (40, 80, 160))
        @test abs(r[end] - p) < 0.3
    end
    # the production Verlet loop converges to the exact propagator (second order in the substep)
    m = LA.staggered_mode(0.9; C_sound = 3.0, C_adv = 0.4)
    tab = TS.mis_tableau(:LRK4)
    Rex = LA.step_matrix(m, 1.0, tab; fast = :exact)
    e = [norm(LA.step_matrix(m, 1.0, tab; nsub = n, fast = :verlet) - Rex) for n in (24, 48)]
    @test rate(e...) > 1.9
    # Yoshida/Suzuki compositions of the Verlet loop: orders 4 and 6
    @test TS.composition(:hevi_s4)[1] == :hevi && TS.composition_cost(:verlet_y6) == 7
    @test sum(TS.composition(:verlet_y4)[2]) ≈ 1 && sum(TS.composition(:verlet_y6)[2]) ≈ 1
    for (fast, p) in ((:verlet_s4, 4), (:verlet_y4, 4), (:verlet_y6, 6))
        e = [norm(LA.step_matrix(m, 1.0, tab; nsub = n, fast) - Rex) for n in (12, 24)]
        @test abs(log2(e[1] / e[2]) - p) < 0.2
    end
    # stable where MIS is: MISDG grid, C_sound = 7
    @test LA.max_amplification(θ -> LA.staggered_mode(θ; C_sound = 7.0, C_adv = 1.0), tab; fast = :verlet, nsub = 12) < 1
end

@testset "TimeStepping: ETD in stage-ODE form (polynomial slow forcing)" begin
    LA = MultiRate.LinearAnalysis
    tab = TS.mis_tableau(:EB4)
    @test TS.fast_time(tab) ≈ 3
    @test TS.rk_order(TS.mis_rk_tableau(tab)[1:2]...) == 4                    # Krogstad ETDRK4, L = 0
    for (name, p) in ((:EB1, 1), (:ETD2, 2), (:EB4, 4))
        _, r = TS.mis_observed_order(TS.mis_tableau(name); n = (20, 40, 80))
        @test abs(r[end] - p) < 0.15
    end
    # fast loops with time-dependent forcing: Verlet order 2, compositions order 4 and 6
    m = LA.staggered_mode(0.9; C_sound = 3.0, C_adv = 0.4)
    Rex = LA.step_matrix(m, 1.0, tab; fast = :exact)
    for (fast, p) in ((:verlet, 2), (:verlet_s4, 4), (:verlet_y6, 6))
        e = [norm(LA.step_matrix(m, 1.0, tab; nsub = n, fast) - Rex) for n in (12, 24)]
        @test abs(log2(e[1] / e[2]) - p) < 0.2
    end
end

@testset "TimeStepping: EPIRK(-W) and exponential Adams–Bashforth in stage-ODE form" begin
    LA = MultiRate.LinearAnalysis
    # orders with an arbitrary L (not the Jacobian): EPIRK-W 3, EAB k; classical EPIRK4 drops to 1
    for (name, p) in ((:EPIRKW3a, 3), (:EPIRKW3b, 3), (:EAB2, 2), (:EAB3, 3), (:EAB4, 4), (:EPIRK4, 1))
        _, r = TS.mis_observed_order(TS.mis_tableau(name); n = (20, 40, 80))
        @test abs(r[end] - p) < 0.15
    end
    # EPIRK4 with L = Jacobian of the full right-hand side: order 4
    L0, G0 = TS.ORDER_TEST_L, TS.order_test_G
    J(y) = L0 + [0.5cos(y[1]) 2y[2]; -0.3y[2]-sin(y[1]) -0.3y[1]]
    qref = TS._rk4_reference(L0, G0, [1.0, 0.5], 1.0, 20000)
    e4 = map((20, 40)) do m
        y = [1.0, 0.5]
        for _ in 1:m
            A = J(y); y = TS.mis_step_exact(A, x -> L0 * x + G0(x) - A * x, y, 1 / m, TS.mis_tableau(:EPIRK4))
        end
        norm(y - qref)
    end
    @test log2(e4[1] / e4[2]) > 3.6
    @test TS.fast_time(TS.mis_tableau(:EPIRKW3b)) ≈ 1 + 0.34706341174296320958
    @test TS.fast_time(TS.mis_tableau(:EAB3)) == 1
    # production stage-ODE step converges to the exact φ-functions (Suzuki loop: order 4)
    m = LA.staggered_mode(0.9; C_sound = 1.0, C_adv = 0.1)
    for name in (:EPIRKW3a, :EPIRKW3b, :EAB3)
        tab = TS.mis_tableau(name)
        Rex = LA.step_matrix(m, 1.0, tab; fast = :exact)
        e = [norm(LA.step_matrix(m, 1.0, tab; nsub = n, fast = :verlet_s4) - Rex) for n in (4, 8)]
        @test abs(log2(e[1] / e[2]) - 4) < 0.3
    end
    # stable for slow steps with small Courant numbers only (coupling with the fast acoustics)
    amp(name, ca) = LA.max_amplification(θ -> LA.staggered_mode(θ; C_sound = 7ca, C_adv = ca),
                                         TS.mis_tableau(name); fast = :exact)
    @test amp(:EPIRKW3a, 0.2) <= 1 + 1e-6 && amp(:EPIRKW3a, 0.5) > 1.5
    @test amp(:EAB3, 0.05) < 1.001 && amp(:EAB3, 0.2) > 1.5
    # multistep bookkeeping: the start-up steps, then EAB; reset! starts again
    tab = TS.mis_tableau(:EAB3)
    w = TS.split_work(tab, 4)
    @test w.nhist == 0
    sF = TS.fast_operator(m, zeros(4))
    q = [1.0, 0.0, 0.5, 0.0]
    for _ in 1:3
        TS.mis_step!(q, m, q, 1.0, tab, sF, w; nsub = 4, fast = :verlet)
    end
    @test w.nhist == 3
    @test TS.reset!(w).nhist == 0
    @test_throws ArgumentError LA.linearized_step(m, zeros(4), 1.0, tab)
end

@testset "TimeStepping: MIS with a frozen part of the right-hand side" begin
    # orders with exact fast stages (RK4, 400 substeps) against a fine RK4 solution
    function frozen_rates(m, name)
        Nf(q) = (C = zeros(1); S = zeros(2); TS.frozen_data!(C, S, m, q); TS.frozen_fast!(zeros(2), m, C, q, :all, S))
        q0 = [1.0, 0.5]; q = copy(q0); h = 1 / 4000
        for _ in 1:4000
            k1 = Nf(q); k2 = Nf(q + h / 2 * k1); k3 = Nf(q + h / 2 * k2); k4 = Nf(q + h * k3)
            q += h / 6 * (k1 + 2k2 + 2k3 + k4)
        end
        e = [norm(TS.run_frozen(m, q0, 1 / n, n; tab = TS.mis_tableau(name), nsub = 400, fast = :rk4) - q) for n in (20, 40)]
        return rate(e...)
    end
    # additive F(y, z) = S(y) + G(z): the MIS orders
    @test frozen_rates(FrozenToy(true), :MIS54) > 3.8
    @test frozen_rates(FrozenToy(true), :MIS4a) > 2.8
    # coefficient of the fast part frozen at the stage values: MIS54, MIS4a drop to 2, Knoth–Wolke keeps 3
    @test 1.8 < frozen_rates(FrozenToy(false), :MIS54) < 2.3
    @test 1.8 < frozen_rates(FrozenToy(false), :MIS4a) < 2.4
    @test frozen_rates(FrozenToy(false), :MIS_KW3) > 2.8
    # MISF53 (MIS54 projected onto the bicoloured conditions ≤ 3): frozen order 3
    @test frozen_rates(FrozenToy(false), :MISF53) > 2.8
    @test frozen_rates(FrozenToy(false), :MIS_FV53) > 2.8   # designed for the FV frozen split
    @test frozen_rates(FrozenToy(false), :MIS_DG53) > 2.8   # designed for the DG frozen split
    # Wicker–Skamarock RK3: order 2; Lawson and ETD with the fast part frozen at qⁿ: ETD keeps its
    # order with a nonlinear fast flow, Lawson drops to 2 (φ(X + ΔS) ≠ φ(X) + e^{τL} ΔS)
    for add in (true, false)
        @test 1.8 < frozen_rates(FrozenToy(add), :WSRK3) < 2.3
        @test frozen_rates(FrozenToy(add), :EB4) > 3.8
        @test 1.8 < frozen_rates(FrozenToy(add), :ETD2) < 2.3
        @test frozen_rates(FrozenToy(add), :ETD3) > 2.8
        @test 0.9 < frozen_rates(FrozenToy(add), :EB1) < 1.2
        @test frozen_rates(FrozenToy(add), :LRK4) > 1.8
        @test frozen_rates(FrozenToy(add), :LRK3) > 1.8
    end
    # FV (flat): F(q, q) = N(q), parts, gravity wave against RK4
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas)
    gw = FV.FVGrid(40, 10, 100e3, 10e3)
    # no background state (full p and ρ)
    mw = FV.FVModel(gw, gas, ρr, Θr)
    q0 = CM.project2d(mw, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * sin(π * z / 10e3) / (1 + ((x - 50e3) / 5e3)^2));
                                     (ρ, 20ρ, 0.0, Θr(z))))
    C = zeros(TS.frozen_ncoef(mw)); S = zeros(mw.nq); out = zeros(mw.nq)
    TS.frozen_data!(C, S, mw, q0)
    TS.frozen_fast!(out, mw, C, q0, :all, S)
    N0 = TS.rhs_nonlinear(mw, q0)
    @test maximum(abs.(out .- N0)) < 1e-14 + 1e-10 * maximum(abs, N0)
    parts = sum(TS.frozen_fast!(zeros(mw.nq), mw, C, q0, p, S) for p in (:hthermo, :hmomentum, :vertical))
    @test parts ≈ out
    @test TS.frozen_fast!(zeros(mw.nq), mw, C, q0, :thermo, S) + TS.frozen_fast!(zeros(mw.nq), mw, C, q0, :momentum, S) ≈ out
    c = CM.sound_speed_max(mw, q0)
    nref = ceil(Int, 200 / (0.3 * gw.dz / (c + 20)))
    qref = TS.run_rk4(mw, q0, 200 / nref, nref)
    dev = norm(qref .- q0)
    for (fast, nsub, lim) in ((:verlet, 12, gw.dz), (:hevi, 12, gw.dz))
        Δt1 = nsub * 0.8 * lim / c
        errs = [norm(TS.run_frozen(mw, q0, Δt, round(Int, 200 / Δt); tab = TS.mis_tableau(:MIS54), nsub, fast) .- qref)
                for Δt in (200 / ceil(200 / Δt1), 100 / ceil(200 / Δt1))]
        # background-free on this coarse grid (Δz = 1 km) the unbalanced rest state dominates the time
        # error: errors decrease, but slowly
        @test errs[2] < 0.1 * dev && errs[2] < errs[1]
    end
    # product-form scheme (FV_PRESSURE_FORM = :product): -(dp/dΘ)_f ∇Θ with the face mean of γ p/Θ,
    # no background state. Exact frozen split with dp/dΘ frozen, linear fast part, exact column
    # solve, converges to its own reference; rejects models with a subtracted reference
    try
        FV.FV_PRESSURE_FORM[] = :product
        mref = (FV.FV_HYDRO_REFERENCE[] = true; try FV.FVModel(gw, gas, ρr, Θr) finally FV.FV_HYDRO_REFERENCE[] = false end)
        @test_throws ErrorException TS.rhs_nonlinear(mref, q0)
        mp = mw
        N0p = TS.rhs_nonlinear(mp, q0)
        C = zeros(TS.frozen_ncoef(mp)); S = zeros(mp.nq)
        TS.frozen_data!(C, S, mp, q0)
        o = TS.frozen_fast!(zeros(mp.nq), mp, C, q0, :all, S)
        @test maximum(abs.(o .- N0p)) < 1e-14 + 1e-10 * maximum(abs, N0p)
        s = FV.frozen_stage_begin!(mp, C, q0)
        R = 1e-3 .* sin.(1:mp.nq); Y = zeros(mp.nq)
        FV._frozen_column_solve!(Y, s, mp, 3.0, R)
        Gv(x) = TS.frozen_fast!(zeros(mp.nq), mp, C, x, :vertical, S)
        @test maximum(abs.(Y .- 3.0 .* (Gv(q0 .+ Y) .- Gv(q0)) .- R)) < 1e-10 * maximum(abs, R)
        # time error of the frozen MIS54 run, background-free, the same for both schemes
        function frozen_err(form)
            FV.FV_PRESSURE_FORM[] = form
            qr = TS.run_rk4(mp, q0, 200 / nref, nref)
            Δt1 = 12 * 0.8 * gw.dz / c
            e = [norm(TS.run_frozen(mp, q0, Δt, round(Int, 200 / Δt); tab = TS.mis_tableau(:MIS54), nsub = 12, fast = :hevi) .- qr)
                 for Δt in (200 / ceil(200 / Δt1), 100 / ceil(200 / Δt1))]
            return e ./ norm(qr .- q0)
        end
        ep, eg = frozen_err(:product), frozen_err(:gradp)
        @test ep[2] < 0.1 && ep[2] < ep[1]
        @test isapprox(ep, eg; rtol = 0.1)
    finally
        FV.FV_PRESSURE_FORM[] = :gradp
    end
    # terrain following: :gradp frozen with p(Θ_z) is the Gal-Chen operator rhs_terrain!; :product
    # parts and exact column solve
    gt = FV.FVGrid(16, 10, 40e3, 10e3; zs = x -> 800 / (1 + ((x - 20e3) / 4e3)^2))
    bt = FV.FVModel(gt, gas, ρr, Θr)
    qt = CM.project2d(bt, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 40e3)); (ρ, 10ρ + sin(x / 5e3), 0.3 * sin(π * z / 10e3) * cos(x / 7e3), Θr(z) * (1 + 1e-3 * cos(x / 6e3)))))
    Ct = zeros(TS.frozen_ncoef(bt)); St = zeros(bt.nq)
    TS.frozen_data!(Ct, St, bt, qt)
    Nt = TS.rhs_nonlinear(bt, qt)
    @test maximum(abs, TS.frozen_fast!(zeros(bt.nq), bt, Ct, qt, :all, St) - Nt) < 1e-10 * maximum(abs, Nt)
    try
        FV.FV_PRESSURE_FORM[] = :product
        TS.frozen_data!(Ct, St, bt, qt)
        ot(p, z = qt) = TS.frozen_fast!(zeros(bt.nq), bt, Ct, z, p, St)
        Nt = ot(:all)
        @test maximum(abs, ot(:hthermo) + ot(:hmomentum) + ot(:vertical) - Nt) < 1e-13 * maximum(abs, Nt)
        vt = 1e-3 .* sin.(1:bt.nq)
        @test maximum(abs, ot(:all, qt .+ vt) + ot(:all, qt .- vt) - 2Nt) < 1e-11 * maximum(abs, Nt)
        st = FV.frozen_stage_begin!(bt, Ct, qt)
        Rt = 1e-3 .* sin.(1:bt.nq); Yt = zeros(bt.nq)
        FV._frozen_column_solve!(Yt, st, bt, 3.0, Rt)
        Gt(x) = TS.frozen_fast!(zeros(bt.nq), bt, Ct, x, :vertical, St)
        @test maximum(abs.(Yt .- 3.0 .* (Gt(qt .+ Yt) .- Gt(qt)) .- Rt)) < 1e-10 * maximum(abs, Rt)
    finally
        FV.FV_PRESSURE_FORM[] = :gradp
    end
end

@testset "FV2D: Exner pressure as an additional equation (frozen split)" begin
    gas = CM.Gas()
    try
        FV.FV_PRESSURE_FORM[] = :exner
        ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
        g = FV.FVGrid(12, 8, 12e3, 8e3); b = FV.FVModel(g, gas, ρr, Θr); m = FV.ExnerModel(b)
        q = CM.project2d(b, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 12e3)); (ρ, 10ρ + sin(x / 2e3), 0.5 * sin(π * z / 8e3) * cos(x / 3e3), Θr(z) * (1 + 1e-3 * cos(x / 2e3)))))
        qe = FV.exner_state(m, q)
        @test FV.base_state(m, qe) == q
        # on π = π(Θ): the base rows are the base operator with the :exner pressure gradient, the π
        # rows are (dπ/dΘ) times the Θ rows
        N = TS.rhs_nonlinear(b, q); Ne = TS.rhs_nonlinear(m, qe)
        @test maximum(abs, Ne[1:b.nq] - N) < 1e-12 * maximum(abs, N)
        @test maximum(abs(Ne[FV.iπ(m, i, k)] - (gas.γ - 1) * qe[FV.iπ(m, i, k)] / q[FV.iΘ(g, i, k)] * N[FV.iΘ(g, i, k)]) for i in 1:12, k in 1:8) <
              1e-12 * maximum(abs, Ne[b.nq+1:end])
        # parts; the fast part is linear in z; exact column solve
        C = zeros(TS.frozen_ncoef(m)); S = zeros(m.nq); TS.frozen_data!(C, S, m, qe)
        o(p, z = qe) = TS.frozen_fast!(zeros(m.nq), m, C, z, p, S)
        @test o(:all) == Ne
        @test maximum(abs, o(:thermo) + o(:momentum) - Ne) < 1e-13 * maximum(abs, Ne)
        @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - Ne) < 1e-13 * maximum(abs, Ne)
        v = 1e-2 .* sin.(1:m.nq)
        @test maximum(abs, o(:all, qe .+ v) + o(:all, qe .- v) - 2Ne) < 1e-11 * maximum(abs, Ne)
        TS.frozen_stage_begin!(m, C, qe)
        R = 1e-3 .* cos.(1:m.nq); Y = zeros(m.nq); FV.exner_column_solve!(Y, m, 3.0, R)
        G(z) = TS.frozen_fast!(zeros(m.nq), m, C, z, :vertical, S)
        @test maximum(abs.(Y .- 3.0 .* (G(qe .+ Y) .- G(qe)) .- R)) < 1e-10 * maximum(abs, R)
        # isentropic atmosphere: the analytic state is exactly balanced (no background state)
        ρi, Θi, θi = CM.stratified_reference(gas; θ0 = 300.0, N = 0.0)
        bi = FV.FVModel(g, gas, ρi, Θi); mi = FV.ExnerModel(bi)
        qi = FV.exner_state(mi, CM.project2d(bi, (x, z) -> (ρi(z), 0.0, 0.0, Θi(z))))
        @test maximum(abs, TS.rhs_nonlinear(mi, qi)) < 1e-11
        # frozen MIS (π recomputed after every step): converges to RK4 of the base model,
        # mass and Θ conserved, π = π(Θ) at the end
        q0 = CM.project2d(bi, (x, z) -> (θ = θi(z) + 0.5 * exp(-((x - 6e3)^2 + (z - 3e3)^2) / 1.5e3^2); (Θi(z) / θ, 0.0, 0.0, Θi(z))))
        c = FV.sound_speed_max(bi, q0); T = 60.0
        qr = TS.run_rk4(bi, q0, T / 600, 600)
        errs = map((6, 12)) do n
            Δt = T / n
            qn = TS.run_frozen(mi, FV.exner_state(mi, q0), Δt, n; tab = TS.mis_tableau(:MIS54), nsub = ceil(Int, Δt / (0.7 * g.dx / c)), fast = :hevi)
            @test maximum(abs(qn[FV.iπ(mi, i, k)] - FV.exner(gas, qn[FV.iΘ(g, i, k)])) for i in 1:12, k in 1:8) == 0
            @test abs(CM.total2d(bi, qn, 1) / CM.total2d(bi, q0, 1) - 1) < 1e-13
            @test abs(sum(qn[FV.iΘ(g, i, k)] for i in 1:12, k in 1:8) / sum(q0[FV.iΘ(g, i, k)] for i in 1:12, k in 1:8) - 1) < 1e-13
            norm(FV.base_state(mi, qn) - qr) / norm(qr - q0)
        end
        @test errs[2] < 0.03 && errs[2] < 0.7 * errs[1]
        # divergence damping by extrapolation (smdiv): β = 0 is the undamped loop; with β = 0.1 still
        # convergent to the same solution, mass conserved
        let n = 12, Δt = T / 12, nsub = ceil(Int, Δt / (0.7 * g.dx / c)), x0 = FV.exner_state(mi, q0)
            q00 = TS.run_frozen(mi, x0, Δt, n; tab = TS.mis_tableau(:MIS54), nsub, fast = :hevi)
            q0s = TS.run_frozen(mi, x0, Δt, n; tab = TS.mis_tableau(:MIS54), nsub, fast = :hevi, smdiv = 0.0)
            @test q0s == q00
            for fast in (:hevi, :verlet)
                qd = TS.run_frozen(mi, x0, Δt, n; tab = TS.mis_tableau(:MIS54), nsub = fast === :verlet ? 4nsub : nsub, fast, smdiv = 0.1)
                @test abs(CM.total2d(bi, qd, 1) / CM.total2d(bi, q0, 1) - 1) < 1e-13
                @test norm(FV.base_state(mi, qd) - qr) / norm(qr - q0) < 0.05
            end
        end
        @test_throws ErrorException TS.frozen_data!(zeros(TS.frozen_ncoef(bi)), zeros(bi.nq), bi, q0)   # the base frozen split has no :exner form
    finally
        FV.FV_PRESSURE_FORM[] = :gradp
    end
end

@testset "FV2D: pressure as the additional variable (pvar = :p), dry and moist" begin
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    g = FV.FVGrid(12, 8, 12e3, 8e3)
    b = FV.FVModel(g, gas, ρr, Θr)                                   # FV_PRESSURE_FORM = :gradp (δp)
    m = FV.ExnerModel(b; pvar = :p)
    q = CM.project2d(b, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 12e3)); (ρ, 10ρ + sin(x / 2e3), 0.5 * sin(π * z / 8e3) * cos(x / 3e3), Θr(z) * (1 + 1e-3 * cos(x / 2e3)))))
    qe = FV.exner_state(m, q)
    @test qe[FV.iπ(m, 2, 3)] ≈ CM.pressure(gas, q[FV.iΘ(g, 2, 3)])
    N = TS.rhs_nonlinear(b, q); Ne = TS.rhs_nonlinear(m, qe)
    @test maximum(abs, Ne[1:b.nq] - N) < 1e-12 * maximum(abs, N)      # the base operator with δp(Θ)
    @test maximum(abs(Ne[FV.iπ(m, i, k)] - gas.γ * qe[FV.iπ(m, i, k)] / q[FV.iΘ(g, i, k)] * N[FV.iΘ(g, i, k)]) for i in 1:12, k in 1:8) <
          1e-12 * maximum(abs, Ne[b.nq+1:end])                          # p_t = (γ p/Θ) Θ_t
    C = zeros(TS.frozen_ncoef(m)); S = zeros(m.nq); TS.frozen_data!(C, S, m, qe)
    o(p, z = qe) = TS.frozen_fast!(zeros(m.nq), m, C, z, p, S)
    @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - Ne) < 1e-13 * maximum(abs, Ne)
    v = 1e-2 .* sin.(1:m.nq)
    @test maximum(abs, o(:all, qe .+ v) + o(:all, qe .- v) - 2Ne) < 1e-11 * maximum(abs, Ne)
    TS.frozen_stage_begin!(m, C, qe)
    R = 1e-3 .* cos.(1:m.nq); Y = zeros(m.nq); FV.exner_column_solve!(Y, m, 3.0, R)
    G(z) = TS.frozen_fast!(zeros(m.nq), m, C, z, :vertical, S)
    @test maximum(abs.(Y .- 3.0 .* (G(qe .+ Y) .- G(qe)) .- R)) < 1e-10 * maximum(abs, R)
    # moist with q_t = 0 and pvar = :p is the dry pvar = :p model (gas with Rd, cpd)
    mg = FV.MoistGas()
    gm = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0)
    bm = FV.FVModel(g, gm, ρr, Θr); md = FV.ExnerModel(bm; pvar = :p); mm = FV.MoistExnerModel(bm; mgas = mg, pvar = :p, micro = :slow)
    Nd = TS.rhs_nonlinear(md, FV.exner_state(md, q)); Nm = TS.rhs_nonlinear(mm, FV.moist_exner_state(mm, q, zeros(12, 8)))
    @test maximum(abs, Nm[1:bm.nq] - Nd[1:bm.nq]) < 1e-9 * maximum(abs, Nd)
    @test maximum(abs(Nm[FV.iπ(mm, i, k)] - Nd[FV.iπ(md, i, k)]) for i in 1:12, k in 1:8) < 1e-6 * maximum(abs, Nd[bm.nq+1:end])
    # moist state: parts, exact column solve, p = p_eos after the step end
    x = FV.moist_exner_state(mm, q, [0.015 * q[FV.iρ(g, i, k)] for i in 1:12, k in 1:8])
    Cm = zeros(TS.frozen_ncoef(mm)); Sm = zeros(mm.nq); TS.frozen_data!(Cm, Sm, mm, x)
    om(p) = TS.frozen_fast!(zeros(mm.nq), mm, Cm, x, p, Sm)
    Nx = TS.rhs_nonlinear(mm, x)
    @test maximum(abs, om(:hthermo) + om(:hmomentum) + om(:vertical) - Nx) < 1e-13 * maximum(abs, Nx)
    TS.frozen_stage_begin!(mm, Cm, x)
    Rm = 1e-3 .* cos.(1:mm.nq); Ym = zeros(mm.nq); FV.moist_column_solve!(Ym, mm, 3.0, Rm)
    Gm(z) = TS.frozen_fast!(zeros(mm.nq), mm, Cm, z, :vertical, Sm)
    @test maximum(abs.(Ym .- 3.0 .* (Gm(x .+ Ym) .- Gm(x)) .- Rm)) < 1e-10 * maximum(abs, Rm)
    xs = copy(x); xs[FV.iπ(mm, 3, 3)] += 1.0; TS.frozen_step_end!(mm, xs)
    @test xs == x
end

@testset "FV2D: moist air, Exner pressure (rho theta_rho, rho q_v, rho q_c, Fischer-Burmeister relaxation)" begin
    mg = FV.MoistGas()
    # equation of state: saturated parcels exactly at saturation, unsaturated ones without condensate
    ρ = 1.1; Θ = 300.0 * ρ
    T, ρv, ρc, p = FV.moist_state(mg, ρ, Θ, 0.02ρ)
    @test ρc > 0 && abs(ρv * mg.Rv * T / FV.psat(mg, T) - 1) < 1e-12
    @test FV.moist_state(mg, ρ, Θ, 0.002ρ)[3] == 0
    # p = p0 (Rd Θ/p0)^(1/(1-κ_M))
    ρd = ρ - 0.02ρ; κ = (mg.Rd * ρd + mg.Rv * ρv) / (mg.Cpd * ρd + mg.Cpv * ρv + mg.Cpl * ρc)
    @test p ≈ mg.p0 * (mg.Rd * Θ / mg.p0)^(1 / (1 - κ)) rtol = 1e-12
    # dry limit: the moist model with q_t = 0 is the dry ExnerModel (gas with Rd, cpd)
    gas = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    g = FV.FVGrid(12, 8, 12e3, 8e3)
    try
        FV.FV_PRESSURE_FORM[] = :exner
        b = FV.FVModel(g, gas, ρr, Θr); md = FV.ExnerModel(b); mm = FV.MoistExnerModel(b; mgas = mg, micro = :slow)
        q = CM.project2d(b, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 12e3)); (ρ, 10ρ + sin(x / 2e3), 0.5 * sin(π * z / 8e3) * cos(x / 3e3), Θr(z) * (1 + 1e-3 * cos(x / 2e3)))))
        qd = FV.exner_state(md, q); qm = FV.moist_exner_state(mm, q, zeros(12, 8))
        Nd = TS.rhs_nonlinear(md, qd); Nm = TS.rhs_nonlinear(mm, qm)
        @test maximum(abs, Nm[1:b.nq] - Nd[1:b.nq]) < 1e-9 * maximum(abs, Nd)
        @test maximum(abs(Nm[FV.iπ(mm, i, k)] - Nd[FV.iπ(md, i, k)]) for i in 1:12, k in 1:8) < 1e-6 * maximum(abs, Nd[b.nq+1:end])
        # moist state: parts, linear fast part, exact column solve, π = π_eos after the step end
        Q = [0.015 * q[FV.iρ(g, i, k)] for i in 1:12, k in 1:8]
        x = FV.moist_exner_state(mm, q, Q)
        C = zeros(TS.frozen_ncoef(mm)); S = zeros(mm.nq); TS.frozen_data!(C, S, mm, x)
        o(p, z = x) = TS.frozen_fast!(zeros(mm.nq), mm, C, z, p, S)
        N = TS.rhs_nonlinear(mm, x)
        @test o(:all) == N
        @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - N) < 1e-13 * maximum(abs, N)
        @test maximum(abs, o(:thermo) + o(:momentum) - N) < 1e-13 * maximum(abs, N)
        v = 1e-3 .* sin.(1:mm.nq)
        @test maximum(abs, o(:all, x .+ v) + o(:all, x .- v) - 2N) < 1e-11 * maximum(abs, N)
        TS.frozen_stage_begin!(mm, C, x)
        R = 1e-3 .* cos.(1:mm.nq); Y = zeros(mm.nq); FV.moist_column_solve!(Y, mm, 3.0, R)
        G(z) = TS.frozen_fast!(zeros(mm.nq), mm, C, z, :vertical, S)
        @test maximum(abs.(Y .- 3.0 .* (G(x .+ Y) .- G(x)) .- R)) < 1e-10 * maximum(abs, R)
        xs = copy(x); xs[FV.iπ(mm, 3, 3)] += 1e-3; TS.frozen_step_end!(mm, xs)
        @test xs == x
        # Fischer–Burmeister phase change: zero in equilibrium, latent heating at fixed θ_e,
        # the same tendency for micro = :slow and :fast
        @test maximum(abs(FV.fischer_burmeister(mg, x[FV.iρ(g, i, k)], x[FV.iΘ(g, i, k)], x[FV.iV(mm, i, k)], x[FV.iC(mm, i, k)])[1]) for i in 1:12, k in 1:8) <
              1e-15 * maximum(x[FV.iρ(g, i, k)] for i in 1:12, k in 1:8)
        ρ1, Θ1, Q1, c1 = x[FV.iρ(g, 2, 3)], x[FV.iΘ(g, 2, 3)], x[FV.iV(mm, 2, 3)], x[FV.iC(mm, 2, 3)]
        T1, ρv1, _, _ = FV.moist_state(mg, ρ1, Θ1, Q1, c1)
        p1 = FV.moist_state(mg, ρ1, Θ1, Q1, c1)[4]
        θS = FV.theta_phase_factor(mg, ρ1, Θ1, T1, p1, ρv1, c1)          # Bryan–Fritsch prefactor
        @test θS > 0
        # through the equation of state: isochoric first law dT/dρc = (L - Rv T)/c_vm, and the
        # closed-form pressure source = chain rule (∂p/∂Θ) θS + ∂p/∂ρc - ∂p/∂ρv
        h = 1e-6 * ρ1
        st(s) = FV.moist_state(mg, ρ1, Θ1 + s * θS * h, Q1 - s * h, c1 + s * h)
        ρd1 = ρ1 - Q1 - c1; Rm1 = mg.Rd * ρd1 + mg.Rv * ρv1
        cvm1 = mg.Cpd * ρd1 + mg.Cpv * ρv1 + mg.Cpl * c1 - Rm1
        @test (st(1)[1] - st(-1)[1]) / 2h ≈ (FV.latent(mg, T1) - mg.Rv * T1) / cvm1 rtol = 1e-6
        @test (st(1)[4] - st(-1)[4]) / 2h ≈ FV.pressure_phase_factor(mg, ρ1, T1, ρv1, c1) rtol = 1e-6
        xsup = copy(x); xsup[FV.iV(mm, 2, 3)] += 0.5c1; xsup[FV.iC(mm, 2, 3)] -= 0.5c1   # supersaturated cell
        mf = FV.MoistExnerModel(b; mgas = mg, micro = :fast)
        Ns, Nf = TS.rhs_nonlinear(mm, xsup), TS.rhs_nonlinear(mf, xsup)
        @test maximum(abs, Ns - Nf) < 1e-12 * maximum(abs, Ns)
        @test mf.τ ≈ 0.25 * 0.85 * 0.92 * g.dx / 350                # default: a quarter of the HEVI acoustic step
        Nn = TS.rhs_nonlinear(FV.MoistExnerModel(b; mgas = mg, micro = :none), xsup)
        Sc = Ns[FV.iC(mm, 2, 3)] - Nn[FV.iC(mm, 2, 3)]
        @test Sc > 0 && abs(Ns[FV.iV(mm, 2, 3)] - Nn[FV.iV(mm, 2, 3)] + Sc) < 1e-12 * Sc           # condensation, vapour sink
        # frozen RK3 (Wicker–Skamarock), Lawson and ETD steps of the moist model (fast phase change),
        # supersaturated start: against RK4, error decreasing with Δt, conserved mass and water
        xr = TS.run_rk4(mf, xsup, 0.05, 1200)
        tot(y, f) = sum(y[f(i, k)] for i in 1:12, k in 1:8)
        for name in (:WSRK3, :LRK4, :EB4)
            e = map((10.0, 5.0)) do Δt
                y = TS.run_frozen(mf, xsup, Δt, round(Int, 60 / Δt); tab = TS.mis_tableau(name), nsub = ceil(Int, Δt / 0.5), fast = :hevi)
                @test abs(tot(y, (i, k) -> FV.iρ(g, i, k)) / tot(xsup, (i, k) -> FV.iρ(g, i, k)) - 1) < 1e-13
                @test abs((tot(y, (i, k) -> FV.iV(mm, i, k)) + tot(y, (i, k) -> FV.iC(mm, i, k))) /
                          (tot(xsup, (i, k) -> FV.iV(mm, i, k)) + tot(xsup, (i, k) -> FV.iC(mm, i, k))) - 1) < 1e-13
                norm(y[1:b.nq] - xr[1:b.nq]) / norm(xr[1:b.nq] - xsup[1:b.nq])
            end
            @test e[2] < e[1] && e[2] < 0.05
        end
    finally
        FV.FV_PRESSURE_FORM[] = :gradp
    end
end

@testset "FV2D: Exner and moist frozen split on the terrain-following grid" begin
    gas = CM.Gas(); mg = FV.MoistGas()
    gm = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0)
    ρr, Θr, θr = CM.stratified_reference(gm; θ0 = 300.0, N = 0.01)
    g = FV.FVGrid(16, 10, 40e3, 10e3; zs = x -> 800 / (1 + ((x - 20e3) / 4e3)^2))
    @test g.terrain
    b = FV.FVModel(g, gm, ρr, Θr)
    q = CM.project2d(b, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 40e3)); (ρ, 10ρ + sin(x / 5e3), 0.3 * sin(π * z / 10e3) * cos(x / 7e3), Θr(z) * (1 + 1e-3 * cos(x / 6e3)))))
    # pvar = :p with p = P(Θ): the frozen operator is the Gal-Chen operator of the base model
    # (same fluxes Jx U and Ω = W - z_x Ū, same metric pressure gradient)
    try
        FV.FV_PRESSURE_FORM[] = :gradp; FV.FV_TERRAIN_FORM[] = :flux
        mp = FV.ExnerModel(b; pvar = :p)
        xp = FV.exner_state(mp, q)
        Nb = TS.rhs_nonlinear(b, q); Np = TS.rhs_nonlinear(mp, xp)
        @test maximum(abs, Np[1:b.nq] - Nb) < 1e-10 * maximum(abs, Nb)
    finally
        FV.FV_PRESSURE_FORM[] = :gradp
    end
    for pvar in (:pi, :p)
        md = FV.ExnerModel(b; pvar); mm = FV.MoistExnerModel(b; mgas = mg, pvar, micro = :slow)
        xd = FV.exner_state(md, q)
        # dry limit of the moist model on terrain
        xm0 = FV.moist_exner_state(mm, q, zeros(16, 10))
        Nd = TS.rhs_nonlinear(md, xd); Nm0 = TS.rhs_nonlinear(mm, xm0)
        @test maximum(abs, Nm0[1:b.nq] - Nd[1:b.nq]) < 1e-9 * maximum(abs, Nd)
        # moist: parts, linear fast part, exact column solve
        x = FV.moist_exner_state(mm, q, [0.012 * q[FV.iρ(g, i, k)] for i in 1:16, k in 1:10])
        C = zeros(TS.frozen_ncoef(mm)); S = zeros(mm.nq); TS.frozen_data!(C, S, mm, x)
        o(p, z = x) = TS.frozen_fast!(zeros(mm.nq), mm, C, z, p, S)
        N = TS.rhs_nonlinear(mm, x)
        @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - N) < 1e-13 * maximum(abs, N)
        @test maximum(abs, o(:thermo) + o(:momentum) - N) < 1e-13 * maximum(abs, N)
        v = 1e-3 .* sin.(1:mm.nq)
        @test maximum(abs, o(:all, x .+ v) + o(:all, x .- v) - 2N) < 1e-11 * maximum(abs, N)
        TS.frozen_stage_begin!(mm, C, x)
        R = 1e-3 .* cos.(1:mm.nq); Y = zeros(mm.nq); FV.moist_column_solve!(Y, mm, 3.0, R)
        Gv(z) = TS.frozen_fast!(zeros(mm.nq), mm, C, z, :vertical, S)
        @test maximum(abs.(Y .- 3.0 .* (Gv(x .+ Y) .- Gv(x)) .- R)) < 1e-10 * maximum(abs, R)
        # dry model: exact column solve on terrain
        Cd = zeros(TS.frozen_ncoef(md)); Sd = zeros(md.nq); TS.frozen_data!(Cd, Sd, md, xd)
        TS.frozen_stage_begin!(md, Cd, xd)
        Rd = 1e-3 .* cos.(1:md.nq); Yd = zeros(md.nq); FV.exner_column_solve!(Yd, md, 3.0, Rd)
        Gd(z) = TS.frozen_fast!(zeros(md.nq), md, Cd, z, :vertical, Sd)
        @test maximum(abs.(Yd .- 3.0 .* (Gd(xd .+ Yd) .- Gd(xd)) .- Rd)) < 1e-10 * maximum(abs, Rd)
        # mass and total water conserved by the frozen step (fluxes J-weighted: Σ J ρ)
        y = TS.run_frozen(mm, x, 2.0, 3; tab = TS.mis_tableau(:MIS54), nsub = 4, fast = :hevi)
        tot(z, f) = sum(g.Jc[i] * z[f(i, k)] for i in 1:16, k in 1:10)
        @test abs(tot(y, (i, k) -> FV.iρ(g, i, k)) / tot(x, (i, k) -> FV.iρ(g, i, k)) - 1) < 1e-13
        @test abs((tot(y, (i, k) -> FV.iV(mm, i, k)) + tot(y, (i, k) -> FV.iC(mm, i, k))) /
                  (tot(x, (i, k) -> FV.iV(mm, i, k)) + tot(x, (i, k) -> FV.iC(mm, i, k))) - 1) < 1e-13
    end
end

@testset "FV2D: moist air with the internal energy or the moist entropy as thermodynamic variable (thvar = $thv)" for thv in (:energy, :entropy)
    mgθ = FV.MoistGas(); mge = FV.MoistGas(thvar = thv); mgE = FV.MoistGas(thvar = :energy)
    # equation of state: the same (T, p) from ρθ_ρ and from ρe, phase equilibrium, isochoric first law
    ρ = 1.1; Θ = 300.0 * ρ
    T, ρv, ρc, p = FV.moist_state(mgθ, ρ, Θ, 0.02ρ)
    E = FV._Θof(mge, ρ, T, ρv, ρc)
    Te, ρve, ρce, pe = FV.moist_state(mge, ρ, E, 0.02ρ)
    @test Te ≈ T rtol = 1e-12
    @test ρve ≈ ρv rtol = 1e-10
    @test pe ≈ p rtol = 1e-12
    @test FV._Θof(mgθ, ρ, T, ρv, ρc) ≈ Θ rtol = 1e-13
    h = 1e-6 * ρ
    ρd = ρ - ρv - ρc; Rm = mgθ.Rd * ρd + mgθ.Rv * ρv; cvm = mgθ.Cpd * ρd + mgθ.Cpv * ρv + mgθ.Cpl * ρc - Rm
    Ee = FV._Θof(mgE, ρ, T, ρv, ρc)
    st(s) = FV.moist_state(mgE, ρ, Ee, ρv - s * h, ρc + s * h)              # condensation at fixed ρ and ρe
    @test (st(1)[1] - st(-1)[1]) / 2h ≈ (FV.latent(mge, T) - mge.Rv * T) / cvm rtol = 1e-6
    @test (st(1)[4] - st(-1)[4]) / 2h ≈ FV.pressure_phase_factor(mge, ρ, T, ρv, ρc) rtol = 1e-6
    if thv === :energy
        @test FV.theta_phase_factor(mge, ρ, E, T, p, ρv, ρc) == 0
    else
        # dρs/dρ_c along the isochoric, energy-conserving phase change; zero at saturation at T0
        Xs(s) = FV._Θof(mge, ρ, st(s)[1], ρv - s * h, ρc + s * h)
        @test (Xs(1) - Xs(-1)) / 2h ≈ FV.theta_phase_factor(mge, ρ, E, T, p, ρv, ρc) rtol = 1e-6
        ρv0 = FV.psat(mge, mge.T0) / (mge.Rv * mge.T0)
        @test abs(FV.theta_phase_factor(mge, 1.0, 0.0, mge.T0, 0.0, ρv0, 1e-4)) < 1e-12
        # the dry limit ρ_v = 0 is regular
        @test FV.moist_state(mge, ρ, FV._Θof(mge, ρ, 290.0, 0.0, 0.0), 0.0, 0.0)[1] ≈ 290.0 rtol = 1e-13
    end
    gas = CM.Gas(R = mgθ.Rd, γ = mgθ.Cpd / (mgθ.Cpd - mgθ.Rd), p00 = mgθ.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    for g in (FV.FVGrid(12, 8, 12e3, 8e3), FV.FVGrid(16, 10, 40e3, 10e3; zs = x -> 800 / (1 + ((x - 20e3) / 4e3)^2))), pvar in (:pi, :p)
        nx, nz = g.nx, g.nz
        b = FV.FVModel(g, gas, ρr, Θr)
        q = CM.project2d(b, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / g.Lx)); (ρ, 10ρ + sin(x / 2e3), 0.5 * sin(π * z / g.H) * cos(x / 3e3), Θr(z) * (1 + 1e-3 * cos(x / 2e3)))))
        Q = [0.012 * q[FV.iρ(g, i, k)] for i in 1:nx, k in 1:nz]
        mθ = FV.MoistExnerModel(b; mgas = mgθ, pvar, micro = :fast); me = FV.MoistExnerModel(b; mgas = mge, pvar, micro = :fast)
        xθ = FV.moist_exner_state(mθ, q, Q)
        x = FV.thermo_convert(mθ, me, xθ)
        # the same temperature, pressure variable and water; converting back reproduces the state
        @test maximum(abs, x[FV.iπ(me, 1, 1):end] - xθ[FV.iπ(mθ, 1, 1):end]) < 1e-12 * maximum(abs, xθ[FV.iπ(mθ, 1, 1):end])
        @test maximum(abs, FV.thermo_convert(me, mθ, x) - xθ) < 1e-11 * maximum(abs, xθ)
        C = zeros(TS.frozen_ncoef(me)); S = zeros(me.nq); TS.frozen_data!(C, S, me, x)
        o(p, z = x) = TS.frozen_fast!(zeros(me.nq), me, C, z, p, S)
        N = TS.rhs_nonlinear(me, x)
        @test o(:all) == N
        @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - N) < 1e-13 * maximum(abs, N)
        # ρ, U, W, V, C rows as with ρθ_ρ (the thermodynamic variable enters only its own row and the pressure row)
        Nθ = TS.rhs_nonlinear(mθ, xθ)
        for f in ((i, k) -> FV.iρ(g, i, k), (i, k) -> FV.iU(g, i, k), (i, k) -> FV.iV(me, i, k), (i, k) -> FV.iC(me, i, k))
            @test maximum(abs(N[f(i, k)] - Nθ[f(i, k)]) for i in 1:nx, k in 1:nz) <= 1e-10 * maximum(abs(Nθ[f(i, k)]) for i in 1:nx, k in 1:nz)
        end
        # pressure row: the same pressure equation up to the discretization of the two forms
        rel = maximum(abs(N[FV.iπ(me, i, k)] - Nθ[FV.iπ(mθ, i, k)]) for i in 1:nx, k in 1:nz) / maximum(abs(Nθ[FV.iπ(mθ, i, k)]) for i in 1:nx, k in 1:nz)
        @test rel < 0.05
        # exact column solve (the vertical part is linear; χ per face and cell)
        TS.frozen_stage_begin!(me, C, x)
        R = 1e-3 .* cos.(1:me.nq); Y = zeros(me.nq); FV.moist_column_solve!(Y, me, 3.0, R)
        G(z) = TS.frozen_fast!(zeros(me.nq), me, C, z, :vertical, S)
        @test maximum(abs.(Y .- 3.0 .* (G(x .+ Y) .- G(x)) .- R)) < 1e-9 * maximum(abs, R)
        # frozen steps (MIS, Lawson, ETD; Verlet EI): mass and water conserved, close to RK4
        xr = TS.run_rk4(me, x, 0.05, 400)
        tot(y, f) = sum(g.Jc[i] * y[f(i, k)] for i in 1:nx, k in 1:nz)
        for name in (:MIS54, :LRK4, :EB4)
            y = TS.run_frozen(me, x, 5.0, 4; tab = TS.mis_tableau(name), nsub = 10, fast = :ei, offcenter = 0.1)
            @test abs(tot(y, (i, k) -> FV.iρ(g, i, k)) / tot(x, (i, k) -> FV.iρ(g, i, k)) - 1) < 1e-13
            @test abs((tot(y, (i, k) -> FV.iV(me, i, k)) + tot(y, (i, k) -> FV.iC(me, i, k))) /
                      (tot(x, (i, k) -> FV.iV(me, i, k)) + tot(x, (i, k) -> FV.iC(me, i, k))) - 1) < 1e-13
            @test norm(y[1:b.nq] - xr[1:b.nq]) < 0.1 * norm(xr[1:b.nq] - x[1:b.nq])
        end
    end
end

@testset "FV2D: gravity explicit (FV_GRAVITY_IMPLICIT[] = false), moist model, terrain, metric term implicit" begin
    mg = FV.MoistGas()
    gas = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    g = FV.FVGrid(16, 10, 40e3, 10e3; zs = x -> 800 / (1 + ((x - 20e3) / 4e3)^2))
    b = FV.FVModel(g, gas, ρr, Θr)
    q = CM.project2d(b, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / g.Lx)); (ρ, 10ρ + sin(x / 2e3), 0.5 * sin(π * z / g.H) * cos(x / 3e3), Θr(z) * (1 + 1e-3 * cos(x / 2e3)))))
    Q = [0.012 * q[FV.iρ(g, i, k)] for i in 1:16, k in 1:10]
    try
        FV.FV_METRIC_IMPLICIT[] = true
        m = FV.MoistExnerModel(b; mgas = mg, pvar = :pi, micro = :fast)
        x = FV.moist_exner_state(m, q, Q)
        C = zeros(TS.frozen_ncoef(m)); S = zeros(m.nq); TS.frozen_data!(C, S, m, x)
        o(p, z = x) = TS.frozen_fast!(zeros(m.nq), m, C, z, p, S)
        N = TS.rhs_nonlinear(m, x)
        V1 = o(:vertical); H1 = o(:hmomentum)
        y1 = TS.run_frozen(m, x, 5.0, 4; tab = TS.mis_tableau(:MIS54), nsub = 10, fast = :cs, offcenter = 0.1, smdiv = 0.1)
        FV.FV_GRAVITY_IMPLICIT[] = false
        @test TS.rhs_nonlinear(m, x) ≈ N rtol = 1e-14
        @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - N) < 1e-13 * maximum(abs, N)
        # gravity moved from the vertical to the explicit momentum part
        Gw = FV.gravity_w!(zeros(m.nq), m, x)
        @test maximum(abs, V1 - o(:vertical) - Gw) < 1e-12 * maximum(abs, V1)
        @test maximum(abs, o(:hmomentum) - H1 - Gw) < 1e-12 * maximum(abs, Gw)
        # column solve of the vertical part without gravity: exact with the explicit metric term (with the
        # implicit one the matrix uses the column-local 1 + z_x², an approximate Jacobian)
        FV.FV_METRIC_IMPLICIT[] = false
        TS.frozen_data!(C, S, m, x); TS.frozen_stage_begin!(m, C, x)
        R = 1e-3 .* cos.(1:m.nq); Y = zeros(m.nq); FV.moist_column_solve!(Y, m, 3.0, R, 2.0)
        G(z) = TS.frozen_fast!(zeros(m.nq), m, C, z, :vertical, S)
        D(v) = FV._scale_tw!(copy(v), g, 3.0, 2.0)
        @test maximum(abs.(Y .- D(G(x .+ Y) .- G(x)) .- R)) < 1e-9 * maximum(abs, R)
        FV.FV_METRIC_IMPLICIT[] = true; TS.frozen_data!(C, S, m, x)
        # the small step :cs and :csfb with explicit gravity: close to implicit gravity and to RK4
        xr = TS.run_rk4(m, x, 0.05, 400)
        for fast in (:cs, :csfb)
            y = TS.run_frozen(m, x, 5.0, 4; tab = TS.mis_tableau(:MIS54), nsub = 10, fast, offcenter = 0.1, smdiv = 0.1)
            @test all(isfinite, y)
            @test norm(y[1:b.nq] - xr[1:b.nq]) < 0.1 * norm(xr[1:b.nq] - x[1:b.nq])
            fast === :cs && @test norm(y[1:b.nq] - y1[1:b.nq]) < 0.05 * norm(xr[1:b.nq] - x[1:b.nq])
        end
    finally
        FV.FV_GRAVITY_IMPLICIT[] = true
        FV.FV_METRIC_IMPLICIT[] = false
    end
end

@testset "Fast substeps per stage: no fast step longer than Δt/nsub" begin
    @test TS.fast_substeps(0.4704, 3) == 2            # MIS2 stage 3 (round gave 1 step of 0.47 Δt)
    @test TS.fast_substeps(0.5, 12) == 6 && TS.fast_substeps(1 / 3, 12) == 4 && TS.fast_substeps(0.01, 12) == 1
    for M in (:MIS2, :MIS3C, :MIS4a, :MIS54), nsub in 1:40
        tab = TS.mis_tableau(M)
        @test all(tab.d ./ TS.stage_substeps(tab, nsub) .<= 1 / nsub + 1e-12)
    end
end

@testset "FV2D: Verlet EI and WRF fast integrators (Imex.pdf version 1 EI; KSD 2007)" begin
    gas = CM.Gas(); mg = FV.MoistGas()
    gm = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0)
    ρr, Θr, θr = CM.stratified_reference(gm; θ0 = 300.0, N = 0.01)
    g = FV.FVGrid(16, 10, 40e3, 10e3; zs = x -> 800 / (1 + ((x - 20e3) / 4e3)^2))
    b = FV.FVModel(g, gm, ρr, Θr)
    q = CM.project2d(b, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 40e3)); (ρ, 10ρ + sin(x / 5e3), 0.3 * sin(π * z / 10e3) * cos(x / 7e3), Θr(z) * (1 + 1e-3 * cos(x / 6e3)))))
    D(v, ct, cw) = FV._scale_tw!(copy(v), g, ct, cw)
    try
        FV.FV_TERRAIN_FORM[] = :flux
        for form in (:product, :gradp, :exner, :pres, :moist)
            FV.FV_PRESSURE_FORM[] = form === :product ? :product : :gradp
            m = form === :exner ? FV.ExnerModel(b; pvar = :pi) : form === :pres ? FV.ExnerModel(b; pvar = :p) :
                form === :moist ? FV.MoistExnerModel(b; mgas = mg, pvar = :pi, micro = :slow) : b
            x = form in (:exner, :pres) ? FV.exner_state(m, q) :
                form === :moist ? FV.moist_exner_state(m, q, [0.012 * q[FV.iρ(g, i, k)] for i in 1:16, k in 1:10]) : copy(q)
            n = length(x)
            C = zeros(TS.frozen_ncoef(m)); S = zeros(n); TS.frozen_data!(C, S, m, x)
            TS.frozen_stage_begin!(m, C, x)
            Gv(z) = TS.frozen_fast!(zeros(n), m, C, z, :vertical, S)
            # two-factor column solve: Y - D J_v Y = R, D = 2 (thermodynamic rows), 5 (W rows)
            if form !== :gradp           # vertical part linear: exact
                R = 1e-3 .* cos.(1:n); Y = zeros(n); FV._vsolve!(Y, m, 2.0, 5.0, R)
                @test maximum(abs.(Y .- D(Gv(x .+ Y) .- Gv(x), 2.0, 5.0) .- R)) < 1e-9 * maximum(abs, R)
            end
            # second order of the EI fast loop (against RK4 with small steps)
            T = 8.0
            run(meth, ns) = TS.frozen_fast_solve!(copy(x), m, C, S, T / ns, ns, zeros(n); method = meth,
                                                  tmp = (zeros(n), zeros(n), zeros(n)))
            xr = run(:rk4, 400)
            e1, e2 = maximum(abs, run(:ei, 4) - xr), maximum(abs, run(:ei, 8) - xr)
            @test log2(e1 / e2) > (form === :gradp ? 1.8 : form === :pres ? 1.6 : 1.9)   # :pres pre-asymptotic at 4 steps
            # merged half steps (EI_MERGE) = separate half steps: exact for the linear vertical parts
            for γ in (0.5, 0.6)
                TS.EI_GAMMA[] = γ
                TS.EI_MERGE[] = false; zu = run(:ei, 16)
                TS.EI_MERGE[] = true;  zm = run(:ei, 16)
                @test maximum(abs, zu - zm) < (form === :gradp ? 1e-5 : 1e-13) * maximum(abs, x)
            end
            TS.EI_GAMMA[] = 0.5
            # WRF forward-backward, vertically implicit: first order (conjugate to Verlet), ε = 0.1
            runo(meth, ns; ε = 0.0, sd = 0.0) = TS.frozen_fast_solve!(copy(x), m, C, S, T / ns, ns, zeros(n); method = meth,
                                                                     tmp = ntuple(_ -> zeros(n), 4), offcenter = ε, smdiv = sd)
            e1, e2 = maximum(abs, runo(:wrf, 16; ε = 0.1) - xr), maximum(abs, runo(:wrf, 32; ε = 0.1) - xr)
            @test 0.85 < log2(e1 / e2) < 1.2
            e1, e2 = maximum(abs, runo(:ei, 16; ε = 0.1) - xr), maximum(abs, runo(:ei, 32; ε = 0.1) - xr)
            @test 0.85 < log2(e1 / e2) < 1.2
            # EI and WRF with the same off-centring ε and extrapolation γd are conjugate:
            # WRF^n E = E EI^n, E the explicit step (1-β)h of the thermodynamic + vertical part
            if form !== :gradp
                for (ε, sd) in ((0.0, 0.0), (0.1, 0.0), (0.1, 0.1)), merge in (true, false)
                    TS.EI_MERGE[] = merge
                    hw = (1 - (1 + ε) / 2) * T / 16
                    Ec(z) = z .+ hw .* TS.frozen_fast!(zeros(n), m, C, z, :hthermo, S) .+ TS.frozen_vertical_rhs!(zeros(n), m, C, z, hw, hw)
                    zw = TS.frozen_fast_solve!(Ec(x), m, C, S, T / 16, 16, zeros(n); method = :wrf, tmp = ntuple(_ -> zeros(n), 4), offcenter = ε, smdiv = sd)
                    @test maximum(abs, zw - Ec(runo(:ei, 16; ε, sd))) < 1e-12 * maximum(abs, x)
                end
                TS.EI_MERGE[] = true
            end
        end
        # purely vertical problem (flat grid, horizontally uniform, U = 0): EI = Crank–Nicolson = :hevi
        FV.FV_PRESSURE_FORM[] = :product
        gf = FV.FVGrid(8, 20, 40e3, 10e3); bf = FV.FVModel(gf, gm, ρr, Θr)
        qf = CM.project2d(bf, (x, z) -> (ρ = ρr(z); (ρ, 0.0, 0.5 * sin(π * z / 10e3), Θr(z) * (1 + 1e-3 * sin(2π * z / 10e3)))))
        nf = length(qf); Cf = zeros(TS.frozen_ncoef(bf)); Sf = zeros(nf); TS.frozen_data!(Cf, Sf, bf, qf); fill!(Sf, 0.0)
        TS.frozen_stage_begin!(bf, Cf, qf)
        tm = (zeros(nf), zeros(nf), zeros(nf))
        ze = TS.frozen_fast_solve!(copy(qf), bf, Cf, Sf, 20.0, 3, zeros(nf); method = :ei, tmp = tm)
        zh = TS.frozen_fast_solve!(copy(qf), bf, Cf, Sf, 20.0, 3, zeros(nf); method = :hevi, tmp = tm)
        @test maximum(abs, ze - zh) < 1e-12 * maximum(abs, qf)
        # WRF without off-centring: the same Crank–Nicolson step
        zw = TS.frozen_fast_solve!(copy(qf), bf, Cf, Sf, 20.0, 3, zeros(nf); method = :wrf, tmp = tm)
        @test maximum(abs, zw - zh) < 1e-12 * maximum(abs, qf)
        # vertical acoustic Courant number c Δτ/Δz = 35: bounded
        @test all(isfinite, ze) && maximum(abs, ze[[FV.iW(gf, i, k) for i in 1:8 for k in 2:20]]) < 2
        # linearized path (FVFast): second order of fast_solve! with :ei, same limit as :hevi_verlet
        FV.FV_PRESSURE_FORM[] = :gradp
        s = TS.fast_operator(b, q; fast = :ei)
        Fs = 1e-4 .* sin.(1:length(q))
        sol(meth, ns) = TS.fast_solve!(zeros(length(q)), s, q, Fs, 8.0 / ns, ns; method = meth)
        xr = sol(:ei, 512)
        @test log2(maximum(abs, sol(:ei, 4) - xr) / maximum(abs, sol(:ei, 8) - xr)) > 1.8
        @test maximum(abs, sol(:hevi_verlet, 512) - xr) < 1e-2 * maximum(abs, sol(:ei, 4) - xr)
        # MIS with the EI fast loop: mass conserved on terrain, close to :hevi
        md = FV.ExnerModel(b; pvar = :pi); xd = FV.exner_state(md, q)
        ye = TS.run_frozen(md, xd, 2.0, 3; tab = TS.mis_tableau(:MIS2), nsub = 4, fast = :ei)
        yh = TS.run_frozen(md, xd, 2.0, 3; tab = TS.mis_tableau(:MIS2), nsub = 4, fast = :hevi)
        tot(z) = sum(g.Jc[i] * z[FV.iρ(g, i, k)] for i in 1:16, k in 1:10)
        @test abs(tot(ye) / tot(xd) - 1) < 1e-13
        @test maximum(abs, ye - yh) < 1e-2 * maximum(abs, ye - xd)          # O(Δτ²): both second order
        yw = TS.run_frozen(md, xd, 2.0, 3; tab = TS.mis_tableau(:MIS2), nsub = 4, fast = :wrf, smdiv = 0.1, offcenter = 0.1)
        @test abs(tot(yw) / tot(xd) - 1) < 1e-13
        @test maximum(abs, yw - yh) < 0.2 * maximum(abs, yw - xd)           # O(Δτ): first order
        # off-centring γ = 0.6 (version III): consistent, first order in Δτ
        TS.EI_GAMMA[] = 0.6
        e1, e2 = maximum(abs, sol(:ei, 16) - xr), maximum(abs, sol(:ei, 32) - xr)
        @test 0.8 < log2(e1 / e2) < 1.3
    finally
        FV.FV_PRESSURE_FORM[] = :gradp; TS.EI_GAMMA[] = 0.5; TS.EI_MERGE[] = true
    end
end

@testset "FV2D: implicit metric term, Ω column solve (FV_METRIC_IMPLICIT)" begin
    mg = FV.MoistGas()
    gm = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0)
    ρr, Θr, θr = CM.stratified_reference(gm; θ0 = 300.0, N = 0.01)
    g = FV.FVGrid(16, 10, 40e3, 10e3; zs = x -> 800 / (1 + ((x - 20e3) / 4e3)^2))
    b = FV.FVModel(g, gm, ρr, Θr)
    q = CM.project2d(b, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 40e3)); (ρ, 10ρ + sin(x / 5e3), 0.3 * sin(π * z / 10e3) * cos(x / 7e3), Θr(z) * (1 + 1e-3 * cos(x / 6e3)))))
    try
        FV.FV_TERRAIN_FORM[] = :flux; FV.FV_PRESSURE_FORM[] = :gradp
        for form in (:exner, :pres, :moist)
            m = form === :exner ? FV.ExnerModel(b; pvar = :pi) : form === :pres ? FV.ExnerModel(b; pvar = :p) :
                FV.MoistExnerModel(b; mgas = mg, pvar = :pi, micro = :slow)
            x = form === :moist ? FV.moist_exner_state(m, q, [0.012 * q[FV.iρ(g, i, k)] for i in 1:16, k in 1:10]) : FV.exner_state(m, q)
            n = length(x); C = zeros(TS.frozen_ncoef(m)); S = zeros(n); TS.frozen_data!(C, S, m, x)
            FV.FV_METRIC_IMPLICIT[] = false
            a0 = TS.frozen_fast!(zeros(n), m, C, x, :all, S)
            FV.FV_METRIC_IMPLICIT[] = true
            # the operator is unchanged, only its split into parts
            @test maximum(abs, TS.frozen_fast!(zeros(n), m, C, x, :all, S) - a0) < 1e-13 * maximum(abs, a0)
            Z0 = zeros(n)
            parts = sum(TS.frozen_fast!(zeros(n), m, C, x, p, Z0) for p in (:hthermo, :hmomentum, :vertical))
            @test maximum(abs, parts - TS.frozen_fast!(zeros(n), m, C, x, :all, Z0)) < 1e-13 * maximum(abs, a0)
            # EI with the metric predictor: second order; WRF first order
            TS.frozen_stage_begin!(m, C, x)
            run(meth, ns) = TS.frozen_fast_solve!(copy(x), m, C, S, 8.0 / ns, ns, zeros(n); method = meth, tmp = ntuple(_ -> zeros(n), 4))
            xr = run(:rk4, 400)
            e1, e2 = maximum(abs, run(:ei, 16) - xr), maximum(abs, run(:ei, 32) - xr)
            @test log2(e1 / e2) > 1.9
            e1, e2 = maximum(abs, run(:wrf, 16) - xr), maximum(abs, run(:wrf, 32) - xr)
            @test 0.9 < log2(e1 / e2) < 1.2
        end
        @test_throws ErrorException TS.frozen_fast!(zeros(b.nq), b, zeros(TS.frozen_ncoef(b)), copy(q), :all, zeros(b.nq))
        # steep terrain (max |z_x| = 0.6, c Δτ |z_x|/Δz = 2.3): explicit metric term unstable, implicit bounded
        gs = FV.FVGrid(32, 40, 32e3, 10e3; zs = x -> 2000 / (1 + ((x - 16e3) / 2e3)^2))
        bs = FV.FVModel(gs, gm, ρr, Θr)
        qs = CM.project2d(bs, (x, z) -> (ρ = ρr(z); (ρ, 10ρ, 0.0, Θr(z))))
        ms = FV.ExnerModel(bs; pvar = :p); xs = FV.exner_state(ms, qs)
        ns = length(xs); Cs = zeros(TS.frozen_ncoef(ms)); Ss = zeros(ns); TS.frozen_data!(Cs, Ss, ms, xs)
        dτ = 0.85 * 0.92 * gs.dx / (1.05 * FV.sound_speed_max(bs, qs))
        δ = 1e-6 .* maximum(abs, xs) .* sin.(0.37 .* (1:ns))
        for imp in (false, true), meth in (:ei, :wrf)
            FV.FV_METRIC_IMPLICIT[] = imp
            TS.frozen_stage_begin!(ms, Cs, xs)
            runs(z0) = TS.frozen_fast_solve!(copy(z0), ms, Cs, Ss, dτ, 200, zeros(ns); method = meth, tmp = ntuple(_ -> zeros(ns), 4),
                                              offcenter = 0.1, smdiv = 0.1)
            growth = maximum(abs, runs(xs .+ δ) .- runs(xs)) / maximum(abs, δ)
            @test imp ? growth < 1e4 : !(growth < 1e10)
        end
    finally
        FV.FV_METRIC_IMPLICIT[] = false; FV.FV_PRESSURE_FORM[] = :gradp
    end
end

@testset "FV3D: 3D C-grid, cut cells, frozen split" begin
    F3 = MultiRate.FV3D
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas)
    try
        # flat 3D with ny = 1 and V = 0 is the 2D FV operator (both pressure forms)
        for form in (:gradp, :product)
            FV.FV_PRESSURE_FORM[] = form; F3.FV3D_PRESSURE_FORM[] = form
            m2 = FV.FVModel(FV.FVGrid(20, 8, 100e3, 10e3), gas, ρr, Θr); g2 = m2.grid
            q2 = CM.project2d(m2, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 100e3)); (ρ, 20ρ, 2.0 * sin(π * z / 10e3), Θr(z))))
            g3 = F3.Grid3D(20, 1, 8, 100e3, 5e3, 10e3); m3 = F3.Model3D(g3, gas, ρr, Θr)
            q3 = zeros(m3.nq)
            for k in 1:8, i in 1:20
                q3[F3.iρ(g3, i, 1, k)] = q2[FV.iρ(g2, i, k)]; q3[F3.iΘ(g3, i, 1, k)] = q2[FV.iΘ(g2, i, k)]
                q3[F3.iU(g3, i, 1, k)] = q2[FV.iU(g2, i, k)]; k > 1 && (q3[F3.iW(g3, i, 1, k)] = q2[FV.iW(g2, i, k)])
            end
            N2 = TS.rhs_nonlinear(m2, q2); N3 = TS.rhs_nonlinear(m3, q3)
            @test maximum(abs(N3[F(g3, i, 1, k)] - N2[G(g2, i, k)]) for (F, G, ks) in
                          ((F3.iρ, FV.iρ, 1:8), (F3.iU, FV.iU, 1:8), (F3.iW, FV.iW, 2:8), (F3.iΘ, FV.iΘ, 1:8)) for k in ks for i in 1:20) <
                  1e-13 * maximum(abs, N2)
        end
        FV.FV_PRESSURE_FORM[] = :gradp
        # common advection (pseudo third order, shift technique): κ = 1/3 on uniform grids, the
        # centred mean at a wall; terrain following with ny = 1 is the 2D terrain-following operator
        @test FV.recon3(1.0, 2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0) ≈ -1 / 6 + 5 / 6 * 2 + 4 / 3
        @test FV.recon3(NaN, 2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, false) ≈ 3.0
        F3.FV3D_PRESSURE_FORM[] = :gradp
        ridge(x) = 400.0 / (1 + ((x - 50e3) / 8e3)^2)
        mr2 = FV.FVModel(FV.FVGrid(20, 8, 100e3, 10e3; zs = ridge), gas, ρr, Θr); gr2 = mr2.grid
        gr3 = F3.Grid3D(20, 1, 8, 100e3, 5e3, 10e3; zs = (x, y) -> ridge(x), terrain = :follow); mr3 = F3.Model3D(gr3, gas, ρr, Θr)
        qr2 = CM.project2d(mr2, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / 100e3)); (ρ, 10ρ + sin(x / 7e3), 0.3 * sin(π * z / 10e3) * cos(x / 9e3), Θr(z) * (1 + 1e-4 * cos(x / 6e3)))))
        qr3 = zeros(mr3.nq)
        for k in 1:8, i in 1:20
            qr3[F3.iρ(gr3, i, 1, k)] = qr2[FV.iρ(gr2, i, k)]; qr3[F3.iΘ(gr3, i, 1, k)] = qr2[FV.iΘ(gr2, i, k)]
            qr3[F3.iU(gr3, i, 1, k)] = qr2[FV.iU(gr2, i, k)]; k > 1 && (qr3[F3.iW(gr3, i, 1, k)] = qr2[FV.iW(gr2, i, k)])
        end
        Nr2 = TS.rhs_nonlinear(mr2, qr2); Nr3 = TS.rhs_nonlinear(mr3, qr3)
        for (F, G, ks) in ((F3.iρ, FV.iρ, 1:8), (F3.iU, FV.iU, 1:8), (F3.iW, FV.iW, 2:8), (F3.iΘ, FV.iΘ, 1:8))
            @test maximum(abs(Nr3[F(gr3, i, 1, k)] - Nr2[G(gr2, i, k)]) for k in ks for i in 1:20) <
                  1e-11 * maximum(abs(Nr2[G(gr2, i, k)]) for k in ks for i in 1:20)
        end
        # cut cells (hill): parts, conservation, exact column solve, flat limit
        F3.FV3D_PRESSURE_FORM[] = :product
        hill(x, y) = 1500.0 * exp(-((x - 10e3)^2 + (y - 10e3)^2) / 3e3^2)
        gc = F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = hill); mc = F3.Model3D(gc, gas, ρr, Θr; implicit = :column)
        @test count(0 .< gc.Vf .< 1) > 0
        q = F3.project3d(mc, (x, y, z) -> (ρr(z) * (1 + 1e-3 * sin(x / 3e3)), 10ρr(z), 5ρr(z), 0.1ρr(z) * sin(π * z / 6e3), Θr(z)))
        C = zeros(TS.frozen_ncoef(mc)); S = zeros(mc.nq); TS.frozen_data!(C, S, mc, q)
        o(p) = TS.frozen_fast!(zeros(mc.nq), mc, C, q, p, S)
        N = TS.rhs_nonlinear(mc, q)
        @test o(:all) == N
        @test maximum(abs, o(:thermo) + o(:momentum) - N) < 1e-14 * maximum(abs, N)
        @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - N) < 1e-14 * maximum(abs, N)
        @test abs(F3.total3d(mc, N, :ρ)) < 1e-14 * F3.total3d(mc, q, :ρ)
        @test abs(F3.total3d(mc, N, :Θ)) < 1e-14 * F3.total3d(mc, q, :Θ)
        s = TS.frozen_stage_begin!(mc, C, q)
        R = 1e-3 .* sin.(1:mc.nq); Y = zeros(mc.nq)
        F3.column_solve!(Y, s, mc, 3.0, R)
        Gv(x) = TS.frozen_fast!(zeros(mc.nq), mc, C, x, :vertical, S)
        @test maximum(abs.(Y .- 3.0 .* (Gv(q .+ Y) .- Gv(q)) .- R)) < 1e-9 * maximum(abs, R)
        # recon = :lsq: least-squares quadratic near cut cells, exact for quadratics in z (means over
        # the true fluid parts of cells and faces)
        gq = F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = hill, recon = :lsq)
        @test length(gq.lsq) == count(gq.mask .& (gq.Vf .> 0))
        φq = [(m = F3._cell_moments(gq, hill, nothing, i, j, k); gq.zc[k]^2 + 2gq.zc[k] * gq.dz * m[4] + gq.dz^2 * m[7])
              for i in 1:8, j in 1:8, k in 1:6]
        eq = maximum(begin
                fm = F3._face_moments(gq, hill, nothing, i, j, k, (f + 1) ÷ 2, isodd(f) ? -1 : 1)
                all(iszero, fm) ? 0.0 : abs(F3.lsq_face(gq, φq, i, j, k, f) - (gq.zc[k]^2 + 2gq.zc[k] * gq.dz * fm[3] + gq.dz^2 * fm[6]))
             end for i in 1:8, j in 1:8, k in 1:6, f in 1:6 if gq.lsqid[i, j, k] > 0)
        @test eq < 1e-12 * maximum(φq)
        mq = F3.Model3D(gq, gas, ρr, Θr)
        # recon = :lin: linear centroid correction of the pseudo third-order values, exact for z
        gl = F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = hill, recon = :lin)
        φl = [(m = F3._cell_moments(gl, hill, nothing, i, j, k); m[1] > 0 ? gl.zc[k] + gl.dz * m[4] : 0.0) for i in 1:8, j in 1:8, k in 1:6]
        el = maximum(begin
                d, s = (f + 1) ÷ 2, isodd(f) ? -1 : 1
                fm = F3._face_moments(gl, hill, nothing, i, j, k, d, s)
                v = d == 1 ? F3.face_x(gl, φl, s > 0 ? mod1(i + 1, 8) : i, j, k, Float64(s)) :
                    d == 2 ? F3.face_y(gl, φl, i, s > 0 ? mod1(j + 1, 8) : j, k, Float64(s)) :
                             (s > 0 ? F3.face_z(gl, φl, i, j, k + 1, 1.0) : F3.face_z(gl, φl, i, j, k, -1.0))
                abs(v - (gl.zc[k] + gl.dz * fm[3]))
             end for i in 1:8, j in 1:8, k in 2:5, f in 1:6
             if gl.lsqid[i, j, k] > 0 && !all(iszero, F3._face_moments(gl, hill, nothing, i, j, k, (f + 1) ÷ 2, isodd(f) ? -1 : 1)))
        @test el < 1e-9 * 6e3
        @test maximum(abs, TS.frozen_fast!(zeros(mq.nq), mq, (Cq = zeros(TS.frozen_ncoef(mq)); Sq = zeros(mq.nq);
              TS.frozen_data!(Cq, Sq, mq, q); Cq), q, :all, Sq) - TS.rhs_nonlinear(mq, q)) < 1e-12 * maximum(abs, N)
        # implicit = :amf (default with cut cells): first-order upwind θ and momentum advection in the mask, in the
        # fast part with its complete Jacobian; mass rows unchanged, conservative, parts consistent, assembled matrix =
        # matrix-free operator = the exact derivative of the masked upwind advection, short run
        ma = F3.Model3D(gc, gas, ρr, Θr); @test ma.implicit === :amf
        Na = TS.rhs_nonlinear(ma, q)
        @test maximum(abs(Na[F3.iρ(gc, i, j, k)] - N[F3.iρ(gc, i, j, k)]) for i in 1:8, j in 1:8, k in 1:6) < 1e-13 * maximum(abs, N)
        @test abs(F3.total3d(ma, Na, :Θ)) < 1e-13 * F3.total3d(ma, q, :Θ)
        Ca = zeros(TS.frozen_ncoef(ma)); Sa = zeros(ma.nq); TS.frozen_data!(Ca, Sa, ma, q)
        oa(p) = TS.frozen_fast!(zeros(ma.nq), ma, Ca, q, p, Sa)
        @test maximum(abs, oa(:all) - Na) < 1e-13 * maximum(abs, N)
        @test maximum(abs, oa(:thermo) + oa(:momentum) - oa(:all)) < 1e-13 * maximum(abs, N)
        @test maximum(abs, oa(:hthermo) + oa(:hmomentum) + oa(:vertical) - oa(:all)) < 1e-13 * maximum(abs, N)
        va = 1e-3 .* cos.(1:ma.nq) .* abs.(q)
        F3.FV3D_MASK_JFORM[] = :conservative                         # complete momentum rows
        Aa = F3.adv_matrix(ma, Ca)
        Ava = F3.adv_theta_masked!(F3.adv_masked!(zeros(ma.nq), ma, Ca, va), ma, Ca, va)
        @test length(Aa.nzval) > 0 && maximum(abs, Aa * va - Ava) < 1e-12 * maximum(abs, Ava)
        # exact Jacobian: masked upwind advection U(y) = A(C(y)) y without the completion terms, and on the Θ rows the
        # upwind Θ flux through the faces of the masked cells (both cells), central differences
        function Uadv(y)
            C = zeros(TS.frozen_ncoef(ma)); TS.frozen_data!(C, zeros(ma.nq), ma, y)
            o = F3.adv_masked!(zeros(ma.nq), ma, C, y)               # the completion terms vanish for z = y
            T = F3.masked_face_transport!(zeros(8, 8, 6), ma, C, y, F3.iΘ, false)
            for k in 1:6, j in 1:8, i in 1:8
                o[F3.iΘ(gc, i, j, k)] += T[i, j, k]
            end
            return o
        end
        rows = unique(Aa.rowval)
        θrows = [F3.iΘ(gc, i, j, k) for i in 1:8, j in 1:8, k in 1:6]
        mrows = setdiff(rows, θrows); trows = intersect(rows, θrows)
        @test length(trows) > count(gc.mask)                           # Θ rows: masked cells and their neighbours
        ε = 1e-5
        # momentum rows: general perturbation (the ρ and mass-flux columns of the complete rows)
        dU = (Uadv(q .+ ε .* va) .- Uadv(q .- ε .* va)) ./ (2ε)
        @test norm((Aa * va .- dU)[mrows]) < 1e-7 * norm(dU[mrows])
        # Θ rows: perturbation of ρ and Θ only (δF = 0): the derivative a± (δΘ - θ δρ) of the upwind Θ flux
        vθ = copy(va); N3 = F3.ncell(gc); vθ[N3+1:end-N3] .= 0
        dθ = (Uadv(q .+ ε .* vθ) .- Uadv(q .- ε .* vθ)) ./ (2ε)
        @test norm((Aa * vθ .- dθ)[trows]) < 1e-7 * norm(dθ[trows])
        @test abs(F3.total3d(ma, Aa * va, :Θ)) < 1e-13 * F3.total3d(ma, abs.(Aa * va), :Θ)   # conservative Θ rows
        F3.FV3D_MASK_JFORM[] = :partial
        ρa = F3.hydrostatic_column(ma, Θr; ρbottom = ρr(gc.zc[1]))
        qa0 = F3.project3d(ma, (x, y, z) -> (ρ = ρa[F3.level(gc, z)]; (ρ, 10ρ, 0.0, 0.0, Θr(gc.zc[F3.level(gc, z)]))))
        qa = TS.run_frozen(ma, qa0, 20.0, 10; tab = TS.mis_tableau(:MIS54), nsub = 4, fast = :hevi)
        @test all(isfinite, qa) && abs(F3.total3d(ma, qa, :ρ) / F3.total3d(ma, qa0, :ρ) - 1) < 1e-12
        # exact masked Jacobian (:conservative) with the coupled sparse solve: the same short run
        try
            F3.FV3D_MASK_JFORM[] = :conservative; F3.FV3D_MASK_SOLVER[] = :lu
            mc2 = F3.Model3D(gc, gas, ρr, Θr)
            qc2 = TS.run_frozen(mc2, qa0, 20.0, 10; tab = TS.mis_tableau(:MIS54), nsub = 4, fast = :hevi)
            @test all(isfinite, qc2) && abs(F3.total3d(mc2, qc2, :ρ) / F3.total3d(mc2, qa0, :ρ) - 1) < 1e-12
            @test maximum(abs, F3.vertical_velocity(mc2, qc2)) < 2maximum(abs, F3.vertical_velocity(ma, qa)) + 1
            # GMRES with the Schwarz preconditioner (2×2 tiles, overlap 1, local AMF or LU; block Jacobi over the columns) = sparse LU
            for (tiles, ov, loc) in (((2, 2), 1, :amf), ((2, 2), 1, :lu), ((8, 8), 0, :amf))
                F3.FV3D_MASK_SOLVER[] = :gmres; F3.FV3D_MASK_TILES[] = tiles; F3.FV3D_MASK_OVERLAP[] = ov; F3.FV3D_MASK_TOL[] = 1e-12
                F3.FV3D_MASK_LOCAL[] = loc
                mc3 = F3.Model3D(gc, gas, ρr, Θr)
                qc3 = TS.run_frozen(mc3, qa0, 20.0, 10; tab = TS.mis_tableau(:MIS54), nsub = 4, fast = :hevi)
                @test maximum(abs, qc3 .- qc2) < 1e-9 * maximum(abs, qc2)
                F3.FV3D_MASK_TOL[] = 1e-3                                # inexact solve: ρ, Θ still conserved
                mc4 = F3.Model3D(gc, gas, ρr, Θr)
                qc4 = TS.run_frozen(mc4, qa0, 20.0, 10; tab = TS.mis_tableau(:MIS54), nsub = 4, fast = :hevi)
                @test abs(F3.total3d(mc4, qc4, :ρ) / F3.total3d(mc4, qa0, :ρ) - 1) < 1e-13
                @test abs(F3.total3d(mc4, qc4, :Θ) / F3.total3d(mc4, qa0, :Θ) - 1) < 1e-13
                F3.FV3D_MASK_TOL[] = 1e-12
                @test abs(F3.total3d(mc3, qc3, :ρ) / F3.total3d(mc3, qa0, :ρ) - 1) < 1e-12
                its = mc3.cache.column[].its
                @test !isempty(its) && maximum(its) < 40
            end
            # :cs small steps with the mask (face weights: h/2 on the ρ, Θ fluxes, (1-β)h, βh on the momenta; the faces of the
            # masked cells linearly implicit Euler with h): the small-step limit of :hevi (first order), ρ and Θ conserved
            for (sol, jf) in ((:amf, :partial), (:gmres, :conservative))
                F3.FV3D_MASK_SOLVER[] = sol; F3.FV3D_MASK_JFORM[] = jf; F3.FV3D_MASK_TILES[] = (2, 2); F3.FV3D_MASK_TOL[] = 1e-12
                mcs = F3.Model3D(gc, gas, ρr, Θr)
                rf(fast, nsub) = TS.run_frozen(mcs, qa0, 20.0, 5; tab = TS.mis_tableau(:MIS54), nsub, fast)
                qh = rf(:hevi, 256)
                e16, e64 = (maximum(abs, rf(:cs, n) .- qh) / maximum(abs, qh) for n in (16, 64))
                @test e64 < 1e-2 && e64 < e16 / 2.5
                q4 = TS.run_frozen(mcs, qa0, 20.0, 10; tab = TS.mis_tableau(:MIS54), nsub = 4, fast = :cs)
                @test all(isfinite, q4)
                @test abs(F3.total3d(mcs, q4, :ρ) / F3.total3d(mcs, qa0, :ρ) - 1) < 1e-13
                @test abs(F3.total3d(mcs, q4, :Θ) / F3.total3d(mcs, qa0, :Θ) - 1) < 1e-13
            end
        finally
            F3.FV3D_MASK_JFORM[] = :partial; F3.FV3D_MASK_SOLVER[] = :amf
            F3.FV3D_MASK_TILES[] = (8, 8); F3.FV3D_MASK_OVERLAP[] = 1; F3.FV3D_MASK_TOL[] = 1e-8; F3.FV3D_MASK_LOCAL[] = :amf
        end
        g0 = F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = (x, y) -> -1.0); gf = F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3)
        qf = F3.project3d(F3.Model3D(gf, gas, ρr, Θr), (x, y, z) -> (ρr(z), 10ρr(z), 5ρr(z), 0.0, Θr(z)))
        @test TS.rhs_nonlinear(F3.Model3D(g0, gas, ρr, Θr), qf) == TS.rhs_nonlinear(F3.Model3D(gf, gas, ρr, Θr), qf)
        # cube resolved by the grid (full and solid cells) and shifted (cut cells): balanced rest
        # state exactly steady, short flow run stable and conservative
        for (box, ncut) in (((8e3, 12e3, 8e3, 12e3, 2000.0), 0), ((8.6e3, 12.6e3, 8.6e3, 12.6e3, 2300.0), -1))
            gb = F3.Grid3D(10, 10, 6, 20e3, 20e3, 6e3; box)
            ncut == 0 ? (@test count(0 .< gb.Vf .< 1) == 0) : (@test count(0 .< gb.Vf .< 1) > 0)
            @test count(gb.Vf .== 0) > 0
            mb = F3.Model3D(gb, gas, ρr, Θr)
            ρb = F3.hydrostatic_column(mb, Θr; ρbottom = ρr(gb.zc[1]))
            st(u) = F3.project3d(mb, (x, y, z) -> (ρ = ρb[F3.level(gb, z)]; (ρ, u * ρ, 0.0, 0.0, Θr(gb.zc[F3.level(gb, z)]))))
            @test maximum(abs, TS.rhs_nonlinear(mb, st(0.0))) < 1e-13
            q0 = st(10.0)
            qe = TS.run_frozen(mb, q0, 20.0, 10; tab = TS.mis_tableau(:MIS54), nsub = 5, fast = :hevi)
            @test all(isfinite, qe) && maximum(abs, F3.vertical_velocity(mb, qe)) < 8
            @test abs(F3.total3d(mb, qe, :ρ) / F3.total3d(mb, q0, :ρ) - 1) < 1e-12
        end
        # mask (cut cells and neighbours) with the implicit masked step: parts, exact implicit matrix
        mk = F3.Model3D(F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = hill), gas, ρr, Θr; implicit = :mask)
        @test count(mk.grid.mask) > count(0 .< mk.grid.Vf .< 1)
        qk = F3.project3d(mk, (x, y, z) -> (ρr(z) * (1 + 1e-3 * sin(x / 3e3)), 10ρr(z), 5ρr(z), 0.1ρr(z) * sin(π * z / 6e3), Θr(z)))
        Ck = zeros(TS.frozen_ncoef(mk)); Sk = zeros(mk.nq); TS.frozen_data!(Ck, Sk, mk, qk)
        ok(p) = TS.frozen_fast!(zeros(mk.nq), mk, Ck, qk, p, Sk)
        @test maximum(abs, ok(:hthermo) + ok(:hmomentum) + ok(:vertical) - ok(:all)) < 1e-14 * maximum(abs, ok(:all))
        sk = TS.frozen_stage_begin!(mk, Ck, qk)
        vk = 1e-3 .* sin.(1:mk.nq)
        Gk(x) = TS.frozen_fast!(zeros(mk.nq), mk, Ck, x, :vertical, Sk)
        @test maximum(abs, sk.J * vk - (Gk(qk .+ vk) - Gk(qk))) < 1e-10 * maximum(abs, sk.J * vk)
        # terrain following (Gal-Chen): zs = 0 is the flat grid, discrete GCL, constant θ, parts,
        # exact column solve, conservation, W at rest in the balanced state
        gt0 = F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = (x, y) -> 0.0, terrain = :follow)
        @test TS.rhs_nonlinear(F3.Model3D(gt0, gas, ρr, Θr), qf) == TS.rhs_nonlinear(F3.Model3D(gf, gas, ρr, Θr), qf)
        gt = F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = (x, y) -> 600.0 * exp(-((x - 10e3)^2 + (y - 10e3)^2) / 3e3^2), terrain = :follow)
        mt = F3.Model3D(gt, gas, ρr, Θr)
        Nc = TS.rhs_nonlinear(mt, F3.project3d(mt, (x, y, z) -> (1.0, 10.0, 5.0, 0.0, 300.0)))
        @test maximum(abs(Nc[F3.iρ(gt, i, j, k)]) for i in 1:8, j in 1:8, k in 2:6) < 1e-15
        @test maximum(abs(Nc[F3.iΘ(gt, i, j, k)] - 300 * Nc[F3.iρ(gt, i, j, k)]) for i in 1:8, j in 1:8, k in 1:6) < 1e-13
        qt = F3.project3d(mt, (x, y, z) -> (ρr(z) * (1 + 1e-3 * sin(x / 3e3)), 10ρr(z), 5ρr(z), 0.1ρr(z) * sin(π * z / 6e3), Θr(z)))
        Ct = zeros(TS.frozen_ncoef(mt)); St = zeros(mt.nq); TS.frozen_data!(Ct, St, mt, qt)
        ot(p) = TS.frozen_fast!(zeros(mt.nq), mt, Ct, qt, p, St)
        Nt = TS.rhs_nonlinear(mt, qt)
        @test maximum(abs, ot(:hthermo) + ot(:hmomentum) + ot(:vertical) - Nt) < 1e-14 * maximum(abs, Nt)
        @test abs(F3.total3d(mt, Nt, :ρ)) < 1e-14 * F3.total3d(mt, qt, :ρ)
        st = TS.frozen_stage_begin!(mt, Ct, qt)
        Yt = zeros(mt.nq); F3.column_solve!(Yt, st, mt, 3.0, R)
        Gt(x) = TS.frozen_fast!(zeros(mt.nq), mt, Ct, x, :vertical, St)
        @test maximum(abs.(Yt .- 3.0 .* (Gt(qt .+ Yt) .- Gt(qt)) .- R)) < 1e-9 * maximum(abs, R)
        Nb = TS.rhs_nonlinear(mt, F3.balanced_state(mt, Θr, ρr))
        @test maximum(abs(Nb[F3.iW(gt, i, j, k)]) for i in 1:8, j in 1:8, k in 2:6) < 1e-13
        # second-order convergence of the frozen MIS54 / HEVI step from a balanced state
        g = F3.Grid3D(20, 2, 8, 100e3, 10e3, 10e3); m = F3.Model3D(g, gas, ρr, Θr)
        ρb = F3.hydrostatic_column(m, Θr; ρbottom = ρr(g.zc[1]))
        q0 = F3.project3d(m, (x, y, z) -> begin
            Θ = Θr(g.zc[F3.level(g, z)]); θ = Θ / ρb[F3.level(g, z)] + 0.01 * sin(π * z / 10e3) / (1 + ((x - 50e3) / 5e3)^2)
            (Θ / θ, 20Θ / θ, 0.0, 0.0, Θ)
        end)
        c = F3.sound_speed_max(m, q0)
        nref = ceil(Int, 300 / (0.3 * g.dz / (c + 20)))
        qref = TS.run_rk4(m, q0, 300 / nref, nref)
        e = [norm(TS.run_frozen(m, q0, Δt, round(Int, 300 / Δt); tab = TS.mis_tableau(:MIS54), nsub = 12, fast = :hevi) .- qref)
             for Δt in (30.0, 15.0)]
        @test rate(e...) > 1.7
    finally
        F3.FV3D_PRESSURE_FORM[] = :gradp
        FV.FV_PRESSURE_FORM[] = :gradp
    end
end

@testset "FV3D: moist air (MoistModel3D, the 2D moist model in 3D)" begin
    F3 = MultiRate.FV3D
    mgθ = FV.MoistGas()
    gas = CM.Gas(R = mgθ.Rd, γ = mgθ.Cpd / (mgθ.Cpd - mgθ.Rd), p00 = mgθ.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    # ny = 1 is the 2D moist model: N, parts, column solve and MIS steps (:hevi, :cs), supersaturated (phase change
    # active), flat and terrain following, all thermodynamic and pressure variables
    nx, nz, L, H = 16, 10, 40e3, 10e3
    ridge(x) = 800 / (1 + ((x - 20e3) / 4e3)^2)
    for (terr, thv, pvar) in ((false, :thetarho, :pi), (true, :thetarho, :p), (true, :energy, :pi), (false, :entropy, :p))
        g2 = terr ? FV.FVGrid(nx, nz, L, H; zs = ridge) : FV.FVGrid(nx, nz, L, H)
        g3 = terr ? F3.Grid3D(nx, 1, nz, L, 5e3, H; zs = (x, y) -> ridge(x), terrain = :follow) : F3.Grid3D(nx, 1, nz, L, 5e3, H)
        b2 = FV.FVModel(g2, gas, ρr, Θr); b3 = F3.Model3D(g3, gas, ρr, Θr)
        mg = FV.MoistGas(thvar = thv)
        m2θ = FV.MoistExnerModel(b2; mgas = mgθ, pvar)
        m2 = FV.MoistExnerModel(b2; mgas = mg, pvar); m3 = F3.MoistModel3D(b3; mgas = mg, pvar)
        @test m3.τ ≈ m2.τ
        q = CM.project2d(b2, (x, z) -> (ρ = ρr(z) * (1 + 1e-3 * sin(2π * x / L)); (ρ, 10ρ + sin(x / 5e3), 0.3 * sin(π * z / H) * cos(x / 7e3), Θr(z) * (1 + 1e-3 * cos(x / 6e3)))))
        x2 = FV.thermo_convert(m2θ, m2, FV.moist_exner_state(m2θ, q, [0.012 * q[FV.iρ(g2, i, k)] for i in 1:nx, k in 1:nz]))
        for i in 1:nx, k in 1:nz
            c = x2[FV.iC(m2, i, k)]; x2[FV.iV(m2, i, k)] += 0.3c; x2[FV.iC(m2, i, k)] -= 0.3c
        end
        TS.frozen_step_end!(m2, x2)
        function to3(x)
            y = zeros(m3.nq)
            for k in 1:nz, i in 1:nx
                y[F3.iρ(g3, i, 1, k)] = x[FV.iρ(g2, i, k)]; y[F3.iΘ(g3, i, 1, k)] = x[FV.iΘ(g2, i, k)]
                y[F3.iU(g3, i, 1, k)] = x[FV.iU(g2, i, k)]; k > 1 && (y[F3.iW(g3, i, 1, k)] = x[FV.iW(g2, i, k)])
                y[F3.iqv(m3, i, 1, k)] = x[FV.iV(m2, i, k)]; y[F3.iqc(m3, i, 1, k)] = x[FV.iC(m2, i, k)]
                y[F3.iπ(m3, i, 1, k)] = x[FV.iπ(m2, i, k)]
            end
            return y
        end
        x3 = to3(x2)
        d(y2, y3) = maximum(abs, to3(y2) - y3) / maximum(abs, y2)
        @test d(TS.rhs_nonlinear(m2, x2), TS.rhs_nonlinear(m3, x3)) < 1e-14
        C2 = zeros(TS.frozen_ncoef(m2)); S2 = zeros(m2.nq); TS.frozen_data!(C2, S2, m2, x2)
        C3 = zeros(TS.frozen_ncoef(m3)); S3 = zeros(m3.nq); TS.frozen_data!(C3, S3, m3, x3)
        for p in (:hthermo, :hmomentum, :vertical, :thermo, :momentum)
            @test d(TS.frozen_fast!(zeros(m2.nq), m2, C2, x2, p, S2), TS.frozen_fast!(zeros(m3.nq), m3, C3, x3, p, S3)) < 1e-14
        end
        TS.frozen_stage_begin!(m2, C2, x2); TS.frozen_stage_begin!(m3, C3, x3)
        R2 = 1e-3 .* cos.(1:m2.nq); Y2 = zeros(m2.nq); Y3 = zeros(m3.nq)
        FV.moist_column_solve!(Y2, m2, 3.0, R2, 1.7); F3.moist_column_solve!(Y3, m3, 3.0, to3(R2), 1.7)
        @test d(Y2, Y3) < 1e-14
        for fast in (:hevi, :cs)
            y2 = TS.run_frozen(m2, x2, 4.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6, fast)
            y3 = TS.run_frozen(m3, x3, 4.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6, fast)
            @test maximum(abs, to3(y2) - y3) < 1e-8 * maximum(abs, y2 - x2)
        end
    end
    # genuine 3D (8 × 8 × 6, flat and a terrain-following hill): parts, linear fast part (micro = :slow), exact column
    # solve, x ↔ y symmetry (flat), conservation of mass and total water by the frozen step (micro = :fast)
    hill(x, y) = 1200.0 * exp(-((x - 10e3)^2 + (y - 10e3)^2) / 4e3^2)
    for terr in (false, true)
        g = terr ? F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = hill, terrain = :follow) : F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3)
        b = F3.Model3D(g, gas, ρr, Θr)
        mθ = F3.MoistModel3D(b; mgas = mgθ, micro = :slow)
        fq(x, y, z) = (ρ = ρr(z) * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)); (ρ, 8ρ + sin(y / 3e3), 5ρ + cos(x / 4e3), 0.2ρ * sin(π * z / 6e3) * cos((x + 2y) / 5e3), Θr(z) * (1 + 1e-3 * cos((x - y) / 3e3))))
        q = F3.project3d(b, fq)
        x = F3.moist_exner_state(mθ, q, [0.013 * q[F3.iρ(g, i, j, k)] for i in 1:8, j in 1:8, k in 1:6])
        C = zeros(TS.frozen_ncoef(mθ)); S = zeros(mθ.nq); TS.frozen_data!(C, S, mθ, x)
        o(p, z = x) = TS.frozen_fast!(zeros(mθ.nq), mθ, C, z, p, S)
        N = TS.rhs_nonlinear(mθ, x)
        @test o(:all) == N
        @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - N) < 1e-13 * maximum(abs, N)
        @test maximum(abs, o(:thermo) + o(:momentum) - N) < 1e-13 * maximum(abs, N)
        v = 1e-3 .* sin.(1:mθ.nq)
        @test maximum(abs, o(:all, x .+ v) + o(:all, x .- v) - 2N) < 1e-11 * maximum(abs, N)
        TS.frozen_stage_begin!(mθ, C, x)
        R = 1e-3 .* cos.(1:mθ.nq); Y = zeros(mθ.nq); F3.moist_column_solve!(Y, mθ, 3.0, R)
        @test maximum(abs.(Y .- 3.0 .* (o(:vertical, x .+ Y) .- o(:vertical, x)) .- R)) < 1e-9 * maximum(abs, R)
        xs = copy(x); xs[F3.iπ(mθ, 3, 2, 3)] += 1e-3; TS.frozen_step_end!(mθ, xs)
        @test xs ≈ x rtol = 1e-15
        if !terr                                              # x ↔ y: transposed state, transposed tendency
            function T(z)
                y = copy(z)
                for k in 1:6, j in 1:8, i in 1:8
                    y[F3.iU(g, i, j, k)] = z[F3.iV(g, j, i, k)]; y[F3.iV(g, i, j, k)] = z[F3.iU(g, j, i, k)]
                    for f in (F3.iρ, F3.iΘ); y[f(g, i, j, k)] = z[f(g, j, i, k)]; end
                    for f in (F3.iqv, F3.iqc, F3.iπ); y[f(mθ, i, j, k)] = z[f(mθ, j, i, k)]; end
                    k > 1 && (y[F3.iW(g, i, j, k)] = z[F3.iW(g, j, i, k)])
                end
                return y
            end
            @test maximum(abs, TS.rhs_nonlinear(mθ, T(x)) - T(N)) < 1e-13 * maximum(abs, N)
        end
        mf = F3.MoistModel3D(b; mgas = mgθ, micro = :fast)
        xsup = copy(x)
        for k in 1:6, j in 1:8, i in 1:8
            c = xsup[F3.iqc(mf, i, j, k)]; xsup[F3.iqv(mf, i, j, k)] += 0.3c; xsup[F3.iqc(mf, i, j, k)] -= 0.3c
        end
        TS.frozen_step_end!(mf, xsup)
        Nf = TS.rhs_nonlinear(mf, xsup)
        Nn = TS.rhs_nonlinear(F3.MoistModel3D(b; mgas = mgθ, micro = :none), xsup)
        @test minimum(Nf[F3.iqc(mf, i, j, k)] - Nn[F3.iqc(mf, i, j, k)]
                      for i in 1:8, j in 1:8, k in 1:6 if xsup[F3.iqc(mf, i, j, k)] > 0) > 0   # condensation in the cloudy cells
        for fast in (:hevi, :cs)
            y = TS.run_frozen(mf, xsup, 3.0, 3; tab = TS.mis_tableau(:MIS54), nsub = 6, fast)
            @test abs(F3.total3d(mf, y, :ρ) / F3.total3d(mf, xsup, :ρ) - 1) < 1e-13
            @test abs((F3.total3d(mf, y, :qv) + F3.total3d(mf, y, :qc)) / (F3.total3d(mf, xsup, :qv) + F3.total3d(mf, xsup, :qc)) - 1) < 1e-13
            @test F3.total3d(mf, y, :qc) > F3.total3d(mf, xsup, :qc)
        end
    end
    @test_throws ErrorException F3.MoistModel3D(F3.Model3D(F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = hill), gas, ρr, Θr; implicit = :column))
end

@testset "FV3D: warm-rain Kessler microphysics (DCMIP), rain = true" begin
    F3 = MultiRate.FV3D
    mgθ = CM.MoistGas()
    gas = CM.Gas(R = mgθ.Rd, γ = mgθ.Cpd / (mgθ.Cpd - mgθ.Rd), p00 = mgθ.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    # rates against the formulas of kessler.f90 (DCMIP 2016) in mixing ratios
    kp = CM.KesslerParams(τr = 1e-12)                       # τr small: limiters inactive
    ρ, ρv, ρc, ρrr, T, p = 1.1, 0.012, 0.003, 0.002, 285.0, 9.0e4
    ρd = ρ - ρv - ρc - ρrr; qc, qr, qv = ρc / ρd, ρrr / ρd, ρv / ρd; r = 0.001ρd
    P, E = CM.kessler_rates(kp, mgθ, ρ, T, p, ρv, ρc, ρrr)
    @test P ≈ ρd * (0.001 * (qc - 0.001) + 2.2 * qc * qr^0.875)
    qvs = CM.psat(mgθ, T) / (mgθ.Rv * T) / ρd
    pc = 3.8 / (p / 100)                                     # kessler.f90: pc = 3.8/(p in mb)
    ern = ((1.6 + 124.9 * (r * qr)^0.2046) * (r * qr)^0.525) / (2550000 * pc / (3.8 * qvs) + 540000) *
          (max(qvs - qv, 0) / (r * qvs))
    @test E ≈ ρd * ern
    @test CM.rain_velocity(kp, ρd, ρrr, 1.2) ≈ 36.34 * (qr * r)^0.1364 * sqrt(1.2 / ρd)
    @test CM.kessler_rates(kp, mgθ, ρ, T, p, 0.02, ρc, ρrr)[2] == 0           # supersaturated: no evaporation
    # liquid content: X(ρ + h, T, ρv, ρl + h) - X(ρ - h, …) / 2h
    for thv in (:thetarho, :energy, :entropy)
        mg = CM.MoistGas(thvar = thv); h = 1e-7
        X = CM._Θof(mg, ρ, T, ρv, ρc + ρrr)
        fd = (CM._Θof(mg, ρ + h, T, ρv, ρc + ρrr + h) - CM._Θof(mg, ρ - h, T, ρv, ρc + ρrr - h)) / 2h
        pp = (mg.Rd * ρd + mg.Rv * ρv) * T
        @test CM.liquid_content(mg, ρ, X, T, pp, ρv, ρc + ρrr) ≈ fd rtol = 1e-6
    end

    hill(x, y) = 1200.0 * exp(-((x - 10e3)^2 + (y - 10e3)^2) / 4e3^2)
    fq(x, y, z) = (ρ = ρr(z) * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)); (ρ, 8ρ + sin(y / 3e3), 5ρ + cos(x / 4e3), 0.2ρ * sin(π * z / 6e3) * cos((x + 2y) / 5e3), Θr(z) * (1 + 1e-3 * cos((x - y) / 3e3))))
    for terr in (false, true), (thv, pvar) in ((:thetarho, :p), (:energy, :pi), (:entropy, :p))
        g = terr ? F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3; zs = hill, terrain = :follow) : F3.Grid3D(8, 8, 6, 20e3, 20e3, 6e3)
        b = F3.Model3D(g, gas, ρr, Θr)
        mg = CM.MoistGas(thvar = thv)
        mθ = F3.MoistModel3D(b; mgas = mgθ, pvar)
        m0 = F3.MoistModel3D(b; mgas = mg, pvar); m1 = F3.MoistModel3D(b; mgas = mg, pvar, rain = true)
        @test m1.nq == m0.nq + F3.ncell(g)
        q = F3.project3d(b, fq)
        x0 = F3.thermo_convert(mθ, m0, F3.moist_exner_state(mθ, q, [0.004 * q[F3.iρ(g, i, j, k)] for i in 1:8, j in 1:8, k in 1:6]))
        # rain = 0 and q_c below the autoconversion threshold: the rain model is the model without rain
        x1 = vcat(x0, zeros(F3.ncell(g)))
        @test all(F3.qc3d(m0, x0) .< 1e-3)
        N0 = TS.rhs_nonlinear(m0, x0); N1 = TS.rhs_nonlinear(m1, x1)
        @test maximum(abs, N1[1:m0.nq] - N0) <= 1e-12 * maximum(abs, N0)
        @test maximum(abs, N1[m0.nq+1:end]) == 0
        # rain present, subsaturated layers: conversion, evaporation, sedimentation
        xr = copy(x1)
        for k in 1:6, j in 1:8, i in 1:8
            ρk = xr[F3.iρ(g, i, j, k)]
            xr[F3.iqr(m1, i, j, k)] = 2e-3 * ρk * (1 + 0.5 * sin(i + 2j + k))
            xr[F3.iqc(m1, i, j, k)] = (k <= 3 ? 2.5e-3 : 0.0) * ρk                     # cloud only below
            xr[F3.iqv(m1, i, j, k)] *= k <= 3 ? 1.0 : 0.8                              # subsaturated aloft
        end
        TS.frozen_step_end!(m1, xr)
        C = zeros(TS.frozen_ncoef(m1)); S = zeros(m1.nq); TS.frozen_data!(C, S, m1, xr)
        N = TS.rhs_nonlinear(m1, xr)
        o(pp, z = xr) = TS.frozen_fast!(zeros(m1.nq), m1, C, z, pp, S)
        @test maximum(abs, o(:hthermo) + o(:hmomentum) + o(:vertical) - N) < 1e-13 * maximum(abs, N)
        @test maximum(abs, o(:thermo) + o(:momentum) - N) < 1e-13 * maximum(abs, N)
        TS.frozen_stage_begin!(m1, C, xr)
        R = 1e-3 .* cos.(1:m1.nq); Y = zeros(m1.nq); F3.moist_column_solve!(Y, m1, 3.0, R)
        @test maximum(abs.(Y .- 3.0 .* (o(:vertical, xr .+ Y) .- o(:vertical, xr)) .- R)) < 1e-9 * maximum(abs, R)
        # budgets of the sedimentation (fast part, msed!) and the slow rain sources: Σ V (sed + S) = -(ground flux) for
        # mass and total water; energy: rain carries c_pl T
        V(i, j, k) = g.Jc[i, j] * g.dx * g.dy * g.dz / g.invV[i, j, k]
        Sr = zeros(m1.nq); TS.frozen_data!(C, Sr, m1, xr)
        sed = F3.msed!(zeros(m1.nq), m1, C, xr; thermo = true, uv = true, w = true)
        Sr .+= sed
        tot(f) = sum(V(i, j, k) * f(i, j, k) for i in 1:8, j in 1:8, k in 1:6)
        A(i, j) = g.dx * g.dy
        Pg = F3.precipitation_rate(m1, xr)
        ground = sum(A(i, j) * Pg[i, j] for i in 1:8, j in 1:8)
        @test tot((i, j, k) -> Sr[F3.iρ(g, i, j, k)]) ≈ -ground rtol = 1e-12
        @test tot((i, j, k) -> Sr[F3.iqv(m1, i, j, k)] + Sr[F3.iqc(m1, i, j, k)] + Sr[F3.iqr(m1, i, j, k)]) ≈ -ground rtol = 1e-12
        if thv === :energy
            Tg = [CM.moist_state(mg, xr[F3.iρ(g, i, j, 1)], xr[F3.iΘ(g, i, j, 1)], xr[F3.iqv(m1, i, j, 1)],
                                 xr[F3.iqc(m1, i, j, 1)] + xr[F3.iqr(m1, i, j, 1)])[1] for i in 1:8, j in 1:8]
            @test tot((i, j, k) -> Sr[F3.iΘ(g, i, j, k)]) ≈ -sum(A(i, j) * Pg[i, j] * mg.Cpl * Tg[i, j] for i in 1:8, j in 1:8) rtol = 1e-12
        end
        # momentum carried by the rain: the U part telescopes to the loss through the ground
        uu(i, j) = xr[F3.iU(g, i, j, 1)] / (0.5 * (xr[F3.iρ(g, mod1(i - 1, 8), j, 1)] + xr[F3.iρ(g, i, j, 1)]))
        Ploc(i, j) = (ρd1 = xr[F3.iρ(g, i, j, 1)] - xr[F3.iqv(m1, i, j, 1)] - xr[F3.iqc(m1, i, j, 1)] - xr[F3.iqr(m1, i, j, 1)];
                      xr[F3.iqr(m1, i, j, 1)] * CM.rain_velocity(m1.kp, ρd1, xr[F3.iqr(m1, i, j, 1)], ρd1))
        dU = sum(sed[F3.iU(g, i, j, k)] * g.Jx[i, j] * g.dz for i in 1:8, j in 1:8, k in 1:6)
        @test dU ≈ -sum(0.5 * (Ploc(mod1(i - 1, 8), j) + Ploc(i, j)) * uu(i, j) for i in 1:8, j in 1:8) rtol = 1e-10
        @test minimum(Sr[F3.iqc(m1, i, j, k)] for i in 1:8, j in 1:8, k in 1:3) < 0           # conversion
        @test maximum(Sr[F3.iqv(m1, i, j, k)] for i in 1:8, j in 1:8, k in 4:6) > 0           # evaporation
        # MIS steps without fall speed (cv = 0): conversion and evaporation conserve mass and total water exactly
        # (the sedimentation fluxes telescope, budgets above; the explicit upwind rain reaches the ground in a few steps)
        water(z) = F3.total3d(m1, z, :qv) + F3.total3d(m1, z, :qc) + F3.total3d(m1, z, :qr)
        m1s = F3.MoistModel3D(b; mgas = mg, pvar, rain = true, kessler = CM.KesslerParams(cv = 0.0))
        xa = xr
        for fast in (:hevi, :cs)
            y = TS.run_frozen(m1s, xa, 2.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6, fast)
            @test abs(F3.total3d(m1, y, :ρ) / F3.total3d(m1, xa, :ρ) - 1) < 1e-13
            @test abs(water(y) / water(xa) - 1) < 1e-13
            @test all(isfinite, y) && minimum(F3.qr3d(m1, y)) >= 0
        end
        # with sedimentation: the rain leaves through the ground, mass and total water decrease by the same amount
        y = TS.run_frozen(m1, xr, 2.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6, fast = :cs)
        dM = F3.total3d(m1, y, :ρ) - F3.total3d(m1, xr, :ρ)
        @test dM < 0 && abs(water(y) - water(xr) - dM) < 1e-8 * abs(dM)   # (dM: difference of totals ~1e12 kg)
        @test 0 < F3.rain_courant(m1, xr, 5.0) < 1
    end
end

@testset "FV3DKA: moist 3D model with KernelAbstractions, x-y tiles (one rank; MPI: test/fv3dka_mpi.jl)" begin
    F3 = MultiRate.FV3D; K = MultiRate.FV3DKA
    mgθ = CM.MoistGas()
    @test isbits(mgθ) && CM.MoistGas(thvar = :energy).thvar === :energy
    gas = CM.Gas(R = mgθ.Rd, γ = mgθ.Cpd / (mgθ.Cpd - mgθ.Rd), p00 = mgθ.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    hill(x, y) = 1200.0 * exp(-((x - 10e3)^2 + (y - 7e3)^2) / 4e3^2)
    nx, ny, nz, L, Ly, H = 10, 8, 6, 20e3, 16e3, 6e3
    for terr in (false, true), (thv, pvar, micro) in ((:thetarho, :p, :fast), (:energy, :pi, :fast), (:entropy, :p, :slow))
        g3 = terr ? F3.Grid3D(nx, ny, nz, L, Ly, H; zs = hill, terrain = :follow) : F3.Grid3D(nx, ny, nz, L, Ly, H)
        b3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.05)
        mg = CM.MoistGas(thvar = thv)
        mθ = F3.MoistModel3D(b3; mgas = mgθ, pvar, micro); m3 = F3.MoistModel3D(b3; mgas = mg, pvar, micro)
        fq(x, y, z) = (ρ = ρr(z) * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)); (ρ, 8ρ + sin(y / 3e3), 5ρ + cos(x / 4e3), 0.2ρ * sin(π * z / H) * cos((x + 2y) / 5e3), Θr(z) * (1 + 1e-3 * cos((x - y) / 3e3))))
        q = F3.project3d(b3, fq)
        x = F3.thermo_convert(mθ, m3, F3.moist_exner_state(mθ, q, [0.013 * q[F3.iρ(g3, i, j, k)] for i in 1:nx, j in 1:ny, k in 1:nz]))
        for k in 1:nz, j in 1:ny, i in 1:nx                   # supersaturated: phase change active
            c = x[F3.iqc(m3, i, j, k)]; x[F3.iqv(m3, i, j, k)] += 0.3c; x[F3.iqc(m3, i, j, k)] -= 0.3c
        end
        TS.frozen_step_end!(m3, x)
        gk = K.KAGrid(nx, ny, nz, L, Ly, H; zs = terr ? hill : nothing)
        I = K.HALO .+ (1:nx); J = K.HALO .+ (1:ny)
        @test Array(gk.Jc)[I, J] == g3.Jc && Array(gk.zxx)[I, J, :] == g3.zxx && Array(gk.zyf)[I, J, :] == g3.zyf
        mk = K.MoistKA(gk; mgas = mg, pvar, micro, sponge = (4e3, 0.05))
        @test mk.τ ≈ m3.τ
        Q = K.ka_state(mk, m3, x)
        @test K.global_state(mk, m3, Q) == x
        Q2 = copy(Q); Q2[1, :, :, :] .= 0; Q2[:, end, :, :] .= 0; K.exchange!(K.SerialDecomp(), Q2, gk)
        @test Q2 == Q                                         # periodic halo
        N3 = TS.rhs_nonlinear(m3, x)
        out = K.ka_zeros(mk); K.ka_rhs!(out, mk, Q)
        # equal up to the finite-difference noise of the frozen ∂p/∂(ρ, X, ρq_v, ρq_c) (h = 1e-6)
        @test maximum(abs, K.global_state(mk, m3, out) - N3) < 1e-9 * maximum(abs, N3)
        y3 = TS.run_frozen(m3, x, 3.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6, fast = :cs)
        K.ka_run!(mk, Q, 3.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6)
        @test maximum(abs, K.global_state(mk, m3, Q) - y3) < 1e-8 * maximum(abs, y3 - x)
        @test abs(K.ka_total(mk, Q, K.VR) / F3.total3d(m3, x, :ρ) - 1) < 1e-13
        @test abs((K.ka_total(mk, Q, K.VQV) + K.ka_total(mk, Q, K.VQC)) / (F3.total3d(m3, x, :qv) + F3.total3d(m3, x, :qc)) - 1) < 1e-13
    end
end

@testset "FV3DKA: dry cut cells (CutKA) equal FV3D Model3D, implicit = :amf, :hevi with GMRES (tile AMF, columns)" begin
    K = MultiRate.FV3DKA; F3 = MultiRate.FV3D
    gas = CM.Gas(); ρr, Θr, θr = CM.stratified_reference(gas)
    pf, jf0, sol0, tol0, til0 = F3.FV3D_PRESSURE_FORM[], F3.FV3D_MASK_JFORM[], F3.FV3D_MASK_SOLVER[], F3.FV3D_MASK_TOL[], F3.FV3D_MASK_TILES[]
    try
        F3.FV3D_PRESSURE_FORM[] = :product
        hill(x, y) = 1500.0 * exp(-((x - 10e3)^2 + (y - 8e3)^2) / 3e3^2)
        g3 = F3.Grid3D(10, 8, 6, 20e3, 16e3, 6e3; zs = hill)
        tab = TS.mis_tableau(:MIS54)
        Random.seed!(3)
        for jf in (:partial, :conservative)
            F3.FV3D_MASK_JFORM[] = jf
            m3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.02)
            ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))
            q0 = F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k], 10ρa[k], -3ρa[k], 0.0, Θr(g3.zc[k]))))
            m = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), jform = jf, prec = :amf)
            # N(q) and the implicit part G_v(v) at a perturbed state
            q = q0 .* (1 .+ 1e-3 .* randn(length(q0)))
            Q = K.ka_state(m, g3, q)
            N = K.global_state(m, g3, K.ka_rhs!(K.ka_zeros(m), m, Q))
            N3 = TS.rhs_nonlinear(m3, q)
            @test maximum(abs, N .- N3) < 1e-13 * maximum(abs, N3)
            C3 = zeros(TS.frozen_ncoef(m3)); S3 = zeros(m3.nq); TS.frozen_data!(C3, S3, m3, q)
            v = 1e-3 .* randn(length(q)) .* abs.(q)
            Cf = K.ka_zeros(m, K.NCC); K.ka_frozen_data!(Cf, K.ka_zeros(m), m, Q)
            V = K.ka_state(m, g3, v); K.exchange!(m.dec, V, m.grid)
            G3 = TS.frozen_fast!(zeros(m3.nq), m3, C3, v, :vertical, S3)
            @test maximum(abs, K.global_state(m, g3, K.ka_vertical!(K.ka_zeros(m), m, Cf, V)) .- G3) < 1e-13 * maximum(abs, G3)
            # MIS54 with the HEVI small steps: GMRES (column preconditioner) = FV3D (GMRES, Schwarz), ρ and Θ conserved
            F3.FV3D_MASK_SOLVER[] = :gmres; F3.FV3D_MASK_TOL[] = 1e-12; F3.FV3D_MASK_TILES[] = (2, 2)
            q3 = TS.run_frozen(m3, q0, 20.0, 6; tab, nsub = 4, fast = :hevi)
            Q = K.ka_state(m, g3, q0)
            K.ka_run!(m, Q, 20.0, 6; tab, nsub = 4, tol = 1e-12)
            qk = K.global_state(m, g3, Q)
            @test maximum(abs, qk .- q3) < 1e-11 * maximum(abs, q3)
            @test abs(F3.total3d(m3, qk, :ρ) / F3.total3d(m3, q0, :ρ) - 1) < 1e-13
            @test abs(F3.total3d(m3, qk, :Θ) / F3.total3d(m3, q0, :Θ) - 1) < 1e-13
            @test maximum(m.its) < 20
            # BiCGStab instead of GMRES: the same run
            Q = K.ka_state(m, g3, q0)
            K.ka_run!(m, Q, 20.0, 6; tab, nsub = 4, tol = 1e-12, krylov = :bicgstab)
            @test maximum(abs, K.global_state(m, g3, Q) .- q3) < 1e-10 * maximum(abs, q3)
            # column preconditioner (default) instead of the tile AMF: the same run, more iterations
            mc = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), jform = jf)
            Q = K.ka_state(mc, g3, q0)
            K.ka_run!(mc, Q, 20.0, 6; tab, nsub = 4, tol = 1e-12)
            @test maximum(abs, K.global_state(mc, g3, Q) .- q3) < 1e-10 * maximum(abs, q3)
            @test sum(m.its) < sum(mc.its) / 2
        end
    finally
        F3.FV3D_PRESSURE_FORM[] = pf; F3.FV3D_MASK_JFORM[] = jf0; F3.FV3D_MASK_SOLVER[] = sol0; F3.FV3D_MASK_TOL[] = tol0
        F3.FV3D_MASK_TILES[] = til0
    end
end

@testset "FV3DKA: warm rain (rain = true) equals FV3D" begin
    F3 = MultiRate.FV3D; K = MultiRate.FV3DKA
    mgθ = CM.MoistGas()
    gas = CM.Gas(R = mgθ.Rd, γ = mgθ.Cpd / (mgθ.Cpd - mgθ.Rd), p00 = mgθ.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    hill(x, y) = 1200.0 * exp(-((x - 10e3)^2 + (y - 7e3)^2) / 4e3^2)
    nx, ny, nz, L, Ly, H = 10, 8, 6, 20e3, 16e3, 6e3
    for terr in (false, true), (thv, pvar, micro) in ((:thetarho, :p, :fast), (:energy, :pi, :slow))
        g3 = terr ? F3.Grid3D(nx, ny, nz, L, Ly, H; zs = hill, terrain = :follow) : F3.Grid3D(nx, ny, nz, L, Ly, H)
        b3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.05)
        mg = CM.MoistGas(thvar = thv)
        mθ = F3.MoistModel3D(b3; mgas = mgθ, pvar, micro, rain = true)
        m3 = F3.MoistModel3D(b3; mgas = mg, pvar, micro, rain = true)
        fq(x, y, z) = (ρ = ρr(z) * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)); (ρ, 8ρ + sin(y / 3e3), 5ρ + cos(x / 4e3), 0.2ρ * sin(π * z / H) * cos((x + 2y) / 5e3), Θr(z) * (1 + 1e-3 * cos((x - y) / 3e3))))
        q = F3.project3d(b3, fq)
        x = F3.thermo_convert(mθ, m3, F3.moist_exner_state(mθ, q, [0.013 * q[F3.iρ(g3, i, j, k)] for i in 1:nx, j in 1:ny, k in 1:nz]))
        for k in 1:nz, j in 1:ny, i in 1:nx
            ρk = x[F3.iρ(g3, i, j, k)]
            x[F3.iqr(m3, i, j, k)] = 2e-3 * ρk * (1 + 0.5 * sin(i + 2j + k))
            k > 3 && (x[F3.iqv(m3, i, j, k)] *= 0.8)                            # subsaturated aloft: evaporation
        end
        TS.frozen_step_end!(m3, x)
        gk = K.KAGrid(nx, ny, nz, L, Ly, H; zs = terr ? hill : nothing)
        mk = K.MoistKA(gk; mgas = mg, pvar, micro, sponge = (4e3, 0.05), rain = true)
        @test mk.nv == 9 && mk.nc == 31
        Q = K.ka_state(mk, m3, x)
        @test K.global_state(mk, m3, Q) == x
        N3 = TS.rhs_nonlinear(m3, x)
        out = K.ka_zeros(mk); K.ka_rhs!(out, mk, Q)
        @test maximum(abs, K.global_state(mk, m3, out) - N3) < 1e-9 * maximum(abs, N3)
        y3 = TS.run_frozen(m3, x, 3.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6, fast = :cs)
        K.ka_run!(mk, Q, 3.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6)
        @test maximum(abs, K.global_state(mk, m3, Q) - y3) < 1e-8 * maximum(abs, y3 - x)
        @test abs(K.ka_total(mk, Q, K.VR) / F3.total3d(m3, y3, :ρ) - 1) < 1e-13
        @test abs(K.ka_total(mk, Q, K.VQR) / F3.total3d(m3, y3, :qr) - 1) < 1e-10
        @test K.ka_precipitation_rate(mk, Q) ≈ F3.precipitation_rate(m3, K.global_state(mk, m3, Q))
    end
end

@testset "FV3D: Smagorinsky–Lilly subgrid-scale model (MoistModel3D, smagorinsky = …)" begin
    F3 = MultiRate.FV3D
    mgθ = CM.MoistGas()
    gas = CM.Gas(R = mgθ.Rd, γ = mgθ.Cpd / (mgθ.Cpd - mgθ.Rd), p00 = mgθ.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    Lx, Ly, H = 20e3, 16e3, 6e3
    kx, ky, kz = 2π / Lx, 2π / Ly, π / H
    hill(x, y) = 600.0 * ((1 + cos(kx * (x - 10e3))) * (1 + cos(ky * (y - 8e3))) / 4)^2      # periodic
    grid(n, terr) = terr ? F3.Grid3D(2n, 2n, n, Lx, Ly, H; zs = hill, terrain = :follow) : F3.Grid3D(2n, 2n, n, Lx, Ly, H)
    function sgs(m, m0, x)                       # the subgrid-scale part of S
        TS.frozen_step_end!(m, x)
        C = zeros(TS.frozen_ncoef(m)); S = zeros(m.nq); S0 = zeros(m.nq)
        TS.frozen_data!(C, S, m, x); TS.frozen_data!(C, S0, m0, x)
        return S - S0
    end
    # constant viscosity (cs = 0, ν0, Pr = 1), ρ = 1: ν ∇²q_v, q_v = f(x, y, η) with ∂η f = 0 at the walls; the ground
    # flux of f (zero in the model) is added to the reference of the lowest cells
    ν = 50.0; sm = F3.Smagorinsky(cs = 0.0, ν0 = ν, Pr = 1.0)
    function escalar(n, terr)
        g = grid(n, terr); b = F3.Model3D(g, gas, ρr, Θr)
        m = F3.MoistModel3D(b; mgas = mgθ, smagorinsky = sm); m0 = F3.MoistModel3D(b; mgas = mgθ)
        zs(x, y) = terr ? hill(x, y) : 0.0
        f(x, y, z) = 0.001 * cos(kx * x) * cos(ky * y + 0.3) * cos(kz * H * (z - zs(x, y)) / (H - zs(x, y)))
        d1(x, y, z, a; h = 0.5) = (e = h .* (a == 1, a == 2, a == 3); (f((x, y, z) .+ e...) - f((x, y, z) .- e...)) / 2h)
        lap(x, y, z; h = 1.0) = sum(f((x, y, z) .+ h .* (a == 1, a == 2, a == 3)...) + f((x, y, z) .- h .* (a == 1, a == 2, a == 3)...) for a in 1:3) / h^2 - 6f(x, y, z) / h^2
        x = vcat(F3.project3d(b, (x, y, z) -> (1.0, 0.0, 0.0, 0.0, 300.0)), zeros(m.nq - b.nq))
        for k in 1:n, j in 1:2n, i in 1:2n
            x[F3.iqv(m, i, j, k)] = 0.005 + f(g.xc[i], g.yc[j], g.Zc[i, j, k])
        end
        d = sgs(m, m0, x)
        function ex(i, j, k)
            xx, yy = g.xc[i], g.yc[j]; r = ν * lap(xx, yy, g.Zc[i, j, k])
            k > 1 && return r
            z = g.Zf[i, j, 1]
            return r + ν * (d1(xx, yy, z, 3) - g.zxf[i, j, 1] * d1(xx, yy, z, 1) - g.zyf[i, j, 1] * d1(xx, yy, z, 2)) / (g.Jc[i, j] * g.dz)
        end
        ref = maximum(abs(ν * lap(g.xc[i], g.yc[j], g.Zc[i, j, k])) for i in 1:2n, j in 1:2n, k in 1:n)
        @test abs(sum(g.Jc[i, j] * d[F3.iqv(m, i, j, k)] for i in 1:2n, j in 1:2n, k in 1:n)) < 1e-14 * sum(abs, d)   # conservative
        @test all(d[F3.iρ(g, i, j, k)] == 0 for i in 1:2n, j in 1:2n, k in 1:n)
        return maximum(abs(d[F3.iqv(m, i, j, k)] - ex(i, j, k)) for i in 1:2n, j in 1:2n, k in 1:n) / ref
    end
    e8, e16, e32 = escalar(8, false), escalar(16, false), escalar(32, false)
    @test e16 < 0.3 * e8 && e32 < 0.3 * e16 && e32 < 1e-3                         # second order
    t16, t32 = escalar(16, true), escalar(32, true)
    @test t32 < 0.6 * t16 && t32 < 4e-3                                                   # terrain following
    # momentum, flat: u = cos(kz z) cos(ky y), w = cos(kx x) sin(kz z): ∂t(ρu) = ν(∇²u + ∂x ∇·u), ∂t(ρw) = ν(∇²w + ∂z ∇·u)
    function emom(n)
        g = grid(n, false); b = F3.Model3D(g, gas, ρr, Θr)
        m = F3.MoistModel3D(b; mgas = mgθ, smagorinsky = sm); m0 = F3.MoistModel3D(b; mgas = mgθ)
        x = vcat(F3.project3d(b, (x, y, z) -> (1.0, cos(kz * z) * cos(ky * y), 0.0, cos(kx * x) * sin(kz * z), 300.0)), zeros(m.nq - b.nq))
        for k in 1:n, j in 1:2n, i in 1:2n; x[F3.iqv(m, i, j, k)] = 0.005; end
        d = sgs(m, m0, x)
        eu = maximum(abs(d[F3.iU(g, i, j, k)] + ν * ((ky^2 + kz^2) * cos(ky * g.yc[j]) + kx * kz * sin(kx * g.xf[i])) * cos(kz * g.zc[k]))
                     for i in 1:2n, j in 1:2n, k in 1:n) / (ν * (ky^2 + kz^2))
        ew = maximum(abs(d[F3.iW(g, i, j, k)] + ν * (kx^2 + 2kz^2) * cos(kx * g.xc[i]) * sin(kz * g.zf[k]))
                     for i in 1:2n, j in 1:2n, k in 3:n-1) / (ν * (kx^2 + 2kz^2))
        return eu, ew
    end
    (u8, w8), (u16, w16) = emom(8), emom(16)
    @test u16 < 0.3 * u8 && w16 < 0.3 * w8 && u16 < 0.015 && w16 < 0.015
    # full model (cs = 0.2, Lilly's stability correction, Scotti length), moist state with rain
    @test F3.filter_length(F3.Smagorinsky(), grid(4, false), 1, 1) ≈ (l1 = log(0.6); l2 = log(0.8); cbrt(2500.0 * 2000.0 * 1500.0) * cosh(sqrt(4 / 27 * (l1^2 - l1 * l2 + l2^2))))
    # neutral background with shear (stable layers keep ν = 0)
    ρn, Θn, _ = CM.stratified_reference(gas; θ0 = 300.0, N = 0.0)
    fq(x, y, z) = (ρ = ρn(z) * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)); (ρ, ρ * (8 + 10 * sin(y / 3e3) * z / H), ρ * (5 + 4 * cos(x / 4e3)), 2ρ * sin(π * z / H) * cos((x + 2y) / 5e3), Θn(z) * (1 + 1e-3 * cos((x - y) / 3e3))))
    for terr in (false, true)
        g = grid(4, terr); b = F3.Model3D(g, gas, ρr, Θr)
        q = F3.project3d(b, fq)
        mθ0 = F3.MoistModel3D(b; mgas = mgθ, rain = true)
        x = F3.moist_exner_state(mθ0, q, [0.004 * q[F3.iρ(g, i, j, k)] for i in 1:8, j in 1:8, k in 1:4])
        for k in 1:4, j in 1:8, i in 1:8
            x[F3.iqr(mθ0, i, j, k)] = 1e-4 * x[F3.iρ(g, i, j, k)] * (1 + sin(i + j))
            x[F3.iqc(mθ0, i, j, k)] = 1e-4 * x[F3.iρ(g, i, j, k)] * (1 + cos(i - 2j + k))
        end
        dθ = nothing
        # stable stratification (N = 0.01), uniform flow: Lilly's correction switches the model off
        xs = F3.moist_exner_state(mθ0, F3.balanced_state(b, Θr, ρr; U = 10.0), [0.004 * ρr(g.Zc[i, j, k]) for i in 1:8, j in 1:8, k in 1:4])
        ms = F3.MoistModel3D(b; mgas = mgθ, rain = true, smagorinsky = F3.Smagorinsky())
        terr || @test maximum(abs, sgs(ms, mθ0, xs)) == 0
        for thv in (:thetarho, :energy, :entropy)
            mg = CM.MoistGas(thvar = thv)
            m = F3.MoistModel3D(b; mgas = mg, rain = true, smagorinsky = F3.Smagorinsky())
            m0 = F3.MoistModel3D(b; mgas = mg, rain = true)
            xt = F3.thermo_convert(mθ0, m, x)
            d = sgs(m, m0, xt)
            νc = F3.eddy_viscosity(m, xt)
            @test maximum(νc) > 0 && minimum(νc) >= 0
            tot(f) = sum(g.Jc[i, j] * f(i, j, k) for i in 1:8, j in 1:8, k in 1:4)
            for idx in (F3.iqv, F3.iqc, F3.iqr)
                @test abs(tot((i, j, k) -> d[idx(m, i, j, k)])) < 1e-13 * tot((i, j, k) -> abs(d[idx(m, i, j, k)]))
            end
            # the tendency of ρθ_ρ is the same for all thermodynamic variables
            ε = 1.0
            mθ = F3.MoistModel3D(b; mgas = mgθ, rain = true)
            Θρ(z) = F3.thermo_convert(m, mθ, z)[[F3.iΘ(g, i, j, k) for i in 1:8, j in 1:8, k in 1:4]]
            dt = (Θρ(xt .+ ε .* d) .- Θρ(xt .- ε .* d)) ./ 2ε
            thv === :thetarho ? (dθ = dt; @test abs(tot((i, j, k) -> d[F3.iΘ(g, i, j, k)])) < 1e-12 * tot((i, j, k) -> abs(d[F3.iΘ(g, i, j, k)]))) :
                                @test maximum(abs, dt - dθ) < 1e-6 * maximum(abs, dθ)
            # π (or p) follows the equation of state
            dπ = [(F3.moist_pvar(m, (xt[F3.iρ(g, i, j, k)], xt[F3.iΘ(g, i, j, k)] + ε * d[F3.iΘ(g, i, j, k)], xt[F3.iqv(m, i, j, k)] + ε * d[F3.iqv(m, i, j, k)],
                    xt[F3.iqc(m, i, j, k)] + xt[F3.iqr(m, i, j, k)] + ε * (d[F3.iqc(m, i, j, k)] + d[F3.iqr(m, i, j, k)]))...) - F3.moist_pvar(m, xt[F3.iρ(g, i, j, k)], xt[F3.iΘ(g, i, j, k)], xt[F3.iqv(m, i, j, k)],
                    xt[F3.iqc(m, i, j, k)] + xt[F3.iqr(m, i, j, k)])) / ε for i in 1:8, j in 1:8, k in 1:4]
            @test maximum(abs, dπ .- [d[F3.iπ(m, i, j, k)] for i in 1:8, j in 1:8, k in 1:4]) < 1e-3 * maximum(abs, dπ)
            # MIS steps
            water(z) = F3.total3d(m, z, :qv) + F3.total3d(m, z, :qc) + F3.total3d(m, z, :qr)
            y = TS.run_frozen(m, xt, 2.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 6, fast = :cs)
            dM = F3.total3d(m, y, :ρ) - F3.total3d(m, xt, :ρ)          # rain leaves through the ground
            @test all(isfinite, y) && abs(water(y) - water(xt) - dM) < 1e-8 * abs(dM)
        end
    end
end

@testset "FV3D: stretched levels (zf), limiters of the face values; CutKA equals FV3D" begin
    F3 = MultiRate.FV3D; K = MultiRate.FV3DKA
    r3 = MultiRate.FV2D.recon3
    # the nonuniform third-order weights are exact for the cell means of a parabola
    Pp(x) = x + x^2 - x^3; av(a, b) = (Pp(b) - Pp(a)) / (b - a)
    @test abs(r3(av(0.0, 0.7), av(0.7, 1.0), av(1.0, 2.3), 1.0, 1.0, 1.0, 1.0, 1.0, true, 0.7, 0.3, 1.3) - (1 + 2 - 3)) < 1e-13
    # Koren (ASAM PhiLim) on a uniform grid is the textbook limiter; van Leer, upwind; extrema first order
    kor(a, b, c) = (δ = b - a; r = (c - b) / δ; b + 0.5 * max(0, min(2r, (1 + 2r) / 3, 2)) * δ)
    @test maximum(abs(r3(a, b, c, 1.0, 1.0, 1.0, 1.0, 1.0, true, 1.0, 1.0, 1.0, 1) - kor(a, b, c))
                  for a in -2:0.37:2, b in -2:0.41:2, c in -2:0.43:2 if a != b) < 1e-14
    @test r3(0.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, true, 1.0, 1.0, 1.0, 2) ≈ 1 + 2 / 3
    @test r3(0.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, true, 1.0, 1.0, 1.0, 1) == 1.0
    @test r3(0.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, true, 1.0, 1.0, 1.0, 3) == 1.0
    @test_throws ErrorException F3.Grid3D(4, 4, 4, 1.0, 1.0, 1.0; limiter = :superbee)
    gas = CM.Gas(); ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    F3.FV3D_PRESSURE_FORM[] = :product
    tab = TS.mis_tableau(:MIS54)
    hill(x, y) = 1500.0 * exp(-((x - 10e3)^2 + (y - 8e3)^2) / 3e3^2)
    # a uniform zf equals the default levels
    ga = F3.Grid3D(12, 8, 6, 20e3, 16e3, 6e3; zs = hill); gb = F3.Grid3D(12, 8, 6, 20e3, 16e3, 6e3; zs = hill, zf = 0.0:1e3:6e3)
    @test gb.uniform && gb.dz == ga.dz && gb.zc == ga.zc && gb.Vf == ga.Vf && gb.Az == ga.Az
    zf = [0.0, 500.0, 1000.0, 1500.0, 2000.0, 2700.0, 3700.0, 6000.0]
    for lim in (:none, :koren)
        g3 = F3.Grid3D(12, 8, 7, 20e3, 16e3, 6e3; zs = hill, zf, limiter = lim)
        @test !g3.uniform && isnan(g3.dz) && g3.dzc == diff(zf) && g3.zc ≈ (zf[1:end-1] .+ zf[2:end]) ./ 2
        @test g3.wlo[6] ≈ 700 / 1700 && g3.dzf[6] ≈ 850
        F3.FV3D_MASK_SOLVER[] = :gmres; F3.FV3D_MASK_TOL[] = 1e-12; F3.FV3D_MASK_TILES[] = (2, 2)
        for (jf, prec) in ((:partial, :columns), (:conservative, :amf))
            F3.FV3D_MASK_JFORM[] = jf
            m3 = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.02)
            ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))
            # the discretely balanced rest state stays at rest on the stretched cut-cell grid
            rest = F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k], 0.0, 0.0, 0.0, Θr(g3.zc[k]))))
            @test maximum(abs, TS.rhs_nonlinear(m3, rest)) < 1e-12
            q0 = F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k] * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)), 10ρa[k],
                                                                      -3ρa[k], 0.0, Θr(g3.zc[k]))))
            q3 = TS.run_frozen(m3, q0, 20.0, 3; tab, nsub = 4, fast = :hevi)
            @test abs(F3.total3d(m3, q3, :ρ) / F3.total3d(m3, q0, :ρ) - 1) < 1e-13
            @test abs(F3.total3d(m3, q3, :Θ) / F3.total3d(m3, q0, :Θ) - 1) < 1e-13
            m = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), jform = jf, prec)
            N3 = TS.rhs_nonlinear(m3, q0)
            Nk = K.global_state(m, g3, K.ka_rhs!(K.ka_zeros(m), m, K.ka_state(m, g3, q0)))
            @test maximum(abs, Nk - N3) < 1e-13 * maximum(abs, N3)
            Q = K.ka_state(m, g3, q0); K.ka_run!(m, Q, 20.0, 3; tab, nsub = 4, tol = 1e-12)
            @test maximum(abs, K.global_state(m, g3, Q) - q3) < 1e-9 * maximum(abs, q3 - q0)
        end
    end
    F3.FV3D_MASK_JFORM[] = :partial
    # nudging zone (slow force), divergence damping and off-centring: CutKA equals FV3D
    g3 = F3.Grid3D(12, 8, 7, 20e3, 16e3, 6e3; zs = hill, zf, limiter = :koren)
    nud = (x0 = 14e3, x1 = 20e3, αmax = 0.01, u = z -> 8.0, v = 1.0, θ = θr)
    m3 = F3.with_nudging(F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.02); nud...)
    ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))
    q0 = F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k] * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)), 10ρa[k],
                                                              -3ρa[k], 0.0, Θr(g3.zc[k]))))
    F3.FV3D_HEVI_BETA[] = 0.6
    m = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), nudge = nud)
    N3 = TS.rhs_nonlinear(m3, q0)
    @test maximum(abs, K.global_state(m, g3, K.ka_rhs!(K.ka_zeros(m), m, K.ka_state(m, g3, q0))) - N3) < 1e-13 * maximum(abs, N3)
    q3 = TS.run_frozen(m3, q0, 20.0, 3; tab, nsub = 4, fast = :hevi, smdiv = 0.1)
    Q = K.ka_state(m, g3, q0); K.ka_run!(m, Q, 20.0, 3; tab, nsub = 4, tol = 1e-12, smdiv = 0.1)
    @test maximum(abs, K.global_state(m, g3, Q) - q3) < 1e-9 * maximum(abs, q3 - q0)
    @test abs(F3.total3d(m3, q3, :ρ) / F3.total3d(m3, q0, :ρ) - 1) < 1e-13         # nudging leaves ρ alone
    F3.FV3D_HEVI_BETA[] = 0.5
    @test_throws ErrorException F3.MoistModel3D(F3.Model3D(F3.Grid3D(4, 4, 3, 4e3, 4e3, 3e3; zf = [0.0, 500.0, 1500.0, 3000.0]),
                                                           gas, ρr, Θr))
end

@testset "Rosenbrock W-methods (ASAM form), acoustic multigrid, masked small step with switchable linear algebra" begin
    F3 = MultiRate.FV3D; K = MultiRate.FV3DKA
    # orders on a stiff ODE: W = J (RosEul 1, Ros2AMF 2, Ros3Pw 3); W-property on a non-stiff ODE (Ros2AMF, Ros3Pw 2)
    mutable struct RosOde; ε::Float64; F::Any; pert::Float64; end
    fo(m, y) = [-(y[1] - cos(y[2])) / m.ε - sin(y[2]) * y[3], y[3], -y[2] + 0.1 * y[1]]
    Jo(m, y) = [-1/m.ε (-sin(y[2]) / m.ε - cos(y[2]) * y[3]) -sin(y[2]); 0 0 1; 0.1 -1 0]
    TS.rhs_nonlinear!(out, m::RosOde, y) = (out .= fo(m, y); out)
    TS.ros_prepare!(m::RosOde, y, γh) = (m.F = lu(I - γh * (Jo(m, y) + m.pert * [0 0.3 0; 0.2 0 0.1; 0.1 0 0.5])); m)
    TS.ros_solve!(k, m::RosOde, r) = (k .= m.F \ r; k)
    y0 = [1.0, 0.5, 0.2]
    for (ε, pert, cases) in ((1e-3, 0.0, ((:RosEul, 1), (:Ros2AMF, 2), (:Ros3Pw, 3))), (1.0, 1.0, ((:Ros2AMF, 2), (:Ros3Pw, 2))))
        m = RosOde(ε, nothing, 0.0)
        ref = TS.run_ros(m, y0, 1 / 8192, 8192; tab = TS.ros_tableau(:Ros3P)); m.pert = pert
        for (name, p) in cases
            e1, e2 = (norm(TS.run_ros(m, y0, 1 / n, n; tab = TS.ros_tableau(name)) - ref) for n in (64, 128))
            @test abs(log2(e1 / e2) - p) < 0.25
        end
    end
    # small hill: FV3D Rosenbrock (exact and AMF) keeps the rest state, conserves; CutKA = FV3D; multigrid iterations
    gas = CM.Gas(); ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    F3.FV3D_PRESSURE_FORM[] = :product; F3.FV3D_MASK_JFORM[] = :conservative
    hill(x, y) = 1500.0 * exp(-((x - 10e3)^2 + (y - 8e3)^2) / 3e3^2)
    zf = [0.0, 500.0, 1000.0, 1500.0, 2000.0, 2700.0, 3700.0, 6000.0]
    g3 = F3.Grid3D(16, 8, 7, 20e3, 16e3, 6e3; zs = hill, zf, limiter = :koren)
    mk(ros) = F3.with_rosenbrock(F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.02); ros)
    m3 = mk(:amf)
    ρa = F3.hydrostatic_column(m3, Θr; ρbottom = ρr(g3.zc[1]))
    rest = F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k], 0.0, 0.0, 0.0, Θr(g3.zc[k]))))
    q0 = F3.project3d(m3, (x, y, z) -> (k = F3.level(g3, z); (ρa[k] * (1 + 1e-3 * sin(x / 3e3) * cos(y / 4e3)), 10ρa[k],
                                                              -3ρa[k], 0.0, Θr(g3.zc[k]))))
    tab = TS.ros_tableau(:Ros2AMF)
    for ros in (:exact, :amf)
        mr = mk(ros)
        @test maximum(abs, TS.run_ros(mr, rest, 60.0, 2; tab) - rest) < 1e-12 * maximum(abs, rest)
        q = TS.run_ros(mr, q0, 10.0, 3; tab)
        @test abs(F3.total3d(mr, q, :ρ) / F3.total3d(mr, q0, :ρ) - 1) < 1e-13
        @test abs(F3.total3d(mr, q, :Θ) / F3.total3d(mr, q0, :Θ) - 1) < 1e-13
    end
    q3 = TS.run_ros(m3, q0, 10.0, 3; tab)
    its = Dict{Symbol,Vector{Int}}()
    for aprec in (:columns, :mg)
        m = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), jform = :conservative)
        Q = K.ka_state(m, g3, q0)
        _, w = K.ka_ros_run!(m, Q, 10.0, 3; tab, tol = 1e-11, aprec)
        @test maximum(abs, K.global_state(m, g3, Q) - q3) < 1e-8 * maximum(abs, q3 - q0)
        its[aprec] = w.its[1:2:end]
    end
    @test maximum(its[:mg]) <= 10 && minimum(its[:columns]) >= 5 * maximum(its[:mg])      # acoustic factor
    # stopping criterion in the acoustic energy norm (residuals in Pa: ρ c², Θ γR, m c): a drop of 1e-3 or an RMS
    # residual of 1e-2 Pa is close to the converged steps with fewer iterations; conservative for any tolerance
    m = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), jform = :conservative)
    Q = K.ka_state(m, g3, q0)
    w = K.CutRos(m, tab.s; aprec = :mg); K.ros_energy_norm!(w)
    for _ in 1:3; K.ka_ros_step!(m, Q, 10.0, tab, w; tol = 1e-3, atol = 1e-2); end
    q = K.global_state(m, g3, Q)
    @test maximum(abs, q - q3) < 1e-2 * maximum(abs, q3 - q0)
    @test abs(F3.total3d(m3, q, :ρ) / F3.total3d(m3, q0, :ρ) - 1) < 1e-13
    @test sum(w.its[1:2:end]) < sum(its[:mg])
    # masked small step (vertical = :explicit, gravity explicit): N(q) = FV3D; the masked AMF preconditioner with its
    # inner solvers gives the same steps as the column preconditioner, with fewer GMRES iterations; conserves
    F3.FV3D_MASK_JFORM[] = :partial
    m3p = F3.with_sponge(F3.Model3D(g3, gas, ρr, Θr), zeros(F3.nunknowns(g3)); zD = 4e3, αtop = 0.02)
    N3 = TS.rhs_nonlinear(m3p, q0)
    mtab = TS.mis_tableau(:MIS54)
    ref = nothing; colits = 0.0
    for (prec, mask) in ((:columns, (;)), (:mask, (acoustic = :hostlu, advection = :hostlu)), (:mask, (acoustic = :cg, advection = :gs, sweeps = 1)),
                         (:mask, (acoustic = :cg, advection = :bicgstab, momenta = :krylov)))
        m = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), vertical = :explicit, prec, mask)
        @test maximum(abs, K.global_state(m, g3, K.ka_rhs!(K.ka_zeros(m), m, K.ka_state(m, g3, q0))) - N3) < 1e-13 * maximum(abs, N3)
        Q = K.ka_state(m, g3, q0); K.ka_run!(m, Q, 2.0, 3; tab = mtab, nsub = 6, tol = 1e-12)
        q = K.global_state(m, g3, Q)
        ref === nothing && (ref = q; colits = sum(m.its) / length(m.its))
        @test maximum(abs, q - ref) < 1e-9 * maximum(abs, ref - q0)
        @test abs(F3.total3d(m3p, q, :ρ) / F3.total3d(m3p, q0, :ρ) - 1) < 1e-13
        @test abs(F3.total3d(m3p, q, :Θ) / F3.total3d(m3p, q0, :Θ) - 1) < 1e-13
        prec === :mask && @test sum(m.its) / length(m.its) < 0.6 * colits
    end
    # the AMF product as the solve (outer = :none, ASAM): conservative, close to the converged step
    m = K.CutKA(K.CutGrid(g3); gas, sponge = (4e3, 0.02), vertical = :explicit, prec = :mask,
                mask = (acoustic = :hostlu, advection = :hostlu, momenta = :krylov, tol = 1e-10, outer = :none))
    Q = K.ka_state(m, g3, q0); K.ka_run!(m, Q, 2.0, 3; tab = mtab, nsub = 6, tol = 1e-12)
    q = K.global_state(m, g3, Q)
    @test maximum(abs, q - ref) < 2e-3 * maximum(abs, ref - q0)
    @test abs(F3.total3d(m3p, q, :Θ) / F3.total3d(m3p, q0, :Θ) - 1) < 1e-13
    @test_throws ErrorException K.CutKA(K.CutGrid(g3); gas, prec = :mask)                      # needs vertical = :explicit
end

@testset "Buildings: CityGML LoD1/LoD2 → level set → cut-cell fractions (stretched levels)" begin
    BD = MultiRate.Buildings
    include(joinpath(@__DIR__, "buildings_citygml.jl"))
    # exact fractions of the linear interpolant: complement, symmetric cases, Monte Carlo
    rng = MersenneTwister(7)
    for _ in 1:20
        v = randn(rng, 4)
        @test BD.tet_fraction(v...) + BD.tet_fraction((-v)...) ≈ 1 atol = 1e-12
        w = randn(rng, 3)
        @test BD.tri_fraction(w...) + BD.tri_fraction((-w)...) ≈ 1 atol = 1e-12
    end
    @test BD.tet_fraction(1.0, 1.0, -1.0, -1.0) ≈ 0.5
    @test BD.tet_fraction(1.0, -1.0, -1.0, -1.0) ≈ 1 / 8
    v = (0.7, -0.2, 0.4, -1.3); n = 200_000; c = 0
    for _ in 1:n
        e = -log.(rand(rng, 4)); λ = e ./ sum(e)                  # uniform in the tetrahedron
        c += sum(λ .* v) > 0
    end
    @test abs(c / n - BD.tet_fraction(v...)) < 5e-3
    # convex decomposition (ear clipping, Hertel–Mehlhorn): L shape, comb
    Lsh = [(0.0, 0.0), (4.0, 0.0), (4.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)]
    cp = BD.convex_pieces(reverse(Lsh))
    @test length(cp) == 2 && sum(BD._area, cp) ≈ 6.0
    comb = [(0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (4.0, 2.0), (4.0, 1.0), (3.0, 1.0), (3.0, 2.0), (2.0, 2.0), (2.0, 1.0),
            (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)]
    cp = BD.convex_pieces(comb)
    @test sum(BD._area, cp) ≈ 8.0 && all(q -> all(BD._turn(q[mod1(t - 1, length(q))], q[t], q[mod1(t + 1, length(q))]) >= -1e-12
                                                   for t in eachindex(q)), cp) && length(cp) <= 4
    # reader and level set: block (LoD1 solid), gable house (LoD2), courtyard block (LoD1, hole); ground 100 m
    f = tempname() * ".gml"; write(f, synthetic_citygml())
    P = BD.read_citygml(f)
    @test length(P) == 15
    φ = BD.building_levelset(P; origin = (400000.0, 5650000.0), L = (120.0, 80.0))
    @test length(φ.solids) == 4                                    # block top, two roof faces, courtyard top
    @test φ.hmax ≈ 15.5 && φ.origin[3] ≈ 100.0 && length(φ.solids[4].holes) == 1
    @test BD.height_at(φ, 40.0, 40.0) ≈ 15.5 && BD.height_at(φ, 76.0, 45.25) ≈ 9.0 && BD.height_at(φ, 76.0, 42.75) ≈ 7.5
    @test BD.height_at(φ, 60.0, 40.0) == 0.0 && BD.height_at(φ, 98.5, 20.5) == 0.0 && BD.height_at(φ, 90.0, 20.5) ≈ 10.0
    # φ near the block: the largest signed plane distance (ASAM), min with the ground, capped by the band
    err = 0.0
    for x in 24.0:0.7:57.0, y in 26.0:0.7:52.0, z in 0.0:0.6:22.0
        ex = min(φ.band, max(30.3 - x, x - 50.7, 32.2 - y, y - 45.9, z - 15.5), z)
        err = max(err, abs(φ(x, y, z) - ex))
    end
    @test err < 1e-9                                               # GIS coordinates ~ 1e5–1e6 m shifted
    @test φ(98.5, 20.5, 5.0) > 0 && φ(90.0, 20.5, 5.0) < 0 && φ(76.0, 45.0, 8.7) < 0 && φ(76.0, 45.0, 9.0) > 0
    # fractions on 4 m columns, stretched levels: volume second order in the sub-cells, sections, solid interior
    zf = BD.stretched_levels(60.0; dz0 = 2.0, z1 = 20.0, r = 1.1)
    @test zf[1] == 0 && zf[end] == 60 && all(diff(zf) .> 0) && all(diff(zf)[1:10] .≈ 2.0) && diff(zf)[end] > 4
    ex = 20.4 * 13.7 * 15.5 + 900 + 3360
    solid(cg) = sum((1 - cg.Vf[i, j, k]) * 16 * (zf[k+1] - zf[k]) for i in 1:cg.nx, j in 1:cg.ny, k in 1:cg.nz)
    cg1 = BD.cut_geometry(φ; Δx = 4.0, zf, nsub = (4, 4), δsub = 1.0)
    cg2 = BD.cut_geometry(φ; Δx = 4.0, zf, nsub = (8, 8), δsub = 0.5)
    e1, e2 = abs(solid(cg1) / ex - 1), abs(solid(cg2) / ex - 1)
    @test e1 < 5e-3 && e2 < 1e-3                                   # errors only at edges and corners
    k6 = findfirst(==(6.0), zf)                                    # section z = 6: block, courtyard, house eaves
    @test abs(sum(1 .- cg2.Az[:, :, k6]) * 16 / (20.4 * 13.7 + 336 + 120) - 1) < 4e-3
    @test cg1.Vf[11, 10, 3] == 0 && cg1.Ax[11, 10, 3] == 0 && cg1.Az[11, 10, 3] == 0   # inside the block
    @test all(cg1.Vf[25, 6, 1:4] .== 1)                            # courtyard (cell 96–100 m × 20–24 m): air
    @test all(cg1.Vf[:, :, end] .== 1) && all(0 .<= cg1.Vf .<= 1) && all(0 .<= cg1.Ax .<= 1)
    # the auxiliary grid (cache) gives the same fractions where it resolves the sub-cells
    ls = BD.levelset(φ; δ = 1.0, δz = 1.0)
    @test maximum(abs, ls.φ[23:59, 27:53, 1:20] .- [φ((i - 1) * ls.δx, (j - 1) * ls.δy, (k - 1) * ls.δz) for i in 23:59, j in 27:53, k in 1:20]) < 1e-5
    zu = collect(0.0:2.0:40.0)
    cu = BD.cut_geometry(φ; Δx = 4.0, zf = zu, nsub = (4, 4), δsub = 1.0)
    cl = BD.cut_geometry(ls; Δx = 4.0, zf = zu, nsub = (4, 4), δsub = 1.0)
    @test maximum(abs, cu.Vf .- cl.Vf) < 1e-5
    # FV3D grid with uniform levels (2 m) from the fractions
    g = BD.fv3d_grid(cu)
    @test g.cut && g.nz == 20 && g.dz ≈ 2.0 && any(g.mask) && all((g.Vf .== 0) .| (g.Vf .== cu.Vf))
    gs = BD.fv3d_grid(cg1)                                         # stretched levels
    @test !gs.uniform && isnan(gs.dz) && gs.dzc ≈ diff(zf) && gs.Vf == cg1.Vf
    # small-cell removal: no cut cell below vmin, no face below amin, faces open only between fluid cells
    cr = deepcopy(cg2)
    nr, vr, nf = BD.remove_small_cells!(cr; vmin = 0.1, amin = 0.1)
    @test nr > 0 && nf > 0 && vr < 1e-3
    @test !any(0 .< cr.Vf .< 0.1) && !any(0 .< cr.Ax .< 0.1) && !any(0 .< cr.Ay .< 0.1) && !any(0 .< cr.Az .< 0.1)
    @test all(cr.Ax[i, j, k] == 0 || (cr.Vf[i, j, k] > 0 && cr.Vf[mod1(i - 1, cr.nx), j, k] > 0)
              for i in 1:cr.nx, j in 1:cr.ny, k in 1:cr.nz)
    @test all(cr.Az[i, j, k] == 0 || (cr.Vf[i, j, k] > 0 && cr.Vf[i, j, k-1] > 0) for i in 1:cr.nx, j in 1:cr.ny, k in 2:cr.nz)
    @test all((cr.Vf .== cg2.Vf) .| (cr.Vf .== 0))
end

@testset "MethodDesign: bicoloured B-series of MIS with frozen first argument" begin
    BC = Main.BiColourMIS
    @test [length(BC.trees(n)) for n in 1:4] == [1, 2, 7, 26]
    @test [count(BC.additive, BC.trees(n)) for n in 1:4] == [1, 2, 6, 18]
    # orders: additive F / non-additive F, as observed with the frozen toy problem
    for (name, pa, pf) in ((:WSRK3, 2, 2), (:MIS4a, 3, 2), (:MIS54, 4, 2), (:MIS_KW3, 3, 3))
        t = TS.mis_tableau(name)
        m = BC.mis_method(name, t.α, t.β, t.γ)
        @test BC.order_of(m, 5; additive_only = true, tol = 1e-9) == pa
        @test BC.order_of(m, 5; tol = 1e-9) == pf
    end
    # exact arithmetic: the WSRK3 residual of [o] vanishes exactly
    t = TS.mis_tableau(:WSRK3)
    m = BC.mis_method(:WSRK3, Rational{BigInt}.(rationalize.(t.α)), Rational{BigInt}.(rationalize.(t.β)), Rational{BigInt}.(rationalize.(t.γ)))
    @test all(iszero(r) for (tr, r) in BC.residuals(m, 2))
end

@testset "Euler2D: frozen split F(q, q) = N(q)" begin
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas)
    for zs in (x -> 0.0, x -> 300 / (1 + ((x - 20e3) / 4e3)^2))
        M = E2.Mesh2D(6, 3, 3, 40e3, 10e3; zs)
        m = E2.Euler2DModel(M, gas, ρr, Θr)
        q = CM.project2d(m, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.5 * exp(-((x - 15e3) / 5e3)^2)); (ρ, 10ρ, 2ρ * sin(x / 5e3), Θr(z))))
        C = zeros(TS.frozen_ncoef(m)); S = zeros(m.nq)
        TS.frozen_data!(C, S, m, q)
        F = TS.frozen_fast!(zeros(m.nq), m, C, q, :all, S)
        N = TS.rhs_nonlinear(m, q)
        @test maximum(abs.(F .- N)) < 1e-11 * maximum(abs, N)
        @test TS.frozen_fast!(zeros(m.nq), m, C, q, :thermo, S) .+ TS.frozen_fast!(zeros(m.nq), m, C, q, :momentum, S) ≈ F
    end
end

@testset "Euler2D: pressure variable, horizontal/vertical split, column-implicit :cs (DGPressureModel)" begin
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    L, H, U = 80e3, 10e3, 20.0
    hill(x) = 400 / (1 + ((x - L / 2) / 5e3)^2)
    dhill(x) = -400 * 2 * (x - L / 2) / 5e3^2 / (1 + ((x - L / 2) / 5e3)^2)^2
    M = E2.Mesh2D(8, 4, 3, L, H; zs = hill)
    base = E2.Euler2DModel(M, gas, ρr, Θr)
    m = E2.DGPressureModel(base)
    qb = CM.project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * exp(-((x - L / 3) / 5e3)^2));
                                        (ρ, ρ * U, ρ * U * dhill(x) * (H - z) / (H - hill(x)), Θr(z))))
    q = E2.pressure_state(m, qb)
    # N of the extended model: base N on the base rows, p_t = P'(Θ) Θ_t
    N = TS.rhs_nonlinear(m, q)
    Nb = TS.rhs_nonlinear(base, qb)
    @test maximum(abs.(N[1:base.nq] .- Nb)) < 1e-11 * maximum(abs, Nb)
    np = M.np
    @test maximum(abs(N[E2.ip_dg(m, e, i, j)] - gas.γ * q[E2.ip_dg(m, e, i, j)] / q[E2.node_iq(np, e, 4, i, j)] * N[E2.node_iq(np, e, 4, i, j)])
                  for e in 1:M.Ne, j in 1:np, i in 1:np) < 1e-10 * maximum(abs, N[base.nq+1:end])
    # parts, kernel and stage matrices
    C = zeros(TS.frozen_ncoef(m)); S = zeros(m.nq); TS.frozen_data!(C, S, m, q)
    z = q .+ 1e-3 .* sin.(1:m.nq) .* abs.(q)
    part(p) = TS.frozen_fast!(zeros(m.nq), m, C, z, p, S)
    F = part(:all)
    @test part(:hthermo) .+ part(:hmomentum) .+ part(:vertical) ≈ F
    @test part(:thermo) .+ part(:momentum) ≈ F
    TS.frozen_stage_begin!(m, C, z)
    for p in (:all, :hthermo, :hmomentum, :vertical)
        Fm = part(p); m.vert.valid = false
        @test maximum(abs.(Fm .- part(p))) < 1e-12 * maximum(abs, Fm)
        m.vert.valid = true
    end
    # exact column solve: z⁺ = z + D G_v(z⁺)
    zz = copy(z); k = zeros(m.nq); y = zeros(m.nq)
    TS.frozen_vertical_implicit!(zz, m, C, 2.0, 1.5, k, y)
    r = zeros(m.nq); TS.frozen_vertical_rhs!(r, m, C, zz, 2.0, 1.5)
    @test norm(zz .- z .- r) < 1e-10 * norm(zz .- z)
    # :cs on terrain: the error drops by 3.1, 2.5 for n = 10, 20, 40 and tends to first order for small h (as before
    # the well-balanced form of the momentum step: ratios 4.2, 3.5, 2.7, 2.2 for n = 10..160; the error now is ≤ the
    # old one at every n, e(160) = 0.37 both)
    TS.frozen_data!(C, S, m, q)
    tmp = ntuple(_ -> zeros(m.nq), 6)
    function fs(meth, n)
        Z = copy(q); TS.frozen_stage_begin!(m, C, Z)
        TS.frozen_fast_solve!(Z, m, C, S, 10.0 / n, n, k; method = meth, tmp)
    end
    zr = fs(:rk4, 400)
    e = [norm(fs(:cs, n) .- zr) for n in (10, 20, 40)]
    @test e[1] / e[2] > 2.9 && e[2] / e[3] > 2.4
    # Bokhove's partition (:csb): the flows A_c, B_c are exact (semigroup), the step is second order
    TS.frozen_stage_begin!(m, C, z)
    for fl in (:a, :b)
        z1 = copy(z); TS.frozen_hflow!(z1, m, C, S, fl, 1.5, z1); TS.frozen_hflow!(z1, m, C, S, fl, 1.5, z1)
        z2 = copy(z); TS.frozen_hflow!(z2, m, C, S, fl, 3.0, z2)
        @test norm(z1 .- z2) < 1e-10 * norm(z2 .- z)
    end
    @test maximum(length, m.vert.groups) <= 6                       # face-local blocks
    eb = [norm(fs(:csb, n) .- zr) for n in (10, 20, 40)]
    @test eb[1] / eb[2] > 3.5 && eb[2] / eb[3] > 3.5
    # MIS54 with :cs at the horizontal limit (4 times the Verlet step) is as accurate as explicit Verlet of the
    # base model (θ changes by 0.8 K; both ≈ 1.5e-2 K from RK4)
    dτv = 0.85 * 0.63 * minimum(M.Z[:, end, :] .- M.Z[:, 1, :]) / (7 * CM.sound_speed_max(base, qb))
    dτh = 0.85 * 0.63 * (L / M.Nex) / (7 * CM.sound_speed_max(base, qb))
    Δt, nst = 30.0, 10
    qv = TS.run_frozen(base, qb, Δt, nst; tab = TS.mis_tableau(:MIS54), nsub = ceil(Int, Δt / dτv), fast = :verlet)
    qc = TS.run_frozen(m, q, Δt, nst; tab = TS.mis_tableau(:MIS54), nsub = ceil(Int, Δt / dτh), fast = :cs)
    qr = TS.run_rk4(base, qb, Δt * nst / 3000, 3000)
    θ(x) = E2.field(base, x[1:base.nq], 4) ./ E2.field(base, x[1:base.nq], 1)
    @test dτh > 3dτv
    ev, ec = maximum(abs.(θ(qv) .- θ(qr))), maximum(abs.(θ(qc) .- θ(qr)))
    @test ev < 2e-2 && ec < 1.6ev
    qb2 = TS.run_frozen(m, q, Δt, nst; tab = TS.mis_tableau(:MIS54), nsub = ceil(Int, Δt / (1.6dτh)), fast = :csb)
    @test maximum(abs.(θ(qb2) .- θ(qr))) < 1.6ev                    # :csb with a 1.6 times larger small step
    @test qc[base.nq+1:end] ≈ [CM.pressure(gas, qc[E2.node_iq(np, e, 4, i, j)]) for e in 1:M.Ne for j in 1:np for i in 1:np]
end

@testset "Euler2D: column solve by static condensation (DGPressureModel, vsolver = :hdg)" begin
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    L, H, U = 80e3, 10e3, 20.0
    hill(x) = 400 / (1 + ((x - L / 2) / 5e3)^2)
    M = E2.Mesh2D(8, 4, 3, L, H; zs = hill)
    base = E2.Euler2DModel(M, gas, ρr, Θr)
    qb = CM.project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * exp(-((x - L / 3) / 5e3)^2)); (ρ, ρ * U, 0.0, Θr(z))))
    for vs in (:full, :contravariant_volume)
        ml = E2.DGPressureModel(base; lmars = :momentum, vsplit = vs)
        mh = E2.DGPressureModel(base; lmars = :momentum, vsplit = vs, vsolver = :hdg)
        z0 = E2.pressure_state(ml, qb); z0 .+= 1e-3 .* sin.(1:ml.nq) .* abs.(z0)
        C0 = zeros(TS.frozen_ncoef(ml)); S0 = zeros(ml.nq); TS.frozen_data!(C0, S0, ml, z0)
        for m in (ml, mh); TS.frozen_stage_begin!(m, C0, z0); end
        # the element rows of the hybrid form with M̂ = s m_f(z) are the LMARS operator
        Jx = mh.aux.vert.Jv; nqp = ml.nq
        dM = Vector(diag(Jx)[nqp+1:end])                              # -1 on the η-face constraint rows, 0 on the ξ-faces
        G = Diagonal([d == 0 ? 0.0 : -1 / d for d in dM]) * Jx[nqp+1:end, 1:nqp]   # s m_f(z) = G z
        @test norm(Jx[1:nqp, 1:nqp] * z0 + Jx[1:nqp, nqp+1:end] * (G * z0) - ml.vert.Jv * z0) < 1e-11 * norm(ml.vert.Jv * z0)
        R = sin.(1:ml.nq) .* abs.(z0)
        y1 = TS.frozen_vertical_solve!(zeros(ml.nq), ml, C0, 0.7, 0.9, R)
        y2 = TS.frozen_vertical_solve!(zeros(ml.nq), mh, C0, 0.7, 0.9, R)
        @test norm(y2 .- y1) < 1e-12 * norm(y1)
    end
end

@testset "Euler2D: contravariant momenta in the fast solver (vsplit = :contravariant_momentum), favoured S2" begin
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    L, H, U = 80e3, 10e3, 20.0
    hill(x) = 400 / (1 + ((x - L / 2) / 5e3)^2)
    isU(k, base) = k <= base.nq && mod1(cld(k, base.mesh.np^2), 4) == 2
    for zs in (x -> 0.0, hill)
        M = E2.Mesh2D(8, 4, 3, L, H; zs)
        base = E2.Euler2DModel(M, gas, ρr, Θr)
        qb = CM.project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * exp(-((x - L / 3) / 5e3)^2)); (ρ, ρ * U, 0.0, Θr(z))))
        mf = E2.DGPressureModel(base; lmars = :momentum)
        mc = E2.DGPressureModel(base; lmars = :momentum, vsplit = :contravariant_momentum)
        mh = E2.DGPressureModel(base; lmars = :momentum, vsplit = :contravariant_momentum, vsolver = :hdg)
        z0 = E2.pressure_state(mf, qb)
        C = zeros(TS.frozen_ncoef(mf)); S = zeros(mf.nq); TS.frozen_data!(C, S, mf, z0)
        for m in (mf, mc, mh); TS.frozen_stage_begin!(m, C, z0); end
        @test maximum(abs, mc.vert.Jh + mc.vert.Jv - mf.vert.Jh - mf.vert.Jv; init = 0.0) < 1e-12 * maximum(abs, mf.vert.Jv)
        Is, Js, Vs = findnz(mc.vert.P * mc.vert.Jv * mc.vert.Pinv)       # no Ũ in the transformed vertical part
        @test maximum(abs, [Vs[t] for t in eachindex(Is) if isU(Is[t], base) || isU(Js[t], base)]; init = 0.0) < 1e-12 * maximum(abs, Vs)
        R = sin.(1:mf.nq) .* abs.(z0)
        y0 = TS.frozen_vertical_solve!(zeros(mf.nq), mf, C, 0.7, 0.9, R)
        y1 = TS.frozen_vertical_solve!(zeros(mf.nq), mc, C, 0.7, 0.9, R)
        y2 = TS.frozen_vertical_solve!(zeros(mf.nq), mh, C, 0.7, 0.9, R)
        zs(L / 2) == 0 && @test norm(y1 .- y0) < 1e-12 * norm(y0)             # flat: identical to :full
        @test norm(y2 .- y1) < 1e-12 * norm(y1)                              # HDG = LU
    end
    # rest over the hill (isothermal, exactly balanced): MIS54 with the favoured S2 keeps it, dry and moist (no water)
    Tiso = 250.0
    mg = CM.MoistGas(); gm = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0)
    M = E2.Mesh2D(8, 4, 3, L, H; zs = hill)
    for (g_, moist) in ((gas, false), (gm, true))
        ρi(z) = g_.p00 / (g_.R * Tiso) * exp(-g_.g * z / (g_.R * Tiso))
        Θi(z) = g_.p00 / g_.R * (g_.R * Tiso * ρi(z) / g_.p00)^(1 / g_.γ)
        base = E2.Euler2DModel(M, g_, ρi, Θi)
        qi = CM.project2d(base, (x, z) -> (ρi(z), 0.0, 0.0, Θi(z)))
        m = E2.DGPressureModel(base; lmars = :momentum, vsplit = :contravariant_momentum, vsolver = :hdg, hpredict = false,
                               moist, mgas = mg)
        q0 = moist ? E2.moist_dg_state(m, qi, (x, z) -> 0.0) : E2.pressure_state(m, qi)
        @test maximum(abs, TS.rhs_nonlinear(m, q0)) < 1e-10                   # the operator is balanced
        mom(q) = maximum(abs, q[E2.node_iq(M.np, e, r, i, j)] for r in (2, 3), e in 1:M.Ne, j in 1:M.np, i in 1:M.np)
        # explicit Störmer–Verlet (S1, small step below the vertical limit) keeps the rest state
        q = TS.run_frozen(m, copy(q0), 10.0, 2; tab = TS.mis_tableau(:MIS54), nsub = 60, fast = :verlet)
        @test mom(q) < 1e-9                                                   # [kg/(m² s)]
        # S2 (:cs, :csfb) as well (well balanced: the horizontal momentum step acts on bⁿ, the explicit vertical force is
        # added after it, and the horizontal mass flux of a⁺ uses the momenta of the column solve with frozen a), also
        # with the midpoint predictor and with vsplit = :full
        for fast in (:cs, :csfb), (vs, hp) in ((:contravariant_momentum, false), (:contravariant_momentum, true), (:full, false))
            mv = E2.DGPressureModel(base; lmars = :momentum, vsplit = vs, vsolver = :hdg, hpredict = hp, moist, mgas = mg)
            q = TS.run_frozen(mv, copy(q0), 30.0, 4; tab = TS.mis_tableau(:MIS54), nsub = 24, fast, offcenter = 0.1, smdiv = 0.1)
            @test mom(q) < 1e-9
        end
    end
    # regression: gravity wave over the hill, favoured setting against RK4 of the same model
    M = E2.Mesh2D(8, 4, 3, L, H; zs = hill)
    base = E2.Euler2DModel(M, gas, ρr, Θr)
    qb = CM.project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.01 * exp(-((x - L / 3) / 5e3)^2)); (ρ, ρ * U, 0.0, Θr(z))))
    m = E2.DGPressureModel(base; lmars = :momentum, vsplit = :contravariant_momentum, vsolver = :hdg, hpredict = false)
    q0 = E2.pressure_state(m, qb)
    T = 600.0; nref = 2400
    qr = TS.run_rk4(m, q0, T / nref, nref)
    q = TS.run_frozen(m, copy(q0), 30.0, 20; tab = TS.mis_tableau(:MIS54), nsub = 24, fast = :cs, offcenter = 0.1, smdiv = 0.1)
    θ(x) = E2.field(base, x[1:base.nq], 4) ./ E2.field(base, x[1:base.nq], 1)
    @test maximum(abs, θ(q) .- θ(qr)) < 0.08     # K; measured 6.5e-2 (the MIS error: S1 and S2 :full give 7.8e-2, 6.7e-2)
end

@testset "Euler2D: moist air with the pressure as additional variable (DGPressureModel, moist = true)" begin
    mg = CM.MoistGas()
    gas = CM.Gas(R = mg.Rd, γ = mg.Cpd / (mg.Cpd - mg.Rd), p00 = mg.p0)
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    L, H = 80e3, 10e3
    M = E2.Mesh2D(8, 4, 3, L, H; zs = x -> 400 / (1 + ((x - L / 2) / 5e3)^2))
    base = E2.Euler2DModel(M, gas, ρr, Θr)
    qb = CM.project2d(base, (x, z) -> (ρ = Θr(z) / (θr(z) + 0.5 * exp(-((x - L / 3) / 5e3)^2 - ((z - 3e3) / 2e3)^2)); (ρ, ρ * 20, 0.0, Θr(z))))
    md = E2.DGPressureModel(base; lmars = :momentum)
    mm = E2.DGPressureModel(base; lmars = :momentum, moist = true, mgas = mg)
    # no water: the dry model (same gas constants)
    zd = E2.pressure_state(md, qb); z0 = E2.moist_dg_state(mm, qb, (x, z) -> 0.0)
    fd = TS.rhs_nonlinear!(zeros(md.nq), md, zd); f0 = TS.rhs_nonlinear!(zeros(mm.nq), mm, z0)
    @test norm(f0[1:md.nq] .- fd) < 1e-8 * norm(fd)                    # ∂p/∂Θ by central differences
    # with water (partly saturated): stage matrices = kernel, split parts add up, total water and mass conserved
    zq = E2.moist_dg_state(mm, qb, (x, z) -> 0.012 * exp(-z / 3000))
    N = M.np^2 * M.Ne
    @test maximum(zq[base.nq+2N+1:base.nq+3N]) > 0                     # condensate present
    C = zeros(TS.frozen_ncoef(mm)); S = zeros(mm.nq); TS.frozen_data!(C, S, mm, zq)
    y = zq .+ 1e-3 .* sin.(1:mm.nq) .* abs.(zq)
    f1 = TS.frozen_fast!(zeros(mm.nq), mm, C, y, :all, S)
    TS.frozen_stage_begin!(mm, C, zq)
    f2 = TS.frozen_fast!(zeros(mm.nq), mm, C, y, :all, S)
    @test norm(f1 .- f2) < 1e-12 * norm(f1)
    f3 = sum(TS.frozen_fast!(zeros(mm.nq), mm, C, y, part, S) for part in (:hthermo, :hmomentum, :vertical))
    @test norm(f3 .- f2) < 1e-12 * norm(f2)
    wgt = [M.J[i, j, e] * M.wg[i] * M.wg[j] for i in 1:M.np, j in 1:M.np, e in 1:M.Ne]
    @test abs(sum(vec(wgt) .* (f1[base.nq+N+1:base.nq+2N] .+ f1[base.nq+2N+1:base.nq+3N]))) <
          1e-10 * sum(vec(wgt) .* abs.(f1[base.nq+N+1:base.nq+2N]))
    @test abs(sum(wgt[i, j, e] * f1[E2.node_iq(M.np, e, 1, i, j)] for i in 1:M.np, j in 1:M.np, e in 1:M.Ne)) < 1e-12 * sum(wgt)
    # p tendency = d p_eos/dt along the full tendency (O(h) finite difference)
    fq = TS.rhs_nonlinear!(zeros(mm.nq), mm, zq); h = 1e-3; z2 = zq .+ h .* fq
    pe = [(E2._dgp_peos(mm, z2, e, i, j) - E2._dgp_peos(mm, zq, e, i, j)) / h for i in 1:M.np, j in 1:M.np, e in 1:M.Ne]
    @test maximum(abs, fq[base.nq+1:base.nq+N] .- vec(pe)) < 1e-5 * maximum(abs, pe)
    # condensed column solve
    mh = E2.DGPressureModel(base; lmars = :momentum, moist = true, mgas = mg, vsolver = :hdg)
    TS.frozen_stage_begin!(mh, C, zq)
    R = sin.(1:mm.nq) .* abs.(zq)
    y1 = TS.frozen_vertical_solve!(zeros(mm.nq), mm, C, 0.7, 0.9, R)
    y2 = TS.frozen_vertical_solve!(zeros(mm.nq), mh, C, 0.7, 0.9, R)
    @test norm(y2 .- y1) < 1e-12 * norm(y1)
end

@testset "LinearAnalysis: MIS amplification (Knoth–Wensch, MISDG grid)" begin
    LA = MultiRate.LinearAnalysis
    tab(n) = TS.mis_tableau(n)
    kw(n, cs, ca) = LA.max_amplification(θ -> LA.kw_mode(θ; C_sound = cs, C_adv = ca), tab(n); fast = :exact)
    @test kw(:WSRK3, 0.0, 1.62) <= 1 + 1e-10 && kw(:WSRK3, 0.0, 1.64) > 1      # pure advection limit 1.63
    @test kw(:MIS54, 0.0, 2.05) <= 1 + 1e-10
    @test kw(:WSRK3, 3.5, 0.5) > 1.01                                       # undamped RK3: C_sound ≲ 3
    @test kw(:MIS54, 12.0, 0.5) <= 1 + 1e-10
    # the real fast loops converge to the exact fast stages
    m = LA.staggered_mode(1.0; C_sound = 3.0, C_adv = 0.4)
    Rex = LA.step_matrix(m, 1.0, tab(:MIS54); fast = :exact)
    e = [norm(LA.step_matrix(m, 1.0, tab(:MIS54); nsub = n, fast = :verlet) - Rex) for n in (24, 48)]
    @test rate(e...) > 1.8
    # WRF divergence damping stabilizes WSRK3 on the MISDG grid (C_sound = 5, nsub = 12)
    sg(n, f) = LA.max_amplification(θ -> LA.staggered_mode(θ; C_sound = 5.0, C_adv = 5 / 7), tab(n); fast = f, nsub = 12)
    @test sg(:WSRK3, :verlet) > 1.05 && sg(:WSRK3, :wrf) < 1
    @test sg(:MIS54, :verlet) < 1
    # 1D pulse: MIS54 fourth order
    T = 1e4
    @test rate(LA.pulse_error(tab(:MIS54), T / 200; fast = :exact), LA.pulse_error(tab(:MIS54), T / 400; fast = :exact)) > 3.8
end

@testset "LinearAnalysis: matrix-free step and Arnoldi" begin
    LA = MultiRate.LinearAnalysis
    m = LA.staggered_mode(0.7; C_sound = 5.0, C_adv = 5 / 7)
    R = LA.step_matrix(m, 1.0, TS.mis_tableau(:WSRK3); nsub = 12, fast = :verlet)
    n = size(R, 1)
    A! = LA.linearized_step(m, zeros(n), 1.0, TS.mis_tableau(:WSRK3); nsub = 12, fast = :verlet)
    v = collect(1.0:n); out = zeros(n)
    @test A!(out, v) ≈ R * v rtol = 1e-6
    B = randn(MersenneTwister(3), 200, 200) / sqrt(200); B[1, 1] = 1.5
    λ, X, res = LA.arnoldi_eigs((o, x) -> mul!(o, B, x), 200; nev = 2, v0 = ones(200))
    ev = eigvals(B)
    @test abs(λ[1]) ≈ maximum(abs, ev) rtol = 1e-6
    @test norm(B * X[:, 1] - λ[1] * X[:, 1]) < 1e-4 * norm(X[:, 1])
end

@testset "LinearAnalysis: Fourier blocks of the FV model, Boussinesq gravity wave" begin
    LA = MultiRate.LinearAnalysis
    gas = CM.Gas()
    ρr, Θr, θr = CM.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    g = FV.FVGrid(8, 4, 80e3, 4e3); m = FV.FVModel(g, gas, ρr, Θr); nz = g.nz
    qb = CM.project2d(m, (x, z) -> (ρr(z), 10ρr(z), 0.0, Θr(z)))
    idx = vcat([[FV.iρ(g, i, k) for i in 1:g.nx] for k in 1:nz], [[FV.iU(g, i, k) for i in 1:g.nx] for k in 1:nz],
               [[FV.iW(g, i, k) for i in 1:g.nx] for k in 2:nz], [[FV.iΘ(g, i, k) for i in 1:g.nx] for k in 1:nz])
    thermo = vcat(trues(nz), falses(nz), falses(nz - 1), trues(nz)); hmom = vcat(falses(nz), trues(nz), falses(nz - 1), falses(nz))
    λb = sort(vcat([eigvals(LA.linearized_mode_model(m, qb, idx, 2π * j / g.nx; thermo, hmom)[3]) for j in 0:g.nx-1]...); by = x -> (imag(x), real(x)))
    λf = sort(eigvals(FV.fast_matrix(FV.fast_operator(m, qb; reuse = false))); by = x -> (imag(x), real(x)))
    @test maximum(abs.(λb .- λf)) < 1e-8 * maximum(abs, λf)
    # the Fourier transform pair
    δ = randn(length(qb))
    @test LA.fourier_synthesis(LA.fourier_coefficients(δ, idx), idx, length(δ)) ≈ δ
    # MIS4 + FV hevi at U Δt/Δz ≈ 17 (Δx = 20 km, Δz = 625 m, Δt = 540 s, U = 20 m/s): unstable with the
    # vertical transport -∂z(ū W') in the slow part, stable with it in L (FV_VADV_FAST)
    g2 = FV.FVGrid(30, 16, 600e3, 10e3); m2 = FV.FVModel(g2, gas, ρr, Θr); nz2 = g2.nz
    qb2 = CM.project2d(m2, (x, z) -> (ρr(z), 20ρr(z), 0.0, Θr(z)))
    idx2 = vcat([[FV.iρ(g2, i, k) for i in 1:g2.nx] for k in 1:nz2], [[FV.iU(g2, i, k) for i in 1:g2.nx] for k in 1:nz2],
                [[FV.iW(g2, i, k) for i in 1:g2.nx] for k in 2:nz2], [[FV.iΘ(g2, i, k) for i in 1:g2.nx] for k in 1:nz2])
    th2 = vcat(trues(nz2), falses(nz2), falses(nz2 - 1), trues(nz2)); hm2 = vcat(falses(nz2), trues(nz2), falses(nz2 - 1), falses(nz2))
    ρ7(v) = (FV.FV_VADV_FAST[] = v;
             LA.spectral_radius(LA.step_matrix(LA.linearized_mode_model(m2, qb2, idx2, 2π * 7 / g2.nx; thermo = th2,
                                hmom = hm2, Δt = 540.0)[1], 1.0, TS.mis_tableau(:MIS4); nsub = 12, fast = :hevi_verlet)))
    @test ρ7(false) > 1.1
    @test ρ7(true) < 1
    FV.FV_VADV_FAST[] = false
    # constant-coefficient Boussinesq long channel: MIS4 with the FV fast loop is accurate at Δt = 540 s
    x, bn, be = LA.gravity_wave_section(TS.mis_tableau(:MIS4), 60000 / 111; nx = 100, fast = :hevi_verlet)
    @test maximum(abs.(bn - be)) < 0.05 * maximum(abs.(be))
end

@testset "MIS: polynomial coupling, order conditions, 1D acoustic-advection test" begin
    MM = MultiRate.MIS
    @test MM.tree_count(5) == [1, 2, 7, 26, 107]
    @test MM.tree_count(5; class = :B) == [1, 2, 6, 20, 73]
    @test MM.tree_count(5; class = :A) == [1, 2, 6, 18, 60]
    @test MM.tree_count(5; class = :L) == [1, 2, 5, 13, 37]
    ts = MM.trees(3); r = MM.residuals(MM.etd2rk(), 3; ts)          # table of the note, ETD2RK
    @test [r[k] for k in eachindex(ts) if ts[k].n == 3] == [1//6, 0, 0, -1//6, 0, 1//12, 0]
    m54 = MM.PolyMIS(TS.mis_tableau(:MIS54))
    @test MM.order(m54) == 2 && MM.order(m54; class = :A) == 4
    @test MM.order(MM.PolyMIS(TS.mis_tableau(:MISF53))) == 3
    @test MM.stage_times(m54)[end] ≈ 1
    A, b = MM.base_method(m54); @test sum(b) ≈ 1 && b' * sum(A; dims = 2)[:] ≈ 1 / 2
    S, L = MM.fv_symbols(1.7)                                       # same step matrix as LinearAnalysis
    R1 = MM.amplification(m54, 1.3 .* S, 6.0 .* L)
    R2 = MultiRate.LinearAnalysis.step_matrix(MultiRate.LinearAnalysis.kw_mode(1.7; C_sound = 6.0, C_adv = 1.3), 1.0,
                                              TS.mis_tableau(:MIS54); fast = :exact)
    @test maximum(abs, eigvals(R1)) ≈ maximum(abs, eigvals(R2)) rtol = 1e-10
    S, L = MM.dg_symbols(0.05; P = 3)                               # consistent semi-discretization
    @test minimum(abs.(eigvals(L) .- 0.0125im)) < 1e-8 && minimum(abs.(eigvals(S) .+ 0.0125im)) < 1e-8
    S, L = MM.dg_symbols(1.1; P = 3, flux = :central)              # substeps converge (Verlet: 2nd order)
    Re = MM.amplification(m54, 0.8S, 3L)
    e = [opnorm(MM.amplification(m54, 0.8S, 3L; inner = (:verlet, n)) - Re) for n in (20, 40)]
    @test 3.5 < e[1] / e[2] < 4.5
    e2 = MM.etd2rk(Float64); Re = MM.amplification(e2, 0.8S, 3L)     # polynomial forcing, RK4 substeps
    e = [opnorm(MM.amplification(e2, 0.8S, 3L; inner = (:rk4, n)) - Re) for n in (20, 40)]
    @test e[1] / e[2] > 14
    @test MM.adv_limit(m54, MM.symbols(:fv); Csound = 0, θs = range(0, π, length = 32)) ≈ 2.05 atol = 0.01
    @test MM.order(MM.cf4()) == 4                                   # commutator-free Lie group methods
    @test MM.order(MM.luscher3()) == 3
    @test MM.order(MM.ck54_cf(); tol = 1e-9) == 4                   # Bazavov's low-storage format, RK(5,4)
    @test all(iszero(r) for (_, r) in MM.restart_conditions(MM.cf4()))          # matrix form, restart class
    rw = MM.restart_conditions(MM.etd2rk())
    @test rw[9][2] == 0 && rw[3][2] != 0          # ETD2RK: coupling condition of order 3 holds, base b'c^2 fails
    @test count(==('\n'), MM.latex_conditions(4)) > 26 && occursin("\\btree", MM.latex_conditions(3; class = :L))
    @test MM.tree_count(5; class = :BA) == [1, 2, 7, 25, 99]                   # F = f(y) + B(y) z + g(z)
    ic = MM.independent_conditions(4; class = :BA, s = 6, K = 1, samples = 2)  # restart class: as for F(y, z)
    @test ic.coupling == [0, 0, 1, 4] && ic.base == [1, 1, 2, 4]
    Ab, bb = MM.base_method(m54); zz = 0.4 + 0.3im                  # base RK: R(z) = 1 + b'z(I - zA)⁻¹1
    @test MM.rk_amplification(m54, fill(zz, 1, 1))[1] ≈ 1 + bb' * ((I - zz * Ab) \ (zz * ones(length(bb))))
    @test MM.max_amplification(m54, MM.symbols(:fv); Cadv = 0.7, Csound = 0) ≈
          MM.max_amplification(m54, MM.symbols(:fv); Cadv = 0.7, Csound = 0, base = true)
end

end
