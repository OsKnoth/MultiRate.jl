# Unit tests (fast). The detailed verification reports are in examples/v*_report.jl.
#
#     julia --project -e 'using Pkg; Pkg.test()'

using MultiRate
using LinearAlgebra, Random, Test

const V1 = MultiRate.HDGRelaxEuler
const V2 = MultiRate.HDGRelaxTheta
const V3 = MultiRate.SplitExplicitEuler
const E2 = MultiRate.Euler2D
const FV = MultiRate.FV2D
const TS = MultiRate.TimeStepping
const CM = MultiRate.Common

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

@testset "MultiRate" begin

@testset "v1 HDGRelaxEuler: periodic normal mode" begin
    atm = V1.Atmosphere(T0 = 250.0, Γ = 0.0)
    Lz = 2000.0
    c = sqrt(V1.c2(atm, 0.0)); g, γ = atm.g, atm.γ
    k = 2π / Lz
    A = [0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 c^2 0.0]
    S = [0.0 0.0 0.0; -g 0.0 0.0; 0.0 (-(γ - 1) * g) 0.0]
    E = eigen(ComplexF64.(-im * k .* A .+ S))
    i = argmax(imag.(E.values)); λ = E.values[i]
    v = E.vectors[:, i] ./ E.vectors[3, i] .* 100
    ex(t) = z -> real.(v .* exp.(im * k .* permutedims(z) .+ λ * t))
    Tend = Lz / c
    errs = map((8, 16)) do Ne
        s = V1.HDGRelax(Ne, 2, 0.0, Lz, atm; bc = :periodic)
        n = ceil(Int, Tend / (s.h / c / 5))
        U = V1.run!(s, V1.initial_state(s, V1.project(s, ex(0.0))), Tend / n, n)
        V1.l2_error(s, U[1:s.nq], ex(Tend))[3]
    end
    @test rate(errs...) > 2.5
end

@testset "v2 HDGRelaxTheta: fast operator" begin
    atm = V2.Atmosphere(T0 = 288.0, Γ = 0.0065)
    for pfac in (1.0, 1.4)
        fr = V2.frozen_from_atmosphere(atm; pfac = pfac)
        s = V2.HDGRelaxTh(10, 3, 0.0, 1e4, fr; bc = :wall)
        λ = eigvals(Diagonal(1 ./ s.Mdiag) * Matrix(s.K + s.G * s.H))
        @test maximum(real.(λ)) < 1e-12                                  # energy stable (split form)
        Z = V2.nodes(s)                                                    # nodal constructor ≈ function one
        s2 = V2.HDGRelaxTh(10, 3, 0.0, 1e4, V2.coef_a.(Ref(fr), Z), fr.theta.(Z), V2.coef_b.(Ref(fr), Z); g = fr.g)
        @test norm(s2.K - s.K) / norm(s.K) < 1e-9
        qb = V2.balanced_state(s, z -> V2.atm_Theta(atm, z))                    # well balanced
        U = V2.run!(s, V2.initial_state(s, qb), 0.05, 20)
        @test maximum(abs.(U[1:s.nq] .- qb) ./ max.(abs.(qb), 1.0)) < 1e-10
    end
    s = V2.HDGRelaxTh(8, 2, 0.0, 1e4, V2.frozen_from_atmosphere(atm); bc = :wall)
    s_std = V2.HDGRelaxTh(8, 2, 0.0, 1e4, V2.frozen_from_atmosphere(atm); bc = :wall, form = :standard)
    λstd = eigvals(Diagonal(1 ./ s_std.Mdiag) * Matrix(s_std.K + s_std.G * s_std.H))
    @test maximum(real.(λstd)) > 1e-6                                     # standard form is not
    q = randn(MersenneTwister(1), s.nq)                                                       # split-explicit hook
    U = V2.run!(s, V2.initial_state(s, q), 0.5, 10; method = :imex, F_slow = -V2.tendency(s, q))
    @test maximum(abs.(U[1:s.nq] .- q)) < 1e-10
end

@testset "v3 SplitExplicitEuler: MIS for nonlinear Euler" begin
    atm = V3.Atmosphere(T0 = 288.0, Γ = 0.0065)
    eu = V3.Euler1D(10, 3, 0.0, 1e4)
    qb = V3.hydrostatic_state(eu, z -> V3.atm_Theta(atm, z))
    @test maximum(abs.(V3.rhs_nonlinear(eu, qb))) < 1e-10
    dτ = 0.1 * eu.h / (V3.sound_speed_max(eu, qb) * (2eu.p + 1))     # acoustic substep
    q = V3.run_mis(eu, qb, 12dτ, 5; tab = V3.mis_tableau(:MIS4a), nsub = 12)
    @test maximum(abs.(q .- qb) ./ max.(abs.(qb), 1.0)) < 1e-11

    pert(z) = begin
        a = 1.4 .* V3.atm_p.(Ref(atm), z) ./ V3.atm_Theta.(Ref(atm), z)
        thp = 0.5 .* exp.(-((z .- 6000) ./ 1000) .^ 2)
        vcat(permutedims(-V3.atm_rho.(Ref(atm), z) .* thp ./ V3.atm_theta.(Ref(atm), z)), zeros(1, length(z)),
             permutedims(100.0 .* exp.(-((z .- 3000) ./ 1000) .^ 2) ./ a))
    end
    q0 = qb .+ V3.project(eu, pert)
    Tend = 4.0
    dtr = 0.2 * eu.h / (V3.sound_speed_max(eu, q0) * (2eu.p + 1))
    nref = ceil(Int, Tend / dtr)
    qref = V3.run_rk4(eu, q0, Tend / nref, nref)
    for (name, order) in ((:MIS2, 2), (:MIS4, 3), (:WSRK3, 2))
        e = [V3.l2_error(eu, V3.run_mis(eu, q0, Δt, round(Int, Tend / Δt); tab = V3.mis_tableau(name),
                                        nsub = 24) .- qref)[2] for Δt in (1.0, 0.5)]
        @test rate(e...) > min(order, 2) - 0.4      # fixed ratio nsub: the 2nd-order fast integrator limits the rate
    end
    q = V3.run_mis(eu, q0, 12dτ, 4; tab = V3.mis_tableau(:MIS4), nsub = 12)
    @test abs(V3.total(eu, q, 1) - V3.total(eu, q0, 1)) < 1e-12 * V3.total(eu, q0, 1)
    @test abs(V3.total(eu, q, 3) - V3.total(eu, q0, 3)) < 1e-12 * V3.total(eu, q0, 3)

    # linear stability of N about the hydrostatic state: two-point gravity is neutral, the
    # pointwise source -g ρ_i has a real positive eigenvalue O(h)
    @test jacobian_max_re(q -> V3.rhs_nonlinear(eu, q), qb) < 1e-10
    eup = V3.Euler1D(10, 3, 0.0, 1e4; volume_flux = :kg, gravity = :pointwise)
    @test jacobian_max_re(q -> V3.rhs_nonlinear(eup, q), V3.hydrostatic_state(eup, z -> V3.atm_Theta(atm, z))) > 1e-6

    s = V3.fast_operator(eu, q)                                              # re-frozen, non-smooth state
    λ = eigvals(Diagonal(1 ./ s.Mdiag) * Matrix(s.K + s.G * s.H))
    @test maximum(real.(λ)) < 1e-12
end

@testset "Euler2D: x–z slice with terrain" begin
    gas = V3.Gas()
    ρr, Θr, θr = E2.stratified_reference(gas; θ0 = 300.0, N = 0.01)
    for zs in (x -> 0.0, x -> 400.0 * exp(-((x - 10e3) / 2e3)^2))
        M = E2.Mesh2D(5, 3, 3, 20e3, 10e3; zs = zs)
        @test maximum(abs(sum(M.D[i, k] * M.Jξx[k, j, e] + M.D[j, k] * M.Jηx[i, k, e] for k in 1:M.np))
                      for i in 1:M.np, j in 1:M.np, e in 1:M.Ne) < 1e-9                    # metric identities
        m = E2.Euler2DModel(M, gas, ρr, Θr)
        qr = E2.project2d(m, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
        @test maximum(abs.(E2.rhs_nonlinear(m, qr))) < 1e-12                               # rest state
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
        E2.hevi_solve!(Y, sv, E2.hevi_factor(sv, 3.0 * V2.IMEX_GM), Ys)   # factor of I - c A_v, c = dt γ
        @test norm(Y .- (I - 3.0 * V2.IMEX_GM * Av) \ Ys) < 1e-12 * norm(Y)
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
    qref = V3.run_rk4(m, q0, 100 / nref, nref)
    dev = norm(qref .- q0)
    for (fast, Δts) in ((:imex, (4.0, 2.0)), (:hevi, (10.0, 5.0)))                   # nsub = 12: dτ refined with Δt
        e = [norm(V3.run_mis(m, q0, Δt, round(Int, 100 / Δt); tab = V3.mis_tableau(:MIS4), nsub = 12,
                             fast = fast) .- qref) for Δt in Δts]
        @test e[2] < 0.05 * dev
        @test rate(e...) > 1.7
    end
    qb = E2.project2d(m, (x, z) -> (ρr(z), 0.0, 0.0, Θr(z)))
    q = V3.run_mis(m, qb, 10.0, 3; tab = V3.mis_tableau(:MIS4), nsub = 12, fast = :hevi)
    @test maximum(abs.(q .- qb) ./ max.(abs.(qb), 1.0)) < 1e-10                              # well balanced

    # thermodynamic/momentum split with the interface momentum, Verlet and forward-backward
    let m = m, q0 = q0
        s2 = E2.fast_operator(m, q0; reuse = false)
        kf = zeros(m.nq); kT = zeros(m.nq); km = zeros(m.nq)
        V2.fast_apply!(kf, s2, q0, s2.fb)
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
                e[k] = 1.0; V2.fast_solve!(y, ss, e, F, dt, 1; method = method); A[:, k] .= y; e[k] = 0.0
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
            e[k] = 1.0; V2.fast_solve!(y, sc, e, F, dt, 1; method = :verlet); A[:, k] .= y; e[k] = 0.0
        end
        μ = abs.(eigvals(A))
        @test maximum(abs.(μ .- 1)) < 1e-8
        qq = copy(q0)
        @test allocs(() -> V2.fast_solve!(qq, s2, q0, nothing, 0.05, 5; method = :verlet)) == 0
        @test allocs(() -> V2.fast_solve!(qq, s2, q0, nothing, 0.05, 5; method = :fb)) == 0
        sh2 = E2.fast_operator(m, q0; fast = :hevi, reuse = false)
        @test allocs(() -> V2.fast_solve!(qq, sh2, q0, nothing, 0.5, 5; method = :hevi)) < 50_000
    end
    e = [norm(V3.run_mis(m, q0, Δt, round(Int, 100 / Δt); tab = V3.mis_tableau(:MIS4), nsub = 12,
                         fast = :verlet) .- qref) for Δt in (8.0, 4.0)]
    @test e[2] < 0.05 * dev
    @test rate(e...) > 1.7
    q = V3.run_mis(m, qb, 8.0, 3; tab = V3.mis_tableau(:MIS4), nsub = 12, fast = :verlet)
    @test maximum(abs.(q .- qb) ./ max.(abs.(qb), 1.0)) < 1e-10

    # allocation-free kernels (in a let block: the testset reassigns m, s, q, which would be boxed)
    let m = m, q0 = q0
        out = zeros(m.nq)
        @test allocs(() -> V3.rhs_nonlinear!(out, m, q0)) == 0
        sh = E2.fast_operator(m, q0; fast = :hevi)
        @test allocs(() -> V2.fast_apply!(out, sh, q0, sh.fb)) == 0
        qq = copy(q0)
        @test allocs(() -> V2.fast_solve!(qq, sh, q0, nothing, 0.1, 5; method = :imex)) == 0
        tab = V3.mis_tableau(:MIS4); w = V3.MISWork(m.nq, 4)
        @test allocs(() -> V3.mis_step!(qq, m, q0, 4.0, tab, sh, w; nsub = 12, fast = :imex)) == 0
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

end
