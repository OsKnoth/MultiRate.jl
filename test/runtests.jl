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
    qb = E2.project2d(m, (x, z) -> (ρi(z), 0.0, 0.0, Θi(z)))                                  # isothermal rest
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
    @test_throws ErrorException TS.frozen_ncoef(FV.FVModel(FV.FVGrid(8, 4, 40e3, 10e3; zs = x -> 100 * exp(-(x - 2e4)^2 / 1e8)), gas, ρr, Θr))
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
        @test mf.τ ≈ 0.25 * 0.85 * 0.92 * min(g.dx, g.dz) / 350                # default: a quarter of the acoustic step
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
        # implicit = :amf (default with cut cells): masked momentum advection (simple upwind) in the
        # fast part; N unchanged, parts consistent, assembled matrix = matrix-free operator, short run
        ma = F3.Model3D(gc, gas, ρr, Θr); @test ma.implicit === :amf
        @test maximum(abs, TS.rhs_nonlinear(ma, q) - N) < 1e-12 * maximum(abs, N)
        Ca = zeros(TS.frozen_ncoef(ma)); Sa = zeros(ma.nq); TS.frozen_data!(Ca, Sa, ma, q)
        oa(p) = TS.frozen_fast!(zeros(ma.nq), ma, Ca, q, p, Sa)
        @test maximum(abs, oa(:thermo) + oa(:momentum) - oa(:all)) < 1e-13 * maximum(abs, N)
        @test maximum(abs, oa(:hthermo) + oa(:hmomentum) + oa(:vertical) - oa(:all)) < 1e-13 * maximum(abs, N)
        va = 1e-3 .* cos.(1:ma.nq)
        Aa = F3.adv_matrix(ma, Ca)
        @test length(Aa.nzval) > 0 && maximum(abs, Aa * va - F3.adv_masked!(zeros(ma.nq), ma, Ca, va)) < 1e-12 * maximum(abs, Aa * va)
        ρa = F3.hydrostatic_column(ma, Θr; ρbottom = ρr(gc.zc[1]))
        qa0 = F3.project3d(ma, (x, y, z) -> (ρ = ρa[F3.level(gc, z)]; (ρ, 10ρ, 0.0, 0.0, Θr(gc.zc[F3.level(gc, z)]))))
        qa = TS.run_frozen(ma, qa0, 20.0, 10; tab = TS.mis_tableau(:MIS54), nsub = 4, fast = :hevi)
        @test all(isfinite, qa) && abs(F3.total3d(ma, qa, :ρ) / F3.total3d(ma, qa0, :ρ) - 1) < 1e-12
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
