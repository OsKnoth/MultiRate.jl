# Conversion between the tile state Q and the global state vector of FV3D.MoistModel3D (same variables, same
# pressure variable): initial states, tests, output.

"""
    ka_state(m::MoistKA, m3::FV3D.MoistModel3D, x)

Tile state Q (on the device of m, halo included) of the global state vector x of the CPU model m3 (same grid).
"""
function ka_state(m::MoistKA, m3::FV3D.MoistModel3D, x::AbstractVector)
    g, g3 = m.grid, m3.base.grid
    (g3.nx, g3.ny, g3.nz) == (g.nx, g.ny, g.nz) || error("ka_state: different grids")
    Q = zeros(g.nxh, g.nyh, g.nz + 1, NV)
    for k in 1:g.nz, J in 1:g.nyh, I in 1:g.nxh
        i, j = mod1(g.i0 + I - HALO, g.nx), mod1(g.j0 + J - HALO, g.ny)
        Q[I, J, k, VR] = x[FV3D.iρ(g3, i, j, k)]; Q[I, J, k, VU] = x[FV3D.iU(g3, i, j, k)]
        Q[I, J, k, VV] = x[FV3D.iV(g3, i, j, k)]; Q[I, J, k, VX] = x[FV3D.iΘ(g3, i, j, k)]
        Q[I, J, k, VQV] = x[FV3D.iqv(m3, i, j, k)]; Q[I, J, k, VQC] = x[FV3D.iqc(m3, i, j, k)]
        Q[I, J, k, VP] = x[FV3D.iπ(m3, i, j, k)]
        k >= 2 && (Q[I, J, k, VW] = x[FV3D.iW(g3, i, j, k)])
    end
    D = ka_zeros(m); copyto!(D, Q)
    return D
end

"""
    global_state(m, m3, Q; fill = 0.0)

Global state vector of the CPU model m3 with the interior of the tile Q (other tiles: `fill`; with MPI the sum over
the ranks, `allreduce_sum`, gives the whole state).
"""
function global_state(m::MoistKA, m3::FV3D.MoistModel3D, Q; fill::Real = 0.0)
    g, g3 = m.grid, m3.base.grid
    KA.synchronize(KA.get_backend(m))
    A = Array(Q)
    x = Base.fill(Float64(fill), m3.nq)
    for k in 1:g.nz, j in 1:g.nyl, i in 1:g.nxl
        I, J = i + HALO, j + HALO; gi, gj = g.i0 + i, g.j0 + j
        x[FV3D.iρ(g3, gi, gj, k)] = A[I, J, k, VR]; x[FV3D.iU(g3, gi, gj, k)] = A[I, J, k, VU]
        x[FV3D.iV(g3, gi, gj, k)] = A[I, J, k, VV]; x[FV3D.iΘ(g3, gi, gj, k)] = A[I, J, k, VX]
        x[FV3D.iqv(m3, gi, gj, k)] = A[I, J, k, VQV]; x[FV3D.iqc(m3, gi, gj, k)] = A[I, J, k, VQC]
        x[FV3D.iπ(m3, gi, gj, k)] = A[I, J, k, VP]
        k >= 2 && (x[FV3D.iW(g3, gi, gj, k)] = A[I, J, k, VW])
    end
    return x
end

"""Σ J ΔV q over the interior of all tiles of the cell variable v (VR, VX, VQV, VQC)."""
function ka_total(m::MoistKA, Q, v::Int)
    g = m.grid
    KA.synchronize(KA.get_backend(m))
    A = Array(view(Q, HALO+1:HALO+g.nxl, HALO+1:HALO+g.nyl, 1:g.nz, v))
    J = Array(view(g.Jc, HALO+1:HALO+g.nxl, HALO+1:HALO+g.nyl))
    dV = g.dx * g.dy * g.dz
    return allreduce_sum(m.dec, sum(A[i, j, k] * J[i, j] * dV for i in axes(A, 1), j in axes(A, 2), k in axes(A, 3)))
end

"""
    ka_project(m, f; mgθ = MoistGas())

Tile state from `f(x, y, z) = (ρ, ρu, ρv, ρw, Θ, Q)` (Θ = ρθ_ρ of the gas mgθ, Q = ρq_t) at the physical positions of
the unknowns (cells: centres; momenta: their faces, as `FV3D.project3d`): water split into vapour and condensate of
phase equilibrium (initial states only), the thermodynamic variable of the model's gas and p (π) of its equation of
state. No global state is needed (large grids, MPI).
"""
function ka_project(m::MoistKA, f; mgθ::MoistGas = MoistGas())
    g = m.grid; mg = m.mgas
    Jc = Array(g.Jc); Jx = Array(g.Jx); Jy = Array(g.Jy)
    Q = zeros(g.nxh, g.nyh, g.nz + 1, NV)
    xc(I) = (g.i0 + I - HALO - 0.5) * g.dx; yc(J) = (g.j0 + J - HALO - 0.5) * g.dy
    xf(I) = (g.i0 + I - HALO - 1) * g.dx; yf(J) = (g.j0 + J - HALO - 1) * g.dy
    zc(k) = (k - 0.5) * g.dz
    for k in 1:g.nz, J in 1:g.nyh, I in 1:g.nxh
        Z = g.H * (1 - Jc[I, J]) + zc(k) * Jc[I, J]
        ρ, _, _, _, Θ, Qt = f(xc(I), yc(J), Z)
        T, ρv, ρc, _ = moist_state(mgθ, ρ, Θ, Qt)
        X = _Θof(mg, ρ, T, ρv, ρc)
        Q[I, J, k, VR] = ρ; Q[I, J, k, VX] = X; Q[I, J, k, VQV] = ρv; Q[I, J, k, VQC] = ρc
        Q[I, J, k, VP] = ka_pvar(mg, m.pres, ρ, X, ρv, ρc)
        Q[I, J, k, VU] = f(xf(I), yc(J), g.H * (1 - Jx[I, J]) + zc(k) * Jx[I, J])[2]
        Q[I, J, k, VV] = f(xc(I), yf(J), g.H * (1 - Jy[I, J]) + zc(k) * Jy[I, J])[3]
        k >= 2 && (Q[I, J, k, VW] = f(xc(I), yc(J), g.H * (1 - Jc[I, J]) + g.zf[k] * Jc[I, J])[4])
    end
    D = ka_zeros(m); copyto!(D, Q)
    return D
end

"""Interior fields of the tile for output: w at the interior η-faces (nxl, nyl, nz-1), θ_e and q_c at the centres."""
function ka_fields(m::MoistKA, Q)
    g = m.grid; KA.synchronize(KA.get_backend(m)); A = Array(Q)
    I, J = HALO .+ (1:g.nxl), HALO .+ (1:g.nyl)
    w = [A[i, j, k, VW] / (0.5 * (A[i, j, k-1, VR] + A[i, j, k, VR])) for i in I, j in J, k in 2:g.nz]
    θe = [theta_e(m.mgas, A[i, j, k, VR], A[i, j, k, VX], A[i, j, k, VQV], A[i, j, k, VQC]) for i in I, j in J, k in 1:g.nz]
    qc = [A[i, j, k, VQC] / A[i, j, k, VR] for i in I, j in J, k in 1:g.nz]
    return (; w, θe, qc)
end
