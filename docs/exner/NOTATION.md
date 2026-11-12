# Notation of the moist Exner/p model (src/FV/moist.jl)

Math symbol, code name, and meaning. Column "clash" marks names that are used twice or are
easy to confuse (candidates for renaming).

## Constants (`MoistGas`, CGDycore values)

| math | code | value / meaning |
|---|---|---|
| R_d, R_v | `Rd`, `Rv` | 287, 461 J/(kg K) |
| c_pd, c_pv, c_pl | `Cpd`, `Cpv`, `Cpl` | 1004, 1885, 4186 J/(kg K) |
| L(T) | `latent(mg, T)` | L_0 + (c_pv − c_pl)(T − T_0), `L0V` = 2.5e6 J/kg, `T0` = 273.15 K |
| p_0 | `p0` | 1e5 Pa |
| e_s(T) | `psat(mg, T)` | saturation vapour pressure (as CGDycore) |
| κ_d | `Rd/Cpd` | dry exponent of the Exner pressure |
| g | `base.gas.g` | gravity |

## Densities and thermodynamics (per cell)

| math | code | meaning | clash |
|---|---|---|---|
| ρ | `ρ`, `iρ` | total density | |
| ρ_v = V = ρ q_v | `ρv`, `V`, `iV` | vapour density (prognostic) | |
| ρ_c = C = ρ q_c | `ρc`, `C`, `iC` | cloud water density (prognostic) | **C** is also the coefficient vector C(y) |
| ρ_d | `ρd` | ρ − ρ_v − ρ_c | |
| Q = ρ_v + ρ_c | `Q` | total water density (initial state only, not prognostic) | |
| q_v, q_c | `φq`, `φc` (fields) | specific humidities V/ρ, C/ρ (face values for transport) | φ is also φ_FB |
| R_m | `Rm` | R_d ρ_d + R_v ρ_v (**density weighted**, J/(m³ K)) | |
| c_pm, c_vm | `cpm` (`Cpml` in CGDycore), `cvm` | c_pd ρ_d + c_pv ρ_v + c_pl ρ_c, c_pm − R_m (density weighted) | |
| κ_M | `κ` | R_m/c_pm, moist exponent | κ also face coefficient κ_f |
| T | `T` | temperature, from (ρ, Θ, V, C) explicitly | |
| p | `p` | p = R_m T = p_0 (R_d Θ/p_0)^(1/(1−κ_M)) | |
| θ_ρ | — | density potential temperature with κ_M | |
| Θ = ρ θ_ρ | `Θ`, `iΘ` | prognostic thermodynamic variable | |
| π | `π`, `iπ` | (p/p_0)^κ_d, Exner pressure with the dry exponent (pvar = :pi) | iπ holds p when pvar = :p |
| Θ̂ | `Θh`, `Θ̂x`, `Θ̂z` | p/(R_d π): ∇p = c_pd Θ̂ ∇π | |
| θ_e | `theta_e` | equivalent potential temperature (poster eq. 17), diagnostic only | |

## Phase change

| math | code | meaning | clash |
|---|---|---|---|
| φ_FB(a, b) | `fischer_burmeister` | ½(a + b − √(a² + b²)), a = ρ_vs(T) − ρ_v, b = ρ_c | **a, b** also ∂π/∂ρ, ∂π/∂Θ |
| ρ_vs | — | e_s(T)/(R_v T) | |
| S_c | `Sc` | −φ_FB/τ, condensation rate (C += S_c, V −= S_c); CGDycore F_Ph = −S_c | |
| τ | `m.τ` | relaxation time | τ is also the local time in ETD stages |
| τ_fac, c_s, w_τ | `τfac`, `cs`, `wτ` | τ = τ_fac·0.85·0.92·min(Δx, Δz)/c_s (default 0.25), or min(Δx, Δz)/w_τ | |
| P | `P` in `theta_phase_factor` | Bryan–Fritsch prefactor | |
| Q_θ | `theta_phase_factor` → `mθS` | −Θ P S_c, source of Θ | named Q_θ, but it is the Θ equation |
| Q_p, Q_π | `pressure_phase_factor` → `mκS` | [R_m (L − R_v T)/c_vm − R_v T] S_c; Q_π = κ_d (π/p) Q_p | code name κS |
| micro | `micro` | :fast (S_c in every small step, default), :slow (at the slow stages), :none | |

## Pressure equation and frozen split

| math | code | meaning | clash |
|---|---|---|---|
| (a, b, c, d) | `abc[1:4, i, k]`, `exner_derivatives` | ∂π/∂(ρ, Θ, V, C) (or ∂p/… for pvar = :p), frozen from y | a, b clash with φ_FB; c with c_p |
| κ_f | `κxL`, `κxR`, `κzB`, `κzT` | a + b θ_f + c q_v,f + d q_c,f per face and neighbour cell | |
| θ_f, q_v,f, q_c,f | `θx`, `qvx`, `qcx`, `θz`, … | upwind face values | |
| F | `Fx`, `Fz` | mass fluxes at faces | F is also the forcing of the fast loop |
| y, z | `y`, `z` | stage value (frozen) and fast state | |
| C(y), S(y) | `C`, `S` | coefficient vector and slow tendency (`frozen_data!`) | C clash with ρ_c |
| F_f(C, z) | `frozen_fast!` | fast part, linear in z apart from the :fast phase source | |
| pvar | `pvar` | :pi or :p | |

## Time stepping

| math | code | meaning |
|---|---|---|
| Δt, n_sub, Δτ | `Δt`, `nsub`, `Δt/nsub` | slow step, small steps per slow step, small step |
| α, β, γ, d_i | `MISTableau` | MIS coefficients, d_i = Σ_j β_ij |
| A, b, c | `LawsonTableau` | Butcher tableau (Lawson) — clashes with (a, b, c, d) |
| c_i, a[i,j,k], P[i,j,k] | `ETDTableau` | ETD nodes, φ-coefficients, polynomial forcing |
| S̃(Y_j) | `_frozen_remainder!` | N(Y_j) − F_f(C(qⁿ), Y_j) for Lawson/ETD |

## State vector (flat FVGrid, nx × nz cells, N = nx·nz)

```
q = [ ρ | U | W | Θ | V | C | π ]
      N   N   nx(nz−1)   N   N   N   N        length 7N − nx
```

| block | index | location | offset |
|---|---|---|---|
| ρ | `iρ(g, i, k)` | cell (i, k) | 0 |
| U = ρu | `iU(g, i, k)` | x-face i = west face of cell i (between i−1 and i), periodic in x | N |
| W = ρw | `iW(g, i, k)`, k = 2…nz | z-face k (between cells k−1 and k); walls k = 1, nz+1 not stored (W = 0) | 2N |
| Θ = ρθ_ρ | `iΘ(g, i, k)` | cell | 2N + nx(nz−1) |
| V = ρ_v | `iV(m, i, k)` | cell | base.nq = 4N − nx |
| C = ρ_c | `iC(m, i, k)` | cell | base.nq + N |
| π (or p) | `iπ(m, i, k)` | cell | base.nq + 2N |

Within each block the cell index is `i + (k − 1)·nx` (i fastest, `ip(g, i) = mod1(i, nx)`).
The first four blocks are the dry `FVModel` state (`base.nq`); `moist_exner_state(m, q, Q)` appends
V, C (phase equilibrium split of Q) and π of the equation of state. π is recomputed from
(ρ, Θ, V, C) after every MIS step (`frozen_step_end!`).

Coefficient vector C(y) (`frozen_ncoef = 8N + 6Nz`, Nz = nx(nz+1)): x-faces 6 blocks of N
(θx, qvx, Θ̂x, κxL, κxR, qcx), z-faces 6 blocks of Nz (θz, qvz, Θ̂z, κzB, κzT, qcz), cells 2 blocks
of N (Q_θ/S_c, Q_π/S_c).
