

## 11. MIS with a frozen part of the right-hand side (flat FV)

**Idea** (Knoth & Wensch, *B-series and split-explicit methods*, github.com/OsKnoth/B-Series;
MATLAB `SplitExplicit.m`): instead of adding and subtracting a linearized fast operator
(S = N − L_n, fast Z' = L_n Z + forcing), write the right-hand side with two arguments,
N(q) = F(q, q), the first frozen at the stage values, the second evolving in the fast
integration:
Z_i' = Σ_j w_ij F(Y_j, Z_i) + Σ_j γ_ij/(d_iΔt)(Y_{j+1} − qⁿ), w_ij = β_ij/d_i, over d_iΔt from
qⁿ + Σ_j α_ij(Y_{j+1} − qⁿ) (`TimeStepping.run_frozen`, `frozen_mis_step!`). For an additive
F(y, z) = S(y) + G(z) this is the usual MIS step.

**FV split** (`src/FV/frozen.jl`, flat grids only):
- fast, evolving z: ρ_t = −∂xU − ∂zW, Θ_t = −∂x(θx U) − ∂z(θz W), U_t = −∂x p'(Θ),
  W_t = −∂z p'(Θ) − g ρ', with the full nonlinear p(Θ), and the sponge;
- frozen coefficients C(y): the upwinded (3rd-order) face values θx, θz of the full operator,
  upwind direction from y; linear in C, so Σ_j w_j F(Y_j, Z) = F_f(Σ_j w_j C(Y_j), Z) + Σ_j w_j S(Y_j);
- slow, frozen y: momentum advection.
F(q, q) = N(q) to rounding (4e-16 relative). Nothing is linearized and there is no reference state.

**Fast loops:** `:verlet` (thermo/momentum, as `SplitExplicit.m`), `:fb`, `:rk4`, compositions
`:verlet_s4` …, and `:hevi`: Strang, horizontal Verlet half steps around a linearly implicit
vertical Crank–Nicolson step z ← z + (I − dt/2 J_v)⁻¹ dt G_v(z), where G_v keeps the nonlinear
p(Θ) and J_v uses a = γp/Θ from the start of the stage and the frozen θz (column solver of `FVFast`).

**Orders** (toy problem: oscillator with coefficient c(y) = 1 + 0.5 sin y₁ frozen, nonlinear slow
part; exact fast stages):

| | additive F = S(y) + G(z) | frozen coefficient in the fast part |
|---|---|---|
| WSRK3, MIS2, MIS3C | 2 | 2 |
| MIS4, MIS4a | 3 | **2** |
| MIS54 | 4 | **2** |
| MIS_KW3 (Knoth–Wolke) | 3 | 3 |

With a non-additive F the mixed trees of the B-series (F_(1,1)FF, …) give further order
conditions; MIS4/4a/54 do not satisfy them. Methods that do (time-dependent weights a_ij^p of
the B-series note) are to be constructed with the Fortran program.

**Gravity wave, 300 km** (`examples/frozen_gravity_wave.jl`, wave error [K], nsub at the substep limit):

| Δt [s] (UΔt/Δx) | 67 (0.53) | 100 (0.8) | 130 (1.04) | 167 (1.33) | 200 (1.6) |
|---|---|---|---|---|---|
| MIS54 frozen `hevi` | 1.1e-4 | 9.6e-5 | 1.6e-4 | 1.2e-4 | **1.6e-4** |
| MIS54 frozen `verlet` | 7.4e-5 | 9.2e-5 | 2.1e-4 | 7.1e-5 | 2.1e-4 |
| MIS54 linearized `hevi` | 1.1e-4 | 9.6e-5 | 2.0e-4 | 2.4e-4 | 3.1e-4 |
| MIS4 frozen `hevi` | 2.3e-4 | 3.1e-4 | 3.1e-4 | 3.3e-4 | 1.6e-3 |
| MIS4 linearized `hevi` | 2.3e-4 | 3.1e-4 | 5.9e-4 | 2.7e-2 | unstable |
| WSRK3 frozen `hevi` | 2.5e-4 | 4.2e-4 | 5.1e-4 | 6.6e-4 | 2.9e-3 |
| WSRK3 linearized `hevi` | 2.5e-4 | 4.2e-4 | 5.2e-4 | 5.6e-3 | unstable |
| MIS4a frozen `hevi` | 0.15 | 2.0e-4 | 1.8e-4 | 3.2e-4 | 4.7e-4 |
| MIS_KW3 (both) | unstable | unstable | unstable | 0.2 | unstable |

At UΔt/Δx ≲ 1 frozen and linearized agree (the fast-loop error dominates). At larger steps the frozen
split stays stable and accurate where the linearized one degrades or blows up (MIS4, WSRK3), and
MIS54 halves its error at 200 s. A plausible reason: the linearized split puts the Θ transport by
the slow velocity, −∂x(U(θ − θ̄_n)) and the advection of the pressure perturbation, into S and
couples them explicitly to the acoustics; in the frozen split all Θ and mass fluxes are in the
fast loop with the full U. MIS4a fails at single step sizes (67 s `hevi`, 130 s `verlet`) in both splits;
MIS_KW3 is stable only at small steps despite its order 3.

![](images/frozen_gravity_wave.png)
![](images/frozen_gravity_wave_sections.png)

**Next:** MIS coefficients for the frozen form (order 3/4 with the mixed trees, time-dependent
weights a_ij^p), the frozen split for DG and with cut cells, a frozen θ_z in a fully implicit
vertical step (Newton for p(Θ)).
