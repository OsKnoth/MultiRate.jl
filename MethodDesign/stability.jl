# Linear stability of MIS methods on the acoustic-advection test problem of Knoth & Wensch (2014),
# eq. (10): u_t + U u_x = -c Π_x, Π_t + U Π_x = -c u_x on a C-grid (2nd-order acoustics, 3rd-order
# upwind advection), exact fast stages (`MultiRate.LinearAnalysis.kw_mode`, `max_amplification`).
# The MISDG grid (4th-order acoustics, 5th-order upwind) is available as `grid = :misdg`.

using MultiRate, Printf
const LA = MultiRate.LinearAnalysis
const TS = MultiRate.TimeStepping

_mode(grid, θ, cs, ca) = grid === :kw ? LA.kw_mode(θ; C_sound = cs, C_adv = ca) : LA.staggered_mode(θ; C_sound = cs, C_adv = ca)

"""Maximum amplification over the Fourier modes for the Courant numbers (C_sound, C_adv)."""
amplification(tab, cs, ca; grid = :kw, nmodes = 24) =
    LA.max_amplification(θ -> _mode(grid, θ, cs, ca), tab; fast = :exact, nmodes)

"""Largest C_adv ≤ camax (resolution dc) with max amplification ≤ 1 + tol for this C_sound."""
function adv_limit(tab, cs; grid = :kw, camax = 3.0, dc = 0.05, tol = 1e-10, nmodes = 24)
    ca = dc
    while ca <= camax + 1e-12
        amplification(tab, cs, ca; grid, nmodes) > 1 + tol && return ca - dc
        ca += dc
    end
    return camax
end

"""Stability penalty Σ max(0, ρ - 1)² over the grid of Courant numbers (C_sound, C_adv)."""
stability_penalty(tab; Cs = (0.0, 2.0, 4.0, 8.0, 12.0), Ca = (0.4, 0.8, 1.2, 1.6), grid = :kw, nmodes = 16) =
    sum(max(0.0, amplification(tab, cs, ca; grid, nmodes) - 1)^2 for cs in Cs, ca in Ca)

if abspath(PROGRAM_FILE) == @__FILE__
    Cs = (0.0, 2.0, 4.0, 8.0, 12.0)
    println("Advective stability limit (C_adv) per C_sound $(Cs), Knoth–Wensch C-grid, exact fast stages:")
    for name in (:WSRK3, :MIS2, :MIS3C, :MIS4, :MIS4a, :MIS54, :MIS_KW3)
        tab = TS.mis_tableau(name)
        @printf("%-8s %s\n", name, join([@sprintf("%.2f", adv_limit(tab, cs)) for cs in Cs], "  "))
    end
end
