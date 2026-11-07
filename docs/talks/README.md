# Talks and notes

## `misdg/` — linear acoustics with advection and multirate time integration

Copied from the MISDG project (`MISDG/Presentation`, October 2026). Only sources, data and the
compiled PDFs are kept; LaTeX build files are not.

**`misdg/acoustic_advection_talk/`** (current version, 36 slides,
[PDF](misdg/acoustic_advection_talk/talk_acoustic_advection_multirate.pdf)):
split-explicit methods as exponential (ETD) methods; Lawson, averaging and resonance-based
methods; staggered grids with high-order advection; WRF's split-explicit RK3 and its Fourier
stability analysis; MIS methods (RK3, MIS-KW3, MIS54) on the 1D test and the Knoth–Wensch
stability regions.

- Compile: `pdflatex talk_acoustic_advection_multirate.tex` twice in that folder (pgfplots reads
  the `data_*.csv` files there). The author on the title page is still the placeholder "Your Name".
- `run_experiments.jl` rewrites the `data_*.csv` files (Julia standard library only, about 2 min);
  `test_fast.jl` gives the table "Accuracy of the fast integrator"; `fast_fb.jl` holds the
  algorithms; `fast_mirror.py` is a NumPy version of the fast integrator. See its `README.txt`.

**`misdg/earlier/`**: earlier versions of the same talk (`LinearAcoustic.tex`,
`talk_acoustic_advection_multirate_2.tex`, with `fig_*.pdf` from `make_figures.py` and
`lawson_rk4_acoustic.py`), and `resonance.tex`, a note on a second-order uniformly accurate
resonance integrator (after Baumstark, Faou & Schratz 2018) with Krylov evaluation of the φ-functions.

### What is in MultiRate.jl

| talk | MultiRate.jl |
|---|---|
| MIS-KW3, MIS54 tableaus | `TimeStepping.mis_tableau(:MIS_KW3)`, `(:MIS54)`; order check `mis_observed_order` |
| Knoth–Wensch stability regions (`data_kw_stab.csv`) | `examples/linear_analysis/mis_stability.jl`, part 1 (`LinearAnalysis.kw_mode`) |
| staggered 4th-order acoustics, 5th-order upwind advection | `LinearAnalysis.staggered_mode`, part 2 |
| 1D pressure pulse, MIS convergence (`data_conv_mis.csv`) | `LinearAnalysis.pulse_error`, part 3 |
| WRF split-explicit RK3 with divergence damping | WSRK3 with `fast = :wrf` |
| high-order (Yoshida) fast integrator, Lawson-RK4, ETDRK4 | not yet (plan phases 5 and 6) |

The results obtained with MultiRate.jl, and how they compare with the talk, are in
[../FINDINGS.md](../FINDINGS.md) (§4, §5).
