Compile the talk:  pdflatex talk_acoustic_advection_multirate.tex   (twice; needs the data_*.csv files in the same folder)
Julia (1.12, standard library only):
  julia test_fast.jl          accuracy of exp(hL) and phi_k(hL) from the fast integrator (table on the slide "Accuracy of the fast integrator")
  julia run_experiments.jl    rewrites the data_*.csv files used by the slides (about 2 min):
                              Lawson-RK4 / ETDRK4 with the high-order fast integrator; the split-explicit RK3 of WRF
                              (forward-backward, with and without pressure extrapolation) and its Fourier stability analysis;
                              MIS methods (RK3, MIS-KW3, MIS54) on the 1D test (data_conv_mis.csv) and the stability
                              regions of the Knoth-Wensch linear acoustics test (data_kw_stab.csv)
fast_mirror.py is the NumPy version of the same algorithm (without split-explicit RK3 and MIS); it reproduces the test_fast.jl table.
MIS coefficients: MIS-KW3 from the Knoth-Wolke RK3 tableau (as printed in Sexton & Reynolds, arXiv:1808.03718),
MIS54 from Bauer & Knoth, J. Comput. Appl. Math. 387 (2021), Table 2.
