# Writes docs/mis_polynomial/mis_tableaus.tex: one page per MIS tableau of `TimeStepping.mis_tableau`
# (coefficients, stage times, orders, stability region on the Knoth–Wensch FV test).
# Needs the regions of examples/mis_stability_fv.jl (docs/mis_polynomial/stability_fv/regions.bin).
#
#     julia --project=examples examples/mis_tableaus_tex.jl

using MultiRate, Printf, Serialization
const M = MultiRate.MIS
const TS = MultiRate.TimeStepping

dir = joinpath(@__DIR__, "..", "docs", "mis_polynomial")
reg = deserialize(joinpath(dir, "stability_fv", "regions.bin"))

src = Dict(
    :WSRK3 => "Wicker \\& Skamarock (2002), RK3 split-explicit scheme: \$\\beta=\\mathrm{diag}(1/3,1/2,1)\$, \$\\alpha=\\gamma=0\$.",
    :MIS2 => "Knoth \\& Wensch (2014).",
    :MIS3C => "Knoth \\& Wensch (2014).",
    :MIS4 => "Knoth \\& Wensch (2014).",
    :MIS4a => "Knoth \\& Wensch (2014), with the authors' corrected \$\\beta_{41}\$ (values as in ClimaTimeSteppers.jl).",
    :MIS_KW3 => "Knoth--Wolke RK3 (\$c=0,1/3,3/4\$) in MIS form with recursive stages (\$\\alpha_{i,i-1}=1\$, \$\\gamma=0\$).",
    :MIS54 => "Bauer \\& Knoth, J.~Comput.\\ Appl.\\ Math.\\ 387 (2021), Table~2.",
    :MISF53 => "This project (\\texttt{MethodDesign/}): MIS54 projected (minimum norm) onto all bicoloured order conditions \$\\le3\$.",
    :MIS_FV42 => "This project (\\texttt{MethodDesign/search\\_app.jl}): from MIS4, stable on the column Fourier blocks of the frozen FV split.",
    :MIS_FV53 => "This project (\\texttt{MethodDesign/search\\_app.jl}): from MISF53, stable on the column Fourier blocks of the frozen FV split.",
    :MIS_DG53 => "This project (\\texttt{MethodDesign/search\\_app.jl}): from MISF53, stable on the column Fourier blocks of the frozen DG split.",
    :TVDMISB => "Knoth \\& Wensch (2014), as in Breeze.jl. The published coefficients have about 6 digits: the residual of \\(\\bullet[w]\\) is \\(2\\cdot10^{-6}\\).",
    :SSPMIS43 => "Breeze.jl (branch \\texttt{glw/mis-tableau-search}): coupled order 2, slow order 3, SSP slow method (SSP coefficient 1.79), CMA-ES search for the cost per simulated time with Breeze's own fast integrator.",
    :ConvexMIS52 => "Breeze.jl (branch \\texttt{glw/mis-tableau-search}): coupled order 2, convex stage start states, \$\\beta\\ge0\$, Shu--Osher coefficient 1.5; constrained search for the stable step per cost.",
    :ConvexMIS52r20 => "Breeze.jl (branch \\texttt{glw/mis-tableau-search}): as ConvexMIS52 with Shu--Osher coefficient 2.0.",
    :ConvexMIS42 => "Breeze.jl (branch \\texttt{glw/mis-tableau-search}): 4 stages, as ConvexMIS52 (Shu--Osher coefficient 1.5).")

fmt(x) = abs(x) < 5e-16 ? "0" : @sprintf("%.10f", x)
function matrix(A; lower::Bool)
    s = size(A, 1)
    rows = [join((j < i || (!lower && j == i) ? fmt(A[i, j]) : "" for j in 1:s), " & ") for i in 1:s]
    return "\\left(\\begin{array}{" * "r"^s * "}" * join(rows, "\\\\\n") * "\\end{array}\\right)"
end
"""Largest C_adv such that every C_adv' ≤ C_adv is stable, at each C_sound of the grid."""
function limit(ρ, i)
    j = findfirst(>(1 + 1e-8), ρ[i, :])
    return j === nothing ? reg.Ca[end] : j == 1 ? 0.0 : reg.Ca[j-1]
end
"""Largest C_sound such that every C_sound' ≤ C_sound is stable, at C_adv = 0."""
function slimit(ρ)
    i = findfirst(>(1 + 1e-8), ρ[:, 1])
    return i === nothing ? reg.Cs[end] : i == 1 ? 0.0 : reg.Cs[i-1]
end

io = IOBuffer()
print(io, raw"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref}
\title{MIS tableaus of \texttt{MultiRate.TimeStepping.mis\_tableau}:\\coefficients and stability on the acoustic-advection test}
\author{MultiRate notes}
\date{\today}
\begin{document}
\sloppy
\maketitle
\noindent\textbf{Convention} (Knoth \& Wensch 2014, ClimaTimeSteppers.jl, Breeze.jl). For $y'=f_{\rm fast}(y)+f_{\rm slow}(y)$,
$Y_0=y^n$, $\Delta Y_j=Y_j-y^n$, stage $i=1,\dots,s$:
\begin{align*}
V(0)&=y^n+\sum_{j<i}\alpha_{ij}\Delta Y_j,\\
V'(\tau)&=f_{\rm fast}(V)+\frac1{d_i}\Big[\sum_{j<i}\gamma_{ij}\frac{\Delta Y_j}{\Delta t}+\sum_{j\le i}\beta_{ij}f_{\rm slow}(Y_{j-1})\Big],\qquad
0\le\tau\le d_i\Delta t,
\end{align*}
$d_i=\sum_j\beta_{ij}$, $Y_i=V(d_i\Delta t)$, $y^{n+1}=Y_s$. $\Sigma d=\sum_id_i$ is the fast integration time per step (in $\Delta t$).
In the notation of \texttt{MISPolynomial.tex}: constant coupling $p_{i+1,j}=\beta_{ij}$, $g_{i+1,j+1}=\gamma_{ij}$,
$\alpha_{i+1,j+1}=\alpha_{ij}$. Stage times $c_i$: the time reached by stage $i$.

\noindent\textbf{Orders} from the bicoloured B-series (\texttt{MultiRate.MIS.order}, residuals $\le10^{-5}$) for the problem
classes: G general $F(y,z)$ (the frozen form), B affine in $z$, A additive $f(y)+g(z)$, L $f(y)+Lz$.

\noindent\textbf{Stability} (Knoth \& Wensch 2014, eq.~(10)): $u_t+Uu_x=-cp_x$, $p_t+Up_x=-cu_x$, periodic C-grid, centred
second-order acoustics (fast), third-order upwind advection (slow), exact fast part; stable region (blue) of
$\max_\theta\rho\le1+10^{-8}$ over $C_{\rm sound}=c\Delta t/\Delta x$ and $C_{\rm adv}=U\Delta t/\Delta x$
(\texttt{examples/mis\_stability\_fv.jl}); orange: the base Runge--Kutta method; dashed: $U/c=1/6$. The computation is
described below.

\subsection*{How the stability plots are computed}
\begin{enumerate}
\item \emph{Symbols.} With $p$ in the cells, $u$ on the faces and the Fourier ansatz $p_k=\hat p\,e^{\mathrm ik\theta}$,
  $u_{k+1/2}=\hat u\,e^{\mathrm i(k+1/2)\theta}$, $\theta\in[0,\pi]$, the semi-discretisation becomes
  $(\hat p,\hat u)'=(C_{\rm adv}S_1(\theta)+C_{\rm sound}L_1(\theta))(\hat p,\hat u)$ per time step $\Delta t=1$, with the
  centred staggered acoustics
  \[
   L_1(\theta)=\begin{pmatrix}0&-(e^{\mathrm i\theta}-1)\\-(1-e^{-\mathrm i\theta})&0\end{pmatrix},\qquad
   \text{eigenvalues }\pm2\mathrm i\sin(\theta/2),
  \]
  and the third-order upwind flux-form advection of both variables,
  $S_1(\theta)=\nu(\theta)I$, $\nu(\theta)=-(2e^{\mathrm i\theta}+3-6e^{-\mathrm i\theta}+e^{-2\mathrm i\theta})/6$
  (\texttt{MultiRate.MIS.fv\_symbols}). The acoustic part is the fast part $L$, the advection the slow part $S$.
\item \emph{MIS step matrix.} Stage $i$ solves $V'=d_iLV+r_i$ on $[0,1]$, $V(0)=I+\sum_j\alpha_{ij}(Y_j-I)$, with the
  constant forcing $r_i=\sum_j\gamma_{ij}(Y_j-I)+\sum_j\beta_{ij}SY_j$; the inner problem is solved exactly (matrix
  exponential of the system augmented by the forcing), i.e.\ the small steps are not part of the analysis. The stage
  recursion gives the $2\times2$ step matrix $R(\theta)$ (\texttt{MultiRate.MIS.amplification}).
\item \emph{Criterion.} A point $(C_{\rm sound},C_{\rm adv})$ is stable if $\max_\theta\rho(R(\theta))\le1+10^{-8}$, with
  40 equidistant $\theta\in[0,\pi]$ (\texttt{max\_amplification}). Grid: $\Delta C_{\rm sound}=0.125$ on $[0,24]$,
  $\Delta C_{\rm adv}=0.05$ on $[0,4]$; blue are the stable grid points, the black line is the level $\rho=1+10^{-8}$.
\item \emph{Base Runge--Kutta method} (orange). The explicit method $(A,b)$ that the MIS method reduces to if $F$ does not
  depend on its second argument (\texttt{base\_method}), applied to the whole operator without fast integration:
  $R=I+b^TZ(I-AZ)^{-1}\mathbf 1$, $Z=C_{\rm adv}S_1+C_{\rm sound}L_1$ (\texttt{rk\_amplification}). At $C_{\rm sound}=0$
  both methods coincide, so the regions agree on the $C_{\rm adv}$ axis; the distance between the orange and the blue
  region in $C_{\rm sound}$ is the gain of the multirate integration.
\item \emph{Line $U/c=1/6$} (dashed), $C_{\rm adv}=C_{\rm sound}/6$: a typical upper bound of the Mach number in the
  atmosphere ($U\approx55$\,m/s, $c\approx340$\,m/s). On this line both Courant numbers grow with $\Delta t$; the value
  ``$U/c=1/6$'' given for each method is the largest $C_{\rm sound}$ (step $1/16$) up to which all points of the line
  are stable, i.e.\ the largest stable time step $\Delta t=C_{\rm sound}\Delta x/c$ at this Mach number.
\item \emph{Tables.} For given $C_{\rm sound}$ the largest $C_{\rm adv}$ such that all smaller $C_{\rm adv}$ are stable
  (from the grid).
\end{enumerate}

\begin{center}\includegraphics[width=\textwidth]{stability_fv/all.png}\end{center}
\tableofcontents
""")
order_names = [:WSRK3, :MIS2, :MIS3C, :MIS4, :MIS4a, :MIS_KW3, :MIS54, :MISF53, :MIS_FV42, :MIS_FV53, :MIS_DG53,
               :TVDMISB, :SSPMIS43, :ConvexMIS52, :ConvexMIS52r20, :ConvexMIS42]
for n in filter(in(keys(reg.res)), order_names)
    t = TS.mis_tableau(n); m = M.PolyMIS(t); s = length(t.d)
    o = [M.order(m; class = c, tol = 1e-5) for c in (:G, :B, :A, :L)]
    c = M.stage_times(m)[2:end]
    ρ = reg.res[n]
    css = (0, 2, 4, 8, 12, 16, 20, 24); lims = [limit(ρ, findmin(abs.(reg.Cs .- x))[2]) for x in css]
    print(io, "\\clearpage\n\\section{$(replace(string(n), "_" => "\\_"))}\n", src[n], "\n\n")
    print(io, "\\noindent \$s=$s\$, \$\\Sigma d=", @sprintf("%.4f", sum(t.d)), "\$, orders G/B/A/L: ", join(o, " / "),
          ".\\\\ Stage times \$c=(", join((@sprintf("%.4f", x) for x in c), ", "), ")\$.\n\n")
    a, g = s == 1 ? (zeros(0, 0), zeros(0, 0)) : (t.α[2:s, 1:s-1], t.γ[2:s, 1:s-1])
    print(io, "{\\footnotesize\\[(\\alpha_{ij})_{i\\ge2}=", matrix(a; lower = false),
          "\\]\n\\[\\beta=", matrix(t.β; lower = false),
          "\\]\n\\[(\\gamma_{ij})_{i\\ge2}=", matrix(g; lower = false), "\\]}\n\n")
    print(io, "\\begin{center}\\includegraphics[width=0.8\\textwidth]{stability_fv/$n.png}\n\n\\small\\begin{tabular}{l",
          "c"^length(css), "}\\toprule \$C_{\\rm sound}\$ & ", join(css, " & "), "\\\\\\midrule largest \$C_{\\rm adv}\$ & ",
          join((@sprintf("%.2f", x) for x in lims), " & "), "\\\\\\bottomrule\\end{tabular}\\end{center}\n")
    ρrk = reg.resrk[n]
    rel = reg.line[n] >= reg.Cs[end] - 1e-12 ? "\\ge" : "="                 # stable to the end of the range
    print(io, "\\noindent On the line \$U/c=1/6\$ stable up to \$C_{\\rm sound}", rel, @sprintf("%.2f", reg.line[n]),
          "\$ (\$C_{\\rm adv}", rel, @sprintf("%.2f", reg.line[n] / 6), "\$). Base Runge--Kutta method, all explicit: ",
          "\$C_{\\rm adv}\\le", @sprintf("%.2f", limit(ρrk, 1)), "\$ at \$C_{\\rm sound}=0\$, \$C_{\\rm sound}\\le",
          @sprintf("%.3f", slimit(ρrk)), "\$ at \$C_{\\rm adv}=0\$.\n")
end
print(io, "\\end{document}\n")
write(joinpath(dir, "mis_tableaus.tex"), String(take!(io)))
println("wrote ", joinpath(dir, "mis_tableaus.tex"))
