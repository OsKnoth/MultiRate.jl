import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lawson_rk4_acoustic import *

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
m = Model()                                   # L=1000 km, c=350 m/s, U=50 m/s, nx=400
T = 3.5 * m.Lx / m.c                          # (c+U)T = 4 L, (c-U)T = 3 L  -> pulses return
x0, w = 0.25 * m.Lx, m.Lx / 16
p0 = np.exp(-((m.xc - x0) / w) ** 2); u0 = np.zeros(m.nx)

n = 200; h = T / n                            # c h/dx = 7, U h/dx = 1
p, u, snap = LawsonRK4(m, h).run(p0, u0, n, snapshots=(0, n // 2, n))
labels = [("$t=0$", 0), (r"$t=T/2$", n // 2), (r"$t=T$", n)]

fig, axs = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
for ax, (lab, k) in zip(axs, labels):
    ax.plot(m.xc / 1e3, p0, color="0.6", ls="--", lw=1.2, label="initial")
    ax.plot(m.xc / 1e3, snap[k][0], color="#1f5fa8", lw=2, label="Lawson-RK4")
    ax.set_title(lab); ax.set_xlabel("x [km]"); ax.set_xlim(0, m.Lx / 1e3)
axs[0].set_ylabel("pressure perturbation $p$ [rel.]"); axs[0].legend(frameon=False, loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig("fig_pressure_1d.pdf"); plt.close(fig)

# convergence against the exact semi-discrete solution
pe, _ = exact(m, p0, u0, T)
ns = [200, 400, 800, 1600, 3200]; errs = []
for nn in ns:
    pp, _, _ = LawsonRK4(m, T / nn).run(p0, u0, nn); errs.append(np.abs(pp - pe).max())
hs = np.array([T / nn for nn in ns]); errs = np.array(errs)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.loglog(hs, errs, "o-", color="#1f5fa8", lw=2, label="Lawson-RK4")
ax.loglog(hs, errs[0] * (hs / hs[0]) ** 4, "k--", lw=1, label=r"$h^4$")
ax.set_xlabel("time step $h$ [s]"); ax.set_ylabel(r"max-error in $p$ at $t=T$"); ax.legend(frameon=False)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout(); fig.savefig("fig_convergence_1d.pdf"); plt.close(fig)
print("T =", T, "n =", n, "h =", h, "errs =", errs)
print("max p at T/2 (should be ~0.5):", snap[n // 2][0].max(), " max|p(T)-p0|:", np.abs(snap[n][0] - p0).max())
