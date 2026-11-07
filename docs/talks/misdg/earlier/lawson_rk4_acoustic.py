"""
Lawson-RK4 for 1D linear acoustics with advection on a periodic staggered grid.

    p_t + U p_x + rho c^2 u_x = 0
    u_t + U u_x + p_x / rho   = 0

p lives at cell centers x_j+1/2, u at faces x_j.
  fast part  L: acoustic operator, 4th order staggered differences
  slow part  N: advection, 5th order upwind-biased (flux form), real space
The exponential exp(h/2 L) is applied in Fourier space as a precomputed
2x2 block per wavenumber:  E = [[C, a*S], [b*S, C]].
"""
import numpy as np

# ------------------------------------------------------------------ setup
class Model:
    def __init__(self, nx=400, Lx=1.0e6, c=350.0, U=50.0, rho=1.0):
        self.nx, self.Lx, self.c, self.U, self.rho = nx, Lx, c, U, rho
        self.dx = Lx / nx
        self.xc = (np.arange(nx) + 0.5) * self.dx      # p points
        self.xf = np.arange(nx) * self.dx              # u points
        th = 2 * np.pi * np.fft.rfftfreq(nx)           # theta = k dx
        z = np.exp(1j * th)
        # staggered 4th order derivatives (symbols w.r.t. index shifts)
        self.D1 = (9/8 * (z - 1) - 1/24 * (z**2 - 1/z)) / self.dx   # u (faces)   -> centers
        self.D2 = (9/8 * (1 - 1/z) - 1/24 * (z - 1/z**2)) / self.dx  # p (centers) -> faces
        self.a = -rho * c**2 * self.D1                  # p_t = a u
        self.b = -self.D2 / rho                         # u_t = b p
        self.w2 = -(c**2) * (self.D1 * self.D2).real    # omega^2 = c^2 s^2 >= 0
        # symbol of the 5th order upwind advection operator (for the exact solution)
        wts = np.array([2, -13, 47, 27, -3]) / 60.0     # f_{j-2..j+2}
        r = sum(w * z**m for w, m in zip(wts, range(-2, 3)))
        self.nsym = -(U / self.dx) * r * (1 - 1/z)

    # slow operator N (real space, same stencil for p and u)
    def N(self, f):
        fh = (2*np.roll(f, 2) - 13*np.roll(f, 1) + 47*f
              + 27*np.roll(f, -1) - 3*np.roll(f, -2)) / 60.0   # f_{j+1/2}
        return -(self.U / self.dx) * (fh - np.roll(fh, 1))

    def exp_blocks(self, tau):
        """Precompute exp(tau L) as (C, aS, bS) per wavenumber."""
        om = np.sqrt(self.w2)
        C = np.cos(om * tau)
        S = tau * np.sinc(om * tau / np.pi)              # sin(om tau)/om, safe at 0
        return C, self.a * S, self.b * S

# ------------------------------------------------------------------ Lawson-RK4
class LawsonRK4:
    def __init__(self, m, h):
        self.m, self.h = m, h
        self.Eh = m.exp_blocks(h / 2)                    # E = exp(h/2 L)
        self.nx = m.nx

    def E(self, p, u):
        """Apply E to the pair (p, u)  -- one rfft/irfft per field."""
        C, aS, bS = self.Eh
        ph, uh = np.fft.rfft(p), np.fft.rfft(u)
        return (np.fft.irfft(C * ph + aS * uh, self.nx),
                np.fft.irfft(bS * ph + C * uh, self.nx))

    def N(self, p, u):
        return self.m.N(p), self.m.N(u)

    def step(self, p, u):
        """One step: 4 evaluations of N and 4 applications of E (not 8-10)."""
        h = self.h
        ap, au = self.N(p, u)                                    # k1
        wp, wu = self.E(p, u)                                    # w  = E y
        y2p, y2u = self.E(p + h/2*ap, u + h/2*au)                # Y2 = E(y + h/2 a)
        bp, bu = self.N(y2p, y2u)                                # k2
        y3p, y3u = wp + h/2*bp, wu + h/2*bu                      # Y3 = w + h/2 b
        cp, cu = self.N(y3p, y3u)                                # k3
        y4p, y4u = self.E(wp + h*cp, wu + h*cu)                  # Y4 = E(w + h c)
        dp, du = self.N(y4p, y4u)                                # k4
        # y_new = E[ w + (Y2-w)/3 + h/3 (b+c) ] + h/6 d
        sp = wp + (y2p - wp)/3 + h/3*(bp + cp)
        su = wu + (y2u - wu)/3 + h/3*(bu + cu)
        np_, nu_ = self.E(sp, su)
        return np_ + h/6*dp, nu_ + h/6*du

    def run(self, p, u, nsteps, snapshots=()):
        out = {}
        for n in range(nsteps + 1):
            if n in snapshots:
                out[n] = (p.copy(), u.copy())
            if n < nsteps:
                p, u = self.step(p, u)
        return p, u, out

# ------------------------------------------------------------------ exact solution
def exact(m, p0, u0, t):
    """Exact solution of the semi-discrete system (L and N are circulant -> commute)."""
    C, aS, bS = m.exp_blocks(t)
    ph, uh = np.fft.rfft(p0), np.fft.rfft(u0)
    g = np.exp(t * m.nsym)
    return (np.fft.irfft(g * (C*ph + aS*uh), m.nx),
            np.fft.irfft(g * (bS*ph + C*uh), m.nx))
