"""NumPy mirror of fast_fb.jl : high-order forward-backward (leapfrog-type) fast integrator.

State (p,u,t).  Vector field split  F = F_A + F_B :
  F_A:  p' = -rho c^2 D u + s_p(t),  t' = 1     (u frozen)      -> p-kick of size alpha
  F_B:  u' = -D p / rho   + s_u(t),  t' = 0     (p, t frozen)   -> u-kick of size beta
Both flows are exactly solvable.  A symmetric composition  A(a0)B(b1)A(a1)...B(bn)A(an)
(Strang / Yoshida) gives order 2, 4, 6.  Source: s(t)=sum_k G_k t^(k-1)/(k-1)!, so that
   z(h) = exp(hL) z0 + sum_k h^k phi_k(hL) G_k .
"""
import numpy as np
from math import factorial

class Model:
    def __init__(s, nx=400, Lx=1.0e6, c=350.0, U=50.0, rho=1.0):
        s.nx, s.Lx, s.c, s.U, s.rho = nx, Lx, c, U, rho
        s.dx = Lx/nx
        s.xc = (np.arange(nx)+0.5)*s.dx

def div_u(m, u):   return (9/8*(np.roll(u,-1)-u) - 1/24*(np.roll(u,-2)-np.roll(u,1)))/m.dx
def grad_p(m, p):  return (9/8*(p-np.roll(p,1)) - 1/24*(np.roll(p,-1)-np.roll(p,2)))/m.dx
def adv(m, f):
    fh = (2*np.roll(f,2)-13*np.roll(f,1)+47*f+27*np.roll(f,-1)-3*np.roll(f,-2))/60
    return -(m.U/m.dx)*(fh-np.roll(fh,1))

# ---- composition coefficients
def stage_weights(order):
    if order == 2: return [1.0]
    if order == 4:
        g1 = 1/(2-2**(1/3)); return [g1, 1-2*g1, g1]
    if order == 6:
        w1, w2, w3 = -1.17767998417887, 0.235573213359357, 0.784513610477560
        w0 = 1-2*(w1+w2+w3); return [w3, w2, w1, w0, w1, w2, w3]
    raise ValueError
def coeffs(order):
    w = stage_weights(order); n = len(w)
    b = np.array(w); a = np.empty(n+1)
    a[0] = w[0]/2; a[n] = w[n-1]/2
    for i in range(1, n): a[i] = (w[i-1]+w[i])/2
    return a, b

def fast(m, p, u, h, nsub, order=4, G=None):
    """integrate z'=Lz+s(t) on [0,h] from z0=(p,u).  G: list of (Gp_k,Gu_k), k=1..r, or None."""
    a, b = coeffs(order); n = len(b); tau = h/nsub
    p = p.copy(); u = u.copy(); t = 0.0
    r = 0 if G is None else len(G)
    def kickA(alpha):
        nonlocal p, t
        p = p - alpha*m.rho*m.c**2*div_u(m,u)
        for k in range(1, r+1):
            p = p + G[k-1][0]*(((t+alpha)**k - t**k)/factorial(k))
        t += alpha
    def kickB(beta):
        nonlocal u
        u = u - beta*grad_p(m,p)/m.rho
        for k in range(1, r+1):
            u = u + beta*G[k-1][1]*(t**(k-1)/factorial(k-1))
    pend = a[0]*tau
    for s in range(nsub):
        if s > 0: pend += a[0]*tau
        for i in range(n):
            kickA(pend); kickB(b[i]*tau); pend = a[i+1]*tau
    kickA(pend)
    return p, u

# ---- Lawson-RK4 with fast integrator as E
def lawson_rk4(m, p, u, h, nsub, order=4):
    E = lambda x, y: fast(m, x, y, h/2, nsub, order)
    ap, au = adv(m,p), adv(m,u)
    wp, wu = E(p,u)
    y2p, y2u = E(p+h/2*ap, u+h/2*au)
    bp, bu = adv(m,y2p), adv(m,y2u)
    cp, cu = adv(m,wp+h/2*bp), adv(m,wu+h/2*bu)
    y4p, y4u = E(wp+h*cp, wu+h*cu)
    dp, du = adv(m,y4p), adv(m,y4u)
    sp = wp+(y2p-wp)/3+h/3*(bp+cp); su = wu+(y2u-wu)/3+h/3*(bu+cu)
    sp, su = E(sp, su)
    return sp+h/6*dp, su+h/6*du

# ---- ETDRK4 (Cox-Matthews): all exp, phi1..phi3 from the fast integrator via polynomial sources
def etdrk4(m, p, u, h, nsub, order=4):
    N = lambda x, y: (adv(m,x), adv(m,y))
    g1 = N(p,u)
    # a = e^{h/2 L} y + h/2 phi1(h/2 L) N(y)          (source constant, G1 = g1, duration h/2)
    ap_, au_ = fast(m,p,u,h/2,nsub,order,[g1])
    g2 = N(ap_,au_)
    bp_, bu_ = fast(m,p,u,h/2,nsub,order,[g2])
    g3 = N(bp_,bu_)
    cp_, cu_ = fast(m,ap_,au_,h/2,nsub,order,[(2*g3[0]-g1[0], 2*g3[1]-g1[1])])
    g4 = N(cp_,cu_)
    G1 = g1
    G2 = tuple((-3*g1[i]+2*(g2[i]+g3[i])-g4[i])/h for i in (0,1))
    G3 = tuple(4*(g1[i]-g2[i]-g3[i]+g4[i])/h**2 for i in (0,1))
    return fast(m,p,u,h,2*nsub,order,[G1,G2,G3])

# ---- exact references
def symbols(m):
    th = 2*np.pi*np.fft.rfftfreq(m.nx); z = np.exp(1j*th)
    D1 = (9/8*(z-1)-1/24*(z**2-1/z))/m.dx; D2 = (9/8*(1-1/z)-1/24*(z-1/z**2))/m.dx
    a = -m.rho*m.c**2*D1; b = -D2/m.rho; w2 = -(m.c**2)*(D1*D2).real
    r = sum(w*z**k for w,k in zip(np.array([2,-13,47,27,-3])/60, range(-2,3)))
    return a, b, np.sqrt(w2), -(m.U/m.dx)*r*(1-1/z)

def exact(m, p0, u0, t):
    a,b,om,ns = symbols(m); C=np.cos(om*t); S=t*np.sinc(om*t/np.pi)
    ph,uh = np.fft.rfft(p0), np.fft.rfft(u0); g=np.exp(t*ns)
    return np.fft.irfft(g*(C*ph+a*S*uh),m.nx), np.fft.irfft(g*(b*S*ph+C*uh),m.nx)

def phi_scalar(k, z):
    z = np.asarray(z, dtype=complex); out = np.empty_like(z)
    small = np.abs(z) < 3
    zz = z[small]; acc = np.zeros_like(zz); term = np.ones_like(zz)/factorial(k)
    for j in range(60):
        acc += term; term = term*zz/(j+k+1)
    out[small] = acc
    big = z[~small]; f = (np.exp(big)-1)/big
    for j in range(1, k): f = (f - 1/factorial(j))/big
    out[~small] = f if k > 0 else np.exp(big)
    return out

def exact_phi(m, k, gp, gu, h):
    """phi_k(hL) g for L alone."""
    a,b,om,_ = symbols(m); th = om*h
    fp, fm = phi_scalar(k, 1j*th), phi_scalar(k, -1j*th)
    alpha = ((fp+fm)/2).real
    beta  = ((fp-fm)/(2j*np.where(th==0,1,th))).real       # coefficient of M = hL
    ph, uh = np.fft.rfft(gp), np.fft.rfft(gu)
    return (np.fft.irfft(alpha*ph + beta*h*a*uh, m.nx), np.fft.irfft(alpha*uh + beta*h*b*ph, m.nx))
