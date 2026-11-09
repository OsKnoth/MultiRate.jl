# Cut-cell finite-volume model operators. State variables use physical cell/face momenta;
# geometric apertures enter conservative fluxes and inactive faces are constrained to zero.

mutable struct CutCellFVFast <: AbstractFastOperator
    grid::CutCellFVGrid
    gas::Gas
    nq::Int
    a::Matrix{Float64}
    b::Matrix{Float64}
    θx::Matrix{Float64}
    θz::Matrix{Float64}
    fb::Vector{Float64}
    Mdiag::Vector{Float64}
    work::Vector{Vector{Float64}}
    lucache::Dict{Float64,Any}
    cmax::Float64
    colbuf::Vector{Float64}
end

function CutCellFVFast(m::FVModel{<:CutCellFVGrid})
    g, nq = m.grid, m.nq
    return CutCellFVFast(g, m.gas, nq, zeros(g.nx,g.nz), zeros(g.nx,g.nz),
                         zeros(g.nx,g.nz), zeros(g.nx,g.nz+1), zeros(nq), ones(nq),
                         [zeros(nq) for _ in 1:7], Dict{Float64,Any}(), 0.0, zeros(g.nz+1))
end

function cutcell_fast_operator(m::FVModel{<:CutCellFVGrid}, q; pfac=m.gas.γ, fast=:hevi,
                               reuse=true, kwargs...)
    fast == :rosenbrock_amf && return cutcell_amf_operator(m, q; pfac, reuse, kwargs...)
    s = reuse && m.fastref[] isa CutCellFVFast ? m.fastref[]::CutCellFVFast : CutCellFVFast(m)
    refreeze_cutcell!(s, m, q, pfac)
    reuse && (m.fastref[] = s)
    return s
end

function refreeze_cutcell!(s::CutCellFVFast, m::FVModel{<:CutCellFVGrid}, q, pfac)
    g = s.grid
    s.cmax = 0.0
    @inbounds for k in 1:g.nz, i in 1:g.nx
        ρ = max(q[iρ(g,i,k)], eps(Float64))
        Θ = max(q[iΘ(g,i,k)], eps(Float64))
        p = pressure(m.gas, Θ)
        s.a[i,k] = pfac * p / Θ
        s.b[i,k] = (1-pfac) * p
        if g.Vol[i,k] > 0
            s.cmax = max(s.cmax, sqrt(m.gas.γ*p/ρ))
        end
    end
    @inbounds for k in 1:g.nz, i in 1:g.nx
        il, ir = ip(g,i-1), ip(g,i+1)
        θc = q[iΘ(g,i,k)] / max(q[iρ(g,i,k)],eps(Float64))
        θl = g.Vol[il,k] > 0 ? q[iΘ(g,il,k)]/q[iρ(g,il,k)] : θc
        θr = g.Vol[i,k] > 0 ? θc : θl
        s.θx[i,k] = g.FU[i,k] == 0 ? θc : (θl+θr)/2
        if k == 1
            s.θz[i,1] = θc
        end
        if k < g.nz
            θu = g.Vol[i,k+1] > 0 ? q[iΘ(g,i,k+1)]/q[iρ(g,i,k+1)] : θc
            s.θz[i,k+1] = g.FW[i,k+1] == 0 ? θc : (θc+θu)/2
        end
        k == g.nz && (s.θz[i,g.nz+1] = θc)
    end
    fill!(s.fb, 0.0)
    @inbounds for k in 1:g.nz, i in 1:g.nx
        il, ir = ip(g,i-1), ip(g,i)
        # Constant part of p = a Θ + b and hydrostatic reference contribution.
        s.fb[iU(g,i,k)] = -(s.b[ir,k] - s.b[il,k])/g.dx
        if 2 <= k <= g.nz && g.FW[i,k] > 0
            s.fb[iW(g,i,k)] = -(s.b[i,k]-s.b[i,k-1])/g.dz +
                (m.pbal[i,k]-m.pbal[i,k-1])/g.dz +
                m.gas.g*0.5*(m.ρbal[i,k]+m.ρbal[i,k-1])
        end
    end
    for (c,F) in s.lucache
        factor_cutcell_vertical!(F,s,c)
    end
    return s
end

@inline cutcell_theta(g, s, q, i, k) =
    g.Vol[i,k] > 0 ? q[iΘ(g,i,k)]/max(q[iρ(g,i,k)],eps(Float64)) : 0.0

function cutcell_fast!(out, s::CutCellFVFast, q, f, horizontal::Bool, vertical::Bool,
                       rows::NTuple{4,Bool}, frows::NTuple{4,Bool})
    g=s.grid; nx=g.nx; nz=g.nz; grav=s.gas.g
    rρ,rU,rW,rΘ=rows
    fill!(out,0.0)
    @inbounds for k in 1:nz, i in 1:nx
        V=g.Vol[i,k]
        V > 0 || continue
        if horizontal
            il,ir=ip(g,i-1),i
            Ul=q[iU(g,i,k)]; Ur=q[iU(g,i+1,k)]
            rρ && (out[iρ(g,i,k)] -= (g.FU[i+1,k]*Ur-g.FU[i,k]*Ul)/V)
            if rΘ
                θl=cutcell_theta(g,s,q,il,k); θr=cutcell_theta(g,s,q,i,k)
                out[iΘ(g,i,k)] -= (g.FU[i+1,k]*s.θx[ip(g,i+1),k]*Ur-
                                    g.FU[i,k]*s.θx[i,k]*Ul)/V
            end
            if rU && g.FU[i,k] > 0
                pL=s.a[il,k]*q[iΘ(g,il,k)]; pR=s.a[i,k]*q[iΘ(g,i,k)]
                out[iU(g,i,k)] -= (pR-pL)/g.dx
            end
        end
        if vertical
            Wb=Wface(g,q,i,k); Wt=Wface(g,q,i,k+1)
            rρ && (out[iρ(g,i,k)] -= (g.FW[i,k+1]*Wt-g.FW[i,k]*Wb)/V)
            if rΘ
                θb=s.θz[i,k]; θt=s.θz[i,k+1]
                out[iΘ(g,i,k)] -= (g.FW[i,k+1]*θt*Wt-g.FW[i,k]*θb*Wb)/V
            end
        end
    end
    if vertical && rW
        @inbounds for k in 2:nz, i in 1:nx
            g.FW[i,k] > 0 || continue
            out[iW(g,i,k)] = -(s.a[i,k]*q[iΘ(g,i,k)]-s.a[i,k-1]*q[iΘ(g,i,k-1)])/g.dz -
                grav*0.5*((q[iρ(g,i,k-1)]-mrefρ(s,i,k-1))+(q[iρ(g,i,k)]-mrefρ(s,i,k)))
        end
    end
    fρ,fU,fW,fΘ=frows
    @inbounds for k in 1:nz, i in 1:nx
        fρ && g.Vol[i,k]>0 && (out[iρ(g,i,k)] += f[iρ(g,i,k)])
        fΘ && g.Vol[i,k]>0 && (out[iΘ(g,i,k)] += f[iΘ(g,i,k)])
        fU && g.FU[i,k]>0 && (out[iU(g,i,k)] += f[iU(g,i,k)])
        fW && 1<k<=nz && g.FW[i,k]>0 && (out[iW(g,i,k)] += f[iW(g,i,k)])
    end
    return out
end

@inline mrefρ(s::CutCellFVFast, i, k) = 0.0

@inline cut_upwind(v, left, right) = v >= 0 ? left : right

# ---------------------------------------------------------------------------------------------
# Third-order cut-cell advection after MATLAB `AdvectionCutThird`, `ShiftFaceCellCut` and
# `ShiftCellFaceCut`. Mixing ratios c/ρ are reconstructed upwind with the volume-weighted
# coefficients k1, k2 (1/3 and 1/6 on a uniform grid). The MATLAB ratio form (k1 r + k2)(cC-cL)
# with r = (cR-cC)/(cC-cL) is written as k1 (cR-cC) + k2 (cC-cL). Neighbours smaller than the
# upwind cell are blended towards cC, and the correction is switched off when the upwind cell's
# opposite face is closed. Walls act as ghosts equal to the upwind cell; the bottom wall
# aperture counts as closed (MATLAB sets FW(:,1) = 0).
# ---------------------------------------------------------------------------------------------

@inline cut_k1(hL, hC, hR) = (hC / (hC + hR + eps())) * ((hL + hC) / (hL + hC + hR + eps()))
@inline cut_k2(hL, hC, hR) = (hC / (hL + hC + eps())) * (hR / (hL + hC + hR + eps()))
@inline cut_fw(g::CutCellFVGrid, i, k) = k == 1 ? 0.0 : g.FW[i,k]

"""Face value reconstructed in upwind cell C from upstream U and downstream D (mixing ratios,
volumes) with open fraction `open` of C's upstream face."""
@inline function cut_face3(cU, cC, cD, VU, VC, VD, open, keepdown::Bool = false)
    VU < VC && (cU = ((VC - VU) * cC + VU * cU) / VC)
    VD < VC && (cD = ((VC - VD) * cC + VD * cD) / VC)
    keepdown && return cC + cut_k1(VU, VC, VD) * (cD - cC) + open * cut_k2(VU, VC, VD) * (cC - cU)
    return cC + open * (cut_k1(VU, VC, VD) * (cD - cC) + cut_k2(VU, VC, VD) * (cC - cU))
end

"""Advective tendency -(1/V) ∮ (ρv·n) (c/ρ) of the cell quantity `c` (nx, nz) into `fc`,
using the face momenta in `q` and the third-order upwind face values."""
function cut_advect3!(fc, c, q, g::CutCellFVGrid, bc = nothing, bu = nothing, bw = nothing; keepdown::Bool = false)
    nx, nz = g.nx, g.nz
    fill!(fc, 0.0)
    # Optional background mixing ratio: reconstruct c/ρ - bc in the cells and add the exact
    # background bu (x-faces) / bw (z-faces) at the face, so a background profile is not smeared.
    r(i, k) = c[i,k] / (q[iρ(g,i,k)] + eps()) - (bc === nothing ? 0.0 : bc[i,k])
    fbu(f, k) = bu === nothing ? 0.0 : bu[f,k]
    fbw(i, k) = bw === nothing ? 0.0 : bw[i,k]
    @inbounds for k in 1:nz, f in 1:nx            # x-face f between cells f-1 and f
        A = g.FU[f,k]; A > 0 || continue
        U = q[iU(g,f,k)]
        il, ir = ip(g,f-1), f
        if U >= 0       # upwind il: upstream il-1, downstream ir; upstream face of il is f-1
            iu = ip(g,il-1)
            cf = cut_face3(r(iu,k), r(il,k), r(ir,k), g.Vol[iu,k], g.Vol[il,k], g.Vol[ir,k],
                           g.FU[ip(g,f-1),k] / (g.FU[ip(g,f-1),k] + eps()), keepdown)
        else            # upwind ir: upstream ir+1, downstream il; upstream face of ir is f+1
            iu = ip(g,ir+1)
            cf = cut_face3(r(iu,k), r(ir,k), r(il,k), g.Vol[iu,k], g.Vol[ir,k], g.Vol[il,k],
                           g.FU[f+1,k] / (g.FU[f+1,k] + eps()), keepdown)
        end
        F = A * U * (cf + fbu(f,k))
        fc[il,k] -= F; fc[ir,k] += F
    end
    @inbounds for k in 2:nz, i in 1:nx            # z-face k between cells k-1 and k
        A = g.FW[i,k]; A > 0 || continue
        W = q[iW(g,i,k)]
        kb, kt = k-1, k
        if W >= 0       # upwind kb: upstream kb-1 (ghost at the bottom wall), face k-1
            ku = max(kb-1, 1)
            Vu = kb == 1 ? g.Vol[i,kb] : g.Vol[i,ku]
            cf = cut_face3(r(i,ku), r(i,kb), r(i,kt), Vu, g.Vol[i,kb], g.Vol[i,kt],
                           cut_fw(g,i,k-1) / (cut_fw(g,i,k-1) + eps()), keepdown)
        else            # upwind kt: upstream kt+1 (ghost at the top wall), face k+1
            ku = min(kt+1, nz)
            Vu = kt == nz ? g.Vol[i,kt] : g.Vol[i,ku]
            cf = cut_face3(r(i,ku), r(i,kt), r(i,kb), Vu, g.Vol[i,kt], g.Vol[i,kb],
                           g.FW[i,k+1] / (g.FW[i,k+1] + eps()), keepdown)
        end
        F = A * W * (cf + fbw(i,k))
        fc[i,kb] -= F; fc[i,kt] += F
    end
    @inbounds for k in 1:nz, i in 1:nx
        fc[i,k] = g.Vol[i,k] > 0 ? fc[i,k] / g.Vol[i,k] : 0.0
    end
    return fc
end

@inline function cut_shiftvel(AL, AR, vL, vR)
    AL > AR ? (vL, (AR * vR + (AL - AR) * vL) / (AL + eps())) :
              ((AL * vL + (AR - AL) * vR) / (AR + eps()), vR)
end

"""Cell-centred left/right momentum parts from face momenta (MATLAB `ShiftFaceCellCut`)."""
function cut_face_to_cell!(uCL, uCR, wCL, wCR, q, g::CutCellFVGrid)
    @inbounds for k in 1:g.nz, i in 1:g.nx
        uCL[i,k], uCR[i,k] = cut_shiftvel(g.FU[i,k], g.FU[i+1,k], q[iU(g,i,k)], q[iU(g,i+1,k)])
        wCL[i,k], wCR[i,k] = cut_shiftvel(cut_fw(g,i,k), g.FW[i,k+1], Wface(g,q,i,k), Wface(g,q,i,k+1))
    end
    return nothing
end

"""Volume-weighted face tendency from the cell tendencies of the two adjacent momentum parts
(MATLAB `ShiftCellFaceCut`): x-face f uses FuCL of cell f and FuCR of cell f-1."""
@inline function cut_cell_to_uface(g::CutCellFVGrid, FuCL, FuCR, f, k)
    il, ir = ip(g,f-1), f
    A = g.FU[f,k]
    VR = g.Vol[ir,k] * A / (A + g.FU[f+1,k] + eps())
    VL = g.Vol[il,k] * A / (A + g.FU[ip(g,f-1),k] + eps())
    return (VR * FuCL[ir,k] + VL * FuCR[il,k]) / (VL + VR + eps())
end

@inline function cut_cell_to_wface(g::CutCellFVGrid, FwCL, FwCR, i, k)
    A = g.FW[i,k]
    VR = g.Vol[i,k] * A / (A + g.FW[i,k+1] + eps())
    VL = g.Vol[i,k-1] * A / (A + cut_fw(g,i,k-1) + eps())
    return (VR * FwCL[i,k] + VL * FwCR[i,k-1]) / (VL + VR + eps())
end

"""
Θ flux reconstruction at faces whose upstream neighbour is missing (bottom/top wall, closed
upstream aperture), where MATLAB `AdvectionCutThird` falls back to donor cell:
- `:donor`        first-order donor cell (MATLAB). For stratified θ this is off by Δz/2·∂θ/∂z at
                  every z-face above a wall cell, ≈ 0.37 K at Δz = 244 m, N = 0.01 (warm layer).
- `:onesided`     keep the downstream term: volume-weighted linear interpolation between the
                  upwind and the downstream cell (second order, no background state).
- `:perturbation` reconstruct θ' = θ - θ̄(z) and add θ̄ at the face height (donor fallback kept).
"""
const CUT_THETA_RECON = Ref(:donor)

"""Third-order advective tendencies of Θ and the momentum parts into the cache arrays."""
function cut_advection_third!(m, q)
    g = m.grid; cc = m.cache.cc
    uCL, uCR, wCL, wCR, FuCL, FuCR, FwCL, FwCR, Fθ, Θc = cc
    @inbounds for k in 1:g.nz, i in 1:g.nx
        Θc[i,k] = q[iΘ(g,i,k)]
    end
    mode = CUT_THETA_RECON[]
    if mode === :perturbation && length(cc) >= 13
        cut_advect3!(Fθ, Θc, q, g, cc[11], cc[12], cc[13])
    else
        mode in (:donor, :onesided) || throw(ArgumentError("CUT_THETA_RECON = $mode"))
        cut_advect3!(Fθ, Θc, q, g; keepdown = mode === :onesided)
    end
    cut_face_to_cell!(uCL, uCR, wCL, wCR, q, g)
    cut_advect3!(FuCL, uCL, q, g); cut_advect3!(FuCR, uCR, q, g)
    cut_advect3!(FwCL, wCL, q, g); cut_advect3!(FwCR, wCR, q, g)
    return nothing
end

function rhs_cutcell!(out,m::FVModel{<:CutCellFVGrid},q)
    g=m.grid; nx=g.nx; nz=g.nz
    c=m.cache; θ=c.θ; p=c.pp; u=c.u; w=c.w; mx=c.mx; mz=c.mz
    fx=c.fx; fz=c.fz; fxu=c.fxu; fxw=c.fxw; fzu=c.fzu; fzw=c.fzw
    mdx=c.mdx; mdz=c.mdz
    fill!(u,0); fill!(w,0); fill!(fx,0); fill!(fz,0); fill!(fxu,0); fill!(fxw,0)
    fill!(fzu,0); fill!(fzw,0); fill!(mdx,0); fill!(mdz,0)
    @inbounds for k in 1:nz, i in 1:nx
        ρ=q[iρ(g,i,k)]; Θ=q[iΘ(g,i,k)]
        θ[i,k]=Θ/max(ρ,eps(Float64))
        p[i,k]=pressure(m.gas,Θ)-m.pbal[i,k]
        mx[i,k]=0.5*(q[iU(g,i,k)]+q[iU(g,i+1,k)])
        mz[i,k]=0.5*(Wface(g,q,i,k)+Wface(g,q,i,k+1))
    end
    @inbounds for k in 1:nz, i in 1:nx
        il=ip(g,i-1); ir=i
        ρf=0.5*(q[iρ(g,il,k)]+q[iρ(g,ir,k)])
        u[i,k]=q[iU(g,i,k)]/max(ρf,eps(Float64))
    end
    @inbounds for k in 2:nz, i in 1:nx
        ρf=0.5*(q[iρ(g,i,k-1)]+q[iρ(g,i,k)])
        w[i,k]=g.FW[i,k]>0 ? q[iW(g,i,k)]/max(ρf,eps(Float64)) : 0.0
    end
    @inbounds for k in 1:nz, i in 1:nx+1
        ii=ip(g,i-1)
        fx[i,k]=g.FU[i,k]*q[iU(g,i,k)]
        fxu[i,k]=fx[i,k]*cut_upwind(q[iU(g,i,k)],mx[ii,k]/max(q[iρ(g,ii,k)],eps(Float64)),
                                                               mx[ip(g,ii+1),k]/max(q[iρ(g,ip(g,ii+1),k)],eps(Float64)))
        fxw[i,k]=fx[i,k]*cut_upwind(q[iU(g,i,k)],mz[ii,k]/max(q[iρ(g,ii,k)],eps(Float64)),
                                                               mz[ip(g,ii+1),k]/max(q[iρ(g,ip(g,ii+1),k)],eps(Float64)))
    end
    @inbounds for k in 1:nz+1, i in 1:nx
        fz[i,k]=g.FW[i,k]*Wface(g,q,i,k)
        if 1<k<=nz
            fzu[i,k]=fz[i,k]*cut_upwind(Wface(g,q,i,k),mx[i,k-1]/max(q[iρ(g,i,k-1)],eps(Float64)),
                                                                       mx[i,k]/max(q[iρ(g,i,k)],eps(Float64)))
            fzw[i,k]=fz[i,k]*cut_upwind(Wface(g,q,i,k),mz[i,k-1]/max(q[iρ(g,i,k-1)],eps(Float64)),
                                                                       mz[i,k]/max(q[iρ(g,i,k)],eps(Float64)))
        end
    end
    fill!(out,0.0)
    third = g.advection == :third
    third && cut_advection_third!(m,q)
    @inbounds for k in 1:nz, i in 1:nx
        V=g.Vol[i,k]
        V>0 || continue
        il=ip(g,i-1); ir=ip(g,i+1)
        Fxl=fx[i,k]; Fxr=fx[i+1,k]
        Fzb=fz[i,k]; Fzt=fz[i,k+1]
        out[iρ(g,i,k)]=-(Fxr-Fxl+Fzt-Fzb)/V
        if third
            out[iΘ(g,i,k)]=c.cc[9][i,k]
            continue
        end
        θl=cut_upwind(q[iU(g,i,k)],θ[il,k],θ[i,k])
        θr=cut_upwind(q[iU(g,i+1,k)],θ[i,k],θ[ir,k])
        θb=k==1 ? θ[i,k] : cut_upwind(Wface(g,q,i,k),θ[i,k-1],θ[i,k])
        θt=k==nz ? θ[i,k] : cut_upwind(Wface(g,q,i,k+1),θ[i,k],θ[i,k+1])
        out[iΘ(g,i,k)]=-(Fxr*θr-Fxl*θl+Fzt*θt-Fzb*θb)/V
        mdx[i,k]=-(fxu[i+1,k]-fxu[i,k]+fzu[i,k+1]-fzu[i,k])/V
        mdz[i,k]=-(fxw[i+1,k]-fxw[i,k]+fzw[i,k+1]-fzw[i,k])/V
    end
    @inbounds for k in 1:nz, i in 1:nx
        il=ip(g,i-1); ir=i
        if g.FU[i,k]>0
            vl,vr=g.Vol[il,k],g.Vol[ir,k]
            den=vl+vr
            adv=third ? cut_cell_to_uface(g,c.cc[5],c.cc[6],i,k) :
                den>0 ? (vl*mdx[il,k]+vr*mdx[ir,k])/den : 0.0
            force=-(p[ir,k]-p[il,k])/g.dx
            out[iU(g,i,k)]=adv+force
        end
        if 1<k<=nz && g.FW[i,k]>0
            vb,vt=g.Vol[i,k-1],g.Vol[i,k]
            den=vb+vt
            adv=third ? cut_cell_to_wface(g,c.cc[7],c.cc[8],i,k) :
                den>0 ? (vb*mdz[i,k-1]+vt*mdz[i,k])/den : 0.0
            ρpb=q[iρ(g,i,k-1)]-m.ρbal[i,k-1]
            ρpt=q[iρ(g,i,k)]-m.ρbal[i,k]
            force=-(p[i,k]-p[i,k-1])/g.dz-m.gas.g*0.5*(ρpb+ρpt)
            out[iW(g,i,k)]=adv+force
        end
    end
    sponge!(out, m, q)
    return out
end

const CUT_ALL=(true,true,true,true)
const CUT_NONE=(false,false,false,false)
const CUT_THERMO=(true,false,false,true)
const CUT_MOM=(false,true,true,false)
const CUT_HMOM=(false,true,false,false)
const CUT_NOT_HMOM=(true,false,true,true)

fast_apply!(out,s::CutCellFVFast,q,f)=cutcell_fast!(out,s,q,f,true,true,CUT_ALL,CUT_ALL)
fast_rhs_thermo!(out,s::CutCellFVFast,q,f)=cutcell_fast!(out,s,q,f,true,true,CUT_THERMO,CUT_THERMO)
fast_rhs_momentum!(out,s::CutCellFVFast,q,f)=cutcell_fast!(out,s,q,f,true,true,CUT_MOM,CUT_MOM)
fast_rhs_horizontal!(out,s::CutCellFVFast,q,f)=cutcell_fast!(out,s,q,f,true,false,CUT_ALL,CUT_ALL)
vertical_apply!(out,s::CutCellFVFast,q)=cutcell_fast!(out,s,q,s.fb,false,true,CUT_ALL,CUT_NONE)
fast_rhs_hmomentum!(out,s::CutCellFVFast,q,f)=cutcell_fast!(out,s,q,f,true,false,CUT_HMOM,CUT_HMOM)
fast_rhs_hthermo!(out,s::CutCellFVFast,q,f)=cutcell_fast!(out,s,q,f,true,false,CUT_THERMO,CUT_NOT_HMOM)

"""Tridiagonal factorization of the cut-cell vertical `(I-c A_v)` column system."""
struct CutCellVerticalFactor
    c::Float64
    sub::Matrix{Float64}
    cp::Matrix{Float64}
    iden::Matrix{Float64}
end

function factor_cutcell_vertical!(F::CutCellVerticalFactor,s::CutCellFVFast,c)
    g=s.grid; nz=g.nz
    @inbounds for i in 1:g.nx, k in 2:nz
        if g.FW[i,k] == 0
            F.sub[k,i]=0; F.cp[k,i]=0; F.iden[k,i]=1
            continue
        end
        # The pressure row uses the two adjacent cell responses separately.
        cell_response(cellk,j,thermo) = begin
            V=g.Vol[i,cellk]
            V>0 || return 0.0
            bot=g.FW[i,cellk]; top=g.FW[i,cellk+1]
            fac=thermo ? s.θz[i,cellk] : 1.0
            -c/V*((j==cellk+1 ? top*fac : 0.0)-(j==cellk ? bot*fac : 0.0))
        end
        function rowcoef(j)
            dΘu=cell_response(k,j,true); dΘd=cell_response(k-1,j,true)
            dρu=cell_response(k,j,false); dρd=cell_response(k-1,j,false)
            (j==k ? 1.0 : 0.0) + c/g.dz*(s.a[i,k]*dΘu-s.a[i,k-1]*dΘd) +
                c*s.gas.g/2*(dρu+dρd)
        end
        sub= k>2 ? rowcoef(k-1) : 0.0
        diag=rowcoef(k)
        sup= k<nz ? rowcoef(k+1) : 0.0
        den=k==2 ? diag : diag-sub*F.cp[k-1,i]
        F.sub[k,i]=sub
        F.iden[k,i]=1/den
        F.cp[k,i]=sup/den
    end
    return F
end

function vertical_factor(s::CutCellFVFast,c)
    get!(s.lucache,Float64(c)) do
        g=s.grid
        F=CutCellVerticalFactor(Float64(c),zeros(g.nz+1,g.nx),zeros(g.nz+1,g.nx),zeros(g.nz+1,g.nx))
        factor_cutcell_vertical!(F,s,c)
    end
end

function vertical_solve!(Y,s::CutCellFVFast,F::CutCellVerticalFactor,Ys)
    g=s.grid; nz=g.nz; c=F.c; W=s.colbuf
    @inbounds for i in 1:g.nx
        for k in 2:nz
            if g.FW[i,k]==0
                W[k]=0.0
                continue
            end
            rhs=Ys[iW(g,i,k)] - c*(s.a[i,k]*Ys[iΘ(g,i,k)]-
                    s.a[i,k-1]*Ys[iΘ(g,i,k-1)])/g.dz -
                    c*s.gas.g/2*(Ys[iρ(g,i,k)]+Ys[iρ(g,i,k-1)])
            W[k]=(k==2 ? rhs : rhs-F.sub[k,i]*W[k-1])*F.iden[k,i]
        end
        for k in nz-1:-1:2
            g.FW[i,k]>0 && (W[k]-=F.cp[k,i]*W[k+1])
        end
        for k in 1:nz
            if g.Vol[i,k]==0
                Y[iρ(g,i,k)]=Ys[iρ(g,i,k)]; Y[iΘ(g,i,k)]=Ys[iΘ(g,i,k)]
                continue
            end
            Wb=k==1 ? 0.0 : W[k]
            Wt=k==nz ? 0.0 : W[k+1]
            ρ=Ys[iρ(g,i,k)]-c/g.Vol[i,k]*(g.FW[i,k+1]*Wt-g.FW[i,k]*Wb)
            Θ=Ys[iΘ(g,i,k)]-c/g.Vol[i,k]*(g.FW[i,k+1]*s.θz[i,k+1]*Wt-
                                           g.FW[i,k]*s.θz[i,k]*Wb)
            Y[iρ(g,i,k)]=ρ; Y[iΘ(g,i,k)]=Θ
            Y[iU(g,i,k)]=Ys[iU(g,i,k)]
        end
        for k in 2:nz
            Y[iW(g,i,k)]=g.FW[i,k]>0 ? W[k] : 0.0
        end
    end
    return Y
end

function hevi_loop!(q,s::CutCellFVFast,f,dt,nsteps,Y,k1,k2,tmp,rhs)
    F=vertical_factor(s,dt/2); halfdt=dt/2
    for _ in 1:nsteps
        fast_rhs_hthermo!(k1,s,q,f); @. q += (halfdt/2)*k1
        fast_rhs_hmomentum!(k2,s,q,f); @. q += halfdt*k2
        fast_rhs_hthermo!(k1,s,q,f); @. q += (halfdt/2)*k1
        vertical_apply!(tmp,s,q); @. rhs=q+(dt/2)*tmp
        vertical_solve!(Y,s,F,rhs); q .= Y
        fast_rhs_hthermo!(k1,s,q,f); @. q += (halfdt/2)*k1
        fast_rhs_hmomentum!(k2,s,q,f); @. q += halfdt*k2
        fast_rhs_hthermo!(k1,s,q,f); @. q += (halfdt/2)*k1
    end
    return q
end
