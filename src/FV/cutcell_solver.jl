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
    maskU::BitMatrix              # x-faces of the HEVI mask (nx+1, nz): implicit horizontal acoustics
    maskW::BitMatrix              # z-faces of the mask (nx, nz+1): with maskU the faces of the advection Jacobian
    wexp::Matrix{Float64}         # 1 on the explicit (unmasked) x-faces, 0 on the masked ones
    lumask::Dict{Float64,Any}     # sparse LU of I - c (A_v + A_h,mask) per c
end

"""
HEVI mask of the cut-cell grid: the cut cells (0 < α < 1) and their fluid neighbours within one
cell (x and z, as `mask_width = 1` in 3D); masked x- and z-faces are the open faces of masked
cells (returned as `(maskU, maskW)`). The
HEVI fast loop treats the horizontal acoustics through these faces implicitly, together with the
vertical part, so the explicit horizontal half steps only see full cells: the fast step is limited
by the regular Δx, not by the small cut cells (true fluid volumes in the divisions).
"""
function cutcell_hevi_mask(g::CutCellFVGrid)
    nx, nz = g.nx, g.nz
    cut = (0 .< g.alpha .< 1 - 64eps())
    cell = falses(nx, nz)
    for k in 1:nz, i in 1:nx
        g.alpha[i,k] > 0 || continue
        cell[i,k] = any(cut[ip(g,i+a), kk] for a in -1:1, kk in max(1,k-1):min(nz,k+1))
    end
    maskU = falses(nx+1, nz)
    for k in 1:nz, f in 1:nx+1
        maskU[f,k] = g.FU[f,k] > 0 && (cell[ip(g,f-1),k] || cell[ip(g,f),k])
    end
    maskW = falses(nx, nz+1)
    for k in 2:nz, i in 1:nx
        maskW[i,k] = g.FW[i,k] > 0 && (cell[i,k-1] || cell[i,k])
    end
    return maskU, maskW
end

function CutCellFVFast(m::FVModel{<:CutCellFVGrid})
    g, nq = m.grid, m.nq
    maskU, maskW = cutcell_hevi_mask(g)
    return CutCellFVFast(g, m.gas, nq, zeros(g.nx,g.nz), zeros(g.nx,g.nz),
                         zeros(g.nx,g.nz), zeros(g.nx,g.nz+1), zeros(nq), ones(nq),
                         [zeros(nq) for _ in 1:7], Dict{Float64,Any}(), 0.0, zeros(g.nz+1),
                         maskU, maskW, Float64.(.!maskU), Dict{Float64,Any}())
end

function cutcell_fast_operator(m::FVModel{<:CutCellFVGrid}, q; pfac=m.gas.γ, fast=:hevi,
                               reuse=true, kwargs...)
    fast in (:rosenbrock_amf, :hevi_amf) && return cutcell_amf_operator(m, q; pfac, reuse, kwargs...)
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
    empty!(s.lumask)
    return s
end

@inline cutcell_theta(g, s, q, i, k) =
    g.Vol[i,k] > 0 ? q[iΘ(g,i,k)]/max(q[iρ(g,i,k)],eps(Float64)) : 0.0

function cutcell_fast!(out, s::CutCellFVFast, q, f, horizontal::Bool, vertical::Bool,
                       rows::NTuple{4,Bool}, frows::NTuple{4,Bool}, explicit_only::Bool = false)
    g=s.grid; nx=g.nx; nz=g.nz; grav=s.gas.g
    # explicit_only: the horizontal terms through the unmasked x-faces (HEVI half steps)
    wf(f,k) = explicit_only ? s.wexp[f,k] : 1.0
    rρ,rU,rW,rΘ=rows
    fill!(out,0.0)
    @inbounds for k in 1:nz, i in 1:nx
        V=g.Vol[i,k]
        V > 0 || continue
        if horizontal
            il,ir=ip(g,i-1),i
            Ul=wf(i,k)*q[iU(g,i,k)]; Ur=wf(i+1,k)*q[iU(g,i+1,k)]
            rρ && (out[iρ(g,i,k)] -= (g.FU[i+1,k]*Ur-g.FU[i,k]*Ul)/V)
            if rΘ
                θl=cutcell_theta(g,s,q,il,k); θr=cutcell_theta(g,s,q,i,k)
                out[iΘ(g,i,k)] -= (g.FU[i+1,k]*s.θx[ip(g,i+1),k]*Ur-
                                    g.FU[i,k]*s.θx[i,k]*Ul)/V
            end
            if rU && g.FU[i,k] > 0
                pL=s.a[il,k]*q[iΘ(g,il,k)]; pR=s.a[i,k]*q[iΘ(g,i,k)]
                out[iU(g,i,k)] -= wf(i,k)*(pR-pL)/g.dx
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

# Third-order cut-cell advection (MATLAB `AdvectionCutThird`, `ShiftFaceCellCut`, `ShiftCellFaceCut`):
# the common scheme of all FV grids, shift_advection.jl (virtual grid sizes from volumes and faces).

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
    a = third ? shift_advection!(c.adv, g, q) : c.adv     # θ face values, momentum (shift_advection.jl)
    @inbounds for k in 1:nz, i in 1:nx
        V=g.Vol[i,k]
        V>0 || continue
        il=ip(g,i-1); ir=ip(g,i+1)
        Fxl=fx[i,k]; Fxr=fx[i+1,k]
        Fzb=fz[i,k]; Fzt=fz[i,k+1]
        out[iρ(g,i,k)]=-(Fxr-Fxl+Fzt-Fzb)/V
        if third
            out[iΘ(g,i,k)]=-(Fxr*a.θfx[ip(g,i+1),k]-Fxl*a.θfx[i,k]+Fzt*a.θfz[i,k+1]-Fzb*a.θfz[i,k])/V
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
            adv=third ? a.SU[i,k] :
                den>0 ? (vl*mdx[il,k]+vr*mdx[ir,k])/den : 0.0
            force=-(p[ir,k]-p[il,k])/g.dx
            out[iU(g,i,k)]=adv+force
        end
        if 1<k<=nz && g.FW[i,k]>0
            vb,vt=g.Vol[i,k-1],g.Vol[i,k]
            den=vb+vt
            adv=third ? a.SW[i,k] :
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

"""Sparse matrix A_v + A_h,mask of the implicit HEVI step (linear fast operator without the
forcing): vertical acoustics and gravity everywhere, horizontal acoustics on the masked x-faces."""
function cutcell_implicit_matrix(s::CutCellFVFast)
    g = s.grid; nx, nz = g.nx, g.nz
    I = Int[]; J = Int[]; V = Float64[]
    add!(r, c, v) = (push!(I, r); push!(J, c); push!(V, v))
    @inbounds for k in 1:nz, i in 1:nx
        Vc = g.Vol[i,k]
        Vc > 0 || continue
        for (kf, sg) in ((k, 1.0), (k+1, -1.0))          # lower face: inflow, upper: outflow
            2 <= kf <= nz && g.FW[i,kf] > 0 || continue
            add!(iρ(g,i,k), iW(g,i,kf), sg * g.FW[i,kf] / Vc)
            add!(iΘ(g,i,k), iW(g,i,kf), sg * g.FW[i,kf] * s.θz[i,kf] / Vc)
        end
    end
    @inbounds for k in 2:nz, i in 1:nx
        g.FW[i,k] > 0 || continue
        add!(iW(g,i,k), iΘ(g,i,k), -s.a[i,k] / g.dz); add!(iW(g,i,k), iΘ(g,i,k-1), s.a[i,k-1] / g.dz)
        add!(iW(g,i,k), iρ(g,i,k), -s.gas.g / 2);    add!(iW(g,i,k), iρ(g,i,k-1), -s.gas.g / 2)
    end
    @inbounds for k in 1:nz, f in 1:nx
        s.maskU[f,k] || continue
        il, ir = ip(g,f-1), f; A = g.FU[f,k]; θf = s.θx[f,k]
        if g.Vol[il,k] > 0
            add!(iρ(g,il,k), iU(g,f,k), -A / g.Vol[il,k]); add!(iΘ(g,il,k), iU(g,f,k), -A * θf / g.Vol[il,k])
        end
        if g.Vol[ir,k] > 0
            add!(iρ(g,ir,k), iU(g,f,k), A / g.Vol[ir,k]);  add!(iΘ(g,ir,k), iU(g,f,k), A * θf / g.Vol[ir,k])
        end
        add!(iU(g,f,k), iΘ(g,il,k), s.a[il,k] / g.dx); add!(iU(g,f,k), iΘ(g,ir,k), -s.a[ir,k] / g.dx)
    end
    return sparse(I, J, V, s.nq, s.nq)
end

"""(A, LU of I - c A) of the implicit masked step, cached per c until the next re-freeze."""
cutcell_mask_factor(s::CutCellFVFast, c) = get!(s.lumask, Float64(c)) do
    A = cutcell_implicit_matrix(s)
    (A = A, F = lu(sparse(1.0I, s.nq, s.nq) - c * A))
end

"""
HEVI fast loop: Strang splitting of explicit horizontal half steps (Verlet-type, through the
unmasked x-faces) and a Crank–Nicolson step of the vertical part. With a non-empty HEVI mask
(`cutcell_hevi_mask`) the implicit step also contains the horizontal acoustics through the masked
x-faces (sparse LU), so the small cut cells do not limit the fast step.
"""
function hevi_loop!(q,s::CutCellFVFast,f,dt,nsteps,Y,k1,k2,tmp,rhs)
    halfdt=dt/2
    masked = any(s.maskU)
    ex = masked
    if masked
        M = cutcell_mask_factor(s, dt/2)
    else
        F = vertical_factor(s,dt/2)
    end
    for _ in 1:nsteps
        hthermo!(k1,s,q,f,ex); @. q += (halfdt/2)*k1
        hmomentum!(k2,s,q,f,ex); @. q += halfdt*k2
        hthermo!(k1,s,q,f,ex); @. q += (halfdt/2)*k1
        if masked
            mul!(tmp, M.A, q); @. rhs = q + (dt/2)*tmp
            ldiv!(Y, M.F, rhs); q .= Y
        else
            vertical_apply!(tmp,s,q); @. rhs=q+(dt/2)*tmp
            vertical_solve!(Y,s,F,rhs); q .= Y
        end
        hthermo!(k1,s,q,f,ex); @. q += (halfdt/2)*k1
        hmomentum!(k2,s,q,f,ex); @. q += halfdt*k2
        hthermo!(k1,s,q,f,ex); @. q += (halfdt/2)*k1
    end
    return q
end
hmomentum!(out,s::CutCellFVFast,q,f,ex)=cutcell_fast!(out,s,q,f,true,false,CUT_HMOM,CUT_HMOM,ex)
hthermo!(out,s::CutCellFVFast,q,f,ex)=cutcell_fast!(out,s,q,f,true,false,CUT_THERMO,CUT_NOT_HMOM,ex)
