# Partially implicit Rosenbrock-W fast solver for cut-cell FV.
# The frozen fast Jacobian is acoustic over the domain and additionally contains a
# low-order advection Jacobian in the cut-cell mask. Its approximate factors are
# (1) vertical acoustics everywhere plus horizontal acoustics on masked faces and
# (2) masked-cell advection.

mutable struct CutCellFVAMF <: AbstractFastOperator
    model::Any
    acoustic::CutCellFVFast
    Aadv::SparseMatrixCSC{Float64,Int}
    Hmask::SparseMatrixCSC{Float64,Int}
    advforcing::Vector{Float64}
    work::Vector{Vector{Float64}}
    factors::Dict{Float64,Any}
end

fast_forcing(s::CutCellFVAMF) = s.advforcing
fast_mass(s::CutCellFVAMF) = s.acoustic.Mdiag
fast_workspace(s::CutCellFVAMF) = s.acoustic.work

function fast_apply!(out, s::CutCellFVAMF, q, f)
    z = s.work[9]
    fill!(z, 0.0)
    fast_apply!(out, s.acoustic, q, z)
    mul!(out, s.Aadv, q, 1.0, 1.0)
    out .+= f
    return out
end

function cutcell_amf_operator(m::FVModel{<:CutCellFVGrid}, q; pfac=m.gas.γ, reuse=true, kwargs...)
    s = reuse && m.fastref[] isa CutCellFVAMF ? m.fastref[]::CutCellFVAMF :
        CutCellFVAMF(m, CutCellFVFast(m), spzeros(m.nq,m.nq), spzeros(m.nq,m.nq),
                     zeros(m.nq), [zeros(m.nq) for _ in 1:9], Dict{Float64,Any}())
    refreeze_cutcell!(s.acoustic, m, q, pfac)
    build_cutcell_amf!(s, q)
    empty!(s.factors)
    reuse && (m.fastref[] = s)
    return s
end

function cutcell_amf_indices(g::CutCellFVGrid)
    nx,nz=g.nx,g.nz
    cellactive=falses(nx,nz)
    for k in 1:nz, i in 1:nx
        cellactive[i,k] = g.Vol[i,k] > 0 && (g.alpha[i,k] < 1-64eps(Float64) || g.MaskVol[i,k])
    end
    rows=BitSet(); cols=BitSet()
    for k in 1:nz, i in 1:nx
        cellactive[i,k] || continue
        push!(rows,iρ(g,i,k)); push!(rows,iΘ(g,i,k))
        # One-cell stencil closure for the frozen first-order advection Jacobian.
        for dk in -1:1, di in -1:1
            kk=k+dk; 1<=kk<=nz || continue
            ii=ip(g,i+di)
            push!(cols,iρ(g,ii,kk)); push!(cols,iΘ(g,ii,kk))
            push!(cols,iU(g,ii,kk)); push!(cols,iU(g,ii+1,kk))
            1<kk<=nz && push!(cols,iW(g,ii,kk))
            1<kk+1<=nz && push!(cols,iW(g,ii,kk+1))
        end
    end
    for k in 1:nz, f in 1:nx
        if g.MaskU[f,k] && g.FU[f,k]>0
            push!(rows,iU(g,f,k)); push!(cols,iU(g,f,k))
        end
    end
    for k in 2:nz, i in 1:nx
        g.MaskW[i,k] && g.FW[i,k]>0 && push!(rows,iW(g,i,k))
    end
    return collect(rows), collect(cols)
end

function cutcell_masked_horizontal_acoustics(s::CutCellFVAMF)
    g=s.acoustic.grid; n=s.acoustic.nq
    I=Int[]; J=Int[]; V=Float64[]
    for k in 1:g.nz, f in 1:g.nx
        g.MaskU[f,k] && g.FU[f,k]>0 || continue
        left,right=ip(g,f-1),f
        Ul=iU(g,f,k); area=g.FU[f,k]
        # Face f is the right face of `left` (outflow for U > 0) and the left face of `right`.
        if g.Vol[left,k]>0
            push!(I,iρ(g,left,k)); push!(J,Ul); push!(V,-area/g.Vol[left,k])
            push!(I,iΘ(g,left,k)); push!(J,Ul); push!(V,-area*s.acoustic.θx[f,k]/g.Vol[left,k])
        end
        if g.Vol[right,k]>0
            push!(I,iρ(g,right,k)); push!(J,Ul); push!(V, area/g.Vol[right,k])
            push!(I,iΘ(g,right,k)); push!(J,Ul); push!(V, area*s.acoustic.θx[f,k]/g.Vol[right,k])
        end
        push!(I,Ul); push!(J,iΘ(g,left,k)); push!(V,s.acoustic.a[left,k]/g.dx)
        push!(I,Ul); push!(J,iΘ(g,right,k)); push!(V,-s.acoustic.a[right,k]/g.dx)
    end
    return sparse(I,J,V,n,n)
end

function build_cutcell_amf!(s::CutCellFVAMF,q)
    m=s.model; base=s.acoustic; n=length(q)
    rows,cols=cutcell_amf_indices(base.grid)
    r0=similar(q); a0=similar(q); z=s.work[9]
    rhs_cutcell!(r0,m,q)
    fill!(z,0.0); fast_apply!(a0,base,q,z)
    @. r0 = r0 - a0 - base.fb
    I=Int[]; J=Int[]; V=Float64[]
    qp=copy(q); rp=similar(q); ap=similar(q)
    @inbounds for j in cols
        h=cbrt(eps(Float64))*max(abs(q[j]),1.0)
        qp[j]=q[j]+h
        rhs_cutcell!(rp,m,qp)
        fill!(z,0.0); fast_apply!(ap,base,qp,z)
        for i in rows
            value=((rp[i]-ap[i]-base.fb[i])-r0[i])/h
            abs(value)>1e-11 && (push!(I,i); push!(J,j); push!(V,value))
        end
        qp[j]=q[j]
    end
    s.Aadv=sparse(I,J,V,n,n)
    advvalue=s.work[8]
    mul!(advvalue,s.Aadv,q)
    s.advforcing.=base.fb
    @inbounds for i in rows
        s.advforcing[i]+=r0[i]-advvalue[i]
    end
    s.Hmask=cutcell_masked_horizontal_acoustics(s)
    return s
end

function cutcell_amf_factors(s::CutCellFVAMF,c::Float64)
    get!(s.factors,c) do
        n=s.acoustic.nq
        Isp=spdiagm(0=>ones(n))
        (vertical=vertical_factor(s.acoustic,c),
         horizontal=lu(Isp-c*s.Hmask),
         advection=lu(Isp-c*s.Aadv))
    end
end

function cutcell_amf_solve!(out,s::CutCellFVAMF,F,rhs,tmp,c)
    fac=cutcell_amf_factors(s,c)
    vertical_solve!(tmp,s.acoustic,fac.vertical,F)
    ldiv!(rhs,fac.horizontal,tmp)
    ldiv!(out,fac.advection,rhs)
    return out
end

"""Three-stage Rosenbrock-W fast integration with vertical/masked acoustics and masked advection factors."""
function fast_rosenbrock_amf!(qout,s::CutCellFVAMF,q0,f,dt,nsteps)
    # RosRK3 coefficients from the MATLAB reference (β₀ = 1).
    α11=1/3; α21=11/54; α22=1/2
    α31=-17/27; α32=11/4; α33=1.0
    # β31 = +17/27: order 3 for W = J and W = 0, order 2 for general W (checked on y' = λy).
    β21=-11/27; β31=17/27; β32=-11/4
    c=Float64(dt)
    q, q0s, stage, k1, k2, k3, rhs, tmp, tmp2=s.work
    q.=q0
    for _ in 1:nsteps
        q0s.=q
        fast_apply!(rhs,s,q0s,f); rhs .*= dt
        cutcell_amf_solve!(k1,s,rhs,tmp,tmp2,c)

        @. stage=q0s+α11*k1
        fast_apply!(rhs,s,stage,f); rhs .*= dt
        @. rhs+=β21*k1
        cutcell_amf_solve!(k2,s,rhs,tmp,tmp2,c)

        @. stage=q0s+α21*k1+α22*k2
        fast_apply!(rhs,s,stage,f); rhs .*= dt
        @. rhs+=β31*k1+β32*k2
        cutcell_amf_solve!(k3,s,rhs,tmp,tmp2,c)

        @. q=q0s+α31*k1+α32*k2+α33*k3
    end
    qout.=q
    return qout
end
