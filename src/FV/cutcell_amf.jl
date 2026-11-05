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

# Simplified masked advection Jacobian (MATLAB `JacAdvecCut`, `JacCACut`): first-order upwind
# transport of conserved cell quantities c = ρφ with face momenta and ρ frozen at qⁿ, restricted
# to masked open faces. ρ and Θ use it directly; the momenta are split into cell parts, advected
# with the same matrix and mapped back to the faces (`JacShiftFaceCellCut`/`JacShiftCellFaceCut`).
# It only approximates the third-order right-hand side; the W-method order does not depend on it.

@inline cut_cell(g::CutCellFVGrid, i, k) = ip(g,i) + (k - 1) * g.nx
@inline cut_ul(g::CutCellFVGrid, f, k) = ip(g,f) + (k - 1) * g.nx      # U face f = 1..nx+1
@inline cut_wl(g::CutCellFVGrid, i, k) = i + (k - 2) * g.nx            # W face k = 2..nz

"""Cell-to-cell upwind transport matrix on masked faces, frozen at state `q`."""
function cutcell_transport_matrix(g::CutCellFVGrid, q)
    I = Int[]; J = Int[]; V = Float64[]
    function add!(cl, cr, flux, cup)   # flux > 0 from cl to cr; cup upwind cell index pair
        iu, ku = cup
        a = flux / max(q[iρ(g,iu,ku)], eps())
        g.Vol[cl...] > 0 && (push!(I, cut_cell(g, cl...)); push!(J, cut_cell(g, iu, ku)); push!(V, -a / g.Vol[cl...]))
        g.Vol[cr...] > 0 && (push!(I, cut_cell(g, cr...)); push!(J, cut_cell(g, iu, ku)); push!(V,  a / g.Vol[cr...]))
    end
    @inbounds for k in 1:g.nz, f in 1:g.nx
        g.MaskU[f,k] && g.FU[f,k] > 0 || continue
        U = q[iU(g,f,k)]; il = ip(g,f-1)
        add!((il,k), (f,k), g.FU[f,k] * U, U >= 0 ? (il,k) : (f,k))
    end
    @inbounds for k in 2:g.nz, i in 1:g.nx
        g.MaskW[i,k] && g.FW[i,k] > 0 || continue
        W = q[iW(g,i,k)]
        add!((i,k-1), (i,k), g.FW[i,k] * W, W >= 0 ? (i,k-1) : (i,k))
    end
    nc = g.nx * g.nz
    return sparse(I, J, V, nc, nc)
end

"""Linear maps face momenta → cell parts [left; right] and back (MATLAB shift Jacobians)."""
function cutcell_shift_matrices(g::CutCellFVGrid)
    nx, nz = g.nx, g.nz; nc = nx * nz; nU = nx * nz; nW = nx * (nz - 1)
    coef(AL, AR) = AL > AR ? ((1.0, 0.0), ((AL - AR) / (AL + eps()), AR / (AL + eps()))) :
                             ((AL / (AR + eps()), (AR - AL) / (AR + eps())), (0.0, 1.0))
    Iu = Int[]; Ju = Int[]; Vu = Float64[]; Iw = Int[]; Jw = Int[]; Vw = Float64[]
    push3!(I, J, V, r, c, v) = (v != 0 && (push!(I, r); push!(J, c); push!(V, v)))
    for k in 1:nz, i in 1:nx
        c = cut_cell(g, i, k)
        (lL, lR), (rL, rR) = coef(g.FU[i,k], g.FU[i+1,k])      # uCL = lL UL + lR UR, uCR = rL UL + rR UR
        push3!(Iu, Ju, Vu, c, cut_ul(g,i,k), lL);      push3!(Iu, Ju, Vu, c, cut_ul(g,i+1,k), lR)
        push3!(Iu, Ju, Vu, nc + c, cut_ul(g,i,k), rL); push3!(Iu, Ju, Vu, nc + c, cut_ul(g,i+1,k), rR)
        (lL, lR), (rL, rR) = coef(cut_fw(g,i,k), g.FW[i,k+1])
        k > 1  && (push3!(Iw, Jw, Vw, c, cut_wl(g,i,k), lL);  push3!(Iw, Jw, Vw, nc + c, cut_wl(g,i,k), rL))
        k < nz && (push3!(Iw, Jw, Vw, c, cut_wl(g,i,k+1), lR); push3!(Iw, Jw, Vw, nc + c, cut_wl(g,i,k+1), rR))
    end
    Sfu = sparse(Iu, Ju, Vu, 2nc, nU); Sfw = sparse(Iw, Jw, Vw, 2nc, nW)
    Iu = Int[]; Ju = Int[]; Vu = Float64[]; Iw = Int[]; Jw = Int[]; Vw = Float64[]
    for k in 1:nz, f in 1:nx                                   # x-face: FuCL of cell f, FuCR of f-1
        A = g.FU[f,k]; A > 0 || continue
        il = ip(g,f-1)
        VR = g.Vol[f,k] * A / (A + g.FU[f+1,k] + eps()); VL = g.Vol[il,k] * A / (A + g.FU[il,k] + eps())
        den = VL + VR + eps(); r = cut_ul(g,f,k)
        push3!(Iu, Ju, Vu, r, cut_cell(g,f,k), VR / den); push3!(Iu, Ju, Vu, r, nc + cut_cell(g,il,k), VL / den)
    end
    for k in 2:nz, i in 1:nx
        A = g.FW[i,k]; A > 0 || continue
        VR = g.Vol[i,k] * A / (A + g.FW[i,k+1] + eps()); VL = g.Vol[i,k-1] * A / (A + cut_fw(g,i,k-1) + eps())
        den = VL + VR + eps(); r = cut_wl(g,i,k)
        push3!(Iw, Jw, Vw, r, cut_cell(g,i,k), VR / den); push3!(Iw, Jw, Vw, r, nc + cut_cell(g,i,k-1), VL / den)
    end
    return Sfu, sparse(Iu, Ju, Vu, nU, 2nc), Sfw, sparse(Iw, Jw, Vw, nW, 2nc)
end

"""Assemble the simplified masked advection Jacobian in global unknown ordering."""
function cutcell_advection_jacobian(g::CutCellFVGrid, q, nq)
    nc = g.nx * g.nz
    A = cutcell_transport_matrix(g, q)
    A2 = blockdiag(A, A)
    Sfu, Scu, Sfw, Scw = cutcell_shift_matrices(g)
    Auu = Scu * A2 * Sfu
    Aww = Scw * A2 * Sfw
    # global offsets (see iρ, iU, iW, iΘ): ρ cells, U faces, W faces, Θ cells
    oρ, oU, oW, oΘ = 0, nc, 2nc, 2nc + g.nx * (g.nz - 1)
    I = Int[]; J = Int[]; V = Float64[]
    for (M, ro, co) in ((A, oρ, oρ), (A, oΘ, oΘ), (Auu, oU, oU), (Aww, oW, oW))
        Ii, Jj, Vv = findnz(M)
        append!(I, Ii .+ ro); append!(J, Jj .+ co); append!(V, Vv)
    end
    return sparse(I, J, V, nq, nq)
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
    s.Aadv=cutcell_advection_jacobian(s.acoustic.grid,q,length(q))
    # A constant fast forcing cancels in the MIS stage (it enters S = N - L q - f as well), so
    # the acoustic forcing is sufficient.
    s.advforcing.=s.acoustic.fb
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
