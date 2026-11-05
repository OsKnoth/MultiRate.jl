"""
Makie plots for Euler2D: each element is resampled (exactly) to an equispaced (2p+1)² grid and
drawn as a triangle mesh with per-vertex colours, so curved terrain-following elements and the
polynomial solution inside each element are shown without an intermediate regular grid.
Works with any Makie backend (CairoMakie for files, GLMakie for screens).
"""
module MultiRateMakieExt

using Makie
using MultiRate.Common: output_fields
using MultiRate.Euler2D: Euler2DModel
using MultiRate.FV2D: FVModel, CutCellFVGrid
using MultiRate.HDGRelaxTheta: lagrange_basis
import MultiRate.Common: plot_fields

const DIVERGING = ("u_prime", "w", "rho_prime", "theta_prime", "p_prime")

function element_triangulation(m::Euler2DModel, nsub)
    M = m.mesh
    V, _ = lagrange_basis(M.xg, collect(range(-1, 1; length = nsub)))
    npe = nsub^2
    pts = zeros(Float32, npe * M.Ne, 2)
    faces = zeros(Int, 2 * (nsub - 1)^2 * M.Ne, 3)
    lin(i, j) = i + (j - 1) * nsub
    k = 0
    for e in 1:M.Ne
        o = (e - 1) * npe
        pts[o+1:o+npe, 1] = vec(V * M.X[:, :, e] * V')
        pts[o+1:o+npe, 2] = vec(V * M.Z[:, :, e] * V')
        for j in 1:nsub-1, i in 1:nsub-1
            a, b, c, d = o + lin(i, j), o + lin(i + 1, j), o + lin(i + 1, j + 1), o + lin(i, j + 1)
            faces[k+1, :] = [a, b, c]
            faces[k+2, :] = [a, c, d]
            k += 2
        end
    end
    return V, pts, faces
end

function plot_fields(m::Euler2DModel, q; names = ["theta_prime", "w"], title = "", zoom = nothing,
                     nsub = 2 * m.mesh.p + 1, xscale = 1e-3, zscale = 1e-3, size = (1100, 380 * length(names)))
    M = m.mesh
    V, pts, faces = element_triangulation(m, nsub)
    pts[:, 1] .*= xscale
    pts[:, 2] .*= zscale
    F = Dict(output_fields(m, q))
    fig = Figure(; size = size)
    for (row, name) in enumerate(names)
        haskey(F, name) || error("unknown field $name; available: $(join(first.(output_fields(m, q)), ", "))")
        vals = reduce(vcat, [vec(V * F[name][:, :, e] * V') for e in 1:M.Ne])
        if name in DIVERGING
            a = maximum(abs, vals)
            a = a > 0 ? a : 1.0
            cmap, crange = :balance, (-a, a)
        else
            cmap, crange = :viridis, extrema(vals)
        end
        ax = Axis(fig[row, 1]; title = row == 1 && !isempty(title) ? "$title — $name" : name,
                  xlabel = xscale == 1e-3 ? "x [km]" : "x", ylabel = zscale == 1e-3 ? "z [km]" : "z")
        plt = mesh!(ax, pts, faces; color = Float32.(vals), colormap = cmap, colorrange = crange,
                    shading = NoShading)
        if zoom !== nothing
            xlims!(ax, zoom[1] .* xscale)
            ylims!(ax, zoom[2] .* zscale)
        end
        Colorbar(fig[row, 2], plt)
    end
    return fig
end

"""FV2D: quadrilateral cell values on the physical (possibly terrain-following) mesh."""
function plot_fields(m::FVModel, q; names = ["theta_prime", "w"], title = "", zoom = nothing,
                     xscale = 1e-3, zscale = 1e-3, size = (1100, 380 * length(names)))
    g = m.grid
    F = Dict(output_fields(m, q))
    ncell = g.nx * g.nz
    pts = zeros(Float32, 4ncell, 2)
    faces = zeros(Int, 2ncell, 3)
    cellid(i, k) = i + (k - 1) * g.nx
    vertex_values(Fv) = begin
        v = zeros(Float32, 4ncell)
        for k in 1:g.nz, i in 1:g.nx
            c = cellid(i, k)
            o = 4(c - 1)
            il = i
            ir = i + 1
            xr = i == g.nx ? g.x0 + g.Lx : g.xf[ir]
            pts[o+1, :] .= (g.xf[il], g.Zv[il, k])
            pts[o+2, :] .= (xr, g.Zv[ir, k])
            pts[o+3, :] .= (xr, g.Zv[ir, k+1])
            pts[o+4, :] .= (g.xf[il], g.Zv[il, k+1])
            faces[2c-1, :] .= (o+1, o+2, o+3)
            faces[2c, :] .= (o+1, o+3, o+4)
            v[o+1:o+4] .= Float32(Fv[i, k])
        end
        v
    end
    pts[:, 1] .*= xscale
    pts[:, 2] .*= zscale
    fig = Figure(; size = size)
    for (row, name) in enumerate(names)
        haskey(F, name) || error("unknown field $name")
        vals = F[name]
        if name in DIVERGING
            a = maximum(abs, vals)
            a = a > 0 ? a : 1.0
            cmap, crange = :balance, (-a, a)
        else
            cmap, crange = :viridis, extrema(vals)
        end
        ax = Axis(fig[row, 1]; title = row == 1 && !isempty(title) ? "$title — $name" : name,
                  xlabel = xscale == 1e-3 ? "x [km]" : "x", ylabel = zscale == 1e-3 ? "z [km]" : "z")
        hm = mesh!(ax, pts, faces; color = vertex_values(vals), colormap = cmap,
                   colorrange = crange, shading = NoShading)
        if zoom !== nothing
            xlims!(ax, zoom[1] .* xscale)
            ylims!(ax, zoom[2] .* zscale)
        end
        Colorbar(fig[row, 2], hm)
    end
    return fig
end

"""FV2D cut-cell fields drawn on the actual clipped fluid polygons."""
function plot_fields(m::FVModel{<:CutCellFVGrid}, q; names = ["theta_prime", "w"], title = "", zoom = nothing,
                     xscale = 1e-3, zscale = 1e-3, size = (1100, 380 * length(names)))
    g=m.grid; F=Dict(output_fields(m,q))
    active=[(i,k) for k in 1:g.nz for i in 1:g.nx if g.Vol[i,k]>0 && length(g.polygons[i,k])>=3]
    npts=sum(length(g.polygons[i,k]) for (i,k) in active)
    ntri=sum(length(g.polygons[i,k])-2 for (i,k) in active)
    pts=zeros(Float32,npts,2); faces=zeros(Int,ntri,3)
    vertex_values(vals)=begin
        out=zeros(Float32,npts); point=0
        for (i,k) in active
            poly=g.polygons[i,k]
            for _ in poly
                point+=1; out[point]=Float32(vals[i,k])
            end
        end
        out
    end
    point=0; face=0
    for (i,k) in active
        poly=g.polygons[i,k]; first=point+1
        for (x,z) in poly
            point+=1; pts[point,:].=(x*xscale,z*zscale)
        end
        for j in 2:length(poly)-1
            face+=1; faces[face,:].=(first,first+j-1,first+j)
        end
    end
    fig=Figure(;size=size)
    for (row,name) in enumerate(names)
        haskey(F,name) || error("unknown field $name")
        vals=F[name]
        if name in DIVERGING
            a=maximum(abs,vals); a=a>0 ? a : 1.0; cmap,crange=:balance,(-a,a)
        else
            cmap,crange=:viridis,extrema(vals)
        end
        ax=Axis(fig[row,1];title=row==1 && !isempty(title) ? "$title — $name" : name,
                xlabel=xscale==1e-3 ? "x [km]" : "x",ylabel=zscale==1e-3 ? "z [km]" : "z")
        hm=mesh!(ax,pts,faces;color=vertex_values(vals),colormap=cmap,colorrange=crange,shading=NoShading)
        if zoom!==nothing
            xlims!(ax,zoom[1].*xscale); ylims!(ax,zoom[2].*zscale)
        end
        Colorbar(fig[row,2],hm)
    end
    return fig
end

end # module
