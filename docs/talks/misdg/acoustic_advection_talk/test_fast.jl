# test_fast.jl -- checks of the fast integrator against exact exp(hL) and phi_k(hL)
# usage:  julia test_fast.jl
include("fast_fb.jl")
using .FastFB
using Printf

m  = Model()
m0 = Model(U = 0.0)                       # acoustic part only (L); advection switched off
F  = dft_matrix(m.nx)
x0 = 0.25 * m.Lx
p0 = [exp(-((x - x0) / (m.Lx / 16))^2) for x in m.xc]
u0 = zeros(m.nx)
h  = 50.0                                 # c h / dx = 7

# exp(hL): error vs exact, order 2/4/6 and substeps per h
pe, ue = exact_solution(m0, p0, u0, h; F = F)
println("exp(hL): max error vs exact  (h = $h s, acoustic Courant number c h/dx = $(m.c*h/m.dx))")
for order in (2, 4, 6)
    cf = Coeffs(order)
    errs = Float64[]
    for nsub in (16, 32, 64, 128)
        p = copy(p0); u = copy(u0)
        fast!(p, u, m0, h, nsub, cf, Vector{Vector{Float64}}(), Vector{Vector{Float64}}())
        push!(errs, max(maximum(abs.(p .- pe)), maximum(abs.(u .- ue))))
    end
    @printf("  order %d: %s\n", order, join([@sprintf("%.1e", e) for e in errs], "  "))
end

# phi_k(hL) g via polynomial source  s(t) = g t^(k-1)/(k-1)!  :  z(h) = h^k phi_k(hL) g
fh = zeros(m.nx)
gp = zeros(m.nx); adv!(gp, p0, m, fh)
gu = 1e-3 .* sin.(2pi .* m.xc ./ m.Lx)
println("phi_k(hL) g: relative error vs exact")
for k in 1:3
    fe_p, fe_u = exact_phi(m0, k, gp, gu, h; F = F)
    for order in (2, 4, 6)
        cf = Coeffs(order)
        errs = Float64[]
        for nsub in (16, 32, 64, 128)
            G_p = [zeros(m.nx) for _ in 1:k]; G_p[k] .= gp
            G_u = [zeros(m.nx) for _ in 1:k]; G_u[k] .= gu
            p = zeros(m.nx); u = zeros(m.nx)
            fast!(p, u, m0, h, nsub, cf, G_p, G_u)
            p ./= h^k; u ./= h^k
            push!(errs, max(maximum(abs.(p .- fe_p)) / maximum(abs.(fe_p)),
                            maximum(abs.(u .- fe_u)) / maximum(abs.(fe_u))))
        end
        @printf("  phi_%d order %d: %s\n", k, order, join([@sprintf("%.1e", e) for e in errs], "  "))
    end
end
