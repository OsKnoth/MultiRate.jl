# Warm bubble at constant pressure in a stably stratified atmosphere.
# Compare DGSEM + HDG (including optional face relaxation) with the finite-volume C-grid.
#
#     julia --project=examples examples/intro_2d/warm_bubble.jl dg [T] [hevi|hevi_relax|fb|verlet]
#     julia --project=examples examples/intro_2d/warm_bubble.jl fv [T] [hevi|wrf|fb|verlet]
#
# Configure with LX, H, NEX, NEZ, P, BUBBLE_X, BUBBLE_Z, BUBBLE_RX, BUBBLE_RZ,
# DELTA_THETA, N, U, NSUB and EPS_IF. Both use MIS4 with HEVI fast steps by default.

include(joinpath(@__DIR__, "../euler2d_common.jl"))

disc = isempty(ARGS) ? :dg : Symbol(ARGS[1])
disc in (:dg, :fv) || error("discretization must be dg or fv")
LX = parse(Float64, get(ENV, "LX", "20000"))
H = parse(Float64, get(ENV, "H", "10000"))
P = parse(Int, get(ENV, "P", "4"))
NEX = parse(Int, get(ENV, "NEX", "10"))
NEZ = parse(Int, get(ENV, "NEZ", "6"))
N = parse(Float64, get(ENV, "N", "0.01"))
U = parse(Float64, get(ENV, "U", "0"))
Δθ = parse(Float64, get(ENV, "DELTA_THETA", "2"))
rx = parse(Float64, get(ENV, "BUBBLE_RX", "2000"))
rz = parse(Float64, get(ENV, "BUBBLE_RZ", "2000"))
xc = parse(Float64, get(ENV, "BUBBLE_X", string(LX / 2)))
zc = parse(Float64, get(ENV, "BUBBLE_Z", "2000"))
T = length(ARGS) > 1 && ARGS[2] != "-" ? parse(Float64, ARGS[2]) : 120.0
table, nsub_default, fast_default = default_scheme(disc)
NSUB = parse(Int, get(ENV, "NSUB", string(nsub_default)))
FAST = length(ARGS) > 2 ? Symbol(ARGS[3]) : fast_default
eps_if = parse(Float64, get(ENV, "EPS_IF", "0"))
0 <= zc <= H || error("BUBBLE_Z must lie between 0 and H")

gas = Gas()
ρr, Θr, θr = E2.stratified_reference(gas; θ0 = 300.0, N = N)
model = build_model(disc; L = LX, H = H, P, NEX, NEZ, ρr, Θr, gas)
q0 = project2d(model, (x, z) -> begin
    θ = θr(z) + Δθ * exp(-((x - xc) / rx)^2 - ((z - zc) / rz)^2)
    (Θr(z) / θ, U * Θr(z) / θ, 0.0, Θr(z))
end)
Δt = T / ceil(T / (NSUB * fast_step(model, q0, FAST)))
name = "intro_bubble_$(disc)_$(FAST)"
@printf("warm bubble: %s, %.1f × %.1f km, Δθ = %.2f K, T = %.1f s\n",
        grid_info(model), LX / 1e3, H / 1e3, Δθ, T)
run_with_output(model, q0; name, Δt, T, nsub = NSUB, table, fast = FAST,
                 fast_options = (; eps_if), nout = 10, plot_names = ["theta_prime", "w"])
