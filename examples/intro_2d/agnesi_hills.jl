# Agnesi hills for the split-explicit MIS introduction.
# Run one of the three scales: a1, a10, a100 (half-widths 1, 10, 100 km).
# The channel is 40 a wide; the default 50 elements keep nodal Δx = 0.2 a at P = 4.
# HEVI is the default. `hevi_relax` compares HEVI with relaxed horizontal DG face momentum;
# `fb` demonstrates the separate forward-backward relaxed update. `fvc` selects cut-cell FV.
#
#     julia --project=examples examples/intro_2d/agnesi_hills.jl a1  [T|-] [fast] [dg|fv|fvc]
#     julia --project=examples examples/intro_2d/agnesi_hills.jl a10 [T|-] [fast] [dg|fv|fvc]
#     julia --project=examples examples/intro_2d/agnesi_hills.jl a100 [T|-] [fast] [dg|fv|fvc]
#
# Set NEX, NEZ, P, LX, LFACTOR, EPS_IF, H0 and VADV (auto|0|1) in the environment to change the run.

include(joinpath(@__DIR__, "../euler2d_common.jl"))

case = isempty(ARGS) ? "a10" : ARGS[1]
a = get(Dict("a1" => 1e3, "a10" => 10e3, "a100" => 100e3), case) do
    error("case must be a1, a10 or a100")
end
U, NBV, θ0 = parse(Float64, get(ENV, "U0", "10")), 0.01, 300.0
disc = length(ARGS) > 3 ? Symbol(ARGS[4]) : :dg
disc in (:dg, :fv, :fvc) || error("discretization must be dg, fv or fvc")
h0 = parse(Float64, get(ENV, "H0", "400"))
P = parse(Int, get(ENV, "P", "4"))
L = parse(Float64, get(ENV, "LX", string(parse(Float64, get(ENV, "LFACTOR", "40")) * a)))
NEX = parse(Int, get(ENV, "NEX", "50"))
H = parse(Float64, get(ENV, "H", "15600"))
NEZ = parse(Int, get(ENV, "NEZ", "16"))
ZD = parse(Float64, get(ENV, "ZD", "9000"))
T = length(ARGS) > 1 && ARGS[2] != "-" ? parse(Float64, ARGS[2]) : 21.6a / max(U, 10.0)
table, nsub_default, fast_default = default_scheme(disc)
FAST = length(ARGS) > 2 ? Symbol(ARGS[3]) : fast_default
NSUB = parse(Int, get(ENV, "NSUB", string(nsub_default)))
eps_if = parse(Float64, get(ENV, "EPS_IF", "0"))
if disc == :dg
    FAST in (:hevi, :hevi_verlet, :hevi_relax, :fb, :verlet) || error("DG hill fast method must be hevi, hevi_verlet, hevi_relax, fb or verlet")
elseif disc == :fvc
    FAST in (:hevi, :fb, :verlet, :rosenbrock_amf) || error("cut-cell FV hill fast method must be hevi, fb, verlet or rosenbrock_amf")
else
    FAST in (:hevi, :wrf, :fb, :verlet) || error("FV hill fast method must be hevi, wrf, fb or verlet")
end

gas = Gas()
ρr, Θr, _ = E2.stratified_reference(gas; θ0 = θ0, N = NBV)
xm = L / 2
zs(x) = h0 / (1 + ((x - xm) / a)^2)
model0 = build_model(disc; L, H, P, NEX, NEZ, ρr, Θr, gas, zs)
q0 = project2d(model0, (x, z) -> (ρr(z), ρr(z) * U, 0.0, Θr(z)))
α = 5U * min(1.0, U / (NBV * a)) / (H - ZD)
model = with_sponge(model0, q0; zD = ZD, αtop = α)
Δt = T / ceil(T / (NSUB * fast_step(model, q0, FAST)))
# Vertical transport of momentum by the fast vertical velocity in L only when needed: vertical
# Courant number of the slow step U Δt/Δz (a10 ≈ 1.3 runs without, a100 ≈ 13 does not).
# VADV = 0 / 1 overrides the automatic choice.
set_vadv!(U, Δt, H / (P * NEZ))
name = "intro_agnesi_$(case)_$(round(Int, L / a))a_$(disc)_$(FAST)"
@printf("Agnesi hill %s: a = %.0f km, x domain = %.0f km (%.0f a), H = %.1f km\n",
        case, a / 1e3, L / 1e3, L / a, H / 1e3)
@printf("%s, T = %.0f s, %s, nsub = %d, fast = %s, eps_if = %.3g s\n",
        grid_info(model), T, table, NSUB, FAST, eps_if)
run_with_output(model, q0; name, Δt, T, nsub = NSUB, table, fast = FAST,
                 fast_options = (; eps_if), refreeze_every = 10, nout = 10,
                 plot_names = ["w", "theta_prime"],
                 zoom = ((xm - min(15a, L / 2), xm + min(15a, L / 2)), (0.0, 10e3)))
