# 2D warm bubble at constant pressure in a stably stratified atmosphere.
# Both discretizations use MIS4 with HEVI fast steps by default.
#
#     julia --project=examples examples/euler2d_bubble.jl [dg|fv] [T|-] [fast]
#
# Configure with environment variables: LX, H, NEX, NEZ, P, BUBBLE_X, BUBBLE_Z,
# BUBBLE_RX, BUBBLE_RZ, DELTA_THETA, N, U, NSUB. Outputs use the common VTKHDF/Makie path.

include(joinpath(@__DIR__, "euler2d_common.jl"))

disc = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :dg
const LX = parse(Float64, get(ENV, "LX", "20000"))
const H = parse(Float64, get(ENV, "H", "10000"))
const P = parse(Int, get(ENV, "P", "4"))
const NEX = parse(Int, get(ENV, "NEX", "10"))
const NEZ = parse(Int, get(ENV, "NEZ", "6"))
const N = parse(Float64, get(ENV, "N", "0.01"))
const U = parse(Float64, get(ENV, "U", "0"))
const Δθ = parse(Float64, get(ENV, "DELTA_THETA", "2.0"))
const rx = parse(Float64, get(ENV, "BUBBLE_RX", "2000"))
const rz = parse(Float64, get(ENV, "BUBBLE_RZ", "2000"))
const xc = parse(Float64, get(ENV, "BUBBLE_X", string(LX / 2)))
const zc = parse(Float64, get(ENV, "BUBBLE_Z", "2000"))
const T = length(ARGS) > 1 && ARGS[2] != "-" ? parse(Float64, ARGS[2]) : 120.0
const table_default, nsub_default, fast_default = default_scheme(disc)
const NSUB = parse(Int, get(ENV, "NSUB", string(nsub_default)))
const TABLE = Symbol(get(ENV, "TABLE", string(table_default)))
const FAST = length(ARGS) > 2 ? Symbol(ARGS[3]) : fast_default

all(x -> x > 0, (LX, H, P, NEX, NEZ, NSUB, rx, rz)) || error("lengths, degrees, and counts must be positive")
0 <= zc <= H || error("BUBBLE_Z must lie between 0 and H")

gas = Gas()
ρr, Θr, θr = E2.stratified_reference(gas; θ0 = 300.0, N = N)
model = build_model(disc; L = LX, H = H, P = P, NEX = NEX, NEZ = NEZ, ρr = ρr, Θr = Θr, gas = gas)

# A potential-temperature perturbation at the reference pressure: Θ remains Θ̄ and rho = Θ̄/θ.
q0 = project2d(model, (x, z) -> begin
    θ = θr(z) + Δθ * exp(-((x - xc) / rx)^2 - ((z - zc) / rz)^2)
    (Θr(z) / θ, U * Θr(z) / θ, 0.0, Θr(z))
end)
Δt = T / ceil(T / (NSUB * fast_step(model, q0, FAST)))
name = "bubble_$(disc)_L$(round(Int, LX / 1e3))km_$(NEX)x$(NEZ)_p$(P)_$(FAST)"
@printf("warm bubble (%s): domain %.1f km × %.1f km, centre (%.1f, %.1f) km, radii (%.1f, %.1f) km, Δθ = %.2f K\n",
        disc, LX / 1e3, H / 1e3, xc / 1e3, zc / 1e3, rx / 1e3, rz / 1e3, Δθ)
@printf("%s, U = %.1f m/s, N = %.3f 1/s, T = %.1f s; %s, nsub = %d\n",
        grid_info(model), U, N, T, TABLE, NSUB)
run_with_output(model, q0; name = name, Δt = Δt, T = T, nsub = NSUB, table = TABLE, fast = FAST,
                 nout = 10, plot_names = ["theta_prime", "w"])
