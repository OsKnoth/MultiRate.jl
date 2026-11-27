# Rosenbrock (W-)methods in the form of ASAM (SRC/ADVECT/IntRos_Mod.f90): per step one approximation W of the Jacobian
# at qⁿ (prepared once), stages
#     rᵢ = Δt N(Yᵢ) + Σ_{j<i} βᵢⱼ kⱼ,    (I - γ Δt W) kᵢ = rᵢ,    Yᵢ₊₁ = qⁿ + Σ_{j≤i} αᵢⱼ kⱼ,
# Y₁ = qⁿ, qⁿ⁺¹ = Y_{s+1}. W-methods keep their order for any W (here: the global acoustic matrix plus a first-order
# upwind advection matrix, possibly factorised (AMF), FV3D/rosenbrock.jl); the other methods need W = J.
#
# Model interface: `ros_prepare!(model, q, γΔt)` (freeze the coefficients and build or factorise I - γ Δt W at q),
# `ros_solve!(k, model, r)` (k = (I - γ Δt W)⁻¹ r) and `rhs_nonlinear!`.

"""Rosenbrock tableau in the ASAM form: s stages, γ, α (s × s, lower triangular incl. diagonal), β (s × s, strictly
lower); `w`: W-method (order independent of W)."""
struct RosTableau
    name::Symbol
    s::Int
    γ::Float64
    α::Matrix{Float64}
    β::Matrix{Float64}
    order::Int
    w::Bool
end

"""
    ros_tableau(name)

:RosEul (linearly implicit Euler, order 1, W), :Ros2AMF (2 stages, γ = 1/2 + √3/6, order 2, W), :Ros3P (Lang–Verwer,
order 3 with W = J, order 1 otherwise), :Ros3Pw (Rang–Angermann, order 3 with W = J, 2 as W-method), :RosRK3, :RosTVD
(3 stages), :RosRK4 (4 stages) — the coefficients of ASAM `InitRos` (checked on ODEs: RosEul 1, Ros2AMF 2, Ros3P and
Ros3Pw 3 with W = J; with W ≠ J Ros2AMF and Ros3Pw 2; RosRK3, RosTVD, RosRK4 below their nominal order in this
autonomous form).
"""
function ros_tableau(name::Symbol)
    α = zeros(4, 4); β = zeros(4, 4)
    if name === :RosEul
        s, γ, ord, w = 1, 1.0, 1, true
        α[1, 1] = 1.0
    elseif name === :Ros2AMF
        s, γ, ord, w = 2, 0.5 + sqrt(3.0) / 6, 2, true
        α[1, 1] = 2 / 3; α[2, 1] = 5 / 4; α[2, 2] = 3 / 4
        β[2, 1] = -4 / 3
    elseif name === :Ros3P
        s, γ, ord, w = 3, 7.886751345948129e-1, 3, false
        α[1, 1] = 1.267949192431123; α[2, 1] = 1.267949192431123
        α[3, 1] = 2.0; α[3, 2] = 5.773502691896258e-1; α[3, 3] = 4.226497308103742e-1
        β[2, 1] = -1.607695154586736; β[3, 1] = -3.464101615137755; β[3, 2] = -1.732050807568877
        α .*= γ; β .*= γ
    elseif name === :Ros3Pw
        s, γ, ord, w = 3, 7.88675134594812865529e-1, 3, true
        α[1, 1] = 2.0; α[2, 1] = 6.33974596215561403412e-1
        α[3, 1] = 1.63397459621556140341; α[3, 2] = 2.94228634059947813384e-1; α[3, 3] = 1.07179676972449078320
        β[2, 1] = -2.53589838486224561365; β[3, 1] = -1.62740473580835520728; β[3, 2] = -2.74519052838329002952e-1
        α .*= γ; β .*= γ
    elseif name === :RosRK3
        s, γ, ord, w = 3, 1.0, 3, false
        b = γ
        α[1, 1] = 1 / (3b); α[2, 1] = (-1 + 12b^2) / (18b^2 * (-1 + 4b)); α[2, 2] = 1 / (2b)
        α[3, 1] = (1 - 3b * (7 + 16b * (-2 + 3b))) / (36b^3 * (-1 + 4b)); α[3, 2] = (-1 + 12b) / (4b^2); α[3, 3] = 1 / b
        β[2, 1] = (1 - 12b^2) / (b^2 * (-9 + 36b)); β[3, 1] = (-1 + 3b * (7 + 16b * (-2 + 3b))) / (36b^3 * (-1 + 4b))
        β[3, 2] = (0.25 - 3b) / b^2
        α .*= γ; β .*= γ
    elseif name === :RosTVD
        s, γ, ord, w = 3, 0.5, 3, false
        b = γ
        α[1, 1] = 1 / b; α[2, 1] = (1 + b * (-9 + 16b)) / (4b^2 * (-1 + 4b)); α[2, 2] = 1 / (4b)
        α[3, 1] = (1 + b * (-3 + 2b * (-17 + 44b))) / (12b^3 * (-1 + 4b)); α[3, 2] = (1 + 6b) / (12b^2); α[3, 3] = 2 / (3b)
        β[2, 1] = b^-2 - 4 / b + 4 / (-1 + 4b); β[3, 1] = (-1 + b * (5 + 8 * (2 - 7b) * b)) / (8b^3 * (-1 + 4b))
        β[3, 2] = -(1 + 4b) / (8b^2)
        α .*= γ; β .*= γ
    elseif name === :RosRK4
        s, γ, ord, w = 4, 0.25, 4, false
        α[1, 1] = 2; α[2, 1] = 2; α[2, 2] = 2; α[3, 1] = 6; α[3, 2] = 10; α[3, 3] = 4
        α[4, 1] = 14 / 3; α[4, 2] = 20 / 3; α[4, 3] = 4 / 3; α[4, 4] = 2 / 3
        β[2, 1] = -4; β[3, 1] = -6; β[3, 2] = -10; β[4, 1] = -4; β[4, 2] = -12
        α .*= γ; β .*= γ
    else
        error("ros_tableau: :RosEul, :Ros2AMF, :Ros3P, :Ros3Pw, :RosRK3, :RosTVD or :RosRK4")
    end
    return RosTableau(name, s, γ, α[1:s, 1:s], β[1:s, 1:s], ord, w)
end

"""Prepare the stage matrix I - γΔt W at q (model interface)."""
function ros_prepare! end
"""k = (I - γΔt W)⁻¹ r with the prepared matrix (model interface)."""
function ros_solve! end

"""Work vectors of a Rosenbrock step."""
struct RosWork
    k::Vector{Vector{Float64}}
    Y::Vector{Float64}
    r::Vector{Float64}
    f::Vector{Float64}
end
RosWork(n::Int, s::Int) = RosWork([zeros(n) for _ in 1:s], zeros(n), zeros(n), zeros(n))

"""
    ros_step!(q, model, Δt, tab, w = RosWork(length(q), tab.s))

One Rosenbrock step q ← qⁿ⁺¹ (in place): W prepared once at qⁿ (`ros_prepare!`), s linear solves (`ros_solve!`).
"""
function ros_step!(q, model, Δt, tab::RosTableau, w::RosWork = RosWork(length(q), tab.s))
    ros_prepare!(model, q, tab.γ * Δt)
    w.Y .= q
    for i in 1:tab.s
        rhs_nonlinear!(w.f, model, w.Y)
        @. w.r = Δt * w.f
        for j in 1:i-1
            b = tab.β[i, j]
            b != 0 && (@. w.r += b * w.k[j])
        end
        ros_solve!(w.k[i], model, w.r)
        w.Y .= q
        for j in 1:i
            a = tab.α[i, j]
            a != 0 && (@. w.Y += a * w.k[j])
        end
    end
    q .= w.Y
    return q
end

"""nsteps Rosenbrock steps from q0; `monitor(n, q)` every `every` steps."""
function run_ros(model, q0, Δt, nsteps; tab::RosTableau = ros_tableau(:Ros3Pw), monitor = nothing, every::Int = 1)
    q = Vector{Float64}(q0)
    w = RosWork(length(q), tab.s)
    for n in 1:nsteps
        ros_step!(q, model, Δt, tab, w)
        monitor !== nothing && n % every == 0 && monitor(n, q)
    end
    return q
end
