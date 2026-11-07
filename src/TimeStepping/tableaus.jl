# MIS tableaus

"""MIS tableau in the ClimaTimeSteppers convention: row i is paper stage i+1, column j multiplies
(Y_{j+1} - qⁿ) for α, γ and S(Y_j) for β."""
struct MISTableau
    name::Symbol
    α::Matrix{Float64}
    β::Matrix{Float64}
    γ::Matrix{Float64}
    d::Vector{Float64}
end
MISTableau(name, α, β, γ) = MISTableau(name, α, β, γ, vec(sum(β; dims = 2)))
nstages(t::MISTableau) = length(t.d)

"""
    mis_from_paper(name, α, β, γ)

Tableau from the (s+1)×(s+1) strictly lower triangular matrices of the papers (Wensch, Knoth &
Galant 2009; Knoth & Wensch 2014), where α_ij and γ_ij multiply (Y_j - qⁿ) and β_ij multiplies
S(Y_j), j < i, with Y_1 = qⁿ. The first column of α and γ multiplies Y_1 - qⁿ = 0 and is dropped.
"""
function mis_from_paper(name::Symbol, α::AbstractMatrix, β::AbstractMatrix, γ::AbstractMatrix)
    n = size(β, 1)
    size(α) == size(β) == size(γ) == (n, n) || throw(ArgumentError("α, β, γ must be square and of equal size"))
    return MISTableau(name, Matrix{Float64}(α[2:n, 2:n]), Matrix{Float64}(β[2:n, 1:n-1]),
                      Matrix{Float64}(γ[2:n, 2:n]))
end

"""
    mis_kw(name, A, b)

MIS form with recursive stages (α_{i,i-1} = 1, γ = 0) of the explicit RK method (A, b): stage i
starts from Y_{i-1} and integrates over (c_i - c_{i-1}) Δt with the forcing Σ_j (a_ij - a_{i-1,j}) S_j.
"""
function mis_kw(name::Symbol, A::AbstractMatrix, b::AbstractVector)
    s = length(b)
    Ae = zeros(s + 1, s + 1); Ae[1:s, 1:s] = A; Ae[s+1, 1:s] = b
    α = zeros(s + 1, s + 1); β = zeros(s + 1, s + 1)
    for i in 2:s+1
        β[i, :] = Ae[i, :] - Ae[i-1, :]
        α[i, i-1] = 1
    end
    return mis_from_paper(name, α, β, zeros(s + 1, s + 1))
end

"""Tables: :MIS2, :MIS3C, :MIS4, :MIS4a (Knoth & Wensch 2014, values as in ClimaTimeSteppers.jl,
MIS4a with the authors' corrected β₅₁), :WSRK3 (Wicker–Skamarock RK3: β = diag(1/3, 1/2, 1)),
:MIS_KW3 (Knoth–Wolke RK3 in MIS form, c = 0, 1/3, 3/4), :MIS54 (5 stages, Bauer & Knoth,
J. Comput. Appl. Math. 387 (2021), Table 2)."""
function mis_tableau(name::Symbol)
    name == :WSRK3 && return MISTableau(name, zeros(3, 3), [1/3 0 0; 0 1/2 0; 0 0 1.0], zeros(3, 3))
    name == :MIS2 && return MISTableau(name,
        [0 0 0; 0.536946566710 0 0; 0.480892968551 0.500561163566 0],
        [0.126848494553 0 0; -0.784838278826 1.37442675268 0; -0.0456727081749 -0.00875082271190 0.524775788629],
        [0 0 0; 0.652465126004 0 0; -0.0732769849457 0.144902430420 0])
    name == :MIS3C && return MISTableau(name,
        [0 0 0; 0.589557277145 0 0; 0.544036601551 0.565511042564 0],
        [0.397525189225 0 0; -0.227036463644 0.624528794618 0; -0.00295238076840 -0.270971764284 0.671323159437],
        [0 0 0; 0.142798786398 0 0; -0.0428918957402 0.0202720980282 0])
    name == :MIS4 && return MISTableau(name,
        [0 0 0 0; 0.914092810304 0 0 0; 1.14274417397 -0.295211246188 0 0;
         0.112965282231 0.337369411296 0.503747183119 0],
        [0.136296478423 0 0 0; 0.280462398979 -0.0160351333596 0 0;
         0.904713355208 -1.04011183154 0.652337563489 0;
         0.0671969845546 -0.365621862610 -0.154861470835 0.970362444469],
        [0 0 0 0; 0.678951983291 0 0 0; -1.38974164070 0.503864576302 0 0;
         -0.375328608282 0.320925021109 -0.158259688945 0])
    name == :MIS4a && return MISTableau(name,
        [0 0 0 0; 0.52349249922385610 0 0 0; 1.1683374366893629 -0.75762080241712637 0 0;
         -0.036477233846797109 0.56936148730740477 0.47746263002599681 0],
        [0.38758444641450318 0 0 0; -0.025318448354142823 0.38668943087310403 0 0;
         0.20899983523553325 -0.45856648476371231 0.43423187573425748 0;
         -0.10048822195663100 -0.46186171956333327 0.83045062122462809 0.27014914900250392],
        [0 0 0 0; 0.13145089796226542 0 0 0; -0.36855857648747881 0.33159232636600550 0 0;
         -0.065767130537473045 0.040591093109036858 0.064902111640806712 0])
    name == :MIS_KW3 && return mis_kw(name, [0 0 0; 1/3 0 0; -3/16 15/16 0], [1/6, 3/10, 8/15])
    if name == :MIS54
        α = zeros(6, 6); β = zeros(6, 6); γ = zeros(6, 6)
        α[2,1] = -0.056843003311023; γ[2,1] =  0.168489083931286; β[2,1] =  0.219579314792533
        α[3,1] =  0.071035715986068; γ[3,1] = -0.025097850341834; β[3,1] = -0.032864918414060
        α[3,2] =  0.050143439731979; γ[3,2] =  0.025515704040468; β[3,2] =  0.634699918767414
        α[4,1] =  0.021491523917140; γ[4,1] =  0.106139356407192; β[4,1] = -0.241761887431829
        α[4,2] =  0.287530720188756; γ[4,2] =  0.264445452990869; β[4,2] = -0.120631540663984
        α[4,3] =  0.239030810792355; γ[4,3] =  0.402246482358727; β[4,3] =  0.374686620841487
        α[5,1] =  0.027558616966568; γ[5,1] = -0.031464053194458; β[5,1] = -0.058474324094343
        α[5,2] =  0.382675659910308; γ[5,2] = -0.068258296801680; β[5,2] =  0.351217252190521
        α[5,3] =  0.177185696263246; γ[5,3] =  0.027558616966568; β[5,3] =  0.309657030167295
        α[5,4] = -0.314894383613333; γ[5,4] =  0.015830368641068; β[5,4] =  0.168604799122988
        α[6,1] =  0.065158401284120; γ[6,1] =  0.150547662349659; β[6,1] = -0.056205055946158
        α[6,2] =  0.079591607322196; γ[6,2] =  0.088610905686011; β[6,2] = -0.068390330952311
        α[6,3] =  0.459806401597571; γ[6,3] =  0.067880982803316; β[6,3] = -0.086209210260269
        α[6,4] =  0.086725275506356; γ[6,4] = -0.297416190393485; β[6,4] =  0.034904705602768
        α[6,5] =  0.439945196292364; γ[6,5] =  0.148246909195494; β[6,5] =  0.448964988009822
        return mis_from_paper(name, α, β, γ)
    end
    error("unknown MIS tableau $name")
end
