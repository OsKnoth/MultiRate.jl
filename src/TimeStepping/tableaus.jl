# MIS and Lawson tableaus

"""Tableau of a split step for q' = N(q) with frozen fast part L: `MISTableau` or `LawsonTableau`.
`mis_step!` dispatches on it, so `run_mis`, the drivers and `LinearAnalysis` accept both."""
abstract type SplitTableau end

"""MIS tableau in the ClimaTimeSteppers convention: row i is paper stage i+1, column j multiplies
(Y_{j+1} - qⁿ) for α, γ and S(Y_j) for β."""
struct MISTableau <: SplitTableau
    name::Symbol
    α::Matrix{Float64}
    β::Matrix{Float64}
    γ::Matrix{Float64}
    d::Vector{Float64}
end
MISTableau(name, α, β, γ) = MISTableau(name, α, β, γ, vec(sum(β; dims = 2)))
nstages(t::MISTableau) = length(t.d)

"""
    LawsonTableau(name, A, b)

Lawson (integrating-factor) form of the explicit RK method (A, b, c = A·1) for q' = N(q) with the
fast part L frozen at qⁿ and S = N - L: with the fast flow φ_τ (Z' = L Z, including the constant
forcing of L),
    Y_i = φ_{c_i Δt}(qⁿ) + Δt Σ_j a_ij e^{(c_i - c_j) Δt L} S(Y_j),
    qⁿ⁺¹ = φ_Δt(qⁿ) + Δt Σ_j b_j e^{(1 - c_j) Δt L} S(Y_j).
Each sum is evaluated by propagating through the distinct nodes in increasing order, adding the
terms of the stages at each node; φ_τ is the fast integrator of the MIS step (`fast_solve!` with
no slow forcing). Needs c_j ≤ c_i whenever a_ij ≠ 0.
"""
struct LawsonTableau <: SplitTableau
    name::Symbol
    A::Matrix{Float64}
    b::Vector{Float64}
    c::Vector{Float64}
    nodes::Vector{Float64}          # distinct nodes, increasing, nodes[1] = 0
    function LawsonTableau(name::Symbol, A::AbstractMatrix, b::AbstractVector)
        c = vec(sum(A; dims = 2))
        s = length(b)
        all(A[i, j] == 0 || c[j] <= c[i] for i in 1:s, j in 1:s) ||
            throw(ArgumentError("Lawson form needs c_j ≤ c_i for a_ij ≠ 0"))
        nodes = sort(unique(vcat(0.0, c)))
        new(name, Matrix{Float64}(A), Vector{Float64}(b), c, nodes)
    end
end
nstages(t::LawsonTableau) = length(t.b)

"""Fast integration time per large step (in units of Δt): Σ_i d_i for MIS; for Lawson the
propagations of the stages, where φ_{cΔt}(qⁿ) is computed once per node."""
fast_time(t::MISTableau) = sum(t.d)
function fast_time(t::LawsonTableau)
    st = [lawson_start(t, t.A[i, :], t.c[i]) for i in 2:nstages(t)]
    sf = lawson_start(t, t.b, 1.0)
    return maximum(vcat(st, sf)) + sum(t.c[2:end] .- st) + (1 - sf)
end

"""Node from which a Lawson stage with weights w and node c starts: the smallest c_j with w_j ≠ 0
(the stage begins with the cached φ_{c_j Δt}(qⁿ)), or c itself if all w_j vanish."""
lawson_start(t::LawsonTableau, w, c) = (js = findall(!iszero, w); isempty(js) ? c : minimum(t.c[js]))

"""
    ETDTableau(name, c, a)

Exponential RK (ETD) method in the stage-ODE form of the MATLAB code (`SplitExplicitETD.m`,
`MethodETD.m`): stage i (i = 1 … s, the last one gives qⁿ⁺¹, c[s] = 1) starts from qⁿ and
integrates the fast system with a slow forcing that is a polynomial in the local time,
    Z' = L Z + Σ_{j ≤ i} Σ_k P[i, j, k] τ^{k-1} S(Y_j),   τ = t/T ∈ [0, 1],   T = c_i Δt,
so that with exact fast stages Y_{i+1} = e^{TL} qⁿ + Δt Σ_j Σ_k a[i, j, k] φ_k(TL) S(Y_j).
`a[i, j, k]` are the φ_k coefficients (MATLAB `Koeff`), P = a/((k-1)! c_i). S(Y_j) uses the
frozen L of qⁿ, Y_1 = qⁿ.
"""
struct ETDTableau <: SplitTableau
    name::Symbol
    c::Vector{Float64}              # nodes of the stages 2 … s+1 (c[end] = 1)
    a::Array{Float64,3}             # φ_k coefficients a[i, j, k]
    P::Array{Float64,3}             # polynomial coefficients of the forcing
    function ETDTableau(name::Symbol, c::AbstractVector, a::AbstractArray{<:Real,3})
        s, K = length(c), size(a, 3)
        size(a, 1) == size(a, 2) == s || throw(ArgumentError("a must be s × s × K"))
        K <= s || throw(ArgumentError("at most s φ-functions (work arrays)"))
        P = [a[i, j, k] / (factorial(k - 1) * c[i]) for i in 1:s, j in 1:s, k in 1:K]
        new(name, Vector{Float64}(c), Array{Float64,3}(a), P)
    end
end
nstages(t::ETDTableau) = length(t.c)
fast_time(t::ETDTableau) = sum(t.c)

"""ETD tableaus (φ-coefficients as in the MATLAB `MethodETD.m`): :EB1 exponential Euler, :ETD2
(Cox–Matthews ETD2RK), :EB4 (Krogstad's ETDRK4, c = 1/2, 1/2, 1, 1)."""
function etd_tableau(name::Symbol)
    if name == :EB1
        a = zeros(1, 1, 1); a[1, 1, 1] = 1
        return ETDTableau(name, [1.0], a)
    elseif name == :ETD2
        a = zeros(2, 2, 2)
        a[1, 1, 1] = 1                                    # Y2 = e^{hL}qⁿ + h φ1 S1
        a[2, 1, 1] = 1; a[2, 1, 2] = -1; a[2, 2, 2] = 1   # qⁿ⁺¹: h(φ1 - φ2) S1 + h φ2 S2
        return ETDTableau(name, [1.0, 1.0], a)
    elseif name == :EB4
        a = zeros(4, 4, 3)
        a[1, 1, 1] = 0.5
        a[2, 1, 1] = 0.5; a[2, 1, 2] = -1; a[2, 2, 2] = 1
        a[3, 1, 1] = 1; a[3, 1, 2] = -2; a[3, 3, 2] = 2
        a[4, 1, :] = [1, -3, 4]; a[4, 2, :] = [0, 2, -4]; a[4, 3, :] = [0, 2, -4]; a[4, 4, :] = [0, -1, 4]
        return ETDTableau(name, [0.5, 0.5, 1.0, 1.0], a)
    end
    error("unknown ETD tableau $name (:EB1, :ETD2, :EB4)")
end

"""Lawson tableaus: :LRK4 (classical RK4), :LRK3 (Kutta's RK3, c = 0, 1/2, 1), :LRK2 (midpoint)."""
function lawson_tableau(name::Symbol)
    name == :LRK4 && return LawsonTableau(name, [0 0 0 0; 1/2 0 0 0; 0 1/2 0 0; 0 0 1 0], [1/6, 1/3, 1/3, 1/6])
    name == :LRK3 && return LawsonTableau(name, [0 0 0; 1/2 0 0; -1 2 0], [1/6, 2/3, 1/6])
    name == :LRK2 && return LawsonTableau(name, [0 0; 1/2 0], [0.0, 1.0])
    error("unknown Lawson tableau $name (:LRK4, :LRK3, :LRK2)")
end

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
J. Comput. Appl. Math. 387 (2021), Table 2), :MISF53 (MIS54 projected onto the frozen-order-3
conditions, MethodDesign/), :MIS_FV42, :MIS_FV53 (designed for the frozen FV split), :MIS_DG53 (for the frozen DG split). :LRK4, :LRK3, :LRK2 give Lawson tableaus
(`lawson_tableau`), :EB1, :ETD2, :EB4 ETD tableaus (`etd_tableau`), :EPIRKW3a, :EPIRKW3b,
:EPIRK4 (`epirk_tableau`), :EAB1 … :EAB4 exponential Adams–Bashforth (`eab_tableau`), so the drivers can select
them by name."""
function mis_tableau(name::Symbol)
    name in (:LRK4, :LRK3, :LRK2) && return lawson_tableau(name)
    name in (:EB1, :ETD2, :EB4) && return etd_tableau(name)
    name in (:EPIRKW3a, :EPIRKW3b, :EPIRK4) && return epirk_tableau(name)
    name in (:EAB1, :EAB2, :EAB3, :EAB4) && return eab_tableau(name)
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
    # MISF53: MIS54 projected (minimum-norm, |Δx| ≈ 0.1) onto all bicoloured order conditions ≤ 3,
    # i.e. order 3 also for the frozen split N(q) = F(q, q) (MethodDesign/); additive order 3.
    # Knoth–Wensch C_adv limit ≥ 2.05 for C_sound ≤ 12 (MethodDesign/stability.jl).
    name == :MISF53 && return MISTableau(name,
        [0 0 0 0 0;
         0.051563690870730 0 0 0 0;
         0.288176208387839 0.237366040355735 0 0 0;
         0.386426181933435 0.226884120668535 -0.311883003737751 0 0;
         0.071336206140529 0.439759102553759 0.100672171718086 0.430813910589328 0],
        [0.224169820125832 0 0 0 0;
         -0.030923474116757 0.636449295829766 0 0 0;
         -0.212661135267493 -0.090680928053002 0.403342063320495 0 0;
         -0.046027730295941 0.354986906394194 0.301752054981687 0.164177159708017 0;
         -0.039824363933280 -0.062654571166931 -0.092900063675952 0.033041549509142 0.440487189534418],
        [0 0 0 0 0;
         0.026055100529186 0 0 0 0;
         0.259743921769648 0.385911008665460 0 0 0;
         -0.084812012173229 0.017527045095579 -0.004196238099816 0 0;
         0.085121165877309 0.059732128761052 -0.256253527165363 0.156391645563470 0])
    # MIS_FV42 (frozen order 2) and MIS_FV53 (frozen order 3): designed for the frozen split of the
    # flat FV model, stable on its column Fourier blocks (U = 10/20/40 m/s, C_adv ≤ 1.6 resp. 2,
    # ρ - 1 ≤ 1e-3; MethodDesign/app_stability.jl, search_app.jl). MIS_FV42 from MIS4,
    # MIS_FV53 from MISF53.
    name == :MIS_FV42 && return MISTableau(name,
        [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.939533903827697 0.000000000000000 0.000000000000000 0.000000000000000;
         0.714544756572902 -0.070082160801396 0.000000000000000 0.000000000000000;
         0.208703128247405 0.332618690409608 0.462937874164909 0.000000000000000],
        [0.100325676538648 0.000000000000000 0.000000000000000 0.000000000000000;
         0.076338118175952 0.090855340227310 0.000000000000000 0.000000000000000;
         0.882700604458283 -1.088480191507675 0.670028426682587 0.000000000000000;
         -0.067674258571199 -0.335969633007350 -0.089273326401108 1.137180639859496],
        [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.777710991038699 0.000000000000000 0.000000000000000 0.000000000000000;
         -2.087281808914354 0.472229017716966 0.000000000000000 0.000000000000000;
         -0.390834095521825 0.427739691715592 -0.212672879879356 0.000000000000000])
    name == :MIS_FV53 && return MISTableau(name,
        [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.044030774579850 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.266839689693391 0.244055370250012 0.000000000000000 0.000000000000000 0.000000000000000;
         0.435230617072292 0.252428843512367 -0.261477641628986 0.000000000000000 0.000000000000000;
         0.080887206335825 0.476180741542896 0.100363203631178 0.357390962631500 0.000000000000000],
        [0.204134136538802 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         -0.054934478315271 0.643569907663564 0.000000000000000 0.000000000000000 0.000000000000000;
         -0.169252640387369 -0.076862927492187 0.455702932593571 0.000000000000000 0.000000000000000;
         -0.055026466182087 0.360867202895980 0.278826092231427 0.144871900467681 0.000000000000000;
         -0.025601114903931 -0.074775175830476 -0.122462039113681 0.025476599648309 0.532620103384200],
        [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.024215733381041 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.250810014707105 0.368151016082687 0.000000000000000 0.000000000000000 0.000000000000000;
         -0.090138073115189 0.009276861688934 -0.025200656837333 0.000000000000000 0.000000000000000;
         0.095566927483628 0.091432615657518 -0.247303120156212 0.151081198733907 0.000000000000000])
    # MIS_DG53 (frozen order 3): designed for the frozen split of the DG model (column Fourier
    # blocks, U = 20/40 m/s, C_adv ≤ 1.2, ρ - 1 ≤ 2e-4; from MISF53).
    name == :MIS_DG53 && return MISTableau(name,
        [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.050356665774274 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.206086955177855 0.290494330240659 0.000000000000000 0.000000000000000 0.000000000000000;
         0.803758663428190 0.161844319173536 -0.260474109544308 0.000000000000000 0.000000000000000;
         0.131047546788960 0.472884953300344 0.225172645413382 0.266611702575936 0.000000000000000],
        [0.243052398948416 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.017540935080244 0.530822192762747 0.000000000000000 0.000000000000000 0.000000000000000;
         -0.153987750812032 -0.078501823216492 0.332489574028524 0.000000000000000 0.000000000000000;
         -0.185953283984562 0.493775831606032 0.238212174485881 0.204364828280765 0.000000000000000;
         -0.008173625293666 -0.059969944513749 -0.107415926612826 0.123033005205737 0.328307340541675],
        [0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.054404302113180 0.000000000000000 0.000000000000000 0.000000000000000 0.000000000000000;
         0.142925754117780 0.440470615020949 0.000000000000000 0.000000000000000 0.000000000000000;
         -0.099802377710638 -0.024861708911963 0.005252594432547 0.000000000000000 0.000000000000000;
         0.050776274182808 -0.059588348739411 -0.288721115996752 0.302343882383795 0.000000000000000])
    error("unknown MIS tableau $name")
end
