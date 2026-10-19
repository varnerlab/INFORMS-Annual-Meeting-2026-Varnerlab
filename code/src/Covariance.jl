"""
    AbstractCovarianceModel

Supertype of growth-covariance models. Covariances are in yr⁻² (growth rates in yr⁻¹).
"""
abstract type AbstractCovarianceModel end

"C1: Single Index Model covariance, Σ_g = σ_m²ββᵀ + diag(σ_ε²)."
struct MySIMCovariance <: AbstractCovarianceModel end

"""
    MyShrunkResidualCovariance

C2: σ_m²ββᵀ + D^{1/2}·R_λ·D^{1/2}, R_λ = (1−λ)·cor(residuals over trailing `window` days) + λI.
`λ` (dimensionless, 0..1); `window` (trading days, 126).
"""
mutable struct MyShrunkResidualCovariance <: AbstractCovarianceModel
    λ::Float64
    window::Int
    MyShrunkResidualCovariance() = new()
end

function build(::Type{MyShrunkResidualCovariance}, data::NamedTuple)::MyShrunkResidualCovariance
    model = MyShrunkResidualCovariance()
    haskey(data, :λ) || throw(ArgumentError("Cannot build MyShrunkResidualCovariance: missing field λ"))
    model.λ = Float64(data.λ)
    model.window = Int(get(data, :window, 126))
    return model
end

"""
    growth_covariance(model, β, σ_ε, σ_m; residuals=nothing)::Matrix{Float64}

Growth covariance (yr⁻²). `β` dimensionless, `σ_ε` and `σ_m` in yr⁻¹. For C2, `residuals`
(days × names, yr⁻¹) supplies the residual correlation; the caller passes the trailing window.
"""
function growth_covariance(::MySIMCovariance, β::AbstractVector, σ_ε::AbstractVector, σ_m::Real;
        residuals::Union{Nothing,AbstractMatrix}=nothing)::Matrix{Float64}
    return Matrix{Float64}(σ_m^2 .* (β * β') .+ Diagonal(σ_ε .^ 2))
end

function growth_covariance(model::MyShrunkResidualCovariance, β::AbstractVector, σ_ε::AbstractVector, σ_m::Real;
        residuals::Union{Nothing,AbstractMatrix}=nothing)::Matrix{Float64}
    residuals === nothing && throw(ArgumentError("MyShrunkResidualCovariance requires `residuals`"))
    size(residuals, 2) == length(σ_ε) || throw(DimensionMismatch("residuals must be days × names"))
    n = length(σ_ε)
    R = (1 - model.λ) .* cor(residuals) .+ model.λ .* Matrix(I, n, n)
    R[isnan.(R)] .= 0.0                    # constant residual series: no correlation information
    R[diagind(R)] .= 1.0
    D = Diagonal(Float64.(σ_ε))
    Σ = σ_m^2 .* (β * β') .+ D * R * D
    return Matrix{Float64}((Σ + Σ') / 2)
end

"""
    residual_matrix(g, g_m, α, β)::Matrix{Float64}

εᵢₜ = gᵢₜ − αᵢ − βᵢ·g_mₜ for `g` days × names (yr⁻¹).
"""
function residual_matrix(g::AbstractMatrix, g_m::AbstractVector, α::AbstractVector, β::AbstractVector)::Matrix{Float64}
    return Matrix{Float64}(g .- α' .- g_m * β')
end

# Basket of up to `n` names, drawn round-robin across sectors (without replacement) so sectors
# are balanced: 3 per sector when there are 11 sectors.
function _draw_basket(rng, sector_idx::Vector{Vector{Int}}, n::Int)::Vector{Int}
    pools = [shuffle(rng, s) for s in sector_idx if !isempty(s)]
    basket = Int[]
    while length(basket) < n && any(!isempty, pools)
        for p in pools
            (isempty(p) || length(basket) >= n) && continue
            push!(basket, pop!(p))
        end
    end
    return basket
end

"""
    shrinkage_loss(md, start, stop; grid=0:0.1:1, n_portfolios=200, seed=5660)::Vector{Float64}

Mean QLIKE (r/p − log(r/p) − 1) per λ ∈ `grid` of the RESIDUAL-portfolio variance: predicted
p = wᵀD^{1/2}R_λD^{1/2}w (D = diag(σ_ε²) from the anchor's OLS), realized r = sample variance over
the next 21 trading days of wᵀε_t (ε from the anchor's OLS α, β). Monthly (21-trading-day) anchors
run from `start` + 252 trading days through `stop` − 21. At each anchor β, α, σ_ε come from OLS on
the trailing 252 days and the residual correlation from the trailing 126 days. Baskets: up to 33
names (round-robin across sectors), Dirichlet(1) weights. Portfolios with non-finite or
non-positive r are skipped. Uses data through `stop` only.
"""
function shrinkage_loss(md::MyMarketData, start::Date, stop::Date;
        grid=0:0.1:1, n_portfolios::Int=200, seed::Int=5660)::Vector{Float64}
    rng = MersenneTwister(seed)
    g = growth_rates(md.close); g_m = growth_rates(md.spy_close)    # g[k] is the return into dates[k+1]
    nt = size(g, 2)
    i0 = searchsortedfirst(md.dates, start); i1 = searchsortedlast(md.dates, stop)
    anchors = collect((i0 + 252):21:(i1 - 21))
    isempty(anchors) && throw(ArgumentError("window too short: need at least 252 + 21 trading days"))
    sector_idx = [findall(==(s), md.sector) for s in unique(md.sector)]
    λs = collect(Float64, grid)
    loss = zeros(length(λs)); count = 0
    for a in anchors
        tr = (a - 252):(a - 1); tr126 = (a - 126):(a - 1); fw = a:(a + 20)
        fits = [estimate_sim(g_m[tr], g[tr, j]) for j in 1:nt]
        α = [f.α for f in fits]; β = [f.β for f in fits]; σε = [f.σ_ε for f in fits]
        E = residual_matrix(g[tr126, :], g_m[tr126], α, β)
        Ef = residual_matrix(g[fw, :], g_m[fw], α, β)
        C = cor(E); C[isnan.(C)] .= 0.0; C[diagind(C)] .= 1.0
        D = Diagonal(σε)
        Σs = [D * ((1 - λ) .* C .+ λ .* Matrix(I, nt, nt)) * D for λ in λs]
        for _ in 1:n_portfolios
            idx = _draw_basket(rng, sector_idx, 33)
            w = randexp(rng, length(idx)); w ./= sum(w)       # Dirichlet(1)
            r = var(Ef[:, idx] * w)
            (isfinite(r) && r > 0) || continue
            count += 1
            for (k, Σ) in enumerate(Σs)
                q = r / dot(w, Σ[idx, idx] * w)
                loss[k] += q - log(q) - 1
            end
        end
    end
    count > 0 || throw(ArgumentError("no valid portfolios"))
    loss ./= count
    all(isfinite, loss) || error("non-finite shrinkage loss")
    return loss
end

"""
    choose_shrinkage(md, start, stop; grid=0:0.1:1, n_portfolios=200, seed=5660)::Float64

The λ ∈ `grid` with the lowest `shrinkage_loss`; ties go to the larger λ.
"""
function choose_shrinkage(md::MyMarketData, start::Date, stop::Date;
        grid=0:0.1:1, n_portfolios::Int=200, seed::Int=5660)::Float64
    L = shrinkage_loss(md, start, stop; grid=grid, n_portfolios=n_portfolios, seed=seed)
    λs = collect(Float64, grid)
    return λs[findlast(l -> l <= minimum(L), L)]
end
