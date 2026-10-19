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
    choose_shrinkage(md, start, stop; grid=0:0.1:1, n_portfolios=200, seed=5660)::Float64

Pick the C2 shrinkage λ ∈ `grid` minimizing mean QLIKE (r/p − log(r/p) − 1) of 21-day-ahead
realized portfolio variance r vs. predicted p = wᵀΣ_g(λ)w, over monthly (21-trading-day) anchors
from `start` + 252 trading days through `stop` − 21. At each anchor β, σ_ε come from OLS on the
trailing 252 days, σ_m is the sample SD of SPY growth over the same days, and the residual
correlation uses the trailing 126 days. Baskets: up to 33 names, Dirichlet(1) weights.
Ties go to the larger λ.
"""
function choose_shrinkage(md::MyMarketData, start::Date, stop::Date;
        grid=0:0.1:1, n_portfolios::Int=200, seed::Int=5660)::Float64
    rng = MersenneTwister(seed)
    g = growth_rates(md.close); g_m = growth_rates(md.spy_close)    # g[k] is the return into dates[k+1]
    nt = size(g, 2)
    i0 = searchsortedfirst(md.dates, start); i1 = searchsortedlast(md.dates, stop)
    anchors = collect((i0 + 252):21:(i1 - 21))
    isempty(anchors) && throw(ArgumentError("window too short: need at least 252 + 21 trading days"))
    secs = unique(md.sector)
    sector_idx = [findall(==(s), md.sector) for s in secs]
    λs = collect(Float64, grid)
    loss = zeros(length(λs)); count = 0
    for a in anchors
        tr = (a - 252):(a - 1); tr126 = (a - 126):(a - 1); fw = a:(a + 20)
        fits = [estimate_sim(g_m[tr], g[tr, j]) for j in 1:nt]
        α = [f.α for f in fits]; β = [f.β for f in fits]; σε = [f.σ_ε for f in fits]
        σm = std(g_m[tr])
        E = residual_matrix(g[tr126, :], g_m[tr126], α, β)
        Σs = [growth_covariance(build(MyShrunkResidualCovariance, (λ=λ, window=126)), β, σε, σm; residuals=E) for λ in λs]
        for _ in 1:n_portfolios
            idx = _draw_basket(rng, sector_idx, 33)
            w = -log.(rand(rng, length(idx))); w ./= sum(w)       # Dirichlet(1)
            r = var(g[fw, idx] * w)
            count += 1
            for (k, Σ) in enumerate(Σs)
                p = dot(w, Σ[idx, idx] * w); q = r / p
                loss[k] += q - log(q) - 1
            end
        end
    end
    loss ./= count
    best = minimum(loss)
    return λs[findlast(l -> l <= best, loss)]
end
