using Test, Dates, Random, Statistics, LinearAlgebra
using VLINFORMS2026

# Synthetic market: 2014–2018 weekdays, 4 sectors × 12 names, β = 1, within-sector residual
# correlation ρ, SPY as the market factor. Growth units yr⁻¹: market SD 2.5, residual SD 4.
function synthetic_md(; ρ::Float64, seed::Int=1)
    rng = MersenneTwister(seed)
    dates = filter(d -> dayofweek(d) <= 5, collect(Date(2014,1,1):Day(1):Date(2018,12,31)))
    n = length(dates); Δt = 1/252
    sectors = ["S$(s)" for s in 1:4 for _ in 1:12]
    gm = 2.5 .* randn(rng, n - 1)
    z = randn(rng, n - 1, 4); u = randn(rng, n - 1, 48)
    E = 4.0 .* [sqrt(ρ) * z[t, parse(Int, sectors[i][2:end])] + sqrt(1 - ρ) * u[t, i] for t in 1:n-1, i in 1:48]
    g = gm .+ E
    cl = 100 .* exp.(vcat(zeros(1, 48), cumsum(g .* Δt; dims=1)))
    spy = 100 .* exp.(vcat(0.0, cumsum(gm .* Δt)))
    return build(MyMarketData, (dates=dates, tickers=["T$(lpad(i,2,'0'))" for i in 1:48], open=cl, close=cl,
        volume=ones(n, 48), spy_open=spy, spy_close=spy, sector=sectors, g_f=zeros(n)))
end

@testset "SIM growth covariance (C1, C2)" begin
    β = [1.0, 1.2, 0.8]; σε = [2.0, 3.0, 2.5]; σm = 2.7
    E = randn(MersenneTwister(5660), 500, 3) * cholesky([1 .6 0; .6 1 0; 0 0 1]).U
    Σ1 = growth_covariance(MySIMCovariance(), β, σε, σm)
    @test Σ1 ≈ σm^2 .* β * β' .+ Diagonal(σε .^ 2)
    c2(λ) = growth_covariance(build(MyShrunkResidualCovariance, (λ=λ, window=126)), β, σε, σm; residuals=E)
    @test c2(1.0) ≈ Σ1
    @test isposdef(c2(0.0)) && c2(0.0)[1,2] - Σ1[1,2] ≈ cor(E)[1,2] * σε[1] * σε[2]
    @test sqrt((1/252) * ([.5,.5,0]' * Σ1 * [.5,.5,0])) < 0.5
    @test_throws ArgumentError growth_covariance(build(MyShrunkResidualCovariance, (λ=0.5, window=126)), β, σε, σm)
    g = [1.0 2.0; 3.0 5.0]; gm = [1.0, 2.0]
    @test residual_matrix(g, gm, [0.5, 0.0], [1.0, 2.0]) ≈ [-0.5 0.0; 0.5 1.0]
end

@testset "choose_shrinkage recovers structure" begin
    t = @elapsed begin
        λ_hi = choose_shrinkage(synthetic_md(ρ=0.8), Date(2015,1,2), Date(2018,12,31))
        λ_0 = choose_shrinkage(synthetic_md(ρ=0.0), Date(2015,1,2), Date(2018,12,31))
    end
    @test λ_hi ≤ 0.3
    @test λ_0 ≥ 0.7
end
