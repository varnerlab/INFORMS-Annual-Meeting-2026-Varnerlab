using Test, Dates, Random, Statistics, LinearAlgebra
using VLINFORMS2026
include("helpers.jl")

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

@testset "shrinkage_loss / choose_shrinkage" begin
    grid = 0:0.1:1; s, e = Date(2015,1,2), Date(2018,12,31)
    L8 = shrinkage_loss(synthetic_md(ρ=0.8), s, e)
    @test grid[argmin(L8)] ≤ 0.3 && (L8[end] - L8[1]) / L8[end] > 0.2
    L0 = shrinkage_loss(synthetic_md(ρ=0.0), s, e)
    @test (L0[end] - minimum(L0)) / L0[end] < 0.02
    @test shrinkage_loss(perturb_after(synthetic_md(ρ=0.8), e), s, e) == L8   # look-ahead canary
    @test choose_shrinkage(synthetic_md(ρ=0.8), s, e) == grid[argmin(L8)]
end
