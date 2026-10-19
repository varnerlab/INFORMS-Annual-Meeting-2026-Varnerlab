using Test, Random, LinearAlgebra, Statistics
using VLINFORMS2026

@testset "project_gbm" begin
    Δt = 1/252; Σg = [0.04/Δt;;]    # 20% annual vol
    t = @elapsed p = project_gbm([10.0], [100.0], 0.0, [0.10], Σg, 0.0; T=21, N=200_000, rng=MersenneTwister(5660))
    @test t < 10
    m21 = 1000 * exp(0.10*21Δt + 0.5*0.04*21Δt); s21 = m21 * sqrt(exp(0.04*21Δt) - 1)
    @test length(p.μ) == 21 && length(p.σ) == 21
    @test isapprox(p.μ[21], m21; rtol=0.005) && isapprox(p.σ[21], s21; rtol=0.02)
    @test project_gbm([-10.0],[100.0],2000.0,[0.10],Σg,0.0; rng=MersenneTwister(1)).μ[21] < 1000.0
    a = project_gbm([1.,2.],[50.,60.],10.,[.1,.2],[9. 2; 2 16],.04; rng=MersenneTwister(7))
    b = project_gbm([1.,2.],[50.,60.],10.,[.1,.2],[9. 2; 2 16],.04; rng=MersenneTwister(7))
    @test a.μ == b.μ && a.σ == b.σ

    # cash-only: deterministic growth at g_f, zero spread
    c = project_gbm([0.0],[100.0],1000.0,[0.1],Σg,0.05; rng=MersenneTwister(2))
    @test c.μ[21] ≈ 1000*exp(0.05*21Δt) && c.σ[21] < 1e-6

    # growth-rate reproduction (spec 6a): one-step growth mean ≈ μ_g, covariance ≈ Σ_g
    Σ2 = [9. 2; 2 16]; μ2 = [0.1, 0.2]; N = 400_000
    g = VLINFORMS2026.simulate_growth_step(μ2, Σ2, N, MersenneTwister(11))
    @test vec(mean(g; dims=2)) ≈ μ2 atol=0.02
    @test cov(g; dims=2) ≈ Σ2 rtol=0.02
end
