using Test, Dates, VLINFORMS2026, Random, Statistics
include("helpers.jl")

@testset "drift EMA" begin
    z = randn(Xoshiro(1), 4_000_000); a = ema_rate(252)
    x = drift_path(z, 252, 2.0)
    @test x[1] == 2.0 * sqrt(a / (2 - a)) * z[1]
    @test isapprox(std(x), 2.0 * sqrt(a / (2 - a)); rtol=0.03)           # stationary SD
    @test isapprox(cor(x[1:end-252], x[253:end]), 0.5; atol=0.05)          # (1-a)^252 = 0.5
    @test drift_path(z, 63, 3.0) ≈ 3.0 .* drift_path(z, 63, 1.0)
    @test all(iszero, drift_path(z[1:100], 63, 0.0))
end
@testset "path seeds and growth windows" begin
    ids = vcat(1:200, 1_000_001:1_000_020)
    seeds = [path_seed(p, :market) for p in ids]
    append!(seeds, [path_seed(p, :sector, s) for p in ids for s in 1:11])
    append!(seeds, [path_seed(p, :name, j) for p in ids for j in 1:400])
    append!(seeds, [path_seed(p, :drift) for p in ids])
    @test allunique(seeds)
    @test_throws ArgumentError path_seed(1, :bogus)
    d = [Date(2014,1,2) + Day(i) for i in 0:20]
    @test growth_window(d, d[3], d[10]) == 3:9
    @test_throws ArgumentError growth_window(d, Date(2013,1,1), d[10])
end
