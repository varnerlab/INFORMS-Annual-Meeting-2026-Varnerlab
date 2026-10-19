using Test, Dates, Random, Statistics
using VLINFORMS2026

@testset "SIM / EWLS / market state" begin
    gm = randn(MersenneTwister(5660), 2000) .* 3; gi = 0.5 .+ 1.2 .* gm .+ randn(MersenneTwister(1), 2000)
    @test estimate_sim(gm, gi).β ≈ 1.2 atol=0.05
    s = ewls_init(0.1, 1.1, 2.0); @test (s.α, s.β, s.σ_ε) == (0.1, 1.1, 2.0)
    s = ewls_init(0.0, 0.0, 1.0; half_life=1e12, prior_weight=1e-12)
    foreach(t -> ewls_update!(s, gi[t], gm[t]), 1:2000)
    @test s.β ≈ estimate_sim(gm, gi).β rtol=1e-6
    v = ewvar_init(2.7); @test ewvar_update!(v, 0.0) < 2.7
    v = ewvar_init(1.0; half_life=1e12, prior_weight=1e-12); foreach(x -> ewvar_update!(v, x), gm)
    @test ewvar_update!(v, gm[end]) ≈ std(vcat(gm, gm[end]); corrected=false) rtol=1e-3
    @test preference_weights([0.2], [0.5], 1.0, 0.0) ≈ [tanh(0.2 + 0.5)]
    @test preference_weights([0.2], [0.5], 1.0, 1.0) ≈ [tanh(0.2/0.5 + 1.0)]
    dates = collect(Date(2025,1,1):Day(1):Date(2025,6,30)); spy = 100 .* exp.(cumsum(0.001 .+ 0.01 .* randn(MersenneTwister(2), length(dates))))
    d = Date(2025,5,1); spy2 = copy(spy); spy2[findfirst(==(d), dates)+1:end] .*= 2
    @test market_state(spy, dates, d) == market_state(spy2, dates, d)
    @test_throws ArgumentError market_state(spy, dates, Date(2024,1,1))
    @test_throws ArgumentError market_state(spy, dates, dates[1])
    i = findfirst(==(d), dates)
    @test market_state(spy, dates, d).g_m ≈ 252 * log(spy[i] / spy[i-1])

    @testset "fit_base_sim recovers betas" begin
        n = 300; dts = collect(Date(2024,1,1):Day(1):Date(2024,1,1)+Day(n-1))
        spy_c = 100 .* exp.(cumsum(0.01 .* randn(MersenneTwister(3), n)))
        gmm = diff(log.(spy_c))
        cl = hcat(50 .* exp.(cumsum(vcat(0.0, 1.5 .* gmm .+ 0.001 .* randn(MersenneTwister(4), n-1)))),
                  50 .* exp.(cumsum(vcat(0.0, 0.5 .* gmm .+ 0.001 .* randn(MersenneTwister(5), n-1)))))
        md = build(MyMarketData, (dates = dts, tickers = ["A","B"], open = cl, close = cl,
            volume = ones(n, 2), spy_open = spy_c, spy_close = spy_c, sector = ["X","X"], g_f = zeros(n)))
        fit = fit_base_sim(md, dts[11], dts[250])
        @test fit.params["A"].β ≈ 1.5 atol=0.02
        @test fit.params["B"].β ≈ 0.5 atol=0.02
        @test fit.σ_m ≈ std(growth_rates(spy_c[11:250]))
    end
end
