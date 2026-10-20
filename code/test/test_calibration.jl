using Test, Dates, Statistics, VLINFORMS2026
include("helpers.jl")
@testset "calibration" begin
    md = synthetic_md(ρ=0.3)
    a, b = md.dates[1], md.dates[1000]
    c = calibrate(md; train_start=a, train_stop=b, n_portfolios=50)
    fit = fit_base_sim(md, a, b)
    @test all(c.base_sim[t].β == fit.params[t].β && c.base_sim[t].α == fit.params[t].α for t in md.tickers)
    @test (c.σ_m, c.m_m, c.v_m) == (fit.σ_m, fit.m_m, fit.v_m)
    @test c.λ == choose_shrinkage(md, a, b; n_portfolios=50, seed=5660)
    @test (c.train_start, c.train_stop) == (a, b)
    path = joinpath(mktempdir(), "cal.jld2"); save_calibration(path, c); d = load_calibration(path)
    @test d.λ == c.λ && d.σ_m == c.σ_m && d.base_sim[md.tickers[3]].β == c.base_sim[md.tickers[3]].β
    @test_throws ArgumentError build(MyCalibration, (σ_m=1.0,))
end
