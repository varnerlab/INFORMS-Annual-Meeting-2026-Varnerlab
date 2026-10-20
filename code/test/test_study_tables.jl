using Test, Dates, DataFrames, VLINFORMS2026
include("helpers.jl")
@testset "comparison tables" begin
    df = DataFrame(allocator=fill("CD", 7), screen=[:bandit, :bandit, :random, :random, :gamma, :bandit, :random],
        hedge=[true, true, true, true, true, false, false], repick_on_band_exit=trues(7), seed=[1, 2, 1, 2, 5660, 1, 1],
        sharpe=[0.9, 0.2, 0.5, 0.4, 0.45, 0.7, 0.6], max_drawdown=[0.1, 0.2, 0.15, 0.12, 0.11, 0.2, 0.25],
        alpha=zeros(7), alpha_t=zeros(7), realized_growth=zeros(7))
    w = paired_win_rate(df; metric=:sharpe)
    row = only(eachrow(w[w.hedge, :]))
    @test row.n_pairs == 2 && row.win_rate == 0.5 && row.median_diff ≈ 0.1      # +0.4, −0.2
    @test only(gamma_percentile(df; metric=:sharpe)[!, :percentile]) == 0.5          # 0.45 between 0.4 and 0.5
    h = hedge_effect(df; metric=:max_drawdown, higher_is_better=false)
    @test only(h[h.screen .== :bandit, :win_rate]) == 1.0                             # 0.1 < 0.2 at seed 1
    g = group_summary(df)
    @test g[(g.screen .== :bandit) .& g.hedge, :sharpe_median] == [0.55]
    md = bt_md(); pr = passive_references(md, md.dates[1], md.dates[end])
    @test pr.name == ["SPY buy-and-hold", "equal-weight universe"]
    @test pr.realized_growth[1] ≈ log(md.spy_close[end] / md.spy_close[1]) / ((length(md.dates) - 1) / 252)
    # CSV-style String screens and the long-vs-short pairing
    d2 = DataFrame(allocator=["CD", "LS", "CD", "LS"], screen=fill("bandit", 4), hedge=[true, false, true, false],
        repick_on_band_exit=trues(4), seed=[1, 1, 2, 2], sharpe=[0.5, 0.3, 0.2, 0.4], status=fill("ok", 4))
    l = long_vs_short_hedge(d2; metric=:sharpe)
    @test only(l.screen) == :bandit && only(l.n_pairs) == 2 && only(l.win_rate) == 0.5 && only(l.median_diff) ≈ 0.0
end
