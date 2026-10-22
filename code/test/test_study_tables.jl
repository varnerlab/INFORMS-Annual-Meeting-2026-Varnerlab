using Test, Dates, DataFrames, VLINFORMS2026
include("helpers.jl")
@testset "comparison tables" begin
    df = DataFrame(allocator=fill("CD", 7), screen=[:bandit, :bandit, :random, :random, :gamma, :bandit, :random],
        hedge=[true, true, true, true, true, false, false], repick_on_band_exit=trues(7), seed=[1, 2, 1, 2, 5660, 1, 1],
        sharpe=[0.9, 0.2, 0.5, 0.4, 0.45, 0.7, 0.6], max_drawdown=[0.1, 0.2, 0.15, 0.12, 0.11, 0.2, 0.25],
        alpha=zeros(7), alpha_t=zeros(7), realized_growth=zeros(7), beta=[0.8, 0.6, 0.7, 0.9, 0.5, 1.0, 1.1],
        mean_invested=[0.9, 0.7, 0.8, 0.6, 0.5, 1.0, 0.95], cap_bind_share=[0.2, 0.4, NaN, 0.1, 0.3, NaN, NaN],
        mean_hedge_weight=[0.05, 0.01, 0.0, 0.02, 0.03, 0.0, 0.0])
    w = paired_win_rate(df; metric=:sharpe)
    row = only(eachrow(w[w.hedge, :]))
    @test row.n_pairs == 2 && row.win_rate == 0.5 && row.median_diff ≈ 0.1      # +0.4, −0.2
    @test only(gamma_percentile(df; metric=:sharpe)[!, :percentile]) == 0.5          # 0.45 between 0.4 and 0.5
    h = hedge_effect(df; metric=:max_drawdown, higher_is_better=false)
    @test only(h[h.screen .== :bandit, :win_rate]) == 1.0                             # 0.1 < 0.2 at seed 1
    g = group_summary(df)
    @test g[(g.screen .== :bandit) .& g.hedge, :sharpe_median] == [0.55]
    # the invested fraction, β and the hedge statistics sit next to every scorecard (base spec §6, §12)
    bh = g[(g.screen .== :bandit) .& g.hedge, :]
    @test bh.mean_invested_median ≈ [0.8] && bh.beta_median ≈ [0.7] && bh.cap_bind_share_median ≈ [0.3]
    @test bh.mean_hedge_weight_median ≈ [0.03]
    @test isnan(only(g[(g.screen .== :bandit) .& .!g.hedge, :cap_bind_share_median]))   # hedge off: NaN, no error
    @test only(g[(g.screen .== :random) .& g.hedge, :cap_bind_share_median]) == 0.1     # NaN rows skipped
    md = bt_md(); pr = passive_references(md, md.dates[1], md.dates[end])
    @test pr.name == ["SPY buy-and-hold", "equal-weight universe"]
    @test pr.realized_growth[1] ≈ log(md.spy_close[end] / md.spy_close[1]) / ((length(md.dates) - 1) / 252)
    # CSV-style String screens and the long-vs-short pairing
    d2 = DataFrame(allocator=["CD", "LS", "CD", "LS"], screen=fill("bandit", 4), hedge=[true, false, true, false],
        repick_on_band_exit=trues(4), seed=[1, 1, 2, 2], sharpe=[0.5, 0.3, 0.2, 0.4], status=fill("ok", 4))
    l = long_vs_short_hedge(d2; metric=:sharpe)
    @test only(l.screen) == :bandit && only(l.n_pairs) == 2 && only(l.win_rate) == 0.5 && only(l.median_diff) ≈ 0.0
    mt = markdown_table(DataFrame(a=[1, 2], b=[0.123456, 2.5]))
    ls = split(mt, "\n"; keepempty=false)
    @test ls[1] == "| a | b |" && ls[2] == "|---|---|" && ls[3] == "| 1 | 0.1235 |" && length(ls) == 4
end

@testset "matched-control percentile and summary comparison" begin
    df = DataFrame(allocator=fill("CD", 4), screen=["global_gamma", "random_global", "random_global", "random_global"],
        hedge=trues(4), basket_size=fill(13, 4), seed=[5660, 1, 2, 3], sharpe=[0.5, 0.2, 0.6, 0.4])
    p = gamma_percentile(df; screen=:global_gamma, control=:random_global, by=[:allocator, :basket_size, :hedge])
    @test only(p.percentile) ≈ 2 / 3 && only(p.gamma_value) == 0.5
    @test only(p.share_of_controls_beaten) ≈ 2 / 3                                # Sharpe: controls below
    dd = DataFrame(allocator=fill("CD", 5), screen=["global_gamma", fill("random_global", 4)...], hedge=trues(5),
        basket_size=fill(13, 5), seed=[5660, 1, 2, 3, 4], max_drawdown=[0.10, 0.05, 0.10, 0.20, 0.30])
    q = gamma_percentile(dd; metric=:max_drawdown, screen=:global_gamma, control=:random_global,
        by=[:allocator, :basket_size, :hedge], higher_is_better=false)
    @test only(q.percentile) ≈ 1.5 / 4                                          # below 0.10: one, tie half
    @test only(q.share_of_controls_beaten) ≈ 2.5 / 4                            # deeper drawdown: two, tie half
    old = DataFrame(label=["a", "b"], sharpe=[0.1, NaN]); new = DataFrame(label=["a", "b"], sharpe=[0.1, NaN])
    @test compare_summaries(old, new; cols=[:sharpe]) == (n_matched=2, n_missing=0, max_abs_diff=0.0)
end

@testset "compare_summaries fails on real differences" begin
    old = DataFrame(label=["a", "b"], sharpe=[0.1, 0.2])
    r = compare_summaries(old, DataFrame(label=["a", "b"], sharpe=[0.1, 0.25]); cols=[:sharpe])
    @test r.max_abs_diff ≈ 0.05
    r = compare_summaries(old, DataFrame(label=["a"], sharpe=[0.1]); cols=[:sharpe])
    @test r.n_missing == 1
    @test r.n_matched == 1
    r = compare_summaries(old, DataFrame(label=["a", "b"], sharpe=[0.1, missing]); cols=[:sharpe])
    @test r.max_abs_diff == Inf
    r = compare_summaries(old, DataFrame(label=["a", "b"], sharpe=[0.1, NaN]); cols=[:sharpe])
    @test r.max_abs_diff == Inf
end

@testset "tables are sorted deterministically" begin
    t = DataFrame(allocator=["LS", "CD", "CD", "CD", "CES", "CD", "CD"], screen=[:gamma, :random, :gamma, :gamma, :bandit, :zz, :gamma],
        hedge=[true, true, false, true, true, true, true], repick_on_band_exit=[true, true, true, false, true, true, true], x=1:7)
    s = sort_table(t)
    # allocator order, then screen order (unknown screens last), hedge on first, re-pick on band exit first
    @test s.x == [7, 4, 3, 2, 6, 5, 1]
    @test sort_table(DataFrame(allocator=["LS", "CESw", "CES", "CD"])).allocator == ["CD", "CES", "CESw", "LS"]
    @test sort_table(t[end:-1:1, :]) == s                                       # independent of the input order
    l = DataFrame(screen=[:random, :bandit, :bandit], repick_on_band_exit=[true, false, true], n=1:3)
    @test sort_table(l).n == [3, 2, 1]
    @test sort_table(DataFrame(name=["b", "a"])).name == ["b", "a"]            # no sort columns: unchanged
end
