using Test, Dates, DataFrames, CSV, Statistics, VLINFORMS2026
include("helpers.jl")
@testset "study runner" begin
    s = MyStudySettings(); grid = (n_seeds=30, base_seed=5660, allocators=["CD", "CES", "SG", "MinVar", "Tangent", "LS"],
        screens=[:bandit, :gamma, :random], hedge=[true, false])
    runs = tier1_runs(s, grid)
    @test length(runs) == 732
    @test count(r -> r.screen == :gamma, runs) == 12
    @test length(unique(study_hash.(runs))) == 732
    fx = fixed_schedule_runs(s, grid)
    @test length(fx) == 366
    @test all(r -> r.settings.hedge && !r.settings.repick_on_band_exit, fx)
    a = runs[1]; b = build(MyStudyRun, (settings=a.settings, allocator=a.allocator, screen=a.screen, seed=a.seed, label="other"))
    @test study_hash(a) == study_hash(b)

    md, ctx = study_ctx()
    r1 = study_run(ctx, "CD", :gamma); r2 = study_run(ctx, "MinVar", :random; seed=3)
    bad = build(MyStudyRun, (settings=with_settings(ctx.settings; bandit_lookback=10_000), allocator="CD", screen=:bandit, seed=1))
    out = mktempdir()
    g1 = @test_logs (:warn, r"run failed") match_mode=:any run_study_grid(md, ctx.calib, [r1, r2, bad]; outdir=out)
    fp = g1.fingerprint
    @test fp == data_fingerprint(md, ctx.calib)
    @test length(g1.ran) == 2
    @test g1.errors == [study_hash(bad; fingerprint=fp)]
    @test isfile(joinpath(out, "runs", study_hash(bad; fingerprint=fp) * ".error.txt"))
    g2 = @test_logs (:warn, r"run failed") match_mode=:any run_study_grid(md, ctx.calib, [r1, r2, bad]; outdir=out)
    @test sort(g2.skipped) == sort(study_hash.([r1, r2]; fingerprint=fp))
    g3 = run_study_grid(md, ctx.calib, [r1, r2]; outdir=out, fresh=true)          # --fresh ignores the cache
    @test isempty(g3.skipped) && sort(g3.ran) == sort(study_hash.([r1, r2]; fingerprint=fp))
    df = rebuild_study_summary(out, "t")
    @test all(c -> c in names(df), ["mean_eff_names", "n_failed", "n_fallback", "n_almost_optimal", "g_bull", "spy_g_bear"])
    @test nrow(df) == 3
    @test isfile(joinpath(out, "t_summary.csv"))
    @test all(c -> c in names(df), ["allocator", "screen", "seed", "hedge", "sharpe", "max_drawdown", "alpha", "n_repicks", "cap_bind_share"])
    @test !any(occursin("return", c) for c in names(df))

    res = run_study(ctx, r1); st = study_stats(res)
    @test st.n_decisions == nrow(res.decisions)
    @test st.n_repicks == nrow(res.repicks) - 1
    @test st.n_basket_changes == count(res.repicks.changed[2:end])
end

@testset "addendum run lists" begin
    s = MyStudySettings()
    g = (n_seeds=30, base_seed=5660, allocators=["CD", "CES", "SG", "MinVar", "Tangent", "LS"], screens=[:bandit, :gamma, :random],
        hedge=[true, false], K=[5, 8, 13, 21, 34],
        universe_screens=[:global_gamma, :liquid_gamma, :corr_gamma, :random_global, :random_liquid],
        fixed_lists=["l8b_13" => s.fixed_tickers, "l8b_15" => vcat(s.fixed_tickers, ["WMT", "PG"])])
    u = universe_runs(s, g); f = fixed_list_runs(s, g)
    @test length(u) == 3780 && length(f) == 24                      # 3·5·6·2 + 2·5·30·6·2; 2·6·2
    @test length(unique(study_hash.(vcat(u, f, tier1_runs(s, g))))) == 3780 + 24 + 732
    @test all(r -> r.settings.basket_K == 15, filter(r -> occursin("l8b_15", r.label), f))
    md, ctx = study_ctx(); res = run_study(ctx, study_run(ctx, "CD", :gamma))
    @test study_stats(res).basket_size == median(length.(res.repicks.basket))
end

@testset "run identity: canonical key and data fingerprint" begin
    run = build(MyStudyRun, (settings=MyStudySettings(), allocator="CD", screen=:gamma, seed=5660, label="x"))
    # the hash does not depend on whether the caller's Main has `using Dates` (a fresh process has not)
    code = "using VLINFORMS2026; print(study_hash(build(MyStudyRun, (settings=MyStudySettings(), allocator=\"CD\", " *
        "screen=:gamma, seed=5660, label=\"x\"))))"
    fresh = read(`$(Base.julia_cmd()) --startup-file=no --project=$(normpath(joinpath(@__DIR__, ".."))) -e $(code)`, String)
    @test fresh == study_hash(run)
    @test study_hash(run; fingerprint="abc") != study_hash(run)
    # the fingerprint sees the prices and the calibration
    md, ctx = study_ctx(); c = ctx.calib
    fp = data_fingerprint(md, c)
    @test fp == data_fingerprint(deepcopy(md), deepcopy(c)) && length(fp) == 40
    md2 = deepcopy(md); md2.close[end, 1] *= 1 + 1e-12
    @test data_fingerprint(md2, c) != fp
    c2 = deepcopy(c); c2.λ = 0.4
    @test data_fingerprint(md, c2) != fp
    c3 = deepcopy(c); t = first(sort(collect(keys(c3.base_sim)))); c3.base_sim[t].α += 1e-9
    @test data_fingerprint(md, c3) != fp
    # a data fix invalidates the cached runs: nothing is skipped
    out = mktempdir(); r1 = study_run(ctx, "CD", :gamma)
    @test length(run_study_grid(md, c, [r1]; outdir=out).ran) == 1
    @test length(run_study_grid(md, c, [r1]; outdir=out).skipped) == 1
    g = run_study_grid(md2, c, [r1]; outdir=out)
    @test isempty(g.skipped) && length(g.ran) == 1 && g.fingerprint == data_fingerprint(md2, c)
end

@testset "concentration and solver counts (hand-built result)" begin
    d = Date(2025, 1, 2) .+ Day.(0:3)
    dec = DataFrame(date=d[1:3], reason=[:initial, :band_exit, :cash_revisit],
        status=[:optimal, :solver_failed, :optimal], n_names=[2, 1, 0], invested_frac=[0.9, 0.5, 0.005],
        μ_T=zeros(3), σ_T=zeros(3), beta_target=zeros(3), beta_max=fill(Inf, 3), cap_binds=falses(3),
        hedge_weight=zeros(3), solver=[:clarabel_almost, :failed, :failed])
    targets = Dict(d[1] => Dict("A" => 0.5, "B" => 0.5, "C" => 0.0005, "D" => -0.1),   # C below 0.1%, D short
        d[2] => Dict("A" => 0.5), d[3] => Dict("A" => 0.005))                           # d[3]: long 0.005 < 1%
    st = study_stats(hand_result(dates=collect(d), decisions=dec, targets=targets))
    @test st.mean_eff_names ≈ 1.5 && st.n_eff_skipped == 1            # decisions 1 and 2: 2 names and 1
    @test st.n_failed == 1                                             # the :solver_failed decision
    @test st.n_fallback == 2      # decision 1 (ALMOST_OPTIMAL) and decision 3 (failed hedge sub-solve, Review Focus 4)
    @test st.n_almost_optimal == 1
end

@testset "market-state growth: day t labeled by ξ on day t−1" begin
    d = collect(Date(2025, 1, 2) .+ Day.(0:4))
    res = hand_result(dates=d, equity=[100.0, 101.0, 100.0, 102.0, 102.0], xi=[-0.5, 0.3, 0.0, -0.2, 0.9],
        decisions=DataFrame(date=Date[], reason=Symbol[], status=Symbol[], n_names=Int[], invested_frac=Float64[],
            μ_T=Float64[], σ_T=Float64[], beta_target=Float64[], beta_max=Float64[], cap_binds=Bool[],
            hedge_weight=Float64[], solver=Symbol[]))
    spy = [10.0, 10.0, 11.0, 11.0, 10.0]
    g(a, b) = log(b / a) * 252
    r = regime_stats(res, spy)                    # days 2, 5 bullish; day 3 bearish; day 4 dropped (ξ = 0)
    @test r.n_bull == 2 && r.n_bear == 1
    @test r.g_bull ≈ (g(100, 101) + g(102, 102)) / 2 && r.g_bear ≈ g(101, 100)
    @test r.spy_g_bull ≈ (g(10, 10) + g(11, 10)) / 2 && r.spy_g_bear ≈ g(10, 11)
    res.daily.xi .= -1.0
    @test isnan(regime_stats(res, spy).g_bear) && regime_stats(res, spy).n_bear == 0
end
