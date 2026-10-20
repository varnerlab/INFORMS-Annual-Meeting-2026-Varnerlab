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
    @test length(g1.ran) == 2
    @test g1.errors == [study_hash(bad)]
    @test isfile(joinpath(out, "runs", study_hash(bad) * ".error.txt"))
    g2 = @test_logs (:warn, r"run failed") match_mode=:any run_study_grid(md, ctx.calib, [r1, r2, bad]; outdir=out)
    @test sort(g2.skipped) == sort(study_hash.([r1, r2]))
    df = rebuild_study_summary(out, "t")
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
