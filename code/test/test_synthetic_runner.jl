using Test, Dates, VLINFORMS2026, CSV
include("helpers.jl")

@testset "market path stats" begin
    gen = toy_generator(); st = market_path_stats(gen, 1:5; window=(gen.dates[1001], gen.dates[1400]), train=(gen.dates[1], gen.dates[1000]))
    md = synthetic_market(gen, 3).md; i0, i1 = 1001, 1400
    S = md.spy_close[i0:i1]
    @test st.window_max_drawdown[3] ≈ maximum(1 .- S ./ accumulate(max, S))
    @test st.train_growth[3] ≈ log(md.spy_close[1000] / md.spy_close[1]) / (999 / 252)
    @test !any(st.has_burst)                                     # toy market is untuned: ϵ = 0
end
@testset "path sets" begin
    df = DataFrame(path_id=1:1000, window_max_drawdown=rand(Xoshiro(2), 1000), has_burst=(1:1000) .% 10 .== 0)
    ps = path_sets(df; n_burst=5, n_bear=5, n_typical=10)
    b = df[df.has_burst, :]; @test sort(ps.path_id[ps.burst_stress]) == sort(b.path_id[sortperm(b.window_max_drawdown; rev=true)[1:5]])
    @test count(ps.bear_stress) == 5 && count(ps.typical) == 10 && !any(ps.bear_stress .& ps.has_burst)
    ro = collect(skipmissing(ps.run_order)); @test sort(ro) == 1:length(ro)
    first4 = ps[coalesce.(ps.run_order, 10^9) .<= 4, :]
    @test any(first4.burst_stress) && any(first4.bear_stress) && any(first4.typical)
    @test_throws ArgumentError path_sets(df; n_burst=500)
end
@testset "synthetic run list" begin
    s, grid = load_study(joinpath(@__DIR__, "..", "..", "config", "study.toml"))
    runs = synthetic_runs(s, grid)
    @test length(runs) == 532 && allunique(r.label for r in runs)
    @test all(r.seed in 5660:5664 for r in runs)
    @test count(r -> r.settings.basket_K in (5, 34) && r.screen in (:global_gamma, :random_global), runs) == 168
end
@testset "path rows match the cached runner" begin
    md, ctx = study_ctx()
    runs = [study_run(ctx, "CD", :gamma), study_run(ctx, "CD", :random; seed=5661)]
    rows = run_synthetic_path(md, ctx.calib, runs)
    tmp = mktempdir()
    run_study_grid(md, ctx.calib, runs; outdir=tmp)
    ref = rebuild_study_summary(tmp, "t")
    @test names(rows) == vcat(names(ref), ["error"])
    @test nrow(rows) == 2 && all(rows.status .== "ok") && all(ismissing, rows.error) && issorted(rows.hash)
    differ = [c for c in names(ref) if c != "wall_seconds" && !isequal(rows[!, c], ref[!, c])]
    @test isempty(differ)
end
@testset "error rows and resumable writes" begin
    md, ctx = study_ctx()
    bad = build(MyStudyRun, (settings=with_settings(ctx.settings; fixed_tickers=["T01", "NOPE"]), allocator="CD",
        screen=:fixed, seed=5660, label="CD|fixed:nope"))
    rows = run_synthetic_path(md, ctx.calib, [study_run(ctx, "CD", :gamma), bad])
    b = rows[rows.label .== "CD|fixed:nope", :]
    @test nrow(b) == 1 && b.status[1] == "error" && occursin("NOPE", b.error[1])
    @test ismissing(b.terminal[1]) && ismissing(b.allocator[1]) && ismissing(b.wall_seconds[1])
    @test all(rows.status[rows.label .!= "CD|fixed:nope"] .== "ok")
    tmp = mktempdir()
    touch(joinpath(tmp, "path_4.csv.tmp"))                      # a write that stopped before its move
    @test !path_done(tmp, 4)
    bench = passive_references(md, ctx.settings.start, ctx.settings.stop)
    meta = DataFrame(path_id=[4], n_errors=[1])
    write_path_files(tmp, 4, rows, bench, meta)
    @test path_done(tmp, 4)
    @test sort(readdir(tmp)) == ["bench_4.csv", "meta_4.csv", "path_4.csv"]
    @test nrow(CSV.read(joinpath(tmp, "path_4.csv"), DataFrame)) == 2
    @test CSV.read(joinpath(tmp, "meta_4.csv"), DataFrame).n_errors == [1]
end
