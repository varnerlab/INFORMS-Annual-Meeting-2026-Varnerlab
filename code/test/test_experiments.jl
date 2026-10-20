using Test, Dates, DataFrames, CSV, JLD2
using VLINFORMS2026
include("helpers.jl")

const _EXP_COLS = ["hash", "label", "rule", "covariance", "λ", "w_max", "σ_g_max", "s_max", "z", "T",
    "basket_kind", "seed", "start", "stop", "status", "wall_seconds",
    "sharpe", "max_drawdown", "alpha_ann", "alpha_t", "beta", "ann_return", "ann_vol",
    "turnover", "total_cost", "borrow_fees",
    "n_decisions", "n_band_exit", "n_drawdown", "n_horizon", "n_cash_revisit", "mean_names", "mean_invested"]

@testset "experiment runner" begin
    md = bt_md(); cfg = bt_cfg(md)

    @testset "hash" begin
        @test config_hash(cfg) == config_hash(bt_cfg(md))
        @test config_hash(cfg) != config_hash(with(cfg; w_max=0.5))
        @test config_hash(cfg) != config_hash(with(cfg; basket_schedule=[md.dates[21] => ["AAA", "BBB"]]))
        @test !haskey(config_key(cfg), :base_sim)
        @test config_key(cfg).basket_schedule isa String && length(config_key(cfg).basket_schedule) == 40   # SHA1
        @test config_hash(cfg) == config_hash(with(cfg; base_sim=bt_base_sim(α=[0.9, 0.9, 0.9])))   # base_sim excluded
        @test length(config_hash(cfg)) == 40
        @test config_hash(cfg) == config_hash(with(cfg; label="other"))   # R33: label is not identity
        @test !haskey(config_key(cfg), :label)
    end

    @testset "run_and_save, resume, partial tmp, summary" begin
        mktempdir() do out
            cfgs = [cfg, with(cfg; w_max=0.5, label="b"), with(cfg; covariance=MyShrunkResidualCovariance(), label="c")]
            cfgs[3].covariance = build(MyShrunkResidualCovariance, (λ=0.3,))
            r1 = run_grid(md, cfgs; outdir=out)
            @test r1.ran == sort(config_hash.(cfgs)) && isempty(r1.skipped) && isempty(r1.errors)
            runs = joinpath(out, "runs")
            for c in cfgs
                @test isfile(joinpath(runs, config_hash(c) * ".jld2"))
            end
            @test !any(endswith(f, ".tmp") for f in readdir(runs))
            mt = Dict(f => mtime(joinpath(runs, f)) for f in readdir(runs))
            r2 = run_grid(md, cfgs; outdir=out)   # nothing to do
            @test isempty(r2.ran) && r2.skipped == sort(config_hash.(cfgs))
            @test all(mtime(joinpath(runs, f)) == mt[f] for f in readdir(runs))

            # partial tmp is rerun
            h = config_hash(cfgs[2]); f = joinpath(runs, h * ".jld2")
            rm(f); write(f * ".tmp", "garbage")
            r3 = run_grid(md, cfgs; outdir=out)
            @test r3.ran == [h] && length(r3.skipped) == 2
            @test isfile(f) && filesize(f) > 100
            @test !isfile(f * ".tmp")

            rebuild_summary(out, "unit")
            df = CSV.read(joinpath(out, "unit_summary.csv"), DataFrame)
            @test names(df) == _EXP_COLS
            @test nrow(df) == 3 && all(df.status .== "ok") && all(df.wall_seconds .>= 0)
            @test Set(df.hash) == Set(config_hash.(cfgs))
            r = df[df.label .== "c", :][1, :]
            @test r.covariance == "C2" && r.λ == 0.3 && r.rule == "MyCobbDouglasRule" && r.z == 1.96 && r.T == 21
            @test df[df.label .== "bt", :λ][1] === missing || isnan(df[df.label .== "bt", :λ][1])
            @test all(df.turnover .> 0) && all(df.n_decisions .>= 1) && all(df.mean_names .> 0)
            @test all(df.n_decisions .>= df.n_band_exit .+ df.n_drawdown .+ df.n_horizon .+ df.n_cash_revisit)
            # stored payload
            d = JLD2.load(f)
            @test Set(keys(d)) ⊇ Set(["config_key", "result", "scorecard"])
        end
    end

    @testset "erroring config does not stop the sweep" begin
        mktempdir() do out
            bad = with(cfg; basket_schedule=[md.dates[21] => ["AAA", "ZZZ"]], label="bad")
            good = with(cfg; label="good")
            r = @test_logs (:warn, r"run failed") match_mode=:any run_grid(md, [bad, good]; outdir=out)
            @test r.errors == [config_hash(bad)] && r.ran == [config_hash(good)]
            runs = joinpath(out, "runs")
            @test isfile(joinpath(runs, config_hash(good) * ".jld2"))
            @test !isfile(joinpath(runs, config_hash(bad) * ".jld2"))
            @test isfile(joinpath(runs, config_hash(bad) * ".error.txt"))
            rebuild_summary(out, "err")
            df = CSV.read(joinpath(out, "err_summary.csv"), DataFrame)
            @test nrow(df) == 2 && names(df) == _EXP_COLS
            @test df[df.label .== "bad", :status] == ["error"] && df[df.label .== "good", :status] == ["ok"]
            @test ismissing(df[df.label .== "bad", :sharpe][1])
            # retry: error text is replaced once the config succeeds (here: re-run still errors, file persists)
            @test_logs (:warn, r"run failed") match_mode=:any run_grid(md, [bad]; outdir=out)
            @test isfile(joinpath(runs, config_hash(bad) * ".error.txt"))
            # success removes a stale error file
            h = config_hash(good); write(joinpath(runs, h * ".error.txt"), "stale"); rm(joinpath(runs, h * ".jld2"))
            run_grid(md, [good]; outdir=out)
            @test !isfile(joinpath(runs, h * ".error.txt")) && isfile(joinpath(runs, h * ".jld2"))
        end
    end

    @testset "load_experiment" begin
        mktempdir() do dir
            p = joinpath(dir, "e.toml")
            write(p, """
            rules = ["CD", "CES"]
            covariance = "C2"
            lambda = [0.0, 0.5]
            w_max = [0.1, 0.2]
            sigma_g_max_annual = 0.12
            """)
            cfgs = load_experiment(p; md=md, base_sim=bt_base_sim(), start=md.dates[21], stop=md.dates[end])
            @test length(cfgs) == 2 * 2 * 2
            @test length(unique(config_hash.(cfgs))) == 8
            @test all(c -> c.covariance isa MyShrunkResidualCovariance, cfgs)
            @test all(c -> c.σ_g_max ≈ 0.12 * sqrt(252), cfgs)
        end
    end
end
