using Test, Dates, Statistics, LinearAlgebra, DataFrames, Random, VLINFORMS2026
include("helpers.jl")
@testset "study loop: accounting and first day" begin
    md, ctx = study_ctx()
    res = run_study(ctx, study_run(ctx, "CD", :gamma))
    @test res.dates == md.dates[ctx.rows] && res.equity[1] == 100_000.0
    @test all(isapprox.(res.equity, res.cash .+ held_value(res, md); rtol=1e-10)) && all(res.cash .>= 0)
    @test res.decisions.reason[1] == :initial && res.repicks.trigger[1] == :initial && res.repicks.date[1] == res.dates[1]
    @test minimum(res.trades.date) == res.dates[2]
    @test run_study(ctx, study_run(ctx, "CD", :gamma)).equity == res.equity                     # deterministic
end
@testset "study loop: the first decision is the documented problem" begin
    md, ctx = study_ctx(); run = study_run(ctx, "CD", :gamma); res = run_study(ctx, run)
    t = first(ctx.rows); basket = [findfirst(==(x), md.tickers) for x in res.repicks.basket[1]]
    cols, hedge = VLINFORMS2026._study_problem_columns(MyCobbDouglasRule(), basket, ctx.pool[1], Int[], ctx.γ[1, :])
    p = VLINFORMS2026._study_problem(ctx, run, t, cols, hedge, zeros(Int, length(md.tickers)), 100_000.0)
    @test p.γ == ctx.γ[1, cols] && p.μ_g ≈ ctx.sim.α[1, cols] .+ ctx.sim.β[1, cols] .* ctx.g̃[t]
    @test p.constraints.β == ctx.sim.β[1, cols] && p.constraints.β_max == beta_cap(ctx.ξ[t])
    @test p.constraints.σ_g_max == ctx.calib.σ_m && p.constraints.c̄ ≈ 5e-4 * mean(md.close[t, cols])
    E = residual_matrix(growth_rates(md.close[t-126:t, cols]), growth_rates(md.spy_close[t-126:t]), ctx.sim.α[1, cols], ctx.sim.β[1, cols])
    @test p.Σ_g ≈ ctx.sim.σ_m[1]^2 .* (ctx.sim.β[1, cols] * ctx.sim.β[1, cols]') .+ 0.5 .* cov(E) .+ 0.5 .* Diagonal(var.(eachcol(E)))
    w = allocate(MyCobbDouglasRule(), p).w
    @test all(isapprox(res.targets[res.dates[1]][md.tickers[c]], w[k]; rtol=1e-6) for (k, c) in enumerate(cols))
end
@testset "study loop: problem columns" begin                         # Review Focus 3 and 4
    γ = [0.3, -0.2, 0.1, -0.4, 0.2, -0.1]
    @test VLINFORMS2026._study_problem_columns(MyCobbDouglasRule(), [1, 2, 3], [3, 4], Int[], γ) == ([1, 2, 3, 4], [false, false, true, true])
    @test VLINFORMS2026._study_problem_columns(MyLongShortCDRule(), [1, 2, 3], [4], [5, 6], γ) == ([1, 3, 4, 6], [false, false, true, false])
end
@testset "study loop: re-pick timer and basket changes" begin
    md, ctx = study_ctx(z=1e6, T=10_000, D_max=1.0, repick_on_band_exit=false)
    res = run_study(ctx, study_run(ctx, "CD", :random))
    @test res.repicks.date == res.dates[1:21:end]
    @test Set(res.decisions.reason) ⊆ Set([:initial, :basket_change])
    @test all(d in res.decisions.date for d in res.repicks.date[res.repicks.changed])
end
@testset "study loop: band exits re-pick only when enabled" begin
    md, ctx = study_ctx(z=0.0)
    on = run_study(ctx, study_run(ctx, "CD", :random))
    @test all(d in on.repicks.date for d in on.decisions.date[on.decisions.reason .== :band_exit])
    md, ctx = study_ctx(z=0.0, repick_on_band_exit=false)
    off = run_study(ctx, study_run(ctx, "CD", :random))
    @test any(off.decisions.reason .== :band_exit) && off.repicks.date == off.dates[1:21:end]
end
@testset "study loop: hedge on and off" begin
    for allocator in ("CD", "CES", "SG", "MinVar", "Tangent", "LS")
        md, ctx = study_ctx()
        res = run_study(ctx, study_run(ctx, allocator, :gamma))
        ok = in.(res.decisions.status, Ref((:optimal, :fallback_minvar, :long_only_fallback)))
        @test all(res.decisions.beta_target[ok] .<= res.decisions.beta_max[ok] .+ 1e-6)
        allocator == "LS" || @test all(res.cash .>= 0)
        md, ctx = study_ctx(hedge=false)
        r0 = run_study(ctx, study_run(ctx, allocator, :gamma))
        @test all(isinf, r0.decisions.beta_max) && all(r0.daily.pool_size .== 0)
    end
end
@testset "study loop: 130/30 shorts" begin                           # Review Focus 4
    md, ctx = study_ctx(); res = run_study(ctx, study_run(ctx, "LS", :gamma))
    pos = positions_by_day(res)
    allowed = Set(vcat(res.repicks.shorts...)); pool = Set(vcat(res.repicks.pool...))
    @test all(n >= 0 || t in allowed for p in pos for (t, n) in p)
    @test all(n >= 0 for p in pos for (t, n) in p if t in pool)
end
@testset "study loop: costs come from settings" begin
    md, ctx = study_ctx(half_spread_bps=0.0, impact_kappa=0.0)
    @test sum(run_study(ctx, study_run(ctx, "CD", :gamma)).trades.cost) == 0.0
end
@testset "study loop: look-ahead canary" begin
    md, ctx = study_ctx(); res = run_study(ctx, study_run(ctx, "CD", :bandit))
    d = res.dates[60]
    _, ctxp = study_ctx(; md=perturb_after(md, d))
    rp = run_study(ctxp, study_run(ctxp, "CD", :bandit))
    @test rp.equity[1:60] == res.equity[1:60]
    k = res.decisions.date .<= d
    @test rp.decisions.date[rp.decisions.date .<= d] == res.decisions.date[k]
    @test rp.repicks.basket[rp.repicks.date .<= d] == res.repicks.basket[res.repicks.date .<= d]
end
@testset "study loop: a binding β cap holds; run validation" begin   # added: the default fixture's cap never binds
    md, ctx = study_ctx(beta_ref=0.0)                                # β_max = max(0, −ξ)
    for allocator in ("CD", "MinVar")
        res = run_study(ctx, study_run(ctx, allocator, :random))
        ok = in.(res.decisions.status, Ref((:optimal, :fallback_minvar, :long_only_fallback)))
        @test any(res.decisions.cap_binds[ok])
        @test all(res.decisions.beta_target[ok] .<= res.decisions.beta_max[ok] .+ 1e-6)
    end
    s = ctx.settings
    @test_throws ArgumentError build(MyStudyRun, (settings=s, allocator="XX", screen=:gamma, seed=1))
    @test_throws ArgumentError build(MyStudyRun, (settings=s, allocator="CD", screen=:other, seed=1))
    @test_throws ArgumentError build(MyStudyRun, (settings=s, allocator="CD", screen=:gamma, seed=-1))
    @test_throws ArgumentError build(MyStudyRun, (settings=s, allocator="CD", screen=:gamma))
    @test build(MyStudyRun, (settings=s, allocator="CD", screen=:gamma, seed=1)).label == ""
    @test_throws ArgumentError run_study(ctx, build(MyStudyRun, (settings=with_settings(s; pool_H=3),
        allocator="CD", screen=:gamma, seed=1)))                     # settings of another context
end
