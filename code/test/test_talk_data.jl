using Test, Dates, VLINFORMS2026
include("helpers.jl")

# One hand-made real-run summary row. `w` is the ending wealth ÷ start (W0 = 100,000 USD).
function talk_row(screen::String, alloc::String, seed::Int, hedge::Bool, w::Float64; K::Int=13, list::String="")
    label = "$(alloc)|$(screen)|K=$(K)|hedge=$(hedge)|seed=$(seed)"
    if screen == "fixed"
        label = "$(alloc)|fixed:$(list)|hedge=$(hedge)|seed=$(seed)"
    elseif screen == "gamma" || screen == "bandit" || screen == "random"
        label = "$(alloc)|$(screen)|hedge=$(hedge)|seed=$(seed)"
    end
    return (label=label, status="ok", allocator=alloc, screen=screen, seed=seed, hedge=hedge, basket_K=K,
        W0=100_000.0, terminal=100_000.0 * w, n_decisions=28)
end

# A full real-run table: every TALK_REALRUN_ROWS picker × 7 utilities × 2 hedges × seeds 5660–5689, with
# terminal = W0·(1 + 0.001·(seed − 5660)), so every cell's median is 1.0145.
function full_rows()::DataFrame
    rows = []
    for (key, label, group) in TALK_REALRUN_ROWS
        screen = key
        K = 13
        list = ""
        if startswith(key, "fixed:")
            screen = "fixed"
            list = key[7:end]
        elseif occursin("|K=", key)
            parts = split(key, "|K=")
            screen = String(parts[1])
            K = parse(Int, parts[2])
        end
        for hedge in (false, true)
            for alloc in ["CD", "CES", "CESw", "SG", "MinVar", "Tangent", "LS"]
                for seed in 5660:5689
                    push!(rows, talk_row(screen, alloc, seed, hedge, 1.0 + 0.001 * (seed - 5660); K=K, list=list))
                end
            end
        end
    end
    return DataFrame(rows)
end

# Large-book rows: K = 34 and 100 × CD and SG, hedge off; each (K, utility) has global γ ending at 1.2 and 1.3 and
# random picks ending at 1.1, 1.25 and 1.4.
function largek_rows()::DataFrame
    rows = []
    for K in (34, 100)
        for alloc in ("CD", "SG")
            for (i, w) in enumerate((1.2, 1.3))
                push!(rows, talk_row("global_gamma", alloc, 5659 + i, false, w; K=K))
            end
            for (i, w) in enumerate((1.1, 1.25, 1.4))
                push!(rows, talk_row("random_global", alloc, 5659 + i, false, w; K=K))
            end
        end
    end
    return DataFrame(rows)
end

# A 12-trading-day result: decisions on day 0 (initial) and day 3 (band_exit). The day-0 forecast has mean
# W0·(1 + 0.01τ) and SD 0.01·W0 (USD) for τ = 1..21; the day-3 forecast starts from 0.95·W0.
function tiny_result()::MyStudyResult
    W0 = 100_000.0
    dates = Date[]
    d = Date(2025, 1, 2)
    while length(dates) < 12
        if dayofweek(d) <= 5
            push!(dates, d)
        end
        d += Day(1)
    end
    equity = W0 .* [1.00, 1.005, 1.012, 0.95, 0.96, 0.97, 0.965, 0.98, 0.99, 1.0, 1.01, 1.02]
    decisions = DataFrame(date=[dates[1], dates[4]], reason=[:initial, :band_exit])
    τ = collect(1:21)
    projections = Dict(dates[1] => build(MyProjection, (μ=W0 .* (1 .+ 0.01 .* τ), σ=fill(0.01 * W0, 21))),
        dates[4] => build(MyProjection, (μ=0.95 * W0 .* (1 .+ 0.005 .* τ), σ=fill(0.01 * W0, 21))))
    return build(MyStudyResult, (dates=dates, equity=equity, cash=zeros(12), borrow_fees=zeros(12), trades=DataFrame(),
        decisions=decisions, repicks=DataFrame(), daily=DataFrame(), targets=Dict{Date,Dict{String,Float64}}(),
        projections=projections))
end

@testset "talk data" begin
    @testset "real-run grid" begin
        rows = full_rows()
        g = talk_realrun_grid(rows)
        @test nrow(g) == 140
        @test g.label[1] == "γ per sector" && g.hedge[1] == false && g.allocator[1] == "CD"
        @test all(g.n_seeds .== 30) && all(isapprox.(g.value, 1.0145; atol=1e-12))
        @test g.row[1] == 1 && g.group_index[1] == 0 && maximum(g.row) == 10 && maximum(g.group_index) == 2
        @test g.group[g.label .== "Hand-picked 13 + WMT, PG"][1] == "Hand-picked"
        # the seed-5660 rows repeated (as in the tier1 and universe summaries) are counted once
        dup = vcat(rows, rows[rows.seed .== 5660, :])
        @test all(talk_realrun_grid(dup).n_seeds .== 30)
        # a repeated label with a different terminal is an error
        bad = vcat(rows, rows[1:1, :])
        bad.terminal[end] *= 1.01
        @test_throws ArgumentError talk_realrun_grid(bad)
        # a missing seed is an error
        @test_throws ArgumentError talk_realrun_grid(rows[2:end, :])
        # bars: lowest and highest dot per row and hedge setting
        r = talk_row_ranges(g)
        @test nrow(r) == 20 && all(r.lo .<= r.hi)
    end

    @testset "large-book shares" begin
        lk = largek_rows()
        s = talk_largek_shares(lk)
        # global [1.2, 1.3] vs random [1.1, 1.25, 1.4]: 3 of 6 pairs end higher
        @test s.share[(s.K .== 34) .& (s.allocator .== "CD")] == [0.5]
        @test s.global_median[(s.K .== 34) .& (s.allocator .== "CD")] == [1.25]
        @test !any((s.K .== 100) .& (s.allocator .== "SG")) && any((s.K .== 34) .& (s.allocator .== "SG"))
        @test_throws ArgumentError talk_largek_shares(lk[lk.screen .!= "random_global", :])
    end

    @testset "text numbers" begin
        g2 = copy(talk_realrun_grid(full_rows()))
        g2.value .= 1.2
        sector_off = (g2.picker .== "gamma") .& (g2.hedge .== false)
        g2.value[sector_off] = [1.086, 1.219, 1.221, 1.234, 1.090, 1.227, 1.110]
        g2.value[(g2.picker .== "random_global|K=13") .& (g2.hedge .== false) .& (g2.allocator .== "LS")] .= 1.031
        g2.value[(g2.picker .== "fixed:l8b_13") .& (g2.hedge .== false) .& (g2.allocator .== "Tangent")] .= 1.697
        h2h = DataFrame(comparison=fill("Global γ (13) minus Random, whole universe (13)", 3), metric="wealth",
            hedge="off", share_above_0=[0.6233, 0.7117, 0.65])
        n = Dict(talk_numbers(g2, [27, 28, 29], h2h))
        @test n["SectorLo"] == "1.09" && n["SectorHi"] == "1.23" && n["BasketLo"] == "1.03" && n["BasketHi"] == "1.70"
        @test n["Rebalances"] == "27" && n["TypicalLo"] == "62" && n["TypicalHi"] == "71"
        path = tempname()
        write_tex_macros(path, ["SectorLo" => "1.09", "Rebalances" => "27"])
        @test readlines(path) == ["\\newcommand{\\SectorLo}{1.09}", "\\newcommand{\\Rebalances}{27}"]
    end

    @testset "MPC episode" begin
        res = tiny_result()
        ep = talk_mpc_episode(res, res.dates[1], res.dates[12])
        @test nrow(ep.path) == 12 && ep.path.w[4] ≈ 0.95 && ep.path.day[1] == 0
        c1 = ep.cones[ep.cones.cone .== 1, :]
        # τ aligned with the trigger: day 3 is compared with μ[3]
        @test c1.day == [0, 1, 2, 3] && c1.sd[1] == 0.0 && c1.mean[1] ≈ 1.0 && c1.mean[4] ≈ 1.03 && c1.sd[4] ≈ 0.01
        @test ep.events.reason == ["band_exit"] && ep.events.day == [3] && ep.events.z[1] ≈ (0.95 - 1.03) / 0.01
        c2 = ep.cones[ep.cones.cone .== 2, :]
        @test c2.mean[1] ≈ 0.95 && c2.day[1] == 3 && c2.day[end] == 11
        # the episode must start on a decision with a stored forecast
        @test_throws ArgumentError talk_mpc_episode(res, res.dates[2], res.dates[12])
        @test_throws ArgumentError talk_mpc_episode(res, res.dates[1], Date(2030, 1, 2))
    end

    @testset "stylized facts" begin
        x = randn(MersenneTwister(1), 20_000)
        sf = talk_stylized_facts(x, [x, x])
        @test abs(sf.acf.real[1]) < 0.03 && abs(sf.summary.kurtosis[1] - 3) < 0.15 && abs(sf.abs_acf.real[1]) < 0.03
        @test sf.summary.series == ["real", "synthetic median"] && nrow(sf.acf) == 20
        w = diff(collect(sf.density.x))[1]
        @test isapprox(sum(filter(!isnan, sf.density.real)) * w, 1.0; atol=0.01)
        @test isapprox(sf.density.normal[findmin(abs.(sf.density.x))[2]], 1 / sqrt(2π); atol=0.01)
        # volatility in blocks of 50 days, SD 1 then 3: clustering and fat tails
        σt = [isodd(div(t - 1, 50)) ? 3.0 : 1.0 for t in 1:20_000]
        y = σt .* randn(MersenneTwister(2), 20_000)
        sy = talk_stylized_facts(y, [y])
        @test sy.abs_acf.real[1] > 0.15 && sy.summary.kurtosis[1] > 3.5
        @test sy.abs_acf.syn_p50[1] ≈ sy.abs_acf.real[1]
    end
end
