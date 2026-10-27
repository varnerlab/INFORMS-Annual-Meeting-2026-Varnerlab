using Test, Dates, VLINFORMS2026, JLD2, Random, Statistics
include("helpers.jl")
const SURROGATE = expanduser("~/Desktop/julia_work/eCornell-AI-finance-lectures/code/src/data/pretrained-jumphmm-market-surrogate.jld2")
@testset "stored market surrogate loads" begin
    if isfile(SURROGATE)
        m = load(SURROGATE, "model")
        @test m isa VLINFORMS2026.JumpHMM.JumpHiddenMarkovModel
        @test m.rf == 0.0 && m.dt ≈ 1/252 && m.partition.N == 50     # observations are plain growth rates
        r = VLINFORMS2026.JumpHMM.simulate(m, 252; n_paths=1, seed=1)
        @test length(r.paths[1].observations) == 252 && all(isfinite, r.paths[1].observations)
    else
        @test_skip "surrogate file missing"
    end
end


@testset "drift EMA" begin
    z = randn(Xoshiro(1), 4_000_000); a = ema_rate(252)
    x = drift_path(z, 252, 2.0)
    @test x[1] == 2.0 * sqrt(a / (2 - a)) * z[1]
    @test isapprox(std(x), 2.0 * sqrt(a / (2 - a)); rtol=0.03)           # stationary SD
    @test isapprox(cor(x[1:end-252], x[253:end]), 0.5; atol=0.05)          # (1-a)^252 = 0.5
    @test drift_path(z, 63, 3.0) ≈ 3.0 .* drift_path(z, 63, 1.0)
    @test all(iszero, drift_path(z[1:100], 63, 0.0))
end
@testset "path seeds and growth windows" begin
    ids = vcat(1:200, 1_000_001:1_000_020)
    seeds = [path_seed(p, :market) for p in ids]
    append!(seeds, [path_seed(p, :sector, s) for p in ids for s in 1:11])
    append!(seeds, [path_seed(p, :name, j) for p in ids for j in 1:400])
    append!(seeds, [path_seed(p, :drift) for p in ids])
    @test allunique(seeds)
    @test_throws ArgumentError path_seed(1, :bogus)
    centers = vcat([center_seed(:name, j) for j in 1:400], [center_seed(:sector, s) for s in 1:11])
    @test allunique(centers) && isdisjoint(seeds, centers)                # long-run-mean draws never reuse a path seed
    d = [Date(2014,1,2) + Day(i) for i in 0:20]
    @test growth_window(d, d[3], d[10]) == 3:9
    @test_throws ArgumentError growth_window(d, Date(2013,1,1), d[10])
end

@testset "sector factors and loadings" begin
    md = synthetic_md(ρ=0.5)
    fit = fit_base_sim(md, md.dates[1], md.dates[end])
    E = sim_residuals(md, fit.params, md.dates[1], md.dates[end])
    @test size(E) == (length(md.dates) - 1, 48)
    @test maximum(abs.(vec(mean(E; dims=1)))) < 1e-8            # OLS residuals have zero mean
    sf = sector_factors(E, md.sector)
    @test sf.sectors == ["S1", "S2", "S3", "S4"]
    @test sf.F[:, 1] ≈ vec(mean(E[:, 1:12]; dims=2))
    @test isapprox(median(loo_loadings(E, md.sector)), 0.5 / (0.5 + 0.5 / 11); atol=0.03)   # the leave-one-out mean's own noise attenuates the slope
    rng = Xoshiro(3); U = randn(rng, 5000, 12)                          # no common factor
    @test abs(mean(loo_loadings(U, fill("A", 12)))) < 0.05              # full-mean slope would be ≈ 1
    @test_throws ArgumentError loo_loadings(U[:, 1:2], ["A", "B"])
end

@testset "composition" begin
    n = 200_000
    rng = Xoshiro(11)
    gm = 2.5 .* randn(rng, n)
    f = randn(rng, n)
    ε = 4.0 .* randn(rng, n)
    c = compose_name(fill(0.0, n), fill(1.2, n), 0.8, gm, f, ε, 6.25, 1.0, 16.0, 0.3; ε_center=0.0)
    @test isapprox(var(c.g), 16.0; rtol=0.02) && isapprox(cov(c.g, gm) / var(gm), 1.2; atol=0.02) && c.n_clipped == 0
    for b in (3.0, -3.0)                                    # ρ = (9·6.25 + 0.64)/16 > 0.9: clipped, sign kept
        c = compose_name(fill(0.0, n), fill(b, n), 0.8, gm, f, ε, 6.25, 1.0, 16.0, 0.3; ε_center=0.0)
        ρ = (9 * 6.25 + 0.64) / 16
        @test all(c.β_eff .≈ b * sqrt(0.9 / ρ)) && c.n_clipped == n && isapprox(var(c.g), 16.0; rtol=0.03)
    end
    βt = repeat([0.5, 1.4], inner=1000, outer=n ÷ 2000)    # drifting β, no day clips (ρ = 0.806 at 1.4)
    c = compose_name(fill(0.0, n), βt, 0.8, gm, f, ε, 6.25, 1.0, 16.0, 0.3; ε_center=0.0)
    @test isapprox(var(c.g), 16.0; rtol=0.03) && mean(c.β_eff) ≈ mean(βt) && c.n_clipped == 0
    c = compose_name(fill(0.0, n), fill(1.0, n), 0.0, gm, f, ε, 6.25, 1.0, 16.0, 0.9; ε_center=0.0)   # R² branch
    @test isapprox(var(c.g), 6.25 / 0.9; rtol=0.03)
    c = compose_name(fill(0.3, n), fill(1.0, n), 0.8, gm, f, ε .+ 5.0, 6.25, 1.0, 16.0, 0.3; ε_center=5.0)
    s² = 1 - (1.0^2 * 6.25 + 0.8^2 * 1.0) / 16.0                                       # no clip, no R² branch
    @test isapprox(mean(c.g), 0.3 + mean(gm) + 0.8 * mean(f) + sqrt(s²) * (mean(ε .+ 5.0) - 5.0); atol=1e-10)   # ε centered on ε_center, α passes through
end

@testset "synthetic market path" begin
    gen = toy_generator(); n = length(gen.dates) - 1
    a = synthetic_market(gen, 7); b = synthetic_market(gen, 7); c = synthetic_market(gen, 8)
    for fld in fieldnames(MyMarketData)                                   # V7: bitwise
        @test getfield(a.md, fld) == getfield(b.md, fld)
    end
    @test a.md.close != c.md.close
    J = VLINFORMS2026.JumpHMM
    @test a.draws.g_m == J.simulate(gen.market, n; n_paths=1, seed=path_seed(7, :market)).paths[1].observations
    for j in (1, 48)                                                      # independent of threading
        @test a.draws.E[:, j] == J.simulate(gen.names[j], n; n_paths=1, seed=path_seed(7, :name, j)).paths[1].observations
    end
    @test size(a.md.close) == (n + 1, 48) && a.md.open == a.md.close
    @test growth_rates(a.md.close) ≈ synthetic_growth(gen, a.draws).G
    @test growth_rates(a.md.spy_close) ≈ a.draws.g_m
    g0 = with_generator(gen; k_β=zeros(48), k_α=0.0)
    s0 = synthetic_growth(g0, a.draws)                                    # toy β ≈ 1 never clips
    @test s0.clip_share == 0.0 && s0.β_eff == repeat(gen.β_center', n)
    G = synthetic_growth(gen, a.draws).G; G[5, 3] = -Inf
    @test_throws ArgumentError market_data_from_growth(gen, a.draws.g_m, G)
end
@testset "name noise is not tied across windows" begin
    # Regression: demeaning a name's draw over its whole path forces Σ ε̃ = 0, so the noise mean in the first
    # 1000 growth rows and in the rest offset each other (cor ≈ −1 across paths).
    gen = with_generator(toy_generator(); α_center=zeros(48), k_β=zeros(48), k_α=0.0); i = 5; s = gen.sector_index[i]
    m1 = zeros(100); m2 = zeros(100)
    for (q, p) in enumerate(1_000_001:1_000_100)
        d = synthetic_draws(gen, p)
        noise = synthetic_growth(gen, d).G[:, i] .- gen.β_center[i] .* d.g_m .- gen.δ[i] .* d.F[:, s]   # toy β ≈ 1 never clips
        m1[q] = mean(noise[1:1000]); m2[q] = mean(noise[1001:end])
    end
    @test abs(cor(m1, m2)) < 0.25
end
