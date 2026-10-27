using Test, Dates, VLINFORMS2026, Statistics
include("helpers.jl")

@testset "ewls β path matches online_sim" begin
    md = synthetic_md(ρ=0.3); f = fit_base_sim(md, md.dates[1], md.dates[800])
    c = build(MyCalibration, (base_sim=f.params, σ_m=f.σ_m, m_m=f.m_m, v_m=f.v_m, λ=0.5,
        train_start=md.dates[1], train_stop=md.dates[800]))
    sim = online_sim(md, c; start=md.dates[2], stop=md.dates[800])
    G = growth_rates(md.close); gm = growth_rates(md.spy_close)
    @test ewls_beta_path(G[1:799, 5], gm[1:799], f.params[md.tickers[5]], f.m_m, f.v_m) ≈ sim.β[:, 5]
end
@testset "β drift recovery" begin
    gen = toy_generator(); a = ema_rate(126); kt = 0.3 / sqrt(a / (2 - a))
    tr = (gen.dates[1], gen.dates[end])
    truth = with_generator(gen; h_β=126.0, k_β=fill(kt, 48), k_α=0.0)
    tg = bank_beta_stats(truth, [synthetic_draws(truth, p) for p in 1_000_001:1_000_010]; train=tr)
    fit = fit_beta_drift(with_generator(gen; k_β=zeros(48), k_α=0.0),
        [synthetic_draws(gen, p) for p in 1_000_011:1_000_020], tg; h_grid=[63.0, 126.0, 252.0], train=tr)
    @test fit.h_β == 126.0 && isapprox(median(fit.k_β), kt; rtol=0.15)
    low = (sd_β=fill(1e-6, 48), acf252=tg.acf252)
    @test all(iszero, fit_beta_drift(with_generator(gen; k_β=zeros(48), k_α=0.0),
        [synthetic_draws(gen, p) for p in 1_000_011:1_000_012], low; h_grid=[126.0], train=tr).k_β)  # at or below noise → k = 0
end
@testset "α decomposition is exact, and φ is recovered" begin
    gen = toy_generator(); d = synthetic_draws(gen, 1_000_011); i = 9
    wins = ((gen.dates[1], gen.dates[end]), (gen.dates[1], gen.dates[700]), (gen.dates[701], gen.dates[end]))
    g1 = with_generator(gen; α_center=fill(0.07, 48), k_α=0.8)
    G = synthetic_growth(g1, d).G; w = growth_window(gen.dates, wins[1]...)
    p = alpha_parts(gen, d, i; windows=wins)
    @test estimate_sim(d.g_m[w], G[w, i]).α ≈ 0.07 + p.rest[1] + 0.8 * p.unit[1] atol=1e-8
    # recovery: truth φ = 0.55 on centers c = ā + φ(α̂ − ā); fit on a separate bank → |φ̂ − 0.55| ≤ 0.10.
    # The "real" α̂ is hand-made with a wide spread (the toy's own α̂ are pure noise around α = 0).
    α_full = collect(range(-0.5, 0.5, length=48)); ā = mean(α_full)
    a = ema_rate(252); kα = 0.06 / sqrt(a / (2 - a))                  # stationary α drift SD 0.06 yr⁻¹
    truth = with_generator(gen; α_center=ā .+ 0.55 .* (α_full .- ā), k_α=kα)
    spread = 0.0; corr = 0.0                                          # truth targets, measured as fit_alpha_drift does
    for q in 1_000_001:1_000_010
        dq = synthetic_draws(truth, q)
        A = zeros(48, 3)
        for j in 1:48
            pj = alpha_parts(truth, dq, j; windows=wins)
            A[j, :] = truth.α_center[j] .+ pj.rest .+ kα .* pj.unit
        end
        spread += std(A[:, 1]) / 10
        corr += cor(A[:, 2], A[:, 3]) / 10
    end
    fit = fit_alpha_drift(with_generator(gen; k_α=0.0), [synthetic_draws(gen, q) for q in 1_000_011:1_000_020],
        (α_full=α_full, spread=spread, corr=corr); windows=wins)
    @test abs(fit.φ - 0.55) <= 0.10
end
@testset "model draw SD gate" begin
    md = synthetic_md(ρ=0.3); m = fit_name_model(md.close[:, 1]; seed=1, N=10, tune=false)
    @test 0.7 < model_sd_ratio(m, growth_rates(md.close[:, 1]); seed=1) < 1.3
end
