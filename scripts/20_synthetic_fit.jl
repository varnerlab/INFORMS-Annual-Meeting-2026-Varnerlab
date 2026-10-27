# Task 6 of the synthetic-market plan: fit the generator on the real 2014–2024 data. Refits the JumpHMM model of every
# name and every sector, fits the β and α drift so the study's own estimators see the real drift on synthetic data,
# and writes results/synthetic/generator.jld2 (key "generator") and results/synthetic/generator_summary.toml.
# Run with threads: julia --project=code -t 30 scripts/20_synthetic_fit.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, JLD2, TOML

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const OUT_DIR = joinpath(REPO, "results", "synthetic")
const CALIB = joinpath(REPO, "results", "calibration.jld2")
const MANIFEST = joinpath(REPO, "code", "Manifest.toml")
const SURROGATE = expanduser("~/Desktop/julia_work/eCornell-AI-finance-lectures/code/src/data/pretrained-jumphmm-market-surrogate.jld2")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const BANK = 1_000_001:1_000_020                 # path ids of the calibration bank
const SD_GATE = (0.5, 2.0)                       # allowed range of the model draw SD ratio
const SHOW_TICKERS = ["BF.B", "CHD", "CTSH", "EOG", "GOOG", "MA", "PPG", "TECH", "UNP"]
const Δt = 1 / 252                               # yr

"""
    alpha_with_se(g_m::AbstractVector, g_i::AbstractVector)::NamedTuple{(:α, :se)}

OLS intercept α (yr⁻¹) of g_i = α + β·g_m + e (growth rates, yr⁻¹) and its standard error (yr⁻¹), derived from the
OLS residuals e: SE² = s²·(1/n + ḡ_m²/Σ(g_m − ḡ_m)²), where s² = Σe²/(n − 2) and ḡ_m is the mean of g_m.
"""
function alpha_with_se(g_m::AbstractVector, g_i::AbstractVector)::NamedTuple{(:α, :se)}
    fit = estimate_sim(g_m, g_i)
    n = length(g_m)
    m̄ = mean(g_m)
    ssr = 0.0
    sxx = 0.0
    for t in 1:n
        e = g_i[t] - fit.α - fit.β * g_m[t]
        ssr += e^2
        sxx += (g_m[t] - m̄)^2
    end
    s² = ssr / (n - 2)
    return (α=fit.α, se=sqrt(s² * (1 / n + m̄^2 / sxx)))
end

"""
    pooled_alpha_stat(g_m::AbstractVector, G::AbstractMatrix, h1::UnitRange{Int}, h2::UnitRange{Int})::Float64

The spec's pooled α change statistic (yr⁻²), reported for reference only: mean over names of Δᵢ² minus the mean of
its noise. Δᵢ = α̂ᵢ(h2) − α̂ᵢ(h1) is the change in name i's OLS α (yr⁻¹) between the growth rows `h1` and `h2` of
`g_m` (yr⁻¹) and column i of `G` (yr⁻¹), and the noise is SEᵢ,₁² + SEᵢ,₂² from `alpha_with_se`. Not floored at 0,
so a negative value means the change is smaller than its noise.
"""
function pooled_alpha_stat(g_m::AbstractVector, G::AbstractMatrix, h1::UnitRange{Int}, h2::UnitRange{Int})::Float64
    N = size(G, 2)
    Δ² = 0.0
    noise = 0.0
    for i in 1:N
        a1 = alpha_with_se(g_m[h1], G[h1, i])
        a2 = alpha_with_se(g_m[h2], G[h2, i])
        Δ² += (a2.α - a1.α)^2 / N
        noise += (a1.se^2 + a2.se^2) / N
    end
    return Δ² - noise
end

function main()
    t0 = time()

    # Step 1: real data and the 2014–2024 calibration
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    calib = load_calibration(CALIB)
    if calib.train_start != SYNTH_TRAIN[1] || calib.train_stop != SYNTH_TRAIN[2]
        error("calibration window is $(calib.train_start) → $(calib.train_stop); expected $(SYNTH_TRAIN[1]) → $(SYNTH_TRAIN[2])")
    end
    if md.dates[1] != SYNTH_TRAIN[1]
        error("market data start on $(md.dates[1]); expected $(SYNTH_TRAIN[1])")
    end
    tickers = md.tickers
    N = length(tickers)
    i0 = index_of(md, SYNTH_TRAIN[1])
    i1 = index_of(md, SYNTH_TRAIN[2])
    rows = growth_window(md.dates, SYNTH_TRAIN[1], SYNTH_TRAIN[2])     # growth rows of the training window
    G = growth_rates(md.close)
    g_m = growth_rates(md.spy_close)
    println(N, " tickers, ", length(md.dates), " dates, ", length(unique(md.sector)), " sectors, ",
        Threads.nthreads(), " threads")

    # Step 2: refit the name models on the training closes; the SD draw reuses the fit seed, and each model's
    # long-run mean (the center of its noise, yr⁻¹) comes from a 1,000,000-step draw seeded by center_seed
    t_names = time()
    names = Vector{VLINFORMS2026.JumpHMM.JumpHiddenMarkovModel}(undef, N)
    ratio = zeros(Float64, N)
    ε_center = zeros(Float64, N)
    Threads.@threads for j in 1:N
        names[j] = fit_name_model(md.close[i0:i1, j]; seed=5660 + j)
        ratio[j] = model_sd_ratio(names[j], G[rows, j]; seed=5660 + j)
        ε_center[j] = model_long_run_mean(names[j]; seed=center_seed(:name, j))
    end
    println("name models: ", round(time() - t_names; digits=1), " s")
    println("model SD ratios:")
    for t in SHOW_TICKERS
        j = findfirst(==(t), tickers)
        if j === nothing
            println("  ", t, ": not in the universe")
        else
            println("  ", t, ": ", round(ratio[j]; digits=4))
        end
    end
    println("  median ", round(median(ratio); digits=4), ", range [", round(minimum(ratio); digits=4), ", ",
        round(maximum(ratio); digits=4), "]")
    bad = Int[]
    for j in 1:N
        if ratio[j] < SD_GATE[1] || ratio[j] > SD_GATE[2]
            push!(bad, j)
        end
    end
    if !isempty(bad)
        error("model SD gate: ", length(bad), " ratios outside [", SD_GATE[1], ", ", SD_GATE[2], "]: ",
            join(["$(tickers[j]) $(round(ratio[j]; digits=4))" for j in bad], ", "))
    end

    # Step 3: residuals, sector factors, loadings and the fixed real variances
    ols = fit_base_sim(md, SYNTH_TRAIN[1], SYNTH_TRAIN[2])
    gap = 0.0
    for t in tickers
        gap = max(gap, abs(ols.params[t].α - calib.base_sim[t].α), abs(ols.params[t].β - calib.base_sim[t].β))
    end
    println("largest |OLS − calibration| in α or β: ", gap)
    E = sim_residuals(md, ols.params, SYNTH_TRAIN[1], SYNTH_TRAIN[2])
    sf = sector_factors(E, md.sector)
    S = length(sf.sectors)
    δ = loo_loadings(E, md.sector)
    σ²_m = var(g_m[rows])
    σ²_s = [var(sf.F[:, s]) for s in 1:S]
    σ²_gen = [var(G[rows, j]) for j in 1:N]
    r² = [calib.base_sim[t].r² for t in tickers]

    # Step 4: sector models on the compounded sector factors, and each one's long-run mean (yr⁻¹, diagnostic only)
    factors = Vector{VLINFORMS2026.JumpHMM.JumpHiddenMarkovModel}(undef, S)
    sector_mean = zeros(Float64, S)
    Threads.@threads for s in 1:S
        factors[s] = fit_name_model(exp.(vcat(0.0, cumsum(sf.F[:, s] .* Δt))); seed=7000 + s)
        sector_mean[s] = model_long_run_mean(factors[s]; seed=center_seed(:sector, s))
    end
    # noise center against each name's mean real training growth (yr⁻¹)
    center_gap = zeros(Float64, N)
    for j in 1:N
        center_gap[j] = ε_center[j] - mean(G[rows, j])
    end
    println("ε_center − mean real training growth: median ", round(median(center_gap); digits=4), ", range [",
        round(minimum(center_gap); digits=4), ", ", round(maximum(center_gap); digits=4), "] yr⁻¹")
    println("sector model long-run means (yr⁻¹): ", join(["$(sf.sectors[s]) $(round(sector_mean[s]; digits=4))" for s in 1:S], ", "))

    # Step 5: the stored market surrogate
    market = JLD2.load(SURROGATE, "model")

    # Step 6: the generator with no drift (h_β = 252 is a placeholder that the β fit replaces)
    sector_index = zeros(Int, N)
    for j in 1:N
        sector_index[j] = findfirst(==(md.sector[j]), sf.sectors)
    end
    gen0 = build(MySyntheticGenerator, (dates=md.dates, tickers=tickers, sector=md.sector, sectors=sf.sectors,
        sector_index=sector_index, market=market, names=names, factors=factors,
        α_center=[ols.params[t].α for t in tickers], β_center=[ols.params[t].β for t in tickers],
        δ=δ, r²=r², σ²_m=σ²_m, σ²_s=σ²_s, σ²_gen=σ²_gen, ε_center=ε_center,
        h_β=252.0, k_β=zeros(N), h_α=252.0, k_α=0.0, φ=1.0,
        close0=md.close[1, :], spy0=md.spy_close[1], volume=vec(median(md.volume[i0:i1, :]; dims=1)), g_f=md.g_f))

    # Step 7: real targets, the bank, then the β and α drift fits
    targets = real_drift_targets(md, calib)
    println("real targets: α spread ", round(targets.spread; digits=4), " yr⁻¹ (expected ≈ 0.076), α corr ",
        round(targets.corr; digits=4), " (expected ≈ 0.18), β̂ acf252 median ", round(targets.acf252; digits=4),
        ", β̂ SD median ", round(median(targets.sd_β); digits=4))
    t_bank = time()
    bank = [synthetic_draws(gen0, p) for p in BANK]
    println("bank of ", length(bank), " paths: ", round(time() - t_bank; digits=1), " s")
    t_beta = time()
    fb = fit_beta_drift(gen0, bank, targets)
    println("β drift fit: ", round(time() - t_beta; digits=1), " s")
    println(fb.table)
    # names whose stationary β drift SD reached the bisection's upper end (sd_max = 1.5)
    a_β = ema_rate(fb.h_β)
    n_cap = 0
    for k in fb.k_β
        if k * sqrt(a_β / (2 - a_β)) >= 1.5 * (1 - 1e-6)
            n_cap += 1
        end
    end
    println("h_β = ", fb.h_β, ", k_β median ", round(median(fb.k_β); digits=4), ", share k_β = 0 ",
        round(count(iszero, fb.k_β) / N; digits=4), ", names at the sd_max cap ", n_cap)
    gen1 = with_generator(gen0; h_β=fb.h_β, k_β=fb.k_β)
    fa = fit_alpha_drift(gen1, bank, targets)
    a_α = ema_rate(gen1.h_α)
    sd_α = fa.k_α * sqrt(a_α / (2 - a_α))
    println("α drift fit: φ = ", fa.φ, ", k_α = ", round(fa.k_α; digits=4), " (stationary SD ", round(sd_α; digits=4),
        " yr⁻¹), spread ", round(fa.spread; digits=4), " vs ", round(targets.spread; digits=4), ", corr ",
        round(fa.corr; digits=4), " vs ", round(targets.corr; digits=4), ", loss ", round(fa.loss; digits=4))
    if fa.k_α == 0
        error("α drift fit gave k_α = 0: decision 4 says α drifts, and the data leave no room for it. Stopping.")
    end
    ā = mean(targets.α_full)
    gen = with_generator(gen1; α_center=ā .+ fa.φ .* (targets.α_full .- ā), k_α=fa.k_α, φ=fa.φ)

    # Step 8: save the generator, then the summary
    mkpath(OUT_DIR)
    jldsave(joinpath(OUT_DIR, "generator.jld2"); generator=gen)
    h1 = growth_window(md.dates, SYNTH_ALPHA_WINDOWS[2][1], SYNTH_ALPHA_WINDOWS[2][2])
    h2 = growth_window(md.dates, SYNTH_ALPHA_WINDOWS[3][1], SYNTH_ALPHA_WINDOWS[3][2])
    pooled_real = pooled_alpha_stat(g_m, G, h1, h2)
    pooled_bank = 0.0
    clip = 0.0
    for d in bank
        gr = synthetic_growth(gen, d)
        pooled_bank += pooled_alpha_stat(d.g_m, gr.G, h1, h2) / length(bank)
        clip += gr.clip_share / length(bank)
    end
    println("pooled α statistic: real ", pooled_real, " yr⁻², fitted bank ", pooled_bank, " yr⁻²; bank clip share ", clip)
    sectors = Dict{String,Any}()
    for s in 1:S
        members = findall(==(s), sector_index)
        sectors[sf.sectors[s]] = Dict{String,Any}("delta_median" => median(δ[members]), "sigma_s" => sqrt(σ²_s[s]),
            "model_mean" => sector_mean[s])
    end
    table = [Dict{String,Any}("h" => r.h, "k_median" => r.k_median, "zero_share" => r.zero_share, "acf252" => r.acf252)
        for r in eachrow(fb.table)]
    rev = TOML.parsefile(MANIFEST)["deps"]["JumpHMM"][1]["repo-rev"]
    wall = time() - t0
    summary = Dict{String,Any}(
        "jumphmm_rev" => rev,
        "wall_seconds" => wall,
        "beta" => Dict{String,Any}("h_beta" => fb.h_β, "k_median" => median(fb.k_β),
            "k_q25" => quantile(fb.k_β, 0.25), "k_q75" => quantile(fb.k_β, 0.75),
            "zero_share" => count(iszero, fb.k_β) / N, "table" => table),
        "alpha" => Dict{String,Any}("phi" => fa.φ, "k_alpha" => fa.k_α, "sd_alpha" => sd_α, "h_alpha" => gen.h_α,
            "spread_fit" => fa.spread, "spread_target" => targets.spread, "corr_fit" => fa.corr,
            "corr_target" => targets.corr, "loss" => fa.loss,
            "pooled_stat_real" => pooled_real, "pooled_stat_bank" => pooled_bank),
        "targets" => Dict{String,Any}("alpha_spread" => targets.spread, "alpha_corr" => targets.corr,
            "beta_acf252" => targets.acf252, "beta_sd_median" => median(targets.sd_β)),
        "sectors" => sectors,
        "model_sd_ratio" => Dict{String,Any}("median" => median(ratio), "min" => minimum(ratio), "max" => maximum(ratio)),
        "eps_center_gap" => Dict{String,Any}("median" => median(center_gap), "min" => minimum(center_gap), "max" => maximum(center_gap)))
    io = open(joinpath(OUT_DIR, "generator_summary.toml"), "w")
    TOML.print(io, summary; sorted=true)
    close(io)
    println("wrote ", joinpath(OUT_DIR, "generator.jld2"), " and generator_summary.toml; wall time ",
        round(wall; digits=1), " s")
    return nothing
end

main()
