# Synthetic-market generator: draws synthetic price histories for the stock universe.

"""
    path_seed(p::Int, stream::Symbol, j::Int=0)::Int

Deterministic RNG seed for synthetic path `p` (1 ≤ p ≤ 10⁷) and random stream `stream`. The streams are
`:market` (p), `:sector` (10¹² + 10⁴p + j), `:name` (2·10¹² + 10⁴p + j) and `:drift` (3·10¹² + p), where
`j` (0 ≤ j < 10⁴) indexes the sector or the ticker. Seeds are distinct across paths, streams and `j`.
Throws `ArgumentError` for an unknown stream, `p > 10⁷` or `j ≥ 10⁴`.
"""
function path_seed(p::Int, stream::Symbol, j::Int = 0)::Int
    p <= 10^7 || throw(ArgumentError("path index p = $(p) exceeds 10^7"))
    j < 10^4 || throw(ArgumentError("index j = $(j) must be below 10^4"))
    if stream == :market
        return p
    elseif stream == :sector
        return 10^12 + 10^4 * p + j
    elseif stream == :name
        return 2 * 10^12 + 10^4 * p + j
    elseif stream == :drift
        return 3 * 10^12 + p
    end
    throw(ArgumentError("unknown stream :$(stream); use :market, :sector, :name or :drift"))
end

"""
    center_seed(kind::Symbol, j::Int)::Int

Fixed seed (the same for every path) of the long-run-mean draw of name model `j` (`kind = :name`: 4·10¹² + j) or
sector model `j` (`kind = :sector`: 4·10¹² + 10⁴ + j), with 1 ≤ j < 10⁴. Every `path_seed` stream lies below
4·10¹² (the largest is `:drift`, at most 3·10¹² + 10⁷), so these seeds never coincide with a path draw's seed.
Throws `ArgumentError` for an unknown kind or `j` outside [1, 10⁴).
"""
function center_seed(kind::Symbol, j::Int)::Int
    if !(1 <= j < 10^4)
        throw(ArgumentError("index j = $(j) must lie in [1, 10^4)"))
    end
    if kind == :name
        return 4 * 10^12 + j
    elseif kind == :sector
        return 4 * 10^12 + 10^4 + j
    end
    throw(ArgumentError("unknown kind :$(kind); use :name or :sector"))
end

"""
    model_long_run_mean(model; seed::Int, n_steps::Int=1_000_000)::Float64

Long-run mean (yr⁻¹) of the JumpHMM `model`'s draws: the mean of `observations` in one `n_steps`-day draw seeded
with `seed` (use `center_seed`).
"""
function model_long_run_mean(model; seed::Int, n_steps::Int=1_000_000)::Float64
    return mean(JumpHMM.simulate(model, n_steps; n_paths=1, seed=seed).paths[1].observations)
end

"""
    growth_window(dates::Vector{Date}, start::Date, stop::Date)::UnitRange{Int}

Rows of the growth-rate matrix that cover the trading days in [`start`, `stop`]. Row k is the growth from
`dates[k]` to `dates[k+1]` (yr⁻¹), so the window is `index(start):(index(stop) − 1)`.
Throws `ArgumentError` if `start` or `stop` is not in `dates`.
"""
function growth_window(dates::Vector{Date}, start::Date, stop::Date)::UnitRange{Int}
    i = findfirst(==(start), dates)
    i === nothing && throw(ArgumentError("start date $(start) is not a trading day in dates"))
    k = findfirst(==(stop), dates)
    k === nothing && throw(ArgumentError("stop date $(stop) is not a trading day in dates"))
    return i:(k - 1)
end

"""
    ema_rate(h::Real)::Float64

EMA weight a = 1 − 2^(−1/h) for half-life `h` (trading days). It equals 1 − η, where η = 2^(−1/h) is the
per-observation decay that `ewls_init` uses for the same half-life.
"""
function ema_rate(h::Real)::Float64
    h > 0 || throw(ArgumentError("half-life h = $(h) must be positive"))
    return 1.0 - 2.0^(-1.0 / h)
end

"""
    drift_path(z::AbstractVector{<:Real}, h::Real, k::Real)::Vector{Float64}

EMA of the shocks `z` (standard normal), scaled by `k` (same units as the drifting α or β), with half-life
`h` (trading days). With a = `ema_rate(h)`: x₁ = k·√(a/(2−a))·z₁ and xₜ = (1−a)·xₜ₋₁ + a·k·zₜ. The start
is drawn from the stationary distribution, so every xₜ has SD k·√(a/(2−a)). The output is linear in `k`,
so one shock bank serves every calibration value of `k`.
"""
function drift_path(z::AbstractVector{<:Real}, h::Real, k::Real)::Vector{Float64}
    a = ema_rate(h)
    x = zeros(Float64, length(z))
    if isempty(z)
        return x
    end
    x[1] = k * sqrt(a / (2 - a)) * z[1]
    for t in 2:length(z)
        x[t] = (1 - a) * x[t-1] + a * k * z[t]
    end
    return x
end

"""
    drift_variance(h::Real, k::Real)::Float64

Stationary variance k²·a/(2−a) of `drift_path` with half-life `h` (trading days) and scale `k`, where a =
`ema_rate(h)` (units of k squared).
"""
function drift_variance(h::Real, k::Real)::Float64
    a = ema_rate(h)
    return k^2 * a / (2 - a)
end

"""
    sim_residuals(md::MyMarketData, params::Dict{String,MySIMParameterEstimate}, start::Date, stop::Date)::Matrix{Float64}

SIM residuals eᵢ = gᵢ − αᵢ − βᵢ·g_m (yr⁻¹) for every ticker, on the growth rows `growth_window(md.dates, start, stop)`.
Columns follow `md.tickers`. `params` holds αᵢ (yr⁻¹) and βᵢ for each ticker, and g_m is the SPY growth rate (yr⁻¹).
"""
function sim_residuals(md::MyMarketData, params::Dict{String,MySIMParameterEstimate}, start::Date, stop::Date)::Matrix{Float64}
    rows = growth_window(md.dates, start, stop)
    g_m = growth_rates(md.spy_close)[rows]
    G = growth_rates(md.close)[rows, :]
    E = zeros(Float64, length(rows), length(md.tickers))
    for (j, t) in enumerate(md.tickers)
        p = params[t]
        for k in 1:length(rows)
            E[k, j] = G[k, j] - p.α - p.β * g_m[k]
        end
    end
    return E
end

"""
    sector_factors(E::AbstractMatrix, sector::Vector{String})::NamedTuple{(:sectors, :F)}

Equal-weight sector factors from the residual matrix `E` (yr⁻¹, days × tickers). `sectors` is the sorted unique
sector labels, and column s of `F` (yr⁻¹, days × sectors) is the mean of the columns of `E` whose entry in
`sector` is `sectors[s]`.
"""
function sector_factors(E::AbstractMatrix, sector::Vector{String})
    size(E, 2) == length(sector) || throw(ArgumentError("E has $(size(E, 2)) columns but sector has $(length(sector)) labels"))
    sectors = sort(unique(sector))
    F = zeros(Float64, size(E, 1), length(sectors))
    for (s, name) in enumerate(sectors)
        members = findall(==(name), sector)
        for k in 1:size(E, 1)
            total = 0.0
            for i in members
                total += E[k, i]
            end
            F[k, s] = total / length(members)
        end
    end
    return (sectors = sectors, F = F)
end

"""
    loo_loadings(E::AbstractMatrix, sector::Vector{String})::Vector{Float64}

Leave-one-out sector loadings δᵢ (dimensionless). δᵢ is the OLS slope, with an intercept, of `E[:, i]` (yr⁻¹) on
the mean of the other columns in the same sector (yr⁻¹). Leaving column i out of the mean keeps its own noise out
of the regressor. Throws `ArgumentError` for a sector with one name.
"""
function loo_loadings(E::AbstractMatrix, sector::Vector{String})::Vector{Float64}
    size(E, 2) == length(sector) || throw(ArgumentError("E has $(size(E, 2)) columns but sector has $(length(sector)) labels"))
    T = size(E, 1)
    δ = zeros(Float64, size(E, 2))
    for name in sort(unique(sector))
        members = findall(==(name), sector)
        length(members) >= 2 || throw(ArgumentError("sector $(name) has one name; a leave-one-out mean needs at least two"))
        for i in members
            # mean of the other names in the sector
            x = zeros(Float64, T)
            for k in 1:T
                total = 0.0
                for j in members
                    if j != i
                        total += E[k, j]
                    end
                end
                x[k] = total / (length(members) - 1)
            end
            # OLS slope with intercept
            x̄ = mean(x)
            ȳ = mean(view(E, :, i))
            sxy = 0.0
            sxx = 0.0
            for k in 1:T
                sxy += (x[k] - x̄) * (E[k, i] - ȳ)
                sxx += (x[k] - x̄)^2
            end
            δ[i] = sxy / sxx
        end
    end
    return δ
end

"""
    sector_loadings(E::AbstractMatrix, sector::Vector{String})::Vector{Float64}

Sector loadings δᵢ (dimensionless) for the generator. Each sector's `loo_loadings` are multiplied by one scale κ,
chosen so that the shared co-movement the generator builds, δᵢ·δⱼ·var(f_s) averaged over the pairs i ≠ j in the
sector, equals the mean pairwise covariance of the columns of `E` (yr⁻², days × tickers). f_s is the full
equal-weight sector mean (`sector_factors`), the series the sector model is fit to. The leave-one-out slopes keep
each name's relative loading; the scale corrects their level, which is biased low when paired with var(f_s).
Throws `ArgumentError` for a sector where either mean pairwise covariance is not positive.
"""
function sector_loadings(E::AbstractMatrix, sector::Vector{String})::Vector{Float64}
    δ = loo_loadings(E, sector)
    sf = sector_factors(E, sector)
    for (s, name) in enumerate(sf.sectors)
        members = findall(==(name), sector)
        n = length(members)
        # mean pairwise covariance: Σ_{i≠j} cov(eᵢ, eⱼ) = var(Σ eᵢ) − Σ var(eᵢ), and Σ eᵢ = n·f_s
        own = 0.0
        for i in members
            own += var(view(E, :, i))
        end
        real_cov = (n^2 * var(view(sf.F, :, s)) - own) / (n * (n - 1))
        real_cov > 0 || throw(ArgumentError("sector $(name) has mean pairwise residual covariance $(real_cov); it must be positive"))
        # the same mean for the leave-one-out loadings: Σ_{i≠j} δᵢδⱼ = (Σ δᵢ)² − Σ δᵢ²
        total = 0.0
        squares = 0.0
        for i in members
            total += δ[i]
            squares += δ[i]^2
        end
        model_cov = (total^2 - squares) * var(view(sf.F, :, s)) / (n * (n - 1))
        model_cov > 0 || throw(ArgumentError("sector $(name) leave-one-out loadings give mean pairwise covariance $(model_cov); it must be positive"))
        κ = sqrt(real_cov / model_cov)
        for i in members
            δ[i] = κ * δ[i]
        end
    end
    return δ
end

"""
    SYNTH_FLOOR

Smallest share of a synthetic name's variance that stays idiosyncratic (dimensionless).
"""
const SYNTH_FLOOR = 0.10

"""
    SYNTH_R2_PRESERVE

Real-data R² (dimensionless) at or above which a name keeps its real R² instead of matching total variance.
"""
const SYNTH_R2_PRESERVE = 0.80

"""
    compose_name(α, β, δ, g_m, f, ε, σ²_m, σ²_s, σ²_gen, r²; ε_center, β²_mean, floor=SYNTH_FLOOR, r2_threshold=SYNTH_R2_PRESERVE)::NamedTuple{(:g, :β_eff, :n_clipped)}

Compose one synthetic name's growth rate g[t] = α[t] + c·β[t]·g_m[t] + c·δ·f[t] + √s²·ε̃[t] (yr⁻¹) for every day t.
α (yr⁻¹) and β (dimensionless) are per-day paths, δ is the sector loading (dimensionless), g_m, f and ε are the
market growth, sector factor and raw JumpHMM draw (yr⁻¹), and σ²_m, σ²_s, σ²_gen are the fixed variances of g_m, f
and ε (yr⁻²). r² is the real R² (dimensionless). `β²_mean` (dimensionless, required) is the long-run mean of β[t]²,
β̄² plus `drift_variance` for a drifting β.

ε̃ = ε − `ε_center`, where `ε_center` (yr⁻¹, required) is the long-run mean of the model that drew ε
(`model_long_run_mean`). The center is fixed, not the path's own mean: removing the path mean forces Σ ε̃ = 0 over
the whole path, which ties the trading-window noise to the training-window noise (a name lucky in one window is
unlucky in the other).

The variance is matched on average over the β drift, not day by day: v̄ = β²_mean·σ²_m + δ²σ²_s and ρ̄ = v̄/σ²_gen,
and s² and c are the same on every day. If r² ≥ r2_threshold, s² = v̄(1 − r²)/r²/σ²_gen and c = 1. Otherwise, if
ρ̄ ≤ 1 − floor, s² = 1 − ρ̄ and c = 1. Otherwise c = √((1 − floor)/ρ̄), which keeps the sign of β, s² = floor, and
every day counts as clipped. A day's variance then rises and falls with β[t]², as a real name's does. Matching each
day instead shrank or clipped the sector term on the days β drifted high, which cut the sector co-movement.

Returns the growth path `g` (yr⁻¹), the per-day β after clipping `β_eff` = c·β (dimensionless) and the number of
clipped days `n_clipped` (0 or the path length).
"""
function compose_name(α::AbstractVector, β::AbstractVector, δ::Real, g_m::AbstractVector, f::AbstractVector, ε::AbstractVector,
                      σ²_m::Real, σ²_s::Real, σ²_gen::Real, r²::Real; ε_center::Real, β²_mean::Real, floor::Real=SYNTH_FLOOR,
                      r2_threshold::Real=SYNTH_R2_PRESERVE)::NamedTuple{(:g, :β_eff, :n_clipped)}
    T = length(g_m)
    if !(length(α) == T && length(β) == T && length(f) == T && length(ε) == T)
        throw(ArgumentError("α, β, g_m, f and ε must share one length; got $(length(α)), $(length(β)), $(T), $(length(f)), $(length(ε))"))
    end
    0 < floor < 1 || throw(ArgumentError("floor must lie in (0, 1); got $(floor)"))
    σ²_gen > 0 || throw(ArgumentError("σ²_gen must be positive; got $(σ²_gen)"))
    r² > 0 || throw(ArgumentError("r² must be positive; got $(r²)"))
    β²_mean >= 0 || throw(ArgumentError("β²_mean must be non-negative; got $(β²_mean)"))
    # one variance split for the whole path, from the long-run mean of β²
    v̄ = β²_mean * σ²_m + δ^2 * σ²_s
    ρ̄ = v̄ / σ²_gen
    c = 1.0
    n_clipped = 0
    if r² >= r2_threshold
        s² = v̄ * (1 - r²) / r² / σ²_gen
    elseif ρ̄ <= 1 - floor
        s² = 1 - ρ̄
    else
        c = sqrt((1 - floor) / ρ̄)
        s² = floor
        n_clipped = T
    end
    g = zeros(Float64, T)
    β_eff = zeros(Float64, T)
    for t in 1:T
        β_eff[t] = c * β[t]
        g[t] = α[t] + β_eff[t] * g_m[t] + c * δ * f[t] + sqrt(s²) * (ε[t] - ε_center)
    end
    return (g=g, β_eff=β_eff, n_clipped=n_clipped)
end

"""
    MySyntheticGenerator

Everything needed to draw synthetic paths of the stock universe. Built with `build(MySyntheticGenerator, nt)`.

Fields: `dates` (SPY trading days), `tickers`, `sector` (one per ticker), `sectors` (sorted unique sector labels),
`sector_index` (position of each ticker's sector in `sectors`); `market`, `names`, `factors` (JumpHMM models for
the market, each ticker and each sector); `α_center` (yr⁻¹), `β_center`, `δ` (sector loading), `r²` (real R²), all
one per ticker; `σ²_m`, `σ²_s` (one per sector), `σ²_gen` (one per ticker) (yr⁻²); `ε_center` (yr⁻¹, one per
ticker: the long-run mean of each name model, `model_long_run_mean` with `center_seed(:name, j)`); `h_β`, `h_α` (EMA half-lives,
trading days), `k_β` (one per ticker), `k_α` (drift scales, same units as β and α yr⁻¹), `φ`; `close0` (USD/share,
one per ticker), `spy0` (USD/share), `volume` (shares/day, one per ticker), `g_f` (yr⁻¹, one per date).
"""
mutable struct MySyntheticGenerator
    dates::Vector{Date}
    tickers::Vector{String}
    sector::Vector{String}
    sectors::Vector{String}
    sector_index::Vector{Int}
    market::Any
    names::Vector
    factors::Vector
    α_center::Vector{Float64}
    β_center::Vector{Float64}
    δ::Vector{Float64}
    r²::Vector{Float64}
    σ²_m::Float64
    σ²_s::Vector{Float64}
    σ²_gen::Vector{Float64}
    ε_center::Vector{Float64}
    h_β::Float64
    k_β::Vector{Float64}
    h_α::Float64
    k_α::Float64
    φ::Float64
    close0::Vector{Float64}
    spy0::Float64
    volume::Vector{Float64}
    g_f::Vector{Float64}
    MySyntheticGenerator() = new()
end

"""
    build(::Type{MySyntheticGenerator}, data::NamedTuple)::MySyntheticGenerator

Build a `MySyntheticGenerator` from a NamedTuple holding every field (units as in the type docstring).
Throws `ArgumentError` naming a missing field.
"""
function build(::Type{MySyntheticGenerator}, data::NamedTuple)::MySyntheticGenerator
    model = MySyntheticGenerator()
    for f ∈ fieldnames(MySyntheticGenerator)
        haskey(data, f) || throw(ArgumentError("Cannot build MySyntheticGenerator: missing field $(f)"))
        setfield!(model, f, data[f])
    end
    return model
end

"""
    with_generator(gen::MySyntheticGenerator; kw...)::MySyntheticGenerator

Copy of `gen` with the named fields replaced. Unchanged fields are shared with `gen`, not copied.
Throws `ArgumentError` for a keyword that is not a field of `MySyntheticGenerator`.
"""
function with_generator(gen::MySyntheticGenerator; kw...)::MySyntheticGenerator
    fields = fieldnames(MySyntheticGenerator)
    for k in keys(kw)
        k ∈ fields || throw(ArgumentError("with_generator: $(k) is not a field of MySyntheticGenerator"))
    end
    return build(MySyntheticGenerator, merge((; (f => getfield(gen, f) for f in fields)...), NamedTuple(kw)))
end

"""
    synthetic_draws(gen::MySyntheticGenerator, p::Int)::NamedTuple{(:g_m, :jumps, :F, :E, :Zβ, :Zα)}

Raw random draws for synthetic path `p`, with n = `length(gen.dates) − 1` growth days and N tickers. `g_m` (yr⁻¹, n)
and `jumps` (n) are the market draw. `F` (yr⁻¹, n × sectors) holds one sector-factor draw per column and `E`
(yr⁻¹, n × N) one ticker draw per column. `Zβ` and `Zα` (n × N) are standard normals for the β and α drifts.
Every stream is seeded from `path_seed(p, ...)`, so the draws do not depend on the thread count.
"""
function synthetic_draws(gen::MySyntheticGenerator, p::Int)::NamedTuple{(:g_m, :jumps, :F, :E, :Zβ, :Zα)}
    n = length(gen.dates) - 1
    N = length(gen.tickers)
    S = length(gen.sectors)
    # market draw
    mkt = JumpHMM.simulate(gen.market, n; n_paths=1, seed=path_seed(p, :market)).paths[1]
    # sector factor draws
    F = zeros(Float64, n, S)
    for s in 1:S
        F[:, s] = JumpHMM.simulate(gen.factors[s], n; n_paths=1, seed=path_seed(p, :sector, s)).paths[1].observations
    end
    # ticker draws, threaded over tickers
    E = zeros(Float64, n, N)
    Threads.@threads for j in 1:N
        E[:, j] = JumpHMM.simulate(gen.names[j], n; n_paths=1, seed=path_seed(p, :name, j)).paths[1].observations
    end
    # drift shocks: all of Zβ, then all of Zα, from one seeded stream
    rng = Xoshiro(path_seed(p, :drift))
    Zβ = randn(rng, n, N)
    Zα = randn(rng, n, N)
    return (g_m=mkt.observations, jumps=mkt.jumps, F=F, E=E, Zβ=Zβ, Zα=Zα)
end

"""
    synthetic_growth(gen::MySyntheticGenerator, d::NamedTuple)::NamedTuple{(:G, :β_eff, :clip_share)}

Growth rates of every ticker for the draws `d` from `synthetic_draws`. For ticker i, α(t) = `α_center[i]` plus the
α drift and β(t) = `β_center[i]` plus the β drift, and `compose_name` builds the growth. `G` (yr⁻¹, n × N) holds the
growth rates, `β_eff` (n × N) the per-day β after clipping, and `clip_share` (dimensionless) the share of the n·N
name-days that were clipped.
"""
function synthetic_growth(gen::MySyntheticGenerator, d::NamedTuple)::NamedTuple{(:G, :β_eff, :clip_share)}
    n, N = size(d.E)
    G = zeros(Float64, n, N)
    β_eff = zeros(Float64, n, N)
    clipped = zeros(Int, N)
    Threads.@threads for i in 1:N
        s = gen.sector_index[i]
        α = gen.α_center[i] .+ drift_path(d.Zα[:, i], gen.h_α, gen.k_α)
        β = gen.β_center[i] .+ drift_path(d.Zβ[:, i], gen.h_β, gen.k_β[i])
        out = compose_name(α, β, gen.δ[i], d.g_m, d.F[:, s], d.E[:, i], gen.σ²_m, gen.σ²_s[s], gen.σ²_gen[i], gen.r²[i];
            ε_center=gen.ε_center[i], β²_mean=gen.β_center[i]^2 + drift_variance(gen.h_β, gen.k_β[i]))
        G[:, i] = out.g
        β_eff[:, i] = out.β_eff
        clipped[i] = out.n_clipped
    end
    return (G=G, β_eff=β_eff, clip_share=sum(clipped) / (n * N))
end

"""
    market_data_from_growth(gen::MySyntheticGenerator, g_m::Vector, G::Matrix)::MyMarketData

Turn the market growth `g_m` (yr⁻¹, n) and ticker growth `G` (yr⁻¹, n × N) into a `MyMarketData` on `gen.dates`.
Closes start at `gen.close0` and `gen.spy0` (USD/share) and compound as exp(g·Δt) with Δt = 1/252 yr; opens equal
closes; volume is `gen.volume` (shares/day) on every date. Throws `ArgumentError` naming the first ticker whose
close is not finite or not positive.
"""
function market_data_from_growth(gen::MySyntheticGenerator, g_m::Vector, G::Matrix)::MyMarketData
    Δt = 1 / 252
    n, N = size(G)
    close = gen.close0' .* exp.(vcat(zeros(1, N), cumsum(G .* Δt; dims=1)))
    spy = gen.spy0 .* exp.(vcat(0.0, cumsum(g_m .* Δt)))
    for j in 1:N
        for t in 1:(n + 1)
            if !(isfinite(close[t, j]) && close[t, j] > 0)
                throw(ArgumentError("ticker $(gen.tickers[j]) has close $(close[t, j]) on day $(t); closes must be finite and positive"))
            end
        end
    end
    return build(MyMarketData, (dates=gen.dates, tickers=gen.tickers, open=close, close=close,
        volume=repeat(gen.volume', n + 1), spy_open=spy, spy_close=spy, sector=gen.sector, g_f=gen.g_f))
end

"""
    synthetic_market(gen::MySyntheticGenerator, p::Int)::NamedTuple{(:md, :draws, :β_eff, :clip_share)}

Synthetic path `p`: the draws, the composed growth and the resulting `MyMarketData` `md`. `β_eff` (n × N) and
`clip_share` (dimensionless) come from `synthetic_growth`.
"""
function synthetic_market(gen::MySyntheticGenerator, p::Int)::NamedTuple{(:md, :draws, :β_eff, :clip_share)}
    draws = synthetic_draws(gen, p)
    gr = synthetic_growth(gen, draws)
    md = market_data_from_growth(gen, draws.g_m, gr.G)
    return (md=md, draws=draws, β_eff=gr.β_eff, clip_share=gr.clip_share)
end
