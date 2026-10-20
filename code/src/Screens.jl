# Ticker screens (N names per GICS sector at each re-pick date) and 130/30 short candidates.
# The bandit is a port of the eCornell S4 per-sector ε-greedy bandit (`train_sector_bandit`),
# changed to single-ticker arms and a continuous reward: the ticker's 21-day growth rate minus SPY's
# (yr⁻¹). All growth rates are g = ln(S_end/S_start)/(k·Δt) with Δt = 1/252.

"""
    AbstractScreen

Supertype of the ticker screens: `MyBanditScreen`, `MyGammaScreen`, `MyRandomScreen`.
"""
abstract type AbstractScreen end

"""
    MyBanditScreen <: AbstractScreen

Per-sector ε-greedy bandit over single-ticker arms. Fields: `lookback` (trading days; reward windows
must end within the trailing `lookback` days), `horizon` (trading days of one reward window),
`ε_floor` (dimensionless), `iters_per_arm`, `min_iters`, `max_iters` (pulls). Inner defaults
252, 21, 0.05, 50, 500, 5000 **(plan decision)**.
"""
mutable struct MyBanditScreen <: AbstractScreen
    lookback::Int
    horizon::Int
    ε_floor::Float64
    iters_per_arm::Int
    min_iters::Int
    max_iters::Int
    function MyBanditScreen()
        return new(252, 21, 0.05, 50, 500, 5000)
    end
end

"""
    MyGammaScreen <: AbstractScreen

Keeps the N highest-γ names per sector. No fields.
"""
mutable struct MyGammaScreen <: AbstractScreen
    MyGammaScreen() = new()
end

"""
    MyRandomScreen <: AbstractScreen

Draws N names per sector from a seeded stream. No fields.
"""
mutable struct MyRandomScreen <: AbstractScreen
    MyRandomScreen() = new()
end

"""
    build(::Type{MyBanditScreen}, data::NamedTuple)::MyBanditScreen

Start from the defaults and override any of the fields named in `data`.
"""
function build(::Type{MyBanditScreen}, data::NamedTuple)::MyBanditScreen
    m = MyBanditScreen()
    for f in fieldnames(MyBanditScreen)
        haskey(data, f) && setfield!(m, f, getfield(data, f))
    end
    return m
end

"""
    build(::Type{MyGammaScreen}, data::NamedTuple=NamedTuple())::MyGammaScreen

The γ screen has no fields; `data` is ignored.
"""
build(::Type{MyGammaScreen}, data::NamedTuple=NamedTuple())::MyGammaScreen = MyGammaScreen()

"""
    build(::Type{MyRandomScreen}, data::NamedTuple=NamedTuple())::MyRandomScreen

The random screen has no fields; `data` is ignored.
"""
build(::Type{MyRandomScreen}, data::NamedTuple=NamedTuple())::MyRandomScreen = MyRandomScreen()

"""
    excess_growth_windows(md::MyMarketData; horizon::Int=21, Δt::Float64=1/252)::Matrix{Float64}

Excess growth rate (yr⁻¹) of every ticker over SPY on every `horizon`-day window. Size
`(length(md.dates) - horizon) × length(md.tickers)`; row `d` is the window from the close of day `d`
to the close of day `d + horizon`:
`X[d, j] = (ln(close[d+h, j]/close[d, j]) - ln(spy[d+h]/spy[d])) / (h·Δt)`.
"""
function excess_growth_windows(md::MyMarketData; horizon::Int=21, Δt::Float64=1/252)::Matrix{Float64}
    n = length(md.dates) - horizon
    n >= 1 || throw(ArgumentError("need more than $horizon days, got $(length(md.dates))"))
    num = log.(md.close[(1 + horizon):end, :] ./ md.close[1:n, :]) .-
          log.(md.spy_close[(1 + horizon):end] ./ md.spy_close[1:n])
    return num ./ (horizon * Δt)
end

"""
    train_ticker_bandit(arms::Vector{Int}, days::UnitRange{Int}, X::AbstractMatrix; seed::Int,
        ε_floor=0.05, iters_per_arm=50, min_iters=500, max_iters=5000)::NamedTuple{(:mean, :count, :tiebreak)}

ε-greedy bandit with one arm per ticker. `arms` are column indices of `X`; a pull draws a day
uniformly from `days` and receives `X[day, arms[a]]` (excess growth rate, yr⁻¹). Pulls total
`clamp(iters_per_arm·length(arms), min_iters, max_iters)`. Returns the running mean reward (yr⁻¹,
0 for an unpulled arm), the pull count and the random tiebreak rank, all aligned with `arms`.
"""
function train_ticker_bandit(arms::Vector{Int}, days::UnitRange{Int}, X::AbstractMatrix; seed::Int,
        ε_floor::Real=0.05, iters_per_arm::Int=50, min_iters::Int=500, max_iters::Int=5000)
    rng = MersenneTwister(seed)
    A = length(arms)
    tiebreak = randperm(rng, A)
    iters = clamp(iters_per_arm * A, min_iters, max_iters)
    μ = zeros(A); cnt = zeros(Int, A)
    for k in 1:iters
        ε = max(ε_floor, k > 1 ? min(1.0, k^(-1 / 3) * (A * log(k))^(1 / 3)) : 1.0)
        pulled = findall(>(0), cnt)
        a = if rand(rng) < ε || isempty(pulled)
            rand(rng, 1:A)
        else
            pulled[argmax([(μ[i], -tiebreak[i]) for i in pulled])]
        end
        d = rand(rng, days)
        r = X[d, arms[a]]
        cnt[a] += 1
        μ[a] += (r - μ[a]) / cnt[a]
    end
    return (mean=μ, count=cnt, tiebreak=tiebreak)
end

"""
    rank_arms(res)::Vector{Int}

Positions (into the `arms` passed to `train_ticker_bandit`) ordered by: pulled arms first, then mean
reward descending, then `tiebreak` ascending.
"""
function rank_arms(res)::Vector{Int}
    return sortperm(collect(zip(res.count .== 0, -res.mean, res.tiebreak)))
end

"""
    pick(screen::AbstractScreen, md::MyMarketData, t::Int; X, γ_t, N::Int, seed::Int)::Vector{Int}

Sorted column indices of the names kept after day `t`'s close: `N` per sector (all of a sector with
at most `N` names), sectors in sorted-name order.

- `MyBanditScreen`: per-sector bandit on `X` (excess growth rates, yr⁻¹, from `excess_growth_windows`
  with the screen's `horizon`). Training windows start on days `t-lookback+1-horizon … t-horizon`, so every
  window ends in `(t-lookback, t]`. Seed of sector rank `s`: `_stream_seed(seed, md.dates[t], s)`.
- `MyGammaScreen`: the `N` highest `γ_t` (yr⁻¹) per sector, ties by column index.
- `MyRandomScreen`: one `MersenneTwister(_stream_seed(seed, md.dates[t], 0))` stream, sectors in order.
"""
function pick(screen::MyBanditScreen, md::MyMarketData, t::Int; X::AbstractMatrix,
        γ_t::AbstractVector, N::Int, seed::Int)::Vector{Int}
    size(X, 1) == length(md.dates) - screen.horizon ||
        throw(ArgumentError("X must have length(md.dates) - horizon = $(length(md.dates) - screen.horizon) rows, got $(size(X, 1))"))
    days = (t - screen.lookback + 1 - screen.horizon):(t - screen.horizon)
    first(days) >= 1 || throw(ArgumentError("training days start at $(first(days)) < 1 for t = $t"))
    groups = _sector_groups(md)
    keep = Int[]
    for (s, sector) in enumerate(sort(collect(keys(groups))))
        pool = groups[sector]
        if length(pool) <= N
            append!(keep, pool)
            continue
        end
        res = train_ticker_bandit(pool, days, X; seed=_stream_seed(seed, md.dates[t], s),
            ε_floor=screen.ε_floor, iters_per_arm=screen.iters_per_arm,
            min_iters=screen.min_iters, max_iters=screen.max_iters)
        append!(keep, pool[rank_arms(res)[1:N]])
    end
    return sort(keep)
end

function pick(screen::MyGammaScreen, md::MyMarketData, t::Int; X::AbstractMatrix,
        γ_t::AbstractVector, N::Int, seed::Int)::Vector{Int}
    groups = _sector_groups(md)
    keep = Int[]
    for sector in sort(collect(keys(groups)))
        pool = groups[sector]
        order = sortperm(collect(zip(-γ_t[pool], pool)))
        append!(keep, pool[order[1:min(N, length(pool))]])
    end
    return sort(keep)
end

function pick(screen::MyRandomScreen, md::MyMarketData, t::Int; X::AbstractMatrix,
        γ_t::AbstractVector, N::Int, seed::Int)::Vector{Int}
    groups = _sector_groups(md)
    rng = MersenneTwister(_stream_seed(seed, md.dates[t], 0))
    keep = Int[]
    for sector in sort(collect(keys(groups)))
        append!(keep, _sample_without_replacement(rng, groups[sector], N))
    end
    return sort(keep)
end

"""
    short_candidates(γ_t::AbstractVector, md::MyMarketData; N::Int, exclude::AbstractVector{Int}=Int[])::Vector{Int}

130/30 short candidates: per sector, the up to `N` names with `γ_t < 0` (yr⁻¹) that are not in
`exclude`, most negative first (ties by column index). Returns sorted column indices.
"""
function short_candidates(γ_t::AbstractVector, md::MyMarketData; N::Int,
        exclude::AbstractVector{Int}=Int[])::Vector{Int}
    groups = _sector_groups(md)
    out = Int[]
    for sector in sort(collect(keys(groups)))
        pool = [j for j in groups[sector] if γ_t[j] < 0 && !(j in exclude)]
        order = sortperm(collect(zip(γ_t[pool], pool)))
        append!(out, pool[order[1:min(N, length(pool))]])
    end
    return sort(out)
end
