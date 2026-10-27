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
