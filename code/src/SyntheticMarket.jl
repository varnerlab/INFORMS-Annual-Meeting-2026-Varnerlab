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
