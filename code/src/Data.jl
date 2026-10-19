"""Price-file names (calibration, 2025, 2026 windows), in chronological order."""
const DATA_FILES = [
    "SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2",
    "SP500-Daily-OHLC-1-2-2025-to-12-31-2025.jld2",
    "SP500-Daily-OHLC-1-2-2026-to-09-04-2026.jld2",
]

const SECTORS_FILE = "sp500-sectors.csv"
const DTB3_FILE = "DTB3.csv"
const DEFAULT_G_F = 0.04   # yr⁻¹, used when DTB3.csv is absent

"""
    growth_rates(x::AbstractVecOrMat; Δt=1/252)

Annualized growth rates (yr⁻¹) `log(x[t]/x[t-1])/Δt` along rows, with Δt in years.
Returns `size(x,1) - 1` rows.
"""
function growth_rates(x::AbstractVecOrMat; Δt::Real=1/252)
    r = size(x, 1)
    return log.(selectdim(x, 1, 2:r) ./ selectdim(x, 1, 1:r-1)) ./ Δt
end

"""
    index_of(md::MyMarketData, d::Date)::Int

Row index of date `d` in `md.dates`. Throws `ArgumentError` if `d` is not a trading day.
"""
function index_of(md::MyMarketData, d::Date)::Int
    i = searchsortedfirst(md.dates, d)
    (i <= length(md.dates) && md.dates[i] == d) || throw(ArgumentError("$(d) is not a trading day"))
    return i
end

# Per-ticker (open, close, volume) series keyed by date, restricted to [start, stop].
function _slice(df::DataFrame, start::Date, stop::Date)::Dict{Date,NTuple{3,Float64}}
    d = Date.(df.timestamp)
    keep = (d .>= start) .& (d .<= stop)
    return Dict{Date,NTuple{3,Float64}}(d[i] => (Float64(df.open[i]), Float64(df.close[i]), Float64(df.volume[i])) for i in findall(keep))
end

"""
    _risk_free(dir::String, dates::Vector{Date})::Vector{Float64}

Risk-free growth rate g_f = log(1 + DTB3/100) (yr⁻¹) on `dates`, forward-filled from FRED DTB3
(leading days take the first observation). Constant 0.04 yr⁻¹ with a warning if the file is absent.
"""
function _risk_free(dir::String, dates::Vector{Date})::Vector{Float64}
    path = joinpath(dir, DTB3_FILE)
    if !isfile(path)
        @warn "DTB3.csv not found; using constant risk-free rate $(DEFAULT_G_F) yr⁻¹"
        return fill(DEFAULT_G_F, length(dates))
    end
    df = CSV.read(path, DataFrame; missingstring=[".", "", "NA"])
    obs = Date.(df[!, 1])
    rate = df[!, 2]
    ok = findall(!ismissing, rate)
    obs, rate = obs[ok], Float64.(rate[ok]) ./ 100
    return [log(1 + rate[clamp(searchsortedlast(obs, d), 1, length(obs))]) for d in dates]
end

"""
    load_market_data(dir::String; start::Date, stop::Date)::MyMarketData

Load the three price files and the sector CSV from `dir`, over dates in [start, stop].
Trading days are SPY's. The universe is the tickers present in all three files, with a row on
every trading day, all opens and closes > 0, and a sector in the CSV (sorted, SPY excluded).
Prices are USD/share, volume is shares/day, g_f is yr⁻¹.
"""
function load_market_data(dir::String; start::Date, stop::Date)::MyMarketData
    sets = [JLD2.load(joinpath(dir, f))["dataset"] for f in DATA_FILES]
    spy = merge((_slice(s["SPY"], start, stop) for s in sets)...)
    dates = sort!(collect(keys(spy)))
    isempty(dates) && throw(ArgumentError("no SPY trading days in [$(start), $(stop)]"))

    sectors = CSV.read(joinpath(dir, SECTORS_FILE), DataFrame)
    sector_of = Dict(String(r.Symbol) => String(r."GICS Sector") for r in eachrow(sectors))

    candidates = sort!(collect(intersect((Set(keys(s)) for s in sets)...)))
    series = Dict{String,Dict{Date,NTuple{3,Float64}}}()
    for t in candidates
        (t == "SPY" || !haskey(sector_of, t)) && continue
        rows = merge((_slice(s[t], start, stop) for s in sets)...)
        all(d -> haskey(rows, d) && rows[d][1] > 0 && rows[d][2] > 0, dates) && (series[t] = rows)
    end
    tickers = sort!(collect(keys(series)))

    pick(k) = [series[t][d][k] for d in dates, t in tickers]
    return build(MyMarketData, (
        dates = dates,
        tickers = tickers,
        open = pick(1), close = pick(2), volume = pick(3),
        spy_open = [spy[d][1] for d in dates],
        spy_close = [spy[d][2] for d in dates],
        sector = [sector_of[t] for t in tickers],
        g_f = _risk_free(dir, dates)))
end
