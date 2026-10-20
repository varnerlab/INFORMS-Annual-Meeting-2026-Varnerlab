# Diagnostic (addendum §1): does the study engine reproduce CHEME-5660 L8b's monthly Cobb-Douglas result on the
# user's 13-name list, and which study setting moves it? The list (AAPL, MSFT, INTC, MU, AMD, NVDA, GS, BAC, WFC,
# C, F, GM, JNJ) was chosen by the user in advance, without 2025 data.
#
# Runs `run_study` (CD, γ screen with N_per_sector = 6, so every name with γ > 0 is held) on a market restricted
# to the list, first with L8b-like settings (SIM frozen at the 2014–2024 fit through a huge EWLS half-life and
# prior weight, no per-stock or volatility cap, monthly MPC horizon T = 21 with the band and drawdown triggers
# off via z = 1e6 and D_max = 1, hedge off), then adds the study settings one at a time (the per-stock cap is
# the addendum's max(w_max, w_max_scale/K) for K screen picks, as in the study's fixed-list arm). Prints the Sharpe
# ratio, max drawdown, realized growth rate (1/yr), decision count and mean invested fraction, for 2025 alone
# (L8b's window) and for the study window, next to SPY buy-and-hold. Every metric is growth-rate based.
# Read-only: writes nothing. Usage (from the repo root): julia scripts/dev/l8b_reproduction.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, Printf, DataFrames

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const DATA_DIR = joinpath(REPO, "data")
const CONFIG = joinpath(REPO, "config", "study.toml")
const CALIB = joinpath(REPO, "results", "calibration.jld2")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const LIST_NAME = "l8b_13"                      # [grid.fixed_lists] entry of config/study.toml
const ALLOCATOR = "CD"
const WINDOWS = [("2025 only (L8b window)", Date(2025, 12, 31)),
    ("2025-01-02 → 2026-09-04 (study window)", Date(2026, 9, 4))]
# L8b-like settings: frozen SIM (EWLS half-life and prior weight 1e12 trading days), no caps (w_max = 1,
# vol cap 1e6 yr⁻¹ᐟ²), T = 21 trading days, band and drawdown triggers off (z = 1e6, D_max = 1), hedge off.
const L8B = (N_per_sector=6, hedge=false, w_max=1.0, vol_cap_annual=1e6, ewls_half_life=1e12, ewls_prior_weight=1e12,
    T=21, z=1e6, D_max=1.0)
const ONLINE = (ewls_half_life=63.0, ewls_prior_weight=63.0)
const STEPS = [
    ("A  L8b-like: fixed 2014-24 SIM, no caps, monthly (T=21)", L8B),
    ("A' same, daily (T=1)", merge(L8B, (T=1,))),
    ("B  + online SIM (63/63)", merge(L8B, ONLINE)),
    ("C  + per-stock cap max(10%, 1.5/K), K = picks", merge(L8B, ONLINE, (w_max=0.10,))),
    ("D  + vol cap = SPY 2014-24", merge(L8B, ONLINE, (w_max=0.10, vol_cap_annual=NaN))),
    ("E  + MPC triggers (z 1.96, D_max 0.08) = study, hedge off",
        merge(L8B, ONLINE, (w_max=0.10, vol_cap_annual=NaN, z=1.96, D_max=0.08))),
    ("F  + hedge (β cap + pool) = study, hedge on",
        merge(L8B, ONLINE, (w_max=0.10, vol_cap_annual=NaN, z=1.96, D_max=0.08, hedge=true))),
]

# The full market restricted to `list` (prices, volume and sectors of those tickers; SPY and g_f unchanged).
function restrict(full::MyMarketData, list::Vector{String})::MyMarketData
    idx = [something(findfirst(==(t), full.tickers), 0) for t in list]
    all(>(0), idx) || throw(ArgumentError("not in the universe: $(list[idx .== 0])"))
    return build(MyMarketData, (dates=full.dates, tickers=full.tickers[idx], open=full.open[:, idx],
        close=full.close[:, idx], volume=full.volume[:, idx], spy_open=full.spy_open, spy_close=full.spy_close,
        sector=full.sector[idx], g_f=full.g_f))
end

# Growth scorecard, decision count and mean invested fraction of one run on `md` with settings `s`.
function score(md::MyMarketData, calib::MyCalibration, s::MyStudySettings)
    ctx = build_context(md, calib, s)
    res = run_study(ctx, build(MyStudyRun, (settings=s, allocator=ALLOCATOR, screen=:gamma, seed=5660, label="")))
    i0 = index_of(md, res.dates[1]); i1 = index_of(md, res.dates[end])
    sc = growth_scorecard(res.equity, md.spy_close[i0:i1], md.g_f[i0:i1])
    return (sc=sc, n=nrow(res.decisions), invested=mean(res.decisions.invested_frac))
end

function main()
    full = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    calib = load_calibration(CALIB)
    base, grid = load_study(CONFIG)
    list = Dict(grid.fixed_lists)[LIST_NAME]
    md = restrict(full, list)
    println("list ", LIST_NAME, " (chosen in advance, without 2025 data): ",
        join(["$(t)=$(s)" for (t, s) in zip(md.tickers, md.sector)], ", "))
    for (label, stop) in WINDOWS
        println("\n=== ", label, " ===")
        a = index_of(md, base.start); b = searchsortedlast(md.dates, stop)
        spy = growth_scorecard(md.spy_close[a:b] ./ md.spy_close[a] .* base.W0, md.spy_close[a:b], md.g_f[a:b])
        @printf("%-58s Sharpe %6.3f  maxDD %5.1f%%  growth %6.3f/yr\n", "SPY buy-and-hold", spy.sharpe,
            100spy.max_drawdown, spy.realized_growth)
        for (name, kw) in STEPS
            r = score(md, calib, with_settings(base; stop=stop, kw...))
            @printf("%-58s Sharpe %6.3f  maxDD %5.1f%%  growth %6.3f/yr  decisions %3d  invested %.2f\n", name,
                r.sc.sharpe, 100r.sc.max_drawdown, r.sc.realized_growth, r.n, r.invested)
        end
    end
    return nothing
end

main()
