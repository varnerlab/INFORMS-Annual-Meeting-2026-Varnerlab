# Task 13: calibrate on the 2014-2024 window; writes results/calibration.jld2 and calibration_summary.toml.
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, TOML, Statistics

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const OUT_DIR = joinpath(REPO, "results")
const CONFIG = joinpath(REPO, "config", "study.toml")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const Δt = 1 / 252                      # yr

function main()
    mkpath(OUT_DIR)
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    s, _ = load_study(CONFIG)
    c = calibrate(md; train_start=s.train_start, train_stop=s.train_stop)
    save_calibration(joinpath(OUT_DIR, "calibration.jld2"), c)
    i0, i1 = index_of(md, s.train_start), index_of(md, s.train_stop)
    spy = md.spy_close[i0:i1]
    G = xi_gain(spy; short=s.xi_short, long=s.xi_long, q=s.xi_quantile)
    sectors = Dict{String,Int}()
    for x in md.sector
        sectors[x] = get(sectors, x, 0) + 1
    end
    summary = Dict{String,Any}(
        "lambda" => c.λ, "sigma_m" => c.σ_m, "vol_cap_annual" => c.σ_m * sqrt(Δt),
        "m_m" => c.m_m, "v_m" => c.v_m,
        "xi_short" => s.xi_short, "xi_long" => s.xi_long, "xi_quantile" => s.xi_quantile, "G" => G,
        "train_start" => string(s.train_start), "train_stop" => string(s.train_stop),
        "universe_size" => length(md.tickers), "sector_counts" => sectors)
    open(joinpath(OUT_DIR, "calibration_summary.toml"), "w") do io
        TOML.print(io, summary)
    end
    for k in sort!(collect(keys(summary)))
        println(rpad(k, 16), summary[k])
    end
    return nothing
end

main()
