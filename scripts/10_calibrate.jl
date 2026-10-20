# Task 13: calibrate on the 2014-2024 window; writes results/calibration.jld2 and calibration_summary.toml.
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, TOML

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const OUT_DIR = joinpath(REPO, "results")
const CONFIG = joinpath(REPO, "config", "study.toml")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)

function main()
    mkpath(OUT_DIR)
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    s, _ = load_study(CONFIG)
    c = calibrate(md; train_start=s.train_start, train_stop=s.train_stop)
    save_calibration(joinpath(OUT_DIR, "calibration.jld2"), c)
    summary = calibration_summary(md, c, s)
    open(joinpath(OUT_DIR, "calibration_summary.toml"), "w") do io
        TOML.print(io, summary; sorted=true)
    end
    for k in sort!(collect(keys(summary)))
        println(rpad(k, 16), summary[k])
    end
    return nothing
end

main()
