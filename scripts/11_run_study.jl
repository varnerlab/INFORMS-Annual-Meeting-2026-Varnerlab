# Task 13: run the Tier 1 grid (default), the fixed-schedule grid (--fixed) or a 2-run smoke test (--smoke).
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const OUT_DIR = joinpath(REPO, "results")
const SMOKE_DIR = joinpath(REPO, "results", "smoke")
const CONFIG = joinpath(REPO, "config", "study.toml")
const CALIB = joinpath(REPO, "results", "calibration.jld2")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)

function main(args=ARGS)
    fixed, smoke = "--fixed" in args, "--smoke" in args
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    s, grid = load_study(CONFIG)
    calib = load_calibration(CALIB)
    if smoke
        runs = filter(r -> r.allocator == "CD" && r.screen == :gamma, tier1_runs(s, grid))
        outdir, experiment = SMOKE_DIR, "smoke"
    elseif fixed
        runs, outdir, experiment = fixed_schedule_runs(s, grid), OUT_DIR, "fixed"
    else
        runs, outdir, experiment = tier1_runs(s, grid), OUT_DIR, "tier1"
    end
    println(length(runs), " runs, ", Threads.nthreads(), " threads, experiment ", experiment)
    res = run_study_grid(md, calib, runs; outdir=outdir)
    df = rebuild_study_summary(outdir, experiment; hashes=study_hash.(runs))
    println("ran=", length(res.ran), " skipped=", length(res.skipped), " errors=", length(res.errors))
    ok = df[string.(df.status) .== "ok", :]
    isempty(ok) || println("mean wall seconds per run: ", mean(skipmissing(ok.wall_seconds)))
    return nothing
end

main()
