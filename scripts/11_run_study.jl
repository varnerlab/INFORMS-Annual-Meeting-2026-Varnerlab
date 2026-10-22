# Task 13: run the Tier 1 grid (default), the fixed-schedule grid (--fixed), the addendum's size sweep and
# fixed lists (--universe) or a 2-run smoke test (--smoke), the η sweep (--eta) or the clean budget-only limits on
# the l8b_13 list (--limits). A run's cache key holds its settings and a
# fingerprint of the market data and calibration, but not the code: after a code change pass --fresh, which
# reruns the selected runs instead of reusing results/runs/, and check that the run reports skipped = 0.
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const OUT_DIR = joinpath(REPO, "results")
const SMOKE_DIR = joinpath(REPO, "results", "smoke")
const CONFIG = joinpath(REPO, "config", "study.toml")
const CALIB = joinpath(REPO, "results", "calibration.jld2")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)

# Step (a): the budget-only dial on one day, written to results/limits_day.csv and gated on the closed forms.
function limits_day_step(md, calib, runs)
    ctx = build_context(md, calib, runs[1].settings)
    cols = sort!([findfirst(==(tk), md.tickers) for tk in runs[1].settings.fixed_tickers])
    s = limits_row(ctx.γ, cols)
    day = limits_day(ctx, runs, s)
    mkpath(OUT_DIR)
    CSV.write(joinpath(OUT_DIR, "limits_day.csv"), day)
    println("limits day: ", md.dates[ctx.rows[s]], " (run row ", s, ")")
    s > 1 && println("(fewer than 5 of the 13 names had γ > 0 on the run start)")
    gap = abs.(day.weight .- day.closed_form)
    println("largest |weight - closed_form| = ", maximum(gap))
    bad = limits_failures(day; tol=1e-3)
    isempty(bad) || error("limits day: solver weight not finite, not optimal, or more than 1e-3 from the closed form:\n",
        join(("$(r.label) $(r.ticker) weight=$(r.weight) closed_form=$(r.closed_form) status=$(r.status)" for r in eachrow(bad)), "\n"))
    return day
end

function main(args=ARGS)
    fixed, smoke, universe, fresh = "--fixed" in args, "--smoke" in args, "--universe" in args, "--fresh" in args
    eta, limits = "--eta" in args, "--limits" in args
    bad = filter(a -> !(a in ("--fixed", "--smoke", "--universe", "--fresh", "--eta", "--limits")), args)
    isempty(bad) || error("unknown arguments: ", join(bad, " "))
    count((fixed, smoke, universe, eta, limits)) <= 1 ||
        error("pass at most one of --fixed, --universe, --smoke, --eta, --limits")
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    s, grid = load_study(CONFIG)
    calib = load_calibration(CALIB)
    if smoke
        runs = filter(r -> r.allocator == "CD" && r.screen == :gamma, tier1_runs(s, grid))
        length(runs) == 2 || error("smoke filter must yield exactly 2 runs, got ", length(runs))
        outdir, experiment = SMOKE_DIR, "smoke"
    elseif fixed
        runs, outdir, experiment = fixed_schedule_runs(s, grid), OUT_DIR, "fixed"
    elseif universe
        runs, outdir, experiment = vcat(universe_runs(s, grid), fixed_list_runs(s, grid)), OUT_DIR, "universe"
    elseif eta
        runs, outdir, experiment = eta_runs(s, grid), OUT_DIR, "eta"
    elseif limits
        runs, outdir, experiment = limits_runs(s, grid), OUT_DIR, "limits"
    else
        runs, outdir, experiment = tier1_runs(s, grid), OUT_DIR, "tier1"
    end
    println(length(runs), " runs, ", Threads.nthreads(), " threads, experiment ", experiment, fresh ? ", fresh" : "")
    res = run_study_grid(md, calib, runs; outdir=outdir, fresh=fresh)
    df = rebuild_study_summary(outdir, experiment; hashes=study_hash.(runs; fingerprint=res.fingerprint))
    println("ran=", length(res.ran), " skipped=", length(res.skipped), " errors=", length(res.errors),
        " (data fingerprint ", res.fingerprint, ")")
    if !isempty(res.skipped)
        println("=" ^ 72)
        println("SKIPPED ", length(res.skipped), " of ", length(unique(study_hash.(runs; fingerprint=res.fingerprint))),
            " runs: their cached results in ", joinpath(outdir, "runs"), " were reused.")
        println("The cache cannot see code changes; after one, rerun with --fresh and check skipped = 0.")
        println("=" ^ 72)
    end
    ok = df[string.(df.status) .== "ok", :]
    isempty(ok) || println("mean wall seconds per run: ", mean(skipmissing(ok.wall_seconds)))
    isempty(res.errors) || (println(stderr, "study finished with ", length(res.errors), " errors"); exit(1))
    limits && limits_day_step(md, calib, runs)
    return nothing
end

main()
