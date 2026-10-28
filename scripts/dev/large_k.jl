# Diagnostic (talk Q&A, 2026-10-28): does the global γ edge over random picks hold for large books? Runs the
# universe-wide screens global γ and random (whole universe) at K = 100 and 200 names on the real 2025–26 window,
# every allocator, hedge on and off, with the study's settings and the study's 30 seeds for both screens (global γ
# runs 30 times, as in scripts/dev/gamma_seeds.jl, so that each B2 cell compares 30 runs with 30). The study's own
# grid stops at K = 34. Writes results/large_k/large_k_summary.csv (run cache in results/large_k/runs/);
# config/study.toml and the committed results are not touched.
# Usage (from the repo root): OMP_NUM_THREADS=1 KMP_GTID_MODE=2 julia -t 30 scripts/dev/large_k.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const OUT_DIR = joinpath(REPO, "results", "large_k")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)

"""
    with_seeds(runs, seeds)::Vector{MyStudyRun}

One copy of each run per seed in `seeds`, with the seed and the label's `seed=` part replaced (as in
scripts/dev/gamma_seeds.jl).
"""
function with_seeds(runs, seeds)::Vector{MyStudyRun}
    out = MyStudyRun[]
    for r in runs
        for seed in seeds
            label = replace(r.label, r"seed=\d+" => "seed=$(seed)")
            push!(out, build(MyStudyRun, (settings=r.settings, allocator=r.allocator, screen=r.screen,
                seed=seed, label=label)))
        end
    end
    return out
end

function main()
    # SCS_jll's libomp needs these, as in script 22
    if get(ENV, "OMP_NUM_THREADS", "") != "1" || get(ENV, "KMP_GTID_MODE", "") != "2"
        error("set OMP_NUM_THREADS=1 KMP_GTID_MODE=2 before starting Julia")
    end

    # load the real data, the study settings and the calibration, as script 11 does
    md = load_market_data(joinpath(REPO, "data"); start=MD_START, stop=MD_STOP)
    s, grid = load_study(joinpath(REPO, "config", "study.toml"))
    calib = load_calibration(joinpath(REPO, "results", "calibration.jld2"))

    # the study grid with only the two large basket sizes and the two universe-wide screens; the study builds random
    # picks with its 30 seeds and global γ once (seed base_seed), so global γ gets the same 30 seeds here
    seeds = collect(grid.base_seed .+ (0:grid.n_seeds-1))
    big = merge(grid, (K=[100, 200],))
    random_runs = universe_runs(s, merge(big, (universe_screens=[:random_global],)))
    global_runs = with_seeds(universe_runs(s, merge(big, (universe_screens=[:global_gamma],))), seeds)
    runs = vcat(global_runs, random_runs)
    println(length(runs), " runs, ", Threads.nthreads(), " threads")

    # run, then write one summary row per run
    res = run_study_grid(md, calib, runs; outdir=OUT_DIR)
    df = rebuild_study_summary(OUT_DIR, "large_k"; hashes=study_hash.(runs; fingerprint=res.fingerprint))
    println("ran=", length(res.ran), " skipped=", length(res.skipped), " errors=", length(res.errors))
    ok = df[string.(df.status) .== "ok", :]
    if !isempty(ok)
        println("mean wall seconds per run: ", mean(skipmissing(ok.wall_seconds)))
    end
    if !isempty(res.errors)
        println(stderr, "finished with ", length(res.errors), " errors")
        exit(1)
    end
    return nothing
end

main()
