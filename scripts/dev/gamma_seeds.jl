# Diagnostic (talk, 2026-10-28): how much does the MPC projection seed move the deterministic pickers? The study runs
# γ per sector, global γ, correlation-aware γ, liquid then γ and the two fixed lists once (seed base_seed), while the
# random pickers and the bandit get 30 seeds. For these pickers the seed only drives the projection paths behind the
# MPC band. This reruns them on the real 2025–26 window with the same 30 seeds as the random pickers: γ per sector
# (33 names), correlation-aware γ and liquid then γ at K = 13, global γ at every study K (5, 8, 13, 21, 34) and the
# two fixed lists; every allocator, hedge on and off. Writes results/gamma_seeds/gamma_seeds_summary.csv (run cache
# in results/gamma_seeds/runs/); config/study.toml and the committed results are not touched.
# Usage (from the repo root): OMP_NUM_THREADS=1 KMP_GTID_MODE=2 julia -t 30 scripts/dev/gamma_seeds.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const OUT_DIR = joinpath(REPO, "results", "gamma_seeds")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)

"""
    with_seeds(runs, seeds)::Vector{MyStudyRun}

One copy of each run per seed in `seeds`, with the seed and the label's `seed=` part replaced.
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
    seeds = collect(grid.base_seed .+ (0:grid.n_seeds-1))

    # the deterministic pickers, each built once by the study's own run builders
    sector = filter(r -> r.screen == :gamma, tier1_runs(s, grid))
    global_all = universe_runs(s, merge(grid, (universe_screens=[:global_gamma],)))
    other13 = universe_runs(s, merge(grid, (K=[13], universe_screens=[:corr_gamma, :liquid_gamma])))
    fixed = fixed_list_runs(s, grid)
    base = vcat(sector, global_all, other13, fixed)

    # every one of them with the 30 seeds of the random pickers
    runs = with_seeds(base, seeds)
    println(length(base), " picker cells × ", length(seeds), " seeds = ", length(runs), " runs, ",
        Threads.nthreads(), " threads")

    # run, then write one summary row per run
    res = run_study_grid(md, calib, runs; outdir=OUT_DIR)
    df = rebuild_study_summary(OUT_DIR, "gamma_seeds"; hashes=study_hash.(runs; fingerprint=res.fingerprint))
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
