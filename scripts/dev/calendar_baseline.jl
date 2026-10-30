# Diagnostic (talk, 2026-10-30): did the re-plan triggers do better than re-planning on the calendar alone? This reruns
# the real-data grid behind the backtest slide (talk slide 6: ten pickers × seven utilities × hedge on and off × seeds
# 5660–5689) with the band exit and the drawdown stop switched off (z = Inf, D_max = Inf), so a book re-plans only when
# the 21-day horizon elapses, its basket changes on the 21-day re-pick timer, or an all-cash book reaches its 21-day
# revisit. Each calendar run is paired with its triggered twin (same picker, utility, hedge and seed) from the stored
# study summaries; the script first checks that every twin's hash is in those summaries, so the pairs differ only in
# z and D_max. Writes results/calendar_baseline/calendar_baseline_summary.csv (run cache in
# results/calendar_baseline/runs/) and results/calendar_baseline/calendar_vs_triggers.csv (one row per picker,
# utility and hedge cell); config/study.toml and the committed results are not touched.
# Usage (from the repo root): OMP_NUM_THREADS=1 KMP_GTID_MODE=2 julia -t 30 scripts/dev/calendar_baseline.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const OUT_DIR = joinpath(REPO, "results", "calendar_baseline")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const STORED = [joinpath(REPO, "results", "tier1_summary.csv"), joinpath(REPO, "results", "universe_summary.csv"),
    joinpath(REPO, "results", "gamma_seeds", "gamma_seeds_summary.csv")]

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

"""
    calendar_only(runs)::Vector{MyStudyRun}

A copy of each run with the band exit and the drawdown stop switched off (`z = Inf`, `D_max = Inf`); every other
setting, the seed and the label (plus the suffix `|calendar`) are kept.
"""
function calendar_only(runs)::Vector{MyStudyRun}
    out = MyStudyRun[]
    for r in runs
        sc = with_settings(r.settings; z=Inf, D_max=Inf)
        push!(out, build(MyStudyRun, (settings=sc, allocator=r.allocator, screen=r.screen, seed=r.seed,
            label=r.label * "|calendar")))
    end
    return out
end

"""
    cell_name(label::AbstractString)::String

A run label without its `seed=` part and without the `|calendar` suffix, e.g. `"CD|global_gamma|K=13|hedge=false"`.
"""
function cell_name(label::AbstractString)::String
    parts = String[]
    for part in split(label, "|")
        if startswith(part, "seed=") || part == "calendar"
            continue
        end
        push!(parts, String(part))
    end
    return join(parts, "|")
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

    # the slide's ten pickers, one run per picker, utility and hedge setting, built by the study's own run builders
    sector = filter(r -> r.seed == grid.base_seed, tier1_runs(s, grid))
    universe = filter(r -> r.seed == grid.base_seed, universe_runs(s, merge(grid, (K=[13],))))
    fixed = filter(r -> occursin("fixed:l8b_13|", r.label) || occursin("fixed:l8b_15|", r.label),
        fixed_list_runs(s, grid))
    base = vcat(sector, universe, fixed)
    if length(base) != 140
        error("expected 140 picker cells (10 pickers × 7 utilities × 2 hedge settings), built ", length(base))
    end

    # the triggered runs with the 30 seeds, and their calendar-only twins in the same order
    triggered = with_seeds(base, seeds)
    calendar = calendar_only(triggered)
    println(length(base), " picker cells × ", length(seeds), " seeds = ", length(calendar), " calendar runs, ",
        Threads.nthreads(), " threads")

    # every triggered run must already be stored: same settings, seed and data, so a pair differs only in z and D_max
    fp = data_fingerprint(md, calib)
    stored = DataFrame()
    for path in STORED
        stored = vcat(stored, CSV.read(path, DataFrame); cols=:union)
    end
    stored = stored[string.(stored.status) .== "ok", :]
    row_of = Dict{String,Int}()
    for i in 1:nrow(stored)
        row_of[string(stored.hash[i])] = i
    end
    missing_twins = 0
    for r in triggered
        if !haskey(row_of, study_hash(r; fingerprint=fp))
            missing_twins += 1
        end
    end
    if missing_twins > 0
        error(missing_twins, " triggered runs are not in the stored summaries; the grid does not match slide 6")
    end
    println("all ", length(triggered), " triggered twins found in the stored summaries")

    # run the calendar-only grid, then write one summary row per run
    res = run_study_grid(md, calib, calendar; outdir=OUT_DIR)
    df = rebuild_study_summary(OUT_DIR, "calendar_baseline"; hashes=study_hash.(calendar; fingerprint=fp))
    println("ran=", length(res.ran), " skipped=", length(res.skipped), " errors=", length(res.errors))
    if !isempty(res.errors)
        println(stderr, "finished with ", length(res.errors), " errors")
        exit(1)
    end
    cal_row = Dict{String,Int}()
    for i in 1:nrow(df)
        cal_row[string(df.hash[i])] = i
    end

    # pair each calendar run with its triggered twin: ending wealth ÷ start, worst drawdown, and re-plan count
    pairs = DataFrame(cell=String[], seed=Int[], hedge=Bool[], wealth_trig=Float64[], wealth_cal=Float64[],
        dd_trig=Float64[], dd_cal=Float64[], replans_trig=Int[], replans_cal=Int[])
    for k in eachindex(triggered)
        t = row_of[study_hash(triggered[k]; fingerprint=fp)]
        c = cal_row[study_hash(calendar[k]; fingerprint=fp)]
        push!(pairs, (cell_name(triggered[k].label), triggered[k].seed, triggered[k].settings.hedge,
            stored.terminal[t] / stored.W0[t], df.terminal[c] / df.W0[c],
            stored.max_drawdown[t], df.max_drawdown[c],
            Int(stored.n_decisions[t]) - 1, Int(df.n_decisions[c]) - 1))
    end

    # one row per cell: the medians over the 30 seeds, and how often the triggered run ended higher
    cells = DataFrame(cell=String[], hedge=Bool[], wealth_trig=Float64[], wealth_cal=Float64[],
        trig_higher_share=Float64[], dd_trig=Float64[], dd_cal=Float64[], replans_trig=Float64[],
        replans_cal=Float64[])
    for g in groupby(pairs, :cell; sort=true)
        higher = 0
        for i in 1:nrow(g)
            if g.wealth_trig[i] > g.wealth_cal[i]
                higher += 1
            end
        end
        push!(cells, (g.cell[1], g.hedge[1], median(g.wealth_trig), median(g.wealth_cal), higher / nrow(g),
            median(g.dd_trig), median(g.dd_cal), median(g.replans_trig), median(g.replans_cal)))
    end
    CSV.write(joinpath(OUT_DIR, "calendar_vs_triggers.csv"), cells)
    println("wrote ", joinpath(OUT_DIR, "calendar_vs_triggers.csv"), " (", nrow(cells), " cells)")

    # the headline counts, hedge off and hedge on
    for hedge in (false, true)
        h = cells[cells.hedge .== hedge, :]
        p = pairs[pairs.hedge .== hedge, :]
        wins = 0
        for i in 1:nrow(h)
            if h.wealth_trig[i] > h.wealth_cal[i]
                wins += 1
            end
        end
        println("hedge ", hedge ? "on " : "off", ": triggered median ends higher in ", wins, " of ", nrow(h),
            " cells; median paired difference (triggered − calendar) ", round(median(p.wealth_trig .- p.wealth_cal);
            digits=3), "; median worst drawdown ", round(median(p.dd_trig); digits=3), " triggered vs ",
            round(median(p.dd_cal); digits=3), " calendar; median re-plans ", median(p.replans_trig), " vs ",
            median(p.replans_cal))
    end
    return nothing
end

main()
