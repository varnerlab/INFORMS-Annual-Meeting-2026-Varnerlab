# Diagnostic (talk, 2026-10-30): the calendar-only comparison of scripts/dev/calendar_baseline.jl, on the simulated
# sell-offs. On each of the 400 stress paths of results/synthetic/paths.csv (200 burst, 200 bear), this rebuilds the
# market and its calibration as script 22 did, takes the runs of the sell-off slide (talk slide 9: the ten pickers of
# TALK_REALRUN_ROWS × seven utilities × hedge on and off; 5 seeds for the bandit and random pickers, 1 otherwise) and
# reruns them with the band exit and the drawdown stop switched off (z = Inf, D_max = Inf). Before running a path it
# checks that every triggered run's hash is an ok row of that path's results/synthetic/rows/path_<p>.csv, so the
# market, the calibration and the settings match and a pair differs only in z and D_max.
# Writes results/calendar_baseline/synthetic/cal_<p>.csv (one row per calendar run; a path whose file exists is
# skipped, so a stopped run picks up where it left off) and, once every stress path is done,
# results/calendar_baseline/synthetic_vs_triggers.csv (one row per set, picker, utility and hedge setting).
# Arguments (at most one):
#   --limit N   run only the first N stress paths by run order, then summarize those
#   --summary   run nothing; summarize the paths already written
# Usage (from the repo root):
#   OMP_NUM_THREADS=1 KMP_GTID_MODE=2 julia --gcthreads=1 -t 30 scripts/dev/calendar_baseline_synthetic.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV, JLD2

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const SYN_DIR = joinpath(REPO, "results", "synthetic")
const ROWS_DIR = joinpath(SYN_DIR, "rows")
const OUT_DIR = joinpath(REPO, "results", "calendar_baseline", "synthetic")
const SUMMARY = joinpath(REPO, "results", "calendar_baseline", "synthetic_vs_triggers.csv")
const SETS = ["burst_stress", "bear_stress"]

"""
    calendar_only(runs)::Vector{MyStudyRun}

A copy of each run with the band exit and the drawdown stop switched off (`z = Inf`, `D_max = Inf`); every other
setting, the seed and the label (plus the suffix `|calendar`) are kept. The same as in calendar_baseline.jl.
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
    slide_runs(runs)::Vector{MyStudyRun}

The runs of `runs` whose picker is one of the sell-off slide's ten (`TALK_REALRUN_ROWS`).
"""
function slide_runs(runs)::Vector{MyStudyRun}
    keys_ = Set{String}()
    for row in TALK_REALRUN_ROWS
        push!(keys_, row[1])
    end
    out = MyStudyRun[]
    for r in runs
        key = picker_key((screen=r.screen, basket_K=r.settings.basket_K, label=r.label))
        if key in keys_
            push!(out, r)
        end
    end
    return out
end

"""
    run_path(gen, s, triggered, p::Int)::DataFrame

The calendar-only twins of `triggered` on stress path `p`: the market from `synthetic_market`, calibrated as script
22 calibrated it, after checking that every triggered run's hash is an ok row of `path_<p>.csv`. Throws an error if
one is not.
"""
function run_path(gen, s, triggered, p::Int)::DataFrame
    sm = synthetic_market(gen, p)
    calib = calibrate(sm.md; train_start=s.train_start, train_stop=s.train_stop, window=s.omega_window)
    fp = data_fingerprint(sm.md, calib)
    # every triggered run must be a stored ok row of this path
    stored = CSV.read(joinpath(ROWS_DIR, "path_$(p).csv"), DataFrame)
    ok = Set{String}()
    for i in 1:nrow(stored)
        if string(stored.status[i]) == "ok"
            push!(ok, string(stored.hash[i]))
        end
    end
    n_missing = 0
    for r in triggered
        if !(study_hash(r; fingerprint=fp) in ok)
            n_missing += 1
        end
    end
    if n_missing > 0
        error("path $(p): $(n_missing) of $(length(triggered)) triggered runs are not ok rows of path_$(p).csv")
    end
    # the calendar-only twins
    rows = run_synthetic_path(sm.md, calib, calendar_only(triggered))
    insertcols!(rows, 1, :path_id => p)
    return rows
end

"""
    path_pairs(p::Int, n_runs::Int)::DataFrame

One row per triggered run of path `p` paired with its calendar-only twin by label: `path_id`, `picker`, `allocator`,
`hedge`, `seed`, ending wealth ÷ start (`wealth_trig`, `wealth_cal`), worst drawdown as a fraction (`dd_trig`,
`dd_cal`) and re-plans after the first decision (`replans_trig`, `replans_cal`). Throws an error unless all `n_runs`
calendar rows are ok and each has its triggered twin.
"""
function path_pairs(p::Int, n_runs::Int)::DataFrame
    cal = CSV.read(joinpath(OUT_DIR, "cal_$(p).csv"), DataFrame)
    trig = CSV.read(joinpath(ROWS_DIR, "path_$(p).csv"), DataFrame)
    n_ok = count(==("ok"), string.(cal.status))
    if nrow(cal) != n_runs || n_ok != n_runs
        error("path $(p): cal_$(p).csv has $(nrow(cal)) rows, $(n_ok) ok; expected $(n_runs)")
    end
    trig_row = Dict{String,Int}()
    for i in 1:nrow(trig)
        if string(trig.status[i]) == "ok"
            trig_row[string(trig.label[i])] = i
        end
    end
    out = DataFrame(path_id=Int[], picker=String[], allocator=String[], hedge=Bool[], seed=Int[],
        wealth_trig=Float64[], wealth_cal=Float64[], dd_trig=Float64[], dd_cal=Float64[],
        replans_trig=Int[], replans_cal=Int[])
    for c in 1:nrow(cal)
        label = replace(string(cal.label[c]), "|calendar" => "")
        t = trig_row[label]
        push!(out, (p, picker_key(trig[t, :]), string(trig.allocator[t]), Bool(trig.hedge[t]), Int(trig.seed[t]),
            trig.terminal[t] / trig.W0[t], cal.terminal[c] / cal.W0[c], trig.max_drawdown[t], cal.max_drawdown[c],
            Int(trig.n_decisions[t]) - 1, Int(cal.n_decisions[c]) - 1))
    end
    return out
end

"""
    summarize(paths::DataFrame, ids::Vector{Int}, n_runs::Int)::Nothing

Pair the triggered and calendar-only runs of the stress paths `ids`, take the median over seeds within each path's
cell, and write `SUMMARY`: one row per set, picker, utility and hedge setting with the median across paths of each
measure, the share of paths where the triggered run ended higher (`trig_higher_share`) and the share where its worst
drawdown was smaller (`dd_lower_share`). Prints the headline counts for each set and hedge setting.
"""
function summarize(paths::DataFrame, ids::Vector{Int}, n_runs::Int)::Nothing
    # the set of each path
    set_of = Dict{Int,String}()
    for i in 1:nrow(paths)
        if paths.burst_stress[i]
            set_of[paths.path_id[i]] = "burst_stress"
        elseif paths.bear_stress[i]
            set_of[paths.path_id[i]] = "bear_stress"
        end
    end
    # within a path: the median over the seeds of each cell
    cells = DataFrame(set=String[], path_id=Int[], picker=String[], allocator=String[], hedge=Bool[],
        wealth_trig=Float64[], wealth_cal=Float64[], dd_trig=Float64[], dd_cal=Float64[],
        replans_trig=Float64[], replans_cal=Float64[])
    for p in ids
        pairs = path_pairs(p, n_runs)
        for g in groupby(pairs, [:picker, :allocator, :hedge]; sort=true)
            push!(cells, (set_of[p], p, g.picker[1], g.allocator[1], g.hedge[1], median(g.wealth_trig),
                median(g.wealth_cal), median(g.dd_trig), median(g.dd_cal), median(g.replans_trig),
                median(g.replans_cal)))
        end
    end
    # across paths: one row per set, picker, utility and hedge setting
    out = DataFrame(set=String[], picker=String[], allocator=String[], hedge=Bool[], n_paths=Int[],
        wealth_trig=Float64[], wealth_cal=Float64[], trig_higher_share=Float64[], dd_trig=Float64[],
        dd_cal=Float64[], dd_lower_share=Float64[], replans_trig=Float64[], replans_cal=Float64[])
    for g in groupby(cells, [:set, :picker, :allocator, :hedge]; sort=true)
        higher = 0
        lower = 0
        for i in 1:nrow(g)
            if g.wealth_trig[i] > g.wealth_cal[i]
                higher += 1
            end
            if g.dd_trig[i] < g.dd_cal[i]
                lower += 1
            end
        end
        push!(out, (g.set[1], g.picker[1], g.allocator[1], g.hedge[1], nrow(g), median(g.wealth_trig),
            median(g.wealth_cal), higher / nrow(g), median(g.dd_trig), median(g.dd_cal), lower / nrow(g),
            median(g.replans_trig), median(g.replans_cal)))
    end
    CSV.write(SUMMARY, out)
    println("wrote ", SUMMARY, " (", nrow(out), " rows, ", length(ids), " paths)")

    # the headline counts, each set and hedge setting
    for set in SETS
        for hedge in (false, true)
            h = out[(out.set .== set) .& (out.hedge .== hedge), :]
            c = cells[(cells.set .== set) .& (cells.hedge .== hedge), :]
            if nrow(h) == 0
                continue
            end
            wins = 0
            safer = 0
            for i in 1:nrow(h)
                if h.wealth_trig[i] > h.wealth_cal[i]
                    wins += 1
                end
                if h.dd_trig[i] < h.dd_cal[i]
                    safer += 1
                end
            end
            println(set, ", hedge ", hedge ? "on " : "off", " (", h.n_paths[1], " paths): triggered median ends higher in ",
                wins, " of ", nrow(h), " cells, median paired difference (triggered − calendar) ",
                round(median(c.wealth_trig .- c.wealth_cal); digits=3), "; worst drawdown smaller in ", safer, " of ",
                nrow(h), " cells, median ", round(median(c.dd_trig); digits=3), " triggered vs ",
                round(median(c.dd_cal); digits=3), " calendar; median re-plans ", median(c.replans_trig), " vs ",
                median(c.replans_cal))
        end
    end
    return nothing
end

function main(args=ARGS)
    # SCS_jll's libomp needs these, as in script 22
    if get(ENV, "OMP_NUM_THREADS", "") != "1" || get(ENV, "KMP_GTID_MODE", "") != "2"
        error("set OMP_NUM_THREADS=1 KMP_GTID_MODE=2 before starting Julia")
    end
    limit = nothing
    summary_only = false
    if length(args) == 2 && args[1] == "--limit"
        limit = parse(Int, args[2])
    elseif length(args) == 1 && args[1] == "--summary"
        summary_only = true
    elseif !isempty(args)
        error("use no argument, --limit N or --summary")
    end

    # the generator, the path table, the study settings and the slide's runs, as script 22 builds them
    gen = JLD2.load(joinpath(SYN_DIR, "generator.jld2"), "generator")
    paths = CSV.read(joinpath(SYN_DIR, "paths.csv"), DataFrame)
    s, grid = load_study(joinpath(REPO, "config", "study.toml"))
    if s.train_start != SYNTH_TRAIN[1] || s.train_stop != SYNTH_TRAIN[2]
        error("study training window is $(s.train_start) → $(s.train_stop); expected $(SYNTH_TRAIN[1]) → $(SYNTH_TRAIN[2])")
    end
    triggered = slide_runs(synthetic_runs(s, grid))
    if length(triggered) != 364
        error("expected 364 slide runs per path (6 pickers × 14 + 4 seeded pickers × 14 × 5), built ", length(triggered))
    end

    # the stress paths in run order
    stress = Int[]
    for i in 1:nrow(paths)
        if paths.burst_stress[i] || paths.bear_stress[i]
            push!(stress, i)
        end
    end
    stress = stress[sortperm(paths.run_order[stress])]
    ids = paths.path_id[stress]
    if limit !== nothing
        ids = ids[1:limit]
    end

    # one path at a time; a path whose file exists is skipped
    if !summary_only
        mkpath(OUT_DIR)
        todo = Int[]
        for p in ids
            if !isfile(joinpath(OUT_DIR, "cal_$(p).csv"))
                push!(todo, p)
            end
        end
        println(length(ids), " stress paths, ", length(ids) - length(todo), " already written, ", length(todo),
            " to run; ", length(triggered), " calendar runs per path; ", Threads.nthreads(), " threads")
        flush(stdout)
        walls = Float64[]
        for (k, p) in enumerate(todo)
            t0 = time()
            rows = run_path(gen, s, triggered, p)
            final = joinpath(OUT_DIR, "cal_$(p).csv")
            CSV.write(final * ".tmp", rows)
            mv(final * ".tmp", final; force=true)
            push!(walls, time() - t0)
            n_errors = count(==("error"), string.(rows.status))
            eta = now() + Second(round(Int, mean(walls) * (length(todo) - k)))
            println("path ", p, " (", k, " of ", length(todo), "): ", round(walls[end]; digits=1), " s, ", n_errors,
                " error rows; ETA ", Dates.format(eta, "yyyy-mm-dd HH:MM"))
            flush(stdout)
        end
    end

    # the summary over the paths written
    done = Int[]
    for p in ids
        if isfile(joinpath(OUT_DIR, "cal_$(p).csv"))
            push!(done, p)
        end
    end
    summarize(paths, done, length(triggered))
    return nothing
end

main()
