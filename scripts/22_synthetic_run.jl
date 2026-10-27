# Task 9 of the synthetic-market plan: the study on each synthetic market path (spec §5). The paths come from
# results/synthetic/paths.csv in run order. For each path the script builds the market, recalibrates it as the study
# would, runs the 532 study runs of `synthetic_runs` and writes three files:
#   path_<p>.csv   one summary row per run (the columns of the real summaries, plus path_id, the set flags and error);
#   bench_<p>.csv  SPY buy-and-hold and equal-weight universe buy-and-hold on the path;
#   meta_<p>.csv   λ, timings (s), clip share, names with online β̂ ≤ 0, error rows and threads.
# A path whose path_<p>.csv exists is skipped, so a stopped run picks up where it left off. A path whose work throws
# gets error_<p>.txt (the message and backtrace) and no path_<p>.csv; the run moves on, and the next launch retries it.
# A calibration over 30 s switches the remaining paths to the fixed-λ calibration (λ = 0). That switch holds only
# within one process: a relaunch starts again with the full calibration.
#
# Arguments (at most one):
#   --smoke     the first 20 paths by run order, written to results/synthetic/smoke/rows/
#   --limit N   shrink the typical set to typical ranks 1…N; both stress sets stay whole
#   --extra M   after every set path, also run typical ranks 601…600+M, by rank; they join the typical set
# Without --smoke the files go to results/synthetic/rows/. The `typical` flag written with each path is
# typical_rank ≤ 600, or ≤ N with --limit N, or ≤ 600 + M with --extra M.
#
# Run: OMP_NUM_THREADS=1 KMP_GTID_MODE=2 julia --gcthreads=1 --project=code -t 30 scripts/22_synthetic_run.jl [--smoke | --limit N | --extra M]

# The SCS fallback solver loads libomp. Without these two settings, libomp gives one thread id to several Julia
# threads and crashes, or the run hangs. Stop before any package loads unless both are set.
if get(ENV, "OMP_NUM_THREADS", "") != "1" || get(ENV, "KMP_GTID_MODE", "") != "2"
    error("script 22 needs OMP_NUM_THREADS=1 and KMP_GTID_MODE=2 (got OMP_NUM_THREADS=",
        get(ENV, "OMP_NUM_THREADS", "(unset)"), ", KMP_GTID_MODE=", get(ENV, "KMP_GTID_MODE", "(unset)"), ").\n",
        "Without them the libomp that the SCS solver loads gives one thread id to several Julia threads, which crashes ",
        "the run (an OMP assertion or a segfault), and its oversubscribed OpenMP threads can hang it.\n",
        "Launch with:\n",
        "OMP_NUM_THREADS=1 KMP_GTID_MODE=2 nohup julia --gcthreads=1 --project=code -t 30 scripts/22_synthetic_run.jl ",
        "> results/synthetic/run.log 2>&1 &")
end

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV, JLD2

const REPO = normpath(joinpath(@__DIR__, ".."))
const SYN_DIR = joinpath(REPO, "results", "synthetic")
const ROWS_DIR = joinpath(SYN_DIR, "rows")
const SMOKE_DIR = joinpath(SYN_DIR, "smoke", "rows")
const CONFIG = joinpath(REPO, "config", "study.toml")
const N_SMOKE = 20        # paths in the smoke run
const N_TYPICAL = 600     # typical ranks in the typical set unless --limit or --extra changes it
const CALIB_MAX = 30.0    # s; after a slower calibration the rest of the run uses the fixed-λ calibration

"""
    parse_args(args::Vector{String})::NamedTuple{(:smoke, :limit, :extra)}

The script's options: `smoke` (true for `--smoke`), `limit` (the N of `--limit N`, else `nothing`) and `extra` (the M
of `--extra M`, else 0). Throws `ArgumentError` for an unknown argument, a count that is missing or not a whole
number ≥ 0, a `--limit` above 600, or more than one option.
"""
function parse_args(args::Vector{String})
    smoke = false
    limit = nothing
    extra = 0
    n_options = 0
    k = 1
    while k <= length(args)
        a = args[k]
        if a == "--smoke"
            smoke = true
            k += 1
        elseif a == "--limit" || a == "--extra"
            if k == length(args)
                throw(ArgumentError("$(a) needs a count"))
            end
            n = tryparse(Int, args[k + 1])
            if n === nothing || n < 0
                throw(ArgumentError("$(a) needs a whole number ≥ 0, got $(args[k + 1])"))
            end
            if a == "--limit"
                limit = n
            else
                extra = n
            end
            k += 2
        else
            throw(ArgumentError("unknown argument $(a); use --smoke, --limit N or --extra M"))
        end
        n_options += 1
    end
    if n_options > 1
        throw(ArgumentError("pass at most one of --smoke, --limit N and --extra M"))
    end
    if limit !== nothing && limit > N_TYPICAL
        throw(ArgumentError("--limit $(limit) is above the $(N_TYPICAL) typical ranks; use --extra to grow the typical set"))
    end
    return (smoke=smoke, limit=limit, extra=extra)
end

"""
    select_paths(paths::DataFrame, opts)::NamedTuple{(:rows, :n_typical)}

The rows of `paths` (from paths.csv) to run, in the order to run them, and the size of the typical set (typical ranks
1…`n_typical`). The set paths run in `run_order`. With `--smoke` only the first 20 run. With `--limit N` a set path
runs only if it is in a stress set or its typical rank is ≤ N. With `--extra M` the paths in no set with typical
ranks 601…600+M follow, by rank.
"""
function select_paths(paths::DataFrame, opts)
    # the set paths, in run order
    in_set = Int[]
    for i in 1:nrow(paths)
        if !ismissing(paths.run_order[i])
            push!(in_set, i)
        end
    end
    order = in_set[sortperm(collect(skipmissing(paths.run_order[in_set])))]
    if opts.smoke
        return (rows=order[1:N_SMOKE], n_typical=N_TYPICAL)
    end
    # --limit N: the stress paths and typical ranks 1…N
    if opts.limit !== nothing
        keep = Int[]
        for i in order
            if paths.burst_stress[i] || paths.bear_stress[i] || paths.typical_rank[i] <= opts.limit
                push!(keep, i)
            end
        end
        return (rows=keep, n_typical=opts.limit)
    end
    # --extra M: typical ranks 601…600+M of the paths in no set, by rank, after every set path
    extra = Int[]
    for i in 1:nrow(paths)
        rank = paths.typical_rank[i]
        if ismissing(paths.run_order[i]) && rank > N_TYPICAL && rank <= N_TYPICAL + opts.extra
            push!(extra, i)
        end
    end
    extra = extra[sortperm(paths.typical_rank[extra])]
    return (rows=vcat(order, extra), n_typical=N_TYPICAL + opts.extra)
end

"""
    run_path(gen, s, runs, row, n_typical::Int, fixed_λ::Bool)::NamedTuple{(:rows, :bench, :meta)}

The study on path `row.path_id` (`row` is a row of paths.csv): the market from `synthetic_market`, its calibration
(`calibrate` on the study's training window, or with `fixed_λ` the SIM fit alone with λ = 0), the study runs `runs`
from `run_synthetic_path`, and the buy-and-hold references from `passive_references`. `rows` and `bench` lead with
`path_id`, and `rows` also carries the set flags (`typical` is typical rank ≤ `n_typical`). `meta` is one row:
`path_id`, the set flags, `lambda` (dimensionless), `lambda_fixed`, `gen_seconds`, `calib_seconds` and
`grid_seconds` (s), `clip_share` (dimensionless), `n_beta_le0` (names whose online β̂ ≤ 0 on some day of the trading
window), `n_errors` (error rows) and `threads`.
"""
function run_path(gen, s, runs, row, n_typical::Int, fixed_λ::Bool)
    p = row.path_id
    burst = row.burst_stress
    bear = row.bear_stress
    typical = row.typical_rank <= n_typical
    # 1. the market
    t0 = time()
    sm = synthetic_market(gen, p)
    gen_seconds = time() - t0
    # 2. the calibration, as the study calibrates the real market, or the fixed-λ fallback
    t1 = time()
    if fixed_λ
        fit = fit_base_sim(sm.md, s.train_start, s.train_stop)
        calib = build(MyCalibration, (base_sim=fit.params, σ_m=fit.σ_m, m_m=fit.m_m, v_m=fit.v_m, λ=0.0,
            train_start=s.train_start, train_stop=s.train_stop))
    else
        calib = calibrate(sm.md; train_start=s.train_start, train_stop=s.train_stop, window=s.omega_window)
    end
    calib_seconds = time() - t1
    # names whose online β̂ reaches zero on some day of the trading window
    n_beta_le0 = VLINFORMS2026.count_beta_le0(online_sim(sm.md, calib; start=s.start, stop=s.stop))
    # 3. the study runs, threaded within the path
    t2 = time()
    rows = run_synthetic_path(sm.md, calib, runs)
    grid_seconds = time() - t2
    insertcols!(rows, 1, :path_id => p, :burst_stress => burst, :bear_stress => bear, :typical => typical)
    # 4. the buy-and-hold references
    bench = passive_references(sm.md, s.start, s.stop; W0=s.W0)
    insertcols!(bench, 1, :path_id => p)
    # 5. the meta row
    n_errors = count(==("error"), rows.status)
    meta = DataFrame(path_id=[p], burst_stress=[burst], bear_stress=[bear], typical=[typical], lambda=[calib.λ],
        lambda_fixed=[fixed_λ], gen_seconds=[gen_seconds], calib_seconds=[calib_seconds], grid_seconds=[grid_seconds],
        clip_share=[sm.clip_share], n_beta_le0=[n_beta_le0], n_errors=[n_errors], threads=[Threads.nthreads()])
    return (rows=rows, bench=bench, meta=meta)
end

"""
    write_path_error(dir::AbstractString, p::Int, text::AbstractString)::Nothing

Write `text` (why path `p` failed: the message and backtrace) to `dir/error_<p>.txt`, through `error_<p>.txt.tmp`
and a move. No `path_<p>.csv` is written, so `path_done` stays false and the next launch retries the path.
"""
function write_path_error(dir::AbstractString, p::Int, text::AbstractString)::Nothing
    mkpath(dir)
    final = joinpath(dir, "error_$(p).txt")
    tmp = final * ".tmp"
    write(tmp, text)
    mv(tmp, final; force=true)
    return nothing
end

function main(args=ARGS)
    t_start = time()
    opts = parse_args(collect(String, args))
    if opts.smoke
        outdir = SMOKE_DIR
    else
        outdir = ROWS_DIR
    end

    # the generator, the path table, the study settings and the run list
    gen = JLD2.load(joinpath(SYN_DIR, "generator.jld2"), "generator")
    paths = CSV.read(joinpath(SYN_DIR, "paths.csv"), DataFrame)
    s, grid = load_study(CONFIG)
    if s.train_start != SYNTH_TRAIN[1] || s.train_stop != SYNTH_TRAIN[2]
        error("study training window is $(s.train_start) → $(s.train_stop); expected $(SYNTH_TRAIN[1]) → $(SYNTH_TRAIN[2])")
    end
    runs = synthetic_runs(s, grid)

    # the paths to run; the ones already written are skipped
    sel = select_paths(paths, opts)
    todo = Int[]
    n_skipped = 0
    for i in sel.rows
        if path_done(outdir, paths.path_id[i])
            n_skipped += 1
        else
            push!(todo, i)
        end
    end
    println(length(sel.rows), " paths selected (typical ranks 1…", sel.n_typical, "), ", n_skipped,
        " skipped (already written), ", length(todo), " to run; ", length(runs), " runs per path; ",
        Threads.nthreads(), " threads, ", Threads.ngcthreads(), " GC threads; writing to ", outdir)
    flush(stdout)

    # one path at a time
    fixed_λ = false
    walls = Float64[]
    n_error_rows = 0
    n_failed = 0
    for (k, i) in enumerate(todo)
        row = paths[i, :]
        t0 = time()
        # the path's work; a failure is logged to error_<p>.txt and the run moves on (Ctrl-C still stops it)
        res = nothing
        failure = ""
        try
            res = run_path(gen, s, runs, row, sel.n_typical, fixed_λ)
            write_path_files(outdir, row.path_id, res.rows, res.bench, res.meta)
        catch err
            if err isa InterruptException
                rethrow()
            end
            bt = catch_backtrace()
            failure = "path $(row.path_id) failed:\n" * sprint(showerror, err) * "\n" * sprint(Base.show_backtrace, bt) * "\n"
        end
        if failure != ""
            n_failed += 1
            write_path_error(outdir, row.path_id, failure)
            println("path ", row.path_id, " (", k, " of ", length(todo), "): FAILED; no path file written, the next launch ",
                "retries it; details in error_", row.path_id, ".txt")
            print(failure)
            flush(stdout)
            continue
        end
        # the path succeeded: drop a stale error file from an earlier failed attempt
        rm(joinpath(outdir, "error_$(row.path_id).txt"); force=true)
        wall = time() - t0
        push!(walls, wall)
        m = res.meta[1, :]
        n_error_rows += m.n_errors
        remaining = mean(walls) * (length(todo) - k)
        λ_note = ""
        if m.lambda_fixed
            λ_note = " (fixed)"
        end
        println("path ", row.path_id, " (", k, " of ", length(todo), "): λ = ", round(m.lambda; digits=4),
            λ_note, ", gen ", round(m.gen_seconds; digits=1), " s, calib ",
            round(m.calib_seconds; digits=1), " s, grid ", round(m.grid_seconds; digits=1), " s, total ",
            round(wall; digits=1), " s, ", m.n_errors, " error rows; ETA ",
            Dates.format(now() + Second(round(Int, remaining)), "yyyy-mm-dd HH:MM"))
        # the first few error messages of the path
        n_shown = 0
        for r in eachrow(res.rows)
            if r.status == "error" && n_shown < 5
                println("  error in ", r.label, ": ", r.error)
                n_shown += 1
            end
        end
        # a slow calibration switches the rest of the run to fixed λ
        if !fixed_λ && m.calib_seconds > CALIB_MAX
            fixed_λ = true
            println("calibration took ", round(m.calib_seconds; digits=1), " s (> ", CALIB_MAX,
                " s): the remaining paths use the fixed-λ calibration (λ = 0)")
        end
        flush(stdout)
    end

    # the totals
    println("ran ", length(walls), ", failed ", n_failed, ", skipped ", n_skipped, ", error rows ", n_error_rows)
    if !isempty(walls)
        println("seconds per path: mean ", round(mean(walls); digits=1), ", median ", round(median(walls); digits=1),
            ", first path ", round(walls[1]; digits=1))
    end
    println("wall time: ", round(time() - t_start; digits=1), " s")
    return nothing
end

main()
