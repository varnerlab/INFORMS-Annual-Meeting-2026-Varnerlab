# Task 11 of the synthetic-market plan: the stress-test tables (spec §7) from the per-path files of script 22.
# Reads a rows directory (default results/synthetic/rows/) and writes tables.md and tables/*.csv in its parent
# directory (results/synthetic/ for the default, results/synthetic/smoke/ for the smoke rows).
# A path counts only when path_<p>.csv, bench_<p>.csv and meta_<p>.csv all exist and path_<p>.csv holds 532 run rows.
# inflight_<p>.txt markers are ignored. Nothing is optimized here, so no thread settings are needed.
#
# Run: julia --project=code scripts/23_synthetic_tables.jl [rows directory]
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV

const REPO = normpath(joinpath(@__DIR__, ".."))
const SYN_DIR = joinpath(REPO, "results", "synthetic")
const RESULTS = joinpath(REPO, "results")
const DATA_DIR = joinpath(REPO, "data")
const CONFIG = joinpath(REPO, "config", "study.toml")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const N_RUNS = 532            # run rows in a complete path file (synthetic_runs with 5 seeds)
const REAL_SEEDS = 5660:5664  # the seeds of the synthetic runs, kept from the real summaries
# The columns of a run row that the tables use.
const ROW_COLUMNS = [:path_id, :label, :status, :allocator, :screen, :seed, :hedge, :basket_K, :W0, :terminal,
    :realized_growth, :max_drawdown, :sharpe]
# The sets: the meta flag, the heading and what the set holds.
const SETS = [
    ("typical", "Typical futures", "paths drawn at random from the 10,000 synthetic markets"),
    ("burst_stress", "Burst stress", "the deepest SPY sell-offs in 2025–26 among the futures with a jump burst, " *
        "a run of large daily moves that lasts months"),
    ("bear_stress", "Bear stress", "the deepest SPY sell-offs in 2025–26 among the futures without a jump burst")]
const HEDGES = [(false, "off"), (true, "on")]

"""
    load_paths(dir::AbstractString, W0::Real)::NamedTuple

The complete paths of the rows directory `dir`: a path counts when its path, bench and meta files all exist and its
path file has `N_RUNS` rows. Returns `cells` (`path_cells` of every counted path), `bench` (`bench_cells`, with the
starting wealth `W0` in USD), `meta` (the meta rows), `n_incomplete` (paths with some of the three files that do
not count), `n_run_rows` and `n_error_rows` (run rows of the counted paths, and those with `status == "error"`).
Throws an error if an ok run row has a starting wealth other than `W0`.
"""
function load_paths(dir::AbstractString, W0::Real)::NamedTuple
    # every path id that has a path, bench or meta file
    found = Set{Int}()
    for f in readdir(dir)
        m = match(r"^(path|bench|meta)_(\d+)\.csv$", f)
        if m !== nothing
            push!(found, parse(Int, m.captures[2]))
        end
    end
    seen = sort(collect(found))
    # the complete paths
    cells = DataFrame[]
    bench = DataFrame[]
    meta = DataFrame[]
    n_incomplete = 0
    n_run_rows = 0
    n_error_rows = 0
    for p in seen
        fpath = joinpath(dir, "path_$(p).csv")
        fbench = joinpath(dir, "bench_$(p).csv")
        fmeta = joinpath(dir, "meta_$(p).csv")
        if !(isfile(fpath) && isfile(fbench) && isfile(fmeta))
            n_incomplete += 1
            continue
        end
        rows = CSV.read(fpath, DataFrame; select=ROW_COLUMNS)
        if nrow(rows) != N_RUNS
            n_incomplete += 1
            continue
        end
        for i in 1:nrow(rows)
            if rows.status[i] == "ok" && rows.W0[i] != W0
                error("path $(p): an ok run row starts with $(rows.W0[i]) USD, not $(W0)")
            end
            if rows.status[i] == "error"
                n_error_rows += 1
            end
        end
        n_run_rows += nrow(rows)
        push!(cells, path_cells(rows))
        push!(bench, bench_cells(CSV.read(fbench, DataFrame); W0=W0))
        push!(meta, CSV.read(fmeta, DataFrame))
    end
    if isempty(cells)
        error("no complete path in ", dir)
    end
    return (cells=reduce(vcat, cells), bench=reduce(vcat, bench), meta=reduce(vcat, meta; cols=:union),
        n_incomplete=n_incomplete, n_run_rows=n_run_rows, n_error_rows=n_error_rows)
end

"""
    real_cells(W0::Real)::DataFrame

`path_cells` of the real 2025–26 run: the rows of results/tier1_summary.csv and results/universe_summary.csv with
seeds in `REAL_SEEDS`, as one path with `path_id` 0. Throws an error if a `PICKER_ROWS` picker lacks a cell for some
utility and hedge setting, or a real row has a starting wealth other than `W0` (USD).
"""
function real_cells(W0::Real)::DataFrame
    parts = DataFrame[]
    for f in ("tier1_summary.csv", "universe_summary.csv")
        d = CSV.read(joinpath(RESULTS, f), DataFrame; select=ROW_COLUMNS[2:end])
        keep = Bool[]
        for s in d.seed
            push!(keep, s in REAL_SEEDS)
        end
        push!(parts, d[keep, :])
    end
    d = reduce(vcat, parts; cols=:union)
    if any(d.W0 .!= W0)
        error("a real summary row starts with a wealth other than ", W0, " USD")
    end
    d.path_id = zeros(Int, nrow(d))
    cells = path_cells(d)
    # every picker of the tables has a real cell for every utility and hedge setting
    for (key, name) in PICKER_ROWS
        for a in TABLE_ALLOCATOR_ORDER
            for (hedge, hedge_text) in HEDGES
                n = count((cells.picker .== key) .& (cells.allocator .== a) .& (cells.hedge .== hedge))
                if n != 1
                    error("real run: ", n, " cells for ", name, ", ", a, ", hedge ", hedge_text)
                end
            end
        end
    end
    return cells
end

"""
    check_grid(t::DataFrame, what::AbstractString)::Nothing

Throws an error unless the grid `t` has the 2 reference rows and the 14 picker rows, and a first column plus the 7
utility columns; `what` names the grid in the message.
"""
function check_grid(t::DataFrame, what::AbstractString)::Nothing
    if nrow(t) != 2 + length(PICKER_ROWS) || ncol(t) != 1 + length(TABLE_ALLOCATOR_ORDER)
        error(what, ": expected ", 2 + length(PICKER_ROWS), " rows and ", 1 + length(TABLE_ALLOCATOR_ORDER),
            " columns, got ", nrow(t), " and ", ncol(t))
    end
    return nothing
end

"""
    write_table(io::IO, tabdir::AbstractString, file::AbstractString, title::AbstractString, csv_table, md_table)::Nothing

Write `csv_table` to `tabdir/file.csv`, and `title` in bold followed by `md_table` as Markdown to `io`.
"""
function write_table(io::IO, tabdir::AbstractString, file::AbstractString, title::AbstractString, csv_table,
        md_table)::Nothing
    CSV.write(joinpath(tabdir, file * ".csv"), csv_table)
    println(io, "**", title, "**")
    println(io)
    print(io, markdown_table(md_table))
    println(io)
    return nothing
end

"""
    write_set(io::IO, tabdir::AbstractString, data, set::AbstractString, ids::Vector{Int})::Nothing

The tables of one set (`set` is its meta flag; `ids` its paths): for each hedge setting the median ending wealth, the
share of paths beating SPY, growth, max drawdown and Sharpe as bands, and the head-to-heads; then hedge on minus off.
"""
function write_set(io::IO, tabdir::AbstractString, data, set::AbstractString, ids::Vector{Int})::Nothing
    cells = data.cells
    bench = data.bench
    H = head_to_head(cells; ids=ids)
    CSV.write(joinpath(tabdir, set * "_head_to_head.csv"), H)
    for (hedge, hedge_text) in HEDGES
        stem = set * "_hedge_" * hedge_text
        println(io, "### Hedge ", hedge_text)
        println(io)
        med = grid_table(cells, bench, :wealth; ids=ids, hedge=hedge, stat=:median)
        beat = grid_table(cells, bench, :wealth; ids=ids, hedge=hedge, stat=:beat_spy)
        check_grid(med, stem * " wealth")
        check_grid(beat, stem * " beat SPY")
        write_table(io, tabdir, stem * "_wealth_median", "Ending wealth ÷ start, median across paths. Bold: the " *
            "cell beats that path's SPY on more than half of the paths.", med, format_grid(med; beat=beat))
        write_table(io, tabdir, stem * "_wealth_beat_spy", "Share of paths on which the ending wealth beats that " *
            "path's SPY.", beat, format_grid(beat))
        bands = [(:realized_growth, "growth_band", "Realized growth (yr⁻¹)"),
            (:max_drawdown, "drawdown_band", "Max drawdown (fraction of the peak)"),
            (:sharpe, "sharpe_band", "Sharpe ratio")]
        for (metric, file, title) in bands
            t = grid_table(cells, bench, metric; ids=ids, hedge=hedge, stat=:band)
            check_grid(t, stem * " " * file)
            write_table(io, tabdir, stem * "_" * file, title * ": median (10th–90th percentile) across paths.", t, t)
        end
        h2h = head_to_head_grid(H; hedge=hedge_text, metric="wealth")
        println(io, "**Head-to-heads, ending wealth ÷ start.** On each path, the first picker's cell minus the ",
            "second's. Each cell: the median (10th–90th percentile) across paths; then the share of paths above 0.")
        println(io)
        print(io, markdown_table(h2h))
        println(io)
    end
    println(io, "### Hedge on minus hedge off")
    println(io)
    println(io, "On each path, the hedge-on cell minus the hedge-off cell. Each cell: the median (10th–90th ",
        "percentile) across paths; then the share of paths above 0.")
    println(io)
    hedge_tables = [("realized_growth", "**Realized growth (yr⁻¹).** Above 0: the hedge raised growth."),
        ("max_drawdown", "**Max drawdown (fraction of the peak).** Above 0: the hedge deepened the drawdown.")]
    for (metric, title) in hedge_tables
        t = head_to_head_grid(H; hedge="on minus off", metric=metric)
        if nrow(t) != length(PICKER_ROWS)
            error(set, " hedge on minus off, ", metric, ": expected ", length(PICKER_ROWS), " rows, got ", nrow(t))
        end
        println(io, title)
        println(io)
        print(io, markdown_table(t))
        println(io)
    end
    return nothing
end

function main(args=ARGS)
    t0 = time()
    # the rows directory and where the tables go
    if isempty(args)
        rows_dir = joinpath(SYN_DIR, "rows")
    else
        rows_dir = String(rstrip(abspath(args[1]), '/'))
    end
    out_dir = dirname(rows_dir)
    tabdir = joinpath(out_dir, "tables")
    mkpath(tabdir)

    # the study settings, the synthetic paths, the real run and the generator checks
    s, _ = load_study(CONFIG)
    data = load_paths(rows_dir, s.W0)
    meta = data.meta
    real_run = real_cells(s.W0)
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    pr = passive_references(md, s.start, s.stop; W0=s.W0)
    pr.path_id = zeros(Int, nrow(pr))
    real_bench = bench_cells(pr; W0=s.W0)
    checks = CSV.read(joinpath(SYN_DIR, "checks.csv"), DataFrame)

    # the header numbers
    n_paths = nrow(meta)
    set_ids = Dict{String,Vector{Int}}()
    for (flag, title, what) in SETS
        set_ids[flag] = sort(meta.path_id[meta[!, flag] .== true])
    end
    n_fixed = count(meta.lambda_fixed .== true)
    clip = meta.clip_share
    nb = meta.n_beta_le0
    share_nb = count(nb .> 0) / n_paths
    real_nb = nothing
    for i in 1:nrow(checks)
        if checks.check[i] == "V6" && startswith(checks.statistic[i], "names whose online β̂ ≤ 0")
            real_nb = checks.reference[i]
        end
    end
    if real_nb === nothing
        error("checks.csv has no V6 row on names whose online β̂ ≤ 0")
    end
    n_gated = count(!ismissing, checks.pass)
    n_passed = count(isequal(true), checks.pass)
    failing = String[]
    for i in 1:nrow(checks)
        if isequal(checks.pass[i], false)
            push!(failing, string(checks.check[i], ", ", checks.statistic[i], ": ", round(checks.synthetic[i]; digits=3),
                " against ", round(checks.reference[i]; digits=3)))
        end
    end
    failing_text = ""
    if !isempty(failing)
        failing_text = " Failing: " * join(failing, "; ") * "."
    end
    nb_median = median(nb)
    if isinteger(nb_median)
        nb_median = Int(nb_median)
    end
    nb_text = string("median ", nb_median, " per path, range ", minimum(nb), "–", maximum(nb))

    # tables.md
    io = open(joinpath(out_dir, "tables.md"), "w")
    println(io, "# Synthetic stress test: ticker picker × utility tables")
    println(io)
    println(io, "Written by `scripts/23_synthetic_tables.jl` on ", Dates.today(), " from `",
        relpath(rows_dir, REPO), "`.")
    println(io)
    println(io, "- Paths completed: ", length(set_ids["typical"]), " typical, ", length(set_ids["burst_stress"]),
        " burst stress, ", length(set_ids["bear_stress"]), " bear stress (", n_paths, " paths; a path can be in ",
        "more than one set). Paths skipped as incomplete: ", data.n_incomplete, ". Error rows: ", data.n_error_rows,
        " of ", data.n_run_rows, " run rows.")
    println(io, "- Paths calibrated with the fixed λ = 0 fallback, used after a calibration over 30 s: ", n_fixed,
        " of ", n_paths, ".")
    println(io, "- Share of name-days on which the generator scaled β and the sector loading down to hold a name's ",
        "variance: median ", round(median(clip); digits=4), " (range ", round(minimum(clip); digits=4), "–",
        round(maximum(clip); digits=4), ") across paths.")
    println(io, "- Hedge pool: on ", round(100 * share_nb; digits=1), "% of paths at least one name's online β̂ ",
        "fell to 0 or below on some day of 2025–26. Such names: ", nb_text, ". Real 2025–26: ",
        round(Int, real_nb), " names.")
    println(io, "- Generator checks (`results/synthetic/checks.csv`): ", n_passed, " of ", n_gated, " gated rows pass.",
        failing_text)
    println(io, "- Random and bandit cells are the median over 5 seeds within a path, then the median (or share) ",
        "across paths. The other pickers run once per path.")
    println(io, "- Wealth is ending wealth ÷ starting wealth. Growth is the realized growth rate in yr⁻¹. Max ",
        "drawdown is the largest fall from a peak, as a fraction of the peak.")
    println(io, "- Utilities: CD Cobb-Douglas; CES on share counts; CESw CES on weights; SG Stone–Geary; MinVar ",
        "minimum variance; Tangent the tangency portfolio; LS 130/30 long-short Cobb-Douglas.")
    println(io, "- SPY buy-and-hold and the equal-weight universe (bought once and held) do not depend on the utility ",
        "or the hedge, so each fills its row.")
    println(io)
    for (flag, title, what) in SETS
        ids = set_ids[flag]
        println(io, "## ", title, " (", length(ids), " paths)")
        println(io)
        println(io, title, ": ", what, ".")
        println(io)
        write_set(io, tabdir, data, flag, ids)
    end

    # where the real run falls among the typical paths
    println(io, "## Where the real 2025–26 run falls among the typical futures")
    println(io)
    println(io, "Each cell: the share of typical paths whose ending wealth ÷ start is at or below the real run's; ",
        "0.900 means 90% of typical futures ended at or below the real result. Real picker cells are the median over ",
        "seeds ", first(REAL_SEEDS), "–", last(REAL_SEEDS), " (the 5 seeds of the synthetic runs) of ",
        "`results/tier1_summary.csv` and `results/universe_summary.csv`. Real SPY and equal weight are bought and ",
        "held on the real closes.")
    println(io)
    for (hedge, hedge_text) in HEDGES
        t = real_percentiles(data.cells, real_run; ids=set_ids["typical"], hedge=hedge, bench=data.bench,
            real_bench=real_bench)
        check_grid(t, "real percentiles, hedge " * hedge_text)
        write_table(io, tabdir, "typical_real_percentile_hedge_" * hedge_text, "Hedge " * hedge_text * ".", t,
            format_grid(t))
    end

    # the caveats
    println(io, "## Caveats")
    println(io)
    println(io, "- Names keep their 2014–2024 order of α (growth not explained by the market), pulled in toward the ",
        "average so the spread matches the part that carried over in real data. Synthetic edges persist about as ",
        "much as real ones did within 2014–2024. In real 2025–26 they did not carry over (correlation −0.12).")
    println(io, "- Each name keeps its 2014–2024 average growth, so this is a best case for picking by γ.")
    println(io, "- Sector factors are drawn independently, so there are no rotations between sectors beyond what the ",
        "market explains.")
    println(io, "- Names move with the market on the same day only; no name leads or lags the market.")
    println(io, "- The market model reproduces the regimes of 2014–2024. Worse regimes come only from rare draws, ",
        "which the stress sets collect.")
    println(io, "- The universe is fixed, as in the real run: no names enter or leave.")
    println(io, "- Synthetic paths have few names whose online β̂ goes to 0 or below (", nb_text, "), like ",
        "the real 2014–2024 windows (0–20 names), while real 2025–26 had ", round(Int, real_nb), ". So the hedge ",
        "tables describe a 2014–2024-like world.")
    println(io, "- The results describe this generator plus these strategies, not a forecast.")
    close(io)

    println(n_paths, " paths (", length(set_ids["typical"]), " typical, ", length(set_ids["burst_stress"]),
        " burst stress, ", length(set_ids["bear_stress"]), " bear stress), ", data.n_incomplete,
        " skipped as incomplete, ", data.n_error_rows, " error rows; wrote ", joinpath(out_dir, "tables.md"), " and ",
        length(readdir(tabdir)), " CSV files in ", tabdir, "; wall time ", round(time() - t0; digits=1), " s")
    return nothing
end

main()
