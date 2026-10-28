# Task 11 of the synthetic-market plan: the stress-test tables (spec §7) from the per-path files of script 22.
# Reads a rows directory (default results/synthetic/rows/) and writes tables.md and tables/*.csv in its parent
# directory (results/synthetic/ for the default, results/synthetic/smoke/ for the smoke rows).
# A path counts only when path_<p>.csv, bench_<p>.csv and meta_<p>.csv all exist and path_<p>.csv holds 532 run rows.
# inflight_<p>.txt markers are ignored; error_<p>.txt marks a path that failed. The planned sets come from
# results/synthetic/paths.csv, and the real references in the header and caveats from results/synthetic/checks.csv
# (script 24), results/calibration_summary.toml (the real λ) and results/synthetic/generator.jld2 (how the generator
# sets each name's α). Nothing is optimized here, so no thread settings are needed.
#
# Run: julia --project=code scripts/23_synthetic_tables.jl [rows directory]
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV, JLD2, TOML

const REPO = normpath(joinpath(@__DIR__, ".."))
const SYN_DIR = joinpath(REPO, "results", "synthetic")
const RESULTS = joinpath(REPO, "results")
const DATA_DIR = joinpath(REPO, "data")
const CONFIG = joinpath(REPO, "config", "study.toml")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const N_RUNS = 532            # run rows in a complete path file (synthetic_runs with 5 seeds)
const REAL_SEEDS = 5660:5664  # the seeds of the synthetic runs, kept from the real summaries
# The columns of a run row that the tables use.
const ROW_COLUMNS = [:path_id, :label, :status, :allocator, :screen, :seed, :start, :stop, :hedge, :basket_K, :W0,
    :terminal, :realized_growth, :max_drawdown, :sharpe]
# The sets: the meta flag and the heading.
const SETS = [("typical", "Typical futures"), ("burst_stress", "Burst stress"), ("bear_stress", "Bear stress")]
const HEDGES = [(false, "off"), (true, "on")]
# The check rows of results/synthetic/checks.csv that the header and caveats read, by the start of `statistic`.
const CHECK_BETA_2025 = "names whose online β̂ ≤ 0 on some day in 2025–26"
const CHECK_BETA_WINDOWS = "names whose online β̂ ≤ 0 on some day of a real 2014–2024 window"
const CHECK_BETA_LAST = "names whose online β̂ ≤ 0 on some day of the real window as long as 2025–26 that ends on"
const CHECK_ALPHA_SPREAD = "cross-name SD of OLS α"
const CHECK_ALPHA_HALVES = "correlation of OLS α, 2014–2019 vs 2020–2024"
const CHECK_ALPHA_TRADE = "correlation of OLS α, 2014–2024 vs 2025–26"

"""
    with_commas(n::Integer)::String

`n` with a comma between each group of three digits ("10,000").
"""
function with_commas(n::Integer)::String
    digits = string(abs(n))
    out = ""
    for (k, c) in enumerate(reverse(digits))
        if k > 1 && rem(k - 1, 3) == 0
            out = "," * out
        end
        out = string(c) * out
    end
    if n < 0
        out = "-" * out
    end
    return out
end

"""
    fixed(x::Real, digits::Int)::String

`x` with exactly `digits` decimals and a true minus sign (−) for a negative value ("0.100", "−0.121").
"""
function fixed(x::Real, digits::Int)::String
    scale = 10^digits
    m = round(Int, scale * x)
    sign = ""
    if m < 0
        sign = "−"
    end
    m = abs(m)
    return string(sign, div(m, scale), ".", lpad(string(rem(m, scale)), digits, '0'))
end

"""
    seeded_picker(key::AbstractString)::Bool

True for a `PICKER_ROWS` key whose basket depends on the seed (bandit per sector and every random picker), which
runs with 5 seeds per path; false for a picker that runs once per path.
"""
function seeded_picker(key::AbstractString)::Bool
    return key == "bandit" || startswith(key, "random")
end

"""
    load_paths(dir::AbstractString, W0::Real)::NamedTuple

The complete paths of the rows directory `dir`: a path counts when its path, bench and meta files all exist and its
path file has `N_RUNS` rows. Returns `cells` (`path_cells` of every counted path), `bench` (`bench_cells`, with the
starting wealth `W0` in USD), `meta` (the meta rows), `n_incomplete` (paths with some of the three files that do
not count), `n_failed` (paths with an `error_<p>.txt` file and no counted rows), `n_run_rows` and `n_error_rows` (run
rows of the counted paths, and those with `status == "error"`) and `window` (the trading window's first and last day).
Throws an error if an ok run row has a starting wealth other than `W0`, a seed outside `REAL_SEEDS`, or a trading
window other than the one shared by every ok row, or if a picker that runs once per path (`seeded_picker` false) has a
cell with more than one seed.
"""
function load_paths(dir::AbstractString, W0::Real)::NamedTuple
    # every path id that has a path, bench or meta file, and every path id with an error file
    found = Set{Int}()
    errored = Set{Int}()
    for f in readdir(dir)
        m = match(r"^(path|bench|meta)_(\d+)\.csv$", f)
        if m !== nothing
            push!(found, parse(Int, m.captures[2]))
        end
        e = match(r"^error_(\d+)\.txt$", f)
        if e !== nothing
            push!(errored, parse(Int, e.captures[1]))
        end
    end
    seen = sort(collect(found))
    # the complete paths
    cells = DataFrame[]
    bench = DataFrame[]
    meta = DataFrame[]
    counted = Set{Int}()
    windows = Set{Tuple{Date,Date}}()
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
        # every ok row starts with W0, uses a seed of the synthetic runs and states its trading window
        for i in 1:nrow(rows)
            if rows.status[i] == "error"
                n_error_rows += 1
                continue
            end
            if rows.W0[i] != W0
                error("path $(p): an ok run row starts with $(rows.W0[i]) USD, not $(W0)")
            end
            if !(rows.seed[i] in REAL_SEEDS)
                error("path $(p): an ok run row has seed $(rows.seed[i]), outside $(REAL_SEEDS)")
            end
            push!(windows, (rows.start[i], rows.stop[i]))
        end
        c = path_cells(rows)
        # a picker that runs once per path has one seed in every cell
        for i in 1:nrow(c)
            if !seeded_picker(c.picker[i]) && c.n_seeds[i] != 1
                error("path $(p): ", c.picker[i], ", ", c.allocator[i], " has ", c.n_seeds[i], " seeds, expected 1")
            end
        end
        n_run_rows += nrow(rows)
        push!(counted, p)
        push!(cells, c)
        push!(bench, bench_cells(CSV.read(fbench, DataFrame); W0=W0))
        push!(meta, CSV.read(fmeta, DataFrame))
    end
    if isempty(cells)
        error("no complete path in ", dir)
    end
    if length(windows) != 1
        error("the ok run rows state ", length(windows), " trading windows; expected one")
    end
    n_failed = 0
    for p in errored
        if !(p in counted)
            n_failed += 1
        end
    end
    return (cells=reduce(vcat, cells), bench=reduce(vcat, bench), meta=reduce(vcat, meta; cols=:union),
        n_incomplete=n_incomplete, n_failed=n_failed, n_run_rows=n_run_rows, n_error_rows=n_error_rows,
        window=first(windows))
end

"""
    real_cells(W0::Real)::DataFrame

`path_cells` of the real 2025–26 run: the rows of results/tier1_summary.csv and results/universe_summary.csv with
seeds in `REAL_SEEDS`, as one path with `path_id` 0. Throws an error if a `PICKER_ROWS` picker lacks a cell for some
utility and hedge setting, if such a cell has other than 5 seeds (`seeded_picker`) or 1 seed (the others), or if a
real row has a starting wealth other than `W0` (USD).
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
    # every picker of the tables has one real cell for every utility and hedge setting, over 5 seeds or 1
    for (key, name) in PICKER_ROWS
        n_want = 1
        if seeded_picker(key)
            n_want = length(REAL_SEEDS)
        end
        for a in TABLE_ALLOCATOR_ORDER
            for (hedge, hedge_text) in HEDGES
                k = findall((cells.picker .== key) .& (cells.allocator .== a) .& (cells.hedge .== hedge))
                if length(k) != 1
                    error("real run: ", length(k), " cells for ", name, ", ", a, ", hedge ", hedge_text)
                end
                if cells.n_seeds[k[1]] != n_want
                    error("real run: ", name, ", ", a, ", hedge ", hedge_text, " has ", cells.n_seeds[k[1]],
                        " seeds, expected ", n_want)
                end
            end
        end
    end
    return cells
end

"""
    check_row(checks::DataFrame, prefix::AbstractString)::DataFrameRow

The one row of `checks` (results/synthetic/checks.csv) whose `statistic` starts with `prefix`. Throws an error if
there is no such row or more than one.
"""
function check_row(checks::DataFrame, prefix::AbstractString)
    k = Int[]
    for i in 1:nrow(checks)
        if startswith(checks.statistic[i], prefix)
            push!(k, i)
        end
    end
    if length(k) != 1
        error("checks.csv has ", length(k), " rows whose statistic starts with \"", prefix, "\"; rerun script 24")
    end
    return checks[k[1], :]
end

"""
    plain_check(check::AbstractString, statistic::AbstractString)::String

What a generator check (one row of checks.csv) measures, in plain words; the `statistic` text itself for a row with
no plain version.
"""
function plain_check(check::AbstractString, statistic::AbstractString)::String
    if startswith(statistic, CHECK_ALPHA_SPREAD)
        return "how much names differ in average growth not explained by the market (α), per year"
    elseif startswith(statistic, CHECK_ALPHA_HALVES)
        return "how much a name's α carries over from 2014–2019 to 2020–2024 (correlation)"
    elseif check == "V1"
        return "SPY growth over 2014–2024, per year"
    elseif check == "V2"
        return "the gain from rebalancing an equal-weight portfolio every day, per year"
    elseif startswith(statistic, "variance ÷ variance of the name's own draw")
        return "each name's variance once the market and its sector are added"
    elseif startswith(statistic, "absolute gap between the OLS β")
        return "how far each name's β lands from its target"
    elseif check == "V4"
        return "how long busy and quiet spells last in each name"
    elseif check == "V5"
        sector = replace(statistic, "mean residual correlation within " => "")
        return "how closely names in " * sector * " move together beyond the market"
    elseif startswith(statistic, "SD of the online β̂")
        return "how much each name's β wanders over time"
    elseif check == "V7"
        return "whether the same path comes out the same when built twice"
    end
    return statistic
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
    write_md(io::IO, title::AbstractString, md_table::DataFrame)::Nothing

Write `title` in bold, then `md_table` as Markdown, to `io`.
"""
function write_md(io::IO, title::AbstractString, md_table::DataFrame)::Nothing
    println(io, "**", title, "**")
    println(io)
    print(io, markdown_table(md_table))
    println(io)
    return nothing
end

"""
    write_set(io::IO, tabdir::AbstractString, data, set::AbstractString, ids::Vector{Int})::Nothing

The tables of one set (`set` is its meta flag; `ids` its paths): for each hedge setting the median ending wealth, the
share of paths beating SPY, growth, max drawdown and Sharpe as bands, and picker against picker; then hedge on minus
off. Each grid also goes to `tabdir` as a CSV of numbers (the bands as `band_table`, the head-to-heads as
`head_to_head`).
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
        # ending wealth: the median, bold where the cell beats SPY on more than half of the paths, and that share
        med = grid_table(cells, bench, :wealth; ids=ids, hedge=hedge, stat=:median)
        beat = grid_table(cells, bench, :wealth; ids=ids, hedge=hedge, stat=:beat_spy)
        check_grid(med, stem * " wealth")
        check_grid(beat, stem * " beat SPY")
        CSV.write(joinpath(tabdir, stem * "_wealth_median.csv"), med)
        CSV.write(joinpath(tabdir, stem * "_wealth_beat_spy.csv"), beat)
        write_md(io, "Ending wealth ÷ start, median across paths. Bold: the cell beats that path's SPY on more than " *
            "half of the paths. Bold compares each path with its own SPY, so a bold median can sit below the SPY " *
            "row's median.", format_grid(med; beat=beat))
        write_md(io, "Share of paths on which the ending wealth beats that path's SPY (— for SPY itself).",
            format_grid(beat))
        # growth, max drawdown and Sharpe: median (10th to 90th percentile) across paths
        bands = [(:realized_growth, "growth_band", "Realized growth (yr⁻¹)"),
            (:max_drawdown, "drawdown_band", "Max drawdown (fraction of the peak)"),
            (:sharpe, "sharpe_band", "Sharpe ratio")]
        for (metric, file, title) in bands
            t = grid_table(cells, bench, metric; ids=ids, hedge=hedge, stat=:band)
            check_grid(t, stem * " " * file)
            CSV.write(joinpath(tabdir, stem * "_" * file * ".csv"), band_table(cells, bench, metric; ids=ids,
                hedge=hedge))
            write_md(io, title * ": median across paths (10th to 90th percentile).", t)
        end
        # picker against picker
        write_md(io, "Picker against picker, ending wealth ÷ start: on each path, the first picker's value minus " *
            "the second's. Median across paths (10th to 90th percentile).",
            head_to_head_grid(H; hedge=hedge_text, metric="wealth", stat=:band))
        write_md(io, "Picker against picker: share of paths on which the first picker ended with more wealth than " *
            "the second.",
            head_to_head_grid(H; hedge=hedge_text, metric="wealth", stat=:share_above_0))
    end
    # hedge on against hedge off, per picker
    println(io, "### Hedge on minus hedge off")
    println(io)
    hedge_tables = [
        ("realized_growth", "Realized growth (yr⁻¹)", :share_above_0, "the hedge raised growth"),
        ("max_drawdown", "Max drawdown (fraction of the peak)", :share_below_0, "the hedge made the drawdown smaller")]
    for (metric, title, share_stat, share_text) in hedge_tables
        band = head_to_head_grid(H; hedge="on minus off", metric=metric, stat=:band)
        share = head_to_head_grid(H; hedge="on minus off", metric=metric, stat=share_stat)
        if nrow(band) != length(PICKER_ROWS) || nrow(share) != length(PICKER_ROWS)
            error(set, " hedge on minus off, ", metric, ": expected ", length(PICKER_ROWS), " rows")
        end
        write_md(io, title * ", hedge on minus hedge off: on each path, the hedge-on value minus the hedge-off " *
            "value. Median across paths (10th to 90th percentile).", band)
        write_md(io, "Share of paths on which " * share_text * ".", share)
    end
    return nothing
end

function main(args=ARGS)
    t0 = time()
    # the rows directory and where the tables go
    if isempty(args)
        rows_dir = joinpath(SYN_DIR, "rows")
    else
        rows_dir = String(rstrip(normpath(abspath(args[1])), '/'))
    end
    out_dir = dirname(rows_dir)
    if !(out_dir == SYN_DIR || startswith(out_dir, SYN_DIR * "/"))
        error("the tables go next to the rows directory, so it must sit under ", SYN_DIR, "; got ", rows_dir)
    end
    tabdir = joinpath(out_dir, "tables")
    mkpath(tabdir)

    # the study settings, the planned paths, the synthetic paths, the real run and the generator checks
    s, _ = load_study(CONFIG)
    paths = CSV.read(joinpath(SYN_DIR, "paths.csv"), DataFrame)
    data = load_paths(rows_dir, s.W0)
    meta = data.meta
    real_run = real_cells(s.W0)
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    pr = passive_references(md, s.start, s.stop; W0=s.W0)
    pr.path_id = zeros(Int, nrow(pr))
    real_bench = bench_cells(pr; W0=s.W0)
    checks = CSV.read(joinpath(SYN_DIR, "checks.csv"), DataFrame)
    real_λ = TOML.parsefile(joinpath(RESULTS, "calibration_summary.toml"))["lambda"]
    gen = JLD2.load(joinpath(SYN_DIR, "generator.jld2"), "generator")

    # the paths finished and planned, per set
    n_paths = nrow(meta)
    set_ids = Dict{String,Vector{Int}}()
    planned = Dict{String,Int}()
    for (flag, title) in SETS
        set_ids[flag] = sort(meta.path_id[meta[!, flag] .== true])
        planned[flag] = count(paths[!, flag] .== true)
    end
    n_planned = count((paths.typical .== true) .| (paths.burst_stress .== true) .| (paths.bear_stress .== true))
    # the trading window
    window_years = round(Dates.value(data.window[2] - data.window[1]) / 365.25; digits=1)
    # λ, the clip share and the names whose β reaches 0
    n_fixed = count(meta.lambda_fixed .== true)
    clip = meta.clip_share
    nb = meta.n_beta_le0
    share_nb = count(nb .> 0) / n_paths
    nb_median = median(nb)
    if isinteger(nb_median)
        nb_median = Int(nb_median)
    end
    nb_text = string("median ", nb_median, " per path, range ", minimum(nb), " to ", maximum(nb))
    # the real references from checks.csv
    real_nb = check_row(checks, CHECK_BETA_2025).reference
    win = check_row(checks, CHECK_BETA_WINDOWS)
    win_n = match(r"median over the (\d+) windows", win.statistic)
    win_range = match(r"\(range (\d+)–(\d+)\)", win.statistic)
    win_step = match(r"a multiple of (\d+) trading days", win.statistic)
    if win_n === nothing || win_range === nothing || win_step === nothing
        error("checks.csv: the 2014–2024 window row does not state its window count, range and step: ", win.statistic)
    end
    step_text = string("one ending every ", win_step.captures[1], " trading days")
    if win_step.captures[1] == "21"
        step_text = step_text * ", about a month"
    end
    win_median = win.reference
    if isinteger(win_median)
        win_median = Int(win_median)
    end
    last_win = check_row(checks, CHECK_BETA_LAST)
    last_dates = match(r"ends on (\d{4}-\d{2}-\d{2}) \(from (\d{4}-\d{2}-\d{2})\)", last_win.statistic)
    if last_dates === nothing
        error("checks.csv: the last 2014–2024 window row does not state its dates: ", last_win.statistic)
    end
    # how the generator sets each name's α: its 2014–2024 α moved toward the average of all names, keeping the share
    # φ of its distance from the average, with no drift (k_α = 0)
    if !(0 < gen.φ <= 1) || gen.k_α != 0
        error("the α caveat assumes 0 < φ ≤ 1 and no α drift; the generator has φ = ", gen.φ, ", k_α = ", gen.k_α)
    end
    α_pull = round(Int, 100 * (1 - gen.φ))
    α_spread = check_row(checks, CHECK_ALPHA_SPREAD)
    α_halves = check_row(checks, CHECK_ALPHA_HALVES)
    α_trade = check_row(checks, CHECK_ALPHA_TRADE)
    # the generator checks that are pass/fail, and the ones that fail
    n_gated = count(!ismissing, checks.pass)
    n_passed = count(isequal(true), checks.pass)
    failing = String[]
    for i in 1:nrow(checks)
        if isequal(checks.pass[i], false)
            push!(failing, string(plain_check(checks.check[i], checks.statistic[i]), " (",
                fixed(checks.synthetic[i], 3), " synthetic against ", fixed(checks.reference[i], 3), " real)"))
        end
    end
    failing_text = ""
    if length(failing) == 1
        failing_text = " The one that fails: " * failing[1] * "."
    elseif length(failing) > 1
        failing_text = " The ones that fail: " * join(failing, "; ") * "."
    end
    # random and bandit cells with all 5 seeds finished, against the count a full set of paths holds
    n_seeded = 0
    for (key, name) in PICKER_ROWS
        if seeded_picker(key)
            n_seeded += n_paths * length(TABLE_ALLOCATOR_ORDER) * length(HEDGES)
        end
    end
    n_full = 0
    for i in 1:nrow(data.cells)
        if seeded_picker(data.cells.picker[i]) && data.cells.n_seeds[i] == length(REAL_SEEDS)
            n_full += 1
        end
    end
    n_short = n_seeded - n_full
    long_short = string(round(Int, 100 * s.ls_long), "/", round(Int, 100 * s.ls_short))

    # tables.md: the header
    io = open(joinpath(out_dir, "tables.md"), "w")
    println(io, "# Synthetic stress test: ticker picker × utility tables")
    println(io)
    println(io, "Written by `scripts/23_synthetic_tables.jl` on ", Dates.today(), " from `",
        relpath(rows_dir, REPO), "`.")
    println(io)
    println(io, "- Trading window: ", data.window[1], " to ", data.window[2], ", about ", window_years, " years. ",
        "Wealth is the ending value ÷ the starting value over that window.")
    println(io, "- Paths finished: ", length(set_ids["typical"]), " of ", planned["typical"], " typical, ",
        length(set_ids["burst_stress"]), " of ", planned["burst_stress"], " burst stress, ",
        length(set_ids["bear_stress"]), " of ", planned["bear_stress"], " bear stress (", n_paths, " of ", n_planned,
        " distinct paths; a path can be in more than one set). Paths that failed: ", data.n_failed, ". Paths skipped ",
        "because their files were incomplete: ", data.n_incomplete, ". Runs that failed inside finished paths: ",
        data.n_error_rows, " of ", with_commas(data.n_run_rows), ".")
    if real_λ == 0
        real_λ_text = "For the real market, that step chooses no toning down."
    else
        real_λ_text = string("For the real market, that step tones it down by ", real_λ, " on a scale from 0 (none) ",
            "to 1 (full).")
    end
    println(io, "- On ", n_fixed, " of ", n_paths, " paths, the study skipped the step that chooses how much to tone ",
        "down the co-movement between names, and used it as measured. ", real_λ_text)
    if minimum(clip) == maximum(clip)
        clip_text = string(round(100 * clip[1]; digits=1), "%. This share is the same on every path.")
    else
        clip_text = string(round(100 * median(clip); digits=1), "% (median; range ", round(100 * minimum(clip);
            digits=1), "% to ", round(100 * maximum(clip); digits=1), "% across paths).")
    end
    println(io, "- Share of name-days on which the generator trimmed a name's links to the market and its sector ",
        "to keep the name's variance at its real value: ", clip_text)
    println(io, "- Names whose β, as the study updates it each day, went to 0 or below at some point in 2025–26 ",
        "(the names the hedge can use): at least one on ", round(Int, 100 * share_nb), "% of paths; ", nb_text,
        ". Real 2025–26: ", round(Int, real_nb), " names.")
    println(io, "- Of the ", n_gated, " checks on the generator, ", n_passed, " pass.", failing_text)
    println(io, "- Random and bandit cells: the median over 5 seeds within a path, then the median (or share) ",
        "across paths. Cells with fewer than 5 finished seeds: ", n_short, " of ", with_commas(n_seeded), ". The ",
        "other pickers run once per path.")
    println(io, "- Bold in the wealth tables: the cell beats that path's SPY on more than half of the paths. Bold ",
        "compares each path with its own SPY, so a bold median can sit below the SPY row's median.")
    println(io, "- Growth is the realized growth rate in yr⁻¹. Max drawdown is the largest fall from a peak, as a ",
        "fraction of the peak. — marks a cell with no value.")
    println(io, "- Utilities: CD Cobb-Douglas; CES on share counts; CESw CES on weights; SG Stone–Geary; MinVar ",
        "minimum variance; Tangent the tangency portfolio; LS ", long_short, " long-short Cobb-Douglas.")
    println(io, "- SPY buy-and-hold and the equal-weight universe (bought once and held) do not depend on the utility ",
        "or the hedge, so each fills its row.")
    println(io)

    # the sets
    what = Dict("typical" => "paths drawn at random from the " * with_commas(nrow(paths)) * " synthetic markets",
        "burst_stress" => "the deepest SPY sell-offs in 2025–26 among the futures with a jump burst, a run of " *
            "large daily moves that lasts months",
        "bear_stress" => "the deepest SPY sell-offs in 2025–26 among the futures without a jump burst")
    for (flag, title) in SETS
        ids = set_ids[flag]
        println(io, "## ", title, " (", length(ids), " paths)")
        println(io)
        println(io, title, ": ", what[flag], ".")
        println(io)
        write_set(io, tabdir, data, flag, ids)
    end

    # where the real run falls among the typical paths
    println(io, "## Where the real 2025–26 run falls among the typical futures")
    println(io)
    println(io, "Each cell: the share of typical paths whose ending wealth ÷ start is at or below the real run's; ",
        "0.900 means 90% of typical futures ended at or below the real result. Real picker cells are the median over ",
        "seeds ", first(REAL_SEEDS), " to ", last(REAL_SEEDS), " (the 5 seeds of the synthetic runs) of ",
        "`results/tier1_summary.csv` and `results/universe_summary.csv`. Real SPY and equal weight are bought and ",
        "held on the real closes.")
    println(io)
    for (hedge, hedge_text) in HEDGES
        t = real_percentiles(data.cells, real_run; ids=set_ids["typical"], hedge=hedge, bench=data.bench,
            real_bench=real_bench)
        check_grid(t, "real percentiles, hedge " * hedge_text)
        CSV.write(joinpath(tabdir, "typical_real_percentile_hedge_" * hedge_text * ".csv"), t)
        write_md(io, "Hedge " * hedge_text * ".", format_grid(t))
    end

    # the caveats
    println(io, "## Caveats")
    println(io)
    println(io, "- α is a name's average growth not explained by the market. Names differ in α by ",
        fixed(α_spread.synthetic, 3), " per year on synthetic paths against ", fixed(α_spread.reference, 3),
        " in real 2014–2024. A name's α carries over from 2014–2019 to 2020–2024 with correlation ",
        fixed(α_halves.synthetic, 2), " on synthetic paths against ", fixed(α_halves.reference, 2), " real, and ",
        "from 2014–2024 into 2025–26 with ", fixed(α_trade.synthetic, 2), " synthetic against ",
        fixed(α_trade.reference, 2), " real.")
    println(io, "- Each name keeps its 2014–2024 ranking by α. Its α is moved ", α_pull, "% of the way toward the ",
        "average of all names, so it keeps ", round(Int, 100 * gen.φ), "% of its 2014–2024 distance from the average, ",
        "and it does not drift over time. So this is a best case for picking by γ.")
    println(io, "- Sector factors are drawn independently, so there are no rotations between sectors beyond what the ",
        "market explains.")
    println(io, "- Names move with the market on the same day only; no name leads or lags the market.")
    println(io, "- The market model reproduces the regimes of 2014–2024. Worse regimes come only from rare draws, ",
        "which the stress sets collect.")
    println(io, "- The universe is fixed, as in the real run: no names enter or leave.")
    println(io, "- Names whose β goes to 0 or below (the names the hedge can use): synthetic paths have a median of ",
        nb_median, " in 2025–26 (range ", minimum(nb), " to ", maximum(nb), " across paths). Real periods of the ",
        "same length within 2014–2024 had a median of ", win_median, " (range ", win_range.captures[1], " to ",
        win_range.captures[2], " across ", win_n.captures[1], " overlapping periods, ", step_text, "). The real ",
        "period just before 2025 (", last_dates.captures[2], " to ",
        last_dates.captures[1], ") had ", round(Int, last_win.reference), ", and real 2025–26 had ",
        round(Int, real_nb), ". So the synthetic paths look like a typical 2014–2024 period, while the real market ",
        "already had more such names just before 2025 and many more in 2025–26.")
    println(io, "- The results describe this generator plus these strategies, not a forecast.")
    close(io)

    println(n_paths, " paths (", length(set_ids["typical"]), " typical, ", length(set_ids["burst_stress"]),
        " burst stress, ", length(set_ids["bear_stress"]), " bear stress), ", data.n_failed, " failed, ",
        data.n_incomplete, " skipped as incomplete, ", data.n_error_rows, " error rows; wrote ",
        joinpath(out_dir, "tables.md"), " and ", length(readdir(tabdir)), " CSV files in ", tabdir, "; wall time ",
        round(time() - t0; digits=1), " s")
    return nothing
end

main()
