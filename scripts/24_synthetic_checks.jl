# Task 8 of the synthetic-market plan: checks V1–V9 (spec §8), the gate before the overnight run. Builds the first 50
# paths by typical_rank in results/synthetic/paths.csv, recalibrates each one as the study would, and compares them with
# the real 2014–2024 market. Writes results/synthetic/checks.md and checks.csv, and exits 1 if any gate fails. V8 is
# the full test suite, run on its own. Run with threads: julia --project=code -t 30 scripts/24_synthetic_checks.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV, JLD2

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const OUT_DIR = joinpath(REPO, "results", "synthetic")
const CONFIG = joinpath(REPO, "config", "study.toml")
const CALIB = joinpath(REPO, "results", "calibration.jld2")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const N_CHECK = 50                               # check paths: the first 50 by typical_rank
const TRADE = (Date(2025, 1, 2), Date(2026, 9, 4))  # trading window of the β̂ ≤ 0 rows (first and last trading day)

function main()
    t0 = time()

    # Step 1: the generator, the path table, the real data, the real calibration and the study settings
    gen = JLD2.load(joinpath(OUT_DIR, "generator.jld2"), "generator")
    paths = CSV.read(joinpath(OUT_DIR, "paths.csv"), DataFrame)
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    calib = load_calibration(CALIB)
    s, _ = load_study(CONFIG)
    if s.train_start != SYNTH_TRAIN[1] || s.train_stop != SYNTH_TRAIN[2]
        error("study training window is $(s.train_start) → $(s.train_stop); expected $(SYNTH_TRAIN[1]) → $(SYNTH_TRAIN[2])")
    end
    println(length(gen.tickers), " tickers, ", length(gen.dates), " dates, ", length(gen.sectors), " sectors, ",
        nrow(paths), " paths in paths.csv, ", Threads.nthreads(), " threads")

    # Step 2: the real reference
    ref = real_reference(md, calib, gen; trade=TRADE)

    # Step 3: the check paths; each one is built, recalibrated as the study would and measured (threaded over paths)
    order = sortperm(paths.typical_rank)
    ids = paths.path_id[order[1:N_CHECK]]
    diags = Vector{NamedTuple}(undef, N_CHECK)
    carry = zeros(Float64, N_CHECK)   # each path's OLS α correlation, 2014–2024 vs the trading window
    t_paths = time()
    Threads.@threads for k in 1:N_CHECK
        sm = synthetic_market(gen, ids[k])
        c = calibrate(sm.md; train_start=s.train_start, train_stop=s.train_stop, window=s.omega_window)
        diags[k] = path_diagnostics(gen, sm, c, ref; trade=TRADE)
        carry[k] = alpha_correlation(sm.md, SYNTH_TRAIN, TRADE)
    end
    println(N_CHECK, " check paths: ", round(time() - t_paths; digits=1), " s")

    # Step 4: V1 over every path in paths.csv, then the table of V1–V6 and V9
    v1 = (median_growth=median(paths.train_growth), n_paths=nrow(paths))
    table = check_table(diags, ref, v1)

    # Step 4b: two report-only rows that give the tables their real references.
    # V6: names whose online β̂ ≤ 0, counted in real 2014–2024 windows as long as the trading window, each started from
    # the real calibration as the 2025–26 count is; next to the synthetic 2025–26 count (median over the check paths).
    W = beta_le0_windows(md, calib; trade=TRADE)
    n_syn = Int[]
    for d in diags
        push!(n_syn, d.n_beta_le0)
    end
    row_v6 = ("V6", "names whose online β̂ ≤ 0 on some day of a real 2014–2024 window as long as 2025–26, median over " *
        "the $(nrow(W)) windows from $(W.start[1]) (range $(minimum(W.n_beta_le0))–$(maximum(W.n_beta_le0))); " *
        "synthetic: 2025–26, median over paths", Float64(median(n_syn)), Float64(median(W.n_beta_le0)), "report only",
        missing)
    insert!(table, findlast(==("V6"), table.check) + 1, row_v6)
    # V9: how much each name's α carries from 2014–2024 into the trading window, real against the check paths' median
    row_v9 = ("V9", "correlation of OLS α, 2014–2024 vs 2025–26", median(carry), alpha_correlation(md, SYNTH_TRAIN, TRADE),
        "report only", missing)
    insert!(table, findlast(==("V9"), table.check) + 1, row_v9)
    println("real 2014–2024 windows for the hedge-name count: ", join(string.(W.start, " → ", W.stop, ": ",
        W.n_beta_le0), "; "))

    # Step 5: V7, the first check path built twice, every market-data field compared with ==
    a = synthetic_market(gen, first(ids)).md
    b = synthetic_market(gen, first(ids)).md
    n_diff = 0
    for f in fieldnames(MyMarketData)
        if !(getfield(a, f) == getfield(b, f))
            n_diff += 1
        end
    end
    push!(table, ("V7", "market-data fields that differ when path $(first(ids)) is built twice", Float64(n_diff), 0.0,
        "= 0", n_diff == 0))

    # Step 6: the result, the CSV and the Markdown record
    n_gates = count(!ismissing, table.pass)
    failed = Int[]
    for r in 1:nrow(table)
        if table.pass[r] === false
            push!(failed, r)
        end
    end
    clip = median([d.clip_share for d in diags])
    wall = time() - t0
    if isempty(failed)
        result = "All $(n_gates) gates pass."
    else
        names_failed = join(["$(table.check[r]) ($(table.statistic[r]))" for r in failed], "; ")
        result = "$(n_gates - length(failed)) of $(n_gates) gates pass. Failing: $(names_failed)."
    end
    mkpath(OUT_DIR)
    CSV.write(joinpath(OUT_DIR, "checks.csv"), table)
    io = open(joinpath(OUT_DIR, "checks.md"), "w")
    println(io, "# Synthetic-market checks V1–V9")
    println(io)
    println(io, "Written by `scripts/24_synthetic_checks.jl` on ", Dates.today(), " (wall time ", round(wall; digits=1),
        " s on ", Threads.nthreads(), " threads).")
    println(io)
    println(io, "- Check paths: the first ", N_CHECK, " paths by `typical_rank` in `paths.csv` (ids ", join(ids, ", "), ").")
    println(io, "- Each check path is recalibrated on ", SYNTH_TRAIN[1], " → ", SYNTH_TRAIN[2],
        " with the study's `calibrate`, then compared with the real market over that window. The rows on names",
        " whose online β̂ ≤ 0 use the trading window ", TRADE[1], " → ", TRADE[2], " instead, except that the third",
        " one's real count comes from 2014–2024 windows as long as the trading window, each started from the real",
        " calibration as the 2025–26 count is. The last V9 row correlates each name's α on 2014–2024 with its α on",
        " the trading window.")
    println(io, "- V1 uses all ", nrow(paths), " paths in `paths.csv`.")
    println(io, "- A number with one value per path is the median over the ", N_CHECK, " paths. A number with one value",
        " per name is each name's median over the paths, then the median over the ", length(gen.tickers), " names.")
    println(io, "- `pass` is `missing` for a row that is reported but not gated.")
    println(io, "- V8 is the study's full test suite, run on its own: `julia --project=code code/test/runtests.jl`.")
    println(io, "- Median share of name-days on which the composition scaled β and the sector loading down (clipped): ",
        round(clip; digits=4), ".")
    println(io)
    println(io, "**", result, "**")
    println(io)
    print(io, markdown_table(table))
    close(io)
    println(markdown_table(table))
    println(result)
    println("wrote ", joinpath(OUT_DIR, "checks.md"), " and checks.csv; wall time ", round(wall; digits=1), " s")
    if !isempty(failed)
        exit(1)
    end
    return nothing
end

main()
