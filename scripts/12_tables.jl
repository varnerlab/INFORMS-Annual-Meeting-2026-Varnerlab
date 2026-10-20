# Task 13: comparison tables from the summary CSVs. Usage: 12_tables.jl [dir] [--smoke]
# Addendum Task 5: when <dir>/universe_summary.csv exists, also the size sweep, the matched-control
# percentiles, the K = 13 head-to-head and the sector-quota cost. Every metric is growth-rate based
# (g = ln(S_t/S_{t-1})/Δt in 1/yr, Δt = 1/252). The new screens are exploratory (designed after Tier 1);
# the user's 13- and 15-name lists were chosen in advance without 2025 data.
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, CSV, DataFrames, Statistics

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const CONFIG = joinpath(REPO, "config", "study.toml")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
# Deterministic screen => its matched random control (addendum §4).
const MATCHED = [:global_gamma => :random_global, :corr_gamma => :random_global, :liquid_gamma => :random_liquid]
# Sector-quota cost (addendum §1 question 5): (pair, sector screen, its basket size, global screen, its size).
const QUOTA_PAIRS = [("gamma vs global_gamma", "gamma", 33, "global_gamma", 34),
    ("random vs random_global", "random", 33, "random_global", 34)]
const QUOTA_METRICS = [:sharpe, :max_drawdown, :realized_growth, :alpha]

"""
    load_ok(path)::DataFrame

The `status == "ok"` rows of the summary CSV at `path`, with `allocator` and `screen` as Strings and
`basket_size` (names) as Int when every value is whole.
"""
function load_ok(path::AbstractString)::DataFrame
    df = CSV.read(path, DataFrame)
    df = df[string.(df.status) .== "ok", :]
    df.allocator = String.(string.(df.allocator)); df.screen = String.(string.(df.screen))
    if "basket_size" in names(df) && all(x -> !ismissing(x) && isinteger(x), df.basket_size)
        df.basket_size = Int.(df.basket_size)
    end
    return df
end

"""
    matched_percentile(df)::DataFrame

`gamma_percentile` of each deterministic screen against its matched random control (`MATCHED`), for
Sharpe ratio and max drawdown, by allocator, basket size and hedge. Columns: `pair`, `metric`, the group
columns, `screen_value` (metric units), `n_control` (control seeds), `percentile` (share of control
seeds below the screen's value, ties half; for max drawdown a low percentile means a smaller drawdown) and
`share_of_controls_beaten` (share of control seeds the screen beats, ties half: a higher Sharpe ratio, or a
shallower max drawdown), which reads the same way for both metrics.
"""
function matched_percentile(df::DataFrame)::DataFrame
    parts = DataFrame[]
    for m in (:sharpe, :max_drawdown), (scr, ctl) in MATCHED
        t = gamma_percentile(df; metric=m, screen=scr, control=ctl, by=[:allocator, :basket_size, :hedge],
            higher_is_better=(m != :max_drawdown))
        rename!(t, :gamma_value => :screen_value, :n_random => :n_control)
        insertcols!(t, 1, :pair => fill("$(scr) vs $(ctl)", nrow(t)), :metric => fill(string(m), nrow(t)))
        push!(parts, t)
    end
    return sort_table(reduce(vcat, parts))
end

"""
    quota_cost(sector, uni)::DataFrame

Per allocator, hedge and `QUOTA_PAIRS` entry: median (across seeds) of each `QUOTA_METRICS` metric for the
per-sector screen in `sector` (Tier 1, 33 names) and the universe-wide screen in `uni` (34 names), and
`<metric>_diff` = global − sector, in the metric's units. `n_sector`, `n_global`: runs behind each median.
"""
function quota_cost(sector::DataFrame, uni::DataFrame)::DataFrame
    ok(x) = !ismissing(x) && !(x isa Real && isnan(x))
    med(v) = (w = Float64.(filter(ok, collect(v))); isempty(w) ? NaN : median(w))
    rows = []
    for a in TABLE_ALLOCATOR_ORDER, h in (true, false), (pair, ss, sK, gs, gK) in QUOTA_PAIRS
        S = sector[(sector.allocator .== a) .& (sector.hedge .== h) .& (sector.screen .== ss) .& (sector.basket_size .== sK), :]
        G = uni[(uni.allocator .== a) .& (uni.hedge .== h) .& (uni.screen .== gs) .& (uni.basket_size .== gK), :]
        (isempty(S) || isempty(G)) && error("quota cost: no rows for $(a), hedge=$(h), $(pair)")
        r = Pair{Symbol,Any}[:allocator => a, :hedge => h, :pair => pair, :n_sector => nrow(S), :n_global => nrow(G)]
        for m in QUOTA_METRICS
            ms, mg = med(S[!, m]), med(G[!, m])
            push!(r, Symbol(m, "_sector") => ms, Symbol(m, "_global") => mg, Symbol(m, "_diff") => mg - ms)
        end
        push!(rows, NamedTuple(r))
    end
    return DataFrame(rows)
end

"""
    addendum_tables(tier1, uni)::Vector{Pair{String,DataFrame}}

The addendum's four tables (spec §8) from the Tier 1 and universe summaries, with sanity checks: every
matched-percentile row has 30 control seeds and the head-to-head holds the fixed lists at 13 and 15 names.
"""
function addendum_tables(tier1::DataFrame, uni::DataFrame)::Vector{Pair{String,DataFrame}}
    both = vcat(uni, tier1; cols=:union)
    by = [:allocator, :screen, :basket_size, :hedge]
    size_sweep = sort_table(group_summary(both; by=by))
    mp = matched_percentile(uni)
    all(==(30), mp.n_control) || error("matched_percentile: expected 30 control seeds per row, got ",
        sort(unique(mp.n_control)))
    h2h = sort_table(group_summary(both[in.(both.basket_size, Ref((13, 15))), :]; by=by))
    fx = sort(unique(h2h.basket_size[string.(h2h.screen) .== "fixed"]))
    fx == [13, 15] || error("head_to_head_K13: expected fixed rows at 13 and 15 names, got ", fx)
    return ["size_sweep" => size_sweep, "matched_percentile" => mp, "head_to_head_K13" => h2h,
        "quota_cost" => quota_cost(tier1, uni)]
end

function main(args=ARGS)
    smoke = "--smoke" in args
    pos = filter(a -> !startswith(a, "--"), args)
    dir = isempty(pos) ? joinpath(REPO, "results") : abspath(pos[1])
    outdir = joinpath(dir, "tables")
    mkpath(outdir)
    files = smoke ? ["smoke_summary.csv"] : ["tier1_summary.csv", "fixed_summary.csv"]
    dfs = [CSV.read(joinpath(dir, f), DataFrame) for f in files if isfile(joinpath(dir, f))]
    isempty(dfs) && error("no summary CSV found in ", dir)
    df = reduce((a, b) -> vcat(a, b; cols=:union), dfs)
    s, _ = load_study(CONFIG)
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    tables = [
        "group_summary" => group_summary(df),
        "bandit_vs_random" => paired_win_rate(df),
        "gamma_percentile" => gamma_percentile(df),
        "hedge_effect_sharpe" => hedge_effect(df; metric=:sharpe),
        "hedge_effect_drawdown" => hedge_effect(df; metric=:max_drawdown, higher_is_better=false),
        "long_vs_short_hedge" => long_vs_short_hedge(df; metric=:sharpe),
        "passive" => passive_references(md, s.start, s.stop; W0=s.W0)]
    upath, tpath = joinpath(dir, "universe_summary.csv"), joinpath(dir, "tier1_summary.csv")
    if !smoke && isfile(upath) && isfile(tpath)
        append!(tables, addendum_tables(load_ok(tpath), load_ok(upath)))
    end
    for (name, t0) in tables
        t = sort_table(t0)                     # independent of the summaries' row order (the run hashes)
        CSV.write(joinpath(outdir, name * ".csv"), t)
        println("\n### ", name, "\n\n", markdown_table(t))
    end
    return nothing
end

main()
