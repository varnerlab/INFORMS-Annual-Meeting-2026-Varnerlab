# Task 13: comparison tables from the summary CSVs. Usage: 12_tables.jl [dir] [--smoke]
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, CSV, DataFrames

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const CONFIG = joinpath(REPO, "config", "study.toml")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)

function md_table(name, df)
    println("\n### ", name, "\n")
    cols = names(df)
    println("| ", join(cols, " | "), " |")
    println("|", join(fill("---", length(cols)), "|"), "|")
    for r in eachrow(df)
        println("| ", join([x isa AbstractFloat ? string(round(x; sigdigits=4)) : string(x) for x in r], " | "), " |")
    end
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
    for (name, t) in tables
        CSV.write(joinpath(outdir, name * ".csv"), t)
        md_table(name, t)
    end
    return nothing
end

main()
