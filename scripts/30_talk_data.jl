# INFORMS 2026 talk (docs/superpowers/specs/2026-10-28-informs-talk-slides-design.md): writes the data behind every
# slide figure and the numbers quoted in the slide text to talk/data/, from committed results, and checks the slide
# claims against the data. The deck in talk/ reads only these files.
# Usage (from the repo root): OMP_NUM_THREADS=1 KMP_GTID_MODE=2 julia scripts/30_talk_data.jl
# Ending wealth ÷ start is terminal / W0; growth rates are g = ln(S_t/S_{t-1})/Δt in 1/yr with Δt = 1/252.
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, DataFrames, CSV, JLD2, Dates, Statistics

const REPO = normpath(joinpath(@__DIR__, ".."))
const RESULTS = joinpath(REPO, "results")
const OUT = joinpath(REPO, "talk", "data")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const REAL_START, REAL_STOP = Date(2025, 1, 2), Date(2026, 9, 4)
const REAL_SEEDS = 5660:5689
const ROW_COLUMNS = [:hash, :label, :status, :allocator, :screen, :seed, :hedge, :basket_K, :W0, :terminal, :n_decisions]

"""
    read_rows(path::AbstractString)::DataFrame

The summary rows of `path` with the columns `ROW_COLUMNS`.
"""
function read_rows(path::AbstractString)::DataFrame
    return CSV.read(path, DataFrame; select=ROW_COLUMNS)
end

"""
    wrote(name::AbstractString, df::DataFrame)::Nothing

Write `df` to talk/data/`name` and say so.
"""
function wrote(name::AbstractString, df::DataFrame)::Nothing
    CSV.write(joinpath(OUT, name), df)
    println("wrote talk/data/", name, " (", nrow(df), " rows)")
    return nothing
end

function main()
    mkpath(OUT)

    # the real run: random and bandit pickers from the study summaries, the γ pickers and hand-picked lists from the
    # 30-seed rerun (their seed-5660 rows appear in both and count once)
    real = vcat(read_rows(joinpath(RESULTS, "tier1_summary.csv")), read_rows(joinpath(RESULTS, "universe_summary.csv")),
        read_rows(joinpath(RESULTS, "gamma_seeds", "gamma_seeds_summary.csv")))
    grid = talk_realrun_grid(real; seeds=REAL_SEEDS)
    wrote("realrun_grid.csv", grid)
    wrote("realrun_bars.csv", talk_row_ranges(grid))

    # the passive references over the real window
    md = load_market_data(joinpath(REPO, "data"); start=MD_START, stop=MD_STOP)
    refs = passive_references(md, REAL_START, REAL_STOP)
    wrote("realrun_refs.csv", DataFrame(name=refs.name, value=refs.terminal ./ 100_000.0))

    # global γ against random picks: K = 5–34 from the rerun and the universe summary, K = 100 and 200 from the
    # large-book runs
    big = read_rows(joinpath(RESULTS, "large_k", "large_k_summary.csv"))
    gs = read_rows(joinpath(RESULTS, "gamma_seeds", "gamma_seeds_summary.csv"))
    un = read_rows(joinpath(RESULTS, "universe_summary.csv"))
    head = vcat(gs[string.(gs.screen) .== "global_gamma", :], un[string.(un.screen) .== "random_global", :], big)
    wrote("largek_shares.csv", talk_largek_shares(head))

    # the numbers in the slide text; decisions per run over the runs of the real-run slide
    keys_ = Set(first.(TALK_REALRUN_ROWS))
    n_decisions = Int[]
    seen = Set{String}()
    for i in 1:nrow(real)
        label = string(real.label[i])
        if string(real.status[i]) == "ok" && real.seed[i] in REAL_SEEDS && !(label in seen) &&
                picker_key(real[i, :]) in keys_
            push!(seen, label)
            push!(n_decisions, real.n_decisions[i])
        end
    end
    h2h = CSV.read(joinpath(RESULTS, "synthetic", "tables", "typical_head_to_head.csv"), DataFrame)
    numbers = talk_numbers(grid, n_decisions, h2h)
    write_tex_macros(joinpath(OUT, "numbers.tex"), numbers)
    println("wrote talk/data/numbers.tex: ", join([string(k, "=", v) for (k, v) in numbers], ", "))
    return nothing
end

main()
