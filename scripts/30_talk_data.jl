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
const TRAIN_START, TRAIN_STOP = Date(2014, 1, 3), Date(2024, 12, 31)
const MPC_LABEL = "CD|fixed:l8b_13|hedge=false|seed=5660"     # the MPC slide's run (spec, slide 3)
const MPC_START, MPC_STOP = Date(2025, 10, 10), Date(2025, 12, 22)
const MPC_REASONS = ["horizon_elapsed", "band_exit", "horizon_elapsed"]
const MPC_Z = 1.96                                             # the trigger's band, SDs
const N_SYN_PATHS = 100                                        # generator paths on the JumpHMM slide
const CLAIM_KURTOSIS, CLAIM_ACF1, CLAIM_ABS_ACF = 4.0, 0.15, 0.10   # fat tails, little autocorrelation, clustering
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

Write `df` to talk/data/`name` and say so. A text field that holds a comma is wrapped in braces, as in
`{Random, liquid 100}`: pgfplotstable reads a braced cell as one cell, and it does not understand double quotes.
"""
function wrote(name::AbstractString, df::DataFrame)::Nothing
    CSV.write(joinpath(OUT, name), df; openquotechar='{', closequotechar='}')
    println("wrote talk/data/", name, " (", nrow(df), " rows)")
    return nothing
end

"""
    check_mpc(ep)::Nothing

Throw an error unless the MPC episode shows what the slide says: the decisions `MPC_REASONS`, a band exit beyond
`MPC_Z` SDs, and every day strictly inside each cone within `MPC_Z` SDs of its forecast.
"""
function check_mpc(ep)::Nothing
    if ep.events.reason != MPC_REASONS
        error("MPC episode decisions are ", ep.events.reason, ", expected ", MPC_REASONS)
    end
    k = findfirst(isequal("band_exit"), ep.events.reason)
    if !(abs(ep.events.z[k]) > MPC_Z)
        error("MPC band exit is only ", ep.events.z[k], " SDs from its forecast")
    end
    for c in unique(ep.cones.cone)
        rows = ep.cones[ep.cones.cone .== c, :]
        for r in 2:(nrow(rows) - 1)
            z = (ep.path.w[rows.day[r] + 1] - rows.mean[r]) / rows.sd[r]
            if abs(z) > MPC_Z
                error("MPC cone ", c, " is left on day ", rows.day[r], " (z = ", z, ") without a re-plan")
            end
        end
    end
    return nothing
end

"""
    check_claims(summary::DataFrame, abs_acf::DataFrame)::Nothing

Throw an error unless the data supports what the JumpHMM slide says: both the real series and the synthetic median
show fat tails (kurtosis above `CLAIM_KURTOSIS`) and little autocorrelation (|ACF at lag 1| below `CLAIM_ACF1`);
real SPY shows lasting volatility clustering (ACF of |g| above `CLAIM_ABS_ACF` at lags 1–5); and the generator
shows short-lived clustering (synthetic median ACF of |g| above `CLAIM_ABS_ACF` at lag 1, below it by lag 3). The
slide says the generator's clustering fades after a day while real clustering lasts weeks.
"""
function check_claims(summary::DataFrame, abs_acf::DataFrame)::Nothing
    for i in 1:nrow(summary)
        if !(summary.kurtosis[i] > CLAIM_KURTOSIS) || !(abs(summary.acf1[i]) < CLAIM_ACF1)
            error("stylized-fact claim fails for ", summary.series[i], ": ", summary[i, :])
        end
    end
    for L in 1:5
        if !(abs_acf.real[L] > CLAIM_ABS_ACF)
            error("real volatility clustering claim fails at lag ", L, ": ", abs_acf[L, :])
        end
    end
    if !(abs_acf.syn_p50[1] > CLAIM_ABS_ACF) || !(abs_acf.syn_p50[3] < CLAIM_ABS_ACF)
        error("the generator's clustering is not short-lived as the slide says: ", abs_acf[1:3, :])
    end
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
    ref_table = DataFrame(name=refs.name, value=refs.terminal ./ 100_000.0)
    wrote("realrun_refs.csv", ref_table)

    # the backup number tables: one tabular body per hedge setting
    for (hedge, tag) in ((false, "off"), (true, "on"))
        lines = talk_table_rows(grid, ref_table; hedge=hedge)
        write(joinpath(OUT, "b3_rows_hedge_$(tag).tex"), join(lines, "\n") * "\n")
        println("wrote talk/data/b3_rows_hedge_", tag, ".tex (", length(lines), " rows)")
    end

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

    # the MPC slide: one real episode of the hand-picked 13, read from the run's cache file
    k = findfirst(isequal(MPC_LABEL), string.(gs.label))
    res = JLD2.load(joinpath(RESULTS, "gamma_seeds", "runs", string(gs.hash[k]) * ".jld2"), "result")
    ep = talk_mpc_episode(res, MPC_START, MPC_STOP)
    check_mpc(ep)
    wrote("mpc_path.csv", ep.path)
    wrote("mpc_events.csv", ep.events)
    for c in unique(ep.cones.cone)
        rows = ep.cones[ep.cones.cone .== c, :]
        wrote("mpc_cone_$(c).csv", rows[:, [:day, :mean, :sd]])
    end
    println("MPC events: ", join([string(ep.events.date[i], " ", ep.events.reason[i], " z=", round(ep.events.z[i]; digits=2))
        for i in 1:nrow(ep.events)], "; "))

    # the JumpHMM slide: real SPY 2014–2024 against the first typical paths of this year's generator
    rows = growth_window(md.dates, TRAIN_START, TRAIN_STOP)
    g_real = Float64[]
    for r in rows
        push!(g_real, log(md.spy_close[r + 1] / md.spy_close[r]) * 252)
    end
    gen = JLD2.load(joinpath(RESULTS, "synthetic", "generator.jld2"), "generator")
    paths = CSV.read(joinpath(RESULTS, "synthetic", "paths.csv"), DataFrame)
    ids = sort(paths.path_id[paths.typical .== true])[1:N_SYN_PATHS]
    window = growth_window(gen.dates, TRAIN_START, TRAIN_STOP)
    g_syn = Vector{Float64}[]
    for p in ids
        push!(g_syn, synthetic_draws(gen, p).g_m[window])
    end
    sf = talk_stylized_facts(g_real, g_syn)
    println(sf.summary)
    check_claims(sf.summary, sf.abs_acf)
    wrote("sf_density.csv", sf.density)
    wrote("sf_acf.csv", sf.acf)
    wrote("sf_abs_acf.csv", sf.abs_acf)
    wrote("sf_summary.csv", sf.summary)
    return nothing
end

main()
