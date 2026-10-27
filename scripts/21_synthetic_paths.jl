# Task 7 of the synthetic-market plan: draw 10,000 synthetic market paths over the real 2014–2026 calendar, rank them by
# drawdown in the 2025–26 trading window, and pick the burst-stress (200), bear-stress (200) and typical (600) sets.
# Writes results/synthetic/paths.csv. Run with threads: julia --project=code -t 30 scripts/21_synthetic_paths.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, Statistics, DataFrames, CSV, JLD2

const REPO = normpath(joinpath(@__DIR__, ".."))
const OUT_DIR = joinpath(REPO, "results", "synthetic")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const TRAIN = (Date(2014, 1, 3), Date(2024, 12, 31))
const Δt = 1 / 252                               # yr

t_start = time()
gen = JLD2.load(joinpath(OUT_DIR, "generator.jld2"), "generator")
stats = market_path_stats(gen, 1:10_000)
sets = path_sets(stats)
CSV.write(joinpath(OUT_DIR, "paths.csv"), sets)

# real SPY training growth, yr⁻¹, for the V1 preview
md = load_market_data(joinpath(REPO, "data"); start=MD_START, stop=MD_STOP)
i0 = findfirst(==(TRAIN[1]), md.dates)
i1 = findfirst(==(TRAIN[2]), md.dates)
real_growth = log(md.spy_close[i1] / md.spy_close[i0]) / ((i1 - i0) * Δt)

# share of the 50 deepest drawdowns with a burst
deep = sortperm(sets.window_max_drawdown; rev=true)[1:50]

println("rows: ", nrow(sets))
println("sets: burst_stress ", count(sets.burst_stress), ", bear_stress ", count(sets.bear_stress), ", typical ", count(sets.typical),
    ", paths in any set ", count(!ismissing, sets.run_order))
println("share of windows with a burst: ", round(100 * mean(sets.has_burst); digits=2), " %")
println("share of the 50 deepest drawdowns with a burst: ", round(100 * mean(sets.has_burst[deep]); digits=1), " %")
println("median train_growth, synthetic: ", round(median(sets.train_growth); digits=4), " yr⁻¹; real SPY 2014–2024: ",
    round(real_growth; digits=4), " yr⁻¹")
for (name, mask) in (("burst_stress", sets.burst_stress), ("bear_stress", sets.bear_stress), ("typical", sets.typical))
    d = sets.window_max_drawdown[mask]
    println("drawdown range, ", name, ": ", round(minimum(d); digits=4), " to ", round(maximum(d); digits=4),
        " (median ", round(median(d); digits=4), ")")
end
println("wall time: ", round(time() - t_start; digits=1), " s")
