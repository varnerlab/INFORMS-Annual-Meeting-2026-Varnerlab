# Addendum Task 5: end-to-end regression check of the base grids. Compares the rerun summaries
# (results/tier1_summary.csv, results/fixed_summary.csv) with the versions committed at HEAD, joined on
# `label`, over every growth-scorecard metric and run statistic the committed summaries carry. The new
# settings fields change every run's hash, so the join is on the label, not the hash.
# Exits 1 unless, for both files, every committed label is present (n_missing == 0) and every metric is
# exactly equal (max_abs_diff == 0.0). Usage (from the repo root): julia scripts/15_regression_check.jl
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, CSV, DataFrames

const REPO = normpath(joinpath(@__DIR__, ".."))
const REF = "HEAD"
const FILES = ["results/tier1_summary.csv", "results/fixed_summary.csv"]
# Growth-scorecard metrics (yr⁻¹ or dimensionless) and run statistics present in the committed summaries.
const METRICS = [:sharpe, :max_drawdown, :alpha, :alpha_t, :beta, :realized_growth, :terminal, :vol,
    :turnover, :total_cost, :borrow_fees, :n_decisions, :n_band_exit, :n_drawdown, :n_horizon,
    :n_cash_revisit, :n_basket_change, :n_repicks, :n_basket_changes, :mean_names, :mean_invested,
    :mean_portfolio_beta, :cap_bind_share, :mean_hedge_weight, :mean_pool_size]

function main()
    pass = true
    for f in FILES
        old = CSV.read(IOBuffer(read(Cmd(`git show $(REF):$(f)`; dir=REPO))), DataFrame)
        new = CSV.read(joinpath(REPO, f), DataFrame)
        r = compare_summaries(old, new; key=:label, cols=METRICS)
        nerr = count(!=("ok"), string.(new.status))
        uniq = allunique(old.label) && allunique(new.label)
        println(f, ": committed rows=", nrow(old), " rerun rows=", nrow(new), " rerun errors=", nerr,
            " unique labels=", uniq)
        println("  compare_summaries over ", length(METRICS), " metrics: ", r)
        ok = r.n_missing == 0 && r.max_abs_diff == 0.0 && nrow(old) == nrow(new) && nerr == 0 && uniq
        println("  ", ok ? "PASS" : "FAIL")
        pass &= ok
    end
    pass || (println(stderr, "regression check failed"); exit(1))
    println("regression check passed")
    return nothing
end

main()
