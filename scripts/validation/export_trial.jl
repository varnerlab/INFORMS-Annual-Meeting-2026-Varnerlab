# Export the live trial's tickets and artifacts to data/validation/ (read-only on the trial repo).
# Run in the TRIAL environment: julia --project=$TRIAL/code scripts/validation/export_trial.jl
using LiveCD, JLD2, CSV, DataFrames, Dates

const TRIAL = expanduser("~/Desktop/julia_work/constrained-CD-with-MPC-paper-trade-trial")
const OUT = normpath(joinpath(@__DIR__, "..", "..", "data", "validation"))
const DATES = ["2026-05-29", "2026-06-29", "2026-07-28", "2026-08-26", "2026-09-01"]

function main()
    mkpath(OUT)
    rows = DataFrame(date=Date[], ticker=String[], weight=Float64[])
    for d in DATES
        t = LiveCD.load_ticket(joinpath(TRIAL, "data", "tickets", "submitted-$(d).jld2"))
        p = t.pending
        @assert Date(p.date) == Date(d) "ticket date $(p.date) != $(d)"
        println(d, " reason=", p.trigger_reason, " n=", length(p.tickers))
        for (tk, w) in zip(p.tickers, p.target_weights)
            push!(rows, (Date(d), String(tk), Float64(w)))
        end
    end
    CSV.write(joinpath(OUT, "trial_targets.csv"), rows)
    for f in ["artifacts/sim_base.jld2", "artifacts/frozen_basket.jld2", "data/trial-log.txt", "data/residuals.csv"]
        cp(joinpath(TRIAL, f), joinpath(OUT, basename(f)); force=true)
    end
    println("wrote ", nrow(rows), " target rows to ", OUT)
end

main()
