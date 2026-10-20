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
    # exact EWLS state the trial held after its 2026-05-29 16:15 update (pre-force backup)
    st = LiveCD.load_state(joinpath(TRIAL, "data", "trial-state.backup-2026-05-29-pre-force.jld2"))
    sdf = DataFrame(ticker=String[], Sw=Float64[], Swx=Float64[], Swy=Float64[], Swxx=Float64[],
        Swxy=Float64[], Swyy=Float64[], η=Float64[], α=Float64[], β=Float64[], σ_ε=Float64[])
    for t in st.tickers
        e = st.ewls[t]
        push!(sdf, (String(t), e.Sw, e.Swx, e.Swy, e.Swxx, e.Swxy, e.Swyy, e.η, e.α, e.β, e.σ_ε))
    end
    CSV.write(joinpath(OUT, "trial_ewls_0529.csv"), sdf)
    println("wrote ", nrow(rows), " target rows to ", OUT)
end

main()
