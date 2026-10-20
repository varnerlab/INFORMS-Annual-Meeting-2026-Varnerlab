# Gate evidence: the replay's per-name preference weights γ at the 5 trial decision dates next to the
# sim and trial target weights. Mirrors the engine's EWLS schedule (trial-log calendar). Run after 02_validate_trial.jl.
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..", "code")); io=devnull)
using VLINFORMS2026, Dates, CSV, DataFrames, JLD2

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const START, STOP = Date(2026, 5, 20), Date(2026, 9, 4)
const FORCED = Date.(["2026-05-29", "2026-06-29", "2026-07-28", "2026-08-26", "2026-09-01"])

function main()
    md = load_market_data(joinpath(REPO, "data"); start=Date(2014, 1, 3), stop=STOP)
    vd = joinpath(REPO, "data", "validation")
    sb = load(joinpath(vd, "sim_base.jld2")); fb = load(joinpath(vd, "frozen_basket.jld2"))
    basket = String.(fb["tickers"]); idx = Dict(t => i for (i, t) in enumerate(sb["tickers"]))
    cols = [findfirst(==(t), md.tickers) for t in basket]
    ews = [ewls_init(sb["alpha"][idx[t]], sb["beta"][idx[t]], sb["sigma_eps"][idx[t]]) for t in basket]
    cal = ewls_calendar_from_log(joinpath(vd, "trial-log.txt"), md.dates; start=START, stop=STOP)
    cmp = CSV.read(joinpath(REPO, "results", "validation", "targets_compare.csv"), DataFrame)
    out = DataFrame(date=Date[], ticker=String[], γ_sim=Float64[], γ_logged_market=Float64[], w_sim=Float64[], w_trial=Float64[])
    logged = Dict{Date,NamedTuple}()            # the trial's own [MARKET] gm_t, λ_t per day (last line wins)
    for ln in eachline(joinpath(vd, "trial-log.txt"))
        m = match(r"^(\d{4}-\d{2}-\d{2}) .*\[MARKET\] gm_t=(\S+)\s+λ_t=(\S+)", ln)
        m === nothing || (logged[Date(m[1])] = (g_m=parse(Float64, m[2]), λ=parse(Float64, m[3])))
    end
    i0 = index_of(md, START)
    for t in i0:searchsortedlast(md.dates, STOP)
        d = md.dates[t]
        g_m = log(md.spy_close[t] / md.spy_close[t-1]) * 252
        for _ in 1:count(==(d), cal), (k, j) in enumerate(cols)
            ewls_update!(ews[k], log(md.close[t, j] / md.close[t-1, j]) * 252, g_m)
        end
        d in FORCED || continue
        ms = market_state(md.spy_close, md.dates, d)
        γ = preference_weights([e.α for e in ews], [e.β for e in ews], ms.g_m, ms.λ)
        γL = preference_weights([e.α for e in ews], [e.β for e in ews], logged[d].g_m, logged[d].λ)
        println("   logged market state g_m=", logged[d].g_m, " λ=", logged[d].λ, " #γ>0=", count(>(0), γL))
        c = cmp[cmp.date .== d, :]
        for (k, tk) in enumerate(basket)
            r = c[c.ticker .== tk, :]
            push!(out, (d, tk, γ[k], γL[k], isempty(r) ? 0.0 : r.w_sim[1], isempty(r) ? 0.0 : r.w_trial[1]))
        end
        println(d, " g_m=", round(ms.g_m; digits=2), " λ=", round(ms.λ; digits=3), " #γ>0=", count(>(0), γ))
    end
    CSV.write(joinpath(REPO, "results", "validation", "gamma_compare.csv"), out)
    for d in FORCED
        println(d)
        o = out[out.date .== d, :]; o = o[sortperm(abs.(o.w_sim .- o.w_trial); rev=true), :]
        show(first(o, 4); allrows=true); println()
    end
end

main()
