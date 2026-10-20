# Task 10: replay the live paper trial (2026-05-20 → 2026-09-04) and compare decisions with its tickets.
# Needs data/validation/ from `julia --project=$TRIAL/code scripts/validation/export_trial.jl`.
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "code")); io=devnull)
using VLINFORMS2026, Dates, CSV, DataFrames, JLD2, Statistics

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const VAL_DIR = joinpath(DATA_DIR, "validation")
const OUT_DIR = joinpath(REPO, "results", "validation")
const START, STOP = Date(2026, 5, 20), Date(2026, 9, 4)
const FORCED = Date.(["2026-05-29", "2026-06-29", "2026-07-28", "2026-08-26", "2026-09-01"])
const σ_G_MAX = 0.12 * sqrt(252)        # yr⁻¹ (trial's 0.12 annualized)
const W_MAX = 0.15                      # dimensionless
const W0 = 100_502.72                   # USD, trial equity after the 2026-05-20 liquidation
const TOL = 0.01                        # dimensionless (1 pp)
const λ_C2 = 0.5                        # placeholder shrinkage until Task 13

function make_cfg(base, σ_m, basket, cov; start=START, forced, natural, cal, kw...)
    return build(MyRunConfig, (rule=MyCobbDouglasRule(), covariance=cov, w_max=W_MAX, s_max=1.0,
        σ_g_max=σ_G_MAX, start=start, stop=start == START ? STOP : start, W0=W0, basket_schedule=[start => basket],
        forced_dates=forced, natural_triggers=natural, ewls_calendar=cal, base_sim=base, σ_m=σ_m,
        σ_m_mode=:fixed, kw...))
end

# Gate rows for the decision dates `dates` of one variant; also the per-name weight comparison.
function gate_rows!(chk, cmp, variant, targets, dates, tt, basket, missing_names)
    for d in dates
        trial = Dict{String,Float64}(String(r.ticker) => r.weight for r in eachrow(tt[tt.date .== d, :]))
        sim = Dict{String,Float64}(t => w for (t, w) in targets[d] if t in basket)
        r = compare_targets(sim, trial; tol=TOL)
        push!(chk, (variant, d, r.pass, r.max_abs_diff, r.worst, join(r.only_sim, ";"), join(r.only_trial, ";"),
            join(missing_names, ";")))
        for t in sort!(collect(union(keys(sim), keys(trial))))
            push!(cmp, (variant, d, t, get(sim, t, 0.0), get(trial, t, 0.0)))
        end
        println(variant, " ", d, " pass=", r.pass, " max|Δw|=", round(r.max_abs_diff; digits=5), " worst=", r.worst)
    end
end

function main()
    mkpath(OUT_DIR)
    md = load_market_data(DATA_DIR; start=Date(2014, 1, 3), stop=STOP)
    sb = load(joinpath(VAL_DIR, "sim_base.jld2"))
    fb = load(joinpath(VAL_DIR, "frozen_basket.jld2"))
    base = Dict(t => build(MySIMParameterEstimate, (α=sb["alpha"][i], β=sb["beta"][i],
        σ_ε=sb["sigma_eps"][i], r²=sb["r_squared"][i])) for (i, t) in enumerate(sb["tickers"]))
    σ_m = sb["sigma_market"]
    inuni = Set(md.tickers)
    basket = [t for t in String.(fb["tickers"]) if t in inuni]
    missing_names = [t for t in String.(fb["tickers"]) if !(t in inuni)]
    isempty(missing_names) || @warn "frozen-basket names missing from the universe" missing_names
    logp = joinpath(VAL_DIR, "trial-log.txt")
    cal = ewls_calendar_from_log(logp, md.dates; start=START, stop=STOP)
    lm = logged_market_from_log(logp, md.dates; start=START, stop=STOP)
    println("EWLS calendar entries: ", length(cal))
    tt = CSV.read(joinpath(VAL_DIR, "trial_targets.csv"), DataFrame)
    C1 = MySIMCovariance()

    # ---- allocation gate (forced dates, C1), three variants
    chk = DataFrame(variant=String[], date=Date[], pass=Bool[], max_abs_diff=Float64[], worst=String[],
        only_sim=String[], only_trial=String[], missing_from_universe=String[])
    cmp = DataFrame(variant=String[], date=Date[], ticker=String[], w_sim=Float64[], w_trial=Float64[])
    # (a) eod: the replay as specified, on this study's own closes
    res = run_backtest(md, make_cfg(base, σ_m, basket, C1; forced=FORCED, natural=false, cal=cal))
    gate_rows!(chk, cmp, "eod", res.targets, FORCED, tt, basket, missing_names)
    # (b) logged_spy: market inputs (gm_t, λ_t at decisions; gm_t in each EWLS update) from the trial's log
    dm = Dict(d => lm.decision[d] for d in FORCED)
    res_b = run_backtest(md, make_cfg(base, σ_m, basket, C1; forced=FORCED, natural=false, cal=cal,
        ewls_g_m=lm.ewls_g_m, decision_market=dm))
    gate_rows!(chk, cmp, "logged_spy", res_b.targets, FORCED, tt, basket, missing_names)
    # (c) state_0529: the trial's exact EWLS state after its 05-29 update + logged 05-29 market inputs
    sdf = CSV.read(joinpath(VAL_DIR, "trial_ewls_0529.csv"), DataFrame)
    st0 = Dict{String,MyEWLSState}()
    for r in eachrow(sdf)
        e = MyEWLSState()
        e.Sw, e.Swx, e.Swy, e.Swxx, e.Swxy, e.Swyy = r.Sw, r.Swx, r.Swy, r.Swxx, r.Swxy, r.Swyy
        e.η, e.α, e.β, e.σ_ε = r.η, r.α, r.β, r.σ_ε
        st0[String(r.ticker)] = e
    end
    d0 = FORCED[1]
    res_c = run_backtest(md, make_cfg(base, σ_m, basket, C1; start=d0, forced=Date[], natural=false, cal=Date[],
        decision_market=Dict(d0 => lm.decision[d0]), ewls_state0=st0))
    gate_rows!(chk, cmp, "state_0529", res_c.targets, [d0], tt, basket, missing_names)
    CSV.write(joinpath(OUT_DIR, "allocation_check_variants.csv"), chk)
    main_chk = select(chk[chk.variant .== "eod", :], Not(:variant))
    CSV.write(joinpath(OUT_DIR, "allocation_check.csv"), main_chk)
    CSV.write(joinpath(OUT_DIR, "targets_compare.csv"),
        select(cmp[cmp.variant .== "eod", :], Not(:variant)))      # (date, ticker, w_sim, w_trial) for Task 16
    cmp.abs_diff = abs.(cmp.w_sim .- cmp.w_trial)
    CSV.write(joinpath(OUT_DIR, "per_name_diff.csv"), sort(cmp, :abs_diff; rev=true))

    # ---- equity check against the trial's V_t
    resid = CSV.read(joinpath(VAL_DIR, "residuals.csv"), DataFrame)
    eq = DataFrame(date=Date[], equity_sim=Float64[], V_t_trial=Float64[], diff=Float64[], diff_pct=Float64[])
    for r in eachrow(resid)
        d = Date(r.decision_date)
        i = searchsortedlast(res.dates, d)
        (i >= 1 && res.dates[i] == d) || continue
        push!(eq, (d, res.equity[i], r.V_t, res.equity[i] - r.V_t, 100 * (res.equity[i] / r.V_t - 1)))
    end
    eq.note = fill("trial V_t includes options P&L (total -882 USD realized by 2026-07-17); replay is equity-only", nrow(eq))
    CSV.write(joinpath(OUT_DIR, "equity_check.csv"), eq)
    println("equity diff (sim - trial): mean=", round(mean(eq.diff); digits=2), " last=", round(eq.diff[end]; digits=2),
        " max|Δ|%=", round(maximum(abs.(eq.diff_pct)); digits=3))

    # ---- informational band check: natural triggers, no forced dates
    bc = DataFrame(variant=String[], date=Date[], reason=String[])
    for (name, cov) in (("C1", MySIMCovariance()), ("C2", build(MyShrunkResidualCovariance, (λ=λ_C2, window=126))))
        c = make_cfg(base, σ_m, basket, cov; forced=Date[], natural=true, cal=cal)
        r = run_backtest(md, c)
        for x in eachrow(r.decisions)
            push!(bc, (name, x.date, string(x.reason)))
        end
    end
    for d in FORCED
        push!(bc, ("trial", d, trial_reason(d)))
    end
    CSV.write(joinpath(OUT_DIR, "band_check.csv"), sort(bc, [:variant, :date]))
    println(sort(bc, [:variant, :date]))
    return all(main_chk.pass)
end

trial_reason(d::Date) = Dict(Date(2026,5,29) => "bug_correction", Date(2026,6,29) => "horizon_elapsed",
    Date(2026,7,28) => "horizon_elapsed", Date(2026,8,26) => "horizon_elapsed", Date(2026,9,1) => "band_exit")[d]

main()
