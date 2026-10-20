# Validation helpers: compare a replay's target weights and EWLS-update calendar with the live trial.
# Units: weights are dimensionless fractions of equity.

const _VAL_HELD_W = 1e-3        # |w| above which a name counts as held (dimensionless)

"""
    compare_targets(sim, trial; tol=0.01)

Compare target weights (dimensionless, fraction of equity) name by name. The names with
w > 1e-3 must be the same set in both dicts, and every |Δw| over the union must be ≤ `tol`.
Returns `(pass, only_sim, only_trial, max_abs_diff, worst)`; a missing name counts as weight 0.
"""
function compare_targets(sim::Dict{String,Float64}, trial::Dict{String,Float64}; tol::Float64=0.01)
    held(d) = Set(k for (k, v) in d if v > _VAL_HELD_W)
    only_sim = sort!(collect(setdiff(held(sim), held(trial))))
    only_trial = sort!(collect(setdiff(held(trial), held(sim))))
    mx, worst = 0.0, ""
    for k in sort!(collect(union(keys(sim), keys(trial))))
        δ = abs(get(sim, k, 0.0) - get(trial, k, 0.0))
        if δ > mx
            mx, worst = δ, k
        end
    end
    return (pass = isempty(only_sim) && isempty(only_trial) && mx <= tol,
        only_sim = only_sim, only_trial = only_trial, max_abs_diff = mx, worst = worst)
end

"""
    ewls_calendar_from_log(path, trading_days; start=first(trading_days), stop=last(trading_days))::Vector{Date}

One entry per `[EWLS] updated=` line of the trial log at `path` whose own date lies in
[`start`, `stop`]: the latest trading day on or before that date (holiday lines map to the previous
trading day; repeats kept). `[EWLS] skipped ...` lines are not updates and are ignored (R32), and the
window filter uses the line's raw date, before mapping (R30). `trading_days` is sorted ascending.
"""
function ewls_calendar_from_log(path::String, trading_days::Vector{Date};
        start::Date=first(trading_days), stop::Date=last(trading_days))::Vector{Date}
    out = Date[]
    for line in eachline(path)
        occursin("[EWLS] updated=", line) || continue
        d = Date(line[1:10])
        start <= d <= stop || continue
        k = searchsortedlast(trading_days, d)
        k >= 1 && push!(out, trading_days[k])
    end
    return out
end

"""
    logged_market_from_log(path, trading_days; start, stop)

The trial's own market inputs from its log. Returns `(ewls_g_m, decision)`:
`ewls_g_m::Vector{Float64}` (yr⁻¹) holds, for each entry of `ewls_calendar_from_log` (same filter and
order), the `gm_t` of the latest `[MARKET]` line on the same raw date at or before the `[EWLS] updated=`
line (`NaN` when there is none); `decision::Dict{Date,NamedTuple{(:g_m,:λ)}}` maps each trading day to
the last `[MARKET]` line `(gm_t [yr⁻¹], λ_t [dimensionless])` logged on that day.
"""
function logged_market_from_log(path::String, trading_days::Vector{Date};
        start::Date=first(trading_days), stop::Date=last(trading_days))
    ewls_g_m = Float64[]
    decision = Dict{Date,NamedTuple{(:g_m, :λ),Tuple{Float64,Float64}}}()
    last_day, last_gm = Date(0), NaN
    for line in eachline(path)
        d = tryparse(Date, length(line) >= 10 ? line[1:10] : "")
        d === nothing && continue
        m = match(r"\[MARKET\] gm_t=(\S+)\s+λ_t=(\S+)", line)
        if m !== nothing
            gm, λ = parse(Float64, m[1]), parse(Float64, m[2])
            last_day, last_gm = d, gm
            decision[d] = (g_m = gm, λ = λ)
        elseif occursin("[EWLS] updated=", line) && start <= d <= stop
            push!(ewls_g_m, last_day == d ? last_gm : NaN)
        end
    end
    return (ewls_g_m = ewls_g_m, decision = filter(p -> p.first in trading_days, decision))
end
