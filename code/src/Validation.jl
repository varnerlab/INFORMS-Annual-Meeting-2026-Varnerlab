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

One entry per `[EWLS]` line of the trial log at `path`: the latest trading day on or before the
line's date (holiday lines map to the previous trading day; repeats kept). Entries whose mapped day
lies outside [`start`, `stop`] are dropped. `trading_days` is sorted ascending.
"""
function ewls_calendar_from_log(path::String, trading_days::Vector{Date};
        start::Date=first(trading_days), stop::Date=last(trading_days))::Vector{Date}
    out = Date[]
    for line in eachline(path)
        occursin("[EWLS]", line) || continue
        d = Date(line[1:10])
        k = searchsortedlast(trading_days, d)
        k >= 1 || continue
        m = trading_days[k]
        start <= m <= stop && push!(out, m)
    end
    return out
end
