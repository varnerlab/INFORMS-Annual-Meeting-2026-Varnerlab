# Cross-seed comparison tables and passive references for the screened-panel study.

# Copy of a summary with `screen` as a Symbol and error rows dropped.
function _tables_prepare(df::DataFrame)::DataFrame
    out = "status" in names(df) || :status in propertynames(df) ? df[string.(df.status) .== "ok", :] : copy(df)
    out = copy(out)
    out.screen = Symbol.(out.screen)
    return out
end

_tables_ok(x) = !ismissing(x) && !(x isa Real && isnan(x))

# Pair rows of `left` and `right` on `keys` (plus the group columns `by`), then summarize the metric.
# Returns one row per group with `n_pairs`, `win_rate` (ties count half) and `median_diff` (left − right).
function _tables_paired(left::DataFrame, right::DataFrame, by::Vector{Symbol}, pairkeys::Vector{Symbol},
        metric::Symbol, higher_is_better::Bool)::DataFrame
    keys = vcat(by, pairkeys)
    L = select(left[_tables_ok.(left[!, metric]), :], keys..., metric => :_a)
    R = select(right[_tables_ok.(right[!, metric]), :], keys..., metric => :_b)
    J = innerjoin(L, R; on=keys)
    sgn = higher_is_better ? 1.0 : -1.0
    rows = []
    for sub in groupby(J, by)
        d = Float64.(sub._a) .- Float64.(sub._b)
        win = mean(ifelse.(d .== 0, 0.5, ifelse.(sgn .* d .> 0, 1.0, 0.0)))
        push!(rows, (; (c => sub[1, c] for c in by)..., n_pairs=nrow(sub), win_rate=win, median_diff=median(d)))
    end
    isempty(rows) && return DataFrame([c => Any[] for c in by]..., :n_pairs => Int[], :win_rate => Float64[],
        :median_diff => Float64[])
    return DataFrame(rows)
end

"""Default `group_summary` metrics: the growth scorecard (Sharpe, max drawdown, α in yr⁻¹, its t-stat, β,
realized growth in yr⁻¹) and, next to it, the invested fraction and the hedge statistics (`mean_invested`,
`cap_bind_share`, `mean_hedge_weight`, fractions; `cap_bind_share` is NaN with the hedge off)."""
const SUMMARY_METRICS = [:sharpe, :max_drawdown, :alpha, :alpha_t, :beta, :realized_growth, :mean_invested,
    :cap_bind_share, :mean_hedge_weight]

"""
    group_summary(df; by, metrics=SUMMARY_METRICS) -> DataFrame

Per group of `by`: `n` (rows, count) and, for each metric `m`, `m_median`, `m_q25`, `m_q75`
(median and quartiles across seeds, in the metric's own units; missing and NaN values skipped, and NaN
when a group has none). `screen` is normalized to a `Symbol`; rows with `status != "ok"` are dropped when
that column exists.
"""
function group_summary(df::DataFrame; by::Vector{Symbol}=[:allocator, :screen, :hedge, :repick_on_band_exit],
        metrics::Vector{Symbol}=SUMMARY_METRICS)::DataFrame
    d = _tables_prepare(df)
    rows = []
    for sub in groupby(d, by)
        r = Pair{Symbol,Any}[c => sub[1, c] for c in by]
        push!(r, :n => nrow(sub))
        for m in metrics
            v = Float64.(filter(_tables_ok, collect(sub[!, m])))
            push!(r, Symbol(m, "_median") => (isempty(v) ? NaN : median(v)))
            push!(r, Symbol(m, "_q25") => (isempty(v) ? NaN : quantile(v, 0.25)))
            push!(r, Symbol(m, "_q75") => (isempty(v) ? NaN : quantile(v, 0.75)))
        end
        push!(rows, NamedTuple(r))
    end
    return DataFrame(rows)
end

"""
    paired_win_rate(df; metric=:sharpe, a=:bandit, b=:random, by, higher_is_better=true) -> DataFrame

Pairs screen `a` with screen `b` by `seed` within each group of `by`. Per group: `n_pairs` (count),
`win_rate` (fraction of pairs where `a` beats `b`; ties count half) and `median_diff`
(median of metric(a) − metric(b), in the metric's units). Groups with no pairs are left out.
"""
function paired_win_rate(df::DataFrame; metric::Symbol=:sharpe, a::Symbol=:bandit, b::Symbol=:random,
        by::Vector{Symbol}=[:allocator, :hedge, :repick_on_band_exit], higher_is_better::Bool=true)::DataFrame
    d = _tables_prepare(df)
    return _tables_paired(d[d.screen .== a, :], d[d.screen .== b, :], by, [:seed], metric, higher_is_better)
end

"""
    gamma_percentile(df; metric=:sharpe, by, screen=:gamma, control=:random, higher_is_better=true) -> DataFrame

Per group of `by` that has a `screen` row: `gamma_value` (metric units), `n_random` (count of
control-screen seeds), `percentile` (fraction in [0,1] of the control screen's values below the
screen value; ties count half) and `share_of_controls_beaten` (fraction of control seeds the screen beats,
ties half: `percentile` when `higher_is_better`, else the share of control values above the screen's, e.g.
deeper drawdowns), which reads the same way for every metric. Groups with no screen row or no control
rows are left out.
"""
function gamma_percentile(df::DataFrame; metric::Symbol=:sharpe,
        by::Vector{Symbol}=[:allocator, :hedge, :repick_on_band_exit],
        screen::Symbol=:gamma, control::Symbol=:random, higher_is_better::Bool=true)::DataFrame
    d = _tables_prepare(df)
    d = d[_tables_ok.(d[!, metric]), :]
    rows = []
    for sub in groupby(d[d.screen .== screen, :], by)
        rnd = d[d.screen .== control, :]
        for c in by
            rnd = rnd[rnd[!, c] .== sub[1, c], :]
        end
        isempty(rnd) && continue
        gv = Float64(sub[1, metric]); rv = Float64.(rnd[!, metric])
        tie = 0.5 * count(==(gv), rv)
        pct = (count(<(gv), rv) + tie) / length(rv)
        beaten = higher_is_better ? pct : (count(>(gv), rv) + tie) / length(rv)
        push!(rows, (; (c => sub[1, c] for c in by)..., gamma_value=gv, n_random=length(rv), percentile=pct,
            share_of_controls_beaten=beaten))
    end
    isempty(rows) && return DataFrame([c => Any[] for c in by]..., :gamma_value => Float64[],
        :n_random => Int[], :percentile => Float64[], :share_of_controls_beaten => Float64[])
    return DataFrame(rows)
end

"""
    hedge_effect(df; metric, by, higher_is_better=true) -> DataFrame

Pairs hedge on with hedge off by `seed` within each group of `by`. Per group: `n_pairs` (count),
`win_rate` (fraction of pairs where hedge on is better; ties count half) and `median_diff`
(median of metric(on) − metric(off), in the metric's units).
"""
function hedge_effect(df::DataFrame; metric::Symbol,
        by::Vector{Symbol}=[:allocator, :screen, :repick_on_band_exit], higher_is_better::Bool=true)::DataFrame
    d = _tables_prepare(df)
    return _tables_paired(d[d.hedge .== true, :], d[d.hedge .== false, :], by, [:seed], metric, higher_is_better)
end

"""
    long_vs_short_hedge(df; metric, higher_is_better=true) -> DataFrame

Pairs the long hedge (`allocator == "CD"` with `hedge`) against the 130/30 short hedge
(`allocator == "LS"` with no hedge) by `(screen, seed, repick_on_band_exit)`. Per `(screen,
repick_on_band_exit)`: `n_pairs` (count), `win_rate` (fraction where the long hedge is better; ties
count half) and `median_diff` (metric(CD) − metric(LS), in the metric's units).
"""
function long_vs_short_hedge(df::DataFrame; metric::Symbol, higher_is_better::Bool=true)::DataFrame
    d = _tables_prepare(df)
    cd = d[(string.(d.allocator) .== "CD") .& (d.hedge .== true), :]
    ls = d[(string.(d.allocator) .== "LS") .& (d.hedge .== false), :]
    return _tables_paired(cd, ls, [:screen, :repick_on_band_exit], [:seed], metric, higher_is_better)
end

"""
    passive_references(md, start, stop; W0=100_000.0) -> DataFrame

One `growth_scorecard` row (growth-rate metrics, yr⁻¹ unless dimensionless) per passive reference over
the trading days `start`..`stop`, with a `name` column: "SPY buy-and-hold" (`W0` USD in SPY at the
`start` close) and "equal-weight universe" (`W0/n` USD in each of the `n` tickers at the `start`
close, no rebalancing). Both are valued at the closes; the risk-free growth rate is `md.g_f`.
"""
function passive_references(md::MyMarketData, start::Date, stop::Date; W0::Real=100_000.0)::DataFrame
    i0, i1 = index_of(md, start), index_of(md, stop)
    spy = md.spy_close[i0:i1]
    gf = md.g_f[i0:i1]
    n = length(md.tickers)
    shares = (W0 / n) ./ md.close[i0, :]
    ew = md.close[i0:i1, :] * shares
    rows = [merge((name="SPY buy-and-hold",), growth_scorecard(Float64(W0) .* spy ./ spy[1], spy, gf)),
        merge((name="equal-weight universe",), growth_scorecard(Float64.(ew), spy, gf))]
    return DataFrame(rows)
end

"""Allocator order of the written tables (`sort_table`)."""
const TABLE_ALLOCATOR_ORDER = ["CD", "CES", "SG", "MinVar", "Tangent", "LS"]
"""Screen order of the written tables (`sort_table`); other screens follow, by name."""
const TABLE_SCREEN_ORDER = ["gamma", "bandit", "random", "global_gamma", "liquid_gamma", "corr_gamma",
    "random_global", "random_liquid", "fixed"]

_table_rank(v, order) = (something(findfirst(==(string(v)), order), length(order) + 1), string(v))

"""
    sort_table(t::DataFrame)::DataFrame

A sorted copy of `t`, so a written table does not depend on the row order of the summaries (which follow the
run hashes). Keys, for the columns `t` has: `allocator` (`TABLE_ALLOCATOR_ORDER`), `pair`, `metric`,
`screen` (`TABLE_SCREEN_ORDER`, others after by name), `basket_size` (ascending), `hedge` (on first),
`repick_on_band_exit` (on first). A table with none of them is returned unchanged.
"""
function sort_table(t::DataFrame)::DataFrame
    cols = names(t); ord = Any[]
    "allocator" in cols && push!(ord, order(:allocator; by=a -> _table_rank(a, TABLE_ALLOCATOR_ORDER)))
    "pair" in cols && push!(ord, order(:pair))
    "metric" in cols && push!(ord, order(:metric; by=string))
    "screen" in cols && push!(ord, order(:screen; by=x -> _table_rank(x, TABLE_SCREEN_ORDER)))
    "basket_size" in cols && push!(ord, order(:basket_size))
    "hedge" in cols && push!(ord, order(:hedge; rev=true))
    "repick_on_band_exit" in cols && push!(ord, order(:repick_on_band_exit; rev=true))
    return isempty(ord) ? copy(t) : sort(t, ord)
end

"""
    markdown_table(df::DataFrame)::String

GitHub-style Markdown table of `df`: a header row, a `|---|` separator row and one row per line, each
ending in a newline. Floats print with 4 significant digits; other values print with `string`.
"""
function markdown_table(df::DataFrame)::String
    cols = names(df)
    cell(x) = x isa AbstractFloat ? string(round(x; sigdigits=4)) : string(x)
    io = IOBuffer()
    println(io, "| ", join(cols, " | "), " |")
    println(io, "|", join(fill("---", length(cols)), "|"), "|")
    for r in eachrow(df)
        println(io, "| ", join((cell(x) for x in r), " | "), " |")
    end
    return String(take!(io))
end

"""
    compare_summaries(old, new; key=:label, cols=[...]) -> NamedTuple

Joins two summary tables on `key`. Returns `(n_matched, n_missing, max_abs_diff)`: rows matched, keys of
`old` absent from `new` (counts), and the largest |old - new| over `cols` and matched rows, in each
column's units. A cell that is missing (or NaN) on both sides counts as 0; one missing or NaN on only
one side makes `max_abs_diff` `Inf`, so the comparison fails rather than passing falsely.
"""
function compare_summaries(old::DataFrame, new::DataFrame; key::Symbol=:label,
        cols=[:sharpe, :max_drawdown, :alpha, :alpha_t, :beta, :realized_growth, :terminal, :vol])::NamedTuple{(:n_matched, :n_missing, :max_abs_diff)}
    idx = Dict(k => i for (i, k) in enumerate(new[!, key]))
    nm = 0; nmiss = 0; mx = 0.0
    for (i, k) in enumerate(old[!, key])
        j = get(idx, k, 0)
        if j == 0
            nmiss += 1; continue
        end
        nm += 1
        for c in cols
            a, b = old[i, c], new[j, c]
            bad(x) = ismissing(x) || (x isa Number && isnan(x))
            (bad(a) && bad(b)) && continue
            if bad(a) || bad(b)
                mx = Inf; continue
            end
            mx = max(mx, abs(Float64(a) - Float64(b)))
        end
    end
    return (n_matched=nm, n_missing=nmiss, max_abs_diff=mx)
end
