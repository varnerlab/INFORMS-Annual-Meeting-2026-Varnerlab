# Picker × utility tables of the synthetic stress test (spec §7): the per-path rows of script 22, summarized across
# synthetic paths. Within a path the median over seeds is taken first, then the median (or a share) across paths.

"""
    PICKER_ROWS

The 14 ticker pickers of the synthetic tables, in table order, as `(key, name)`: `key` is what `picker_key` returns
for a run row of that picker, and `name` is its plain name with the name count of its basket.
"""
const PICKER_ROWS = [
    ("gamma", "γ per sector (33)"),
    ("bandit", "Bandit per sector (33)"),
    ("random", "Random per sector (33)"),
    ("global_gamma|K=5", "Global γ (5)"),
    ("global_gamma|K=13", "Global γ (13)"),
    ("global_gamma|K=34", "Global γ (34)"),
    ("corr_gamma|K=13", "Correlation-aware γ (13)"),
    ("liquid_gamma|K=13", "Liquid then γ (13)"),
    ("random_global|K=5", "Random, whole universe (5)"),
    ("random_global|K=13", "Random, whole universe (13)"),
    ("random_global|K=34", "Random, whole universe (34)"),
    ("random_liquid|K=13", "Random, liquid 100 (13)"),
    ("fixed:l8b_13", "User's 13 names (13)"),
    ("fixed:l8b_15", "User's 13 + WMT, PG (15)"),
]

# The two passive reference rows at the top of every grid, as `passive_references` names them.
const _SYN_SPY = "SPY buy-and-hold"
const _SYN_REFERENCES = [_SYN_SPY, "equal-weight universe"]

# The cell metrics: wealth (ending wealth ÷ start), realized growth (yr⁻¹), max drawdown (fraction), Sharpe.
const _SYN_METRICS = [:wealth, :realized_growth, :max_drawdown, :sharpe]

# The head-to-head picker pairs (first minus second), by `picker_key`.
const _SYN_PAIRS = [("global_gamma|K=5", "random_global|K=5"), ("global_gamma|K=13", "random_global|K=13"),
    ("global_gamma|K=34", "random_global|K=34"), ("gamma", "random")]

"""
    picker_key(row)::String

The `PICKER_ROWS` key of a run row (a `DataFrameRow` or a `NamedTuple` with `screen`, `basket_K` and `label`):
`screen` itself for the per-sector screens `gamma`, `bandit` and `random` (their basket is 33 names whatever
`basket_K` says); the `fixed:<list>` part of `label` for screen `fixed`; otherwise `"<screen>|K=<basket_K>"`.
Throws `ArgumentError` for a `fixed` row whose label has no `fixed:` part.
"""
function picker_key(row)::String
    screen = string(row.screen)
    if screen == "gamma" || screen == "bandit" || screen == "random"
        return screen
    end
    if screen == "fixed"
        for part in split(string(row.label), "|")
            if startswith(part, "fixed:")
                return String(part)
            end
        end
        throw(ArgumentError("fixed-list row has no fixed:<list> part in its label $(row.label)"))
    end
    return "$(screen)|K=$(Int(row.basket_K))"
end

# Plain name of a PICKER_ROWS key.
function _syn_picker_name(key::String)::String
    for (k, name) in PICKER_ROWS
        if k == key
            return name
        end
    end
    throw(ArgumentError("picker key $(key) is not in PICKER_ROWS"))
end

# Median of the values that are neither missing nor NaN; NaN when there are none.
function _syn_median(v)::Float64
    x = Float64[]
    for a in v
        if !ismissing(a) && !isnan(a)
            push!(x, Float64(a))
        end
    end
    if isempty(x)
        return NaN
    end
    return median(x)
end

# `x` as text with 3 decimals ("-0.140", "1.200"); "n/a" when `x` is NaN or infinite.
function _syn_fixed3(x::Real)::String
    if !isfinite(x)
        return "n/a"
    end
    m = round(Int, 1000 * x)
    sign = ""
    if m < 0
        sign = "-"
    end
    m = abs(m)
    return string(sign, div(m, 1000), ".", lpad(string(rem(m, 1000)), 3, '0'))
end

# "median (p10–p90)" of `v` with 3 decimals; "n/a" when `v` is empty.
function _syn_band(v::Vector{Float64})::String
    if isempty(v)
        return "n/a"
    end
    return string(_syn_fixed3(median(v)), " (", _syn_fixed3(quantile(v, 0.1)), "–", _syn_fixed3(quantile(v, 0.9)), ")")
end

"""
    path_cells(rows::DataFrame)::DataFrame

One cell per (`path_id`, picker, `allocator`, `hedge`) of the run rows `rows` (path files of script 22, or a real
summary with a `path_id` column added). Only `status == "ok"` rows count. Columns: `path_id`, `picker`
(`picker_key`), `allocator`, `hedge`, `n_seeds` (the ok rows behind the cell, count) and the median over those rows
of `wealth` (`terminal / W0`, ending wealth ÷ start, dimensionless), `realized_growth` (yr⁻¹), `max_drawdown`
(fraction) and `sharpe` (dimensionless). Missing and NaN values are skipped; a metric with none is NaN.
"""
function path_cells(rows::DataFrame)::DataFrame
    # the ok rows of each cell, in order of first appearance
    groups = Dict{Tuple{Int,String,String,Bool},Vector{Int}}()
    order = Tuple{Int,String,String,Bool}[]
    for i in 1:nrow(rows)
        if ismissing(rows.status[i]) || string(rows.status[i]) != "ok"
            continue
        end
        r = rows[i, :]
        key = (Int(r.path_id), picker_key(r), string(r.allocator), Bool(r.hedge))
        if !haskey(groups, key)
            groups[key] = Int[]
            push!(order, key)
        end
        push!(groups[key], i)
    end
    # the median over the cell's rows of each metric
    out = DataFrame(path_id=Int[], picker=String[], allocator=String[], hedge=Bool[], n_seeds=Int[],
        wealth=Float64[], realized_growth=Float64[], max_drawdown=Float64[], sharpe=Float64[])
    for key in order
        idx = groups[key]
        wealth = Float64[]
        for i in idx
            push!(wealth, rows.terminal[i] / rows.W0[i])
        end
        push!(out, (key[1], key[2], key[3], key[4], length(idx), _syn_median(wealth),
            _syn_median(rows.realized_growth[idx]), _syn_median(rows.max_drawdown[idx]), _syn_median(rows.sharpe[idx])))
    end
    return out
end

"""
    bench_cells(bench::DataFrame; W0::Real=100_000.0)::DataFrame

The passive references of each path (bench files of script 22: one row per `path_id` and `name`, "SPY buy-and-hold"
and "equal-weight universe") as cells: `path_id`, `name`, `wealth` (`terminal / W0`, dimensionless, with `W0` the
starting wealth in USD), `realized_growth` (yr⁻¹), `max_drawdown` (fraction) and `sharpe` (dimensionless).
"""
function bench_cells(bench::DataFrame; W0::Real=100_000.0)::DataFrame
    out = DataFrame(path_id=Int[], name=String[], wealth=Float64[], realized_growth=Float64[],
        max_drawdown=Float64[], sharpe=Float64[])
    for i in 1:nrow(bench)
        push!(out, (Int(bench.path_id[i]), string(bench.name[i]), bench.terminal[i] / W0, bench.realized_growth[i],
            bench.max_drawdown[i], bench.sharpe[i]))
    end
    return out
end

# (path_id, picker, allocator) → value of `metric` for the cells with hedge `hedge`; NaN values are left out.
function _syn_cell_lookup(cells::DataFrame, metric::Symbol, hedge::Bool)::Dict{Tuple{Int,String,String},Float64}
    out = Dict{Tuple{Int,String,String},Float64}()
    v = cells[!, metric]
    for i in 1:nrow(cells)
        if cells.hedge[i] == hedge && !isnan(v[i])
            out[(cells.path_id[i], cells.picker[i], cells.allocator[i])] = v[i]
        end
    end
    return out
end

# (path_id, name) → value of `metric` for the reference cells.
function _syn_bench_lookup(bench::DataFrame, metric::Symbol)::Dict{Tuple{Int,String},Float64}
    out = Dict{Tuple{Int,String},Float64}()
    v = bench[!, metric]
    for i in 1:nrow(bench)
        out[(bench.path_id[i], bench.name[i])] = v[i]
    end
    return out
end

# One grid cell from the values `vals` of a cell on its paths and the SPY values `spys` of the same paths.
function _syn_grid_stat(vals::Vector{Float64}, spys::Vector{Float64}, stat::Symbol, lower_is_better::Bool)
    if stat == :band
        return _syn_band(vals)
    end
    if isempty(vals)
        return NaN
    end
    if stat == :median
        return median(vals)
    end
    # :beat_spy, strictly better than the path's SPY
    n = 0
    for k in eachindex(vals)
        if lower_is_better && vals[k] < spys[k]
            n += 1
        elseif !lower_is_better && vals[k] > spys[k]
            n += 1
        end
    end
    return n / length(vals)
end

"""
    grid_table(cells, bench, metric::Symbol; ids, hedge::Bool, stat::Symbol)::DataFrame

The picker × utility grid of `metric` (`:wealth`, ending wealth ÷ start; `:realized_growth`, yr⁻¹; `:max_drawdown`,
fraction; `:sharpe`) across the paths `ids`, from `cells` (`path_cells`) with hedge setting `hedge` and `bench`
(`bench_cells`). Rows: "SPY buy-and-hold" and "equal-weight universe" (the same value in every utility column), then
the names of `PICKER_ROWS`. Columns: `Ticker picker`, then the utilities of `TABLE_ALLOCATOR_ORDER`. A cell uses the
paths of `ids` that have it. `stat`:
- `:median`: the median across those paths (NaN when there are none);
- `:band`: the text "median (p10–p90)" across those paths, with 3 decimals ("n/a" when there are none);
- `:beat_spy`: the share of those paths on which the cell is strictly better than that path's SPY (higher is better,
  except `:max_drawdown`, where lower is better); 0 for the SPY row itself; NaN when there are none.
Throws `ArgumentError` for another `stat` or `metric`, or for a path of `ids` with no SPY row in `bench`.
"""
function grid_table(cells::DataFrame, bench::DataFrame, metric::Symbol; ids::AbstractVector{<:Integer}, hedge::Bool,
        stat::Symbol)::DataFrame
    if !(stat in (:median, :band, :beat_spy))
        throw(ArgumentError("stat must be :median, :band or :beat_spy; got $(stat)"))
    end
    if !(metric in _SYN_METRICS)
        throw(ArgumentError("metric must be one of $(_SYN_METRICS); got $(metric)"))
    end
    lower_is_better = metric == :max_drawdown
    cell = _syn_cell_lookup(cells, metric, hedge)
    ref = _syn_bench_lookup(bench, metric)
    for p in ids
        if !haskey(ref, (p, _SYN_SPY))
            throw(ArgumentError("path $(p) has no \"$(_SYN_SPY)\" row in bench"))
        end
    end
    labels = String[]
    columns = [Any[] for a in TABLE_ALLOCATOR_ORDER]
    # the reference rows, the same in every utility column
    for name in _SYN_REFERENCES
        vals = Float64[]
        spys = Float64[]
        for p in ids
            if haskey(ref, (p, name))
                push!(vals, ref[(p, name)])
                push!(spys, ref[(p, _SYN_SPY)])
            end
        end
        x = _syn_grid_stat(vals, spys, stat, lower_is_better)
        push!(labels, name)
        for k in eachindex(TABLE_ALLOCATOR_ORDER)
            push!(columns[k], x)
        end
    end
    # the picker rows
    for (key, name) in PICKER_ROWS
        push!(labels, name)
        for (k, a) in enumerate(TABLE_ALLOCATOR_ORDER)
            vals = Float64[]
            spys = Float64[]
            for p in ids
                if haskey(cell, (p, key, a))
                    push!(vals, cell[(p, key, a)])
                    push!(spys, ref[(p, _SYN_SPY)])
                end
            end
            push!(columns[k], _syn_grid_stat(vals, spys, stat, lower_is_better))
        end
    end
    out = DataFrame("Ticker picker" => labels)
    for (k, a) in enumerate(TABLE_ALLOCATOR_ORDER)
        out[!, a] = identity.(columns[k])
    end
    return out
end

"""
    format_grid(t::DataFrame; beat=nothing)::DataFrame

A copy of the grid `t` (`grid_table` with `stat = :median` or `:beat_spy`, or `real_percentiles`) with every utility
cell as text with 3 decimals ("n/a" for NaN). With `beat` (the `:beat_spy` grid of the same rows), a cell whose share
of paths beating SPY is above 0.5 is set in bold (`**…**`). Throws `ArgumentError` if `beat` has other rows.
"""
function format_grid(t::DataFrame; beat::Union{Nothing,DataFrame}=nothing)::DataFrame
    if beat !== nothing && beat[!, "Ticker picker"] != t[!, "Ticker picker"]
        throw(ArgumentError("beat must have the same Ticker picker rows as t"))
    end
    out = DataFrame("Ticker picker" => copy(t[!, "Ticker picker"]))
    for a in TABLE_ALLOCATOR_ORDER
        col = String[]
        for i in 1:nrow(t)
            text = _syn_fixed3(t[i, a])
            if beat !== nothing && beat[i, a] > 0.5
                text = "**" * text * "**"
            end
            push!(col, text)
        end
        out[!, a] = col
    end
    return out
end

# A head-to-head row: the per-path differences `d` summarized (n_paths, median, p10, p90, share above 0).
function _syn_diff_row(comparison::String, metric::String, hedge::String, allocator::String, d::Vector{Float64})
    if isempty(d)
        return (comparison=comparison, metric=metric, hedge=hedge, allocator=allocator, n_paths=0, median=NaN,
            p10=NaN, p90=NaN, share_above_0=NaN)
    end
    return (comparison=comparison, metric=metric, hedge=hedge, allocator=allocator, n_paths=length(d),
        median=median(d), p10=quantile(d, 0.1), p90=quantile(d, 0.9), share_above_0=count(>(0), d) / length(d))
end

"""
    head_to_head(cells::DataFrame; ids)::DataFrame

Per-path differences of `cells` (`path_cells`) over the paths `ids`, per utility. Rows, each for every utility of
`TABLE_ALLOCATOR_ORDER`:
- for hedge "off", then "on": global γ minus random, whole universe at K = 5, 13 and 34, then γ per sector minus
  random per sector, in `wealth` (ending wealth ÷ start); `comparison` is "<name> minus <name>";
- for hedge "on minus off": the hedge-on cell minus the hedge-off cell of each `PICKER_ROWS` picker, in
  `realized_growth` (yr⁻¹), then in `max_drawdown` (fraction); `comparison` is the picker's name.
Columns: `comparison`, `metric`, `hedge`, `allocator`, `n_paths` (paths of `ids` with both cells, count), and the
`median`, `p10`, `p90` (10th and 90th percentiles) of the differences, in the metric's units, and `share_above_0`
(fraction of those paths with a difference above 0). With no such path, `n_paths` is 0 and the rest NaN.
"""
function head_to_head(cells::DataFrame; ids::AbstractVector{<:Integer})::DataFrame
    rows = []
    # picker minus picker, ending wealth, per hedge setting
    for (hedge, hedge_text) in ((false, "off"), (true, "on"))
        L = _syn_cell_lookup(cells, :wealth, hedge)
        for (a_key, b_key) in _SYN_PAIRS
            comparison = _syn_picker_name(a_key) * " minus " * _syn_picker_name(b_key)
            for alloc in TABLE_ALLOCATOR_ORDER
                d = Float64[]
                for p in ids
                    if haskey(L, (p, a_key, alloc)) && haskey(L, (p, b_key, alloc))
                        push!(d, L[(p, a_key, alloc)] - L[(p, b_key, alloc)])
                    end
                end
                push!(rows, _syn_diff_row(comparison, "wealth", hedge_text, alloc, d))
            end
        end
    end
    # hedge on minus hedge off, per picker
    for metric in (:realized_growth, :max_drawdown)
        on = _syn_cell_lookup(cells, metric, true)
        off = _syn_cell_lookup(cells, metric, false)
        for (key, name) in PICKER_ROWS
            for alloc in TABLE_ALLOCATOR_ORDER
                d = Float64[]
                for p in ids
                    if haskey(on, (p, key, alloc)) && haskey(off, (p, key, alloc))
                        push!(d, on[(p, key, alloc)] - off[(p, key, alloc)])
                    end
                end
                push!(rows, _syn_diff_row(name, string(metric), "on minus off", alloc, d))
            end
        end
    end
    out = DataFrame()
    for r in rows
        push!(out, r)
    end
    return out
end

"""
    head_to_head_grid(h::DataFrame; hedge::String, metric::String)::DataFrame

The rows of `h` (`head_to_head`) with this `hedge` ("off", "on" or "on minus off") and `metric` as a grid: one row
per `comparison`, in the order of `h`, then one column per utility of `TABLE_ALLOCATOR_ORDER`. Each cell is the text
"median (p10–p90); share above 0", with 3 decimals, or "n/a" when no path has both cells.
"""
function head_to_head_grid(h::DataFrame; hedge::String, metric::String)::DataFrame
    comparisons = String[]
    text = Dict{Tuple{String,String},String}()
    for i in 1:nrow(h)
        if h.hedge[i] != hedge || h.metric[i] != metric
            continue
        end
        if !(h.comparison[i] in comparisons)
            push!(comparisons, h.comparison[i])
        end
        if h.n_paths[i] == 0
            text[(h.comparison[i], h.allocator[i])] = "n/a"
        else
            text[(h.comparison[i], h.allocator[i])] = string(_syn_fixed3(h.median[i]), " (", _syn_fixed3(h.p10[i]), "–",
                _syn_fixed3(h.p90[i]), "); ", _syn_fixed3(h.share_above_0[i]))
        end
    end
    out = DataFrame("Comparison" => comparisons)
    for a in TABLE_ALLOCATOR_ORDER
        col = String[]
        for c in comparisons
            push!(col, get(text, (c, a), "n/a"))
        end
        out[!, a] = col
    end
    return out
end

"""
    real_percentiles(cells, real_cells; ids, hedge::Bool, bench, real_bench)::DataFrame

Where the real 2025–26 run falls among the paths `ids`: the grid (rows and columns of `grid_table`) of the share of
those paths whose ending wealth ÷ start is at or below the real one, with hedge setting `hedge`. `cells` and `bench`
are the synthetic `path_cells` and `bench_cells`; `real_cells` is `path_cells` of the real summary rows (one cell
per picker, utility and hedge setting, whatever its `path_id`) and `real_bench` is `bench_cells` of the real
passive references. A cell with no real value, or no path of `ids` that has it, is NaN. Throws `ArgumentError` if
`real_cells` has two cells for one picker, utility and hedge setting, or `real_bench` two rows for one name.
"""
function real_percentiles(cells::DataFrame, real_cells::DataFrame; ids::AbstractVector{<:Integer}, hedge::Bool,
        bench::DataFrame, real_bench::DataFrame)::DataFrame
    syn = _syn_cell_lookup(cells, :wealth, hedge)
    ref = _syn_bench_lookup(bench, :wealth)
    # the real cells and the real references
    real = Dict{Tuple{String,String},Float64}()
    for i in 1:nrow(real_cells)
        if real_cells.hedge[i] == hedge
            key = (real_cells.picker[i], real_cells.allocator[i])
            if haskey(real, key)
                throw(ArgumentError("real_cells has more than one cell for $(key), hedge = $(hedge)"))
            end
            real[key] = real_cells.wealth[i]
        end
    end
    real_ref = Dict{String,Float64}()
    for i in 1:nrow(real_bench)
        if haskey(real_ref, real_bench.name[i])
            throw(ArgumentError("real_bench has more than one row for $(real_bench.name[i])"))
        end
        real_ref[real_bench.name[i]] = real_bench.wealth[i]
    end
    labels = String[]
    columns = [Float64[] for a in TABLE_ALLOCATOR_ORDER]
    # the reference rows, the same in every utility column
    for name in _SYN_REFERENCES
        share = NaN
        if haskey(real_ref, name)
            n = 0
            n_at_or_below = 0
            for p in ids
                if haskey(ref, (p, name))
                    n += 1
                    if ref[(p, name)] <= real_ref[name]
                        n_at_or_below += 1
                    end
                end
            end
            if n > 0
                share = n_at_or_below / n
            end
        end
        push!(labels, name)
        for k in eachindex(TABLE_ALLOCATOR_ORDER)
            push!(columns[k], share)
        end
    end
    # the picker rows
    for (key, name) in PICKER_ROWS
        push!(labels, name)
        for (k, a) in enumerate(TABLE_ALLOCATOR_ORDER)
            share = NaN
            if haskey(real, (key, a)) && !isnan(real[(key, a)])
                n = 0
                n_at_or_below = 0
                for p in ids
                    if haskey(syn, (p, key, a))
                        n += 1
                        if syn[(p, key, a)] <= real[(key, a)]
                            n_at_or_below += 1
                        end
                    end
                end
                if n > 0
                    share = n_at_or_below / n
                end
            end
            push!(columns[k], share)
        end
    end
    out = DataFrame("Ticker picker" => labels)
    for (k, a) in enumerate(TABLE_ALLOCATOR_ORDER)
        out[!, a] = columns[k]
    end
    return out
end
