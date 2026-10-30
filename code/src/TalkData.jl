# Data behind the INFORMS 2026 talk figures (docs/superpowers/specs/2026-10-28-informs-talk-slides-design.md): the
# real-run picker × utility grid, the large-book head-to-heads and the numbers quoted in the slide text. Every
# function returns plain tables that scripts/30_talk_data.jl writes to talk/data/.

"""
    TALK_REALRUN_ROWS

The ten ticker pickers of the real-run slide, in slide order, as `(key, label, group)`: `key` is what `picker_key`
returns for a run row of that picker, `label` the row name on the slide and `group` the basket-size group above it.
"""
const TALK_REALRUN_ROWS = [
    ("gamma", "γ per sector", "33 names · 3 per sector"),
    ("bandit", "Bandit per sector", "33 names · 3 per sector"),
    ("random", "Random per sector", "33 names · 3 per sector"),
    ("global_gamma|K=13", "Global γ", "13 names"),
    ("corr_gamma|K=13", "Correlation-aware γ", "13 names"),
    ("liquid_gamma|K=13", "Liquid then γ", "13 names"),
    ("random_liquid|K=13", "Random, liquid 100", "13 names"),
    ("random_global|K=13", "Random, whole universe", "13 names"),
    ("fixed:l8b_13", "Hand-picked 13", "Hand-picked"),
    ("fixed:l8b_15", "Hand-picked 13 + WMT, PG", "Hand-picked"),
]

# Row number (1 = top) and group index (0 = first group) of each TALK_REALRUN_ROWS picker, by key.
function _talk_row_positions()::Dict{String,Tuple{Int,Int}}
    out = Dict{String,Tuple{Int,Int}}()
    group_index = -1
    last_group = ""
    for (i, (key, label, group)) in enumerate(TALK_REALRUN_ROWS)
        if group != last_group
            group_index += 1
            last_group = group
        end
        out[key] = (i, group_index)
    end
    return out
end

# The ok rows of `rows` with a seed in `seeds`, each label once. Throws ArgumentError when two copies of a label
# disagree on `terminal` (USD) by more than 1e-9 of its value.
function _talk_unique_rows(rows::DataFrame, seeds)::DataFrame
    first_row = Dict{String,Int}()
    keep = Int[]
    for i in 1:nrow(rows)
        if string(rows.status[i]) != "ok" || !(rows.seed[i] in seeds)
            continue
        end
        label = string(rows.label[i])
        if haskey(first_row, label)
            j = first_row[label]
            if abs(rows.terminal[i] - rows.terminal[j]) > 1e-9 * abs(rows.terminal[j])
                throw(ArgumentError("label $(label) appears twice with different terminal wealth"))
            end
        else
            first_row[label] = i
            push!(keep, i)
        end
    end
    return rows[keep, :]
end

"""
    talk_realrun_grid(rows::DataFrame; seeds=5660:5689)::DataFrame

The real-run dot table: one row per `TALK_REALRUN_ROWS` picker × hedge setting (off, then on) ×
`TABLE_ALLOCATOR_ORDER` utility, in that order, with columns `row` (1 = top), `group_index` (0 = first group),
`label`, `group`, `allocator`, `hedge`, `value` (the median over the cell's runs of `terminal / W0`, ending wealth ÷
start, dimensionless), `picker` (`picker_key`) and `n_seeds` (runs behind the cell, count).

`rows` holds summary rows (`label`, `status`, `allocator`, `screen`, `seed`, `hedge`, `basket_K`, `W0`, `terminal`
in USD) from one or more summary files. Only `status == "ok"` rows with a seed in `seeds` count, and a label that
appears in two files counts once. Throws `ArgumentError` if two copies of a label disagree on `terminal`, or if a
cell does not have exactly `length(seeds)` runs.
"""
function talk_realrun_grid(rows::DataFrame; seeds=5660:5689)::DataFrame
    d = _talk_unique_rows(rows, seeds)
    pos = _talk_row_positions()
    # the ending wealth ÷ start of each run, by (picker, utility, hedge)
    cells = Dict{Tuple{String,String,Bool},Vector{Float64}}()
    for i in 1:nrow(d)
        key = picker_key(d[i, :])
        if !haskey(pos, key)
            continue
        end
        k = (key, string(d.allocator[i]), Bool(d.hedge[i]))
        if !haskey(cells, k)
            cells[k] = Float64[]
        end
        push!(cells[k], d.terminal[i] / d.W0[i])
    end
    out = DataFrame(row=Int[], group_index=Int[], label=String[], group=String[], allocator=String[], hedge=Bool[],
        value=Float64[], picker=String[], n_seeds=Int[])
    for (key, label, group) in TALK_REALRUN_ROWS
        for hedge in (false, true)
            for a in TABLE_ALLOCATOR_ORDER
                v = get(cells, (key, a, hedge), Float64[])
                if length(v) != length(seeds)
                    throw(ArgumentError("$(label), $(a), hedge = $(hedge) has $(length(v)) runs, expected $(length(seeds))"))
                end
                push!(out, (pos[key][1], pos[key][2], label, group, a, hedge, median(v), key, length(v)))
            end
        end
    end
    return out
end

"""
    talk_row_ranges(dots::DataFrame)::DataFrame

The bar table of a dot table (columns `row`, `group_index`, `label`, `group`, `hedge`, `value`): one row per dot
row and hedge setting, in order of first appearance, with columns `row`, `group_index`, `label`, `group`, `hedge`,
`lo` and `hi` (the lowest and highest `value`, in the units of `value`).
"""
function talk_row_ranges(dots::DataFrame)::DataFrame
    out = DataFrame(row=Int[], group_index=Int[], label=String[], group=String[], hedge=Bool[], lo=Float64[],
        hi=Float64[])
    where_ = Dict{Tuple{Int,Bool},Int}()
    for i in 1:nrow(dots)
        k = (dots.row[i], Bool(dots.hedge[i]))
        v = dots.value[i]
        if haskey(where_, k)
            j = where_[k]
            out.lo[j] = min(out.lo[j], v)
            out.hi[j] = max(out.hi[j], v)
        else
            push!(out, (dots.row[i], dots.group_index[i], dots.label[i], dots.group[i], Bool(dots.hedge[i]), v, v))
            where_[k] = nrow(out)
        end
    end
    return out
end

"""
    talk_largek_shares(rows::DataFrame; sg_max_K::Int=34, n_runs::Union{Nothing,Int}=nothing)::DataFrame

Global γ against random picks from the whole universe at each basket size: one row per (K, hedge setting,
utility), ordered by K, then hedge (off first), then `TABLE_ALLOCATOR_ORDER`, with columns `K` (names), `allocator`,
`hedge`, `n_global` and `n_random` (runs, count), `global_median` and `random_median` (median ending wealth ÷ start)
and `share`, the fraction of (global, random) run pairs in which global γ ends strictly higher.

`rows` holds ok summary rows of screens `global_gamma` and `random_global` (other screens are ignored), with
`allocator`, `screen`, `hedge`, `basket_K`, `W0` and `terminal` (USD). Stone–Geary rows with `basket_K > sg_max_K`
are dropped: its 1%-per-name floor uses the whole budget at 100 names. Throws `ArgumentError` when a (K, utility,
hedge setting) has runs of one screen but not the other, or, when `n_runs` is given, a screen with other than `n_runs`
runs (the slide says "30 runs each").
"""
function talk_largek_shares(rows::DataFrame; sg_max_K::Int=34, n_runs::Union{Nothing,Int}=nothing)::DataFrame
    runs = Dict{Tuple{Int,String,Bool,String},Vector{Float64}}()
    for i in 1:nrow(rows)
        screen = string(rows.screen[i])
        if string(rows.status[i]) != "ok" || !(screen == "global_gamma" || screen == "random_global")
            continue
        end
        K = Int(rows.basket_K[i])
        a = string(rows.allocator[i])
        if a == "SG" && K > sg_max_K
            continue
        end
        k = (K, a, Bool(rows.hedge[i]), screen)
        if !haskey(runs, k)
            runs[k] = Float64[]
        end
        push!(runs[k], rows.terminal[i] / rows.W0[i])
    end
    Ks = Int[]
    for k in keys(runs)
        if !(k[1] in Ks)
            push!(Ks, k[1])
        end
    end
    sort!(Ks)
    out = DataFrame(K=Int[], allocator=String[], hedge=Bool[], n_global=Int[], n_random=Int[],
        global_median=Float64[], random_median=Float64[], share=Float64[])
    for K in Ks
        for hedge in (false, true)
            for a in TABLE_ALLOCATOR_ORDER
                has_g = haskey(runs, (K, a, hedge, "global_gamma"))
                has_r = haskey(runs, (K, a, hedge, "random_global"))
                if has_g != has_r
                    throw(ArgumentError("K = $(K), $(a), hedge = $(hedge) has runs of only one screen"))
                end
                if !has_g
                    continue
                end
                g = runs[(K, a, hedge, "global_gamma")]
                r = runs[(K, a, hedge, "random_global")]
                if n_runs !== nothing && (length(g) != n_runs || length(r) != n_runs)
                    throw(ArgumentError("K = $(K), $(a), hedge = $(hedge) has $(length(g)) global γ and " *
                        "$(length(r)) random runs, expected $(n_runs) each"))
                end
                n_higher = 0
                for x in g
                    for y in r
                        if x > y
                            n_higher += 1
                        end
                    end
                end
                push!(out, (K, a, hedge, length(g), length(r), median(g), median(r), n_higher / (length(g) * length(r))))
            end
        end
    end
    return out
end

"""
    talk_from_wide(t::DataFrame; hedge::Bool, scale::Real=1.0)::NamedTuple{(:dots, :refs)}

A wide synthetic table (column `Ticker picker`, then one column per utility) as the talk's dot and reference tables
for one hedge setting:
- `dots`: one row per `TALK_REALRUN_ROWS` picker × utility, in slide order then `TABLE_ALLOCATOR_ORDER`, with
  columns `row`, `group_index`, `label`, `group`, `allocator`, `hedge`, `value` and `picker` (the picker key). The
  table row of each picker is found by its `PICKER_ROWS` name; pickers not on the slide are left out.
- `refs`: columns `name` and `value` for `"SPY buy-and-hold"` and `"equal-weight universe"`.
`value` is `scale` times the table cell, in the units of the table times `scale` (ending wealth ÷ start for the
wealth tables; `scale = 100` turns a fraction into a percentile). Throws `ArgumentError` when a picker or a reference
row is missing.
"""
function talk_from_wide(t::DataFrame; hedge::Bool, scale::Real=1.0)::NamedTuple{(:dots, :refs)}
    # table row of each name
    where_ = Dict{String,Int}()
    names_ = string.(t[!, "Ticker picker"])
    for i in eachindex(names_)
        where_[names_[i]] = i
    end
    table_name = Dict{String,String}()
    for (key, name) in PICKER_ROWS
        table_name[key] = name
    end

    # one dot per slide picker and utility
    pos = _talk_row_positions()
    dots = DataFrame(row=Int[], group_index=Int[], label=String[], group=String[], allocator=String[], hedge=Bool[],
        value=Float64[], picker=String[])
    for (key, label, group) in TALK_REALRUN_ROWS
        name = table_name[key]
        if !haskey(where_, name)
            throw(ArgumentError("talk_from_wide: no row named $(name)"))
        end
        for a in TABLE_ALLOCATOR_ORDER
            push!(dots, (pos[key][1], pos[key][2], label, group, a, hedge, scale * t[where_[name], a], key))
        end
    end

    # the passive references: the same value in every utility column, so the first one is read
    refs = DataFrame(name=String[], value=Float64[])
    for name in ["SPY buy-and-hold", "equal-weight universe"]
        if !haskey(where_, name)
            throw(ArgumentError("talk_from_wide: no row named $(name)"))
        end
        push!(refs, (name, scale * t[where_[name], TABLE_ALLOCATOR_ORDER[1]]))
    end
    return (dots=dots, refs=refs)
end

"""
    talk_largek_dots(shares::DataFrame)::DataFrame

The large-book head-to-heads (`talk_largek_shares` output) as a dot table: one row per basket size K in increasing
order, labeled `"<K> names"`, all in the group `"Global γ against random picks"`, with columns `row`, `group_index`,
`label`, `group`, `allocator`, `hedge`, `value` (100·share, percent of seed pairs in which global γ ends higher) and
`picker`.
"""
function talk_largek_dots(shares::DataFrame)::DataFrame
    Ks = sort(unique(Int.(shares.K)))
    rank = Dict{Int,Int}()
    for (i, K) in enumerate(Ks)
        rank[K] = i
    end
    out = DataFrame(row=Int[], group_index=Int[], label=String[], group=String[], allocator=String[], hedge=Bool[],
        value=Float64[], picker=String[])
    for i in 1:nrow(shares)
        K = Int(shares.K[i])
        push!(out, (rank[K], 0, "$(K) names", "Global γ against random picks", string(shares.allocator[i]),
            Bool(shares.hedge[i]), 100 * shares.share[i], "global_gamma|K=$(K)"))
    end
    sort!(out, [:row]; alg=MergeSort)
    return out
end

# `x` rounded to `digits` decimals as text, with trailing zeros kept (1.7 with 2 digits is "1.70").
function _talk_decimals(x::Real, digits::Int)::String
    text = string(round(Float64(x); digits=digits))
    if !occursin(".", text)
        text = text * "."
    end
    n = length(text) - findfirst('.', text)
    return text * repeat("0", max(digits - n, 0))
end

"""
    talk_numbers(grid::DataFrame, n_decisions::AbstractVector{<:Integer}, h2h::DataFrame)::Vector{Pair{String,String}}

The numbers quoted in the slide text, as LaTeX macro name => text, in this order:
- `SectorLo`, `SectorHi`: lowest and highest hedge-off γ per sector ending wealth ÷ start in `grid`
  (`talk_realrun_grid`), 2 decimals;
- `BasketLo`, `BasketHi`: lowest and highest hedge-off value over every picker of `grid`, 2 decimals;
- `Rebalances`: the median of `n_decisions` (decisions per run, count, the first allocation included) minus 1,
  rounded;
- `TypicalLo`, `TypicalHi`: lowest and highest `100·share_above_0` (percent) over the `h2h` rows (the
  `typical_head_to_head.csv` table) comparing global γ with random picks at 13 names on wealth with the hedge off,
  rounded.
Throws `ArgumentError` if a selection is empty.
"""
function talk_numbers(grid::DataFrame, n_decisions::AbstractVector{<:Integer}, h2h::DataFrame)::Vector{Pair{String,String}}
    sector = Float64[]
    basket = Float64[]
    for i in 1:nrow(grid)
        if !grid.hedge[i]
            push!(basket, grid.value[i])
            if grid.picker[i] == "gamma"
                push!(sector, grid.value[i])
            end
        end
    end
    shares = Float64[]
    for i in 1:nrow(h2h)
        if h2h.comparison[i] == "Global γ (13) minus Random, whole universe (13)" && h2h.metric[i] == "wealth" &&
                h2h.hedge[i] == "off"
            push!(shares, 100 * h2h.share_above_0[i])
        end
    end
    if isempty(sector) || isempty(basket) || isempty(shares) || isempty(n_decisions)
        throw(ArgumentError("talk_numbers: an empty selection"))
    end
    return ["SectorLo" => _talk_decimals(minimum(sector), 2), "SectorHi" => _talk_decimals(maximum(sector), 2),
        "BasketLo" => _talk_decimals(minimum(basket), 2), "BasketHi" => _talk_decimals(maximum(basket), 2),
        "Rebalances" => string(round(Int, median(n_decisions) - 1)),
        "TypicalLo" => string(round(Int, minimum(shares))), "TypicalHi" => string(round(Int, maximum(shares)))]
end

"""
    write_tex_macros(path::AbstractString, pairs::Vector{Pair{String,String}})::Nothing

Write one `\\newcommand{\\<name>}{<text>}` line per pair to `path`.
"""
function write_tex_macros(path::AbstractString, pairs::Vector{Pair{String,String}})::Nothing
    io = open(path, "w")
    try
        for (name, text) in pairs
            println(io, "\\newcommand{\\", name, "}{", text, "}")
        end
    finally
        close(io)
    end
    return nothing
end

"""
    talk_table_rows(grid::DataFrame, refs::DataFrame; hedge::Bool)::Vector{String}

The body lines of a backup number table for one hedge setting: one `label & v_CD & … & v_LS \\\\` line per row of
`grid` (the `talk_realrun_grid` output), in row order, with the utilities in `TABLE_ALLOCATOR_ORDER`, then one line per
row of `refs` (columns `name`, `value`) with its value in every column. Values are ending wealth ÷ start at 3
decimals. Throws `ArgumentError` when a row lacks a utility.
"""
function talk_table_rows(grid::DataFrame, refs::DataFrame; hedge::Bool)::Vector{String}
    # the cells of this hedge setting, keyed by (row, utility)
    labels = Dict{Int,String}()
    cells = Dict{Tuple{Int,String},Float64}()
    for i in 1:nrow(grid)
        if Bool(grid.hedge[i]) == hedge
            labels[grid.row[i]] = string(grid.label[i])
            cells[(grid.row[i], string(grid.allocator[i]))] = grid.value[i]
        end
    end

    # one line per row, utilities in table order
    lines = String[]
    for r in sort(collect(keys(labels)))
        parts = [labels[r]]
        for a in TABLE_ALLOCATOR_ORDER
            if !haskey(cells, (r, a))
                throw(ArgumentError("talk_table_rows: row $(labels[r]) has no $(a) value"))
            end
            push!(parts, _talk_decimals(cells[(r, a)], 3))
        end
        push!(lines, join(parts, " & ") * " \\\\")
    end

    # the references, the same value in every column
    for i in 1:nrow(refs)
        parts = [string(refs.name[i])]
        for a in TABLE_ALLOCATOR_ORDER
            push!(parts, _talk_decimals(refs.value[i], 3))
        end
        push!(lines, join(parts, " & ") * " \\\\")
    end
    return lines
end

# Index of `d` in `dates`, or 0 when it is not there.
function _talk_index(dates::Vector{Date}, d::Date)::Int
    for i in eachindex(dates)
        if dates[i] == d
            return i
        end
    end
    return 0
end

"""
    talk_mpc_episode(res::MyStudyResult, start::Date, stop::Date; W0::Real=100_000.0)::NamedTuple{(:path, :cones, :events)}

The MPC trigger at work over the trading days `start`..`stop` of one run, for the MPC slide:
- `path`: `day` (trading days since `start`), `date` and `w` (equity ÷ `W0`, dimensionless), one row per day;
- `cones`: `cone` (1 for the forecast made on `start`), `day` (since `start`), `mean` and `sd` (the forecast's mean
  and SD of the book's value ÷ `W0`). Each decision date in [`start`, `stop`) with a stored forecast starts a cone
  at τ = 0 with `mean = w` and `sd = 0`; row τ ≥ 1 holds `μ[τ]` and `σ[τ]` of the forecast, as `check_trigger`
  compares them with the value τ trading days after the decision. A cone runs to the next decision, to the end of
  its forecast or to `stop`, whichever comes first;
- `events`: `day`, `date`, `reason` (the trigger, as text) and `z = (w − mean)/sd` of the cone active that day
  (NaN when there is none), one row per decision in (`start`, `stop`].
Throws `ArgumentError` if `start` or `stop` is not a trading day of `res`, `stop` is before `start`, or no forecast
was stored on `start`.
"""
function talk_mpc_episode(res::MyStudyResult, start::Date, stop::Date;
        W0::Real=100_000.0)::NamedTuple{(:path, :cones, :events)}
    i0 = _talk_index(res.dates, start)
    i1 = _talk_index(res.dates, stop)
    if i0 == 0 || i1 == 0 || i1 < i0
        throw(ArgumentError("start $(start) and stop $(stop) must be trading days of the run, in order"))
    end
    if !haskey(res.projections, start)
        throw(ArgumentError("no forecast was stored on $(start)"))
    end
    # the decisions inside the window, by trading day since start
    dec_days = Int[]
    dec_reasons = String[]
    for i in 1:nrow(res.decisions)
        k = _talk_index(res.dates, res.decisions.date[i])
        if k >= i0 && k <= i1
            push!(dec_days, k - i0)
            push!(dec_reasons, string(res.decisions.reason[i]))
        end
    end
    order = sortperm(dec_days)
    dec_days = dec_days[order]
    dec_reasons = dec_reasons[order]
    w = res.equity[i0:i1] ./ W0
    path = DataFrame(day=collect(0:(i1 - i0)), date=res.dates[i0:i1], w=w)
    # one cone per decision with a stored forecast, cut at the next decision
    cones = DataFrame(cone=Int[], day=Int[], mean=Float64[], sd=Float64[])
    n_cones = 0
    for j in eachindex(dec_days)
        d0 = dec_days[j]
        date0 = res.dates[i0 + d0]
        if d0 >= i1 - i0 || !haskey(res.projections, date0)
            continue
        end
        p = res.projections[date0]
        last_day = min(d0 + length(p.μ), i1 - i0)
        if j < length(dec_days)
            last_day = min(last_day, dec_days[j + 1])
        end
        n_cones += 1
        push!(cones, (n_cones, d0, w[d0 + 1], 0.0))
        for τ in 1:(last_day - d0)
            push!(cones, (n_cones, d0 + τ, p.μ[τ] / W0, p.σ[τ] / W0))
        end
    end
    # each decision after start, with its distance from the cone that ends there
    events = DataFrame(day=Int[], date=Date[], reason=String[], z=Float64[])
    for j in eachindex(dec_days)
        d = dec_days[j]
        if d == 0
            continue
        end
        z = NaN
        for r in nrow(cones):-1:1
            if cones.day[r] == d && cones.sd[r] > 0
                z = (w[d + 1] - cones.mean[r]) / cones.sd[r]
                break
            end
        end
        push!(events, (d, res.dates[i0 + d], dec_reasons[j], z))
    end
    return (path=path, cones=cones, events=events)
end

# Autocorrelation (dimensionless) of `x` at lag `L` (days): cor(x[1:end-L], x[1+L:end]).
function _talk_acf(x::AbstractVector, L::Int)::Float64
    return cor(x[1:(end - L)], x[(1 + L):end])
end

# Kurtosis (dimensionless, non-excess: 3 for a normal) m₄/m₂² of `x`, from central moments.
function _talk_kurtosis(x::AbstractVector)::Float64
    m = mean(x)
    m2 = 0.0
    m4 = 0.0
    for v in x
        m2 += (v - m)^2
        m4 += (v - m)^4
    end
    m2 /= length(x)
    m4 /= length(x)
    return m4 / m2^2
end

# Gaussian kernel density of `x` at each point of `grid` (per unit of `x`), with Silverman's bandwidth
# h = 0.9·min(SD, IQR/1.34)·n^(−1/5).
function _talk_kde(x::AbstractVector, grid::AbstractVector)::Vector{Float64}
    n = length(x)
    spread = min(std(x), (quantile(x, 0.75) - quantile(x, 0.25)) / 1.34)
    h = 0.9 * spread * n^(-1 / 5)
    out = Vector{Float64}(undef, length(grid))
    for (k, u) in enumerate(grid)
        acc = 0.0
        for v in x
            acc += exp(-0.5 * ((u - v) / h)^2)
        end
        out[k] = acc / (n * h * sqrt(2π))
    end
    return out
end

# The q-quantile of each row of `M`, ignoring NaN; NaN for a row with no number.
function _talk_row_quantile(M::Matrix{Float64}, q::Real)::Vector{Float64}
    out = Vector{Float64}(undef, size(M, 1))
    for i in 1:size(M, 1)
        v = Float64[]
        for x in M[i, :]
            if !isnan(x)
                push!(v, x)
            end
        end
        if isempty(v)
            out[i] = NaN
        else
            out[i] = quantile(v, q)
        end
    end
    return out
end

"""
    TALK_QQ_PROBS

The probabilities of the JumpHMM slide's quantile panel: the 0.1st to 99.9th percentile, dense in the tails.
"""
const TALK_QQ_PROBS = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995,
    0.9975, 0.999]

"""
    talk_stylized_facts(g_real, g_syn; lags=1:20, abs_lags=1:252, grid=-10.0:0.1:10.0, probs=TALK_QQ_PROBS)
        ::NamedTuple{(:density, :qq, :acf, :abs_acf, :summary)}

The stylized facts of daily growth rates (yr⁻¹) for the JumpHMM slide, after Fig. 3 of the JumpHMM paper: one
real series `g_real` against the synthetic series `g_syn` (a vector of series), as the real value and the 10th,
50th and 90th percentiles across the synthetic series:
- `density`: `x` (yr⁻¹, the points of `grid`), `real`, `syn_p10`, `syn_p50`, `syn_p90`, the Gaussian kernel
  density (per yr⁻¹) of each series, and `normal`, the normal density with the real series' mean and SD;
- `qq`: `p` (the probabilities `probs`), `real`, `syn_p10`, `syn_p50`, `syn_p90`, the p-quantile (yr⁻¹) of each
  series;
- `acf`: `lag` (trading days, `lags`), `real`, `syn_p10`, `syn_p50`, `syn_p90`, the autocorrelation of g;
- `abs_acf`: the same for |g| at the lags `abs_lags`;
- `summary`: `series` (`"real"`, `"synthetic median"`), `kurtosis` (non-excess, 3 for a normal), `acf1` and
  `abs_acf1` (lag 1); the synthetic row is the median across the synthetic series.
"""
function talk_stylized_facts(g_real::AbstractVector{<:Real}, g_syn::AbstractVector{<:AbstractVector{<:Real}};
        lags=1:20, abs_lags=1:252, grid=-10.0:0.1:10.0,
        probs=TALK_QQ_PROBS)::NamedTuple{(:density, :qq, :acf, :abs_acf, :summary)}
    x = collect(grid)
    ns = length(g_syn)
    D = Matrix{Float64}(undef, length(x), ns)
    Q = Matrix{Float64}(undef, length(probs), ns)
    A = Matrix{Float64}(undef, length(lags), ns)
    B = Matrix{Float64}(undef, length(abs_lags), ns)
    kurt = Float64[]
    for j in 1:ns
        D[:, j] = _talk_kde(g_syn[j], x)
        Q[:, j] = quantile(g_syn[j], probs)
        for (k, L) in enumerate(lags)
            A[k, j] = _talk_acf(g_syn[j], L)
        end
        B[:, j] = abs_acf(g_syn[j], abs_lags)
        push!(kurt, _talk_kurtosis(g_syn[j]))
    end
    # the normal reference has the real series' mean and SD
    m = mean(g_real)
    sd = std(g_real)
    normal = Float64[]
    for u in x
        push!(normal, exp(-0.5 * ((u - m) / sd)^2) / (sd * sqrt(2π)))
    end
    real_acf = Float64[]
    for L in lags
        push!(real_acf, _talk_acf(g_real, L))
    end
    real_abs = abs_acf(g_real, abs_lags)
    density = DataFrame(x=x, real=_talk_kde(g_real, x), syn_p10=_talk_row_quantile(D, 0.1),
        syn_p50=_talk_row_quantile(D, 0.5), syn_p90=_talk_row_quantile(D, 0.9), normal=normal)
    qq = DataFrame(p=collect(probs), real=quantile(g_real, probs), syn_p10=_talk_row_quantile(Q, 0.1),
        syn_p50=_talk_row_quantile(Q, 0.5), syn_p90=_talk_row_quantile(Q, 0.9))
    acf = DataFrame(lag=collect(lags), real=real_acf, syn_p10=_talk_row_quantile(A, 0.1),
        syn_p50=_talk_row_quantile(A, 0.5), syn_p90=_talk_row_quantile(A, 0.9))
    absacf = DataFrame(lag=collect(abs_lags), real=real_abs, syn_p10=_talk_row_quantile(B, 0.1),
        syn_p50=_talk_row_quantile(B, 0.5), syn_p90=_talk_row_quantile(B, 0.9))
    summary = DataFrame(series=["real", "synthetic median"], kurtosis=[_talk_kurtosis(g_real), median(kurt)],
        acf1=[real_acf[1], median(A[1, :])], abs_acf1=[real_abs[1], median(B[1, :])])
    return (density=density, qq=qq, acf=acf, abs_acf=absacf, summary=summary)
end
