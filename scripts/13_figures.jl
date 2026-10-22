# Task 14: slide figures (PNG, 1600×900) from results/*_summary.csv and results/runs/, written to
# results/figures/. Addendum Task 5 adds figs 6–7 (size sweep) when results/universe_summary.csv exists. Settings come from config/study.toml and λ from results/calibration_summary.toml.
# Usage (from the repo root): julia --project=scripts scripts/13_figures.jl
# One-time setup: julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
# Every performance number is a growth-rate metric: g = ln(S_t/S_{t-1})/Δt in 1/yr with Δt = 1/252.
using Pkg
Pkg.activate(@__DIR__; io=devnull)
using VLINFORMS2026, Plots, DataFrames, CSV, JLD2, Dates, Statistics, Random, TOML, Printf

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const RESULTS = joinpath(REPO, "results")
const FIG_DIR = joinpath(RESULTS, "figures")
const CONFIG = joinpath(REPO, "config", "study.toml")
const CALIB_TOML = joinpath(RESULTS, "calibration_summary.toml")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const SIZE = (1600, 900)

# Chart chrome and categorical slots: the dataviz reference palette on its light surface. The first three
# slots pass the all-pairs CVD and normal-vision gates (validate_palette.js --pairs all); slot 3 (aqua) is
# below 3:1 contrast, so the γ screen also has its own marker shape and every value is in results/tables/.
# The screen → slot map is fixed, so a screen has the same color in every figure. Screens sit left to
# right in SCREENS order within each allocator slot, and the legend lists them in the same order.
const SURFACE = "#fcfcfb"
const INK, INK2 = "#0b0b0b", "#52514e"
const GRID, AXIS = "#e1e0d9", "#c3c2b7"
const SLOT = ("#2a78d6", "#eb6834", "#1baf7a")
const SCREENS = ("bandit", "gamma", "random")
const SCREEN_COLOR = Dict("bandit" => SLOT[1], "random" => SLOT[2], "gamma" => SLOT[3])
const SCREEN_OFFSET = Dict("bandit" => -0.24, "gamma" => 0.0, "random" => 0.24)
const LEGEND_X = Dict("bandit" => 0.03, "gamma" => 0.265, "random" => 0.556, "median" => 0.801)
const ALLOCATORS = ["CD", "CES", "CESw", "SG", "MinVar", "Tangent", "LS"]
const ALLOC_NAME = Dict("CD" => "Cobb-Douglas", "CES" => "CES (share counts)", "CESw" => "CES (weights)",
    "SG" => "Stone-Geary", "MinVar" => "min variance", "Tangent" => "tangency", "LS" => "130/30 CD")
# Two-line allocator names for the seven-slot x axes of the strip figures.
const TICK_NAME = Dict("CD" => "Cobb-\nDouglas", "CES" => "CES\n(share\ncounts)", "CESw" => "CES\n(weights)",
    "SG" => "Stone-\nGeary", "MinVar" => "min\nvariance", "Tangent" => "tangency", "LS" => "130/30\nCD")

"""
    style!()

Set the Plots defaults for every figure: light surface, recessive solid hairline grid, sans text in
ink tokens, font sizes for a 1600×900 px slide. No units.
"""
function style!()
    gr()
    default(size=SIZE, dpi=100, background_color=SURFACE, background_color_inside=SURFACE,
        foreground_color_grid=GRID, gridalpha=1.0, gridlinewidth=1, gridstyle=:solid,
        foreground_color_axis=AXIS, foreground_color_border=AXIS, foreground_color_text=INK2,
        tickfontcolor=INK2, guidefontcolor=INK2, titlefontcolor=INK, legendfontcolor=INK2,
        fontfamily="sans-serif", titlefontsize=20, guidefontsize=16, tickfontsize=14, legendfontsize=14,
        titlelocation=:left, framestyle=:axes, grid=:y, foreground_color_legend=nothing,
        background_color_legend=nothing, linewidth=2, markerstrokecolor=SURFACE, markerstrokewidth=2)
    return nothing
end

"""
    num(x::Real)::String

A setting value as caption text: whole numbers without a decimal point (1.0 → "1"), others as `string`.
Units are the setting's own.
"""
num(x::Real)::String = isinteger(x) ? string(Int(x)) : string(x)

"""
    beta_cap_text(s::MyStudySettings)::String

The β-cap rule `max(β_min, β_ref − κ_β·ξ)` (dimensionless) written with the values in `s`, e.g.
"max(0, 1 − ξ)" for the defaults.
"""
function beta_cap_text(s::MyStudySettings)::String
    kξ = s.kappa_beta == 1 ? "ξ" : "$(num(s.kappa_beta))·ξ"
    return "max($(num(s.beta_min)), $(num(s.beta_ref)) − $(kξ))"
end

"""
    lambda_note(λ::Real, window::Int)::String

Caption sentence for the calibrated residual-correlation shrinkage λ (dimensionless, grid 0..1) over the
`window`-day (trading days) residual window. It says "on the grid boundary" only when λ is 0 or 1.
"""
function lambda_note(λ::Real, window::Int)::String
    λ == 0 && return "Residual covariance: calibrated shrinkage λ = 0, on the grid boundary " *
        "(the unshrunk $(window)-day residual correlation)."
    λ == 1 && return "Residual covariance: calibrated shrinkage λ = 1, on the grid boundary " *
        "(the $(window)-day residual correlation fully shrunk to the identity)."
    return "Residual covariance: calibrated shrinkage λ = $(num(λ)) of the $(window)-day residual correlation " *
        "toward the identity."
end

"""
    caption_panel(txt::AbstractString)

A frameless strip that carries a note (one or more lines) under a figure, in secondary ink (`INK2`, the
darker of the two text inks below `INK`, for contrast on the slide surface). No units.
"""
caption_panel(txt::AbstractString) = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1),
    annotations=(0.0, 0.5, text(txt, 13, INK2, :left)))

"""
    spacer_panel()

An empty frameless strip, stacked above a figure so its top titles are not flush with the image edge.
No units.
"""
spacer_panel() = plot(framestyle=:none, grid=false, legend=false)

"""
    screen_label(scr, counts)::Union{Nothing,String}

Legend text for screen `scr` from `counts[scr]`, the number of values plotted per allocator: "bandit
screen (30 seeds)", "γ screen (one deterministic run)", or a range such as "28–30 seeds" when the
allocators differ. `nothing` when the screen was not plotted.
"""
function screen_label(scr::String, counts::Dict{String,Vector{Int}})
    c = get(counts, scr, Int[])
    isempty(c) && return nothing
    lo, hi = extrema(c)
    n = lo == hi ? string(lo) : "$(lo)–$(hi)"
    scr == "gamma" && return hi == 1 ? "γ screen (one deterministic run)" : "γ screen ($(n) runs)"
    return "$(scr) screen ($(n) $(hi == 1 ? "seed" : "seeds"))"
end

"""
    legend_strip(counts)

A frameless strip holding the one legend shared by the panels of a seed-distribution figure: one key
per plotted screen, in the screens' left-to-right order within an allocator slot, then the median tick.
Key text comes from `screen_label(scr, counts)`; keys sit at fixed fractions of the width (`LEGEND_X`),
so the legend never runs past the figure edge. No units.
"""
function legend_strip(counts::Dict{String,Vector{Int}})
    p = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1))
    for scr in SCREENS
        lab = screen_label(scr, counts)
        lab === nothing && continue
        x = LEGEND_X[scr]
        scatter!(p, [x], [0.5]; color=SCREEN_COLOR[scr], marker=scr == "gamma" ? :diamond : :circle,
            markersize=scr == "gamma" ? 11 : 8)
        annotate!(p, x + 0.02, 0.5, text(lab, 15, INK2, :left, :vcenter))
    end
    x = LEGEND_X["median"]
    plot!(p, [x - 0.012, x + 0.012], [0.5, 0.5]; color=INK, linewidth=3)
    annotate!(p, x + 0.02, 0.5, text("median across seeds", 15, INK2, :left, :vcenter))
    return p
end

"""
    jitter(n::Int, rng; width=0.08)::Vector{Float64}

`n` horizontal offsets (x-axis units; allocator slots are 1 apart) spread evenly over ±`width` and
shuffled with `rng`, so a 30-seed strip never piles its dots on one x.
"""
jitter(n::Int, rng; width::Float64=0.08)::Vector{Float64} =
    n == 1 ? [0.0] : shuffle(rng, collect(range(-width, width; length=n)))

"""
    strips!(p, d, col)::Dict{String,Vector{Int}}

Draw, per allocator in `ALLOCATORS`, the bandit and random screens' finite values of column `col` of
`d` as dot strips with a median tick and the γ screen as diamonds (values in the column's units).
The result holds, per screen, the number of values drawn for each allocator that has any.
"""
function strips!(p, d::DataFrame, col::Symbol)::Dict{String,Vector{Int}}
    rng = MersenneTwister(5660)
    counts = Dict{String,Vector{Int}}()
    for scr in SCREENS, (i, a) in enumerate(ALLOCATORS)
        v = Float64.(d[(d.allocator .== a) .& (d.screen .== scr), col])
        v = v[isfinite.(v)]
        isempty(v) && continue
        push!(get!(counts, scr, Int[]), length(v))
        x0 = i + SCREEN_OFFSET[scr]
        if scr == "gamma"
            scatter!(p, fill(x0, length(v)), v; marker=:diamond, markersize=11, color=SCREEN_COLOR[scr], label="")
        else
            scatter!(p, x0 .+ jitter(length(v), rng), v; markersize=6, color=SCREEN_COLOR[scr], label="")
            m = median(v)
            plot!(p, [x0 - 0.12, x0 + 0.12], [m, m]; color=INK, linewidth=3, label="")
        end
    end
    return counts
end

"""
    strip_figure(d, metric; scale, ylabel, title, refs, fmt, ylims, caption)

Seed distributions per allocator (see `strips!`) of `metric` times `scale` (e.g. 100 for %), on the
y-range `ylims` (scaled units), with the passive references `refs` (`name => value`, already scaled) as
horizontal lines labeled `fmt(value)` in a gutter right of the last allocator: the higher label above its
line, the lower one below it, so they never collide.
"""
function strip_figure(d::DataFrame, metric::Symbol; scale::Real=1.0, ylabel::String, title::String,
        refs::Vector{<:Pair}, fmt::Function, ylims::Tuple, caption::String)
    d = copy(d); d._y = scale .* Float64.(d[!, metric])
    nA = length(ALLOCATORS)
    p = plot(; xlims=(0.45, nA + 1.6), ylims=ylims, xticks=(1:nA, [TICK_NAME[a] for a in ALLOCATORS]),
        ylabel=ylabel, title=title, legend=false, left_margin=10Plots.mm, bottom_margin=8Plots.mm)
    for (k, (name, val)) in enumerate(sort(refs; by=last, rev=true))
        hline!(p, [val]; color=INK2, linewidth=1.5, label="")
        lab = replace(name, " universe" => "\nuniverse", " buy-and-hold" => "\nbuy-and-hold") * " " * fmt(val)
        annotate!(p, nA + 0.55, val, text(lab, 13, INK2, :left, k == 1 ? :bottom : :top))
    end
    counts = strips!(p, d, :_y)
    return plot(legend_strip(counts), p, caption_panel(caption); layout=grid(3, 1; heights=[0.06, 0.82, 0.12]))
end

"""
    shared_ylims(dfs, metric; scale=1.0, refs=Float64[])::Tuple{Float64,Float64}

One y-range (metric units times `scale`) that holds every finite value of `metric` in every DataFrame
of `dfs` and every reference value in `refs`, padded by 5% of its span on each side, so figures drawn
from different summaries share an axis.
"""
function shared_ylims(dfs::Vector{DataFrame}, metric::Symbol; scale::Real=1.0, refs::Vector{Float64}=Float64[])
    v = vcat([scale .* Float64.(d[!, metric]) for d in dfs]..., refs)
    v = v[isfinite.(v)]
    lo, hi = extrema(v)
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)
end

"""
    hedge_pairs(df)::DataFrame

Hedge-on minus hedge-off differences paired on `(allocator, screen, seed)`: columns `allocator,
screen, seed, d_sharpe` (dimensionless) and `d_mdd` (percentage points of peak wealth, max drawdown on −
off; negative means the hedge lowered the drawdown). Pass one re-pick schedule only.
"""
function hedge_pairs(df::DataFrame)::DataFrame
    pair_cols = [:allocator, :screen, :seed]
    on = select(df[df.hedge .== true, :], pair_cols..., :sharpe => :s_on, :max_drawdown => :m_on)
    off = select(df[df.hedge .== false, :], pair_cols..., :sharpe => :s_off, :max_drawdown => :m_off)
    j = innerjoin(on, off; on=pair_cols)
    return DataFrame(allocator=j.allocator, screen=j.screen, seed=j.seed,
        d_sharpe=Float64.(j.s_on) .- Float64.(j.s_off), d_mdd=100 .* (Float64.(j.m_on) .- Float64.(j.m_off)))
end

"""
    hedge_figure(pairs, caption)

Two panels side by side, by allocator: the paired hedge on − off differences in Sharpe ratio
(dimensionless) and in max drawdown (percentage points), drawn as in `strips!`, each with a zero line.
"""
function hedge_figure(pairs::DataFrame, caption::String)
    nA = length(ALLOCATORS)
    counts = Dict{String,Vector{Int}}()
    panels = map(((:d_sharpe, "Δ Sharpe ratio (on − off)", "Sharpe ratio, hedge on − off (higher is better)"),
            (:d_mdd, "Δ max drawdown (percentage points)", "Max drawdown, hedge on − off (lower is better)"))) do (m, yl, tl)
        p = plot(; xlims=(0.45, nA + 0.55), xticks=(1:nA, [TICK_NAME[a] for a in ALLOCATORS]), ylabel=yl, title=tl,
            titlefontsize=18, legend=false, left_margin=10Plots.mm, bottom_margin=6Plots.mm)
        hline!(p, [0.0]; color=INK2, linewidth=1.5, label="")
        mergewith!(vcat, counts, strips!(p, pairs, m))
        p
    end
    body = plot(panels...; layout=(1, 2))
    return plot(legend_strip(counts), body, caption_panel(caption); layout=grid(3, 1; heights=[0.06, 0.85, 0.09]))
end

"""
    date_ticks(d0, d1, step)

Tick positions (the first of every `step`-th month from January of `d0`'s year, the first one moved to
`d0` if it falls before it) inside `d0..d1` and their "Mon yyyy" labels, for a date axis.
"""
function date_ticks(d0::Date, d1::Date, step::Int)
    t = unique(max.(collect(Date(year(d0), 1, 1):Month(step):d1), d0))
    return (t, Dates.format.(t, "u yyyy"))
end

"""
    example_figure(res, md, s, label)

A 2×2 panel of one run: (a) wealth (thousand USD) vs SPY buy-and-hold scaled to the same starting
wealth, (b) portfolio β (Σ wᵢβᵢ at the close, online β) vs the cap β_max(ξ) (dimensionless), (c) hedge-pool
size (names) and (d) wealth held in pool names (% of wealth), all against the trading date. Caption text
(the cap rule, the pool size and lookback) comes from the settings `s`.
"""
function example_figure(res, md, s::MyStudySettings, label::String)
    dy = res.daily
    i0, i1 = index_of(md, res.dates[1]), index_of(md, res.dates[end])
    spy = md.spy_close[i0:i1]
    spyW = res.equity[1] .* spy ./ spy[1]
    tk = date_ticks(res.dates[1], res.dates[end], 4)
    common = (; xlims=(res.dates[1], res.dates[end]), xticks=tk, left_margin=10Plots.mm, right_margin=10Plots.mm,
        top_margin=5Plots.mm, bottom_margin=4Plots.mm, titlefontsize=17)
    pa = plot(res.dates, res.equity ./ 1e3; color=SLOT[1], label="portfolio", ylabel="wealth (thousand USD)",
        title="(a) Wealth vs SPY buy-and-hold", legend=:bottomright, common...)
    plot!(pa, res.dates, spyW ./ 1e3; color=SLOT[2], label="SPY buy-and-hold")
    pb = plot(dy.date, dy.beta_max; color=INK2, label="β cap = $(beta_cap_text(s))", ylabel="β (dimensionless)",
        title="(b) Portfolio β vs the β cap", legend=:bottomright, common...)
    plot!(pb, dy.date, dy.portfolio_beta; color=SLOT[1], label="portfolio β (online SIM)")
    pc = plot(dy.date, dy.pool_size; seriestype=:steppost, color=SLOT[1], ylabel="names", legend=false,
        title="(c) Hedge-pool size", ylims=(0, maximum(dy.pool_size) + 1), common...)
    pd = plot(dy.date, 100 .* dy.hedge_weight; color=SLOT[1], fillrange=0, fillalpha=0.10, fillcolor=SLOT[1],
        ylabel="% of wealth", legend=false, title="(d) Wealth held in hedge-pool names", common...)
    note = "One run: $(label). The cap is enforced at decisions only, so the daily β can drift above it between them.\n" *
        "Pool: up to $(s.pool_H) names with online β ≤ 0 and down-day β ≤ 0 (down days over the last $(s.downday_lookback) trading days)."
    return plot(spacer_panel(), plot(pa, pb, pc, pd; layout=(2, 2)), caption_panel(note);
        layout=grid(3, 1; heights=[0.03, 0.88, 0.09]))
end

"""
    market_figure(res, md, s, G, period)

Two stacked panels over the run: ξ (dimensionless market-state signal, −G·(EMA_short/EMA_long − 1) of SPY
closes, G dimensionless) and g̃_M, the span-`s.gtilde_window` EMA of SPY's daily growth rate (1/yr), each
with a zero line, plus a caption with the run `period` and G.
"""
function market_figure(res, md, s::MyStudySettings, G::Real, period::String)
    dy = res.daily
    i0, i1 = index_of(md, res.dates[1]), index_of(md, res.dates[end])
    g̃ = gtilde_series(md.spy_close; window=s.gtilde_window)[i0:i1]
    common = (; xlims=(res.dates[1], res.dates[end]), xticks=date_ticks(res.dates[1], res.dates[end], 2),
        legend=false, left_margin=10Plots.mm, right_margin=10Plots.mm, titlefontsize=18)
    p1 = plot(dy.date, dy.xi; color=SLOT[1], ylabel="ξ (dimensionless)", top_margin=5Plots.mm,
        title="ξ, market-state signal (ξ > 0 bearish: short SPY EMA below long; β cap = $(beta_cap_text(s)))", common...)
    hline!(p1, [0.0]; color=INK2, linewidth=1.5)
    p2 = plot(res.dates, g̃; color=SLOT[1], ylabel="growth rate (1/yr)", top_margin=8Plots.mm, bottom_margin=4Plots.mm,
        title="g̃_M, SPY growth rate smoothed by a $(s.gtilde_window)-day EMA", common...)
    hline!(p2, [0.0]; color=INK2, linewidth=1.5)
    note = "Run $(period). ξ = −G·(EMA_short/EMA_long − 1) of SPY closes, EMA spans $(s.xi_short) and $(s.xi_long) " *
        "trading days, gain G = $(@sprintf("%.2f", G)):\n|ξ| ≤ 1 on $(num(round(100 * s.xi_quantile; digits=6)))% of training days " *
        "($(year(s.train_start))–$(year(s.train_stop))). Both series use closes through each day only."
    return plot(spacer_panel(), plot(p1, p2; layout=(2, 1)), caption_panel(note);
        layout=grid(3, 1; heights=[0.03, 0.88, 0.09]))
end

# Size sweep (addendum figs 6–7). Each universe-wide γ screen keeps one categorical slot and its own marker;
# a random control is drawn as a dashed median line over an IQR wash in the slot of the γ screen it is
# matched with (whole universe: global γ's blue; liquid shortlist: liquid γ's orange). The fixed lists are
# ink markers and the passive references gray horizontal lines (SPY solid and thin, equal-weight dotted and
# thicker), each labeled at the right edge of every panel, the higher label above its line and the lower below,
# so the two stay apart when their values are close.
const UNI_SCREENS = ("global_gamma", "liquid_gamma", "corr_gamma")
const UNI_COLOR = Dict("global_gamma" => SLOT[1], "liquid_gamma" => SLOT[2], "corr_gamma" => SLOT[3])
const UNI_MARKER = Dict("global_gamma" => :circle, "liquid_gamma" => :rect, "corr_gamma" => :diamond)
const CONTROLS = ("random_global", "random_liquid")
const CONTROL_COLOR = Dict("random_global" => SLOT[1], "random_liquid" => SLOT[2])
const FIXED_SIZES = (13, 15)
const FIXED_MARKER = Dict(13 => :star5, 15 => :utriangle)
# Passive references: (line style, width in px, direct label).
const REF_STYLE = Dict("SPY buy-and-hold" => (:solid, 1.5, "SPY"), "equal-weight universe" => (:dot, 3.0, "EW"))

"""
    sweep_values(d, a, scr, K, col)::Vector{Float64}

The finite values of column `col` (its units) of `d` for allocator `a`, screen `scr` and basket size `K`
(names).
"""
function sweep_values(d::DataFrame, a::String, scr::String, K::Int, col::Symbol)::Vector{Float64}
    v = Float64.(d[(d.allocator .== a) .& (d.screen .== scr) .& (d.basket_size .== K), col])
    return v[isfinite.(v)]
end

"""
    sweep_key!(p, x, y, kind, color, marker, label)::Float64

One legend key at (`x`, `y`) in a frameless 0..1 strip: `kind` is `:line` (line and marker), `:band`
(IQR wash with a dashed median line), `:marker`, or `:ref` with `marker` a passive-reference name (its
`REF_STYLE` line),
followed by `label` in secondary ink. Returns the x (strip units) where the next key may start, from an
estimate of the label's width (`KEY_CHAR_W` per character at 14 pt). No units.
"""
function sweep_key!(p, x::Real, y::Real, kind::Symbol, color, marker, label::String)::Float64
    w = 0.016
    if kind == :band
        plot!(p, Shape([x - w, x + w, x + w, x - w], [y - 0.17, y - 0.17, y + 0.17, y + 0.17]);
            fillcolor=color, fillalpha=0.18, linewidth=0, label="")
        plot!(p, [x - w, x + w], [y, y]; color=color, linewidth=2.5, linestyle=:dash)
    elseif kind == :line
        plot!(p, [x - w, x + w], [y, y]; color=color, linewidth=2.5)
        scatter!(p, [x], [y]; color=color, marker=marker, markersize=8)
    elseif kind == :marker
        scatter!(p, [x], [y]; color=color, marker=marker, markersize=marker == :star5 ? 15 : 11)
    else
        ls, lw, _ = REF_STYLE[marker]
        plot!(p, [x - w, x + w], [y, y]; color=INK2, linewidth=lw, linestyle=ls)
    end
    annotate!(p, x + w + 0.006, y, text(label, 14, INK2, :left, :vcenter))
    return x + 2w + 0.006 + KEY_CHAR_W * length(label) + 0.035
end

const KEY_CHAR_W = 0.0078

"""
    sweep_legend(refs, fmt)

The legend strip shared by the panels of a size-sweep figure. Top row: the three γ screens and the two
random controls. Bottom row: the two fixed lists and the passive references `refs` (`name => value`,
already scaled, labeled with `fmt(value)`). Keys in a row sit one after another. No units.
"""
function sweep_legend(refs::Vector{<:Pair}, fmt::Function)
    p = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1))
    ref = Dict(refs)
    rows = [(0.75, [(:line, UNI_COLOR["global_gamma"], UNI_MARKER["global_gamma"], "global γ"),
                (:line, UNI_COLOR["liquid_gamma"], UNI_MARKER["liquid_gamma"], "liquid γ"),
                (:line, UNI_COLOR["corr_gamma"], UNI_MARKER["corr_gamma"], "correlation-aware γ"),
                (:band, CONTROL_COLOR["random_global"], nothing, "random, whole universe"),
                (:band, CONTROL_COLOR["random_liquid"], nothing, "random, liquid shortlist")]),
            (0.22, [(:marker, INK, FIXED_MARKER[13], "user's 13-name list"),
                (:marker, INK, FIXED_MARKER[15], "user's 15-name list"),
                (:ref, nothing, "SPY buy-and-hold", "SPY buy-and-hold " * fmt(ref["SPY buy-and-hold"])),
                (:ref, nothing, "equal-weight universe", "equal-weight universe (EW) " * fmt(ref["equal-weight universe"]))])]
    for (y, keys) in rows
        x = 0.02
        for (kind, color, marker, label) in keys
            x = sweep_key!(p, x, y, kind, color, marker, label)
        end
    end
    return p
end

"""
    sweep_figure(d, metric; scale, ylabel, title, refs, fmt, caption)

Small multiples, one panel per allocator (hedge-on rows of the universe summary `d`): `metric` times
`scale` (e.g. 100 for %) against basket size K (names, log axis). Each γ screen is a line through its
one deterministic run per K; each random control is its median across seeds with the interquartile band;
the fixed lists are markers at their sizes; `refs` (`name => value`, scaled) are horizontal lines in their
`REF_STYLE`, each with its short label in a gutter right of K = 34 (the higher label above its line, the lower
below it). All panels share one y-range.
"""
function sweep_figure(d::DataFrame, metric::Symbol; scale::Real=1.0, ylabel::String, title::String,
        refs::Vector{<:Pair}, fmt::Function, caption::String)
    Ks = sort(unique(Int.(d.basket_size[in.(d.screen, Ref(UNI_SCREENS))])))
    d = copy(d); d._y = scale .* Float64.(d[!, metric])
    seeds = Int[]; allv = Float64[last.(refs)...]
    series = Dict{Tuple{String,String},Any}()
    for a in ALLOCATORS
        for scr in UNI_SCREENS
            v = [only(sweep_values(d, a, scr, K, :_y)) for K in Ks]
            series[(a, scr)] = v; append!(allv, v)
        end
        for c in CONTROLS
            vs = [sweep_values(d, a, c, K, :_y) for K in Ks]
            append!(seeds, length.(vs))
            q = [quantile(v, [0.25, 0.5, 0.75]) for v in vs]
            series[(a, c)] = (lo=getindex.(q, 1), med=getindex.(q, 2), hi=getindex.(q, 3))
            append!(allv, getindex.(q, 1), getindex.(q, 3))
        end
        for n in FIXED_SIZES
            v = sweep_values(d, a, "fixed", n, :_y)
            series[(a, "fixed$(n)")] = v; append!(allv, v)
        end
    end
    allequal(seeds) || error("random controls have unequal seed counts: ", sort(unique(seeds)))
    lo, hi = extrema(allv); pad = 0.06 * (hi - lo)
    ylims = (lo - pad, hi + pad)
    panels = map(enumerate(ALLOCATORS)) do (i, a)
        p = plot(; xscale=:log10, xlims=(4.4, 52), xticks=(Ks, string.(Ks)), minorgrid=false, ylims=ylims,
            title=ALLOC_NAME[a], titlefontsize=17, legend=false, ylabel=i in (1, 5) ? ylabel : "",
            xlabel=i >= 5 ? "basket size K (names, log scale)" : i == 4 ? "K (names, log scale)" : "", left_margin=i in (1, 5) ? 12Plots.mm : 3Plots.mm,
            bottom_margin=i >= 4 ? 6Plots.mm : 2Plots.mm, guidefontsize=15, tickfontsize=13)
        for (k, (name, val)) in enumerate(sort(refs; by=last, rev=true))
            ls, lw, short = REF_STYLE[name]
            hline!(p, [val]; color=INK2, linewidth=lw, linestyle=ls)
            annotate!(p, 38, val, text(short, 12, INK2, :left, k == 1 ? :bottom : :top))
        end
        for c in CONTROLS
            s = series[(a, c)]
            plot!(p, Ks, s.med; ribbon=(s.med .- s.lo, s.hi .- s.med), fillcolor=CONTROL_COLOR[c], fillalpha=0.14,
                color=CONTROL_COLOR[c], linestyle=:dash, linewidth=2.5)
        end
        for scr in UNI_SCREENS
            plot!(p, Ks, series[(a, scr)]; color=UNI_COLOR[scr], linewidth=2.5, marker=UNI_MARKER[scr], markersize=7)
        end
        for n in FIXED_SIZES
            v = series[(a, "fixed$(n)")]
            isempty(v) || scatter!(p, fill(n, length(v)), v; color=INK, marker=FIXED_MARKER[n],
                markersize=FIXED_MARKER[n] == :star5 ? 15 : 11)
        end
        p
    end
    head = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1),
        annotations=(0.0, 0.4, text(title, 20, INK, :left)))
    body = plot(panels..., spacer_panel(); layout=(2, 4))   # seven allocators fill 2×4 less one slot
    return plot(head, sweep_legend(refs, fmt), body, caption_panel(caption);
        layout=grid(4, 1; heights=[0.04, 0.08, 0.73, 0.15]))
end

# η-dial figures (CES / CESw): x axes are η on a log scale ticked at the grid values and at 1. The two CES
# versions keep one categorical slot each in every figure; Cobb-Douglas (η = 1) is an ink diamond; the
# headline η = 0.5 is a dotted vertical line.
const VERSIONS = ("CES", "CESw")
const VERSION_COLOR = Dict("CES" => SLOT[1], "CESw" => SLOT[2])
const VERSION_MARKER = Dict("CES" => :circle, "CESw" => :rect)
const DIAL_ETAS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0]
const SWEEP_ETAS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]
const HEADLINE_ETA = 0.5
const LEFT_OUT_NOTE = "CES (share counts) at η = 0.1 is left out: ill-conditioned when a limit binds."
REF_STYLE["headline"] = (:dot, 2.5, "")

"""
    eta_axis(etas; kw...)

Keyword arguments for a log-η x axis: limits a little beyond the grid `etas`, ticks at every grid value
(including 1, the Cobb-Douglas point). η is dimensionless.
"""
eta_axis(etas::Vector{Float64}) = (; xscale=:log10, xlims=(minimum(etas) / 1.25, maximum(etas) * 1.25),
    xticks=(etas, num.(etas)), minorgrid=false)

"""
    eta_legend(; band)

A frameless strip holding the legend of an η figure: the two CES versions (with an interquartile wash when
`band`), the Cobb-Douglas point at η = 1 and the dotted headline η = 0.5. No units.
"""
function eta_legend(; band::Bool=false)
    p = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1))
    x = 0.02
    for v in VERSIONS
        x = sweep_key!(p, x, 0.5, :line, VERSION_COLOR[v], VERSION_MARKER[v], ALLOC_NAME[v])
    end
    band && (x = sweep_key!(p, x, 0.5, :band, INK2, nothing, "IQR band"))
    x = sweep_key!(p, x, 0.5, :marker, INK, :diamond, "CD (η = 1)")
    sweep_key!(p, x, 0.5, :ref, nothing, "headline", "headline η = 0.5")
    return p
end

"""
    day_figure(d, caption)

Two panels (CES share counts, CES weights): each name's weight (fraction of wealth) on the one-day solve
against η (log axis), with the Cobb-Douglas weights as the η = 1 point of both. One colored line per name.
"""
function day_figure(d::DataFrame, caption::String)
    tickers = sort(unique(String.(string.(d.ticker))))
    cols = [HSV(360 * (i - 1) / length(tickers), 0.75, iseven(i) ? 0.62 : 0.88) for i in eachindex(tickers)]
    cd = d[string.(d.allocator) .== "CD", :]
    ymax = 1.03 * maximum(d.weight)
    panels = map(enumerate(VERSIONS)) do (j, v)
        p = plot(; eta_axis(DIAL_ETAS)..., xlims=(0.04, 48.0), ylims=(0, ymax), title=ALLOC_NAME[v], titlefontsize=18, legend=false,
            ylabel=j == 1 ? "weight (fraction of wealth)" : "", xlabel="η (log scale)",
            left_margin=j == 1 ? 12Plots.mm : 3Plots.mm, bottom_margin=6Plots.mm)
        vline!(p, [1.0]; color=INK2, linewidth=1.5, linestyle=:dot, label="")
        dv = d[string.(d.allocator) .== v, :]
        for (i, t) in enumerate(tickers)
            a = sort(vcat(dv[string.(dv.ticker) .== t, [:eta, :weight]], cd[string.(cd.ticker) .== t, [:eta, :weight]]), :eta)
            plot!(p, a.eta, a.weight; color=cols[i], linewidth=2, marker=:circle, markersize=5, label="")
        end
        # Direct labels at η = 20, stacked upward with a minimum gap so none overlap.
        end_w = [only(dv[(string.(dv.ticker) .== t) .& (dv.eta .== maximum(DIAL_ETAS)), :weight]) for t in tickers]
        order = sortperm(end_w); gap = 0.045 * ymax; ly = zeros(length(tickers)); prev = -Inf
        for i in order
            prev = ly[i] = max(end_w[i], prev + gap)
        end
        for i in eachindex(tickers)
            plot!(p, [maximum(DIAL_ETAS), 26.0], [end_w[i], ly[i]]; color=cols[i], linewidth=1.5, label="")
            annotate!(p, 27.5, ly[i], text(tickers[i], 12, INK, :left, :vcenter))
        end
        p
    end
    legend = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1))
    per_row = cld(length(tickers), 2)
    for (i, t) in enumerate(tickers)
        r, c = fldmod1(i, per_row)
        x = 0.02 + (c - 1) * 0.1
        y = r == 1 ? 0.72 : 0.25
        plot!(legend, [x - 0.012, x + 0.012], [y, y]; color=cols[i], linewidth=2.5)
        scatter!(legend, [x], [y]; color=cols[i], markersize=6)
        annotate!(legend, x + 0.02, y, text(t, 14, INK2, :left, :vcenter))
    end
    return plot(legend, plot(panels...; layout=(1, 2)), caption_panel(caption);
        layout=grid(3, 1; heights=[0.1, 0.77, 0.13]))
end

"""
    dial_figure(dial, caption)

Five panels of the clean-limits runs against η (log axis): realized growth rate (1/yr), volatility (1/yr),
max drawdown (% of peak wealth), effective names (count) and invested fraction. The CES and CESw lines
share the Cobb-Douglas point at η = 1; the sixth slot is the legend.
"""
function dial_figure(dial::DataFrame, caption::String)
    dial = copy(dial); dial.allocator = String.(string.(dial.allocator))
    cd = dial[dial.allocator .== "CD", :]
    specs = ((:realized_growth, 1.0, "growth rate (1/yr)", "Realized growth rate", nothing),
        (:vol, 1.0, "volatility (1/yr)", "Volatility", nothing),
        (:max_drawdown, 100.0, "max drawdown (% of peak)", "Max drawdown (lower is better)", nothing),
        (:mean_eff_names, 1.0, "effective names (count)", "Effective number of names", nothing),
        (:mean_invested, 1.0, "invested fraction", "Invested fraction", (0.0, 1.1)))
    panels = map(enumerate(specs)) do (i, (m, sc, yl, ttl, yli))
        p = plot(; eta_axis(DIAL_ETAS)..., title=ttl, titlefontsize=m == :max_drawdown ? 15 : 17, legend=false, ylabel=yl,
            xlabel=i >= 3 ? "η (log scale)" : "", left_margin=14Plots.mm, right_margin=5Plots.mm, bottom_margin=5Plots.mm,
            guidefontsize=13, tickfontsize=12, xtickfontsize=10,
            (yli === nothing ? (;) : (; ylims=yli))...)
        vline!(p, [HEADLINE_ETA]; color=INK2, linewidth=1.5, linestyle=:dot, label="")
        for v in VERSIONS
            a = sort(vcat(dial[dial.allocator .== v, :], cd), :eta)
            plot!(p, a.eta, sc .* a[!, m]; color=VERSION_COLOR[v], linewidth=2.5, marker=VERSION_MARKER[v], markersize=7)
        end
        scatter!(p, cd.eta, sc .* cd[!, m]; color=INK, marker=:diamond, markersize=11)
        p
    end
    key = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1))
    for (y, args) in ((0.8, (:line, VERSION_COLOR["CES"], VERSION_MARKER["CES"], ALLOC_NAME["CES"])),
            (0.62, (:line, VERSION_COLOR["CESw"], VERSION_MARKER["CESw"], ALLOC_NAME["CESw"])),
            (0.44, (:marker, INK, :diamond, "Cobb-Douglas (η = 1)")),
            (0.26, (:ref, nothing, "headline", "headline η = 0.5")))
        sweep_key!(key, 0.08, y, args...)
    end
    return plot(spacer_panel(), plot(panels..., key; layout=(2, 3)), caption_panel(caption);
        layout=grid(3, 1; heights=[0.03, 0.82, 0.15]))
end

"""
    eta_cell(E, a, cap, scr, col)

Rows of the η sweep table `E` for allocator `a`, cap setting `cap` ("on"/"off"), screen `scr`, hedge on,
in η order, with columns `eta`, `med`, `lo`, `hi` (the median and the 25th and 75th percentiles of metric
`col`, in the metric's own units).
"""
function eta_cell(E::DataFrame, a::String, cap::String, scr::String, col::Symbol)
    for suffix in ("_median", "_q25", "_q75")
        string(col, suffix) in names(E) || error("eta_sweep.csv has no column ", col, suffix, " (needed for the η figures)")
    end
    r = E[(E.allocator .== a) .& (E.cap .== cap) .& (E.screen .== scr) .& (E.hedge .== true), :]
    r = sort(r, :eta)
    return DataFrame(eta=Float64.(r.eta), med=Float64.(r[!, Symbol(col, :_median)]),
        lo=Float64.(r[!, Symbol(col, :_q25)]), hi=Float64.(r[!, Symbol(col, :_q75)]))
end

"""
    eta_figure(E, metric; scale, ylabel, title, caption)

2×3 panels of `metric` times `scale` against η (log axis): rows are the per-name cap on and off, columns the
bandit, γ and random screens, hedge on. CES and CESw lines are medians across seeds, with an interquartile
band for the bandit and random screens; CD is the diamond at η = 1 and η = 0.5 the dotted headline. All
panels share one y-range.
"""
function eta_figure(E::DataFrame, metric::Symbol; scale::Real=1.0, ylabel::String, title::String, caption::String)
    scrs = ("bandit", "gamma", "random")
    cells = Dict((a, cap, scr) => eta_cell(E, a, cap, scr, metric) for a in (VERSIONS..., "CD"), cap in ("on", "off"), scr in scrs)
    allv = vcat([scale .* [c.lo; c.hi] for c in values(cells)]...)
    lo, hi = extrema(allv[isfinite.(allv)]); pad = 0.06 * (hi - lo)
    panels = map(Iterators.product(scrs, ("on", "off"))) do (scr, cap)
        row, col = cap == "on" ? 1 : 2, findfirst(==(scr), scrs)
        p = plot(; eta_axis(SWEEP_ETAS)..., ylims=(lo - pad, hi + pad), legend=false, titlefontsize=16,
            title=row == 1 ? (scr == "gamma" ? "γ screen" : "$(scr) screen") * ", cap on" : "cap off",
            ylabel=col == 1 ? ylabel : "", xlabel=row == 2 ? "η (log scale)" : "", guidefontsize=12, tickfontsize=12,
            left_margin=col == 1 ? 12Plots.mm : 3Plots.mm, bottom_margin=row == 2 ? 6Plots.mm : 2Plots.mm)
        vline!(p, [HEADLINE_ETA]; color=INK2, linewidth=1.5, linestyle=:dot, label="")
        for v in VERSIONS
            c = cells[(v, cap, scr)]
            isempty(c) && continue
            rib = scr == "gamma" ? (;) : (; ribbon=(scale .* (c.med .- c.lo), scale .* (c.hi .- c.med)))
            plot!(p, c.eta, scale .* c.med; color=VERSION_COLOR[v], fillcolor=VERSION_COLOR[v], fillalpha=0.16,
                linewidth=2.5, marker=VERSION_MARKER[v], markersize=6, rib...)
        end
        c = cells[("CD", cap, scr)]
        isempty(c) || scatter!(p, c.eta, scale .* c.med; color=INK, marker=:diamond, markersize=10,
            markerstrokecolor=INK, markerstrokewidth=2,
            (scr == "gamma" ? (;) : (; yerror=(scale .* (c.med .- c.lo), scale .* (c.hi .- c.med))))...)
        (scr, cap) => p
    end
    ps = [last(panels[findfirst(x -> first(x) == (scr, cap), panels)]) for cap in ("on", "off") for scr in scrs]
    head = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1),
        annotations=(0.0, 0.4, text(title, 20, INK, :left)))
    return plot(head, eta_legend(band=true), plot(ps...; layout=(2, 3)), caption_panel(caption);
        layout=grid(4, 1; heights=[0.04, 0.07, 0.73, 0.16]))
end

"""
    frontier_figure(E, caption)

Two panels (per-name cap on, off): median growth rate (1/yr) against median max drawdown (% of peak wealth)
across the random screen's seeds, hedge on. Each CES version's points are joined in η order and labeled
with η; the Cobb-Douglas point is the diamond.
"""
function frontier_figure(E::DataFrame, caption::String)
    cells = Dict((a, cap) => (g=eta_cell(E, a, cap, "random", :realized_growth), d=eta_cell(E, a, cap, "random", :max_drawdown))
        for a in (VERSIONS..., "CD"), cap in ("on", "off"))
    ys = vcat([c.g.med for c in values(cells)]...)
    yl = extrema(ys); py = 0.1 * (yl[2] - yl[1])
    panels = map(enumerate(("on", "off"))) do (j, cap)
        xc = vcat([100 .* cells[(a, cap)].d.med for a in (VERSIONS..., "CD")]...)
        xl = extrema(xc); px = 0.1 * (xl[2] - xl[1])
        p = plot(; xlims=(xl[1] - px, xl[2] + px), ylims=(yl[1] - py, yl[2] + py), legend=false, grid=:xy,
            title="per-name cap $(cap)", titlefontsize=18, xlabel="median max drawdown (% of peak wealth)",
            ylabel=j == 1 ? "median realized growth rate (1/yr)" : "", left_margin=j == 1 ? 12Plots.mm : 3Plots.mm,
            bottom_margin=6Plots.mm)
        dx, dy = 0.03 * (xl[2] - xl[1] + 2px), 0.05 * (yl[2] - yl[1] + 2py)
        for (v, sx, sy) in (("CES", 1, -1), ("CESw", -1, 1))
            c = cells[(v, cap)]
            isempty(c.g) && continue
            x, y = 100 .* c.d.med, c.g.med
            plot!(p, x, y; color=VERSION_COLOR[v], linewidth=2.5, marker=VERSION_MARKER[v], markersize=8)
            for k in eachindex(x)
                annotate!(p, x[k] + sx * dx, y[k] + sy * dy, text(num(c.g.eta[k]), 12, INK2, :center, :center))
            end
            h = findfirst(==(HEADLINE_ETA), c.g.eta)
            if h !== nothing   # ring: an ink circle under the point redrawn in its own color
                scatter!(p, [x[h]], [y[h]]; color=INK, marker=:circle, markersize=17)
                scatter!(p, [x[h]], [y[h]]; color=VERSION_COLOR[v], marker=VERSION_MARKER[v], markersize=8)
            end
        end
        c = cells[("CD", cap)]
        isempty(c.g) || scatter!(p, 100 .* c.d.med, c.g.med; color=INK, marker=:diamond, markersize=12)
        p
    end
    head = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1),
        annotations=(0.0, 0.4, text("Growth rate against max drawdown across η, random screen, hedge on", 20, INK, :left)))
    key = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1))
    x = 0.02
    for v in VERSIONS
        x = sweep_key!(key, x, 0.5, :line, VERSION_COLOR[v], VERSION_MARKER[v], ALLOC_NAME[v])
    end
    sweep_key!(key, x, 0.5, :marker, INK, :diamond, "CD (η = 1)")
    return plot(head, key, plot(panels...; layout=(1, 2)), caption_panel(caption);
        layout=grid(4, 1; heights=[0.04, 0.07, 0.73, 0.16]))
end

"""
    load_table(file)::Union{Nothing,DataFrame}

Read `results/<file>` with `allocator` as String; `nothing` if the file is missing.
"""
function load_table(file::AbstractString)
    path = joinpath(RESULTS, file)
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame)
    df.allocator = String.(string.(df.allocator))
    return df
end

"""
    load_ok(file)::Union{Nothing,DataFrame}

Read `results/<file>` and keep the `status == "ok"` rows, with `screen` and `allocator` as Strings;
`nothing` if the file is missing.
"""
function load_ok(file::AbstractString)
    path = joinpath(RESULTS, file)
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame)
    df = df[string.(df.status) .== "ok", :]
    df.screen = String.(string.(df.screen)); df.allocator = String.(string.(df.allocator))
    return df
end

function main()
    style!()
    mkpath(FIG_DIR)
    s, _ = load_study(CONFIG)
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    passive = passive_references(md, s.start, s.stop; W0=s.W0)
    pref(m; scale=1.0) = [r.name => scale * getproperty(r, m) for r in eachrow(passive)]
    λ = TOML.parsefile(CALIB_TOML)["lambda"]
    tr = index_of(md, s.train_start):searchsortedlast(md.dates, s.train_stop)
    G = xi_gain(md.spy_close[tr]; short=s.xi_short, long=s.xi_long, q=s.xi_quantile)   # as in build_context
    period = "$(Dates.format(s.start, "yyyy-mm-dd")) to $(Dates.format(s.stop, "yyyy-mm-dd"))"
    covnote = lambda_note(λ, s.omega_window)
    tier1 = load_ok("tier1_summary.csv")
    tier1 === nothing && error("no tier1_summary.csv in ", RESULTS)
    fixed = load_ok("fixed_summary.csv")
    written = String[]
    save_fig!(fig, name) = (f = joinpath(FIG_DIR, name); savefig(fig, f); push!(written, f))
    sets = [(df[df.hedge .== true, :], tag, sched) for (df, tag, sched) in
        ((tier1, "", "re-pick every $(s.repick_interval) days or at a band exit"),
         (fixed, "_fixed", "fixed $(s.repick_interval)-day re-pick schedule, no band-exit re-picks")) if df !== nothing]
    ons = DataFrame[first(x) for x in sets]
    ylim_sharpe = shared_ylims(ons, :sharpe; refs=last.(pref(:sharpe)))
    ylim_mdd = shared_ylims(ons, :max_drawdown; scale=100, refs=last.(pref(:max_drawdown; scale=100)))
    for (on, tag, sched) in sets
        cap = "Run $(period), hedge on (β cap and hedge pool), $(sched).\n" *
            "Each dot is one seed; the horizontal lines are the passive references.\n$(covnote)"
        save_fig!(strip_figure(on, :sharpe; ylabel="Sharpe ratio (from daily growth rates)", ylims=ylim_sharpe,
            title="Sharpe ratio by allocator and screen", refs=pref(:sharpe), fmt=v -> @sprintf("%.2f", v),
            caption=cap), "fig1_sharpe$(tag).png")
        save_fig!(strip_figure(on, :max_drawdown; scale=100, ylabel="max drawdown (% of peak wealth)", ylims=ylim_mdd,
            title="Max drawdown by allocator and screen (lower is better)", refs=pref(:max_drawdown; scale=100),
            fmt=v -> @sprintf("%.1f%%", v), caption=cap), "fig2_max_drawdown$(tag).png")
    end
    t1 = tier1[tier1.repick_on_band_exit .== true, :]
    hcap = "Run $(period). Each dot is one seed: the hedge-on run minus its hedge-off twin (same allocator, screen, " *
        "seed and settings,\nbut no β cap and no hedge pool). Band exits trigger re-picks, so the twins' re-pick " *
        "dates, and hence their baskets, can differ."
    save_fig!(hedge_figure(hedge_pairs(t1), hcap), "fig3_hedge_effect.png")
    ex = t1[(t1.allocator .== "CD") .& (t1.screen .== "gamma") .& (t1.hedge .== true), :]
    nrow(ex) == 1 || error("expected one CD / γ / hedge-on run, found ", nrow(ex))
    res = load(joinpath(RESULTS, "runs", ex.hash[1] * ".jld2"))["result"]
    save_fig!(example_figure(res, md, s, "Cobb-Douglas, γ screen, hedge on"), "fig4_example_run.png")
    save_fig!(market_figure(res, md, s, G, period), "fig5_market_state.png")
    uni = load_ok("universe_summary.csv")
    if uni !== nothing
        on = uni[(uni.hedge .== true) .& (uni.repick_on_band_exit .== true), :]
        nseed = nrow(on[(on.screen .== "random_global") .& (on.allocator .== "CD") .& (on.basket_size .== s.basket_K), :])
        ucap = "Run $(period), hedge on (β cap and hedge pool), re-pick every $(s.repick_interval) days or at a band exit. " *
            "A screen pick's cap is max($(num(100 * s.w_max))%, $(num(s.w_max_scale))/K).\n" *
            "Solid lines: one deterministic run per screen and K. Dashed lines and bands: random-control median and " *
            "interquartile range over $(nseed) seeds.\n" *
            "Liquid γ and its control use the top $(s.liquidity_shortlist) names by $(s.liquidity_window)-day median dollar " *
            "volume. Correlation-aware γ: pairwise growth-rate correlation ≤ $(num(s.corr_max)).\n" *
            "The universe-wide screens are exploratory: they, their thresholds and the K grid were designed after the " *
            "Tier 1 results were known.\nThe user's 13- and 15-name lists were chosen in advance, without 2025 data."
        save_fig!(sweep_figure(on, :sharpe; ylabel="Sharpe ratio",
            title="Sharpe ratio (from daily growth rates) vs basket size K, hedge on", refs=pref(:sharpe),
            fmt=v -> @sprintf("%.2f", v),
            caption=ucap), "fig6_sharpe_vs_K.png")
        save_fig!(sweep_figure(on, :max_drawdown; scale=100, ylabel="max drawdown (%)",
            title="Max drawdown (% of peak wealth) vs basket size K, hedge on (lower is better)",
            refs=pref(:max_drawdown; scale=100),
            fmt=v -> @sprintf("%.1f%%", v), caption=ucap), "fig7_max_drawdown_vs_K.png")
    end
    lday, ldial, esweep = load_table("limits_day.csv"), load_table("tables/limits_dial.csv"), load_table("tables/eta_sweep.csv")
    if lday !== nothing
        day = join(string.(unique(lday.date)), ", ")
        save_fig!(day_figure(lday, "One-day solve on $(day), budget only, L8b 13-name list, chosen in advance. Each line is one name's weight (fraction of wealth).\n" *
            "Cobb-Douglas is η = 1 (dotted line). η is dimensionless, on a log axis ticked at the grid values and at 1."),
            "fig_limits_day.png")
    end
    if ldial !== nothing
        save_fig!(dial_figure(ldial, "Run $(period), budget only, L8b 13-name list, chosen in advance, hedge off. Growth rates and volatility in 1/yr.\n" *
            "CES (share counts) and CES (weights) lines share the Cobb-Douglas point (diamond) at η = 1; the dotted line marks the headline η = 0.5."),
            "fig_limits_dial.png")
    end
    if esweep !== nothing
        ecap = "Run $(period), hedge on. Lines: median across seeds; bandit and random: IQR band; γ: one run.\n" *
            "CD = Cobb-Douglas, the diamond at η = 1 (its bar is the IQR); the dotted line is the headline η = 0.5.\n" *
            "Re-picks follow each run's own band exits, so the two versions can hold different baskets.\n" * LEFT_OUT_NOTE
        METRIC_LABEL = (realized_growth=(1.0, "growth rate (1/yr)", "Realized growth rate"),
            sharpe=(1.0, "Sharpe ratio", "Sharpe ratio (from daily growth rates)"),
            max_drawdown=(100.0, "max drawdown (% of peak)", "Max drawdown (lower is better)"),
            vol=(1.0, "volatility (1/yr)", "Volatility"),
            mean_eff_names=(1.0, "effective names (count)", "Effective number of names"),
            mean_invested=(1.0, "invested fraction", "Invested fraction"),
            turnover=(1.0, "turnover (1/yr)", "Turnover"))
        for m in ETA_METRICS
            haskey(METRIC_LABEL, m) || error("no figure label for η-sweep metric ", m, "; add it to METRIC_LABEL")
            sc, yl, ttl = METRIC_LABEL[m]
            save_fig!(eta_figure(esweep, m; scale=sc, ylabel=yl, title="$(ttl) against η, CES (share counts) and CES (weights), hedge on",
                caption=ecap), "fig_eta_$(m).png")
        end
        save_fig!(frontier_figure(esweep, "Run $(period), hedge on, random screen: median across seeds; up and to the left is better.\nPoints are joined in η order and labeled with η; ringed points are the headline η = 0.5.\n" *
            "Re-picks follow each run's own band exits, so the two versions can hold different baskets.\n" * LEFT_OUT_NOTE),
            "fig_eta_frontier.png")
    end
    foreach(println, written)
    return nothing
end

main()
