# Task 14: slide figures (PNG, 1600×900) from the summary CSVs and the run files.
# Usage: julia scripts/13_figures.jl [dir]   (default dir = results/; reads <dir>/*_summary.csv and
# <dir>/runs/, writes <dir>/figures/*.png). Every performance number is a growth-rate metric:
# g = ln(S_t/S_{t-1})/Δt in 1/yr with Δt = 1/252.
using Pkg
Pkg.activate(@__DIR__; io=devnull)
using VLINFORMS2026, Plots, DataFrames, CSV, JLD2, Dates, Statistics, Random

const REPO = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(REPO, "data")
const CONFIG = joinpath(REPO, "config", "study.toml")
const CALIB_TOML = joinpath(REPO, "results", "calibration_summary.toml")
const MD_START, MD_STOP = Date(2014, 1, 3), Date(2026, 9, 4)
const SIZE = (1600, 900)

# Chart chrome and categorical slots: the dataviz reference palette on its light surface. The first three
# slots pass the all-pairs CVD and normal-vision gates (validate_palette.js --pairs all); slot 3 (aqua) is
# below 3:1 contrast, so the γ screen also has its own marker shape and every value is in results/tables/.
# The screen → slot map is fixed, so a screen has the same color in every figure.
const SURFACE = "#fcfcfb"
const INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
const GRID, AXIS = "#e1e0d9", "#c3c2b7"
const SLOT = ("#2a78d6", "#eb6834", "#1baf7a")
const SCREENS = ("bandit", "random", "gamma")
const SCREEN_COLOR = Dict("bandit" => SLOT[1], "random" => SLOT[2], "gamma" => SLOT[3])
const SCREEN_OFFSET = Dict("bandit" => -0.24, "gamma" => 0.0, "random" => 0.24)
const ALLOCATORS = ["CD", "CES", "SG", "MinVar", "Tangent", "LS"]
const ALLOC_NAME = Dict("CD" => "Cobb-Douglas", "CES" => "CES", "SG" => "Stone-Geary",
    "MinVar" => "min variance", "Tangent" => "tangency", "LS" => "130/30 CD")

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
    caption_panel(txt::AbstractString)

A frameless strip that carries a note (one or more lines) under a figure, in muted ink. No units.
"""
caption_panel(txt::AbstractString) = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1),
    annotations=(0.0, 0.5, text(txt, 13, MUTED, :left)))

"""
    legend_strip(n_seeds::Int)

A frameless strip holding the one legend shared by the panels of a seed-distribution figure: the bandit
and random screens (`n_seeds` dots each), the γ screen (one diamond) and the median tick. Keys are placed
by hand at fixed fractions of the width, so the legend never runs past the figure edge. No units.
"""
function legend_strip(n_seeds::Int)
    p = plot(framestyle=:none, grid=false, legend=false, xlims=(0, 1), ylims=(0, 1))
    keys = (("bandit screen ($(n_seeds) seeds)", :dot, SLOT[1]), ("random screen ($(n_seeds) seeds)", :dot, SLOT[2]),
        ("γ screen (one deterministic run)", :diamond, SLOT[3]), ("median across seeds", :line, INK))
    for ((lab, kind, col), x) in zip(keys, (0.03, 0.255, 0.495, 0.79))
        if kind == :line
            plot!(p, [x - 0.012, x + 0.012], [0.5, 0.5]; color=col, linewidth=3)
        else
            scatter!(p, [x], [0.5]; color=col, marker=kind == :diamond ? :diamond : :circle,
                markersize=kind == :diamond ? 11 : 8)
        end
        annotate!(p, x + 0.02, 0.5, text(lab, 15, INK2, :left, :vcenter))
    end
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
    strips!(p, d, col)

Draw, per allocator in `ALLOCATORS`, the bandit and random screens' values of column `col` of `d`
as dot strips with a median tick and the γ screen as one diamond (values in the column's units).
"""
function strips!(p, d::DataFrame, col::Symbol)
    rng = MersenneTwister(5660)
    for scr in SCREENS, (i, a) in enumerate(ALLOCATORS)
        v = Float64.(d[(d.allocator .== a) .& (d.screen .== scr), col])
        v = v[isfinite.(v)]
        isempty(v) && continue
        x0 = i + SCREEN_OFFSET[scr]
        if scr == "gamma"
            scatter!(p, fill(x0, length(v)), v; marker=:diamond, markersize=11, color=SCREEN_COLOR[scr], label="")
        else
            scatter!(p, x0 .+ jitter(length(v), rng), v; markersize=6, color=SCREEN_COLOR[scr], label="")
            m = median(v)
            plot!(p, [x0 - 0.12, x0 + 0.12], [m, m]; color=INK, linewidth=3, label="")
        end
    end
    return p
end

"""
    strip_figure(d, metric; scale, ylabel, title, refs, caption, n_seeds)

Seed distributions per allocator (see `strips!`) of `metric` times `scale` (e.g. 100 for %), with the
passive references `refs` (`name => value`, already scaled) as horizontal lines labeled in a gutter right
of the last allocator: the higher label above its line, the lower one below it, so they never collide.
"""
function strip_figure(d::DataFrame, metric::Symbol; scale::Real=1.0, ylabel::String, title::String,
        refs::Vector{<:Pair}, caption::String, n_seeds::Int)
    d = copy(d); d._y = scale .* Float64.(d[!, metric])
    nA = length(ALLOCATORS)
    p = plot(; xlims=(0.45, nA + 1.5), xticks=(1:nA, [ALLOC_NAME[a] for a in ALLOCATORS]),
        ylabel=ylabel, title=title, legend=false, left_margin=10Plots.mm, bottom_margin=4Plots.mm)
    for (k, (name, val)) in enumerate(sort(refs; by=last, rev=true))
        hline!(p, [val]; color=INK2, linewidth=1.5, label="")
        lab = replace(name, " universe" => "\nuniverse", " buy-and-hold" => "\nbuy-and-hold") * " $(round(val; digits=2))"
        annotate!(p, nA + 0.6, val, text(lab, 13, INK2, :left, k == 1 ? :bottom : :top))
    end
    strips!(p, d, :_y)
    return plot(legend_strip(n_seeds), p, caption_panel(caption); layout=grid(3, 1; heights=[0.06, 0.82, 0.12]))
end

"""
    hedge_pairs(df)::DataFrame

Hedge-on minus hedge-off differences paired on `(allocator, screen, seed)`: columns `allocator,
screen, seed, d_sharpe` (dimensionless) and `d_mdd` (percentage points of peak wealth, max drawdown on −
off; negative means the hedge lowered the drawdown). Pass one re-pick schedule only.
"""
function hedge_pairs(df::DataFrame)::DataFrame
    keys = [:allocator, :screen, :seed]
    on = select(df[df.hedge .== true, :], keys..., :sharpe => :s_on, :max_drawdown => :m_on)
    off = select(df[df.hedge .== false, :], keys..., :sharpe => :s_off, :max_drawdown => :m_off)
    j = innerjoin(on, off; on=keys)
    return DataFrame(allocator=j.allocator, screen=j.screen, seed=j.seed,
        d_sharpe=Float64.(j.s_on) .- Float64.(j.s_off), d_mdd=100 .* (Float64.(j.m_on) .- Float64.(j.m_off)))
end

"""
    hedge_figure(pairs, caption, n_seeds)

Two panels side by side, by allocator: the paired hedge on − off differences in Sharpe ratio
(dimensionless) and in max drawdown (percentage points), drawn as in `strips!`, each with a zero line.
"""
function hedge_figure(pairs::DataFrame, caption::String, n_seeds::Int)
    nA = length(ALLOCATORS)
    short = Dict("CD" => "Cobb-\nDouglas", "CES" => "CES", "SG" => "Stone-\nGeary", "MinVar" => "min\nvariance",
        "Tangent" => "tangency", "LS" => "130/30\nCD")
    panels = map(((:d_sharpe, "Δ Sharpe ratio (on − off)", "Sharpe ratio, hedge on − off (higher is better)"),
            (:d_mdd, "Δ max drawdown (percentage points)", "Max drawdown, hedge on − off (lower is better)"))) do (m, yl, tl)
        p = plot(; xlims=(0.45, nA + 0.55), xticks=(1:nA, [short[a] for a in ALLOCATORS]), ylabel=yl, title=tl,
            titlefontsize=18, legend=false, left_margin=10Plots.mm, bottom_margin=6Plots.mm)
        hline!(p, [0.0]; color=INK2, linewidth=1.5, label="")
        strips!(p, pairs, m)
    end
    body = plot(panels...; layout=(1, 2))
    return plot(legend_strip(n_seeds), body, caption_panel(caption); layout=grid(3, 1; heights=[0.06, 0.85, 0.09]))
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
    example_figure(res, md, label)

A 2×2 panel of one run: (a) wealth (thousand USD) vs SPY buy-and-hold scaled to the same starting
wealth, (b) portfolio β (Σ wᵢβᵢ at the close, online β) vs the cap β_max(ξ) (dimensionless), (c) hedge-pool
size (names) and (d) wealth held in pool names (% of wealth), all against the trading date.
"""
function example_figure(res, md, label::String)
    dy = res.daily
    i0, i1 = index_of(md, res.dates[1]), index_of(md, res.dates[end])
    spy = md.spy_close[i0:i1]
    spyW = res.equity[1] .* spy ./ spy[1]
    tk = date_ticks(res.dates[1], res.dates[end], 4)
    common = (; xlims=(res.dates[1], res.dates[end]), xticks=tk, left_margin=10Plots.mm, right_margin=10Plots.mm,
        bottom_margin=4Plots.mm, titlefontsize=17)
    pa = plot(res.dates, res.equity ./ 1e3; color=SLOT[1], label="portfolio", ylabel="wealth (thousand USD)",
        title="(a) Wealth vs SPY buy-and-hold", legend=:bottomright, common...)
    plot!(pa, res.dates, spyW ./ 1e3; color=SLOT[2], label="SPY buy-and-hold")
    pb = plot(dy.date, dy.beta_max; color=INK2, label="β cap = max(0, 1 − ξ)", ylabel="β (dimensionless)",
        title="(b) Portfolio β vs the β cap", legend=:bottomright, common...)
    plot!(pb, dy.date, dy.portfolio_beta; color=SLOT[1], label="portfolio β (online SIM)")
    pc = plot(dy.date, dy.pool_size; seriestype=:steppost, color=SLOT[1], ylabel="names", legend=false,
        title="(c) Hedge-pool size", ylims=(0, maximum(dy.pool_size) + 1), common...)
    pd = plot(dy.date, 100 .* dy.hedge_weight; color=SLOT[1], fillrange=0, fillalpha=0.10, fillcolor=SLOT[1],
        ylabel="% of wealth", legend=false, title="(d) Wealth held in hedge-pool names", common...)
    note = "One run: $(label). The cap is enforced at decisions only, so the daily β can drift above it between them.\n" *
        "Pool: up to 5 names with online β ≤ 0 and down-day β ≤ 0."
    return plot(plot(pa, pb, pc, pd; layout=(2, 2)), caption_panel(note); layout=grid(2, 1; heights=[0.91, 0.09]))
end

"""
    market_figure(res, md, gtilde_window)

Two stacked panels over the run: ξ (dimensionless market-state signal, −G·(EMA₂₁/EMA₆₃ − 1) of SPY) and
g̃_M, the span-`gtilde_window` EMA of SPY's daily growth rate (1/yr), each with a zero line.
"""
function market_figure(res, md, gtilde_window::Int)
    dy = res.daily
    i0, i1 = index_of(md, res.dates[1]), index_of(md, res.dates[end])
    g̃ = gtilde_series(md.spy_close; window=gtilde_window)[i0:i1]
    common = (; xlims=(res.dates[1], res.dates[end]), xticks=date_ticks(res.dates[1], res.dates[end], 2),
        legend=false, left_margin=10Plots.mm, right_margin=10Plots.mm, titlefontsize=18)
    p1 = plot(dy.date, dy.xi; color=SLOT[1], ylabel="ξ (dimensionless)",
        title="ξ, market-state signal (ξ > 0 bearish: short SPY average below long; β cap = max(0, 1 − ξ))", common...)
    hline!(p1, [0.0]; color=INK2, linewidth=1.5)
    p2 = plot(res.dates, g̃; color=SLOT[1], ylabel="growth rate (1/yr)", bottom_margin=6Plots.mm,
        title="g̃_M, SPY growth rate smoothed by a $(gtilde_window)-day EMA", common...)
    hline!(p2, [0.0]; color=INK2, linewidth=1.5)
    return plot(p1, p2; layout=(2, 1))
end

"""
    load_ok(dir, file)::Union{Nothing,DataFrame}

Read `<dir>/<file>` and keep the `status == "ok"` rows, with `screen` and `allocator` as Strings;
`nothing` if the file is missing.
"""
function load_ok(dir::AbstractString, file::AbstractString)
    path = joinpath(dir, file)
    isfile(path) || return nothing
    df = CSV.read(path, DataFrame)
    df = df[string.(df.status) .== "ok", :]
    df.screen = String.(string.(df.screen)); df.allocator = String.(string.(df.allocator))
    return df
end

function main(args=ARGS)
    style!()
    pos = filter(a -> !startswith(a, "--"), args)
    dir = isempty(pos) ? joinpath(REPO, "results") : abspath(pos[1])
    figdir = joinpath(dir, "figures"); mkpath(figdir)
    s, grid = load_study(CONFIG)
    md = load_market_data(DATA_DIR; start=MD_START, stop=MD_STOP)
    passive = passive_references(md, s.start, s.stop; W0=s.W0)
    pref(m; scale=1.0) = [r.name => scale * getproperty(r, m) for r in eachrow(passive)]
    lam = isfile(CALIB_TOML) ? match(r"lambda\s*=\s*([0-9.eE+-]+)", read(CALIB_TOML, String)).captures[1] : "?"
    period = "$(Dates.format(s.start, "yyyy-mm-dd")) to $(Dates.format(s.stop, "yyyy-mm-dd"))"
    covnote = "Residual covariance: calibrated shrinkage λ = $(lam), the grid boundary (the unshrunk " *
        "$(s.omega_window)-day residual correlation)."
    tier1 = load_ok(dir, "tier1_summary.csv")
    tier1 === nothing && error("no tier1_summary.csv in ", dir)
    written = String[]
    save(fig, name) = (f = joinpath(figdir, name); savefig(fig, f); push!(written, f))
    for (df, tag, sched) in ((tier1, "", "re-pick every $(s.repick_interval) days or at a band exit"),
            (load_ok(dir, "fixed_summary.csv"), "_fixed", "fixed $(s.repick_interval)-day re-pick schedule, no band-exit re-picks"))
        df === nothing && continue
        on = df[df.hedge .== true, :]
        cap = "Run $(period), hedge on (β cap and hedge pool), $(sched).\n" *
            "Each dot is one seed; the horizontal lines are the passive references.\n$(covnote)"
        save(strip_figure(on, :sharpe; ylabel="Sharpe ratio (from daily growth rates)", n_seeds=grid.n_seeds,
            title="Sharpe ratio by allocator and screen", refs=pref(:sharpe), caption=cap), "fig1_sharpe$(tag).png")
        save(strip_figure(on, :max_drawdown; scale=100, ylabel="max drawdown (% of peak wealth)", n_seeds=grid.n_seeds,
            title="Max drawdown by allocator and screen (lower is better)", refs=pref(:max_drawdown; scale=100),
            caption=cap), "fig2_max_drawdown$(tag).png")
    end
    t1 = tier1[tier1.repick_on_band_exit .== true, :]
    hcap = "Run $(period). Each dot is one seed: the hedge-on run minus its hedge-off twin (same allocator, " *
        "screen and seed).\nHedge off removes the β cap and the hedge pool and is otherwise identical."
    save(hedge_figure(hedge_pairs(t1), hcap, grid.n_seeds), "fig3_hedge_effect.png")
    ex = t1[(t1.allocator .== "CD") .& (t1.screen .== "gamma") .& (t1.hedge .== true), :]
    nrow(ex) == 1 || error("expected one CD / γ / hedge-on run, found ", nrow(ex))
    res = load(joinpath(dir, "runs", ex.hash[1] * ".jld2"))["result"]
    save(example_figure(res, md, "Cobb-Douglas, γ screen, hedge on"), "fig4_example_run.png")
    save(market_figure(res, md, s.gtilde_window), "fig5_market_state.png")
    foreach(println, written)
    return nothing
end

main()
