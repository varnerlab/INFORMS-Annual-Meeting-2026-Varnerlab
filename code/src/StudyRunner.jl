# Resumable, threaded runner for the screened-utility panel study: one JLD2 per run (keyed by a hash of
# the run's settings, allocator, screen and seed), one summary CSV. Shares `_run_items` with `run_grid`.

"""
    study_key(run::MyStudyRun)::NamedTuple

Identity of a study run: every `MyStudySettings` field of `run.settings` (units as in `MyStudySettings`)
followed by `allocator`, `screen` and `seed`. `label` is excluded.
"""
function study_key(run::MyStudyRun)::NamedTuple
    fs = fieldnames(MyStudySettings)
    return NamedTuple{(fs..., :allocator, :screen, :seed)}(
        (Tuple(getfield(run.settings, f) for f in fs)..., run.allocator, run.screen, run.seed))
end

"""
    study_hash(run::MyStudyRun)::String

`bytes2hex(sha1(repr(study_key(run))))`, the 40-character file stem of the run's results.
"""
study_hash(run::MyStudyRun)::String = bytes2hex(sha1(repr(study_key(run))))

"""
    study_stats(res::MyStudyResult)::NamedTuple

Per-run activity statistics. `turnover` (yr⁻¹) = Σ|qty·price| over trades with qty ≠ 0 divided by the
mean equity (USD) and by the run length in years, n_days/252 with n_days = length(dates) − 1;
`total_cost` and `borrow_fees` in USD. `n_decisions` counts decisions and `n_band_exit`, `n_drawdown`,
`n_horizon`, `n_cash_revisit`, `n_basket_change` count them by `reason` (`:band_exit`, `:drawdown`,
`:horizon_elapsed`, `:cash_revisit`, `:basket_change`). `n_repicks` counts re-picks and `n_basket_changes`
those that changed the basket, both excluding the initial re-pick. `mean_names` (names) and
`mean_invested` (fraction of wealth) average over decisions. `mean_portfolio_beta` (dimensionless),
`mean_hedge_weight` (fraction of wealth) and `mean_pool_size` (names) average over days.
`cap_bind_share` is the share of decisions with `cap_binds` (dimensionless), `NaN` when the β cap is off
(no decision has a finite cap). Reading "share of days the cap binds" as decision days, the only days the
cap is enforced, is a **(plan decision)**.
"""
function study_stats(res::MyStudyResult)::NamedTuple
    tr = res.trades; live = tr.qty .!= 0
    traded = sum(abs.(tr.qty[live] .* tr.price[live]); init=0.0)
    years = (length(res.dates) - 1) / 252
    dec = res.decisions; rp = res.repicks; dy = res.daily
    cnt(r) = count(==(r), dec.reason)
    mn(x) = isempty(x) ? NaN : mean(x)
    capped = any(isfinite, dec.beta_max)
    return (turnover=years > 0 ? traded / mean(res.equity) / years : NaN,
        total_cost=sum(tr.cost; init=0.0), borrow_fees=sum(res.borrow_fees),
        n_decisions=nrow(dec), n_band_exit=cnt(:band_exit), n_drawdown=cnt(:drawdown),
        n_horizon=cnt(:horizon_elapsed), n_cash_revisit=cnt(:cash_revisit), n_basket_change=cnt(:basket_change),
        n_repicks=max(nrow(rp) - 1, 0), n_basket_changes=count(rp.changed[2:end]),
        mean_names=mn(dec.n_names), mean_invested=mn(dec.invested_frac),
        mean_portfolio_beta=mn(dy.portfolio_beta),
        cap_bind_share=capped ? mean(dec.cap_binds) : NaN,
        mean_hedge_weight=mn(dy.hedge_weight), mean_pool_size=mn(dy.pool_size))
end

"""
    run_and_save_study(ctx::MyStudyContext, run::MyStudyRun; outdir="results")::String

Run `run_study(ctx, run)`, score it with `growth_scorecard(equity, spy_close, g_f)` on the result's dates
(equity in USD, SPY close in USD, g_f in yr⁻¹) and write `outdir/runs/<hash>.jld2` with keys
`study_key`, `result`, `scorecard`, `stats` (`study_stats`), `wall_seconds` (s) and `label`. The file is
written to `<hash>.jld2.tmp` and then renamed, so a reader never sees a partial file. Returns the hash.
Throws whatever the run throws (`run_study_grid` catches it).
"""
function run_and_save_study(ctx::MyStudyContext, run::MyStudyRun; outdir::AbstractString="results")::String
    h = study_hash(run)
    runs = _runs_dir(outdir); mkpath(runs)
    t0 = time_ns()
    res = run_study(ctx, run)
    md = ctx.md
    i0 = index_of(md, res.dates[1]); i1 = index_of(md, res.dates[end])
    sc = growth_scorecard(res.equity, md.spy_close[i0:i1], md.g_f[i0:i1])
    st = study_stats(res)
    wall = (time_ns() - t0) / 1e9
    final = joinpath(runs, h * ".jld2"); tmp = final * ".tmp"
    jldsave(tmp; study_key=study_key(run), result=res, scorecard=sc, stats=st, wall_seconds=wall, label=run.label)
    mv(tmp, final; force=true)
    return h
end

"""
    run_study_grid(md::MyMarketData, calib::MyCalibration, runs::Vector{MyStudyRun}; outdir="results")

Run every study run whose `outdir/runs/<hash>.jld2` is missing, threaded as in `run_grid`. Duplicate
runs (equal hash) run once; the first in `runs` determines the stored label. One `MyStudyContext` is
built per distinct `context_key` that still has work and shared by its runs (a context that fails to
build fails its runs, each writing an error file). A failure writes `<hash>.error.txt` and the sweep
continues; errored hashes are retried on the next call. Returns `(ran, skipped, errors)`, each a sorted
`Vector{String}` of hashes.
"""
function run_study_grid(md::MyMarketData, calib::MyCalibration, runs::Vector{MyStudyRun};
        outdir::AbstractString="results")
    rdir = _runs_dir(outdir); mkpath(rdir)
    seen = Set{String}(); todo = Tuple{String,MyStudyRun}[]; skipped = String[]
    for r in runs
        h = study_hash(r)
        h in seen && continue
        push!(seen, h)
        isfile(joinpath(rdir, h * ".jld2")) ? push!(skipped, h) : push!(todo, (h, r))
    end
    ctxs = Dict{Any,Any}()
    for (_, r) in todo
        k = context_key(r.settings)
        haskey(ctxs, k) && continue
        ctxs[k] = try
            build_context(md, calib, r.settings)
        catch err
            err
        end
    end
    function run_one(h, r)
        c = ctxs[context_key(r.settings)]
        c isa Exception && throw(c)
        run_and_save_study(c, r; outdir=outdir)
    end
    out = _run_items(todo, run_one; outdir=outdir, label_of=r -> r.label, key_repr=r -> repr(study_key(r)))
    return (ran=out.ran, skipped=sort!(skipped), errors=out.errors)
end

_study_label(a::AbstractString, scr, hedge::Bool, seed::Integer) = "$(a)|$(scr)|hedge=$(hedge)|seed=$(seed)"

"""
    tier1_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}

Tier 1 runs: allocators × screens × hedge, each on `s` with `hedge` set, with seeds
`base_seed .+ (0:n_seeds-1)`; the deterministic screens (`_STUDY_DETERMINISTIC`) get `[base_seed]` only. `grid` is the
`(n_seeds, base_seed, allocators, screens, hedge)` tuple of `load_study`. Labels look like
`"CD|bandit|hedge=true|seed=5660"`.
"""
function tier1_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}
    out = MyStudyRun[]
    for h in grid.hedge, a in grid.allocators, scr in grid.screens
        seeds = scr in _STUDY_DETERMINISTIC ? [grid.base_seed] : collect(grid.base_seed .+ (0:grid.n_seeds-1))
        sh = with_settings(s; hedge=h)
        for seed in seeds
            push!(out, build(MyStudyRun, (settings=sh, allocator=a, screen=scr, seed=seed,
                label=_study_label(a, scr, h, seed))))
        end
    end
    return out
end

"""
    fixed_schedule_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}

The hedge-on subset of `tier1_runs(s, grid)` with `repick_on_band_exit = false`: re-picks only on the
`repick_interval` timer. Labels add the suffix `|fixed`.
"""
function fixed_schedule_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}
    sf = with_settings(s; repick_on_band_exit=false)
    out = MyStudyRun[]
    for r in tier1_runs(sf, (; grid..., hedge=[true]))
        push!(out, build(MyStudyRun, (settings=r.settings, allocator=r.allocator, screen=r.screen, seed=r.seed,
            label=r.label * "|fixed")))
    end
    return out
end

const _STUDY_SCORE_FIELDS = (:sharpe, :max_drawdown, :alpha, :alpha_t, :beta, :realized_growth, :terminal, :vol)
const _STUDY_STAT_FIELDS = (:turnover, :total_cost, :borrow_fees, :n_decisions, :n_band_exit, :n_drawdown,
    :n_horizon, :n_cash_revisit, :n_basket_change, :n_repicks, :n_basket_changes, :mean_names, :mean_invested,
    :mean_portfolio_beta, :cap_bind_share, :mean_hedge_weight, :mean_pool_size)
const _STUDY_SUMMARY_COLUMNS = vcat([:hash, :label, :status, :allocator, :screen, :seed],
    collect(fieldnames(MyStudySettings)), [:wall_seconds], collect(_STUDY_SCORE_FIELDS), collect(_STUDY_STAT_FIELDS))

function _study_summary_row(file::AbstractString)
    h = first(split(basename(file), "."))
    d = load(file)
    k = d["study_key"]; sc = d["scorecard"]; st = d["stats"]
    row = Dict{Symbol,Any}(:hash => h, :label => get(d, "label", ""), :status => "ok",
        :allocator => k.allocator, :screen => String(k.screen), :seed => k.seed,
        :wall_seconds => get(d, "wall_seconds", missing))
    for f in fieldnames(MyStudySettings)
        row[f] = getfield(k, f)
    end
    for f in _STUDY_SCORE_FIELDS
        row[f] = getfield(sc, f)
    end
    for f in _STUDY_STAT_FIELDS
        row[f] = getfield(st, f)
    end
    return row
end

"""
    rebuild_study_summary(outdir, experiment; hashes=nothing)::DataFrame

Scan `outdir/runs/` and write `outdir/<experiment>_summary.csv`: one row per `<hash>.jld2`
(`status = "ok"`) and one per `<hash>.error.txt` with no matching `.jld2` (`status = "error"`, other
columns missing), sorted by hash and restricted to `hashes` when given. Columns: `hash, label, status,
allocator, screen` (a `String`), `seed`, every `MyStudySettings` field, `wall_seconds` (s), the
growth-scorecard fields (`sharpe, max_drawdown, alpha, alpha_t, beta, realized_growth, terminal, vol`)
and the `study_stats` fields. Returns the DataFrame.
"""
function rebuild_study_summary(outdir::AbstractString, experiment::AbstractString;
        hashes::Union{Nothing,AbstractVector{<:AbstractString}}=nothing)
    runs = _runs_dir(outdir)
    files = isdir(runs) ? sort(readdir(runs; join=true)) : String[]
    keep(f) = hashes === nothing || first(split(basename(f), ".")) in hashes
    rows = Dict{Symbol,Any}[]; ok = Set{String}()
    for f in files
        (endswith(f, ".jld2") && keep(f)) || continue
        push!(rows, _study_summary_row(f)); push!(ok, first(split(basename(f), ".")))
    end
    for f in files
        (endswith(f, ".error.txt") && keep(f)) || continue
        first(split(basename(f), ".")) in ok && continue
        push!(rows, _error_row(f))
    end
    sort!(rows; by=r -> r[:hash])
    df = DataFrame([c => Any[get(r, c, missing) for r in rows] for c in _STUDY_SUMMARY_COLUMNS]...)
    for c in names(df)
        df[!, c] = identity.(df[!, c])
    end
    CSV.write(joinpath(outdir, "$(experiment)_summary.csv"), df)
    return df
end
