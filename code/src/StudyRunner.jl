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

# Canonical text of a study-key value: independent of the caller's imports (a `Date` is its ISO string,
# never `repr`, which prints `Date(…)` or `Dates.Date(…)` depending on what `Main` has loaded).
_canon(x::Date) = string(x)
_canon(x::Bool) = string(x)
_canon(x::Integer) = string(x)
_canon(x::AbstractFloat) = repr(Float64(x))
_canon(x::Symbol) = ":" * String(x)
_canon(x::AbstractString) = repr(String(x))
_canon(x::AbstractVector) = "[" * join((_canon(v) for v in x), ",") * "]"
_canon(x) = repr(x)

"""
    study_key_string(run::MyStudyRun)::String

Canonical serialization of `study_key(run)`: `name=value` pairs joined by `;`, with dates as ISO strings,
so the text does not depend on the caller's `using` statements.
"""
study_key_string(run::MyStudyRun)::String = join(("$(k)=$(_canon(v))" for (k, v) in pairs(study_key(run))), ";")

"""
    study_hash(run::MyStudyRun; fingerprint::AbstractString="")::String

The 40-character file stem of the run's results: `bytes2hex(sha1(study_key_string(run)))`, with
`"|data=" * fingerprint` appended when `fingerprint` (a `data_fingerprint`) is non-empty. `run_study_grid`
and the scripts pass the fingerprint, so a change to the market data or the calibration gives new hashes.
"""
function study_hash(run::MyStudyRun; fingerprint::AbstractString="")::String
    txt = study_key_string(run)
    isempty(fingerprint) || (txt *= "|data=" * fingerprint)
    return bytes2hex(sha1(txt))
end

"""
    data_fingerprint(md::MyMarketData, calib::MyCalibration)::String

SHA-1 (40 hex characters) of everything a run reads besides its settings: every field of `md` (dates as
ISO strings, tickers, sectors, and the open, close, volume, SPY and g_f arrays by their Float64 bit
patterns, with their sizes) and of `calib` (`base_sim` by sorted ticker, `σ_m`, `m_m`, `v_m`, `λ`,
`train_start`, `train_stop`). Part of each run's identity in `run_study_grid`.
"""
function data_fingerprint(md::MyMarketData, calib::MyCalibration)::String
    ctx = SHA.SHA1_CTX()
    txt(x::AbstractString) = SHA.update!(ctx, Vector{UInt8}(codeunits(x * "\x1f")))
    num(x::AbstractArray{<:Real}) = (txt(string(size(x))); SHA.update!(ctx, collect(reinterpret(UInt8, Float64.(vec(x))))))
    txt("dates"); foreach(d -> txt(string(d)), md.dates)
    txt("tickers"); foreach(txt, md.tickers)
    txt("sector"); foreach(txt, md.sector)
    for f in (:open, :close, :volume, :spy_open, :spy_close, :g_f)
        txt(String(f)); num(getfield(md, f))
    end
    txt("base_sim")
    for t in sort!(collect(keys(calib.base_sim)))
        e = calib.base_sim[t]
        txt(t); num([e.α, e.β, e.σ_ε, e.r²])
    end
    txt("calib"); num([calib.σ_m, calib.m_m, calib.v_m, calib.λ])
    txt(string(calib.train_start)); txt(string(calib.train_stop))
    return bytes2hex(SHA.digest!(ctx))
end

# Effective number of names (names) of target weights w (dimensionless): 1/Σw̃ᵢ² with w̃ the entries above
# `_BT_HELD_W` rescaled to sum to 1; NaN when those weights sum to less than 0.01 (skipped decision).
function _eff_names(w::AbstractVector{<:Real})::Float64
    long = [x for x in w if x > _BT_HELD_W]
    tot = sum(long; init=0.0)
    tot < 0.01 && return NaN
    return 1 / sum(abs2, long ./ tot)
end

"""
    study_stats(res::MyStudyResult)::NamedTuple

Per-run activity statistics. `turnover` (yr⁻¹) = Σ|qty·price| over trades with qty ≠ 0 divided by the
mean equity (USD) and by the run length in years, n_days/252 with n_days = length(dates) − 1;
`total_cost` and `borrow_fees` in USD. `n_decisions` counts decisions and `n_band_exit`, `n_drawdown`,
`n_horizon`, `n_cash_revisit`, `n_basket_change` count them by `reason` (`:band_exit`, `:drawdown`,
`:horizon_elapsed`, `:cash_revisit`, `:basket_change`). `n_repicks` counts re-picks and `n_basket_changes`
those that changed the basket, both excluding the initial re-pick. `mean_names` (names) and
`mean_invested` (fraction of wealth) average over decisions. `mean_portfolio_beta` (dimensionless),
`mean_hedge_weight` (fraction of wealth) and `mean_pool_size` (names) average over days. `basket_size` (names) is the median pick count over re-picks.
`cap_bind_share` is the share of decisions with `cap_binds` (dimensionless), `NaN` when the β cap is off
(no decision has a finite cap). Reading "share of days the cap binds" as decision days, the only days the
cap is enforced, is a **(plan decision)**.

`mean_eff_names` (names) is the mean over decisions of the effective number of names 1/Σw̃ᵢ², where w̃ are
the long target weights above `_BT_HELD_W` (dimensionless) rescaled to sum to 1; a decision whose such
weights sum to less than 0.01 is skipped and counted in `n_eff_skipped` (decisions). Rescaling the long
weights and skipping near-cash decisions is a **(plan decision)**. `n_failed` (decisions) counts
decisions with status `:infeasible` or `:solver_failed`. `n_fallback` (decisions) counts decisions that did
not fail but whose `solver` is `:clarabel_almost`, `:scs` or `:failed` (a failed sub-solve such as the hedge
step), and `n_almost_optimal` (decisions) those whose `solver` is `:clarabel_almost`. Counting every
non-Clarabel-optimal outcome of an unfailed decision as a fallback is a **(plan decision)**.
"""
function study_stats(res::MyStudyResult)::NamedTuple
    tr = res.trades; live = tr.qty .!= 0
    traded = sum(abs.(tr.qty[live] .* tr.price[live]); init=0.0)
    years = (length(res.dates) - 1) / 252
    dec = res.decisions; rp = res.repicks; dy = res.daily
    cnt(r) = count(==(r), dec.reason)
    mn(x) = isempty(x) ? NaN : mean(x)
    capped = any(isfinite, dec.beta_max)
    eff = [_eff_names(collect(values(res.targets[d]))) for d in dec.date]
    ok = .!(dec.status .∈ Ref(_STUDY_FAILED))
    return (turnover=years > 0 ? traded / mean(res.equity) / years : NaN,
        total_cost=sum(tr.cost; init=0.0), borrow_fees=sum(res.borrow_fees),
        n_decisions=nrow(dec), n_band_exit=cnt(:band_exit), n_drawdown=cnt(:drawdown),
        n_horizon=cnt(:horizon_elapsed), n_cash_revisit=cnt(:cash_revisit), n_basket_change=cnt(:basket_change),
        n_repicks=max(nrow(rp) - 1, 0), n_basket_changes=count(rp.changed[2:end]),
        mean_names=mn(dec.n_names), mean_invested=mn(dec.invested_frac),
        mean_portfolio_beta=mn(dy.portfolio_beta),
        cap_bind_share=capped ? mean(dec.cap_binds) : NaN,
        mean_hedge_weight=mn(dy.hedge_weight), mean_pool_size=mn(dy.pool_size),
        basket_size=median(length.(rp.basket)),
        mean_eff_names=mn(filter(!isnan, eff)), n_eff_skipped=count(isnan, eff),
        n_failed=count(!, ok), n_fallback=count(ok .& (dec.solver .∈ Ref((:clarabel_almost, :scs, :failed)))),
        n_almost_optimal=count(ok .& (dec.solver .== :clarabel_almost)))
end

"""
    regime_stats(res::MyStudyResult, spy_close::AbstractVector{<:Real})::NamedTuple

Mean growth by market state (spec §7). `spy_close` (USD) has one entry per entry of `res.dates`. Each
run day k ≥ 2 has growth g_k = ln(V_k/V_{k−1})·252 (yr⁻¹) for the portfolio (V = `res.equity`, USD) and
for SPY (V = `spy_close`). Day k is bullish when ξ on day k−1 (`res.daily.xi[k−1]`, dimensionless) is
negative, bearish when it is positive, and dropped when it is 0; labeling day k by the previous day's ξ,
which was known before day k opened, is a **(plan decision)**. Returns `g_bull`, `g_bear`, `spy_g_bull`,
`spy_g_bear` (yr⁻¹; means over each label's days, `NaN` when there are none) and `n_bull`, `n_bear`
(days).
"""
function regime_stats(res::MyStudyResult, spy_close::AbstractVector{<:Real})::NamedTuple
    n = length(res.dates)
    length(spy_close) == n || throw(ArgumentError("spy_close needs one entry per run date"))
    gp = [log(res.equity[k] / res.equity[k-1]) * 252 for k in 2:n]
    gs = [log(spy_close[k] / spy_close[k-1]) * 252 for k in 2:n]
    xi = res.daily.xi[1:n-1]
    bull = xi .< 0; bear = xi .> 0
    mn(x) = isempty(x) ? NaN : mean(x)
    return (g_bull=mn(gp[bull]), g_bear=mn(gp[bear]), n_bull=count(bull), n_bear=count(bear),
        spy_g_bull=mn(gs[bull]), spy_g_bear=mn(gs[bear]))
end

"""
    run_and_save_study(ctx::MyStudyContext, run::MyStudyRun; outdir="results",
        fingerprint=data_fingerprint(ctx.md, ctx.calib))::String

Run `run_study(ctx, run)`, score it with `growth_scorecard(equity, spy_close, g_f)` on the result's dates
(equity in USD, SPY close in USD, g_f in yr⁻¹) and write `outdir/runs/<hash>.jld2`, with
hash = `study_hash(run; fingerprint)`, holding `study_key`, `data_fingerprint`, `result`, `scorecard`,
`stats` (`study_stats`), `wall_seconds` (s) and `label`. The file is written to `<hash>.jld2.tmp` and then
renamed, so a reader never sees a partial file. Returns the hash. Throws whatever the run throws
(`run_study_grid` catches it).
"""
function run_and_save_study(ctx::MyStudyContext, run::MyStudyRun; outdir::AbstractString="results",
        fingerprint::AbstractString=data_fingerprint(ctx.md, ctx.calib))::String
    h = study_hash(run; fingerprint=fingerprint)
    runs = _runs_dir(outdir); mkpath(runs)
    t0 = time_ns()
    res = run_study(ctx, run)
    md = ctx.md
    i0 = index_of(md, res.dates[1]); i1 = index_of(md, res.dates[end])
    sc = growth_scorecard(res.equity, md.spy_close[i0:i1], md.g_f[i0:i1])
    st = merge(study_stats(res), regime_stats(res, md.spy_close[i0:i1]))
    wall = (time_ns() - t0) / 1e9
    final = joinpath(runs, h * ".jld2"); tmp = final * ".tmp"
    jldsave(tmp; study_key=study_key(run), data_fingerprint=String(fingerprint), result=res, scorecard=sc, stats=st,
        wall_seconds=wall, label=run.label)
    mv(tmp, final; force=true)
    return h
end

"""
    run_study_grid(md::MyMarketData, calib::MyCalibration, runs::Vector{MyStudyRun}; outdir="results", fresh=false)

Run every study run whose `outdir/runs/<hash>.jld2` is missing, threaded as in `run_grid`. The hash is
`study_hash(run; fingerprint)` with `fingerprint = data_fingerprint(md, calib)`, computed once, so cached
runs from other data or another calibration are not reused. With `fresh = true` the cache is ignored for
the selected runs: their existing `<hash>.jld2` and `<hash>.error.txt` are deleted and every run reruns
(use it after a code change, which the hash cannot see). Duplicate runs (equal hash) run once; the first
in `runs` determines the stored label. One `MyStudyContext` is built per distinct `context_key` that still
has work and shared by its runs (a context that fails to build fails its runs, each writing an error
file). A failure writes `<hash>.error.txt` and the sweep continues; errored hashes are retried on the
next call. Returns `(ran, skipped, errors, fingerprint)`: sorted `Vector{String}`s of hashes and the
fingerprint (pass it to `study_hash` for `rebuild_study_summary(...; hashes)`).
"""
function run_study_grid(md::MyMarketData, calib::MyCalibration, runs::Vector{MyStudyRun};
        outdir::AbstractString="results", fresh::Bool=false)
    rdir = _runs_dir(outdir); mkpath(rdir)
    fp = data_fingerprint(md, calib)
    seen = Set{String}(); todo = Tuple{String,MyStudyRun}[]; skipped = String[]
    for r in runs
        h = study_hash(r; fingerprint=fp)
        h in seen && continue
        push!(seen, h)
        if fresh
            rm(joinpath(rdir, h * ".jld2"); force=true); rm(joinpath(rdir, h * ".error.txt"); force=true)
        end
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
        run_and_save_study(c, r; outdir=outdir, fingerprint=fp)
    end
    out = _run_items(todo, run_one; outdir=outdir, label_of=r -> r.label,
        key_repr=r -> study_key_string(r) * "|data=" * fp)
    return (ran=out.ran, skipped=sort!(skipped), errors=out.errors, fingerprint=fp)
end

_study_label(a::AbstractString, scr, hedge::Bool, seed::Integer) = "$(a)|$(scr)|hedge=$(hedge)|seed=$(seed)"

"""
    _eta_text(η::Real)::String

An elasticity η (dimensionless) as label text: whole numbers without a decimal point (`2.0 → "2"`,
`0.05 → "0.05"`).
"""
_eta_text(η::Real) = isinteger(η) ? string(Int(η)) : string(Float64(η))

"""
    eta_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}

The η sweep inside the study **(plan decision)**: 2,562 runs with the default grid. Cap on: `"CES"` and `"CESw"` at
every `grid.eta` except `s.ces_eta`, on `with_settings(s; ces_eta=η)`. Cap off (`w_max = 1`): `"CD"` once, plus `"CES"` and
`"CESw"` at every `grid.eta`. Each is `tier1_runs` with `allocators=[a]`, labeled
`<tier1 label>|eta=<η>|cap=<on|off>`; CD's label carries η = 1 but its settings keep `ces_eta = s.ces_eta`, which CD ignores.
Share-count `"CES"` (never `"CESw"`) is left out at every η in `grid.eta_ces_drop`, cap on and off **(user, 2026-10-22)**: it
is ill-conditioned when a limit binds. A grid without that field drops nothing.
"""
function eta_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}
    out = MyStudyRun[]
    add!(sv, a, η, cap) = for r in tier1_runs(sv, (; grid..., allocators=[a]))
        push!(out, build(MyStudyRun, (settings=r.settings, allocator=r.allocator, screen=r.screen, seed=r.seed,
            label=r.label * "|eta=$(_eta_text(η))|cap=$(cap)")))
    end
    drop = get(grid, :eta_ces_drop, Float64[])
    for a in ("CES", "CESw"), η in grid.eta
        (a == "CES" && η in drop) && continue
        η == s.ces_eta || add!(with_settings(s; ces_eta=η), a, η, "on")
    end
    add!(with_settings(s; w_max=1.0), "CD", 1, "off")
    for a in ("CES", "CESw"), η in grid.eta
        (a == "CES" && η in drop) && continue
        add!(with_settings(s; ces_eta=η, w_max=1.0), a, η, "off")
    end
    return out
end

"""
    limits_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}

The clean budget-only limits on the fixed list `grid.limits_list` **(plan decision)**: 17 runs, `"CD"` (η = 1 in the
label) plus `"CES"` and `"CESw"` at every `grid.eta_limits`. Each uses screen `:fixed`, seed `grid.base_seed` and
`with_settings(s; fixed_tickers=list, basket_K=length(list), hedge=false, w_max=1.0, vol_cap_annual=Inf, s_max=1.0)`
(plus `ces_eta=η` for CES and CESw; CD keeps `s.ces_eta`). Labels look like
`"CES|fixed:l8b_13|hedge=false|seed=5660|eta=0.05|limits"`.
"""
function limits_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}
    list = collect(String, last(grid.fixed_lists[findfirst(p -> first(p) == grid.limits_list, grid.fixed_lists)]))
    sl = with_settings(s; fixed_tickers=list, basket_K=length(list), hedge=false, w_max=1.0, vol_cap_annual=Inf, s_max=1.0)
    out = MyStudyRun[]
    for (a, η) in vcat(["CD" => 1.0], [(a => e) for a in ("CES", "CESw") for e in grid.eta_limits])
        sv = a == "CD" ? sl : with_settings(sl; ces_eta=η)
        push!(out, build(MyStudyRun, (settings=sv, allocator=a, screen=:fixed, seed=grid.base_seed,
            label=_study_label(a, "fixed:$(grid.limits_list)", false, grid.base_seed) * "|eta=$(_eta_text(η))|limits")))
    end
    return out
end

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
    universe_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}

Size-sweep runs: `grid.K` × `grid.hedge` × `grid.allocators` × `grid.universe_screens`, each on
`with_settings(s; basket_K=K, hedge=h)`. Deterministic screens (`_STUDY_DETERMINISTIC`) get seed
`[base_seed]`, the others `base_seed .+ (0:n_seeds-1)`. Labels look like
`"CD|global_gamma|K=13|hedge=true|seed=5660"`.
"""
function universe_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}
    out = MyStudyRun[]
    for K in grid.K, h in grid.hedge, a in grid.allocators, scr in grid.universe_screens
        seeds = scr in _STUDY_DETERMINISTIC ? [grid.base_seed] : collect(grid.base_seed .+ (0:grid.n_seeds-1))
        sh = with_settings(s; basket_K=K, hedge=h)
        for seed in seeds
            push!(out, build(MyStudyRun, (settings=sh, allocator=a, screen=scr, seed=seed,
                label="$(a)|$(scr)|K=$(K)|hedge=$(h)|seed=$(seed)")))
        end
    end
    return out
end

"""
    fixed_list_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}

Runs of the fixed lists: `grid.fixed_lists` (name => tickers) × `grid.hedge` × `grid.allocators` with
screen `:fixed`, `fixed_tickers = list`, `basket_K = length(list)` and seed `base_seed`. Labels look like
`"CD|fixed:l8b_13|hedge=true|seed=5660"`.
"""
function fixed_list_runs(s::MyStudySettings, grid)::Vector{MyStudyRun}
    out = MyStudyRun[]
    for (name, list) in grid.fixed_lists, h in grid.hedge, a in grid.allocators
        sh = with_settings(s; fixed_tickers=collect(String, list), basket_K=length(list), hedge=h)
        push!(out, build(MyStudyRun, (settings=sh, allocator=a, screen=:fixed, seed=grid.base_seed,
            label=_study_label(a, "fixed:$(name)", h, grid.base_seed))))
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
    :mean_portfolio_beta, :cap_bind_share, :mean_hedge_weight, :mean_pool_size, :basket_size,
    :mean_eff_names, :n_eff_skipped, :n_failed, :n_fallback, :n_almost_optimal,
    :g_bull, :g_bear, :n_bull, :n_bear, :spy_g_bull, :spy_g_bear)
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
        v = getfield(k, f)
        row[f] = v isa AbstractVector ? join(v, ";") : v
    end
    for f in _STUDY_SCORE_FIELDS
        row[f] = getfield(sc, f)
    end
    for f in _STUDY_STAT_FIELDS
        row[f] = get(st, f, missing)
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
