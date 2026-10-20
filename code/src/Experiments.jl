# Resumable, threaded experiment runner: one JLD2 per run (keyed by a config hash), one summary CSV.

# Hook fields that hold dictionaries are reduced to key-sorted pairs so the repr does not depend on
# insertion order.
_key_hook(x) = x === nothing ? nothing : sort!(collect(x); by=first)

"""
    config_key(cfg::MyRunConfig)::NamedTuple

Identity of a run: every field of `cfg` except `base_sim`, with `basket_schedule` reduced to the hex
SHA1 of its repr (a `String`). The validation-only hooks (`ewls_g_m`, `decision_market`,
`ewls_state0`) are always present (as `nothing` by default); dictionary hooks are stored as
key-sorted pairs. Dimensions are those of the fields (yr⁻¹, USD, shares, dates).
"""
function config_key(cfg::MyRunConfig)::NamedTuple
    fields = Symbol[]; vals = Any[]
    for f in fieldnames(MyRunConfig)
        f === :base_sim && continue
        v = getfield(cfg, f)
        f === :basket_schedule && (v = bytes2hex(sha1(repr(v))))
        f in (:decision_market, :ewls_state0) && (v = _key_hook(v))
        push!(fields, f); push!(vals, v)
    end
    return NamedTuple{Tuple(fields)}(Tuple(vals))
end

"""
    config_hash(cfg::MyRunConfig)::String

`bytes2hex(sha1(repr(config_key(cfg))))`, the 40-character file stem of the run's results.
"""
config_hash(cfg::MyRunConfig)::String = bytes2hex(sha1(repr(config_key(cfg))))

_runs_dir(outdir::AbstractString) = joinpath(outdir, "runs")

"""
    run_and_save(md::MyMarketData, cfg::MyRunConfig; outdir="results")::String

Run `run_backtest(md, cfg)`, score it with `scorecard(equity, spy_close, g_f)` on the result's dates
(equity in USD, SPY close in USD, g_f in yr⁻¹), and write `outdir/runs/<hash>.jld2` with keys
`config_key`, `result`, `scorecard`, `wall_seconds` (s), `basket_kind` (`"static"` for a one-entry
basket schedule, else `"scheduled"`). The file is written to `<hash>.jld2.tmp`
and then renamed (`mv(...; force=true)`), so a reader never sees a partial file. Returns the hash.
Throws whatever the backtest throws (`run_grid` catches it).
"""
function run_and_save(md::MyMarketData, cfg::MyRunConfig; outdir::AbstractString="results")::String
    h = config_hash(cfg)
    runs = _runs_dir(outdir); mkpath(runs)
    t0 = time_ns()
    res = run_backtest(md, cfg)
    i0 = index_of(md, res.dates[1]); i1 = index_of(md, res.dates[end])
    sc = scorecard(res.equity, md.spy_close[i0:i1], md.g_f[i0:i1])
    wall = (time_ns() - t0) / 1e9
    final = joinpath(runs, h * ".jld2"); tmp = final * ".tmp"
    jldsave(tmp; config_key=config_key(cfg), result=res, scorecard=sc, wall_seconds=wall,
        basket_kind=length(cfg.basket_schedule) == 1 ? "static" : "scheduled")
    mv(tmp, final; force=true)
    return h
end

function _write_error(runs::AbstractString, h::AbstractString, cfg::MyRunConfig, err, bt)
    path = joinpath(runs, h * ".error.txt"); tmp = path * ".tmp"
    open(tmp, "w") do io
        println(io, "label: ", cfg.label)
        println(io, "hash: ", h)
        println(io, "config_key: ", repr(config_key(cfg)))
        println(io, "exception: ")
        showerror(io, err, bt)
        println(io)
    end
    mv(tmp, path; force=true)
end

"""
    run_grid(md::MyMarketData, cfgs::Vector{MyRunConfig}; outdir="results")

Run every config whose `outdir/runs/<hash>.jld2` is missing (a stray `.tmp` does not count), using
`Threads.@threads`. Duplicate configs (equal hash) run once. When `Threads.nthreads() > 1`, BLAS is
limited to one thread. Each run is wrapped in `try/catch`: a failure writes
`<hash>.error.txt` (exception, stacktrace, config key) and the sweep continues; errored hashes are
retried on the next call, and a success deletes the stale error file. Returns
`(ran=Vector{String}, skipped=Vector{String}, errors=Vector{String})` of hashes.
"""
function run_grid(md::MyMarketData, cfgs::Vector{MyRunConfig}; outdir::AbstractString="results")
    Threads.nthreads() > 1 && BLAS.set_num_threads(1)
    runs = _runs_dir(outdir); mkpath(runs)
    seen = Set{String}(); todo = Tuple{String,MyRunConfig}[]; skipped = String[]
    for c in cfgs
        h = config_hash(c)
        h in seen && continue
        push!(seen, h)
        isfile(joinpath(runs, h * ".jld2")) ? push!(skipped, h) : push!(todo, (h, c))
    end
    ran = String[]; errors = String[]; lk = ReentrantLock()
    Threads.@threads :dynamic for k in eachindex(todo)
        h, c = todo[k]
        try
            run_and_save(md, c; outdir=outdir)
            rm(joinpath(runs, h * ".error.txt"); force=true)
            lock(() -> push!(ran, h), lk)
        catch err
            bt = catch_backtrace()
            _write_error(runs, h, c, err, bt)
            lock(lk) do
                push!(errors, h)
                @warn "run failed" hash = h label = c.label exception = (err, bt)
            end
        end
    end
    return (ran=ran, skipped=skipped, errors=errors)
end

# Rule name with its parameters, e.g. "MyCESRule(η=2.0)"; parameter-free rules give just the type name.
function _rule_name(rule)::String
    T = typeof(rule); fs = [f for f in fieldnames(T) if isdefined(rule, f)]
    isempty(fs) && return string(nameof(T))
    return string(nameof(T), "(", join(("$(f)=$(getfield(rule, f))" for f in fs), ", "), ")")
end

const _SUMMARY_METRICS = (:sharpe, :max_drawdown, :alpha_ann, :alpha_t, :beta, :ann_return, :ann_vol)

"""
    run_stats(res::MyBacktestResult)::NamedTuple

Per-run activity statistics. `turnover` = Σ|qty·price| (USD) over trades with qty ≠ 0, divided by the
mean equity (USD) and by the run length in years ((n_days)/252), i.e. fraction of equity traded per
year (yr⁻¹). `total_cost` = Σ cost (USD); `borrow_fees` = Σ daily borrow fees (USD); `n_*` count
decisions by `reason`; `mean_names` = mean `n_names`; `mean_invested` = mean `invested_frac`.
"""
function run_stats(res::MyBacktestResult)::NamedTuple
    tr = res.trades; live = tr.qty .!= 0
    traded = sum(abs.(tr.qty[live] .* tr.price[live]); init=0.0)
    years = (length(res.dates) - 1) / 252
    dec = res.decisions
    cnt(r) = count(==(r), dec.reason)
    return (turnover=years > 0 ? traded / mean(res.equity) / years : NaN,
        total_cost=sum(tr.cost; init=0.0), borrow_fees=sum(res.borrow_fees),
        n_decisions=nrow(dec), n_band_exit=cnt(:band_exit), n_drawdown=cnt(:drawdown),
        n_horizon=cnt(:horizon_elapsed), n_cash_revisit=cnt(:cash_revisit),
        mean_names=isempty(dec.n_names) ? NaN : mean(dec.n_names),
        mean_invested=isempty(dec.invested_frac) ? NaN : mean(dec.invested_frac))
end

const _SUMMARY_COLUMNS = [:hash, :label, :rule, :covariance, :λ, :w_max, :σ_g_max, :s_max, :z, :T,
    :basket_kind, :seed, :start, :stop, :status, :wall_seconds,
    :sharpe, :max_drawdown, :alpha_ann, :alpha_t, :beta, :ann_return, :ann_vol,
    :turnover, :total_cost, :borrow_fees,
    :n_decisions, :n_band_exit, :n_drawdown, :n_horizon, :n_cash_revisit, :mean_names, :mean_invested]

function _summary_row(file::AbstractString)
    h = first(split(basename(file), "."))
    d = load(file)
    k = d["config_key"]; sc = d["scorecard"]; st = run_stats(d["result"])
    cov = k.covariance
    base = Dict{Symbol,Any}(:hash => h, :label => k.label, :rule => _rule_name(k.rule),
        :covariance => cov isa MyShrunkResidualCovariance ? "C2" : "C1",
        :λ => cov isa MyShrunkResidualCovariance ? cov.λ : missing,
        :w_max => k.w_max, :σ_g_max => k.σ_g_max, :s_max => k.s_max, :z => k.mpc.z, :T => k.mpc.T,
        :basket_kind => get(d, "basket_kind", missing), :seed => k.seed, :start => k.start, :stop => k.stop, :status => "ok",
        :wall_seconds => get(d, "wall_seconds", missing))
    for m in _SUMMARY_METRICS
        base[m] = getfield(sc, m)
    end
    for m in keys(st)
        base[m] = getfield(st, m)
    end
    return base
end

# Parse the `label:` line of an error file; the hash is the file stem.
function _error_row(file::AbstractString)
    h = first(split(basename(file), "."))
    label = ""
    for line in eachline(file)
        startswith(line, "label: ") && (label = line[8:end]; break)
    end
    return Dict{Symbol,Any}(:hash => h, :label => label, :status => "error")
end

"""
    rebuild_summary(outdir::String, experiment::String)::DataFrame

Scan `outdir/runs/` and write `outdir/<experiment>_summary.csv`: one row per `<hash>.jld2`
(`status = "ok"`) and one per `<hash>.error.txt` with no matching `.jld2` (`status = "error"`,
metrics missing), sorted by hash. Columns: `hash, label, rule, covariance, λ, w_max, σ_g_max, s_max,
z, T, basket_kind, seed, start, stop, status, wall_seconds`, the scorecard (`sharpe, max_drawdown,
alpha_ann, alpha_t, beta, ann_return, ann_vol`) and `run_stats` (`turnover, total_cost, borrow_fees,
n_decisions, n_band_exit, n_drawdown, n_horizon, n_cash_revisit, mean_names, mean_invested`).
`λ` is missing for C1. `basket_kind` is the value stored by `run_and_save`. Returns the DataFrame.
"""
function rebuild_summary(outdir::AbstractString, experiment::AbstractString)
    runs = _runs_dir(outdir)
    files = isdir(runs) ? sort(readdir(runs; join=true)) : String[]
    rows = Dict{Symbol,Any}[]
    okhashes = Set{String}()
    for f in files
        endswith(f, ".jld2") || continue
        push!(rows, _summary_row(f)); push!(okhashes, first(split(basename(f), ".")))
    end
    for f in files
        endswith(f, ".error.txt") || continue
        first(split(basename(f), ".")) in okhashes && continue
        push!(rows, _error_row(f))
    end
    sort!(rows; by=r -> r[:hash])
    df = DataFrame([c => Any[get(r, c, missing) for r in rows] for c in _SUMMARY_COLUMNS]...)
    for c in names(df)
        df[!, c] = identity.(df[!, c])
    end
    CSV.write(joinpath(outdir, "$(experiment)_summary.csv"), df)
    return df
end

const _RULE_CTORS = Dict{String,Function}(
    "CD" => (p) -> MyCobbDouglasRule(),
    "CES" => (p) -> build(MyCESRule, (η=get(p, :η, 2.0),)),
    "SG" => (p) -> build(MyStoneGearyRule, (baseline_frac=get(p, :baseline_frac, 0.01),)),
    "MinVar" => (p) -> MyMinVarRule(),
    "Tangent" => (p) -> MyTangentRule(),
    "LS" => (p) -> MyLongShortCDRule())

# Values of a TOML key as a vector (a scalar becomes a one-element vector).
_as_vec(x) = x isa AbstractVector ? collect(x) : [x]

"""
    load_experiment(path; md, base_sim, start, stop, σ_m=2.0, W0=100_000.0)::Vector{MyRunConfig}

Read a TOML file and expand it into the cartesian product of its array-valued parameters (a scalar is
a one-element array). Minimal by design: Julia scripts usually build configs directly. Keys:

| Key | Meaning | Default |
|---|---|---|
| `rules` | strings from `CD, CES, SG, MinVar, Tangent, LS` | `["CD"]` |
| `eta` | CES elasticity η (dimensionless, > 1) | 2.0 |
| `baseline_frac` | Stone–Geary baseline fraction | 0.01 |
| `covariance` | `"C1"` or `"C2"` | `"C1"` |
| `lambda` (or quoted `"λ"`) | C2 shrinkage (0..1; ignored for C1) | 0.5 |
| `w_max`, `s_max` | per-name / sector caps (dimensionless) | 0.1, 1.0 |
| `sigma_g_max_annual` | annualized vol cap; σ_g,max = value·√252 (yr⁻¹) | 0.12 |
| `z`, `T`, `seed` | band width, horizon (trading days), RNG seed | 1.96, 21, 5660 |
| `basket` | `"all"` (every ticker of `md`) or a list of tickers | `"all"` |

`md`, `base_sim`, `start`, `stop` (a trading day with a previous one), `σ_m` (yr⁻¹) and `W0` (USD)
come from the caller, since they are data rather than parameters. Each config's `label` is its
`key=value` list. Unknown TOML keys throw `ArgumentError`.
"""
function load_experiment(path::AbstractString; md::MyMarketData, base_sim, start::Date, stop::Date,
        σ_m::Real=2.0, W0::Real=100_000.0)::Vector{MyRunConfig}
    t = TOML.parsefile(path)
    haskey(t, "lambda") && (t["λ"] = pop!(t, "lambda"))   # TOML bare keys are ASCII; "λ" must be quoted
    known = ("rules", "eta", "baseline_frac", "covariance", "λ", "w_max", "s_max", "sigma_g_max_annual",
        "z", "T", "seed", "basket")
    bad = setdiff(keys(t), known)
    isempty(bad) || throw(ArgumentError("unknown experiment key(s) $(sort!(collect(bad))); known: $(known)"))
    axes = [(k, _as_vec(get(t, k, d))) for (k, d) in (("rules", "CD"), ("eta", 2.0), ("baseline_frac", 0.01),
        ("covariance", "C1"), ("λ", 0.5), ("w_max", 0.1), ("s_max", 1.0), ("sigma_g_max_annual", 0.12),
        ("z", 1.96), ("T", 21), ("seed", 5660), ("basket", "all"))]
    out = MyRunConfig[]
    for combo in Iterators.product((last(a) for a in axes)...)
        p = Dict(Symbol(first(a)) => v for (a, v) in zip(axes, combo))
        rname = p[:rules]
        haskey(_RULE_CTORS, rname) || throw(ArgumentError("unknown rule \"$(rname)\"; known: $(sort!(collect(keys(_RULE_CTORS))))"))
        p[:covariance] in ("C1", "C2") || throw(ArgumentError("covariance must be \"C1\" or \"C2\", got \"$(p[:covariance])\""))
        cov = p[:covariance] == "C1" ? MySIMCovariance() : build(MyShrunkResidualCovariance, (λ=Float64(p[:λ]),))
        names = p[:basket] == "all" ? copy(md.tickers) : String.(_as_vec(p[:basket]))
        label = join(("$(k)=$(p[Symbol(k)])" for (k, _) in axes
            if length(last(only(filter(a -> first(a) == k, axes)))) > 1), ", ")
        push!(out, build(MyRunConfig, (rule=_RULE_CTORS[rname](p), covariance=cov, w_max=Float64(p[:w_max]),
            s_max=Float64(p[:s_max]), σ_g_max=Float64(p[:sigma_g_max_annual]) * sqrt(252),
            mpc=build(MyMPCSpec, (z=Float64(p[:z]), T=Int(p[:T]))), start=start, stop=stop, W0=Float64(W0),
            seed=Int(p[:seed]), basket_schedule=[start => names], base_sim=base_sim, σ_m=Float64(σ_m),
            label=label)))
    end
    return out
end
