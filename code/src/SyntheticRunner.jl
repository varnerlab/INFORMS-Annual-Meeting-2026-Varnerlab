"""
    market_path_stats(gen::MySyntheticGenerator, ids::AbstractVector{Int}; window=(Date(2025,1,2), Date(2026,9,4)),
        train=(Date(2014,1,3), Date(2024,12,31)))::DataFrame

One row per path id in `ids`, in the order of `ids`. The market is drawn exactly as `synthetic_draws` draws it, and the
SPY closes (USD/share) compound from `gen.spy0` as exp(g_m·Δt) with Δt = 1/252 yr, as `market_data_from_growth` does.
Columns: `path_id`; `window_max_drawdown` (dimensionless fraction, peak reset at the window start, on the closes of
the window's dates); `window_growth` and `train_growth` (yr⁻¹, ln(S_end/S_start)/(trading days/252) over the window
and over the training window); `has_burst` (true if any jump flag falls on the window's growth rows). Threaded over ids.
"""
function market_path_stats(gen::MySyntheticGenerator, ids::AbstractVector{Int};
        window::Tuple{Date,Date}=(Date(2025, 1, 2), Date(2026, 9, 4)),
        train::Tuple{Date,Date}=(Date(2014, 1, 3), Date(2024, 12, 31)))::DataFrame
    Δt = 1 / 252
    n = length(gen.dates) - 1
    w0 = findfirst(==(window[1]), gen.dates)
    w1 = findfirst(==(window[2]), gen.dates)
    t0 = findfirst(==(train[1]), gen.dates)
    t1 = findfirst(==(train[2]), gen.dates)
    if w0 === nothing || w1 === nothing || t0 === nothing || t1 === nothing
        throw(ArgumentError("window $(window) and train $(train) must both start and end on dates of gen.dates"))
    end
    rows = growth_window(gen.dates, window[1], window[2])
    m = length(ids)
    drawdown = zeros(Float64, m)
    wgrowth = zeros(Float64, m)
    tgrowth = zeros(Float64, m)
    burst = fill(false, m)
    Threads.@threads for k in 1:m
        mkt = JumpHMM.simulate(gen.market, n; n_paths=1, seed=path_seed(ids[k], :market)).paths[1]
        S = gen.spy0 .* exp.(vcat(0.0, cumsum(mkt.observations .* Δt)))
        W = S[w0:w1]
        drawdown[k] = maximum(1 .- W ./ accumulate(max, W))
        wgrowth[k] = log(S[w1] / S[w0]) / ((w1 - w0) * Δt)
        tgrowth[k] = log(S[t1] / S[t0]) / ((t1 - t0) * Δt)
        burst[k] = any(mkt.jumps[rows])
    end
    return DataFrame(path_id=collect(ids), window_max_drawdown=drawdown, window_growth=wgrowth,
        train_growth=tgrowth, has_burst=burst)
end

"""
    path_sets(stats::DataFrame; n_burst=200, n_bear=200, n_typical=600, seed=5660)::DataFrame

Add the three study sets to `stats` (from `market_path_stats`). `burst_stress` marks the `n_burst` deepest
`window_max_drawdown` among `has_burst` paths, `bear_stress` the `n_bear` deepest among the others. `typical_rank` is the
position of the row in `randperm(MersenneTwister(seed), nrow(stats))` and `typical` is `typical_rank ≤ n_typical`.
`run_order` is missing for a path in no set. Otherwise it numbers the paths 1…m by (priority, path_id), where priority is
the smallest of rank-in-set / set-size over the path's sets (rank 1 is the deepest drawdown in a stress set and
`typical_rank` in the typical set), so stopping early leaves every set equally complete. Throws `ArgumentError` if a
stress pool is smaller than its quota.
"""
function path_sets(stats::DataFrame; n_burst::Int=200, n_bear::Int=200, n_typical::Int=600, seed::Int=5660)::DataFrame
    out = copy(stats)
    m = nrow(out)
    burst_pool = findall(out.has_burst)
    bear_pool = findall(.!out.has_burst)
    if length(burst_pool) < n_burst
        throw(ArgumentError("burst pool has $(length(burst_pool)) paths, fewer than n_burst = $(n_burst)"))
    end
    if length(bear_pool) < n_bear
        throw(ArgumentError("bear pool has $(length(bear_pool)) paths, fewer than n_bear = $(n_bear)"))
    end
    if n_typical > m
        throw(ArgumentError("n_typical = $(n_typical) exceeds the $(m) paths"))
    end
    priority = fill(Inf, m)
    burst_stress = fill(false, m)
    bear_stress = fill(false, m)
    # stress sets: deepest first, rank 1 is the deepest
    burst_top = burst_pool[sortperm(out.window_max_drawdown[burst_pool]; rev=true)[1:n_burst]]
    for (r, i) in enumerate(burst_top)
        burst_stress[i] = true
        priority[i] = min(priority[i], r / n_burst)
    end
    bear_top = bear_pool[sortperm(out.window_max_drawdown[bear_pool]; rev=true)[1:n_bear]]
    for (r, i) in enumerate(bear_top)
        bear_stress[i] = true
        priority[i] = min(priority[i], r / n_bear)
    end
    # typical set: position in a seeded permutation
    typical_rank = randperm(MersenneTwister(seed), m)
    typical = typical_rank .<= n_typical
    for i in 1:m
        if typical[i]
            priority[i] = min(priority[i], typical_rank[i] / n_typical)
        end
    end
    # run order over the paths in at least one set
    members = findall(isfinite, priority)
    order = sortperm(members; by=i -> (priority[i], out.path_id[i]))
    run_order = Vector{Union{Missing,Int}}(missing, m)
    for (pos, k) in enumerate(order)
        run_order[members[k]] = pos
    end
    out.burst_stress = burst_stress
    out.bear_stress = bear_stress
    out.typical_rank = typical_rank
    out.typical = typical
    out.run_order = run_order
    return out
end

"""
    synthetic_runs(s::MyStudySettings, grid; n_seeds::Int=5)::Vector{MyStudyRun}

The study runs made on every synthetic path: 532 with the default grid and `n_seeds = 5`. `grid` is from
`load_study`, and seeds are `grid.base_seed .+ (0:n_seeds-1)` for the random screens. The runs are, in this order:
- `tier1_runs` (154 runs);
- `universe_runs` at K = 13 with every `grid.universe_screens` entry (182 runs);
- `universe_runs` at K = 5 and K = 34 with screens `:global_gamma` and `:random_global` only (168 runs);
- `fixed_list_runs` (28 runs).
"""
function synthetic_runs(s::MyStudySettings, grid; n_seeds::Int=5)::Vector{MyStudyRun}
    g = merge(grid, (n_seeds=n_seeds,))
    out = MyStudyRun[]
    append!(out, tier1_runs(s, g))
    append!(out, universe_runs(s, merge(g, (K=[13],))))
    append!(out, universe_runs(s, merge(g, (K=[5, 34], universe_screens=[:global_gamma, :random_global]))))
    append!(out, fixed_list_runs(s, g))
    return out
end

"""
    run_synthetic_path(md::MyMarketData, calib::MyCalibration, runs::Vector{MyStudyRun})::DataFrame

Every run in `runs` on the market `md` with the calibration `calib`, scored as `run_and_save_study` scores it, with
no file written. One `MyStudyContext` is built per distinct `context_key` and shared by its runs; a context that
fails to build fails its runs. The runs are threaded with `Threads.@threads :dynamic`, with BLAS on one thread when
`Threads.nthreads() > 1`.

Returns one row per run, sorted by `hash` (`study_hash(run; fingerprint=data_fingerprint(md, calib))`, as
`run_study_grid` hashes it). The columns are those of `rebuild_study_summary`, then `error`. A run that finishes has
`status = "ok"` and `error` missing. A run that throws has `status = "error"`, the exception text in `error`, its
`hash` and `label`, and every other column missing.
"""
function run_synthetic_path(md::MyMarketData, calib::MyCalibration, runs::Vector{MyStudyRun})::DataFrame
    fp = data_fingerprint(md, calib)
    # one context per distinct context key; a context that fails to build is stored as its exception
    ctxs = Dict{Any,Any}()
    for r in runs
        k = context_key(r.settings)
        if !haskey(ctxs, k)
            try
                ctxs[k] = build_context(md, calib, r.settings)
            catch err
                ctxs[k] = err
            end
        end
    end
    # the runs, threaded; run k fills slot k
    if Threads.nthreads() > 1
        BLAS.set_num_threads(1)
    end
    rows = Vector{Dict{Symbol,Any}}(undef, length(runs))
    Threads.@threads :dynamic for k in eachindex(runs)
        r = runs[k]
        h = study_hash(r; fingerprint=fp)
        try
            c = ctxs[context_key(r.settings)]
            if c isa Exception
                throw(c)
            end
            out = _score_study(c, r)
            row = _study_row(study_key(r), r.label, out.scorecard, out.stats, out.wall)
            row[:hash] = h
            row[:status] = "ok"
            rows[k] = row
        catch err
            rows[k] = Dict{Symbol,Any}(:hash => h, :label => r.label, :status => "error",
                :error => sprint(showerror, err))
        end
    end
    # sorted by hash, in the columns of rebuild_study_summary plus error
    hashes = String[]
    for row in rows
        push!(hashes, row[:hash])
    end
    rows = rows[sortperm(hashes)]
    df = DataFrame()
    for c in vcat(_STUDY_SUMMARY_COLUMNS, [:error])
        df[!, c] = Any[get(row, c, missing) for row in rows]
    end
    for c in names(df)
        df[!, c] = identity.(df[!, c])
    end
    return df
end

"""
    path_done(dir::AbstractString, p::Int)::Bool

True when `dir/path_<p>.csv` exists, that is, when `write_path_files` has finished path `p`.
"""
function path_done(dir::AbstractString, p::Int)::Bool
    return isfile(joinpath(dir, "path_$(p).csv"))
end

"""
    write_path_files(dir::AbstractString, p::Int, rows, bench, meta)::Nothing

Write the tables of path `p` to `dir` (created if missing): `bench` to `bench_<p>.csv`, `meta` to `meta_<p>.csv`,
then `rows` to `path_<p>.csv` last, so `path_done(dir, p)` is true only once all three are written. Each table is
anything `CSV.write` accepts. Each file is written to `<name>.tmp` first and then moved into place, so a reader never
sees a partial file.
"""
function write_path_files(dir::AbstractString, p::Int, rows, bench, meta)::Nothing
    mkpath(dir)
    for (stem, table) in (("bench", bench), ("meta", meta), ("path", rows))
        final = joinpath(dir, "$(stem)_$(p).csv")
        tmp = final * ".tmp"
        CSV.write(tmp, table)
        mv(tmp, final; force=true)
    end
    return nothing
end
