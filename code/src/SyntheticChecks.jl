# Checks V1–V9 of the synthetic market (spec §8): synthetic paths, each recalibrated as the study would recalibrate
# it, against the real 2014–2024 market.

"""
    SYNTH_ACF_LAGS

Lags (trading days) of the abs-growth autocorrelations in check V4: 1, 5 and 20.
"""
const SYNTH_ACF_LAGS = [1, 5, 20]

"""
    ew_rebalance_gap(G::AbstractMatrix; Δt=1/252)::Float64

Growth (yr⁻¹) of the equal-weight portfolio rebalanced every day minus the growth of the equal-weight portfolio
bought once and held, over the rows of the growth matrix `G` (yr⁻¹, days × names), with Δt in yr.
Rebalanced: Σₜ log(meanᵢ exp(G[t, i]·Δt)). Buy-and-hold: log(meanᵢ exp(Σₜ G[t, i]·Δt)). Both are divided by
(rows·Δt). Throws `ArgumentError` if `G` has no rows or no columns.
"""
function ew_rebalance_gap(G::AbstractMatrix; Δt::Real=1/252)::Float64
    T, N = size(G)
    if T == 0 || N == 0
        throw(ArgumentError("G must have at least one row and one column; got size $(size(G))"))
    end
    # rebalanced every day: the log of the mean one-day growth factor, summed over the days
    reb = 0.0
    for t in 1:T
        total = 0.0
        for i in 1:N
            total += exp(G[t, i] * Δt)
        end
        reb += log(total / N)
    end
    # bought once and held: the log of the mean whole-window growth factor
    total = 0.0
    for i in 1:N
        x = 0.0
        for t in 1:T
            x += G[t, i] * Δt
        end
        total += exp(x)
    end
    bh = log(total / N)
    return (reb - bh) / (T * Δt)
end

"""
    abs_acf(x::AbstractVector, lags)::Vector{Float64}

Autocorrelation (dimensionless) of abs.(x) at each lag L in `lags` (trading days):
`cor(abs.(x[1:end-L]), abs.(x[1+L:end]))`. Throws `ArgumentError` for a lag below 1 or a lag that leaves fewer than
2 pairs.
"""
function abs_acf(x::AbstractVector, lags)::Vector{Float64}
    a = abs.(x)
    n = length(a)
    out = zeros(Float64, length(lags))
    for (k, L) in enumerate(lags)
        if L < 1 || n - L < 2
            throw(ArgumentError("lag $(L) must be at least 1 and leave 2 or more pairs in $(n) values"))
        end
        out[k] = cor(a[1:(n - L)], a[(1 + L):n])
    end
    return out
end

"""
    mean_pairwise_corr(R::AbstractMatrix)::Float64

Mean (dimensionless) of the off-diagonal entries of `cor(R)`, the correlations between the columns of `R`.
Throws `ArgumentError` if `R` has fewer than 2 columns.
"""
function mean_pairwise_corr(R::AbstractMatrix)::Float64
    k = size(R, 2)
    if k < 2
        throw(ArgumentError("R needs at least 2 columns for a pairwise correlation; got $(k)"))
    end
    C = cor(R)
    total = 0.0
    for i in 1:k
        for j in 1:k
            if i != j
                total += C[i, j]
            end
        end
    end
    return total / (k * (k - 1))
end

"""
    residual_correlations(R::AbstractMatrix, sector_index::Vector{Int}, S::Int)::NamedTuple{(:sector_corr, :cross_corr)}

Residual co-movement for check V5, from the SIM residuals `R` (yr⁻¹, days × names). `sector_corr[s]` is the
`mean_pairwise_corr` of the columns whose `sector_index` is s, for s = 1…S, and `cross_corr` is the mean correlation
over the pairs of names in different sectors (both dimensionless). Throws `ArgumentError` if `sector_index` does not
have one entry per column or no pair of names spans two sectors.
"""
function residual_correlations(R::AbstractMatrix, sector_index::Vector{Int}, S::Int)::NamedTuple{(:sector_corr, :cross_corr)}
    N = size(R, 2)
    if length(sector_index) != N
        throw(ArgumentError("R has $(N) columns but sector_index has $(length(sector_index)) entries"))
    end
    # within each sector
    sector_corr = zeros(Float64, S)
    for s in 1:S
        members = findall(==(s), sector_index)
        sector_corr[s] = mean_pairwise_corr(R[:, members])
    end
    # across sectors: every pair of names in different sectors
    C = cor(R)
    total = 0.0
    n_pairs = 0
    for i in 1:N
        for j in (i + 1):N
            if sector_index[i] != sector_index[j]
                total += C[i, j]
                n_pairs += 1
            end
        end
    end
    if n_pairs == 0
        throw(ArgumentError("no pair of names spans two sectors"))
    end
    return (sector_corr=sector_corr, cross_corr=total / n_pairs)
end

"""
    abs_acf_matrix(G::AbstractMatrix)::Matrix{Float64}

`abs_acf` of each column of `G` (yr⁻¹, days × names) at `SYNTH_ACF_LAGS`: row i holds name i's autocorrelations
(dimensionless), one column per lag.
"""
function abs_acf_matrix(G::AbstractMatrix)::Matrix{Float64}
    N = size(G, 2)
    A = zeros(Float64, N, length(SYNTH_ACF_LAGS))
    for i in 1:N
        A[i, :] = abs_acf(view(G, :, i), SYNTH_ACF_LAGS)
    end
    return A
end

"""
    count_beta_le0(sim::MyOnlineSIM)::Int

Number of names (columns of `sim.β`) whose online β̂ (dimensionless) is ≤ 0 on at least one day.
"""
function count_beta_le0(sim::MyOnlineSIM)::Int
    n = 0
    for j in 1:size(sim.β, 2)
        if any(view(sim.β, :, j) .<= 0)
            n += 1
        end
    end
    return n
end

"""
    beta_le0_windows(md::MyMarketData, calib::MyCalibration; train=SYNTH_TRAIN,
        trade=(Date(2025, 1, 2), Date(2026, 9, 4)))::DataFrame

`count_beta_le0` of `online_sim(md, calib)` in each window of the training period `train` that is as long as the
trading window `trade`. Each window's online pass starts from `calib`, as the count over `trade` does. The trading
window holds L trading days (`trade[1]` through `trade[2]`). The windows hold L trading days each, follow one another
without overlap, start on the trading day after `train[1]` (`online_sim` needs a previous day) and end on or before
`train[2]`; days left over at the end are not used. Columns: `start` and `stop` (each window's first and last trading
day) and `n_beta_le0` (names). Throws `ArgumentError` if a date of `train` or `trade` is not a trading day of `md`,
or no window fits.
"""
function beta_le0_windows(md::MyMarketData, calib::MyCalibration; train::Tuple{Date,Date}=SYNTH_TRAIN,
        trade::Tuple{Date,Date}=(Date(2025, 1, 2), Date(2026, 9, 4)))::DataFrame
    L = index_of(md, trade[2]) - index_of(md, trade[1]) + 1
    first_row = index_of(md, train[1]) + 1
    last_row = index_of(md, train[2])
    out = DataFrame(start=Date[], stop=Date[], n_beta_le0=Int[])
    s = first_row
    while s + L - 1 <= last_row
        start = md.dates[s]
        stop = md.dates[s + L - 1]
        push!(out, (start, stop, count_beta_le0(online_sim(md, calib; start=start, stop=stop))))
        s += L
    end
    if nrow(out) == 0
        throw(ArgumentError("no window of $(L) trading days fits in $(train[1]) → $(train[2])"))
    end
    return out
end

"""
    alpha_correlation(md::MyMarketData, a::Tuple{Date,Date}, b::Tuple{Date,Date})::Float64

Correlation (dimensionless) across the names of `md` of each name's OLS α (yr⁻¹) from `fit_base_sim` on window `a`
against window `b`, the way `real_drift_targets` correlates the two halves of the training window.
"""
function alpha_correlation(md::MyMarketData, a::Tuple{Date,Date}, b::Tuple{Date,Date})::Float64
    fa = fit_base_sim(md, a[1], a[2])
    fb = fit_base_sim(md, b[1], b[2])
    αa = zeros(Float64, length(md.tickers))
    αb = zeros(Float64, length(md.tickers))
    for (j, t) in enumerate(md.tickers)
        αa[j] = fa.params[t].α
        αb[j] = fb.params[t].α
    end
    return cor(αa, αb)
end

"""
    real_reference(md_real::MyMarketData, calib_real::MyCalibration, gen::MySyntheticGenerator;
        trade=(Date(2025, 1, 2), Date(2026, 9, 4)))::NamedTuple

Everything real that the checks compare against. It is measured on the training rows w =
`growth_window(md_real.dates, SYNTH_TRAIN...)`, the same way `path_diagnostics` measures a synthetic path. Fields:
- `G` (yr⁻¹, days × names) and `g_m` (yr⁻¹): the real growth of the names and of SPY on w;
- `sectors` (`gen.sectors`), `sector_corr` (one per sector) and `cross_corr` (dimensionless): the mean pairwise
  correlation of the SIM residuals from `calib_real.base_sim`, within each sector and across sectors
  (`residual_correlations`, with the names grouped by `gen.sector_index`);
- `sd_β` (one per name) and `acf252` from `real_drift_targets(md_real, calib_real)`;
- `acf` (names × 3): each name's `abs_acf` at lags 1, 5 and 20 (trading days);
- `ew_gap` (yr⁻¹): `ew_rebalance_gap` of G;
- `spread` (yr⁻¹) and `corr` from `real_drift_targets`;
- `spy_growth` (yr⁻¹): log(S_stop/S_start)/((i1 − i0)/252), with S the SPY close on the first (row i0) and last
  (row i1) training day;
- `n_beta_le0`: the number of names whose β̂ from `online_sim(md_real, calib_real)` over the trading window `trade`
  is ≤ 0 on any day.
Throws `ArgumentError` if `md_real` and `gen` differ in dates or tickers, or `calib_real` was not fitted on
`SYNTH_TRAIN`.
"""
function real_reference(md_real::MyMarketData, calib_real::MyCalibration, gen::MySyntheticGenerator;
        trade::Tuple{Date,Date}=(Date(2025, 1, 2), Date(2026, 9, 4)))::NamedTuple
    if md_real.dates != gen.dates || md_real.tickers != gen.tickers
        throw(ArgumentError("md_real and gen must share dates and tickers"))
    end
    if calib_real.train_start != SYNTH_TRAIN[1] || calib_real.train_stop != SYNTH_TRAIN[2]
        throw(ArgumentError("calib_real covers $(calib_real.train_start) → $(calib_real.train_stop); expected $(SYNTH_TRAIN[1]) → $(SYNTH_TRAIN[2])"))
    end
    # real training growth
    w = growth_window(md_real.dates, SYNTH_TRAIN[1], SYNTH_TRAIN[2])
    G = growth_rates(md_real.close)[w, :]
    g_m = growth_rates(md_real.spy_close)[w]
    # residual co-movement within and across sectors
    R = sim_residuals(md_real, calib_real.base_sim, SYNTH_TRAIN[1], SYNTH_TRAIN[2])
    rc = residual_correlations(R, gen.sector_index, length(gen.sectors))
    # β̂ drift and α targets
    targets = real_drift_targets(md_real, calib_real)
    # SPY growth over the training window
    i0 = index_of(md_real, SYNTH_TRAIN[1])
    i1 = index_of(md_real, SYNTH_TRAIN[2])
    spy_growth = log(md_real.spy_close[i1] / md_real.spy_close[i0]) / ((i1 - i0) / 252)
    # names whose online β̂ reaches zero in the trading window
    sim = online_sim(md_real, calib_real; start=trade[1], stop=trade[2])
    return (G=G, g_m=g_m, sectors=gen.sectors, sector_corr=rc.sector_corr, cross_corr=rc.cross_corr,
        sd_β=targets.sd_β, acf252=targets.acf252, acf=abs_acf_matrix(G), ew_gap=ew_rebalance_gap(G),
        spread=targets.spread, corr=targets.corr, spy_growth=spy_growth, n_beta_le0=count_beta_le0(sim))
end

"""
    path_diagnostics(gen::MySyntheticGenerator, sm::NamedTuple, calib_syn::MyCalibration, ref::NamedTuple;
        trade=(Date(2025, 1, 2), Date(2026, 9, 4)))::NamedTuple

Check diagnostics of one synthetic path `sm` (from `synthetic_market`), recalibrated as `calib_syn` (from `calibrate`
on `sm.md` over `SYNTH_TRAIN`). `ref` is from `real_reference`. Everything is on the training rows w =
`growth_window(gen.dates, SYNTH_TRAIN...)` unless noted. G and g_m are the path's name and SPY growth (yr⁻¹) on w,
and E = `sm.draws.E` holds each name's raw JumpHMM draw (yr⁻¹). Fields:
- V2: `ew_gap` (yr⁻¹), the `ew_rebalance_gap` of G;
- V3, one per name: `var_ratio_draw` = var(G[:, i]) ÷ var(E[w, i]), `var_ratio_real` = var(G[:, i]) ÷ `gen.σ²_gen[i]`,
  and `β_diff` = the OLS β of G[:, i] on g_m minus mean(`sm.β_eff[w, i]`);
- V4, names × 3 (lags 1, 5, 20): `acf_syn` (`abs_acf` of G[:, i]), `acf_draw` (of E[w, i]) and `acf_real` (`ref.acf`);
- V5: `sector_corr` (one per sector) and `cross_corr`, measured as `real_reference` measures them, from the SIM
  residuals of `calib_syn.base_sim`;
- V6: `ewls_sd` (the `sd_β` per name) and `acf252` from `real_drift_targets(sm.md, calib_syn)`, and `n_beta_le0`,
  the number of names whose β̂ from `online_sim(sm.md, calib_syn)` over the trading window `trade` is ≤ 0 on any day;
- V9: `α_spread` (yr⁻¹) and `α_corr`, the `spread` and `corr` of `real_drift_targets(sm.md, calib_syn)`;
- `clip_share`: `sm.clip_share`.
Throws `ArgumentError` if `calib_syn` was not fitted on `SYNTH_TRAIN`.
"""
function path_diagnostics(gen::MySyntheticGenerator, sm::NamedTuple, calib_syn::MyCalibration, ref::NamedTuple;
        trade::Tuple{Date,Date}=(Date(2025, 1, 2), Date(2026, 9, 4)))::NamedTuple
    if calib_syn.train_start != SYNTH_TRAIN[1] || calib_syn.train_stop != SYNTH_TRAIN[2]
        throw(ArgumentError("calib_syn covers $(calib_syn.train_start) → $(calib_syn.train_stop); expected $(SYNTH_TRAIN[1]) → $(SYNTH_TRAIN[2])"))
    end
    md = sm.md
    N = length(gen.tickers)
    # synthetic training growth and the raw draws on the same rows
    w = growth_window(gen.dates, SYNTH_TRAIN[1], SYNTH_TRAIN[2])
    G = growth_rates(md.close)[w, :]
    g_m = growth_rates(md.spy_close)[w]
    E = sm.draws.E[w, :]
    # V3: variance against the name's own draw and against the real variance, and β against the mean β_eff
    var_ratio_draw = zeros(Float64, N)
    var_ratio_real = zeros(Float64, N)
    β_diff = zeros(Float64, N)
    for i in 1:N
        v = var(view(G, :, i))
        var_ratio_draw[i] = v / var(view(E, :, i))
        var_ratio_real[i] = v / gen.σ²_gen[i]
        β_diff[i] = estimate_sim(g_m, G[:, i]).β - mean(view(sm.β_eff, w, i))
    end
    # V5: residual co-movement under the path's own calibration
    R = sim_residuals(md, calib_syn.base_sim, SYNTH_TRAIN[1], SYNTH_TRAIN[2])
    rc = residual_correlations(R, gen.sector_index, length(gen.sectors))
    # V6 and V9: β̂ drift and α, measured as the real targets are measured
    targets = real_drift_targets(md, calib_syn)
    # V6: names whose online β̂ reaches zero in the trading window
    sim = online_sim(md, calib_syn; start=trade[1], stop=trade[2])
    return (ew_gap=ew_rebalance_gap(G), var_ratio_draw=var_ratio_draw, var_ratio_real=var_ratio_real, β_diff=β_diff,
        acf_syn=abs_acf_matrix(G), acf_draw=abs_acf_matrix(E), acf_real=ref.acf,
        sector_corr=rc.sector_corr, cross_corr=rc.cross_corr,
        ewls_sd=targets.sd_β, acf252=targets.acf252, n_beta_le0=count_beta_le0(sim),
        α_spread=targets.spread, α_corr=targets.corr, clip_share=sm.clip_share)
end

"""
    name_medians(X::AbstractMatrix)::Vector{Float64}

Each name's median over paths: entry i is the median of column i of `X` (paths × names).
"""
function name_medians(X::AbstractMatrix)::Vector{Float64}
    out = zeros(Float64, size(X, 2))
    for i in 1:size(X, 2)
        out[i] = median(view(X, :, i))
    end
    return out
end

"""
    check_table(diags::Vector, ref::NamedTuple, v1::NamedTuple)::DataFrame

The check table for the path diagnostics `diags` (from `path_diagnostics`) against the real reference `ref` (from
`real_reference`). `v1` holds `median_growth`, the median SPY training growth (yr⁻¹) over `n_paths` paths.
Columns: `check`, `statistic`, `synthetic`, `reference`, `threshold` and `pass` (`missing` for a row that is
reported but not gated).

A number with one value per path is summarized by its median over paths. A number with one value per name and path
is summarized by each name's median over paths, then the median of those over names.

Gates:
- V1: |median growth − real SPY growth| ≤ 0.02;
- V2: |median ew_gap − real gap| ≤ 0.02;
- V3: median `var_ratio_draw` within [0.95, 1.05], and median |β_diff| ≤ 0.05;
- V4: |median(acf_syn − acf_draw)| ≤ 0.05 at each lag;
- V5: |median sector_corr − real| ≤ 0.05 for every sector;
- V6: median over names of (median-over-paths `ewls_sd` ÷ `sd_β`) within [0.9, 1.1];
- V9: |median α_spread − real spread| ≤ 0.01 (yr⁻¹) and |median α_corr − real corr| ≤ 0.05.
Reported only: V3 median `var_ratio_real`; V4 median(acf_syn − acf_real) at each lag; V5 median `cross_corr`;
V6 median `acf252`; the share of paths with `n_beta_le0` ≥ 1, next to the real market's own yes/no (1.0 if its
`n_beta_le0` ≥ 1, else 0.0); and the median over paths of `n_beta_le0`, with its range, next to the real count.
Throws `ArgumentError` for an empty `diags`.
"""
function check_table(diags::Vector, ref::NamedTuple, v1::NamedTuple)::DataFrame
    P = length(diags)
    if P == 0
        throw(ArgumentError("diags is empty; the checks need at least one path"))
    end
    N = length(ref.sd_β)
    S = length(ref.sectors)
    L = length(SYNTH_ACF_LAGS)
    # gather every path's values: one entry per path, or paths × names
    ew_gap = zeros(Float64, P)
    cross_corr = zeros(Float64, P)
    acf252 = zeros(Float64, P)
    α_spread = zeros(Float64, P)
    α_corr = zeros(Float64, P)
    sector_corr = zeros(Float64, P, S)
    var_draw = zeros(Float64, P, N)
    var_real = zeros(Float64, P, N)
    β_abs = zeros(Float64, P, N)
    sd_ratio = zeros(Float64, P, N)
    acf_vs_draw = zeros(Float64, P, N, L)
    acf_vs_real = zeros(Float64, P, N, L)
    n_le0 = zeros(Int, P)
    n_with_le0 = 0
    for p in 1:P
        d = diags[p]
        ew_gap[p] = d.ew_gap
        cross_corr[p] = d.cross_corr
        acf252[p] = d.acf252
        α_spread[p] = d.α_spread
        α_corr[p] = d.α_corr
        for s in 1:S
            sector_corr[p, s] = d.sector_corr[s]
        end
        for i in 1:N
            var_draw[p, i] = d.var_ratio_draw[i]
            var_real[p, i] = d.var_ratio_real[i]
            β_abs[p, i] = abs(d.β_diff[i])
            sd_ratio[p, i] = d.ewls_sd[i] / ref.sd_β[i]
            for l in 1:L
                acf_vs_draw[p, i, l] = d.acf_syn[i, l] - d.acf_draw[i, l]
                acf_vs_real[p, i, l] = d.acf_syn[i, l] - d.acf_real[i, l]
            end
        end
        n_le0[p] = d.n_beta_le0
        if d.n_beta_le0 >= 1
            n_with_le0 += 1
        end
    end
    # one row per statistic: (check, statistic, synthetic, reference, threshold, pass)
    t = DataFrame(check=String[], statistic=String[], synthetic=Float64[], reference=Float64[], threshold=String[],
        pass=Union{Missing,Bool}[])
    # V1: SPY growth over the training window
    x = v1.median_growth
    push!(t, ("V1", "SPY growth 2014–2024, median over $(v1.n_paths) paths (yr⁻¹)", x, ref.spy_growth,
        "within 0.02 of reference", abs(x - ref.spy_growth) <= 0.02))
    # V2: equal-weight rebalancing gap
    x = median(ew_gap)
    push!(t, ("V2", "equal-weight growth, rebalanced daily minus bought and held (yr⁻¹)", x, ref.ew_gap,
        "within 0.02 of reference", abs(x - ref.ew_gap) <= 0.02))
    # V3: variance and β of each name after composition
    x = median(name_medians(var_draw))
    push!(t, ("V3", "variance ÷ variance of the name's own draw", x, 1.0, "in [0.95, 1.05]",
        0.95 <= x <= 1.05))
    x = median(name_medians(β_abs))
    push!(t, ("V3", "absolute gap between the OLS β and the mean composed β", x, 0.0, "≤ 0.05", x <= 0.05))
    x = median(name_medians(var_real))
    push!(t, ("V3", "variance ÷ real variance", x, 1.0, "report only", missing))
    # V4: clustering of abs growth, against the name's own draw (gated) and against real (reported)
    for l in 1:L
        x = median(name_medians(acf_vs_draw[:, :, l]))
        push!(t, ("V4", "abs-growth autocorrelation at lag $(SYNTH_ACF_LAGS[l]), synthetic − own draw", x, 0.0,
            "within 0.05 of reference", abs(x) <= 0.05))
    end
    for l in 1:L
        x = median(name_medians(acf_vs_real[:, :, l]))
        push!(t, ("V4", "abs-growth autocorrelation at lag $(SYNTH_ACF_LAGS[l]), synthetic − real", x, 0.0,
            "report only", missing))
    end
    # V5: residual co-movement within each sector (gated) and across sectors (reported)
    for s in 1:S
        x = median(sector_corr[:, s])
        push!(t, ("V5", "mean residual correlation within $(ref.sectors[s])", x, ref.sector_corr[s],
            "within 0.05 of reference", abs(x - ref.sector_corr[s]) <= 0.05))
    end
    push!(t, ("V5", "mean residual correlation across sectors", median(cross_corr), ref.cross_corr, "report only",
        missing))
    # V6: β̂ drift
    x = median(name_medians(sd_ratio))
    push!(t, ("V6", "SD of the online β̂ ÷ the real SD", x, 1.0, "in [0.9, 1.1]", 0.9 <= x <= 1.1))
    push!(t, ("V6", "lag-252 autocorrelation of the online β̂", median(acf252), ref.acf252, "report only", missing))
    # the real market's own yes/no: does it have a name whose β̂ ≤ 0 on some day?
    if ref.n_beta_le0 >= 1
        real_has_le0 = 1.0
    else
        real_has_le0 = 0.0
    end
    push!(t, ("V6", "share of paths with a name whose online β̂ ≤ 0 on some day in 2025–26 " *
        "(reference: the real market's own yes/no, 1 = yes)", n_with_le0 / P, real_has_le0, "report only", missing))
    push!(t, ("V6", "names whose online β̂ ≤ 0 on some day in 2025–26, median over paths " *
        "(range $(minimum(n_le0))–$(maximum(n_le0)))", median(n_le0), Float64(ref.n_beta_le0), "report only", missing))
    # V9: α spread and persistence
    x = median(α_spread)
    push!(t, ("V9", "cross-name SD of OLS α, 2014–2024 (yr⁻¹)", x, ref.spread, "within 0.01 of reference",
        abs(x - ref.spread) <= 0.01))
    x = median(α_corr)
    push!(t, ("V9", "correlation of OLS α, 2014–2019 vs 2020–2024", x, ref.corr, "within 0.05 of reference",
        abs(x - ref.corr) <= 0.05))
    return t
end
