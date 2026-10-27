# Fitting the synthetic-market generator: JumpHMM refits and the β and α drift calibration.

"""
    SYNTH_HMM

JumpHMM settings for every name and sector fit: `N` (states), `ν` (Student-t degrees of freedom), `rf` (yr⁻¹),
`dt` (yr), the tuning grids `ϵ_range` (jump probability per trading day) and `λ_range` (mean jump run length,
trading days), and `n_paths` (simulated paths per tuning grid point).
"""
const SYNTH_HMM = (N=50, ν=5.0, rf=0.0, dt=1/252, ϵ_range=range(1e-4, 2.5e-2, length=10),
    λ_range=range(10.0, 120.0, length=10), n_paths=100)

"""
    SYNTH_TRAIN

The real training window (first and last trading day), 2014-01-03 → 2024-12-31.
"""
const SYNTH_TRAIN = (Date(2014, 1, 3), Date(2024, 12, 31))

"""
    SYNTH_ALPHA_WINDOWS

The three windows (start, stop) of the α targets: the full training window 2014-01-03 → 2024-12-31, its first
half 2014-01-03 → 2019-12-31 and its second half 2020-01-02 → 2024-12-31.
"""
const SYNTH_ALPHA_WINDOWS = ((Date(2014, 1, 3), Date(2024, 12, 31)), (Date(2014, 1, 3), Date(2019, 12, 31)),
    (Date(2020, 1, 2), Date(2024, 12, 31)))

"""
    fit_name_model(prices::AbstractVector{<:Real}; seed::Int, N::Int=SYNTH_HMM.N, tune::Bool=true)

JumpHMM model of one price series (USD/share). `JumpHMM.fit` with `N` states and the `SYNTH_HMM` settings
(ν, rf, dt), then, if `tune` is true, `JumpHMM.tune` over the `SYNTH_HMM` grids with `n_paths` and `seed`.
The model's draws are growth rates (yr⁻¹), since rf = 0.
"""
function fit_name_model(prices::AbstractVector{<:Real}; seed::Int, N::Int=SYNTH_HMM.N, tune::Bool=true)
    x = Float64.(prices)
    model = JumpHMM.fit(JumpHMM.JumpHiddenMarkovModel, x; rf=SYNTH_HMM.rf, N=N, ν=SYNTH_HMM.ν, dt=SYNTH_HMM.dt)
    if tune
        model = JumpHMM.tune(model, x; ϵ_range=SYNTH_HMM.ϵ_range, λ_range=SYNTH_HMM.λ_range,
            n_paths=SYNTH_HMM.n_paths, seed=seed)
    end
    return model
end

"""
    model_sd_ratio(model, g_real::AbstractVector; seed::Int, n_steps::Int=100_000)::Float64

SD of one `n_steps`-day draw from the JumpHMM `model` (seeded with `seed`) divided by the SD of the real growth
rates `g_real` (both yr⁻¹). A ratio near 1 means the model reproduces the real day-to-day spread.
"""
function model_sd_ratio(model, g_real::AbstractVector; seed::Int, n_steps::Int=100_000)::Float64
    draw = JumpHMM.simulate(model, n_steps; n_paths=1, seed=seed).paths[1].observations
    return std(draw) / std(g_real)
end

"""
    ewls_beta_path(g_i, g_m, prior::MySIMParameterEstimate, m_m, v_m; half_life=63.0, prior_weight=63.0)::Vector{Float64}

β̂ (dimensionless) after each EWLS update on the growth rates `g_i` and `g_m` (yr⁻¹), the same recursion as
`online_sim`. The state starts from `prior` (α yr⁻¹, β, σ_ε yr⁻¹) with market mean `m_m` (yr⁻¹) and population
variance `v_m` (yr⁻²); `half_life` and `prior_weight` are in trading days. Entry t is β̂ after update t.
"""
function ewls_beta_path(g_i::AbstractVector, g_m::AbstractVector, prior::MySIMParameterEstimate, m_m::Real, v_m::Real;
                        half_life::Real=63.0, prior_weight::Real=63.0)::Vector{Float64}
    if length(g_i) != length(g_m)
        throw(ArgumentError("g_i has $(length(g_i)) days but g_m has $(length(g_m))"))
    end
    s = ewls_init(prior.α, prior.β, prior.σ_ε; half_life=half_life, prior_weight=prior_weight,
        market_mean=m_m, market_variance=v_m)
    b = zeros(Float64, length(g_i))
    for t in 1:length(g_i)
        ewls_update!(s, g_i[t], g_m[t])
        b[t] = s.β
    end
    return b
end

"""
    beta_path_stats(b::AbstractVector)::NamedTuple{(:sd, :acf252)}

SD of the β̂ path `b` (dimensionless) and its lag-252 autocorrelation `cor(b[1:end-252], b[253:end])`.
Throws `ArgumentError` if `b` has 254 or fewer days.
"""
function beta_path_stats(b::AbstractVector)::NamedTuple{(:sd, :acf252)}
    if length(b) <= 254
        throw(ArgumentError("a lag-252 autocorrelation needs more than 254 days; got $(length(b))"))
    end
    return (sd=std(b), acf252=cor(b[1:end-252], b[253:end]))
end

"""
    real_drift_targets(md::MyMarketData, calib::MyCalibration)::NamedTuple{(:sd_β, :acf252, :α_full, :α_h1, :α_h2, :spread, :corr)}

Drift targets from the real data, one entry per ticker of `md`:
- `sd_β`: SD of each name's β̂ path from `online_sim(md, calib)` over the training window (from the trading day
  after 2014-01-03, which is 2014-01-06, to 2024-12-31), and `acf252`: the median over names of its lag-252
  autocorrelation;
- `α_full`, `α_h1`, `α_h2` (yr⁻¹): each name's OLS α from `fit_base_sim` on the three `SYNTH_ALPHA_WINDOWS`;
- `spread = std(α_full)` (yr⁻¹) and `corr = cor(α_h1, α_h2)` (dimensionless).
"""
function real_drift_targets(md::MyMarketData, calib::MyCalibration)::NamedTuple{(:sd_β, :acf252, :α_full, :α_h1, :α_h2, :spread, :corr)}
    # β̂ paths of the study's own EWLS over the training window
    start = md.dates[index_of(md, SYNTH_TRAIN[1]) + 1]
    sim = online_sim(md, calib; start=start, stop=SYNTH_TRAIN[2])
    N = length(md.tickers)
    sd_β = zeros(Float64, N)
    acf = zeros(Float64, N)
    for j in 1:N
        st = beta_path_stats(sim.β[:, j])
        sd_β[j] = st.sd
        acf[j] = st.acf252
    end
    # OLS α in the full window and in each half
    α = [zeros(Float64, N) for _ in 1:3]
    for w in 1:3
        fit = fit_base_sim(md, SYNTH_ALPHA_WINDOWS[w][1], SYNTH_ALPHA_WINDOWS[w][2])
        for j in 1:N
            α[w][j] = fit.params[md.tickers[j]].α
        end
    end
    return (sd_β=sd_β, acf252=median(acf), α_full=α[1], α_h1=α[2], α_h2=α[3], spread=std(α[1]), corr=cor(α[2], α[3]))
end

"""
    name_growth(gen::MySyntheticGenerator, d::NamedTuple, i::Int, α_c::Real, k_α::Real, h_β::Real, k_β::Real)::Vector{Float64}

Growth path (yr⁻¹, n days) of ticker `i` on the draws `d`, composed as `synthetic_growth` does but with the α
center `α_c` (yr⁻¹), the α drift scale `k_α` (yr⁻¹) and the β drift half-life `h_β` (trading days) and scale `k_β`
given here. Every other setting comes from `gen`.
"""
function name_growth(gen::MySyntheticGenerator, d::NamedTuple, i::Int, α_c::Real, k_α::Real, h_β::Real, k_β::Real)::Vector{Float64}
    s = gen.sector_index[i]
    α = α_c .+ drift_path(d.Zα[:, i], gen.h_α, k_α)
    β = gen.β_center[i] .+ drift_path(d.Zβ[:, i], h_β, k_β)
    out = compose_name(α, β, gen.δ[i], d.g_m, d.F[:, s], d.E[:, i], gen.σ²_m, gen.σ²_s[s], gen.σ²_gen[i], gen.r²[i];
        ε_center=gen.ε_center[i], β²_mean=gen.β_center[i]^2 + drift_variance(h_β, k_β))
    return out.g
end

"""
    name_bank_beta_stats(gen::MySyntheticGenerator, bank::Vector, i::Int, rows::UnitRange{Int}, h_β::Real, k_β::Real)::NamedTuple{(:sd, :acf252)}

Mean over the bank paths of ticker `i`'s β̂ path statistics (`beta_path_stats`) on the growth rows `rows`, with β
drift half-life `h_β` (trading days) and scale `k_β`. On each path: compose the name with `gen`'s α settings, fit
the EWLS prior with `estimate_sim` on `rows`, take the market mean (yr⁻¹) and population variance (yr⁻²) of the
path's g_m on `rows`, and run `ewls_beta_path` over `rows`.
"""
function name_bank_beta_stats(gen::MySyntheticGenerator, bank::Vector, i::Int, rows::UnitRange{Int}, h_β::Real, k_β::Real)::NamedTuple{(:sd, :acf252)}
    sd = 0.0
    acf = 0.0
    for d in bank
        g = name_growth(gen, d, i, gen.α_center[i], gen.k_α, h_β, k_β)
        g_i = g[rows]
        g_m = d.g_m[rows]
        prior = estimate_sim(g_m, g_i)
        b = ewls_beta_path(g_i, g_m, prior, mean(g_m), var(g_m; corrected=false))
        st = beta_path_stats(b)
        sd += st.sd / length(bank)
        acf += st.acf252 / length(bank)
    end
    return (sd=sd, acf252=acf)
end

"""
    bank_beta_stats(gen::MySyntheticGenerator, bank::Vector; train=SYNTH_TRAIN)::NamedTuple{(:sd_β, :acf252)}

β̂ drift statistics of `gen` on the bank of draws `bank` (from `synthetic_draws`), measured as `real_drift_targets`
measures the real data. On each path, each name is composed, its EWLS prior is fitted with `estimate_sim` on the
training rows `growth_window(gen.dates, train...)`, the market moments come from the path's g_m on those rows, and
`ewls_beta_path` runs over them. `sd_β` is each name's SD of β̂ (dimensionless), averaged over the bank, and
`acf252` is the median over names of each name's bank-mean lag-252 autocorrelation.
"""
function bank_beta_stats(gen::MySyntheticGenerator, bank::Vector; train=SYNTH_TRAIN)::NamedTuple{(:sd_β, :acf252)}
    rows = growth_window(gen.dates, train[1], train[2])
    N = length(gen.tickers)
    sd_β = zeros(Float64, N)
    acf = zeros(Float64, N)
    Threads.@threads for i in 1:N
        st = name_bank_beta_stats(gen, bank, i, rows, gen.h_β, gen.k_β[i])
        sd_β[i] = st.sd
        acf[i] = st.acf252
    end
    return (sd_β=sd_β, acf252=median(acf))
end

"""
    fit_beta_drift(gen0::MySyntheticGenerator, bank::Vector, targets; h_grid=[63.0, 126.0, 252.0, 504.0, 1008.0, 2016.0],
        sd_max=1.5, train=SYNTH_TRAIN)::NamedTuple{(:h_β, :k_β, :table)}

Fit one β drift half-life `h_β` (trading days) and a drift scale k per name, so that `bank_beta_stats` on the bank
reproduces `targets.sd_β` (one per name) and `targets.acf252`. α is held at `gen0.α_center` with k_α = 0.

For each h in `h_grid` and each name (threaded): if the bank-mean SD of β̂ at k = 0 already reaches the name's
target, k = 0. Otherwise bisect, with 30 halvings, on the stationary drift SD σₓ ∈ [0, `sd_max`], with
k = σₓ/√(a/(2−a)) and a = `ema_rate(h)`. The bank is reused at every candidate (common random numbers). Then pick
the h whose median over names of the bank-mean acf252 is closest to `targets.acf252`.

`k_β` holds the k per name at that h. `table` (a DataFrame) has one row per h: `h`, `k_median`, `zero_share` (share
of names with k = 0) and `acf252` (median synthetic acf252).
"""
function fit_beta_drift(gen0::MySyntheticGenerator, bank::Vector, targets; h_grid=[63.0, 126.0, 252.0, 504.0, 1008.0, 2016.0],
                        sd_max=1.5, train=SYNTH_TRAIN)::NamedTuple{(:h_β, :k_β, :table)}
    N = length(gen0.tickers)
    if length(targets.sd_β) != N
        throw(ArgumentError("targets.sd_β has $(length(targets.sd_β)) entries but the generator has $(N) tickers"))
    end
    gen = with_generator(gen0; k_α=0.0)
    rows = growth_window(gen.dates, train[1], train[2])
    H = length(h_grid)
    K = zeros(Float64, H, N)
    A = zeros(Float64, H, N)
    for q in 1:H
        h = h_grid[q]
        a = ema_rate(h)
        unit_sd = sqrt(a / (2 - a))
        Threads.@threads for i in 1:N
            target = targets.sd_β[i]
            st0 = name_bank_beta_stats(gen, bank, i, rows, h, 0.0)
            if st0.sd >= target
                # the estimation noise alone reaches the target: no drift
                K[q, i] = 0.0
                A[q, i] = st0.acf252
            else
                # bisect on the stationary drift SD
                lo = 0.0
                hi = Float64(sd_max)
                for _ in 1:30
                    mid = (lo + hi) / 2
                    st = name_bank_beta_stats(gen, bank, i, rows, h, mid / unit_sd)
                    if st.sd < target
                        lo = mid
                    else
                        hi = mid
                    end
                end
                k = ((lo + hi) / 2) / unit_sd
                K[q, i] = k
                A[q, i] = name_bank_beta_stats(gen, bank, i, rows, h, k).acf252
            end
        end
    end
    # pick the half-life whose median acf252 is closest to the target
    acf_median = zeros(Float64, H)
    for q in 1:H
        acf_median[q] = median(A[q, :])
    end
    best = argmin(abs.(acf_median .- targets.acf252))
    table = DataFrame(h=Float64.(h_grid), k_median=[median(K[q, :]) for q in 1:H],
        zero_share=[count(iszero, K[q, :]) / N for q in 1:H], acf252=acf_median)
    return (h_β=Float64(h_grid[best]), k_β=K[best, :], table=table)
end

"""
    alpha_parts(gen1::MySyntheticGenerator, d::NamedTuple, i::Int; windows=SYNTH_ALPHA_WINDOWS)::NamedTuple{(:rest, :unit)}

The two parts of ticker `i`'s OLS α̂ (yr⁻¹) in the three `windows` (full, first half, second half) on the draws
`d`. `rest` is α̂ of the name composed with α center 0 and no α drift (β drift as in `gen1`). `unit` is α̂ of the
unit α drift path `drift_path(Zα[:, i], h_α, 1.0)` alone. OLS is linear in gᵢ and α does not enter the clipping,
so a name with center c and drift scale k has α̂ = c + rest + k·unit in each window.
"""
function alpha_parts(gen1::MySyntheticGenerator, d::NamedTuple, i::Int; windows=SYNTH_ALPHA_WINDOWS)::NamedTuple{(:rest, :unit)}
    g = name_growth(gen1, d, i, 0.0, 0.0, gen1.h_β, gen1.k_β[i])
    u = drift_path(d.Zα[:, i], gen1.h_α, 1.0)
    rest = zeros(Float64, 3)
    unit = zeros(Float64, 3)
    for w in 1:3
        rows = growth_window(gen1.dates, windows[w][1], windows[w][2])
        rest[w] = estimate_sim(d.g_m[rows], g[rows]).α
        unit[w] = estimate_sim(d.g_m[rows], u[rows]).α
    end
    return (rest=rest, unit=unit)
end

"""
    fit_alpha_drift(gen1::MySyntheticGenerator, bank::Vector, targets; φ_grid=0.0:0.01:1.5, sdα_grid=0.0:0.0025:0.15,
        windows=SYNTH_ALPHA_WINDOWS)::NamedTuple{(:φ, :k_α, :spread, :corr, :loss)}

Grid fit of the α center pull φ (dimensionless) and the α drift scale `k_α` (yr⁻¹), so that the synthetic OLS α̂
reproduce `targets.spread` (std of the full-window α̂ across names, yr⁻¹) and `targets.corr` (correlation of the
two half-window α̂ across names). Name i's center is cᵢ(φ) = ā + φ(α̂ᵢ − ā), with α̂ᵢ = `targets.α_full[i]` and
ā their mean.

`alpha_parts` is computed once for every bank path and name; then α̂ᵢ,w = cᵢ(φ) + restᵢ,w + k·unitᵢ,w. At each grid
point (φ, σ_α), with k = σ_α/√(a/(2−a)) and a = `ema_rate(gen1.h_α)`, the spread and corr are taken per path and
averaged over paths. The loss is ((spread − targets.spread)/0.01)² + ((corr − targets.corr)/0.05)². Returns the
grid minimizer: `φ`, `k_α`, its `spread` (yr⁻¹), `corr` and `loss`. The caller sets `α_center` = c(φ) in the
generator. `sdα_grid=[0.0]` fits φ alone with no α drift.
"""
function fit_alpha_drift(gen1::MySyntheticGenerator, bank::Vector, targets; φ_grid=0.0:0.01:1.5, sdα_grid=0.0:0.0025:0.15,
                         windows=SYNTH_ALPHA_WINDOWS)::NamedTuple{(:φ, :k_α, :spread, :corr, :loss)}
    N = length(gen1.tickers)
    P = length(bank)
    if length(targets.α_full) != N
        throw(ArgumentError("targets.α_full has $(length(targets.α_full)) entries but the generator has $(N) tickers"))
    end
    ā = mean(targets.α_full)
    # α̂ parts for every path (rows: names, columns: windows)
    R = [zeros(Float64, N, 3) for _ in 1:P]
    U = [zeros(Float64, N, 3) for _ in 1:P]
    for p in 1:P
        Threads.@threads for i in 1:N
            parts = alpha_parts(gen1, bank[p], i; windows=windows)
            R[p][i, :] = parts.rest
            U[p][i, :] = parts.unit
        end
    end
    # grid search over (φ, σ_α)
    a = ema_rate(gen1.h_α)
    unit_sd = sqrt(a / (2 - a))
    α̂ = zeros(Float64, N, 3)
    best = (φ=NaN, k_α=NaN, spread=NaN, corr=NaN, loss=Inf)
    for φ in φ_grid
        for σ_α in sdα_grid
            k = σ_α / unit_sd
            spread = 0.0
            corr = 0.0
            for p in 1:P
                for w in 1:3
                    for i in 1:N
                        α̂[i, w] = ā + φ * (targets.α_full[i] - ā) + R[p][i, w] + k * U[p][i, w]
                    end
                end
                spread += std(view(α̂, :, 1)) / P
                corr += cor(view(α̂, :, 2), view(α̂, :, 3)) / P
            end
            loss = ((spread - targets.spread) / 0.01)^2 + ((corr - targets.corr) / 0.05)^2
            if loss < best.loss
                best = (φ=Float64(φ), k_α=k, spread=spread, corr=corr, loss=loss)
            end
        end
    end
    return best
end
