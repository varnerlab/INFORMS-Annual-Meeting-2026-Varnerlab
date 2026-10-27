# Synthetic-market stress test: design

Date: 2026-10-26. Status: builder check done and the user's two questions answered; awaiting approval.
Builds on `2026-10-20-screened-utility-panel-study-design.md` (the study) and its addenda. The study's strategies,
settings and code are used unchanged.

Items marked **(my call)** were chosen by Claude, not the user. They are defaults the user can overturn. Everything
listed in §3 was decided by the user in the 2026-10-26 conversation.

## 1. Why

Every result in the study comes from one real history, 2025-01-02 → 2026-09-04: about 420 trading days with one
sell-off (Feb–Apr 2025). The point of this work is to show how the strategies could perform if the market enters
regimes we have not seen yet. That we haven't seen a regime doesn't mean we won't.

We make many synthetic futures and run the whole study on each one as if it were real:
- the calibration,
- the daily α and β updates,
- γ,
- the ticker pickers,
- the re-plan triggers,
- the hedge,
- every utility.

Two sets of futures are used. A **typical set** shows the spread of outcomes. A **stress set** keeps only the futures
with the worst sell-offs.

**Success:** picker × utility tables, in the same layout as the real-run tables, built across many synthetic futures.
- One version for typical futures and one for the worst sell-offs.
- Hedge off and hedge on, each with SPY and equal-weight rows.
- The generator first passes the checks in §8, in particular that it does not over-predict.

### Context: the earlier synthetic over-prediction

The eCornell S2 Monte Carlo built on the same hybrid generator over-predicted the rebalancing engine's growth by about
28 points a year. `~/Desktop/papers/hybrid-jump-hmm-paper/bias_correction_issue.md` blamed iid residuals plus daily
rebalancing.

On 2026-10-26 we found the cause in the eCornell engine instead:
- `compute_market_growth` stores the growth from day t to t+1 at index t.
- `allocate_shares` reads index t on day t, so every rebalance sees part of the move it is about to earn.

With the notebook's settings (22 tickers, 5 bps, 15% drawdown, 50% turnover), here is the engine's growth as written
and with the signal shifted back one day:

| | as written | shifted one day | market |
|---|---|---|---|
| synthetic, median of 300 paths | 46.8%/yr | 8.7%/yr | 9.9%/yr |
| real 2025–26 | 26.3%/yr | 8.9%/yr | 24.2%/yr (SPY) |

The synthetic market was not more predictable from day to day than the real one: the lag-1 autocorrelation is −0.04
synthetic and −0.11 real.

This study's signals use closes through the decision day only, and its test suite has look-ahead canaries, so we
expect no inflation. Check V2 in §8 tests this directly.

## 2. Goals, non-goals, anti-goals

Goals
- A synthetic-market generator (§4) that gives every universe name its own JumpHMM path.
  - Names are tied together through the market and their sector.
  - α and β drift over time.
- Running the unchanged study on each synthetic history, recalibrated on that history (§5).
- Tables for a typical set and a stress set of futures (§6, §7).
- Checks that the synthetic market is sane before any result is reported (§8).

Non-goals
- No change to the study's strategies, settings or real-data results.
- No copula and no cross-sectional rank reorder, anywhere.
- No links between sector factors in this version (they are drawn independently).
- No lagged market dependence (lead-lag).
- No names entering or leaving the universe.
- No hand-built crash scenarios or variance scaling in this version (§12 lists them as later dials).
- No figures until the user says what each should show.

Anti-goals: things that would make this a failure even if it runs.
- A synthetic market that over-predicts, as the eCornell Monte Carlo did.
- A reorder or any other step that destroys a name's own volatility clustering and jump timing.
- A synthetic world whose true α and β are exactly what the strategies already know. That hands γ the answer.
- Results presented as forecasts of returns, or as single pooled numbers that mix pickers and utilities.
- Returns instead of growth rates anywhere.

## 3. What the user decided (2026-10-26)

1. Synthetic data comes from JumpHMM for every ticker, composed with the market through the paper's variance-corrected
   SIM (`~/Desktop/papers/hybrid-jump-hmm-paper/paper/sections/method.tex`).
2. No copula reorder. It destroys each name's volatility clustering, and the paper skips it.
3. Recalibrate everything on the synthetic history. In the user's words: "update all of our parameters using that
   synthetic truth".
4. β drifts over time, and α drifts too. Both drifts are EMAs, like the rest of the approach.
5. Add co-movement within sectors with a sector factor. Start with independent sector factors.
6. The purpose is §1: how the strategies could perform in regimes we have not seen yet.
7. The user rejected a "null world" in which every name has the same true α. Don't use it.
8. The user approved the generator description in §4 ("looks right"), including the calls marked there.
9. Each name's α centers on its 2014–2024 estimate pulled toward the average, the option closer to real-market
   behavior (§4.3).
10. Results must be ready for the Oct 30 slides, with as many paths as the machine can run (§6, §10).

## 4. The generator

All growth rates are continuously compounded and annualized: g = ln(Sₜ/Sₜ₋₁)/Δt with Δt = 1/252 yr. Each synthetic
path covers the real trading calendar, every SPY trading day from 2014-01-03 to 2026-09-04, so every date-based
setting of the study works unchanged. The universe is the study's 373 names, with their real GICS sectors.

### 4.1 Market

g_m(t) is drawn from the stored JumpHMM market surrogate, which was fit to SPY over 2014–2024. It has regimes, jumps
and fat tails. Synthetic SPY is this market path, starting at the real SPY close on 2014-01-03.

- **File:** `~/Desktop/julia_work/eCornell-AI-finance-lectures/code/src/data/pretrained-jumphmm-market-surrogate.jld2`,
  key `"model"`.

### 4.2 Sector factors (independent)

1. **Real residuals.** For 2014–2024, compute each name's residual eᵢ(t) = gᵢ(t) − α̂ᵢ − β̂ᵢ·g_m(t), using the
   study's 2014–2024 SIM fit (`fit_base_sim`).
2. **Sector factor.** For each of the 11 sectors, the factor f_s(t) is the equal-weight mean of eᵢ(t) over the sector's
   names.
3. **Loading.** Each name's δᵢ is the OLS slope of eᵢ on the leave-one-out sector mean, i.e. the mean without name i
   **(my call:** this avoids the self-inclusion bias in small sectors, which have 15–17 names**)**.
4. **Fit.** Fit one JumpHMM per sector to the index exp(Σ f_s·Δt). Use the same JumpHMM settings as the stored
   per-ticker surrogates **(my call)**.
5. **Draw.** On each path the 11 sector paths are drawn independently of each other and of the market (user).

### 4.3 Drifting α and β

Each name's α and β follow an EMA of random shocks around its 2014–2024 value, the same recursion the study uses
everywhere, fed with N(0,1) draws instead of prices:

    xᵢ(t) = (1 − a)·xᵢ(t−1) + a·kᵢ·zᵢ(t),   a = 1 − 2^(−1/h)
    βᵢ(t) = β̄ᵢ + xᵢ(t)

α has the same form with its own x, h and k. The shocks z are independent across names, across α and β, and of every
other draw. Each x starts from its stationary distribution, which has mean 0 and SD kᵢ·√(a/(2 − a)), so the synthetic
2014–2024 already contains drift.

The calibration targets are below. The fitting procedure is left to the builder; the steps listed are defaults.

**β (default method, my call).**
1. Run the study's own EWLS on real 2014-01-03 → 2024-12-31: half-life 63, prior weight 63, starting at each name's
   2014–2024 fit.
2. Record each name's SD of β̂ᵢ(t), plus the median across names of the lag-252 autocorrelation of β̂ᵢ(t).
3. Choose one h_β for all names and a kᵢ per name, so that the same EWLS run on synthetic 2014–2024 reproduces both
   numbers. A name whose real SD is at or below the pure-noise level gets kᵢ = 0.

Matching the EWLS output on both sides takes care of estimation noise: the synthetic EWLS has the same noise as the
real one.

**α (default method, my call).** A 63-day EWLS α̂ is mostly noise, so the target uses long windows.
1. For each name, take Δᵢ = α̂ᵢ(2020–2024) − α̂ᵢ(2014–2019) by OLS, and its noise SEᵢ,₁² + SEᵢ,₂².
2. The pooled target is mean(Δᵢ²) − mean(noise), with a floor of 0.
3. Use h_α = 252 trading days and one k_α for all names, chosen so that the same statistic computed on synthetic data
   matches the real one.

Hedge names then arise naturally. A name whose β drifts below zero on a path is a real hedge on that path, and the
study's daily update has to find it.

**The level each name's α drifts around: pulled toward the average (user, 2026-10-26).** The user chose the option
closer to the real market. Real 2014–2024 data, measured 2026-10-26:
- The OLS α spread across names is 0.076/yr, against an RMS standard error of 0.075/yr.
- α on 2014–2019 vs 2020–2024 correlates 0.18 across names. Their covariance implies a persistent spread of about
  0.042/yr.
- α on 2014–2024 vs 2025–26 correlates −0.12.

With the raw estimates as centers, a rough estimate gives synthetic α̂ a spread of about 0.10/yr and a carry-over of
about 0.4. That is more spread and more persistence than the real market shows.

The rule is ᾱᵢ = ā + φ·(α̂ᵢ − ā), where α̂ᵢ is the name's 2014–2024 OLS α and ā their universe mean **(my call:
universe mean)**.
- Fit φ jointly with the α drift's k_α, so the synthetic data reproduce check V9: a spread of 0.076/yr and a
  carry-over of 0.18.
- Start the fit from φ ≈ 0.042/0.076 ≈ 0.55.
- Names keep their order: NVDA stays above KR, by less.

### 4.4 Each name's own noise

ε̃ᵢ(t) is a draw from the name's JumpHMM marginal, fit to its full growth over 2014–2024, and demeaned over the path.
It is not reordered, so the name keeps its own clustering and jumps.

- **Refit every name's model on the study's own data (my call).** The stored eCornell models
  (`~/Desktop/julia_work/eCornell-AI-finance-lectures/code/src/data/pretrained-portfolio-surrogate.jld2`, trained by
  `eCornell-AI-finance-lectures/code/scripts/train-portfolio-surrogate.jl`) were fit on a price file that still holds
  unadjusted splits for nine names: BF.B, CHD, CTSH, EOG, GOOG, MA, PPG, TECH and UNP.
  - Checked 2026-10-26: MA's training file has a one-day log move of 2.32 (a 10:1 split).
  - Its model's draws have a growth SD of 19.2/yr.
- Refit all 373 names on the study's split-reconciled 2014–2024 data, rather than patching nine. Use the stored
  models' settings (50 states, ν = 5, the same jump-tuning grid). The builder check measured about 10 s per name.
- Keep the stored market model: its SPY data is identical to the study's.

### 4.5 Composition and variance correction

    gᵢ(t) = αᵢ(t) + βᵢ(t)·g_m(t) + δᵢ·f_s(t) + sᵢ(t)·ε̃ᵢ(t)

This extends the paper's correction with the sector term. Its discussion section notes that the correction generalizes
by replacing βᵢ²σ_m² with the variance of the whole factor part.

σ_m², σ_s² and σ_gen,i² are fixed at their real 2014–2024 values, not taken from each path **(my call)**. These are
SPY's growth variance, the real sector factor's variance and the name's growth variance. With per-path sample
variances, a path with a jump burst (§6) would shrink every name's own noise and clip more β over the whole path,
calibration years included. With fixed values, a burst raises names' variance through the market term only.

    ρᵢ(t) = (βᵢ(t)²·σ_m² + δᵢ²·σ_s²) / σ_gen,i²
    if ρᵢ(t) ≤ 1 − f:  sᵢ(t)² = 1 − ρᵢ(t)
    else:              scale βᵢ(t) and δᵢ for that day by √((1 − f)/ρᵢ(t)), and sᵢ(t)² = f

f = 0.10 as in the paper. The sign is kept, so negative-β names stay hedges.
- sᵢ(t) is recomputed every day, so each name's total variance stays at its JumpHMM level as β moves **(my call)**.
- Log the share of name-days that are clipped.
- Keep the paper's R²-preserving branch for any name with real R² ≥ 0.80. None is expected among single stocks.

### 4.6 Building the data set

The generator returns a `MyMarketData` built directly, not through `load_market_data`:
- **close:** each name starts at its real close on 2014-01-03, on the loader's reconciled split basis. Then
  closeᵢ(t) = closeᵢ(t−1)·exp(gᵢ(t)·Δt), where gᵢ(t) is the growth into day t.
- **open:** set equal to close **(my call)**. The study doesn't use opens; only the legacy `Backtest.jl` and the data
  fingerprint read them.
- **volume:** each name's real median daily share volume over 2014–2024, held constant **(my call)**. The
  liquidity-based pickers then rank on dollar volume, which moves with the synthetic price.
- **spy_open / spy_close:** the market path from §4.1.
- **sector:** the real sectors.
- **g_f:** the real 3-month T-bill series, the same as in the real run **(my call)**.

## 5. Running the study on one synthetic path

1. **Calibrate.** Run `calibrate(md_synth; train_start = 2014-01-03, train_stop = 2024-12-31, window = omega_window)`.
   This is the user's "update all our parameters". It covers the SIM fit, the market moments and the shrinkage λ.
   - If `choose_shrinkage` makes a path too slow, the builder may fix λ at the real-data value (0.0), and must log it.
   - G and ξ are recomputed inside `build_context`, as in the real run.
2. **Build the context** with the unchanged `config/study.toml` settings, including `drawdown_reset = true`.
3. **Run a trimmed grid (my call).** All 7 utilities, hedge on and off, for these pickers:
   - γ per sector, bandit per sector and random per sector (33 names). Bandit and random use 5 seeds instead of 30.
   - At K = 13: global γ, correlation-aware γ, liquid then γ, random whole universe and random liquid 100. The random
     pickers use 5 seeds.
   - At K = 5 and K = 34: global γ and random whole universe only, to test the small-K result.
   - The fixed lists: the user's 13 names, and the 13 plus WMT and PG.
   - That is about 530 runs per path.
4. **Benchmarks per path:** synthetic SPY buy-and-hold and equal-weight universe buy-and-hold.
5. **Save** the same summary columns as `results/tier1_summary.csv`, plus the path id and set.
   - Keep only summary rows, not per-run files **(my call:** otherwise more than 100K run files**)**.
   - Results go under `results/synthetic/`. The real-run results are not touched.

## 6. Which futures

1. Draw 10,000 market paths over the full calendar, seeded by path id 1…10,000. Rank them by synthetic SPY's maximum
   drawdown within the trading window 2025-01-02 → 2026-09-04.
   - Tag each window by whether it contains a **jump burst**, using the simulation's jump flags.
   - Builder check, 2026-10-26, on the stored market model: a burst starts with probability about 0.0001 per day and
     lasts about 119 days. Burst days are ±3–3.5% with almost no net drift: a four-month whipsaw, not a slow bear market.
   - 5.4% of windows contain a burst. So do 68% of the 50 deepest window drawdowns.
2. **Three sets, about 1,000 paths in all (my call; the user asked for as many as the machine can run):**
   - **Burst stress:** the 200 deepest drawdowns among windows with a burst.
   - **Bear stress:** the 200 deepest drawdowns among windows without one.
   - **Typical:** 600 paths drawn at random from all 10,000.
   - A path can be in more than one set. Tables are made per set.
   - If the overnight run finishes early, the typical set grows; the stress sets stay fixed.
3. Only for these paths, draw the sector factors, the α and β drifts and the names' own noise. Their seeds are derived
   from the path id, so any path can be rebuilt bit for bit.
   - The non-market draws are independent of the market, so selecting on the market alone is valid.
   - Price paths are not stored; seeds are **(my call)**.

## 7. Outputs

Make tables only; figures come after the user says what each should show. For each set (typical, stress) and each
hedge setting:
- **Picker × utility table of median ending wealth ÷ start across paths,** in the layout of the real-run tables:
  - Rows are the pickers in plain names, with name counts. SPY and equal-weight rows go at the top.
  - Within a path, take the median over seeds first, then the median across paths.
- **The same table as the share of paths in which the cell beats that path's SPY.**
- **The same table for growth (yr⁻¹), max drawdown and Sharpe, each with its 10th–90th percentile across paths.**
- **Head-to-head distributions across paths:** global γ minus random whole universe at K = 5, 13 and 34; γ per sector
  minus random per sector; hedge on minus hedge off for growth and max drawdown.
- **For each cell, where the real 2025–26 result falls within the typical set,** as a percentile.

## 8. Checks before any result is reported

| # | Check | Pass |
|---|---|---|
| V1 | Synthetic SPY growth over 2014–2024, median across paths, against real | within 2 points/yr |
| V2 | Over-prediction guard: equal-weight universe, daily-rebalanced minus buy-and-hold growth, synthetic vs real 2014–2024 | gaps within 2 points/yr of each other |
| V3 | Each name: variance of gᵢ vs σ_gen,i², and OLS β on synthetic 2014–2024 vs the mean of βᵢ(t); variance vs real reported alongside | path median within 5%; β within 0.05 |
| V4 | Clustering kept: autocorrelation of abs(gᵢ) at lags 1, 5 and 20, synthetic vs the name's own JumpHMM draw, median over names; the same against real data, reported, not gated | median within 0.05 at each lag |
| V5 | Sector co-movement: mean pairwise residual correlation within each sector, synthetic vs real 2014–2024; across sectors reported | within 0.05 per sector |
| V6 | Drift: synthetic EWLS β̂ spread per name vs its target; share of paths with at least one name whose online β̂ ≤ 0 in the trading window, shown next to real 2025–26 | spread within 10%; share reported |
| V7 | Same path id → identical `MyMarketData` | bitwise |
| V8 | The study's full test suite, including the look-ahead canaries | passes unchanged |
| V9 | α spread and persistence: cross-name SD of OLS α on synthetic 2014–2024 (real 0.076/yr), and the correlation of α on 2014–2019 vs 2020–2024 (real 0.18) | within 0.01/yr; within 0.05 |

V3 and V4 gate on the composition, not the models: they check that composing doesn't destroy what each name's model
produces. How well JumpHMM itself matches real names is the paper's question. Here it is reported but not gated.

The builder check measured the stored models against real data:
- Their variance runs a median 7% above real.
- They cluster less: autocorrelation of abs(g) is lower than real by a median of 0.06 at lag 1, 0.17 at lag 5 and 0.10
  at lag 20.

The refit (§4.4) may change both.

The thresholds are my call. If a check fails, stop and report it to the user rather than tuning to pass.

## 9. Code layout and reuse (my call)

- **Dependency:** add JumpHMM.jl to `code/Project.toml`, pinned to the revision in the eCornell session-2 Manifest
  (`varnerlab/JumpHMM.jl`, `git-tree-sha1 3b1e4e55…`), so the stored models load. Use the same revision for the
  sector-factor fits.
- **Generator code:** a new `code/src/SyntheticMarket.jl` with the drift EMA, sector-factor fitting, composition and
  the `MyMarketData` builder. Compose with the paper's `compose_hybrid` logic. Do **not** call JumpHMM.jl's
  `HybridSingleIndexModel` or eCornell's `generate_hybrid_scenario`: both reorder with a copula.
- **Scripts:**
  - `scripts/20_synthetic_fit.jl`: refits the 373 name JumpHMMs (§4.4), fits the sector-factor JumpHMMs and the drift
    parameters, and writes them to `results/synthetic/generator.jld2` plus a summary TOML.
  - `scripts/21_synthetic_paths.jl`: draws the 10,000 market paths, ranks them and writes both sets.
  - `scripts/22_synthetic_run.jl`: runs each path in turn. Resumable: a path is skipped once its rows are written.
  - `scripts/23_synthetic_tables.jl`: builds the tables.
  - `scripts/24_synthetic_checks.jl`: runs §8.
- **Tests:** the variance target and β recovery on a toy universe; the drift EMA's stationary SD; leave-one-out δ;
  reproducibility from seed.
- **Branch:** `feat/synthetic-stress`, off `feat/ces-eta-sweep`.

## 10. Cost and timing

- **Machine:** Apple M3 Ultra, 32 cores, 512 GB.
- **Per path, measured 2026-10-26 on real data:** calibration 3 s and context 0.7 s. Generation is estimated at about
  10 s. The grid takes about 25 s on 24–32 threads; the last real reruns averaged 1.4 s per Tier 1 run and 0.55 s per
  universe run with 24 in flight. That is about 40 s per path.
- **Refitting the 373 name models:** about 10 s per name, a few minutes across 32 cores.
- **About 1,000 paths:** about 11 hours, one overnight run.
- **Schedule, for the Oct 30 slides (user):**
  - **Tue Oct 27:** the generator, the refits, the drift and sector fits, tests, and checks V1–V9.
  - **Wed Oct 28:** the per-path runner and a 20-path smoke run end to end, then launch the full run overnight.
  - **Thu Oct 29:** tables.
  - If a check fails on Tuesday, stop and report it rather than tuning to pass. That risks the Friday date.

## 11. Caveats to state with the results

- Names keep their 2014–2024 order of α, pulled in to the spread that carries over in real data. Synthetic edges
  persist about as much as real ones did within 2014–2024; real 2025–26 showed no carry-over (−0.12).
- Sector factors are independent, so there are no cross-sector rotations beyond what the market explains.
- Market dependence is contemporaneous only, as in the paper.
- The market surrogate reproduces 2014–2024's regimes. Worse regimes come only from tail draws (the stress set).
- The universe is fixed, as in the real run.
- The results describe this generator plus this strategy, not a forecast.

## 12. Left to the builder, and later dials

**Left to the builder:**
- type and file names;
- the exact fitting procedure for h and k within the §4.3 targets;
- the seed scheme;
- threading and resumability;
- table formatting in the style of the real-run tables;
- whether to time `choose_shrinkage` and fall back to the fixed λ (log it).

**Later dials, not in this version:**
- linked sector factors;
- scaling market variance (the paper's `scale_market`, 2× and 3×);
- a larger drift spread to stress β further;
- the full 30-seed grid on the stress set.

## 13. Open questions for the user

Both questions were answered on 2026-10-26:
- **Q1:** pull α toward the average (§4.3).
- **Q2:** results for the Oct 30 slides, with as many paths as the machine can run (§6, §10).
