# Screened utility-panel study: design

Date: 2026-10-20. Status: approved by the user 2026-10-20.
Supersedes `2026-10-19-eod-utility-variation-study-design.md`.
Repo: `varnerlab/INFORMS-Annual-Meeting-2026-Varnerlab` (private).

Items marked **(my call)** were chosen by Claude, not the user. They are defaults the user can overturn. Everything else in §4 was decided by the user in the 2026-10-20 conversation.

## 1. Why

The live paper trial (`constrained-CD-with-MPC-paper-trade-trial`) runs one configuration: a constrained Cobb-Douglas allocator with an MPC rebalance trigger, on a basket frozen once by a bandit. A live trial can test only one configuration at a time, and only in real time.

This study replays the trial's machinery on end-of-day data, from 2025-01-02 to the end of the course data (2026-09-04), with three things the trial does not have:

- **A panel of allocators** instead of Cobb-Douglas alone.
- **A ticker screen that re-picks the basket** every 21 trading days or at a band exit, instead of a basket frozen once. Three screens are compared: a bandit that looks at past growth, a γ screen that looks forward through the SIM, and a random control.
- **A market hedge:** a cap on the portfolio's β that tightens when ξ turns bearish, plus a standing pool of negative-β names that every basket can draw on to meet it.

It is the material for the Varnerlab talk at the **2026 INFORMS Annual Meeting** (session "Computational Portfolio Optimization and Asset Allocation", **Tue Nov 3, 2:45–4:00 PM**, Moscone South-310), and for a paper afterwards.

**Success:** a table and figures comparing every allocator × screen combination on Sharpe ratio, max drawdown and alpha vs SPY, all computed from growth rates. The comparisons must be free of look-ahead and must hold up across seeds, so they can be presented without caveats that undercut them. The headline questions are:

1. Does re-picking the basket with a screen beat a random pick within each sector?
2. Does the forward-looking γ screen beat the backward-looking bandit?
3. How do the allocators compare under identical limits and inputs, each running with the same screen? (Exactly identical baskets across allocators come from the fixed-schedule sensitivity run in §10.)
4. Does the β cap with the hedge pool lower drawdown without giving up Sharpe, compared with the same runs without it? And is a long hedge (negative-β names) a better hedge than the 130/30 short book? **(my call: question 4 is how I would frame the hedge result)**

## 2. Goals, non-goals, anti-goals

Goals
- Run six allocators under identical data, timing, limits and costs (§6).
- Compare three ticker screens that re-pick the basket during the run (§8).
- Hedge market exposure with a ξ-driven cap on portfolio β and a standing pool of negative-β names (§6, §8), and measure what it does against the same runs without it (§10).
- Make every window, lag and setting configurable, with defaults, so the user can vary them (§11).

Non-goals
- No live or paper order execution, no broker, no options, no taxes.
- No intraday data. Decisions use daily bars.
- No changes to the live trial.
- No JumpHMM. The band comes from a multi-asset GBM (§7).

Anti-goals (these make the study a failure even if the code works)
- Look-ahead: any decision, screen or setting that uses data after the decision day; in particular, any setting chosen by looking at 2025–2026 results.
- Allocator comparisons that differ in their limits or inputs, not their objective.
- Claiming a screen adds value when it does not beat the random screen across seeds.
- Anything computed or labeled as a "return". Every quantity is a continuously compounded growth rate (§3).

## 3. Constraints

- **Time:** Tier 1 results (§10) are needed for slides by **Fri Oct 30** **(my call)**.
- **Growth rates only (user, 2026-10-20):** the growth rate over k trading days is g = ln(S_end/S_start)/(k·Δt), in 1/yr, with Δt = 1/252. Nothing is computed as a simple or log return: not the bandit reward, not Sharpe, not alpha, not any label or table heading.
- **Language and conventions:** Julia 1.12, CHEME-5660 conventions: `My*` types built with `build(T, NamedTuple)`, loaded through `Include.jl`; full docstrings with units; seeded RNG, default seed 5660.
- **Data:** large files are never committed. A fetch script fills a gitignored `data/` folder.
- **Configurable:** every setting in §11 is read from a config file. No window, lag or limit is hard-coded.

## 4. Decided by the user (2026-10-20)

- **Run window:** 2025-01-02 → 2026-09-04. Data from 2014–2024 only establish baseline parameter values.
- **Machinery:** the trial's: constrained allocator, MPC triggers, decide after the close, fill at the next open, trial costs.
- **SIM parameters update online** during the run, as in the eCornell S4 daily baseline and the trial.
- **γ input:** the market growth term is lagged by a long window: the 252-day moving average of SPY's daily growth, as in CHEME-5660 L8b, not a single day as in the trial.
- **Allocator panel:** five long-only rules (Cobb-Douglas, CES, Stone–Geary, min-variance, tangent) under the same limits, plus a 130/30 long-short Cobb-Douglas.
- **Covariance:** SIM market part plus a residual block estimated from data, shrunk toward its diagonal; the residual block is re-estimated during the run.
- **Projection:** multi-asset GBM using that covariance.
- **Screens:** (1) bandit, (2) γ screen, (3) random within each sector. Each picks the top N tickers per sector.
- **Bandit:** based on the eCornell S4 per-sector bandit, but scored on 21-day growth against SPY, learning from the trailing 252 trading days.
- **γ screen:** compute γ for every ticker in the universe and keep the N highest per sector. Because γ depends on ξ, the screen reads the market state.
- **Re-pick:** every 21 trading days, or at a band exit.
- **N:** 3 per sector by default (same as the trial's 33), configurable.
- **130/30 shorts:** the bottom N γ per sector.
- **Per-stock cap:** 10%.
- **Vol cap:** SPY's 2014–2024 growth-rate volatility, expressed as an annualized volatility (about 17%).
- **β cap (hedge):** every allocator carries a cap on the portfolio's β, Σ wᵢβᵢ ≤ β_max(ξ), that tightens when ξ turns bearish (§6). The cap uses each name's signed β.
- **Standing hedge pool:** names whose β is ≤ 0 both overall and on SPY's down days are eligible for every basket, on top of the top N per sector, under every screen (§8).
- **Everything adjustable:** all lags, windows and settings are configurable.

## 5. Daily loop

On each trading day t of the run, after the close:

1. **Online SIM update.** For every universe ticker, update (α, β, σ_ε) with day t's growth rates by exponentially weighted least squares (EWLS): half-life 63 updates, prior weight 63, seeded from the 2014–2024 fit. This is the trial's and eCornell's `ewls_update!`. The prior's sufficient statistics use SPY's 2014–2024 mean and variance of daily growth, as in CHEME-5660's `ewls_init`, not the trial's unit-variance seed, which makes the prior about 7× weaker than its nominal weight in growth units **(my call)**. The market variance σ²_M is also updated online as an exponentially weighted variance of SPY growth with the same half-life, seeded from the 2014–2024 fit (carried over from the user's 2026-10-19 decision).
2. **Market inputs**, from SPY closes through day t:
   - ξ_t = −G·(S̄_short/S̄_long − 1), the crossover of the 21-day and 63-day moving averages of SPY's close. The gain G is set on 2014–2024 so that |ξ| ≤ 1 on 99% of training days, as in L8b **(my call: L8b's calibrated gain, not the trial's fixed G = 20)**.
   - g̃_M,t, the 252-day moving average of SPY's daily growth rate.
   - Inputs use data through day t's close. There is no look-ahead because orders fill at day t+1's open **(my call; L8b lags one more day because it fills at the same close)**.
3. **Preference weights:** γᵢ = tanh( g̃ᵢ / max(|βᵢ|, β_floor)^ξ ), with g̃ᵢ = αᵢ + βᵢ·g̃_M and β_floor = 0.1 **(my call)**.
   - The numerator is the name's expected growth with its signed β, so a negative-β name is charged, not credited, when g̃_M > 0. The trial's form, tanh(αᵢ/|βᵢ|^ξ + |βᵢ|^(1−ξ)·g̃_M), replaces β by |β| in the numerator as well and treats β = −0.5 as +0.5.
   - The denominator scales by how strongly the name moves with the market, in either direction. Direction is handled by the β cap (§6).
   - β_floor keeps names with β near zero from blowing up the ratio.
   - For β ≥ β_floor this is exactly CHEME-5660 L8b's γ.
   - Online β goes ≤ 0 for a group of defensive names during the run (utilities, staples, exchanges); on a median day about 9 names, at the peak over 100.
4. **Mark equity** at day t's close.
5. **Triggers**, in the trial's priority order: cash revisit, drawdown > D_max, τ ≥ T, band exit (equity outside μ_τ ± z·σ_τ). The drawdown peak is the running maximum of equity and is never reset, as in the trial.
6. **Re-pick** (§8) if 21 trading days have passed since the last re-pick, or if the band-exit trigger fired. The hedge pool is refreshed at the same time. If the basket (screen picks plus pool) changes, a rebalance is forced with reason `basket_change`. **(my call:** the re-pick counter runs from the last re-pick, not the last rebalance; drawdown and cash-revisit rebalances keep the current basket.)
7. **Solve** the run's allocator (§6) if any trigger fired, then re-project the band (§7).
8. **Fill** at day t+1's open in whole shares, with the trial's cost model: half-spread 5 bps plus impact κ·(q/ADV). τ counts trading days.
9. **Cash** earns the risk-free growth rate (FRED DTB3; a constant 4%/yr if unavailable).

Starting capital is $100,000 in cash; the first decision is made after the close of 2025-01-02, using a basket picked by the screen from data through that day **(my call)**.

## 6. Allocators

### Shared limits (all five long-only rules)

Ported from the trial engine's `solve_constrained_cd` constraint set:

- **Budget:** Σ nᵢSᵢ ≤ W; the rest is cash.
- **Long-only:** nᵢ ≥ 0.
- **Per-stock cap:** wᵢ ≤ 0.10.
- **Sector cap:** off by default (s_max = 1), configurable.
- **Vol cap:** wᵀΣ_g w ≤ σ²_g,max, where σ_g,max is SPY's 2014–2024 growth-rate standard deviation; the equivalent annualized volatility is σ_g,max·√Δt (about 17%).
- **Turnover budget:** the trial's c̄·Σ|Δnᵢ| ≤ 0.10·W, with c̄ = 5 bps × the basket's mean price.
- **β cap (user, 2026-10-20):** Σ wᵢβᵢ ≤ β_max(ξ_t), with wᵢ = nᵢSᵢ/W at day t's close, each name's signed online β, and cash at β = 0. The cap applies to every allocator, including the 130/30 (on net weights).
  - β_max(ξ) = max(β_min, β_ref − κ_β·ξ), with defaults β_ref = 1.0, κ_β = 1.0, β_min = 0, so β_max = max(0, 1 − ξ) **(my call)**.
  - At ξ = 0 the portfolio may carry at most SPY's market exposure; at ξ = 0.5 (bearish) at most half; at ξ = −0.5 (bullish) up to 1.5, which rarely binds.
  - Over the run ξ ranges from about −0.75 to +0.85, so β_max ranges from about 1.75 down to 0.15.
  - Holding all cash always satisfies the cap, so it never makes the long-only problems infeasible.

### How each rule treats γ ≤ 0 names (my call)

- **Cobb-Douglas and CES:** names with γ ≤ 0 are held at the floor ε, as in the trial engine.
- **Stone–Geary:** every basket name, including γ ≤ 0 names, is held at least at its baseline bᵢ; γ splits the rest.
- **Min-variance and tangent:** classical rules that do not use γ, as in L8b. They may hold any basket name.
- **Hedge-pool names with γ ≤ 0** are the exception in the γ-based rules: instead of being pinned at ε, they are free variables with lower bound ε (bᵢ for Stone–Geary) and no utility term. Because the budget binds in these rules, the solver holds more than the floor only when the β cap or the vol cap makes the name worth its cost in budget. Pool names with γ > 0 are treated like any other basket name. Without this exception the trial engine's ε pin would keep the γ-based rules from using the pool at all.

Under the bandit and random screens some picked names will have γ ≤ 0 on a given day, so the γ-based rules hold fewer names than the basket size. This is part of what the screen comparison measures, not a defect: the γ screen picks names the γ-based rules want, and the other screens do not.

### Objectives

- **Cobb-Douglas:** maximize Σ_{γᵢ>0} γᵢ log nᵢ. The trial's rule.
- **CES:** CHEME-5660 `allocate_ces` parameterization, under the shared limits. Default η = 0.5, as in L8b **(my call)**. At η = 1 it is Cobb-Douglas.
- **Stone–Geary:** maximize Σ_{γᵢ>0} γᵢ log(nᵢ − bᵢ), with a baseline of 1% of equity per basket name, bᵢ = 0.01·W/Sᵢ rounded down **(my call)**. If the baselines and limits leave no feasible allocation, keep the previous holdings and record `:infeasible`.
- **Min-variance:** minimize wᵀΣ_g w under the shared limits.
- **Tangent:** maximize (μ_gᵀw − g_f)/√(wᵀΣ_g w), with μ_g,i = αᵢ + βᵢ·g̃_M. If no feasible portfolio beats g_f, fall back to min-variance and log it **(my call)**.
- **Investment level for min-variance and tangent** (two-fund rule, carried over): choose the mix at the largest investment the per-stock cap, sector cap, β cap and turnover allow, ignoring the vol cap; then scale it by min(1, σ_g,max/σ(mix)) and hold the rest in cash. Report the invested fraction next to every scorecard.

**Test oracles:** with every limit except the budget switched off, the CD and CES solvers reproduce CHEME-5660's `allocate_cobb_douglas` and `allocate_ces`, and Stone–Geary reproduces the formulas in the course note.

### 130/30 long-short Cobb-Douglas

- **Long side L:** the screen's basket names with γ > 0, plus the hedge pool (pool names with γ ≤ 0 are free long variables with no utility term, as above).
- **Short side S:** the bottom N γ names per sector with γ < 0, under every screen, so the screens differ only in their longs **(my call)**. Hedge-pool names are excluded from S, because shorting a hedge raises the portfolio's β **(my call)**. If S is empty, the rule runs as long-only CD and logs it.
- Maximize Σ_L γᵢ log nᵢ⁺ + Σ_S |γᵢ| log nᵢ⁻; long weights ≤ 1.30, short weights ≤ 0.30, net exposure ≤ 1.00, cash ≥ 0.
- Per-stock cap on |wᵢ|; sector cap (when on) on each sector's net exposure; vol cap, β cap and turnover on the signed weights.
- Short-sale proceeds fund the extra longs. Borrow fee 0.75%/yr on short notional, accrued daily. No margin calls (gross 160% is within Reg-T) **(my calls, carried over)**.

## 7. Covariance and GBM band

**Covariance:** Σ_g = σ²_M·ββᵀ + Ω_ε,λ, in 1/yr², with

Ω_ε,λ = (1 − λ)·Ω̂_ε + λ·diag(Ω̂_ε)

as in CHEME-5660 `lectures/week-7/L7a/SIM-RESIDUAL-CORRELATION-DISCUSSION.md`.

- β and σ²_M are the day's online estimates (§5).
- Ω̂_ε is re-estimated at every decision: the sample covariance of the SIM residual growth rates of the names in play (screen picks and hedge pool, plus shorts for 130/30), over the trailing 126 trading days, using the day's α and β **(my call: 126 days)**.
- λ is chosen once on 2014–2024 and frozen **(my call: same method as the 2026-10-19 build)**. At each month-end, fit Ω̂_ε on the trailing 126 days, predict the next 21 days' residual variance of random long-only portfolios from random sector baskets, and choose the λ in {0, 0.1, …, 1} with the lowest forecast loss.
- The same Σ_g is used for the vol cap, the min-variance and tangent objectives, and the GBM band.

**GBM band:** after each rebalance, project the holdings T trading days forward:

- log S_{k+1} = log S_k + Δt·μ_g + √Δt·L·Z_k, with Z_k ~ N(0, I) and LLᵀ = Δt·Σ_g, so simulated growth rates reproduce μ_g and Σ_g.
- Drift μ_g,i = αᵢ + βᵢ·g̃_M; no −σ²/2 correction, because g is a log growth rate.
- Portfolio value per path = shares × simulated prices + cash (cash at g_f; shorts enter with negative share counts). μ_τ and σ_τ are the mean and standard deviation across paths.
- 1,000 paths, seeded from the run seed.
- Check: for one stock, μ_τ and σ_τ match the lognormal closed form.

## 8. Ticker screens

Every screen re-picks at the times in §5, step 6, from the **universe**: tickers present in all three price files (2014–2024, 2025, 2026) with a GICS sector in `sp500-sectors.csv`. Each picks N names per sector (default 3, so 33 names).

### Screen 1: bandit (backward-looking)

Based on the eCornell S4 per-sector ε-greedy bandit (`lectures/session-4/scripts/bandit/per_sector_bandit.jl`), with three changes:

- **Arms are single tickers**, and the screen keeps the top N per sector, instead of one arm per N-subset.
- **Reward** for ticker i on a window starting at day d: g_i(d, d+h) − g_SPY(d, d+h), with g(d, d+h) = ln(S_{d+h}/S_d)/(h·Δt) and h = 21. S4 subtracts the sector's equal-weight average; here SPY.
- **Data:** at a re-pick on day t, window start days d are drawn uniformly from those whose window ends within the trailing 252 trading days and no later than day t, so every window is complete by day t. The bandit is retrained from scratch at each re-pick **(my call)**.

Unchanged from S4: one bandit per sector; exploration rate ε_k = max(0.05, k^(−1/3)·(A·ln k)^(1/3)), where A is the number of arms in the sector; iteration budget 50 × A, clamped to [500, 5000]; running-mean arm estimates. Ranking is by estimated mean reward; never-pulled arms rank last; ties are broken by the seed.

Note for the talk: with one ticker per arm, every arm's reward is observable on every day, so the bandit converges to ranking tickers by their average 21-day excess growth over SPY in the trailing year, which makes it a sector-neutral momentum screen.

### Screen 2: γ screen (forward-looking through the SIM)

At a re-pick, compute γᵢ (§5, step 3) for every universe ticker with the day's online parameters and inputs, and keep the N highest per sector. Because γᵢ = tanh(g̃ᵢ/max(|βᵢ|, β_floor)^ξ), the ranking is expected growth per unit of |βᵢ|^ξ: in a falling market (ξ > 0) it favors a sector's names that move least with the market, and in a rising one those that move most. Negative-β names rank low whenever g̃_M > 0, which is most of the run; they reach the basket through the hedge pool. Deterministic.

### Screen 3: random (control)

At each re-pick, draw N names per sector uniformly at random, independently of the previous basket **(my call: a fresh draw, with turnover and costs reported so the screens' trading can be compared)**.

### Standing hedge pool (all screens)

The user decided (2026-10-20) that every basket can draw on a pool of hedge names, on top of the screen's N per sector. The pool depends only on market data and the online SIM, so it is the same under every screen, allocator and seed on a given day.

- **Eligible** at a re-pick on day t: a universe name whose online β on day t is ≤ 0 **and** whose down-day β is ≤ 0. The down-day β is the OLS slope, with intercept, of the name's daily growth on SPY's daily growth, using only the days in the trailing 252 trading days (through day t) on which SPY's growth was negative. The second test drops names that look negative only because of up days; in 2025–26 it keeps KR, AWK, ED and CBOE and drops VZ, T and NEM.
- **Size:** up to H = 5 names, most negative down-day β first **(my call: at the 10% per-stock cap, five names can hold up to half of equity)**. Names already picked by the screen count once.
- **Refreshed** at every re-pick (§5, step 6). The pool can be empty; then the β cap is met by trimming positive-β names or holding cash.
- **Not shorted:** pool names are excluded from the 130/30 short side (§6).
- **Expected size:** seeded from 2014–2024 fits, no online β is ≤ 0 at the start of the run, so the pool is empty until April 2025 and missed the Feb–Apr 2025 sell-off. On 21-day re-pick dates it then held a median of 5 eligible names, rising to more than 50 in mid-2026.

## 9. Data and windows

- **Prices:** CHEME-5660 `code/src/data/`: `SP500-Daily-OHLC-1-3-2014-to-12-31-2024.jld2`, `…-1-2-2025-to-12-31-2025.jld2`, `…-1-2-2026-to-09-04-2026.jld2`. Split-adjusted, not dividend-adjusted; price growth is used for the portfolios and SPY alike.
- **Sectors:** `sp500-sectors.csv` (current GICS membership).
- **Risk-free:** FRED DTB3, cached in `data/`.
- **Calibration, 2014–2024** (baseline values only, no backtests): the SIM fit (α, β, σ_ε, σ²_M) for every universe ticker; the ξ gain G; the vol cap σ_g,max; the shrinkage λ.
- **Run, 2025-01-02 → 2026-09-04:** about 420 trading days. The first bandit training uses the trailing 252 days, which fall in 2024.

## 10. Runs

**Tier 1 (for the talk):** 6 allocators × 3 screens × hedge on/off, default settings.

- Bandit and random screens: 30 seeds each (S4 used 30). γ screen: deterministic basket, one run per allocator.
- Hedge on is the β cap and hedge pool as specified. Hedge off removes both (no β cap, no pool) and is otherwise identical, with the same seeds, so each hedge-on run has a paired hedge-off run **(my call)**.
- 6 × (30 + 30 + 1) × 2 = 732 backtests.
- Question 4's long-hedge vs short-hedge comparison needs no extra runs: hedge-on long-only CD against hedge-off 130/30 CD.
- Passive references, not run through the allocators: SPY buy-and-hold, and an equal-weight portfolio of the whole universe.
- Re-pick dates depend on each run's own band exits, so baskets can differ across allocators even within a seed. Pairing is by seed.

**Fixed-schedule run (Tier 1 if time allows) (my call):** the 366 hedge-on runs with band-exit re-picks turned off, so every run re-picks on the same 21-day schedule. The γ screen depends only on market data and the online SIM, and the bandit and random screens are seeded by (seed, re-pick date), so within a seed every allocator holds exactly the same baskets. This gives the clean allocator comparison for headline question 3.

**Sensitivity (as time allows, then for the paper):** rerun Tier 1 with one setting changed at a time, e.g. the vol cap, N, the bandit lookback, the g̃_M window, T and z. Every setting tried is reported; none is chosen for looking best.

The config-driven, resumable, threaded runner keys each run by a hash of its config and writes one result row plus the daily wealth series.

## 11. Configuration (defaults)

| Group | Setting | Default |
|---|---|---|
| Market inputs | ξ short / long moving-average windows | 21 / 63 days |
| | ξ gain G | calibrated (99% rule) |
| | g̃_M moving-average window | 252 days |
| | γ β_floor | 0.1 |
| SIM | EWLS half-life / prior weight | 63 / 63 |
| | EWLS prior seeding | SPY 2014–2024 mean and variance (CHEME-5660 `ewls_init`) |
| Covariance | Ω̂_ε window | 126 days |
| | shrinkage λ | calibrated, then frozen |
| Triggers | T / z / D_max / cash revisit | 21 / 1.96 / 0.08 / 21 |
| Projection | GBM paths | 1,000 |
| Screens | N per sector | 3 |
| | re-pick interval | 21 days |
| | re-pick on band exit | on |
| | bandit lookback / reward horizon | 252 / 21 days |
| | bandit ε floor, iterations per arm, min / max iterations | 0.05, 50, 500 / 5000 |
| | seeds | 30 |
| Limits | per-stock cap | 0.10 |
| | vol cap | SPY 2014–2024 (calibrated) |
| | sector cap | off |
| | turnover budget | 0.10 of equity |
| Hedge | β cap / hedge pool | on / on |
| | β_ref / κ_β / β_min | 1.0 / 1.0 / 0 |
| | pool size H | 5 |
| | down-day β lookback | 252 days |
| Allocators | CES η | 0.5 |
| | Stone–Geary baseline | 1% of equity per name |
| | 130/30 long / short / net | 1.30 / 0.30 / 1.00 |
| | borrow fee | 0.75%/yr |
| Costs | half-spread / impact κ | 5 bps / trial value |
| Account | starting capital | $100,000 |

## 12. Scorecard

From the daily wealth series over the run, all in growth rates:

- **Sharpe ratio:** with g_W,t = ln(W_t/W_{t−1})/Δt, Sharpe = mean(g_W − g_f)/(√Δt·std(g_W − g_f)), i.e. excess growth over the annualized volatility σ·√Δt. This matches CHEME-5660 `realized_scorecard`.
- **Max drawdown:** peak to trough on wealth.
- **Alpha vs SPY:** OLS of (g_W − g_f) = α + β·(g_SPY − g_f) + ε on daily growth rates. Report α in 1/yr, its t-stat, and β.
- **Secondary:** realized growth ln(W_T/W_0)/T, terminal wealth, turnover, costs, borrow fees, trigger counts by reason, re-pick count, basket changes, names held, invested fraction.
- **Hedge:** daily portfolio β (Σ wᵢβᵢ at the online β) against β_max(ξ), the share of days the β cap binds, the weight held in hedge-pool names, and the pool's size over time.
- **Across seeds:** median and IQR. For the bandit vs random, paired win rates by seed. For the γ screen, its percentile within the random screen's 30-seed distribution.

## 13. What changes from the 2026-10-19 design

Dropped:
- **The live-trial validation gate** **(my call):** the new strategy differs from the trial (252-day g̃_M, calibrated G, re-picking screens), so the trial can no longer validate it. Component tests remain.
- **Stage 0 tuning** on 2016–2024: the vol cap and per-stock cap are now picked (§4).
- **The plain-SIM covariance (C1)** as a separate option; λ = 1 still reproduces it if wanted.
- **The quarterly-retrained, N-subset, win/loss bandit** and its matched-turnover random baseline.
- **Simple returns** in the scorecard.

Kept, from the existing `feat/eod-study` code where it matches this design **(my call: reuse the tested components rather than rebuild them)**: the shared constraint set and six allocators, the GBM projection, costs, triggers, EWLS and SIM fit, the covariance and λ choice, and the experiment runner. New or rewritten: the market inputs and γ, the β cap and hedge pool (including freeing pool names from the ε pin), the three screens and the re-pick step, the backtest loop's re-pick logic, the config file, and the growth-based scorecard.

## 14. Known limitations (state them in the talk)

- **Survivorship bias:** the universe holds only tickers with complete histories.
- **Sector assignments** are current GICS membership, not point-in-time.
- **The GBM band** has no jumps, regimes or fat tails, so it can be too narrow in a crash.
- **Drawdown trigger:** because the peak is never reset, once equity is more than D_max below its peak the trigger fires every day until equity recovers, as in the trial.
- **Shorting:** no borrow-availability limits or recall risk; flat fee.
- **Costs** are a parametric estimate, not real fills.
- **Dividends** are not included, in the portfolios or in SPY.
- **The hedge was designed after looking at 2025–26 data.** The β cap and pool were added after inspecting the run's online betas and how KR, AWK and ED behaved in the Feb–Apr 2025 sell-off. No backtest was run, and each decision uses only trailing data, but the design is not blind to the run period. Report hedge-on and hedge-off side by side and say this in the talk.
- **Online β is noisy** at a 63-day half-life. The down-day test filters some false hedges, but a name can pass both tests by chance, especially when the pool grows past 50 eligible names in mid-2026.

## 15. Left to the builder

- Module and file layout within CHEME-5660 conventions, and the config file format.
- Solver formulations (CES and tangent cones, homogenization).
- Plot styling for the talk figures.
