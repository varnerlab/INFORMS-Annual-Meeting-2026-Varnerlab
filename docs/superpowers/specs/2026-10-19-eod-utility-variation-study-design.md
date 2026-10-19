# End-of-day utility-variation study: design

Date: 2026-10-19. Status: draft for user approval.
Repo: `varnerlab/INFORMS-Annual-Meeting-2026-Varnerlab` (private). All study material lives here.

Items marked **(my call)** were chosen by Claude, not the user. They are defaults the user can overturn.

## 1. Why

The live paper trial (`constrained-CD-with-MPC-paper-trade-trial`) runs one configuration: constrained Cobb-Douglas (CD) allocation with an MPC rebalance trigger, on a frozen 33-stock basket. From 06-01 to 10-16 it gained +7.6% while SPY gained +0.65%. An analysis on 2026-10-19 found two problems:

- **The rebalance band is too narrow.** All four band-exit rebalances after 09-01 happened once the optimizer had concentrated into 4–6 stocks. The single-index covariance assumes stock-specific moves are uncorrelated. In fact PSX and VLO have a residual correlation of +0.87, and CHD and PG +0.63. As a result, the model's daily σ for the portfolio was 0.64%, against 0.78% actual.
- **The volatility cap leaks for the same reason.** The model put the portfolio at 10.2% annualized volatility, against a cap of 12%. The actual figure was 12.3%.

A live trial can test only one configuration at a time, and only in real time. This study replays the strategy on end-of-day data so that utility functions and operating rules can be compared side by side. It is the material for a Varnerlab presentation at the **2026 INFORMS Annual Meeting (San Francisco)**, and for a paper afterwards.
- Session: "Computational Portfolio Optimization and Asset Allocation" (contributed, Finance), **Tuesday Nov 3, 2:45–4:00 PM**, Moscone South-310. Chair: Jiming Pan.

**Success:** a table, and figures, comparing each configuration on **Sharpe ratio, max drawdown, and alpha vs SPY**. The comparisons must hold up across bandit seeds and be free of look-ahead, so they can be presented without caveats that undercut them.

## 2. Goals, non-goals, anti-goals

Goals
- Compare six portfolio rules under identical data, timing, basket, constraints and costs (§5).
- Test the operational fixes suggested by the trial: sector-aware covariance, sector caps, band width and horizon (§8).
- Include a bandit "ticker step" that picks the basket, benchmarked against a random pick within each sector (§7).
- Validate the simulator against the live trial before trusting any other result (§9).

Non-goals
- No live or paper order execution, no broker integration, no options.
- No intraday data. Decisions use daily bars only.
- No changes to the live trial.
- No tax modeling. The engine repo's `Tax.jl` is out of scope.
- No fixing JumpHMM. That is separate work, and JumpHMM is not used anywhere in this study.

Anti-goals (these would make the study a failure even if the code works)
- Results inflated by look-ahead: filling at the same bar the signal used, or fitting parameters on the full sample and then scoring inside it.
- Allocator comparisons that differ in their constraints or inputs, not their utility.
- Claiming a bandit "edge" that does not beat a random pick within each sector. eCornell S4 already showed this happening (it beat random in 14 of 30 seeds).

## 3. Constraints

- **Time:** the talk is Tue Nov 3. The slides need Tier 1 results by **Fri Oct 30**, which leaves time to build them **(my call)**. Tier 2 runs before the talk if time allows; otherwise it goes into the paper, along with Tier 3.
- **Language:** Julia 1.12, following CHEME-5660 conventions:
  - `My*` types built with `build(T, NamedTuple)`, loaded through `Include.jl`
  - full docstrings with units
  - growth rates g = log(S_t/S_{t−1})/Δt in 1/yr, with Δt = 1/252
  - seeded RNG, default seed 5660
- **Codebase:** CHEME-5660 is the initial base (user's decision). Reuse code by copying the needed functions into this repo, not by depending on the 5660 package. That package is a large teaching monolith that is slow to load. **(my call)**
- **Data:** large data files are never committed. GitHub blocks files over 100 MB, and the 2014–2024 price file is 88 MB. A fetch script fills a gitignored `data/` folder. **(my call)**

## 4. Decided by the user

- New repo `INFORMS-Annual-Meeting-2026-Varnerlab`, using the CHEME-5660 codebase at least initially.
- Data arrives at the end of the day: decisions are made after the close.
- Six portfolios: Cobb-Douglas, CES, Stone–Geary, min-variance and tangent, **all five under the same constraints**, plus a **130/30 long-short Cobb-Douglas**.
- **Windows:** parameters are estimated on CHEME-5660's 2014–2024 data. The simulation starts in January 2025 and runs to the end of the 5660 2026 data (2026-09-04).
- **Ticker step:** the eCornell per-sector bandit, changed to look for baskets that **beat SPY over a 21-day window** (§7).
- **Deliverables:** slides for the Nov 3 talk, plus a paper.
- Forward projection: a **multi-asset GBM** replaces the trial's JumpHMM (§6a). JumpHMM has an unresolved issue with portfolios and is not used.
- Scorecard: Sharpe ratio, max drawdown, alpha vs SPY.

## 5. Portfolio rules

Notation:

| Symbol | Meaning |
|---|---|
| nᵢ | shares held |
| Sᵢ | price |
| W | equity |
| wᵢ = nᵢSᵢ/W | dollar weight |
| γᵢ | preference weight, from the trial's formula `compute_preference_weights` (engine `SIM.jl:135`): γᵢ = tanh(αᵢ/\|βᵢ\|^λ + \|βᵢ\|^(1−λ)·g_m) |
| Σ_g | **growth-rate covariance** in yr⁻², from the selected SIM model (§6). Defined as in CHEME-5660 L6b/L7a and the nomenclature notebook, with g = Δlog S/Δt and Δt = 1/252. The covariance rate is C = Δt·Σ_g, in yr⁻¹. |
| μ_g | expected growth rate, μ_g,i = αᵢ + βᵢ·ḡ_M, in yr⁻¹ (L7a) |

### Shared constraint set (all five long-only rules)

Port this from the engine's `solve_constrained_cd` (`modified_cobb_douglas_example/code/src/Allocator.jl:72`), separated from its objective so the five rules can share it:

- **Budget:** Σ nᵢSᵢ ≤ W. The remainder is cash.
- **Long-only:** nᵢ ≥ 0.
- **Per-stock cap:** wᵢ ≤ w_max.
- **Sector cap:** Σ_{i∈s} wᵢ ≤ s_max for each GICS sector s. This is new; "none" means s_max = 1.
- **Volatility cap:** wᵀΣ_g w ≤ σ²_g,max, a cap on the portfolio's growth-rate standard deviation in yr⁻¹.
  - The equivalent annualized volatility is σ_g,max·√Δt. So the trial's 0.12 annualized equals σ_g,max = 0.12·√252 ≈ 1.90 yr⁻¹.
  - The trial's engine applied the cap to C = Δt·Σ_g, which is the same constraint after rescaling.
  - The rest of this doc writes σ_max as shorthand for σ_g,max.
- **Turnover budget:** copied exactly from the trial: c̄·Σ|Δnᵢ| ≤ 0.10·W, with c̄ = 5 bps × the basket's mean price. This caps estimated trading cost, not trade size, and it rarely binds **(builder's call, accepted)**.
- **Floor for the log utilities:** stocks with γ ≤ 0 are held at the floor ε. This matches the engine.
- **How much to invest (my call):** every rule invests up to 100% of equity and holds cash only when the caps or σ_max stop it.
  - CD does this naturally, because its log utility keeps buying until a limit binds. The trial sat at 60% when only 4 stocks had γ > 0 and each was capped at 15%.
  - Min-variance would otherwise go to all cash, and tangent has no defined size. Both rules therefore follow two-fund separation (ruling made during the build, 2026-10-19):
    - First choose the **mix** at the largest investment the per-name cap, sector cap and turnover allow, ignoring the volatility cap. This is usually 100%.
    - Then **scale** that mix down by c = min(1, σ_g,max/σ(mix)), holding the rest in cash.
    - An earlier version forced Σw to the largest investment the volatility cap allowed. That left a single feasible portfolio whenever the cap bound, so min-variance and tangent were identical across most of the Stage 0 grid.
  - This keeps the comparison about the utility, not about different investment levels.

### Objectives

- **Cobb-Douglas:** maximize Σ_{γᵢ>0} γᵢ log nᵢ. This is the trial's rule.
- **CES:** CES utility with elasticity η, using the parameterization of CHEME-5660 `allocate_ces` (`AdaptivePortfolio.jl:446`). η = 1 reduces to Cobb-Douglas. Default η = 2 **(my call)**.
- **Stone–Geary:** maximize Σ_{γᵢ>0} γᵢ log(nᵢ − bᵢ), with nᵢ ≥ bᵢ for every stock. This follows CHEME-5660 `lectures/instructor/STONE-GEARY-UTILITY-NOTE.md`. The baseline is a **dollar baseline of 1% of equity for every basket stock**, bᵢ = 0.01·W/Sᵢ, rounded down to whole shares **(my call)**.
  - In effect, every stock gets a 1% core position (33% for a 33-stock basket), and the preferences γ split the remaining 67%.
  - A baseline the size of CD's floor (ε ≈ 0.001 shares) would make Stone–Geary a copy of CD.
  - The 1% core also directly limits the concentration found on 10-19.
  - If the baselines plus the limits leave no feasible allocation, the allocator keeps the previous holdings and records status `:infeasible`, the same as every other rule (ruling made during the build, 2026-10-19).
  - Tier 2 tests baselines of 0.5% and 2%.
- **Min-variance:** minimize wᵀΣ_g w, as in L6b/L7a, under the shared constraints.
- **Tangent (max Sharpe):** maximize (μ_gᵀw − g_f)/√(wᵀΣ_g w). This uses L7a's excess-growth form, with e = μ_g − g_f·1 and g_f the risk-free growth rate.
  - Solve it with the standard homogenization that turns it into a cone program. CHEME-5660 `Sharpe.jl` has a COSMO SOCP version.
  - If no feasible portfolio has μᵀw > r_f, fall back to min-variance and log that it happened **(my call)**.
  - Expected growth: μᵢ = αᵢ + βᵢ·ḡ_m, where ḡ_m is the trailing 252-day mean market growth rate **(my call)**.
  - Tier 2 adds a check variant with α set to zero.

**Test oracles:** with every constraint except the budget switched off, the CD, CES and Stone–Geary solvers must reproduce CHEME-5660's closed forms. Those are `allocate_cobb_douglas`, `allocate_ces`, and the Stone–Geary formulas in the note.

### 130/30 long-short Cobb-Douglas

- The long side L is the stocks with γ > 0, and the short side S is the stocks with γ < 0.
- Maximize Σ_L γᵢ log nᵢ⁺ + Σ_S |γᵢ| log nᵢ⁻.
- Long weights sum to at most 1.30 and short weights to at most 0.30. These are inequalities so the problem stays feasible under the caps; the log utility pushes both sides to their limits.
- Per-stock cap on |wᵢ|.
- The sector cap applies to the sector's net exposure \|Σ_{i∈s} wᵢ\|, which allows hedged pairs within a sector **(my call)**.
- The σ_max and turnover constraints apply to the signed weights.
- If S is empty, the portfolio runs as long-only CD at 100% and logs that it did.
- **Financing (my call):** this follows the standard 130/30 structure.
  - Short-sale proceeds pay for the extra long positions.
  - Net exposure (long − short) ≤ 1.00.
  - Cash ≥ 0: no borrowing beyond the short proceeds, and no negative cash.
  - If σ_max binds, the optimizer may shrink either side, but net exposure never exceeds 100%. A net-long book therefore cannot inflate alpha.
  - Any positive cash earns r_f.
- **Short costs:** borrow fee of 0.75% a year on short notional, accrued daily **(my call)**.
- **Margin calls:** not modeled. Gross exposure of 160% is within Reg-T.

## 6. Covariance options

The selected model must be used consistently in three places: (a) the σ_max constraint, (b) the min-variance and tangent objectives, and (c) the GBM forward projection that sets the rebalance band (§6a). Because the GBM draws directly from Σ_g, (c) follows automatically.

Both options use the SIM growth covariance of the CHEME-5660 notes, in yr⁻² (user's decision).

- **C1, SIM (L6b/L7a):**
  - Σ̂_g,SIM = s²_g,M·β̂β̂ᵀ + D̂_g, where D̂_g = diag(s²_g,ε,i).
  - The parameters are the online EWLS estimates of β and σ_ε at each decision.
  - The market variance σ²_g,M is also updated online (user's decision, 2026-10-19). It is an exponentially weighted variance of SPY growth, with the same half-life (63) and prior weight (63) as EWLS, seeded from the base fit.
  - The trial kept σ_m fixed at its 2014–2024 value. That value (≈17% annualized) overstated the 2026 market (≈10%), which is consistent with the trial's band being too wide before 09-01.
  - Only the §9 validation replay keeps σ_m fixed, to reproduce the trial's inputs.
  - This is the trial's model, without its extra ×Δt factor.
- **C2, SIM with shrunk residual covariance:**
  - This follows the extension in CHEME-5660 `lectures/week-7/L7a/SIM-RESIDUAL-CORRELATION-DISCUSSION.md` (2026-10-06): Σ_g = s²_g,M·ββᵀ + Ω_ε,λ, with Ω_ε,λ = (1−λ)Ω̂_ε + λ·diag(Ω̂_ε).
  - Ω̂_ε is the sample covariance of the basket's SIM residual growth rates over a trailing 126 trading days **(my call)**.
  - **Choosing λ (my call):** λ is picked once on 2014–2024, before Stage 0, by a rolling forecast check.
    - At each month-end, fit Ω̂_ε on the trailing 126 days.
    - Predict the next 21 days' **residual** variance, var(wᵀε), for random long-only portfolios drawn from random sector baskets. The prediction is wᵀD^{1/2}R_λD^{1/2}w.
    - Scoring residual variance rather than total variance keeps market-variance forecast errors out of the choice of λ (ruling made during the build, 2026-10-19).
    - Choose the λ in {0, 0.1, …, 1} that minimizes the forecast error, then freeze it.
    - The note itself says to choose λ on training data and keep the test period for evaluation.
  - λ = 1 reproduces C1.
  - This targets the specific failure found on 10-19.

## 6a. Forward projection: multi-asset GBM (replaces JumpHMM)

After each rebalance, project the new holdings forward T trading days to get the band μ_τ ± zσ_τ for τ = 1…T. This is the role of the trial's `forward_project`.

- **Model:** the multi-asset GBM in CHEME-5660's growth-rate form.
  - Each step, log Sₖ₊₁ = log Sₖ + Δt·μ_g + √Δt·L·Zₖ, with Zₖ ~ N(0, I) and LLᵀ = C = Δt·Σ_g.
  - The log-price shock covariance over a step is therefore Δt·C = Δt²·Σ_g.
  - The simulated growth rates g = Δlog S/Δt reproduce μ_g and Σ_g, which is the notes' own check.
  - Σ_g is the run's selected covariance (C1 or C2).
- **Drift:** μ_g,i = αᵢ + βᵢ·ḡ_M, the same expected-growth input as the tangent rule (§5). Because g is a log-growth rate, no −σ²/2 correction is applied.
- **Portfolio value per path:** shares × simulated prices + cash (cash earns r_f). μ_τ and σ_τ are the mean and standard deviation across paths at each τ.
- **Paths:** N = 1000, as in the trial. Seeded from the run seed, so runs are reproducible.
- **Short side:** for the 130/30 rule, short positions enter the portfolio value with negative share counts.
- **Check:** for a single stock, μ_τ and σ_τ must match the lognormal closed form. The engine's `MPC.jl` `forward_project_closed_form` is a reference.
- **Known gap:** GBM has no jumps, regimes or fat tails, so the band may be too narrow in a crash. This is stated as a limitation.

## 7. Ticker step (bandit)

- **Start point:** the eCornell S4 per-sector bandit (`eCornell-AI-finance-lectures/lectures/session-4/scripts/bandit/per_sector_bandit.jl`). The engine repo's `Bandit.jl` `train_sector_bandit` is the same algorithm.
  - An arm is a subset of q_s tickers within a GICS sector.
  - The reward is the 21-day log return of the CD-weighted sub-basket minus the sector's equal-weight return.
  - Exploration is ε-greedy with a floor of 0.05.
  - Quotas come from `assign_quotas`.
- **The change (user's decision):** the reward measures beating SPY, not the sector average.
  - For a pull on training day d, the reward is 1 if the CD-weighted sub-basket's 21-day log return from d beats SPY's 21-day log return over the same days, and 0 otherwise.
  - CHEME-5660's ticker picker uses the risk-free rate as its yardstick, and eCornell S4 uses the sector's equal-weight return. Here the bandit's target is the same benchmark the study is scored against.
- **Arm estimate (my call):** each arm's win rate is tracked as a Beta(1 + wins, 1 + losses) posterior, as in CHEME-5660's Beta-arm bandit. Exploration stays ε-greedy with eCornell's floor of 0.05. The mean 21-day excess return over SPY is also logged per arm, for diagnostics only.
- **Arms and quotas are unchanged:** each sector's bandit still chooses among q_s-subsets of that sector. A sector that lags SPY across the board still fills its quota with its most-often-winning subset.
- **Walk-forward:**
  - Retrain at the first trading day of each quarter in the out-of-sample window, using only data before that day. SIM parameters for γ are refit on trailing data, not the full sample **(my call)**.
  - A basket change forces a rebalance, with the new trigger reason `basket_change`.
- **Basket size:** K = 33, the same as the trial **(my call)**.
- **Pairing:** within a seed, all six portfolio rules use the same bandit basket, so their comparisons are paired. Because the bandit's reward is CD-weighted, the baskets may slightly favor CD; the random-pick baseline controls for this.
- **Baselines:**
  - **Random pick within each sector, with matched turnover (my call).**
    - The first draw is random, with the bandit's quotas.
    - At each quarterly retrain, it swaps out the same number of names per sector that the bandit swapped, choosing the replacements at random.
    - Rotation costs are therefore equal, so "the bandit beats random" measures stock-picking alone.
- **Passive benchmarks, reported but not run through the allocators:** SPY, and an equal-weight portfolio of the whole universe.

## 8. Simulation, data and variation plan

**Daily loop** (end-of-day timing):
1. After the close on day t, update the online SIM estimates (EWLS) with day t's growth. Half-life is 63 calendar days and prior weight 63, matching the trial.
2. Compute g_m and λ from SPY data through day t, then compute γ.
3. Mark equity at day t's close.
4. Check triggers in priority order: cash_revisit, then drawdown > D_max, then τ ≥ T, then band exit (μ ± zσ), then basket_change.
5. If a trigger fires, solve the selected rule.
6. Orders fill at the **open of day t+1** in whole shares, with costs from the engine's `Costs.jl`: a half-spread of 5 bps plus impact κ(q/ADV). τ counts actual trading days in the data.
7. Cash earns r_f.

**Data:**
- **Prices:** CHEME-5660 `code/src/data/` daily OHLC from Polygon:
  - 2014–2024: 424 tickers with complete histories
  - 2025: 483 tickers
  - 2026-01-02 → 09-04: 475 tickers
- **No extension past 2026-09-04.** The run ends where the 5660 data ends (user's decision), so no Alpaca data and no credentials are needed **(my call)**.
- **Universe:** tickers present in every file. This carries survivorship bias, listed under limitations.
- **Sectors:** `sp500-sectors.csv` (current GICS membership).
- **Dividends:** prices are split-adjusted but not dividend-adjusted. Price returns are used for both the portfolios and SPY. Dividends are a known difference in the §9 equity check **(builder's call, accepted)**.
- **r_f:** FRED DTB3 (3-month T-bill) cached into `data/`. If unavailable, use a constant 4%/yr **(my call)**.

**Windows:**
- **Calibration, 2014-01-02 → 2024-12-31:** base SIM fit (α, β, σ_ε, σ_m) and initial bandit training. GBM inputs are recomputed at each decision from trailing data, so the GBM needs no separate calibration.
- **Scored out-of-sample window, 2025-01-02 → 2026-09-04** (user's decision). This gives about 420 trading days and 6 quarterly bandit retrains.
- **Validation window:** the live-trial overlap that falls inside the data, 2026-05-29 → 09-04 (§9).
- **σ_max and w_max are chosen from 2014–2024 (user's decision).** The trial's values (0.12 and 0.15) were picked by sweeps over 2025-01 → 2026-04, which overlaps the scored window. They are used only in the §9 validation. The other settings are the trial's defaults and were never tuned on the scored window: z 1.96, T 21, D_max 0.08, cash_revisit 21, and the turnover budget, which rarely binds.

**Stage 0: tune σ_max and w_max (runs before Tier 1).** All choices below are **(my call)**.
- **Window:**
  - Fit base SIM parameters on 2014–2015 (warm-up).
  - Tune on 2016-01 → 2024-12, with walk-forward EWLS updates.
  - This prevents the tuning from being fooled by SIM parameters that were fit on the same days.
- **Grid:** σ_g,max ∈ {1.27, 1.59, 1.90, 2.38, 3.17} yr⁻¹ (annualized 0.08, 0.10, 0.12, 0.15, 0.20) × w_max ∈ {0.05, 0.10, 0.15, 0.20}, which is 20 pairs.
- **Baskets:** random-within-sector, 5 seeds. The bandit is not used, because it was trained on the same years.
- **Covariance:** tuned separately for C1 and C2. The same σ_max is a tighter limit under C2, because C2 sees more risk.
- **Selection:**
  - One shared (σ_max, w_max) pair per covariance option, consistent with "same constraints for all rules".
  - Pick the pair with the highest median Sharpe across all six rules; ties go to lower median max drawdown.
  - Tuning on CD alone is avoided, because it would favor CD.
- **Size:** 2 × 20 × 6 × 5 = 1,200 backtests.
- **Freeze:** the chosen pairs are written to a config file and frozen **before any run touches 2025**.

**Variation plan** (tiers, run in order):
- **Tier 1, needed for the talk:**
  - The §9 validation replay.
  - Stage 0 tuning, as above.
  - All 6 rules × {C1, C2}, using each covariance option's Stage 0 pair (σ_max, w_max). Otherwise: no sector cap, z 1.96, T 21, D_max 0.08.
  - Each combination runs on two kinds of basket: the bandit (10 seeds) and the random-within-sector baseline (10 seeds).
  - Baskets are newly picked. The trial's 33 stocks are not used in scoring (user's decision, 2026-10-19); they appear only in the §9 code check.
  - 240 backtests in total.
- **Tier 2, one setting at a time around the Stage 0 settings, for both C1 and C2, 30 seeds.** Tier 1's results are not used to choose the covariance option, because that would be choosing on the scored window. These sweeps run on the **2014–2024 window**, so they choose settings without touching the scored window. The chosen settings are then rescored on 2025-01 → 2026-09-04:
  - z ∈ {1.96, 2.5, no band}
  - T ∈ {21, 63}
  - s_max ∈ {none, 0.30}
  - CES η ∈ {0.5, 2, 5}
  - Stone–Geary baseline ∈ {0.5%, 1%, 2%}
  - tangent with α set to zero
- **Tier 3, for the paper:** a factorial design over the 2–3 settings that mattered most in Tier 2.

## 9. Validation against the live trial (gate for everything else)

The trial's band came from JumpHMM, which this study does not use. So validation **forces rebalances on the trial's actual decision dates** and checks everything that does not depend on the projection **(my call)**.

- **Setup:** replay 2026-05-29 → 2026-09-04 (the end of the data) with the trial's settings, its frozen 33 and its `sim_base` artifact. Force a rebalance on each of the trial's four decision dates in that window: 06-29, 07-28, 08-26 and 09-01. The 09-01 decision is the one that concentrated the trial into 6 names, so the most important event is covered.
- **Allocation check (the gate):**
  - At every forced date, the simulated target weights must match the trial's ticket `target_weights`: same names, each weight within 1pp.
  - Allocations do not depend on the projection. A mismatch therefore means a bug in γ, EWLS, the constraint set or the CD solver.
- **Equity check:** report simulated equity next to the account's `V_t` from `data/residuals.csv`, after removing the options sleeve's realized P&L (−$882). Differences must be explained, not necessarily zero; fill prices and the decide that did not run on 08-20 are known sources.
- **Band check (informational):**
  - Rerun the same window with triggers free and the GBM band, once with C1 and once with C2.
  - Report on which dates each fires, next to the trial's JumpHMM band exits.
  - The trial's band was too narrow because it ignored within-sector correlation. If C2 is the fix, its band should fire less often than C1's.

## 10. Scorecard

Computed on daily equity over the scored window:
- **Sharpe ratio:** √252 · mean(r − r_f) / std(r − r_f), using daily simple returns.
- **Max drawdown:** peak to trough on equity.
- **Alpha vs SPY:** Jensen's alpha from OLS of (r_p − r_f) = α + β(r_SPY − r_f) + ε. Report α annualized ×252, its t-stat, and β.
- **Secondary:** turnover, total costs, borrow fees, trigger counts by reason, and average number of names held.
- **Across seeds:** report the median and IQR, and win-rates for paired comparisons against the random-within-sector baseline.

Each run writes one result row (config, seed, metrics) plus its daily equity series, keyed by a hash of the config. Figures for the talk are generated from those results.

## 11. Where code comes from

| Need | Source |
|---|---|
| SIM growth covariance Σ̂_g,SIM, min-var and tangent in growth-rate form, multi-asset GBM in growth-rate form | CHEME-5660 L6b/L7a lectures, the nomenclature notebook, and `lectures/week-7/L7a/SIM-RESIDUAL-CORRELATION-DISCUSSION.md` (C2) |
| Type/build pattern, SIM fit, EWLS, CD/CES closed forms (oracles), tangent SOCP, min-var, data loaders, scorecard | CHEME-5660 `code/src` (`Types.jl`, `Factory.jl`, `AdaptivePortfolio.jl`, `Sharpe.jl`, `Files.jl`) |
| Constrained solver constraint set, γ and λ, MPC trigger logic, closed-form projection reference, costs, backtest loop reference | `modified_cobb_douglas_example/code/src` (`Allocator.jl`, `SIM.jl`, `MPC.jl`, `Costs.jl`, `Backtest.jl`). Its JumpHMM projection is **not** ported; the GBM projection (§6a) is new code. |
| Per-sector bandit | eCornell S4 `per_sector_bandit.jl`, which is the same as engine `Bandit.jl` |
| Trial settings, frozen basket, `sim_base` artifact, tickets, residuals.csv, live equity | `constrained-CD-with-MPC-paper-trade-trial` (`config/trial-config.toml`, `artifacts/`, `data/`) |

## 12. Known limitations (state them in the talk)

- **Survivorship bias:** CHEME-5660's 2014–2024 file holds only tickers with complete histories.
- **Sector assignments:** GICS membership is current, not point-in-time.
- **Projection model:** the GBM projection has no jumps, regimes or fat tails, so its band can be too narrow in a crash.
- **Validation:** the replay cannot reproduce the trial's own band exits, because those came from the JumpHMM projection this study does not use.
- **Shorting:** no borrow-availability limits or recall risk; the borrow fee is flat.
- **Costs:** the cost model is a parametric estimate, not real fills.

## 13. Open questions (user)

None. All three were resolved by the user on 2026-10-19:
- bandit change → §7
- scored window → §8
- talk and deliverables → §1 and §3

## 14. Left to the builder

- Module name and file layout inside `src/`, within CHEME-5660 conventions.
- The conic formulation for CES and the tangent homogenization.
- How the config sweep driver is built: TOML per experiment, resumable, parallel over runs.
- Plot styling for the talk figures.
