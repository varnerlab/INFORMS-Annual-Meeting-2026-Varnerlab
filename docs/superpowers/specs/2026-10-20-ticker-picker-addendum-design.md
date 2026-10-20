# Ticker-picker addendum: universe-wide screens, Fibonacci basket sizes, fixed arms

Date: 2026-10-20. Status: approved by the user 2026-10-20.
Extends `2026-10-20-screened-utility-panel-study-design.md` (the "base spec"); everything there still holds unless this document changes it. § numbers refer to the base spec.

Items marked **(my call)** were chosen by Claude, not the user. Everything else in §3 was decided by the user in the 2026-10-20 conversation.

## 1. Why

> **Note (2026-10-20, added after the split reconciliation):** the diagnostic figures in this section (SPY 0.74, the L8b-list Sharpe ratios 1.36, 1.35 and 1.39, and the quota-free γ screen's 0.39 → 0.64) were computed before the data loader reconciled the split basis across the three price files (ORLY, NFLX, NOW and FAST at 2025-01-02; KLAC, MNST and APH at 2026-01-02). They are kept unchanged as the record of what motivated this addendum. For current numbers see `results/tables/head_to_head_K13.csv` (and `results/tables/group_summary.csv` for Tier 1).

Tier 1 (732 runs, base spec §10) found every allocator × screen median Sharpe below SPY buy-and-hold over 2025-01-02 → 2026-09-04 (SPY 0.74). A diagnostic then showed the machinery is not the cause:

- The study engine run on the user's CHEME-5660 L8b 13-name list (AAPL, MSFT, INTC, MU, AMD, NVDA, GS, BAC, WFC, C, F, GM, JNJ) reproduces L8b's 2025 monthly Cobb-Douglas result (Sharpe 1.36 vs L8b's 1.35). With every study limit and the hedge on, the list still gets 1.39 over the run window.
- The user chose that list on their own from sectors they follow (chips, finance, software, consumer), without any 2025 data.
- The base spec's screens all pick 3 names in each of 11 GICS sectors. Over the run window the median stock lagged SPY in 10 of the 11 sectors (Energy was the exception), so the quota puts most of the basket into laggards.
- A throwaway check with the quota removed raised the γ screen's Sharpe from 0.39 to 0.64 (CD, hedge on). That is still below SPY and far below the user's list.

This addendum asks which mechanical ticker pickers come close to the user's list, and how basket size matters.

**New headline questions** (added to base spec §1):

5. What does the sector quota cost? Compare sector γ (33 names) with global γ (34), and sector random with global random.
6. How do results change with basket size K across Fibonacci sizes?
7. At K = 13, can a causal screen match the user's 13-name list under identical rules?

## 2. Goals, non-goals, anti-goals

Goals
- Add three universe-wide screens, two matched random controls and a fixed-list arm. Run them at five basket sizes under the base spec's machinery, allocators, limits and hedge settings.
- Keep the base spec's 732 Tier 1 and 366 fixed-schedule runs comparable: same code path, same results directory, same tables.

Non-goals
- No new allocators, triggers, costs or hedge rules. The only allocator change is per-name per-stock caps (§5).
- No market-cap or fundamental data. Dollar volume stands in for size.
- No tuning of the new screens' settings on the run window. Every threshold below is fixed before any run.

Anti-goals
- Calling the user's list hindsight-biased. It was chosen in advance, without 2025 data (user, 2026-10-20).
- Presenting the new screens as out-of-sample discoveries. They were designed after Tier 1, so the talk must call their comparison exploratory (§8).
- Anything computed or labeled as a "return" (base spec §3).

## 3. Decided by the user (2026-10-20)

- **New screens:**
  1. Global γ: the top K γ across the whole universe.
  2. Liquid then γ: the top 100 names by dollar volume, then the top K γ among them.
  3. Correlation-aware γ.
- **Basket sizes:** Fibonacci, K ∈ {5, 8, 13, 21, 34}.
- **Per-stock cap scales with basket size**, so a small basket is not forced into cash. It applies to the K screen picks only; hedge-pool names and 130/30 short names keep the base 10% cap.
- **Fixed arm:** the user's 13-name L8b list (Fibonacci size).
- **Sensitivity:** the user's 15-name list from L8b (the 13 plus WMT, PG) was checked by the user's request. It performs about the same and is more balanced.

## 4. Screens (extends base spec §8)

All new screens re-pick on the base spec's schedule (every 21 trading days, or at a band exit when `repick_on_band_exit`). Each uses only data dated ≤ t. Each returns K column indices, sorted. γ_t is the day's γ for every universe ticker (base spec §5, step 3). Ties always break by column index.

| Screen | Symbol | Rule |
|---|---|---|
| Global γ | `:global_gamma` | The K highest γ_t over the whole universe. |
| Liquid then γ | `:liquid_gamma` | Shortlist = the top `liquidity_shortlist` = 100 names by dollar volume. A name's dollar volume is the median of close × volume (USD/day) over the trailing `liquidity_window` = 63 trading days ending at t. Then take the K highest γ_t in the shortlist. **(my call: 63 days, median, top 100)** |
| Correlation-aware γ | `:corr_gamma` | Walk down the global γ_t ranking. Accept a name if its correlation with every name already accepted is ≤ `corr_max` = 0.7. Correlation is the Pearson correlation of daily growth rates over the trailing `corr_window` = 252 trading days (growth into days t−251 … t). If the full pass accepts fewer than K, fill the rest with the highest-γ names not yet accepted. **(my call: 0.7, fixed before any run)** |
| Random, whole universe | `:random_global` | K names uniformly from the universe. A fresh draw at each re-pick from `MersenneTwister(_stream_seed(seed, date, 0))`, as the base spec's random screen. |
| Random, liquid shortlist | `:random_liquid` | K names uniformly from the same liquid shortlist as `:liquid_gamma`, with the same seeding. |
| Fixed list | `:fixed` | The configured ticker list. Every re-pick returns the same list, so re-picks never change the basket (the hedge pool may still change). |

Controls pair as follows:
- `:global_gamma` and `:corr_gamma` against `:random_global`.
- `:liquid_gamma` against `:random_liquid`.

What stays the same as the base spec:
- **Hedge pool:** unchanged. Pool names are added on top of the K picks.
- **130/30 shorts:** unchanged under every screen, i.e. the bottom N = 3 γ < 0 per sector, excluding the pool. So the screens differ only in their longs **(my call: consistent with base spec §6)**.

## 5. Per-stock cap

Effective cap for a **screen pick** (a name in the screen's basket, or in the fixed list): w_max,eff = max(`w_max`, `w_max_scale`/K), with `w_max` = 0.10 and `w_max_scale` = 1.5.

- K is the basket's size before the hedge pool: the screen's K, the fixed list's length, or 33 for the per-sector screens.
- **Hedge-pool names and 130/30 short names keep `w_max` = 10% (user, 2026-10-20).** So the pool keeps the base spec's design: 5 names can hold up to half of equity. A name that is both a screen pick and a pool name gets the pick's cap **(my call)**.
- At K = 33 the rule gives 10%, so the base spec's runs are unchanged.
- **Implementation consequence:** the per-stock cap becomes per name. `MyConstraintSet` carries a vector of caps, defaulting to the scalar for every name, so old problems are unchanged. This change has to reach:
  - the shared constraint rows, long-only and signed;
  - the `max_investable` and two-fund min-variance/tangent path;
  - the 130/30 rule.
  
  All existing allocator tests must still pass bit-identically when every cap equals the scalar.

| K | 5 | 8 | 13 | 15 | 21 | 33 / 34 |
|---|---|---|---|---|---|---|
| w_max,eff | 30% | 18.75% | 11.54% | 10% | 10% | 10% |

**(my call: the 1.5 factor, so a K-name basket can be fully invested with room to tilt)**

## 6. Runs (extends base spec §10)

| Block | Count |
|---|---|
| New deterministic screens: 3 × 5 K × 6 allocators × hedge on/off | 180 |
| Random controls: 2 × 5 K × 30 seeds × 6 allocators × hedge on/off | 3,600 |
| Fixed arms: the 13-name list (main) and the 15-name list (sensitivity) × 6 allocators × hedge on/off | 24 |
| **New total** | **3,804** |

Notes:
- **Seeds:** random controls use the base spec's seeds (5660 + 0 … 29), so hedge-on and hedge-off runs pair by seed.
- **Fixed schedule:** the new screens are not given a fixed-schedule variant **(my call: time; the base spec's fixed-schedule run already answers headline question 3)**.
- **Re-running the base runs:** adding settings fields changes every run's hash, so the 732 Tier 1 and 366 fixed-schedule runs are re-run under the new code. Their metrics must match the committed summaries exactly (the cap is still 10% at K = 33). This doubles as a regression check.
- **Compute:** about 4,900 runs, roughly 15 minutes on 24 threads at the last sweep's speed.

## 7. Configuration (extends base spec §11)

New settings, all in `config/study.toml`:

| Setting | Default |
|---|---|
| `basket_K` | 13. The universe-wide screens' size; the grid sweeps it. |
| `liquidity_shortlist` / `liquidity_window` | 100 / 63 days |
| `corr_max` / `corr_window` | 0.7 / 252 days |
| `w_max_scale` | 1.5 |
| `fixed_tickers` | the user's 13-name list. Used only by `:fixed`. |

New grid entries: `K = [5, 8, 13, 21, 34]`, `universe_screens = ["global_gamma", "liquid_gamma", "corr_gamma", "random_global", "random_liquid"]`, and the two fixed lists (13 main, 15 sensitivity).

## 8. Results and reporting

Tables (all growth-rate based, as base spec §12):
- **Size sweep:** median and IQR of Sharpe, max drawdown, alpha and realized growth, by screen × K × allocator × hedge.
- **Matched-control percentile:** each deterministic screen's value within its random control's 30 seeds, at each K.
- **Head-to-head at K = 13:** the user's 13-name list against every screen and control at K = 13. The 15-name list is shown as a sensitivity row.
- **Quota cost:** sector γ (33) vs global γ (34), and sector random vs global random (medians).

Figures:
- Sharpe vs K, one line per deterministic screen, with the random controls as median + IQR bands. SPY and the equal-weight universe are horizontal lines, and the fixed lists are markers at K = 13 and 15.
- The same for max drawdown.

**Limitations to state in the talk:**
- The new screens, their thresholds and the K grid were chosen after the Tier 1 results were known. Their comparison is exploratory, not out-of-sample.
- The user's 13- and 15-name lists were chosen in advance, without 2025 data.
- Base spec §14 (survivorship, GICS membership, GBM, dividends) applies unchanged.

## 9. Left to the builder

- Type names and file placement for the new screens, following the existing `AbstractScreen`/`pick` pattern in `code/src/Screens.jl`.
- How `pick` receives K: a settings field, or a field on the screen type.
- Table and figure layout.
