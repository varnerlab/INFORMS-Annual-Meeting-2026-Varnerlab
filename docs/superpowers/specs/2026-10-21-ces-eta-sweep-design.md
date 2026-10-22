# CES elasticity sweep: two CES versions, the clean limits, and the η dial in the study

Date: 2026-10-21. Status: approved by the user on 2026-10-22.
Extends `2026-10-20-screened-utility-panel-study-design.md` (the "base spec") and
`2026-10-20-ticker-picker-addendum-design.md` (the "addendum"); everything there still holds unless this document
changes it. The CHEME-5660 reference is L8b, `lectures/week-8/L8b/CHEME-5660-L8b-Lecture-Utility-Allocation-Bets-Fall-2026.ipynb`
in the course repository, and its CES companion notebook `advanced/adaptive_utility/CHEME-5660-L8b-Advanced-CES-Limits-Fall-2026.ipynb`.

Items marked **(my call)** were chosen by Claude, not the user. Everything in §3 was decided by the user in the
2026-10-21 conversation.

## 1. Why

The study's CES allocator uses one elasticity, η = 0.5, which was picked without a reason. CES is the one
utility in the panel with a free shape parameter, and L8b presents that parameter as a concentration dial with
three limits. Over share counts nᵢ, with preference weights γᵢ and prices Sᵢ, under a budget alone:

| η | Share-count CES (L8b), nᵢ ∝ (γᵢ/Sᵢ)^η | Weight-based CES, wᵢ ∝ γᵢ^η |
|---|---|---|
| → 0 | equal share counts (a price-weighted book, like the Dow) | equal weights (1/N) |
| = 1 | Cobb-Douglas, wᵢ ∝ γᵢ | Cobb-Douglas, wᵢ ∝ γᵢ |
| → ∞ | everything in the largest γᵢ/Sᵢ ("bang for the buck") | everything in the largest γᵢ |

The share-count version's dollar weights are wᵢ ∝ γᵢ^η·Sᵢ^(1−η), so for η ≠ 1 they depend on each stock's
nominal share price. L8b's CES companion notebook already states this. The share-count version is the course's
theory and stays. The weight-based version is added beside it as the price-invariant comparison. At the same η
and on the same basket, the two decisions differ only in how share price enters.

Every rule sees the study's split-adjusted prices (user, §3). The study compares methods on this dataset; it is
not a reconstruction of trading history.

The talk's story (user, §3): show the limits in data, then show how much of the dial survives the study's real
limits, then pose a market-responsive η, set by the market state the way ξ tilts γ, as a hypothesis.

**New headline questions** (added to base spec §1 and addendum §1):

8. Does the shape of the utility matter? Along η, does performance change a lot or a little?
9. How do growth, drawdown and concentration trade off as η moves (η as a risk setting)?
10. How large is the nominal-price gap between the two CES versions at each η?
11. Does the best η differ between bullish and bearish market states (motivation for η(ξ))?

## 2. Goals, non-goals, anti-goals

Goals
- Add the weight-based CES allocator, `CESw`, with every rule the share-count `CES` has.
- Show the three limits cleanly: budget only, on the user's L8b 13-name list (step (a), §5).
- Sweep η inside the full study's Tier 1 grid, with the 10% per-stock cap on and off (step (b), §6).
- Split the sweep's results by market state (§7).
- Rerun every grid once, fresh, with this change and the 2026-10-21 fixes (§9).

Non-goals
- No market-responsive η rule is built or tested. η(ξ) is stated as a hypothesis only (§7).
- No change to `CES` (share counts), to γ, to the screens, triggers, costs or hedge rules.
- No η sweep in the size-sweep (addendum) or fixed-schedule grids; they get `CESw` at the headline η only (§8).

Anti-goals
- Choosing a headline η, or an η(ξ) rule, from 2025–26 results. The run window is the test window.
- Calling the share-count CES wrong. It is the course's theory, and its price dependence is documented there.
- Reconstructing as-traded prices, or calling the split-adjusted prices look-ahead (user, §3).
- Calling the user's 13-name list hindsight-biased. It was chosen in advance, without 2025 data (user, 2026-10-20).
- Anything computed or labeled as a "return": performance is growth rates g = ln(S_end/S_start)/(k·Δt), Δt = 1/252.

## 3. Decided by the user (2026-10-21)

- Keep the share-count CES for its theory (the Cobb-Douglas and bang-for-the-buck limits), and add a weight-based
  CES as a second version.
- Sweep η, since 0.5 was just picked.
- The story: show the limits; then hypothesize that η could be made market-responsive, like ξ.
- Two steps: (a) the clean limits, budget only, on the L8b 13-name list; (b) the same sweep inside the full study.
- Step (b) runs twice: with the 10% per-stock cap on and with it off.
- Both CES versions use the split-adjusted prices, like every other rule. The study is a comparison of methods
  on the split-adjusted data, not a reconstruction of history, so no as-traded prices or split calendar.
- The turnover rule c̄·Σ|Δnᵢ| ≤ `turnover_frac`·W is relabeled a spread-cost budget in the comments and docstrings
  (user, 2026-10-22). It is kept from the trial and the code is unchanged. At 5 bps, for a long-only book, it can bind
  only when a basket's mean share price is more than 100 times its cheapest name's price.
- Also decided on 2026-10-21 and committed on branch `fix/fractional-shares-dtb3` (d0e6074, c5dc6b0, 684f7e6;
  not part of this design, but part of the same rerun): fractional-share fills, the DTB3 bank-discount
  conversion with a one-day publication lag, and λ calibrated with the study's own residual covariance (λ is
  still 0).

## 4. The weight-based CES allocator

`CESw` maximizes U = (Σ_{γᵢ>0} γᵢ·wᵢ^ρ)^(1/ρ) with ρ = 1 − 1/η and wᵢ = nᵢSᵢ/W, the dollar weight of name i.
Under a budget alone its optimum is wᵢ ∝ γᵢ^η over the names with γᵢ > 0, after the floors (below) are paid
for. η = 1 is Cobb-Douglas exactly, as for `CES`.

Every other rule is `CES`'s, unchanged: names with γ ≤ 0 pinned at ε shares, free hedge-pool names (nᵢ ≥ ε, no
utility term), every shared limit (per-stock caps, sector cap, volatility cap, β cap, turnover budget), the
lexicographic hedge solve, `:no_preferred` when no γ > 0, and η = 1 delegating to Cobb-Douglas. The only change
is the utility's argument: weights instead of share counts. It is a new rule type (e.g. `MyCESWeightRule`), built
from `(η = …,)`, with allocator label `"CESw"` in the study and `"CES"` keeping its meaning (share counts). Both
read the same `ces_eta` setting.

Conic form: the existing `CES` power-cone formulation with the cone's variable taken as wᵢ = nᵢSᵢ/W. The
`CES` implementation rescales its share variable by a typical share count c so the objective stays O(1); the
weight version needs the same care, with a typical weight in place of c **(builder: pick the scale; see §10)**.

## 5. Step (a): the clean limits, budget only

Setting: the `:fixed` screen with the user's 13-name list (`grid.fixed_lists.l8b_13`), hedge off, per-stock cap
off (`w_max` = 1, so every cap is 1), volatility cap off (none), sector cap off (`s_max` = 1, already the
default). The turnover budget stays (it never binds). Costs, triggers and fills are the study's. One seed (the
screen is deterministic).

η grid **(my call)**: {0.05, 0.1, 0.25, 0.5, 2, 4, 10, 20} for both versions, plus the Cobb-Douglas run as
η = 1. These are 17 runs, a new experiment `limits` with its own summary `results/limits_summary.csv`.

Outputs:
1. **The dial, one day.** On the run's first decision day (2025-01-02), the allocation of each version at every
   η in the grid: each name's dollar weight against η (log axis), one panel per version. The solver's weights
   are checked against L8b's closed form (budget only, floors paid first) at every η; a mismatch beyond 1e-3
   in any weight is an error, not a footnote. If fewer than 5 of the 13 names have γ > 0 that day, use the
   first decision day with the most names at γ > 0 instead, and state the date **(my call)**.
2. **The dial over the run.** For each version and η: realized growth rate, growth-rate volatility, maximum
   drawdown, mean effective number of names (§6) and mean invested fraction, over 2025-01-02 → 2026-09-04.

## 6. Step (b): the η sweep in the study

Cells: the Tier 1 grid exactly (screens bandit, γ, random; hedge on and off; 30 seeds for the random screens, 1
for the deterministic ones; 122 runs per allocator and setting).

η grid **(my call)**: {0.1, 0.25, 0.5, 2, 4, 10}, evenly spaced in log η around 1, with Cobb-Douglas as η = 1.

Two cap settings (user, §3):
- **Cap on:** the study's limits, unchanged (`w_max` = 0.10). At large η this fills the best names to 10% each,
  in order of γᵢ/Sᵢ (`CES`) or γᵢ (`CESw`). The η = 0.5 runs of both versions and the η = 1 run (CD) are the
  Tier 1 runs themselves (§8), so this setting adds `CES` and `CESw` at {0.1, 0.25, 2, 4, 10}: 1,220 runs.
- **Cap off:** `w_max` = 1 (so a pick's cap and a pool name's cap are 1 too); every other limit, the hedge and the
  hedge-off twin unchanged. Concentration then runs until the volatility cap (or the β cap) stops it. This
  setting needs its own η = 1 point, Cobb-Douglas with the cap off: `CD`, plus `CES` and `CESw` at all six η,
  is 13 × 122 = 1,586 runs.

These 2,806 runs are a new experiment `eta` with summary `results/eta_summary.csv`. Run labels add the η and the
cap setting, e.g. `CESw|bandit|hedge=true|seed=5660|eta=2|cap=off` **(my call on the format)**.

New run statistics **(my call)**, added to `study_stats` so every grid reports them:
- `mean_eff_names`, a concentration measure that, unlike `n_names`, does not count tiny holdings. Per decision:
  take the long target weights above the 0.1% hold threshold, rescale them to sum to 1, and compute 1/Σwᵢ².
  Cash is not counted as diversification (the mean invested fraction reports it). Decisions whose long weights
  sum to less than 1% (an all-cash book) are skipped, and `n_eff_skipped` counts them. For 130/30, the long side.
- `n_failed`, the decisions with status `:infeasible` or `:solver_failed` (the book is kept), and `n_fallback`,
  the decisions accepted from the SCS fallback solver or at Clarabel's `ALMOST_OPTIMAL`. The extreme η are
  where these are most likely. They are reported in the sweep tables; any η point where they exceed 1% of
  decisions is flagged to the user, who decides whether to keep it. No point is dropped automatically.

Baskets: re-picks fire on each run's own band exits, so at the same seed the two CES versions (and different η)
can hold different baskets after the first band exit, as every Tier 1 allocator pair already can. Step (b)'s
gaps between the versions therefore include basket and timing differences **(my call: accepted, and said so on
the slide)**. The same-basket comparison of the two versions comes from step (a) (fixed list) and from the
fixed-schedule grid at η = 0.5 (timer re-picks only, so the same baskets).

Outputs, per version × η × cap setting × hedge (medians over seeds within each screen, as the Tier 1 tables do):
realized growth rate, Sharpe ratio, maximum drawdown, growth-rate volatility, `mean_eff_names`, mean invested
fraction, turnover. Figures: each statistic against η (log axis), both versions, cap on and off, Cobb-Douglas
marked at η = 1; and growth against drawdown traced along η.

## 7. Market-state split (motivation for η(ξ))

For every run of the η sweep (both cap settings) and of step (a): each trading day t of the run after the first
is labeled by ξ_{t−1}, the `daily.xi` of the previous trading day (computed from closes through t − 1, the state
the last decision before day t could see): bullish if ξ_{t−1} < 0, bearish if ξ_{t−1} > 0, dropped if it is 0
**(my call: the same-day ξ_t moves with day t's own SPY close, which would push down days toward "bearish")**. Per run, the growth rate over each label's days is the mean of
the daily growth rates ln(V_t/V_{t−1})/Δt over those days (yr⁻¹). The table reports, per version × η × cap ×
hedge × screen, the median across seeds of each regime's growth rate, the days in each regime, and SPY's
growth rate over the same days.

This is exploratory and in-sample. The talk may say "the best η differs (or does not differ) between bullish
and bearish days in 2025–26, which motivates (or does not motivate) η(ξ)". It may not propose a specific η(ξ)
rule from these numbers. A rule would be fitted on 2014–2024, as λ and the ξ gain were, in later work.

## 8. The other grids

`CESw` joins `grid.allocators`, at the headline η, so Tier 1 (854 runs), the fixed-schedule grid (427) and the
addendum's size sweep and fixed lists (4,438) all report both CES versions.

Headline η **(my call)**: 0.5 for both versions in every main table and figure. 0.5 was set for `CES` before any
run; `CESw` uses the same value so the two versions are compared at one η. The sweep is reported as sensitivity
around it, never used to replace it.

## 9. The rerun

After this change, every grid reruns with `--fresh`: Tier 1, fixed-schedule, universe, `eta`, `limits`. This is
also the rerun for the fractional-share, DTB3 and λ changes, so it is done once. Then `12_tables.jl` and
`13_figures.jl` regenerate every table and figure. All committed numbers move; the README's dated notes record
why (fractional shares, DTB3, `CESw` added). About 8,500 runs at ~4–5 s each, roughly 11 CPU-hours.

## 10. Tests and acceptance

- `CESw`, budget only: matches wᵢ ∝ γᵢ^η (floors paid first) at every η in both grids, including 0.05 and 20.
- `CESw` at η = 1 equals Cobb-Douglas.
- `CES`, budget only: matches nᵢ ∝ (γᵢ/Sᵢ)^η at every η in both grids, including 0.05 and 20. The existing
  tests cover only η = 2 and 1.
- Price invariance: multiplying one name's price by k (a split) leaves `CESw`'s weights unchanged and changes
  `CES`'s (η ≠ 1).
- `CESw` with every shared limit binding, as the existing `CES` tests do (vol cap, sector cap, β cap with a
  hedge name, γ ≤ 0 pins, `:no_preferred`).
- The run builders for `eta` and `limits` produce the counts and labels in §5–6.
- `mean_eff_names` (with an all-cash decision skipped), `n_failed` and `n_fallback` on hand-built results.
- A real-data smoke run of a few `eta` and `limits` runs, with the accounting checks used for the
  fractional-share change (cash ≥ 0, equity = cash + holdings).

**Solver accuracy at the extreme η** is the main risk. η = 0.05 means a power cone with exponent 0.05
(ρ = −19); η = 20 means ρ = 0.95. If the solver cannot meet the closed-form checks at an η after rescaling,
that η is dropped from its grid and the user is told which and why. No η is dropped silently, and no tolerance
is loosened to make a check pass without saying so.

## 11. Left to the builder

- The scale for `CESw`'s cone variable, and any rescaling `CES` needs at the extreme η (§4, §10).
- How `eta` and `limits` are selected on the command line of `11_run_study.jl`, and where their grids live in
  `config/study.toml`.
- Table file names and figure layout, within the outputs listed in §5–7.
- Whether the step (a) single-day picture is computed through the study's problem builder or directly from the
  allocator; either way it uses the run's γ and prices for that day.
