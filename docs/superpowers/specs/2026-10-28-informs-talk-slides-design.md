# INFORMS 2026 talk slides: design

Date: 2026-10-28. Status: written from the brainstorming session of 2026-10-28; the user will iterate on it.
Items marked **(my call)** were decided by the writer, not the user; everything else is the user's decision.

## The talk

- Title: "Coupling Interpretable Allocation, Model-Predictive Control, and Tail-Aware Forecasting for Live Portfolio
  Management" (keep it unchanged). Speaker: Jeffrey Varner, Cornell University.
- INFORMS Annual Meeting 2026, San Francisco. Session TD47, Moscone South 310, Tuesday, November 3; talk 5 of 6.
- 15 minutes: about 10 minutes of talk and 5 of questions. Slides are due **October 30**.
- Program book: `~/Downloads/2026_Annual_San_Francisco_Program_Book.pdf`.

## Intent

The audience is mathematically fluent and includes people from quantitative firms (Citadel, Millennium, Renaissance
and similar). The user wants them to leave thinking the work is serious and sophisticated. The four things they
should remember:

1. Utility functions such as Cobb-Douglas turn a preference score per stock into portfolio weights that maximize the
   investor's satisfaction.
2. A model-predictive-control style trigger decides when to rebalance.
3. The JumpHMM market generator.
4. The slides themselves are beautiful and precise.

The sophistication should come from precision and restraint: exact statements, clean figures, every number traceable
to data. Not from hype.

## Goals, non-goals, anti-goals

Goals:
- Eight main slides in the order below, plus three backup slides after the acknowledgements.
- Every plotted or quoted number comes from a committed results file through a script, and the build can be rerun.

Non-goals:
- The live (paper-trading) trial. It broke because of an options issue and is not part of this talk.
- New experiments. Every figure reads results that already exist; the MPC slide reads the forecasts stored in the
  run files.

Anti-goals (a failure even if the deck compiles):
- Anything that presents backtest results as live trading. Results are labeled as a real-data backtest.
- Hype or "jazz hands" wording. Slide text is plain and specific.
- Internal labels on slides (`global_gamma`, `l8b_13`, "SCS share", run labels). Use the display names below.
- Pooled medians across utilities. Results are shown as picker × utility, one dot per utility, with SPY shown.
- The word "returns" for performance. The study uses growth rates, g = ln(S_t/S_{t−1})/Δt with Δt = 1/252; ending
  wealth ÷ starting wealth is fine.
- Calling the hand-picked lists "hindsight": the user picked those names from sectors they follow, without 2025 data.
- Lecture furniture from the course template: no disclaimer slide, no learning objectives, no lecture footer.

## Style

- CHEME 5660 Beamer style: `~/Desktop/julia_work/CHEME-5660-CourseRepository-Fall-2026/lectures/templates/vnslides.sty`.
  XeLaTeX, 16:9. Fonts: Helvetica Neue Light, Medium and Bold. Colors: ink #333333 (headings), body #5E5E5E, coral
  accent #FF644E (`\hilite`, the title rule, the SPY line), link blue #579ACA.
- The template insists on a disclaimer slide and a lecture footer. Drop both; keep the title page, the frame title
  with the coral rule, the fonts and the colors.
- Forecast-band figures follow the CHEME 5660 L5a style
  (`~/Desktop/julia_work/CHEME-5660-CourseRepository-Fall-2026/lectures/week-5/L5a/`): 68/95/99% bands in
  `deepskyblue1` with transparency, the realized path in red at line width 3, a `gray95` plot background.
- Dot-plot grammar, shared by the real-run, synthetic and backup slides:
  - rows are ticker pickers, grouped by basket size, with small uppercase group labels in link blue;
  - one dot per utility;
  - hedge off is a filled blue dot (#579ACA), drawn just above the row line;
  - hedge on is a hollow dot with a darker blue outline (#2F6FAE), drawn just below it;
  - a pale bar (#DBE7F3) spans each row's lowest to highest dot;
  - SPY buy-and-hold is a vertical coral dashed line, and the equal-weight universe is a grey dotted line;
  - the legend reads "hedge off", "hedge on", "SPY buy-and-hold", "equal-weight universe" and "one dot per utility".
- Reference mockups, HTML generators written during brainstorming:
  - real-run dot plot: `.superpowers/brainstorm/96992-1793220151/content/real-run-dotplot-hedge.html`;
  - cleaned-up synthetic option A: `synthetic-a-v3.html` in the same folder.

  These folders are git-ignored scratch and may disappear, so treat this document as the authority. The mockups'
  numbers are out of date (see "Numbers" below).
- **(my call)** Slide numbers: small, bottom right, in grey on every slide except the title. No other footer.
- **(my call)** Draw the dot plots, the MPC strip and the stylized-fact panels in TikZ/pgfplots from small CSVs,
  so figure text uses the slide fonts and stays vector. A Julia script, `scripts/30_talk_figures.jl` in the
  `scripts/` environment (which already has Plots, CSV and DataFrames), writes those CSVs from the results. The
  builder may draw figures another way if the figure text matches the slide fonts and stays sharp.
- **(my call)** The deck lives in `talk/`: `talk/informs2026.tex`, a copy of `vnslides.sty` (so the repo stands
  alone once public), `talk/data/` for the plotted CSVs, and `talk/figs/` for TikZ sources.

## Slides

The time budget is about 10 minutes over six content slides, roughly 1.5 minutes each. Slide titles below are
placeholders unless quoted as decided.

### 1. Title

Talk title, speaker, Cornell University, "INFORMS Annual Meeting 2026 · San Francisco · Session TD47", "November 3,
2026". Use the template's title page.

### 2. Utility allocation: the general problem, then the closed form

Show the general constrained problem first, then the closed form under a budget alone. The closed form is why
allocation is fast.

- Preference weight per stock, from a single-index model of daily growth rates (`code/src/MarketInputs.jl`):
  γᵢ = tanh((αᵢ + βᵢ·g̃_M) / max(|βᵢ|, β_floor)^ξ). Names with γᵢ ≤ 0 are held at a tiny floor.
- The general problem (`code/src/Allocators.jl` docstrings): choose shares nᵢ to maximize a utility U(n; γ), subject
  to the constraints active in the study:
  - the budget;
  - a per-name cap of 10% of wealth;
  - a cap on the portfolio's growth-rate volatility, set to SPY's 2014–2024 level;
  - a spread-cost turnover budget per rebalance;
  - with the hedge on, a cap on the portfolio β.

  **(my call)** The sector cap exists in code but is off in the study (`s_max = 1`), so leave it off the slide.
- Utilities on the slide:
  - Cobb-Douglas: Σ γᵢ log nᵢ;
  - CES: (Σ γᵢ nᵢ^ρ)^(1/ρ) with ρ = 1 − 1/η;
  - CES on weights: the same with wᵢ = nᵢSᵢ/W;
  - Stone–Geary: Σ γᵢ log(nᵢ − bᵢ).
- Each solve is a convex cone program: exponential cones for the log utilities, power cones for CES.
- The comparison rules are minimum variance, the tangency portfolio and 130/30 long-short Cobb-Douglas. Name them in
  one line so the seven dots on later slides make sense.
- Closed form under a budget alone (`budget_only_weights`):
  - Cobb-Douglas: wᵢ ∝ γᵢ;
  - CES on weights: wᵢ ∝ γᵢ^η;
  - CES on shares: nᵢ ∝ (γᵢ/Sᵢ)^η.
- No speed claim in milliseconds unless the builder measures it.
- **(my call)** One slide with two builds, so the deck keeps its eight main slides:
  - first build: γ and the general problem, with the constraints written as words in one list rather than five
    equations;
  - second build: the closed forms.

  The comparison rules get one short line at the bottom.

### 3. MPC rebalance trigger (one strip over time, decided)

The user chose option C from the mockups: one timeline with successive forecasts, drawn in the L5a style from a real
episode. This replaces the earlier two-panel plan.

- After each rebalance the controller projects the book's value 21 trading days ahead with 1,000 multi-asset GBM paths
  (`project_gbm` in `code/src/Projection.jl`). The bands are mean ± 1, 1.96 and 2.576 SD of that projection, shown as
  68/95/99%.
- It re-plans early when the value leaves the 95% band (z = 1.96) or falls 8% below its peak. Otherwise it waits out
  the 21 days (`code/src/Triggers.jl`, `MyMPCSpec`).
- The y axis is scaled wealth, **W / W₀** (user decision), and the realized value is the red line.
- Each re-plan starts a new forecast cone from the value the market actually delivered. A cone cut short by a band
  exit ends at the exit day, and the rest of the abandoned forecast is not drawn.
- Coral ring markers at each re-plan carry a short label, e.g. "Nov 20: below the 95% band → re-plan".
- **The user's framing, which goes on the slide:** the controller's model is a simple multi-asset GBM, while the
  "plant" is real prices (or, on the stress-test slide, JumpHMM futures). This is model-plant mismatch on purpose.
  The mockup puts it in one line under the title: "Controller's model: multi-asset GBM, 1,000 paths, 21 trading days.
  Plant: real prices."
- **Real data:** the run files already store every forecast.
  - `MyStudyResult.projections::Dict{Date,MyProjection}` holds the mean `μ` and SD `σ` in USD for τ = 1..21 at each
    decision date.
  - `decisions` holds `date` and `reason` (`initial`, `horizon_elapsed`, `band_exit`, `drawdown`, `basket_change`).
  - `equity` and `dates` give the realized path.

  No logging hook or rerun is needed.
- **Episode (my call, from the mockup):**
  - the run: hand-picked 13 names, Cobb-Douglas, hedge off, seed 5660, cache file
    `results/gamma_seeds/runs/36b34a2c14db9505912bdbd832ed8b61cb1d1399.jld2` (seed 5660 matches the committed run);
  - the window, Oct 10 – Dec 22, 2025, three forecasts:
    - from Oct 10, it runs its 21 days, and the re-plan comes on Nov 10;
    - from Nov 10, it leaves the 95% band on day 8 (z = −2.1), and the re-plan comes on Nov 20;
    - from Nov 20, it runs its 21 days, and the re-plan comes on Dec 22;
  - the footnote: "Real-data backtest: hand-picked 13 names, Cobb-Douglas, hedge off. Bands are the forecasts stored at
    each rebalance."
- Mockup generator: `mock_mpc_real.py` in the session scratchpad, rendered as
  `.superpowers/brainstorm/96992-1793220151/content/mpc-real.html`. Both are scratch; this section is the authority.
- How many rebalances: across the runs on slide 5, the median `n_decisions` is 28. That includes the first
  allocation, so it is 27 re-plans in 419 trading days. The headline run has the same count: 28 decisions, which is
  27 re-plans. Say "about 27 rebalances after the first allocation" and take the number from `n_decisions − 1`. Do
  not use `n_repicks`, which counts basket re-picks.

### 4. JumpHMM generator

- Left: the model architecture figure from the JumpHMM paper. The TikZ source is
  `~/Desktop/julia_work/HMM-w-jumps-paper/arxiv-paper/sections/main_figures.tex`, the `fig:model_architecture` figure
  (around lines 21–126). Restyle it to the deck's fonts and colors.
- Right: real SPY daily growth rates over 2014–2024 against this year's generator (`results/synthetic/generator.jld2`,
  its market growth draws, e.g. `synthetic_draws(gen, p).g_m` restricted to the 2014–2024 dates). Three small
  panels:
  - the distribution (log density, so the tails show);
  - the autocorrelation of g;
  - the autocorrelation of |g|.

  **(my call)** Use 100 generator paths and show the real series as a line against the generator's 10th–90th
  percentile band.
- The point to land: the generator reproduces fat tails, little autocorrelation, and volatility clustering.

### 5. Real run (dot plot, decided)

- Label: "Real-data backtest, Jan 2025 – Sep 2026" (decided). The trading window is 2025-01-02 to 2026-09-04.
- **(my call)** A footnote on this slide: "A paper-trading deployment of this system is running; it is not part of
  this talk." The user approved one line to this effect; the placement is mine.
- Rows, in three groups:
  - **33 names, 3 per sector:**
    - γ per sector;
    - Bandit per sector;
    - Random per sector.
  - **13 names:**
    - Global γ;
    - Correlation-aware γ;
    - Liquid then γ;
    - Random, liquid 100;
    - Random, whole universe.
  - **Hand-picked:**
    - Hand-picked 13;
    - Hand-picked 13 + WMT, PG.
- Seven dots per row and hedge state: CD, CES, CESw, SG, MinVar, Tangent, LS. The x axis is ending wealth ÷ start.
- Reference lines: SPY buy-and-hold 1.317 and the equal-weight universe 1.296 (equal dollars in all 373 names, bought
  at the start and held).
- **(my call)** The legend glosses the hedge as "hedge on: up to 5 names with negative β join the basket and the
  book's β is capped".

### 6. Synthetic stress test (option A, decided)

- Three side-by-side panels with the same rows and dot grammar as slide 5:
  - Typical, 600 futures;
  - Burst stress, 200 futures;
  - Bear stress, 200 futures.
- Each dot is the median ending wealth across that set's futures for one utility. Each panel has its own x axis with
  ticks every 0.1, and its own SPY and equal-weight lines.
- The three panels share one set of row labels on the left, as in the mockup.
- All ten slide rows, plus the SPY and equal-weight rows, are in all six tables: three sets × hedge off/on. The
  tables also hold Global γ (5) and Global γ (34) rows; those are not plotted.
- One shared axis title: "ending wealth ÷ start, median across futures".
- Data: `results/synthetic/tables/{typical,burst_stress,bear_stress}_hedge_{off,on}_wealth_median.csv`.
- Placeholder title: "978 synthetic futures in three regimes". Do not title it "same ranking in every regime": that
  is false, because in bear futures random picks match or beat γ for most utilities.

### 7. Conclusions

Three takeaways in the lecture-deck format. This is the user-approved text; refresh the numbers from the 30-seed
medians (see "Numbers").

- **Utility functions set the weights.** We compared seven weighting rules: four utility functions (Cobb-Douglas, CES
  on shares, CES on weights, Stone–Geary), minimum variance, the tangency portfolio, and a 130/30 long-short
  Cobb-Douglas. Under a budget alone, the utility rules have a closed-form answer. With limits on single names,
  sectors, volatility and turnover, the problem stays convex and a standard solver handles it. In the real run, the
  stocks we picked mattered more than the rule: on one 33-stock basket the seven rules ended between 1.09 and 1.23
  times the starting wealth, while different baskets ranged from 1.03 to 1.70.
- **The MPC trigger decides when to rebalance.** After each rebalance we simulate 1,000 paths of the portfolio's
  value over the next 21 trading days. We rebalance early if the value leaves the 95% band or falls 8% below its peak,
  and otherwise wait out the 21 days. In the real run that came to about 27 rebalances in 419 trading days.
- **We tested the method on simulated markets.** A JumpHMM fitted to 2014–2024 data reproduces fat tails, weak
  autocorrelation and volatility clustering. We used it to simulate 978 markets: 600 typical, 200 with jump bursts and
  200 deep bear markets. Picking 13 stocks by γ beat 13 random stocks in 62–71% of typical markets. In bear markets,
  random picks did as well or better for most weighting rules.
  - **(my call, a draft the user has not approved)** Add: "On the real data, picking by γ beat random picks clearly
    up to 13 names, less often at 34, and about half the time at 100 and 200 names." Keep it off the slide until the
    user approves the wording.
  - Takeaway 2 keeps "about 27 rebalances", meaning re-plans after the first allocation (see slide 3).

Two notes on the numbers. Takeaway 1's ranges are hedge off: γ per sector spans 1.086–1.234 across the seven rules;
across the slide's rows, the range runs from 1.031 (random whole-universe 13, LS) to 1.697 (hand-picked 13,
Tangent). Takeaway 1 also mentions sector limits, which the study did not turn on; the builder should flag the
wording to the user rather than change it.

### 8. Acknowledgements

People:
- Abdulrahman Alswaidan, MS 2026: co-author on the JumpHMM paper.
- Aaron Wheeler, PhD 2024: "helpful discussions" (my wording).
- Evan Vu, undergraduate, class of 2026, now at Bridgewater: an early version of the Cobb-Douglas allocation and the
  bandit picker.

Alpaca Markets, in plain text with no logo **(my call)**.

Two QR codes side by side:
- "Paper": https://arxiv.org/abs/2603.10202, labeled "JumpHMM paper · arXiv:2603.10202". No paper title, because a
  retitled replacement is prepared but may not be live yet.
- "Code": https://github.com/varnerlab/INFORMS-Annual-Meeting-2026-Varnerlab. Making the repo public is a separate
  step that needs the user's go-ahead.

Use the `qrcode` LaTeX package (installed with TeX Live 2025). **(my call)** Layout: people and Alpaca on the left
two-thirds, the two QR codes on the right.

### Backup slides (after the acknowledgements)

Each is titled with the question it answers.

- **B1. "Was 2025 a lucky year?"** Where each real result falls among the 600 typical futures: rows and dots as on
  slide 5, with the x axis as the percentile from 0 to 100%, and a 50% line instead of the SPY and equal-weight lines.
  SPY's own percentile (70th) is shown as a row. The committed tables
  (`results/synthetic/tables/typical_real_percentile_hedge_{off,on}.csv`) used the single-seed real values for the γ
  pickers and the hand-picked lists; recompute those rows with the 30-seed real medians.
- **B2. "Does this work for a real-sized book?"** Global γ against random picks from the whole universe at 5, 8, 13,
  21, 34, 100 and 200 names, on real data, with 30 seeds on both sides.
  - Show it per utility, hedge off and on, as the share of seed pairs in which global γ ends higher, with a 50% line.
  - Leave Stone–Geary out at 100 and 200 names, with a footnote. Its 1%-per-name floor uses the whole budget at 100
    names and twice the budget at 200, so most decisions fail and the 200-name book sits in cash.
- **B3. "Show me the numbers."** The exact picker × utility tables behind slide 5, hedge off and hedge on, with SPY
  and equal-weight rows. These are 30-seed medians, written to two decimals or three.

## Numbers

- Real run, every row: the median over seeds 5660–5689 for each picker × utility × hedge cell.
  - γ pickers and hand-picked lists: `results/gamma_seeds/gamma_seeds_summary.csv`, from `scripts/dev/gamma_seeds.jl`.
    It has 4,200 runs; seed 5660 reproduces the committed single-seed runs exactly.
  - Random and bandit rows: `results/tier1_summary.csv`, `results/universe_summary.csv` (random, liquid 100 and whole
    universe at 13 names), and `results/fixed_summary.csv` for reference.
- Ending wealth is `terminal / W0` (W0 = 100,000 USD). SPY and the equal-weight universe come from
  `passive_references` in `code/src/StudyTables.jl`.
- Large books: `results/large_k/large_k_summary.csv`, from `scripts/dev/large_k.jl`. It has 30 seeds for both
  pickers at 100 and 200 names.
- Synthetic: `results/synthetic/tables/` and `results/synthetic/tables.md` (commit 33fbb2e).
- **Not yet committed:** `scripts/dev/gamma_seeds.jl`, `scripts/dev/large_k.jl` and their summary CSVs. Commit the
  scripts and summaries before building on them, but not the per-run caches.
- Expected real-run medians, hedge off, as a check on the figure script:

  | Picker | CD | CES | CESw | SG | MinVar | Tangent | LS |
  |---|---|---|---|---|---|---|---|
  | γ per sector | 1.230 | 1.219 | 1.221 | 1.234 | 1.086 | 1.227 | 1.110 |
  | Global γ (13) | 1.520 | 1.384 | 1.460 | 1.441 | 1.355 | 1.341 | 1.248 |
  | Correlation-aware γ (13) | 1.352 | 1.386 | 1.421 | 1.380 | 1.284 | 1.404 | 1.219 |
  | Liquid then γ (13) | 1.333 | 1.310 | 1.379 | 1.350 | 1.238 | 1.361 | 1.305 |
  | Hand-picked 13 | 1.548 | 1.571 | 1.614 | 1.631 | 1.443 | 1.697 | 1.549 |
  | Hand-picked 13 + WMT, PG | 1.616 | 1.646 | 1.637 | 1.639 | 1.362 | 1.642 | 1.622 |
  | Random, whole universe (13) | 1.098 | 1.132 | 1.138 | 1.112 | 1.104 | 1.160 | 1.031 |

## Constraints

- Due October 30. Main slides first, backups second.
- Runs need `OMP_NUM_THREADS=1 KMP_GTID_MODE=2`, a workaround for SCS's OpenMP library issue.
- Nothing is pushed, tagged, merged or made public without the user's explicit go-ahead. The work stays on
  `feat/synthetic-stress` or a branch off it.
- The user designs the figures. Ask before changing a decided figure design. Placeholder titles and items marked
  "my call" are open to the user.

## Left to the builder

- Exact slide titles (placeholders are fine until the user reviews), speaker notes, and transitions. Keep transitions
  minimal.
- Figure tooling and layout details within the style rules above.
- A different MPC episode, only if the chosen one fails to render cleanly. It must still show a horizon re-plan
  followed by a band exit, from a run shown on slide 5.
- How the backup slides lay out their content, within the dot-plot grammar.
