# Interpretable allocation and model-predictive rebalancing of equity portfolios

This repository holds the code, results, slides and paper for the INFORMS Annual Meeting 2026 talk
"Coupling Interpretable Allocation, Model-Predictive Control, and Tail-Aware Forecasting for Live Portfolio
Management" (Jeffrey D. Varner, Cornell University; session TD47, San Francisco, November 3, 2026). The companion
paper, "Interpretable Allocation and Model-Predictive Rebalancing of Equity Portfolios, Tested on Real and Synthetic
Markets", is on arXiv as [arXiv:2611.03022](https://arxiv.org/abs/2611.03022); its sources are in `arxiv-paper/`.

## What the study does

The study tests one way to run a stock portfolio on daily S&P 500 data. It has three parts.

1. **What to hold.** Each stock gets a score, γ, from a single-index model fit: its growth beyond what the market
   explains, scaled by its market exposure β. How strongly β counts depends on the market state, which compares a
   21-day and a 63-day moving average of SPY. A picker chooses a basket of names: by γ, at random, with a bandit,
   or from a fixed list chosen in advance. Then the book is weighted by maximizing a utility function under trading
   limits. Seven weighting rules are compared: Cobb–Douglas, CES on share counts, CES on weights, Stone–Geary,
   minimum variance, the tangency portfolio and a 130/30 long-short book.
2. **When to trade.** The book re-plans when 21 trading days have passed, when its value leaves a forecast band
   (model-predictive control with a GBM forecast), or when it falls 8% below its peak.
3. **How to judge it.** The main test is a backtest on real data from January 2, 2025 to September 4, 2026, after
   calibration on 2014–2024. A backtest is a single historical path, so the study also runs on synthetic markets
   built by a JumpHMM generator fit to the same stocks: 600 typical futures and two sets of 200 simulated sell-offs,
   978 markets in all, because some markets fall in more than one set. Each synthetic market covers 2014–2026; the
   study calibrates on its 2014–2024 and trades its 2025–2026.

Each configuration runs twice: once with a market hedge and once without. The hedge caps the book's β, with a
tighter cap as the market state turns bearish, and it can hold up to five names whose β is zero or negative.

The universe is the 373 S&P 500 names with a complete daily history from 2014 to 2026 and a current GICS sector.
Performance is measured in growth rates, g = ln(S_end/S_start)/(k·Δt) per year with Δt = 1/252, and in ending
wealth ÷ starting wealth. Simple and log returns are never used.

## Main findings

- **Allocation:** which stocks were picked mattered more than how they were weighted. On one basket, the seven
  weighting rules ended close together, while different baskets ended far apart.
- **Rebalancing:** the book re-planned at the 21-day horizon, at a band exit or at a drawdown stop. About a third of
  the re-plans came before the horizon. Compared with re-planning on the calendar alone, ending wealth on real data
  came out about the same, and the worst drawdown was about one point smaller with the triggers.
- **Evaluation:** in both sell-off sets, every median book ended above SPY. In typical synthetic futures, 13 names
  picked by γ beat 13 random names in about two-thirds of the futures, for every weighting rule. In bear futures,
  random picks did as well or better for most rules.

The numbers behind each finding are in `results/` (below), the slides and the paper.

## What is in the repository

| Folder | What it holds |
|---|---|
| `code/` | The Julia package `VLINFORMS2026`: data loading, calibration, the pickers, the weighting problems, the backtest engine, the synthetic-market generator, and the tests. |
| `scripts/` | Numbered scripts that run the study from start to finish (below). `scripts/dev/` holds the extra checks the talk and paper use. |
| `config/study.toml` | Every study setting, with its units. |
| `results/` | The committed summaries, tables and figures. Per-run files are not committed. |
| `talk/` | The INFORMS deck (XeLaTeX Beamer). `talk/informs2026.pdf` is the built deck. |
| `arxiv-paper/` | The paper (pdfLaTeX). `arxiv-paper/Paper_v1.pdf` is the built paper. |
| `docs/design-notes.md` | Every modeling choice, solver ruling and data fix in the real-data study, and what each run step writes. |
| `docs/superpowers/` | The design spec and the implementation plan for each piece of work, by date. |

## Setup

1. **Julia 1.12.** The manifests were built with Julia 1.12.7.
2. **Packages.** Install both environments once:

   ```
   julia --project=code -e 'using Pkg; Pkg.instantiate()'
   julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
   ```

   `JumpHMM` installs from [varnerlab/JumpHMM.jl](https://github.com/varnerlab/JumpHMM.jl) at the commit pinned
   in `code/Manifest.toml`. The `scripts` environment is only for the figure script, `13_figures.jl`, which uses
   Plots.jl.
3. **Price data.** The prices are three daily OHLC files for the S&P 500, from Polygon.io, covering 2014-01-03 to
   2026-09-04, plus a sector file. They come from the public
   [CHEME-5660 course repository](https://github.com/varnerlab/CHEME-5660-CourseRepository-Fall-2026), in
   `code/src/data/`. Clone that repository, then run `julia scripts/00_fetch_data.jl`. The script copies the files
   into `data/` and downloads the 3-month T-bill rate (DTB3) from FRED. If the course repository is not at
   `~/Desktop/julia_work/CHEME-5660-CourseRepository-Fall-2026`, set `CHEME5660_DATA` to its `code/src/data`
   folder. `data/` is never committed.
4. **Thread settings.** Set these two variables in every shell that runs the study:

   ```
   export OMP_NUM_THREADS=1 KMP_GTID_MODE=2
   ```

   The backup solver (SCS) loads an OpenMP library that gives several Julia threads the same thread id. Without
   these settings, threaded runs crash or hang. Script 22 refuses to start unless both are set.

Every script activates its own environment, so run each one from the repository root as `julia scripts/<name>.jl`.
`-t 30` below matches the 32-core machine that made the committed results. Use `-t auto` or your own core count.

## Running the study

### 1. Real-data study

| Step | Command | Writes |
|---|---|---|
| Copy the data | `julia scripts/00_fetch_data.jl` | `data/` |
| Calibrate on 2014–2024 | `julia scripts/10_calibrate.jl` | `results/calibration.jld2`, `results/calibration_summary.toml` |
| Main grid | `julia -t 30 scripts/11_run_study.jl` | `results/tier1_summary.csv` |
| Fixed-schedule grid | `julia -t 30 scripts/11_run_study.jl --fixed` | `results/fixed_summary.csv` |
| Ticker-picker grid (4,438 runs) | `julia -t 30 scripts/11_run_study.jl --universe` | `results/universe_summary.csv` |
| CES elasticity sweep (2,562 runs) | `julia -t 30 scripts/11_run_study.jl --eta` | `results/eta_summary.csv` |
| Budget-only limits dial (17 runs) | `julia scripts/11_run_study.jl --limits` | `results/limits_summary.csv`, `results/limits_day.csv` |
| Tables | `julia scripts/12_tables.jl` | `results/tables/*.csv` |
| Figures | `julia --project=scripts scripts/13_figures.jl` | `results/figures/*.png` |

- `julia scripts/11_run_study.jl --smoke` runs two runs into `results/smoke/` as a quick check.
- Each run writes its own file to `results/runs/`. The file name is a hash of the run's settings, the market data
  and the calibration, so a rerun skips the runs it has already finished. The hash does not see code changes.
  After changing the code, rerun every grid with `--fresh` and check that the printed `skipped` count is 0.
- If a threaded run crashes, rerun the same command without `--fresh`, and it picks up where it stopped.
- `julia scripts/15_regression_check.jl [rev]` compares rerun summaries with the ones committed at git revision
  `rev` (default `HEAD`). It exits 1 unless every value matches exactly. Use it after a code change that should not
  move the results.

### 2. Synthetic stress test

Run these steps in this order. Each script's header explains what it does and what it reads.

| Step | Command | Writes |
|---|---|---|
| Fit the generator to real 2014–2024 data | `julia -t 30 scripts/20_synthetic_fit.jl` | `results/synthetic/generator.jld2` (not committed), `results/synthetic/generator_summary.toml` |
| Draw 10,000 markets and pick the typical and sell-off sets | `julia -t 30 scripts/21_synthetic_paths.jl` | `results/synthetic/paths.csv` |
| Compare the generator with the real market (22 checks) | `julia -t 30 scripts/24_synthetic_checks.jl` | `results/synthetic/checks.md`, `results/synthetic/checks.csv` |
| Run the study on each market (532 runs per market) | `julia --gcthreads=1 -t 30 scripts/22_synthetic_run.jl` | `results/synthetic/rows/` (not committed) |
| Tables | `julia scripts/23_synthetic_tables.jl` | `results/synthetic/tables.md`, `results/synthetic/tables/*.csv` |

The study run takes about 24 seconds per market on 30 threads, so the 978 markets take about 6.5 hours. To run it
overnight, use the wrapper, which relaunches the run after a crash and kills an attempt that stalls for 10 minutes:

```
caffeinate -i nohup zsh scripts/22_relaunch.sh > /dev/null 2>&1 &
```

Its logs go to `results/synthetic/rows/logs/`. `--smoke` runs the first 20 markets only.

### 3. Extra checks used by the talk and paper

| Script | What it checks | Writes |
|---|---|---|
| `scripts/dev/gamma_seeds.jl` | Reruns the γ pickers and the fixed lists over the same 30 seeds as the random pickers | `results/gamma_seeds/` |
| `scripts/dev/large_k.jl` | Whether picking by γ still beats random picks at 100 and 200 names | `results/large_k/` |
| `scripts/dev/calendar_baseline.jl` | The re-plan triggers against re-planning on the calendar alone, on real data | `results/calendar_baseline/calendar_vs_triggers.csv` |
| `scripts/dev/calendar_baseline_synthetic.jl` | The same comparison on the 400 sell-off markets (needs the step 2 rows) | `results/calendar_baseline/synthetic_vs_triggers.csv` |
| `scripts/dev/l8b_reproduction.jl` | Whether the engine reproduces the CHEME-5660 monthly Cobb–Douglas example | prints only |

Run each one as `julia -t 30 scripts/dev/<name>.jl`. The sell-off comparison also takes `--gcthreads=1`, like
script 22.

Two parts of the repository need private repositories and cannot be rerun from this one:

- `scripts/02_validate_trial.jl` and `scripts/validation/export_trial.jl` replay the live paper trial (May 20 to
  September 4, 2026) against its own order tickets. Their committed output is in `results/validation/`.
- `scripts/dev/make_engine_fixture.jl` builds the test fixture `code/test/fixtures/engine_cd_case.jld2` from the
  original allocation engine. The fixture is committed.

### 4. Slides and paper

1. `julia scripts/30_talk_data.jl` writes the data behind every slide figure to `talk/data/`, using only committed
   results, and checks every number quoted on the slides.
2. **Slides:** `cd talk && make` builds `informs2026.pdf` with XeLaTeX. It needs latexmk and the Helvetica Neue
   fonts that ship with macOS. `make notes` builds the speaker notes, and `bash talk/check_deck.sh 16` builds the
   deck and checks it: 16 pages, no overfull boxes, no missing glyphs.
3. **Paper:** `cd arxiv-paper && make` builds `Paper_v1.pdf` with pdfLaTeX and BibTeX. `make figs` redraws the
   figures from the slides' figure code and data. `make source` packs the arXiv upload (comments stripped, no `.bib`).

## Tests

```
julia --project=code code/test/runtests.jl
```

## Data limitations

- **Survivorship.** The universe holds only names with a complete 2014–2026 history. Sectors are current
  membership, not point-in-time.
- **No dividends.** Dividends are left out, for both the portfolios and SPY.
- **Splits between files.** Each of the three price files is split-adjusted only as of the date it was made. At
  each file boundary, the loader puts the earlier rows on the newest file's basis; seven names are adjusted.
  Spin-offs and ticker reuse are not adjusted. Two remain, both in the training window: IR on 2020-03-02 and NI on
  2015-07-02.
- **Exploratory screens.** The universe-wide screens and the basket sizes were designed after the first results
  were known, so that comparison is exploratory. The 13-name and 15-name fixed lists were chosen in advance,
  without 2025 data.
- **Best case for γ in the synthetic markets.** Each synthetic stock keeps its 2014–2024 average growth, so the
  stress test is a best case for picking by γ.

`docs/design-notes.md` has the details of each item.

## License

MIT License. See `LICENSE`.
