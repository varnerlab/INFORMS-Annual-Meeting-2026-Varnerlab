# INFORMS 2026 talk feedback: Bob's questions and our response

Recorded 2026-11-04. Bob raised these questions at the talk and in a follow-up email. Our assessment is checked
against the code (`code/src/`) and the results files. The reply we sent is below, and a draft test plan follows it.
**Still open:** whether the tests belong in a revision of this paper (arXiv 2611.03022) or in a follow-up paper.

## Questions and assessment

1. **Is a cash term missing from the budget constraint?** Not mathematically. Cash is the slack of
   `Σ nᵢSᵢ ≤ W`: `c* = W − Σ nᵢSᵢ`, which the forecast already uses (Eq. mpc-value). Writing
   `Σ nᵢSᵢ + c = W, c ≥ 0` would make it visible. Trading costs are not in the budget. Targets are priced at the
   close, and the fill at the next open scales buys so that `c ≥ 0`.

2. **At a re-plan, do we liquidate and rebuy, or does the turnover limit mix the old and new books?** The new
   book is the new optimum.
   - The orders are `n* − n_prev`, and held names outside the new name set are sold. At equal prices this is
     equivalent to selling everything and rebuying, except that only the changed shares pay the spread.
   - The "turnover" limit is really a cap on spread cost, `5 bps × S̄ × Σ|Δn| ≤ 0.10 W`. A full round trip costs
     about 0.1% of W, so the cap never binds (`Allocators.jl:378–395`).
   - No utility depends on `n_prev`.
   - Known untidiness: the limit uses the mean price `S̄` rather than each `Sᵢ`, and it counts only the solver's
     free names.

3. **In a bear market, the book can stay high-beta until a trigger fires.** This is valid, and it's the main point.
   - The beta cap `β̄ = max(0, 1 − ξ)` applies only to the targets of a re-plan. Between re-plans the shares are
     held and the book's beta is not controlled.
   - The band exit catches fast sell-offs and the 8% drawdown stop catches large ones. But a slow decline that
     stays inside the band (which widens like √τ) can be held for up to 21 trading days.
   - None of the triggers looks at the market signal ξ.
   - The paper and talk wording was fixed on branch `docs/hedge-wording-review-fixes`.

4. **Why 21 days? Why not daily or every minute?** We picked it (one trading month) and never tuned it.
   - It is the longest hold, not a fixed schedule. In the 4,200 main-comparison runs, 39% of re-plans after the
     first came early: 28% from band exits and 11% from drawdown stops. Another 56% came at the 21-day horizon and
     5% from basket changes on the 21-day re-pick timer.
   - Arguments against much faster rebalancing: the estimates change slowly (63-day half-life, 252-day market
     average), and costs grow with frequency. Davis–Norman also says a no-trade region beats continuous trading
     under proportional costs. The model uses daily closes only.
   - Bob was not worried about costs.

5. **(Email) The algorithm would be better if it morphed into a short position as the market became more
   bearish.** It doesn't, for two reasons.
   - The beta cap stops at 0, so a re-plan can push the book toward neutral but never net short.
   - The 130/30 rule shorts up to 30%, but its shorts are stock picks: the most negative γ per sector, at a fixed
     cap that does not depend on ξ. Its shorts also change only at a trigger.
   - With no positive score, every rule goes to cash, not short.

## Reply sent (2026-11-04)

> Hi Bob —
>
> It was great to chat with you - when this draft gets cleaned up, I'll be happy to send it along.
>
> Re transition -
> That is a super interesting suggestion. Right now, the algorithm doesn't respond directly to a falling market.
> It only changes the portfolio when a trigger fires: a month has passed, the portfolio's value has strayed from
> what we forecast, or the portfolio has lost 8% from its peak. Until then, it holds what it has, even if that's
> a high-risk portfolio in a falling market.
>
> When a trigger fires and the hedged variant is active, the new portfolio must stay under a market-exposure limit
> that is lower when the market is weaker. It meets that limit by holding cash or a few stocks that don't move
> with the market or move against it. But the limit stops at zero, so it can push the portfolio toward neutral but
> never toward net short.
>
> We did test a long/short version (a standard 130/30 design), where we short up to 30% of the portfolio in stocks
> our score expects to fall. But the short side doesn't grow as the market weakens, and it also only changes when
> a trigger fires.
>
> There are a few directions to explore: First, we could add a trigger that rebalances as soon as our market
> signal turns bearish, so the portfolio doesn't sit exposed until month-end. Then we could build on the
> long/short version so the short side grows and the portfolio can go net short as the signal gets more bearish.
> Our market signal lags the market, so a short position could still be in place when the market turns back up.
>
> Another point you raised is also interesting: why we wait 21 trading days. We picked it and never tuned it.
> About 40% of our rebalances already came early because a trigger fired, but we could test shorter and longer
> limits too.
>
> Thank you for taking the time to speak with me; there are many new directions to explore.
>
> Stay in touch.
>
> cheers — jeff

## Draft test plan

Each test runs on real prices (2025-01-02 to 2026-09-04) and on the three synthetic sets: typical, jump bursts
(burst stress) and steady decline (bear stress). It uses the main comparison's selection and weighting rules,
hedge on, 30 seeds, and pairs each run with the current design under the same seed.

Report for each test:
- terminal wealth
- maximum drawdown
- realized beta
- number of re-plans by trigger
- trading cost
- days on which the held book's beta exceeded the current cap

Report the jump-bursts set separately, since its paths include rebounds.

- **E1. Beta-violation trigger (Bob's main point).**
  - Each day, compute the held book's beta `Σ βᵢ(t) wᵢ(t)` with the current online loadings and weights.
  - Re-plan when it exceeds `β̄(ξₜ)` by a margin δ (e.g. 0, 0.1, 0.2).
  - Variant: re-plan when ξ crosses zero upward.
  - Small change: one new reason in `check_trigger` (`code/src/Triggers.jl`) plus the daily beta, which
    `StudyBacktest.jl` already computes as `held_β`.
- **E2. Net-short beta cap.**
  - Let the cap go negative: `β̄ = max(β_min, 1 − ξ)` with `β_min` in {0, −0.25, −0.5}.
  - Instruments: (a) short SPY as a single hedge instrument, with the borrow fee; (b) a 130/30 whose short limit
    grows with ξ.
  - Main risk: ξ lags (EMA21/EMA63), so a short can still be on in a rebound. Measure losses on rebounds directly.
  - Run with E1. Without the new trigger, a negative cap only acts at the next scheduled re-plan.
- **E3. Horizon sweep.**
  - T ∈ {5, 10, 21, 42, 63} trading days, triggered and calendar-only (the calendar-only setup exists:
    `scripts/dev/calendar_baseline.jl`).
  - T now sets three things: forecast horizon, longest hold and re-pick interval. Vary them separately as well as
    together.
- **E4. Paper housekeeping (no new runs).**
  - Write cash explicitly in the budget (`Σ nᵢSᵢ + c = W, c ≥ 0`).
  - Rename the turnover limit as a spread-cost cap, or drop it. It never binds.

**This paper or the next?** A suggestion for discussion:
- E4 and possibly E1 fit a revision of this paper. E1 closes a stated limitation with one new trigger.
- E2 and E3 change the design and need new grids, so they fit a follow-up paper on market-state control.
