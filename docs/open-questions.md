# Open questions

## 2026-10-20: four decisions before results-producing runs (Tasks 13–15)

Asked after Tasks 1–12 were built and reviewed (branch `feat/eod-study`, 754 tests passing).

1. **Accept the validation?** Starting from the trial's own saved state, the simulator reproduces the trial's 05-29 decision to within 1e-6. With the trial's logged SPY inputs, 4 of 5 dates match within 1pp. 09-01 can't match: it is a knife edge, and the trial made two runs during market hours that end-of-day data can't replicate. Recommendation: yes, with 09-01 stated as a limitation in the talk.
2. **Reset the drawdown peak after a drawdown rebalance?** As written, once equity falls 8% below its peak, the trial re-solves every day. Over 2016–2024 that cost about $52k on $100k. Recommendation: yes.
3. **Use a 21-day SPY average for γ instead of a single day?** Every "go to cash" decision fell on a day SPY was down. The live trial has the same behavior. Recommendation: yes.
4. **Switch the bandit to one arm per stock?** The 3-stock-subset version can't learn: there are thousands of arms per sector and only 5,000 draws, so its baskets are no better than random. A per-stock arm that scores a stock's beat-SPY win rate, picking the top 3 per sector, keeps the idea and can learn. Recommendation: yes.

### Response (2026-10-20)

User: "question 2 clearly indicates that you are not doing what I want"

None of the four is answered. The build stays paused until the mismatch is resolved.

## 2026-10-20: resolved by restarting the design

The user restarted the design instead of answering these questions. The approved design is `docs/superpowers/specs/2026-10-20-screened-utility-panel-study-design.md`, which replaces the 10-19 design and keeps the trial's rules (drawdown peak never reset), uses a 252-day SPY growth average in γ, and re-picks the basket with single-ticker screens.

## 2026-10-25: drawdown peak reset adopted

This supersedes the line above that the design keeps the trial's never-reset drawdown peak. With that peak, the drawdown trigger fired on every day the book sat more than 8% under its peak: 88.5% of all Tier 1 decisions. A Tier 1 trial run of the reset cut decisions per run from 158 to 27 (median), left max drawdown unchanged, and cut costs by about 14%. The user adopted it: each drawdown decision sets the peak back to current equity (`drawdown_reset = true` in `config/study.toml`). Every grid is rerun under the new rule.

## 2026-11-04: test Bob's talk feedback here or in a follow-up paper?

Bob (INFORMS 2026 talk and a follow-up email) said the portfolio should move toward a net short position as the
market turns bearish. He also noted that it holds a possibly high-beta book until a trigger fires, and asked why we
rebalance every 21 days. His questions, our assessment, the reply we sent and a draft test plan (E1–E4) are in
`docs/2026-11-04-talk-feedback-bob.md`. Undecided: which tests go in a revision of this paper and which in a
follow-up paper.
