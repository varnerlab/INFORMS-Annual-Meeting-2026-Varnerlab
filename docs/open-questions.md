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
