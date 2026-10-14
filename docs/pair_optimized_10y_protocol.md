# Pair-optimized 10-year / 3-year rolling protocol

This document freezes the V6 research protocol defined in
`configs/pair_optimized_10y_3y.json`. The study changes the optimization target
from an aggregate portfolio return to the net return of each equity-pair sleeve.
It is additive: none of the earlier reports or their artifacts are inputs to a
new performance claim, except for the frozen S&P 500 constituent/GICS snapshot.

## Timeline and rolling decisions

The market-data request covers 2016-08-26 through 2026-08-26, with 2026-08-27
as the download's exclusive end date. Yahoo adjusted closes are downloaded as a
complete ten-year snapshot rather than prepending two years to the old file;
historical adjustments can be revised by the provider.

Every decision uses the actual Yahoo sessions in the half-open interval from
the rebalance date minus three calendar years through the session immediately
before the rebalance date. The observation count therefore varies with the
calendar and available data; it is not forced to 756. Training is rolling, not
expanding. Rebalances occur on a calendar-quarter cadence. Each outer evaluation
starts on its rebalance date and ends immediately before the next calendar-
quarter boundary, using however many Yahoo sessions actually occur in that
interval. The final interval is truncated at the study end. Positions start
flat for each outer pair vintage and are liquidated, with cost, at its terminal
boundary.

At every rebalance, all different-issuer within-sector pairs are screened again.
The tradable set contains only pairs passing the strict 5% within-sector
Benjamini-Hochberg FDR rule together with the stored I(1), positive hedge-ratio,
half-life and Hurst gates. The minimum price-observation count is 90% of the
actual training-window session count, rounded upward. A shortage of eligible
pairs is valid: the study never inserts a quota filler or an exploratory
fallback merely to reach ten names.

## Pair-level optimization objective

Each eligible pair is its own independently capitalized, one-times-gross
sleeve. Hyperparameters are selected by maximizing that pair's net CAGR in
purged chronological validation data. No portfolio CAGR, covariance allocator,
sector quota or shared-name cap enters the primary ranking. After tuning, up to
ten strict pairs with the strongest training-only scores are evaluated during
the next outer quarter. If fewer than ten satisfy the gates and activity
requirements, fewer are evaluated.

The 96 predeclared signal rules combine:

- SMA and EWMA z-scores;
- 10, 20, 40 and 60-session lookbacks;
- 1.25, 1.75 and 2.25 entry thresholds;
- 0.25 and 0.50 exit thresholds; and
- 10 and 30-session maximum holding periods.

The z-score stop remains fixed at 4.0. The implemented warm-ups are 10, 16, 32
and 48 observations for the 10, 20, 40 and 60-session lookbacks, respectively:
the lookback itself when it is at most 10, otherwise 80% of the lookback rounded
upward. A fixed SMA-60, 2.0-entry, 0.5-exit, 60-session-hold rule remains the
baseline comparator and retains its original 40-observation warm-up.

Within each calendar-exact three-year outer training window, inner folds begin
with 378 training sessions and use 63-session validation blocks stepped forward
by 63 sessions. A one-session purge protects the next-session return label; no
additional embargo is needed because validation blocks do not overlap. The
spread relation is refit using each fold's training dates, every validation fold
starts flat, and every terminal position is liquidated. These state and cost
boundaries are essential—a trailing indicator alone does not make a fold
auditable.

Rules need at least ten trades in the full outer training window, six pooled
inner-validation trades, and activity in at least three inner folds. The winner
uses a deterministic one-standard-error rule rather than mechanically accepting
the noisiest maximum. Among configurations within one standard error of the raw
best, the implemented order favors lower validation-fold annual-log-return
dispersion, lower annualized turnover, shallower drawdown, higher pooled
validation net CAGR and finally the stable parameter ID.

## Cost convention

The primary result charges 2 basis points per dollar traded on one-way absolute
leg turnover. This is the conservative end of the assumed 1–2 bp range. A 1 bp
sensitivity reruns the same selected pairs and frozen parameters; costs are not
used to retune a second winner. Launch trades and terminal liquidations are
charged, and short exposure retains the existing 30 bp annual borrow proxy.
Cash earns zero in the pair-sleeve comparison.

This flat rate is a scenario, not an execution-grade estimate. Yahoo adjusted
closes do not provide historical bid/ask quotes, market impact, locates or
security-specific borrow fees. Lowering the flat cost therefore does not remove
those omitted frictions.

## Leakage and interpretation controls

- Each formation set contains only actual Yahoo sessions in `[rebalance date −
  3 calendar years, rebalance date)`, and screening, pair ranking and
  hyperparameter selection use only those dates.
- Each outer evaluation is `[rebalance date, next calendar-quarter boundary)`;
  no fixed 63-observation outer window is manufactured.
- Hedge coefficients are refit inside each inner training fold and once on the
  full rolling training window before the outer quarter.
- The primary 2 bp cost, the grid and all activity thresholds are frozen before
  outer evaluation.
- The 1 bp sensitivity reuses the primary selections and parameters.
- Full-ten-year cointegration or return rankings may be descriptive only; they
  cannot select a traded pair.
- FRED series are reserved for post-hoc regime attribution. Standard FRED
  downloads contain revised observations rather than a point-in-time vintage.
- Rolling-screen caches must be keyed by formation bounds, the universe hash and
  the screening-settings hash, so an eight-year or differently configured cache
  cannot be silently reused.

The primary output is a pair-vintage ledger and linked individual-pair metrics.
This study does not calculate or report an aggregate portfolio curve: combining
pair sleeves would introduce an allocation decision outside the study's stated
pair-return objective. Likewise, a leaderboard formed after viewing all outer
returns is descriptive; it is not a deployable selection rule.

## Remaining credibility limits

Ten years of prices does not correct the frozen universe's survivorship and
current-GICS bias. Delisted securities, historical membership and historical
industry classifications remain absent. Quarterly rolling windows and repeated
pair/grid tests are dependent, so naive pair-count significance is invalid.
Hyperparameter search also creates winner's-curse risk even with chronological
folds and a one-standard-error rule. Results require parameter-stability,
baseline and data-snooping diagnostics, followed by a genuinely new paper-
trading interval before any deployment decision.
