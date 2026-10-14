# Point-in-time pair-research data contract

This additive study rejects current-constituent and current-classification
substitutes. The full historical replay remains blocked until all seven files
below pass `audit_point_in_time_contract` and the unchanged V6 engine has been
replayed on those inputs.

## Required files

Place licensed extracts under `data/point_in_time/` using the filenames in
`configs/pair_identity_pit_selection_correction.json`.

| File | Required content |
| --- | --- |
| `security_master.csv` | Permanent security and issuer identifiers, effective-dated names, share class and exchange |
| `sp500_membership_history.csv` | Effective-dated S&P 500 membership and source release date |
| `gics_history.csv` | Effective-dated issuer GICS sector/sub-industry and source release date |
| `daily_security_returns.csv.gz` | Daily total returns including distributions, with delisting-return flag |
| `delisting_events.csv` | Delisting return, reason, payment and successor security |
| `corporate_actions.csv` | Splits, distributions, mergers and successor mappings |
| `pair_first_knowable_dates.csv` | Frozen pair identity, first knowable date, rule and provenance |

Exact column requirements are defined in `src/equity_pairs/pit_data.py`. The
audit also rejects invalid dates, duplicate keys, overlapping effective-date
intervals, unknown security identifiers and insufficient date coverage.

## Information timing

- A membership or GICS record can be used only after its `source_release_date`.
- A pair holds cash before `first_knowable_date`; later-discovered identities
  are never backfilled.
- Delisted securities and merger successors remain in the history through the
  event return and exit mechanics.
- A valid zero-signal quarter is distinct from unavailable data.

## Inference

The selection layer uses synchronized four-quarter circular blocks and
anniversary-year blocks. Each draw resamples the same time blocks for all 114
pairs, reconstructs every path and reranks the full family. Romano-Wolf max-t
is the primary pair-level family-wise correction; White's Reality Check is a
global sensitivity test.

These corrections are conditional on the frozen pair family. They do not cure
researcher-path selection or an ex-post universe definition.

## Run

```bash
PYTHONPATH=src python scripts/run_pit_selection_correction.py \
  --config configs/pair_identity_pit_selection_correction.json
```

Until the licensed data and exact PIT V6 replay both pass, every
`promotion_eligible` value must remain `false`.
