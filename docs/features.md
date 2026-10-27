# Feature dictionary (gold layer)

Built by `notebooks/databricks/04_gold_features.py`; mirrored in
`src/sentinel/features/reference.py` for testing.

**Point-in-time rule:** every history feature looks only at the same card's transactions
*strictly before* the current one. Window frames end at `-1` (previous row, or one second
before). Transactions in the same second as the current one are excluded from time windows.
CI enforces this: Spark output must equal the independent pandas implementation, and
changing future transactions must not change any earlier row's features.

## Amount vs. the card's own history

| Feature | Definition | Why it matters |
|---|---|---|
| `amount`, `log_amount` | transaction amount, `log1p(amount)` | fraud skews to larger amounts |
| `card_prior_txn_count` | number of earlier transactions on the card | cold-start indicator; ratios are noisy when small |
| `card_prior_amt_mean` / `_std` / `_max` | mean, sample std, max of earlier amounts | the card's normal spending |
| `amt_to_prior_mean` | `amount / card_prior_amt_mean` | "unusually high for this customer" |
| `amt_to_prior_max` | `amount / card_prior_amt_max` | > 1 means a new personal maximum |
| `amt_zscore` | `(amount − mean) / std` (null if std = 0 or < 2 prior txns) | standardised surprise |

## Velocity (the burst signal)

| Feature | Definition |
|---|---|
| `cnt_1h`, `cnt_24h`, `cnt_7d` | count of the card's earlier transactions in the last 1 h / 24 h / 7 d |
| `amt_1h`, `amt_24h`, `amt_7d` | sum of their amounts |
| `secs_since_prev_txn` | seconds since the card's previous transaction (null for the first) |

Each compromised card in this data has one burst of ~10 frauds over ~2 days, so these are
expected to be the strongest features.

## Time, place, customer

| Feature | Definition |
|---|---|
| `hour_of_day`, `day_of_week` | UTC hour; Spark `dayofweek` (1 = Sunday … 7 = Saturday) |
| `is_night` | 1 if hour ≥ 22 or < 4 |
| `home_merch_km` | haversine distance between customer home and merchant location |
| `age_at_txn` | customer age at transaction time — **protected: fairness checks only, not a model input** |
| `log_city_pop` | `log1p(city_pop)` |
| `category`, `state` | categorical; one-hot encoded in the modelling step |
| `gender` | **protected: fairness checks only, not a model input** |

## Watchlist (point-in-time join in silver)

| Feature | Definition |
|---|---|
| `wl_card_hit` | card is on the hot-list at transaction time |
| `wl_merchant_hit` | merchant is on the risk list at transaction time |
| `wl_any_hit` | either of the above |

## Labels (in `gold_training_set` only, never in `gold_features`)

| Column | Meaning | Use |
|---|---|---|
| `label_at_cutoff` | fraud label known by the training cutoff (end of the gap month) | **training** |
| `label_final` | every fraud label known now | **evaluation only** |
| `label_known_ts`, `label_source` | when and how the label arrived | analysis of label delay |
| `split` | train / gap / valid / test / out_of_range by event time | chronological split |

Deliberately **not** used: `merchant_id` / `merchant_name` as raw features (high-cardinality,
memorises merchants), `job` and `city` (high-cardinality, weak and potentially sensitive), and
any label-derived aggregate such as a merchant's historical fraud rate (would need careful
as-of handling; the watchlist already provides that signal point-in-time).
