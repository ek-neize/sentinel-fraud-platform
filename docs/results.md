# Sentinel — test-period results

Held-out test period **2020-07-01 to 2020-12-31**: 525,661 transactions, 2,012 frauds (~10.9/day). Thresholds were fixed on validation (Apr–Jun 2020) and applied unchanged. 95% intervals: day-level bootstrap.

## Headline

With a review budget of **20 alerts/day**, the XGBoost champion caught **89.3%** of fraud (95% CI 85.8%–92.4%) and **93.0% of fraud dollars**, with precision 66.8% (CI 62.9%–70.2%) and 1.7 false positives per 1,000 transactions. Net benefit vs. no model: **$574,827** (CI $509,925–$634,923); the watchlist rule with the same budget caught 20.4% for $103,208.

## Strategies at the operating budget

| strategy                                       |   alerts_per_day | recall   | precision   | amount_recall   |   fp_per_1000_txns | cost_total   | net_benefit   |
|:-----------------------------------------------|-----------------:|:---------|:------------|:----------------|-------------------:|:-------------|:--------------|
| No model (every fraud missed)                  |              0   | 0.0%     | –           | 0.0%            |                0   | $931,222     | $0            |
| Watchlist rule: review every hit               |             89.7 | 25.6%    | 3.1%        | 29.3%           |               30.4 | $1,216,587   | $-285,365     |
| Watchlist rule: top 20/day by amount           |             20   | 20.4%    | 11.1%       | 29.1%           |                6.2 | $828,015     | $103,208      |
| Logistic regression: threshold, max 20/day     |             15.1 | 68.8%    | 49.9%       | 84.3%           |                2.6 | $407,151     | $524,071      |
| XGBoost champion: threshold, max 20/day        |             14.6 | 89.3%    | 66.8%       | 93.0%           |                1.7 | $356,395     | $574,827      |
| XGBoost champion: threshold, no cap (overtime) |             16.7 | 96.5%    | 63.3%       | 98.9%           |                2.1 | $328,315     | $602,908      |
| XGBoost champion: daily top-20 (upper bound)   |             20   | 89.8%    | 49.1%       | 93.5%           |                3.6 | $383,010     | $548,213      |

## Capacity curve (champion, deployable policy)

|   capacity_k | recall   | precision   | amount_recall   |   fp_per_1000_txns | net_benefit   |
|-------------:|:---------|:------------|:----------------|-------------------:|:--------------|
|            5 | 22.6%    | 100.0%      | 27.8%           |                0   | $187,148      |
|           10 | 53.5%    | 99.4%       | 63.0%           |                0   | $419,065      |
|           15 | 79.1%    | 88.2%       | 84.3%           |                0.4 | $542,111      |
|           20 | 89.3%    | 66.8%       | 93.0%           |                1.7 | $574,827      |
|           25 | 93.6%    | 52.8%       | 96.8%           |                3.2 | $574,339      |
|           30 | 96.2%    | 43.6%       | 98.6%           |                4.8 | $559,584      |
|           40 | 98.2%    | 32.4%       | 99.7%           |                7.8 | $516,903      |
|           50 | 98.9%    | 25.4%       | 99.8%           |               11.1 | $465,294      |
|           75 | 99.1%    | 16.6%       | 99.9%           |               19.1 | $339,145      |
|          100 | 99.4%    | 12.4%       | 100.0%          |               27   | $215,330      |

Operating budget = highest net benefit (ties → smaller team): **20/day**.

## Busy days

The fixed threshold exceeded 20 alerts on 49 of 184 days (max 44/day). Capping review at 20/day left 379 alerts unreviewed, of which 145 were fraud.

## Fairness check (protected attributes are not model inputs)

| attribute   | group   |   txns | fraud_rate   | alert_rate   | recall   | false_positive_rate   |   alert_rate_ratio |   fpr_ratio |
|:------------|:--------|-------:|:-------------|:-------------|:---------|:----------------------|-------------------:|------------:|
| age_band    | 30-44   | 175166 | 0.35%        | 0.46%        | 88.2%    | 0.15%                 |               0.9  |        0.91 |
| age_band    | 45-59   | 145709 | 0.39%        | 0.51%        | 87.7%    | 0.17%                 |               1    |        0.98 |
| age_band    | 60+     | 116785 | 0.45%        | 0.60%        | 92.4%    | 0.18%                 |               1.16 |        1.06 |
| age_band    | <30     |  88001 | 0.35%        | 0.51%        | 89.3%    | 0.19%                 |               0.99 |        1.13 |
| gender      | F       | 288312 | 0.37%        | 0.50%        | 85.2%    | 0.19%                 |               0.98 |        1.09 |
| gender      | M       | 237349 | 0.40%        | 0.52%        | 94.0%    | 0.15%                 |               1.02 |        0.89 |

Ratios compare each group with the overall rate; outside 0.8–1.25 warrants review. Differences in alert rate partly reflect different fraud rates per group.

## Cost assumptions (config/assumptions.yaml)

Review $25/alert · escalation $75/caught fraud · customer friction $5/false positive · missed fraud loses 85.0% of its amount + $15 · caught fraud recovers 90.0%.

*Synthetic data (Sparkov): fraud patterns are cleaner than in production, so absolute numbers are optimistic; the method and the comparisons are the point.*
