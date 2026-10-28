---
id: SOP-002
title: Interpreting model scores and reason codes
owner: Fraud Analytics (synthetic)
version: "2.0"
effective: "2020-04-01"
---
# SOP-002 Interpreting model scores and reason codes

## §1 What the score means
The model score is an estimated probability that a transaction is fraudulent, based on
patterns in historical labelled data. It is a model signal, not evidence. A score of 0.99
means transactions like this one were usually fraud in the past; it does not establish that
this transaction is fraud. Never record the score as a finding.

## §2 Reason codes
Each alert shows up to three reason codes: the inputs that pushed this transaction's score up
the most. Reason codes describe why the model reacted. Each reason that refers to a recorded
value (amount, recent spend, time since the previous transaction, watchlist membership) must
be checked against the card's transaction records before it is used in a case note. If the
records do not match the reason code, note the discrepancy and report it to Fraud Analytics.

## §3 Context-only signals
Some inputs carry little information on their own and must be treated as context only, never
as a reason to escalate or confirm: customer city population, customer state, day of week,
the number of earlier transactions on the card, and distance between the customer's home and
the merchant. When one of these appears as a reason code, mention it only as background and
do not rely on it.

## §4 Threshold and risk bands
The operating threshold is set from validation data so that the expected alert volume matches
the review team's capacity. Risk bands shown in tools: High (score at or above the larger of
0.90 and the threshold), Elevated (at or above the threshold), Low (below the threshold).
Bands help order work; they do not change the evidence required for a decision.
