---
id: SOP-012
title: False-positive closure and feedback
owner: Fraud Operations (synthetic)
version: "1.1"
effective: "2020-02-15"
---
# SOP-012 False-positive closure and feedback

## §1 Closing criteria
An alert may be closed as `cleared` when the records show a purchase consistent with the
card's history (similar amounts, categories and timing), there is no burst, no hot-list hit,
and no unresolved indicator. A high model score alone is not a reason to keep a case open.

## §2 Feedback
Record the main reason for clearing (for example "regular grocery purchase, in pattern").
Cleared alerts feed back to Fraud Analytics. Labels can change later: chargebacks can arrive
up to 60 days after a transaction, and a cleared alert that later receives a chargeback is
reopened and reviewed.
