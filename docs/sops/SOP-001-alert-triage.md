---
id: SOP-001
title: Alert triage and case opening
owner: Fraud Operations (synthetic)
version: "1.3"
effective: "2020-01-15"
---
# SOP-001 Alert triage and case opening

## §1 Scope
Applies to every alert raised by the transaction-monitoring model (Sentinel) or by a
watchlist rule. An alert is a request to review one transaction; it is not a finding of
fraud. All alerts are worked in the investigation queue in the order set by §2.

## §2 Order of work
Investigators take the open alert with the highest priority first. Priority is the model's
fraud probability unless operations has switched the queue to expected loss. The team's
daily review budget is fixed; alerts not reviewed on their first day stay in the backlog and
expire unreviewed after 3 days. Before taking a new alert, check whether another open alert
exists on the same card: if so, work them together as one case (see SOP-003 §4).

## §3 Opening a case
Record in the case file: alert ID, transaction ID, card ID, transaction time (UTC), amount,
merchant name and category, the model score and threshold in force, and the reason codes
shown with the alert. Record the card's transactions for the previous 7 days up to and
including the alerted transaction. Do not record information that was not available at the
time of the alert (for example, later transactions or chargebacks received afterwards).

## §4 Time targets
High-risk alerts (score at or above 0.90) should be reviewed the same day. A case that cannot
be resolved within 24 hours because information is pending (for example, the customer has
not responded) stays open with a note of what is outstanding.
