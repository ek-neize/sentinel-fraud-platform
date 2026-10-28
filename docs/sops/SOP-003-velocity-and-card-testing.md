---
id: SOP-003
title: Velocity bursts and card testing
owner: Fraud Operations (synthetic)
version: "1.4"
effective: "2019-11-01"
---
# SOP-003 Velocity bursts and card testing

## §1 Burst indicators
A compromised card is typically used many times in a short period: several purchases within
24 hours, often at different merchants and categories, with total spend far above the card's
normal level. Indicators: three or more transactions in the previous 24 hours, 24-hour spend
well above the card's usual daily spend, or only minutes between consecutive transactions.

## §2 Card testing
Fraudsters often test a card with one or more small purchases before larger ones. A purchase
far below the card's average amount, followed or preceded by large purchases within hours,
fits this pattern. Small grocery or fuel purchases at unusual hours are common test patterns.

## §3 Review steps
List every transaction on the card in the 24 hours up to the alerted transaction, with time,
merchant, category and amount. Mark which ones the model also scored above the threshold.
Compare total 24-hour spend with the card's typical daily spend over the previous 7 days.
Record these as facts in the case file; the pattern assessment is the investigator's
judgement and must be labelled as such.

## §4 Linking burst transactions
When several alerts belong to the same burst on one card, work them as one case and record
all linked alert IDs. A disposition reached for the burst applies to each linked alert only
after the investigator has checked that each transaction belongs to the same pattern.
