---
id: SOP-007
title: Time-of-day and location anomalies
owner: Fraud Operations (synthetic)
version: "1.0"
effective: "2019-08-01"
---
# SOP-007 Time-of-day and location anomalies

## §1 Night-time activity
Purchases between 22:00 and 04:00 (UTC, as recorded) are more often fraudulent than daytime
purchases in historical data. Night-time is an indicator to combine with others (bursts,
out-of-pattern amounts); many customers legitimately shop at night.

## §2 Distance from home
The distance between the customer's home and the merchant is recorded, but in the current
data it carries little information and is treated as context only (see SOP-002 §3). Do not
escalate on distance alone.

## §3 Combining indicators
Record which time and location indicators are present as facts, and state in the case note
how they combine with the other indicators. A single weak indicator is never sufficient.
