---
id: SOP-004
title: Card-not-present (online) transactions
owner: Fraud Operations (synthetic)
version: "1.1"
effective: "2019-09-01"
---
# SOP-004 Card-not-present (online) transactions

## §1 Indicators
Card-not-present (CNP) transactions are those in online categories (category names ending in
`_net`, such as `shopping_net`, `misc_net` and `grocery_net`). CNP fraud needs only the card
details, not the physical card, so it is common after a data breach. CNP purchases that are
large, out of pattern for the card, or part of a burst deserve particular attention.

## §2 Verification
Device, IP address and shipping address are not available in the Sentinel case data. Do not
infer them. If the merchant participates in the merchant-verification programme, request the
order details through the merchant channel and record the response as a fact with its date.

## §3 Customer confirmation
For CNP alerts above $500, or CNP alerts that are part of a burst, contact the customer using
SOP-008 before deciding, unless the card is already blocked for confirmed fraud.
