**v1 → v2.** I reviewed the 29 briefs that v1's automatic checks flagged, one by one:

- *Checks too strict (fixed in the checks, re-scored without new API calls):*
  "confirmed fraud" inside a quoted SOP condition ("unless the card is already blocked for
  confirmed fraud") and in open questions was counted as a verdict (9 briefs); signal
  statements that compared a reason code with the recorded facts were scored only against
  the signal they cited, because the schema did not let them cite facts (28 briefs). No
  number was invented in any brief.
- *Real issues (fixed in the case file and the prompt, which needs a new run):* reason codes
  say "transaction at 01:00" meaning the hour of day, which Claude read as a discrepancy
  with 01:35 (now worded "in the 01:00 hour of the day"); the prompt asked to check every
  reason code, which produced over-cautious notes ("$881 is only 1.1x the average, check
  it"); the judge found that ~15% of citations were definitions attached to procedure steps
  (now: cite the section that prescribes the step); one brief missed a prior purchase in
  the transaction list (now: check every fact before saying something is not recorded).
