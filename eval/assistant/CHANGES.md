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

**v2 result and what is left.** The judge's "cited section supports the step" rose from 85%
to 95% and unsupported statements fell from 1.1% to 0.6%, with every brief passing the
automatic checks. The trade-off: citation recall against the gold SOPs fell from 100% to
82%, almost all of it SOP-007 (night-time, 18 of 36 cases). With the stricter citation
rule Claude often skips SOP-007 §3 ("record which time indicators are present and how they
combine"), which is a real omission, while part of the drop is the gold rule counting an
indicator-only SOP as required. Of about 465 statements the judge flagged 3: one real error
in the cautious direction (a 24-hour transaction count labelled "context only"), one on a
deliberately contradictory adversarial reason code, and one strict reading of a hedge.
Next candidates: a dedicated "indicators present" list in the brief schema, and
expert-labelled gold SOPs instead of rules.
