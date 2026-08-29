# DigitalOcean six-hour variation and gap-closure plan

This is a plan-only package. It has not contacted DigitalOcean and it is not a record of a live
run.

The guaranteed core is seven matched low-load panels across exactly six hours. Each panel sends
four samples for every one of the 11 DigitalOcean-hosted models and four workload shapes: short
input/short output, 100K input/short output (50K for the registered MiniMax route), short
input/long output, and a deterministic mixed workload. Two samples reuse stable exact prompts
across panels; two use panel-unique prompts so automatic prefix caching cannot masquerade as
time variation. Every row records that cache condition. That is 176 open-loop arrivals per panel
and 1,232 requests across the study. The 176 launches span 175 seconds at one request per second;
with the 360-second hard full-stream timeout, every panel has a conservative 535-second drain
bound inside its 600-second deadline. Each endpoint/workload cell receives 28 observations: 14
stable-prompt and 14 panel-unique cache-cold observations.

Panel starts are hourly from hour 0 through hour 6. The process has a separate bounded drain and
finalization tail of at most 15 minutes so the hour-6 streams can finish cleanly; that tail is not
presented as additional time-of-day evidence.

Gap work is deliberately secondary. Between protected panels, the runner attempts only the
registered unresolved cells: 25 capacity cells, six caching gaps, and the unresolved capability,
context, output-length, and vision checks. A job starts only when its strict bound fits before the
next panel guard. Any gap still pending at the cutoff is labeled untested; it does not turn a
completed six-hour study into a failed campaign.

The historical fixed-rate rows are retained as existing evidence but are not rerun in this live
plan. They used an older exact workload recipe; applying their selection to the corrected 100K and
newer recipes would measure different cells and silently relabel them. The 100K input recipe is a
distinct workload identity and is never presented as the historical 32K recipe.

Regenerate and validate the package without credentials or network traffic:

```text
python -m inference_bench.cli plan-digitalocean-closure examples/digitalocean-hosted-2026-08-27.yaml reports/digitalocean --output examples/digitalocean-six-hour-gap-closure
python -m inference_bench.cli plan examples/digitalocean-six-hour-gap-closure/digitalocean-six-hour-gap-closure.yaml
```

The immutable selection, source hashes, timing proof, request ceilings, and conservative cost
ceiling are in `plan.json`. Live execution remains a separate, explicit command and is not part of
this package.
