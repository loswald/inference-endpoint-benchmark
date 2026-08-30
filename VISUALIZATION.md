# Visualization contract

A benchmark figure exists to support one engineering decision. If a chart needs a paragraph to
explain which lines matter, redesign it.

## Required rules

- One estimand and one unit per axis. Never combine RPM, TPM, latency, quality, and cost on one scale.
- Compare matched endpoint × workload cells only. Do not rank heterogeneous global averages.
- Use endpoint small multiples or aligned horizontal dot plots; never connect AIMD epochs from
  different endpoints or phases into a looping trajectory.
- Put the sampling unit and interval meaning in the caption: request, epoch, block, or matched pair.
- Show right-censoring explicitly with an arrow or “tested lower bound.” A missing knee is not zero.
- Leave missing cells blank and label why. Do not draw zero bars for untested or inconclusive cells.
- Use log axes only when orders of magnitude matter; show major ticks in ordinary engineering units.
- Direct-label important series where possible. Keep palettes color-blind-safe and reserve red for
  actual failure, not merely a low rank.
- Draw stable-prefix and cache-cold time observations as separate series. Never connect a point in
  one cache stratum to a point in another.
- Show qualified outliers; do not silently trim, winsorize, clip, or hide them.
- Every public PDF page must be rendered and inspected at normal reading size before publication.

## Surprising-number audit

Before displaying an extreme tokens/second or latency value, bind it to one request and verify:

1. usage is provider-reported, positive, and belongs to one choice rather than an aggregate;
2. TTFT is a streamed content event, not buffered headers or the final response;
3. timestamps are monotonic and use one clock and one unit;
4. the post-TTFT denominator spans at least one second; shorter bursts are audited but censored;
5. hidden reasoning is zero or explicitly stratified for visible decode comparisons;
6. retries, queue delay, and drain are included in the correct end-to-end estimand;
7. no token count was estimated from bytes and then labelled provider usage;
8. the observation remains traceable in the audit table even when excluded from one metric.

The report should name a rate an end-to-end proxy unless the provider exposes the corresponding
server-internal timing. A qualified extreme may be real; the audit exists to distinguish that from a
denominator or aggregation bug.

## Recommended figure set

For each campaign, prefer a compact set:

1. coverage by endpoint and dimension;
2. highest confirmed AIMD lower bound by matched workload;
3. achieved fixed-rate stability RPM with pass/fail/gated status and block-level intervals;
4. effective input TPM and output TPM in separate figures;
5. TTFT and end-to-end latency by matched workload;
6. documented versus observed context/output boundaries;
7. quality change at low load versus sustained load;
8. timing/outlier qualification counts.

Tables remain better than charts for exact capability states, validation reasons, documented limits,
and per-endpoint production instructions.

Never color a fixed-rate test as successful merely because execution completed. Success requires the
registered acceptance predicate. Measured failures and tests that could not start reliably need their
own explicit states. Contiguous time blocks are not independent repeats and their intervals must be
labelled exploratory.
