# Methods

## Route identity

A route is more specific than a model name. The benchmark binds every request to provider, adapter,
base URL, model identifier/version, API family/version, region, service-specific upstream, pricing,
and declared limits. Changing any serving field changes the route identity hash.

## Workload design

The default campaign combines:

1. low-load latency baselines for four input/output shapes;
2. capability requests for streaming, non-streaming, tools, JSON output, vision, log probabilities,
   stop, seed, and parameter boundaries;
3. a strength-two covering array for sampling/stream/output interactions;
4. context anchors at 1%, 10%, 25%, 50%, 75%, 90%, 95%, 99%, and just over the documented limit,
   with separated retrieval markers;
5. requested-output anchors from 32 through the documented maximum and one value above;
6. deterministic reasoning, instruction, extraction, and retrieval quality tasks;
7. matched cached/uncached-prefix trials;
8. open-loop AIMD saturation and endpoint-isolated sustained soaks.

The planner is deterministic under the campaign seed. Comparable routes receive identical target
shapes. Provider-reported usage is authoritative; synthetic prompt “token” counts are scheduling and
reservation estimates because tokenizer behavior can differ.

## AIMD

Each route and workload shape is isolated from other capacity traffic. Poisson arrivals are scheduled
from an epoch clock independently of completions. A semaphore is only a safety ceiling: waiting at the
ceiling produces measured queue delay.

After a healthy epoch, offered RPS increases by a fixed additive amount. After two unhealthy epochs,
it is multiplied by the configured decrease factor. The highest healthy candidate receives three
confirmation epochs separated by low-load epochs. A 50%-rate recovery epoch is run and labelled only
when a two-epoch overload was actually observed.

Default epoch health requires:

- at least 99% of completed logical requests ending successfully;
- no more than 1% of physical attempts rate-limited, including intermediate retries;
- no more than 1% of physical attempts ending in a server error, timeout, or transport error;
- p95 TTFT and end-to-end latency no greater than twice their low-load baselines when those baseline
  metrics are observable;
- end-of-epoch drain no larger than 10% of the epoch or one second.

These short epochs locate a knee. They do not establish sustained capacity.

## Sustained soaks

A soak consists of four 30-second analysis blocks by default. The runner records the tested rate and
block health, but the report does not automatically certify sustainable capacity. Capacity above or
below the tested rate is not inferred. The current runner also does not construct paired low-load and
near-load quality trials, so low-load quality scores must not be interpreted as evidence that quality
is unchanged under saturation.

## Metrics

All clocks are client-side monotonic clocks. Units are stored and printed explicitly.

- **TTFT:** request start to first content-bearing SSE event.
- **End-to-end latency:** request start through complete response drain.
- **Decode proxy:** provider-reported completion tokens divided by `(end-to-end seconds − TTFT)`.
  This includes client transport/drain overhead and is not direct server decode compute. The
  primary proxy requires at least eight billed completion tokens, two content-bearing events, and
  10 ms after TTFT; observations failing the gate remain recorded but are not headline decode data.
- **Offered RPM:** scheduled arrivals divided by the scheduled arrival-window minutes.
- **Completed/effective RPM:** completed or successful requests divided by full block wall minutes,
  including response drain after the arrival window.
- **Physical-attempt RPM:** every provider send, including intermediate retries, divided by the same
  full block wall minutes. Attempt-level 429, server-error, timeout, and transport-error counts are
  reported separately from final logical-request outcomes.
- **Effective input/output TPM:** successful provider-reported tokens divided by full block wall
  minutes, including drain. A block with any successful request missing usage is censored from TPM.
- **Goodput:** successful requests or tokens per wall minute, including queueing and failures in the
  denominator interval.

Quality-adjusted goodput is not emitted because the current runner does not produce a valid paired
quality-under-load estimand.

SSE event spans are never converted to token/s: an SSE event can contain zero, one, or many tokens.
Fewer than two content events cannot even define an event span. The number and timestamps of events
remain diagnostic evidence only.

## Confidence intervals

- Request latency and decode-proxy summaries use request-level percentile bootstrap intervals.
- Success probabilities use Wilson 95% intervals.
- Load rates and load success proportions use a paired epoch/block bootstrap of the ratio of summed
  units to summed denominators. They do not use individual requests as independent load samples.
- Tokens within a response are never treated as independent observations.
- p99 is withheld below 1,000 eligible requests.

Intervals characterize the sampled route, time, region, account, workload, and load. They do not
erase systematic provider/time effects. The block bootstrap assumes block-level exchangeability;
adjacent soak blocks and adaptively selected AIMD epochs can remain temporally correlated, so sparse
intervals are conditional and exploratory rather than proof of day-wide capacity.

## Validity and outliers

Every terminal attempt is preserved. Metric-specific eligibility prevents one defect from erasing
otherwise useful evidence: a response with missing usage may support latency but not TPM. Invalid,
censored, and anomalous observations appear in `outlier-audit.jsonl`. A matched-cell 3×IQR rule flags
valid extremes for investigation but never removes them. No winsorization, trimming, or undocumented
timeout deletion occurs. Decode proxies already classified as anomalous are excluded from the
primary decode summary but retained in the audit; matched-cell valid extremes remain included.

## Cache stratification

Cache behavior can be automatic and non-disableable. `cached_trial` reuses an identical prefix;
`uncached_trial` deterministically randomizes it. The ledger separately stores provider-reported
cached tokens. Cached, uncached, and uncontrolled rows are never pooled in TTFT/prefill comparisons.

## What the client cannot claim

The client cannot directly measure server-side prefill compute, GPU utilization, batching, queue
depth, or routing unless the provider exposes trustworthy server timings. Context rejection after a
429/timeout/5xx is inconclusive, not a hard context boundary. A route-specific capability rejection
does not describe other checkpoints or providers. The current report is an evidence package, not a
PDF, production recommendation, publication gate, or completeness certificate.
