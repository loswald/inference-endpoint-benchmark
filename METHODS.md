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
separated confirmation epochs followed by a 50%-rate recovery epoch.

Default epoch health requires:

- at least 99% successful completed requests;
- no more than 1% rate-limited requests;
- no more than 1% combined server errors and timeouts;
- end-of-epoch drain no larger than 10% of the epoch or one second.

These short epochs locate a knee. They do not establish sustained capacity.

## Sustained soaks

A soak consists of four independent 30-second analysis blocks by default. All blocks must pass the
health gate before the tested rate may be described as soak-verified. Capacity above or below the
tested rate is not inferred. Quality tasks can be paired at baseline and load by adding the same task
cell to both phases.

## Metrics

All clocks are client-side monotonic clocks. Units are stored and printed explicitly.

- **TTFT:** request start to first content-bearing SSE event.
- **End-to-end latency:** request start through complete response drain.
- **Decode proxy:** provider-reported completion tokens divided by `(end-to-end seconds − TTFT)`.
  This includes client transport/drain overhead and is not direct server decode compute.
- **Effective input/output TPM:** successful provider-reported tokens divided by analysis-block wall
  minutes.
- **Goodput:** successful requests or tokens per wall minute, including queueing and failures in the
  denominator interval.
- **Quality-adjusted goodput:** successful units multiplied by deterministic task score, per minute.

SSE event spans are never converted to token/s: an SSE event can contain zero, one, or many tokens.
Fewer than two content events cannot even define an event span. The number and timestamps of events
remain diagnostic evidence only.

## Confidence intervals

- Request latency and decode-proxy summaries use request-level percentile bootstrap intervals.
- Success probabilities use Wilson 95% intervals.
- Load rates use independent epochs or soak blocks as bootstrap units.
- Tokens within a response are never treated as independent observations.
- p99 is withheld below 1,000 eligible requests.

Intervals characterize the sampled route, time, region, account, workload, and load. They do not
erase systematic provider/time effects.

## Validity and outliers

Every terminal attempt is preserved. Metric-specific eligibility prevents one defect from erasing
otherwise useful evidence: a response with missing usage may support latency but not TPM. Invalid,
censored, and anomalous observations appear in `outlier-audit.jsonl`. A matched-cell 3×IQR rule flags
valid extremes for investigation but never removes them. No winsorization, trimming, or undocumented
timeout deletion occurs.

## Cache stratification

Cache behavior can be automatic and non-disableable. `cached_trial` reuses an identical prefix;
`uncached_trial` deterministically randomizes it. The ledger separately stores provider-reported
cached tokens. Cached, uncached, and uncontrolled rows are never pooled in TTFT/prefill comparisons.

## What the client cannot claim

The client cannot directly measure server-side prefill compute, GPU utilization, batching, queue
depth, or routing unless the provider exposes trustworthy server timings. Context rejection after a
429/timeout/5xx is inconclusive, not a hard context boundary. A route-specific capability rejection
does not describe other checkpoints or providers.

