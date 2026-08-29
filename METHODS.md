# Methods

## Route identity

A route is more specific than a model name. The benchmark binds every request to provider, adapter,
base URL, model identifier/version, API family/version, region, service-specific upstream, pricing,
declared limits, output-limit field, HTTP version, connection reuse, and opaque account/project/quota
scope. Client location is bound into campaign identity. Changing any of these fields changes the
applicable identity hash.

The route also binds safe documentation/pricing source URLs, a UTC retrieval time, an
owner-declared expected SHA-256 for a separately retained evidence bundle, the fixed
`Accept-Encoding: identity` transport profile, and the hard complete-stream request timeout. The
harness labels the external bundle declaration unverified until a release process hashes its bytes.

An output fallback is a finite local screening ceiling, never a documented provider limit. A
just-above rejection probe is generated only when the route declares an explicit output limit.

## Workload design

The default campaign combines:

1. low-load latency baselines for four input/output shapes;
2. functional capability requests for streaming, non-streaming, tools, JSON output, and vision,
   plus explicitly acceptance-only requests for log probabilities, untriggered stop, seed, and
   parameter boundaries;
3. a strength-two covering array for sampling/stream/output interactions, constructed after
   route-limit realization so clipping cannot collapse the advertised factor coverage;
4. fixed context anchors at 1%, 10%, 25%, 50%, 75%, 90%, 95%, 99%, and one nominal
   `documented + 1` synthetic target, with three independently generated opaque retrieval markers;
5. requested-output anchors from 32 through the documented maximum and one value above;
6. deterministic reasoning, instruction, extraction, and retrieval quality tasks;
7. matched cached/uncached-prefix trials;
8. open-loop AIMD saturation and endpoint-isolated sustained soaks.

The planner is deterministic under the campaign seed. Comparable routes receive identical target
shapes. Provider-reported usage is authoritative; synthetic prompt “token” counts are scheduling and
reservation estimates because tokenizer behavior can differ. Before each claim, the runner
materializes the exact JSON bytes, hashes those bytes and the generator version, and reserves the
larger of the planned count or a UTF-8 byte upper bound plus route overhead, multiplied by the
declared safety factor.

Hard-coded temperature/top-p values are nominal OpenAI-compatible transport screens, not documented
provider ranges. They are labelled acceptance-only and never interpreted as correct enforcement.
Likewise, the `context_tokens + 1` synthetic prompt is a nominal target: acceptance does not prove a
documented context boundary was exceeded unless provider-reported input usage establishes that.

## AIMD

Each route and workload shape is isolated from other capacity traffic. Poisson arrivals are scheduled
from an epoch clock independently of completions. A semaphore is only a safety ceiling: waiting at the
ceiling produces measured queue delay.

Within AIMD and soak suites, endpoint × shape blocks execute sequentially but in a deterministic
seeded shuffle rather than route-config order. AIMD and soak receive independent shuffles, and the
realized orders are persisted as events. This reduces systematic time-order confounding without
overlapping capacity traffic.

Static measured endpoint × suite blocks and the logical cells inside each block receive independent
deterministic seeded shuffles. Their complete realized order is persisted before execution; resume
removes completed logical IDs without reordering the remainder. Standalone warmup diagnostics retain
diagnostic-first order and do not confer a controlled warm state on later measurements.

The first configured healthy increases use a bounded geometric multiplier to bracket capacity
rapidly; later healthy epochs increase by a fixed additive amount. After two consecutive unhealthy
epochs, offered RPS is multiplied by the configured decrease factor and the controller continues in
additive-increase/multiplicative-decrease mode. The highest healthy candidate receives three
confirmation epochs separated by low-load epochs. A 50%-rate recovery epoch is run and labelled only
when a two-epoch overload was actually observed. If no such overload is observed before the epoch or
maximum-rate ceiling, the result is explicitly right-censored as the highest tested healthy rate;
it is not described as a knee or capacity ceiling.

Low-load baselines and confirmation separators use an exact, deterministic sample count (20 by
default and never fewer than 20), evenly spaced in time. If a baseline RPS is configured, the
baseline duration is extended as needed so that all samples fit without exceeding that rate. This
avoids zero-arrival Poisson baselines and makes the p95 sample size explicit; load-bearing AIMD,
confirmation, recovery, and soak epochs retain seeded open-loop Poisson arrivals.

Long-load size is not fixed at 32K. The `warmup`, `latency`, `aimd`, and `soak` suites can bind
`long_input_tokens` and `long_output_tokens` into campaign identity. `long_short` uses the configured
input target with a short output; `short_long` uses the configured output target with a short input;
the heterogeneous mix uses the same targets whenever it selects either subtype. Explicit targets
fail planning when they exceed the documented combined-context/output allowance unless the matching
`long_*_overflow: clip` policy is deliberately selected. Clipping changes the realized matched-cell
identity and is recorded in request metadata. Planning, request materialization, pre-send token/cost
reservation, execution, and reporting all consume that same realized target.

The separately configured `warmup` suite is a transport/availability diagnostic, not a paired warm
latency control. It may be separated from measured blocks by other work and a resumed process does
not replay its already-terminal identities. Reports therefore label every non-warmup cell
`warm_state=uncontrolled_not_paired`; no warm/cold latency claim is made.

An interrupted partial epoch is never replayed. It is scientifically censored and does not count as
healthy or as congestion. The same applies to a Poisson epoch with zero scheduled arrivals and to a
cost/time/402/reservation-guarded epoch. A censored baseline stops that endpoint × shape controller;
a completed but unhealthy low-load baseline also stops and explicitly censors the controller as
`unhealthy_low_load_baseline`; it cannot define latency thresholds or capacity. A censored ramp
epoch leaves the tested rate and best-healthy candidate unchanged and breaks consecutiveness of
unhealthy evidence. Censored blocks remain in the coverage/audit ledger but are excluded from
capacity estimates and confidence intervals.

Default epoch health requires:

- at least 99% of all scheduled logical requests ending successfully;
- no more than 1% of physical attempts rate-limited, including intermediate retries;
- no more than 1% of physical attempts ending in a server error, timeout, or transport error;
- p95 TTFT and end-to-end latency no greater than twice their low-load baselines when those baseline
  metrics are observable;
- end-of-epoch drain no larger than 10% of the epoch or one second.

These short epochs seek a knee. Only two consecutive scientifically eligible unhealthy epochs
bracket one; otherwise the result is explicitly left- or right-censored. They do not establish
sustained capacity.

## Sustained soaks

A soak consists of four 30-second analysis blocks by default. The runner records the tested rate and
block health, but the report does not automatically certify sustainable capacity. Capacity above or
below the tested rate is not inferred. The current runner also does not construct paired low-load and
near-load quality trials, so low-load quality scores must not be interpreted as evidence that quality
is unchanged under saturation.

## Metrics

All clocks are client-side monotonic clocks. Units are stored and printed explicitly.

- **TTFT:** request start to first content-bearing SSE event, summarized only across successful final
  logical outcomes with an observed TTFT. Failures and successful empty/EOS outputs remain in
  reliability counts but are not silently included in the TTFT sample size.
- **Arrival-to-completion latency (headline):** among successful final logical outcomes only,
  scheduled open-loop arrival through completion, including event-loop/semaphore queueing, local
  pre-send work, all physical attempts, retry backoff, and complete response drain. The report gives
  this success-conditioned population and its `n` explicitly; failure latency is not claimed by this
  estimand.
- **Successful final-attempt service time:** one successful final-logical send start through its
  complete response drain. This request-level distribution excludes failed and intermediate retry
  attempts; physical-attempt counts and errors are reported separately.
- **Decode proxy:** provider-reported completion tokens divided by `(end-to-end seconds − TTFT)`.
  This includes client transport/drain overhead and is not direct server decode compute. The
  primary proxy requires at least eight billed completion tokens, two content-bearing events, and
  one second after TTFT; shorter bursts are preserved in the audit but censored from tokens/second
  summaries because buffering and denominator instability dominate them.
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
- **Logical success rate under load:** successful final outcomes divided by all scheduled open-loop
  arrivals. Unlaunched, failed, and unknown arrivals remain in the denominator.
- **Deterministic quality score:** mean score across every predeclared quality trial. A non-success
  receives zero; unrelated timing or usage invalidity cannot silently remove a scored response.
  Unknown claimed outcomes and claimed incomplete retry sequences also receive explicit zeros.
  Reports disclose total scored trials, successful-response trials, non-success zeros,
  incomplete-retry zeros, and unscored requests. Because all bundled deterministic scorers are
  binary, quality confidence intervals use the Wilson binomial interval with logical trials as units.

Quality-adjusted goodput is not emitted because the current runner does not produce a valid paired
quality-under-load estimand.

Retry sequences are stratified by their final logical outcome and predeclared cache trial, then every
physical attempt and its conservative/settled cost is charged to that same unconditional base cell.
An intermediate 429 with unknown usage cannot create a second logical observation or hide its cost
from the final-success cell.

SSE event spans are never converted to token/s: an SSE event can contain zero, one, or many tokens.
Fewer than two content events cannot even define an event span. The number and timestamps of events
remain diagnostic evidence only.

Completion-token usage can include hidden reasoning. The primary visible post-TTFT proxy therefore
requires an explicit provider-reported reasoning-token count of zero. Positive and unknown counts
remain visible as counts and are censored from that proxy; reliability, cost, latency, and usage are
not conditioned on this post-outcome state. Aggregate billed-output TPM remains labelled as billed
token goodput.

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
censored, and anomalous observations appear in `outlier-audit.jsonl`. A matched-cell 3×IQR rule
flags valid extremes; when the central IQR is exactly zero, deviations from that central value are
flagged explicitly. Neither rule removes data. No winsorization, trimming, or undocumented timeout
deletion occurs. Decode proxies already classified as anomalous are excluded from the primary
decode summary but retained in the audit; matched-cell valid extremes remain included.
Deterministic quality is deliberately orthogonal to transport/usage eligibility: if a predeclared
scorer can evaluate the retained outcome it stays in the quality denominator, and provider
non-success is scored zero.

Matched-cell accounting reports `settled_usd_sum`, `unknown_reserved_usd_sum`, and their
`conservative_exposure_usd_sum` separately. Derived per-success and per-effective-token costs use the
conservative exposure and are named accordingly.

A validation-class HTTP status on a deliberately invalid probe is not labelled correct enforcement.
Without a retained, allow-listed provider error reason plus a matched successful control, it is a
neutral censored observation and cannot establish a hard boundary.

## Cache stratification

Cache behavior can be automatic and non-disableable. `cached_trial` reuses an identical prefix;
`uncached_trial` deterministically randomizes it. The ledger separately stores provider-reported
cached tokens. Cached, uncached, and uncontrolled rows are never pooled in TTFT/prefill comparisons.

## What the client cannot claim

The client cannot directly measure server-side prefill compute, GPU utilization, batching, queue
depth, or routing unless the provider exposes trustworthy server timings. Context rejection after a
429/timeout/5xx is inconclusive, not a hard context boundary. A route-specific capability rejection
does not describe other checkpoints or providers. Fixed anchors do not locate an exact context
boundary, and this harness does not isolate tool-schema or image contributions to a combined context
limit. A single-provider report is an evidence package, not a production recommendation,
publication gate, or completeness certificate. The matrix atlas is a reading layer over terminal
provider reports: it adds provider and run identity, combines only matched tables, and keeps the
request as the low-load sampling unit and the epoch/block as the capacity sampling unit. Its PDF
never substitutes for the row-level ledger or turns missing evidence into a zero.

## Cost, resume, and terminal-state contract

Configuration, adapter construction, credentials, HTTP/2 availability, JSON serialization, and the
exact byte reservation all fail before a request claim. Each physical retry receives a deterministic
identity and its own atomic reservation. A crash-recovered claimed send becomes a final `unknown`
outcome whose reservation remains in exposure and reports; it is never retried automatically.
Epoch/block summaries use idempotency keys and reject identity drift on resume. Reports require a
terminal, internally consistent SQLite/JSONL snapshot.

SQLite is authoritative if appending the derived JSONL projection fails after a transaction commits.
The ledger marks that projection dirty, continues without changing the settled outcome, and safely
rebuilds the complete ordered projection before reporting.
