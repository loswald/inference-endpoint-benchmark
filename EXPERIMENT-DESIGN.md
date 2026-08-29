# Experiment design

The design maps an endpoint's operating envelope instead of benchmarking one convenient prompt.
It is finite, repeatable, and organized around decisions an AI engineer must make.

## 1. Admit the exact route

Record the provider, endpoint/deployment, model version, API family, region, client location,
documented context and output limits, price, quota scope, and supported features. A successful model
catalog lookup is discovery—not evidence that the inference route works.

Run one short streaming control before the main campaign. Verify status, usage fields, response
parsing, request identifiers, and any routing attribution.

## 2. Establish low-load behavior

For every route, measure identical randomized prompts at a rate far below expected capacity:

- time to first visible token;
- end-to-end and service latency;
- visible decode proxy when token accounting permits it;
- request success and deterministic quality;
- provider-reported input and output tokens;
- cost per successful request.

Warmups are diagnostics and remain separate. A measured request is not retroactively called warm or
cold unless that state was explicitly controlled.

## 3. Map the four capacity shapes

| Shape | Main pressure | Typical target |
|---|---|---:|
| short / short | request rate and queueing | 256 input / 128 output |
| long / short | input TPM and prefill behavior | 32K or 100K input / 128 output |
| short / long | output TPM and decode behavior | 256 input / 4K output |
| heterogeneous mixed | production-like contention | deterministic mixture |

Use the same target token counts and prompt identities for comparable routes. If one route cannot
fit the target, report the unmatched route-relative cell separately.

## 4. Find the knee with open-loop AIMD

Open-loop means arrivals are scheduled from a clock, not from the completion of the previous
request. A separate concurrency ceiling prevents unlimited client memory growth. When the endpoint
slows, arrivals wait and their queue delay remains part of end-to-end latency.

The controller follows this sequence:

1. Measure a low-rate baseline. If it is unhealthy, keep halving the offered rate through the
   declared floor before calling the search unresolved.
2. Increase geometrically to bracket the neighborhood quickly.
3. Increase by a fixed additive step after a healthy epoch.
4. Multiply the offered rate down after congestion, normally by 0.5.
5. Treat two consecutive unhealthy epochs as an overload bracket.
6. Confirm the best healthy rate in three separated epochs.
7. After observed overload, test recovery at half that rate.

An epoch is healthy only when its predeclared reliability, latency, queue-growth, and throttling
criteria all pass. If the highest configured rate remains healthy, report “at least this rate”;
there is no observed knee.

TTFT availability is reported separately from capacity health. A successful response can lack a
visible first-output event; that makes TTFT and decode-rate metrics unavailable, but does not by
itself turn successful, timely service into a capacity failure.

## 5. Verify sustained behavior

AIMD epochs are short controller observations. They do not prove sustained capacity.

For every endpoint × shape, run the selected candidate rate as four 30-second blocks. Report every
block and the aggregate:

- offered, completed, and successful RPM;
- successful input and output TPM;
- success, throttle, timeout, and server-error rates;
- p95 TTFT and end-to-end latency;
- queue drain after the arrival window;
- deterministic quality under load. A low-load-vs-loaded quality-delta claim additionally requires
  the same predeclared tasks to be paired across conditions; the current built-in fixed-rate runner
  does not yet provide that paired estimand.

Call the tested rate a fixed-rate pass only if every required block completes and all health gates pass. If a
budget, deadline, crash, or missing-usage condition removes a block, label the cell incomplete.

## 6. Exercise capabilities, not just request acceptance

The capability suite includes functional controls for:

- streaming and non-streaming;
- tool selection, argument validity, nested schemas, and parallel calls;
- JSON and JSON-schema validity;
- a generated solid-color image with a deterministic answer;
- short and long answers;
- reasoning, coding with executable checks, extraction, summarization, and instruction following;
- stop, seed, log probabilities, temperature, and top-p;
- pairwise parameter interactions plus selected three-way performance corners.

A 2xx proves transport acceptance. Functional support requires the deterministic behavior check.
A validation 4xx supports a boundary only when matched before/after controls still succeed.

## 7. Map context and output envelopes

Probe fixed anchors and percentages of the documented context window: 1%, 10%, 25%, 50%, 75%,
90%, 95%, and 99%. Long prompts contain independent markers near the beginning, middle, and end.
Acceptance without correct retrieval is not long-context success.

Probe requested output at small, medium, large, near-limit, limit, and just-over-limit anchors.
Separate:

1. request-limit acceptance;
2. realized output length;
3. prompt-plus-output enforcement;
4. EOS, truncation, timeout, and infrastructure termination.

## 8. Measure time variation separately

Run a dedicated low-load sentinel campaign. Repeat identical prompts at fixed offsets across the
desired horizon and randomize route order within each panel. Do not overlap capacity or capability
traffic from the same provider account.

For a 24-hour screen, use 12 panels two hours apart or 24 hourly panels. Report the observed span,
panel count, per-panel sample count, and uncertainty. Adjacent panels are temporally correlated;
they are not independent days.

## 9. Parallelize at the correct level

Run independent providers concurrently. Keep capacity sweeps endpoint-isolated inside each
provider. Low-load time panels can include all endpoints serially in randomized order. This gives
fast execution without turning shared quota contention into a false endpoint property.

## 10. Statistical units

- request metrics: final logical requests;
- AIMD confirmation: epochs;
- fixed-rate stability: analysis blocks;
- time variation: matched requests within panels, with panels retained explicitly;
- quality change: matched task pairs.

Tokens are measurements, not independent samples. Use Wilson intervals for binary proportions and
request or block bootstrap intervals for medians, quantiles, and rates. Withhold p99 below 1,000
eligible observations. Publish sample size, unit, interval bounds, and method next to every estimate.

## Production handoff

Recommend only a workload-matched rate that passed the fixed-rate stability test. If no such rate exists, publish the highest
observed healthy screen and say what remains unverified. Production clients should retain bounded
concurrency, exponential retry backoff with jitter, and an AIMD-style adaptive offered-rate limit.

Re-run relevant cells after a model, deployment, API version, region, quota, or provider routing
change. A provider name is not a performance guarantee; the exact route identity is the result.
