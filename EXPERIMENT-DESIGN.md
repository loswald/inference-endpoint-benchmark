# Experiment design

This is a practical campaign, not an unbounded Cartesian product. It maps the important operating
envelope first, then spends extra samples where the decision is uncertain.

## Workload shapes

Use at least four matched shapes for every route:

| Shape | Primary pressure | Example anchors |
|---|---|---|
| short input / short output | request rate and queueing | 256 in / 128 out |
| long input / short output | input TPM and end-to-end prefill proxy | 32K, then route-relative 10–99% |
| short input / long output | output TPM and post-TTFT decoding proxy | 1K–4K realized output |
| heterogeneous mixed | production-like interference | deterministic mixture of task families and sizes |

Keep task content and realized token anchors identical across comparable routes. If a documented
limit forces clipping, the clipped cell receives a different identity and is not pooled with the
original target.

## Recommended 24-hour sequence

1. **Admission and warmups.** Serial controls, transport parsing, usage settlement, and explicit
   warmup labels.
2. **Low-load baselines.** Randomized endpoint blocks for TTFT, end-to-end latency, output rate,
   reliability, cost, and deterministic task quality.
3. **Isolated AIMD.** Open-loop arrivals plus a separate concurrency ceiling. Geometrically bracket,
   then additively increase and multiplicatively decrease. Two consecutive unhealthy epochs bracket
   congestion; healthy termination at the configured ceiling is a right-censored lower bound.
4. **Sustained soaks.** For each endpoint × shape, run a predeclared candidate rate in multiple
   analysis blocks. Report achieved rate, TPM, reliability, latency, queue growth, quality, and
   between-block intervals. AIMD alone is never called sustained capacity.
5. **Capability matrix.** Test exact states for streaming, tools, parallel calls, structured output,
   vision, stop, seed, logprobs, and sampling controls. Use a strength-two covering array for numeric
   and categorical interactions rather than a full Cartesian product.
6. **Context and output envelopes.** Probe route-relative percentages, fixed anchors, just-below,
   at, and just-above boundaries. Acceptance without retrieval or realized output is not success.
7. **Matched-control closure.** Re-run unresolved capability cells as control-before → probe →
   control-after. A 400/413/422 is a parameter rejection only when both controls pass. Authentication,
   quota, timeout, or 5xx failures remain inconclusive capability evidence.
8. **Recovery and time variation.** After overload, fall to half the candidate rate and measure
   recovery. Repeat a sparse sentinel panel over the desired time horizon; do not call a partial
   day “24-hour variability.”

## Statistical units

- requests for request latency and deterministic quality;
- AIMD epochs for capacity confirmation;
- predeclared soak blocks for sustained-load uncertainty;
- matched pair IDs for quality-versus-load changes.

Tokens are never independent samples. Suppress p99 unless roughly 1,000 eligible observations exist.
Use Wilson intervals for binary rates and bootstrap or t intervals only when their sampling-unit
assumptions are stated. Adjacent time blocks can be correlated, so label four-block intervals
exploratory rather than pretending they are independent days.

## Production handoff

Publish tested anchors, not invented headroom. Production should begin below the matched workload
anchor, retain a separate concurrency ceiling, and use retry/backoff plus AIMD. Re-run admission and
the relevant cells after any model version, region, quota, API, or serving-stack change.
