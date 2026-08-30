# Inference Endpoint Benchmark

A reproducible laboratory for answering a practical question:

> How will this exact hosted-model endpoint behave for the workloads our engineers actually run?

It measures low-load responsiveness, load capacity, token throughput, reliability, long-context
behavior, output length, tools, structured output, vision, sampling controls, quality under load,
recovery after overload, and matched variation within a declared observation window. Results stay
separated by endpoint and workload; the software does not manufacture a single global score.

## Evidence status

Implementation, a one-request transport check, an active campaign, and a terminal benchmark are
four different claims. Result columns count only checked-in public evidence; active work is named
only to explain why no terminal result appears. “Not yet established” means exactly that, not
failure and not zero performance. A campaign that is still running is listed as non-terminal rather
than promoted to a result.

| Provider | Adapter | Live transport proof | Static / capabilities | Adaptive load | Fixed-rate stability | Time variation | Public report |
|---|---|---|---|---|---|---|---|
| DigitalOcean hosted open models | implemented | established | partial: 564 / 2,891 planned cells | partial: 24 / 44 repeatedly confirmed; 20 need lower-rate closure | 48 / 48 executed: 3 pass, 38 tested-rate non-pass, 3 no-valid-test | complete: 7 matched panels, 1,232 required observations | withheld pending corrected closure and graphics |
| Alibaba Cloud Model Studio, Singapore pay-as-you-go | Chat, Responses, and embeddings implemented | established for `qwen3.8-flash` by direct and packaged-library calls; the eight-route load campaign is underway but non-terminal | not yet established by a terminal public run | underway; no terminal result published | not yet established | not yet established | not published |
| Amazon Bedrock | Chat and Responses implemented | established for exact `zai.glm-4.7-flash` Mantle route in `us-east-1` | not yet established by a terminal public run | underway; no terminal result published | not yet established | not yet established | not published |
| Azure AI Foundry | Chat, Responses, and embeddings implemented | established for five exact text routes; embedding transport also live-proved | not yet established by a terminal public run | underway; no terminal result published | not yet established | not yet established | not published |
| Google Vertex AI | Chat implemented | established for exact `google/gemini-3.6-flash` global OpenAI-compatible route | not yet established by a terminal public run | underway; no terminal result published | not yet established | not yet established | not published |
| OpenRouter | Chat implemented | not established by this repository package | not run here | not run here | not run here | not run here | not published |

For DigitalOcean, the retained [method and data guide](reports/digitalocean/README.md) states exactly
what each measured state establishes. Provider admission receipts prove only that one exact
transport worked with the tested parser and accounting path; they are not latency, reliability,
quality, or capacity evidence. For Alibaba, the public
[provider contract](docs/provider-contracts/alibaba-model-studio-2026-08-29.yaml) and its sanitized
[receipts](docs/provider-contracts/receipts/) contain hash-bound proof metadata without credentials,
prompts, or model output.

The evidence files are observations from exact routes and test windows, not permanent provider-wide
rankings. A blank or “not established” cell is deliberately different from a measured zero.

## What is implemented

| Provider transport | Adapter | Status |
|---|---|---|
| Amazon Bedrock Mantle Chat Completions | `bedrock_mantle` | implemented |
| Amazon Bedrock Mantle Responses | `bedrock_mantle_responses` | implemented |
| Azure AI Foundry Chat Completions | `azure_model_inference` / `azure_openai` | implemented |
| Azure AI Foundry Responses | `azure_responses` | implemented |
| Google Vertex AI OpenAI-compatible Chat Completions | `vertex_openai` | implemented, renewable OAuth |
| OpenRouter Chat Completions | `openrouter` | implemented, exact upstream attested |
| Alibaba Model Studio Chat Completions | `alibaba_model_studio` | implemented, region and pay-as-you-go isolated |
| Alibaba Model Studio Responses | `alibaba_model_studio_responses` | implemented as a separate API contract |
| Azure AI Foundry OpenAI-compatible embeddings | `openai_compatible_embeddings` | implemented as a separate embedding lane |
| Alibaba Model Studio OpenAI-compatible embeddings | `openai_compatible_embeddings` | implemented as a separate embedding lane |
| DigitalOcean Chat Completions | `openai_compatible` | implemented through a provider profile |
| Generic OpenAI-compatible Chat Completions | `openai_compatible` | implemented |

Native Bedrock Converse and native Gemini `generateContent` are not silently emulated. They remain
explicit placeholders until a native-only capability makes their additional request and response
surface worth maintaining. The main provider matrix uses the highest-level compatible API that can
faithfully exercise the workload.

OpenRouter is stricter than a request-side provider hint. Every request disables fallbacks,
requires supported parameters, asks for routing metadata, and verifies the selected and attempted
upstream provider in the response. A missing or mismatched attestation is a failed measurement.

## Questions the benchmark answers

- How fast is the first visible token at low load?
- How quickly are visible output tokens delivered after the first token?
- What successful RPM, input TPM, and output TPM are achieved—not merely offered?
- Where does reliability or latency begin to deteriorate as load rises?
- Does a candidate rate remain healthy for a 120-second fixed-rate stability test, block by block?
- How does the answer change for short requests, long prompts, long outputs, and a mixed workload?
- Which tool, JSON, vision, streaming, and parameter combinations work functionally?
- At which context anchors are requests accepted, and can the model still retrieve separated facts?
- Does performance vary across matched low-load panels during the declared run window?
- Does deterministic task quality deteriorate near saturation?

## Workload map

The same four capacity shapes are used for every admitted route:

| Shape | Default target | What it stresses |
|---|---:|---|
| `short_short` | 256 input / 128 output | request rate, queueing, fixed overhead |
| `long_short` | configured long input / 128 output | input TPM and end-to-end prefill behavior |
| `short_long` | 256 input / 4K output | output TPM and decoding |
| `mixed` | deterministic mixture | production-like contention |

Long-input and long-output targets are configuration fields. A campaign can use 100K, 256K, or
larger prompt anchors when the documented route window permits it. Targets that do not fit fail at
planning time unless the configuration explicitly asks to clip them; clipped and unclipped cells
are never pooled.

The non-load suites cover:

- short and long-context latency baselines;
- fixed context percentages and retrieval anchors;
- requested and realized output-length anchors;
- streaming and non-streaming;
- tools, tool selection, parallel calls, and schema complexity;
- JSON and JSON-schema output;
- vision with a generated, byte-stable image and deterministic answer;
- parameter boundaries and pairwise interactions;
- reasoning, coding, extraction, summarization, instruction-following, and long-context quality;
- cached and deliberately uncached matched prompts.

## AIMD in plain language

AIMD is a feedback controller, not a score.

1. Establish a healthy low-load reference. If the starting probe is unhealthy, halve the rate and
   keep trying through the declared floor; stopping above that floor is a configuration error.
2. Increase quickly until the endpoint shows stress.
3. Continue with smaller additive increases while healthy.
4. After congestion, multiply the offered rate downward—normally by one half, never below the
   declared floor.
5. Confirm the best healthy rate in three separated epochs, stepping lower and retrying when a
   confirmation candidate is unhealthy.
6. After a real overload, fall back and measure recovery.

Arrivals are open-loop: requests are scheduled independently of previous completions. If the
endpoint slows, requests queue and that delay remains in the measurement. This avoids coordinated
omission, where a closed-loop client appears healthy merely because it stops offering work while
the server is slow.

An adaptive load search finds a transition region or a highest-tested repeatedly passing rate. It
does **not** by itself prove sustained capacity. The fixed-rate stability suite (internally named
`soak`) tests one candidate rate in multiple 30-second blocks. A passing claim requires every
registered reliability, latency, queue, usage, quality, and recovery criterion to hold. Merely
finishing the program is not a pass.

“Unhealthy” is a composite controller state, not a synonym for “the endpoint failed.” Reliability,
throttling, queue delay, and end-to-end latency drive the load controller. Missing TTFT remains an
explicit observability gap but does not erase an otherwise successful capacity observation. If no
healthy reference is found through the declared floor, the result says exactly that lower rates were
not tested; it does not claim that the endpoint is unusable.

## Parallelism without contaminating capacity

Independent providers run concurrently with `run-matrix`. Within a provider, endpoint capacity
sweeps remain isolated. This is deliberate: running two endpoints against a shared account quota
would make neither endpoint's ceiling interpretable.

```yaml
version: 1
max_parallel_providers: 4
campaigns:
  - {name: bedrock, provider: amazon-bedrock, config: bedrock.yaml, output: bedrock}
  - {name: azure, provider: azure-ai-foundry, config: azure.yaml, output: azure}
  - {name: vertex, provider: google-vertex-ai, config: vertex.yaml, output: vertex}
  - {name: openrouter, provider: openrouter, config: openrouter.yaml, output: openrouter}
```

Each provider appears once in a matrix. Its config may contain many endpoints. Capacity cell order
is randomized and recorded, but endpoint sweeps execute one at a time inside that provider.

## Measuring matched variation within a run

`time_variation` is a dedicated low-load campaign. It repeats identical route-neutral prompts at
fixed offsets from the run ledger's original start time—for example, seven hourly panels over six
hours—with deterministic randomized route order inside each panel. Panel arrivals are open-loop:
the sender launches them at their scheduled times without waiting for earlier responses. Planning
rejects a campaign whose concurrency, timeout, panel deadline, send cutoff, or final drain window
cannot preserve that schedule.

Stable exact-prefix requests and panel-unique cache-cold requests are separate registered strata.
They receive distinct identities, summaries, and plotted series; they are never pooled or connected
across strata. Optional work may fill otherwise idle time only when it can stop before the next
protected panel. It cannot change panel semantics or delay a required panel. No other workload may
overlap this campaign for the same provider.

```yaml
campaign:
  max_wall_seconds: 22500
  concurrency: 256
  retries: 0
suites:
  time_variation:
    enabled: true
    panels: 7
    interval_minutes: 60
    samples_per_route_shape: 4
    stable_exact_prompt_repeats: 2
    panel_unique_cache_cold_repeats: 2
    shapes: [short_short, long_short, short_long, mixed]
    offered_rps: 1
```

The report plots matched p50 latency, TTFT, success rate, and paired cache-cold minus stable-prefix
differences through time with 95% intervals. A partial run is labelled with its observed span; a
six-hour run is not called a 24-hour, daily, diurnal, or indefinite sustainability study.

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.lock
pip install -e . --no-deps
```

A Git checkout is not required at runtime. Built wheels carry the exact dependency lock; an
installed wheel or unpacked source archive identifies itself by hashing its package bytes, version,
and lockfile. Clean Git checkouts retain the commit-based identity. Both forms are written into run
and report provenance and are checked again before terminal publication.

For development:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Running one provider

Start from [examples/digitalocean.yaml](examples/digitalocean.yaml), the
[Alibaba Singapore provider profile](examples/provider-profiles/alibaba-model-studio-singapore.yaml),
or a private provider config.
A route records the exact provider, model/deployment, API family, region, context/output limits,
prices, authentication environment-variable name, documented capabilities, and dated source URLs.
Credentials remain environment variables and never enter a report.

```bash
# Compose a reusable provider catalog with a provider-neutral experiment. This reads no credential
# and sends no traffic.
inference-bench compile-profile \
  examples/provider-profiles/alibaba-model-studio-singapore.yaml \
  examples/experiment-profiles/standard-static-and-capacity.yaml \
  --output .private/alibaba-comprehensive.yaml

# No credential lookup and no provider traffic:
inference-bench plan .private/azure.yaml

# Live traffic:
inference-bench run .private/azure.yaml --output runs/azure-001 --confirm-live

# Derive tables, audits, and figures from the terminal ledger:
inference-bench report runs/azure-001
```

The monetary cap is a rough safety brake, not the experiment's organizing principle. Token usage
reported by the provider is used when complete; missing usage is not invented. The benchmark's
scientific design is driven by workload coverage, replications, and load behavior.

## Running several providers in parallel

```bash
inference-bench plan-matrix .private/providers.yaml
inference-bench run-matrix .private/providers.yaml \
  --output-root runs/multi-provider-001 --confirm-live

# Combine a main adaptive-load run, a follow-up fixed-rate run, and optional time panels:
inference-bench report-matrix .private/providers.yaml \
  --run-root runs/multi-provider-001 \
  --run-root runs/multi-provider-soak-001 \
  --output published/atlas

# Turn every observed endpoint × workload adaptive-load bound into a two-minute fixed-rate campaign.
# The command retains its historical `derive-soak` name for compatibility.
# The generated .rates.json explains the evidence used for each candidate rate.
inference-bench derive-soak provider.yaml runs/provider/report/controller-summary.csv \
  --output .private/provider-soak.yaml

# Rebuild and verify the final DigitalOcean package from a verified aggregate run directory.
PYTHONPATH=src python scripts/build-digitalocean-final-report.py \
  --run-dir private/do-six-hour-variation-20260828-r1 \
  --summary-dir reports/digitalocean \
  --output published/digitalocean-final-report
PYTHONPATH=src python scripts/verify-digitalocean-final-publication.py \
  published/digitalocean-final-report
```

Every provider gets its own configuration, wall-clock cap, output directory, and durable ledger.
One provider failing does not merge or relabel another provider's evidence.

`report-matrix` regenerates each terminal provider report, combines only matched tables, and builds
`inference-endpoint-evidence-atlas.pdf`. The atlas starts with method and coverage pages, then shows
separate figures for low-load latency, AIMD bounds, sustained RPM/TPM, context retrieval, and
functional capabilities. It ends with a one-page operating-evidence sheet for every exact route.
Missing measurements receive an explicit state and plain-language explanation; they never become
blank cells or zeroes.

## Extending the library

Providers and experiments are separate. A `provider-profile/v1` file defines exact route identity,
transport, region, billing channel, prices, limits, and credential environment-variable names. A
`benchmark-experiment/v1` file defines the measurement design. `compile-profile` combines them
deterministically, rejects unknown or duplicate fields, emits a canonical campaign, and never reads
credentials or sends traffic.

Private or third-party transports can register an adapter factory through the Python entry-point
group `inference_endpoint_benchmark.adapters`. New experiment suites use
`inference_endpoint_benchmark.suites` and return versioned `SuitePlugin` objects. Neither extension
requires editing the execution kernel.

The `inference_bench.reporting` package is provider-neutral: a `report-profile/v1` file maps source
tables, endpoint labels, workload recipes, metrics, intervals, and measured-state rules into typed
evidence cells. Its validators reject missing matrix cells, impossible intervals, duplicate cell
identities, and unexplained states before publication. See
[examples/report-profiles](examples/report-profiles/) and the public API in
[`inference_bench.reporting`](src/inference_bench/reporting/__init__.py).

When an older adaptive search did not reach a healthy low-load reference, a
`capacity-closure-profile/v1` selects only the unresolved route × workload cells and maps them back
to current shapes. The credential-free compiler refuses ambiguous mappings and emits a new plan
without replaying completed request IDs:

```bash
inference-bench plan-capacity-closure \
  provider.yaml prior-run/report/controller-summary.csv closure-profile.yaml \
  --output closure-plan
```

## Output that engineers can use

Every report contains:

- `matched-cell-summary.csv` — request-level latency, TTFT, quality, reliability, token usage, cost,
  sample size, units, and 95% intervals;
- `load-block-summary.csv` — offered and achieved RPM, successful input/output TPM, errors, queue
  drain, and block-level intervals;
- `controller-summary.csv` — adaptive-load bracket, confirmations, recovery, and fixed-rate test completion for every
  endpoint × workload;
- `time-variation-summary.csv` — matched low-load panels within the declared observation window,
  with cache strata kept separate;
- `coverage-ledger.csv` — every planned cell and its completed, unsupported, inconclusive, or
  untested disposition;
- `outlier-audit.jsonl` — the exact reason every suspicious observation was kept, excluded from a
  particular metric, or censored;
- route-specific figures with units, sample counts, and uncertainty.
- `inference-endpoint-evidence-atlas.pdf` — combined multi-provider guide with one endpoint page per
  route and no heterogeneous global score.

Figures use small multiples and matched cells. Capacity points are not connected into looping
spaghetti. Different workloads are not averaged together. Log scales are used only when a panel
spans at least a twenty-fold range and are labelled explicitly.

## How to read the core metrics

- **TTFT:** request start to first visible streamed content. Lower is better.
- **End-to-end latency:** scheduled arrival to final response, including client queueing, retries,
  backoff, and drain. This is what the application experiences.
- **Decode proxy:** provider-reported visible completion tokens divided by time after TTFT. It is a
  client-observed proxy, not direct GPU speed, and is withheld when hidden reasoning makes it
  incomparable.
- **Successful RPM:** successful logical requests divided by the full arrival window plus drain.
- **Effective input/output TPM:** successful provider-reported tokens over the same wall time.
- **Goodput:** useful successful work per minute; failed and missing arrivals remain visible.
- **95% interval:** a range reflecting sampling uncertainty. Request metrics use requests; load
  metrics use epochs or soak blocks. Individual tokens are never treated as independent samples.

p99 is withheld below 1,000 eligible observations. Nothing is silently trimmed. Plausible extremes
remain in the estimate and audit; impossible timing or token arithmetic is retained but excluded
only from the metric it cannot support.

## Reproducibility and safe resumption

SQLite is authoritative. A deterministic request ID is recorded before each send. After a crash,
an ambiguous in-flight request becomes `unknown` and is not replayed automatically. Completed work
is skipped on resume. The run records the source commit, dependency lock, route identity, realized
execution order, response timing, usage, status, and sanitized request/header identifiers.

Prompts and model outputs are used in memory for deterministic scoring but are not written to the
public ledger. Error bodies are represented by a digest. See [DATA-HANDLING.md](DATA-HANDLING.md).

## Method references

- [Experiment design](EXPERIMENT-DESIGN.md)
- [Adapter behavior](ADAPTERS.md)
- [Metric definitions](METHODS.md)
- [Visualization contract](VISUALIZATION.md)
- [Provider admission checklist](PROVIDER-ADMISSION.md)

This repository contains the benchmark engine, not a universal benchmark result. A result belongs
to the exact provider × endpoint × model version × API × region × client location × time window that
was measured.
