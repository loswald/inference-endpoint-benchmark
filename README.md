# Inference Endpoint Benchmark

A reproducible laboratory for answering a practical question:

> How will this exact hosted-model endpoint behave for the workloads our engineers actually run?

It measures low-load responsiveness, load capacity, token throughput, reliability, long-context
behavior, output length, tools, structured output, vision, sampling controls, quality under load,
recovery after overload, and variation across the day. Results stay separated by endpoint and
workload; the software does not manufacture a single global score.

## What is implemented

| Provider transport | Adapter | Status |
|---|---|---|
| Amazon Bedrock Mantle Chat Completions | `bedrock_mantle` | implemented |
| Amazon Bedrock Mantle Responses | `bedrock_mantle_responses` | implemented |
| Azure AI Foundry Chat Completions | `azure_model_inference` / `azure_openai` | implemented |
| Azure AI Foundry Responses | `azure_responses` | implemented |
| Google Vertex AI OpenAI-compatible Chat Completions | `vertex_openai` | implemented, renewable OAuth |
| OpenRouter Chat Completions | `openrouter` | implemented, exact upstream attested |
| DigitalOcean Chat Completions | `digitalocean` | implemented |
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
- Does a candidate rate remain healthy for a sustained two-minute test, block by block?
- How does the answer change for short requests, long prompts, long outputs, and a mixed workload?
- Which tool, JSON, vision, streaming, and parameter combinations work functionally?
- At which context anchors are requests accepted, and can the model still retrieve separated facts?
- Does performance vary across matched low-load panels during the day?
- Does deterministic task quality deteriorate near saturation?

## Workload map

The same four capacity shapes are used for every admitted route:

| Shape | Default target | What it stresses |
|---|---:|---|
| `short_short` | 256 input / 128 output | request rate, queueing, fixed overhead |
| `long_short` | 32K input / 128 output | input TPM and end-to-end prefill behavior |
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

1. Start at a low offered request rate.
2. Increase quickly until the endpoint shows stress.
3. Continue with smaller additive increases while healthy.
4. After congestion, multiply the offered rate downward—normally by one half.
5. Confirm the best healthy rate in three separated epochs.
6. After a real overload, fall back and measure recovery.

Arrivals are open-loop: requests are scheduled independently of previous completions. If the
endpoint slows, requests queue and that delay remains in the measurement. This avoids coordinated
omission, where a closed-loop client appears healthy merely because it stops offering work while
the server is slow.

An AIMD sweep finds a knee or a highest-tested healthy lower bound. It does **not** by itself prove
sustained capacity. The soak suite tests a fixed candidate rate in multiple 30-second blocks. A
sustained claim requires every planned block to complete with the declared reliability, latency,
queue, and usage criteria.

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

## Measuring variation across the day

`time_variation` is a dedicated low-load campaign. It repeats identical route-neutral prompts at
fixed offsets—for example, 12 panels two hours apart—with randomized route order inside each panel.
It cannot be enabled beside AIMD, soak, or capability traffic, because overlapping load would make
the time-of-day result uninterpretable.

```yaml
suites:
  time_variation:
    enabled: true
    panels: 12
    interval_minutes: 120
    samples_per_route_shape: 5
    shapes: [short_short, long_short]
    offered_rps: 0.2
```

The report plots matched p50 latency, TTFT, and success rate through time with 95% intervals. A
partial run is labelled as the observed time span; it is never called “24-hour variability.”

## Installation

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.lock
pip install -e . --no-deps
```

For development:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Running one provider

Start from [examples/digitalocean.yaml](examples/digitalocean.yaml) or a private provider config.
A route records the exact provider, model/deployment, API family, region, context/output limits,
prices, authentication environment-variable name, documented capabilities, and dated source URLs.
Credentials remain environment variables and never enter a report.

```bash
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

# Combine a main AIMD run, a follow-up soak run, and optional time panels:
inference-bench report-matrix .private/providers.yaml \
  --run-root runs/multi-provider-001 \
  --run-root runs/multi-provider-soak-001 \
  --output published/atlas

# Turn every observed endpoint × workload AIMD lower bound into a two-minute soak campaign.
# The generated .rates.json explains the evidence used for each candidate rate.
inference-bench derive-soak provider.yaml runs/provider/report/controller-summary.csv \
  --output .private/provider-soak.yaml

# Re-render a sanitized DigitalOcean direct-run summary package with the clean atlas style.
inference-bench report-digitalocean-summary private/do-summary \
  --capacity-source do-sixhour-aimd-20260824-r1 \
  --soak-source do-direct-soak-20260823-r1 \
  --output published/digitalocean-atlas
```

Every provider gets its own configuration, wall-clock cap, output directory, and durable ledger.
One provider failing does not merge or relabel another provider's evidence.

`report-matrix` regenerates each terminal provider report, combines only matched tables, and builds
`inference-endpoint-evidence-atlas.pdf`. The atlas starts with method and coverage pages, then shows
separate figures for low-load latency, AIMD bounds, sustained RPM/TPM, context retrieval, and
functional capabilities. It ends with a one-page operating-evidence sheet for every exact route.
Missing measurements stay blank and labelled; they never become zeroes.

## Output that engineers can use

Every report contains:

- `matched-cell-summary.csv` — request-level latency, TTFT, quality, reliability, token usage, cost,
  sample size, units, and 95% intervals;
- `load-block-summary.csv` — offered and achieved RPM, successful input/output TPM, errors, queue
  drain, and block-level intervals;
- `controller-summary.csv` — AIMD bracket, confirmations, recovery, and soak completion for every
  endpoint × workload;
- `time-variation-summary.csv` — matched low-load panels across the day;
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
