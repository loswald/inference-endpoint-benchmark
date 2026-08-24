# Inference Endpoint Benchmark

A small, provider-neutral harness for measuring hosted language-model endpoints under realistic
load. It records request-level latency and token usage, maps context/output/capability boundaries,
runs open-loop AIMD saturation tests and sustained soaks, scores deterministic quality tasks, and
builds matched-cell reports with uncertainty intervals.

The harness is designed to answer engineering questions—not to manufacture a leaderboard:

- What request rate and token throughput did this exact route sustain for this exact workload?
- How do time to first token, decode speed, reliability, and quality change near saturation?
- Which documented capabilities work through the tested API surface?
- Where do context and requested-output limits accept, degrade, or reject?
- Which conclusions are measured, censored, anomalous, or still untested?

## What “provider neutral” means

The benchmark core never imports a cloud SDK. An adapter translates one normalized request into a
provider call and returns one normalized result. The included `openai_compatible` adapter covers:

- DigitalOcean Gradient AI inference;
- ordinary OpenAI-compatible endpoints;
- Azure OpenAI-compatible deployments when the route URL and API version are supplied;
- Vertex OpenAI-compatible endpoints when a bearer token is supplied by the caller environment;
- an OpenRouter request mapper that pins one upstream and disables fallbacks. Live OpenRouter
  execution deliberately fails closed until an adapter verifies the actual upstream generation.

`bedrock_native`, `vertex_native`, and evidence-bearing `openrouter` are explicit fail-closed
placeholders. They are not labelled implemented. Add their SDK translation or generation lookup
behind the same adapter protocol and contract tests before using those API families for evidence.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp examples/digitalocean.yaml campaign.yaml
export DIGITALOCEAN_API_TOKEN="..."

# No credentials loaded and no provider traffic:
inference-bench plan campaign.yaml

# Live traffic; requires --confirm-live deliberately:
inference-bench run campaign.yaml --output runs/do-001 --confirm-live
inference-bench report runs/do-001
```

The `plan` command prints a credential-free workload count and conservative worst-case token cost.
The `run` command refuses to start without an explicit live confirmation, a positive wall-time cap,
and a positive cost cap.

## Campaign structure

Routes define exact serving identity: provider, model, API family/version, region, endpoint URL,
authentication environment-variable name, declared limits, prices, and capabilities. Suites define
load shape and replication. See [examples/digitalocean.yaml](examples/digitalocean.yaml).

The standard workload shapes are:

| Shape | Approximate input | Requested output | Main pressure |
|---|---:|---:|---|
| `short_short` | 256 tokens | 128 tokens | RPM and queueing |
| `long_short` | route-relative, capped by config | 128 tokens | input TPM / prefill proxy |
| `short_long` | 256 tokens | route-relative, capped by config | output TPM / decoding |
| `mixed` | deterministic heterogeneous mix | mixed | production-like contention |

Actual provider-reported token usage—not prompt estimates—drives settled cost and effective TPM.

## Scientific boundaries

- Arrivals are open-loop: scheduled arrival time is independent of completion time. Queue delay is
  retained, so coordinated omission cannot hide overload.
- AIMD epochs locate a capacity knee. They are not called “sustainable” by themselves.
- A fixed-rate sustained claim requires a completed soak and all predeclared health gates.
- Confidence intervals use independent requests for request metrics and independent epochs/blocks
  for load metrics. Output tokens are never treated as independent samples.
- p99 is withheld below 1,000 eligible observations.
- The client cannot directly observe server-side prefill compute. `TTFT - transport baseline` is at
  most an end-to-end prefill proxy and is labelled that way.
- Capability rejection is route/API-specific. A rejected image request does not prove that another
  checkpoint or provider supports no vision.

## Data-quality contract

Nothing is silently trimmed. Every terminal attempt is classified as `valid`, `anomalous`,
`invalid`, or `censored`. Missing usage, impossible timing order, non-monotonic stream timestamps,
zero/near-zero decode duration with multiple output tokens, and inconsistent units are retained and
reported. Legitimate slow or fast extremes remain in the primary sample and are also listed in the
outlier audit. Invalid observations are excluded only from the estimand they cannot support; their
counts and reasons remain visible.

## Durable execution

SQLite is authoritative. Before a send, the runner atomically reserves the deterministic request ID
and worst-case cost. A crash leaves an `in_flight` row; on resume it becomes `unknown` and is never
automatically replayed. A prompt-free JSONL event projection is fsynced for easy inspection and can
be regenerated from SQLite. See [DATA-HANDLING.md](DATA-HANDLING.md).

## Reports

Reports compare matched endpoint × suite × workload cells only. Figures carry units, sample counts,
and 95% intervals. Capacity trajectories use route-specific small multiples. The report generator
does not create heterogeneous global scores or rank endpoints tested on different workloads.

## Development

```bash
ruff check .
pytest
```

This repository contains no benchmark results, credentials, or DigitalOcean-specific conclusions.
