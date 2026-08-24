# Inference Endpoint Benchmark

A small, provider-neutral harness for measuring hosted language-model endpoints under realistic
load. It records request-level latency and token usage, screens fixed context/output anchors,
runs open-loop AIMD saturation tests and fixed-rate soak blocks, scores deterministic low-load
quality tasks, and builds matched-cell reports with uncertainty intervals.

The harness is designed to answer engineering questions—not to manufacture a leaderboard:

- What request rate and token throughput did this exact route sustain for this exact workload?
- How do time to first token, decode speed, and reliability change across offered load?
- Which documented capabilities are functionally verified, merely transport-accepted, or rejected?
- At which tested context and requested-output anchors does the route accept, degrade, or reject?
- Which conclusions are measured, censored, anomalous, or still untested?

## What “provider neutral” means

The benchmark core never imports a cloud SDK. An adapter translates one normalized request into a
provider call and returns one normalized result. The included `openai_compatible` adapter covers:

- DigitalOcean Gradient AI inference;
- ordinary OpenAI-compatible endpoints;
- Azure OpenAI-compatible deployments when the route URL and API version are supplied;
- an OpenRouter request mapper that pins one upstream and disables fallbacks. Live OpenRouter
  execution deliberately fails closed until an adapter verifies the actual upstream generation.

`bedrock_native`, `vertex_openai`, `vertex_native`, and evidence-bearing `openrouter` are explicit
fail-closed placeholders. They are not labelled implemented. In particular, a static Vertex access
token is unsafe for a multi-hour campaign; admission requires tested ADC/service-account refresh
and expiry handling. Add each SDK/OAuth translation or generation lookup behind the same adapter
protocol and contract tests before using those API families for evidence.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.lock
pip install -e . --no-deps
# Contributors may add the dev extra. Manifests capture only the locked runtime closure and this
# package; unrelated ambient/development distributions are deliberately omitted.
pip install -e ".[dev]"
mkdir -p .private
cp examples/digitalocean.yaml .private/campaign.yaml
# PowerShell: New-Item -ItemType Directory -Force .private; Copy-Item examples/digitalocean.yaml .private/campaign.yaml
# Edit every replace-with-* value and both null prices from current provider documentation.
# The template deliberately fails closed until this is done.
export DIGITALOCEAN_API_TOKEN="..."

# No credentials loaded and no provider traffic:
inference-bench plan .private/campaign.yaml

# Live traffic; requires --confirm-live deliberately:
inference-bench run .private/campaign.yaml --output runs/do-001 --confirm-live
inference-bench report runs/do-001
```

The `plan` command prints a credential-free workload count and conservative worst-case token cost.
The shipped DigitalOcean example contains null prices, so planning refuses it until exact dated
input and output prices are supplied; illustrative executable prices would undermine the cost cap.
The `run` command refuses to start without an explicit live confirmation, a positive wall-time cap,
and a positive cost cap. Before the first durable request claim, every route must also pass adapter,
transport, credential, exact-payload serialization, and conservative reservation preflight.
`.private/` is intentionally ignored so a private campaign file does not invalidate the clean-source
preflight; preserve that file securely outside public artifacts for exact resume. Tracked source
changes still fail closed.
Each non-placeholder route must bind an owner-declared expected SHA-256 for a separately retained
documentation/pricing bundle, public source URLs, and UTC retrieval time. The harness deliberately
labels that declaration unverified; a publication process must hash the external bytes before making
documented-limit or price claims. Its `request_timeout_seconds` is the hard full-stream deadline used
by every generated workload; raise it explicitly for slow long-output or long-context routes. The
plan reports each timeout and the possible final-request drain. HTTP requests explicitly use
`Accept-Encoding: identity`, so optional ambient compression packages cannot change wire behavior.

Seed, untriggered stop, and log-probability probes are explicitly labelled
`parameter_acceptance_only`; this harness does not claim those feature behaviors work because it
does not yet perform paired seed comparison, forced-stop scoring, or log-probability parsing.

## Campaign structure

Routes define exact serving identity: provider, model, API family/version, region, endpoint URL,
authentication environment-variable name, declared limits, prices, and capabilities. Suites define
load shape and replication. See [examples/digitalocean.yaml](examples/digitalocean.yaml).

The standard workload shapes are:

| Shape | Approximate input | Requested output | Main pressure |
|---|---:|---:|---|
| `short_short` | 256 tokens | 128 tokens | RPM and queueing |
| `long_short` | configured target; route-relative default up to 32,768 | 128 tokens | input TPM / prefill proxy |
| `short_long` | 256 tokens | configured target; 4,096 default | output TPM / decoding |
| `mixed` | deterministic heterogeneous mix | mixed | production-like contention |

`warmup`, `latency`, `aimd`, and `soak` accept identity-bound `long_input_tokens` and
`long_output_tokens`. This permits 100K-class long-input AIMD/soak cells and larger realized-output
cells when the exact route limits allow them. An explicit target defaults to
`long_input_overflow: fail` / `long_output_overflow: fail`: planning stops if the target cannot fit
the documented prompt-plus-output context or output limit. Choosing `clip` is deliberate and
visible—the realized token target changes the matched cell ID and metadata, so clipped and
unclipped routes are never silently pooled. Configured long targets require the corresponding
documented route limits. The unconfigured defaults are safely route-clipped and retain their
requested and realized targets separately.

Provider-reported token usage drives usage-based settled cost and effective TPM when it is complete.
The hard USD guard models prompt/input, cached-input, and completion/output token prices only.
Accordingly, generic `request_defaults` currently allow only the non-spend-multiplying `user` field;
audio/image generation, hosted search/tools, service tiers, multiple candidates, and vendor-specific
billable features fail configuration until their own reservation/pricing contract is implemented.
Missing usage is conservatively costed and censored from TPM; it is never converted to zero tokens.
The reservation binds the exact UTF-8 body and generator version, uses a byte-count upper bound plus
route overhead, and includes the full retry ceiling in plan totals.

## Scientific boundaries

- Arrivals are open-loop: scheduled arrival time is independent of completion time. Queue delay is
  retained, so coordinated omission cannot hide overload. Headline latency begins at the scheduled
  arrival and ends after the final attempt drains, including client queueing, retries, and backoff.
- AIMD first uses a bounded geometric bracket and then additive increases/multiplicative decreases
  to seek a capacity knee. A knee is bracketed only after two consecutive unhealthy epochs. If the
  configured epoch/rate ceiling stays healthy, the result is a right-censored highest-tested lower
  bound—not a knee or capacity ceiling. AIMD epochs are never called “sustainable” by themselves.
- Capacity blocks remain endpoint-isolated and sequential, but endpoint × shape order is a
  recorded deterministic seed shuffle (independent for AIMD and soak) to reduce time-order bias.
- Low-load measured suite blocks and their cells are also deterministically seed-shuffled, with the
  realized logical-request order persisted. Standalone warmup diagnostics remain first. A resume
  skips completed logical IDs without changing the relative order of pending cells.
- Soak blocks are fixed-rate evidence, not an automatic sustainable-capacity certificate. A
  sustained claim requires a predeclared endpoint × workload rate, adequate independent blocks,
  complete usage, and all health gates; the report generator does not make that claim for you.
- Zero-arrival, crash-partial, cost/time-guarded, 402-latched, and reservation-overrun epochs are
  retained as censored blocks. They never change AIMD state or enter provider-capacity intervals.
- Confidence intervals use requests for request metrics and epochs/blocks for load metrics. Output
  tokens are never treated as independent samples; load intervals carry an explicit block-level
  exchangeability assumption.
- p99 is withheld below 1,000 eligible observations.
- The client cannot directly observe server-side prefill compute. `TTFT - transport baseline` is at
  most an end-to-end prefill proxy and is labelled that way.
- Capability rejection is route/API-specific. A rejected image request does not prove that another
  checkpoint or provider supports no vision.
- Context results are fixed-anchor acceptance and three-marker retrieval screens. This release does
  not implement adaptive exact-boundary search or separately identify tool-schema/image token
  contributions; it does not make those claims.
- The `warmup` suite is standalone diagnostic traffic. It is not paired atomically with measured
  latency blocks, so every measured cell is reported as `warm_state=uncontrolled_not_paired`; the
  harness makes no warm- or cold-endpoint latency claim, including after a resume.
- A configured fallback output ceiling is only a local screen boundary. The harness emits a
  just-above expected-rejection probe only when the route has an explicit documented output limit;
  missing limits remain visible as inconclusive configuration cells.

## Data-quality contract

Nothing is silently trimmed. Every terminal attempt is classified as `valid`, `anomalous`,
`invalid`, or `censored`. Missing usage, impossible timing order, non-monotonic stream timestamps,
zero/near-zero decode duration with multiple output tokens, and inconsistent units are retained and
reported. Legitimate slow or fast extremes remain in the primary sample and are also listed in the
outlier audit. Invalid observations are excluded only from the estimand they cannot support; their
counts and reasons remain visible.

Hidden reasoning tokens are parsed when reported. Visible post-TTFT decode proxies are comparable
only in the explicit `reasoning_tokens == 0` stratum. Positive and unreported reasoning counts are
retained as counts and censored from that proxy; base reliability, cost, latency, and usage summaries
remain unconditional on this response-derived state.

Request-level TTFT and arrival/service latency percentiles are explicitly success-conditioned:
they include successful final logical outcomes only (and TTFT additionally requires an observed
first visible event). Their `n` is published. Failures remain in reliability, attempt, cost, and
coverage counts; this harness does not mislabel the success-conditioned latency distribution as an
all-outcome completion-time distribution.

Deterministic quality means end-to-end quality over every predeclared scored trial. Provider
non-success receives zero, and an unrelated timing/usage defect cannot remove a score from the
denominator. Unknown-after-claim and incomplete-retry trials also receive explicit zeros. The report
publishes scored, successful-response, non-success-zero, incomplete-retry-zero, and unscored counts;
binary-score 95% intervals use Wilson's binomial method rather than a degenerate bootstrap.

Matched-cell cost columns separate settled provider-priced cost from reservations retained for
unknown outcomes. Any per-success or per-token derived cost is explicitly named conservative
exposure and uses their sum; an unknown reservation is never mislabeled settled spend.

## Durable execution

SQLite is authoritative. Before a send, the runner atomically reserves the deterministic request ID
and worst-case cost. A crash leaves an `in_flight` row; on resume it becomes `unknown` and is never
automatically replayed. A prompt-free JSONL event projection is fsynced for easy inspection and can
be regenerated from SQLite. Every planned request/epoch is also registered before execution, so
completed, unsupported, inconclusive, untested, and cap/time-censored coverage remains explicit.
See [DATA-HANDLING.md](DATA-HANDLING.md).

## Reports

Reports compare matched endpoint × suite × workload cells only. Figures carry units, sample counts,
and 95% intervals. Capacity trajectories use route-specific small multiples. The report generator
does not create heterogeneous global scores or rank endpoints tested on different workloads.

The generated package is Markdown, CSV, JSON/JSONL, and PNG. It includes a reproducibility manifest
with artifact hashes and the software identifiers available to the reporting process. It does not
build a PDF, certify publication readiness, or replace a human claim/privacy/secret review.
Report generation fails closed unless SQLite has zero in-flight attempts, exactly one canonical
terminal event, and a complete ordered JSONL projection.

A 400/413/422 response on a deliberately over-limit probe is recorded as an observed validation
status only. Status alone does not prove that the intended parameter/context/output boundary was
enforced; that boundary stays inconclusive without a bound provider error reason and matched
successful control.

`campaign.public.json` is deliberately sanitized through an explicit allowlist. It preserves route
identity hashes but omits arbitrary request defaults, header values, authentication transport
details, URL queries/credentials, and unknown extension fields. Keep the private source campaign
configuration separately if exact operational reproduction requires omitted fields.

HTTP/2 is explicit and route-identity-bound (`http2: false` by default); it is never silently
enabled. Connection reuse, output-limit field, opaque quota scope, configurable quota-header names,
and client location are likewise part of the measurement contract. Configuration is strict:
unknown fields and zero/negative coverage counts fail before planning.

## Development

```bash
ruff check .
pytest
```

This repository contains no benchmark results, credentials, or DigitalOcean-specific conclusions.
