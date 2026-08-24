# Adapter contract

An adapter implements a preflight plus two async methods:

```python
class Adapter(Protocol):
    def preflight(self, route: RouteConfig) -> None: ...
    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult: ...
    async def close(self) -> None: ...
```

The adapter owns translation and transport only. It must:

- obtain credentials solely from the environment variable named by the route;
- fail credential and transport checks before a durable request claim;
- use a client monotonic clock;
- stream the entire response or return a timeout/transport result;
- return provider-reported input, output, and cached-input usage without fabricating missing values;
- retain only allow-listed quota/request headers;
- always retain `Retry-After` so reporting-header customization cannot disable provider-directed
  retry backoff;
- return a SHA-256 digest rather than an error response body;
- keep response text in memory for deterministic scoring but never add it to the ledger;
- identify fallback/upstream behavior when a router is involved.

The OpenAI-compatible adapter additionally exposes `prepare`/`infer_prepared`: `prepare` produces
the exact canonical UTF-8 bytes, payload/generator hashes, and headers before a claim; the send uses
those same bytes without reserialization. Other adapters must provide an equivalent preclaim
materialization contract before they can be admitted.

HTTP/2 is opt-in (`false` by default) and identity-bound. Connection reuse and the connection-pool
ceiling are also explicit. HTTP transports use `trust_env=false`, so ambient proxies, netrc files,
and CA-bundle variables cannot silently change the measured route. Requests explicitly set
`Accept-Encoding: identity`; the profile is identity- and manifest-bound so optional Brotli/Zstd
installations cannot change response decoding. Non-streaming requests have
no TTFT; their header time and total completion time remain separate. A non-empty streaming result
requires a recognized content, refusal, reasoning, or tool-delta event **and** an explicit terminal
signal: either `[DONE]` or one terminal choice with a non-empty finish reason. A finish-only choice
is a valid empty model response. Exactly one choice is admitted: an absent choice index is treated
as positional zero, while a present malformed or nonzero choice index is a protocol error. Split
tool calls are reconstructed by tool index. A missing tool index uses its list position only for
compatibility; any present malformed tool index is a protocol error. Reasoning-only
streams are successful transport outcomes with no visible-answer TTFT; refusals are successful
transport outcomes that can fail the task-quality scorer. A bare `[DONE]`, clean EOF after partial
semantic output, conflicting/repeated finish choice, or malformed event is a protocol error, not a
success. Because `stream_options.include_usage` is not
universal, every route chooses the identity-bound `stream_usage_mode`: `omit` maximizes transport
compatibility and censors TPM when usage is absent, `try` requests usage but accepts it as missing,
and `required` additionally records missing usage as a contract violation. No adapter performs a
hidden fallback send. The route also declares an identity-bound `request_timeout_seconds`; it is a
hard deadline for the complete response stream, not merely response headers. Raise it deliberately
for slow long-output or very-long-context cells instead of editing adapter code.

## Included adapters

| Adapter | Status | Notes |
|---|---|---|
| `openai_compatible` | implemented | Generic Chat Completions transport |
| `digitalocean` | implemented | Same transport; use DO endpoint and token env |
| `azure_openai` | implemented | Route must contain the deployment URL/API version and API-key header |
| `vertex_openai` | **not implemented** | Fails closed until ADC/service-account OAuth refresh and expiry tests exist |
| `openrouter` | **request mapper only; live fail-closed** | Builds an exact pinned request, but needs generation lookup/actual-upstream verification before evidence-bearing use |
| `bedrock_native` | **not implemented** | Fails closed; add exact Converse/native model-region mapping and tests |
| `vertex_native` | **not implemented** | Fails closed; add Gemini-native content/tool/usage mapping and ADC refresh tests |
| `azure_model_inference_native` | **not implemented** | Fails closed; add exact Azure Model Inference mapping and tests |

The OpenAI-compatible path does not make native APIs equivalent. A provider/model/API/region route is
admitted only after an adapter contract test demonstrates request mapping, stream parsing, usage,
errors, and route identity.

## Adding an adapter

1. Implement the protocol without importing it into the benchmark core.
2. Add a registry name in `adapters/base.py`.
3. Test success, streaming, single/batched/empty/malformed SSE events, missing/nonintegral usage,
   hidden-reasoning and cache usage, split tools, structured output, provider errors, numeric and
   HTTP-date `Retry-After`, timeout, authentication omission, and body/header redaction.
4. Document which API family, model types, and regions were actually tested.
5. Keep unsupported capability states explicit; never translate a missing feature into apparent
   success.
