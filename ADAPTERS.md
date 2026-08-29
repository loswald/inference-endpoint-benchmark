# Provider adapters

Adapters translate the common benchmark request into a provider transport and normalize the result.
They do not change the workload, retry policy, AIMD controller, statistics, or report.

## Implemented transports

| Adapter | API | Authentication | Provider-specific checks |
|---|---|---|---|
| `digitalocean` | Chat Completions | bearer token | official inference host supplied by route |
| `bedrock_mantle` | Chat Completions | Bedrock API key | `api.aws` host and exact chat path |
| `bedrock_mantle_responses` | Responses | Bedrock API key | `api.aws` host and exact Responses path |
| `azure_model_inference` / `azure_openai` | Chat Completions | `api-key` or bearer | Azure Foundry/OpenAI host and exact chat path |
| `azure_responses` | Responses | `api-key` or bearer | Azure host and exact Responses path |
| `vertex_openai` | Chat Completions | service-account file or ADC | renewable OAuth and Google API host |
| `openrouter` | Chat Completions | bearer token | exact upstream pin, fallbacks off, response metadata attested |
| `openai_compatible` | Chat Completions | configurable header | generic protocol behavior only |

`bedrock_native`, `vertex_native`, and `azure_model_inference_native` are explicit placeholders.
Live preflight refuses them. The compatible transports above are real provider implementations, not
aliases that bypass credential refresh or provider routing checks.

## Transport behavior

Before a request begins, an adapter:

- loads or refreshes its credential;
- validates the provider hostname and API path;
- builds the final JSON body;
- validates the connection settings.

During the request it measures with a monotonic clock, consumes the complete response stream, and
records header time, first visible content, content-event offsets, final completion, status, usage,
finish reason, and allow-listed request/quota headers. The error body itself is never retained; only
its SHA-256 digest is recorded.

Chat streaming requires valid server-sent events and an explicit terminal signal. Split tool calls
are reconstructed by index. Refusals are successful transport outcomes and can still receive a
quality score of zero. Reasoning-only streams are transport successes but provide no visible-answer
TTFT.

Responses streaming recognizes output-text deltas and terminal `response.completed` or
`response.incomplete` events. Non-streaming Responses output reconstructs text and function calls
from the output-item array.

## OpenRouter attribution

The request contains exactly one upstream in `only` and `order`, with `allow_fallbacks: false` and
`require_parameters: true`. The adapter also enables OpenRouter routing metadata. It verifies that:

- exactly one endpoint is marked selected;
- the selected provider matches the configured upstream;
- every reported attempt remains inside that same pin.

Missing or mismatched metadata is a protocol failure. Performance is never attributed to an
unverified router path.

## Vertex credentials

`vertex_openai` uses the service-account file named by the route's authentication environment
variable when present. Otherwise it uses Google Application Default Credentials. Short-lived OAuth
tokens are refreshed before the request is admitted, so a multi-hour run does not depend on a
pasted access token.

## Adding a provider-native API

Add a native adapter only when it exposes a workload the compatible API cannot represent. A useful
adapter must have contract tests for:

- text and image input;
- tools and structured output;
- streaming and non-streaming;
- usage and cached/reasoning-token fields;
- throttling, retry hints, timeouts, and malformed responses;
- exact model/API/region identity;
- credential renewal over the intended run duration.

Catalog presence alone is not an implementation. A model becomes an admitted route only after its
exact provider/API/region call succeeds and its current limits and price are recorded.
