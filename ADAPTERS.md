# Adapter contract

An adapter implements two async methods:

```python
class Adapter(Protocol):
    async def infer(self, route: RouteConfig, request: RequestSpec) -> InferenceResult: ...
    async def close(self) -> None: ...
```

The adapter owns translation and transport only. It must:

- obtain credentials solely from the environment variable named by the route;
- use a client monotonic clock;
- stream the entire response or return a timeout/transport result;
- return provider-reported input, output, and cached-input usage without fabricating missing values;
- retain only allow-listed quota/request headers;
- return a SHA-256 digest rather than an error response body;
- keep response text in memory for deterministic scoring but never add it to the ledger;
- identify fallback/upstream behavior when a router is involved.

## Included adapters

| Adapter | Status | Notes |
|---|---|---|
| `openai_compatible` | implemented | Generic Chat Completions transport |
| `digitalocean` | implemented | Same transport; use DO endpoint and token env |
| `azure_openai` | implemented | Route must contain the deployment URL/API version and API-key header |
| `vertex_openai` | implemented | Caller supplies a current access token through the named env var |
| `openrouter` | **request mapper only; live fail-closed** | Builds an exact pinned request, but needs generation lookup/actual-upstream verification before evidence-bearing use |
| `bedrock_native` | **not implemented** | Fails closed; add exact Converse/native model-region mapping and tests |
| `vertex_native` | **not implemented** | Fails closed; add Gemini-native content/tool/usage mapping and tests |
| `azure_model_inference_native` | **not implemented** | Fails closed; add exact Azure Model Inference mapping and tests |

The OpenAI-compatible path does not make native APIs equivalent. A provider/model/API/region route is
admitted only after an adapter contract test demonstrates request mapping, stream parsing, usage,
errors, and route identity.

## Adding an adapter

1. Implement the protocol without importing it into the benchmark core.
2. Add a registry name in `adapters/base.py`.
3. Test success, streaming, single/batched SSE events, missing usage, tools, structured output,
   provider errors, 429 retry metadata, timeout, authentication omission, and body/header redaction.
4. Document which API family, model types, and regions were actually tested.
5. Keep unsupported capability states explicit; never translate a missing feature into apparent
   success.
