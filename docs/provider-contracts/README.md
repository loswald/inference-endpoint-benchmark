# Provider contracts

These files separate three different claims that must never be collapsed:

1. **Documented** means the provider's primary documentation states the property.
2. **Live-proved** means a sanitized receipt confirms one exact model, API family, and region.
3. **Runnable** means this library has an executable adapter for that exact transport and a
   credential-free profile can pass planning before any output directory or provider request is
   created.

The Amazon Bedrock contract is runnable only for `zai.glm-4.7-flash` through Mantle Chat
Completions in `us-east-1`. The Google Vertex contract is runnable only for
`google/gemini-3.6-flash` through the global OpenAI-compatible Chat Completions endpoint. Both
Vertex streaming and nonstreaming transports passed sanitized canaries on 2026-08-30; this proves
transport and complete usage reporting, not benchmark performance. The separate native
`generateContent` transport remains outside the executable profile.

The Azure contract admits five exact text deployments and one separately planned
`text-embedding-3-large` route. Its embedding adapter and public 11-cell contract profile are
implemented, but no terminal embedding benchmark is published. The Alibaba profile contains eight
exact Singapore text routes plus a separate `text-embedding-v4` profile. Checked-in public receipts
currently live-prove only `qwen3.8-flash`; the other seven text routes are documentation-grounded
and present in the authenticated catalog, which is not proof that they are callable. The active
eight-route campaign remains non-terminal and is not evidence until its ledger closes and passes
review.

Compile the runnable Bedrock route without resolving a credential:

```console
inference-bench compile-profile \
  examples/provider-profiles/amazon-bedrock-mantle-us-east-1.yaml \
  examples/experiment-profiles/standard-static-and-capacity.yaml \
  --output .private/amazon-bedrock-glm47-flash.yaml

inference-bench compile-profile \
  examples/provider-profiles/google-vertex-gemini36-flash-global.yaml \
  examples/experiment-profiles/standard-static-and-capacity.yaml \
  --output .private/google-vertex-gemini36-flash.yaml
```

Planning or running the compiled file still requires the normal experiment review. Live execution
also requires `BEDROCK_API_KEY` for Bedrock or Google Application Default Credentials for Vertex;
credential values never belong in profiles, contracts, commands, or receipts.
