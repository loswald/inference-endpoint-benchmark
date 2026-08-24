# Data handling and release-review notice

The default workloads are synthetic and deterministic. Users may replace them, but doing so changes
the data-governance boundary.

The durable ledger stores:

- route, suite, cell, and deterministic request identities;
- SHA-256 hashes of the exact request bytes, generator-bound payload, and text-plus-tool output;
- timestamps, status classes, allow-listed headers, token usage, latency, quality score, cost, and
  validity flags;
- no prompt text, image bytes, model output, raw body, credential, or unrestricted header map.

Provider finish reasons are reduced at adapter, result, ledger, and report boundaries to a fixed
public enum; unknown provider-controlled text becomes `other` and cannot enter the prompt-free event
projection or aggregate CSV.

Provider-reported cached-input and hidden-reasoning token counts are retained as separate nullable
fields. An explicit zero is different from missing/unknown. Wrong-type, fractional, negative, or
otherwise inconsistent usage is retained as a parse/validity error and cannot become TPM or
usage-priced cost.

The live process necessarily holds request and response content in memory long enough to send and
score it. Custom scorers should return a scalar and diagnostics without embedding response text.

The generated report is a release candidate, not publication approval. Before any external release:

1. inspect the exact run configuration and route labels for private account information;
2. recursively scan every artifact for secrets and local paths;
3. verify that prompt/output hashes cannot expose a low-entropy private value by enumeration;
4. publish only the sanitized report, aggregate tables, declared configuration, code revision, and
   any request-level records authorized for release;
5. state all censored, invalid, anomalous, and untested cells.

No PDF builder, publication gate, legal review, or organization-specific secret scanner is included.
The public campaign serializer uses an explicit allowlist and omits operational fields, but this is
defense in depth—not proof that a full run directory is safe to publish.
