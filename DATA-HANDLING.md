# Data handling and publication notice

The default workloads are synthetic and deterministic. Users may replace them, but doing so changes
the data-governance boundary.

The durable ledger stores:

- route, suite, cell, and deterministic request identities;
- SHA-256 hashes of request payloads and outputs;
- timestamps, status classes, allow-listed headers, token usage, latency, quality score, cost, and
  validity flags;
- no prompt text, image bytes, model output, raw body, credential, or unrestricted header map.

The live process necessarily holds request and response content in memory long enough to send and
score it. Custom scorers should return a scalar and diagnostics without embedding response text.

Before publication:

1. inspect the exact run configuration and route labels for private account information;
2. recursively scan every artifact for secrets and local paths;
3. verify that prompt/output hashes cannot expose a low-entropy private value by enumeration;
4. publish only the sanitized report, aggregate tables, declared configuration, code revision, and
   any request-level records authorized for release;
5. state all censored, invalid, anomalous, and untested cells.

