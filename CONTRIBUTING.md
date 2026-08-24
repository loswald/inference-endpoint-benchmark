# Contributing

Keep changes small, provider-neutral, and evidence-oriented. A new adapter or metric must include:

- an exact contract and unit definition;
- deterministic tests for success, errors, missing usage, streaming, and redaction;
- an honest unsupported state for surfaces it does not implement;
- no credential lookup during planning or reporting;
- no silent trimming, fallback, route aliasing, or fabricated token usage.

Before opening a pull request:

```bash
ruff check .
pytest
inference-bench plan examples/provider-stubs.yaml
```

`inference-bench plan examples/digitalocean.yaml` is a deliberate fail-closed check: the public
template has null prices and placeholder route metadata, so it must refuse planning until a local
copy is filled from current provider documentation.

Never commit a live run directory, credential, private prompt/output, provider response body, raw
header dump, account identifier, or signed URL.
