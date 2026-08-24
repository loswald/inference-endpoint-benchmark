# Reproducibility checklist

Retain the following for every evidence-bearing campaign:

- exact Git commit and Python/dependency versions;
- sanitized campaign configuration and its identity hash;
- exact route/model/API/version/region identifiers;
- documented capability, limit, and pricing snapshots with retrieval dates;
- campaign seed, start/end UTC, client region, and connection-reuse policy;
- authoritative SQLite ledger plus its SHA-256 digest;
- prompt-free JSONL event projection;
- matched-cell summary, metric contract, request-level outlier audit, and report;
- all stopped/censored/untested cells and the reason for each;
- cumulative cost/time exposure including unknown or failed requests.

Run `plan` on the same configuration before reproduction. A changed provider alias, deployment,
upstream, API version, price, or capability snapshot is a new route—not a repeat.

