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
- generated `reproducibility-manifest.json`, including hashes of produced artifacts and explicit
  `not_recorded` values where the runtime could not determine a revision or termination state;
- all stopped/censored/untested cells and the reason for each;
- cumulative cost/time exposure including unknown or failed requests.

Run `plan` on the same configuration before reproduction. A changed provider alias, deployment,
upstream, API version, price, or capability snapshot is a new route—not a repeat.

The generated public campaign file is intentionally not a lossless operational configuration. It
omits arbitrary defaults, headers, authentication transport details, URL query data, and unknown
extensions. Preserve the private input configuration and a separately reviewed dependency lock or
container digest for exact reruns. The generated manifest records direct installed distributions;
it is not a transitive lockfile.
