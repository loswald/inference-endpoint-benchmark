# Reproducibility checklist

Retain the following for every evidence-bearing campaign:

- exact clean Git commit, clean-tree verification hash, Python/dependency versions, and
  dependency-lock hash;
- sanitized campaign configuration and its identity hash;
- exact route/model/API/version/region identifiers;
- an owner-declared expected lowercase SHA-256 for a separately retained capability/limit/pricing
  bundle per route, separate public documentation and pricing source URLs, and an exact UTC
  retrieval timestamp;
- normalized exact invocation plus a raw-invocation hash, campaign seed, start/end UTC, client
  region, HTTP/2 choice, connection-reuse policy, explicit connection-pool ceiling, and
  `trust_env=false` transport policy;
- authoritative SQLite ledger plus its SHA-256 digest;
- prompt-free JSONL event projection;
- matched-cell summary, metric contract, request-level outlier audit, and report;
- generated `reproducibility-manifest.json`, including hashes of produced artifacts, the exact run
  source/environment, and the separately captured report-generator source/environment;
- all stopped/censored/untested cells and the reason for each;
- cumulative cost/time exposure including unknown or failed requests.

Run `plan` on the same configuration before reproduction. A changed provider alias, deployment,
upstream, API version, price, or capability snapshot is a new route—not a repeat.

The declared evidence-bundle digest, safe source locators, retrieval timestamp, and full-stream
request timeout are route-identity fields and are emitted in both the sanitized campaign and
reproducibility manifest. This harness does **not** open or hash the external evidence bundle; it
labels the declaration unverified. Retain the exact bundle outside public run artifacts and verify
its bytes against the declared digest before publishing documented-limit or price claims.

The generated public campaign file is intentionally not a lossless operational configuration. It
omits arbitrary defaults, headers, authentication transport details, URL query data, and unknown
extensions. Preserve the private input configuration and a separately reviewed dependency lock or
container digest for exact reruns. This repository includes a hash-bound reference
`requirements.lock`; install it before the editable package as shown in the quick start. The
generated run and report manifests fail closed unless every hash-bound runtime/transitive pin is
installed at the exact version, and record that verified lock closure plus this package. Unrelated
ambient package names are deliberately excluded from public-candidate artifacts. Dirty source is
refused rather than represented by an unreconstructible digest. The normalized invocation replaces local
configuration/output paths, while the raw form is retained only as a SHA-256 digest.

Live execution and reporting require clean committed source and fail closed when their Git revision,
dependency lock, or required runtime identity cannot be resolved. The transport fixes
`Accept-Encoding: identity`, preventing ambient optional compression packages from changing wire
behavior under the same identity. The live runner rechecks the exact
identity after lazy adapter preflight and again at terminal; a drift event makes the report refuse the
campaign. SQLite is authoritative; a failed prompt-free
JSONL append marks the projection dirty and the report path rebuilds it deterministically rather than
reclassifying a settled provider request.

The report acquires the same kernel-backed exclusive campaign lease as the live runner, checkpoints
the SQLite WAL before hashing, and re-verifies an identical clean source/locked environment before
derivation, before manifest export, and after export. A concurrent runner or source transition makes
report generation fail; it is never attributed to whichever revision happened to be visible last.
