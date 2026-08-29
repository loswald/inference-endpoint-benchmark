# DigitalOcean hosted open-model inference benchmark

This directory retains the dated machine-readable evidence for the 11 DigitalOcean-hosted
open-model endpoints frozen for measurement on 28-29 August 2026. **Publication is currently
withheld.** The previous PDF and figures mislabeled unresolved lower-rate searches as endpoint
failures and collapsed distinct fixed-rate states into a single red failure state; they have been
removed until regenerated from the corrected semantics. The tables below remain auditable partial
evidence, not a completeness certificate or production recommendation.

## What the current evidence establishes

| Evidence layer | Current evidence |
|---|---|
| Six-hour matched low-load study | 7/7 hourly panels, 1,232/1,232 scheduled requests, all 11 endpoints and four workload families, stable repeated-prefix and panel-unique fresh-prefix strata |
| Adaptive load search | 24/44 endpoint-by-workload cells have repeated numeric evidence: 19 tested lower bounds and 5 confirmed brackets. The other 20 searches stopped after one unhealthy 0.5 req/s starting probe; lower rates were not tested and those cells are unresolved, not endpoint failures. |
| Two-minute fixed-rate test | All 44 cells have an execution record: 3 passed every registered condition, 38 did not meet every condition at the tested rate, and 3 were transport-gated with no scientific pass/fail result. |
| Static and capability refresh | 564/2,891 planned cells completed, 2 were inconclusive, and 2,325 were stopped by the phase time limit. Preserve these as partial evidence, not a complete capability matrix. |
| Context and output probes | One explicit measured state per endpoint; accepted values remain lower bounds unless an accept/reject interval was observed |

The six-hour study completed 1,135 of 1,232 requests successfully (92.1%; Wilson 95% interval
90.5-93.5%). Eight endpoints completed all 112 of their scheduled requests. Gemma 4 31B completed
105/112, Qwen3.5 397B A17B completed 72/112, and Nemotron 3 Super 120B completed 62/112.
Rate limits and timeouts are counted as outcomes, not erased as missing rows.

## How to read the results

- **Adaptive load search** increases offered traffic while an endpoint remains healthy, backs down
  after degradation, and requires three separated healthy confirmations before reporting a tested
  lower bound or bracket. These are observations for the exact request recipe, not theoretical maxima
  or automatically safe production settings.
- **Two-minute fixed-rate test** runs one candidate rate for four adjacent 30-second analysis blocks.
  A pass requires every registered reliability, latency, queueing, usage, quality, and recovery check.
  The block intervals are exploratory because adjacent blocks are not independent repeats.
- **Six-hour matched low-load study** repeats the same endpoint-by-workload design once per hour over
  a single six-hour window. It estimates variation within that run. It does not establish a daily,
  diurnal, or indefinite production pattern.
- **Stable prefix / fresh prefix** compares an exact repeated token prefix with a panel-unique fresh
  prefix. DigitalOcean documents caching as automatic and best effort. The report shows cached-token,
  latency, and cost effects separately by endpoint and recipe rather than pooling unlike workloads.
- **Eligible decode rate** is reported only when timestamps and token usage are complete, the decode
  window is at least one second, and at least 16 output tokens were produced. Every exclusion and
  reason remains in the audit table, preventing impossible token-per-second outliers.

## Product contracts kept separate from live probes

DigitalOcean documents automatic best-effort exact-prefix prompt caching for hosted open models and
documents batch inference as unsupported for hosted open models. Vision labels distinguish the
DigitalOcean catalog contract from live probe behavior; a failed endpoint probe does not prove that
an upstream model family lacks that capability.

## Machine-readable package

The evidence directory contains:

- endpoint inventory, adaptive-load, fixed-rate, capability, context/output, quality, recovery, and
  six-hour variation tables;
- request-level and panel-level 95% intervals, matched cache comparisons, and an outlier audit;
- source/provenance manifests.

No blank cell is silently interpreted as zero or success. The tables use plain states such as
`passed`, `tested-rate non-pass`, `transport-gated`, `unsupported by product contract`,
`inconclusive`, `stopped by time limit`, and `not run for this exact recipe`.

## Reproduce and verify

From the repository root, with the verified aggregate run directory available:

```powershell
$env:PYTHONPATH = "src"
python scripts/build-digitalocean-final-report.py `
  --run-dir <verified-six-hour-run-directory> `
  --summary-dir reports/digitalocean `
  --output published/digitalocean-final-report
python -m pytest tests/test_digitalocean_final.py tests/test_digitalocean_variation.py -q
python scripts/verify-digitalocean-final-publication.py published/digitalocean-final-report
```

The eventual build command fails closed on unsafe public content and writes both the recursive
safety receipt and deterministic SHA-256 publication manifest. The verification command checks the complete
endpoint/workload matrices, all six-hour tables, nonblank coverage and limit states, PDF page and
endpoint coverage, figure inventory and dimensions, safety receipt, and every published file hash.
The source commit, campaign hash, estimator definitions, and sampling units are recorded in the PDF
and machine-readable manifests.
