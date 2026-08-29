# DigitalOcean hosted open-model inference benchmark

This is the dated, reproducible evidence package for the 11 DigitalOcean-hosted open-model
endpoints frozen for measurement on 28-29 August 2026. It is designed for engineers choosing an
endpoint and for researchers auditing exactly what was tested. It is not a claim about models added
to DigitalOcean after the catalog freeze.

The finished PDF is
[`digitalocean-inference-endpoints-technical-benchmark-2026-08-29.pdf`](../../published/digitalocean-final-report/digitalocean-inference-endpoints-technical-benchmark-2026-08-29.pdf).

## What is complete

| Evidence layer | Finished evidence |
|---|---|
| Six-hour matched low-load study | 7/7 hourly panels, 1,232/1,232 scheduled requests, all 11 endpoints and four workload families, stable repeated-prefix and panel-unique fresh-prefix strata |
| Adaptive load search | All 44 endpoint-by-workload cells represented: 19 repeatedly passing tested lower bounds, 5 repeatedly confirmed brackets, and 20 measured failures at the lowest tested rate |
| Two-minute fixed-rate test | All 44 cells represented: 3 passed every registered condition, 38 measured failures, and 3 cases where a reliable baseline could not be established |
| Static and capability refresh | 564 completed cells retained with explicit measured, unsupported, inconclusive, or stopped-by-time-limit states |
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

The publication directory contains:

- the PDF report and directly labelled figures;
- endpoint inventory, adaptive-load, fixed-rate, capability, context/output, quality, recovery, and
  six-hour variation tables;
- request-level and panel-level 95% intervals, matched cache comparisons, and an outlier audit;
- source/provenance manifests and a recursive public-safety scan.

No blank cell is silently interpreted as zero or success. The tables use plain states such as
`passed`, `measured failure`, `unsupported by product contract`, `inconclusive`, and `not run for this
exact recipe`.

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

The build command fails closed on unsafe public content and writes both the recursive safety receipt
and deterministic SHA-256 publication manifest. The verification command checks the complete
endpoint/workload matrices, all six-hour tables, nonblank coverage and limit states, PDF page and
endpoint coverage, figure inventory and dimensions, safety receipt, and every published file hash.
The source commit, campaign hash, estimator definitions, and sampling units are recorded in the PDF
and machine-readable manifests.
