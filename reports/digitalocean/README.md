# DigitalOcean hosted inference: current evidence status

**This benchmark is not complete and is not a production-qualification report.** The current PDF
atlas has been withdrawn as a decision aid while two publication defects are corrected: it counted
finished fixed-rate tests as successes even when they failed the registered acceptance criteria,
and one comparison combined 32K-input and 100K-input experiments under a 32K label.

The evidence package covers the 11 DigitalOcean-hosted open-model endpoints that were in scope.
Commercial pass-through routes, including Arcee, are excluded everywhere.

The current clean rebuild is the
[interim evidence atlas](../../published/digitalocean-atlas/digitalocean-hosted-inference-evidence-atlas.pdf).
It is intentionally explicit about unfinished evidence; the six-hour matched-panel study described
in the atlas is the next live experiment, not a result already obtained.

## What is actually established

| Experiment | Established result | Unresolved or failed |
|---|---:|---:|
| Adaptive load search, 44 endpoint-by-workload cells | 19 repeatedly confirmed numeric rate bounds | 9 single observations; 15 failed at the lowest tested rate; 1 no numeric bound |
| 120-second fixed-rate stability tests | 3 passed | 38 measured failures; 3 could not establish a reliable baseline |
| Static/capability refresh, 2,891 planned cells | 564 completed | 2 inconclusive; 2,325 stopped by the time limit |
| Published endpoint-by-dimension matrix, 176 rows | 100 completed; 7 documented unsupported | 69 inconclusive |
| Prompt-cache matched pairs | 5 of 11 endpoints measured | 6 endpoints not reached |

The three fixed-rate passes were only for the short-input/short-output workload at 1 request per
second: DeepSeek V4 Flash, Gemma 4 31B, and Qwen3.8 Max. No long-input, long-output, or mixed-workload
cell passed the full registered stability criteria. A finished test is not a passing test.

## Plain-language test names

- **Adaptive load search** (internal code name: `aimd`) increases traffic while an endpoint remains
  healthy, reduces traffic after degradation, then requires three separated confirmations before a
  rate is called repeatedly passing.
- **120-second fixed-rate stability test** (internal code name: `soak`) sends requests at one target
  rate for four contiguous 30-second analysis blocks. Its four-block intervals are exploratory:
  the blocks are adjacent in time and serial correlation was not modelled.
- **Passed** means the registered reliability, latency, queueing, usage, quality, and recovery
  criteria all held. **Measured failure** means the test ran but one or more criteria failed.

Until the missing experiments are run and the contradictory support matrices are reconciled, use
these files as an auditable partial dataset—not as a provider-wide guarantee or a production
deployment recommendation.

## What the evidence contains

- Capacity: all 44 exact endpoint × workload rows are represented. Representation is not scientific
  completion. The combined capacity table
  uses 21 controllers from the corrected 2026-08-28 AIMD closure and 23 exact matched cells from the
  earlier verified six-hour campaign. The correction run's four-hour guard censored those 23 cells
  before start; the report does not relabel them as new evidence.
- Fixed-rate stability: all 44 current hosted-model × workload cells finished execution bookkeeping.
  Forty-one yielded analyzable outcomes: 3 passes and 38 measured failures. Three could not establish
  a reliable baseline. Intervals use four predeclared, contiguous 30-second analysis blocks and are
  exploratory rather than independent-repeat confidence intervals.
- Static verification: 217 matched verification cells across 34 endpoint × suite groups and 644
  attempts provide independent caching, capability, context, output, quality, interaction, latency,
  and warm-up evidence. Unreached cells remain labelled as not measured.
- Prompt caching: DigitalOcean's hosted open models use automatic, best-effort exact-prefix caching.
  Matched pairs report observed cached-token counters, TTFT when exposed, and settled-cost ratios;
  missing endpoint pairs are not interpreted as lack of support.
- Batch inference: DigitalOcean documents batch inference as unavailable for these open-model
  endpoints. This product-contract state is separate from measured request failures.

## Machine-readable tables

- `endpoint-inventory.csv` and `endpoint-summary.csv`: exact route identity and endpoint summaries;
- `capacity-summary.csv`: combined AIMD evidence by endpoint × workload;
- `capacity-provenance-manifest.json`: per-campaign capacity-cell provenance and source hashes;
- `capacity-controller-summary-20260828.csv`, `capacity-load-block-summary-20260828.csv`, and
  `capacity-coverage-ledger-20260828.csv`: corrected closure controller, epoch, and coverage audit;
- `soak-cell-summary.csv` and `soak-block-summary.csv`: 120-second fixed-rate outcomes and
  block-level uncertainty;
- `static-verification-summary.csv`, `cache-verification-pairs.csv`, and
  `static-verification-manifest.json`: independent static and cache verification;
- `capability-evidence.csv` and `observed-limits.csv`: functional results and measured boundaries;
- `quality-pair-summary.csv` and `recovery-summary.csv`: matched quality and overload recovery;
- `coverage-matrix.csv` and `scope-exclusions.csv`: measured, unsupported, transport-gated,
  censored, and not-measured states;
- `cache-state-metrics.csv`: cache strata retained from the source evidence;
- `public-safety-scan.json`: recursive publication-safety scan result.

Read capacity only at the exact endpoint × workload level. Confirmed AIMD points are observed
healthy lower bounds or brackets, not theoretical maxima. Exploratory observations remain labelled.
Only rows with `soak_acceptance_pass=true` support a passing fixed-rate claim. Request and epoch
intervals use their declared sampling units. Four-block fixed-rate intervals are exploratory because
the blocks are contiguous; tokens are never treated as independent samples. Throughput figures
exclude rows that fail the explicit eligibility rules, and
the accompanying audit tables retain the exclusion reasons. No missing value is silently plotted as
zero or presented as a successful measurement.

The previously published PDF SHA-256
`bc02e51369a8687b204f10839c1f354517aad8fd38cc74be964f073f4a476375` is retained only for audit and
must not be used for engineering decisions.
