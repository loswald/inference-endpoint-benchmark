# DigitalOcean hosted inference evidence atlas

The [PDF atlas](digitalocean-hosted-inference-evidence-atlas.pdf) is the readable entry point. It
covers the 11 current DigitalOcean-hosted open-model endpoints in this evidence package. Commercial
pass-through routes are excluded everywhere.

## What the evidence contains

- Capacity: all 44 exact endpoint × workload cells are represented. The combined capacity table
  uses 21 controllers from the corrected 2026-08-28 AIMD closure and 23 exact matched cells from the
  earlier verified six-hour campaign. The correction run's four-hour guard censored those 23 cells
  before start; the report does not relabel them as new evidence.
- Sustained load: all 44 current hosted-model × workload soak cells finished execution. Forty-one
  are scientifically complete; three are explicitly baseline transport-gated. Intervals use the four predeclared
  30-second blocks as the sampling units.
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
- `soak-cell-summary.csv` and `soak-block-summary.csv`: two-minute sustained-load outcomes and
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
Only completed fixed-rate soak blocks support sustained-load statements. Request, epoch, and block
confidence intervals use their declared independent sampling units; tokens are never treated as
independent samples. Throughput figures exclude rows that fail the explicit eligibility rules, and
the accompanying audit tables retain the exclusion reasons. No missing value is silently plotted as
zero or presented as a successful measurement.

PDF SHA-256: `bc02e51369a8687b204f10839c1f354517aad8fd38cc74be964f073f4a476375`.
The PDF writer runs in invariant mode: identical evidence tables and source code reproduce identical
PDF bytes.
