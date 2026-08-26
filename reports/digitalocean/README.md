# DigitalOcean hosted inference evidence atlas

The [PDF atlas](digitalocean-hosted-inference-evidence-atlas.pdf) is the readable entry point. It
contains a decision map, matched capacity and sustained-soak plates, a capability matrix, and one
evidence sheet for each of the 12 exact DigitalOcean-hosted endpoints in the campaign.

The adjacent CSV files are the compact, machine-readable evidence behind the figures:

- `endpoint-inventory.csv` and `endpoint-summary.csv`: route identity and endpoint-level counts;
- `capacity-summary.csv`: AIMD healthy observations and bounds by exact workload;
- `soak-cell-summary.csv` and `soak-block-summary.csv`: two-minute fixed-rate outcomes and block
  uncertainty;
- `capability-evidence.csv`: transport acceptance and deterministic functional scoring;
- `observed-limits.csv`: context, output, tool, and vision boundary observations;
- `quality-pair-summary.csv`: matched low-load versus stressed task quality;
- `recovery-summary.csv`: post-overload recovery evidence;
- `coverage-matrix.csv` and `scope-exclusions.csv`: what was measured and what was not;
- `cache-state-metrics.csv`: explicitly separated cache strata;
- `public-safety-scan.json`: recursive publication scan result for the source summary package.

Interpret figures at the endpoint × workload level. AIMD points are bounds on the observed knee;
only completed fixed-rate soak blocks support sustained-capacity language. Missing evidence is
labelled instead of imputed, and tokens are never treated as independent statistical samples.

PDF SHA-256: `c091dfbe420bd3363d8f6e24de69ec2343f98776bc01dc97bd67d78a7dabd558`.
The PDF writer runs in invariant mode: identical evidence tables and source code reproduce the same
PDF bytes, not merely an equivalent visual document.
