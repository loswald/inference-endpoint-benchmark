# Six-hour within-run variation profile

`six-hour-variation.yaml` is one portable experiment profile for the five admitted provider
catalogs. It measures seven matched panels: one at launch and one each hour for six hours. Every
panel contains four requests for every endpoint and workload shape—two stable exact-prefix
requests and two panel-unique cache-cold requests—launched on one global open-loop schedule at one
arrival per second with no retries.

## Schedule contract

- All arrivals in a panel are concurrent-capable (`concurrency: 256`); the runner never turns a
  panel into a serial loop.
- Provider sends stop at 21,840 seconds. The campaign wall/drain limit is 22,500 seconds.
- The process-independent schedule is anchored to the immutable `started_at_utc` in the campaign
  ledger. A resumed process waits for future registered panels and rejects an overdue panel; it
  does not silently start a new six-hour window.
- A restart inside a started panel retains the original per-request arrival offsets. Terminal or
  unknown request IDs are never replayed. If any still-unsent arrival is already in the past, the
  indivisible remainder is marked `time_censored` and the panel records an explicit
  `time_variation_panel_censored` event instead of launching those requests late.
- The registered 0.25-second lateness tolerance absorbs only ordinary scheduler wake-up jitter;
  it is shorter than the one-second inter-arrival interval. A panel already overdue when either a
  fresh or resumed invocation reaches it is censored even if no panel-start event was written.
- Credential-free compilation rejects a plan unless panel concurrency admits every arrival, each
  panel's launch span plus its longest route timeout fits its deadline, every final-panel arrival
  launches before the send cutoff, and the final response can drain before the wall limit.
- `interleave_gap_work` changes only whether optional benchmark suites may use protected idle
  intervals. It never changes how the required panels execute.

## Variation-only transport overrides

Provider catalogs retain their general-purpose request timeout. The experiment profile uses the
portable `provider_route_overrides` layer to tighten only this bounded panel study:

| Provider profile identity | Panel request timeout |
|---|---:|
| `digitalocean` | 360 seconds |
| `azure-ai-foundry` | 800 seconds |
| `amazon-bedrock` | 180 seconds |
| `google-vertex-ai` | 800 seconds |
| `alibaba-model-studio` | 720 seconds |

The same layer raises this experiment's connection-pool ceiling to 256. Composition precedence is
provider-profile defaults, catalog route, provider-scoped experiment override, then an optional
exact-route override. Every effective route setting is validated and bound into campaign identity.

Compile the profile with any admitted provider catalog using the normal credential-free command:

```text
inference-bench compile-profile PROVIDER_PROFILE \
  examples/experiment-profiles/six-hour-variation.yaml --output COMPILED_CAMPAIGN
```

No credential is read during composition or planning. The exact planned panel rows are 1,232 for
DigitalOcean, 560 for Azure AI Foundry, 112 for Amazon Bedrock, 112 for Google Vertex AI, and 896
for Alibaba Model Studio.

## Reporting contract

`stable_prefix` and `panel_unique_cold` are persisted request strata and part of both logical and
matched-cell identity. Reports estimate them separately, then compute explicitly labeled
panel-matched `panel_unique_cold minus stable_prefix` contrasts. Charts connect observations only
within the same stratum. The result supports a within-run six-hour variation statement—not a
24-hour, daily, diurnal, or indefinite production-sustainability claim.
