# DigitalOcean hosted inference: interim technical evidence

> **Not complete. Not a production qualification.** This report describes only the
> experiments that produced auditable evidence. A completed request schedule is never
> counted as a passing experiment.

## Executive truth

- **19/44** endpoint-workload cells have a repeat-confirmed adaptive-load rate bound.
- **3/44** 120-second fixed-rate tests passed every registered acceptance check; **38 failed** and **3 could not establish a reliable baseline**.
- **100/176** endpoint-capability checks produced complete evidence; **69 were inconclusive** and **7 were documented as unsupported**.
- **Six-hour time-of-day variation has not been measured yet.** A reserved section below defines the required matched panel; this report makes no full-day or diurnal claim.

## What the tests mean

- **Adaptive load search** raises offered request rate while the endpoint is healthy, reduces it after degradation, and requires three separated healthy confirmations before publishing a numeric bound. A confirmed lower bound is not a theoretical maximum.
- **120-second fixed-rate stability test** holds one candidate rate for four adjacent 30-second analysis blocks and then checks recovery. A pass requires every registered reliability, latency, queueing, usage, quality, and recovery condition to pass.
- **95% intervals** use the sampling unit printed with each result. The four stability blocks are adjacent in time, so their Student-t intervals are exploratory and do not model serial correlation.

## Exact workload recipes

The combined adaptive-load table contains two separately identified source recipes. They are shown separately and must not be compared as if 32K and 100K prompts were the same workload.

| Source | Workload | Registered recipe |
|---|---|---|
| `do-capacity-20260828-r2` | short prompt / short answer | 256-token prompt -> 128-token answer target |
| `do-capacity-20260828-r2` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) |
| `do-capacity-20260828-r2` | short prompt / long answer | 256-token prompt -> 4,096-token answer target |
| `do-capacity-20260828-r2` | seeded multi-workload mix | seeded four-way mix: 256->128, 100K/50K->128, 256->4,096, and 1,024->512 JSON |
| `do-sixhour-aimd-20260824-r1` | short prompt / short answer | short exact-answer prompt -> 64-token answer ceiling |
| `do-sixhour-aimd-20260824-r1` | 32K-token prompt / short answer | 32,000-token prompt -> 64-token answer ceiling |
| `do-sixhour-aimd-20260824-r1` | short prompt / long answer | short prompt -> 1,024-word target / 2,048-token ceiling |
| `do-sixhour-aimd-20260824-r1` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling |

The fixed-rate campaign used the historical 32K-era recipes:

| Workload | Registered recipe |
|---|---|
| short prompt / short answer | short exact-answer prompt -> 64-token answer ceiling |
| 32K-token prompt / short answer | 32,000-token prompt -> 64-token answer ceiling |
| short prompt / long answer | short prompt -> 1,024-word target / 2,048-token ceiling |
| seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling |

## Adaptive-load results

| Endpoint | Workload | Exact recipe | Result | Evidence source |
|---|---|---|---|---|
| `deepseek-v4-flash-0731` | short prompt / short answer | short exact-answer prompt -> 64-token answer ceiling | repeatedly passed through at least 32 RPS; ceiling not found | `do-sixhour-aimd-20260824-r1` |
| `deepseek-v4-flash-0731` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) | repeatedly passed through at least 0.5 RPS; ceiling not found | `do-capacity-20260828-r2` |
| `deepseek-v4-flash-0731` | short prompt / long answer | 256-token prompt -> 4,096-token answer target | repeatedly passed through at least 1 RPS; ceiling not found | `do-capacity-20260828-r2` |
| `deepseek-v4-flash-0731` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | repeatedly passed at 0.0666667 RPS; degraded by 0.2 RPS | `do-sixhour-aimd-20260824-r1` |
| `gemma-4-31B-it` | short prompt / short answer | 256-token prompt -> 128-token answer target | repeatedly passed at 0.25 RPS; degraded by 2.5 RPS | `do-capacity-20260828-r2` |
| `gemma-4-31B-it` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `gemma-4-31B-it` | short prompt / long answer | 256-token prompt -> 4,096-token answer target | repeatedly passed through at least 1 RPS; ceiling not found | `do-capacity-20260828-r2` |
| `gemma-4-31B-it` | seeded multi-workload mix | seeded four-way mix: 256->128, 100K/50K->128, 256->4,096, and 1,024->512 JSON | repeatedly passed through at least 2 RPS; ceiling not found | `do-capacity-20260828-r2` |
| `glm-5.2` | short prompt / short answer | 256-token prompt -> 128-token answer target | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `glm-5.2` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `glm-5.2` | short prompt / long answer | short prompt -> 1,024-word target / 2,048-token ceiling | healthy at 0.2 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |
| `glm-5.2` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | repeatedly passed at 0.0666667 RPS; degraded by 0.2 RPS | `do-sixhour-aimd-20260824-r1` |
| `kimi-k3` | short prompt / short answer | 256-token prompt -> 128-token answer target | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `kimi-k3` | 32K-token prompt / short answer | 32,000-token prompt -> 64-token answer ceiling | repeatedly passed through at least 0.8 RPS; ceiling not found | `do-sixhour-aimd-20260824-r1` |
| `kimi-k3` | short prompt / long answer | short prompt -> 1,024-word target / 2,048-token ceiling | healthy at 0.466667 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |
| `kimi-k3` | seeded multi-workload mix | seeded four-way mix: 256->128, 100K/50K->128, 256->4,096, and 1,024->512 JSON | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `mimo-v2.5-pro` | short prompt / short answer | short exact-answer prompt -> 64-token answer ceiling | repeatedly passed through at least 8 RPS; ceiling not found | `do-sixhour-aimd-20260824-r1` |
| `mimo-v2.5-pro` | 32K-token prompt / short answer | 32,000-token prompt -> 64-token answer ceiling | healthy at 0.8 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |
| `mimo-v2.5-pro` | short prompt / long answer | 256-token prompt -> 4,096-token answer target | repeatedly passed through at least 1 RPS; ceiling not found | `do-capacity-20260828-r2` |
| `mimo-v2.5-pro` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | healthy at 0.302163 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |
| `minimax-m2.5` | short prompt / short answer | 256-token prompt -> 128-token answer target | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `minimax-m2.5` | 32K-token prompt / short answer | 32,000-token prompt -> 64-token answer ceiling | repeatedly passed at 0.466667 RPS; degraded by 0.8 RPS | `do-sixhour-aimd-20260824-r1` |
| `minimax-m2.5` | short prompt / long answer | short prompt -> 1,024-word target / 2,048-token ceiling | repeatedly passed through at least 0.466667 RPS; ceiling not found | `do-sixhour-aimd-20260824-r1` |
| `minimax-m2.5` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | repeatedly passed through at least 1 RPS; ceiling not found | `do-sixhour-aimd-20260824-r1` |
| `nemotron-3-ultra-550b` | short prompt / short answer | short exact-answer prompt -> 64-token answer ceiling | healthy at 0.99393 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |
| `nemotron-3-ultra-550b` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `nemotron-3-ultra-550b` | short prompt / long answer | 256-token prompt -> 4,096-token answer target | repeatedly passed through at least 1 RPS; ceiling not found | `do-capacity-20260828-r2` |
| `nemotron-3-ultra-550b` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | repeatedly passed through at least 0.2 RPS; ceiling not found | `do-sixhour-aimd-20260824-r1` |
| `nvidia-nemotron-3-super-120b` | short prompt / short answer | 256-token prompt -> 128-token answer target | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `nvidia-nemotron-3-super-120b` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `nvidia-nemotron-3-super-120b` | short prompt / long answer | short prompt -> 1,024-word target / 2,048-token ceiling | healthy at 0.024582 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |
| `nvidia-nemotron-3-super-120b` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | no healthy result at the lowest tested rate (0.0666667 RPS) | `do-sixhour-aimd-20260824-r1` |
| `openai-gpt-oss-120b` | short prompt / short answer | short exact-answer prompt -> 64-token answer ceiling | repeatedly passed through at least 8 RPS; ceiling not found | `do-sixhour-aimd-20260824-r1` |
| `openai-gpt-oss-120b` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `openai-gpt-oss-120b` | short prompt / long answer | 256-token prompt -> 4,096-token answer target | test ran, but no defensible numeric rate was established | `do-capacity-20260828-r2` |
| `openai-gpt-oss-120b` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | repeatedly passed at 0.0666667 RPS; degraded by 0.2 RPS | `do-sixhour-aimd-20260824-r1` |
| `qwen3.5-397b-a17b` | short prompt / short answer | short exact-answer prompt -> 64-token answer ceiling | no healthy result at the lowest tested rate (1 RPS) | `do-sixhour-aimd-20260824-r1` |
| `qwen3.5-397b-a17b` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `qwen3.5-397b-a17b` | short prompt / long answer | 256-token prompt -> 4,096-token answer target | no healthy result at the lowest tested rate (0.25 RPS) | `do-capacity-20260828-r2` |
| `qwen3.5-397b-a17b` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | no healthy result at the lowest tested rate (0.0666667 RPS) | `do-sixhour-aimd-20260824-r1` |
| `qwen3.8-max` | short prompt / short answer | short exact-answer prompt -> 64-token answer ceiling | healthy at 2.46713 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |
| `qwen3.8-max` | 100K-token prompt / short answer | 100,000-token prompt -> 128-token answer target (50,000-token prompt for Minimax M2.5) | repeatedly passed through at least 0.5 RPS; ceiling not found | `do-capacity-20260828-r2` |
| `qwen3.8-max` | short prompt / long answer | short prompt -> 1,024-word target / 2,048-token ceiling | healthy at 0.913063 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |
| `qwen3.8-max` | seeded multi-workload mix | seeded five-way mix: short exact answer, 4,096-token context, 512-word answer, JSON, and tool call; 1,024-token ceiling | healthy at 2.00595 RPS once; not repeated | `do-sixhour-aimd-20260824-r1` |

## 120-second fixed-rate stability results

| Endpoint | Workload | Candidate rate | Outcome | Why it failed |
|---|---|---:|---|---|
| `deepseek-v4-flash-0731` | short prompt / short answer | 1 RPS | **passed** | - |
| `deepseek-v4-flash-0731` | 32K-token prompt / short answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `deepseek-v4-flash-0731` | short prompt / long answer | 1.6 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `deepseek-v4-flash-0731` | seeded multi-workload mix | 1.6 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `gemma-4-31B-it` | short prompt / short answer | 1 RPS | **passed** | - |
| `gemma-4-31B-it` | 32K-token prompt / short answer | 0.4 RPS | **failed** | p95 end-to-end latency exceeded 2x the low-load reference; p95 time to first token exceeded 2x the low-load reference |
| `gemma-4-31B-it` | short prompt / long answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; p95 end-to-end latency exceeded 2x the low-load reference; post-load recovery answers did not all pass quality checks |
| `gemma-4-31B-it` | seeded multi-workload mix | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `glm-5.2` | short prompt / short answer | 1 RPS | **failed** | p95 end-to-end latency exceeded 2x the low-load reference; loaded answer failed its quality check; quality fell under load; low-load reference answer failed its quality check; post-load recovery answers did not all pass quality checks; post-load quality remained more than 5 percentage points below low load |
| `glm-5.2` | 32K-token prompt / short answer | 0.4 RPS | **failed** | p95 time to first token exceeded 2x the low-load reference; p95 end-to-end latency exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `glm-5.2` | short prompt / long answer | 0.4 RPS | **failed** | p95 time to first token exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; p95 end-to-end latency exceeded 2x the low-load reference; post-load recovery answers did not all pass quality checks |
| `glm-5.2` | seeded multi-workload mix | 0.1 RPS | **failed** | p95 time to first token exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check |
| `kimi-k3` | short prompt / short answer | 1 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; p95 time to first token exceeded 2x the low-load reference; post-load recovery answers did not all pass quality checks; post-load quality remained more than 5 percentage points below low load |
| `kimi-k3` | 32K-token prompt / short answer | 2.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `kimi-k3` | short prompt / long answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `kimi-k3` | seeded multi-workload mix | 3.2 RPS | **failed** | success rate fell below 99%; more than 1% of requests were rate limited; low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `mimo-v2.5-pro` | short prompt / short answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; p95 time to first token exceeded 2x the low-load reference; post-load recovery answers did not all pass quality checks |
| `mimo-v2.5-pro` | 32K-token prompt / short answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; quality fell under load; post-load recovery answers did not all pass quality checks; post-load quality remained more than 5 percentage points below low load; post-load p95 time to first token remained above 2x the low-load reference |
| `mimo-v2.5-pro` | short prompt / long answer | 0.4 RPS | **failed** | p95 time to first token exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `mimo-v2.5-pro` | seeded multi-workload mix | 0.2 RPS | **failed** | p95 time to first token exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks; post-load quality remained more than 5 percentage points below low load |
| `minimax-m2.5` | short prompt / short answer | 16 RPS | **failed** | success rate fell below 99%; more than 1% of requests were rate limited; p95 time to first token exceeded 2x the low-load reference; p95 end-to-end latency exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; the request queue kept growing; post-load recovery answers did not all pass quality checks |
| `minimax-m2.5` | 32K-token prompt / short answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks; post-load p95 latency remained above 2x the low-load reference |
| `minimax-m2.5` | short prompt / long answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `minimax-m2.5` | seeded multi-workload mix | 0.2 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks; post-load quality remained more than 5 percentage points below low load |
| `nemotron-3-ultra-550b` | short prompt / short answer | 1 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; p95 end-to-end latency exceeded 2x the low-load reference; post-load recovery answers did not all pass quality checks; post-load p95 time to first token remained above 2x the low-load reference; post-load p95 latency remained above 2x the low-load reference |
| `nemotron-3-ultra-550b` | 32K-token prompt / short answer | 0.8 RPS | **failed** | p95 end-to-end latency exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `nemotron-3-ultra-550b` | short prompt / long answer | 1 RPS | **failed** | p95 end-to-end latency exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; p95 time to first token exceeded 2x the low-load reference; post-load recovery answers did not all pass quality checks |
| `nemotron-3-ultra-550b` | seeded multi-workload mix | 0.4 RPS | **failed** | p95 end-to-end latency exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; p95 time to first token exceeded 2x the low-load reference; post-load recovery answers did not all pass quality checks; post-load p95 latency remained above 2x the low-load reference |
| `nvidia-nemotron-3-super-120b` | short prompt / short answer | - | **could not start** | a reliable low-load baseline could not be established |
| `nvidia-nemotron-3-super-120b` | 32K-token prompt / short answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `nvidia-nemotron-3-super-120b` | short prompt / long answer | 0.1 RPS | **failed** | p95 end-to-end latency exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `nvidia-nemotron-3-super-120b` | seeded multi-workload mix | - | **could not start** | a reliable low-load baseline could not be established |
| `openai-gpt-oss-120b` | short prompt / short answer | 32 RPS | **failed** | p95 time to first token exceeded 2x the low-load reference; p95 end-to-end latency exceeded 2x the low-load reference; loaded answer failed its quality check; quality fell under load; the request queue kept growing; low-load reference answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `openai-gpt-oss-120b` | 32K-token prompt / short answer | 2.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `openai-gpt-oss-120b` | short prompt / long answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `openai-gpt-oss-120b` | seeded multi-workload mix | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `qwen3.5-397b-a17b` | short prompt / short answer | 16 RPS | **failed** | success rate fell below 99%; more than 1% of requests were rate limited; low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `qwen3.5-397b-a17b` | 32K-token prompt / short answer | 0.4 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks; post-load p95 time to first token remained above 2x the low-load reference |
| `qwen3.5-397b-a17b` | short prompt / long answer | 0.2 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; p95 time to first token exceeded 2x the low-load reference; post-load recovery answers did not all pass quality checks |
| `qwen3.5-397b-a17b` | seeded multi-workload mix | - | **could not start** | a reliable low-load baseline could not be established |
| `qwen3.8-max` | short prompt / short answer | 1 RPS | **passed** | - |
| `qwen3.8-max` | 32K-token prompt / short answer | 0.2 RPS | **failed** | low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `qwen3.8-max` | short prompt / long answer | 0.4 RPS | **failed** | p95 time to first token exceeded 2x the low-load reference; p95 end-to-end latency exceeded 2x the low-load reference; low-load reference answer failed its quality check; loaded answer failed its quality check; post-load recovery answers did not all pass quality checks |
| `qwen3.8-max` | seeded multi-workload mix | 0.1 RPS | **failed** | low-load reference answer failed its quality check; p95 end-to-end latency exceeded 2x the low-load reference; loaded answer failed its quality check |

## Six-hour matched variation panel — pending

No six-hour panel is present in the retained evidence. The closure experiment must repeat the same endpoint, workload recipe, offered rate, region, and acceptance checks at predeclared times across a six-hour window. Results belong here only after all matched panels and their request-level receipts are verified.

This six-hour panel can support a six-hour within-run variation statement. It cannot support a 24-hour, daily, or diurnal claim.

## Engineering use

The current evidence is useful for reproducing observed behavior and selecting cells for follow-up. It is not sufficient for a provider-wide production approval. Treat every unconfirmed, failed, could-not-start, inconclusive, or unmeasured cell as an open engineering risk—not as zero performance and not as implicit support.
