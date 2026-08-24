# Provider admission

Run this gate separately for every provider account and exact serving route. A model name alone is
not a route identity.

## 1. Freeze the route

Record, with a retrieval timestamp and hash of the retained documentation bundle:

- provider, account/project, model ID, version or snapshot, API family and API version;
- region or global routing mode, deployment/endpoint ID, client location, and HTTP version;
- input, cached-input, output, image, tool, search, and other billable prices;
- context, output, image, request-rate, token-rate, concurrency, payload-size, and timeout limits;
- streaming, tools, structured output, vision, caching, logprobs, seed, stop, and sampling support;
- authentication type, token lifetime, refresh behavior, and quota/reset header semantics.

Unknowns stay unknown. Do not copy a limit or feature from another provider hosting the same model.

## 2. Admit the adapter

The adapter must prove, without using benchmark results as its own test oracle:

1. credential resolution does not print or persist the secret;
2. the exact serialized request matches the route contract;
3. a streamed control returns the expected marker and complete token usage;
4. request ID, model/version identity, finish reason, timing, usage, and safe quota signals parse;
5. a known validation error is classified separately from authentication, quota, and provider errors;
6. an interrupted request becomes `unknown` and is never automatically replayed;
7. every possible request is conservatively reserved under the campaign cost cap.

For router services, retain proof of the actual upstream. A requested provider preference is not
proof that the response came from that provider.

## 3. Preflight controls

Before paid breadth or load traffic, run a serial three-part control:

- control-plane/catalog request;
- short streamed exact-marker inference;
- repeat of the marker request on the same route.

Stop the lane on 401/403, 402, inconsistent model identity, incomplete settlement usage, or a failed
marker. These are route-access incidents, not capability observations.

## 4. Plan the envelope

The plan must state wall-time and cost caps, full retry ceilings, hard stream timeouts, maximum
simultaneous reservations, deterministic request identities, and the exact cells that can be
censored by a deadline or budget guard. Planning loads no credential and sends no traffic.

Only after these checks pass should `--confirm-live` be used.
