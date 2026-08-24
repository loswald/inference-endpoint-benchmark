# Security

## Reporting a vulnerability

Please report security issues privately to the repository owner. Do not open a public issue that
contains credentials, signed URLs, private endpoint names, account identifiers, prompts, outputs,
or response bodies.

## Secret handling

- Authentication configuration stores an environment-variable name, never its value. The sanitized
  public configuration additionally omits arbitrary headers/defaults and authentication transport
  details; the private input configuration still requires normal secret hygiene.
- Route identity hashes commit to omitted operational fields. Hashing is not secret storage:
  low-entropy private values may be guessable by enumeration, so credentials and private identifiers
  do not belong in request defaults or arbitrary headers.
- The planner never reads credentials.
- The runner reads the named variable during preflight before the first durable request claim.
- Credentials, prompts, media, model outputs, raw response bodies, and raw headers are not written
  to the public ledger.
- Error bodies are represented by a SHA-256 digest plus a sanitized error class.
- Only a fixed built-in allowlist of rate-limit, retry, and request-ID header names can be retained;
  configuration cannot add arbitrary header names.

Run directories should be treated as private until a separate recursive secret scan and human claim
review appropriate to the organization releasing them have passed. Report generation is not a
publication gate.
