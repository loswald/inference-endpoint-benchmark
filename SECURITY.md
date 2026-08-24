# Security

## Reporting a vulnerability

Please report security issues privately to the repository owner. Do not open a public issue that
contains credentials, signed URLs, private endpoint names, account identifiers, prompts, outputs,
or response bodies.

## Secret handling

- Configuration stores only an environment-variable name, never its value.
- The planner never reads credentials.
- The runner reads the named variable only immediately before live execution.
- Credentials, prompts, media, model outputs, raw response bodies, and raw headers are not written
  to the public ledger.
- Error bodies are represented by a SHA-256 digest plus a sanitized error class.
- Only allow-listed rate-limit and request-ID header names are retained.

Run directories should be treated as private until the report and ledgers pass a separate recursive
secret scan appropriate to the organization publishing them.

