# Routing request-context plan

- Preserve the complete latest user message and append a small, bounded window of recent prior user messages for follow-ups. Do not add model calls or I/O, keep the existing no-user fallback, and leave recognized machine messages unchanged so anchored matching continues to work.
- Key cached decisions by a SHA-256 digest of the exact routing text sent to selection, retaining the current user scope. Keep reuse for identical requests and separate requests whose meaningful text differs after the former prefix.
- Add controlled-classifier tests through HTTP and WebSocket endpoints, including processed media, long inputs, prior user context, no-user routing, cache separation/reuse, and existing machine-route fixture preservation. First run changed-behavior tests against the current implementation to establish bounded red proofs.
