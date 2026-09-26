# Fork changeset

`swxtch` is upstream [`ulab-uiuc/LLMRouter`](https://github.com/ulab-uiuc/LLMRouter) `main` plus the changes below.
Changes are never pushed upstream; the `upstream` remote's push URL is disabled.

Status: **fork-only** means the change exists only here; **upstreamable** means it fixes upstream's own code and
could be offered back.

## Upstream syncs

- **2026-09-26** Fork base: `swxtch` branched from upstream `main` at `d1490a3`
  (Merge pull request #199 from onepounchman/feat/racer-router).

## Fork changes

### OpenClaw router

- **#1** `openclaw_router/opencode.yaml` routes opencode requests by complexity across luna-max, glm-5.3-flash
  and glm-5.3, classified by gpt-oss-120b at low effort (classifier only, never a target;
  `scripts/eval_opencode_classifier.py` scores candidates). Router: configurable classifier prompt, token budget
  and field, temperature (or none), `extra_body`, timeout and fallback; a decision cache
  (`cache_size`, `cache_ttl`) keyed per process by (user, query prefix) that coalesces concurrent
  same-key requests into one classifier call, never stores a fallback, and expires unused entries.
  Backends: per-model `extra_body`, `timeout` and `max_tokens_param`, a LiteLLM
  backend (`provider_type: litellm`) for Responses-API-only models, and forwarding of standard sampling
  params and `reasoning_content`. Every response and stream chunk reports the serving backend's id
  (`llms.<name>.served_model`, default `model`) in `model`, which cost trackers price by, and a request
  naming a served id pins that backend, over HTTP and `/v1/chat/ws` alike; both forward the sampling
  params. The `litellm` floor is 1.101.0, the lowest release verified to run the `openai/responses/<model>`
  bridge with function tools and `reasoning_effort: max` on Python 3.10. The classifier reply
  parser resolves a reply naming several tiers to the recommended one. _Fork-only._
- **#1** `max_tokens` was clamped to 100 for any model missing from the built-in context table once a
  prompt passed 32k tokens; per-model `context_limit` was parsed but never used. Launcher defaults
  overrode the config's `serve` values: `python -m openclaw_router` and `python openclaw_router/server.py`
  bound every config to 0.0.0.0:8000, `scripts/start-openclaw.sh` always passed `--port 8000`, and
  `llmrouter serve` ignored an explicit `--port 0`. Each now binds `serve.host`/`serve.port` unless a
  flag is given.
  `/v1/chat/ws` dropped `tools`, `tool_choice` and messages' tool-call fields that HTTP forwards.
  _Upstreamable._
