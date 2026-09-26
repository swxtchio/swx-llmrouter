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

- **opencode-routing** (WIP) Route opencode requests across luna-max, glm-5.3-flash and glm-5.3 by complexity.
