"""Score classifier candidates for openclaw_router/opencode.yaml on labeled coding requests.

Each candidate replaces only the config's `router` connection settings, so every candidate sees
the same prompt, the same parser and the same code path the server uses (select_by_llm).

    python scripts/eval_opencode_classifier.py --reps 2
    python scripts/eval_opencode_classifier.py --only oss120b-low luna-low

Needs the API keys the candidates use (FIREWORKS_API_KEY, AZURE_OPENAI_API_KEY). The labels are
hand-assigned from the tier definitions in the prompt; they are a smoke benchmark, not ground truth.
"""

import argparse
import asyncio
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from openclaw_router import routers  # noqa: E402
from openclaw_router.config import OpenClawConfig  # noqa: E402

CONFIG = os.path.join(REPO, "openclaw_router", "opencode.yaml")
LOW, MID, HIGH = "luna-max", "glm-5.3-flash", "sol-high"

CASES = [
    (LOW, "What does the --no-prefix flag do?"),
    (LOW, "Rename the variable `cfg` to `config` in server.py"),
    (LOW, "Write a commit message for the staged changes"),
    (LOW, "Where is adjust_max_tokens defined?"),
    (LOW, "Summarize what this PR changes"),
    (LOW, "Fix the typo 'recieve' in the README"),
    (LOW, "Generate a title for this conversation"),
    (LOW, "Explain what the decision cache in routers.py does"),
    (MID, "Add a --timeout flag to the CLI that sets the per-request timeout, and update its tests"),
    (MID, "Write unit tests for parse_router_choice covering edge cases"),
    (MID, "The test_ws test fails with 'async def functions are not natively supported'; fix it"),
    (MID, "Add a /v1/models endpoint that lists the configured backends with their descriptions"),
    (MID, "Implement retry with exponential backoff for 429 responses in the backend client"),
    (MID, "Add YAML schema validation for the llms section and report clear errors"),
    (MID, "Port the start-openclaw.sh script options to a Python click CLI"),
    (MID, "Add a Prometheus /metrics endpoint counting requests per routed model"),
    (HIGH, "Our WebSocket reconnect path has an intermittent race that duplicates in-flight tool calls "
           "under load; redesign the session state machine to make it lock-free and provably correct"),
    (HIGH, "Design a multi-tenant architecture for the router with per-tenant budgets, isolation and "
           "quota enforcement"),
    (HIGH, "Memory grows unbounded after about 6 hours in production but only under streaming load; "
           "find the leak"),
    (HIGH, "Refactor the whole backend layer into a provider plugin system without breaking any "
           "existing configs"),
    (HIGH, "Audit the auth and key-handling paths for security vulnerabilities, including key leakage "
           "through logs and errors"),
    (HIGH, "p99 latency doubled after the httpx upgrade; profile it and fix the regression"),
    (HIGH, "Streaming responses occasionally interleave chunks from two concurrent requests; find the "
           "root cause"),
    (HIGH, "Design a migration that moves routing memory from JSONL to Postgres with zero downtime"),
]

FIREWORKS = {"provider": "fireworks", "base_url": "https://api.fireworks.ai/inference/v1",
             "max_tokens_param": "max_tokens", "temperature": 0.0}
AZURE = {"provider": "azure", "base_url": "https://swxcodexmodels.openai.azure.com/openai/v1",
         "max_tokens_param": "max_completion_tokens", "temperature": None}

CANDIDATES = {
    "oss120b-low": dict(FIREWORKS, model="accounts/fireworks/models/gpt-oss-120b", max_tokens=1024,
                        extra_body={"reasoning_effort": "low"}),
    "oss120b-medium": dict(FIREWORKS, model="accounts/fireworks/models/gpt-oss-120b", max_tokens=2048,
                           extra_body={"reasoning_effort": "medium"}),
    "luna-low": dict(AZURE, model="gpt-6-luna", max_tokens=512, extra_body={"reasoning_effort": "low"}),
}


def load(overrides):
    config = OpenClawConfig.from_yaml(CONFIG)
    config.router.cache_size = 0
    for key, value in overrides.items():
        setattr(config.router, key, value)
    return config


async def evaluate(name, config, reps):
    models = list(config.llms)
    unparsed = 0
    parse = routers.parse_router_choice

    def counting_parse(content, names):
        nonlocal unparsed
        choice = parse(content, names)
        unparsed += choice is None
        return choice

    routers.parse_router_choice = counting_parse
    try:
        rows, latencies = [], []
        for _ in range(reps):
            for label, query in CASES:
                start = time.perf_counter()
                got = await routers.select_by_llm(query, models, config)
                latencies.append(time.perf_counter() - start)
                rows.append((label, got, query))
    finally:
        routers.parse_router_choice = parse

    n = len(rows)
    correct = sum(label == got for label, got, _ in rows)
    under = sum(models.index(got) < models.index(label) for label, got, _ in rows)
    over = sum(models.index(got) > models.index(label) for label, got, _ in rows)
    latencies.sort()
    print(f"\n== {name}: accuracy {correct}/{n} = {correct / n:.0%}  under-routed {under}  "
          f"over-routed {over}  unparsed (fallback) {unparsed}")
    print(f"   latency median {statistics.median(latencies):.2f}s  "
          f"p90 {latencies[int(0.9 * (n - 1))]:.2f}s  max {latencies[-1]:.2f}s")
    for label, got, query in sorted({row for row in rows if row[0] != row[1]}):
        print(f"   want {label:14} got {got:14} {query[:70]}")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=positive_int, default=2, help="passes over the cases (>= 1)")
    parser.add_argument("--only", nargs="*", choices=sorted(CANDIDATES), help="candidates to run")
    return parser.parse_args(argv)


async def main():
    args = parse_args()
    for name in args.only or CANDIDATES:
        await evaluate(name, load(CANDIDATES[name]), args.reps)


if __name__ == "__main__":
    asyncio.run(main())
