import asyncio
import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from openclaw_router.config import LLMConfig, MediaConfig, OpenClawConfig, RouterConfig
from openclaw_router.routers import OpenClawRouter, parse_router_choice, select_by_llm
from openclaw_router.server import adjust_max_tokens, clean_response, clean_streaming_chunk, create_app

from tests.test_openclaw_http_tool_calls import RecordingAsyncClient

TIERS = ["luna-max", "glm-5.3-flash", "glm-5.3"]
OPENCODE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "openclaw_router", "opencode.yaml")


def make_llm(name, **kwargs):
    return LLMConfig(name=name, provider="mock", model_id=name, base_url="https://example.test/v1", **kwargs)


def make_config(router=None, **llm_kwargs):
    return OpenClawConfig(
        show_model_prefix=False,
        router=router or RouterConfig(strategy="random"),
        media=MediaConfig(enabled=False),
        api_keys={"mock": "test-key"},
        llms={name: make_llm(name, **llm_kwargs.get(name, {})) for name in TIERS},
    )


class RouterReplyClient:
    """httpx.AsyncClient stand-in for the classifier call."""

    reply = ""
    calls = 0
    last_json = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        type(self).calls += 1
        type(self).last_json = json
        return _Reply({"choices": [{"message": {"content": type(self).reply}}]})


class _Reply:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data


class MaxTokensTests(unittest.TestCase):
    def test_configured_context_limit_keeps_large_prompts_unclamped(self):
        # ~50k tokens: over the 32k fallback, which used to clamp max_tokens to 100.
        messages = [{"role": "user", "content": "x" * 200_000}]
        self.assertEqual(adjust_max_tokens(messages, "unknown-model", 32000), 100)
        self.assertEqual(adjust_max_tokens(messages, "unknown-model", 32000, 1_000_000), 32000)

    def test_unset_context_limit_keeps_builtin_table(self):
        messages = [{"role": "user", "content": "hi"}]
        # google/gemma-2-9b-it is 8192 in MODEL_CONTEXT_LIMITS.
        self.assertLess(adjust_max_tokens(messages, "google/gemma-2-9b-it", 32000, None), 8192)


class BackendBodyTests(unittest.TestCase):
    def setUp(self):
        RecordingAsyncClient.response_json = {
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]
        }
        RecordingAsyncClient.stream_lines = ['data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}', "data: [DONE]"]
        config = make_config(**{
            "luna-max": {
                "max_tokens": 32000,
                "context_limit": 400000,
                "extra_body": {"reasoning_effort": "max"},
                "max_tokens_param": "max_completion_tokens",
            },
        })
        self.client = TestClient(create_app(config=config))

    def _payload(self, **extra):
        payload = {"model": "luna-max", "messages": [{"role": "user", "content": "hi"}]}
        payload.update(extra)
        return payload

    def test_extra_body_and_model_max_tokens_reach_backend(self):
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = self.client.post("/v1/chat/completions", json=self._payload(top_p=0.9))
        self.assertEqual(response.status_code, 200)
        body = RecordingAsyncClient.last_post_json
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual(body["max_completion_tokens"], 32000)
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["top_p"], 0.9)
        self.assertNotIn("seed", body)

    def test_streaming_forwards_extra_body_and_client_max_tokens(self):
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = self.client.post("/v1/chat/completions", json=self._payload(stream=True, max_tokens=1234))
        self.assertEqual(response.status_code, 200)
        body = RecordingAsyncClient.last_stream_json
        self.assertEqual(body["reasoning_effort"], "max")
        self.assertEqual(body["max_completion_tokens"], 1234)

    def test_model_extra_body_overrides_client_params(self):
        config = make_config(**{"luna-max": {"extra_body": {"top_p": 1.0}}})
        client = TestClient(create_app(config=config))
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            client.post("/v1/chat/completions", json=self._payload(top_p=0.5))
        self.assertEqual(RecordingAsyncClient.last_post_json["top_p"], 1.0)


class _Dumpable:
    def __init__(self, data):
        self._data = data

    def model_dump(self, exclude_none=False):
        return self._data


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield _Dumpable(chunk)


class LiteLLMBackendTests(unittest.TestCase):
    TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]

    def setUp(self):
        config = make_config(**{"luna-max": {
            "provider_type": "litellm",
            "max_tokens": 32000,
            "context_limit": 400000,
            "extra_body": {"reasoning_effort": "max"},
            "timeout": 600,
        }})
        self.client = TestClient(create_app(config=config))
        self.calls = []

    def _fake(self, result):
        async def acompletion(**kwargs):
            self.calls.append(kwargs)
            return result
        return acompletion

    def test_streaming_tool_calls_bridge_through_litellm(self):
        chunks = [
            {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": ""}}]}}]},
            {"id": "c1", "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"path":"README.md"}'}}]}}]},
            {"id": "c1", "choices": [{"index": 0, "finish_reason": "tool_calls", "delta": {}}]},
        ]
        with patch("litellm.acompletion", self._fake(_Stream(chunks))):
            response = self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "stream": True, "tools": self.TOOLS,
                "messages": [{"role": "user", "content": "read the readme"}],
            })
        lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(lines[-1], "data: [DONE]")
        events = [json.loads(line[6:]) for line in lines[:-1]]
        self.assertEqual({event["model"] for event in events}, {"luna-max"})
        self.assertEqual(events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(events[2]["choices"][0]["finish_reason"], "tool_calls")

        kwargs = self.calls[0]
        self.assertEqual(kwargs["model"], "luna-max")
        self.assertEqual(kwargs["api_base"], "https://example.test/v1")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["reasoning_effort"], "max")
        self.assertEqual(kwargs["max_tokens"], 32000)
        self.assertEqual(kwargs["timeout"], 600)
        self.assertEqual(kwargs["tools"], self.TOOLS)
        self.assertTrue(kwargs["stream_options"]["include_usage"])

    def test_non_streaming_response_is_cleaned(self):
        result = _Dumpable({"id": "r1", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})
        with patch("litellm.acompletion", self._fake(result)):
            response = self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "ok")
        self.assertEqual(response.json()["model"], "luna-max")

    def test_streaming_error_is_reported_in_stream(self):
        async def failing(**kwargs):
            raise RuntimeError("upstream refused")
        with patch("litellm.acompletion", failing):
            response = self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("upstream refused", response.text)


class ServedModelTests(unittest.TestCase):
    """Cost trackers price by the response `model`, so it must name the backend that served."""

    def setUp(self):
        RecordingAsyncClient.response_json = {
            "model": "upstream-dated-snapshot",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        }
        RecordingAsyncClient.stream_lines = [
            'data: {"model":"upstream-dated-snapshot","choices":[{"index":0,"delta":{"content":"o"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
            "data: [DONE]",
        ]
        config = make_config(**{"glm-5.3": {"served_model": "accounts/fireworks/models/glm-5p3"}})
        self.client = TestClient(create_app(config=config))

    def _post(self, **payload):
        body = {"model": "glm-5.3", "messages": [{"role": "user", "content": "hi"}]}
        body.update(payload)
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            return self.client.post("/v1/chat/completions", json=body)

    def test_non_streaming_reports_served_model(self):
        self.assertEqual(self._post().json()["model"], "accounts/fireworks/models/glm-5p3")

    def test_every_streamed_chunk_reports_served_model(self):
        lines = [l for l in self._post(stream=True).text.splitlines() if l.startswith("data: ")]
        self.assertEqual(lines[-1], "data: [DONE]")
        models = [json.loads(l[6:]).get("model") for l in lines[:-1]]
        self.assertEqual(models, ["accounts/fireworks/models/glm-5p3"] * 2)

    def test_request_by_served_id_pins_that_backend(self):
        self._post(model="accounts/fireworks/models/glm-5p3")
        self.assertEqual(RecordingAsyncClient.last_post_json["model"], "glm-5.3")

    def test_served_id_defaults_to_model(self):
        self.assertEqual(make_llm("x").served_id, "x")
        self.assertEqual(make_llm("x", served_model="y").served_id, "y")


class ReasoningContentTests(unittest.TestCase):
    def test_reasoning_content_survives_cleaning(self):
        chunk = {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]}
        self.assertEqual(clean_streaming_chunk(chunk)["choices"][0]["delta"]["reasoning_content"], "thinking")
        result = {"choices": [{"message": {"role": "assistant", "content": "a", "reasoning_content": "r"}}]}
        self.assertEqual(clean_response(result)["choices"][0]["message"]["reasoning_content"], "r")


class ParseRouterChoiceTests(unittest.TestCase):
    def test_exact_and_decorated_names(self):
        self.assertEqual(parse_router_choice("glm-5.3", TIERS), "glm-5.3")
        self.assertEqual(parse_router_choice("**glm-5.3-flash**.", TIERS), "glm-5.3-flash")
        self.assertEqual(parse_router_choice("`luna-max`", TIERS), "luna-max")

    def test_fuzzy_prefers_longest_name(self):
        self.assertEqual(parse_router_choice("I would pick glm-5.3-flash here", TIERS), "glm-5.3-flash")

    def test_think_block_is_ignored(self):
        self.assertEqual(parse_router_choice("<think>maybe glm-5.3</think>luna-max", TIERS), "luna-max")
        self.assertIsNone(parse_router_choice("<think>glm-5.3 is best but", TIERS))

    def test_empty_or_unknown(self):
        self.assertIsNone(parse_router_choice("", TIERS))
        self.assertIsNone(parse_router_choice(None, TIERS))
        self.assertIsNone(parse_router_choice("gpt-4o", TIERS))


class SelectByLlmTests(unittest.TestCase):
    def setUp(self):
        RouterReplyClient.calls = 0
        RouterReplyClient.last_json = None

    def _router(self, **kwargs):
        defaults = dict(strategy="llm", provider="mock", base_url="https://example.test/v1", model="classifier")
        defaults.update(kwargs)
        return RouterConfig(**defaults)

    def _select(self, config, query):
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            return asyncio.run(select_by_llm(query, TIERS, config))

    def test_custom_prompt_budget_and_extra_body(self):
        RouterReplyClient.reply = "glm-5.3"
        config = make_config(router=self._router(
            prompt="Pick from {model_names}.\n{models}\nQ: {query}",
            max_tokens=512,
            extra_body={"reasoning_effort": "low"},
        ))
        self.assertEqual(self._select(config, "fix {this} race"), "glm-5.3")
        body = RouterReplyClient.last_json
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["temperature"], 0.0)
        prompt = body["messages"][0]["content"]
        self.assertIn("Pick from luna-max, glm-5.3-flash, glm-5.3.", prompt)
        self.assertIn("Q: fix {this} race", prompt)

    def test_reasoning_model_classifier_body(self):
        RouterReplyClient.reply = "luna-max"
        config = make_config(router=self._router(max_tokens=512, max_tokens_param="max_completion_tokens",
                                                 temperature=None))
        self.assertEqual(self._select(config, "rename a variable"), "luna-max")
        body = RouterReplyClient.last_json
        self.assertEqual(body["max_completion_tokens"], 512)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("temperature", body)

    def test_unparseable_reply_uses_fallback(self):
        RouterReplyClient.reply = ""
        config = make_config(router=self._router(fallback="glm-5.3-flash"))
        self.assertEqual(self._select(config, "hello"), "glm-5.3-flash")

    def test_unknown_fallback_uses_first_model(self):
        RouterReplyClient.reply = "nonsense"
        config = make_config(router=self._router(fallback="not-a-model"))
        self.assertEqual(self._select(config, "hello"), "luna-max")


class DecisionCacheTests(unittest.TestCase):
    def setUp(self):
        RouterReplyClient.calls = 0
        RouterReplyClient.reply = "glm-5.3"

    def _router(self, cache_size):
        return OpenClawRouter(make_config(router=RouterConfig(
            strategy="llm", provider="mock", base_url="https://example.test/v1", model="c", cache_size=cache_size,
        )))

    def _select_many(self, router, queries, user=None):
        async def run():
            return [await router.select_model(q, user=user) for q in queries]
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            return asyncio.run(run())

    def test_tool_loop_classified_once(self):
        router = self._router(cache_size=8)
        self.assertEqual(self._select_many(router, ["same turn"] * 5), ["glm-5.3"] * 5)
        self.assertEqual(RouterReplyClient.calls, 1)

    def test_cache_is_bounded_and_per_user(self):
        router = self._router(cache_size=2)
        self._select_many(router, ["a", "b", "c", "a"])
        self.assertEqual(RouterReplyClient.calls, 4)  # "a" was evicted
        self._select_many(router, ["a"], user="other")
        self.assertEqual(RouterReplyClient.calls, 5)

    def test_disabled_by_default(self):
        router = self._router(cache_size=0)
        self._select_many(router, ["same"] * 3)
        self.assertEqual(RouterReplyClient.calls, 3)


class OpencodeConfigTests(unittest.TestCase):
    def test_opencode_yaml_parses_with_tier_order(self):
        with patch.dict(os.environ, {"FIREWORKS_API_KEY": "fw", "AZURE_OPENAI_API_KEY": "az"}):
            config = OpenClawConfig.from_yaml(OPENCODE_CONFIG)
        self.assertEqual(list(config.llms), TIERS)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertFalse(config.show_model_prefix)
        self.assertEqual(config.router.strategy, "llm")
        self.assertEqual(config.router.fallback, "glm-5.3-flash")
        self.assertGreater(config.router.cache_size, 0)
        self.assertEqual(config.router.model, "gpt-6-luna")
        self.assertEqual(config.router.extra_body, {"reasoning_effort": "low"})
        self.assertEqual(config.router.max_tokens_param, "max_completion_tokens")
        self.assertIsNone(config.router.temperature)
        self.assertEqual(config.get_api_key(config.router.provider), "az")
        for placeholder in ("{models}", "{model_names}", "{memory}", "{query}"):
            self.assertIn(placeholder, config.router.prompt)
        luna = config.llms["luna-max"]
        self.assertEqual(luna.extra_body, {"reasoning_effort": "max"})
        self.assertEqual(luna.provider_type, "litellm")
        self.assertEqual(luna.model_id, "openai/responses/gpt-6-luna")
        self.assertEqual(config.llms["glm-5.3"].provider_type, "openai_compatible")
        # Served ids must match what opencode stores in responseModelIDs and firstmate prices.
        self.assertEqual(
            [llm.served_id for llm in config.llms.values()],
            ["gpt-6-luna", "accounts/fireworks/models/glm-5p3-flash", "accounts/fireworks/models/glm-5p3"],
        )
        self.assertEqual(config.get_api_key(luna.provider, luna), "az")
        self.assertEqual(config.llms["glm-5.3"].context_limit, 1000000)


if __name__ == "__main__":
    unittest.main()
