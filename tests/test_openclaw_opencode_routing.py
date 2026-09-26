import asyncio
import importlib.util
import io
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from openclaw_router.config import LLMConfig, MediaConfig, OpenClawConfig, RouterConfig
from openclaw_router.routers import OpenClawRouter, parse_router_choice, select_by_llm
from openclaw_router.server import adjust_max_tokens, clean_response, clean_streaming_chunk, create_app

from tests.test_openclaw_http_tool_calls import RecordingAsyncClient

# ~50k estimated tokens: over the 32768 default a model missing from MODEL_CONTEXT_LIMITS gets,
# where adjust_max_tokens clamps max_tokens to 100 unless the model's context_limit is used.
LARGE_PROMPT = "x" * 200_000

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
    status_code = 200
    error = None  # raised by post() when set
    calls = 0
    last_json = None
    last_timeout = None

    @classmethod
    def reset(cls, reply=""):
        cls.reply, cls.status_code, cls.error = reply, 200, None
        cls.calls, cls.last_json, cls.last_timeout = 0, None, None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None, timeout=None):
        cls = type(self)
        cls.calls += 1
        cls.last_json = json
        cls.last_timeout = timeout
        # Suspend like a real network call, so concurrent callers interleave here.
        await asyncio.sleep(0)
        if cls.error is not None:
            raise cls.error
        return _Reply({"choices": [{"message": {"content": cls.reply}}]}, cls.status_code)


class _Reply:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

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
        RecordingAsyncClient.reset_capture()
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
                "timeout": 42.0,
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

    def test_large_prompt_keeps_max_tokens_under_configured_context_limit(self):
        messages = [{"role": "user", "content": LARGE_PROMPT}]
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            self.client.post("/v1/chat/completions", json=self._payload(messages=messages))
            self.client.post("/v1/chat/completions", json=self._payload(messages=messages, stream=True))
        self.assertEqual(RecordingAsyncClient.last_post_json["max_completion_tokens"], 32000)
        self.assertEqual(RecordingAsyncClient.last_stream_json["max_completion_tokens"], 32000)

    def test_streaming_forwards_passthrough_params(self):
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            self.client.post("/v1/chat/completions", json=self._payload(stream=True, top_p=0.9, seed=7))
        body = RecordingAsyncClient.last_stream_json
        self.assertEqual((body["top_p"], body["seed"]), (0.9, 7))

    def test_backend_calls_use_model_timeout(self):
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            self.client.post("/v1/chat/completions", json=self._payload())
            self.client.post("/v1/chat/completions", json=self._payload(stream=True))
        self.assertEqual(RecordingAsyncClient.last_post_timeout, 42.0)
        self.assertEqual(RecordingAsyncClient.last_stream_timeout, 42.0)

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
            "served_model": "gpt-6-luna",
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
        self.assertEqual([event["model"] for event in events], ["gpt-6-luna"] * 3)
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
        self.assertEqual(response.json()["model"], "gpt-6-luna")

    def test_large_prompt_and_passthrough_params_reach_litellm(self):
        result = _Dumpable({"id": "r1", "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]})
        with patch("litellm.acompletion", self._fake(result)):
            self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "top_p": 0.9, "seed": 7,
                "messages": [{"role": "user", "content": LARGE_PROMPT}]})
        kwargs = self.calls[0]
        self.assertEqual(kwargs["max_tokens"], 32000)
        self.assertEqual((kwargs["top_p"], kwargs["seed"]), (0.9, 7))

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
        RecordingAsyncClient.reset_capture()
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
        # A router that would pick another tier, so only pinning can reach glm-5.3.
        with patch("openclaw_router.server.OpenClawRouter.select_model",
                   AsyncMock(return_value="luna-max")) as select_model:
            self._post(model="accounts/fireworks/models/glm-5p3")
        select_model.assert_not_awaited()
        self.assertEqual(RecordingAsyncClient.last_post_json["model"], "glm-5.3")

    def test_websocket_pins_served_id_and_forwards_request_fields(self):
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
        tool_call = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        payload = {
            "model": "accounts/fireworks/models/glm-5p3",
            "messages": [
                {"role": "user", "content": "read it"},
                {"role": "assistant", "content": "", "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file text"},
            ],
            "tools": tools,
            "tool_choice": "auto",
            "top_p": 0.9,
        }
        with patch("openclaw_router.server.OpenClawRouter.select_model",
                   AsyncMock(return_value="luna-max")) as select_model, \
                patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            with self.client.websocket_connect("/v1/chat/ws") as websocket:
                websocket.send_json(payload)
                while "[DONE]" not in websocket.receive_text():
                    pass
        select_model.assert_not_awaited()
        body = RecordingAsyncClient.last_stream_json
        self.assertEqual(body["model"], "glm-5.3")
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["top_p"], 0.9)
        self.assertEqual(body["messages"][1]["tool_calls"], [tool_call])
        self.assertEqual(body["messages"][2]["tool_call_id"], "call_1")

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

    def test_several_tiers_resolve_to_the_recommended_one(self):
        self.assertEqual(parse_router_choice("Not luna-max; glm-5.3", TIERS), "glm-5.3")
        self.assertEqual(parse_router_choice("use glm-5.3 here, not glm-5.3-flash", TIERS), "glm-5.3")
        self.assertEqual(parse_router_choice("I'd say glm-5.3.", TIERS), "glm-5.3")
        self.assertIsNone(parse_router_choice("not luna-max", TIERS))

    def test_think_block_is_ignored(self):
        self.assertEqual(parse_router_choice("<think>maybe glm-5.3</think>luna-max", TIERS), "luna-max")
        self.assertIsNone(parse_router_choice("<think>glm-5.3 is best but", TIERS))

    def test_empty_or_unknown(self):
        self.assertIsNone(parse_router_choice("", TIERS))
        self.assertIsNone(parse_router_choice(None, TIERS))
        self.assertIsNone(parse_router_choice("gpt-4o", TIERS))


class SelectByLlmTests(unittest.TestCase):
    def setUp(self):
        RouterReplyClient.reset()

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

    # The configured fallback, not models[0] (luna-max at max effort), covers a classifier outage.
    def test_classifier_error_status_uses_fallback(self):
        RouterReplyClient.reply = "glm-5.3"
        RouterReplyClient.status_code = 429
        config = make_config(router=self._router(fallback="glm-5.3-flash"))
        self.assertEqual(self._select(config, "hello"), "glm-5.3-flash")
        self.assertEqual(RouterReplyClient.calls, 1)

    def test_classifier_exception_uses_fallback(self):
        RouterReplyClient.error = RuntimeError("read timeout")
        config = make_config(router=self._router(fallback="glm-5.3-flash"))
        self.assertEqual(self._select(config, "hello"), "glm-5.3-flash")
        self.assertEqual(RouterReplyClient.calls, 1)

    def test_missing_api_key_uses_fallback(self):
        RouterReplyClient.reply = "glm-5.3"
        config = make_config(router=self._router(fallback="glm-5.3-flash"))
        config.api_keys = {}
        self.assertEqual(self._select(config, "hello"), "glm-5.3-flash")
        self.assertEqual(RouterReplyClient.calls, 0)

    def test_classifier_call_uses_router_timeout(self):
        RouterReplyClient.reply = "glm-5.3"
        config = make_config(router=self._router(timeout=7.5))
        self._select(config, "hello")
        self.assertEqual(RouterReplyClient.last_timeout, 7.5)

    def test_substituted_text_is_not_rescanned(self):
        RouterReplyClient.reply = "glm-5.3"
        config = make_config(router=self._router(prompt="{memory}\nQ: {query}"))
        memory = [{"query": "explain {query} and {models}", "model": "luna-max"}]
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            asyncio.run(select_by_llm("NEW QUERY", TIERS, config, memory_items=memory))
        prompt = RouterReplyClient.last_json["messages"][0]["content"]
        self.assertIn("explain {query} and {models}", prompt)
        self.assertEqual(prompt.count("NEW QUERY"), 1)


class DecisionCacheTests(unittest.TestCase):
    def setUp(self):
        RouterReplyClient.reset(reply="glm-5.3")

    def _router(self, cache_size, **kwargs):
        return OpenClawRouter(make_config(router=RouterConfig(
            strategy="llm", provider="mock", base_url="https://example.test/v1", model="c", cache_size=cache_size,
            fallback="glm-5.3-flash", **kwargs,
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

    def test_concurrent_same_key_requests_share_one_classifier_call(self):
        router = self._router(cache_size=8)

        async def run():
            return await asyncio.gather(*[router.select_model("same turn") for _ in range(20)])
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            selected = asyncio.run(run())
        self.assertEqual(selected, ["glm-5.3"] * 20)
        self.assertEqual(RouterReplyClient.calls, 1)

    def test_fallback_decision_is_not_reused(self):
        router = self._router(cache_size=8)
        RouterReplyClient.status_code = 503
        self.assertEqual(self._select_many(router, ["turn"]), ["glm-5.3-flash"])
        RouterReplyClient.status_code = 200
        self.assertEqual(self._select_many(router, ["turn"]), ["glm-5.3"])
        self.assertEqual(RouterReplyClient.calls, 2)

    def test_unused_entry_expires(self):
        router = self._router(cache_size=8, cache_ttl=60)
        clock = [1000.0]
        with patch("openclaw_router.routers.time.monotonic", lambda: clock[0]):
            self._select_many(router, ["turn"])
            clock[0] += 59
            self._select_many(router, ["turn"])  # within the TTL: reused, and refreshed
            self.assertEqual(RouterReplyClient.calls, 1)
            clock[0] += 59
            self._select_many(router, ["turn"])  # 59s since last use: still live
            self.assertEqual(RouterReplyClient.calls, 1)
            clock[0] += 61
            self._select_many(router, ["turn"])
        self.assertEqual(RouterReplyClient.calls, 2)


class EvalScriptTests(unittest.TestCase):
    def _script(self):
        path = os.path.join(os.path.dirname(__file__), "..", "scripts", "eval_opencode_classifier.py")
        spec = importlib.util.spec_from_file_location("eval_opencode_classifier", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_reps_below_one_is_rejected(self):
        script = self._script()
        for reps in ("0", "-1"):
            with self.assertRaises(SystemExit) as raised, \
                    patch("sys.stderr", io.StringIO()) as stderr:
                script.parse_args(["--reps", reps])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("--reps: must be at least 1", stderr.getvalue())
        self.assertEqual(script.parse_args(["--reps", "1"]).reps, 1)


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
        self.assertEqual(config.router.model, "accounts/fireworks/models/gpt-oss-120b")
        self.assertEqual(config.router.extra_body, {"reasoning_effort": "low"})
        self.assertEqual(config.get_api_key(config.router.provider), "fw")
        # The classifier is never a routing target.
        self.assertNotIn(config.router.model, [llm.served_id for llm in config.llms.values()])
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
        self.assertEqual([llm.context_limit for llm in config.llms.values()], [1000000] * 3)


if __name__ == "__main__":
    unittest.main()
