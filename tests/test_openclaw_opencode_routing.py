import asyncio
import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi.testclient import TestClient

from openclaw_router.config import LLMConfig, MediaConfig, OpenClawConfig, RouterConfig
from openclaw_router.routers import OpenClawRouter, parse_router_choice, select_by_llm
from openclaw_router.server import adjust_max_tokens, clean_response, clean_streaming_chunk, create_app

from tests.test_openclaw_http_tool_calls import RecordingAsyncClient

# ~50k estimated tokens: over the 32768 default a model missing from MODEL_CONTEXT_LIMITS gets,
# where adjust_max_tokens clamps max_tokens to 100 unless the model's context_limit is used.
LARGE_PROMPT = "x" * 200_000

TIERS = ["luna-max", "glm-5.3-flash", "sol-high"]
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
    gate = None  # asyncio.Event that post() waits on when set
    calls = 0
    last_json = None
    last_timeout = None

    @classmethod
    def reset(cls, reply=""):
        cls.reply, cls.status_code, cls.error, cls.gate = reply, 200, None, None
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
        if cls.gate is not None:
            await cls.gate.wait()
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

    def test_every_standard_sampling_param_is_forwarded(self):
        params = {
            "top_p": 0.9, "stop": ["\n\n"], "seed": 7, "response_format": {"type": "json_object"},
            "parallel_tool_calls": False, "presence_penalty": 0.5, "frequency_penalty": -0.5,
        }
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            self.client.post("/v1/chat/completions", json=self._payload(**params))
        body = RecordingAsyncClient.last_post_json
        self.assertEqual({name: body.get(name) for name in params}, params)

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
        config = make_config(**{"sol-high": {"served_model": "gpt-6-sol"}})
        self.client = TestClient(create_app(config=config))

    def _post(self, **payload):
        body = {"model": "sol-high", "messages": [{"role": "user", "content": "hi"}]}
        body.update(payload)
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            return self.client.post("/v1/chat/completions", json=body)

    def test_non_streaming_reports_served_model(self):
        self.assertEqual(self._post().json()["model"], "gpt-6-sol")

    def test_every_streamed_chunk_reports_served_model(self):
        lines = [l for l in self._post(stream=True).text.splitlines() if l.startswith("data: ")]
        self.assertEqual(lines[-1], "data: [DONE]")
        models = [json.loads(l[6:]).get("model") for l in lines[:-1]]
        self.assertEqual(models, ["gpt-6-sol"] * 2)

    def test_request_by_served_id_pins_that_backend(self):
        # A router that would pick another tier, so only pinning can reach sol-high.
        with patch("openclaw_router.server.OpenClawRouter.select_model",
                   AsyncMock(return_value="luna-max")) as select_model:
            self._post(model="gpt-6-sol")
        select_model.assert_not_awaited()
        self.assertEqual(RecordingAsyncClient.last_post_json["model"], "sol-high")

    def test_websocket_pins_served_id_and_forwards_request_fields(self):
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
        tool_call = {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        payload = {
            "model": "gpt-6-sol",
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
        self.assertEqual(body["model"], "sol-high")
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["top_p"], 0.9)
        self.assertEqual(body["messages"][1]["tool_calls"], [tool_call])
        self.assertEqual(body["messages"][2]["tool_call_id"], "call_1")

    def test_relayed_stream_chunks_report_served_model(self):
        # The HTTP stream and the WebSocket relay rebuild chunks when the [model] prefix is on.
        served = "gpt-6-sol"
        for prefix in (False, True):
            config = make_config(**{"sol-high": {"served_model": served}})
            config.show_model_prefix = prefix
            client = TestClient(create_app(config=config))
            payload = {"model": "sol-high", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
            with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
                http_lines = client.post("/v1/chat/completions", json=payload).text.splitlines()
                with client.websocket_connect("/v1/chat/ws") as websocket:
                    websocket.send_json(payload)
                    ws_lines = []
                    while True:
                        message = websocket.receive_text()
                        if "[DONE]" in message:
                            break
                        ws_lines.append(message if message.startswith("data: ") else "data: " + message)
            for entry, lines in (("http", http_lines), ("ws", ws_lines)):
                with self.subTest(entry=entry, prefix=prefix):
                    chunks = [json.loads(line[6:]) for line in lines
                              if line.startswith("data: ") and "[DONE]" not in line]
                    self.assertEqual(len(chunks), 2)
                    self.assertEqual([chunk.get("model") for chunk in chunks], [served] * 2)

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
        self.assertEqual(parse_router_choice("sol-high", TIERS), "sol-high")
        self.assertEqual(parse_router_choice("**glm-5.3-flash**.", TIERS), "glm-5.3-flash")
        self.assertEqual(parse_router_choice("`luna-max`", TIERS), "luna-max")

    def test_fuzzy_prefers_longest_name(self):
        self.assertEqual(parse_router_choice("I would pick glm-5.3-flash here", TIERS), "glm-5.3-flash")

    def test_several_tiers_resolve_to_the_recommended_one(self):
        self.assertEqual(parse_router_choice("Not luna-max; sol-high", TIERS), "sol-high")
        self.assertEqual(parse_router_choice("use sol-high here, not glm-5.3-flash", TIERS), "sol-high")
        self.assertEqual(parse_router_choice("I'd say sol-high.", TIERS), "sol-high")
        self.assertIsNone(parse_router_choice("not luna-max", TIERS))

    def test_think_block_is_ignored(self):
        self.assertEqual(parse_router_choice("<think>maybe sol-high</think>luna-max", TIERS), "luna-max")
        self.assertIsNone(parse_router_choice("<think>sol-high is best but", TIERS))

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
        RouterReplyClient.reply = "sol-high"
        config = make_config(router=self._router(
            prompt="Pick from {model_names}.\n{models}\nQ: {query}",
            max_tokens=512,
            extra_body={"reasoning_effort": "low"},
        ))
        self.assertEqual(self._select(config, "fix {this} race"), "sol-high")
        body = RouterReplyClient.last_json
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertEqual(body["temperature"], 0.0)
        prompt = body["messages"][0]["content"]
        self.assertIn("Pick from luna-max, glm-5.3-flash, sol-high.", prompt)
        self.assertIn("\n- luna-max\n- glm-5.3-flash\n- sol-high\n", prompt)
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
        RouterReplyClient.reply = "sol-high"
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
        RouterReplyClient.reply = "sol-high"
        config = make_config(router=self._router(fallback="glm-5.3-flash"))
        config.api_keys = {}
        self.assertEqual(self._select(config, "hello"), "glm-5.3-flash")
        self.assertEqual(RouterReplyClient.calls, 0)

    def test_classifier_call_uses_router_timeout(self):
        RouterReplyClient.reply = "sol-high"
        config = make_config(router=self._router(timeout=7.5))
        self._select(config, "hello")
        self.assertEqual(RouterReplyClient.last_timeout, 7.5)

    def test_substituted_text_is_not_rescanned(self):
        RouterReplyClient.reply = "sol-high"
        config = make_config(router=self._router(prompt="{memory}\nQ: {query}"))
        memory = [{"query": "explain {query} and {models}", "model": "luna-max"}]
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            asyncio.run(select_by_llm("NEW QUERY", TIERS, config, memory_items=memory))
        prompt = RouterReplyClient.last_json["messages"][0]["content"]
        self.assertIn("explain {query} and {models}", prompt)
        self.assertEqual(prompt.count("NEW QUERY"), 1)


class DecisionCacheTests(unittest.TestCase):
    def setUp(self):
        RouterReplyClient.reset(reply="sol-high")

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
        self.assertEqual(self._select_many(router, ["same turn"] * 5), ["sol-high"] * 5)
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
        self.assertEqual(selected, ["sol-high"] * 20)
        self.assertEqual(RouterReplyClient.calls, 1)

    def test_cancelled_first_caller_does_not_fail_the_others(self):
        router = self._router(cache_size=8)

        async def run():
            RouterReplyClient.gate = asyncio.Event()
            first = asyncio.ensure_future(router.select_model("same turn"))
            second = asyncio.ensure_future(router.select_model("same turn"))
            for _ in range(100):  # until the classifier call is in flight
                if RouterReplyClient.calls:
                    break
                await asyncio.sleep(0)
            first.cancel()
            RouterReplyClient.gate.set()
            return await second, first.cancelled()
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            self.assertEqual(asyncio.run(run()), ("sol-high", True))
        self.assertEqual(RouterReplyClient.calls, 1)

    def test_fallback_decision_is_not_reused(self):
        # Every route_by_llm fallback exit: set the failure, then restore a healthy classifier.
        def error_status():
            RouterReplyClient.status_code = 503

        def unparseable():
            RouterReplyClient.reply = "I cannot decide"

        def raises():
            RouterReplyClient.error = RuntimeError("connection reset")

        def times_out():
            RouterReplyClient.error = httpx.ReadTimeout("classifier timed out")

        exits = {"non-200": error_status, "unparseable reply": unparseable, "exception": raises,
                 "timeout": times_out, "missing API key": None}
        for name, fail in exits.items():
            with self.subTest(name):
                RouterReplyClient.reset(reply="sol-high")
                router = self._router(cache_size=8)
                if fail is None:
                    router.config.api_keys = {}
                else:
                    fail()
                self.assertEqual(self._select_many(router, ["turn"]), ["glm-5.3-flash"])
                calls_while_failing = RouterReplyClient.calls
                RouterReplyClient.reset(reply="sol-high")
                router.config.api_keys = {"mock": "test-key"}
                self.assertEqual(self._select_many(router, ["turn", "turn"]), ["sol-high"] * 2)
                # Classified again after the fallback, then that real decision is cached.
                self.assertEqual(RouterReplyClient.calls, 1, calls_while_failing)

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


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERVE_YAML = 'serve:\n  host: "127.0.0.9"\n  port: 8123\nllms: {}\n'


class LauncherBindTests(unittest.TestCase):
    """Every documented launcher binds the config's serve.host/port unless a flag is given."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = os.path.join(self.tmp.name, "serve.yaml")
        with open(self.config, "w") as handle:
            handle.write(SERVE_YAML)

    def _bind_in_process(self, entry, argv):
        with patch("sys.argv", argv), patch("uvicorn.run") as run, patch("sys.stdout", io.StringIO()):
            entry()
        run.assert_called_once()
        return run.call_args.kwargs["host"], run.call_args.kwargs["port"]

    def _python_m(self, *flags):
        from openclaw_router.__main__ import main
        return self._bind_in_process(main, ["openclaw_router", "--config", self.config, *flags])

    def _llmrouter_serve(self, *flags):
        from llmrouter.cli.router_main import main
        return self._bind_in_process(main, ["llmrouter", "serve", "--config", self.config, *flags])

    def _server_script(self, *flags):
        # `python openclaw_router/server.py` in a child, with uvicorn stubbed to report its bind.
        stub = os.path.join(self.tmp.name, "stub")
        os.makedirs(stub, exist_ok=True)
        with open(os.path.join(stub, "uvicorn.py"), "w") as handle:
            handle.write("import json\ndef run(app, host, port, **kwargs):\n    print('BIND ' + json.dumps([host, port]))\n")
        env = dict(os.environ, PYTHONPATH=stub)
        result = subprocess.run(
            [sys.executable, os.path.join(REPO, "openclaw_router", "server.py"), "--config", self.config, *flags],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
        )
        binds = [line[5:] for line in result.stdout.splitlines() if line.startswith("BIND ")]
        self.assertEqual(len(binds), 1, result.stdout + result.stderr)
        return tuple(json.loads(binds[0]))

    def test_config_serve_values_bind_without_flags(self):
        for launch in (self._python_m, self._llmrouter_serve, self._server_script):
            with self.subTest(launch.__name__):
                self.assertEqual(launch(), ("127.0.0.9", 8123))

    def test_explicit_flags_override_config(self):
        for launch in (self._python_m, self._llmrouter_serve, self._server_script):
            with self.subTest(launch.__name__):
                self.assertEqual(launch("--host", "0.0.0.0", "--port", "0"), ("0.0.0.0", 0))

    def _start_script(self, host, config_port, *flags, gateway=False):
        """Run scripts/start-openclaw.sh with its external commands stubbed.

        `python -m` records the router launch instead of starting one; curl records its
        arguments and then really probes, so a probe succeeds only against a live listener.
        Returns the recorded launch and probe lines, the script's output and the config path.
        """
        config = os.path.join(self.tmp.name, "start.yaml")
        with open(config, "w") as handle:
            handle.write(SERVE_YAML.replace("127.0.0.9", host).replace("8123", str(config_port)))
        stub = os.path.join(self.tmp.name, "bin")
        os.makedirs(stub, exist_ok=True)
        calls = os.path.join(self.tmp.name, "calls")
        gateway_pid = os.path.join(self.tmp.name, "gateway.pid")
        for leftover in (calls, gateway_pid):
            if os.path.exists(leftover):
                os.unlink(leftover)
        real_sleep = shutil.which("sleep")
        stubs = {
            "python": f'if [ "$1" = "-m" ]; then echo "python $*" >> "{calls}"; exit 0; fi\nexec "{sys.executable}" "$@"',
            "curl": f'echo "curl $*" >> "{calls}"\nexec "{shutil.which("curl")}" --max-time 5 "$@"',
            "lsof": "exit 1",
            "pkill": "exit 0",
            "tail": "exit 0",
            # The gateway stub fails at once; `sleep 3`, the script's wait before checking it,
            # returns once that process is gone (bounded), so the failure branch is taken.
            "openclaw": f'echo $$ > "{gateway_pid}"\nexit 1',
            "sleep": (f'[ "$1" = 3 ] || exit 0\nfor _ in $(seq 500); do\n'
                      f'  [ -s "{gateway_pid}" ] && ! kill -0 "$(cat "{gateway_pid}")" 2>/dev/null && exit 0\n'
                      f'  "{real_sleep}" 0.01\ndone'),
        }
        if not gateway:
            del stubs["openclaw"]
            flags = ("--no-gateway",) + flags
        for name, body in stubs.items():
            path = os.path.join(stub, name)
            with open(path, "w") as handle:
                handle.write("#!/bin/bash\n" + body + "\n")
            os.chmod(path, 0o755)
        env = dict(os.environ, PATH=stub + os.pathsep + os.environ["PATH"],
                   ROUTER_LOG=os.path.join(self.tmp.name, "router.log"),
                   GATEWAY_LOG=os.path.join(self.tmp.name, "gateway.log"))
        result = subprocess.run(
            ["bash", os.path.join(REPO, "scripts", "start-openclaw.sh"), "-c", config, *flags],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # The router is launched with `nohup ... &` and never waited for, so its record can land
        # after the script exits: poll for it, bounded, instead of reading once.
        deadline = time.monotonic() + 10
        while True:
            lines = open(calls).read().splitlines() if os.path.exists(calls) else []
            launch = [line for line in lines if line.startswith("python -m openclaw_router")]
            if launch or time.monotonic() > deadline:
                break
            time.sleep(0.02)
        probes = [line for line in lines if line.startswith("curl ")]
        return launch, probes, result.stdout, config

    def test_start_script_leaves_the_bind_to_the_config(self):
        with http_listener("127.0.0.9") as port:
            launch, probes, output, config = self._start_script("127.0.0.9", port)
        self.assertEqual(launch, [f"python -m openclaw_router --config {config}"])
        self.assertEqual(probes, [f"curl -s http://127.0.0.9:{port}/health"])
        self.assertIn(f"API: http://127.0.0.9:{port}/v1/chat/completions", output)
        self.assertIn(f"OpenClaw Router: http://127.0.0.9:{port}\n", output)

    def test_start_script_port_flag_overrides_config(self):
        # Nothing listens on the config's port 1; the router binds the -p port.
        with http_listener("127.0.0.9") as port:
            launch, probes, output, config = self._start_script("127.0.0.9", 1, "-p", str(port))
        self.assertEqual(launch, [f"python -m openclaw_router --config {config} --port {port}"])
        self.assertEqual(probes, [f"curl -s http://127.0.0.9:{port}/health"])
        self.assertIn(f"OpenClaw Router: http://127.0.0.9:{port}\n", output)

    def test_start_script_probes_a_wildcard_bind_on_loopback_of_its_family(self):
        # (serve.host, address the router's listener binds, host the probe must use)
        cases = [("0.0.0.0", "0.0.0.0", "127.0.0.1"), ("::", "::", "[::1]"), ("[::]", "::", "[::1]")]
        for host, listen, probe_host in cases:
            with self.subTest(host), http_listener(listen) as port:
                _, probes, output, _ = self._start_script(host, port)
                self.assertEqual(probes, [f"curl -s http://{probe_host}:{port}/health"])
                self.assertIn(f"OpenClaw Router: http://{probe_host}:{port}\n", output)

    def test_start_script_brackets_an_ipv6_literal(self):
        for host in ("::1", "[::1]"):
            with self.subTest(host), http_listener("::1") as port:
                _, probes, output, _ = self._start_script(host, port)
                self.assertEqual(probes, [f"curl -s http://[::1]:{port}/health"])
                self.assertIn(f"API: http://[::1]:{port}/v1/chat/completions", output)

    def test_start_script_gateway_hint_names_the_router_bind(self):
        with http_listener("127.0.0.9") as port:
            _, _, output, _ = self._start_script("127.0.0.9", port, gateway=True)
        self.assertIn("OpenClaw Gateway failed to start", output)
        self.assertIn(f"models.providers.openclaw.baseUrl (http://127.0.0.9:{port}/v1)", output)


@contextlib.contextmanager
def http_listener(host):
    """Answer HTTP 200 on host:<ephemeral port>, listening the way uvicorn does.

    uvicorn.run binds with loop.create_server(host=...), which makes an IPv6 socket
    IPv6-only, so a "::" listener refuses 127.0.0.1 exactly as the router would.
    """

    class Reply(asyncio.Protocol):
        def connection_made(self, transport):
            self.transport = transport

        def data_received(self, data):
            self.transport.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            self.transport.close()

    loop = asyncio.new_event_loop()
    ready = threading.Event()
    state = {}

    def serve():
        asyncio.set_event_loop(loop)
        try:
            server = loop.run_until_complete(loop.create_server(Reply, host=host, port=0))
            state["port"] = server.sockets[0].getsockname()[1]
        except OSError as error:
            state["error"] = error
            ready.set()
            return
        ready.set()
        loop.run_forever()
        server.close()
        loop.run_until_complete(server.wait_closed())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    if not ready.wait(10):
        raise RuntimeError(f"listener on {host} did not start")
    if "error" in state:
        raise state["error"]
    try:
        yield state["port"]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()


class YamlFieldsTests(unittest.TestCase):
    """Each router/llm field this change added is read from YAML (every value here is non-default)."""

    YAML = """
router:
  strategy: llm
  prompt: "P {query}"
  max_tokens: 777
  max_tokens_param: max_completion_tokens
  temperature: null
  extra_body: {reasoning_effort: low}
  timeout: 9.5
  cache_size: 3
  cache_ttl: 42
  fallback: b
llms:
  a:
    model: route/a
    served_model: served-a
    provider_type: litellm
    max_tokens: 1234
    context_limit: 5555
    extra_body: {reasoning_effort: max}
    timeout: 33
    max_tokens_param: max_completion_tokens
  b:
    model: b
"""

    def test_new_fields_are_parsed(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(self.YAML)
        self.addCleanup(os.unlink, handle.name)
        config = OpenClawConfig.from_yaml(handle.name)
        router = config.router
        self.assertEqual(
            (router.prompt, router.max_tokens, router.max_tokens_param, router.temperature, router.extra_body,
             router.timeout, router.cache_size, router.cache_ttl, router.fallback),
            ("P {query}", 777, "max_completion_tokens", None, {"reasoning_effort": "low"}, 9.5, 3, 42.0, "b"),
        )
        a, b = config.llms["a"], config.llms["b"]
        self.assertEqual(
            (a.model_id, a.served_id, a.provider_type, a.max_tokens, a.context_limit, a.extra_body, a.timeout,
             a.max_tokens_param),
            ("route/a", "served-a", "litellm", 1234, 5555, {"reasoning_effort": "max"}, 33.0, "max_completion_tokens"),
        )
        # Unset fields: context_limit falls back to the built-in table, served id to the model.
        self.assertEqual((b.context_limit, b.served_id), (None, "b"))


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
        self.assertEqual(config.router.cache_ttl, 1800)
        self.assertEqual((config.router.max_tokens, config.router.timeout), (1024, 20))
        self.assertEqual([llm.timeout for llm in config.llms.values()], [600, 300, 600])
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
        sol = config.llms["sol-high"]
        self.assertEqual(sol.extra_body, {"reasoning_effort": "high"})
        self.assertEqual((sol.provider_type, sol.served_id), ("litellm", "gpt-6-sol"))
        self.assertEqual(luna.model_id, "openai/responses/gpt-6-luna")
        # Served ids must match what opencode stores in responseModelIDs and firstmate prices.
        self.assertEqual(
            [llm.served_id for llm in config.llms.values()],
            ["gpt-6-luna", "accounts/fireworks/models/glm-5p3-flash", "gpt-6-sol"],
        )
        self.assertEqual(config.get_api_key(luna.provider, luna), "az")
        self.assertEqual([llm.context_limit for llm in config.llms.values()], [1000000] * 3)


if __name__ == "__main__":
    unittest.main()
