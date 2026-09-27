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
import textwrap
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import tiktoken
from fastapi import HTTPException
from fastapi.testclient import TestClient

from openclaw_router.config import (
    LLMConfig,
    MediaConfig,
    MemoryConfig,
    OpenClawConfig,
    RouterConfig,
)
from openclaw_router import routers as router_module
from openclaw_router.memory import MemoryBank
from openclaw_router.routers import OpenClawRouter, parse_router_choice, select_by_llm
from openclaw_router.server import (
    _input_token_encoding,
    adjust_max_tokens,
    clean_response,
    clean_streaming_chunk,
    create_app,
    estimate_tokens,
)

from tests.test_openclaw_http_tool_calls import RecordingAsyncClient

_HTTPX_ASYNC_CLIENT = httpx.AsyncClient

# A large readable body exercises the model context fallback without a pathological character run.
LARGE_PROMPT = "word " * 60_000

TIERS = ["luna-max", "glm-5.3-flash", "sol-high"]
DEFAULT_USAGE_LOG_PATH = "~/.local/state/openclaw-router/classifier-usage.jsonl"
OPENCODE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "openclaw_router", "opencode.yaml")
SYSTEM_MESSAGE_CASES = (
    ("MAIN-OPENCODE-PROMPT", "AXI-AMBIENT-CONTEXT"),
    ("MAIN-OPENCODE-PROMPT", "AXI-AMBIENT-CONTEXT", "MAIN-OPENCODE-PROMPT"),
    ("SYSTEM-INSTRUCTION-A", "SYSTEM-INSTRUCTION-B", "SYSTEM-INSTRUCTION-C"),
)


def make_llm(name, **kwargs):
    return LLMConfig(name=name, provider="mock", model_id=name, base_url="https://example.test/v1", **kwargs)


def make_config(router=None, **llm_kwargs):
    config = OpenClawConfig(
        show_model_prefix=False,
        router=router or RouterConfig(strategy="random"),
        media=MediaConfig(enabled=False),
        api_keys={"mock": "test-key"},
        llms={name: make_llm(name, **llm_kwargs.get(name, {})) for name in TIERS},
    )
    config.router.classifier_usage_log_path = os.devnull
    return config


def configure_machine_routing(config):
    """Load the production machine-route policy into an isolated test router."""
    with patch.dict(os.environ, {"FIREWORKS_API_KEY": "test-fireworks", "AZURE_OPENAI_API_KEY": "test-azure"}):
        machine_config = OpenClawConfig.from_yaml(OPENCODE_CONFIG)
    config.router.machine_model = machine_config.router.machine_model
    config.router.machine_patterns = machine_config.router.machine_patterns
    return config


# Each captured input is the OpenCode database's exact user-message text after the server's [:500] routing slice.
with open(os.path.join(os.path.dirname(__file__), "fixtures", "openclaw_q2_routing_windows.json"), encoding="utf-8") as fixture:
    Q2_MACHINE_ROUTING_EXAMPLES = tuple(
        (item["name"], item["routing_text"], item["marker"]) for item in json.load(fixture)
    )
Q2_T08_HEARTBEAT = next(query for name, query, _ in Q2_MACHINE_ROUTING_EXAMPLES if name == "fleet heartbeat T08")
Q2_T46_HEARTBEAT = next(query for name, query, _ in Q2_MACHINE_ROUTING_EXAMPLES if name == "fleet heartbeat T46")
with open(os.path.join(os.path.dirname(__file__), "fixtures", "firstmate_heartbeat_generator_windows.json"), encoding="utf-8") as fixture:
    HEARTBEAT_GENERATOR_WINDOWS = tuple(json.load(fixture))
with open(os.path.join(os.path.dirname(__file__), "fixtures", "firstmate_dedicated_send_generator_windows.json"), encoding="utf-8") as fixture:
    DEDICATED_GENERATOR_WINDOWS = tuple(json.load(fixture))
with open(os.path.join(os.path.dirname(__file__), "fixtures", "firstmate_turnend_guard_prefixes.json"), encoding="utf-8") as fixture:
    TURNEND_GUARD_WINDOWS = {item["name"]: item["routing_text"] for item in json.load(fixture)}


class RouterReplyClient:
    """httpx.AsyncClient stand-in for the classifier call."""

    reply = ""
    status_code = 200
    error = None  # raised by post() when set
    gate = None  # asyncio.Event that post() waits on when set
    usage = None
    calls = 0
    last_json = None
    last_timeout = None

    @classmethod
    def reset(cls, reply="", usage=None):
        cls.reply, cls.status_code, cls.error, cls.gate, cls.usage = reply, 200, None, None, usage
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
        data = {"choices": [{"message": {"content": cls.reply}}]}
        if cls.usage is not None:
            data["usage"] = cls.usage
        return _Reply(data, cls.status_code)


class PendingRouterReplyClient(RouterReplyClient):
    gates = []

    async def post(self, url, headers=None, json=None, timeout=None):
        cls = type(self)
        gate_index = cls.calls
        cls.calls += 1
        cls.last_json = json
        cls.last_timeout = timeout
        await cls.gates[gate_index].wait()
        if cls.error is not None:
            raise cls.error
        data = {"choices": [{"message": {"content": cls.reply}}]}
        if cls.usage is not None:
            data["usage"] = cls.usage
        return _Reply(data, cls.status_code)


class _Reply:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


class MaxTokensTests(unittest.TestCase):
    def test_configured_context_limit_keeps_large_prompts_unclamped(self):
        messages = [{"role": "user", "content": LARGE_PROMPT}]
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
        self.config = config
        self.client = TestClient(create_app(config=config))

    def _payload(self, **extra):
        payload = {"model": "luna-max", "messages": [{"role": "user", "content": "hi"}]}
        payload.update(extra)
        return payload

    def _post_with_upstream_content(self, payload, status_code, content, content_type, client=None):
        def response_for_request(request):
            return httpx.Response(status_code, content=content, headers={"content-type": content_type},
                                  request=request)

        def async_client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(response_for_request)
            return _HTTPX_ASYNC_CLIENT(*args, **kwargs)

        with patch("openclaw_router.server.httpx.AsyncClient", side_effect=async_client):
            return (client or self.client).post("/v1/chat/completions", json=payload)

    def _post_with_upstream_error(self, payload, status_code, body):
        return self._post_with_upstream_content(
            payload, status_code, json.dumps(body).encode(), "application/json")

    def _post_with_upstream_stream(self, payload, lines, client=None):
        content = ("\n\n".join(lines) + "\n\n").encode()
        return self._post_with_upstream_content(
            payload, 200, content, "text/event-stream", client=client)

    def _stream_overflow(self, prefix):
        return {
            "message": f"{prefix}: Your input exceeds the context window.",
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "input",
        }

    def _assert_non_streaming_http_error(self, status_code, upstream_error):
        response = self._post_with_upstream_error(
            self._payload(), status_code, {"error": upstream_error})

        self.assertEqual(response.status_code, status_code)
        self.assertEqual(response.json(), {"error": upstream_error})

    def test_non_streaming_http_client_error_keeps_complete_message(self):
        self._assert_non_streaming_http_error(400, {
            "message": "DIRECT-HTTP-OVERFLOW: Your input exceeds the context window. " + "x" * 2400,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "input",
        })

    def test_non_streaming_http_server_error_keeps_complete_message(self):
        self._assert_non_streaming_http_error(503, {
            "message": "DIRECT-HTTP-UNAVAILABLE: upstream is unavailable. " + "y" * 2400,
            "type": "server_error",
            "code": "upstream_unavailable",
            "param": None,
        })

    def test_streaming_http_overflow_before_first_chunk_is_an_http_error(self):
        upstream_error = {
            "message": "DIRECT-STREAM-OVERFLOW: Your input exceeds the context window. " + "z" * 2400,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "input",
        }
        response = self._post_with_upstream_error(
            self._payload(stream=True), 400, {"error": upstream_error})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": upstream_error})

    def _assert_streaming_http_overflow_after_prelude(self, prelude, prefix):
        upstream_error = self._stream_overflow(prefix)
        response = self._post_with_upstream_stream(self._payload(stream=True), [
            f"data: {json.dumps(prelude)}",
            f"data: {json.dumps({'error': upstream_error})}",
        ])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": upstream_error})

    def test_streaming_http_200_error_event_before_any_prelude_is_an_http_error(self):
        upstream_error = self._stream_overflow("DIRECT-SSE-FIRST")
        response = self._post_with_upstream_stream(self._payload(stream=True), [
            f"data: {json.dumps({'error': upstream_error})}",
        ])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": upstream_error})

    def test_streaming_http_role_prelude_then_overflow_is_an_http_error(self):
        self._assert_streaming_http_overflow_after_prelude({
            "id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        }, "DIRECT-SSE-ROLE")

    def test_streaming_http_empty_choices_prelude_then_overflow_is_an_http_error(self):
        self._assert_streaming_http_overflow_after_prelude({"id": "c1", "choices": []}, "DIRECT-SSE-EMPTY")

    def test_streaming_http_usage_prelude_then_overflow_is_an_http_error(self):
        self._assert_streaming_http_overflow_after_prelude({
            "id": "c1", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4},
        }, "DIRECT-SSE-USAGE")

    def test_streaming_http_usage_prelude_with_prefix_then_overflow_is_an_http_error(self):
        config = make_config()
        config.show_model_prefix = True
        client = TestClient(create_app(config=config))
        upstream_error = self._stream_overflow("DIRECT-SSE-PREFIX-USAGE")
        response = self._post_with_upstream_stream(self._payload(stream=True), [
            'data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
            'data: {"id":"c1","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":0,"total_tokens":4}}',
            f"data: {json.dumps({'error': upstream_error})}",
        ], client=client)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": upstream_error})

    def test_streaming_http_flushes_prelude_when_content_arrives(self):
        prelude = {"id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
        content = {"id": "c1", "choices": [{"index": 0, "delta": {"content": "ready"}}]}
        response = self._post_with_upstream_stream(self._payload(stream=True), [
            f"data: {json.dumps(prelude)}",
            f"data: {json.dumps(content)}",
            "data: [DONE]",
        ])

        data_lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
        events = [json.loads(line[6:]) for line in data_lines if line[6:] != "[DONE]"]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(events[1]["choices"][0]["delta"]["content"], "ready")
        self.assertEqual(data_lines[-1], "data: [DONE]")

    def test_streaming_http_failure_after_content_is_an_openai_error_event(self):
        message = "DIRECT-SSE-AFTER-CONTENT: upstream failed"
        content = {"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]}
        response = self._post_with_upstream_stream(self._payload(stream=True), [
            f"data: {json.dumps(content)}",
            f"data: {json.dumps({'error': message})}",
        ])

        events = [line for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(events[0][6:])["choices"][0]["delta"]["content"], "partial")
        self.assertEqual(json.loads(events[1][6:]), {"error": {
            "message": message,
            "type": "api_error",
            "code": None,
            "param": None,
        }})

    def test_streaming_http_prefix_buffered_content_then_overflow_is_an_http_error(self):
        config = make_config()
        config.show_model_prefix = True
        client = TestClient(create_app(config=config))
        upstream_error = self._stream_overflow("DIRECT-SSE-PREFIX")
        response = self._post_with_upstream_stream(self._payload(stream=True), [
            'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"["}}]}',
            f"data: {json.dumps({'error': upstream_error})}",
        ], client=client)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": upstream_error})

    def test_streaming_http_non_utf8_error_body_keeps_upstream_status(self):
        body = b"upstream rejected request: \xff"
        response = self._post_with_upstream_content(
            self._payload(stream=True), 503, body, "application/json")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"error": {
            "message": body.decode("utf-8", errors="replace"),
            "type": "api_error",
            "code": None,
            "param": None,
        }})

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

    def test_tool_schemas_reduce_output_budget_for_sync_and_streaming(self):
        self.config.llms["luna-max"].context_limit = 2000
        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Read the requested reference and return the relevant details. " * 120,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }]

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            responses = [
                self.client.post("/v1/chat/completions", json=self._payload(tools=tools, max_tokens=1500)),
                self.client.post(
                    "/v1/chat/completions",
                    json=self._payload(tools=tools, max_tokens=1500, stream=True),
                ),
            ]

        self.assertEqual([response.status_code for response in responses], [200, 200])
        for body in (RecordingAsyncClient.last_post_json, RecordingAsyncClient.last_stream_json):
            self.assertEqual(body["tools"], tools)
            self.assertLessEqual(body["max_completion_tokens"], 900)

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

    def test_tool_schemas_reduce_compatible_backend_output_budget(self):
        self.config.llms["luna-max"].context_limit = 2000
        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Reference detail safely. " * 200,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }]
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            sync_response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(tools=tools, max_tokens=1500),
            )
            stream_response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(tools=tools, max_tokens=1500, stream=True),
            )

        self.assertEqual((sync_response.status_code, stream_response.status_code), (200, 200))
        for stream, body in (
            (False, RecordingAsyncClient.last_post_json),
            (True, RecordingAsyncClient.last_stream_json),
        ):
            with self.subTest(stream=stream):
                self.assertEqual(body["tools"], tools)
                self.assertLessEqual(body["max_completion_tokens"], 1300)

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

    def test_system_messages_reach_openai_compatible_backend_in_order(self):
        for stream in (False, True):
            for system_messages in SYSTEM_MESSAGE_CASES:
                with self.subTest(stream=stream, system_messages=system_messages):
                    messages = ([{"role": "system", "content": content} for content in system_messages]
                                + [{"role": "user", "content": "hi"}])
                    with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
                        response = self.client.post(
                            "/v1/chat/completions", json=self._payload(messages=messages, stream=stream))
                    self.assertEqual(response.status_code, 200)
                    body = (RecordingAsyncClient.last_stream_json if stream
                            else RecordingAsyncClient.last_post_json)
                    self.assertEqual(
                        body["messages"],
                        [
                            {"role": "system", "content": "\n\n".join(system_messages)},
                            {"role": "user", "content": "hi"},
                        ],
                    )

    def test_empty_system_message_does_not_emit_system_backend_message(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ]
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = self.client.post("/v1/chat/completions", json=self._payload(messages=messages))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(RecordingAsyncClient.last_post_json["messages"], [
            {"role": "user", "content": "hi"},
        ])

    def test_empty_system_message_does_not_wrap_model_without_system_role_user_body(self):
        config = make_config()
        config.llms["sol-high"].model_id = "meta/llama-3.1-8b-instruct"
        client = TestClient(create_app(config=config))
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "hi"},
        ]
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = client.post("/v1/chat/completions", json={"model": "sol-high", "messages": messages})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(RecordingAsyncClient.last_post_json["messages"], [
            {"role": "user", "content": "hi"},
        ])

    def test_empty_system_message_between_instructions_adds_no_blank_run(self):
        messages = [
            {"role": "system", "content": "SYSTEM-INSTRUCTION-A"},
            {"role": "system", "content": ""},
            {"role": "system", "content": "SYSTEM-INSTRUCTION-B"},
            {"role": "user", "content": "hi"},
        ]
        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = self.client.post("/v1/chat/completions", json=self._payload(messages=messages))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(RecordingAsyncClient.last_post_json["messages"], [
            {"role": "system", "content": "SYSTEM-INSTRUCTION-A\n\nSYSTEM-INSTRUCTION-B"},
            {"role": "user", "content": "hi"},
        ])

    def test_system_messages_for_model_without_system_role_reach_user_body(self):
        config = make_config()
        config.llms["sol-high"].model_id = "meta/llama-3.1-8b-instruct"
        client = TestClient(create_app(config=config))
        for system_messages in SYSTEM_MESSAGE_CASES:
            with self.subTest(system_messages=system_messages):
                messages = ([{"role": "system", "content": content} for content in system_messages]
                            + [{"role": "user", "content": "hi"}])
                with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
                    response = client.post("/v1/chat/completions", json={
                        "model": "sol-high", "messages": messages,
                    })
                self.assertEqual(response.status_code, 200)
                body = RecordingAsyncClient.last_post_json
                self.assertEqual(len(body["messages"]), 1)
                self.assertEqual(body["messages"][0]["role"], "user")
                self.assertEqual(
                    body["messages"][0]["content"],
                    "[System Instructions]\n" + "\n\n".join(system_messages)
                    + "\n\n[User Message]\nhi",
                )

    def test_websocket_forwards_two_and_three_system_messages_to_backend_body(self):
        for system_messages in SYSTEM_MESSAGE_CASES:
            with self.subTest(system_messages=system_messages):
                messages = ([{"role": "system", "content": content} for content in system_messages]
                            + [{"role": "user", "content": "hi"}])
                payload = {"model": "luna-max", "messages": messages}
                with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
                    with self.client.websocket_connect("/v1/chat/ws") as websocket:
                        websocket.send_json(payload)
                        while "[DONE]" not in websocket.receive_text():
                            pass
                body = RecordingAsyncClient.last_stream_json
                self.assertEqual(
                    body["messages"],
                    [
                        {"role": "system", "content": "\n\n".join(system_messages)},
                        {"role": "user", "content": "hi"},
                    ],
                )


class ContextLimitHTTPTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config(**{
            "luna-max": {"served_model": "gpt-6-luna", "context_limit": 40},
            "sol-high": {"served_model": "gpt-6-sol", "context_limit": 80},
        })
        self.client = TestClient(create_app(config=self.config))

    def _payload(self, model="gpt-6-luna", messages=None, **extra):
        payload = {
            "model": model,
            "messages": messages or [{
                "role": "user",
                "content": "This request contains many distinct words and phrases. " * 20,
            }],
        }
        payload.update(extra)
        return payload

    def _assert_error_payload(self, body, limit):
        error = body["error"]
        self.assertEqual(set(error), {"message", "type", "code", "param"})
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertEqual(error["code"], "context_length_exceeded")
        self.assertEqual(error["param"], "messages")
        self.assertIn(str(limit), error["message"])

    def _assert_context_error(self, response, limit=40):
        self.assertEqual(response.status_code, 400, response.text)
        self._assert_error_payload(response.json(), limit)

    def _tool_schema(self, repetitions=50):
        return [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find reference details safely. " * repetitions,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }]

    def test_oversized_served_id_returns_openai_error_without_backend_call(self):
        with patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call:
            backend_call.return_value = {"choices": []}
            response = self.client.post("/v1/chat/completions", json=self._payload())

        backend_call.assert_not_awaited()
        self._assert_context_error(response)

    def test_oversized_auto_stream_is_rejected_before_sse_or_backend_call(self):
        async def completed_stream():
            yield "data: [DONE]\n\n"

        with (
            patch("openclaw_router.server.OpenClawRouter.select_model", new_callable=AsyncMock,
                  return_value="sol-high") as select_model,
            patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call,
        ):
            backend_call.return_value = completed_stream()
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(
                    model="auto",
                    messages=[{
                        "role": "user",
                        "content": "This request contains many distinct words and phrases. " * 40,
                    }],
                    stream=True,
                ),
            )

        select_model.assert_awaited_once()
        backend_call.assert_not_awaited()
        self._assert_context_error(response, limit=80)
        self.assertEqual(response.headers["content-type"], "application/json")

    def test_under_limit_served_id_reaches_selected_backend(self):
        with patch(
            "openclaw_router.server.LLMBackend.call",
            new_callable=AsyncMock,
            return_value={"id": "ok", "choices": []},
        ) as backend_call:
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(messages=[{"role": "user", "content": "short"}]),
            )

        self.assertEqual(response.status_code, 200, response.text)
        backend_call.assert_awaited_once()
        self.assertIn("luna-max", backend_call.await_args.args)

    def test_tool_call_arguments_count_toward_context_limit(self):
        self.config.llms["luna-max"].context_limit = 50
        messages = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": "x" * 160},
            }]},
        ]
        with patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call, \
                redirect_stdout(io.StringIO()):
            backend_call.return_value = {"choices": []}
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(messages=messages),
            )

        backend_call.assert_not_awaited()
        self._assert_context_error(response, limit=50)

    def test_tokenizer_failure_warns_once_for_byte_count_fallback(self):
        _input_token_encoding.cache_clear()
        sample = "café 🚀"
        try:
            with patch(
                "openclaw_router.server.tiktoken.get_encoding",
                side_effect=OSError("encoding fetch failed"),
            ) as get_encoding, warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                expected = len(sample.encode("utf-8", errors="surrogatepass"))
                self.assertEqual(estimate_tokens(sample), expected)
                self.assertEqual(estimate_tokens(sample), expected)

            get_encoding.assert_called_once_with("o200k_base")
            self.assertEqual(len(captured), 1)
            self.assertIn("UTF-8 byte-count fallback", str(captured[0].message))
        finally:
            _input_token_encoding.cache_clear()

    def test_tool_result_content_alone_pushes_input_over_limit(self):
        self.config.llms["luna-max"].context_limit = 100
        messages = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "y" * 400},
        ]
        with patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call, \
                redirect_stdout(io.StringIO()):
            backend_call.return_value = {"choices": []}
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(messages=messages),
            )

        backend_call.assert_not_awaited()
        self._assert_context_error(response, limit=100)

    def test_request_tool_schemas_count_toward_context_limit(self):
        with patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call:
            backend_call.return_value = {"choices": []}
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(
                    messages=[{"role": "user", "content": "ok"}],
                    tools=self._tool_schema(50),
                ),
            )

        backend_call.assert_not_awaited()
        self._assert_context_error(response)

    def test_websocket_preflights_tool_schemas_before_backend_call(self):
        async def completed_stream():
            yield json.dumps({"choices": []})

        with patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call:
            backend_call.return_value = completed_stream()
            with self.client.websocket_connect("/v1/chat/ws") as websocket:
                websocket.send_json(self._payload(
                    messages=[{"role": "user", "content": "ok"}],
                    tools=self._tool_schema(50),
                ))
                body = websocket.receive_json()

        backend_call.assert_not_awaited()
        self._assert_error_payload(body, limit=40)

    def test_unset_tier_limit_uses_the_output_budget_fallback(self):
        llm = self.config.llms["glm-5.3-flash"]
        llm.context_limit = None
        llm.model_id = "unlisted-round-one-model"
        self.assertIsNone(llm.context_limit)
        with patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call:
            backend_call.return_value = {"choices": []}
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(
                    model="glm-5.3-flash",
                    messages=[{"role": "user", "content": "word " * 40000}],
                ),
            )

        backend_call.assert_not_awaited()
        self._assert_context_error(response, limit=32768)

    def test_code_estimate_is_calibrated_and_keeps_safe_side_headroom(self):
        source = Path(__file__).resolve().parents[1] / "openclaw_router" / "server.py"
        code = source.read_text(encoding="utf-8")
        messages = [{"role": "user", "content": code}]
        serialized = json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":"))
        encoding = tiktoken.get_encoding("o200k_base")
        tokenizer_count = len(encoding.encode(serialized, disallowed_special=()))

        self.config.llms["luna-max"].context_limit = (tokenizer_count * 11 + 9) // 10
        with patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call, \
                redirect_stdout(io.StringIO()):
            backend_call.return_value = {"choices": []}
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(messages=messages),
            )

        self.assertEqual(response.status_code, 200, response.text)
        backend_call.assert_awaited_once()

        safety_messages = [{"role": "user", "content": "short"}]
        safety_input = json.dumps({"messages": safety_messages}, ensure_ascii=False, separators=(",", ":"))
        safety_token_count = len(encoding.encode(safety_input, disallowed_special=()))
        self.config.llms["luna-max"].context_limit = safety_token_count
        with patch("openclaw_router.server.LLMBackend.call", new_callable=AsyncMock) as backend_call, \
                redirect_stdout(io.StringIO()):
            backend_call.return_value = {"choices": []}
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(messages=safety_messages),
            )

        backend_call.assert_not_awaited()
        self._assert_context_error(response, limit=safety_token_count)

    def test_output_budget_accounts_for_tool_payloads(self):
        self.config.llms["luna-max"].context_limit = 1800
        self.config.llms["luna-max"].max_tokens = 4096
        RecordingAsyncClient.reset_capture()
        RecordingAsyncClient.response_json = {"choices": []}
        messages = [
            {"role": "user", "content": "x" * 40},
            {"role": "assistant", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "lookup", "arguments": "y" * 2000},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "z" * 200},
        ]

        with patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient):
            response = self.client.post(
                "/v1/chat/completions",
                json=self._payload(messages=messages, max_tokens=1500),
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertLessEqual(RecordingAsyncClient.last_post_json["max_tokens"], 1050)


class _ResponsesRequestHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, request))
        if request.get("stream"):
            body = b"data: [DONE]\n\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        else:
            body = json.dumps({
                "id": "resp_local",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-6-luna",
                "output": [{
                    "id": "msg_local", "type": "message", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class LiteLLMResponsesWireTests(unittest.TestCase):
    def setUp(self):
        _ResponsesRequestHandler.requests = []
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _ResponsesRequestHandler)
        self.thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.thread.start()
        config = make_config()
        config.llms["luna-max"].provider_type = "litellm"
        config.llms["luna-max"].model_id = "openai/responses/gpt-6-luna"
        config.llms["luna-max"].served_model = "gpt-6-luna"
        config.llms["luna-max"].base_url = f"http://127.0.0.1:{self.upstream.server_port}/v1"
        config.llms["luna-max"].max_tokens = 32
        config.llms["luna-max"].context_limit = 400000
        self.client = TestClient(create_app(config=config))

    def tearDown(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        self.thread.join(3)

    def test_litellm_responses_wire_keeps_all_system_messages_for_sync_and_stream(self):
        for stream in (False, True):
            for system_messages in SYSTEM_MESSAGE_CASES:
                with self.subTest(stream=stream, system_messages=system_messages):
                    messages = ([{"role": "system", "content": content} for content in system_messages]
                                + [{"role": "user", "content": "hi"}])
                    response = self.client.post("/v1/chat/completions", json={
                        "model": "luna-max", "stream": stream, "messages": messages,
                    })
                    self.assertEqual(response.status_code, 200, response.text)
                    path, body = _ResponsesRequestHandler.requests[-1]
                    self.assertEqual(path, "/v1/responses")
                    instruction = body.get("instructions", "")
                    self.assertEqual(instruction, "\n\n".join(system_messages))


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


class _FailingStream:
    def __init__(self, chunks, error):
        self._chunks = chunks
        self._error = error

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for chunk in self._chunks:
            yield _Dumpable(chunk)
        raise self._error


def _litellm_api_error(message, status_code, body=None):
    from litellm.exceptions import APIError

    error = APIError(
        status_code=status_code,
        message=message,
        llm_provider="azure",
        model="gpt-6-luna",
    )
    error.body = body
    return error


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
        self.config = config
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

    def test_non_streaming_litellm_failure_returns_complete_openai_error(self):
        upstream_error = {
            "message": "LITELLM-SYNC-OVERFLOW: Your input exceeds the context window. " + "s" * 2400,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "input",
        }

        async def failing(**kwargs):
            raise _litellm_api_error(upstream_error["message"], 400, {"error": upstream_error})

        with patch("litellm.acompletion", failing):
            response = self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "messages": [{"role": "user", "content": "hi"}]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": upstream_error})

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

    def test_tool_schemas_reduce_litellm_output_budget_for_sync_and_streaming(self):
        self.config.llms["luna-max"].context_limit = 2000
        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Reference detail safely. " * 200,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }]
        for stream in (False, True):
            with self.subTest(stream=stream):
                result = _Stream([]) if stream else _Dumpable({"id": "r1", "choices": []})
                with patch("litellm.acompletion", self._fake(result)):
                    response = self.client.post("/v1/chat/completions", json={
                        "model": "gpt-6-luna",
                        "stream": stream,
                        "messages": [{"role": "user", "content": "short"}],
                        "tools": tools,
                        "max_tokens": 1500,
                    })

                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(self.calls[-1]["tools"], tools)
                self.assertLessEqual(self.calls[-1]["max_tokens"], 1300)

    def test_streaming_overflow_during_iteration_before_first_chunk_is_http_error(self):
        message = "LITELLM-STREAM-OVERFLOW: Your input exceeds the context window. " + "t" * 2400
        error = _litellm_api_error(message, 400)
        expected_message = str(error)

        async def failing(**kwargs):
            return _FailingStream([], error)

        with patch("litellm.acompletion", failing):
            response = self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "stream": True, "messages": [{"role": "user", "content": "hi"}]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": {
            "message": expected_message,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": None,
        }})

    def test_streaming_litellm_role_prelude_then_overflow_is_http_error(self):
        message = "LITELLM-STREAM-PRELUDE: Your input exceeds the context window."
        error = _litellm_api_error(message, 400)
        expected_message = str(error)

        async def acompletion(**kwargs):
            return _FailingStream([{
                "id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            }], error)

        with patch("litellm.acompletion", acompletion):
            response = self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "stream": True, "messages": [{"role": "user", "content": "hi"}]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"error": {
            "message": expected_message,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": None,
        }})

    def test_streaming_failure_after_success_chunk_emits_openai_error_event(self):
        upstream_error = {
            "message": "LITELLM-STREAM-AFTER-CHUNK: upstream failed. " + "u" * 2400,
            "type": "server_error",
            "code": "upstream_unavailable",
            "param": None,
        }

        async def acompletion(**kwargs):
            return _FailingStream([{
                "id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}],
            }], _litellm_api_error(upstream_error["message"], 503, {"error": upstream_error}))

        with patch("litellm.acompletion", acompletion):
            response = self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "stream": True, "messages": [{"role": "user", "content": "hi"}]})

        lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
        events = [json.loads(line[6:]) for line in lines]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "partial")
        self.assertEqual(events[1], {"error": upstream_error})

    def test_streaming_empty_backend_error_message_uses_exception_name(self):
        async def acompletion(**kwargs):
            return _FailingStream([{
                "id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}],
            }], RuntimeError(""))

        with patch("litellm.acompletion", acompletion):
            response = self.client.post("/v1/chat/completions", json={
                "model": "luna-max", "stream": True, "messages": [{"role": "user", "content": "hi"}]})

        lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
        events = [json.loads(line[6:]) for line in lines]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "partial")
        self.assertEqual(events[1], {"error": {
            "message": "RuntimeError",
            "type": "api_error",
            "code": None,
            "param": None,
        }})


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

    def _machine_client(self):
        config = make_config(router=RouterConfig(
            strategy="llm", provider="mock", base_url="https://example.test/v1", model="classifier",
            cache_size=8, fallback="sol-high",
        ))
        configure_machine_routing(config)
        config.llms["luna-max"].served_model = "gpt-6-luna"
        return TestClient(create_app(config=config))

    def test_non_streaming_reports_served_model(self):
        self.assertEqual(self._post().json()["model"], "gpt-6-sol")

    def test_every_streamed_chunk_reports_served_model(self):
        lines = [l for l in self._post(stream=True).text.splitlines() if l.startswith("data: ")]
        self.assertEqual(lines[-1], "data: [DONE]")
        models = [json.loads(l[6:]).get("model") for l in lines[:-1]]
        self.assertEqual(models, ["gpt-6-sol"] * 2)

    def test_machine_route_stamps_http_response_and_every_stream_chunk(self):
        client = self._machine_client()
        request = {"model": "auto", "messages": [{"role": "user", "content": Q2_T08_HEARTBEAT}]}
        output = io.StringIO()
        with patch("openclaw_router.routers.route_by_llm", AsyncMock(return_value=("sol-high", False))) as classify, \
                patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient), \
                redirect_stdout(output):
            response = client.post("/v1/chat/completions", json=request)
            stream = client.post("/v1/chat/completions", json={**request, "stream": True})

        classify.assert_not_awaited()
        self.assertEqual(response.json()["model"], "gpt-6-luna")
        self.assertEqual(RecordingAsyncClient.last_post_json["model"], "luna-max")
        lines = [line for line in stream.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(lines[-1], "data: [DONE]")
        self.assertEqual([json.loads(line[6:]).get("model") for line in lines[:-1]], ["gpt-6-luna"] * 2)
        self.assertIn("[Router] Machine -> luna-max (marker=Fleet heartbeat. Run one supervision cycle)", output.getvalue())
        self.assertNotIn("Strategy=llm", output.getvalue())

    def test_machine_route_reaches_websocket_backend_without_classifier(self):
        client = self._machine_client()
        request = {"model": "auto", "messages": [{"role": "user", "content": Q2_T46_HEARTBEAT}]}
        output = io.StringIO()
        chunks = []
        with patch("openclaw_router.routers.route_by_llm", AsyncMock(return_value=("sol-high", False))) as classify, \
                patch("openclaw_router.server.httpx.AsyncClient", RecordingAsyncClient), \
                redirect_stdout(output):
            with client.websocket_connect("/v1/chat/ws") as websocket:
                websocket.send_json(request)
                while True:
                    line = websocket.receive_text()
                    if "[DONE]" in line:
                        break
                    if line.startswith("data: "):
                        chunks.append(json.loads(line[6:]))
                    else:
                        chunks.append(json.loads(line))

        classify.assert_not_awaited()
        self.assertEqual(RecordingAsyncClient.last_stream_json["model"], "luna-max")
        self.assertEqual([chunk.get("model") for chunk in chunks], ["gpt-6-luna"] * 2)
        self.assertIn("[Router] Machine -> luna-max (marker=Fleet heartbeat. Run one supervision cycle)", output.getvalue())
        self.assertNotIn("Strategy=llm", output.getvalue())

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

    def _auto_route_client(self, **llm_options):
        config = make_config(router=RouterConfig(
            strategy="llm", provider="mock", base_url="https://example.test/v1", model="c",
            cache_size=8, fallback="glm-5.3-flash",
        ), **llm_options)
        return TestClient(create_app(config=config))

    def _auto_route_request(self, content="same turn", stream=False):
        return {
            "model": "auto",
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
            "user": "decision-cache-user",
        }

    def _post_auto_route(self, client, payload, backend_call):
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient), \
                patch("openclaw_router.server.LLMBackend.call", new=backend_call):
            return client.post("/v1/chat/completions", json=payload)

    def test_routed_client_error_reclassifies_the_same_cache_key(self):
        for status in (400, 422):
            with self.subTest(status=status):
                RouterReplyClient.reset(reply="sol-high")
                client = self._auto_route_client()
                attempted_models = []

                async def backend_call(_backend, model, *args, **kwargs):
                    attempted_models.append(model)
                    if len(attempted_models) == 1:
                        raise HTTPException(status_code=status, detail="routed client rejection")
                    return {"choices": []}

                request = self._auto_route_request()
                failed = self._post_auto_route(client, request, backend_call)
                self.assertEqual(failed.status_code, status)

                RouterReplyClient.reply = "luna-max"
                retried = self._post_auto_route(client, request, backend_call)

                self.assertEqual(retried.status_code, 200)
                self.assertEqual(attempted_models, ["sol-high", "luna-max"])
                self.assertEqual(RouterReplyClient.calls, 2)

    def test_routed_429_and_server_errors_keep_the_cached_decision(self):
        for status in (429, 503):
            with self.subTest(status=status):
                RouterReplyClient.reset(reply="sol-high")
                client = self._auto_route_client()
                attempted_models = []

                async def backend_call(_backend, model, *args, **kwargs):
                    attempted_models.append(model)
                    if len(attempted_models) == 1:
                        raise HTTPException(status_code=status, detail="routed retryable failure")
                    return {"choices": []}

                request = self._auto_route_request()
                failed = self._post_auto_route(client, request, backend_call)
                self.assertEqual(failed.status_code, status)

                RouterReplyClient.reply = "luna-max"
                retried = self._post_auto_route(client, request, backend_call)

                self.assertEqual(retried.status_code, 200)
                self.assertEqual(attempted_models, ["sol-high", "sol-high"])
                self.assertEqual(RouterReplyClient.calls, 1)

    def test_successful_routed_attempt_keeps_the_decision_cached(self):
        RouterReplyClient.reset(reply="sol-high")
        client = self._auto_route_client()
        attempted_models = []

        async def backend_call(_backend, model, *args, **kwargs):
            attempted_models.append(model)
            return {"choices": []}

        request = self._auto_route_request()
        first = self._post_auto_route(client, request, backend_call)
        RouterReplyClient.reply = "luna-max"
        second = self._post_auto_route(client, request, backend_call)

        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertEqual(attempted_models, ["sol-high", "sol-high"])
        self.assertEqual(RouterReplyClient.calls, 1)

    def test_context_precheck_reclassifies_without_calling_the_backend(self):
        RouterReplyClient.reset(reply="luna-max")
        client = self._auto_route_client(**{
            "luna-max": {"context_limit": 40},
            "sol-high": {"context_limit": 100_000},
        })
        attempted_models = []

        async def backend_call(_backend, model, *args, **kwargs):
            attempted_models.append(model)
            return {"choices": []}

        request = self._auto_route_request("This request contains many distinct words and phrases. " * 20)
        rejected = self._post_auto_route(client, request, backend_call)
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(attempted_models, [])

        RouterReplyClient.reply = "sol-high"
        retried = self._post_auto_route(client, request, backend_call)

        self.assertEqual(retried.status_code, 200)
        self.assertEqual(attempted_models, ["sol-high"])
        self.assertEqual(RouterReplyClient.calls, 2)

    def test_streaming_client_error_before_first_output_reclassifies(self):
        RouterReplyClient.reset(reply="sol-high")
        client = self._auto_route_client()
        attempted_models = []

        async def backend_call(_backend, model, *args, **kwargs):
            attempted_models.append(model)
            if len(attempted_models) == 1:
                async def error_event():
                    yield 'data: {"error":{"message":"stream rejected","status_code":400}}\n\n'
                return error_event()

            async def success_stream():
                yield 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
                yield "data: [DONE]\n\n"
            return success_stream()

        request = self._auto_route_request(stream=True)
        failed = self._post_auto_route(client, request, backend_call)
        self.assertEqual(failed.status_code, 400)

        RouterReplyClient.reply = "luna-max"
        retried = self._post_auto_route(client, request, backend_call)

        self.assertEqual(retried.status_code, 200)
        self.assertEqual(attempted_models, ["sol-high", "luna-max"])
        self.assertEqual(RouterReplyClient.calls, 2)

    def test_streaming_client_error_after_content_reclassifies(self):
        RouterReplyClient.reset(reply="sol-high")
        client = self._auto_route_client()
        attempted_models = []

        async def backend_call(_backend, model, *args, **kwargs):
            attempted_models.append(model)
            if len(attempted_models) == 1:
                async def failing_stream():
                    yield 'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
                    raise HTTPException(status_code=400, detail="stream rejected after content")
                return failing_stream()

            async def success_stream():
                yield 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
                yield "data: [DONE]\n\n"
            return success_stream()

        request = self._auto_route_request(stream=True)
        failed = self._post_auto_route(client, request, backend_call)
        self.assertEqual(failed.status_code, 200)
        self.assertEqual(attempted_models, ["sol-high"])

        RouterReplyClient.reply = "luna-max"
        retried = self._post_auto_route(client, request, backend_call)

        self.assertEqual(retried.status_code, 200)
        self.assertEqual(attempted_models, ["sol-high", "luna-max"])
        self.assertEqual(RouterReplyClient.calls, 2)

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

    def test_q2_machine_examples_select_configured_tier_without_classifier(self):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient), \
                patch("openclaw_router.routers._safe_log") as log:
            selected = self._select_many(router, [query[:500] for _, query, _ in Q2_MACHINE_ROUTING_EXAMPLES])

        self.assertEqual(selected, ["luna-max"] * len(Q2_MACHINE_ROUTING_EXAMPLES))
        self.assertEqual(RouterReplyClient.calls, 0)
        decision_lines = [call.args[0] for call in log.call_args_list]
        self.assertEqual(len(decision_lines), len(Q2_MACHINE_ROUTING_EXAMPLES))
        for line, (_, _, marker) in zip(decision_lines, Q2_MACHINE_ROUTING_EXAMPLES):
            self.assertIn("[Router] Machine -> luna-max", line)
            self.assertIn(f"marker={marker.encode('unicode_escape').decode('ascii')}", line)
            self.assertNotIn("Strategy=llm", line)

    def test_fleet_heartbeat_matches_firstmate_generator_grammar(self):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)

        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            selected = self._select_many(
                router, [item["routing_text"] for item in HEARTBEAT_GENERATOR_WINDOWS]
            )

        self.assertEqual(selected, ["luna-max"] * len(HEARTBEAT_GENERATOR_WINDOWS))
        self.assertEqual(RouterReplyClient.calls, 0)

    def test_dedicated_heartbeat_envelope_matches_arbitrary_row_text(self):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)

        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            selected = self._select_many(
                router, [item["routing_text"] for item in DEDICATED_GENERATOR_WINDOWS]
            )

        self.assertEqual(selected, ["luna-max"] * len(DEDICATED_GENERATOR_WINDOWS))
        self.assertEqual(RouterReplyClient.calls, 0)

    def test_from_firstmate_and_away_daemon_markers_route_by_prefix(self):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)
        messages = (
            "[fm-from-firstmate]\x1f review this item",
            "\x1f daemon escalation summary",
        )

        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient), \
                patch("openclaw_router.routers._safe_log") as log:
            selected = self._select_many(router, messages)

        self.assertEqual(selected, ["luna-max", "luna-max"])
        self.assertEqual(RouterReplyClient.calls, 0)
        decision_lines = [call.args[0] for call in log.call_args_list]
        self.assertIn("marker=[fm-from-firstmate]\\x1f", decision_lines[0])
        self.assertIn("marker=\\x1f", decision_lines[1])
        self.assertTrue(all("Strategy=llm" not in line for line in decision_lines))

    def _assert_turnend_generator_prefix_routes(self, name):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient), \
                patch("openclaw_router.routers._safe_log") as log:
            selected = self._select_many(router, [TURNEND_GUARD_WINDOWS[name]])

        self.assertEqual(selected, ["luna-max"])
        self.assertEqual(RouterReplyClient.calls, 0)
        decision_lines = [call.args[0] for call in log.call_args_list]
        self.assertEqual(len(decision_lines), 1)
        self.assertIn("[Router] Machine -> luna-max (marker=TURN WOULD END)", decision_lines[0])
        self.assertNotIn("Strategy=llm", decision_lines[0])

    def test_shell_blind_turnend_banner_routes_as_machine(self):
        self._assert_turnend_generator_prefix_routes("shell-blind")

    def test_shell_invalid_home_turnend_banner_routes_as_machine(self):
        self._assert_turnend_generator_prefix_routes("shell-invalid-home")

    def test_opencode_blind_turnend_message_routes_as_machine(self):
        self._assert_turnend_generator_prefix_routes("opencode-blind")

    def test_opencode_invalid_home_turnend_message_routes_as_machine(self):
        self._assert_turnend_generator_prefix_routes("opencode-invalid-home")

    def test_shell_warning_before_blind_turnend_banner_routes_as_machine(self):
        self._assert_turnend_generator_prefix_routes("shell-warning-blind")

    def test_opencode_warning_before_blind_turnend_message_routes_as_machine(self):
        self._assert_turnend_generator_prefix_routes("opencode-warning-blind")

    def test_human_mid_sentence_turnend_banner_quotes_classify(self):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)
        human_quotes = [
            f"Please review this quoted warning before continuing: {message} The quote is only context."
            for message in TURNEND_GUARD_WINDOWS.values()
        ]

        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            selected = self._select_many(router, human_quotes)

        self.assertEqual(selected, ["sol-high"] * len(TURNEND_GUARD_WINDOWS))
        self.assertEqual(RouterReplyClient.calls, len(TURNEND_GUARD_WINDOWS))

    def test_human_mid_sentence_operational_home_warning_quote_classifies_and_reuses_cache(self):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)
        quote = (
            "Please explain this warning quoted in a conversation: "
            f"{TURNEND_GUARD_WINDOWS['shell-warning-blind']} That is only context."
        )

        selected = self._select_many(router, [quote, quote])

        self.assertEqual(selected, ["sol-high", "sol-high"])
        self.assertEqual(RouterReplyClient.calls, 1)

    def test_machine_route_precedes_a_conflicting_cached_decision(self):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)
        query = Q2_T08_HEARTBEAT
        router._decision_cache[("", query)] = ("sol-high", time.monotonic())

        self.assertEqual(self._select_many(router, [query]), ["luna-max"])
        self.assertEqual(RouterReplyClient.calls, 0)

    def test_human_mid_sentence_marker_quotes_classify_and_reuse_cache(self):
        router = self._router(cache_size=8)
        configure_machine_routing(router.config)
        heartbeat_quote = next(
            item["routing_text"] for item in HEARTBEAT_GENERATOR_WINDOWS
            if item["name"] == "timestamp-0-ram-0-unavailable"
        )
        dedicated_quote = DEDICATED_GENERATOR_WINDOWS[0]["routing_text"]
        human_quote = (
            "Please explain these quoted markers [fm-from-peer]\x1f, [fm-from-firstmate]\x1f, \x1f, "
            "WATCHER FIRED [, and OBSERVER:; quote this dedicated payload: "
            + dedicated_quote
            + "; and this fleet payload: "
            + heartbeat_quote
        )
        human_receipt_mid_sentence = "Please explain [fm-heartbeat-receipt:hb-human-quote] as a marker inside this sentence."

        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            selected = self._select_many(
                router,
                [Q2_T08_HEARTBEAT, human_quote[:500], human_quote[:500], human_receipt_mid_sentence],
            )

        self.assertEqual(selected, ["luna-max", "sol-high", "sol-high", "sol-high"])
        self.assertEqual(RouterReplyClient.calls, 2)

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


class ClassifierUsageLogTests(unittest.TestCase):
    USAGE = {"prompt_tokens": 7312, "completion_tokens": 24}

    def setUp(self):
        RouterReplyClient.reset(reply="sol-high", usage=dict(self.USAGE))
        self.tmp = tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.dirname(__file__)))
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "classifier-usage.jsonl")

    def _router(self, cache_size=0):
        config = make_config(router=RouterConfig(
            strategy="llm", provider="mock", base_url="https://example.test/v1",
            model="accounts/fireworks/models/gpt-oss-120b", cache_size=cache_size,
            fallback="glm-5.3-flash",
        ))
        config.router.classifier_usage_log_path = self.path
        config.llms["sol-high"].model_id = "openai/responses/gpt-6-sol"
        config.llms["sol-high"].served_model = "gpt-6-sol"
        config.llms["glm-5.3-flash"].model_id = "accounts/fireworks/models/glm-5p3-flash"
        return OpenClawRouter(config)

    def _run_and_join_writers(self, operation):
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            result = operation()
        router_module._flush_classifier_usage_records()
        return result

    def _select(self, router, query):
        return self._run_and_join_writers(lambda: asyncio.run(router.select_model(query)))

    def _records(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    def test_provider_usage_record_names_classifier_and_served_model(self):
        router = self._router()
        timer = SimpleNamespace(time=Mock(return_value=123456789.25), monotonic=Mock(side_effect=[10.0, 11.25]))
        with patch.object(router_module, "time", timer):
            self.assertEqual(self._select(router, "hard task"), "sol-high")
        records = self._records()
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(set(record), {
            "ts", "model", "in_tokens", "out_tokens", "latency_ms", "fallback", "served_model",
        })
        self.assertEqual(record["ts"], 123456789.25)
        self.assertEqual(record["model"], "accounts/fireworks/models/gpt-oss-120b")
        self.assertEqual(record["in_tokens"], 7312)
        self.assertEqual(record["out_tokens"], 24)
        self.assertIsInstance(record["latency_ms"], (int, float))
        self.assertEqual(record["latency_ms"], 1250.0)
        self.assertFalse(record["fallback"])
        self.assertEqual(record["served_model"], router.config.llms["sol-high"].served_id)
        self.assertEqual(record["served_model"], "gpt-6-sol")
        self.assertNotEqual(record["served_model"], router.config.llms["sol-high"].model_id)

    def test_unparseable_and_unknown_choices_keep_usage_on_fallback_records(self):
        for index, reply in enumerate(("", "not-a-configured-model")):
            with self.subTest(reply=reply):
                self.path = os.path.join(self.tmp.name, f"classifier-usage-{index}.jsonl")
                RouterReplyClient.reset(reply=reply, usage=dict(self.USAGE))
                router = self._router()
                self.assertEqual(self._select(router, "fallback case"), "glm-5.3-flash")
                record, = self._records()
                self.assertEqual((record["in_tokens"], record["out_tokens"]), (7312, 24))
                self.assertTrue(record["fallback"])
                self.assertEqual(record["served_model"], "accounts/fireworks/models/glm-5p3-flash")

    def test_cache_hit_adds_no_record_after_classifier_miss(self):
        router = self._router(cache_size=8)
        self.assertEqual(self._select(router, "same turn"), "sol-high")
        after_miss = self._records()
        self.assertEqual(len(after_miss), 1)
        self.assertEqual(self._select(router, "same turn"), "sol-high")
        self.assertEqual(RouterReplyClient.calls, 1)
        self.assertEqual(self._records(), after_miss)

    def test_machine_route_makes_no_classifier_call_or_usage_record(self):
        router = self._router()
        configure_machine_routing(router.config)

        self.assertEqual(self._select(router, Q2_T46_HEARTBEAT), "luna-max")
        self.assertEqual(RouterReplyClient.calls, 0)
        self.assertEqual(self._records(), [])

    def test_coalesced_burst_writes_once_for_one_classifier_call(self):
        router = self._router(cache_size=8)

        async def run():
            return await asyncio.gather(*(router.select_model("same turn") for _ in range(20)))

        selected = self._run_and_join_writers(lambda: asyncio.run(run()))
        self.assertEqual(selected, ["sol-high"] * 20)
        self.assertEqual(RouterReplyClient.calls, 1)
        self.assertEqual(len(self._records()), 1)

    def test_provider_failure_without_usage_records_null_tokens_and_fallback(self):
        RouterReplyClient.error = RuntimeError("connection failed before a response")
        router = self._router()
        self.assertEqual(self._select(router, "failed call"), "glm-5.3-flash")
        self.assertEqual(RouterReplyClient.calls, 1)
        record, = self._records()
        self.assertIsNone(record["in_tokens"])
        self.assertIsNone(record["out_tokens"])
        self.assertTrue(record["fallback"])
        self.assertEqual(record["served_model"], "accounts/fireworks/models/glm-5p3-flash")

    def test_partial_and_absent_provider_usage_never_becomes_zero(self):
        cases = (
            ({"prompt_tokens": 17}, 17, None),
            (None, None, None),
        )
        for index, (usage, expected_in, expected_out) in enumerate(cases):
            with self.subTest(usage=usage):
                self.path = os.path.join(self.tmp.name, f"missing-usage-{index}.jsonl")
                RouterReplyClient.reset(reply="sol-high", usage=usage)
                router = self._router()
                self.assertEqual(self._select(router, "usage gap"), "sol-high")
                record, = self._records()
                self.assertEqual((record["in_tokens"], record["out_tokens"]), (expected_in, expected_out))
                self.assertNotEqual(record["in_tokens"], 0)
                self.assertNotEqual(record["out_tokens"], 0)

    def test_non_200_response_usage_is_preserved_on_fallback(self):
        RouterReplyClient.reset(reply="sol-high", usage={"prompt_tokens": 41, "completion_tokens": 7})
        RouterReplyClient.status_code = 503
        router = self._router()
        self.assertEqual(self._select(router, "provider error with usage"), "glm-5.3-flash")
        record, = self._records()
        self.assertEqual((record["in_tokens"], record["out_tokens"]), (41, 7))
        self.assertTrue(record["fallback"])
        self.assertEqual(record["served_model"], "accounts/fireworks/models/glm-5p3-flash")

    def test_model_field_matches_model_sent_after_extra_body_override(self):
        router = self._router()
        override = "accounts/fireworks/models/gpt-oss-120b-override"
        router.config.router.extra_body = {"model": override}
        self.assertEqual(self._select(router, "model override"), "sol-high")
        record, = self._records()
        self.assertEqual(RouterReplyClient.last_json["model"], override)
        self.assertEqual(record["model"], RouterReplyClient.last_json["model"])

    def test_cancelled_classifier_call_does_not_claim_a_fallback_model(self):
        router = self._router()

        async def run():
            RouterReplyClient.gate = asyncio.Event()
            request = asyncio.create_task(router.select_model("cancel in flight"))
            for _ in range(100):
                if RouterReplyClient.calls:
                    break
                await asyncio.sleep(0)
            self.assertEqual(RouterReplyClient.calls, 1)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request

        self._run_and_join_writers(lambda: asyncio.run(run()))
        record, = self._records()
        self.assertTrue(record["cancelled"])
        self.assertFalse(record["fallback"])
        self.assertNotIn("served_model", record)
        self.assertIsNone(record["in_tokens"])
        self.assertIsNone(record["out_tokens"])

    def test_no_classifier_dispatch_adds_no_record_after_a_classifier_record(self):
        router = self._router()
        self.assertEqual(self._select(router, "classifier miss"), "sol-high")
        after_classifier = self._records()
        self.assertEqual(len(after_classifier), 1)

        router.config.llms = {"only-model": make_llm("only-model")}
        self.assertEqual(self._select(router, "direct route"), "only-model")
        self.assertEqual(RouterReplyClient.calls, 1)
        self.assertEqual(self._records(), after_classifier)

        pre_dispatch = self._router()
        pre_dispatch.config.api_keys = {}
        self.assertEqual(self._select(pre_dispatch, "no key"), "glm-5.3-flash")
        self.assertEqual(RouterReplyClient.calls, 1)
        self.assertEqual(self._records(), after_classifier)

    def test_unwritable_path_reports_error_without_failing_routing(self):
        blocker = os.path.join(self.tmp.name, "not-a-directory")
        with open(blocker, "w", encoding="utf-8") as output:
            output.write("file")
        router = self._router()
        router.config.router.classifier_usage_log_path = os.path.join(blocker, "usage.jsonl")
        messages = []
        write_failure_reported = threading.Event()
        safe_log = router_module._safe_log

        def capture(message):
            message = str(message)
            messages.append(message)
            if "Classifier usage logging failed" in message:
                write_failure_reported.set()
            safe_log(message)

        with patch("openclaw_router.routers._safe_log", capture):
            self.assertEqual(self._select(router, "logging fails"), "sol-high")
            self.assertTrue(write_failure_reported.wait(3), "write failure was not reported")
        self.assertTrue(any("dropped 1 record(s)" in message for message in messages))

    def test_stalled_writer_preserves_routing_and_bounds_shutdown_drops(self):
        router = self._router()
        router_module._flush_classifier_usage_records()
        test_queue_limit = 16
        writer_entered = threading.Event()
        release_writer = threading.Event()
        writer_finished = threading.Event()
        route_finished = threading.Event()
        overflow_reported = threading.Event()
        shutdown_reported = threading.Event()
        shutdown_finished = threading.Event()
        drop_messages = []
        result = {}
        shutdown_thread = None
        append = router_module._append_classifier_usage_record
        route_count = test_queue_limit + 2

        def stalled_append(path, record):
            writer_entered.set()
            try:
                release_writer.wait()
                append(path, record)
            finally:
                writer_finished.set()

        def route():
            try:
                async def run():
                    gate = asyncio.Event()
                    RouterReplyClient.gate = gate
                    requests = [
                        asyncio.create_task(router.select_model(f"stalled log {index}"))
                        for index in range(route_count)
                    ]
                    for _ in range(100):
                        if RouterReplyClient.calls == route_count:
                            break
                        await asyncio.sleep(0)
                    self.assertEqual(RouterReplyClient.calls, route_count)
                    gate.set()
                    return await asyncio.gather(*requests)

                result["selected"] = asyncio.run(run())
            finally:
                route_finished.set()

        def capture_log(message):
            message = str(message)
            drop_messages.append(message)
            if "Classifier usage logging failed: dropped" in message and "bounded writer buffer is full" in message:
                overflow_reported.set()
            if "shutdown drain timed out" in message:
                shutdown_reported.set()

        with patch.object(router_module, "_CLASSIFIER_USAGE_QUEUE_LIMIT", test_queue_limit), \
                patch.object(router_module._classifier_usage_queue, "maxsize", test_queue_limit), \
                patch("openclaw_router.routers._append_classifier_usage_record", stalled_append), \
                patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient), \
                patch("openclaw_router.routers._safe_log", capture_log):
            worker = threading.Thread(target=route, daemon=True)
            baseline_threads = set(threading.enumerate())
            unexpected_threads = []
            shutdown_thread = None
            worker.start()
            try:
                self.assertTrue(writer_entered.wait(3), "the append worker did not start")
                self.assertTrue(route_finished.wait(3), "the stalled append held the route result")
                self.assertEqual(result.get("selected"), ["sol-high"] * route_count)
                self.assertEqual(RouterReplyClient.calls, route_count)
                self.assertTrue(overflow_reported.wait(3), "writer overflow was not reported")
                self.assertTrue(
                    router_module._wait_for_classifier_usage_reports(3),
                    "writer overflow report did not finish before shutdown accounting",
                )
                overflow_reports = [
                    message for message in drop_messages
                    if "bounded writer buffer is full" in message and "dropped " in message
                ]
                self.assertTrue(overflow_reports, "writer overflow did not include its dropped-record count")
                writer = router_module._classifier_usage_writer_thread
                writers = [thread for thread in threading.enumerate() if thread.name == "classifier-usage-writer"]
                self.assertEqual(len(writers), 1)
                reporters = [thread for thread in threading.enumerate() if thread.name == "classifier-usage-reporter"]
                self.assertEqual(len(reporters), 1)
                allowed_threads = baseline_threads | {worker}
                if writer is not None:
                    allowed_threads.add(writer)
                allowed_threads.update(reporters)
                unexpected_threads = list(set(threading.enumerate()) - allowed_threads)
                self.assertEqual(unexpected_threads, [])
                self.assertLessEqual(
                    router_module._classifier_usage_queue.qsize(),
                    router_module._classifier_usage_queue.maxsize,
                )
                queued_before_shutdown = router_module._classifier_usage_queue.qsize()
                self.assertGreater(queued_before_shutdown, 0)
                def shutdown():
                    try:
                        router_module._shutdown_classifier_usage_writer()
                    finally:
                        shutdown_finished.set()

                with patch.object(router_module, "_CLASSIFIER_USAGE_SHUTDOWN_TIMEOUT_SECONDS", 0.05):
                    shutdown_thread = threading.Thread(target=shutdown, daemon=True)
                    shutdown_thread.start()
                    self.assertTrue(shutdown_finished.wait(3), "shutdown waited indefinitely for the stalled writer")
                    self.assertTrue(shutdown_reported.wait(3), "shutdown did not report unflushed records")
                    shutdown_reports = [message for message in drop_messages if "shutdown drain timed out" in message]
                    self.assertTrue(
                        any(f"dropped {queued_before_shutdown} record(s)" in message for message in shutdown_reports),
                        repr(shutdown_reports),
                    )
                    self.assertTrue(
                        any("unconfirmed 1 record(s)" in message for message in shutdown_reports),
                        repr(shutdown_reports),
                    )
                    shutdown_thread.join(3)
            finally:
                release_writer.set()
                worker.join(3)
                if shutdown_thread is not None:
                    shutdown_thread.join(3)
                router_module._flush_classifier_usage_records()
            worker.join(3)
            self.assertFalse(worker.is_alive(), "route thread did not finish")
            self.assertTrue(writer_finished.wait(3), "the append worker did not finish after release")
            for thread in unexpected_threads:
                thread.join(3)
        writer = router_module._classifier_usage_writer_thread
        if writer is not None:
            writer.join(3)
            self.assertFalse(writer.is_alive(), "writer did not stop after consuming the shutdown marker")
        router_module._flush_classifier_usage_records()
        self.assertEqual(len(self._records()), 1)

    def test_shutdown_join_is_bounded_with_active_write(self):
        router = self._router()
        writer_entered = threading.Event()
        release_writer = threading.Event()
        route_finished = threading.Event()
        shutdown_finished = threading.Event()
        shutdown_reported = threading.Event()
        messages = []
        append = router_module._append_classifier_usage_record
        route_result = {}

        def stalled_append(path, record):
            writer_entered.set()
            try:
                release_writer.wait()
                append(path, record)
            finally:
                route_finished.set()

        def route():
            try:
                route_result["selected"] = asyncio.run(router.select_model("shutdown with active write"))
            finally:
                route_finished.set()

        def capture_log(message):
            message = str(message)
            messages.append(message)
            if "unconfirmed 1 record(s)" in message and "shutdown drain timed out" in message:
                shutdown_reported.set()

        with patch("openclaw_router.routers._append_classifier_usage_record", stalled_append), \
                patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient), \
                patch("openclaw_router.routers._safe_log", capture_log):
            route_thread = threading.Thread(target=route, daemon=True)
            route_thread.start()
            shutdown_thread = None
            try:
                self.assertTrue(writer_entered.wait(3), "writer did not enter the stalled append")
                self.assertTrue(route_finished.wait(3), "the stalled append held the route")
                self.assertEqual(route_result.get("selected"), "sol-high")
                self.assertEqual(router_module._classifier_usage_queue.qsize(), 0)

                def shutdown():
                    try:
                        router_module._shutdown_classifier_usage_writer()
                    finally:
                        shutdown_finished.set()

                with patch.object(router_module, "_CLASSIFIER_USAGE_SHUTDOWN_TIMEOUT_SECONDS", 0.05):
                    shutdown_thread = threading.Thread(target=shutdown, daemon=True)
                    shutdown_thread.start()
                    self.assertTrue(shutdown_finished.wait(3), "shutdown join waited on the stalled write")
                    self.assertTrue(shutdown_reported.wait(3), "shutdown did not mark its in-flight record unconfirmed")
                    shutdown_reports = [message for message in messages if "shutdown drain timed out" in message]
                    self.assertFalse(any("dropped 1 record(s)" in message for message in shutdown_reports))
            finally:
                release_writer.set()
                route_thread.join(3)
                if shutdown_thread is not None:
                    shutdown_thread.join(3)
        writer = router_module._classifier_usage_writer_thread
        if writer is not None:
            writer.join(3)
        router_module._flush_classifier_usage_records()

    def test_child_process_flushes_record_during_normal_shutdown(self):
        path = os.path.join(self.tmp.name, "child-classifier-usage.jsonl")
        script = textwrap.dedent("""\
            import asyncio
            import sys
            import threading
            from openclaw_router import routers
            from tests.test_openclaw_opencode_routing import RouterReplyClient, make_config
            from openclaw_router.config import RouterConfig
            from openclaw_router.routers import OpenClawRouter

            config = make_config(router=RouterConfig(
                strategy="llm", provider="mock", base_url="https://example.test/v1",
                model="accounts/fireworks/models/gpt-oss-120b", fallback="glm-5.3-flash",
            ))
            config.router.classifier_usage_log_path = sys.argv[1]
            RouterReplyClient.reset(reply="sol-high", usage={"prompt_tokens": 19, "completion_tokens": 4})
            routers.httpx.AsyncClient = RouterReplyClient
            append = routers._append_classifier_usage_record
            release_writer = threading.Event()

            class ShutdownAwareQueue:
                def __init__(self, delegate):
                    self.delegate = delegate

                def __getattr__(self, name):
                    return getattr(self.delegate, name)

                def put(self, item, *args, **kwargs):
                    if item is routers._CLASSIFIER_USAGE_STOP:
                        release_writer.set()
                    return self.delegate.put(item, *args, **kwargs)

            routers._classifier_usage_queue = ShutdownAwareQueue(routers._classifier_usage_queue)

            def gated_append(path, record):
                release_writer.wait()
                append(path, record)

            routers._append_classifier_usage_record = gated_append
            asyncio.run(OpenClawRouter(config).select_model("exit after routing"))
        """)
        repo = os.path.dirname(os.path.dirname(__file__))
        subprocess.run([sys.executable, "-c", script, path], cwd=repo, check=True)
        records = self._records_from(path) if os.path.exists(path) else []
        self.assertEqual(len(records), 1)
        record, = records
        self.assertEqual(record["model"], "accounts/fireworks/models/gpt-oss-120b")
        self.assertEqual((record["in_tokens"], record["out_tokens"]), (19, 4))
        self.assertEqual(record["served_model"], "sol-high")

    def test_classifier_record_after_shutdown_is_restarted_or_reported(self):
        router = self._router()
        self.assertEqual(self._select(router, "before shutdown"), "sol-high")
        before = self._records()
        self.assertEqual(len(before), 1)
        old_writer = router_module._classifier_usage_writer_thread
        self.assertIsNotNone(old_writer)
        router_module._shutdown_classifier_usage_writer()
        self.assertFalse(old_writer.is_alive())

        RouterReplyClient.reset(reply="sol-high", usage=dict(self.USAGE))
        drop_reported = threading.Event()
        reports = []

        def capture_log(message):
            message = str(message)
            reports.append(message)
            if "Classifier usage logging failed: dropped" in message:
                drop_reported.set()

        new_writer = None
        try:
            with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient), \
                    patch("openclaw_router.routers._safe_log", capture_log):
                self.assertEqual(asyncio.run(router.select_model("after shutdown")), "sol-high")
                new_writer = router_module._classifier_usage_writer_thread
                if new_writer is not old_writer and new_writer.is_alive():
                    router_module._flush_classifier_usage_records()
                    self.assertEqual(len(self._records()), 2)
                else:
                    self.assertTrue(drop_reported.wait(3), "record was queued to a dead writer without a drop report")
                    self.assertEqual(self._records(), before)
        finally:
            if new_writer is old_writer or new_writer is None or not new_writer.is_alive():
                router_module._shutdown_classifier_usage_writer()

    def _records_from(self, path):
        with open(path, encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    def test_healthy_writer_keeps_all_records_for_simultaneous_burst(self):
        router = self._router()
        router_module._flush_classifier_usage_records()
        router_module._shutdown_classifier_usage_writer()
        writer_start = threading.Event()
        real_thread = threading.Thread
        count = 300
        with router_module._classifier_usage_report_condition:
            drops_before = router_module._classifier_usage_drop_total

        def gated_thread(*args, **kwargs):
            target = kwargs.get("target")
            if target is router_module._classifier_usage_writer:
                def delayed_writer(*target_args, **target_kwargs):
                    writer_start.wait()
                    target(*target_args, **target_kwargs)

                kwargs["target"] = delayed_writer
            return real_thread(*args, **kwargs)

        async def run_burst():
            PendingRouterReplyClient.reset(reply="sol-high", usage=dict(self.USAGE))
            PendingRouterReplyClient.gates = [asyncio.Event() for _ in range(count)]
            tasks = [
                asyncio.create_task(router.select_model(f"healthy burst {index}"))
                for index in range(count)
            ]
            for _ in range(1000):
                if PendingRouterReplyClient.calls == count:
                    break
                await asyncio.sleep(0)
            self.assertEqual(PendingRouterReplyClient.calls, count)
            for gate in PendingRouterReplyClient.gates:
                gate.set()
            return await asyncio.gather(*tasks)

        try:
            with patch("openclaw_router.routers.httpx.AsyncClient", PendingRouterReplyClient), \
                    patch("openclaw_router.routers.threading.Thread", gated_thread), \
                    patch("openclaw_router.routers._safe_log", lambda message: None):
                try:
                    selected = asyncio.run(run_burst())
                finally:
                    writer_start.set()
                    router_module._flush_classifier_usage_records()
        finally:
            writer_start.set()

        self.assertEqual(selected, ["sol-high"] * count)
        self.assertEqual(PendingRouterReplyClient.calls, count)
        records = self._records()
        self.assertEqual(len(records), count)
        with router_module._classifier_usage_report_condition:
            self.assertEqual(router_module._classifier_usage_drop_total - drops_before, 0)

    def test_full_writer_buffer_does_not_skip_classifier_dispatch(self):
        router = self._router()
        usage_queue = router_module._classifier_usage_queue
        router_module._flush_classifier_usage_records()
        original_maxsize = usage_queue.maxsize
        test_buffer_size = 8
        writer_entered = threading.Event()
        release_writer = threading.Event()
        overflow_reported = threading.Event()
        append = router_module._append_classifier_usage_record

        def stalled_append(path, record):
            writer_entered.set()
            release_writer.wait()
            append(path, record)

        def capture_log(message):
            message = str(message)
            if "dropped" in message and "bounded writer buffer is full" in message:
                overflow_reported.set()

        usage_queue.maxsize = test_buffer_size
        try:
            with patch("openclaw_router.routers._append_classifier_usage_record", stalled_append), \
                    patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient), \
                    patch("openclaw_router.routers._safe_log", capture_log):
                self.assertEqual(asyncio.run(router.select_model("start the writer")), "sol-high")
                self.assertTrue(writer_entered.wait(3), "writer did not enter the stalled append")
                for index in range(test_buffer_size):
                    record = router_module._classifier_usage_record(
                        router.config,
                        "accounts/fireworks/models/gpt-oss-120b",
                        dict(self.USAGE),
                        123456789.0 + index,
                        1.0,
                        False,
                        "sol-high",
                    )
                    router_module._queue_classifier_usage_record(router.config, record)
                self.assertTrue(usage_queue.full(), "test did not fill the writer buffer before the next route")

                self.assertEqual(asyncio.run(router.select_model("classifier with full log buffer")), "sol-high")
                self.assertEqual(RouterReplyClient.calls, 2)
                self.assertTrue(overflow_reported.wait(3), "the full-buffer record drop was not reported")
        finally:
            release_writer.set()
            router_module._flush_classifier_usage_records()
            usage_queue.maxsize = original_maxsize

    def test_concurrent_distinct_calls_append_independently_parseable_lines(self):
        router = self._router()
        count = 300

        async def run():
            PendingRouterReplyClient.reset(reply="sol-high", usage=dict(self.USAGE))
            PendingRouterReplyClient.gates = [asyncio.Event() for _ in range(count)]
            tasks = [asyncio.create_task(router.select_model(f"query {index}")) for index in range(count)]
            for _ in range(1000):
                if PendingRouterReplyClient.calls == count:
                    break
                await asyncio.sleep(0)
            self.assertEqual(PendingRouterReplyClient.calls, count)
            PendingRouterReplyClient.gates[0].set()
            selected = [await tasks[0]]
            router_module._flush_classifier_usage_records()
            for gate in PendingRouterReplyClient.gates[1:]:
                gate.set()
            selected.extend(await asyncio.gather(*tasks[1:]))
            router_module._flush_classifier_usage_records()
            return selected

        with patch("openclaw_router.routers.httpx.AsyncClient", PendingRouterReplyClient), \
                patch("openclaw_router.routers._safe_log", lambda message: None):
            try:
                selected = asyncio.run(run())
            finally:
                router_module._flush_classifier_usage_records()
        self.assertEqual(selected, ["sol-high"] * count)
        self.assertEqual(PendingRouterReplyClient.calls, count)
        with open(self.path, "rb") as source:
            payload = source.read()
        self.assertTrue(payload.endswith(b"\n"))
        lines = payload.splitlines()
        self.assertEqual(len(lines), count)
        records = [json.loads(line) for line in lines]
        self.assertEqual({record["model"] for record in records}, {"accounts/fireworks/models/gpt-oss-120b"})
        self.assertEqual({record["served_model"] for record in records}, {"gpt-6-sol"})

    def test_append_lock_serializes_contending_writes_at_the_file_boundary(self):
        append_lock = router_module._classifier_usage_append_lock
        first_prefix_written = threading.Event()
        first_release = threading.Event()
        first_finished = threading.Event()
        second_lock_attempt = threading.Event()
        second_write_entered = threading.Event()
        second_prefix_written = threading.Event()
        second_release = threading.Event()
        second_finished = threading.Event()
        output_path = os.path.join(self.tmp.name, "contended.jsonl")
        open_file = open
        config = self._router().config
        record_a = router_module._classifier_usage_record(
            config,
            "accounts/fireworks/models/gpt-oss-120b",
            {"prompt_tokens": 3, "completion_tokens": 1},
            123456789.0,
            1.25,
            False,
            "sol-high",
        )
        record_b = router_module._classifier_usage_record(
            config,
            "accounts/fireworks/models/glm-5p3-flash",
            {"prompt_tokens": 5, "completion_tokens": 2},
            123456790.0,
            2.5,
            False,
            "sol-high",
        )

        class TrackingAppendLock:
            def __enter__(inner_self):
                if threading.current_thread().name == "append-second":
                    second_lock_attempt.set()
                return append_lock.__enter__()

            def __exit__(inner_self, exc_type, exc_value, traceback):
                return append_lock.__exit__(exc_type, exc_value, traceback)

        class SplitOutput:
            def __init__(self, output):
                self.output = output

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.output.close()

            def write(self, line):
                record = json.loads(line)
                split = len(line) // 2
                if record["model"] == "accounts/fireworks/models/gpt-oss-120b":
                    self.output.write(line[:split])
                    self.output.flush()
                    first_prefix_written.set()
                    first_release.wait()
                    self.output.write(line[split:])
                    self.output.flush()
                    first_finished.set()
                else:
                    second_write_entered.set()
                    self.output.write(line[:split])
                    self.output.flush()
                    second_prefix_written.set()
                    second_release.wait()
                    self.output.write(line[split:])
                    self.output.flush()
                    second_finished.set()
                return len(line)

        def controlled_open(path, *args, **kwargs):
            output = open_file(path, *args, **kwargs)
            if os.fspath(path) == output_path:
                return SplitOutput(output)
            return output

        first = threading.Thread(
            target=router_module._append_classifier_usage_record,
            args=(output_path, record_a),
            name="append-first",
        )
        second = threading.Thread(
            target=router_module._append_classifier_usage_record,
            args=(output_path, record_b),
            name="append-second",
        )
        with patch.object(router_module, "_classifier_usage_append_lock", TrackingAppendLock()), \
                patch("builtins.open", controlled_open):
            first.start()
            try:
                self.assertTrue(first_prefix_written.wait(3), "first append did not reach write boundary")
                second.start()
                self.assertTrue(second_lock_attempt.wait(3), "second append did not attempt the lock")
                try:
                    acquired_without_wait = append_lock.acquire(blocking=False)
                except AttributeError:
                    acquired_without_wait = True
                else:
                    if acquired_without_wait:
                        append_lock.release()
                if acquired_without_wait:
                    self.assertTrue(second_write_entered.wait(3), "unlocked second append did not enter write")
                    self.assertTrue(second_prefix_written.wait(3), "unlocked second append did not write its prefix")
                    first_release.set()
                    self.assertTrue(first_finished.wait(3), "first append did not finish")
                else:
                    first_release.set()
                    self.assertTrue(first_finished.wait(3), "first append did not finish")
                self.assertTrue(second_write_entered.wait(3), "second append did not enter the serialized write")
                self.assertTrue(second_prefix_written.wait(3), "second append did not write its prefix")
                second_release.set()
                self.assertTrue(second_finished.wait(3), "second append did not finish")
            finally:
                first_release.set()
                second_release.set()
                first.join(3)
                second.join(3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        with open_file(output_path, encoding="utf-8") as source:
            lines = source.read().splitlines()
        records = [json.loads(line) for line in lines]
        self.assertEqual(
            {record["model"] for record in records},
            {"accounts/fireworks/models/gpt-oss-120b", "accounts/fireworks/models/glm-5p3-flash"},
        )


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
  machine_model: a
  machine_patterns:
    - kind: prefix
      pattern: "[machine]"
      marker: "[machine]"
  classifier_usage_log_path: ${CLASSIFIER_USAGE_DIR}/classifier.jsonl
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

    def _load_yaml(self, source):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.dirname(__file__))) as temp_dir:
            path = os.path.join(temp_dir, "router.yaml")
            with open(path, "w", encoding="utf-8") as output:
                output.write(source)
            return OpenClawConfig.from_yaml(path)

    def test_new_fields_are_parsed(self):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.dirname(__file__))) as temp_dir:
            config_dir = os.path.join(temp_dir, "config")
            os.mkdir(config_dir)
            path = os.path.join(config_dir, "router.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.YAML)
            with patch.dict(os.environ, {"CLASSIFIER_USAGE_DIR": "usage"}):
                config = OpenClawConfig.from_yaml(path)
                memory = MemoryBank(
                    MemoryConfig(path="${CLASSIFIER_USAGE_DIR}/memory.jsonl"),
                    config_dir=config.config_dir,
                )
        router = config.router
        self.assertEqual(
            (router.prompt, router.max_tokens, router.max_tokens_param, router.temperature, router.extra_body,
             router.timeout, router.cache_size, router.cache_ttl, router.fallback, router.classifier_usage_log_path),
            ("P {query}", 777, "max_completion_tokens", None, {"reasoning_effort": "low"}, 9.5, 3, 42.0, "b",
             "usage/classifier.jsonl"),
        )
        self.assertEqual((router.machine_model, router.machine_patterns), (
            "a", [{"kind": "prefix", "pattern": "[machine]", "marker": "[machine]"}],
        ))
        self.assertEqual(
            router_module._classifier_usage_log_path(config),
            os.path.join(config.config_dir, "usage", "classifier.jsonl"),
        )
        self.assertEqual(
            os.fspath(memory.path),
            os.path.join(config.config_dir, "usage", "memory.jsonl"),
        )
        a, b = config.llms["a"], config.llms["b"]
        self.assertEqual(
            (a.model_id, a.served_id, a.provider_type, a.max_tokens, a.context_limit, a.extra_body, a.timeout,
             a.max_tokens_param),
            ("route/a", "served-a", "litellm", 1234, 5555, {"reasoning_effort": "max"}, 33.0, "max_completion_tokens"),
        )
        # Unset fields: context_limit falls back to the built-in table, served id to the model.
        self.assertEqual((b.context_limit, b.served_id), (None, "b"))

    def test_omitted_machine_tier_uses_first_configured_llm(self):
        RouterReplyClient.reset(reply="sol-high")
        self.assertIsNone(RouterConfig().machine_model)
        self.assertEqual(RouterConfig().machine_patterns, [])
        config = make_config(router=RouterConfig(
            strategy="llm", provider="mock", base_url="https://example.test/v1", model="classifier",
        ))
        configure_machine_routing(config)
        config.router.machine_model = None
        self.assertIsNone(config.router.machine_model)
        with patch("openclaw_router.routers.httpx.AsyncClient", RouterReplyClient):
            selected = asyncio.run(OpenClawRouter(config).select_model(Q2_T08_HEARTBEAT))
        self.assertEqual(selected, "luna-max")
        self.assertEqual(RouterReplyClient.calls, 0)

    def test_machine_model_must_name_a_configured_llm(self):
        with self.assertRaisesRegex(ValueError, "router.machine_model must name a configured llm"):
            self._load_yaml("router:\n  machine_model: missing\nllms:\n  a: {model: a}\n")

    def test_unknown_machine_pattern_kind_fails_during_yaml_load(self):
        with self.assertRaisesRegex(ValueError, r"router.machine_patterns\[0\]\.kind must be"):
            self._load_yaml(
                "router:\n  machine_patterns:\n    - kind: contains\n      pattern: marker\n"
                "llms:\n  a: {model: a}\n"
            )

    def test_invalid_machine_regex_fails_during_yaml_load(self):
        with self.assertRaisesRegex(ValueError, r"router.machine_patterns\[0\]\.pattern is an invalid regex"):
            self._load_yaml(
                "router:\n  machine_patterns:\n    - kind: regex\n      pattern: '['\n"
                "llms:\n  a: {model: a}\n"
            )

    def test_malformed_machine_pattern_shapes_fail_during_yaml_load(self):
        cases = (
            (
                "router:\n  machine_patterns: prefix\nllms:\n  a: {model: a}\n",
                "router.machine_patterns must be a YAML list",
            ),
            (
                "router:\n  machine_patterns:\n    - marker\nllms:\n  a: {model: a}\n",
                r"router.machine_patterns\[0\] must be a mapping",
            ),
            (
                "router:\n  machine_patterns:\n    - kind: prefix\n      pattern: 7\n"
                "llms:\n  a: {model: a}\n",
                r"router.machine_patterns\[0\]\.pattern must be a non-empty string",
            ),
        )
        for source, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self._load_yaml(source)


class OpencodeConfigTests(unittest.TestCase):
    def test_classifier_usage_log_path_has_stable_default(self):
        self.assertEqual(RouterConfig().classifier_usage_log_path, DEFAULT_USAGE_LOG_PATH)
        config = OpenClawConfig()
        self.assertEqual(
            router_module._classifier_usage_log_path(config),
            os.path.abspath(os.path.expanduser(DEFAULT_USAGE_LOG_PATH)),
        )

    def test_yaml_without_classifier_log_override_uses_default(self):
        with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.dirname(__file__))) as temp_dir:
            path = os.path.join(temp_dir, "router.yaml")
            with open(path, "w", encoding="utf-8") as output:
                output.write("router:\n  strategy: llm\n")
            config = OpenClawConfig.from_yaml(path)
        self.assertEqual(config.router.classifier_usage_log_path, DEFAULT_USAGE_LOG_PATH)
        self.assertEqual(
            router_module._classifier_usage_log_path(config),
            os.path.abspath(os.path.expanduser(DEFAULT_USAGE_LOG_PATH)),
        )

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
        self.assertEqual(
            config.router.classifier_usage_log_path,
            "~/.local/state/openclaw-router/opencode-classifier-usage.jsonl",
        )
        self.assertEqual(config.router.extra_body, {"reasoning_effort": "low"})
        self.assertEqual(config.get_api_key(config.router.provider), "fw")
        self.assertEqual(config.router.machine_model, "luna-max")
        self.assertEqual(
            [(entry["kind"], entry["marker"]) for entry in config.router.machine_patterns],
            [
                ("prefix", "[fm-from-peer]\x1f"),
                ("regex", "Fleet heartbeat. Run one supervision cycle"),
                ("regex", "[fm-heartbeat-receipt:"),
                ("prefix", "WATCHER FIRED ["),
                ("prefix", "OBSERVER: "),
                ("prefix", "[fm-from-firstmate]\x1f"),
                ("prefix", "\x1f"),
                ("regex", "TURN WOULD END"),
            ],
        )
        self.assertEqual(len(config.router.machine_patterns), 8)
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
        self.assertEqual(
            {name: config.llms[name].context_limit for name in TIERS},
            {"luna-max": 900000, "glm-5.3-flash": 1000000, "sol-high": 900000},
        )


if __name__ == "__main__":
    unittest.main()
