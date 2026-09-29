"""
OpenClaw Router Server
======================
OpenAI-compatible API server with intelligent LLM routing.

Usage:
    llmrouter serve --config configs/openclaw_example.yaml

Or directly:
    python server.py --config config.yaml
"""

import json
import math
import os
import re
import sys
import warnings
from collections.abc import Mapping
from functools import lru_cache
from typing import AsyncGenerator, Optional, Dict, Any, List, Callable

import tiktoken

TOKEN_CHUNK_CHARS = 8192
TOKEN_ESTIMATE_SAFETY_FACTOR = 1.02
ROUTING_CONTEXT_MESSAGE_WINDOW = 256
ROUTING_CONTEXT_MAX_USER_TURNS = 3
ROUTING_CONTEXT_MAX_CHARS = 4000
ROUTING_CONTEXT_MAX_PARTS = 64

# Check dependencies
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
    import httpx
    import uvicorn
except ImportError:
    print("Please install: pip install fastapi uvicorn httpx pydantic")
    sys.exit(1)

# Handle both relative and direct imports
try:
    from .config import OpenClawConfig, LLMConfig, MODELS_WITHOUT_SYSTEM_ROLE, MODEL_CONTEXT_LIMITS
    from .routers import OpenClawRouter, _safe_log
    from .media import process_multimodal_content, MediaConfig
except ImportError:
    from config import OpenClawConfig, LLMConfig, MODELS_WITHOUT_SYSTEM_ROLE, MODEL_CONTEXT_LIMITS
    from routers import OpenClawRouter, _safe_log
    from media import process_multimodal_content, MediaConfig


# ============================================================
# Request/Response Models
# ============================================================

class Message(BaseModel):
    role: str
    content: Optional[Any] = None  # Can be string or list (multimodal)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None  # None: the selected model's configured max_tokens
    stream: Optional[bool] = False
    user: Optional[str] = None  # Optional user id (used for memory scoping if enabled)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stream_options: Optional[Dict[str, Any]] = None
    top_p: Optional[float] = None
    stop: Optional[Any] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


# Standard OpenAI sampling/format fields forwarded to the backend when the client sets them.
PASSTHROUGH_FIELDS = (
    "top_p",
    "stop",
    "seed",
    "response_format",
    "parallel_tool_calls",
    "presence_penalty",
    "frequency_penalty",
)


def resolve_requested_model(config: OpenClawConfig, requested: str) -> str:
    """Map a requested served id (e.g. "gpt-6-luna") to its llm name so it pins that backend."""
    if requested in config.llms:
        return requested
    for name, llm in config.llms.items():
        if llm.served_id == requested:
            return name
    return requested


def request_messages(request: "ChatRequest") -> List[Dict[str, Any]]:
    """The request's messages as backend dicts, keeping tool-call fields."""
    messages = []
    for message in request.messages:
        message_payload = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls is not None:
            message_payload["tool_calls"] = message.tool_calls
        if message.tool_call_id is not None:
            message_payload["tool_call_id"] = message.tool_call_id
        if message.function_call is not None:
            message_payload["function_call"] = message.function_call
        messages.append(message_payload)
    return messages


def passthrough_params(request: "ChatRequest") -> Dict[str, Any]:
    params = {}
    for name in PASSTHROUGH_FIELDS:
        value = getattr(request, name, None)
        if value is not None:
            params[name] = value
    return params


# ============================================================
# Message Processing
# ============================================================

def normalize_content(content: Any) -> str:
    """Convert multimodal content to plain string"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif "text" in part:
                    text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content) if content else ""


def _bounded_routing_context_text(content: Any, max_chars: int) -> str:
    """Keep prior multimodal content from expanding the small routing-context budget."""
    if max_chars <= 0:
        return ""
    if isinstance(content, str):
        return content[:max_chars]
    if isinstance(content, list):
        parts = []
        remaining = max_chars
        for index, part in enumerate(content):
            if index >= ROUTING_CONTEXT_MAX_PARTS:
                break
            if isinstance(part, dict):
                text = part.get("text", "") if part.get("type") == "text" or "text" in part else ""
            elif isinstance(part, str):
                text = part
            else:
                continue
            if not isinstance(text, str) or not text:
                continue
            separator = "\n" if parts else ""
            available = remaining - len(separator)
            if available <= 0:
                break
            fragment = text[:available]
            parts.append(separator + fragment)
            remaining -= len(separator) + len(fragment)
            if len(fragment) < len(text):
                break
        return "".join(parts)
    return ""


def _build_routing_query(
    messages: List[Dict[str, Any]], last_user_idx: int, latest_query: str, router: OpenClawRouter
) -> str:
    """Keep follow-up routing grounded in a small tail of earlier user work."""
    if not latest_query or router._matching_machine_pattern(latest_query) is not None:
        return latest_query

    history_start = max(0, last_user_idx - ROUTING_CONTEXT_MESSAGE_WINDOW)
    context_parts = []
    remaining = ROUTING_CONTEXT_MAX_CHARS
    for message in reversed(messages[history_start:last_user_idx]):
        if message.get("role") != "user":
            continue
        separator = "\n\n" if context_parts else ""
        available = remaining - len(separator)
        if available <= 0:
            break
        text = _bounded_routing_context_text(message.get("content", ""), available)
        if not text:
            continue
        context_parts.append(text)
        remaining -= len(separator) + len(text)
        if len(context_parts) >= ROUTING_CONTEXT_MAX_USER_TURNS or remaining <= 0:
            break

    if not context_parts:
        return latest_query
    context_parts.reverse()
    return f"{latest_query}\n\nRecent user context:\n" + "\n\n".join(context_parts)


def normalize_messages(messages: List[Dict], model_id: str = "") -> List[Dict]:
    """Normalize message format for compatibility"""
    normalized = []
    system_contents = []

    for msg in messages:
        role = msg.get("role", "user")
        content = normalize_content(msg.get("content", ""))
        normalized_msg = {"role": role, "content": content}

        if msg.get("tool_calls") is not None:
            normalized_msg["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id") is not None:
            normalized_msg["tool_call_id"] = msg["tool_call_id"]
        if msg.get("function_call") is not None:
            normalized_msg["function_call"] = msg["function_call"]

        if role == "system":
            if content:
                system_contents.append(content)
        else:
            normalized.append(normalized_msg)

    system_content = "\n\n".join(system_contents)

    # Handle models without system role support
    if system_contents and model_id in MODELS_WITHOUT_SYSTEM_ROLE:
        if normalized and normalized[0]["role"] == "user":
            normalized[0]["content"] = f"[System Instructions]\n{system_content}\n\n[User Message]\n{normalized[0]['content']}"
        else:
            normalized.insert(0, {"role": "user", "content": f"[System Instructions]\n{system_content}"})
    elif system_contents:
        normalized.insert(0, {"role": "system", "content": system_content})

    return normalized


@lru_cache(maxsize=1)
def _input_token_encoding():
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        warnings.warn(
            "Tokenizer loading failed; using the UTF-8 byte-count fallback for context estimates.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def estimate_tokens(text: str) -> int:
    """Estimate model input with headroom so the estimate errs on the safe side."""
    encoding = _input_token_encoding()
    if encoding is None:
        return len(text.encode("utf-8", errors="surrogatepass"))
    try:
        token_count = sum(
            len(encoding.encode(text[offset:offset + TOKEN_CHUNK_CHARS], disallowed_special=()))
            for offset in range(0, len(text), TOKEN_CHUNK_CHARS)
        )
    except Exception:
        return len(text.encode("utf-8", errors="surrogatepass"))
    return math.ceil(token_count * TOKEN_ESTIMATE_SAFETY_FACTOR)


def estimate_input_tokens(messages: List[Dict], tools: Optional[List[Dict[str, Any]]] = None) -> int:
    """Estimate the complete serialized message and tool-schema input."""
    request_input = {"messages": messages}
    if tools is not None:
        request_input["tools"] = tools
    serialized_input = json.dumps(request_input, ensure_ascii=False, separators=(",", ":"), default=str)
    return estimate_tokens(serialized_input)


def resolve_context_limit(model_id: str, configured_limit: Optional[int]) -> int:
    """Use the configured limit or the same built-in fallback as output budgeting."""
    return configured_limit or MODEL_CONTEXT_LIMITS.get(model_id, 32768)


def context_length_error(llm: LLMConfig, context_limit: int) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "message": (
                f"Estimated input exceeds the context limit of "
                f"{context_limit} tokens for model '{llm.served_id}'."
            ),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "messages",
        },
    )


def adjust_max_tokens(messages: List[Dict], model_id: str, requested_max: int,
                      context_limit: Optional[int] = None,
                      tools: Optional[List[Dict[str, Any]]] = None) -> int:
    """Adjust max_tokens based on context limit"""
    context_limit = resolve_context_limit(model_id, context_limit)

    input_tokens = estimate_input_tokens(messages, tools)

    available = context_limit - input_tokens - 100
    if available < 100:
        available = 100

    result = min(requested_max, available)

    # NVIDIA API limits max_tokens to 1024
    if model_id in MODELS_WITHOUT_SYSTEM_ROLE:
        result = min(result, 1024)

    return result


def clean_response(result: Dict) -> Dict:
    """Clean response for OpenAI compatibility"""
    usage = _clean_usage(result.get("usage"))

    cleaned = {
        "id": result.get("id", ""),
        "object": result.get("object", "chat.completion"),
        "model": result.get("model", ""),
        "choices": [],
        "usage": usage
    }

    for choice in result.get("choices", []):
        cleaned_choice = {
            "index": choice.get("index", 0),
            "finish_reason": choice.get("finish_reason", "stop")
        }
        if "message" in choice:
            msg = choice["message"]
            cleaned_choice["message"] = {
                "role": msg.get("role", "assistant"),
                "content": msg.get("content")
            }
            if msg.get("reasoning_content") is not None:
                cleaned_choice["message"]["reasoning_content"] = msg["reasoning_content"]
            if msg.get("tool_calls") is not None:
                cleaned_choice["message"]["tool_calls"] = msg["tool_calls"]
            if msg.get("function_call") is not None:
                cleaned_choice["message"]["function_call"] = msg["function_call"]
        cleaned["choices"].append(cleaned_choice)

    return cleaned


def _message_has_tool_calls(message: Optional[Dict[str, Any]]) -> bool:
    return bool(message and (message.get("tool_calls") or message.get("function_call")))


def _delta_has_tool_calls(delta: Optional[Dict[str, Any]]) -> bool:
    return bool(delta and (delta.get("tool_calls") or delta.get("function_call")))


def _clean_usage_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            cleaned_item = _clean_usage_value(item)
            if cleaned_item is not None:
                cleaned[key] = cleaned_item
        return cleaned
    if isinstance(value, list):
        cleaned = []
        for item in value:
            cleaned_item = _clean_usage_value(item)
            if cleaned_item is not None:
                cleaned.append(cleaned_item)
        return cleaned
    if value is None:
        return None
    return value


def _clean_usage(usage_raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not usage_raw:
        return {}
    if not isinstance(usage_raw, dict):
        return {}
    cleaned_usage = _clean_usage_value(usage_raw)
    return cleaned_usage if isinstance(cleaned_usage, dict) else {}


def _merge_stream_options(stream_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(stream_options or {})
    merged.setdefault("include_usage", True)
    return merged


def stamp_served_model(line: str, served_id: str) -> str:
    """Set `model` on one SSE data line so every chunk names the model that served it."""
    if not line.startswith("data: ") or line.strip() == "data: [DONE]":
        return line
    try:
        data = json.loads(line[6:])
    except ValueError:
        return line
    if not isinstance(data, dict) or "error" in data:
        return line
    data["model"] = served_id
    return f"data: {json.dumps(data)}"


def clean_streaming_chunk(chunk: Dict) -> Optional[Dict]:
    """Clean streaming chunk for OpenAI compatibility"""
    choices = chunk.get("choices", [])
    usage = _clean_usage(chunk.get("usage"))
    if not choices and not usage:
        return None

    cleaned = {
        "id": chunk.get("id", ""),
        "object": chunk.get("object", "chat.completion.chunk"),
        "choices": []
    }
    if "model" in chunk:
        cleaned["model"] = chunk["model"]
    if usage:
        cleaned["usage"] = usage

    for choice in choices:
        finish_reason = choice.get("finish_reason")
        cleaned_choice = {
            "index": choice.get("index", 0),
            "finish_reason": finish_reason
        }

        if "delta" in choice:
            delta = choice["delta"]
            if finish_reason == "stop":
                cleaned_choice["delta"] = {}
            else:
                cleaned_delta = {}
                if "role" in delta:
                    cleaned_delta["role"] = delta["role"]
                if "content" in delta:
                    cleaned_delta["content"] = delta["content"]
                if "reasoning_content" in delta:
                    cleaned_delta["reasoning_content"] = delta["reasoning_content"]
                if "tool_calls" in delta:
                    cleaned_delta["tool_calls"] = delta["tool_calls"]
                if "function_call" in delta:
                    cleaned_delta["function_call"] = delta["function_call"]
                cleaned_choice["delta"] = cleaned_delta
        else:
            cleaned_choice["delta"] = {}

        cleaned["choices"].append(cleaned_choice)

    return cleaned


LOCAL_PROVIDER_HINTS = {
    "sglang",
    "vllm",
    "llama.cpp",
    "llama_cpp",
    "lmstudio",
    "lm_studio",
    "huggingface_cli",
}


def _is_local_base_url(base_url: str) -> bool:
    if not base_url:
        return False
    lower = base_url.lower()
    return (
        "localhost" in lower
        or "127.0.0.1" in lower
        or lower.startswith("http://0.0.0.0")
    )


def _resolve_auth_mode(provider: str, base_url: str, auth_mode: str = "auto", local: Optional[bool] = None) -> str:
    mode = (auth_mode or "auto").strip().lower()
    if mode in ("none", "bearer"):
        return mode

    provider_norm = (provider or "").strip().lower()
    is_local = bool(local) if local is not None else _is_local_base_url(base_url)
    if provider_norm in LOCAL_PROVIDER_HINTS or is_local:
        return "none"
    return "bearer"


def _build_chat_url(base_url: str, chat_path: str) -> str:
    path = (chat_path or "/chat/completions").strip()
    if not path.startswith("/"):
        path = "/" + path
    return f"{(base_url or '').rstrip('/')}{path}"


_CONTEXT_OVERFLOW_RE = re.compile(
    r"context[_ -]+length[_ -]+exceeded|exceeds the context window", re.IGNORECASE)


def _error_source(value: Any):
    """Extract an upstream error object or unstructured message from one error source."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}, value
    if not isinstance(value, Mapping):
        return {}, None

    if "error" in value:
        upstream_error = value["error"]
        if isinstance(upstream_error, Mapping):
            return upstream_error, None
        if isinstance(upstream_error, str):
            return {"message": upstream_error}, None
    if "message" in value:
        return value, None
    if "detail" in value:
        return _error_source(value["detail"])
    return {}, None


def _backend_error_response_data(error: Exception):
    """Return an OpenAI-compatible error body and an available upstream error status."""
    sources = [getattr(error, "body", None), getattr(error, "detail", None), getattr(error, "error", None)]
    response = getattr(error, "response", None)
    response_json = getattr(response, "json", None)
    if callable(response_json):
        try:
            sources.append(response_json())
        except Exception:
            pass

    metadata = {}
    raw_message = None
    for source in sources:
        source_metadata, source_message = _error_source(source)
        for field in ("message", "type", "code", "param"):
            if metadata.get(field) is None and source_metadata.get(field) is not None:
                metadata[field] = source_metadata[field]
        if raw_message is None and source_message is not None:
            raw_message = source_message

    message = metadata.get("message")
    if not isinstance(message, str):
        message = raw_message
    if not isinstance(message, str):
        message = getattr(error, "message", None)
    if not isinstance(message, str):
        message = str(error)
    if not message:
        message = type(error).__name__

    code = metadata.get("code")
    if not isinstance(code, str) or not code:
        code = getattr(error, "code", None)
    if not isinstance(code, str) or not code:
        code = None
    is_context_overflow = (code is not None and code.lower() == "context_length_exceeded") or bool(
        _CONTEXT_OVERFLOW_RE.search(message))
    if is_context_overflow and code is None:
        code = "context_length_exceeded"

    status = getattr(error, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool) or not 400 <= status < 600:
        status = getattr(response, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool) or not 400 <= status < 600:
        status = 400 if is_context_overflow else 502

    error_type = metadata.get("type")
    if not isinstance(error_type, str):
        error_type = getattr(error, "type", None)
    if not isinstance(error_type, str) or not error_type:
        error_type = "invalid_request_error" if status < 500 else "api_error"
    if is_context_overflow and status < 500:
        error_type = "invalid_request_error"

    param = metadata.get("param")
    if not isinstance(param, str):
        param = getattr(error, "param", None)
    if not isinstance(param, str):
        param = None

    return status, {"error": {"message": message, "type": error_type, "code": code, "param": param}}


def _invalidate_routed_decision_for_status(
    router: OpenClawRouter,
    auto_routed: bool,
    query: str,
    user: Optional[str],
    selected_model: Optional[str],
    decision_identity: Optional[int],
    status: int,
) -> None:
    if (
        auto_routed
        and selected_model is not None
        and decision_identity is not None
        and 400 <= status < 500
        and status != 429
    ):
        router.invalidate_cached_decision(query, user, selected_model, decision_identity)


def _backend_error_response(
    error: Exception, on_status: Optional[Callable[[int], None]] = None
) -> JSONResponse:
    status, body = _backend_error_response_data(error)
    if on_status is not None:
        on_status(status)
    return JSONResponse(content=body, status_code=status)


def _backend_stream_error_event(
    error: Exception, on_status: Optional[Callable[[int], None]] = None
) -> str:
    status, body = _backend_error_response_data(error)
    if on_status is not None:
        on_status(status)
    return f"data: {json.dumps(body)}\n\n"


def _stream_error_exception(chunk: str):
    """Convert an upstream SSE error object into the exception shape used by HTTP errors."""
    if not isinstance(chunk, str):
        return None
    payload = chunk[6:] if chunk.startswith("data: ") else chunk
    if '"error"' not in payload:
        return None
    try:
        data = json.loads(payload.strip())
    except (TypeError, ValueError):
        return None
    if not isinstance(data, Mapping) or "error" not in data:
        return None

    error = data["error"]
    detail = {"error": error if isinstance(error, (Mapping, str)) else {"message": str(error)}}
    status = data.get("status_code")
    if not isinstance(status, int) or isinstance(status, bool) or not 400 <= status < 600:
        status = error.get("status_code") if isinstance(error, Mapping) else None
    overflow_code = error.get("code") if isinstance(error, Mapping) else None
    if not isinstance(status, int) or isinstance(status, bool) or not 400 <= status < 600:
        status = 400 if _CONTEXT_OVERFLOW_RE.search(str(error)) or (
            isinstance(overflow_code, str) and overflow_code.lower() == "context_length_exceeded") else 502
    return HTTPException(status_code=status, detail=detail)


def _stream_chunk_is_useful(chunk: str) -> bool:
    """Recognize stream output that warrants committing the HTTP response."""
    if not isinstance(chunk, str):
        return False
    payload = chunk[6:] if chunk.startswith("data: ") else chunk
    if payload.strip() == "[DONE]":
        return True
    try:
        data = json.loads(payload.strip())
    except (TypeError, ValueError):
        return False
    if not isinstance(data, Mapping):
        return False

    choices = data.get("choices")
    if not isinstance(choices, list):
        return False

    def has_content(value: Any) -> bool:
        if isinstance(value, str):
            value = re.sub(r'^\[[\w\-\.]+\]\s*', '', value)
        return bool(value)

    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        if choice.get("finish_reason") is not None or has_content(choice.get("text")):
            return True
        delta = choice.get("delta")
        if isinstance(delta, Mapping) and any(
            key != "role" and has_content(value) for key, value in delta.items()
        ):
            return True
    return False


# ============================================================
# LLM Backend
# ============================================================

class LLMBackend:
    """LLM API caller"""

    def __init__(self, config: OpenClawConfig):
        self.config = config

    async def call(self, llm_name: str, messages: List[Dict], max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None, stream: bool = False,
                   tools: Optional[List[Dict[str, Any]]] = None,
                   tool_choice: Optional[Any] = None,
                   stream_options: Optional[Dict[str, Any]] = None,
                   extra_params: Optional[Dict[str, Any]] = None):
        """Call LLM API"""
        if llm_name not in self.config.llms:
            raise HTTPException(status_code=404, detail=f"LLM '{llm_name}' not found")

        llm_config = self.config.llms[llm_name]
        api_key = self.config.get_api_key(llm_config.provider, llm_config)
        if max_tokens is None:
            max_tokens = llm_config.max_tokens

        if llm_config.provider_type == "litellm":
            return await self._call_litellm(
                llm_config, messages, max_tokens, temperature, api_key, stream,
                tools, tool_choice, stream_options, extra_params,
            )

        if stream:
            return self._call_streaming(
                llm_config,
                messages,
                max_tokens,
                temperature,
                api_key,
                tools,
                tool_choice,
                stream_options,
                extra_params,
            )
        else:
            return await self._call_sync(
                llm_config, messages, max_tokens, temperature, api_key, tools, tool_choice, extra_params
            )

    async def _call_litellm(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                            temperature: Optional[float], api_key: Optional[str], stream: bool,
                            tools: Optional[List[Dict[str, Any]]] = None,
                            tool_choice: Optional[Any] = None,
                            stream_options: Optional[Dict[str, Any]] = None,
                            extra_params: Optional[Dict[str, Any]] = None):
        """Call through LiteLLM, for backends that are not Chat Completions servers.

        `model` is a LiteLLM model string. "openai/responses/<id>" bridges to the OpenAI
        Responses API, which some reasoning models require for function tools.
        """
        import litellm

        normalized = normalize_messages(messages, llm.model_id)
        kwargs: Dict[str, Any] = {
            "model": llm.model_id,
            "messages": normalized,
            "max_tokens": adjust_max_tokens(
                normalized, llm.model_id, max_tokens, llm.context_limit, tools=tools
            ),
            "api_base": llm.base_url,
            "timeout": llm.timeout,
            "stream": stream,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if stream:
            kwargs["stream_options"] = _merge_stream_options(stream_options)
        kwargs.update(extra_params or {})
        kwargs.update(llm.extra_body or {})

        if not stream:
            response = await litellm.acompletion(**kwargs)
            result = clean_response(response.model_dump(exclude_none=True))
            result["model"] = llm.served_id
            return result

        async def generate() -> AsyncGenerator:
            response = await litellm.acompletion(**kwargs)
            async for chunk in response:
                data = chunk.model_dump(exclude_none=True)
                data["model"] = llm.served_id
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return generate()

    async def _call_sync(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                         temperature: Optional[float], api_key: Optional[str],
                         tools: Optional[List[Dict[str, Any]]] = None,
                         tool_choice: Optional[Any] = None,
                         extra_params: Optional[Dict[str, Any]] = None) -> Dict:
        """Synchronous API call"""
        normalized = normalize_messages(messages, llm.model_id)
        adjusted_max = adjust_max_tokens(
            normalized, llm.model_id, max_tokens, llm.context_limit, tools=tools
        )
        auth_mode = _resolve_auth_mode(llm.provider, llm.base_url, llm.auth_mode, llm.local)
        chat_url = _build_chat_url(llm.base_url, llm.chat_path)


        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if auth_mode == "bearer" and api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": normalized,
                llm.max_tokens_param or "max_tokens": adjusted_max,
            }
            if temperature is not None:
                body["temperature"] = temperature
            if tools is not None:
                body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
            body.update(extra_params or {})
            body.update(llm.extra_body or {})

            resp = await client.post(
                chat_url,
                headers=headers,
                json=body,
                timeout=llm.timeout
            )

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)

            result = clean_response(resp.json())
            result["model"] = llm.served_id
            return result

    async def _call_streaming(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                          temperature: Optional[float], api_key: Optional[str],
                          tools: Optional[List[Dict[str, Any]]] = None,
                          tool_choice: Optional[Any] = None,
                          stream_options: Optional[Dict[str, Any]] = None,
                          extra_params: Optional[Dict[str, Any]] = None) -> AsyncGenerator:
        """Streaming API call"""
        normalized = normalize_messages(messages, llm.model_id)
        adjusted_max = adjust_max_tokens(
            normalized, llm.model_id, max_tokens, llm.context_limit, tools=tools
        )
        auth_mode = _resolve_auth_mode(llm.provider, llm.base_url, llm.auth_mode, llm.local)
        chat_url = _build_chat_url(llm.base_url, llm.chat_path)

        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if auth_mode == "bearer" and api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": normalized,
                llm.max_tokens_param or "max_tokens": adjusted_max,
                "stream": True,
                "stream_options": _merge_stream_options(stream_options),
            }
            if temperature is not None:
                body["temperature"] = temperature
            if tools is not None:
                body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
            body.update(extra_params or {})
            body.update(llm.extra_body or {})

            async with client.stream(
                "POST",
                chat_url,
                headers=headers,
                json=body,
                timeout=llm.timeout
            ) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    error_text = error.decode("utf-8", errors="replace")
                    print(f"[Backend Streaming] Error {resp.status_code}: {error_text[:200]}")
                    raise HTTPException(status_code=resp.status_code, detail=error_text)

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        yield stamp_served_model(line, llm.served_id) + "\n\n"


# ============================================================
# FastAPI App Factory
# ============================================================

def create_app(config: OpenClawConfig = None, config_path: str = None) -> FastAPI:
    """Create FastAPI application"""
    if config is None and config_path:
        config = OpenClawConfig.from_yaml(config_path)
    elif config is None:
        config = OpenClawConfig()

    app = FastAPI(
        title="OpenClaw Router",
        description="OpenAI-compatible API with intelligent LLM routing",
        version="1.0.0"
    )

    # Initialize components
    router = OpenClawRouter(config)
    backend = LLMBackend(config)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "strategy": config.router.strategy,
            "llms": list(config.llms.keys())
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "description": llm.description}
                for name, llm in config.llms.items()
            ] + [{"id": "auto", "object": "model", "description": "Auto router"}]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        print(f"============\n")
        messages = request_messages(request)

        # Extract user query for routing (with optional media understanding)
        user_query = ""
        media_description = None

        # Find and process the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            raw_content = messages[last_user_idx]["content"]

            # Process multimodal content if media is enabled
            # Supports both OpenAI format (list) and OpenClaw format (string with [media attached:...])
            if config.media.enabled:
                # Use together API key as fallback
                together_key = config.api_keys.get("together")
                processed_text, media_desc = await process_multimodal_content(
                    raw_content, config.media, fallback_key=together_key
                )
                user_query = processed_text
                media_description = media_desc
                if media_desc:
                    print(f"[Media] Processed: {media_desc[:80]}...")
                    # IMPORTANT: Replace the message content with processed text
                    # so LLM sees the image description instead of [media attached: ...]
                    messages[last_user_idx]["content"] = processed_text
            else:
                user_query = normalize_content(raw_content)

            user_query = _build_routing_query(messages, last_user_idx, user_query, router)

        if not user_query:
            user_query = "general query"

        # Select model
        available_models = list(config.llms.keys())
        request.model = resolve_requested_model(config, request.model)
        auto_routed = request.model == "auto" or request.model not in available_models
        decision_identity = None
        if auto_routed:
            selected_model = await router.select_model(user_query, user=request.user)
            decision_identity = router.decision_cache_identity(
                user_query, request.user, selected_model
            )
            # ASCII-only log to avoid Windows GBK UnicodeEncodeError.
            # print(f"[Router] Query: '{user_query[:50]}...' -> {selected_model}")
            print(f"[Router] Query: '{user_query}' -> {selected_model}")
        else:
            selected_model = request.model
            print(f"[Specified] Query: '{user_query}' -> {selected_model}")

        def invalidate_route_for_status(status):
            _invalidate_routed_decision_for_status(
                router, auto_routed, user_query, request.user, selected_model,
                decision_identity, status,
            )

        def routed_error_response(error):
            return _backend_error_response(error, on_status=invalidate_route_for_status)

        def routed_stream_error_event(error):
            return _backend_stream_error_event(error, on_status=invalidate_route_for_status)

        selected_llm = config.llms.get(selected_model)
        if selected_llm:
            context_limit = resolve_context_limit(selected_llm.model_id, selected_llm.context_limit)
            normalized_messages = normalize_messages(messages, selected_llm.model_id)
            estimated_input_tokens = estimate_input_tokens(normalized_messages, request.tools)
            if estimated_input_tokens > context_limit:
                return routed_error_response(context_length_error(selected_llm, context_limit))

        # Handle streaming
        if request.stream:
            try:
                stream_gen = await backend.call(
                    selected_model, messages, request.max_tokens,
                    request.temperature, stream=True,
                    tools=request.tools,
                    tool_choice=request.tool_choice,
                    stream_options=request.stream_options,
                    extra_params=passthrough_params(request),
                )
                stream_iterator = stream_gen.__aiter__()
                try:
                    first_chunk = await stream_iterator.__anext__()
                except StopAsyncIteration:
                    first_chunk = None
            except Exception as error:
                return routed_error_response(error)

            first_chunk_error = _stream_error_exception(first_chunk)
            if first_chunk_error is not None:
                return routed_error_response(first_chunk_error)

            async def transformed_stream():
                prefix_sent = False
                content_buffer = ""
                buffered_chunks = []

                async def stream_with_first_chunk():
                    if first_chunk is not None:
                        yield first_chunk
                    async for chunk in stream_iterator:
                        yield chunk

                def flush_buffered_prefix() -> Optional[str]:
                    nonlocal prefix_sent, content_buffer, buffered_chunks
                    if not buffered_chunks or prefix_sent:
                        return None

                    content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                    first = buffered_chunks[0]
                    try:
                        first_json = first[6:] if first.startswith("data: ") else first
                        first_data = json.loads(first_json.strip())
                        if first_data.get("choices") and first_data["choices"][0].get("delta"):
                            first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                            prefix_sent = True
                            buffered_chunks = []
                            return f"data: {json.dumps(first_data)}\n\n"
                    except:
                        pass
                    return None

                try:
                    prefix_disabled = False
                    async for chunk in stream_with_first_chunk():
                        stream_error = _stream_error_exception(chunk)
                        if stream_error is not None:
                            raise stream_error

                        if not config.show_model_prefix:
                            yield chunk
                            continue

                        if prefix_disabled:
                            if "[DONE]" in chunk:
                                yield chunk
                                continue
                            try:
                                json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                                data = json.loads(json_str.strip())
                                cleaned = clean_streaming_chunk(data)
                                if cleaned:
                                    yield f"data: {json.dumps(cleaned)}\n\n"
                                    continue
                            except:
                                pass
                            yield chunk
                            continue

                        # Add model prefix to first content chunk
                        if "[DONE]" in chunk:
                            # Flush buffer before DONE
                            if buffered_chunks and not prefix_sent:
                                flushed_chunk = flush_buffered_prefix()
                                if flushed_chunk:
                                    yield flushed_chunk
                            yield chunk
                        else:
                            try:
                                json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                                data = json.loads(json_str.strip())
                                cleaned = clean_streaming_chunk(data)

                                if cleaned:
                                    if cleaned.get("usage") and not cleaned.get("choices"):
                                        if buffered_chunks and not prefix_sent:
                                            flushed_chunk = flush_buffered_prefix()
                                            if flushed_chunk:
                                                yield flushed_chunk
                                        yield f"data: {json.dumps(cleaned)}\n\n"
                                        continue

                                    choices = cleaned.get("choices", [])
                                    if choices and "delta" in choices[0]:
                                        delta = choices[0]["delta"]

                                        if _delta_has_tool_calls(delta):
                                            if buffered_chunks and not prefix_sent:
                                                for buffered_chunk in buffered_chunks:
                                                    try:
                                                        buffered_json = buffered_chunk[6:] if buffered_chunk.startswith("data: ") else buffered_chunk
                                                        buffered_data = json.loads(buffered_json.strip())
                                                        buffered_cleaned = clean_streaming_chunk(buffered_data)
                                                        if buffered_cleaned:
                                                            yield f"data: {json.dumps(buffered_cleaned)}\n\n"
                                                        else:
                                                            yield buffered_chunk
                                                    except:
                                                        yield buffered_chunk
                                                buffered_chunks = []
                                            prefix_disabled = True
                                            yield f"data: {json.dumps(cleaned)}\n\n"
                                            continue

                                        content = delta.get("content", "")

                                        if not prefix_sent:
                                            content_buffer += content
                                            buffered_chunks.append(chunk)

                                            if len(content_buffer) > 30 or (content_buffer and not content_buffer.startswith("[")):
                                                content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                                first = buffered_chunks[0]
                                                first_data = json.loads(first[6:] if first.startswith("data: ") else first)
                                                if first_data.get("choices") and first_data["choices"][0].get("delta"):
                                                    first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                                    yield f"data: {json.dumps(first_data)}\n\n"
                                                    prefix_sent = True
                                                    buffered_chunks = []
                                        else:
                                            yield f"data: {json.dumps(cleaned)}\n\n"
                                    else:
                                        if prefix_sent:
                                            yield f"data: {json.dumps(cleaned)}\n\n"
                            except:
                                yield chunk
                except Exception as e:
                    print(f"[Stream Error] {type(e).__name__}: {e}")
                    raise

            async def generate():
                buffered_preludes = []
                useful_output_seen = False
                try:
                    async for chunk in transformed_stream():
                        if useful_output_seen:
                            yield chunk
                            continue

                        if _stream_chunk_is_useful(chunk):
                            useful_output_seen = True
                            for prelude in buffered_preludes:
                                yield prelude
                            buffered_preludes.clear()
                            yield chunk
                        else:
                            buffered_preludes.append(chunk)
                except Exception as error:
                    if not useful_output_seen:
                        raise
                    yield routed_stream_error_event(error)
                    return

                for prelude in buffered_preludes:
                    yield prelude

            stream_output = generate().__aiter__()
            try:
                first_output = await stream_output.__anext__()
            except StopAsyncIteration:
                first_output = None
            except Exception as error:
                return routed_error_response(error)

            async def response_stream():
                if first_output is not None:
                    yield first_output
                async for chunk in stream_output:
                    yield chunk

            return StreamingResponse(response_stream(), media_type="text/event-stream")

        else:
            try:
                result = await backend.call(
                    selected_model, messages, request.max_tokens,
                    request.temperature, stream=False,
                    tools=request.tools, tool_choice=request.tool_choice,
                    extra_params=passthrough_params(request),
                )
            except Exception as error:
                return routed_error_response(error)

            # Add model prefix
            if config.show_model_prefix and result.get("choices"):
                message = result["choices"][0].get("message", {})
                content = message.get("content")
                if content and not _message_has_tool_calls(message):
                    # Remove any existing prefix
                    content = re.sub(r'^\[[\w\-\.]+\]\s*', '', content)
                    message["content"] = f"[{selected_model}] {content}"

            return result

    @app.get("/")
    async def root():
        return {
            "name": "OpenClaw Router",
            "version": "1.0.0",
            "strategy": config.router.strategy,
            "llms": list(config.llms.keys()),
            "endpoints": {
                "chat": "POST /v1/chat/completions",
                "models": "GET /v1/models",
                "health": "GET /health"
            }
        }

    @app.get("/routers")
    async def list_routers():
        """List available routing strategies"""
        return {
            "available_routers": router.get_available_routers(),
            "current": config.router.strategy
        }

    @app.websocket("/v1/chat/ws")
    async def chat_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time streaming"""
        await websocket.accept()
        auto_routed = False
        user_query = ""
        routed_user = None
        selected_model = None
        decision_identity = None
        try:
            # Receive request
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            routed_user = request.user
            messages = request_messages(request)

            # Extract user query for routing
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                raw_content = messages[last_user_idx]["content"]
                if config.media.enabled:
                    together_key = config.api_keys.get("together")
                    processed_text, _ = await process_multimodal_content(
                        raw_content, config.media, fallback_key=together_key
                    )
                    user_query = processed_text
                    messages[last_user_idx]["content"] = processed_text
                else:
                    user_query = normalize_content(raw_content)

                user_query = _build_routing_query(messages, last_user_idx, user_query, router)

            if not user_query:
                user_query = "general query"

            # Select model
            available_models = list(config.llms.keys())
            request.model = resolve_requested_model(config, request.model)
            auto_routed = request.model == "auto" or request.model not in available_models
            if auto_routed:
                selected_model = await router.select_model(user_query, user=request.user)
                decision_identity = router.decision_cache_identity(
                    user_query, request.user, selected_model
                )
                _safe_log(f"[WS Router] Query: '{user_query[:50]}...' -> {selected_model}")
            else:
                selected_model = request.model

            selected_llm = config.llms.get(selected_model)
            if selected_llm:
                context_limit = resolve_context_limit(selected_llm.model_id, selected_llm.context_limit)
                normalized_messages = normalize_messages(messages, selected_llm.model_id)
                estimated_input_tokens = estimate_input_tokens(normalized_messages, request.tools)
                if estimated_input_tokens > context_limit:
                    error_status, error_body = _backend_error_response_data(
                        context_length_error(selected_llm, context_limit)
                    )
                    _invalidate_routed_decision_for_status(
                        router, auto_routed, user_query, routed_user, selected_model,
                        decision_identity, error_status,
                    )
                    await websocket.send_json(error_body)
                    return

            # Call LLM backend in streaming mode
            prefix_sent = False
            content_buffer = ""
            buffered_chunks = []

            stream_gen = await backend.call(
                selected_model, messages, request.max_tokens,
                request.temperature,
                stream=True,
                tools=request.tools,
                tool_choice=request.tool_choice,
                stream_options=request.stream_options,
                extra_params=passthrough_params(request),
            )

            async for chunk in stream_gen:
                stream_error = _stream_error_exception(chunk)
                if stream_error is not None:
                    error_status, _ = _backend_error_response_data(stream_error)
                    _invalidate_routed_decision_for_status(
                        router, auto_routed, user_query, routed_user, selected_model,
                        decision_identity, error_status,
                    )

                if not config.show_model_prefix:
                    await websocket.send_text(chunk)
                    continue

                if "[DONE]" in chunk:
                    if buffered_chunks and not prefix_sent:
                        content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                        first = buffered_chunks[0]
                        try:
                            data_chunk = json.loads(first[6:]) if first.startswith("data: ") else {}
                            if data_chunk.get("choices") and data_chunk["choices"][0].get("delta"):
                                data_chunk["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                await websocket.send_text(f"data: {json.dumps(data_chunk)}\n\n")
                        except:
                            pass
                    await websocket.send_text(chunk)
                else:
                    try:
                        json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                        data_chunk = json.loads(json_str.strip())
                        cleaned = clean_streaming_chunk(data_chunk)

                        if cleaned:
                            if cleaned.get("usage") and not cleaned.get("choices"):
                                if buffered_chunks and not prefix_sent:
                                    content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                    first = buffered_chunks[0]
                                    try:
                                        first_data = json.loads(first[6:] if first.startswith("data: ") else first)
                                        if first_data.get("choices") and first_data["choices"][0].get("delta"):
                                            first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                            await websocket.send_text(f"data: {json.dumps(first_data)}\n\n")
                                            prefix_sent = True
                                            buffered_chunks = []
                                    except:
                                        pass
                                await websocket.send_json(cleaned)
                                continue

                            choices = cleaned.get("choices", [])
                            if choices and "delta" in choices[0]:
                                content = choices[0]["delta"].get("content", "")

                                if not prefix_sent:
                                    content_buffer += content
                                    buffered_chunks.append(chunk)

                                    if len(content_buffer) > 30 or (content_buffer and not content_buffer.startswith("[")):
                                        content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                        first = buffered_chunks[0]
                                        first_data = json.loads(first[6:] if first.startswith("data: ") else first)
                                        if first_data.get("choices") and first_data["choices"][0].get("delta"):
                                            first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                            await websocket.send_text(f"data: {json.dumps(first_data)}\n\n")
                                            prefix_sent = True
                                            buffered_chunks = []
                                else:
                                    await websocket.send_json(cleaned)
                            else:
                                if prefix_sent:
                                    await websocket.send_json(cleaned)
                    except:
                        await websocket.send_text(chunk)

        except WebSocketDisconnect:
            _safe_log("[WS] Client disconnected")
        except Exception as e:
            _safe_log(f"[WS Error] {type(e).__name__}: {e}")
            try:
                if auto_routed and selected_model is not None:
                    error_status, _ = _backend_error_response_data(e)
                    _invalidate_routed_decision_for_status(
                        router, auto_routed, user_query, routed_user, selected_model,
                        decision_identity, error_status,
                    )
                await websocket.send_json({"error": str(e)})
            except:
                pass
        finally:
            try:
                await websocket.close()
            except:
                pass

    return app


def run_server(app: FastAPI = None, config_path: str = None, host: Optional[str] = None,
               port: Optional[int] = None):
    """Run the server. Unset host/port come from the config's serve section, else 0.0.0.0:8000."""
    if app is None:
        config = OpenClawConfig.from_yaml(config_path) if config_path else OpenClawConfig()
        app = create_app(config=config)
        host = config.host if host is None else host
        port = config.port if port is None else port
    host = "0.0.0.0" if host is None else host
    port = 8000 if port is None else port

    print(f"""
============================================================
  OpenClaw Router
============================================================
  Server: http://{host}:{port}
  API:    http://{host}:{port}/v1/chat/completions
  Health: http://{host}:{port}/health
============================================================
""")

    uvicorn.run(app, host=host, port=port)


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenClaw Router Server")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--host", default=None, help="Host to bind (default: config serve.host, else 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=None,
                        help="Port to bind (default: config serve.port, else 8000)")
    args = parser.parse_args()

    run_server(config_path=args.config, host=args.host, port=args.port)
