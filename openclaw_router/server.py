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
import os
import re
import sys
from typing import AsyncGenerator, Optional, Dict, Any, List

# Check dependencies
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
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


def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 4 chars = 1 token)"""
    return len(text) // 4


def adjust_max_tokens(messages: List[Dict], model_id: str, requested_max: int,
                      context_limit: Optional[int] = None) -> int:
    """Adjust max_tokens based on context limit"""
    if not context_limit:
        context_limit = MODEL_CONTEXT_LIMITS.get(model_id, 32768)

    input_text = " ".join(m.get("content", "") for m in messages)
    input_tokens = estimate_tokens(input_text)

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
            "max_tokens": adjust_max_tokens(normalized, llm.model_id, max_tokens, llm.context_limit),
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
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as error:
                status = getattr(error, "status_code", None) or 502
                raise HTTPException(status_code=status, detail=str(error)[:500])
            result = clean_response(response.model_dump(exclude_none=True))
            result["model"] = llm.served_id
            return result

        async def generate() -> AsyncGenerator:
            try:
                response = await litellm.acompletion(**kwargs)
                async for chunk in response:
                    data = chunk.model_dump(exclude_none=True)
                    data["model"] = llm.served_id
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as error:
                print(f"[Backend LiteLLM] Error: {str(error)[:200]}")
                yield f'data: {json.dumps({"error": str(error)[:200]})}\n\n'

        return generate()

    async def _call_sync(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                         temperature: Optional[float], api_key: Optional[str],
                         tools: Optional[List[Dict[str, Any]]] = None,
                         tool_choice: Optional[Any] = None,
                         extra_params: Optional[Dict[str, Any]] = None) -> Dict:
        """Synchronous API call"""
        normalized = normalize_messages(messages, llm.model_id)
        adjusted_max = adjust_max_tokens(normalized, llm.model_id, max_tokens, llm.context_limit)
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
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

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
        adjusted_max = adjust_max_tokens(normalized, llm.model_id, max_tokens, llm.context_limit)
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
                    print(f"[Backend Streaming] Error {resp.status_code}: {error.decode()[:200]}")
                    yield f'data: {json.dumps({"error": error.decode()[:200]})}\n\n'
                    return

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
                user_query = processed_text[:500]
                media_description = media_desc
                if media_desc:
                    print(f"[Media] Processed: {media_desc[:80]}...")
                    # IMPORTANT: Replace the message content with processed text
                    # so LLM sees the image description instead of [media attached: ...]
                    messages[last_user_idx]["content"] = processed_text
            else:
                user_query = normalize_content(raw_content)[:500]

        if not user_query:
            user_query = "general query"

        # Select model
        available_models = list(config.llms.keys())
        request.model = resolve_requested_model(config, request.model)
        if request.model == "auto" or request.model not in available_models:
            selected_model = await router.select_model(user_query, user=request.user)
            # ASCII-only log to avoid Windows GBK UnicodeEncodeError.
            # print(f"[Router] Query: '{user_query[:50]}...' -> {selected_model}")
            print(f"[Router] Query: '{user_query}' -> {selected_model}")
        else:
            selected_model = request.model
            print(f"[Specified] Query: '{user_query}' -> {selected_model}")

        # Handle streaming
        if request.stream:
            async def generate():
                prefix_sent = False
                content_buffer = ""
                buffered_chunks = []

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

                    stream_gen = await backend.call(
                        selected_model, messages, request.max_tokens,
                        request.temperature, stream=True,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                        stream_options=request.stream_options,
                        extra_params=passthrough_params(request),
                    )
                    async for chunk in stream_gen:
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
                    yield f'data: {json.dumps({"error": str(e)})}\n\n'

            return StreamingResponse(generate(), media_type="text/event-stream")

        else:
            result = await backend.call(
                selected_model, messages, request.max_tokens,
                request.temperature, stream=False,
                tools=request.tools, tool_choice=request.tool_choice,
                extra_params=passthrough_params(request),
            )

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
        try:
            # Receive request
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            messages = request_messages(request)

            # Extract user query for routing
            user_query = ""
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
                    user_query = processed_text[:500]
                    messages[last_user_idx]["content"] = processed_text
                else:
                    user_query = normalize_content(raw_content)[:500]

            if not user_query:
                user_query = "general query"

            # Select model
            available_models = list(config.llms.keys())
            request.model = resolve_requested_model(config, request.model)
            if request.model == "auto" or request.model not in available_models:
                selected_model = await router.select_model(user_query, user=request.user)
                _safe_log(f"[WS Router] Query: '{user_query[:50]}...' -> {selected_model}")
            else:
                selected_model = request.model

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
