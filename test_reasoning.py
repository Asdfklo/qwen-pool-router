from __future__ import annotations

import pytest

from hermes_router import (
    DynamicThinkingConfig,
    ReasoningConfig,
    RouterConfig,
    reasoning_backend_filter,
    rewrite_payload,
)
from reasoning import (
    extract_reasoning_hint,
    normalize_reasoning_effort,
    parse_model_reasoning_suffix,
    resolve_reasoning_request,
    strip_client_reasoning_fields,
)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"reasoningEffort": "high"}, "high"),
        ({"reasoning_effort": "xhigh"}, "xhigh"),
        ({"reasoning": {"effort": "low"}}, "low"),
        ({"reasoning": {"level": "medium"}}, "medium"),
        ({"providerOptions": {"openai": {"reasoningEffort": "medium"}}}, "medium"),
        ({"providerOptions": {"openai": {"reasoning_effort": "medium"}}}, "medium"),
        ({"providerOptions": {"nyx": {"reasoningEffort": "none"}}}, "none"),
        ({"providerOptions": {"nyx": {"reasoning_effort": "none"}}}, "none"),
        ({"providerOptions": {"openaiCompatible": {"reasoningEffort": "high"}}}, "high"),
        ({"providerOptions": {"openaiCompatible": {"reasoning_effort": "high"}}}, "high"),
        ({"providerOptions": {"openai-compatible": {"reasoningEffort": "high"}}}, "high"),
        ({"providerOptions": {"openai-compatible": {"reasoning_effort": "high"}}}, "high"),
        ({"extra_body": {"reasoning_effort": "minimal"}}, "minimal"),
    ],
)
def test_all_reasoning_shapes(payload: dict, expected: str) -> None:
    raw, source = extract_reasoning_hint(payload) or (None, "")
    assert normalize_reasoning_effort(raw, source=source).effort == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("fast", "none"),
        ("deep", "high"),
        ("extreme", "xhigh"),
        (" HIGH ", "high"),
        (False, "none"),
        (0, "none"),
    ],
)
def test_aliases(value: object, expected: str) -> None:
    assert normalize_reasoning_effort(value).effort == expected


def test_missing_and_invalid_defaults() -> None:
    assert normalize_reasoning_effort(None, "low").effort == "low"
    invalid = normalize_reasoning_effort("nonsense", "minimal")
    assert invalid.effort == "minimal"
    assert invalid.used_fallback is True
    assert invalid.warning


@pytest.mark.parametrize(
    ("model", "base", "effort"),
    [
        ("qwen-3.6-35b:high", "qwen-3.6-35b", "high"),
        ("qwen-3.6-35b:none", "qwen-3.6-35b", "none"),
        ("qwen-3.6-35b:deep", "qwen-3.6-35b", "high"),
        ("qwen-3.6-35b", "qwen-3.6-35b", None),
        ("vendor:model", "vendor:model", None),
    ],
)
def test_model_suffix(model: str, base: str, effort: str | None) -> None:
    assert parse_model_reasoning_suffix(model) == (base, effort)


def test_precedence_and_legacy_auto() -> None:
    suffix = resolve_reasoning_request(
        {"reasoningEffort": "none"},
        "qwen-3.6-35b:high",
        model_default="low",
        global_default="minimal",
    )
    assert (suffix.effort, suffix.source) == ("high", "model_suffix")

    body = resolve_reasoning_request(
        {"reasoningEffort": "low"},
        "qwen-3.6-35b",
        model_default="high",
        global_default="minimal",
    )
    assert (body.effort, body.source) == ("low", "reasoningEffort")

    model_default = resolve_reasoning_request({}, "qwen-3.6-35b", model_default="high", global_default="low")
    assert (model_default.effort, model_default.source) == ("high", "model_default")

    global_default = resolve_reasoning_request({}, "qwen-3.6-35b", global_default="low")
    assert (global_default.effort, global_default.source) == ("low", "global_default")

    auto = resolve_reasoning_request({}, "qwen-3.6-35b")
    assert (auto.effort, auto.source) == (None, "dynamic_rules")


def test_stripping_is_surgical() -> None:
    payload = {
        "reasoningEffort": "high",
        "reasoning": {"effort": "low", "summary": "auto"},
        "providerOptions": {
            "openai": {"reasoningEffort": "high", "textVerbosity": "low"},
            "other": {"keep": True},
        },
        "extra_body": {"reasoning_effort": "minimal", "keep": 1},
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function"}],
        "temperature": 0.4,
    }
    stripped = strip_client_reasoning_fields(payload)
    assert "reasoningEffort" not in stripped
    assert stripped["reasoning"] == {"summary": "auto"}
    assert stripped["providerOptions"]["openai"] == {"textVerbosity": "low"}
    assert stripped["providerOptions"]["other"] == {"keep": True}
    assert stripped["extra_body"] == {"keep": 1}
    assert stripped["messages"] == payload["messages"]
    assert stripped["tools"] == payload["tools"]

    empty = strip_client_reasoning_fields(
        {
            "providerOptions": {"openai": {"reasoningEffort": "high"}},
            "extra_body": {"reasoning_effort": "high"},
        }
    )
    assert empty == {}


def test_rewrite_maps_effort_to_qwen_controls() -> None:
    payload = {
        "model": "qwen-3.6-35b:high",
        "reasoningEffort": "high",
        "messages": [{"role": "user", "content": "solve this"}],
        "tools": [{"type": "function"}],
        "max_tokens": 50,
        "stream": True,
    }
    rewritten = rewrite_payload(
        payload,
        "Qwen3.6-35B-A3B",
        dynamic_thinking=DynamicThinkingConfig(),
        reasoning_effort="high",
        reasoning=ReasoningConfig(expose_reasoning_models=True),
    )
    assert rewritten["model"] == "Qwen3.6-35B-A3B"
    assert rewritten["chat_template_kwargs"]["enable_thinking"] is True
    assert rewritten["reasoning_budget"] == 8192
    assert rewritten["stream_options"]["include_usage"] is True
    assert rewritten["tools"] == payload["tools"]
    assert "reasoningEffort" not in rewritten
    assert rewritten["messages"][-1]["content"].endswith("/think")

    no_think = rewrite_payload(
        payload,
        "Qwen3.6-35B-A3B",
        dynamic_thinking=DynamicThinkingConfig(),
        reasoning_effort="none",
        reasoning=ReasoningConfig(),
    )
    assert no_think["chat_template_kwargs"]["enable_thinking"] is False
    assert "reasoning_budget" not in no_think
    assert no_think["messages"][-1]["content"].endswith("/no_think")


def test_old_config_and_optional_backend_routes() -> None:
    legacy = RouterConfig.model_validate(
        {"backends": [{"name": "a", "api_base": "http://127.0.0.1:1/v1"}]}
    )
    assert legacy.reasoning.default_effort is None
    assert reasoning_backend_filter(legacy, "high") == (None, False)

    routed = RouterConfig.model_validate(
        {
            "backends": [
                {"name": "off", "api_base": "http://127.0.0.1:1/v1"},
                {"name": "high", "api_base": "http://127.0.0.1:2/v1"},
            ],
            "reasoning": {"routes": {"none": "off", "high": ["high"]}},
        }
    )
    assert reasoning_backend_filter(routed, "none") == ({"off"}, False)
    assert reasoning_backend_filter(routed, "high") == ({"high"}, False)
    assert reasoning_backend_filter(routed, "medium") == (None, True)


def test_completions_prompt_processing() -> None:
    from hermes_router import (
        _extract_prompt_text_from_payload,
        estimate_prompt_tokens,
        append_text_to_last_user_message,
    )

    # 1. Test _extract_prompt_text_from_payload
    assert _extract_prompt_text_from_payload({"prompt": "Hello"}) == "Hello"
    assert _extract_prompt_text_from_payload({"prompt": ["Hello", "World"]}) == "Hello World"
    assert _extract_prompt_text_from_payload({"prompt": ["Hello", 123, {"text": "Obj"}]}) == "Hello <token_id_123> Obj"
    assert _extract_prompt_text_from_payload({"prompt": {"text": "Object prompt"}}) == "Object prompt"
    assert _extract_prompt_text_from_payload({}) == ""

    # 2. Test estimate_prompt_tokens on completions prompt
    assert estimate_prompt_tokens({"prompt": "A" * 100}) == 25

    # 3. Test append_text_to_last_user_message on completions prompt
    res1 = append_text_to_last_user_message({"prompt": "Hello"}, " /think")
    assert res1["prompt"] == "Hello /think"

    res2 = append_text_to_last_user_message({"prompt": ["Hello", "World"]}, " /think")
    assert res2["prompt"] == ["Hello", "World /think"]


def test_llama_health_caching_and_parallelization() -> None:
    import asyncio
    import time
    from hermes_router import BackendState, BackendConfig, llama_healthy, choose_backend, RouterState
    import httpx

    config = RouterConfig.model_validate({
        "backends": [
            {"name": "b1", "api_base": "http://127.0.0.1:9091/v1"},
            {"name": "b2", "api_base": "http://127.0.0.1:9092/v1"}
        ]
    })
    state = RouterState(config)
    b1 = state.backends[0]
    b1.last_watchdog_status = {"ready": True}
    b2 = state.backends[1]
    b2.last_watchdog_status = {"ready": True}

    call_count = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json={"status": "ok"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
            # First health check (no cache)
            res1 = await llama_healthy(client, b1, cache_seconds=5.0)
            assert res1 is True
            assert call_count == 1

            # Second health check (cached)
            res2 = await llama_healthy(client, b1, cache_seconds=5.0)
            assert res2 is True
            assert call_count == 1  # Should not increase

            # Choose backend parallel check should use cache as well
            res_choose = await choose_backend(
                client=client,
                state=state,
                payload={"prompt": "test"},
                request_id=1
            )
            assert res_choose is not None
            # b1 is cached, but b2 needs check
            assert call_count == 2

    asyncio.run(run())
