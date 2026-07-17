from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from hermes_router import (
    RouterConfig,
    RouterState,
    SemanticHealthConfig,
    acquire_backend_lease,
    active_request_stale_seconds,
    apply_config_reload,
    reap_stale_active_leases,
    release_backend_lease,
    reserve_semantic_backend,
    run_semantic_backend_check,
    validate_config,
)


def make_config(**overrides: object) -> RouterConfig:
    payload: dict[str, object] = {
        "backends": [
            {
                "name": "node-a",
                "api_base": "http://node-a:8081/v1",
                "backend_model": "qwen",
            }
        ]
    }
    payload.update(overrides)
    return RouterConfig.model_validate(payload)


def test_config_validation_accepts_current_shape() -> None:
    validate_config(make_config())


@pytest.mark.parametrize(
    "override",
    [
        {"max_queue_seconds": 0},
        {"retry_after_seconds": 0},
        {
            "backends": [
                {"name": "same", "api_base": "http://a/v1"},
                {"name": "same", "api_base": "http://b/v1"},
            ]
        },
        {"reasoning": {"routes": {"high": "missing"}}},
        {"semantic_health": {"enabled": True, "interval_seconds": 10}},
    ],
)
def test_config_validation_rejects_unsafe_values(override: dict) -> None:
    with pytest.raises(ValueError):
        validate_config(make_config(**override))


def test_mutable_config_reload_preserves_backend_state() -> None:
    current = make_config(max_queue_seconds=5)
    state = RouterState(current)
    original_backend = state.backends[0]
    original_backend.last_success = 123
    app = SimpleNamespace(
        state=SimpleNamespace(router_state=state, status_snapshot={"old": True})
    )
    updated = make_config(
        max_queue_seconds=30,
        retry_after_seconds=30,
        semantic_health={"enabled": True, "interval_seconds": 900},
    )

    changed = asyncio.run(apply_config_reload(app, current, updated))

    assert "max_queue_seconds" in changed
    assert current.max_queue_seconds == 30
    assert current.retry_after_seconds == 30
    assert current.semantic_health.enabled is True
    assert state.backends[0] is original_backend
    assert state.backends[0].last_success == 123
    assert state.config_reload_count == 1
    assert app.state.status_snapshot is None


def test_reload_rejects_restart_required_changes() -> None:
    current = make_config(listen_port=4000)
    state = RouterState(current)
    app = SimpleNamespace(state=SimpleNamespace(router_state=state, status_snapshot=None))
    updated = make_config(listen_port=4001)

    with pytest.raises(ValueError, match="restart required"):
        asyncio.run(apply_config_reload(app, current, updated))
    assert current.listen_port == 4000
    assert state.config_reload_count == 0


def test_reload_rejects_removing_active_backend() -> None:
    current = RouterConfig.model_validate(
        {
            "backends": [
                {"name": "node-a", "api_base": "http://a/v1"},
                {"name": "node-b", "api_base": "http://b/v1"},
            ]
        }
    )
    state = RouterState(current)
    state.backends[1].active_requests = 1
    app = SimpleNamespace(state=SimpleNamespace(router_state=state, status_snapshot=None))
    updated = make_config()

    with pytest.raises(ValueError, match="active backends"):
        asyncio.run(apply_config_reload(app, current, updated))


def test_active_lease_release_preserves_other_active_requests() -> None:
    config = make_config()
    state = RouterState(config)
    backend = state.backends[0]

    first = acquire_backend_lease(backend, 1)
    second = acquire_backend_lease(backend, 2)
    release_backend_lease(backend, first)

    assert backend.active_requests == 1
    assert first not in backend.active_leases
    assert second in backend.active_leases


def test_stale_active_lease_reaper_recovers_busy_backend() -> None:
    config = make_config(request_deadline_seconds=1, request_timeout_seconds=1)
    state = RouterState(config)
    backend = state.backends[0]

    lease_id = acquire_backend_lease(backend, 1)
    backend.active_leases[lease_id] = time_now = 1000.0
    recovered = reap_stale_active_leases(
        backend,
        config,
        now=time_now + active_request_stale_seconds(config) + 1,
    )

    assert recovered == 1
    assert backend.active_requests == 0
    assert backend.stale_active_resets == 1
    assert backend.active_leases == {}


def test_completion_semantic_check() -> None:
    async def run() -> None:
        config = make_config()
        state = RouterState(config)
        backend = state.backends[0]
        backend.last_watchdog_status = {"ready": True}
        semantic = SemanticHealthConfig(idle_seconds=0)

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = {
                "choices": [{"message": {"role": "assistant", "content": "NYX_OK"}}]
            }
            return httpx.Response(200, json=payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await reserve_semantic_backend(state, backend, semantic)
            assert await run_semantic_backend_check(
                client, state, backend, semantic, force_tool_check=False
            )
        assert backend.semantic_healthy is True
        assert backend.semantic_check_count == 1
        assert backend.active_requests == 0

    asyncio.run(run())


def test_tool_semantic_check() -> None:
    async def run() -> None:
        config = make_config()
        state = RouterState(config)
        backend = state.backends[0]
        backend.last_watchdog_status = {"ready": True}
        semantic = SemanticHealthConfig(idle_seconds=0, tool_check_every=1)

        async def handler(request: httpx.Request) -> httpx.Response:
            payload = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {"name": "nyx_health_ping", "arguments": '{"status":"ok"}'},
                                }
                            ],
                        }
                    }
                ]
            }
            return httpx.Response(200, json=payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await reserve_semantic_backend(state, backend, semantic)
            assert await run_semantic_backend_check(
                client, state, backend, semantic, force_tool_check=True
            )
        assert backend.semantic_healthy is True
        assert backend.semantic_tool_healthy is True

    asyncio.run(run())


def test_semantic_failure_is_reported_without_opening_circuit() -> None:
    async def run() -> None:
        config = make_config()
        state = RouterState(config)
        backend = state.backends[0]
        backend.last_watchdog_status = {"ready": True}
        semantic = SemanticHealthConfig(idle_seconds=0, enforce=False)

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "WRONG"}}]},
                request=request,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await reserve_semantic_backend(state, backend, semantic)
            assert not await run_semantic_backend_check(
                client, state, backend, semantic, force_tool_check=False
            )
        assert backend.semantic_healthy is False
        assert backend.semantic_consecutive_failures == 1
        assert backend.circuit_state == "closed"

    asyncio.run(run())


def test_dynamic_props_detection() -> None:
    async def run() -> None:
        config = make_config()
        state = RouterState(config)
        backend = state.backends[0]
        backend.last_props_checked = 0.0

        async def handler(request: httpx.Request) -> httpx.Response:
            url_str = str(request.url)
            if "/props" in url_str:
                props_payload = {
                    "default_generation_settings": {
                        "n_ctx": 4096,
                    },
                    "modalities": {
                        "vision": True,
                    },
                    "model_path": "/models/Qwen-7B-Instruct.gguf"
                }
                return httpx.Response(200, json=props_payload, request=request)
            elif "/metrics" in url_str:
                return httpx.Response(200, text="llamacpp:prompt_tokens_seconds 123.0\nllamacpp:predicted_tokens_seconds 45.0\n", request=request)
            return httpx.Response(404, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            from hermes_router import refresh_backend_statuses
            await refresh_backend_statuses(client, state, config)

        assert backend.max_context_tokens == 4096
        assert backend.has_vision is True
        assert backend.backend_model == "Qwen-7B-Instruct"
        assert backend.last_props_checked > 0.0

    asyncio.run(run())


def test_models_endpoint_has_vision_and_context() -> None:
    config = make_config()
    from hermes_router import create_app
    app = create_app(config)

    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        state = app.state.router_state
        state.backends[0].max_context_tokens = 8192
        state.backends[0].has_vision = True

        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        model = data["data"][0]
        assert model["context_length"] == 8192
        assert model["modalities"]["vision"] is True
        assert "image" in model["architecture"]["input_modalities"]
