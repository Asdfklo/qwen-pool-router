import pytest
import asyncio
import httpx
from fastapi.testclient import TestClient
from hermes_router import (
    create_app,
    RouterConfig,
    RouterState,
    choose_backend,
    is_backend_compatible,
    payload_has_images,
    is_endpoint_allowed,
)

def test_payload_has_images():
    payload_text = {
        "messages": [{"role": "user", "content": "hello"}]
    }
    assert not payload_has_images(payload_text)

    payload_image = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                ]
            }
        ]
    }
    assert payload_has_images(payload_image)


def test_is_endpoint_allowed():
    from hermes_router import BackendConfig
    # Without allowed_endpoints configured, everything is allowed
    cfg_all = BackendConfig(name="b1", api_base="http://127.0.0.1:1/v1")
    assert is_endpoint_allowed(cfg_all, "/v1/chat/completions")
    assert is_endpoint_allowed(cfg_all, "/chat/completions")
    assert is_endpoint_allowed(cfg_all, "/infill")

    # With allowed_endpoints configured
    cfg_rest = BackendConfig(
        name="b1",
        api_base="http://127.0.0.1:1/v1",
        allowed_endpoints=["chat/completions", "/infill"]
    )
    assert is_endpoint_allowed(cfg_rest, "/v1/chat/completions")
    assert is_endpoint_allowed(cfg_rest, "/chat/completions")
    assert is_endpoint_allowed(cfg_rest, "/infill")
    assert not is_endpoint_allowed(cfg_rest, "/completions")
    assert not is_endpoint_allowed(cfg_rest, "/v1/completions")


def test_is_backend_compatible():
    from hermes_router import BackendState, BackendConfig
    cfg = BackendConfig(name="b1", api_base="http://127.0.0.1:1/v1", backend_model="Qwen3.6-35B-A3B")
    backend = BackendState(config=cfg)
    backend.backend_model = "Qwen3.6-35B-A3B"
    backend.has_vision = False

    # Fuzzy case-insensitive substring match
    assert is_backend_compatible(backend, "qwen", "/v1/chat/completions", False)
    assert is_backend_compatible(backend, "Qwen3.6", "/v1/chat/completions", False)
    assert is_backend_compatible(backend, "a3b", "/v1/chat/completions", False)
    assert not is_backend_compatible(backend, "gemma", "/v1/chat/completions", False)

    # Public model name compatibility bypass
    assert is_backend_compatible(backend, "qwen-3.6-35b", "/v1/chat/completions", False, public_model_name="qwen-3.6-35b")

    # Vision requests
    assert not is_backend_compatible(backend, "qwen", "/v1/chat/completions", True)
    backend.has_vision = True
    assert is_backend_compatible(backend, "qwen", "/v1/chat/completions", True)


@pytest.mark.asyncio
async def test_choose_backend_filtering_and_weights():
    config = RouterConfig.model_validate({
        "backends": [
            {
                "name": "gemma-node",
                "api_base": "http://127.0.0.1:9091/v1",
                "backend_model": "gemma-2-9b",
                "weight": 3,
            },
            {
                "name": "qwen-node-1",
                "api_base": "http://127.0.0.1:9092/v1",
                "backend_model": "Qwen3.6-35B-A3B",
                "weight": 1,
            },
            {
                "name": "qwen-node-2",
                "api_base": "http://127.0.0.1:9093/v1",
                "backend_model": "Qwen3.6-35B-A3B",
                "weight": 5,
            }
        ]
    })
    state = RouterState(config)
    for b in state.backends:
        b.last_watchdog_status = {"ready": True}
        b.last_llama_healthy = True

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
        # Route to gemma
        choice_gemma = await choose_backend(client, state, {}, 1, requested_model="gemma")
        assert choice_gemma is not None
        assert choice_gemma[0].config.name == "gemma-node"

        # Route to qwen - should choose qwen-node-2 because of higher weight (5 vs 1)
        choice_qwen = await choose_backend(client, state, {}, 2, requested_model="Qwen")
        assert choice_qwen is not None
        assert choice_qwen[0].config.name == "qwen-node-2"


@pytest.mark.asyncio
async def test_choose_backend_vision_prefer_and_fallback():
    config = RouterConfig.model_validate({
        "backends": [
            {
                "name": "text-node",
                "api_base": "http://127.0.0.1:9091/v1",
                "backend_model": "qwen-35b",
            },
            {
                "name": "vision-node",
                "api_base": "http://127.0.0.1:9092/v1",
                "backend_model": "qwen-35b-vision",
            }
        ]
    })
    state = RouterState(config)
    state.backends[0].has_vision = False
    state.backends[1].has_vision = True
    for b in state.backends:
        b.last_watchdog_status = {"ready": True}
        b.last_llama_healthy = True

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
        # 1. Vision request -> MUST route to vision-node
        payload_vision = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": "xyz"}]
                }
            ]
        }
        choice_vision = await choose_backend(client, state, payload_vision, 1)
        assert choice_vision is not None
        assert choice_vision[0].config.name == "vision-node"

        # 2. Pure text request -> prefers text-node (due to +1000 score preference)
        payload_text = {
            "messages": [{"role": "user", "content": "hello"}]
        }
        choice_text = await choose_backend(client, state, payload_text, 2)
        assert choice_text is not None
        assert choice_text[0].config.name == "text-node"


def test_models_aggregation_replicas():
    config = RouterConfig.model_validate({
        "backends": [
            {"name": "b1", "api_base": "http://127.0.0.1:1/v1", "backend_model": "qwen-35b"},
            {"name": "b2", "api_base": "http://127.0.0.1:2/v1", "backend_model": "qwen-35b"},
            {"name": "b3", "api_base": "http://127.0.0.1:3/v1", "backend_model": "gemma-9b"},
        ]
    })
    app = create_app(config)
    app.state.router_state = RouterState(config)
    app.state.config = config

    with TestClient(app) as client:
        res = client.get("/v1/models")
        assert res.status_code == 200
        data = res.json()["data"]
        models_dict = {item["id"]: item for item in data}

        assert "qwen-35b" in models_dict
        assert "gemma-9b" in models_dict

        # Verify replica counts
        assert models_dict["qwen-35b"]["replicas"] == 2
        assert models_dict["gemma-9b"]["replicas"] == 1


def test_embeddings_and_rerank_routing():
    config = RouterConfig.model_validate({
        "backends": [
            {"name": "general", "api_base": "http://127.0.0.1:1/v1", "backend_model": "qwen-35b"},
            {"name": "embed-node", "api_base": "http://127.0.0.1:2/v1", "backend_model": "bge-embed-v1.5"},
            {"name": "rerank-node", "api_base": "http://127.0.0.1:3/v1", "backend_model": "bge-rerank-v2"},
        ]
    })
    app = create_app(config)
    app.state.router_state = RouterState(config)
    app.state.config = config

    from hermes_router import select_proxy_backend
    state = app.state.router_state

    # Verify model name-based selection
    assert select_proxy_backend(state, "/v1/embeddings").config.name == "embed-node"
    assert select_proxy_backend(state, "/v1/rerank").config.name == "rerank-node"


def test_vision_no_capable_backend_returns_400():
    config = RouterConfig.model_validate({
        "backends": [
            {"name": "text-node-only", "api_base": "http://127.0.0.1:1/v1", "backend_model": "qwen-35b"},
        ]
    })
    app = create_app(config)
    app.state.router_state = RouterState(config)
    app.state.config = config

    with TestClient(app) as client:
        # Request with images should fail with a 400 error because no vision backend exists
        res = client.post("/v1/chat/completions", json={
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": "xyz"}]
                }
            ]
        })
        assert res.status_code == 400
        assert res.json() == {"error": "model does not support vision"}


def test_dashboard_html_renders_with_cpu_backend():
    """Regression: dashboard must render without NameError when CPU backends present."""
    from hermes_router import dashboard_html
    html = dashboard_html("test")
    assert isinstance(html, str)
    assert len(html) > 1000
    # CPU backend shows RAM, GPU shows VRAM in the JS label logic
    assert "RAM" in html
    assert "VRAM" in html
