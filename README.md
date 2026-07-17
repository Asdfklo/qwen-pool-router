# Nyx Router Reasoning Variants

Nyx Router accepts Codex/OpenCode-style reasoning levels and translates them into Qwen controls without forwarding client-only provider fields to llama.cpp.

Supported levels are `none`, `minimal`, `low`, `medium`, `high`, and `xhigh`. Model suffixes have highest precedence, followed by request fields, model defaults, and the global default. For example, `qwen-3.6-35b:high` overrides `reasoningEffort: "none"`.

When no reasoning variant or default is configured, Nyx preserves its existing `/think`, `/no_think`, and keyword-rule auto mode. This is intentional for the current Precision deployment.

## OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "nyx/qwen-3.6-35b",
  "provider": {
    "nyx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Nyx Router",
      "options": {
        "baseURL": "http://192.168.68.99:4000/v1"
      },
      "models": {
        "qwen-3.6-35b": {
          "name": "Nyx Qwen 3.6 35B",
          "variants": {
            "none": { "reasoningEffort": "none", "textVerbosity": "low" },
            "minimal": { "reasoningEffort": "minimal", "textVerbosity": "low" },
            "low": { "reasoningEffort": "low", "textVerbosity": "low" },
            "medium": { "reasoningEffort": "medium", "textVerbosity": "low" },
            "high": { "reasoningEffort": "high", "textVerbosity": "low" },
            "xhigh": { "reasoningEffort": "xhigh", "textVerbosity": "low" }
          }
        }
      }
    }
  }
}
```

```bash
opencode run --model nyx/qwen-3.6-35b --variant high \
  "Reply briefly and solve: what is 17 * 23?"

opencode run --model nyx/qwen-3.6-35b --variant none \
  "Reply briefly and solve: what is 17 * 23?"
```

The same controls are accepted as `reasoningEffort`, `reasoning_effort`, `reasoning.effort`, `reasoning.level`, supported `providerOptions` shapes, and supported `extra_body` shapes.

## Precision Deployment

Precision currently runs Qwen on `8081`; ports `8082` and `8083` host unrelated Gemma and Skyfall models. Nyx therefore uses the existing Qwen replicas on Precision and New-Gaming and applies these per-request controls:

| Effort | Qwen thinking | Budget |
| --- | --- | ---: |
| none | off | 0 |
| minimal | on | 128 |
| low | on | 512 |
| medium | on | 2048 |
| high | on | 8192 |
| xhigh | on | 16384 |

This avoids loading three copies of the 35B model and does not interfere with the other llama.cpp services. The `reasoning.routes` map remains available for fake tests or future dedicated profiles.

## Optional Dedicated Profiles

If future hardware can keep several Qwen instances resident, start separate `none`, `medium`, and `high` llama.cpp profiles with distinct unused ports. Do not use Precision's current `8082` or `8083`.

```bash
# Common arguments abbreviated; each profile needs its own free port.
llama-server -m /mnt/models/qwen36-apex-mtp/Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf \
  --port PORT_NONE --jinja --reasoning off --reasoning-format deepseek --ctx-size 65536

llama-server -m /mnt/models/qwen36-apex-mtp/Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf \
  --port PORT_MEDIUM --jinja --reasoning on --reasoning-format deepseek \
  --reasoning-budget 2048 --ctx-size 65536

llama-server -m /mnt/models/qwen36-apex-mtp/Qwen3.6-35B-A3B-APEX-MTP-I-Compact.gguf \
  --port PORT_HIGH --jinja --reasoning on --reasoning-format deepseek \
  --reasoning-budget 8192 --ctx-size 65536
```

Map `minimal` to the off or medium profile, `low` to medium, and `xhigh` to high until benchmarks justify more resident copies.

## Validation

```bash
pytest -q
./scripts/start-fake-reasoning-upstreams.sh
/opt/hermes-qwen-pool/.venv/bin/python hermes_router.py --config router-beta-fake.yaml
NYX_ROUTER_URL=http://127.0.0.1:4001 ./scripts/test-reasoning-routing.sh
./scripts/start-fake-reasoning-upstreams.sh stop
```

## Configuration Validation and Reload

Validate a config without starting a listener:

```bash
/opt/hermes-qwen-pool/.venv/bin/python hermes_router.py \
  --config /etc/hermes-qwen-pool/router.yaml --check-config
```

Apply mutable changes after validation:

```bash
sudo systemctl kill --kill-who=main -s HUP hermes-router.service
```

Queue limits, retry behavior, reasoning settings, backend settings, circuit breakers, affinity, and semantic-health settings reload atomically. Listener address/port, HTTP-client timeouts, and telemetry paths require a normal restart. A rejected reload leaves the active configuration unchanged and records the error in `/api/status` and the journal.

## Semantic Health

Semantic checks are fixed, no-thinking requests made directly to each backend only after the router has been idle. Completion checks expect `NYX_OK`; every configured Nth pass also verifies a forced `nyx_health_ping` tool call.

```yaml
semantic_health:
  enabled: true
  interval_seconds: 900
  initial_delay_seconds: 120
  idle_seconds: 30
  timeout_seconds: 20
  max_tokens: 12
  expected_response: NYX_OK
  tool_check_every: 4
  failure_threshold: 3
  enforce: false
```

Keep `enforce: false` until the model and template have demonstrated stable canary behavior. Report-only failures appear in `/api/status`, structured logs, and the node table without opening a circuit or removing the backend.
