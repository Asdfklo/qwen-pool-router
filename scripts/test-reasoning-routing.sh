#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${NYX_ROUTER_URL:-http://127.0.0.1:4001}"
MODEL="${NYX_ROUTER_MODEL:-qwen-3.6-35b}"

request() {
  curl -fsS "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"
}

assert_route() {
  local name="$1" payload="$2" expected="$3" response actual
  response="$(request "$payload")"
  actual="$(jq -r '.choices[0].message.content' <<<"$response")"
  [[ "$actual" == "$expected" ]] || { echo "$name: expected $expected, got $actual" >&2; return 1; }
  echo "$name -> $actual"
}

models="$(curl -fsS "$BASE_URL/v1/models")"
jq -e --arg model "$MODEL" '.data | map(.id) | index($model + ":xhigh") != null' <<<"$models" >/dev/null
echo "/v1/models -> base model plus reasoning aliases"

assert_route "none" \
  "{\"model\":\"$MODEL\",\"reasoningEffort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"route test\"}],\"max_tokens\":20}" \
  "FAKE_UPSTREAM_OFF"
assert_route "medium" \
  "{\"model\":\"$MODEL\",\"reasoning_effort\":\"medium\",\"messages\":[{\"role\":\"user\",\"content\":\"route test\"}],\"max_tokens\":20}" \
  "FAKE_UPSTREAM_MEDIUM"
assert_route "high" \
  "{\"model\":\"$MODEL\",\"reasoning\":{\"effort\":\"high\"},\"messages\":[{\"role\":\"user\",\"content\":\"route test\"}],\"max_tokens\":20}" \
  "FAKE_UPSTREAM_HIGH"
assert_route "providerOptions" \
  "{\"model\":\"$MODEL\",\"providerOptions\":{\"openai\":{\"reasoningEffort\":\"high\"}},\"messages\":[{\"role\":\"user\",\"content\":\"route test\"}],\"max_tokens\":20}" \
  "FAKE_UPSTREAM_HIGH"
assert_route "suffix override" \
  "{\"model\":\"$MODEL:high\",\"reasoningEffort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"route test\"}],\"max_tokens\":20}" \
  "FAKE_UPSTREAM_HIGH"

tool_response="$(request "{\"model\":\"$MODEL:none\",\"messages\":[{\"role\":\"user\",\"content\":\"tool test\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"ping\",\"parameters\":{\"type\":\"object\"}}}],\"tool_choice\":\"auto\"}")"
jq -e '.nyx_received.tools[0].type == "function" and .nyx_received.tool_choice == "auto"' <<<"$tool_response" >/dev/null
jq -e '.nyx_received | has("reasoningEffort") | not' <<<"$tool_response" >/dev/null
echo "tool fields preserved; router-only reasoning fields stripped"

stream="$(curl -fsS -N "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL:high\",\"messages\":[{\"role\":\"user\",\"content\":\"stream test\"}],\"stream\":true,\"max_tokens\":20}")"
grep -q 'FAKE_UPSTREAM_HIGH' <<<"$stream"
grep -q '\[DONE\]' <<<"$stream"
echo "streaming high -> valid SSE from FAKE_UPSTREAM_HIGH"

echo "All reasoning routing checks passed"
