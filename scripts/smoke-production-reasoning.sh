#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${NYX_ROUTER_URL:-http://127.0.0.1:4000}"
MODEL="${NYX_ROUTER_MODEL:-qwen-3.6-35b}"

post() {
  curl -fsS --max-time 180 "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$1"
}

assert_content() {
  local name="$1" payload="$2" expected="$3" response actual
  response="$(post "$payload")"
  actual="$(jq -r '.choices[0].message.content' <<<"$response")"
  [[ "$actual" == "$expected" ]] || { echo "$name: expected $expected, got $actual" >&2; exit 1; }
  echo "$name -> $actual"
}

models="$(curl -fsS "$BASE_URL/v1/models")"
for suffix in none minimal low medium high xhigh; do
  jq -e --arg id "$MODEL:$suffix" '.data | map(.id) | index($id) != null' <<<"$models" >/dev/null
done
echo "/v1/models -> all reasoning aliases present"

assert_content "legacy auto" \
  "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly LEGACY_OK and nothing else.\"}],\"temperature\":0,\"max_tokens\":24}" \
  "LEGACY_OK"
assert_content "reasoningEffort none" \
  "{\"model\":\"$MODEL\",\"reasoningEffort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly NONE_OK\"}],\"temperature\":0,\"max_tokens\":24}" \
  "NONE_OK"
assert_content "providerOptions none" \
  "{\"model\":\"$MODEL\",\"providerOptions\":{\"openai\":{\"reasoningEffort\":\"none\"}},\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly PROVIDER_OK\"}],\"temperature\":0,\"max_tokens\":24}" \
  "PROVIDER_OK"
assert_content "suffix overrides body" \
  "{\"model\":\"$MODEL:high\",\"reasoningEffort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"Calculate 17 * 23, then reply with only the number.\"}],\"temperature\":0,\"max_tokens\":128}" \
  "391"

stream="$(curl -fsS -N --max-time 120 "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL:none\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly STREAM_OK\"}],\"temperature\":0,\"max_tokens\":24,\"stream\":true}")"
grep -q 'data: ' <<<"$stream"
grep -q '\[DONE\]' <<<"$stream"
echo "streaming none -> valid SSE through [DONE]"

curl -fsS "$BASE_URL/api/status" | jq -e 'all(.backends[]; .ready == true)' >/dev/null
echo "all configured backends -> ready"
