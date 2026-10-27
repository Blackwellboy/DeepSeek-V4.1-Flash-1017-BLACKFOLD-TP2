#!/bin/bash
# Structured output with thinking on, and max_tokens running out inside reasoning (Blackfold TEST 3).
# Adapted from sushi tests/test_json_schema_thinking.sh (MIT, https://github.com/beamivalice/sushi @ 3fc47d8):
# server start and sushi's [grammar] log checks are removed; point it at a live serve.
#   BASE=http://<head>:18191 MODEL=DeepSeek-v4.1-Flash-1017-heretic-tf tools/test_json_schema_thinking.sh
set -u
BASE=${BASE:-${1:-http://127.0.0.1:18191}}
MODEL=${MODEL:-${2:-DeepSeek-v4.1-Flash-1017-heretic-tf}}
PASS=0
FAIL=0
TOTAL=0
command -v jq >/dev/null 2>&1 || { echo "FAIL: jq is required"; exit 1; }
run_test() {
    TOTAL=$((TOTAL + 1))
    if [ "$2" = PASS ]; then PASS=$((PASS + 1)); echo "  PASS: $1"
    else FAIL=$((FAIL + 1)); echo "  FAIL: $1 — $3"; fi
}

SCHEMA='{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}'
PROMPT="A basket starts with 17 apples. Six are removed, then three groups of two are added. Give the final count."

echo "=== /v1/chat/completions: reasoning_effort + json_schema, non-stream ==="
BODY=$(curl -s -m 300 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\", \"max_tokens\": 512, \"temperature\": 0, \"stream\": false,
    \"reasoning_effort\": \"medium\",
    \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"answer\", \"strict\": true, \"schema\": $SCHEMA}},
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]
}")
CONTENT=$(echo "$BODY" | jq -r '.choices[0].message.content // ""')
if echo "$CONTENT" | jq -e 'has("answer")' >/dev/null 2>&1; then
    run_test "schema JSON lands in content (chat, non-stream)" PASS ""
else
    RC=$(echo "$BODY" | jq -r '.choices[0].message.reasoning_content // ""' | head -c 120)
    run_test "schema JSON lands in content (chat, non-stream)" FAIL "content: '$(echo "$CONTENT" | head -c 120)' reasoning: '$RC'"
fi

echo "=== /v1/chat/completions: same request, stream ==="
STREAM=$(curl -s -m 300 -N "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\", \"max_tokens\": 512, \"temperature\": 0, \"stream\": true,
    \"reasoning_effort\": \"low\",
    \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"answer\", \"strict\": true, \"schema\": $SCHEMA}},
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]
}")
SCONTENT=$(echo "$STREAM" | grep '^data: {' | sed 's/^data: //' | jq -rj '.choices[0].delta.content // empty' 2>/dev/null)
if echo "$SCONTENT" | jq -e 'has("answer")' >/dev/null 2>&1; then
    run_test "schema JSON lands in delta.content (chat, stream)" PASS ""
else
    run_test "schema JSON lands in delta.content (chat, stream)" FAIL "streamed content: '$(echo "$SCONTENT" | head -c 120)'"
fi

echo "=== /v1/responses: reasoning.effort + text.format json_schema ==="
BODY=$(curl -s -m 300 "$BASE/v1/responses" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\", \"max_output_tokens\": 2048, \"temperature\": 0, \"stream\": false,
    \"reasoning\": {\"effort\": \"medium\"},
    \"text\": {\"format\": {\"type\": \"json_schema\", \"name\": \"answer\", \"strict\": true, \"schema\": $SCHEMA}},
    \"input\": \"$PROMPT\"
}")
RTEXT=$(echo "$BODY" | jq -r '[.output[]? | select(.type=="message") | .content[]? | select(.type=="output_text") | .text] | join("")')
if echo "$RTEXT" | jq -e 'has("answer")' >/dev/null 2>&1; then
    run_test "schema JSON lands in output_text (responses)" PASS ""
else
    run_test "schema JSON lands in output_text (responses)" FAIL "output_text: '$(echo "$RTEXT" | head -c 120)'"
fi

echo "=== /v1/messages: finite thinking budget + json_schema, stream ==="
STREAM=$(curl -s -m 300 -N "$BASE/v1/messages" -H "Content-Type: application/json" -d "{
    \"model\": \"$MODEL\", \"max_tokens\": 512, \"temperature\": 0, \"stream\": true,
    \"thinking\": {\"type\": \"enabled\", \"budget_tokens\": 256},
    \"output_config\": {\"format\": {\"type\": \"json_schema\", \"schema\": $SCHEMA}},
    \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]
}")
MTEXT=$(echo "$STREAM" | sed -n 's/^data: //p' | jq -rj 'select(.type=="content_block_delta" and .delta.type=="text_delta") | .delta.text' 2>/dev/null)
if echo "$MTEXT" | jq -e 'has("answer")' >/dev/null 2>&1; then
    run_test "schema JSON lands in text delta (messages, finite budget)" PASS ""
else
    run_test "schema JSON lands in text delta (messages, finite budget)" FAIL "streamed text: '$(echo "$MTEXT" | head -c 120)'"
fi

# Unlike a reasoning cutoff with answer capacity left, the hard completion cap
# must not force a boundary, reserve answer tokens, or move reasoning to content.
# This short request exercises exhaustion inside the prompt-opened think block;
# only routing, accounting, and the finish reason matter, never the model's prose.
for STREAMING in false true; do
    echo "=== /v1/chat/completions: reasoning exhausts max_tokens, stream=$STREAMING ==="
    BODY=$(curl -s -m 300 -N "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{
        \"model\": \"$MODEL\", \"max_tokens\": 64, \"temperature\": 0, \"stream\": $STREAMING,
        \"enable_thinking\": true, \"reasoning_budget_tokens\": -1,
        \"stream_options\": {\"include_usage\": true},
        \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"answer\", \"strict\": true, \"schema\": $SCHEMA}},
        \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}]
    }")
    if [ "$STREAMING" = true ]; then
        RESULT=$(echo "$BODY" | sed -n 's/^data: \({.*}\)$/\1/p' | jq -s '{
            finish_reason: [.[] | .choices[]? | .finish_reason // empty] | last,
            completion_tokens: [.[] | .usage.completion_tokens // empty] | last,
            content: [.[] | .choices[]? | .delta.content // empty] | join(""),
            reasoning: [.[] | .choices[]? | .delta.reasoning_content // empty] | join("")
        }' 2>/dev/null)
    else
        RESULT=$(echo "$BODY" | jq '{
            finish_reason: .choices[0].finish_reason,
            completion_tokens: .usage.completion_tokens,
            content: (.choices[0].message.content // ""),
            reasoning: (.choices[0].message.reasoning_content // "")
        }' 2>/dev/null)
    fi
    # Whether 64 tokens is exhausted mid-thought is the checkpoint's choice:
    # a model that finishes under the cap must land schema JSON in content.
    if echo "$RESULT" | jq -e '(.finish_reason == "length" and .completion_tokens == 64 and
        .content == "" and (.reasoning | length > 0)) or
        (.finish_reason == "stop" and .completion_tokens < 64 and (.content | fromjson | .answer != null))' >/dev/null 2>&1; then
        run_test "reasoning-only length stop respects completion cap (stream=$STREAMING)" PASS ""
    else
        run_test "reasoning-only length stop respects completion cap (stream=$STREAMING)" FAIL "$(echo "$RESULT" | head -c 240)"
    fi
done

echo "=== $PASS/$TOTAL passed ==="
[ "$FAIL" -eq 0 ] || exit 1
