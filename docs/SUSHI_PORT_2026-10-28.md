# sushi port: structured output, visible queue, clean shutdown (28 Oct 2026)

Ported from [sushi](https://github.com/beamivalice/sushi) (MIT) at `3fc47d8a460970b1e86b24de1cab3b0f2067bc72`:
patch [0004](../patches/0004-sushi-port-structured-output-queue-drain.patch), the template in
[templates/dsv41](../templates/dsv41/README.md) and the tests in `tools/test_json_schema_*.sh`.

**Patch base:** 0004 is made against the live candidate-6.1 engine tree (`native-blackfold-cand61`, head binary
sha256 `05defe05...`), which carries server changes beyond `6366d2c` + 0001-0003. It does **not** apply cleanly to
`6366d2c` + 0001-0003 yet (`chat.zig`, `openai.zig`, `zig/build/cuda.zig` conflict).

## What changed

- **response_format no longer HTTP 400.** The native DeepSeek-V4.1 engine has no grammar mask (`Info.structures`
  false), so `response_format` was refused. The server now appends sushi's instruction ("Respond with valid JSON
  only ... MUST conform to this JSON schema" + the schema) to the leading system message before render.
  `json_schema`, `json_object`, `guided_json`, `guided_choice`, `guided_regex` and `structured_outputs` are served;
  `guided_grammar` (EBNF) is still refused. This is **prompting, not enforcement**: the reply follows the schema only as
  well as the model follows the instruction. Validate on the client.
- **Visible queue** (sushi `1d16e45`): `[queue] <id> queued behind R running and W waiting` and `... got a lane after
  waiting Ns` log lines; streamed requests get `: queued behind ...` SSE comments while they wait.
- **Clean shutdown** (sushi `cb2525f`): on SIGTERM, queued and running replies end with HTTP 503 (non-stream) or an
  error event + `[DONE]` (stream) instead of a short 200. A capacity refusal on a non-stream reply now answers 503
  (it was hard-coded 400).
- **Template check**: `zig build test-dsv41-template`, 14/14 byte-exact.

Not ported: sushi's grammar-constrained decoding (`json_schema.zig`, `json_grammar.zig`, `token_mask.zig`,
`regex.zig`, `row_grammar.zig`). TensorFold's CUDA lanes engine samples on the GPU with drafted verify rows across
two ranks; a mask needs a per-row vocab mask uploaded and applied in the dsv41 sampling kernels plus draft-row checks,
which is engine work, not a server port.

## A/B on the live pair (head + worker, same weights, flags and template)

Old = candidate 6.1 (`05defe05...`); new = candidate 6.1 + 0004 (`8e3e85d8...`). Greedy (temperature 0).

| Check | Old | New |
|---|---|---|
| Ivan-style curl, `json_schema` / `json_object` | 400 / 400 | 200 / 200 |
| 12-case schema suite, `reasoning_effort: none` (valid JSON, required keys, no extras, enums/types) | 0/12 (all 400) | 11/12 |
| same suite, `reasoning_effort: high` | 0/12 (all 400) | 12/12 |
| sushi `test_json_schema_thinking.sh` (6 checks, incl. TEST 3 max_tokens inside reasoning) | 0/6 | 6/6 |
| sushi `test_json_schema_enforcement.sh` (6 adversarial cases: prompt demands ```json fences + an extra field) | 0/6 | 0/6 |
| Decode tok/s, 400-token greedy, code / prose / Go | 110.1 / 58.6 / 104.5 (4 runs each, 2 restarts) | 109.5 / 58.1 / 103.5 (7 runs each, 3 restarts); last warm sample 109.9 / 58.4 / 104.0 |
| Shutdown with 2 replies in flight | process killed, replies cut | stream: error event + `[DONE]`; non-stream: 503 |
| 6 concurrent streams on 4 lanes | waits invisible | `[queue]` lines + `: queued behind ...` comments |

The one remaining suite failure and all 6 adversarial failures are the same thing: when the user explicitly asks for a
```json fence or an extra field, the model obeys the user over the system instruction. Only a decode mask fixes that.

Decode speed is within about 1% (old 91.1 and 91.0 tok/s suite means over two restarts; new 90.5, 89.5 and 90.8). The
server change does not touch the decode path; part of the gap may be the rebuilt binary itself (a rebuild of the
unchanged baseline was not measured). Swap downtime was 42-46 s per restart (model loads in ~33 s).
