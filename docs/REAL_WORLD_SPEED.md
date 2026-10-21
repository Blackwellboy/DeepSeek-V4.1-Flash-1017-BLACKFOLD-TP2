# Real-world speed: what to expect, and how to get it

The headline numbers in the README are real measurements, and they are best cases: greedy decoding, thinking off, a
short prompt, one request at a time, after warm-up, with nothing else on the pair. This page shows what changes
when you use BLACKFOLD the way people actually do: thinking on, long context, sampled, agents. It also lists what you
need to get the measured speeds on your own two Sparks, and how to check that you do.

All numbers below come from the same pair and recipe as the README unless marked otherwise. The raw files are in
[results/](../results/RESULTS.md).

## 1. What you ask matters most

Speculative decoding (the model's own draft head) accepts more tokens on predictable text, so decode speed depends on
the kind of answer. These are single-stream rates from tonyd2wild's `v41bench_tf` on our serve (greedy, thinking off),
from `results/raw/tony-v41bench-parsed.json`:

| Category | tok/s |
|---|---:|
| formatting | 114.2 |
| coding | 113.0 |
| math | 107.9 |
| JSON | 100.6 |
| reasoning | 86.4 |
| prose | 65.1 |
| summary | 56.9 |
| narrative | 56.7 |

lucataco's `kit_bench` gives the same picture: code 103.8 tok/s and prose 58.8 tok/s at 384 tokens.

Draft acceptance from the server's own counters: about 78-80% on code (~4.2 tokens per round), and 55-61% on prose
(~2.0 tokens per round).

## 2. Long context costs a little decode, and prefill time grows with it

| Same tasks, single stream (`results/raw/engine-6366d2c-qualification.txt`) | 8K context | 125K context |
|---|---:|---:|
| code, tok/s | 98.0 | 93.1 |
| prose, tok/s | 60.8 | 57.2 |

**Cold prefill** (nothing cached) runs at about 1,700-2,000 tok/s:
- `kit_bench`: 1,690 / 2,016 / 2,008 tok/s at 8K / 32K / 128K.
- A 125K-token prompt takes about **62 s** on two rails, and 67.9 s on one.
- A 250K-token prompt takes about two minutes.

**After the first time, the prompt cache does the work.** The serve keeps prompt states and resumes from them:
- A follow-up turn only processes what is new since the last kept state.
- Kept states sit on 2,048-token boundaries, so expect a few thousand tokens per turn even when little changed.
- In the agent session below, 95% of prompt tokens came from the cache.

## 3. The first long reply after a start is slow

The engine prepares its CUDA graphs the first time it meets a new size, during that request. On a 131K-token prompt,
`kit_bench`'s `depth` test ran at 35.4 tok/s the first time, then 76.7, then 93.1. Before you time anything, send one
request at the context size you care about, or discard the first run.

## 4. Thinking: you pay for the tokens, not just the speed

With thinking on (the serve's default), every reply first writes its reasoning, and those tokens count against
`max_tokens` and against your time.

**How effort maps to budget.** The reasoning effort you send sets a budget the template writes into the prompt:

| You send | Budget |
|---|---:|
| `minimal` | 25 |
| `low` | 50 |
| `medium` | 62 |
| `high` (the default) | 75 |
| `xhigh` | 88 |
| `max` | 100 |
| `none` | thinking off |

**Picking a level:**
- Lower levels reason for fewer tokens and answer sooner.
- For tool-calling agents, `low` is a sensible starting point.
- Raise it for hard problems.

**Keep one thinking setting per conversation.** The effort line is written at the start of the prompt. A request with
a different setting, or with thinking turned off, is a different prompt, and the prompt cache cannot reuse the earlier
one. On a 150K-token conversation, a switch costs a cold re-process: about a minute.

**Set `max_tokens` yourself.**
- A thinking reply that hits the limit before it finishes reasoning comes back with empty `content`.
- The deploy script now defaults to 32,768 when a request sends none; it was 4,096 for the measurements in this repo.
- DeepSeek recommends at least 256K.

## 5. A real agent session

This is one coding-agent session on our serve, observed from its logs; it is not a benchmark:
- 43 calls, thinking on at `xhigh`, temperature 1.0
- context grew from 34K to 196K tokens and stayed between 157K and 196K for most calls

| | |
|---|---:|
| Median call | 12 s |
| Output speed on calls with 500+ output tokens (median, wall time including the uncached part of the prompt) | ~57 tok/s |
| Draft acceptance / tokens per round | 60% / 2.7 |
| Prompt tokens served from the cache | 95% |

That is what thinking text at long context looks like: closer to the prose rows above than to the coding row. Lower
effort (section 4) and earlier summarising of long sessions (decode and every cache miss get slower as context
grows) are the two levers a user controls.

## 6. Getting the measured speeds on your pair

In order of impact:

1. **Two rails.** Use both RoCE ports, MTU 9000, with the GID index aligned on both nodes ([RAILS.md](RAILS.md)). On
   our pair the second rail cut a cold 125K prompt from 67.9 s to 62.0 s, and raised 125K decode by 1.4-2.5%.
2. **Healthy, matched GPUs.** The slower node sets the pace for both.
   - Run `tools/bf_preflight.sh` on each node and compare SM clock, max clock, power, temperature and active throttle
     reasons.
   - A GB10 can latch to a low clock that only a power cycle clears; tonyd2wild documents this.
   - Long uptime with many engine starts can fragment memory: compare the `buddyinfo` line, and reboot if one node
     looks very different.
3. **Nothing else on the pair.** The head has only a few GiB free with the model loaded. Another GPU job, or a
   second serve, costs speed or memory, and can cost the serve itself.
4. **Warm up before you measure** (section 3).
5. **Byte-stable prompts.** Keep the system prompt and tool list identical between turns: no timestamps or changing
   counters near the top. Keep one thinking setting per conversation.
6. **Client settings.**
   - Set `max_tokens`.
   - Leave `top_k` unset or send 1-50. On this engine pin, a sampled request with `top_k` 0 or -1 fails the decode
     round for every request running at that moment; a fix exists in the TensorFold forks and is being upstreamed.
   - DeepSeek recommends temperature 1.0 and top_p 1.0. The serve's defaults are temperature 1.0, top_p 0.95 and
     top_k 20.
7. **Secure the port.** The serve binds `0.0.0.0`. Set `API_KEY_FILE` in `deploy/blackfold.env` if anything beyond
   the machine can reach it.

## 7. Measure it yourself

`tools/bf_lab.py` records a run against your serve, and compares two runs by full token IDs and by the engine's own
counters. See [tools/README.md](../tools/README.md). Record both regimes:

```bash
# the benchmark regime: greedy, thinking off (comparable with the numbers above)
python3 tools/bf_lab.py record --base http://HEAD:18191 --label bench --out bench.json --yes-quiet-window
# the regime people use: sampled, thinking on
python3 tools/bf_lab.py record --base http://HEAD:18191 --label real --temperature 1.0 --thinking low \
  --suite long --out real.json --yes-quiet-window
```

Sampled runs are repeatable: the serve seeds sampling from the prompt. Two runs of the same build must therefore match
token for token in either regime, and any change you make can be checked the same way.
