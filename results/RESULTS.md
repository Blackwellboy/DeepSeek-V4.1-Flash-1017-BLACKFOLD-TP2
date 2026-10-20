# Results and conditions

Serve: two DGX Spark (GB10) nodes, TP2, TensorFold native `629c524` + `patches/0001-0003`, two RoCE
rails, NIGHTFALL weights (LibertAI EXL3 2.9 bpw + refusal patch), context 1,048,576, 4 parallel streams,
replay floor, graphs, L2 prefetch and Engram AIO on. Measured 2026-10-20 with no other traffic.
Comparator figures are each project's own published results for its stock build, on its own hardware.

## Same-tool benchmarks

`kitbench-*.json`: lucataco's `bench/run-bench.sh` unchanged (kit_bench.py at its pinned hash), greedy,
thinking off. Decode is the median of 3 at 384 tokens; concurrency 2 reps; prefill cold, then warm.
`kitbench-depth*.json`: the tool's `depth` command (131,072-token prompt, 256 tokens) has no repeat
option, so it was run three times; the first run after warm-up is consistently slow (35.4 tok/s), the
next two were 76.7 and 93.1. lucataco's published figure is one run (71.1).

`tony-v41bench-parsed.json`: tonyd2wild's `v41bench_tf.py`, `--levels 1,2,3,4 --prefill
2000,8000,32000,64000,128000`, completions route, temperature 0. Totals are tokens over wall time
including time to first token, averaged over 8 categories (the counting ceiling category excluded, as in
his reporting). Prefill is prompt tokens over the full one-token request time.

## Quality

`eval-blackfold.json`, `eval-compare.txt`: lucataco's `eval/quality_eval.py`, seed 20261004, subset digest
`09c2e6cc43e275af` (200 MMLU, 100 GSM8K), thinking off and on, 4 parallel requests, compared with
`compare.py` against the DeepSeek API answers committed in lucataco's repository (same subset digest).
No API was called. Answer caps: MMLU 16 tokens off and 8,192 on; GSM8K 2,048 off and 8,192 on. The MMLU
thinking-on gap (90.5% vs 93.0%, p = 0.12) comes mostly from 8 answers cut at the 8,192-token cap
(API: 3).

## Controlled A/B trials (same pair, same prompts)

Held-out code and prose tasks at 8K and 125K context, solo and two concurrent streams, four repetitions
per cell, temperature 0. Every arm's outputs had to match a saved Python-engine reference by full
generated token IDs (all did: 96 outputs in the three-arm trial, 128 in the four-arm trial).

- `copyoff-ab-summary.json`, `copyoff-ab-round-counters.json`: Python engine / native with patches /
  Python engine, one rail. Native reproduced Python's round counts exactly in all four solo cells and
  beat both Python arms in 9 of 10 cells by mean.
- `rail-ab-summary.json`: Python / native one rail / native two rails / Python. Two rails cut 125K cold
  prefill from 67.9 s to 62.0 s (code) and 67.2 s to 61.2 s (prose) and raised every 125K decode cell by
  1.4 to 2.5%. Cold prefill is one sample per cell.

## Qualification of the patched build

Tool calls 8/8, vision 2/2, strict protocol 10/11 (only a tool-result type check fails, as on the
unpatched engine), reasoning 8/8 at high effort, 17/17 retained-prefix growth and revisit checks, 16/16
held-out outputs identical to the reference, and a shared system prefix reused 98,304 cached tokens on
return. The same checks passed again on the live serve after the switch.

## Limits

One pair of nodes. Two tasks per context size in the A/B trials. Comparators ran on their own hardware
and network: tonyd2wild's lane also uses two rails; lucataco's uses two RoCE ports. Benchmark tools build
prompts differently, which is why our prefill is far ahead on kit_bench and behind on v41bench_tf.
