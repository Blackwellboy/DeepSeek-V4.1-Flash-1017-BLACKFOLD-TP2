# Why native TensorFold took more rounds than the Python engine

On the same 125K-context code prompt, at temperature 0, both engines produced the same 1,238 tokens.
The Python TP2 engine took 300 verification rounds. Native TensorFold `629c524` took 309 to 311, and on
that cell it was 0.8 to 2.5% slower even though each of its rounds was faster. Two separate differences
explain the gap.

## 1. The overhead term (explains all of prose)

Both engines choose how many drafted tokens to verify by maximising expected accepted tokens per
millisecond. The Python engine's DeepSeek decoder (`families/deepseek_v41/cuda/multi.py`) adds a fixed
3.5 ms to each candidate's cost. Native (`core/lanes/windows.zig:allocate`) adds a live overhead estimate
that, on this pair, settles near zero. A larger fixed term favours longer windows. With the fixed 3.5 ms
(patch 0002), native prose matched Python exactly: 434 rounds, 796 drafted, 443 accepted (native counts
one token past the stop).

## 2. Copy windows (explains most of the rest)

With the overhead matched, native code still took 308 rounds. The Python DeepSeek decoder never proposes
copies. Native's lanes engine runs a suffix-lookup proposer: once the output repeats at least 8 tokens of
the context, it proposes the continuation from the context instead of the draft head's tokens, and copy
windows skip the cost-model allocation. DeepSeek's window is 6 rows, so a copy is capped at 5 tokens, the
same as a full draft block. It cannot beat a good draft round, only replace one.

Evidence, from a diagnostic build that logged every allocation:

- Each code request had 24 rounds with no allocation event; prose had none.
- A CPU replay of the suffix-lookup rule (ngram 3, minimum match 4, entry match 8, width 5) on the saved
  prompt and output token IDs finds copy firing inside all 24 unlogged code rounds, at only 2 of 286
  logged head rounds (consistent with the proposer's silencing after failed copies, not modelled), and
  never on the prose output.
- Those copy rounds sat where the preceding head round had chosen all 5 drafts in 14 of 17 cases. Full
  head rounds yielded about 5.2 tokens there; the copy rounds about 3.7. That accounts for roughly 7
  extra rounds, close to the 8 observed. This last figure is an estimate.

## Result

With both patches, native's solo counters match Python's on every tested prompt: 125K code 300 rounds and
1,196 drafted, 125K prose 434 and 796, and the 8K prompts too. In a Python / native / Python trial, all 96
outputs were identical by full token IDs and native was faster in 9 of 10 cells (125K solo code +3.8%,
125K two-stream code +9.0%); 8K solo prose was 1.9% slower on one rail and faster on two.

Limits: one code and one prose prompt per context size, temperature 0. Copy windows may help workloads not
tested here (long verbatim quoting). Both patches are opt-in; upstream defaults are unchanged.
