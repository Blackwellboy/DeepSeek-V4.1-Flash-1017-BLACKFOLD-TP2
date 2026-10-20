# Patches against TensorFold `629c524`

Base: [bertholomus/TensorFold](https://github.com/bertholomus/TensorFold) branch `deepseek-v41-zig`,
commit `629c524a0ea57512a5342abb498d1b7c25353343`. Apply in order:

```bash
git checkout 629c524a0ea57512a5342abb498d1b7c25353343
git apply patches/0001-native-stream-tool-call-whitespace.patch
git apply patches/0002-dsv41-fixed-solo-head-overhead.patch
git apply patches/0003-dsv41-copy-windows-off.patch
```

Applied in this order to a fresh checkout, they reproduce the sources of the qualified build byte for
byte. With all three, upstream's host tests pass 224/224 (the 222 upstream and solo-overhead tests plus
two new ones) and both server suites pass (71 and 30).

## 0001: hold back leading whitespace in streamed tool calls

Changes `zig/src/server/chat.zig`; adds route fixtures to `zig/src/server/stream_preflight_test.zig`.

Some streamed replies sent two newlines as content before an otherwise correct tool call, so a client
that requires a tool-only answer saw stray content. The server now holds leading whitespace until it can
tell a tool call from prose, drops it only after a real call is parsed, and keeps ordinary indentation
and whitespace-only answers. Generation is unchanged. Measured: tool-call cases 4/8 to 8/8, strict
protocol suite 6/11 to 10/11 (the remaining failure is a tool-result type check unrelated to this patch).

## 0002: optional fixed draft allowance for a lone stream

Changes `zig/src/core/lanes/config.zig`, `zig/src/core/lanes/windows.zig`,
`zig/src/families/dsv41/lanes.zig`.

The allocator picks how many drafted tokens to verify by expected tokens per millisecond, adding an
overhead term. Upstream uses a live estimate, which on this pair settles near zero; the Python TP2 engine
uses a fixed 3.5 ms. Adds an optional `solo_head_overhead_ms` (default null, upstream behaviour). The
DeepSeek backend sets it to its existing `draft_ms` (3.5) while its draft head is loaded, and it is used
only for a truly solo head-draft window; shared, copied, forced and other backends' windows keep the live
estimate. Measured alone: 125K prose went from 439 to 434 verification rounds, exactly the Python
engine's count.

## 0003: optional switch for suffix-lookup copy windows

Changes `zig/src/core/lanes/config.zig`, `zig/src/core/lanes/windows.zig`,
`zig/src/families/dsv41/lanes.zig`.

When the output repeats at least eight tokens of context, the native lanes engine proposes a copied
continuation instead of the draft head's tokens. For DeepSeek the window is six rows, so a copy is capped
at five tokens, the same as a full draft block, and it fires in predictable stretches where the draft
head was already filling all five. Adds an optional `copy_proposals` (default true, upstream behaviour);
the DeepSeek backend sets it false while its draft head is loaded, matching the Python engine, which
never proposes copies. Measured with 0002: 125K code went from 308 to 300 rounds, the Python engine's
count, and every output stayed identical. See [`../docs/COPY_WINDOWS.md`](../docs/COPY_WINDOWS.md).

TensorFold's LICENSE and NOTICE at the base revision are in this folder.
