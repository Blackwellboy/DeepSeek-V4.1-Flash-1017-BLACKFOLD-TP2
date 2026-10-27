# DeepSeek-V4.1 chat template from sushi (GitHub issue #1)

`chat_template.jinja` is the DeepSeek-V4.1 chat template transcribed from the release's `encoding/encoding.py`
(chat and thinking modes, reasoning_effort low/high/max, DSML tool markup). `template_cases.json` holds 14 reference
renders produced by the release encoder with `dump_dsv41_template_cases.py`.

Source: [sushi](https://github.com/beamivalice/sushi) (MIT, Copyright (c) 2026 Theinruj Toranavikrai and David Dalcu),
commit `3fc47d8a460970b1e86b24de1cab3b0f2067bc72`, files `src/fixtures/dsv41_chat_template.jinja`,
`src/fixtures/dsv41_template_cases.json`, `tests/dump_dsv41_template_cases.py`. Unmodified.

Checked on 28 Oct 2026 with TensorFold's own Jinja renderer (patch 0004 adds `zig build test-dsv41-template` and the
`tf-dsv41-template-cases` checker):

| Template | 14 encoder cases, byte for byte |
|---|---|
| this `chat_template.jinja` (sushi) | 14/14 |
| the live serve's `chat_template.native.jinja` (Quick start step 4) | 14/14 |

So the template the recipe already builds renders exactly what DeepSeek's encoder produces for these shapes; this file
is an alternative that needs no `sed` rewrite.
