# Deploy files

| File | Where | What |
|---|---|---|
| `blackfold-native` | head node, next to `blackfold.env` | `start`, `stop`, `status` for the TP2 serve |
| `blackfold.env.example` | copy to `blackfold.env` | head/worker addresses, paths, image, optional binary pins |
| `blackfold-rail-align` | `/usr/local/bin/` on both nodes | addresses the twin link and aligns its GID index |
| `blackfold-rail.env.example` | copy to `/etc/blackfold-rail.env` on both nodes | twin device names and this node's twin address |
| `blackfold-rail-align.service` | `/etc/systemd/system/` on both nodes, enabled | runs the align script at boot |
| `90-blackfold-rail2` | `/etc/NetworkManager/dispatcher.d/` on both nodes, 0755 root | re-aligns after NetworkManager re-activates the twin |

Defaults match the serve we measured: context 1,048,576, 4 parallel streams, thinking on, temperature
1.0, top-p 0.95, top-k 20, replay floor, graphs, L2 prefetch and Engram AIO on.

Two settings differ from the measured serve:
- `max_tokens` for requests that send none is now `MAX_TOKENS` (default 32,768; it was 4,096). Thinking counts against
  it, and 4,096 could end a long reasoning reply before any answer.
- `API_KEY_FILE` (optional, a chmod 600 file with one key per line) turns on the engine's API key for every route
  except `/health`. The start script checks readiness through `/health`, so it works with or without a key.
`--restart no`: the serve does not start itself after a reboot.

These are parameterised versions of the scripts that run our pair, with host-specific values moved into
the two `.env` files. The one-rail fallback in `blackfold-native` has not been exercised in this form.
Stop any other serve on the same GPUs and port before `start`; the script only checks for its own
container name.
