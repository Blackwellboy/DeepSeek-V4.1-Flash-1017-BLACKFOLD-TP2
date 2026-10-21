# Tools

Both are standard-library only and read-only toward your nodes, except where noted.

## `bf_lab.py`: record and compare a serve

A run records each case's full generated token IDs, plus the engine's counters: tokens per second, time to first
token, prefill seconds, cached tokens, rounds, drafted and accepted. It writes a JSON receipt.

```bash
python3 tools/bf_lab.py plan --suite default          # what would be sent; sends nothing
python3 tools/bf_lab.py record --base http://HEAD:18191 --label before --out before.json --yes-quiet-window
python3 tools/bf_lab.py record --base http://HEAD:18191 --label after  --out after.json  --yes-quiet-window
python3 tools/bf_lab.py compare before.json after.json --md report.md
python3 tools/bf_lab.py selftest                       # checks the tool itself against a mock server
```

**Comparing runs:**
- `compare` exits 1 if any case's token IDs differ.
- `--rounds-must-match` also fails when a case's verification rounds move. Use it when a change must keep the
  engine's schedule, for example an engine pin moved with the same settings.

**Recording runs:**
- `record` sends nothing without `--yes-quiet-window`. Other traffic on the serve skews the timings and shares the
  draft rounds, so run it when nothing else is using the pair.
- Suites: `quick`, `default` (code, prose, structured, plus 8K and 32K long-context cases) and `long` (8K, 32K, 128K).
- Each request is temperature 0 with thinking off unless you pass `--temperature` and `--thinking`.
  - `--temperature 1.0 --thinking low` is the regime real chat and agent use runs in.
  - The serve seeds sampling from the prompt, so sampled runs are repeatable and the token-ID check still applies.
- It never sends `top_k`, `top_p` or `min_p`; the serve's defaults apply.

## `bf_preflight.sh`: one node's state

```bash
bash tools/bf_preflight.sh --binary <path to tensorfold-native or tf-dsv41-lanes> --engram <Engram folder>
```

It prints `key=value` lines:
- uptime and boot id
- available memory and fragmentation (`buddyinfo` orders 9-10)
- GPU clock, max clock, power, temperature and active throttle reasons
- each RoCE port's state and IPv4 RoCE v2 GID index
- the binary's SHA-256
- where the Engram tables live
- running containers

Run it on both nodes and compare them: the slower node sets the pace for the pair.

`--burn SECONDS` runs a short GPU load to expose a clock stuck low. That is load, so use it only with the serve
stopped.

## `bf_window.sh`: compare serve configurations in one quiet window

```bash
tools/bf_window.sh --arms arms.txt --base http://HEAD:18191 --out window --i-own-this-window
```

`arms.txt` holds one configuration per line: a name, a complete `blackfold.env`, and optional `KEY=VALUE` engine
switches for both ranks. The first line is the reference.

**What it does:**
- Starts each arm in turn and warms it up.
- Records it with `bf_lab.py` at T0 (greedy, thinking off) and T1 (temperature 1.0, thinking on).
- Restores the reference arm at the end and checks it with a live generation. It restores even after a failure.
- Writes `report.md`: every arm against the reference, token for token, with rounds and tok/s.

**When it won't run:** while `/health` shows live connections. It restarts the serve once per arm, so give it a
window nobody else is using.

Engine switches reach the containers through `EXTRA_ENV` in `blackfold.env`.

## `engram_slice.py`: keep only each node's half of the Engram tables

See [docs/ENGRAM_DISK.md](../docs/ENGRAM_DISK.md). It has four modes:
- `--dry-run`: show what each node would keep.
- `--punch-in-place --yes-punch`: release the other rank's rows in place.
- `--out DIR`: copy this rank's rows to new files.
- `--check`: exits 1 unless this node holds every row its rank reads. `deploy/blackfold-native start` runs it on both
  nodes.

`tools/test_engram_slice.py` tests every mode end to end on synthetic tables.

