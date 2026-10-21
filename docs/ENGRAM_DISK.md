# Get ~95 GB of disk back on each node (Engram tables)

DeepSeek-V4.1's Engram tables are two files of ~94 GiB each (shards 47 and 48), and the setup puts both on **both**
nodes. On a TP2 pair, though, each node only ever reads its own half:
- The engine gives each rank half of each table's hash columns.
- Those columns map to one contiguous run of rows.

`tools/engram_slice.py --punch-in-place` releases the half a node never reads. It works in place, with no copy and no
free space needed. Every offset in the files stays the same, so the engine reads exactly the bytes it read before.

On our pair: **94.4 GiB released per node** (95% → 84% full). Four prompts that bypass the prompt cache gave
identical tokens before and after.

## Do it

On each node, with that node's rank. The head is rank 0 and the worker is rank 1, as in `deploy/blackfold-native`.

```bash
# see what would be kept and released (reads headers only)
python3 tools/engram_slice.py --config $MODEL_DIR/config.json --src $ENGRAM_DIR --rank R --world 2 --dry-run
# release the other rank's half
python3 tools/engram_slice.py --config $MODEL_DIR/config.json --src $ENGRAM_DIR --rank R --world 2 \
  --punch-in-place --yes-punch
```

The tool works out the row ranges from the model's own bucket primes. They sum to exactly the config's
`engram_num_embeddings`; if they don't, the tool refuses to run.

## What it protects

- **Untouched:** only the embed weight and scale rows are released. The file headers and the projection weights in
  the same files (`wkv`, `q_weight`, `k_weight`) stay as they are.
- **Checked before and after:** sampled kept rows are compared byte for byte, and the kept range is checked for
  holes.
- **Refusals:** it refuses to run twice for different ranks on one node.
- **Safe while serving:** a rank never reads the rows that get released.

## The start-up check

After this, a node can only serve its own rank. A node run as the other rank would read zeros. So
`deploy/blackfold-native start` now runs `engram_slice.py --check` on both nodes first, and refuses to start if either
node lacks a row its rank uses:

```
refusing: the head's Engram tables lack rank 0's rows
```

The check only reads file headers and hole maps; it takes a second. It passes on untouched tables. `ENGRAM_CHECK=0`
skips it, but don't do that after punching.

## Before you do it

- **This is permanent.** Undoing it means downloading shards 47 and 48 again (~95 GB each, per node).
- **`setup.sh`'s shard check will fail.** The shards no longer match their published SHA-256s.
- **Copy mode needs room.** If you have ~95 GB free and would rather keep the originals, `--out DIR` writes the
  rank's rows to new files instead (point `ENGRAM_DIR` at them).
