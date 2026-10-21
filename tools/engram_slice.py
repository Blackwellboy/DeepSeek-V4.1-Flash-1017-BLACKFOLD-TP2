#!/usr/bin/env python3
"""engram_slice.py: keep only one TP rank's part of DeepSeek-V4.1's Engram tables on a node.

Each TP rank of the native engine reads only the rows its hash columns map to: columns [lo, hi) of each Engram layer,
which are one contiguous run of table rows (offsets[lo] .. offsets[hi-1] + primes[hi-1]). This tool copies that run
of each layer's weight and scale tables into a new safetensors file whose __metadata__ says the run's first row
(engram_first_row). A node then keeps only its own rank's slice: about half of the ~189 GiB on a TP2 pair. The bytes
are the original ones, so the model's outputs cannot change.

  copy mode:     engram_slice.py --config MODEL/config.json --src ENGRAM_DIR --rank R --world W --out OUT_DIR
  in-place mode: engram_slice.py --config MODEL/config.json --src ENGRAM_DIR --rank R --world W --punch-in-place --yes-punch
  check mode:    engram_slice.py --config MODEL/config.json --src ENGRAM_DIR --rank R --world W --check

Copy mode writes the rank's rows to new files in OUT_DIR (point TF_DS_ENGRAM there); it needs room for them.
In-place mode punches out (fallocate PUNCH_HOLE) the rows the rank never reads from the original files: no copy and no
free space needed, every offset unchanged, about half of the tables' disk back. It cannot be undone except by
downloading the shards again, and the shards' published SHA-256 no longer match afterwards. Only the embed weight and
scale rows are touched; the header and the projection weights in the same files are not.
Check mode reads nothing but file headers and hole maps: it exits 0 when this node's tables hold every row rank R
uses (whole tables, this rank's slices, or tables punched for this rank), and 1 otherwise. The deploy script runs it on
both nodes before it starts the serve, so it protects engines without their own check too.
Either way the engine checks at start that its table holds every row its columns can hash to (no slice gap, no hole),
and refuses to start otherwise (EngramRowsMissing): a node prepared for the other rank cannot serve wrong rows.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import struct
import sys

KEY = re.compile(r"layers\.(\d+)\.engram\.embed\.(weight|scale)$")
SIZES = {"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U8": 1, "I8": 1, "BF16": 2, "F16": 2, "F32": 4}


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin with the bases the engine uses (engram.zig isPrime)."""
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def primes(cfg: dict) -> list[list[int]]:
    """Bucket primes [layer][(ngram - 1) * heads], drawn upward from engram_vocab_size, none reused (engram.zig primes)."""
    used: set[int] = set()
    out = []
    for _ in cfg["engram_layer_ids"]:
        row = []
        for _o in range(cfg["engram_max_ngram_size"] - 1):
            cur = cfg["engram_vocab_size"] - 1
            for _h in range(cfg["engram_n_heads"]):
                cur += 1
                while not is_prime(cur) or cur in used:
                    cur += 1
                used.add(cur)
                row.append(cur)
        out.append(row)
    return out


def rank_rows(cfg: dict, rank: int, world: int) -> dict[int, tuple[int, int]]:
    """Layer id -> the whole-table rows [r0, r1) this rank's hash columns use (the engine's EngramHost lo/hi)."""
    ps = primes(cfg)
    totals = cfg.get("engram_num_embeddings")
    out = {}
    for j, layer in enumerate(cfg["engram_layer_ids"]):
        offs, acc = [], 0
        for p in ps[j]:
            offs.append(acc)
            acc += p
        if totals is not None and totals[j] != acc:
            raise SystemExit(f"layer {layer}: primes sum to {acc} rows but config says {totals[j]}; refusing")
        cols = len(ps[j])
        lo, hi = rank * cols // world, (rank + 1) * cols // world
        out[layer] = (offs[lo], offs[hi - 1] + ps[j][hi - 1])
    return out


def header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


def find_tables(src: str) -> dict[int, dict[str, tuple[str, int, dict]]]:
    """Layer id -> {"weight"|"scale": (file, data start, entry)} from every safetensors in src."""
    found: dict[int, dict[str, tuple[str, int, dict]]] = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(src, name)
        start, h = header(path)
        if "engram_first_row" in (h.get("__metadata__") or {}):
            raise SystemExit(f"{path} is already a slice; point --src at the original tables")
        for key, e in h.items():
            m = KEY.search(key)
            if m:
                found.setdefault(int(m.group(1)), {})[m.group(2)] = (path, start, {**e, "key": key})
    return found


def copy_range(src: str, dst, off: int, length: int, chunk: int = 64 << 20) -> None:
    with open(src, "rb") as f:
        f.seek(off)
        left = length
        while left:
            b = f.read(min(chunk, left))
            if not b:
                raise SystemExit(f"{src}: short read at {off + length - left}")
            dst.write(b)
            left -= len(b)


def slice_layer(layer: int, tabs: dict, r0: int, r1: int, out_dir: str, rank: int, world: int) -> str:
    rows = r1 - r0
    plan = []
    data_off = 0
    hdr: dict = {"__metadata__": {"engram_first_row": str(r0), "engram_rank": f"{rank}/{world}"}}
    for which in ("weight", "scale"):
        path, start, e = tabs[which]
        total, cols = e["shape"]
        if r1 > total:
            raise SystemExit(f"layer {layer} {which}: needs rows up to {r1}, table has {total}")
        row_bytes = cols * SIZES[e["dtype"]]
        n = rows * row_bytes
        hdr[e["key"]] = {"dtype": e["dtype"], "shape": [rows, cols], "data_offsets": [data_off, data_off + n]}
        plan.append((path, start + e["data_offsets"][0] + r0 * row_bytes, n, row_bytes, start + e["data_offsets"][0]))
        data_off += n
    raw = json.dumps(hdr, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    out = os.path.join(out_dir, f"engram-layer{layer}-rank{rank}of{world}.safetensors")
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for path, off, n, _rb, _base in plan:
            copy_range(path, f, off, n)
    os.replace(tmp, out)
    return out


def verify(out: str, layer: int, tabs: dict, r0: int, r1: int, samples: int) -> None:
    start, h = header(out)
    meta = h["__metadata__"]
    assert int(meta["engram_first_row"]) == r0
    rng = random.Random(layer * 7919 + r0)
    picks = {r0, r1 - 1} | {rng.randrange(r0, r1) for _ in range(samples)}
    for which in ("weight", "scale"):
        path, sstart, e = tabs[which]
        oe = h[e["key"]]
        row_bytes = oe["shape"][1] * SIZES[oe["dtype"]]
        with open(path, "rb") as fs, open(out, "rb") as fo:
            for r in sorted(picks):
                fs.seek(sstart + e["data_offsets"][0] + r * row_bytes)
                fo.seek(start + oe["data_offsets"][0] + (r - r0) * row_bytes)
                if fs.read(row_bytes) != fo.read(row_bytes):
                    raise SystemExit(f"{out}: {which} row {r} differs from the source")


FALLOC_FL_KEEP_SIZE, FALLOC_FL_PUNCH_HOLE = 0x01, 0x02


def _fallocate():
    import ctypes
    import ctypes.util
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    f = libc.fallocate
    f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64, ctypes.c_int64]
    f.restype = ctypes.c_int
    return f, ctypes


def has_hole(fd: int, lo: int, hi: int) -> bool:
    """Whether bytes [lo, hi) of fd contain a hole (SEEK_HOLE finds the first at or after lo; the file end is one)."""
    try:
        return os.lseek(fd, lo, os.SEEK_HOLE) < hi
    except OSError:
        return False


def punch_layer(layer: int, tabs: dict, r0: int, r1: int, samples: int, chunk: int = 1 << 30) -> int:
    """Punch out every row of this layer's weight and scale tables outside [r0, r1); returns bytes released."""
    fall, ctypes = _fallocate()
    released = 0
    for which in ("weight", "scale"):
        path, start, e = tabs[which]
        total, cols = e["shape"]
        rb = cols * SIZES[e["dtype"]]
        base = start + e["data_offsets"][0]
        if e["data_offsets"][1] - e["data_offsets"][0] != total * rb:
            raise SystemExit(f"{path}: {which} size does not match its shape")
        fd = os.open(path, os.O_RDWR)
        try:
            if has_hole(fd, base + r0 * rb, base + r1 * rb):
                raise SystemExit(f"{path}: {which} rows [{r0}, {r1}) already have holes: this node was prepared for "
                                 "another rank or world size; re-download the shard")
            rng = random.Random(layer * 104729 + r0)
            picks = sorted({r0, r1 - 1} | {rng.randrange(r0, r1) for _ in range(samples)})
            before = [os.pread(fd, rb, base + r * rb) for r in picks]
            blocks0 = os.fstat(fd).st_blocks
            for lo, hi in ((0, r0), (r1, total)):
                off, end = base + lo * rb, base + hi * rb
                while off < end:
                    n = min(chunk, end - off)
                    if fall(fd, FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE, off, n) != 0:
                        raise SystemExit(f"{path}: fallocate punch failed: {os.strerror(ctypes.get_errno())}")
                    off += n
            os.fsync(fd)
            after = [os.pread(fd, rb, base + r * rb) for r in picks]
            if before != after:
                raise SystemExit(f"{path}: kept {which} rows changed while punching; stop and re-download the shard")
            if has_hole(fd, base + r0 * rb, base + r1 * rb):
                raise SystemExit(f"{path}: kept {which} rows have a hole after punching")
            released += (blocks0 - os.fstat(fd).st_blocks) * 512
        finally:
            os.close(fd)
    return released


def check(cfg: dict, src: str, rank: int, world: int) -> list[str]:
    """Problems that would make rank `rank` read rows this node does not hold; empty when the node is ready."""
    problems = []
    ranges = rank_rows(cfg, rank, world)
    found: dict[int, dict[str, tuple[str, int, dict, int]]] = {}
    for name in sorted(os.listdir(src)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(src, name)
        start, h = header(path)
        first = int((h.get("__metadata__") or {}).get("engram_first_row", "0"))
        for key, e in h.items():
            m = KEY.search(key)
            if m:
                found.setdefault(int(m.group(1)), {})[m.group(2)] = (path, start, e, first)
    for layer, (r0, r1) in ranges.items():
        tabs = found.get(layer, {})
        if set(tabs) != {"weight", "scale"}:
            problems.append(f"layer {layer}: weight and scale tables not both in {src}")
            continue
        for which, (path, start, e, first) in sorted(tabs.items()):
            rows, cols = e["shape"]
            if r0 < first or r1 > first + rows:
                problems.append(f"layer {layer} {which}: holds rows [{first}, {first + rows}), rank {rank} needs [{r0}, {r1})")
                continue
            rb = cols * SIZES[e["dtype"]]
            base = start + e["data_offsets"][0]
            fd = os.open(path, os.O_RDONLY)
            try:
                if has_hole(fd, base + (r0 - first) * rb, base + (r1 - first) * rb):
                    problems.append(f"layer {layer} {which}: rows rank {rank} needs were punched out "
                                    "(this node was prepared for another rank)")
            finally:
                os.close(fd)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, help="the model's config.json")
    ap.add_argument("--src", required=True, help="folder with the original Engram tables (shards 47/48)")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--out", help="copy mode: new folder for this rank's slices")
    ap.add_argument("--punch-in-place", action="store_true", help="punch the other ranks' rows out of the --src files")
    ap.add_argument("--yes-punch", action="store_true", help="confirm: in-place punching is undone only by re-download")
    ap.add_argument("--verify-rows", type=int, default=256, help="random rows per tensor compared with the source")
    ap.add_argument("--dry-run", action="store_true", help="print the row ranges and sizes, write nothing")
    ap.add_argument("--check", action="store_true", help="exit 1 unless this node holds every row --rank uses")
    ap.add_argument("--layer", type=int, default=None,
                    help="slice only this Engram layer id (one at a time when the disk cannot hold both copies)")
    a = ap.parse_args(argv)
    if not 0 <= a.rank < a.world:
        raise SystemExit("--rank must be in [0, --world)")
    with open(a.config) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)  # DeepSeek-V4.1 keeps the Engram fields in text_config, as the engine reads them
    if a.check:
        if "engram_layer_ids" not in cfg:
            print("engram check: this config has no Engram tables; nothing to check")
            return 0
        problems = check(cfg, a.src, a.rank, a.world)
        for pr in problems:
            print(f"ENGRAM CHECK FAILED (rank {a.rank}): {pr}", file=sys.stderr)
        if not problems:
            print(f"engram check: this node holds every row rank {a.rank} of {a.world} uses")
        return 1 if problems else 0
    ranges = rank_rows(cfg, a.rank, a.world)
    if a.layer is not None:
        if a.layer not in ranges:
            raise SystemExit(f"--layer {a.layer}: Engram layers are {sorted(ranges)}")
        ranges = {a.layer: ranges[a.layer]}
    tables = find_tables(a.src)
    need = 0
    for layer, (r0, r1) in ranges.items():
        tabs = tables.get(layer, {})
        if set(tabs) != {"weight", "scale"}:
            raise SystemExit(f"layer {layer}: weight and scale tables not both found in {a.src}")
        size = sum((r1 - r0) * t[2]["shape"][1] * SIZES[t[2]["dtype"]] for t in tabs.values())
        whole = sum(t[2]["shape"][0] * t[2]["shape"][1] * SIZES[t[2]["dtype"]] for t in tabs.values())
        need += size
        print(f"layer {layer}: rows [{r0}, {r1}) of {tabs['weight'][2]['shape'][0]}  {size / 2**30:.1f} GiB of {whole / 2**30:.1f}")
    if a.dry_run:
        return 0
    if a.punch_in_place:
        if not a.yes_punch:
            raise SystemExit("refusing: --punch-in-place changes the original files for good; add --yes-punch")
        total = 0
        for layer, (r0, r1) in ranges.items():
            total += punch_layer(layer, tables[layer], r0, r1, a.verify_rows)
            print(f"layer {layer}: kept rows [{r0}, {r1}), the rest punched out (kept rows verified)")
        print(f"released {total / 2**30:.1f} GiB")
        return 0
    if not a.out:
        raise SystemExit("copy mode needs --out (or use --punch-in-place)")
    os.makedirs(a.out, exist_ok=True)
    for layer in ranges:
        if os.path.exists(os.path.join(a.out, f"engram-layer{layer}-rank{a.rank}of{a.world}.safetensors")):
            raise SystemExit(f"{a.out} already holds layer {layer}'s slice")
    others = [n for n in os.listdir(a.out) if n.endswith(".safetensors") and not n.startswith("engram-layer")]
    if others:
        raise SystemExit(f"{a.out} holds other safetensors ({others[0]}); use a folder of slices only")
    free = shutil.disk_usage(a.out).free
    if free < need + (2 << 30):
        raise SystemExit(f"{a.out}: {free / 2**30:.1f} GiB free, need {need / 2**30:.1f} GiB plus 2")
    for layer, (r0, r1) in ranges.items():
        out = slice_layer(layer, tables[layer], r0, r1, a.out, a.rank, a.world)
        verify(out, layer, tables[layer], r0, r1, a.verify_rows)
        print(f"wrote {out} (verified {a.verify_rows} + 2 rows a tensor)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
