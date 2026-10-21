#!/usr/bin/env python3
"""End-to-end test of engram_slice.py on synthetic tables: both ranks' slices put back together are the original
bytes, each slice carries its first row, and the tool refuses a wrong config, a slice as source and a used folder.
Run: python3 -I candidate2/tools/test_engram_slice.py"""

import importlib.util
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("engram_slice", os.path.join(HERE, "engram_slice.py"))
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)


def write_st(path, tensors, meta=None):
    """tensors: [(key, dtype, shape, bytes)] -> a safetensors file."""
    hdr, off, blob = {}, 0, b""
    if meta:
        hdr["__metadata__"] = meta
    for key, dt, shape, data in tensors:
        hdr[key] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
        blob += data
    raw = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + blob)


def read_tensor(path, key):
    start, h = es.header(path)
    e = h[key]
    with open(path, "rb") as f:
        f.seek(start + e["data_offsets"][0])
        return f.read(e["data_offsets"][1] - e["data_offsets"][0]), h


def main():
    cfg = {"engram_layer_ids": [3, 7], "engram_max_ngram_size": 3, "engram_vocab_size": 50, "engram_n_heads": 2}
    ps = es.primes(cfg)
    cfg["engram_num_embeddings"] = [sum(r) for r in ps]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src")
        os.makedirs(src)
        whole = {}
        tensors = []
        for layer, total in zip(cfg["engram_layer_ids"], cfg["engram_num_embeddings"]):
            w = bytes((layer * 31 + i * 7) % 251 for i in range(total * 6))
            s = bytes((layer * 13 + i * 5) % 241 for i in range(total))
            whole[layer] = (w, s)
            tensors += [(f"layers.{layer}.engram.embed.weight", "U8", [total, 6], w),
                        (f"layers.{layer}.engram.embed.scale", "U8", [total, 1], s)]
        write_st(os.path.join(src, "model-00047.safetensors"), tensors[:2] + [("layers.3.mlp.other", "U8", [2, 2], b"abcd")])
        write_st(os.path.join(src, "model-00048.safetensors"), tensors[2:])
        conf = os.path.join(d, "config.json")
        with open(conf, "w") as f:
            json.dump({"text_config": cfg}, f)
        outs = []
        for rank in (0, 1):
            out = os.path.join(d, f"rank{rank}")
            assert es.main(["--config", conf, "--src", src, "--rank", str(rank), "--world", "2", "--out", out,
                            "--verify-rows", "50"]) == 0
            outs.append(out)
        for layer in cfg["engram_layer_ids"]:
            parts_w, parts_s, firsts = b"", b"", []
            for rank, out in enumerate(outs):
                path = os.path.join(out, f"engram-layer{layer}-rank{rank}of2.safetensors")
                w, h = read_tensor(path, f"layers.{layer}.engram.embed.weight")
                s, _ = read_tensor(path, f"layers.{layer}.engram.embed.scale")
                firsts.append(int(h["__metadata__"]["engram_first_row"]))
                assert h[f"layers.{layer}.engram.embed.weight"]["shape"][0] == len(s)
                parts_w += w
                parts_s += s
            assert firsts[0] == 0 and 0 < firsts[1] < len(whole[layer][1]), firsts
            assert firsts[1] == len(read_tensor(os.path.join(outs[0], f"engram-layer{layer}-rank0of2.safetensors"),
                                                f"layers.{layer}.engram.embed.scale")[0])
            assert parts_w == whole[layer][0] and parts_s == whole[layer][1], f"layer {layer}: halves differ from the whole"
            # the engine's view: rank 1's columns hash into its slice exactly
            r0, r1 = es.rank_rows(cfg, 1, 2)[layer]
            assert (r0, r1 - r0) == (firsts[1], len(whole[layer][1]) - firsts[1])
        # check mode on whole tables (either rank) and on each rank's copied slices (only that rank)
        for rank in (0, 1):
            assert es.main(["--config", conf, "--src", src, "--rank", str(rank), "--world", "2", "--check"]) == 0
            assert es.main(["--config", conf, "--src", outs[rank], "--rank", str(rank), "--world", "2", "--check"]) == 0
            assert es.main(["--config", conf, "--src", outs[rank], "--rank", str(1 - rank), "--world", "2", "--check"]) == 1
        # one layer at a time gives the same file as both at once
        one = os.path.join(d, "one")
        for layer in cfg["engram_layer_ids"]:
            assert es.main(["--config", conf, "--src", src, "--rank", "1", "--world", "2", "--out", one, "--layer", str(layer)]) == 0
            name = f"engram-layer{layer}-rank1of2.safetensors"
            with open(os.path.join(one, name), "rb") as f1, open(os.path.join(outs[1], name), "rb") as f2:
                assert f1.read() == f2.read(), f"--layer {layer} differs"
        # refusals
        bad = dict(cfg, engram_num_embeddings=[cfg["engram_num_embeddings"][0] + 1, cfg["engram_num_embeddings"][1]])
        badconf = os.path.join(d, "bad.json")
        with open(badconf, "w") as f:
            json.dump({"text_config": bad}, f)
        for argv, why in (
            (["--config", badconf, "--src", src, "--rank", "0", "--world", "2", "--out", os.path.join(d, "x")], "config"),
            (["--config", conf, "--src", outs[0], "--rank", "0", "--world", "2", "--out", os.path.join(d, "y")], "slice as source"),
            (["--config", conf, "--src", src, "--rank", "0", "--world", "2", "--out", outs[0]], "slice already there"),
            (["--config", conf, "--src", src, "--rank", "0", "--world", "2", "--out", os.path.join(d, "z"), "--layer", "5"],
             "not an Engram layer"),
        ):
            try:
                es.main(argv)
            except SystemExit as e:
                assert e.code not in (0, None), why
            else:
                raise AssertionError(f"no refusal: {why}")
    # in-place mode on wide rows (so whole blocks free): rank 1 keeps its rows byte for byte, the rest reads as zeros
    with tempfile.TemporaryDirectory() as d:
        cfg2 = dict(cfg)
        src = os.path.join(d, "src")
        os.makedirs(src)
        whole = {}
        for layer, total in zip(cfg2["engram_layer_ids"], cfg2["engram_num_embeddings"]):
            w = bytes((layer * 31 + i * 7) % 251 + 1 for i in range(total * 8192))
            s = bytes((layer * 13 + i * 5) % 241 + 1 for i in range(total * 64))
            whole[layer] = (w, s)
            write_st(os.path.join(src, f"shard-{layer}.safetensors"),
                     [(f"layers.{layer}.engram.embed.weight", "U8", [total, 8192], w),
                      (f"layers.{layer}.engram.embed.scale", "U8", [total, 64], s),
                      (f"layers.{layer}.engram.wkv.weight", "U8", [2, 2], b"wkv!")])
        conf = os.path.join(d, "config.json")
        with open(conf, "w") as f:
            json.dump({"text_config": cfg2}, f)
        base = ["--config", conf, "--src", src, "--world", "2", "--punch-in-place"]
        try:
            es.main(base + ["--rank", "1"])
        except SystemExit as e:
            assert e.code not in (0, None)
        else:
            raise AssertionError("punch without --yes-punch was not refused")
        before = sum(os.stat(os.path.join(src, n)).st_blocks for n in os.listdir(src))
        assert es.main(base + ["--rank", "1", "--yes-punch", "--verify-rows", "40"]) == 0
        after = sum(os.stat(os.path.join(src, n)).st_blocks for n in os.listdir(src))
        assert after < before, (before, after)
        for layer in cfg2["engram_layer_ids"]:
            path = os.path.join(src, f"shard-{layer}.safetensors")
            r0, r1 = es.rank_rows(cfg2, 1, 2)[layer]
            w, _ = read_tensor(path, f"layers.{layer}.engram.embed.weight")
            s, _ = read_tensor(path, f"layers.{layer}.engram.embed.scale")
            assert w[r0 * 8192:r1 * 8192] == whole[layer][0][r0 * 8192:r1 * 8192], "kept weight rows changed"
            assert s[r0 * 64:r1 * 64] == whole[layer][1][r0 * 64:r1 * 64], "kept scale rows changed"
            assert w[:r0 * 8192] == bytes(r0 * 8192), "other rank's weight rows not punched"
            assert read_tensor(path, f"layers.{layer}.engram.wkv.weight")[0] == b"wkv!", "projection touched"
        # check mode: this node now serves rank 1 only
        chk = ["--config", conf, "--src", src, "--world", "2", "--check"]
        assert es.main(chk + ["--rank", "1"]) == 0, "check refused the right rank"
        assert es.main(chk + ["--rank", "0"]) == 1, "check passed the wrong rank"
        # the other rank cannot be prepared on this node now: its rows are gone
        try:
            es.main(base + ["--rank", "0", "--yes-punch"])
        except SystemExit as e:
            assert e.code not in (0, None)
        else:
            raise AssertionError("punching rank 0 over rank 1's punched files was not refused")
    print("engram_slice test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
