#!/usr/bin/env python3
"""bf-lab: record a BLACKFOLD serve's outputs and counters, and compare two recordings.

The gate every speed change is held to: greedy outputs identical by full token ids, plus the draft counters that
explain a speed change (rounds, drafted, accepted). One command records a receipt; another compares two receipts.

  bf_lab.py plan    --suite default                         # what would be sent, sends nothing
  bf_lab.py record  --base URL --label A --out a.json --yes-quiet-window [--suite default] [--reps 2] [--concurrency 1]
                    [--temperature 1.0 --thinking minimal]   # the real regime: sampled, thinking on
  bf_lab.py compare a.json b.json [--md r.md] [--rounds-must-match]   # exit 1 if token ids (or rounds) differ
  bf_lab.py selftest                                        # record + compare against an in-process mock server

Reads from the native server's reply: tensorfold.token_ids / token_sha / tokens_per_second / prefill_seconds /
time_to_first_token, speculative.rounds / drafted / accepted, usage.prompt_tokens_details.cached_tokens.

Rules this tool keeps on its own:
  * Sends nothing without --yes-quiet-window (a recording is load on the serve; it needs a quiet window and the owner's OK).
  * temperature 0 and thinking off unless asked; never top_k / top_p / min_p in the request (the server's defaults;
    never top_k 0 or -1).
  * Streams off. One request per case per rep at --concurrency 1.
Standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import http.server
import json
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

VERSION = 1

# ---------------------------------------------------------------------------------------------------------------- suite

_WORDS = ("amber basin cobalt delta ember fjord granite harbor island juniper kestrel lantern meadow nickel orchard "
          "pylon quartz river summit tundra umber valley willow xenon yarrow zephyr").split()


def _filler(words: int, seed: int) -> str:
    """Deterministic prose-like filler of about `words` words (a long-context body that is the same on every run)."""

    rng = random.Random(seed)
    out, line = [], []
    for i in range(words):
        line.append(rng.choice(_WORDS))
        if len(line) >= 12:
            out.append(" ".join(line).capitalize() + ".")
            line = []
        if i % 400 == 399:
            out.append(f"\n\nSection {i // 400 + 1}.\n")
    if line:
        out.append(" ".join(line).capitalize() + ".")
    return " ".join(out)


def suite(name: str) -> list[dict[str, Any]]:
    """The named suite's cases: id, kind, messages, max_tokens."""

    code = [
        "Write a Python function that parses an ISO-8601 duration like P3DT4H5M into seconds, with tests.",
        "Implement an LRU cache in Rust with get/put in O(1). Include a short explanation.",
        "Write a bash script that rotates log files older than 7 days into a dated tar.gz archive.",
        "Write a TypeScript debounce function with leading and trailing options, fully typed.",
    ]
    prose = [
        "Explain how a heat pump moves heat from a cold place to a warm one, for a curious teenager.",
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Summarize the trade-offs between tensor parallelism and pipeline parallelism for LLM serving.",
        "Describe a walk through a market at dawn, in vivid detail.",
    ]
    structured = [
        "Return a JSON array of 8 fictional products with fields name, price, sku and tags. JSON only.",
        "Make a markdown table of the planets with columns: name, mean radius km, moons, notes.",
    ]
    cases: list[dict[str, Any]] = []

    def add(kind: str, i: int, text: str, max_tokens: int, system: str | None = None) -> None:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
        cases.append({"id": f"{kind}-{i}", "kind": kind, "messages": msgs, "max_tokens": max_tokens})

    if name in ("default", "quick"):
        n = 2 if name == "quick" else len(code)
        for i, t in enumerate(code[:n]):
            add("code", i, t, 400)
        for i, t in enumerate(prose[:n]):
            add("prose", i, t, 400)
        for i, t in enumerate(structured[: max(1, n // 2)]):
            add("structured", i, t, 300)
        if name == "default":
            add("long8k", 0, "Read the document and list its section numbers.\n\n" + _filler(6000, 8), 200)
            add("long32k", 0, "Read the document and list its section numbers.\n\n" + _filler(24000, 32), 200)
        return cases
    if name == "long":
        for words, tag in ((6000, "8k"), (24000, "32k"), (96000, "128k")):
            add(f"long{tag}", 0, "Read the document and list its section numbers.\n\n" + _filler(words, words), 200)
        return cases
    raise SystemExit(f"unknown suite {name!r} (default, quick, long, or a JSON file)")


def load_suite(arg: str) -> list[dict[str, Any]]:
    if arg.endswith(".json"):
        with open(arg, encoding="utf-8") as f:
            return json.load(f)
    return suite(arg)


def suite_digest(cases: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------- record

def _post(url: str, body: dict[str, Any], timeout: float, key: str | None) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: float, key: str | None) -> Any:
    req = urllib.request.Request(url)
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            try:
                return json.loads(raw)
            except ValueError:
                return raw.decode(errors="replace")[:2000]
    except (urllib.error.URLError, TimeoutError) as e:
        return {"error": str(e)}


def request_body(case: dict[str, Any], model: str, salt: str, temperature: float = 0.0,
                 thinking: str = "off") -> dict[str, Any]:
    """One case's request. At temperature > 0 the native server seeds sampling from the prompt ids (server.zig
    resolveSampling -> seedFor), so a sampled reply is as repeatable as a greedy one and the token-id gate still holds.
    That is also the only way to exercise the sampler: at temperature 0 the server takes the greedy path."""
    msgs = json.loads(json.dumps(case["messages"]))
    if salt:
        msgs[0]["content"] = f"[{salt}] " + msgs[0]["content"]
    kwargs: dict[str, Any] = {"enable_thinking": False}
    if thinking != "off":
        kwargs = {"enable_thinking": True, "reasoning_effort": thinking}
    return {
        "model": model,
        "messages": msgs,
        # thinking replies spend tokens on reasoning first; give them room for an answer too
        "max_tokens": case["max_tokens"] if thinking == "off" else case["max_tokens"] * 4,
        "temperature": temperature,
        "stream": False,
        "return_token_ids": True,
        "chat_template_kwargs": kwargs,
    }


def one(base: str, model: str, case: dict[str, Any], salt: str, timeout: float, key: str | None,
        temperature: float = 0.0, thinking: str = "off") -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        r = _post(base.rstrip("/") + "/v1/chat/completions", request_body(case, model, salt, temperature, thinking),
                  timeout, key)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return {"error": str(e)}
    wall = time.perf_counter() - t0
    tf = r.get("tensorfold") or {}
    sp = r.get("speculative") or {}
    us = r.get("usage") or {}
    ids = tf.get("token_ids")
    choice = (r.get("choices") or [{}])[0]
    return {
        "wall_s": round(wall, 4),
        "finish_reason": choice.get("finish_reason"),
        "token_sha": tf.get("token_sha") or (hashlib.sha256(json.dumps(ids).encode()).hexdigest() if ids else None),
        "token_ids": ids,
        "completion_tokens": us.get("completion_tokens"),
        "prompt_tokens": us.get("prompt_tokens"),
        "cached_tokens": (us.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "tps": tf.get("tokens_per_second"),
        "ttft_s": tf.get("time_to_first_token"),
        "prefill_s": tf.get("prefill_seconds"),
        "rounds": sp.get("rounds"),
        "drafted": sp.get("drafted"),
        "accepted": sp.get("accepted"),
    }


def record(args: argparse.Namespace) -> int:
    cases = load_suite(args.suite)
    if not args.yes_quiet_window:
        print("refusing to send: a recording is load on the serve. Re-run with --yes-quiet-window once the owner has OK'd "
              "a quiet window. (`plan` shows what would be sent.)", file=sys.stderr)
        return 2
    base, key = args.base, args.api_key
    models = _get(base.rstrip("/") + "/v1/models", 10, key)
    model = args.model or ((models.get("data") or [{}])[0].get("id") if isinstance(models, dict) else None)
    if not model:
        print(f"no model id from {base}/v1/models: {models}", file=sys.stderr)
        return 2
    receipt: dict[str, Any] = {
        "bf_lab": VERSION,
        "label": args.label,
        "notes": args.notes,
        "engine": args.engine,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "suite": args.suite,
        "suite_digest": suite_digest(cases),
        "salt": args.salt,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "reps": args.reps,
        "concurrency": args.concurrency,
        "health_before": _get(base.rstrip("/") + "/health", 10, key),
        "cases": {},
    }
    jobs = [(c, rep) for rep in range(args.reps) for c in cases]
    results: dict[str, list[dict[str, Any]]] = {c["id"]: [] for c in cases}
    with cf.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = {ex.submit(one, base, model, c, args.salt, args.timeout, key, args.temperature, args.thinking): (c, rep)
                for c, rep in jobs}
        for f in cf.as_completed(futs):
            c, rep = futs[f]
            res = f.result()
            res["rep"] = rep
            results[c["id"]].append(res)
            mark = "ERR " + res["error"][:80] if "error" in res else f"{res['completion_tokens']} tok {res['tps']} tok/s rounds={res['rounds']}"
            print(f"  {c['id']:<14} rep {rep}: {mark}", file=sys.stderr)
    for c in cases:
        reps = sorted(results[c["id"]], key=lambda r: r["rep"])
        for r in reps[1:]:
            r.pop("token_ids", None)  # rep 0 keeps the full ids; later reps keep their sha
        receipt["cases"][c["id"]] = {"kind": c["kind"], "max_tokens": c["max_tokens"], "runs": reps}
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    receipt["health_after"] = _get(base.rstrip("/") + "/health", 10, key)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1)
    errors = sum(1 for v in results.values() for r in v if "error" in r)
    unstable = [cid for cid, v in results.items() if len({r.get("token_sha") for r in v if "error" not in r}) > 1]
    print(f"wrote {args.out}: {len(cases)} cases x {args.reps} reps, {errors} errors, "
          f"{len(unstable)} cases whose reps disagree{': ' + ', '.join(unstable) if unstable else ''}", file=sys.stderr)
    return 1 if errors or unstable else 0


# ---------------------------------------------------------------------------------------------------------------- compare

def _med(xs: list[Any]) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def _pct(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return "n/a"
    return f"{(b - a) / a * 100:+.1f}%"


def compare(args: argparse.Namespace) -> int:
    with open(args.a, encoding="utf-8") as f:
        A = json.load(f)
    with open(args.b, encoding="utf-8") as f:
        B = json.load(f)
    lines = [f"# bf-lab compare: {A.get('label')} -> {B.get('label')}", ""]
    if any(A.get(k) != B.get(k) for k in ("suite_digest", "salt", "temperature", "thinking")):
        lines += ["**Different suites, salts, temperatures or thinking settings: token ids are not comparable.**", ""]
    lines += [f"- A: {A.get('label')} engine={A.get('engine')} {A.get('started')} concurrency={A.get('concurrency')}",
              f"- B: {B.get('label')} engine={B.get('engine')} {B.get('started')} concurrency={B.get('concurrency')}", "",
              "| case | ids | rounds A→B | accepted A→B | tok/s A→B | Δ tok/s | ttft A→B (s) | cached B |",
              "|---|---|---|---|---|---|---|---|"]
    mismatch = 0
    rounds_moved = 0
    for cid, ca in A["cases"].items():
        cb = B["cases"].get(cid)
        if cb is None:
            lines.append(f"| {cid} | missing in B | | | | | | |")
            mismatch += 1
            continue
        ra = [r for r in ca["runs"] if "error" not in r]
        rb = [r for r in cb["runs"] if "error" not in r]
        sa, sb = {r.get("token_sha") for r in ra}, {r.get("token_sha") for r in rb}
        same = bool(ra and rb) and sa == sb and len(sa) == 1
        if not same:
            mismatch += 1
            ids_a = ra[0].get("token_ids") if ra else None
            ids_b = rb[0].get("token_ids") if rb else None
            where = ""
            if ids_a and ids_b:
                n = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), min(len(ids_a), len(ids_b)))
                where = f" @{n}"
            verdict = f"DIFF{where}"
        else:
            verdict = "same"
            rda, rdb = _med([r.get("rounds") for r in ra]), _med([r.get("rounds") for r in rb])
            if args.rounds_must_match and rda != rdb:
                rounds_moved += 1
                verdict = "same ids, ROUNDS MOVED"
        ta, tb = _med([r.get("tps") for r in ra]), _med([r.get("tps") for r in rb])
        lines.append(
            f"| {cid} | {verdict} | {_med([r.get('rounds') for r in ra])}→{_med([r.get('rounds') for r in rb])} | "
            f"{_med([r.get('accepted') for r in ra])}→{_med([r.get('accepted') for r in rb])} | "
            f"{ta and round(ta, 1)}→{tb and round(tb, 1)} | {_pct(ta, tb)} | "
            f"{_fmt(_med([r.get('ttft_s') for r in ra]))}→{_fmt(_med([r.get('ttft_s') for r in rb]))} | "
            f"{_med([r.get('cached_tokens') for r in rb])} |")
    ok = mismatch == 0 and rounds_moved == 0
    lines += ["", f"**{'PASS' if ok else 'FAIL'}**: {mismatch} case(s) differ by token ids"
              + (f", {rounds_moved} case(s) moved rounds (--rounds-must-match)" if args.rounds_must_match else "") + "."
              " Rounds that move with identical ids are the speed story (e.g. a build missing TF_DS_SOLO_OVERHEAD /"
              " TF_DS_NO_COPIES is identical but takes more rounds)."]
    text = "\n".join(lines)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if ok else 1


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def plan(args: argparse.Namespace) -> int:
    cases = load_suite(args.suite)
    print(f"suite {args.suite} digest {suite_digest(cases)}: {len(cases)} cases")
    for c in cases:
        n = sum(len(m["content"]) for m in c["messages"])
        print(f"  {c['id']:<14} {c['kind']:<10} {n:>8} chars  max_tokens {c['max_tokens']}")
    print("each request by default: temperature 0, thinking off, stream off, return_token_ids, no top_k/top_p/min_p")
    return 0


# ---------------------------------------------------------------------------------------------------------------- selftest

class _Mock(http.server.BaseHTTPRequestHandler):
    """A stand-in serve: ids are a function of the prompt (and of `shift`, to fake a regression)."""

    shift = 0
    extra_rounds = 0

    def log_message(self, *a: Any) -> None:  # quiet
        pass

    def _send(self, obj: Any) -> None:
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send({"data": [{"id": "mock-blackfold"}]})
        else:
            self._send({"status": "ok"})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert "top_k" not in body and body["return_token_ids"] is True
        key = json.dumps([body["messages"], body["temperature"], body["chat_template_kwargs"]])
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        n = min(body["max_tokens"], 50)
        ids = [(seed + i * 7) % 1000 for i in range(n)]
        if type(self).shift and "short story" in json.dumps(body["messages"]):
            ids[-1] += type(self).shift
        rounds = n // 4 + type(self).extra_rounds
        self._send({
            "choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": n, "prompt_tokens_details": {"cached_tokens": 0}},
            "tensorfold": {"token_ids": ids, "token_sha": hashlib.sha256(str(ids).encode()).hexdigest(),
                           "tokens_per_second": 100.0 - type(self).extra_rounds, "time_to_first_token": 0.1,
                           "prefill_seconds": 0.05},
            "speculative": {"rounds": rounds, "drafted": n, "accepted": n - rounds},
        })


def selftest(_: argparse.Namespace) -> int:
    import os
    import tempfile

    def serve(shift: int, extra: int) -> tuple[http.server.ThreadingHTTPServer, str]:
        h = type("H", (_Mock,), {"shift": shift, "extra_rounds": extra})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"

    tmp = tempfile.mkdtemp(prefix="bf_lab_selftest_")
    p = build_parser()

    def rec(base: str, label: str) -> str:
        out = os.path.join(tmp, f"{label}.json")
        rc = record(p.parse_args(["record", "--base", base, "--label", label, "--out", out, "--suite", "quick",
                                  "--reps", "2", "--concurrency", "2", "--yes-quiet-window"]))
        assert rc == 0, f"record {label} rc={rc}"
        return out

    s1, b1 = serve(0, 0)
    a = rec(b1, "A")
    a2 = rec(b1, "A2")
    s1.shutdown()
    s2, b2 = serve(0, 3)  # same ids, more rounds: the missing-env-var case
    slow = rec(b2, "slow")
    s2.shutdown()
    s3, b3 = serve(5, 0)  # one prose case's last id moves: a correctness regression
    bad = rec(b3, "bad")
    s3.shutdown()

    def cmp(x: str, y: str, *flags: str) -> int:
        return compare(p.parse_args(["compare", x, y, *flags]))

    assert cmp(a, a2) == 0, "identical recordings must pass"
    s4, b4 = serve(0, 0)

    def rec_t(label: str) -> str:
        out = os.path.join(tmp, f"{label}.json")
        rc = record(p.parse_args(["record", "--base", b4, "--label", label, "--out", out, "--suite", "quick", "--reps",
                                  "2", "--temperature", "1.0", "--thinking", "minimal", "--yes-quiet-window"]))
        assert rc == 0, f"record {label} rc={rc}"
        return out

    t1, t2 = rec_t("T1a"), rec_t("T1b")
    s4.shutdown()
    assert cmp(t1, t2) == 0, "two sampled thinking recordings of the same build must pass"
    assert cmp(a, t1) == 1, "greedy vs sampled must not compare as the same"
    assert cmp(a, slow) == 0, "more rounds with identical ids passes the id gate (rounds column shows it)"
    assert cmp(a, slow, "--rounds-must-match") == 1, "with --rounds-must-match, more rounds must fail"
    assert cmp(a, a2, "--rounds-must-match") == 0, "identical recordings pass --rounds-must-match"
    assert cmp(a, bad) == 1, "a moved token id must fail"
    assert record(p.parse_args(["record", "--base", b1, "--label", "x", "--out", os.path.join(tmp, "x.json")])) == 2, \
        "no --yes-quiet-window must refuse"
    print(f"\nselftest PASS ({tmp})")
    return 0


# ---------------------------------------------------------------------------------------------------------------- cli

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="bf_lab", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--base", required=True, help="serve base URL, e.g. http://HEAD:18191")
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--suite", default="default")
    r.add_argument("--reps", type=int, default=2)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--salt", default="", help="prefix for each case's first message (same salt on both arms)")
    r.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy. Real traffic runs at 1.0; only > 0 exercises the sampler (still token-identical)")
    r.add_argument("--thinking", default="off", choices=["off", "minimal", "low", "medium", "high", "xhigh", "max"],
                   help="off, or the reasoning effort to think at (the native server maps minimal..max to 25..100)")
    r.add_argument("--model", default=None)
    r.add_argument("--engine", default=None, help="engine SHA / build id to stamp into the receipt")
    r.add_argument("--notes", default="")
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--api-key", default=None)
    r.add_argument("--yes-quiet-window", action="store_true")
    r.set_defaults(fn=record)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--md", default=None)
    c.add_argument("--rounds-must-match", action="store_true",
                   help="also fail when a case's median rounds move (a pin move that must keep today's schedule)")
    c.set_defaults(fn=compare)
    pl = sub.add_parser("plan")
    pl.add_argument("--suite", default="default")
    pl.set_defaults(fn=plan)
    st = sub.add_parser("selftest")
    st.set_defaults(fn=selftest)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
