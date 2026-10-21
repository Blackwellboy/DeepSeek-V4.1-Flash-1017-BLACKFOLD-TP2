#!/usr/bin/env bash
# bf_window.sh: measure several serve configurations ("arms") in one quiet window, the same way, then restore.
#
#   bf_window.sh --arms arms.txt --base http://HEAD:18191 --out DIR --i-own-this-window \
#     [--script deploy/blackfold-native] [--t0-suite default] [--t1-suite quick] [--reps 2] [--t1-thinking low]
#
# arms.txt, one arm per line ('#' comments allowed):  NAME  ENV_FILE  [KEY=VALUE ...]
#   ENV_FILE is a complete blackfold.env for that arm (its own HEAD_BUILD / WORKER_BUILD for a different engine).
#   KEY=VALUE pairs become EXTRA_ENV: engine switches passed to both ranks.
#   The FIRST arm is the reference every other arm is compared with, and it is the one restored at the end.
#
# For each arm: stop the serve, start it with the arm's config, warm it up, then record with tools/bf_lab.py
# in two regimes: T0 (greedy, thinking off, the benchmark regime) and T1 (temperature 1.0 with thinking on, the
# regime real use runs in). Sampling is seeded from the prompt, so both regimes are repeatable and every arm must
# match the reference token for token; the rounds and tok/s columns show what an arm changes.
#
# This RESTARTS the serve several times. Run it only when nobody is using the pair. It refuses to start while
# /health reports live connections, and restores the reference arm on exit, even after a failure.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ARMS="" BASE="" OUT="" OWN=0 SCRIPT="$here/../deploy/blackfold-native"
T0_SUITE=default T1_SUITE=quick REPS=2 T1_THINKING=low
while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS=$2; shift 2 ;;
    --base) BASE=${2%/}; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --script) SCRIPT=$2; shift 2 ;;
    --t0-suite) T0_SUITE=$2; shift 2 ;;
    --t1-suite) T1_SUITE=$2; shift 2 ;;
    --reps) REPS=$2; shift 2 ;;
    --t1-thinking) T1_THINKING=$2; shift 2 ;;
    --i-own-this-window) OWN=1; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$ARMS" ] && [ -n "$BASE" ] && [ -n "$OUT" ] || { sed -n 2,12p "$0"; exit 2; }
[ "$OWN" = 1 ] || { echo "refusing: this restarts the serve; pass --i-own-this-window when nobody else uses it" >&2; exit 2; }
LAB="python3 $here/bf_lab.py"
export BLACKFOLD_TOOLS="$here"  # the arms' copied deploy scripts find engram_slice.py here
mkdir -p "$OUT"
log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$OUT/window.log" >&2; }

mapfile -t LINES < <(grep -vE '^\s*(#|$)' "$ARMS")
[ "${#LINES[@]}" -ge 2 ] || { echo "need at least two arms" >&2; exit 2; }
names=()
for line in "${LINES[@]}"; do
  read -r name envfile extras <<<"$line"
  [ -f "$envfile" ] || { echo "arm $name: no env file $envfile" >&2; exit 2; }
  d="$OUT/arms/$name"; mkdir -p "$d"
  cp "$SCRIPT" "$d/blackfold-native"; chmod +x "$d/blackfold-native"
  { grep -v '^EXTRA_ENV=' "$envfile"; printf 'EXTRA_ENV="%s"\n' "${extras:-}"; } > "$d/blackfold.env"
  names+=("$name")
done
REF=${names[0]}

health() { curl -s -m 5 "$BASE/health" || true; }
busy() {  # live connections or waiting requests, from /health
  health | python3 -c 'import json,sys
try: l=json.load(sys.stdin).get("live") or {}
except Exception: print(0); sys.exit()
print(int(l.get("connections") or 0)+int(l.get("waiting") or 0))'
}

restored=0
restore() {
  [ "$restored" = 1 ] && return
  restored=1
  log "restore: $REF"
  "$OUT/arms/$REF/blackfold-native" stop >/dev/null 2>&1 || true
  if "$OUT/arms/$REF/blackfold-native" start >>"$OUT/window.log" 2>&1 &&
     $LAB record --base "$BASE" --label restore-check --suite quick --reps 1 --out "$OUT/restore-check.json" \
       --yes-quiet-window >>"$OUT/window.log" 2>&1; then
    log "restore: $REF serving, live generation checked"
  else
    log "RESTORE FAILED: start $REF by hand ($OUT/arms/$REF/blackfold-native start) and check it answers"
  fi
}
trap 'restore' EXIT

n=$(busy); [ "$n" = 0 ] || { log "refusing: /health reports $n live connections or waiting requests"; restored=1; exit 3; }
for name in "${names[@]}"; do
  d="$OUT/arms/$name"
  n=$(busy); [ "$n" = 0 ] || { log "stopping early: $n live connections appeared"; exit 3; }
  log "arm $name: restart"
  "$d/blackfold-native" stop >>"$OUT/window.log" 2>&1 || true
  "$d/blackfold-native" start >>"$OUT/window.log" 2>&1
  [ -x "$here/bf_preflight.sh" ] && bash "$here/bf_preflight.sh" > "$d/preflight-head.txt" 2>&1 || true
  log "arm $name: warm-up"
  $LAB record --base "$BASE" --label "$name-warm" --suite quick --reps 1 --out "$d/warm-t0.json" --yes-quiet-window >>"$OUT/window.log" 2>&1 || true
  $LAB record --base "$BASE" --label "$name-warm" --suite quick --reps 1 --temperature 1.0 --thinking "$T1_THINKING" \
    --out "$d/warm-t1.json" --yes-quiet-window >>"$OUT/window.log" 2>&1 || true
  log "arm $name: T0 ($T0_SUITE x$REPS)"
  $LAB record --base "$BASE" --label "$name" --engine "$name" --suite "$T0_SUITE" --reps "$REPS" --out "$d/t0.json" \
    --yes-quiet-window >>"$OUT/window.log" 2>&1 || log "arm $name: T0 had errors or unstable reps"
  log "arm $name: T1 ($T1_SUITE x$REPS, temperature 1.0, thinking $T1_THINKING)"
  $LAB record --base "$BASE" --label "$name" --engine "$name" --suite "$T1_SUITE" --reps "$REPS" --temperature 1.0 \
    --thinking "$T1_THINKING" --out "$d/t1.json" --yes-quiet-window >>"$OUT/window.log" 2>&1 \
    || log "arm $name: T1 had errors or unstable reps"
done
restore

{
  echo "# bf_window report"
  echo
  echo "Reference arm: \`$REF\`. Every arm must match it token for token in both regimes."
  for name in "${names[@]:1}"; do
    for r in t0 t1; do
      echo; echo "## $name vs $REF, $r"; echo
      $LAB compare "$OUT/arms/$REF/$r.json" "$OUT/arms/$name/$r.json" || true
    done
  done
} > "$OUT/report.md" 2>&1
log "report: $OUT/report.md"
