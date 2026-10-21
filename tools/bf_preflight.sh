#!/usr/bin/env bash
# bf_preflight.sh: one node's state as key=value lines, read-only, so a bench or a start records what the box was
# doing. Run on the head and on the worker; diff the two. Changes nothing, starts nothing, sends no traffic.
#
#   bf_preflight.sh [--binary PATH] [--engram DIR] [--burn SECONDS]
#
# --burn runs a short GPU load to expose a latched clock (GB10 can stick at ~600-950 MHz until power-cycled). It is
# load: only in a quiet window with the serve stopped. Everything else only reads.
set -u
BIN="" ENGRAM="" BURN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --binary) BIN=$2; shift 2 ;;
    --engram) ENGRAM=$2; shift 2 ;;
    --burn) BURN=$2; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
kv() { printf '%s=%s\n' "$1" "$2"; }
have() { command -v "$1" >/dev/null 2>&1; }

kv role_hint "$( [ -n "${BF_ROLE:-}" ] && echo "$BF_ROLE" || echo unset)"
kv time_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
kv uptime_s "$(cut -d. -f1 /proc/uptime)"
kv kernel "$(uname -r)"
kv boot_id "$(cat /proc/sys/kernel/random/boot_id 2>/dev/null)"

# memory: available, fragmentation (orders 9-10 = 2-4 MiB blocks), swap, THP
kv mem_available_gib "$(awk '/MemAvailable/ {printf "%.1f", $2/1048576}' /proc/meminfo)"
kv swap_used_gib "$(awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {printf "%.1f", (t-f)/1048576}' /proc/meminfo)"
kv buddyinfo_normal_o9_o10 "$(awk '$4=="Normal" {print $(5+9) "," $(5+10)}' /proc/buddyinfo | paste -sd';')"
kv thp "$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"

# GPU: driver, clocks, power, temperature, throttle reasons
if have nvidia-smi; then
  kv driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  kv gpu "$(nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,utilization.gpu \
    --format=csv,noheader 2>/dev/null | head -1)"
  kv throttle_active "$(nvidia-smi -q -d PERFORMANCE 2>/dev/null | awk -F: '$2 ~ /^ *Active *$/ {gsub(/ +/,"",$1); printf "%s,", $1}')"
  kv gpu_procs "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | paste -sd';')"
else
  kv driver none
fi
kv cpu_governor "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"

# RDMA rails: state and MTU per device, IPv4 GID index per port
if [ -d /sys/class/infiniband ]; then
  for d in /sys/class/infiniband/*; do
    n=$(basename "$d")
    for p in "$d"/ports/*; do
      st=$(cut -d: -f2 "$p/state" 2>/dev/null | tr -d ' ')
      gid=""
      for t in "$p"/gid_attrs/types/*; do
        [ "$(cat "$t" 2>/dev/null)" = "RoCE v2" ] || continue
        i=$(basename "$t")
        g=$(cat "$p/gids/$i" 2>/dev/null)
        case "$g" in 0000:0000:0000:0000:0000:ffff:*) gid="$i"; break ;; esac
      done
      kv "rail_${n}_p$(basename "$p")" "state=$st ipv4_rocev2_gid=${gid:-none}"
    done
  done
else
  kv rails none
fi

# engine binary and Engram tables
[ -n "$BIN" ] && kv binary_sha256 "$(sha256sum "$BIN" 2>/dev/null | cut -c1-64)"
if [ -n "$ENGRAM" ]; then
  kv engram_fs "$(findmnt -no SOURCE,FSTYPE --target "$ENGRAM" 2>/dev/null)"
  kv engram_gib "$(du -s --block-size=1G "$ENGRAM" 2>/dev/null | cut -f1)"
fi

# the serve container, if any
if have docker; then
  kv containers "$(docker ps --format '{{.Names}}:{{.Status}}' 2>/dev/null | paste -sd';')"
fi

if [ "$BURN" -gt 0 ] 2>/dev/null && have nvidia-smi; then
  if have gpu_burn; then
    timeout "$((BURN + 5))" gpu_burn "$BURN" >/dev/null 2>&1 &
  elif python3 -c 'import torch' 2>/dev/null; then
    timeout "$((BURN + 5))" python3 -c "import torch,time;a=torch.randn(8192,8192,device='cuda',dtype=torch.float16);t=time.time()
while time.time()-t<$BURN: a@a
torch.cuda.synchronize()" >/dev/null 2>&1 &
  else
    kv burn "skipped (no gpu_burn or torch)"; BURN=0
  fi
  if [ "$BURN" -gt 0 ]; then
    sleep $((BURN / 2))
    kv burn_mid "$(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu --format=csv,noheader | head -1)"
    wait
  fi
fi
