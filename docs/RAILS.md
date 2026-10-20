# Two RoCE rails on a DGX Spark pair

Each GB10 node reaches its ConnectX-7 port over two PCIe x4 links, so one cabled port shows up as two
netdevs and two RDMA devices (here `enp1s0f0np0` / `rocep1s0f0` and its twin `enP2p1s0f0np0` /
`roceP2p1s0f0`). One twin carries roughly half the port's bandwidth: `ib_write_bw` between the twins
measured 108.9 Gb/s over 10 s. Giving the twin its own address and listing both devices lets TensorFold
use them as two rails. No extra cable is needed.

## Measured effect (same build, same pair, all outputs identical)

| 125K-context cell | One rail | Two rails |
|---|---:|---:|
| Cold prefill, code prompt | 67.9 s | 62.0 s (-8.7%) |
| Cold prefill, prose prompt | 67.2 s | 61.2 s (-8.9%) |
| Solo code / prose decode, tok/s | 91.5 / 56.2 | 92.8 / 57.1 |
| Two-stream code / prose, per stream | 68.1 / 37.4 | 69.8 / 37.9 |

Cold prefill is one sample per cell; decode is the mean of four.

## The one trap: GID index alignment

The engine takes a single GID index per node (`TF_RDMA_GID_INDEX`, `NCCL_IB_GID_INDEX`) and uses it for
every device. The twin's IPv4 RoCE v2 GID must therefore sit at the same index as the primary's. GID
slots fill lowest-free, so the index depends on how many IPv6 link-local addresses the twin carries when
its IPv4 is added. On our pair the primaries sat at index 5 on one node and 3 on the other, and
NetworkManager's address-generation mode decided where the twin landed. Check with:

```bash
for h in rocep1s0f0 roceP2p1s0f0; do for i in $(seq 0 7); do
  printf '%s %s %s %s\n' $h $i "$(cat /sys/class/infiniband/$h/ports/1/gids/$i 2>/dev/null)" \
    "$(cat /sys/class/infiniband/$h/ports/1/gid_attrs/types/$i 2>/dev/null)"; done; done
```

## Persistent setup used here (both nodes)

1. A NetworkManager profile for the twin that brings the link up with MTU 9000 and no IPv4. Use
   `ipv6.addr-gen-mode eui64` on a node whose primary has one IPv6 link-local, the default on a node
   whose primary has two:

   ```bash
   sudo nmcli connection add type ethernet con-name blackfold-rail2 ifname enP2p1s0f0np0 \
     ipv4.method disabled ipv6.method link-local 802-3-ethernet.mtu 9000 \
     connection.autoconnect yes connection.autoconnect-priority 10
   ```
   Disable autoconnect on any older profile bound to the twin.
2. [`deploy/blackfold-rail-align`](../deploy/blackfold-rail-align) adds the twin's IPv4 from
   `/etc/blackfold-rail.env` and moves it to the primary's GID index (adding or removing an IPv6 filler),
   run at boot by [`blackfold-rail-align.service`](../deploy/blackfold-rail-align.service).
3. NetworkManager strips that address whenever it re-activates the profile, so
   [`deploy/90-blackfold-rail2`](../deploy/90-blackfold-rail2) (a dispatcher hook) re-runs the align unit on
   `up` and `reapply`. Verified with two re-activations per node: address and GIDs back within 10 s.

Limits: if the twin link drops while serving, the engine's second-rail connections stay broken until the
serve is restarted. `deploy/blackfold-native start` falls back to one rail, with a warning, when the
twins cannot be aligned.
