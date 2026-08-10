#!/bin/ash
# === Synology NAS network optimization ===
# Run as root (sudo -i first, then run this script).
# Applies immediately, but needs persistence setup for reboots.

echo "=== Before ==="
echo "rmem_max: $(sysctl -n net.core.rmem_max)"
echo "wmem_max: $(sysctl -n net.core.wmem_max)"
echo "qdisc: $(sysctl -n net.core.default_qdisc)"
echo "slow_start_after_idle: $(sysctl -n net.ipv4.tcp_slow_start_after_idle)"
echo ""

# 1. TCP buffer sizes: 208KB -> 2MB (the biggest bottleneck)
#    BDP at 180ms: 2MB / 0.180s = 89 Mbps (vs 9 Mbps before)
#    BDP at  26ms: 2MB / 0.026s = 615 Mbps (vs 64 Mbps before)
sysctl -w net.core.rmem_max=2097152
sysctl -w net.core.wmem_max=2097152
sysctl -w net.ipv4.tcp_rmem="4096 131072 2097152"
sysctl -w net.ipv4.tcp_wmem="4096 16384 2097152"

# 2. Fair queuing (fq_codel is the best option available — BBR module not compiled)
sysctl -w net.core.default_qdisc=fq_codel

# 3. Disable slow-start-after-idle (kills performance for intermittent WebDAV)
sysctl -w net.ipv4.tcp_slow_start_after_idle=0

# 4. TCP fast open (save 1 RTT on repeat connections — ~26-180ms)
sysctl -w net.ipv4.tcp_fastopen=3

echo "=== After ==="
echo "rmem_max: $(sysctl -n net.core.rmem_max)"
echo "wmem_max: $(sysctl -n net.core.wmem_max)"
echo "qdisc: $(sysctl -n net.core.default_qdisc)"
echo "slow_start_after_idle: $(sysctl -n net.ipv4.tcp_slow_start_after_idle)"
echo "tcp_fastopen: $(sysctl -n net.ipv4.tcp_fastopen)"
echo ""
echo "[OK] TCP tuning applied. These revert on reboot — see below."

# === Persist across reboots ===
# Add to /etc/sysctl.conf (or /etc/sysctl.d/99-tailscale.conf):
#
#   net.core.rmem_max=2097152
#   net.core.wmem_max=2097152
#   net.ipv4.tcp_rmem=4096 131072 2097152
#   net.ipv4.tcp_wmem=4096 16384 2097152
#   net.core.default_qdisc=fq_codel
#   net.ipv4.tcp_slow_start_after_idle=0
#   net.ipv4.tcp_fastopen=3
#
# Then: sysctl -p

# === Tune the macOS client (run on your Mac, NOT on the NAS) ===
# sudo sysctl -w net.inet.tcp.recvspace=2097152
# sudo sysctl -w net.inet.tcp.sendspace=2097152
# sudo sysctl -w net.inet.tcp.slowstart_flightsize=20
