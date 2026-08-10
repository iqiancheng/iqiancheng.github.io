#!/bin/ash
# === Synology NAS security hardening + network optimization ===
# Combines recommendations from two blog posts.
# Usage: ssh opennas -> sudo -i -> paste this script.
#
# Notes:
# 1. DSM control-panel changes still need to be done manually (see checklist at bottom).
# 2. Back up important data before running.

set -e

echo "=========================================="
echo " Synology NAS hardening + network tuning"
echo "=========================================="

# -- 1. SSH hardening ----------------------------------
echo ""
echo "=== [1/6] SSH hardening ==="

SSHD_CONFIG="/etc/ssh/sshd_config"
if [ -f "$SSHD_CONFIG" ]; then
    # Backup
    cp "$SSHD_CONFIG" "${SSHD_CONFIG}.bak.$(date +%Y%m%d)"

    # Update config
    sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' "$SSHD_CONFIG"
    sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' "$SSHD_CONFIG"
    sed -i 's/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/' "$SSHD_CONFIG"
    sed -i 's/^#*Protocol.*/Protocol 2/' "$SSHD_CONFIG"

    # SSH port change must be done via DSM UI; this script only does key/auth hardening.
    echo "[OK] SSH config hardened: PermitRootLogin=no, PasswordAuthentication=no"
    echo "[INFO] Change SSH port to 9525 (and drop 22) in DSM -> Terminal & SNMP"
fi

# -- 2. Disable unnecessary services -------------------
echo ""
echo "=== [2/6] Disable unnecessary services ==="

# rsync daemon
if systemctl is-active rsyncd.service >/dev/null 2>&1; then
    systemctl stop rsyncd.service
    systemctl disable rsyncd.service
    echo "[OK] rsyncd stopped & disabled"
fi

# iSCSI
if systemctl is-active iscsid.service >/dev/null 2>&1; then
    systemctl stop iscsid.service 2>/dev/null
    systemctl disable iscsid.service 2>/dev/null
    echo "[OK] iscsid stopped & disabled"
fi

# SNMP
for svc in snmpd.service synosnmpcd.service; do
    if systemctl is-active "$svc" >/dev/null 2>&1; then
        systemctl stop "$svc"
        systemctl disable "$svc"
        echo "[OK] $svc stopped & disabled"
    fi
done

# -- 3. TCP kernel tuning ------------------------------
echo ""
echo "=== [3/6] TCP kernel tuning ==="

sysctl -w net.core.rmem_max=2097152
sysctl -w net.core.wmem_max=2097152
sysctl -w net.ipv4.tcp_rmem="4096 131072 2097152"
sysctl -w net.ipv4.tcp_wmem="4096 16384 2097152"
sysctl -w net.ipv4.tcp_slow_start_after_idle=0
sysctl -w net.ipv4.tcp_fastopen=3

# Try fq_codel (if compiled into the kernel)
if sysctl -w net.core.default_qdisc=fq_codel 2>/dev/null; then
    echo "[OK] fq_codel set"
else
    sysctl -w net.core.default_qdisc=pfifo_fast 2>/dev/null
    echo "[WARN] fq_codel not available, falling back to pfifo_fast"
fi

# Persist to sysctl.conf
SYSCTL_CONF="/etc/sysctl.conf"
{
    echo "# Synology NAS TCP tuning — added $(date)"
    echo "net.core.rmem_max=2097152"
    echo "net.core.wmem_max=2097152"
    echo "net.ipv4.tcp_rmem=4096 131072 2097152"
    echo "net.ipv4.tcp_wmem=4096 16384 2097152"
    echo "net.ipv4.tcp_slow_start_after_idle=0"
    echo "net.ipv4.tcp_fastopen=3"
    echo "net.core.default_qdisc=pfifo_fast"
} >> "$SYSCTL_CONF"
echo "[OK] TCP settings persisted to $SYSCTL_CONF"

# -- 4. Firewall rules ---------------------------------
echo ""
echo "=== [4/6] Firewall (iptables) ==="

# Allow established/related first
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow SSH (9525) — LAN + Tailscale
iptables -A INPUT -p tcp --dport 9525 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 9525 -s 100.64.0.0/10 -j ACCEPT

# Allow DSM HTTPS (5001) — LAN
iptables -A INPUT -p tcp --dport 5001 -s 192.168.0.0/16 -j ACCEPT

# Allow WebDAV (5006) — Tailscale
iptables -A INPUT -p tcp --dport 5006 -s 100.64.0.0/10 -j ACCEPT

# Allow HTTPS (443) — LAN
iptables -A INPUT -p tcp --dport 443 -s 192.168.0.0/16 -j ACCEPT

# Allow SMB (445/139) — LAN (required for Finder mounts)
iptables -A INPUT -p tcp --dport 445 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 139 -s 192.168.0.0/16 -j ACCEPT

# Allow NetBIOS name service (137-138/udp) — LAN
iptables -A INPUT -p udp --dport 137:138 -s 192.168.0.0/16 -j ACCEPT

# Allow mDNS/Bonjour (5353/udp) — LAN (Finder auto-discovery)
iptables -A INPUT -p udp --dport 5353 -s 192.168.0.0/16 -j ACCEPT

# Allow ICMP (ping)
iptables -A INPUT -p icmp -j ACCEPT

# Default drop
iptables -A INPUT -j DROP

echo "[OK] iptables rules applied"

# -- 5. WebDAV MPM tuning ------------------------------
echo ""
echo "=== [5/6] WebDAV Apache MPM tuning ==="

MPM_CONF="/var/packages/WebDAVServer/target/etc/httpd/conf/extra/httpd-mpm.conf-webdav"
if [ -f "$MPM_CONF" ]; then
    cp "$MPM_CONF" "${MPM_CONF}.bak.$(date +%Y%m%d)"
    cat > "$MPM_CONF" << 'EOFMPM'
<IfModule !mpm_netware_module>
    PidFile /run/httpd/httpd-webdav.pid
</IfModule>

<IfModule mpm_worker_module>
    StartServers          3
    MinSpareThreads       25
    MaxSpareThreads       75
    ThreadsPerChild       25
    MaxRequestsPerChild   0
    ServerLimit           6
    MaxRequestWorkers     150
</IfModule>
EOFMPM
    echo "[OK] MPM config updated"

    # Restart WebDAV
    if /usr/syno/bin/synopkg restart WebDAVServer 2>/dev/null; then
        echo "[OK] WebDAVServer restarted"
    else
        echo "[WARN] synopkg restart failed, check httpd -t for syntax errors"
        echo "       Common issue: non-breaking space (\\xc2\\xa0) in heredoc"
    fi
fi

# -- 6. Tailscale Serve --------------------------------
echo ""
echo "=== [6/6] Tailscale Serve ==="

TAILSCALE_BIN="/var/packages/Tailscale/target/bin/tailscale"
if [ -x "$TAILSCALE_BIN" ]; then
    # Check current serve status
    if ! $TAILSCALE_BIN serve status 2>/dev/null | grep -q "5006"; then
        $TAILSCALE_BIN serve --bg https+insecure://localhost:5006
        echo "[OK] Tailscale Serve enabled for WebDAV"
    else
        echo "[OK] Tailscale Serve already configured"
    fi
    $TAILSCALE_BIN serve status
fi

# -- Audit report --------------------------------------
echo ""
echo "=========================================="
echo " Audit report"
echo "=========================================="
echo ""

echo "=== 1. Telnet ==="
netstat -tlnp 2>/dev/null | grep ":23 " && echo "[FAIL] Telnet is ON" || echo "[PASS] Telnet is OFF"

echo "=== 2. SSH ports ==="
SSH_PORTS=$(netstat -tlnp 2>/dev/null | grep ":22 \|:9525 " | wc -l)
[ "$SSH_PORTS" -le 1 ] && echo "[PASS] SSH single port" || echo "[FAIL] SSH on $SSH_PORTS ports (should be 1)"

echo "=== 3. Firewall ==="
iptables -L INPUT -n 2>/dev/null | grep -q "DROP\|REJECT" && echo "[PASS] Firewall has rules" || echo "[FAIL] Firewall empty"

echo "=== 4. admin account ==="
grep "^admin:" /etc/passwd 2>/dev/null && echo "[WARN] admin account exists (disable in DSM UI)" || echo "[PASS]"

echo "=== 5. HTTP redirect ==="
curl -skI http://localhost:5000 2>/dev/null | grep -q "302\|301" && echo "[PASS] HTTP->HTTPS" || echo "[FAIL] HTTP still accessible (enable in DSM)"

echo "=== 6. Public ports ==="
PUBLIC=$(netstat -tlnp 2>/dev/null | grep "0.0.0.0" | wc -l)
echo "[INFO] $PUBLIC ports listening on 0.0.0.0"

echo ""
echo "=========================================="
echo " Still to do manually in the DSM UI:"
echo "=========================================="
echo "  [ ] Terminal & SNMP -> uncheck Telnet"
echo "  [ ] Terminal & SNMP -> change SSH port to 9525 (drop 22)"
echo "  [ ] Users & Groups -> disable admin account"
echo "  [ ] Users & Groups -> disable guest account"
echo "  [ ] Network -> DSM Settings -> HTTP->HTTPS redirect"
echo "  [ ] Security -> Protection -> auto-block (3/5min -> lock 30min)"
echo "  [ ] Security -> Advanced -> enable CSP / XCTO / HSTS"
echo "  [ ] File Services -> rsync -> uncheck"
echo "  [ ] File Services -> SMB -> enable signing + anti-brute-force"
echo "  [ ] Shared Folders -> grant WebDAV permissions"
echo "=========================================="
