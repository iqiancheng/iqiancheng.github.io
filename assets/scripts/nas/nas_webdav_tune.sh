#!/bin/ash
# === WebDAV Apache MPM tuning ===
# Run as root on the Synology NAS.
# Config file: /var/packages/WebDAVServer/target/etc/httpd/conf/extra/httpd-mpm.conf-webdav

MPM_CONF="/var/packages/WebDAVServer/target/etc/httpd/conf/extra/httpd-mpm.conf-webdav"
BACKUP="${MPM_CONF}.bak.$(date +%Y%m%d)"

echo "Backing up current MPM config to $BACKUP"
cp "$MPM_CONF" "$BACKUP"

# Current default: ThreadsPerChild=1 (essentially single-threaded, terrible).
# Fix: 25 threads per process, better startup/spare settings.

cat > "$MPM_CONF" << 'EOF'
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
EOF

echo "MPM config updated:"
cat "$MPM_CONF"

# Restart WebDAV server
echo "Restarting WebDAV server..."
/usr/syno/bin/synopkg restart WebDAVServer 2>/dev/null || systemctl restart webdavd 2>/dev/null || {
    echo "Could not restart via synopkg. Try manually:"
    echo "  sudo synopkg restart WebDAVServer"
}

echo ""
echo "[OK] Apache MPM tuned. Verify with: ps aux | grep httpd | grep webdav | wc -l"
echo "Should see ~3-6 httpd processes (instead of 2 previously)"
