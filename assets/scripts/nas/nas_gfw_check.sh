#!/bin/bash
# === NAS GFW layered health check ===
# Usage: bash nas_gfw_check.sh
#    or: ssh opennas "bash -s" < nas_gfw_check.sh

set -e

echo "=== NAS GFW health check — $(date) ==="
echo ""

FAILS=0

# -- 1. Docker mirrors --
echo "[1/6] Docker mirrors"
for m in \
  https://docker.1panel.live \
  https://docker.1ms.run \
  https://docker.m.daocloud.io \
  https://mirror.iscas.ac.cn; do
  code=$(curl -sk --connect-timeout 3 -o /dev/null -w '%{http_code}' "$m/v2/" 2>/dev/null)
  case $code in
    200|401) echo "  OK  $m ($code)" ;;
    *)       echo "  FAIL $m ($code)"; FAILS=$((FAILS+1)) ;;
  esac
done

# -- 2. pip mirror --
echo "[2/6] pip mirror"
code=$(curl -sk --connect-timeout 3 -I -o /dev/null -w '%{http_code}' \
  https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null)
if [ "$code" = "200" ]; then
  echo "  OK  mirrors.aliyun.com"
else
  echo "  FAIL mirrors.aliyun.com ($code)"
  FAILS=$((FAILS+1))
fi

# -- 3. npm mirror --
echo "[3/6] npm mirror"
code=$(curl -sk --connect-timeout 3 -I -o /dev/null -w '%{http_code}' \
  https://registry.npmmirror.com/ 2>/dev/null)
if [ "$code" = "200" ]; then
  echo "  OK  registry.npmmirror.com"
else
  echo "  FAIL registry.npmmirror.com ($code)"
  FAILS=$((FAILS+1))
fi

# -- 4. mihomo explicit proxy (port 7890) --
echo "[4/6] mihomo explicit proxy"
for target in https://github.com https://www.google.com https://registry-1.docker.io/v2/; do
  code=$(curl -sk --connect-timeout 5 -x http://127.0.0.1:7890 \
    -o /dev/null -w '%{http_code}' "$target" 2>/dev/null)
  case $code in
    200|301|302|401) echo "  OK  proxy -> $target ($code)" ;;
    *)               echo "  FAIL proxy -> $target ($code)"; FAILS=$((FAILS+1)) ;;
  esac
done

# -- 5. mihomo TUN transparent proxy (no -x flag) --
echo "[5/6] mihomo TUN (transparent)"
for target in https://github.com https://www.googleapis.com/ https://www.google.com; do
  code=$(curl -sk --connect-timeout 5 -o /dev/null -w '%{http_code}' "$target" 2>/dev/null)
  case $code in
    200|301|302|404) echo "  OK  TUN -> $target ($code)" ;;
    *)               echo "  FAIL TUN -> $target ($code)"; FAILS=$((FAILS+1)) ;;
  esac
done

# -- 6. mihomo process (host-level, not Docker) --
echo "[6/6] mihomo process"
if ps aux | grep -v grep | grep -q mihomo-host; then
  echo "  OK  mihomo-host running (systemd)"
else
  echo "  FAIL mihomo-host not running"
  FAILS=$((FAILS+1))
fi

# -- Report --
echo ""
echo "=== Check complete: $FAILS failure(s) ==="
[ "$FAILS" -gt 0 ] && exit 1 || exit 0
