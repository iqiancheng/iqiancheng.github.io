---
layout: post
author: Joseph
title: "家庭 NAS 翻墙实战：GFW 连通性审计 + mihomo 分层代理"
date: 2026-05-10 00:00:00 +0800
categories: [homelab]
tags: [docker, proxy, nas, security, github]
description: >
  对中国大陆环境下的 Synology NAS 进行全量 GFW 连通性审计，找出被墙的关键服务（GitHub、Docker Hub、Google），
  部署 mihomo 代理实现分层访问策略——国内 CDN mirror 处理日常流量，代理解决被墙资源。
toc: true
---
## 背景

[上一篇 Docker 镜像加速](/posts/docker-registry-mirrors-china/) 解决了 `docker pull` 的问题，但 NAS 上还有更多服务受 GFW 影响——Git clone 超时、raw 脚本下载失败、部分容器注册表不可达。

这篇对 NAS 做一个**全量 GFW 连通性审计**，然后部署 mihomo（原 Clash.Meta）作为代理层，最终形成「mirror 加速 + 代理兜底」的分层访问策略。

测试环境：Synology SA6400, DSM 7.x, 中国移动宽带, mihomo 运行在 Docker 中。

---

## 一、GFW 审计方法论

### 1.1 审计范围

覆盖 NAS 日常运维涉及的所有外部服务：

| 类别 | 服务 | 典型场景 |
|------|------|---------|
| 代码托管 | github.com, raw.githubusercontent.com | git clone, 脚本下载, Gitea 镜像同步 |
| 容器注册表 | registry-1.docker.io, ghcr.io, quay.io, gcr.io | docker pull |
| Python 生态 | pypi.org | pip install |
| 搜索引擎 | google.com | 技术搜索 |
| 系统服务 | DSM 更新, NTP, SynoCommunity | 系统维护 |

### 1.2 测试方法

```bash
# 直连测试
curl -sk --connect-timeout 5 -o /dev/null -w '%{http_code}|%{time_total}s' 'https://<target>/'

# 代理测试（部署后）
curl -sk --connect-timeout 5 -x http://127.0.0.1:7890 -o /dev/null -w '%{http_code}|%{time_total}s' 'https://<target>/'
```

HTTP 状态码读法：
- 200/301/302 → 可达
- 401 (Registry) → 可达（需要认证，正常）
- 000 (curl 超时) → **被墙**
- 连接重置 (TCP RST) → **被墙**

---

## 二、审计结果：直连 vs 代理

| 服务 | 直连 | 通过 mihomo | 判定 |
|------|------|------------|------|
| **github.com** | TCP 重置 | 200 (0.52s) | **被墙** |
| **raw.githubusercontent.com** | 超时 | 301 (0.46s) | **被墙** |
| **registry-1.docker.io** | 超时 | 401 (0.79s) | **被墙**（已用 mirror 绕过） |
| **google.com** | 超时 | 301 (1.05s) | **被墙** |
| **gcr.io** | 超时 | 302 (0.42s) | **被墙** |
| ghcr.io | 301 (0.42s) | 301 (0.50s) | 正常 |
| quay.io | 200 (1.51s) | 200 (1.54s) | 正常 |
| pypi.org | 200 (4.08s) | 200 (0.38s) | 正常但慢（走 mirror 更快） |
| **registry.npmjs.org** | 200 (1.82s) | — | **未被墙**（走 mirror 更快，0.11s） |
| DSM 更新 | 404 (0.28s) | — | 正常 |
| NTP | 正常 | — | 正常 |

关键发现：**GitHub 全家桶全封** + Docker Hub 直连被墙 + Google 被墙。npm 官方源和 PyPI 均未被墙，但国内 mirror 延迟更低。ghcr.io 和 quay.io 在国内正常可达。

---

## 三、分层访问策略

mirror 和代理**不是替代关系**——mirror 走国内 CDN 更快，代理解决 mirror 覆盖不到的缺口。

```
第一层：国内 Mirror（快速通道）
  ├── Docker: 1panel.live / 1ms.run / daocloud / iscas / huaweicloud
  ├── pip: mirrors.aliyun.com + pypi.tuna.tsinghua.edu.cn
  └── npm: registry.npmmirror.com

第二层：mihomo 代理（被墙必需）
  ├── GitHub (git clone / raw / releases)
  ├── Docker Hub 直连（mirror 失效时的 fallback）
  └── Google / GCR / 其他被墙服务

第三层：按需使用
  └── curl/wget: 需要时手动 export https_proxy
```

日常 `docker pull` 和 `pip install` 走第一层（国内 CDN 更快），遇到 mirror 没缓存或被墙的资源才走第二层代理。

---

## 四、mihomo 部署步骤

### 4.1 镜像拉取

前提：已按上一篇配置 Docker mirror，否则镜像都拉不下来。

```bash
sudo docker pull metacubex/mihomo:latest
```

### 4.2 准备配置

mihomo 兼容标准 Clash 配置。如果你本地有 Clash Verge 在跑，可以直接用同一份订阅生成的 config，调整以下字段：

```yaml
# NAS 部署需要改的关键项
allow-lan: true              # 允许局域网设备连接
external-controller: 0.0.0.0:9090  # Dashboard 监听
bind-address: '*'
log-level: warning           # NAS 上降噪
tun:
  enable: false              # Docker 容器里不建议开 TUN
```

### 4.3 启动容器

```bash
sudo mkdir -p /volume2/docker/mihomo

sudo docker run -d \
  --name mihomo \
  --restart unless-stopped \
  -p 7890:7890 \
  -p 9090:9090 \
  -v /volume2/docker/mihomo/config.yaml:/root/.config/mihomo/config.yaml:ro \
  metacubex/mihomo:latest
```

> 踩坑：mihomo 优先读取 `~/.config/mihomo/config.yaml`，不是 `/etc/mihomo/config.yaml`。挂载到后者会生成空配置覆盖你的文件。

### 4.4 验证

```bash
$ curl -x http://127.0.0.1:7890 -o /dev/null -w '%{http_code}' https://github.com
200    # ← 通了
```

### 4.5 防火墙放行

给局域网和 Tailscale 设备开放代理端口：

```bash
iptables -I DEFAULT_INPUT 3 -p tcp --dport 7890 -s 192.168.0.0/16 -j ACCEPT
iptables -I DEFAULT_INPUT 4 -p tcp --dport 7890 -s 100.64.0.0/10 -j ACCEPT
```

Dashboard：`external-controller: 0.0.0.0:9090` 暴露的是 mihomo 的 **REST API**，浏览器直接打开只返回 JSON。需要一个前端 SPA 来消费这个 API。

**推荐方案 A（零部署）**：用官方托管版 [metacubexd](https://github.com/MetaCubeX/metacubexd)，浏览器打开 `https://metacubexd.vercel.app/`，在页面设置里填入 mihomo API 地址 `http://<NAS-IP>:9090` 即可。

> 注意：API 地址需要从浏览器端可达。局域网直接 `http://192.168.x.x:9090`，Tailscale 用 `http://100.66.x.x:9090`。如果走 Tailscale Serve 暴露 9090 需要 `tailscale serve --bg https+insecure://localhost:9090`。

**方案 B（自部署）**：在 NAS 上跑 metacubexd 容器，用 Tailscale Serve 暴露：

```bash
sudo docker run -d \
  --name metacubexd \
  --restart unless-stopped \
  -p 127.0.0.1:9080:80 \
  ghcr.io/metacubex/metacubexd:latest

tailscale serve --bg https+insecure://localhost:9080
```

之后通过 `https://<nas>.<tailnet>.ts.net` 访问 Dashboard，自动连接同机 mihomo。

---

## 五、各场景使用方式

### NAS 自身

```bash
# 单次使用
export https_proxy=http://127.0.0.1:7890
curl https://github.com/...

# 或者走 mirror（更快）
# docker pull → 自动走 mirror
# pip install → 自动走 mirror
```

### 局域网设备

WiFi 设置 → HTTP 代理 → `<NAS-IP>:7890`。macOS 在「系统设置 → Wi-Fi → 代理」中配置。

### npm（Claude Code CLI / OpenCode / Codex CLI 等）

npm 官方源未被墙但延迟较高（1.8s）。推荐切到阿里镜像：

```bash
npm config set registry https://registry.npmmirror.com/
```

Claude Code CLI 等 AI coding 工具依赖的 npm 包（`@anthropic-ai/claude-code` 等）在 npmmirror 都有同步。切完验证：

```bash
npm view @anthropic-ai/claude-code version
```

你 Mac 上的 npm 当前走本地 Clash Verge 代理（`~/.npmrc` 中配了 `proxy=http://127.0.0.1:7890`），已切换为 npmmirror 直连。二者效果相近，但 mirror 省一次代理转发。

### Tailscale 远程设备

代理地址填 `http://100.66.x.x:7890`（NAS 的 Tailscale IP），出门在外也能用。

---

## 六、TUN 透明代理：从 Docker 到宿主机

Docker 版 mihomo 提供了 HTTP/SOCKS 代理（`:7890`）和 Dashboard 控制（`:9090`），适合 `curl -x`、`git config http.proxy` 这类**主动声明代理**的场景。

但有些应用不认 `http_proxy` 环境变量——比如 Synology Cloud Sync 的 C++ HTTP 库就完全忽略它。这些"代理无感"的应用需要**透明代理**：在系统层面劫持 TCP 出站流量，按规则分流，应用完全无感知。

mihomo 的 **TUN 模式**就是做这个的。Docker 容器里跑 TUN 有两个障碍：

- 需要 `/dev/net/tun` 设备——Synology 默认没有，但 `tun.ko` 内核模块存在，只是没加载
- 需要 `NET_ADMIN` capability 来创建虚拟网卡和 iptables 规则

### 6.1 加载 tun.ko

```bash
# Synology SA6400 (DSM 7.2) — 模块已在磁盘上，只差 insmod
ls /lib/modules/tun.ko        # 确认存在
sudo insmod /lib/modules/tun.ko
ls -la /dev/net/tun           # crw-rw-rw- — 加载成功
```

### 6.2 从 Docker 迁到宿主机

Docker 容器跑 TUN 技术上可行，但 iptables 规则和网络命名空间隔离太复杂。直接做法：**在宿主机上跑 mihomo**，开启 TUN + gVisor 用户态网络栈。

```bash
# 下载 mihomo 宿主机二进制（linux-amd64 compatible，含 gVisor）
# 在 Mac 上：
curl -sL -o mihomo.gz \
  "https://github.com/MetaCubeX/mihomo/releases/download/v1.19.24/mihomo-linux-amd64-compatible-v1.19.24.gz"
gunzip mihomo.gz && chmod +x mihomo
cat mihomo | ssh opennas "cat > /volume2/docker/mihomo/mihomo-host"
ssh opennas chmod +x /volume2/docker/mihomo/mihomo-host

# 停止 Docker 版 mihomo
ssh opennas sudo docker stop mihomo
ssh opennas sudo docker update --restart=no mihomo
```

> v1.19.24 的 `compatible` 构建版带 `with_gvisor` 标签——gVisor 在用户态实现完整 TCP/IP 协议栈，**不需要内核 TUN/TAP 支持**就能创建虚拟网卡。这对内核裁剪激进的 NAS 系统至关重要。实测 DSM 7.2（内核 5.10.55+）上运行正常，但依然需要先 `insmod tun.ko` 来提供 `/dev/net/tun` 设备节点。

### 6.3 TUN 配置

宿主机配置与 Docker 版共用同一份节点/规则，只增改 TUN 段：

```yaml
# /usr/local/etc/mihomo/config.yaml 新增项
tun:
  enable: true
  stack: gvisor              # 用户态 TCP/IP，无需内核 TUN 驱动
  auto-route: true           # 自动添加路由规则
  auto-redirect: true        # 自动添加 iptables REDIRECT 规则
  auto-detect-interface: true
  bypass-private: false      # 交给规则引擎判断（GEOIP CN → DIRECT）
```

`auto-redirect` 会在 iptables NAT 表中创建以下链：

```
Chain mihomo-output (OUTPUT 链引用)
  REDIRECT tcp → port 38453   # 所有本机出站 TCP 被重定向到 mihomo TUN 入站

Chain mihomo-prerouting (PREROUTING 链引用)
  DNAT udp dpt:53 → 198.18.0.2  # DNS 劫持到 mihomo fake-ip DNS
  REDIRECT tcp → port 38453
```

### 6.4 DNS fake-ip 机制

mihomo TUN 劫持所有 DNS 查询（UDP 53），返回 `198.18.0.0/24` 段的虚拟 IP。当应用向这个虚拟 IP 发起 TCP 连接时，mihomo 根据原始域名匹配规则（GEOIP、DOMAIN-KEYWORD 等），决定走代理还是直连。

```
应用                            TUN 层                        mihomo
  │                               │                             │
  │  DNS query: github.com        │                             │
  ├──────────────────────────────►│  198.18.0.5 ← fake IP       │
  │  198.18.0.5                   │                             │
  │                               │                             │
  │  TCP SYN → 198.18.0.5:443    │                             │
  ├──────────────────────────────►│  REDIRECT → port 38453      │
  │                               ├────────────────────────────►│
  │                               │  查表: github.com → Proxy    │
  │                               │◄────────────────────────────┤
  │◄──────────────────────────────┤                             │
```

这个机制的精妙之处：**Fake IP 和域名一一映射，应用层完全无感知**。

### 6.5 Tailscale 兼容性

TUN 的 `auto-redirect` 只拦截 **TCP** 流量（和 UDP 53 DNS）。Tailscale 的 WireGuard 隧道跑在 **UDP 41641**，不受影响。Tailscale 状态在 TUN 启用后保持正常：

```
$ tailscale status
100.66.51.28  openai  chinayanpeng@  linux  -   # ← 本机，在线
```

> 如果后续需要 Tailscale exit node 功能，注意 exit node 的 TCP 转发同样会被 TUN 拦截——这是预期行为，出口流量本就应该走代理。

---

## 七、Cloud Sync Google Drive 修复

### 7.1 问题诊断

Cloud Sync 是 Synology 官方的云盘同步套件，支持 Google Drive、Dropbox、百度网盘、阿里云 OSS 等。问题出在 Google Drive：

```
症状: Google OAuth 授权成功 → 连接 Google Drive 超时
根因: www.googleapis.com (Google Drive API) 被 GFW TCP 重置
```

排查过程：

```bash
# 1. 确认代理路径可达
curl -x http://127.0.0.1:7890 'https://www.googleapis.com/'  # → 404 (可达)

# 2. DSM 系统代理 (/etc/proxy.conf) → 无效
# Cloud Sync 不读系统代理配置

# 3. 环境变量 http_proxy → 无效
# 修改 start.sh 导出 proxy env vars → 无效
# 进程 environ 中确认有 http_proxy → 仍然超时
# 结论: Cloud Sync 的 C++ HTTP 库不认代理环境变量
```

### 7.2 失败方案：二进制 wrapper

在 `syno-cloud-syncd` 位置放了一个 shell wrapper，导出四种大小写的代理变量后 exec 真实二进制：

```sh
#!/bin/sh
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export no_proxy=localhost,127.0.0.1,192.168.0.0/16
export NO_PROXY=localhost,127.0.0.1,192.168.0.0/16
exec /var/packages/CloudSync/target/sbin/syno-cloud-syncd.real "$@"
```

环境变量确实进了进程（`/proc/PID/environ` 确认），但 Cloud Sync 依然超时。**它的 C++ HTTP 客户端完全忽略系统代理变量**——这在闭源商业软件中很常见。

### 7.3 正确方案：TUN 透明代理

有了第六章的 TUN 透明代理，问题自动消失。Cloud Sync 发起到 `www.googleapis.com:443` 的 TCP 连接时：

1. DNS 查询被劫持到 mihomo fake-ip DNS → 返回 `198.18.0.x` 虚拟 IP
2. TCP SYN 到 `198.18.0.x:443` → iptables REDIRECT 到 mihomo TUN 入站
3. mihomo 查 fake-ip 表还原域名 → 匹配规则 `DOMAIN-SUFFIX,googleapis.com` → 走代理
4. 代理节点完成 TLS 握手 → HTTP CONNECT → Google Drive API 可通

**应用代码一行不改，环境变量一个不设。** 这就是透明代理的价值。

### 7.4 清理痕迹

TUN 生效后，之前的 wrapper 和 start.sh 修改全部撤干净：

```bash
# 停止 Cloud Sync → 删除 wrapper → 恢复原始二进制 → 清理 start.sh → 重启
sudo /usr/syno/bin/synopkg stop CloudSync
sudo rm /volume2/@appstore/CloudSync/sbin/syno-cloud-syncd
sudo mv /volume2/@appstore/CloudSync/sbin/syno-cloud-syncd.real \
        /volume2/@appstore/CloudSync/sbin/syno-cloud-syncd
sudo cp /volume2/@appstore/CloudSync/scripts/start.sh.bak.20260510 \
        /volume2/@appstore/CloudSync/scripts/start.sh
```

重启后 Cloud Sync 进程环境中没有任何 proxy 变量，Google Drive 正常连接。

---

## 八、systemd 开机自启与重启验证

重启后以下组件需要自动恢复：加载 tun 模块、启动 mihomo-host、重建 TUN iptables 规则。

### 8.1 systemd service

```ini
# /etc/systemd/system/mihomo.service
[Unit]
Description=Mihomo transparent proxy (TUN mode)
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/volume2

[Service]
Type=simple
ExecStartPre=-/sbin/insmod /lib/modules/tun.ko
ExecStartPre=/bin/sleep 2
ExecStart=/volume2/docker/mihomo/mihomo-host -d /usr/local/etc/mihomo
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

关键设计：
- `ExecStartPre=-/sbin/insmod` — `-` 前缀表示即使 insmod 失败也继续（模块可能已加载）
- `RequiresMountsFor=/volume2` — 等存储卷挂载完毕才启动（二进制和配置都在 volume2 上）
- `Restart=on-failure` — mihomo 异常退出时 10 秒后自动重启
- TUN iptables 规则由 mihomo 的 `auto-redirect` 在运行时自动创建，**不需要手动管理**

### 8.2 重启验证清单

实际重启 DSM 后的逐项验证结果：

| 检查项 | 命令 | 结果 |
|--------|------|------|
| tun 模块 | `ls /dev/net/tun` | ✅ 已加载 |
| mihomo-host 进程 | `ps aux \| grep mihomo-host` | ✅ PID 12292 |
| systemd 状态 | `systemctl status mihomo` | ✅ active (running) |
| TUN 透明代理 | `curl https://github.com` | ✅ 200 (2.7s) |
| 代理端口 | `curl -x :7890 https://github.com` | ✅ 200 (0.5s) |
| Cloud Sync | `ps aux \| grep cloud-syncd` | ✅ 运行中 |
| Google Drive | DSM UI → Cloud Sync → 连接 | ✅ 正常 |
| Tailscale | `tailscale status` | ✅ 在线 |
| Docker 容器 | `docker ps` | ✅ gitea, open-webui 等全部恢复 |

### 8.3 启用以自举

```bash
sudo systemctl enable mihomo.service
sudo systemctl daemon-reload
```

重启测试耗时约 110 秒，所有反墙组件全自动恢复。不需要任何手动操作。

---

## 九、部署前后对比

| 维度 | 部署前 | 部署后 |
|------|--------|--------|
| github.com | 超时 | **0.52s**（TUN 透明代理，无需 -x） |
| raw.githubusercontent.com | 超时 | **0.46s**（同上） |
| Docker Hub 直连 | 超时 | **0.79s**（mirror 优先，代理备用） |
| gcr.io | 超时 | **0.42s** |
| Google APIs | 超时 | **0.30s**（TUN 透明代理） |
| Cloud Sync Google Drive | 超时 | **正常同步**（TUN 自动劫持） |
| 国内流量 (baidu.com) | 正常 | **0.07s**（GEOIP CN → DIRECT，不走代理） |
| 局域网设备代理 | 无 | HTTP proxy @ NAS:7890 |
| Dashboard | 无 | localhost:9090（或 SSH 隧道） |
| 全局透明代理 | 无 | **TUN 模式**（国内直连、海外代理，非全量劫持） |
| 重启恢复 | — | **systemd 自动启动**（110 秒全量恢复） |

---

## 十、维护清单

GFW 不是静态的——mirror 站停服、代理节点失活、新的域名被加入封锁列表都是常态。每一层都可能独立失效，需要分层巡检。

### 10.1 巡检脚本（一键）

脚本维护在 Mac 上，通过 SSH 在 NAS 远端执行。更新为宿主机版 mihomo：

```bash
#!/bin/bash
# === NAS GFW 分层巡检脚本 ===
echo "=== $(date) ==="

# ── 1. Docker Mirrors ──
echo "[1/6] Docker mirrors"
for m in \
  https://docker.1panel.live \
  https://docker.1ms.run \
  https://docker.m.daocloud.io \
  https://mirror.iscas.ac.cn; do
  code=$(curl -sk --connect-timeout 3 -o /dev/null -w '%{http_code}' "$m/v2/" 2>/dev/null)
  case $code in
    200|401) echo "  OK  $m ($code)" ;;
    *)       echo "  FAIL $m ($code)" ;;
  esac
done

# ── 2. pip Mirror ──
echo "[2/6] pip mirror"
code=$(curl -sk --connect-timeout 3 -I -o /dev/null -w '%{http_code}' \
  https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null)
[ "$code" = "200" ] && echo "  OK  aliyun" || echo "  FAIL aliyun"

# ── 3. npm Mirror ──
echo "[3/6] npm mirror"
code=$(curl -sk --connect-timeout 3 -I -o /dev/null -w '%{http_code}' \
  https://registry.npmmirror.com/ 2>/dev/null)
[ "$code" = "200" ] && echo "  OK  npmmirror" || echo "  FAIL npmmirror"

# ── 4. mihomo Explicit Proxy (port 7890) ──
echo "[4/6] mihomo explicit proxy"
for target in https://github.com https://www.google.com https://registry-1.docker.io/v2/; do
  code=$(curl -sk --connect-timeout 5 -x http://127.0.0.1:7890 \
    -o /dev/null -w '%{http_code}' "$target" 2>/dev/null)
  case $code in
    200|301|302|401) echo "  OK  proxy → $target ($code)" ;;
    *)               echo "  FAIL proxy → $target ($code)" ;;
  esac
done

# ── 5. mihomo TUN Transparent Proxy (no -x) ──
echo "[5/6] mihomo TUN (transparent)"
for target in https://github.com https://www.googleapis.com/ https://www.google.com; do
  code=$(curl -sk --connect-timeout 5 -o /dev/null -w '%{http_code}' "$target" 2>/dev/null)
  case $code in
    200|301|302|404) echo "  OK  TUN → $target ($code)" ;;
    *)               echo "  FAIL TUN → $target ($code)" ;;
  esac
done

# ── 6. mihomo Process ──
echo "[6/6] mihomo process"
if ps aux | grep -v grep | grep -q mihomo-host; then
  echo "  OK  mihomo-host running (systemd)"
else
  echo "  FAIL mihomo-host not running"
fi
echo "=== done ==="
```

用法不变（脚本归档在 [`assets/scripts/nas/nas_gfw_check.sh`]({{ '/assets/scripts/nas/nas_gfw_check.sh' | relative_url }})，下文 `convert_clash_to_mihomo.rb` 同样放在同一目录）：

```bash
# Mac 端远程巡检
cat nas_gfw_check.sh | ssh opennas "bash -s"

# 别名
alias nas-check="cat ~/claudespace/nas_gfw_check.sh | ssh opennas 'bash -s'"
```

### 10.2 分层巡检矩阵

| 巡检层 | 频率 | 典型失效模式 | 测试命令 | 恢复手段 |
|--------|------|-------------|---------|---------|
| Docker mirror | **每月** | mirror 站停服、白名单移除 | `curl -sk 'https://<mirror>/v2/'` | 从社区找新 mirror，更新 `dockerd.json` |
| pip mirror | **每月** | aliyun/tuna 停服 | `curl -skI 'https://mirrors.aliyun.com/pypi/simple/'` | 切到 `pypi.org` 直连（未被墙） |
| npm mirror | **每月** | npmmirror 停服 | `curl -skI 'https://registry.npmmirror.com/'` | 切回 `registry.npmjs.org`（未被墙） |
| mihomo 进程 | **每周** | 进程 crash、OOM | `ps aux \| grep mihomo-host` | `systemctl restart mihomo` |
| mihomo 代理功能 | **每周** | 节点失活、订阅过期 | `curl -x :7890 https://github.com` | 更新 config.yaml → `systemctl restart mihomo` |
| mihomo TUN 透明代理 | **每周** | TUN 接口丢失、iptables 规则丢失 | `curl https://github.com`（不带 -x） | `systemctl restart mihomo`（auto-redirect 重建规则） |
| 代理延迟 | **每月** | 节点劣化、线路变更 | `curl -x :7890 -w '%{time_total}' https://github.com` | 更换优先节点 |
| GFW 新增封锁 | **每季度** | 新域名被墙 | 全量 `curl --connect-timeout 5` | 加入代理规则列表 |
| tun 模块 | **每季度** | 内核升级后模块不兼容 | `ls /dev/net/tun` | `insmod /lib/modules/tun.ko` |
| systemd 服务 | **每季度** | 服务 disabled | `systemctl is-enabled mihomo` | `systemctl enable mihomo` |

### 10.3 失效场景与恢复路径

**场景 1：Docker pull 突然超时**

```
症状: docker pull 报 i/o timeout
排查: curl -sk 'https://docker.1panel.live/v2/' 逐个测试所有 mirror
根因: 全部 mirror 同时失效
恢复:
  方案A — 从社区找新 mirror 加入 dockerd.json
  方案B — TUN 透明代理已部署，给 dockerd.json 加回 proxies 字段：
  { "proxies": { "http-proxy": "http://127.0.0.1:7890", "https-proxy": "http://127.0.0.1:7890" } }
  systemctl restart pkgctl-ContainerManager
```

**场景 2：mihomo-host 停止运行**

```
症状: curl -x :7890 https://github.com 超时，ps aux | grep mihomo-host 无结果
排查:
  1. systemctl status mihomo 看退出原因
  2. journalctl -u mihomo --since "10 min ago"
恢复:
  方案A — systemctl restart mihomo
  方案B — 如果 systemd 也挂了: nohup /volume2/docker/mihomo/mihomo-host -d /usr/local/etc/mihomo &
预防: Restart=on-failure 已配置，异常退出 10 秒后自动重启
```

**场景 3：TUN 透明代理失效**

```
症状: curl https://github.com 超时（不带 -x），但 curl -x :7890 https://github.com 正常
排查:
  1. iptables -t nat -L mihomo-output -n 看 REDIRECT 规则是否存在
  2. ip addr show utun 看 TUN 接口
恢复: systemctl restart mihomo（auto-redirect 会自动重建所有 iptables 规则）
```

**场景 4：Cloud Sync Google Drive 再次超时**

```
症状: 同场景 3（TUN 失效 → Cloud Sync 立即受影响）
排查: 同场景 3
恢复: 同场景 3
注意: Cloud Sync 无需任何特殊配置，TUN 恢复后自动重连
```

**场景 5：重启后 tun 模块未加载**

```
症状: systemctl status mihomo 显示 failed，journalctl 中有 "no such file or directory"
根因: /dev/net/tun 不存在（内核模块加载失败）
排查: insmod /lib/modules/tun.ko 2>&1
恢复: 手动加载后 systemctl restart mihomo
预防: ExecStartPre=-/sbin/insmod /lib/modules/tun.ko 已配置（- 前缀容错）
```

### 10.4 建议的维护节奏

```
每周 (5 min):
  └── ps aux | grep mihomo-host → 确认进程活着
  └── curl -x :7890 github.com → 确认代理通
  └── curl github.com（不带 -x）→ 确认 TUN 通

每月 (15 min):
  └── 执行 nas_gfw_check.sh（含 TUN 检查）
  └── 检查 systemctl status mihomo
  └── 更新 Clash 订阅 → convert_clash_to_mihomo.rb → SCP config 到 NAS
  └── systemctl restart mihomo

每季度 (30 min):
  └── 全量 GFW audit: curl 测试所有关键域名直连状态
  └── 检查 Docker/pip/npm mirror 站存活列表
  └── 检查 tun 模块兼容性（尤其 DSM 升级后）
  └── 清理 docker system prune -f
```

> 巡检脚本放在 Mac 上，不要放 NAS。如果 mihomo 挂了，NAS 上的 cron 也发不出通知。

---

## 总结

与 [Docker 镜像加速](/posts/docker-registry-mirrors-china/) 博文配套食用。整体反墙架构演进：

```
v1: Docker mihomo (HTTP/SOCKS proxy)
    ├── curl -x / git config http.proxy → 手动声明代理
    └── 问题: Cloud Sync 等闭源软件不认代理变量

v2: 宿主机 mihomo + TUN 透明代理（最终形态）
    ├── 国内 CDN mirror  → Docker/pip/npm 加速
    ├── TUN 透明代理     → 所有 TCP 流量按规则自动分流
    │   ├── GEOIP CN     → DIRECT（国内 OSS/网盘直连）
    │   ├── 被墙域名     → Proxy（GitHub / Google / Docker Hub）
    │   └── DNS fake-ip  → 域名级精确路由
    ├── Cloud Sync       → Google Drive 无需配置自动通
    └── systemd 自启     → 重启后 110 秒全量恢复
```

- **Mirror** 解决国内访问速度（日常主力，走国内 CDN）
- **TUN 透明代理** 解决被墙资源（全自动，应用无感知）

二者互补，不是替代。这个分层策略同样适用于任何在中国大陆运行的家用服务器——NAS、软路由、开发机都适用。

> 与 NAS 系列联动：mihomo TUN 打通后，[之前的自建 DERP 计划](/posts/synology-nas-advanced-exploration/)（Task 13）的 AWS EC2 新加坡节点也可以通过代理 SSH 部署。GitHub 可达性为后续 Gitea + Docsify 文档流水线扫清了障碍。Cloud Sync 恢复后，NAS 可以正常同步国内外多网盘。
