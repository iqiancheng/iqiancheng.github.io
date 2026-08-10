---
layout: post
author: Joseph
title: "Tailscale 接家里 NAS 踩坑全记录：SSH 免密 / DERP / NAT 打洞 / Peer Relay"
date: 2026-05-10 00:00:00 +0800
categories: [homelab, networking]
tags: [networking, nas]
description: >
  在华为凌霄子母路由 Q6 + 移动宽带环境下，给 Synology SA6400 NAS 配置 Tailscale 的完整实战记录。
  覆盖 SSH 免密登录、DSM 透明代理踩坑、NAT 打洞失败、DERP vs Peer Relay 原理解析，以及 WebDAV 替代 SMB 的实际性能验证。
toc: true
---

## 背景

家里的 Synology SA6400（DSM 7.x, 内核 5.10.55）搭载在华为凌霄子母路由 Q6 后面，运营商是中国移动。目标是让手机在外面能流畅访问 NAS 文件，同时 NAS 加入 Tailscale 网络便于远程管理。

网络拓扑：

```
Internet
  │
  ├── 手机 (Android, 移动 5G CG-NAT)
  │
  ├── AWS EC2 (ap-southeast-1, 公网 IP)
  │
  └── 华为 Q6 路由 (192.168.x.1)
        └── Synology SA6400 (192.168.x.80, SSH 9525)
```

---

## 一、SSH 免密登录：比你想象的复杂

### 1.1 SSH Config 优化

目标：`ssh opennas` 一键登录。首先配置 `~/.ssh/config`：

```
Host opennas
    HostName 192.168.x.80
    Port 9525
    User <user>
    IdentityFile ~/.ssh/id_rsa
    # 连接复用
    ControlMaster auto
    ControlPath ~/.ssh/.control/%C.sock
    ControlPersist 10m
    # 心跳保活
    ServerAliveInterval 60
    ServerAliveCountMax 5
    # 加速 & 消除警告
    Compression yes
    StrictHostKeyChecking accept-new
```

**连接复用效果（实测）**：第一次连接走完整 SSH 握手，第二次及后续 `ssh opennas` **毫秒级完成**，无需重新认证。`ControlPersist 10m` 保证空闲 10 分钟后自动回收连接，避免 socket 残留。

### 1.2 Synology 的 authorized_keys 陷阱

**排错路径**：`ssh-copy-id` 报权限拒绝 → 怀疑 home 目录权限 → SSH 进 NAS 手动 `chmod` → root 也报 read-only → 怀疑是文件系统级限制而非权限问题 → 查 DSM 文档确认 Btrfs 子卷断电后自动 lock write。

常规 `ssh-copy-id` 在 Synology 上失效。`/var/services/homes/` 是 Btrfs 子卷挂载，断电重启后 Storage Manager 自动锁写——**root 都写不进去**：

```bash
root@OpenNAS:~# echo 'ssh-ed25519 AAAAC3Nza...' >> /volume1/homes/<user>/.ssh/authorized_keys
-ash: /volume1/homes/<user>/.ssh/authorized_keys: Read-only file system

root@OpenNAS:~# chmod 700 /volume2/homes/<user>/.ssh
chmod: changing permissions of '/volume2/homes/<user>/.ssh': Read-only file system
```

> 关键证据：Synology 的 Btrfs 子卷在异常断电后进入只读保护模式，即使 root 也无法写入。必须先通过 DSM Web → 存储管理器解锁。

解决路径：

1. DSM Web → 存储管理器 → 解锁 Volume 写入
2. `ssh-copy-id opennas` 正常执行
3. 验证免密登录：

```bash
$ ssh opennas "uname -a"
Linux OpenNAS 5.10.55+ #69057 SMP ... x86_64 GNU/Linux synology_epyc7002_sa6400
```

---

## 二、Tailscale 安装：透明代理的连环坑

### 2.1 通用脚本不认 Synology

```bash
$ curl -fsSL https://tailscale.com/install.sh | sh
# OS=other-linux
# VERSION=
# PACKAGETYPE=
# UNAME=Linux OpenNAS 5.10.55+ ... synology_epyc7002_sa6400
# No /etc/os-release
#
# → Couldn't determine what kind of Linux is running.
```

Synology 没有 `/etc/os-release`，install.sh 的分发版检测逻辑全部失效。

**正确路径**：从 [pkgs.tailscale.com/stable](https://pkgs.tailscale.com/stable/) 手动下载 DSM7 x86_64 SPK 包：

```
tailscale-x86_64-1.96.4-700096004-dsm7.spk  (35MB)
```

### 2.2 透明代理劫持一切 HTTPS 流量

**排错路径**：`curl` 报 `Failed to connect to 192.168.x.229:7890` → 7890 是 Clash 默认端口 → 怀疑环境变量 `http_proxy` → 检查为空 → 排除环境变量 → 检查 `~/.curlrc` `/etc/wgetrc` `/etc/environment` 全部为空 → 排除应用层 → 怀疑系统级 iptables 透明代理 → 确认是之前 Docker 部署的 Clash 留下的 NAT 重定向规则未清理。

NAS 上所有 curl/wget 请求被 iptables 透明代理劫持到局域网代理服务器：

```bash
$ curl -v http://google.com
* Failed to connect to 192.168.x.229 port 7890 after 4 ms: Error

$ curl -fsSL https://pkgs.tailscale.com/
* Failed to connect to 192.168.x.229 port 7890 after 3093 ms: Error
```

即使 DSM 控制面板中关闭了代理设置，iptables 的透明代理规则仍生效。Docker/Clash 类应用可能留下残留 NAT 规则。

**解法**：本地下载 SPK → 管道传输到 NAS `/tmp/` → 手动安装：

```bash
# 本地
curl -o tailscale-dsm7.spk "https://pkgs.tailscale.com/stable/tailscale-x86_64-1.96.4-700096004-dsm7.spk"

# 传到 NAS（scp 因 home 目录只读也失败，改用 pipe）
cat tailscale-dsm7.spk | ssh opennas "cat > /tmp/tailscale-dsm7.spk"

# NAS 终端安装
sudo synopkg install /tmp/tailscale-dsm7.spk
```

### 2.3 sudo requiretty 问题

Synology 的 sudo 默认 `requiretty`，自动化 SSH 无法执行 sudo 命令：

```bash
$ ssh opennas "sudo whoami"
sudo: a terminal is required to read the password
```

`tailscale up` 又必须 root。只能手动在 NAS 交互终端执行：

```bash
ssh opennas  # 交互式登录
sudo -i
/var/packages/Tailscale/target/bin/tailscale up --auth-key=tskey-auth-xxx
```

Binary 路径为 `/var/packages/Tailscale/target/bin/tailscale`，不在默认 `$PATH` 中。

---

## 三、NAT 打洞与端口转发的真相

**核心问题**：手机到 NAS 为什么走 relay 而不是 direct？

**递归排查思路**：

```
Step 1: tailscale status → 看到 relay → 确认非直连
Step 2: tailscale netcheck → PortMapping 为空 → NAT 遍历能力不足
Step 3: tailscale ping 两端互测 → DNS/路由正常，但打洞失败
Step 4: 检查两侧 NAT 类型 → NAS 侧锥形 NAT（可控），手机侧 CG-NAT（不可控）
Step 5: 结论 → 单向开端口不够，需要双侧可触达
```

### 3.1 Tailscale 网络诊断

安装完成后，先看全网拓扑和节点连通性：

```bash
$ tailscale status
100.66.x.x    <nas-host>       linux    -
100.98.x.x   <mac-device>           macOS    offline, last seen 1d ago
100.117.x.x  ip-172-31-xx-xx  linux    idle; offers exit node
100.98.x.x   <ios-device>    iOS      -
100.112.x.x   <android-device>           android  active; relay "sfo", tx 3660784 rx 367668
```

> 关键证据：手机 `<android-device>` 显示 `relay`，不是 `direct`，流量走 Tailscale 官方 DERP 中继。

**NAT 遍历能力检查**：

```bash
$ tailscale netcheck
* UDP: true
* IPv4: yes, 120.229.xx.xx:9650
* IPv6: no, but OS has support
* MappingVariesByDestIP: false
* PortMapping: UPnP          ← 首次为空，后续重检测发现已可用
* Nearest DERP: Hong Kong
* DERP latency:
    - hkg: 34.1ms  (Hong Kong)
    - sin: 63ms    (Singapore)
    - tok: 71.6ms  (Tokyo)
    - lax: 168.2ms (Los Angeles)
    - sfo: 180.7ms (San Francisco)
    ...
```

关键发现：
- **PortMapping: UPnP** — 华为 Q6 路由器支持 UPnP，已被 Tailscale 检测到（首次检测时为空，可能与路由器状态有关）
- 最近 DERP 是香港（23ms），但实际连接却走了旧金山（180ms+）

### 3.2 端口转发的实验：失败

在华为 Q6 路由器 Web 管理后台手动添加端口转发规则：UDP 41641 → 192.168.x.80:41641。结果：

```bash
$ tailscale netcheck   # 转发前后对比
PortMapping: (空)      # 无变化

$ tailscale ping 100.112.x.x
pong from <android-device> via DERP(sfo) in 371ms
pong from <android-device> via DERP(sfo) in 385ms
pong from <android-device> via DERP(sfo) in 805ms
...
direct connection not established   # 始终无法直连
```

> 关键证据：手动端口转发对 Tailscale 的 NAT 遍历**无效**。Tailscale 使用自己的一套 NAT 穿透算法（STUN + TURN 变体），不依赖系统级端口转发。

### 3.3 端口触发的实验：同样失败

改用端口触发（Port Triggering）策略后：

```bash
$ tailscale status
100.112.x.x  <android-device>  android  idle, tx 153704 rx 37480   # 从 relay 退到 idle

$ tailscale ping 100.112.x.x
pong from <android-device> via DERP(sfo) in 841ms
pong from <android-device> via DERP(sfo) in 844ms
pong from <android-device> via DERP(sfo) in 1.069s
...
direct connection not established   # 依然失败
```

更糟糕的是，中继切换到了更远的 **San Francisco**（原先是 hkg 27ms，现在 sfo 180ms+），ping 从 ~400ms 恶化到 1s+。

**结论：手机侧 CG-NAT 是单向端口转发无法解决的硬伤。** UDP 打洞需要双方都能被外部触达，而运营商级 NAT 让手机成为一个不可预测的黑洞。

---

## 四、DERP vs Peer Relay：原理解析 + 实测对比

### 4.1 Tailscale 三层连接优先级

| 优先级 | 类型 | 路径 | 实测延迟 |
|--------|------|------|----------|
| 1 | **Direct** | P2P UDP 直连 | NAS ↔ EC2: 258ms（跨太平洋） |
| 2 | **Peer Relay** | 经 tailnet 内其他节点中转 | 未触发（见下文） |
| 3 | **DERP** | Tailscale 官方中继服务器 | NAS ↔ 手机: 370ms~1.2s（SFO） |

连接协商流程：

```
启动 → DERP（临时）→ 尝试 Direct → 失败 → 尝试 Peer Relay → 失败 → 回退 DERP
```

### 4.2 Peer Relay 实测：理想丰满，现实骨感

**分析链**：

```
Q: NAS 和 EC2 能直连吗？
A: tailscale ping 100.117.x.x → 258ms via 54.89.xx.xx:48365 → YES

Q: EC2 有公网 IP 吗？
A: 54.89.xx.xx 是 AWS 弹性 IP → YES，天然适合做中转

Q: 为什么 peer relay 没触发？
A: tailscale status 仍显示 "relay" 而非 "peer-relay"

Q: 手机能直连 EC2 吗？
A: 推断不能——手机 CG-NAT 同样阻止了 EC2→手机的打洞
```

核心矛盾：peer relay 要求中转节点对**两端**都能建立直接 UDP 通道，但蜂窝网络 CG-NAT 让手机对任何外部节点都是单向可达。

理论最优路径：

```
NAS ←→ EC2 (ap-southeast-1, 公网 IP) ←→ 手机
      [直连 258ms]          [公网可达]
```

实际验证 NAS ↔ EC2 直连能力：

```bash
$ tailscale ping 100.117.x.x
pong from ip-172-31-xx-xx via DERP(iad) in 718ms   # 初始走中继
pong from ip-172-31-xx-xx via DERP(iad) in 883ms
pong from ip-172-31-xx-xx via 54.89.xx.xx:48365 in 258ms  # 成功升级到直连!
```

> 关键证据：NAS 和 EC2 之间**可以建立直接连接**（258ms），EC2 有公网 IP `54.89.xx.xx`，是天然的 peer relay 候选节点。

但刷新 `tailscale status` 后：

```bash
100.117.x.x  ip-172-31-xx-xx  linux  active; offers exit node; direct 54.89.xx.xx:48365
100.112.x.x   <android-device>           android  active; relay "sfo", tx 3774132 rx 375240
                                      ^^^^^^^^^^^^^
                                      仍然是 DERP relay！
```

Peer relay **未触发**。推测原因：
1. 手机侧 CG-NAT 过于严格，即使公网 EC2 也无法稳定建立双向 UDP 通道
2. Tailscale v1.96.4 的 peer relay 选择算法可能偏向于已经建立的连接（DERP），切换不积极
3. EC2 到手机的 UDP 打洞同样被 CG-NAT 阻断

> 教训：Peer relay 在「家庭宽带 ↔ VPS」场景下效果很好，但在「蜂窝网络 ↔ VPS」场景下仍然受限于运营商 NAT。

### 4.3 DERP 中继延迟测试：不稳定是常态

同一台手机，不同时间的 DERP 中继节点和延迟漂移：

```bash
# 时刻 1: 香港
100.112.x.x  <android-device>  android  active; relay "hkg", tx 147768 rx 43080
# ping: ~390ms

# 时刻 2: 旧金山
100.112.x.x  <android-device>  android  active; relay "sfo", tx 3660784 rx 367668
# ping: ~380ms（相对稳定）

# 时刻 3: 旧金山波动
pong via DERP(sfo) in 371ms
pong via DERP(sfo) in 1.279s    # 波动巨大
pong via DERP(sfo) in 805ms
```

> DERP 节点由 Tailscale 自动选择，不可手动指定。连接质量取决于当前 DERP 服务器的负载和跨洋链路状态。

### 4.4 Exit Node 区域选择：香港 vs 新加坡 vs 东京

当前 EC2 在 us-east-1，NAS 到它的 DERP 延迟 231ms。如果要让 EC2 兼做低延迟 Exit Node（访问 GPT/Claude API + peer relay 候选），区域选择直接影响体验。

从 NAS 实测的 DERP 延迟倒推各区域网络质量：

| 区域 | NAS 延迟 | 推荐度 | 理由 |
|------|---------|--------|------|
| 香港 ap-east-1 | **26ms** | 最优 | 离大陆最近，API 直连无 GFW；需在 AWS Console enable region |
| 新加坡 ap-southeast-1 | **55ms** | 首选 | 移动线路好，无需审批，价格正常 |
| 首尔 ap-northeast-2 | ~60ms | 备选 | 跟新加坡接近 |
| 东京 ap-northeast-1 | 73ms | 备选 | 成熟区域 |
| us-west-2 | ~170ms | 不推荐 | API 近但 NAS 远，手机到 EC2 差 3 倍 |
| us-east-1（当前） | 231ms | 别用了 | 绕地球半圈 |

> 结论：**新加坡 ap-southeast-1**（首选）或**香港 ap-east-1**（如果在意那 29ms 差距且不介意价格高 20-30%）。GPT/Claude API 推理时间 2-10 秒，EC2 到 API 那几十毫秒可忽略——瓶颈在 NAS 到 EC2 这一段。

AWS 没有台湾 region，如果执意要台湾机房，GCP asia-east1（彰化）装 Tailscale 一样用，但体验跟香港/新加坡没区别。

> 补充：X 上常有人推荐 DMIT 等 VPS 商家的香港/东京节点，因为它们走 **CN2 GIA（电信）/ CMI（移动）** 优化线路，高峰期不丢包、延迟稳定在 25-40ms。AWS/GCP 走的是普通 transit，移动用户高峰期容易拥堵（这也解释了 DERP hkg 延迟从 26ms 抖到 400-900ms 的现象）。但 DMIT 价格 $15-30/月起，是 AWS Lightsail 的 3-6 倍。**连回家看文件 500KB/s 够用，AWS 足够了——DMIT 是翻墙用户追求极致稳定的价位。**

---

## 五、应用层优化：甩掉 SMB，上 WebDAV

### 5.1 为什么 SMB 不适合广域网？

SMB 协议的多 TCP 连接 + 大量握手交换模型，在 >100ms 延迟下退化为不可用：

- 每个文件操作触发 10+ 次 round-trip
- 300ms DERP 延迟 → 单次操作 3-6 秒
- 目录列举（大量小文件） → 数十秒甚至超时

### 5.2 WebDAV 部署与验证

**5 分钟部署**：DSM → 套件中心 → 安装 WebDAV Server → 启用 HTTPS 端口 5006。

本地验证：

```bash
$ curl -sk -X PROPFIND -u <user>:xxx https://localhost:5006/
HTTP 403    # WebDAV 根目录不公开暴露，正常

$ curl -sk -X PROPFIND -H 'Depth: 1' -u <user>:xxx https://localhost:5006/volume2/
HTTP 405    # 需在 DSM 共享文件夹中手动为 WebDAV 开权限
```

> WebDAV Server 安装后默认不暴露任何共享文件夹——需要到「控制面板 → 共享文件夹 → 选中目标 → 编辑 → 权限 → 勾选 WebDAV」。

配置完成后，手机 Cx File Explorer 连接：

```
协议:    WebDAV (HTTPS)
地址:    https://100.66.x.x:5006
端口:    5006
用户名:  <user>
```

### 5.3 实际体验对比

| 场景 | SMB over DERP | WebDAV over DERP |
|------|--------------|-----------------|
| 浏览目录（~50 个文件） | 8-15 秒 | 2-4 秒 |
| 下载 10MB 文件 | 不稳定，易断 | 稳定 |
| 上传照片 | 慢 | 可接受 |
| 播放视频 | 卡顿严重 | 可流畅播放 |

---

## 六、网络层优化：TCP 内核调优 + Apache 线程池

解决了应用层协议（SMB → WebDAV）后，下一步深挖 TCP 协议栈和 Web 服务器本身。

### 6.1 TCP 缓冲区：208KB 的隐形天花板

Synology DSM 7.x 的默认 TCP 内核参数出奇保守：

```bash
$ sysctl net.core.rmem_max net.core.wmem_max
net.core.rmem_max = 212992          # ← 208KB
net.core.wmem_max = 212992          # ← 208KB
```

> 关键证据：`rmem_max = wmem_max = 208KB`。这个值直接限制了 TCP 接收窗口上限，进而限制**带宽延迟积（BDP）**。

不同 DERP 节点下的理论最大吞吐量：

| DERP 节点 | 延迟 | 208KB BDP 上限 | 2MB BDP 上限 |
|-----------|------|---------------|-------------|
| hkg (香港) | 26ms | ~64 Mbps | ~615 Mbps |
| sin (新加坡) | 55ms | ~30 Mbps | ~290 Mbps |
| tok (东京) | 73ms | ~23 Mbps | ~219 Mbps |
| lax (洛杉矶) | 168ms | ~10 Mbps | ~95 Mbps |
| sfo (旧金山) | 174ms | ~**9 Mbps** | ~89 Mbps |

> 手机走 LAX/SFO DERP 时，TCP 窗口锁死 ~9Mbps，即使链路实际有 30Mbps+ 也白费。

**修复**（需 root）：

```bash
sysctl -w net.core.rmem_max=2097152
sysctl -w net.core.wmem_max=2097152
sysctl -w net.ipv4.tcp_rmem="4096 131072 2097152"
sysctl -w net.ipv4.tcp_wmem="4096 16384 2097152"
sysctl -w net.ipv4.tcp_slow_start_after_idle=0
sysctl -w net.ipv4.tcp_fastopen=3
```

Synology 内核 5.10.55+ 未编译 BBR 和 fq_codel 模块，`cubic + pfifo_fast` 是唯一可用组合。

### 6.2 Apache MPM：ThreadsPerChild = 1

WebDAV Server 底层是 Apache 2.4.58，MPM worker。Synology 默认配置：

```
ThreadsPerChild        1      # ← 等于没用多线程
MinSpareThreads        1
MaxSpareThreads        1
StartServers           2
```

> 关键证据：`ThreadsPerChild = 1` 退化为 prefork 模式，每个请求 fork 一个 17MB 进程，无法利用线程级并发。

**修复**（需 root，配置文件 `/var/packages/WebDAVServer/target/etc/httpd/conf/extra/httpd-mpm.conf-webdav`）：

```
StartServers          3
MinSpareThreads       25
MaxSpareThreads       75
ThreadsPerChild       25
ServerLimit           6
MaxRequestWorkers     150
```

改完后 `synopkg restart WebDAVServer` 即可生效。

> 踩坑：`synopkg restart` 报 error 272。排查发现粘贴 heredoc 时混入了 non-breaking space（`\xc2\xa0`），Apache 语法检查报 `Invalid command`。`sed -i 's/\xc2\xa0/ /g'` 清理后正常。

### 6.3 实测对比

局域网基准（下载 10MB，macOS → 192.168.x.80:5006）：

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 平均速度 | ~4.2 MB/s | ~**9.8 MB/s** | **2.3x** |
| 最快单次 | 5.3 MB/s | 11.3 MB/s | 2.1x |

> 服务端 localhost 基准（NAS → localhost:5006）为 **687 MB/s**——磁盘和 WebDAV 逻辑不是瓶颈，问题全在网络栈和并发处理。

### 6.4 移动端实测：碎文件 vs 大文件的真相

关掉手机 Exit Node 后，DERP 自动切到 hkg（26ms）：

```
$ tailscale ping 100.112.x.x
pong from v2324a via DERP(hkg) in 405ms
pong from v2324a via DERP(hkg) in 619ms
pong from v2324a via DERP(hkg) in 539ms
direct connection not established
```

实测下载（手机蜂窝 → NAS over DERP hkg）：

| 文件类型 | 速度 | 原因 |
|----------|------|------|
| 碎文件（多个 10-100KB） | **~100KB/s** | 每个文件一次 HTTP GET，RTT 吃掉了 90% 时间 |
| 压缩包（单文件 10MB+） | **~500KB/s** | 一次握手后 TCP 流持续传输，窗口跑满 |

> 关键证据：压缩包 500KB/s 对比优化前的 100KB/s，**TCP 缓冲区 208KB→2MB 在蜂窝链路上确实生效了（5x 提升）**。碎文件慢不是带宽问题——HTTP/1.1 每个 GET 都是独立请求，400-900ms 延迟下，传输 50KB 只需要 0.1 秒，但握手等 RTT 耗了 0.5 秒：
> ```
> 下载 10KB × 10 个文件:
>   传输时间: 0.02s × 10 = 0.2s
>   RTT 开销: 0.5s × 10 = 5s       ← 罪魁祸首
>   实际耗时: ~5.2s, 有效速率 ≈ 20KB/s
> ```

**结论**：蜂窝 DERP 的「最后一公里」延迟是硬天花板，TCP 调优对大文件传输有效（5x），但对碎文件的协议开销无能为力。临时方案：在 NAS 上把碎文件 zip 打包再下载。

### 6.5 Tailscale Serve：减去一层 TLS

Serve 已部署，在 NAS 上运行：

```bash
$ tailscale serve --bg https+insecure://localhost:5006
Available within your tailnet:

https://openai.<tailnet>.ts.net/
|-- proxy https+insecure://localhost:5006
```

链路变化：

```
之前: 手机 → TLS → Apache:5006（自签名证书，curl 得 -k）
现在: 手机 → TLS → Tailscale Serve → localhost:5006（正规证书，本地回环）
             └── Tailscale CA 签发证书     └── 流量不出 NAS
```

好处：
- 证书从 Synology 自签名变成 Tailscale CA 正规签发，手机不再报证书警告
- Serve → WebDAV 走 `localhost`，省掉外部 TLS 加解密开销
- 仍然是 tailnet 内网地址（`*.ts.net` 公网不可解析），**不增加安全风险**
- `https+insecure` 仅指 Serve 不验证后端（localhost 无所谓）

### 6.6 Tailscale Funnel：为什么不启用

Funnel 跟 Serve 的区别在于 **Funnel 把服务暴露到公网**：

| | Serve | Funnel |
|---|---|---|
| 访问范围 | Tailnet 内网 | **公网可访问** |
| 协议路径 | WireGuard/UDP → DERP → Serve | HTTPS/TCP 443 → 边缘节点 → 内网 |
| 运营商视角 | UDP 隧道（可能被 QoS） | 标准 HTTPS（不可区分） |
| 安全风险 | 内网隔离 | 公网暴露，靠 WebDAV Auth 挡 |
| 预期速度 | ~500KB/s（当前） | 可能 2-5 MB/s |

Funnel 走 TCP 443 不受运营商 UDP QoS 影响，速度可能到 2-5 MB/s。但代价是把 WebDAV 暴露到公网——虽然 Basic Auth + 强密码 + 自动封锁在，但攻击面确实比纯 tailnet 大得多。

当前 500KB/s 够日常用，决定不启用 Funnel。

---

## 七、总结：排查方法论

回顾整个过程，每个问题的诊断都遵循同一套方法：

```
现象观察 → 假设驱动 → 交叉验证 → 逐步排除 → 锁定根因 → 对症下药
```

| 现象 | 首轮假设 | 验证手段 | 排除后 | 最终根因 |
|------|---------|---------|--------|---------|
| authorized_keys 写不进去 | 权限问题 | `chmod` 报 read-only | 怀疑文件系统 | Storage Manager Btrfs 锁写 |
| curl 无法访问外网 | http_proxy 环境变量 | `env \| grep proxy` 为空 | 怀疑系统级代理 | iptables 透明代理劫持 |
| `tailscale up` 失败 | 未安装 | `which tailscale` 找不到 | 确认已装但 PATH 不对 | Synology SPK 装到非标路径 |
| tailscale status 显示 relay | 路由器无 UPnP | `netcheck` 确认 PortMapping 空 | 手动端口转发也无效 | 手机侧 CG-NAT |
| peer relay 不触发 | EC2 不在线 | ping EC2 直连可达 | 排除 EC2 侧问题 | 手机 CG-NAT 双向打洞失败 |
| 手机 WebDAV 速度慢 | DERP 延迟高 | 关掉 Exit Node 切到 hkg | DERP 已切 hkg 仍 100KB/s | DERP 共享带宽 + UDP QoS 限速 |
| 局域网也跑不满千兆 | 协议开销 | WebDAV LAN 测试仅 4.2 MB/s | TLS + 单线程 Apache 各占一半 | TCP 208KB + ThreadsPerChild=1 |
| `synopkg restart` 报 error 272 | 配置语法错误 | `httpd -t` 语法检查 | 原配置 OK，修改后才出错 | heredoc 引入 non-breaking space |

### 踩坑清单

1. **Synology Storage Manager 重启后锁写** → `authorized_keys` 写不进去（root 也不行）
2. **透明代理劫持 HTTP/HTTPS** → curl/wget 全部失败，需 `--noproxy '*'`，但 install.sh 内部 curl 无法传参
3. **无 `/etc/os-release`** → Tailscale 通用安装脚本无法识别 Synology
4. **sudo requiretty** → 自动化脚本无法 sudo，需手动交互终端。解决：在 `/etc/sudoers.d/` 中添加 NOPASSWD 规则后可通过 `ssh opennas "sudo ..."` 远程执行（`visudo` 在 DSM 上不可用，直接用 echo 写入）
5. **手工端口转发对 Tailscale 无效** → Tailscale 有自己的 NAT 遍历机制，不依赖系统端口映射
6. **手机 CG-NAT** → 端口转发、端口触发都无法解决直连问题
7. **SMB over 高延迟** → SMB 协议在 >300ms 延迟下基本不可用
8. **Peer relay 未触发** → 即使有公网 VPS 作为中转候选，手机侧 NAT 仍然阻断
9. **TCP rmem/wmem 默认 208KB** → 高延迟 DERP 链路下吞吐上限仅 9Mbps（SFO），调到 2MB 解决
10. **Apache ThreadsPerChild = 1** → WebDAV Server 线程无效化，每个请求 fork 进程，调到 25 后 LAN 速度翻倍
11. **DERP 共享带宽 + UDP QoS** → 蜂窝网络走 DERP 中继仅 100KB/s，TCP 调优无法解决，需自定义 DERP 或 Funnel
12. **non-breaking space 污染** → heredoc 粘贴混入 `\xc2\xa0`，Apache 语法检查报 `Invalid command`，`sed` 修复
13. **fq_codel 确认不可用** → `sysctl -w net.core.default_qdisc=fq_codel` 返回 `No such file or directory`，确认内核未编译 `sch_fq_codel` 模块，回退到 `pfifo_fast`
14. **iptables 白名单遗漏 SMB 端口** → 启用防火墙后 macOS Finder 无法发现和连接 NAS。需放行 TCP 445/139 + UDP 137-138（NetBIOS）+ UDP 5353（mDNS），全部限定局域网来源

### 最佳实践

1. **SSH 免密** — `ControlMaster auto` + `ControlPath` + `ControlPersist`，第二次连接秒进
2. **Tailscale on Synology** — 用 SPK 包手动安装，别走 install.sh
3. **NAT 打洞失败不纠结** — 移动 CG-NAT 和家庭宽带之间的直连是玄学，接受走中继
4. **WebDAV 替代 SMB** — 高延迟链路的决定性优化，HTTP 单连接模型碾压 SMB
5. **Peer Relay 有前提** — 需要中转节点对两端都能直接建立 UDP 通道，蜂窝网络场景下未必可行
6. **关闭透明代理残留** — 检查 `iptables -t nat -L` 清理 Docker/Clash 残留规则
7. **调大 TCP 缓冲区** — `rmem_max/wmem_max` 从默认 208KB → 2MB，高延迟 DERP 链路吞吐上限提升 10 倍
8. **调优 Apache MPM** — `ThreadsPerChild` 从 1 → 25，WebDAV 局域网吞吐翻倍
9. **Android 开 Allow WLAN** — 在家走 WiFi 直连 NAS 不绕 DERP，出门才走蜂窝中继
10. **关掉手机 Exit Node 连 NAS** — Exit Node 会强制流量绕美国再回来，直连 DERP HKG 更快
11. **开启 Tailscale Serve** — `tailscale serve --bg https+insecure://localhost:5006`，Tailscale CA 正规证书 + 省一层 TLS
12. **碎文件先打包** — 高延迟链路上 HTTP/1.1 每个 GET 独立握手，碎文件先 zip 再下载，速度从 100KB/s → 500KB/s
13. **启用远程 sudo 自动化** — DSM 默认 `sudo requiretty`，在 `/etc/sudoers.d/<user>` 中添加 NOPASSWD 规则后，可通过 `ssh opennas "sudo ..."` 远程执行 root 命令，配合一键加固脚本实现无交互审计

### 最终部署形态

```
~/.ssh/config:
  Host opennas → 192.168.x.80:9525
     ControlMaster auto | keep-alive | compression

Synology SA6400:
  ├── SSH: 9525（免密，复用连接）
  ├── sudo: /etc/sudoers.d/<user> NOPASSWD（远程自动化）
  ├── Tailscale: 100.66.x.x（DERP fallback）
  ├── Tailscale Serve: https://openai.<tailnet>.ts.net（TLS 终止 → localhost:5006）
  ├── WebDAV: :5006 HTTPS
  └── iptables: 白名单 + 默认 DROP

手机:
  └── Cx File Explorer → WebDAV → https://openai.<tailnet>.ts.net

AWS EC2（推荐新加坡 ap-southeast-1）:
  └── Tailscale: 100.117.x.x（exit node + peer relay 候选，55ms）
```
