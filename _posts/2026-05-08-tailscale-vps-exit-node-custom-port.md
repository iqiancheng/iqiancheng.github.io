---
layout: post
title: "Tailscale VPS Exit Node 实战：自定义 UDP 端口 + UDP GRO 调优 + DNS 防泄漏"
date: 2026-05-08 00:00:00 +0800
author: Joseph
categories: [networking]
tags: [networking, security, devops]
---
## 背景

手上有一台闲置的境外云厂商 VPS (Ubuntu 24.04)，希望：

1. 把它接入自己的 Tailscale tailnet 作为 **exit node**
2. 让手机（Android）走这台 VPS 出口访问海外流媒体（以 YouTube 为例）
3. **自定义 WireGuard UDP 端口**（>1024 的随机端口，避开常见扫描特征）
4. **防火墙/内核转发**规范配置，避免装完用不了
5. **DNS 不泄漏**给运营商

这篇把从零到手机能刷 YouTube 的整个流程固化下来，含踩坑点。另一篇更偏 overview 的笔记见 [2026-03-03 Tailscale 家庭多设备与 VPS 组网](/posts/tailscale-home-multidevice-vps-gpt/)，本文聚焦**单台 VPS exit node 的极简实操**。

---

## 成品参数（示例）

| 项目 | 值 |
|---|---|
| OS | Ubuntu 24.04 LTS (Noble) |
| Tailscale | 1.96.x |
| WireGuard UDP port | 20000–55000 间随机一个（示例 `<PORT>`） |
| IP forwarding | v4 + v6 持久化 `/etc/sysctl.d/99-tailscale.conf` |
| ufw 入站 | `22/tcp` + `<PORT>/udp` |
| 网卡转发优化 | `rx-udp-gro-forwarding on, rx-gro-list off` |
| 客户端 DNS | Tailnet 全局 `8.8.8.8` + Override local DNS |

---

## 步骤一：探测 VPS 现状

不管怎么自动化，第一步永远是**摸清现状**，避免覆盖已有配置：

```bash
ssh vps 'echo "---OS---"; (cat /etc/os-release 2>/dev/null | grep -E "^(NAME|VERSION)=");
echo "---FW---"; (command -v ufw && sudo ufw status 2>/dev/null | head -5);
(command -v iptables >/dev/null && echo "iptables: yes");
echo "---SYSCTL---"; sysctl net.ipv4.ip_forward net.ipv6.conf.all.forwarding 2>/dev/null;
echo "---TS---"; (command -v tailscale && tailscale version) || echo "tailscale: not installed"'
```

关键事实收集：

- OS → 决定 install.sh 分支
- ufw 是否 active、已放行哪些端口
- `net.ipv4.ip_forward` 是否开（exit node 必须开）
- tailscale 是否已装过（避免重复）

---

## 步骤二：随机端口生成

macOS 默认没 `shuf`，用 awk 替代（这段可以本地跑，只为打一个数字）：

```bash
PORT=$(awk -v min=20000 -v max=55000 'BEGIN{srand(); print int(min+rand()*(max-min))}')
echo "Selected port: $PORT"
```

**为什么要随机端口**？Tailscale 默认 WireGuard UDP 端口是 `41641`。这个端口在扫描器里已经成特征，换到 20000–55000 内一个没有公开服务指纹的端口能减少主动扫描注意力。>1024 是为了避开需要 `CAP_NET_BIND_SERVICE` 的特权端口区间，同时也是所有常见防火墙默认 deny 的区间，需要显式放行。

> 以下示例中都用 `<PORT>` 占位你生成的具体端口。

---

## 步骤三：安装 + 转发 + 防火墙（一次到位）

```bash
ssh vps 'set -e
PORT=<PORT>

# 1. 安装
curl -fsSL https://tailscale.com/install.sh | sh

# 2. 开启 IP forwarding 并持久化
echo "net.ipv4.ip_forward = 1" | sudo tee /etc/sysctl.d/99-tailscale.conf
echo "net.ipv6.conf.all.forwarding = 1" | sudo tee -a /etc/sysctl.d/99-tailscale.conf
sudo sysctl -p /etc/sysctl.d/99-tailscale.conf

# 3. ufw 放行 WireGuard 端口
sudo ufw allow ${PORT}/udp comment "tailscale-wg"
'
```

为什么把转发放在 `/etc/sysctl.d/99-tailscale.conf` 而不是直接改 `/etc/sysctl.conf`？

- Ubuntu 24.04 推荐 drop-in 风格
- 数字越大优先级越高，`99-` 确保覆盖别的默认
- 单独文件，后续清理时 `rm` 一个文件就能回退

---

## 步骤四：自定义端口的**正确姿势**（一个坑）

直觉上以为：

```bash
sudo tailscale up --port=<PORT> --advertise-exit-node --auth-key=...
```

**这是错的**。Tailscale 1.96.x 执行后会报：

```
flag provided but not defined: -port
```

原因：`--port` 是 **`tailscaled` 守护进程**的参数，不是 `tailscale up` 的客户端 flag。在 Debian/Ubuntu 包里，通过 `/etc/default/tailscaled` 的 `PORT` 变量传给 systemd service。

正确做法：

```bash
# 1. 改配置
sudo sed -i "s/^PORT=.*/PORT=\"<PORT>\"/" /etc/default/tailscaled
grep ^PORT /etc/default/tailscaled
# PORT="<PORT>"

# 2. 重启守护进程（让新端口生效）
sudo systemctl restart tailscaled

# 3. 现在才是 tailscale up
sudo tailscale up --auth-key=tskey-auth-XXXX --advertise-exit-node

# 4. 验证监听
sudo ss -ulnp | grep tailscal
# UNCONN 0 0 0.0.0.0:<PORT>   ... users:(("tailscaled",...))
# UNCONN 0 0 0.0.0.0:<RANDOM> ... users:(("tailscaled",...))
```

> **注意**：`ss` 还会显示另一个随机端口，那是 tailscaled 的 portmapper / NAT-PMP 辅助端口，不是 WireGuard 数据面，**不要放行**。

---

## 步骤五：admin console 审批 exit node

即使 `--advertise-exit-node` 成功，在客户端 app 里还是会看到：

> **This machine does not expose any routes**

这是因为 Tailscale 默认**不自动审批** exit node 的 `0.0.0.0/0` + `::/0` 路由——这是一个刻意的安全设计（防止某台机器被入侵后无声无息变成中转出口）。

检查节点**其实在广播**路由：

```bash
sudo tailscale debug prefs | grep -A2 AdvertiseRoutes
# "AdvertiseRoutes": [
#     "0.0.0.0/0",
#     "::/0"
# ],
```

**手动审批**（一次性）：

1. 打开 https://login.tailscale.com/admin/machines
2. 找到对应 hostname → ⋯ → **Edit route settings**
3. 勾选 **Use as exit node** → Save

节点上就会挂上 **Exit node** badge，其他节点 app 里就能选它做出口。

> **未来自动化**：在 tailnet policy 文件里加 `autoApprovers`：
>
> ```json
> "autoApprovers": {
>   "exitNode": ["autogroup:admin"]
> }
> ```
>
> 之后单用户 tailnet 里新增 exit node 就不用再点了。

---

## 步骤六：UDP GRO 转发调优（关键性能项）

`tailscale up` 成功后会给一条警告：

```
Warning: UDP GRO forwarding is suboptimally configured on <IFACE>,
UDP forwarding throughput capability will increase with a configuration change.
See https://tailscale.com/s/ethtool-config-udp-gro
```

这不是可有可无的提示，在高带宽 exit node 场景下**关乎几倍吞吐**（官方 benchmark 从 ~1Gbps 提到线速）。

**一次性应用**：

```bash
IFACE=$(ip -o route get 8.8.8.8 \| awk '{print $5}')
sudo ethtool -K $IFACE rx-udp-gro-forwarding on rx-gro-list off
```

**持久化**（重启后也生效）—— 用 networkd-dispatcher 在接口 routable 时自动触发：

```bash
IFACE=$(ip -o route get 8.8.8.8 \| awk '{print $5}')
sudo tee /etc/networkd-dispatcher/routable.d/50-tailscale-gro >/dev/null <<EOF
#!/bin/sh
ethtool -K ${IFACE} rx-udp-gro-forwarding on rx-gro-list off 2>/dev/null || true
EOF
sudo chmod +x /etc/networkd-dispatcher/routable.d/50-tailscale-gro
```

验证：

```bash
sudo ethtool -k $IFACE | grep -E "rx-udp-gro-forwarding|rx-gro-list"
# rx-gro-list: off
# rx-udp-gro-forwarding: on
```

> **为什么不用 `/etc/rc.local` 或 systemd oneshot**？
>
> - `rc.local` 在 Ubuntu 24.04 默认没有，需要自己造
> - 独立 systemd service 要绑定网卡 ready 状态，写起来繁琐
> - networkd-dispatcher 原生监听 `routable` hook，适配虚拟化云环境网卡可能延迟出现的场景

---

## 步骤七：客户端走出口 + DNS 防泄漏

### 7.1 Android app 开出口

1. Tailscale app → 顶部菜单 / 主界面 → **Use exit node**
2. 列表选你的 VPS hostname
3. 勾选 **Allow LAN access**（保留访问本地路由器 / 打印机能力）

此时 `ifconfig.me` 应该显示 VPS 的公网 IP。

### 7.2 只切出口**还不够** —— DNS 会泄漏

即使流量走 VPS 出口，DNS 查询很可能还是**本地运营商的 DNS**（因为 Android 系统或 app 默认不覆盖 DNS）。这会导致：

- 运营商能看到你访问了哪些域名
- YouTube / Netflix 等基于 **DNS 就近 CDN 调度**的服务可能仍把你判为中国用户

**两层同时开**：

**(A) 客户端侧**：Android Tailscale app → Settings → 打开 **Use Tailscale DNS**（等价 CLI `--accept-dns=true`）

**(B) Tailnet 侧（关键）**：admin console → **DNS**

1. **Add nameserver → Custom** → 填 `8.8.8.8`（或 `1.1.1.1`）
2. **打开 Override local DNS 开关**⚠️ 这一步最容易漏
3. Save

> **为什么优先 `8.8.8.8`**？Google DNS 对 YouTube / Google 服务返回**就近 CDN 节点**。VPS 在海外时，配 `8.8.8.8` 能拿到同区域的 YouTube CDN，延迟最低；反之如果还用国内 DNS，YouTube CDN 可能还分配给中国节点，再绕回海外走出口，RTT 变差。

### 7.3 验证清单

1. **公网 IP 是否切换**：https://ifconfig.me → 显示 VPS IP ✓
2. **DNS 是否泄漏**：https://dnsleaktest.com → Standard test → **只**应看到 Google/Cloudflare，不出现运营商 DNS
3. **YouTube 可打开**：浏览器 + app 都正常

如果 `dnsleaktest` 还看到运营商 DNS：

- 客户端 "Use Tailscale DNS" 是否开着
- Tailnet "Override local DNS" 是否开着
- Android **Private DNS** 设置是否强制覆盖（设置 → 网络 → 私有 DNS → 改为**自动**或**关闭**）

---

## 常见命令速查

```bash
# 状态
sudo tailscale status
sudo tailscale netcheck                    # NAT/防火墙探测
sudo tailscale debug prefs                 # 完整偏好
sudo ss -ulnp | grep tailscal              # 监听端口

# 改 WireGuard 端口
sudo sed -i 's/^PORT=.*/PORT="新端口"/' /etc/default/tailscaled
sudo systemctl restart tailscaled
sudo ufw allow 新端口/udp comment "tailscale-wg"
sudo ufw delete allow 旧端口/udp

# 卸载
sudo tailscale down
sudo apt-get remove --purge tailscale
sudo rm /etc/sysctl.d/99-tailscale.conf /etc/networkd-dispatcher/routable.d/50-tailscale-gro
sudo sysctl -p
```

---

## Admin console 其他常见设置一览

按个人用户实用度排序：

| 功能 | 用途 | 个人实用度 |
|---|---|---|
| **DNS** | 本文已用，全局 nameserver + 防泄漏 | ⭐⭐⭐⭐⭐ |
| **Machines** | 节点列表、审批 exit node / subnet | ⭐⭐⭐⭐⭐ |
| **Keys** | auth key 管理，**用过即弃一定要撤销** | ⭐⭐⭐⭐⭐ |
| **ACLs** | 多用户/多设备隔离策略 | ⭐⭐⭐（单用户用不上） |
| **Webhooks** | 节点掉线 / key 过期时推 Bark/Telegram | ⭐⭐⭐⭐ |
| **Auto-approvers** | 新 exit node 自动审批 | ⭐⭐⭐⭐ |
| **Tailscale SSH** | 基于 tailnet 身份免密 SSH 登录 | ⭐⭐⭐⭐ |
| **Funnel** | 把 tailnet 内部服务暴露到公网 HTTPS | ⭐⭐⭐（有自建服务才用） |
| **Tailnet lock** | 防 Tailscale 控制面被黑后冒充节点 | ⭐⭐（偏执级别） |
| **Services** | 多节点虚拟 IP + DNS 做 HA | ⭐（企业场景） |
| **Apps (OAuth)** | CI / Terraform 动态签发 auth key | ⭐（自动化场景） |
| **Trust credentials** | 企业 CA / 证书 | ⭐（企业合规） |

---

## 踩坑合集

1. **`tailscale up --port` 报错** → `--port` 是 `tailscaled` 的 flag，改 `/etc/default/tailscaled` 然后 `systemctl restart tailscaled`
2. **App 显示 "does not expose any routes"** → 检查 `tailscale debug prefs` 看 `AdvertiseRoutes`，如果有 `0.0.0.0/0` 就是**admin console 没审批**
3. **转发没开导致出口不工作** → `sysctl net.ipv4.ip_forward` 必须是 1，且持久化到 `/etc/sysctl.d/`
4. **UDP GRO 没调** → 高带宽场景吞吐卡在 1Gbps，必调
5. **DNS 泄漏** → 同时要开客户端 `--accept-dns` + tailnet **Override local DNS**，少一个都漏
6. **ss 看到两个 UDP 端口** → 另一个是 portmapper，不用放行
7. **auth key 明文贴到聊天/issue** → 立刻去 admin console `Keys` 里**撤销**，换新的

---

## 最终架构图

```
  [Android phone]                                              [Internet]
       │                                                            ▲
       │ WireGuard (UDP <PORT>)                                     │
       │                                                            │
       ▼                                                            │
 [tailnet 100.x.x.x]                                                │
       │                                                            │
       └──────────▶ [VPS Ubuntu 24.04] ──── NAT/iptables ──────────┘
                    exit-node
                    ip_forward=1, UDP GRO on
                    ufw: 22/tcp, <PORT>/udp
                    DNS: 8.8.8.8 (via tailnet override)
```

## 下一步（可选增强）

- [ ] 配 Webhook → 节点掉线推 Bark/Telegram
- [ ] 在 policy 文件里加 `autoApprovers.exitNode`，省掉下次手点
- [ ] 打开 **Tailscale SSH**，以后 `tailscale ssh vps` 免密登录
- [ ] 定期 `tailscale update` 或启用 `AutoUpdate: {Check: true, Apply: true}`（本文已默认开）

---

_相关笔记_：
- [2026-03-03 Tailscale 家庭多设备与 VPS 组网：GPT 访问、出口节点与文件共享](/posts/tailscale-home-multidevice-vps-gpt/) — 更偏 overview 的家庭组网场景
