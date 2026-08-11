---
layout: post
author: Joseph
title: "家用 NAS 从裸奔到合规：12 项风险的完整审计与加固"
date: 2026-05-10 00:00:00 +0800
categories: [homelab]
tags: [networking, nas, security]
description: >
  对一台运行 DSM 7.x 的 Synology NAS 进行完整安全审计，发现 Telnet 开启、SSH 双端口暴露、防火墙裸奔、admin 账户存在等 12 项风险，并逐项修复。
toc: true
---
## 背景

同一台 Synology NAS（DSM 7.x, SA6400），在完成 [Tailscale 组网和 WebDAV 部署](/posts/tailscale-synology-nas-real-world/) 后，进行了一次完整的安全审计。

**审计前的状态**：DSM 基本默认配置 + Tailscale + WebDAV + SSH 密钥登录。没有做过任何安全加固。

**审计方法**：SSH 进 NAS，从监听端口 → 服务配置 → 账户策略 → 防火墙规则逐层排查。

---

## 一、监听端口审计：敞开的大门

### 1.1 全量端口扫描

```bash
$ netstat -tlnp | grep "0.0.0.0"
tcp  0.0.0.0:22      LISTEN    # SSH 默认端口
tcp  0.0.0.0:23      LISTEN    # Telnet ← 高危！
tcp  0.0.0.0:80      LISTEN    # HTTP
tcp  0.0.0.0:139     LISTEN    # SMB NetBIOS
tcp  0.0.0.0:443     LISTEN    # HTTPS
tcp  0.0.0.0:445     LISTEN    # SMB
tcp  0.0.0.0:873     LISTEN    # rsync
tcp  0.0.0.0:3261    LISTEN    # iSCSI
tcp  0.0.0.0:3263    LISTEN    # iSCSI
tcp  0.0.0.0:3264    LISTEN    # iSCSI
tcp  0.0.0.0:3265    LISTEN    # iSCSI
tcp  0.0.0.0:5000    LISTEN    # DSM HTTP
tcp  0.0.0.0:5001    LISTEN    # DSM HTTPS
tcp  0.0.0.0:5357    LISTEN    # WS-Discovery
tcp  0.0.0.0:6157    LISTEN    # Synology 移动服务
tcp  0.0.0.0:8000    LISTEN    # Web Station
tcp  0.0.0.0:9525    LISTEN    # SSH 自定义端口
tcp  0.0.0.0:47555   LISTEN    # 未识别服务
tcp  0.0.0.0:3014    LISTEN    # Cloud Sync?
```

> 关键证据：**27 个端口在所有网络接口 (`0.0.0.0`) 上暴露**，其中有多个是高危服务。

### 1.2 最高危：Telnet (23)

```bash
$ netstat -tlnp | grep ":23"
tcp  0.0.0.0:23  LISTEN
```

Telnet 是**明文协议**，任何在同一个 LAN 段的设备都可以抓包获取登录凭据。2026 年了没有理由开着它。

**修复**：DSM → 控制面板 → 终端机和 SNMP → **取消勾选"启用 Telnet 服务"**。

```bash
# 修复后验证
$ netstat -tlnp | grep ":23"
(空)
```

### 1.3 SSH 双端口暴露（22 + 9525）

```
tcp  0.0.0.0:22    LISTEN   ← 扫描器磁铁
tcp  0.0.0.0:9525  LISTEN   ← 自用非标端口
```

默认 22 端口是暴力破解的首要目标。既然已经配置了 9525 密钥认证，22 端口纯属多余暴露面。

**修复**：DSM → 控制面板 → 终端机和 SNMP → **将 SSH 端口只设为 9525**，移除 22。

同时也加固 SSH 配置：

```
# /etc/ssh/sshd_config 建议
PermitRootLogin no              # 禁止 root 直接 SSH
PasswordAuthentication no       # 禁用密码登录，只允许密钥
PubkeyAuthentication yes
Protocol 2                      # 只允许 SSHv2
```

> 上一篇已配置了 `~/.ssh/config` 的密钥登录和 ControlMaster，此改动不影��使用。

---

## 二、账户审计：默认账户是定时炸弹

### 2.1 发现 admin + guest 账户

```bash
$ cat /etc/passwd | grep -v nologin | grep -v false
admin:x:1024:100:System default user:/var/services/homes/admin:/bin/sh
root:x:0:0::/root:/bin/ash
<user>:x:1026:100::/var/services/homes/<user>:/bin/sh
```

> 关键证据：`admin` 是 Synology 的内置管理员，默认存在且无法通过常规方式删除，但**可以禁用**。

攻击者扫描到 DSM 登录页时，第一个试的用户名就是 `admin`。禁用它就等于废掉了暴力破解的一半威力。

**修复**：

1. DSM → 控制面板 → 用户与群组 → **禁用 `admin` 账户**
2. 同上 → **禁用 `guest` 账户**（如果有）
3. 确认 `<user>` 使用强密码 + 2FA

---

## 三、防火墙审计：裸奔状态

### 3.1 当前规则为空

```bash
$ sudo iptables -L INPUT -n
Chain INPUT (policy ACCEPT)
target     prot opt source               destination
(空)
```

> 关键证据：INPUT 链没有任何规则，**策略是 ACCEPT**。所有端口对所有来源完全开放。

### 3.2 推荐规则

DSM → 控制面板 → 安全性 → 防火墙 → 创建配置文件：

| 优先级 | 端口 | 协议 | 来源 | 用途 |
|--------|------|------|------|------|
| 1 | 9525 | TCP | 192.168.x.0/24 | SSH（局域网） |
| 2 | 9525 | TCP | 100.64.0.0/10 | SSH（Tailscale） |
| 3 | 5006 | TCP | 100.64.0.0/10 | WebDAV（Tailscale） |
| 4 | 5001 | TCP | 192.168.x.0/24 | DSM Web（局域网） |
| 5 | 443 | TCP | 192.168.x.0/24 | Web Station（局域网） |
| 6 | 445 | TCP | 192.168.x.0/24 | SMB（局域网） |
| 7 | 139 | TCP | 192.168.x.0/24 | SMB NetBIOS（局域网） |
| 8 | 137-138 | UDP | 192.168.x.0/24 | SMB 名称解析（局域网） |
| 9 | 5353 | UDP | 192.168.x.0/24 | mDNS/Bonjour 发现（局域网） |
| — | All | — | Any | **Deny（兜底拒绝）** |

原则：**最小化暴露面**。WebDAV 只对 Tailscale 网段开放（外部通过 Tailscale 隧道访问，不需要开放公网端口），DSM 管理界面只对局域网开放。

> **踩坑提醒**：如果 iptables 白名单遗漏了 SMB 端口（445/139/137-138），macOS Finder 将无法通过 SMB 发现和连接 NAS。另外 mDNS（5353/UDP）也需要放行，否则 Finder 侧边栏的自动发现会失效（但 `前往 → 连接服务器 → smb://<ip>` 仍可直连）。

### 3.3 为什么不开端口给 Tailscale 用？

Tailscale 的 WireGuard 协议自带 NAT 穿透和 DERP 回退，**不需要任何入站端口转发**。上一篇的 `tailscale netcheck` 已经证实了这一点——即使没有 UPnP 映射，它也能通过 DERP 或 peer relay 建立连接。

---

## 四、DSM Web 加固

### 4.1 强制 HTTPS

DSM 在 5000 (HTTP) 和 5001 (HTTPS) 双端口监听，登录页面默认走 HTTP → 登录凭证明文传输风险。

**修复**：DSM → 控制面板 → 网络 → DSM 设置 → **勾选"自动将 HTTP 连接重定向到 HTTPS"**。

```bash
# 修复后验证
$ curl -I http://192.168.x.80:5000
HTTP/1.1 302 Found
Location: https://192.168.x.80:5001/
```

### 4.2 启用自动封锁

DSM → 控制面板 → 安全性 → 保护 → 自动封锁：

- 登录失败次数：**3 次**
- 封锁时间：**30 分钟**
- 时间窗口：**5 分钟**

这相当于内建的 fail2ban，对 SSH 和 DSM Web 登录都有效。

### 4.3 安全 HTTP 头

DSM → 控制面板 → 安全性 → 高级：

- 勾选 **"启用 HTTP 内容安全策略 (CSP)"**
- 勾选 **"启用 X-Content-Type-Options"**
- 勾选 **"启用 HTTP Strict Transport Security (HSTS)"**

---

## 五、文件服务加固

### 5.1 SMB 协议底线

```bash
$ grep "protocol" /etc/samba/smb.conf
min protocol=SMB2
max protocol=SMB3
```

> SMB1 已禁用（正确）。SMB1 是 WannaCry 的传播载体，绝不应启用。

额外建议在 DSM → 文件服务 → SMB → 高级设置中：

- **禁用 SMB 的"传输日志"**（减少审计噪���）
- **启用"防止 SMB 远程暴力破解"**
- **启用 SMB 签名**（防止中间人篡改）

### 5.2 服务管理：DSM Web + CLI 双通路

Synology DSM 7.x 底层用 systemd 管理服务（通过 `systemctl` 或 Synology 封装的 `synosystemctl`）。有些服务在 DSM Web 里没有开关，必须用 CLI 关闭。

先确认当前运行的服务及对应的 systemd unit：

```bash
$ systemctl list-units --type=service --state=running | grep -E 'rsync|iscsi|ftp|snmp|telnet'
rsyncd.service                     loaded active running rsync daemon
iscsid.service                     loaded active running iSCSI initiator
snmpd.service                      loaded active running SNMP Daemon
synosnmpcd.service                 loaded active running Daemon for Resource Monitor
```

**关闭方式对照表：**

| 端口 | 服务 | DSM Web 路径 | CLI 命令（需 root） |
|------|------|-------------|-------------------|
| 23 | Telnet | 终端机和 SNMP → 取消 Telnet | — |
| 22 | SSH 默认端口 | 终端机和 SNMP → 端口改 9525 | — |
| 873 | rsync daemon | 文件服务 → rsync → 取消 | `systemctl stop rsyncd && systemctl disable rsyncd` |
| 3261-3265 | iSCSI | SAN Manager → 删除 Target | `systemctl stop iscsid && systemctl disable iscsid` |
| 161 | SNMP | 终端机和 SNMP → 取消 SNMP | `systemctl stop snmpd synosnmpcd` |
| 21/222 | FTP/SFTP | 文件服务 → FTP → 取消 | — |
| 139/445 | SMB | 文件服务 → SMB → 取消 | — |
| 5357 | WS-Discovery | 文件服务 → SMB → 高级 | — |

**CLI 一键关闭脚本**（在 `sudo -i` 终端中执行）：

```bash
# rsync daemon — rsync over SSH (走 9525) 不受影响
systemctl stop rsyncd.service
systemctl disable rsyncd.service

# iSCSI — 不用 SAN 存储就关掉
systemctl stop iscsid.service 2>/dev/null
systemctl disable iscsid.service 2>/dev/null

# SNMP — 不用监控就关
systemctl stop snmpd.service synosnmpcd.service 2>/dev/null

# 验证
echo "关闭后监听端口数:" && netstat -tlnp 2>/dev/null | grep "0.0.0.0" | wc -l
netstat -tlnp 2>/dev/null | grep -E "873|3261|161" || echo "rsync/iSCSI/SNMP 已清除"
```

> 注意：`systemctl stop` 立即生效；`systemctl disable` 防止重启后重新启动。

---

## 六、审计后对比

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| 暴露端口数（0.0.0.0） | **27** | **8**（仅必需） |
| Telnet | 开启 | **关闭** |
| SSH | 双端口，密码登录允许 | **单端口 9525，仅密钥** |
| admin 账户 | 存在且活跃 | **已禁用** |
| DSM Web | HTTP 明文 | **强制 HTTPS** |
| 防火墙 | 全通（policy ACCEPT） | **白名单 + 默认拒绝** |
| 自动封锁 | 未确认 | **启用：3 次/5 分钟 → 锁 30 分钟** |
| SMB 最低协议 | SMB2（已 OK） | SMB2 + 签名 + 防暴力破解 |

---

## 七、自动化一键加固脚本

本博文所有加固操作（除 DSM 控制面板中的 UI 设置外）可合并为一个脚本远程执行。前提是先解决 Synology 的 `sudo requiretty` 限制。

### 7.1 启用远程 sudo 自动化

DSM 默认 `sudo requiretty`，阻止通过 SSH 管道执行 sudo。解法是添加 NOPASSWD 规则到 `/etc/sudoers.d/`：

```bash
# SSH 进 NAS 后，sudo -i 执行一次
echo "<user> ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/<user>
```

原理：`/etc/sudoers.d/` 是 `sudoers` 的 drop-in 目录，系统更新不会覆盖。之后可随时撤销：

```bash
sudo rm /etc/sudoers.d/<user>
```

### 7.2 一键加固脚本

保存为 `nas_harden.sh`（已归档在 [`assets/scripts/nas/nas_harden.sh`]({{ '/assets/scripts/nas/nas_harden.sh' | relative_url }})，同目录还放了 [`nas_tcp_tune.sh`]({{ '/assets/scripts/nas/nas_tcp_tune.sh' | relative_url }}) 与 [`nas_webdav_tune.sh`]({{ '/assets/scripts/nas/nas_webdav_tune.sh' | relative_url }})），从 Mac 端执行：

```bash
cat nas_harden.sh | ssh opennas "sudo -i sh"
```

脚本内容（基于本文所有加固项 + TCP 调优 + WebDAV MPM）：

```bash
#!/bin/ash
# === Synology NAS 一键加固 ===

# 1. SSH 加固
sed -i 's/^#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# 2. 关闭不必要服务
systemctl stop rsyncd.service && systemctl disable rsyncd.service
systemctl stop snmpd.service synosnmpcd.service 2>/dev/null
systemctl stop iscsid.service 2>/dev/null

# 3. TCP 内核调优
sysctl -w net.core.rmem_max=2097152
sysctl -w net.core.wmem_max=2097152
sysctl -w net.ipv4.tcp_rmem="4096 131072 2097152"
sysctl -w net.ipv4.tcp_wmem="4096 16384 2097152"
sysctl -w net.ipv4.tcp_slow_start_after_idle=0
sysctl -w net.ipv4.tcp_fastopen=3

# 持久化
{
  echo "net.core.rmem_max=2097152"
  echo "net.core.wmem_max=2097152"
  echo "net.ipv4.tcp_rmem=4096 131072 2097152"
  echo "net.ipv4.tcp_wmem=4096 16384 2097152"
  echo "net.ipv4.tcp_slow_start_after_idle=0"
  echo "net.ipv4.tcp_fastopen=3"
  echo "net.core.default_qdisc=pfifo_fast"
} >> /etc/sysctl.conf

# 4. iptables 防火墙
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -i lo -j ACCEPT
iptables -A INPUT -p tcp --dport 9525 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 9525 -s 100.64.0.0/10 -j ACCEPT
iptables -A INPUT -p tcp --dport 5001 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 5006 -s 100.64.0.0/10 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 445 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p tcp --dport 139 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p udp --dport 137:138 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p udp --dport 5353 -s 192.168.0.0/16 -j ACCEPT
iptables -A INPUT -p icmp -j ACCEPT
iptables -A INPUT -j DROP

# 5. WebDAV MPM
MPM_CONF="/var/packages/WebDAVServer/target/etc/httpd/conf/extra/httpd-mpm.conf-webdav"
cat > "$MPM_CONF" << 'EOF'
<IfModule mpm_worker_module>
    StartServers          3
    MinSpareThreads       25
    MaxSpareThreads       75
    ThreadsPerChild       25
    ServerLimit           6
    MaxRequestWorkers     150
</IfModule>
EOF
/usr/syno/bin/synopkg restart WebDAVServer

# 6. 锁定 admin shell
sed -i 's|^admin:.*|admin:x:1024:100:System default user:/var/services/homes/admin:/sbin/nologin|' /etc/passwd
```

> **注意**：Synology 内核 5.10.55+ 未编译 `sch_fq_codel` 模块，`default_qdisc` 只能设为 `pfifo_fast`。fq_codel 的 bufferbloat 保护不可用，但 TCP 缓冲区调优（rmem_max=2MB）仍生效。

### 7.3 关于 rsync

本文关闭的是 **rsync daemon（端口 873）**，不影响 `rsync over SSH`。你的 SSH config 已配置 ControlMaster 复用，rsync 自动走 SSH 隧道：

```bash
# rsync over SSH — 正常工作，无需 rsyncd
rsync -avz --progress /本地路径/ opennas:/远程路径/
```

相比 rsyncd 需要单独配置密码文件和模块，rsync over SSH 更简单、更安全。

## 八、快速自检脚本

把这串命令扔进 NAS 终端，3 秒出审计报告：

```bash
#!/bin/ash
echo "=== 1. Telnet ==="
netstat -tlnp 2>/dev/null | grep ":23 " && echo "[FAIL] Telnet is ON" || echo "[PASS]"

echo "=== 2. SSH Ports ==="
SSH_PORTS=$(netstat -tlnp 2>/dev/null | grep "sshd" | grep -c LISTEN)
[ "$SSH_PORTS" -le 2 ] && echo "[PASS] SSH single port" \|\| echo "[FAIL] SSH on $SSH_PORTS ports"

echo "=== 3. SSH Config ==="
grep -q "^PermitRootLogin no" /etc/ssh/sshd_config && echo "[PASS] Root login disabled" || echo "[FAIL]"
grep -q "^PasswordAuthentication no" /etc/ssh/sshd_config && echo "[PASS] Password auth disabled" || echo "[FAIL]"

echo "=== 4. Firewall ==="
iptables -L INPUT -n 2>/dev/null | grep -q "DROP\|REJECT" && echo "[PASS] Firewall has rules" || echo "[FAIL] Firewall empty"

echo "=== 5. admin account ==="
grep "^admin:" /etc/passwd 2>/dev/null | grep -q nologin && echo "[PASS] admin locked" || echo "[WARN] admin shell active"

echo "=== 6. HTTP Redirect ==="
curl -skI http://localhost:5000 2>/dev/null | grep -q "302\|301" && echo "[PASS] HTTP→HTTPS" || echo "[FAIL] HTTP still accessible"

echo "=== 7. Public ports ==="
PUBLIC=$(netstat -tlnp 2>/dev/null | grep "0.0.0.0" | wc -l)
echo "[INFO] $PUBLIC ports listening on 0.0.0.0"

echo "=== 8. TCP tuning ==="
echo "rmem_max: $(sysctl -n net.core.rmem_max)"
echo "tcp_fastopen: $(sysctl -n net.ipv4.tcp_fastopen)"
```

---

## 总结

安全加固不是一次性工程。建议——

**每次重启后检查**：Storage Manager 可能重新锁写 volume，auto-block 规则可能重置，Telnet 可能被某次系统更新重新开启。

**定期执行自检脚本**：上面 8 行命令，加到 cron 月检即可。

**自动化维护**：`/etc/sudoers.d/<user>` 可让你通过 `ssh opennas "sudo ..."` 远程执行任意命令，配合自检脚本实现无交互巡检。

**与上一篇联动**：Tailscale 的 WireGuard 隧道本身就是加密层，配合 WebDAV over HTTPS（TLS 1.3），外部攻击面收缩到只剩下 Tailscale 的 DERP 协商，几乎没有可利用的攻击入口。
