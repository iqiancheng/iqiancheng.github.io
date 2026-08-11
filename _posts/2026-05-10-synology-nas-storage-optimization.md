---
layout: post
author: Joseph
title: '给 NAS 做一次"体检"：Btrfs 压缩、scrub 排程与 SSD 缓存审计'
date: 2026-05-10 00:00:00 +0800
categories: [homelab]
tags: [nas]
description: >
  在完成网络层 TCP 调优和 WebDAV MPM 优化后，对 Synology SA6400 的存储层进行审计。
  覆盖 Btrfs zstd 压缩、每月数据清理排程、SSD 缓存状态分析与 LINEAR JBOD 风险评估。
toc: true
---
## 背景

[前一篇网络优化](/posts/tailscale-synology-nas-real-world/) 已经把局域网 WebDAV 速度从 4.2 MB/s 推到 9.8 MB/s，[安全加固](/posts/synology-nas-security-hardening/) 把端口从 27 个砍到 16 个。这一篇往下走一层：存储。

NAS 的最终瓶颈永远在磁盘。网络再快、协议再好，落到存储层该有的问题一个不少——静默数据损坏、空间浪费、缓存配置错误。

**测试环境**：Synology SA6400, DSM 7.x, 2×14.6TB HDD + 1×1.8TB NVMe SSD, Btrfs。

---

## 一、存储布局审计

### 1.1 当前拓扑

```
/dev/nvme0n1  1.8TB  →  md2 (RAID1, 单盘降级)  →  vg1-volume_1  →  cachedev_1  →  /volume1  (系统/应用)
/dev/sata1    14.6TB ─┐
                       ├─  md3 (LINEAR, 29TB)    →  vg2-volume_2  →  cachedev_0  →  /volume2  (数据)
/dev/sata2    14.6TB ─┘
```

关键发现：

| 设备 | RAID 类型 | 冗余 | 用途 |
|------|----------|------|------|
| md2 (NVMe) | RAID1 | 无（单盘） | 系统/套件，1.8TB，仅用 460MB |
| md3 (HDD×2) | **LINEAR** | **无** | 数据，29TB，已用 5.9TB |

> 关键证据：`md3 : active linear sata1p3[0] sata2p3[1]`。LINEAR 模式把两块 14.6TB 串成一个 29TB 卷，**零冗余**。任意一块盘故障 = 全部数据丢失。

### 1.2 数据冗余决策

这个 29TB 存储的是公开影视资源，有外部来源可重新获取。因此**不做 RAID 迁移、不做快照**。

如果你的数据不可替代，这篇博文的 RAID 部分到此结束——请在它有冗余之前先做备份。剩下的优化针对**已接受 LINEAR 风险**的场景。

---

## 二、Btrfs 压缩：zstd 启用实测

### 2.1 压缩的价值判断

Btrfs 支持 per-directory 透明压缩。关键问题是**哪些目录值得开**：

| 目录 | 内容 | 压缩收益 |
|------|------|---------|
| `/volume2/homes` | 文档、代码、配置、数据库 | **高（预计 20-40%）** |
| `/volume2/docker` | 容器数据、layer、日志 | **高** |
| `/volume2/downloads` | 影视媒体文件 | **无（已压缩）** |
| `/volume2/NetBackup` | 备份归档 | **无（已压缩或去重）** |

### 2.2 启用命令

```bash
# 对高收益目录开启 zstd
sudo btrfs property set /volume2/homes compression zstd
sudo btrfs property set /volume2/docker compression zstd
```

> **注意**：`btrfs property set` 只影响**之后写入的文件**。已有文件保持原样。如果需要压缩存量数据，需运行 `btrfs filesystem defrag -czstd`（耗时，5.9TB 数据可能需要数十小时）。在决定运行 defrag 之前，确认有足够的空间余量。

### 2.3 验证

```bash
$ btrfs property get /volume2/homes compression
compression=zstd
```

### 2.4 为什么 downloads 设置失败？

```bash
$ sudo btrfs property set /volume2/downloads compression zstd
ERROR: Invalid argument
$ sudo btrfs property set /volume2/downloads/压缩测试 compression zstd
ERROR: Invalid argument
# 子卷内部新建的子目录同样失败
```

`/volume2/downloads` 是 DSM 创建的共享文件夹子卷。进一步的排查发现：**Synology DSM 7.x 阻止了对用户创建的共享文件夹子卷设置 Btrfs 压缩属性**。即使在该子卷内部新建目录再设置，同样返回 `Invalid argument`。

对比之前成功的 `homes` 和 `docker`：
- `/volume2/homes`：系统生成子卷，未被 DSM 锁定
- `/volume2/docker`：Docker 套件创建的子卷，不受此限制
- `/volume2/downloads`：**通过 DSM 控制面板创建的共享文件夹** → 被锁定

对这类 DSM 管理的共享文件夹，目前 DSM Web UI 不暴露压缩选项，CLI 的 `btrfs property set` 也被拦截。**只能接受现状**，等 Synology 后续版本开放此功能。

---

## 三、数据清理（Scrub）：从零到月度排程

### 3.1 当前状态

```bash
$ sudo btrfs scrub status /volume2
no stats available
total bytes scrubbed: 0.00B with 0 errors
```

> 关键证据：这台 NAS 从部署至今，**从未运行过 Btrfs scrub**。5.9TB 数据的完整性从未被校验过。

Btrfs 的 checksum 机制可以在读取时检测静默数据损坏（bit rot），但前提是**数据被读取过**。Scrub 就是强制全量读取 + 校验，把冷数据的潜在损坏提前暴露出来。

### 3.2 排程设计

DSM 7.x 的 Storage Manager 有内置的数据清理排程（Storage Manager → 存储池 → 排程数据清理），但也可以通过 CLI 创建。

在 DSM 排程系统（`/usr/syno/etc/synoschedule.d/`）中创建月度任务：

```
Task ID: 6
类型: monthly
时间: 每月 1 日 03:00
命令: /usr/syno/sbin/btrfs scrub start /volume2
状态: enabled
```

对应的 `/etc/crontab` 条目：

```
0	3	1	*	*	root	/usr/syno/bin/synoschedtask --run id=6
```

### 3.3 设计考量

- **凌晨 3:00**：避开正常使用时段
- **每月一次**：对 5.9TB HDD 足够。过于频繁的 scrub 会拖慢正常 I/O。如果是 SSD 可以更频繁
- **不和 dedup 冲突**：已有关键的 Btrfs 去重任务（Task 3）每天 00:00 运行，scrub 错开时段

> 踩坑：DSM 的 crontab 必须用 **tab 分隔字段**，空格会导致 cron 拒绝解析。检查格式用 `cat -A /etc/crontab`，看到 `^I` 才是 tab。

### 3.4 手动触发测试

```bash
$ sudo /usr/syno/bin/synoschedtask --run id=6
$ sudo btrfs scrub status /volume2
Scrub started:    Sun May 10 07:40:00 2026
Status:           running
Duration:         0:05:00
Time left:        8:30:00
ETA:              Sun May 10 16:10:00 2026
```

29TB 的完整 scrub 预计需要 8-10 小时。期间不影响正常读写（Btrfs scrub 是后台 I/O，优先级低于用户 I/O）。

---

## 四、SSD 缓存：DUMMY 不是 Bug

### 4.1 现象

```bash
$ sudo dmsetup table | grep cache
cachedev_0: ... cache mode(DUMMY)
	total blocks(0), cached blocks(0), cache percent(0)
cachedev_1: ... cache mode(DUMMY)
	total blocks(0), cached blocks(0), cache percent(0)
```

> `cache mode(DUMMY)` 不是错误。它表示 Synology 的 flashcache 框架已初始化，但**没有 SSD 被分配为缓存设备**。

### 4.2 为什么

当前存储拓扑：

```
NVMe (1.8TB)  →  volume1 (系统/套件)   ← 独立卷，不是缓存
HDD×2 (29TB)  →  volume2 (数据)
```

NVMe 被用作 volume1（DSM 系统 + 套件安装位置），而不是 volume2 的缓存。这是 DSM 的默认行为——第一块 SSD 通常被建议作为系统卷。

### 4.3 要不要改？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **保持现状** | 系统/套件在 NVMe 上，DSM Web 响应快 | volume2 无缓存加速 |
| **NVMe 做读写缓存** | 随机读写性能大幅提升 | 需先迁移 volume1（460MB）到 HDD，重建缓存，增加复杂度 |
| **加第二块 NVMe** | 系统和缓存各一块，皆大欢喜 | 需要硬件投入 |

结论：460MB 的系统卷不值得单独占一块 1.8TB NVMe。但目前 volume2 的内容是顺序大文件（影视），SSD 缓存对顺序读写的收益有限。**保持现状**，等有随机 I/O 密集型工作负载时再考虑迁移。

---

## 五、其他发现

### 5.1 Btrfs 去重已在运行

```bash
$ sudo cat /usr/syno/etc/synoschedule.d/root/3.task | grep name
name=Task 3
app=btrfsdedupe
state=enabled
```

DSM 内置的 Btrfs 去重任务（`synobtrfsdedupe`）每天 00:00 运行。这与我们新增的 zstd 压缩是互补的——去重消除冗余块，压缩进一步缩小每个块的体积。

### 5.2 SSD TRIM 正常

```bash
$ sudo cat /usr/syno/etc/synoschedule.d/root/5.task
name=Task 5
app=SYNO.SDS.StorageManager.Volume.Dialog.TrimSupport
state=enabled
```

每天 00:00 对 volume1 (NVMe) 执行 TRIM，正常。

### 5.3 快照数量：0

```bash
$ ls /volume2/@snapshot/ | wc -l
0
```

之前安全审计脚本把 Btrfs 子卷（Docker 容器层 + Synology Drive 的子卷）误计为快照。实际快照数为零。对可重新获取的公开影视数据，不做快照是合理的。

### 5.4 DSM 自动更新：已禁用

```bash
$ sudo cat /usr/syno/etc/synoschedule.d/root/1.task | grep state
state=disabled
```

Task 1（DSM Auto Update）已禁用。目的是避免自动更新引入兼容性问题。建议每季度手动检查一次更新。

---

## 六、优化前后对比

| 维度 | 优化前 | 优化后 |
|------|--------|--------|
| Btrfs 压缩 | 未开启 | **zstd on homes + docker** |
| 数据校验 | 从未 scrub | **每月 1 日凌晨自动** |
| SSD 缓存 | DUMMY（误以为异常） | **确认正常（NVMe 是系统卷）** |
| 冗余风险 | LINEAR JBOD（未知） | **已评估、已接受** |
| 快照 | 0（错误报告为 168） | **确认 0，合理** |
| 自动更新 | 已启用 | **已禁用（手动管理）** |
| 数据去重 | 每日（隐式生效） | **已验证正常运行** |

---

## 总结

存储优化和你前两篇的网络优化是互补的：

- **网络层**（TCP 2MB + WebDAV MPM 25） → 局域网吞吐从 4.2 → 9.8 MB/s
- **存储层**（zstd 压缩 + 月度 scrub） → 空间利用率 + 数据完整性保障
- **安全层**（iptables + 端口收敛） → 攻击面从 27 端口 → 16 端口，全部白名单

三个层次的优化覆盖了「速度、空间、安全」三个维度。

**定期维护清单**：

- [ ] **每月**：Scrub 完成后检查 `btrfs scrub status` 是否有 error
- [ ] **每季度**：手动检查 DSM 更新，确认变更日志无 Breaking Change 后升级
- [ ] **每半年**：检查 Btrfs 空间使用 `btrfs filesystem df /volume2`，关注 metadata 用量
- [ ] **每年**：SMART 全盘检测 + Btrfs defrag（如果 zstd 压缩存量文件）

> 与上两篇联动：网络优化的 TCP 调优（rmem_max=2MB）对高延迟 DERP 链路有效，存储层的 zstd 压缩可以和 `rsync over SSH`（第一篇的 rsyncd 替代方案）配合使用——rsync 的 `-z` 是传输时压缩，zstd 是落盘后压缩，两不冲突。
