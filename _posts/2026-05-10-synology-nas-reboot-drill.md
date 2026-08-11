---
layout: post
author: Joseph
title: "一次重启引发的血案：SATA 瞬断如何拖垮 RAID / scemd / synopkg"
date: 2026-05-10 00:00:00 +0800
categories: [homelab]
tags: [docker, proxy, nas, debugging]
description: >
  NAS 重启后所有依赖 volume2 的包全部挂掉——不是 GFW 的问题，是一块 SATA 盘的瞬时读错误
  触发了从 RAID → scemd → synopkg 的三级雪崩。完整复盘故障链路、恢复步骤、以及挖出来的 Synology 内部机制。
toc: true
---
## 背景

[上一篇 GFW 审计博文](/posts/synology-nas-gfw-audit-mihomo/) 收尾时做了 NAS 重启演练，验证 mihomo TUN、Cloud Sync、Container Manager 等组件在重启后能否自动恢复。

结果没通过。重启后 Package Center 里三个包全挂：
- **Container Manager** error 272（"not startable"），所有 Docker 容器不可用
- **CloudSync** error 272，Google Drive 同步中断
- **DownloadStation** error 273，连带 pgsql-adapter 失败

不是 GFW 的问题。是一块磁盘的瞬时错误触发了三级雪崩。

测试环境：Synology SA6400, DSM 7.2, 2×WD 16TB SATA linear RAID, 1×NVMe SSD 缓存。

---

## 一、故障时间线

```
13:49:26  系统启动，RAID 阵列开始组装
13:49:29  pkg-volume.target 就绪，各包开始启动
13:49:36  scemd 启动
13:50:15  sata1 发生瞬时读错误 → md 驱动将 sata1p3 标记 faulty
13:50:21  scemd 检测到 RAID crash → 将 /volume2 设为 read-only
13:50:25  scemd 开始 unmount 流程，通知所有依赖 /volume2 的服务停止
13:50:29  CloudSync 被停止，mihomo 被停止
13:50:38  ContainerManager 被停止；enabled flag 被清除
          系统尝试重启包，但 /volume2/@docker 仍是 read-only
          dockerd 尝试 chmod /volume2/@docker → "read-only file system"
13:50:38-50  dockerd 连续 crash 6 次
13:50:50  systemd: "start request repeated too quickly" → 放弃
13:50:50  synopkg 将 ContainerManager 标记为 disabled
14:01:39  用户从 Package Center 尝试启动 → feasibility check 失败
          "Volume is not working. Can't start [ContainerManager]"
```

---

## 二、根因分析：瞬时 I/O 错误的三级雪崩

### 2.1 第一级：SATA 层

内核日志中最先出现的是两条 SATA I/O 错误：

```
sd 0:0:0:0: [sata1] tag#23 UNKNOWN(0x2003)
Sense Key : 0x5 [current]     ← ILLEGAL REQUEST
ASC=0x21 ASCQ=0x4             ← Logical block address out of range

blk_update_request: I/O error, dev sata1, sector 5885322120

md_error: sata1p3 is being to be set faulty
md/raid:md3: Disk failure on sata1p3, disabling device.
read error, md3, sata1p3 index [0], sector 5864080264
```

`ASC=0x21 ASCQ=0x4` 是 SCSI 标准的 "Logical Block Address Out of Range" 错误。不是物理坏道，不是磁头故障——是固件/控制器层面的瞬时异常，类似"控制器暂时返回了一个它认为不存在的地址"。两次错误分别发生在不同扇区（5885322120 和 5895686312），间隔 9 秒，之后没有再出现。

SMART 数据也佐证了这一点：0 个重分配扇区、0 个待处理扇区、0 个不可纠正错误。磁盘是健康的。

### 2.2 第二级：Synology md 驱动层

标准 Linux md 驱动在遇到读错误时会尝试重试并从冗余磁盘恢复。但 Synology 的修改版 md 驱动多了一个行为：

```
md_error: sata1p3 is being to be set faulty
```

这个 `(E)` 标记是 **Synology 私有的扩展**，写在 `/proc/mdstat` 中：

```
# 故障状态
md3 : active linear sata1p3[0](E) sata2p3[1]
      31230310400 blocks super 1.2 64k rounding [2/2] [EU]

# 正常状态
md3 : active linear sata1p3[0] sata2p3[1]
      31230310400 blocks super 1.2 64k rounding [2/2] [UU]
```

关键发现：**`(E)` 标记是持久的**。即使内核 sysfs 中 `dev-sata1p3/state=in_sync` 且 `errors=0`，`(E)` 也不会自动清除。标准 `mdadm --remove/--re-add` 对 linear RAID 不可用（linear 无法降级运行，移除任何盘都会导致阵列不完整）。

这个标记在 NAS 正常运行时可以通过正确的 synopkg 启动流程来清除（见第五节），但不通过标准 Linux RAID 工具暴露。

### 2.3 第三级：scemd → synopkg 链

Synology 的服务事件管理守护进程 `scemd` 负责监控硬件状态和卷生命周期。启动时它检测到 `/proc/mdstat` 中的 `(E)` 标记：

```
scemd 日志：
Space abnormal status reason: RAID crash
space_error_log.c:47 command="/bin/mount -o remount,ro /volume2"
```

scemd 做了三件事：
1. **强制 remount ro** — 将 `/volume2` 从 rw 改为 ro
2. **写入状态文件** — `/run/space/volume_state` 写入 `access_type=ro`，`/run/space/volume_status.cache` 写入 `"status":1`（异常）
3. **触发 unmount-volume-end.target** — 通知所有依赖 `/volume2` 的 systemd 服务停止

之后 synopkg 的 feasibility check（`feasibility_check.cpp:129`）读取 scemd 的状态缓存，发现 volume status ≠ 0，拒绝启动任何在该卷上有数据依赖的包：

```
synopkg.log:
Volume is not working. Can't start [ContainerManager], version 24.0.2-1535
Volume is not working. Can't start [CloudSync], version 2.7.2-2714
```

---

## 三、Synology 包管理状态机

这次故障暴露出了几个不为人知的 Synology 内部机制。

### 3.1 包状态码

| Code | 含义 | 日志描述 |
|------|------|---------|
| 0 | 正常启动 | `retrieve from status script` |
| 1 | 单元活跃但状态脚本返回非0 | `the script status is not 0 but the unit is active` |
| 150 | FHS target 读写状态异常 | `failed to get fhs target read write state` |
| 260 | 依赖服务不可用 | `Failed to contact dependee system services` |
| 262 | 包未启用 | `not turned on` |
| 263 | systemd 状态获取失败 | `retrieve from systemd failed status` |
| 272 | 上次启动失败 | `failed to start on previous startup` |
| 273 | systemd 状态无法转换 | `translate from systemd status` |

### 3.2 pkgctl 与 synopkg 的两层架构

Package Center 的启动流程分两层：

```
synopkg start <package>
  │
  ├─ feasibility check (feasibility_check.cpp)
  │   ├─ hard check [0]: volume 状态 (scemd 缓存)
  │   └─ soft check: 包自定义检查
  │
  └─ systemctl start pkgctl-<package>.service
      │
      └─ ExecStart: synopkgctl start <package>
          │
          └─ 实际启动 systemd 服务单元
```

- **`synopkg`**（上层 CLI）：做可行性检查，通过后才调 systemd
- **`synopkgctl`**（下层 CLI）：直接启动，**不做可行性检查**——这就是恢复时可以绕过 error 272 的原因

### 3.3 enabled flag 的生命周期

`/var/packages/<name>/enabled` 文件标记包的启用状态。注意：

- 文件内容为空（零字节）是正常的——文件的存在本身就是标志
- **pkgctl 的 ExecStart 会在启动成功后 touch 这个文件**
- **ExecStop 会在停止时 rm 这个文件**
- 如果包异常退出导致 ExecStop 被执行，文件就被删除——下次 synopkg 就认为包是 disabled

这就是为什么 docker 连续 crash 6 次后，enabled flag 消失了——系统尝试启动包（调用 ExecStart），实际启动 dockerd 后 service 又 crash，导致 systemd 调用了 ExecStop。

### 3.4 FHS target 检查

"FHS" = File Hierarchy Standard。`/var/packages/<name>/target` 是一个符号链接，指向 `/volume2/@appstore/<name>`。feasibility check 会验证这个链路的读写状态。

当 `/volume2` 被 scemd 设为 ro 时，`target → /volume2/@appstore/...` 就是只读的，所以 feasibility check 返回 `broken_by: fhs, status_code: 150`。

---

## 四、恢复流程

### 4.1 失败路径（手动无章法操作）

在理解根因之前走了一些弯路：

- 手动 `mount -o remount,rw /volume2` → 有效，但 scemd 检测到 RAID 异常后又立刻设回 ro
- 手动 `systemctl start dockerd` → dockerd 起来了，但 synopkg 状态没更新，Package Center 仍显示 error
- 创建 `enabled` flag 文件 → 无效，feasibility check 在更早的阶段就拦截了
- `synopkg start` → 反复报 "Volume is not working"

### 4.2 正确路径

完整恢复序列：

```bash
# 1. 停止 scemd（阻止它反复设 ro）
systemctl stop scemd

# 2. 重新挂载为读写
mount -o remount,rw /volume2
mount -o remount,rw /volume2/@docker

# 3. 修复 scemd 缓存的状态文件
# 将 volume_status.cache 中 volume2 的 status 从 1 改为 0
# （或直接删掉 /run/space/volume_status.cache，scemd 后续重建）

# 4. 通知 scemd 这是一次有意的手动操作
spacetool --notify-ro-volume-mount-rw /volume2

# 5. 通过 synopkg 框架启动包（而非手动 systemctl）
synopkgctl start ContainerManager
synopkgctl start CloudSync

# 6. 修复依赖链
systemctl reset-failed pgsql-adapter.service
systemctl start pgsql-adapter.service

# 7. 恢复 scemd（此时 RAID (E) 标记已被步骤 5 清除）
systemctl start scemd
```

### 4.3 (E) 标记的清除时机

最有意思的发现：**RAID 的 `(E)` 标记不是手动清除的，而是在 `synopkgctl start ContainerManager` 的过程中自动消失的**。

当 synopkgctl 执行 ContainerManager 的完整启动流程（包括包启动脚本、资源检查、apparmor 配置等）时，会触发 Synology 的存储子系统重新评估 RAID 状态。因为磁盘实际健康（SMART clean, errors=0, state=in_sync），标记就被清除了。

这个行为暗示 synopkgctl 的启动脚本中包含了类似 `spacetool --assemble-all` 或 RAID 健康重新检查的步骤。

### 4.4 pgsql-adapter 的独立恢复

Download Station 的依赖链中有一个不显眼但关键的服务：`pgsql-adapter.service`（PostgreSQL adapter）。它是一个 oneshot 服务：

```ini
# /usr/lib/systemd/system/pgsql-adapter.service
Requisite=syno-share.target syno-volume.target
ExecStart=/usr/lib/systemd/scripts/pgsql.sh start
```

因为它在 Requisite 中依赖了 `syno-volume.target`，当 volume2 异常时它也挂了。修复方法就是普通的 reset-failed + start。

---

## 五、关键文件与接口速查

| 路径 | 用途 | 备注 |
|------|------|------|
| `/proc/mdstat` | RAID 状态，`(E)` 标记所在 | Synology 扩展了标准 mdstat |
| `/sys/block/mdN/md/dev-*/state` | 内核层设备状态 | 与 mdstat 的 `(E)` 可能不同步 |
| `/sys/block/mdN/md/dev-*/errors` | 内核层错误计数 | 即使 `(E)` 在，errors 也可能为 0 |
| `/run/space/volume_state` | scemd 写入的卷访问模式 | 内容: `access_type=rw/ro` |
| `/run/space/volume_status.cache` | scemd 缓存的卷状态 JSON | `"status":0` 正常，`"status":1` 异常 |
| `/run/space/space_table` | LVM/RAID 设备状态（JSON） | 记录了 `"failed":true` |
| `/var/packages/<pkg>/enabled` | 包启用标志（空文件） | 存在=启用，被 rm=禁用 |
| `/var/packages/<pkg>/target` | → `/volumeN/@appstore/<pkg>` | FHS 可行性检查的目标 |
| `/var/packages/<pkg>/conf/resource` | 包资源声明（JSON） | 定义 feasibility check 插件 |
| `/var/packages/<pkg>/conf/resource.own` | 运行时的资源状态 | 由 synopkg 生成 |
| `/var/cache/synopkg/` | synopkg 缓存目录 | installed, badge_count_records 等 |
| `/var/log/synopkg.log` | synopkg 的主日志 | 记录了所有 start/stop 和 feasibility check |
| `/var/log/synofeasibilitycheck.log` | feasibility check 专有日志 | 记录了 `check [0] failed` 详情 |

命令行工具：

| 命令 | 用途 |
|------|------|
| `synopkg start/stop/status <pkg>` | 高层包管理（带 feasibility check） |
| `synopkgctl start/stop <pkg>` | 低层包控制（无 feasibility check） |
| `spacetool --notify-ro-volume-mount-rw /volumeN` | 通知 scemd 卷被有意设为 rw |
| `spacetool --assemble-all` | 扫描并组装所有存储空间 |
| `spacetool --bootup-assemble` | 系统启动时的组装流程 |
| `synoshare --get <name>` | 查看共享文件夹详情 |
| `mdadm --detail /dev/mdN` | 查看 RAID 阵列详情 |

---

## 六、预防措施

### 6.1 RAID 层面

这次是 linear RAID（JBOD 拼接）。如果丢了一个盘，数据恢复比 RAID1/5/6 更难。建议：

1. **定期 Data Scrubbing**（DSM Storage Manager → 卷 → 管理 → 文件系统检查）。我们已经配了每月 scrub，务必保持。
2. **关注 SMART 属性**：这次磁盘 SMART 全正常但仍有瞬断，所以光看 SMART 不够，还要监控 `dmesg` 中的 I/O error。
3. 如果第二次出现同样磁盘的瞬断，考虑换盘。

### 6.2 scemd 层面

scemd 的保护机制在大方向上是正确的——发现 RAID 异常时把卷切到只读防数据损坏。但对于瞬断类型的错误（几秒内就自愈），它缺少自动重试机制。

可以将恢复脚本放入 systemd boot hook：

```bash
# /usr/local/bin/nas-boot-recovery.sh
# 在启动完成后检查 volume 状态，如果因为 (E) 标记被设 ro，
# 自动执行恢复序列
```

但不建议全自动——如果磁盘真的坏了，自动把 ro 改 rw 会导致数据持续写入而损坏文件系统。

### 6.3 监控建议

```bash
# 加入每周巡检脚本
check_raid_flag() {
  if grep -q '(E)' /proc/mdstat; then
    echo "WARN: RAID (E) flag detected on md3"
    dmesg | grep -i 'md_error\|fault' | tail -5
  fi
}

check_volume_access() {
  if mount | grep '/volume2.*ro,' > /dev/null; then
    echo "CRIT: /volume2 is read-only"
  fi
}

check_synopkg_health() {
  for pkg in ContainerManager CloudSync DownloadStation; do
    status=$(synopkg status "$pkg" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))")
    [ "$status" != "running" ] && echo "FAIL: $pkg status=$status"
  done
}
```

---

## 七、教训总结

1. **磁盘瞬断比完全故障更难排查**。如果是彻底坏了，SMART 会有明确指示；瞬断的 `ASC=0x21 ASCQ=0x4`（LBA out of range）看起来像固件 bug，不是物理损坏。

2. **Synology 的修改版 md 驱动引入了 `(E)` 标记**，它和内核 sysfs 的 `state/errors` 独立维护。标准 Linux RAID 知识不完整适用于 DSM。

3. **scemd 是 Synology 系统中最核心也最不透明的守护进程**。它控制卷生命周期、硬件监控、包依赖——但没有任何文档。理解它的行为全靠读日志和逆向工程。

4. **`synopkgctl` 和 `synopkg` 的区别是关键**。前者绕过可行性检查，是紧急恢复的入口；后者是日常使用的正确方式。遇到变着花样不让你启动的包，试试直调 `synopkgctl`。

5. **一个磁盘 I/O 错误 → RAID 标记 → scemd 保护 → 卷只读 → 所有包挂掉**。这条链上的每一环都不知道下一环会发生什么——md 驱动不知道 scemd 会怎么处理它的标记，scemd 不知道 synopkg 会怎么理解它的卷状态。系统容错设计的经典反例。

6. **重启演练非常有价值**。所有组件在日常运行中都可以手动踩坑修好，但重启后它们必须靠自动机制恢复。这次暴露的三个问题——`(E)` 标记持久化、scemd 把卷设 ro、依赖包状态连锁——在 uptime 正常时是完全不可见的。

归根结底，这次故障的本质是：**一个自愈的磁盘瞬断，碰上了不自愈的 Synology 状态机**。硬件没问题，但 DSM 的内部分层状态追踪没有容错机制，导致一个 0.01 秒的 I/O 瞬断演变成全量服务不可用。
