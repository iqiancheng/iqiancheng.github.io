---
layout: post
author: Joseph
title: "RAID (E) 标记与 volume2 只读：一次 SATA 物理层雪崩的复盘"
date: 2026-05-10 00:00:00 +0800
categories: [homelab]
tags: [nas, hardware]
description: >
  天钡 WTR R1 (Intel N100) + SA6400 loader 重启后 /volume2 只读、所有包无法启动。
  md3 线性阵列中 sata1p3 被标记 (E)。深度诊断确认是 SATA 物理层 ICRC 错误导致的三级雪崩——从线缆接触到 scemd 再到 synopkg feasibility check。
toc: true
---
## 背景

[上一篇 GFW 审计](/posts/synology-nas-gfw-audit-mihomo/) 结束后重启 NAS 做存活验证，结果 Package Center 全线崩溃：

- ContainerManager error 272 → 所有 Docker 容器不可用
- CloudSync error 272 → Google Drive 同步中断
- DownloadStation error 273 → 连带 pgsql-adapter 失败

排查发现 `/volume2` 被强制 mount 为只读。根源在 `/proc/mdstat`：

```
md3 : active linear sata1p3[0](E) sata2p3[1]
                           ^^^
```

这个是群晖 md 驱动的私有 `(E)` 故障标记。两个盘的 md0/md1 系统分区正常 `[UU]`，说明盘没有完全离线。

**环境**：[天钡 Tbao WTR R1](https://www.tianbaoelectronics.com/) 桶形 3 盘位 NAS 主机（Intel N100, 双 2.5G 网口）+ SA6400 loader, DSM 7.2, 2×WD WUH721816ALE6L4 16TB (HC550) LINEAR JBOD, 1×NVMe SSD 缓存。

---

## 一、故障时间线

```
13:49:26 系统启动，各包开始初始化
13:50:15 sata1 出现 I/O 错误 → md 驱动将 sata1p3 标记 (E)
13:50:21 scemd 检测到 RAID crash → mount -o remount,ro /volume2
13:50:25 scemd 通知所有依赖 /volume2 的服务停止
13:50:29 CloudSync 被停，mihomo 被停
13:50:38 ContainerManager 被停 → enabled flag 被 ExecStop 删除
          dockerd 尝试 chmod /volume2/@docker → "read-only file system"
13:50:38-50 dockerd 连续 crash 6 次
13:50:50 systemd: "start request repeated too quickly" → 放弃
13:50:50 synopkg 将 ContainerManager 标记为 disabled
14:01:39 从 Package Center 尝试启动 → feasibility check 失败
         "Volume is not working. Can't start [ContainerManager]"
```

同样的时间线在 14:45（重建后再次重启）100% 复现。

---

## 二、故障链分析：三级雪崩

### 第一级：SATA 物理层

内核日志中的错误签名：

```
sd 0:0:0:0: [sata1] tag#23 UNKNOWN(0x2003) Result: hostbyte=0x00 driverbyte=0x08
Sense Key : 0x5 [current]          ← ILLEGAL REQUEST
ASC=0x21 ASCQ=0x4                  ← LBA out of range

blk_update_request: I/O error, dev sata1, sector 4056170944
md_error: sata1p3 is being to be set faulty
md/raid:md3: Disk failure on sata1p3, disabling device.
```

初期判断有两种可能方向：

| 假设 | 证据 |
|------|------|
| 固件 bug | 两个盘固件版本不同（PCGNW232 vs PCGAW23G），sata1 默认 SMART 不可用 |
| SATA 物理层 | 错误签名 `ASC=0x21 ASCQ=0x4`，每次不同扇区 |

深入调查后排除了固件假设。

### 第二级：Synology md 驱动

Synology 的修改版 md 驱动在遇到 I/O 错误时会设置 `(E)` 标记——这是**标准 Linux md 没有的行为**。标准 md 线性阵列遇到读错误会重试并从其他盘恢复（如果有冗余），但群晖的驱动会直接标记设备故障。

`(E)` 标记是持久的——即使内核 sysfs 中 `dev-sata1p3/state=in_sync` 且 `errors=0`，`(E)` 不会自动清除。

### 第三级：scemd → synopkg

scemd 启动时检测 `/proc/mdstat` 中的 `(E)` → 判定为 RAID crash → 写入 `/run/space/volume_status.cache`（`"status":1`）→ `mount -o remount,ro /volume2`

synopkg 的 feasibility check（`feasibility_check.cpp:129`）读取 volume status 缓存，`[0]` 号硬检查失败 → 拒绝启动所有依赖该卷的包。

---

## 三、关键诊断发现

### 3.1 SMART 数据 "不可用" 是误导

```bash
# 错误方式——走 SCSI 直接访问路径
smartctl -a /dev/sata1
# → "SMART support is: Unavailable - device lacks SMART capability"

# 正确方式——强制走 SCSI-ATA Translation (SAT) 路径
smartctl -d sat -a /dev/sata1
# → SMART support is: Available - device has SMART capability.
# → SMART overall-health self-assessment test result: PASSED
```

`-d sat` 是关键。因为 sata1 的 SCSI INQUIRY 返回的是 "WDC WDC"（SCSI device type）而非 "ATA"（SATA device type），smartctl 默认走 SCSI 路径拿不到 SMART。**sata2 返回 "ATA"，默认就能读 SMART**。

### 3.2 决定性证据：312 次 ICRC 错误

```bash
$ smartctl -d sat -l error /dev/sata1

ATA Error Count: 312 (device log contains only the most recent five errors)

Error 312:
  Error: ICRC, ABRT at LBA = 0x00000000 = 0
  （全部 312 次错误均为 ICRC + ABRT）

Error 311:
  Error: ICRC, ABRT at LBA = 0x00000000 = 0

...（全部同类型）
```

**ICRC = Interface Cyclic Redundancy Check**。这是 SATA **物理层**的校验错误——数据在线缆传输过程中被损坏，与磁盘介质无关。

所有 312 次错误全部是 ICRC/ABRT，零个坏扇区、零个待处理扇区、零个重分配扇区。

### 3.3 SATA 链路降级到 3.0 Gb/s

| 参数 | sata1（故障盘） | sata2（正常盘） |
|------|---------------|---------------|
| 型号 | WUH721816ALE6L4 | WUH721816ALE6L4 |
| 序列号 | 2CKWTD0J | 2CJZKTTN |
| 固件 | **PCGNW232** | PCGAW23G |
| SATA 支持 | 6.0 Gb/s | 6.0 Gb/s |
| 当前链路 | **3.0 Gb/s** | 6.0 Gb/s |
| SMART 默认 | 不可用（需 -d sat） | 可用 |
| SMART 健康 | PASSED | PASSED |
| 坏扇区 | 0 | 0 |
| ATA Error Count | **312**（全部 ICRC） | 0 |

SATA 控制器将链路协商降级到 3.0 Gb/s（SATA II 速度）是**信号完整性问题的自我保护行为**——与 WiFi 信号弱时降速同理。

### 3.4 SMART 属性全绿

```
  1 Raw_Read_Error_Rate          0
  5 Reallocated_Sector_Count     0
  7 Seek_Error_Rate              0
196 Reallocation_Event_Count     0
197 Current_Pending_Sector_Count 0
198 Off-Line_Uncorrectable       0
```

所有坏扇区相关属性为 0。13 次短自检中 11 次通过、2 次被宿主机中断（与盘无关）。

### 3.5 固件版本差异不是原因

| | sata1 | sata2 |
|---|---|---|
| 短标识 | W232 | W23G |
| 完整版本 | **PCGNW232** | **PCGAW23G** |
| 差异 | 可能是 OEM/bulk 版本 | 零售版本 |

两个版本都是 WD 的正式固件。PCGNW232 的 SCSI-ATA 翻译层不完全透明（导致 SMART 需要 `-d sat`），但这**不导致 ICRC 错误**——ICRC 是物理层的校验和错误，固件层面无法产生。

### 3.6 SMART 属性 199：UDMA CRC Error Count

本次诊断未显式查看属性 199（SATA CRC Error Count）。后续应监控这个属性的增长趋势——如果换线后还在涨，说明问题不在线缆。

### 3.7 天钡 WTR R1 物理排查

#### 3.7.1 散热风道不均

关机清灰时发现一个重要细节：**WTR R1 的 3 盘位中，只有一个盘位能直接吹到风扇**。桶形机箱的风扇位于顶部，气流主要覆盖靠近风扇的盘位，其余盘位依赖被动对流。两块 HC550 16TB 企业盘满负载功耗各约 8-12W，如果散热不均，长期积热可能影响 SATA 接口的电气特性。

#### 3.7.2 内部走线与接触

- **SATA 信号线非独立线缆**——WTR R1 等迷你主机没有传统 SATA 数据线，而是从主板上直接焊出一条细线接 SATA HDD 接口。这种硬连接方案**没有屏蔽层**，信号完整性天生不如标准 SATA 线缆，对接口接触状态和走线弯折更敏感
- **天钡桶形小主机内部空间紧凑**，焊出的细线可能在装配时弯折过度
- **SATA 背板信号质量**——这些小 NAS 机箱通常用非屏蔽排线或直接焊线
- **电源适配器裕度**——两块 HC550 启动电流峰值各约 25-30W，50-60W 瞬间负载可能超出适配器能力
- **SATA 控制器 ASPM 电源管理**——某些控制器在低功耗状态下链路不稳定
- 灰尘在接口区域的堆积也可能降低接触可靠性——清灰 + 重新插拔本身就是有效的修复手段

---

## 四、恢复流程

### 4.1 synopkg 和 synopkgctl 的关键区别

```
synopkg start <package>
  → feasibility check (feasibility_check.cpp:129)
    → [0] hard check: volume state   ← 这里被 scemd 缓存拦截
  → systemctl start pkgctl-<pkg>.service
    → synopkgctl start <package>      ← 实际启动

synopkgctl start <package>
  → 直接启动，不做 feasibility check  ← 恢复入口！
```

### 4.2 恢复脚本

```bash
# 1. 停 scemd 阻止复写
systemctl stop scemd

# 2. Remount rw
mount -o remount,rw /volume2
mount -o remount,rw /volume2/@docker

# 3. 修 volume status cache
# 编辑 /run/space/volume_status.cache
# 将 "/dev/vg2/volume_2" 的 status 从 1 改为 0

# 4. 通过 synopkgctl 启动（绕过 feasibility check）
/usr/syno/sbin/synopkgctl start ContainerManager
/usr/syno/sbin/synopkgctl start CloudSync

# 5. 修依赖链
systemctl reset-failed pgsql-adapter.service
systemctl start pgsql-adapter.service

# 6. 高层启动
/usr/syno/bin/synopkg start DownloadStation

# 7. 恢复 scemd
systemctl start scemd
```

`scemd` 重启后不再次设 ro —— sata1 磁盘介质本身完好，恢复后不再产生新的 I/O 错误。

### 4.3 (E) 标记的自动清除

`synopkgctl start ContainerManager` 执行过程中会触发群晖存储子系统的 RAID 状态重新评估。因为磁盘健康（state=in_sync, errors=0），`(E)` 标记自动消失。

---

## 五、修复路径

### 方案 A：重新插拔 SATA 接口（首选，已执行）

ICRC 错误 = 信号路径受损。WTR R1 的 SATA 连接是从主板焊出细线 → 硬盘接口，信号路径上的薄弱点主要在**接口接触面**。

操作步骤：
1. 关机，打开机箱
2. 拔下 SATA 接口（硬盘侧），清理接口和周围灰尘
3. 重新插紧，确保完全入位
4. 开机，`smartctl -d sat -i /dev/sata1` 确认链路协商到 6.0 Gb/s

> 注意：由于 SATA 线是从主板焊出的，无法像传统方案那样换线。如果重新插拔后问题复发，需考虑方案 B（端口互换隔离）或方案 D（换盘/检查焊点）。

### 方案 B：交换 SATA 端口

将 sata1 和 sata2 的端口互换。如果问题跟到另一个盘，是主板侧端口问题；如果留在原盘，是线缆问题。

### 方案 C：软件兜底

在物理修复前，部署 boot 自动恢复服务：

```ini
# /etc/systemd/system/nas-volume-recovery.service
[Unit]
Description=Auto-recover volume2 from ro after boot
After=multi-user.target
Wants=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/nas-volume-recovery.sh
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
```

配合检测脚本检测 `mount | grep '/volume2.*ro,'`。

### 方案 D：换盘

如果换线/换端口后问题仍复现，可能是硬盘的 SATA 接口有物理损伤。HC550 的 MTTF 为 250 万小时，但接口物理损伤与盘体寿命无关。

---

## 六、Synology 内部机制速查

| 组件 | 位置 | 用途 |
|------|------|------|
| `(E)` 标记 | `/proc/mdstat` | 群晖 md 驱动的持久化设备故障标记 |
| scemd | `systemctl status scemd` | 系统事件管理守护进程，控制卷挂载/卸载生命周期 |
| feasibility check | `/var/log/synofeasibilitycheck.log` | 包启动前的硬/软检查 |
| volume_state | `/run/space/volume_state` | 当前卷访问模式（rw/ro） |
| volume_status.cache | `/run/space/volume_status.cache` | scemd 缓存的卷健康状态 JSON |
| enabled flag | `/var/packages/<pkg>/enabled` | 空文件=启用，被 rm=禁用 |
| FHS target | `/var/packages/<pkg>/target` | → `/volumeN/@appstore/<pkg>` |
| pkgctl unit | `/usr/local/lib/systemd/system/pkgctl-*.service` | 包的主控 systemd 单元 |
| synopkg | `/usr/syno/bin/synopkg` | 高层包管理（带 feasibility check） |
| synopkgctl | `/usr/syno/sbin/synopkgctl` | 低层包控制（无 feasibility check） |
| spacetool | `/usr/syno/bin/spacetool` | 存储空间管理 CLI |

### 包状态码

| Code | 含义 |
|------|------|
| 0 | 正常 |
| 1 | unit active 但 status script 返回非0 |
| 150 | FHS target 读写状态检查失败 |
| 260 | 依赖服务不可达 |
| 262 | 包未启用（not turned on） |
| 263 | systemd 状态获取失败 |
| 272 | 上次启动失败（not startable） |
| 273 | systemd 状态无法转换 |

---

## 七、社区搜索关键字

后续追踪时用这些关键字组合搜索：

- `WD HC550 ICRC error SATA cable` — Reddit r/DataHoarder 和 r/linuxquestions 有类似案例
- `smartctl -d sat WDC WDC firmware` — SCSI-ATA 翻译层类型检测
- `Synology linear RAID (E) flag md_error` — 群晖 md 驱动私有标记
- `PCGNW232 firmware ICRC` — sata1 的完整固件版本
- `SATA link downshift 3.0 Gbps` — 链路降级问题
- `天钡 NAS SATA 背板 掉盘` — 同类硬件用户反馈
- `scemd volume remount ro RAID crash` — scemd 保护机制
- `synopkg feasibility_check package_start` — 群晖可行性检查
- `SATA CRC Error Count attribute 199` — ICRC 监控属性
- `WD HC550 startup current 25W` — 电源裕度排查
- `JMB585 ASM1166 ASPM SATA link unstable` — SATA 控制器电源管理问题

### r/linuxquestions 参考案例

社区有同为 ICRC 错误的类似帖子（SSD failing because of ATA Errors - Failed commands due to ICRC errors, reddit.com/r/linuxquestions/comments/1kyfdrw/），该案例中 SATA CRC_Error_Count = 86, SATA_Phy_Error_Count = 86——根源同样是物理层。社区回复第一建议始终是 "check the SATA cable"。

---

## 八、结论

| 项目 | 判定 |
|------|------|
| **根因** | SATA 物理层 ICRC 错误（线缆/背板/端口接触不良） |
| **介质** | 完美——0 坏扇区、0 待处理扇区、SMART PASSED |
| **固件** | PCGNW232 vs PCGAW23G——版本差异不导致 ICRC |
| **复现性** | 每次重启 100%（高 I/O 时触发） |
| **链路** | 从 6.0 Gb/s 降级到 3.0 Gb/s |
| **SMART 读取** | 需 `smartctl -d sat`，不能直接 `smartctl -a` |
| **恢复** | 停 scemd → remount rw → synopkgctl start → 重开 scemd |
| **永久修复** | 更换 SATA 数据线 / 重新插拔 / 检查电源 |

312 次 ICRC 错误、SATA 链路从 6.0 Gb/s 降级到 3.0 Gb/s、不同扇区随机出错——三条线索一致指向 SATA 物理层信号完整性。

---

## 九、物理修复与验证

### 9.1 操作步骤

1. 关机，打开 WTR R1 机箱
2. 三个盘位的 SATA 数据线和电源线全部**重新插拔**（同时清灰）
3. 将两块盘的 **SATA 端口互换**——原 sata1（PCGNW232）换到 sata2 口，原 sata2（PCGAW23G）换到 sata1 口
4. 开机，对照修复前的基线数据逐项验证

### 9.2 修复前后对比

| 检查项 | 修复前 | 修复后 |
|--------|--------|--------|
| sata1 SATA 链路 | **3.0 Gb/s** | **6.0 Gb/s** |
| sata2 SATA 链路 | 6.0 Gb/s | **6.0 Gb/s** |
| sata1 固件 | PCGNW232 | PCGAW23G（端口互换） |
| sata2 固件 | PCGAW23G | PCGNW232（端口互换） |
| /proc/mdstat (E) 标记 | `sata1p3[0](E)` | `[UU]` 无标记 |
| dmesg I/O errors | `blk_update_request: I/O error` | 干净 |
| sata1 ATA Error Count | 312（全部 ICRC） | 0 |
| /volume2 挂载 | ro | **rw** |
| CloudSync | error 272 | running (status 0) |
| ContainerManager | error 272 | status_code 1（unit active，所有容器 healthy） |
| Docker 容器 | 全部挂 | gitea / open-webui / opengist → healthy |

### 9.3 关键结论

**端口互换后，原故障盘（PCGNW232）在新端口上也协商到 6.0 Gb/s，零 ICRC 错误。** 这排除了：
- **磁盘故障**——同一块盘之前 312 次 ICRC，换端口后零错误
- **主板 SATA 端口故障**——原 sata2 口接 PCGNW232 后正常工作
- **固件版本差异**——PCGNW232 在 sata2 口上同样稳定

**根因锁定为 SATA 接口接触不良。** 清灰 + 重新插拔解决了问题——WTR R1 的 SATA 连接是主板焊出细线而非独立线缆，额外增加了信号路径的不确定性。可能的原因包括灰尘导致微氧化、接口未完全入位、或焊出线在机箱内弯折过度造成的间歇性接触不良。

### 9.4 后续监控建议

即使修复后验证通过，仍需持续监控：

```bash
# 每次重启后跑
sudo smartctl -d sat -l error /dev/sata1 | grep "ATA Error Count"
sudo smartctl -d sat -l error /dev/sata2 | grep "ATA Error Count"
# 预期均为 0，如果开始增长说明接触问题复发

# 监控属性 199 (SATA CRC Error Count)
sudo smartctl -d sat -A /dev/sata1 | grep "^199"
sudo smartctl -d sat -A /dev/sata2 | grep "^199"
```

如果问题复发，由于 SATA 线直接从主板焊出无法简单更换，下一步是**检查主板焊点是否虚焊**，或**对风道不好的盘位增加散热**（如加装小型散热片或改进机箱通风）。也可以考虑在硬盘侧加装 SATA 转接板，将焊接走线替换为可更换的 SATA 标准线缆。

---

*调查日期：2026-05-10 | 修复日期：2026-05-10 |
设备：天钡 Tbao WTR R1 桶形 3 盘位 NAS 主机（Intel N100, 双 2.5G 网口）+ SA6400 loader, DSM 7.2 |
磁盘：2×WD WUH721816ALE6L4 16TB HC550 (PCGNW232 + PCGAW23G) |
Linear JBOD, Btrfs on dm-crypt + NVMe cache*
