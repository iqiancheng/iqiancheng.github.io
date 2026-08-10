---
layout: post
author: Joseph
title: "Cloud Sync 静默丢文件：一个 NFD/NFC 编码踩出的坑"
date: 2026-05-10 00:00:00 +0800
categories: [homelab, storage]
tags: [tooling, nas]
description: >
  Cloud Sync 同步 Google Drive 时静默跳过含重音字符的文件，日志提示 "EVENT is not NFC Form"。
  原因是 macOS 使用 NFD 编码文件名而 Linux/Synology 期望 NFC。记录问题根因与解决方案。
toc: true
---

## 问题现象

配置 Cloud Sync 同步 Google Drive 后，发现部分文件**静默丢失**——不报错、不同步、日志里只有一行 warning：

```log
Aug 25 02:10:20 [WARNING] event-manager.cpp(220):
  EVENT is not NFC Form 'Event<EV_ADD> (WAITTING):
  /test/ttéà.docx (server,file) size = 6407,
  hash = b191db19481e35cad2d189799ff88441', skipping...
```

文件名叫 `ttéà.docx`，但 Cloud Sync 直接把它**跳过**了。更糟糕的是——你完全不知道丢了什么。

---

## 根因：Unicode 归一化形式差异

### NFC vs NFD

Unicode 中带重音的字符（如 `é`、`à`、`ü`）有两种合法编码方式：

| 归一化形式 | 编码方式 | 示例 `é` |
|-----------|---------|---------|
| **NFC**（合成，Composed） | 单个码点 | `U+00E9`（1 个字符） |
| **NFD**（分解，Decomposed） | 基字符 + 组合用读音符 | `e` + `◌́`（2 个字符） |

两者**屏幕显示完全一样**，但底层字节不同，字符串比较不相等。

### 各平台默认行为

| 平台 / 服务 | 文件系统 | 默认归一化形式 |
|------------|---------|--------------|
| **macOS** | APFS / HFS+ | **NFD** |
| **Windows** | NTFS | NFC（但允许 NFD） |
| **Linux** | ext4 / Btrfs | 不强制归一化 |
| **Google Drive** | — | **NFC** |
| **Synology DSM** | Btrfs | 期望 NFC |

### 问题链路

```
macOS 创建文件 "café.pdf"（NFD 编码）
        ↓ 上传
   Google Drive 存储（NFC 编码，但保留 NFD 也可）
        ↓ Cloud Sync 下载
   Synology 检查文件名 → "这是 NFD，不是 NFC" → 跳过！
```

Cloud Sync 的 `event-manager.cpp` 在 PushEvent 阶段做了一个**硬编码的 NFC 检查**，NFD 文件名直接丢弃。这是 Synology 的一个久未修复的 bug。

---

## 为什么这个问题被骂了好几年

Synology 社区论坛上最早的相关报告可以追溯到 2019 年甚至更早：

- [社区帖子 1：UTF-8 NFD characters not syncing](https://community.synology.com/enu/forum/17/post/98730)
- [社区帖子 2：Cloud Sync skipping files with accents](https://community.synology.com/enu/forum/17/post/116542)
- [社区帖子 3：Google Drive sync missing files](https://community.synology.com/enu/forum/17/post/50677)

用户反复提交 ticket、社区持续抱怨，Synology 始终没从根本上改 `event-manager.cpp` 的行为逻辑——直到 DSM 7.x 后期才加了一个 workaround 复选框。

---

## 解决方案

### 方案一：勾选 Cloud Sync 的 NFD 转换选项（推荐）

DSM 7.x 某版后在 Cloud Sync 任务设置中增加了一个复选框：

> **Convert NFD characters to NFC during sync**

勾上后，Cloud Sync 同步时会自动将 NFD 文件名转换为 NFC，文件不再被跳过。

### 方案二：在 macOS 端预处理文件名（批量修复存量文件）

如果有大量历史文件需要转换，可以用 `iconv` 或 `convmv`：

```bash
# macOS 上批量转换当前目录下所有文件名 NFD → NFC
brew install convmv
convmv --nfd --nfc -r /path/to/files --notest
```

或者用 Python 脚本：

```python
import unicodedata
import os

def normalize_to_nfc(path):
    """重命名目录树中所有 NFD 文件名为 NFC"""
    for root, dirs, files in os.walk(path):
        for name in files + dirs:
            nfc_name = unicodedata.normalize('NFC', name)
            if nfc_name != name:
                old = os.path.join(root, name)
                new = os.path.join(root, nfc_name)
                os.rename(old, new)
                print(f"  {name} → {nfc_name}")
```

### 方案三：从源头避免

在 macOS 上创建文件时使用 NFC：

```bash
# 创建新文件时强制 NFC 命名
touch "$(echo 'café' | iconv -t UTF-8-MAC | iconv -f UTF-8-MAC)"
```

> **注意**：方案二和方案三只能处理**还未同步**的文件。如果文件已经在 Google Drive 上且是 NFD 编码，需要在 Google Drive 端重命名。

---

## 影响范围

这个问题不仅影响 Google Drive，**所有通过 Cloud Sync 同步的云存储**都可能受影响：

| 云服务 | 是否受影响 | 备注 |
|--------|----------|------|
| Google Drive | ✅ 是 | NFD 文件静默跳过 |
| Dropbox | ✅ 是 | 同样行为 |
| OneDrive | ✅ 是 | 同样行为 |
| Backblaze B2 | ❓ 未确认 | 对象存储，取决于文件名处理 |
| WebDAV 远端 | 取决于服务端 | Linux 服务端通常 NFC |

核心原因不在云服务端，而在 **Synology Cloud Sync 的 event-manager** 里那个 NFC check。

---

## 相关资源

- [前一篇：pool2 掉盘修复](/2026/05/10/synology-nas-pool2-recovery.html)
- [Reddit: Cloud Sync does not synchronize Google Drive files with NFD characters](https://www.reddit.com/r/synology/comments/1f0z9zn/)
- [Unicode 归一化形式规范 (UAX #15)](https://unicode.org/reports/tr15/)
- [Apple 文档：文件系统与 Unicode 归一化](https://developer.apple.com/library/archive/qa/qa1235/_index.html)
