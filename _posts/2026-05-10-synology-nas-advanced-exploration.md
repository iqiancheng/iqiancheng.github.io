---
layout: post
author: Joseph
title: "家用 NAS 进阶路线图：Python API + Docker 生态 + 自托管清单"
date: 2026-05-10 00:00:00 +0800
categories: [homelab]
tags: [tooling, docker, networking, nas, github]
description: >
  基于 SA6400 (DSM 7.x, Epyc, 2×14.6TB HDD + 1.8TB NVMe) 的现有配置和兴趣点，
  对 N4S4/synology-api Python 库、Docker 容器生态、自建服务方向进行全面技术调研，
  梳理短期可执行清单和长期演进路线。
toc: true
---
## 背景

前三篇博文分别覆盖了[组网与 WebDAV 部署](/posts/tailscale-synology-nas-real-world/)、[安全加固](/posts/synology-nas-security-hardening/)和[存储优化](/posts/synology-nas-storage-optimization/)，把 NAS 从裸奔状态推到了"网络-安全-存储"三层基准线。这一篇往上层走：**围绕现有 Docker 服务、Python API 能力和兴趣方向，规划 NAS 作为家庭数据中心的技术演进路线**。

核心问题：除了文件存储和 WebDAV，这台 SA6400 还能做什么？有哪些 API 可以替代 DSM Web 手动操作？已经拉取的 Docker 镜像有哪些值得跑起来？

**测试环境**：Synology SA6400, DSM 7.x, 2×14.6TB HDD (LINEAR, 29TB) + 1×1.8TB NVMe SSD, Btrfs, Tailscale 已部署。

---

## 一、现有基础设施回顾

### 1.1 核心服务

| 服务 | 实现方式 | 端口 | 状态 |
|------|---------|------|------|
| SSH | 密钥认证, ControlMaster 复用 | 9525 | 生产 |
| WebDAV | Apache 2.4 + Tailscale Serve | 5006 / ts.net | 生产 |
| SMB | 局域网, Finder 自动发现 | 139/445 | 生产 |
| Git | Gitea (Docker) | 3014 (Web), 222 (SSH) | 生产 |
| AI Chat | Open-WebUI (Docker) | 8000 | 运行中 |
| Gist/Paste | OpenGist (Docker) | 6157 | 运行中 |
| 系统通知 | SMTP via 163.com:465 | — | 配置完成 |

### 1.2 安全与网络层

```
iptables 白名单 + 默认 DROP
├── 局域网: SSH(9525), DSM Web(5001), HTTPS(443), SMB(139/445/137-138)
├── Tailscale: SSH(9525), WebDAV(5006)
├── ICMP: 全开
└── 其他: DROP (已拦截千级数据包)

TCP 调优: rmem_max=2MB, fastopen=3, slow_start_after_idle=0
Apache MPM: ThreadsPerChild=25 (Worker 模式)
```

### 1.3 存储层

```
NVMe 1.8TB → volume1 (系统/套件, 460MB used)
HDD×2 29TB → volume2 (LINEAR, 5.9TB used)
├── Btrfs zstd: homes + docker ✅
├── Btrfs zstd: downloads ❌ (DSM 锁定共享文件夹子卷)
├── Scrub: 每月1日 03:00 ✅
├── 去重: 每日 00:00 ✅
├── TRIM: 每日 00:00 ✅
└── 快照: 无（公开影视数据无需冗余）
```

---

## 二、Docker 现状审计

### 2.1 运行中的容器

```bash
{% raw %}$ /usr/local/bin/docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}"{% endraw %}
```

| 容器 | 镜像 | 端口映射 | 用途 |
|------|------|---------|------|
| gitea | gitea/gitea:1.22.6 | 0.0.0.0:222→22, 0.0.0.0:3014→3000 | 自建代码仓库 |
| gitea_mysql | mysql:8 | 3306 | Gitea 数据库后端 |
| open-webui | open-webui:main | 0.0.0.0:8000→8080 | AI 对话 Web 前端 |
| opengist | opengist:1 | 0.0.0.0:6157→6157 | 代码片段/Gist 分享 |

Gitea + OpenGist 构成了完整的代码托管 + 片段分享工作流，Open-WebUI 提供了 AI 聊天前端。这三个服务已经在日常使用中。

### 2.2 已下载未运行的镜像

```bash
{% raw %}$ /usr/local/bin/docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"{% endraw %}
```

| 镜像 | 用途 | 可能方向 |
|------|------|---------|
| `minio/minio` | S3 兼容对象存储 | 替代 AWS S3 做本地 API，运行对象存储工作负载 |
| `1panel/maxkb` | 知识库系统 | 配合 Open-WebUI 做 RAG（检索增强生成） |
| `dreamacro/clash` | 代理工具 | 旧 DERP 替代方案的遗留镜像（待评估是否保留） |
| `sujaykumarh/docsify` | 文档站点生成器 | Gitea + Docsify 文档流水线 |
| `moelin/1panel` | 服务器管理面板 | 轻量 Linux 面板（与 DSM 功能重叠，待评估） |

这些镜像已经 `pull` 下来但从未启动过，说明曾经有过相关计划但未执行。下面会逐一评估。

---

## 三、已确认不感兴趣的方向

为避免无效探索，先明确排除以下方向：

- Ollama / 本地大模型推理（硬件无 GPU，不跑本地模型）
- Paperless-ngx / 文档 OCR 管理
- Vaultwarden / 自建密码管理器
- RAID 冗余 / 数据快照（影视数据可重新获取）

---

## 四、N4S4/synology-api：Python 程序化管理的核心

[N4S4/synology-api](https://github.com/N4S4/synology-api) 是一个社区维护的 Python 库，封装了 Synology DSM 的 Web API（50+ 模块，300+ 端点），支持 DSM 6.x 和 7.x。

### 4.1 基础连接模式

```python
from synology_api.filestation import FileStation
fs = FileStation(ip, port, user, password, dsm_version=7)
```

已配置 sudo NOPASSWD (`/etc/sudoers.d/`)，支持 `ssh opennas "sudo ..."` 无交互远程执行，为 Python API 自动化提供了基础。

### 4.2 高优先级模块

| 模块 | 能力概述 | 价值 |
|------|---------|------|
| `docker_api` | 容器全生命周期管理（40+ 方法），可替代 DSM Container Manager UI | **替代 DSM UI 手动操作** |
| `core_storage` | 磁盘/存储池/卷/配额/回收站程序化管理 | **存储健康监控自动化** |

**docker_api 关键方法**：
- `get_container_list()` — 列出所有容器及状态
- `container_start(id)` / `container_stop(id)` / `container_restart(id)`
- `get_container_log(id)` — 获取容器日志
- `get_images_list()` — 列出所有镜像
- `image_pull()` / `image_remove()` — 镜像管理
- `network_list()` / `network_create()` — 网络管理
- `project_list()` / `project_create()` — Docker Compose 项目管理

有了这个模块，可以编写 Python 脚本实现容器状态监控、自动备份 Gitea 仓库、异常容器自愈等自动化运维，不需要打开 DSM Web 界面。

**core_storage 关键方法**：
- `storage_disk_list()` — 所有磁盘信息（型号、序列号、温度、健康状态）
- `storage_pool_list()` — 存储池状态（RAID 类型、容量、使用率）
- `storage_volume_list()` — 卷列表
- `storage_quota_list()` — 配额信息
- `recycle_bin_list()` / `recycle_bin_clear()` — 回收站管理

### 4.3 用户兴趣已确认的模块

以下模块是用户明确表示感兴趣的方向：

| 模块 | 核心能力 | 应用场景 |
|------|---------|---------|
| `cloud_sync` | S3/B2/OneDrive 等云同步任务管理，连接创建/暂停/恢复 | 检查现有云同步状态，管理同步策略 |
| `photos` | 相册操作、条件相册、分享链接 API | 照片管理自动化 |
| `vpn` | VPN Server 管理、连接监控、OpenVPN 配置导出 | VPN 状态监控 |
| `dhcp_server` | DHCP 租约列表、IP 保留、PXE/TFTP 配置 | 家庭网络 IP 管理 |
| `directory_server` | LDAP 用户/组 CRUD、密码重置 | **家庭统一认证基础** |
| `universal_search` | 全盘文件搜索 API（单一方法 `search(keyword)`） | 快速定位文件 |
| `log_center` | 集中日志查询、远程归档、日志转发 | 配合健康检查脚本做告警 |
| `virtualization` | VMM 虚拟机管理（启停、快照、镜像创建） | 实验环境管理 |

### 4.4 中低优先级模块

| 模块 | 能力 | 场景 |
|------|------|------|
| `core_network` | 静态路由、UPnP、WOL、流量控制、网络接口管理 | 网络配置自动化 |
| `core_security` | 防火墙规则、自动封锁、证书管理 | 安全策略 GitOps 化 |
| `core_backup` | Hyper Backup 任务管理、完整性校验 | 备份状态监控 |
| `core_upgrade` | DSM 更新检查/下载/安装 | 已禁用自动更新，手动管理 |
| `task_scheduler` | 排程任务 CRUD | 替代手动创建 task 文件 |

---

## 五、技术方向路线图

```
┌─────────────────────────────────────────────────────┐
│                 NAS 高级用法探索路线                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  1. Python 运维自动化                                 │
│     ├── Docker 容器监控 + 自愈脚本                     │
│     ├── 存储健康检查 + 告警推送                        │
│     └── iptables/安全配置 GitOps 化                   │
│                                                      │
│  2. 存储扩展探索                                      │
│     ├── MinIO S3 对象存储 (已拉取镜像)                 │
│     └── S3 API vs WebDAV 性能对比                     │
│                                                      │
│  3. AI/知识管理                                       │
│     ├── MaxKB + Open-WebUI (RAG 知识库)               │
│     └── 文档向量化 + 私有知识问答                      │
│                                                      │
│  4. 开发工具链                                        │
│     ├── Gitea + Docsify 文档流水线                    │
│     └── OpenGist 代码片段集成                          │
│                                                      │
│  5. 自建 DERP (遗留 Task 13)                          │
│     └── AWS 新加坡 EC2 部署 derper                    │
│                                                      │
│  6. 统一认证 (LDAP)                                   │
│     └── directory_server API → Gitea/OpenWebUI 等     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 5.1 Python 运维自动化

这是最高优先级的方向，因为立即可行且有直接收益。

**Docker 容器监控 + 自愈脚本**：

```python
from synology_api.docker_api import DockerAPI
import smtplib

docker = DockerAPI(ip, port, user, password, dsm_version=7)
containers = docker.get_container_list()
for c in containers:
    if c['status'] != 'running':
        # 尝试重启或发送告警
        send_alert(f"Container {c['name']} is {c['status']}")
```

**存储健康检查 + 告警推送**：结合 `core_storage` 获取磁盘 SMART 状态、存储池使用率、Btrfs scrub 结果，通过 DSM 已配置的邮件通知（163.com:465）发送告警。DSM 系统邮件已由 `[192.168.x.x] OpenAI NAS` 发送者配置完毕。

### 5.2 MinIO S3 对象存储

MinIO 镜像已经 pull 好，部署成本极低。S3 API 相比 WebDAV 的优势：

- 标准的 AWS SDK 支持（Python boto3、Java、Go 等）
- 原生支持分片上传、断点续传
- Bucket 级别的访问控制和版本管理
- 可以作为本地 S3 替代方案，用于开发测试

值得跑起来做一次 S3 API vs WebDAV 的性能对比测试。

### 5.3 MaxKB + Open-WebUI RAG 知识库

[MaxKB](https://github.com/1Panel-dev/MaxKB) 是一个开源的 LLM 知识库问答系统，支持文档向量化存储和检索增强生成（RAG）。结合已运行的 Open-WebUI：

```
文档 (PDF/Markdown/代码) → MaxKB 向量化 → 向量数据库
                                             ↓
用户提问 → Open-WebUI → 检索相关文档 → LLM API → 回答
```

前置问题：需要确认 Open-WebUI 的后端是什么（远程 API 还是本地模型？），以及 LLM API 是否可用于 RAG 场景。

### 5.4 开发工具链：Gitea + Docsify

[Docsify](https://docsify.js.org) 是一个轻量级文档站点生成器（无需构建，纯运行时渲染 Markdown）。与 Gitea 配合的流水线：

```
Gitea 仓库 (Markdown 文档) → Webhook → Docsify 容器 → 文档站点
```

优势是文档即代码，版本管理和协作在 Gitea 中完成，Docsify 只负责渲染。比 Wiki 系统更轻量。

### 5.5 自建 DERP 中继服务器

这是上一篇 Tailscale 实战的遗留问题。当前手机通过 `DERP(sfo)` 中继访问 NAS WebDAV，延迟 400-900ms，吞吐量仅 120-500KB/s。

[Headscale 社区方案](https://tailscale.com/kb/1118/custom-derp-servers/)：在 AWS EC2 新加坡（`ap-southeast-1`）上部署 `derper`，NAS 到新加坡实测延迟仅 55ms。

```
手机 → 自建 DERP (新加坡, ~55ms) → NAS
      ↑                              ↑
      CDN 优化线路                   55ms 直连
```

预计效果：WebDAV 吞吐从 500KB/s → 2-5MB/s，延迟从 400+ms → 200ms 以内。

### 5.6 LDAP 统一认证

Synology 内置的 Directory Server 提供 LDAP 服务，结合 `directory_server` Python API 可以：

- 创建统一的家庭用户目录（区别于 Synology 本地用户）
- Gitea 支持 LDAP 认证
- Open-WebUI 可通过反向代理接入 LDAP
- DSM 本身也可以作为 LDAP 客户端

目标：一个用户名密码登录所有自建服务。不过这个方向投入较高，建议在完成前几个方向后再评估。

---

## 六、短期可执行清单

| # | 方向 | 投入 | 产出 | 阻塞 |
|---|------|------|------|------|
| 1 | `docker_api` Python 脚本 — 列出所有容器状态 + 自动备份 Gitea | 2h | 替代 DSM UI 手动操作 | 无 |
| 2 | MinIO 部署验证 — 跑起来看 S3 API 是否可用 | 1h | 本地 S3 替代方案 | 无 |
| 3 | `core_storage` + `log_center` 健康检查脚本 | 3h | 自动巡检 + 邮件告警 | 需先验证 DSM 邮件通知是否能收到 |
| 4 | `cloud_sync` API — 检查现有云同步状态 | 0.5h | 了解当前云同步连接 | 无 |
| 5 | 自建 DERP (新加坡 EC2) | 4h | 手机 WebDAV 从 500KB/s → 2-5MB/s | 需 AWS EC2 新加坡可用 |
| 6 | MaxKB + Open-WebUI 集成 | 3h | RAG 知识库问答 | Open-WebUI 需确认后端 API |

---

## 七、待回答的问题

- [ ] Open-WebUI 的后端是什么？本地模型还是远程 API？
- [ ] MinIO 当初拉取镜像的原始需求是什么？
- [ ] 有没有云同步任务在跑（Cloud Sync → OneDrive/其他）？
- [ ] 对 LDAP 统一认证有兴趣吗（SSO for Gitea/OpenGist/DSM）？
- [ ] AWS EC2 新加坡现在状态如何？是否可以 SSH 上去？

---

## 总结

前三篇把 NAS 从"裸奔"推到了"能用且安全"的基准线——网络层 TCP 调优 + WebDAV 替代 SMB（吞吐 4.2→9.8 MB/s），安全层 27 端口收敛到 8 个 + 全白名单，存储层 zstd 压缩 + 月度 scrub。这一篇规划的是"从能用到好用"的下一阶段。

核心思路是：**用 Python API 替代 DSM Web 手动操作，用 Docker 容器扩展服务边界，用自建 DERP 解决最后一公里延迟**。

三个层次的演进路径：

- **近期（1-2 天）**：docker_api + core_storage 脚本化，MinIO 部署验证
- **中期（1-2 周）**：自建 DERP 解决手机慢，MaxKB 知识库集成
- **长期（按需）**：LDAP 统一认证，Gitea + Docsify 文档流水线

> 与前三篇联动：本篇的 Python 自动化可以和存储层的 scrub 健康检查（第三篇）、安全层的 iptables 配置 GitOps 化（第二篇）结合。自建 DERP 直接解决第一篇遗留的手机 WebDAV 速度瓶颈。
