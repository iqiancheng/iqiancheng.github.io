---
layout: post
author: Joseph
title: "Docker 镜像加速站 2026 实测：14 站还剩几家能用？"
date: 2026-05-10 00:00:00 +0800
categories: [homelab, devops]
tags: [docker, proxy, nas, devops]
description: >
  在 Synology NAS 上对 14 个 Docker 镜像加速站逐一测试，找出 2026 年仍可用的站，
  给出 Synology DSM 上的配置方法和测试命令。
toc: true
---

## 背景

[前三篇 NAS 博文](/posts/synology-nas-advanced-exploration/) 规划了 Docker 容器扩展路线，但在实际 `docker pull` 时发现 Docker Hub (`registry-1.docker.io`) 被墙，所有国内传统镜像站也大面积停服。需要找到 2026 年仍可用的镜像加速方案。

测试环境：Synology SA6400, DSM 7.x, Docker 24.0.2, 中国移动宽带。

---

## 一、为什么传统镜像站失效？

Docker 的 `registry-mirrors` 机制不是透明代理——它要求 Docker daemon 先到 Docker Hub 完成认证，然后才能从 mirror 拉取 blob。这意味着：

1. Docker Hub 被 DNS 污染 + SNI 阻断 → mirror 机制的第一步就挂了
2. 2024 年中起，国内各大高校和云厂商相继关停 Docker 镜像服务（合规原因）
3. 幸存 mirror 多为白名单模式——只有已缓存的镜像才能拉，新镜像需要 mirror 运营方主动同步

---

## 二、测试方法

Docker Registry v2 API 的 `/v2/` 端点无需认证即可访问。返回值含义：

| HTTP 状态码 | 含义 |
|------------|------|
| 200 | 完全可达，镜像可拉 |
| 401 | 需要认证（正常——registry 在线，Docker daemon 会自动完成认证再拉取） |
| 302 | 重定向到其他站 |
| 403 | 拒绝访问（限制外部访问） |
| 000 (curl 超时) | **不可用** |

测试命令：

```bash
curl -sk --connect-timeout 5 -o /dev/null -w '%{http_code}|%{time_total}' 'https://<mirror-host>/v2/'
```

---

## 三、14 个镜像站逐站测试

以下测试全部从 NAS 直连执行（`curl` 不走代理）。

| # | 镜像站 | HTTP | 延迟 | 判定 |
|---|--------|------|------|------|
| 1 | `docker.1panel.live` | **200** | 0.94s | **可用** |
| 2 | `docker.1ms.run` | 401 | 0.16s | **可用** |
| 3 | `docker.m.daocloud.io` | 401 | 0.17s | **可用** |
| 4 | `mirror.iscas.ac.cn` | 401 | 0.21s | **可用** |
| 5 | `05f073ad3c0010ea0f4bc00b7105ec20.mirror.swr.myhuaweicloud.com` | 401 | 0.22s | **可用** |
| 6 | `hub.rat.dev` | 302 | 0.09s | **重定向到 docker.1ms.run** |
| 7 | `hub-mirror.c.163.com` | 000 | 0.05s | 不可用 |
| 8 | `mirror.ccs.tencentyun.com` | 000 | 0.05s | 不可用 |
| 9 | `registry.docker-cn.com` | 000 | 5.00s | 不可用 |
| 10 | `dockerpull.com` | 000 | 2.51s | 不可用 |
| 11 | `dockerproxy.cn` | 000 | 0.05s | 不可用 |
| 12 | `docker.rainbond.cc` | 000 | 0.06s | 不可用 |
| 13 | `docker.udayun.com` | 000 | 2.52s | 不可用 |
| 14 | `docker.211678.top` | 000 | 2.51s | 不可用 |

**结论：14 站中 5 个可用，1 个是别名，8 个已失效。** 网易、腾讯云、Docker 中国官方、七牛云、Azure 中国等大厂镜像站在 2023-2024 年已相继关闭（资料中确认）。

---

## 四、Synology DSM 上的落地配置

> **关键差异**：Synology Container Manager 的配置在 `/var/packages/ContainerManager/etc/dockerd.json`，**不是** `/etc/docker/daemon.json`。后者的 registry-mirrors 会被前者覆盖。

### 4.1 更新 dockerd.json

```bash
sudo python3 -c "
import json
with open('/var/packages/ContainerManager/etc/dockerd.json') as f:
    config = json.load(f)
config['registry-mirrors'] = [
    'https://docker.1panel.live',
    'https://docker.1ms.run',
    'https://docker.m.daocloud.io',
    'https://mirror.iscas.ac.cn',
    'https://05f073ad3c0010ea0f4bc00b7105ec20.mirror.swr.myhuaweicloud.com'
]
with open('/var/packages/ContainerManager/etc/dockerd.json', 'w') as f:
    json.dump(config, f, indent=2)
"
```

### 4.2 重启 Docker

```bash
sudo systemctl restart pkgctl-ContainerManager
```

### 4.3 验证

```bash
$ sudo docker info | grep -A5 'Registry Mirrors'
 Registry Mirrors:
  https://docker.1panel.live/
  https://docker.1ms.run/
  https://docker.m.daocloud.io/
  https://mirror.iscas.ac.cn/
  ...
```

### 4.4 测试拉取

```bash
sudo docker pull metacubex/mihomo:latest
```

如果成功拉取，配置生效。

---

## 五、排坑记录

### 5.1 Docker daemon 残留代理

旧版 Clash 容器可能在 `dockerd.json` 中残留 `proxies` 配置：

```json
"proxies": {"http-proxy": "192.168.x.x:7890", "https-proxy": "192.168.x.x:7890"}
```

现象：`docker pull` 报 `proxyconnect tcp: dial tcp 192.168.x.x:7890: connect: no route to host`。

修复：删除 `proxies` 字段后重启。

### 5.2 registry-mirrors 空值覆盖

`dockerd.json` 中 `"registry-mirrors":[]` 会覆盖 `/etc/docker/daemon.json` 的配置。
两处必须一致或在 `dockerd.json` 中统一维护。

### 5.3 Mirror 缓存延迟

即使 mirror 可达，某些镜像层可能未缓存（mirror 需要临时从 Docker Hub 同步）。
遇到 `Pulling fs layer` 卡住时等待 1-2 分钟，不要立即 kill。

### 5.4 Synology synopkg restart 不可靠

`synopkg restart ContainerManager` 返回 error 275 但不一定表示失败。
改用 `systemctl restart pkgctl-ContainerManager` 更可靠。

---

## 六、镜像站容错机制

Docker daemon 会按 `registry-mirrors` 顺序**依次尝试**，第一个超时或无缓存则 fallback 到下一个。
因此排序有讲究：

```
1. docker.1panel.live    ← 200 完全可达，做主力
2. docker.1ms.run        ← 延迟最低(0.16s)
3. docker.m.daocloud.io  ← DaoCloud，多注册表支持
4. mirror.iscas.ac.cn    ← 中科院软件所，稳定
5. mirror.swr.myhuaweicloud.com ← 华为云，个人账号专属
```

1panel 在前因为它返回 200（代理模式，不需要 Docker Hub 认证），其余 401 的做 fallback。

---

## 七、长期维护建议

- **每季度**用 curl 跑一轮上述测试，剔除新增失效的站
- **关注社区**：DaoCloud 的 [public-image-mirror](https://github.com/DaoCloud/public-image-mirror) 项目持续更新可用镜像站状态
- **自建代理**：如果所有 mirror 都停服了，最后的兜底方案是 NAS 上部署 mihomo → Docker daemon 通过 HTTP_PROXY 走代理（下一篇会覆盖）

---

## 总结

2026 年中国大陆 Docker 镜像站的现状是**大厂全灭，社区抱团**。14 个站中能用的只有 5 个，全是社区或个人维护的小站。传统推荐官档里的 `mirror.ccs.tencentyun.com`、`registry.docker-cn.com` 等全线阵亡，网上搜到的教程大多已过时。

> 与 NAS 系列的联动：Docker 镜像加速是本 session 部署 mihomo 代理的前置依赖。接下来用 mihomo 跑 Trojan 出口后，可以进一步给 Docker daemon 配置 `HTTP_PROXY` 直连 Docker Hub，彻底绕过 mirror。
