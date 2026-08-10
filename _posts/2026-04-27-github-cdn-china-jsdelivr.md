---
layout: post
title: "大陆高速访问 GitHub 资源：jsDelivr CDN 拼接技巧与镜像实测"
date: 2026-04-27 00:00:00 +0800
author: Joseph
categories: [网络, 工具]
tags: [proxy, github, devops]
---

`raw.githubusercontent.com` 在大陆经常超时或被 SNI 阻断。本文记录通过 jsDelivr 及其多个子域名/镜像，在不挂代理的情况下高速拉取 GitHub 仓库中的静态资源（配置文件、图片、JS 脚本等）的实践。

---

## 一、URL 拼接通用格式

jsDelivr 提供 GitHub 资源的 CDN 加速，统一格式为：

```
https://{host}/gh/{user}/{repo}@{ref}/{path}
```

| 占位符 | 说明 | 示例 |
|--------|------|------|
| `{host}` | CDN 节点域名 | `fastly.jsdelivr.net` |
| `{user}/{repo}` | GitHub 用户名/仓库名 | `pengyanai/misc` |
| `{ref}` | 分支名、tag 或 commit SHA | `main`、`v1.0`、`9974b7b...` |
| `{path}` | 仓库内文件路径 | `collapsed-tiles/img/claude.png` |

示例：

```
https://fastly.jsdelivr.net/gh/pengyanai/misc@main/collapsed-tiles/global-ai-availability.stoverride
```

### ref 的三种写法

| 写法 | 特点 | 适用场景 |
|------|------|----------|
| `@main` | 跟踪分支最新 | 开发阶段，内容会随 push 更新（有缓存延迟） |
| `@v1.0.0` | 跟踪 Git tag | 发布版本，语义化引用 |
| `@9974b7b3a44b...` | 锁定 commit SHA | 需要不可变链接时使用，确保内容永远不变 |

> 锁定 commit SHA 在代理工具配置中尤其有用——避免 CDN 缓存未刷新导致配置不一致。

---

## 二、可用 CDN 节点域名对比

以下域名均遵循同一 URL 格式，只是底层 CDN 提供商不同，大陆可达性差异明显：

| 域名 | CDN 后端 | 大陆可用性 | 备注 |
|------|----------|-----------|------|
| `cdn.jsdelivr.net` | 自动调度（Fastly/CF/Gcore） | 一般 | 官方默认域名，有时被间歇干扰 |
| `fastly.jsdelivr.net` | Fastly | **较好** | 大陆实测稳定，推荐首选 |
| `gcore.jsdelivr.net` | Gcore | **较好** | Gcore 在亚太有节点，速度不错 |
| `testingcf.jsdelivr.net` | Cloudflare | 一般 | CF 节点多但大陆延迟波动较大 |
| `jsd.onmicrosoft.cn` | 国内社区镜像 | **好** | `.cn` 域名，大陆直连最快 |
| `jsd.cdn.zzko.cn` | 国内社区镜像 | 已失效 | 证书过期，不再可用 |

**推荐优先级**（大陆无代理场景）：

```
jsd.onmicrosoft.cn > fastly.jsdelivr.net > gcore.jsdelivr.net > cdn.jsdelivr.net
```

---

## 三、GitHub 原生 raw 域名

```
https://raw.githubusercontent.com/{user}/{repo}/{ref}/{path}
```

- 大陆直连**经常超时**或被 RST，不建议作为生产链接
- 可用 `refs/heads/main` 替代 `main`（完整 ref 路径写法）
- 适合作为 fallback 或仅在代理环境下使用

---

## 四、缓存刷新：purge 接口

jsDelivr 默认缓存 24h，push 新内容后不会立即生效。手动刷新：

```
https://purge.jsdelivr.net/gh/{user}/{repo}@{ref}/{path}
```

用浏览器或 `curl` 访问即可触发，返回 JSON 表示清除状态：

```json
{
  "id": "eIdWlP9dFM3qB2TR",
  "status": "finished",
  "paths": {
    "/gh/pengyanai/misc@main/collapsed-tiles/global-ai-availability.stoverride": {
      "throttled": false,
      "providers": { "CF": true, "FY": true }
    }
  }
}
```

> `CF` = Cloudflare, `FY` = Fastly。两个后端都返回 `true` 表示缓存已清除。

批量刷新多个文件可以写个简单脚本：

```bash
for f in file1.js file2.png file3.yaml; do
  curl -s "https://purge.jsdelivr.net/gh/user/repo@main/$f" | jq .status
done
```

---

## 五、实战：Stash 配置中的图标与脚本引用

在 Stash overlay 配置中引用 GitHub 上的图标和脚本，选择 CDN 域名直接影响大陆用户加载速度：

```yaml
# 推荐：使用 fastly 节点
icon: https://fastly.jsdelivr.net/gh/pengyanai/misc@main/collapsed-tiles/img/chatgpt.png

# 备选：国内镜像（大陆最快）
icon: https://jsd.onmicrosoft.cn/gh/pengyanai/misc@main/collapsed-tiles/img/claude.png

# 不推荐：直接用 raw（大陆可能超时）
icon: https://raw.githubusercontent.com/pengyanai/misc/refs/heads/main/collapsed-tiles/img/grok.png
```

同一个仓库、同一个路径，只需替换域名前缀即可在不同 CDN 之间切换。

---

## 六、注意事项

1. **文件大小限制**：jsDelivr 对单文件有 50MB 限制，超过需拆分或换方案
2. **私有仓库不支持**：jsDelivr 只能加速公开仓库
3. **社区镜像稳定性**：`jsd.onmicrosoft.cn` 等非官方镜像由社区维护，可能随时下线，建议保留 `fastly.jsdelivr.net` 作为 fallback
4. **缓存延迟**：push 后需手动 purge 或等待 ~24h，急用时记得调 purge 接口
5. **commit SHA 锁定**：对于代理工具配置等对一致性要求高的场景，建议用 commit SHA 而非分支名

---

## 七、快速参考卡片

```
# 拉取文件（大陆推荐）
https://fastly.jsdelivr.net/gh/{user}/{repo}@{ref}/{path}
https://jsd.onmicrosoft.cn/gh/{user}/{repo}@{ref}/{path}

# 刷新缓存
https://purge.jsdelivr.net/gh/{user}/{repo}@{ref}/{path}

# GitHub 原生（需代理）
https://raw.githubusercontent.com/{user}/{repo}/{ref}/{path}
```
