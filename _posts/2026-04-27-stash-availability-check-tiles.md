---
layout: post
title: "Stash Availability Check Tiles：一键检测 AI 与流媒体服务可用性"
date: 2026-04-27 00:00:00 +0800
author: Joseph
categories: [networking]
tags: [agent, proxy]
---
切换代理节点后，最常见的问题是"这个节点能用 ChatGPT 吗？Netflix 解锁了吗？"。每次手动打开浏览器逐个测试太麻烦，于是我基于 [StashNetworks/misc](https://github.com/StashNetworks/misc) 的 collapsed-tiles 机制，Vibe Coding 开发了一套覆盖主流 AI 服务和流媒体的一键检测磁贴，安装后直接在 Stash 首页折叠展示各服务的可用状态。

项目地址：[pengyanai/misc](https://github.com/pengyanai/misc)

---

## 一、支持的服务

| AI 服务 | 流媒体 |
|---------|--------|
| ChatGPT Web / App | YouTube |
| Codex (OpenAI) | Netflix |
| Claude | Disney+ |
| Gemini | Spotify |
| Grok | |
| Copilot | |

共 11 个检测项，覆盖日常最常用的 AI 和流媒体服务。

---

## 二、安装方式

打开 Stash → 覆写 → 安装覆写，填入以下任一 URL：

```
# 主线路（Fastly CDN，大陆可直连）
https://fastly.jsdelivr.net/gh/iqiancheng/misc@main/collapsed-tiles/global-ai-availability.stoverride

# 备用1（GitHub Raw，需代理）
https://raw.githubusercontent.com/pengyanai/misc/main/collapsed-tiles/global-ai-availability.stoverride

# 备用2（国内镜像）
https://jsd.onmicrosoft.cn/gh/pengyanai/misc@main/collapsed-tiles/global-ai-availability.stoverride
```

安装后回到首页，即可看到折叠式检测磁贴。点击展开查看各服务状态。

---

## 三、检测结果说明

| 状态 | 含义 |
|------|------|
| `Available (SG)` | 可用，括号内为出口地区国家码 |
| `Available` | 可用，但未能提取地区码 |
| `N/A` / `Not Available` | 当前出口 IP 不支持该服务 |
| `Blocked` | 服务明确拒绝（如 OpenAI 的 `unsupported_country`） |
| `Network Error` | 网络不通或超时 |

---

## 四、检测原理

每个服务对应一个独立的 JS 脚本，通过 Stash 的 `script-providers` 机制执行。核心逻辑是：

1. **发起 HTTP 请求**到目标服务的特定端点
2. **解析响应**判断可用性（状态码、页面关键词、API 返回字段）
3. **提取地区信息**（如果服务返回了地区码）
4. **通过 `$done()` 回传结果**到 Stash 磁贴显示

不同服务的检测策略有所不同：

| 服务 | 检测端点 | 判断方式 |
|------|----------|----------|
| ChatGPT / Codex | `api.openai.com/compliance/cookie_requirements` | 响应中是否包含 `unsupported_country` |
| Claude | `claude.ai` 首页 | 页面是否包含地区限制关键词 |
| Grok | `grok.com` 首页 | 页面关键词 `grok` / `xai` |
| YouTube | `youtube.com/premium` | 解析响应中的 `"GL":"XX"` 地区码 |
| Spotify | `spclient.wg.spotify.com` 注册验证 API | 解析 JSON 中的 `country` 和 `is_country_launched` |
| Netflix | `netflix.com` | HTTP 状态码与页面内容 |
| Disney+ | `disneyplus.com` | 页面是否包含地区限制关键词 |

所有脚本共享相同的 `$httpClient` 封装模式，通过 Stash 的代理策略发起请求，因此检测的是**当前代理出口 IP** 对应的可用性。

---

## 五、stoverride 文件结构

整个配置是一个 YAML 格式的 `.stoverride` 文件，包含两个核心部分：

```yaml
# 磁贴定义
tiles:
  - name: Claude-Check          # 唯一标识
    title: Claude                # 显示名称
    icon: https://...claude.png  # 图标 URL（走 CDN）
    collapsed: true              # 默认折叠
    url: https://claude.ai       # 点击跳转

# 脚本绑定
script-providers:
  Claude-Check:                  # 与 tile name 对应
    url: https://...claude.js    # 检测脚本 URL
```

`tiles` 定义磁贴的外观和交互，`script-providers` 将每个磁贴绑定到对应的检测脚本。图标和脚本都托管在 GitHub 上，通过 jsDelivr CDN 加速分发。

---

## 六、项目目录结构

```
misc/
└── collapsed-tiles/
    ├── global-ai-availability.stoverride   # 主覆写配置
    ├── img/                                 # 服务图标（PNG）
    │   ├── chatgpt.png
    │   ├── claude.png
    │   ├── codex.png
    │   ├── copilot.png
    │   ├── disney.png
    │   ├── gemini.png
    │   ├── grok.png
    │   ├── netflix.png
    │   ├── spotify.png
    │   └── youtube.png
    └── script/                              # 检测脚本（JS）
        ├── chatgpt-web.js
        ├── chatgpt-app.js
        ├── claude.js
        ├── codex.js
        ├── copilot.js
        ├── disney.js
        ├── gemini.js
        ├── grok.js
        ├── netflix.js
        ├── spotify.js
        └── youtube.js
```

---

## 七、相比上游的改进

基于 [StashNetworks/misc](https://github.com/StashNetworks/misc) 的原始 ChatGPT + YouTube + Netflix 三项检测，本项目做了以下扩展：

- **新增 7 项检测**：Codex、Claude、Grok、Gemini、Copilot、Disney+、Spotify
- **改进 YouTube 检测**：通过解析 `"GL":"XX"` 提取出口地区码，而不是简单判断可达性
- **改进 Spotify 检测**：使用注册验证 API 获取 `country` 和 `is_country_launched` 精确判断
- **每个服务独立品牌色**：磁贴背景色与服务品牌色匹配（如 Spotify `#1DB954`、YouTube `#FF0000`）
- **多 CDN 线路**：提供 Fastly、GitHub Raw、国内镜像三条安装线路

---

## 八、扩展自定义检测

如果要新增一个服务检测，只需三步：

1. 在 `script/` 下新建 JS 文件，实现检测逻辑并通过 `$done()` 返回结果
2. 在 `img/` 下放入对应服务图标
3. 在 `.stoverride` 的 `tiles` 和 `script-providers` 中各添加一项

检测脚本模板：

```javascript
async function request(method, params) {
  return new Promise((resolve, reject) => {
    $httpClient[method.toLowerCase()](params, (error, response, data) => {
      resolve({ error, response, data });
    });
  });
}

async function main() {
  const { error, response, data } = await request("GET", "https://example.com");
  if (error) {
    $done({ content: "Network Error", backgroundColor: "" });
    return;
  }
  // 根据 response/data 判断可用性
  $done({ content: "Available", backgroundColor: "#HEXCOLOR" });
}

(async () => { main().catch(() => $done({})); })();
```

Push 到 GitHub 后，通过 `purge.jsdelivr.net` 刷新 CDN 缓存即可生效。
