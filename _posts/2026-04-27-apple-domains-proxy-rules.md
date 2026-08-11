---
layout: post
title: "Apple 子域名功能梳理与代理规则配置指南"
date: 2026-04-27 00:00:00 +0800
author: Joseph
categories: [networking]
tags: [networking, proxy]
---
Apple 生态中大量系统级服务依赖特定子域名通信，了解这些域名的功能归属，有助于在 Stash / Surge 等代理工具中精准编写分流规则，避免误拦截或漏放。本文按服务分组梳理关键域名及其用途，并给出可直接使用的规则片段。

---

## 一、GeoServices 地理服务 (`ls.apple.com`)

地理服务是 Apple 地图、天气、新闻、Spotlight 等多个系统功能的底层依赖，所有请求都走 `*.ls.apple.com`。

| 域名 | 路径 | 功能 |
|------|------|------|
| `gspe1-ssl.ls.apple.com` | `/pep/gcc` | 基于网络的地区检测 (GCC)，被地图/新闻/Spotlight/Watch 等共用 |
| `configuration.ls.apple.com` | `/config/defaults` | 设备信息检测与配置下发 |
| `gspe35-ssl.ls.apple.com` | `/config/announcements` | 地图公告配置 |
| `gspe35-ssl.ls.apple.com` | `/geo_manifest/dynamic/config` | 地图动态配置清单 |
| `gsp-ssl.ls.apple.com` | `/dispatcher.arpc` | 地图调度服务（国际版） |
| `gsp-ssl.ls.apple.com` | `/directions.arpc` | 地图导航/方向服务（国际版） |
| `gspe19-ssl.ls.apple.com` | `/tile.vf` | 地图瓦片（国际版） |
| `gspe19-cn-ssl.ls.apple.com` | `/tiles` | 地图瓦片（高德/中国版） |
| `gspe12-ssl.ls.apple.com` | `/traffic` | 交通流量（国际版） |
| `gspe12-cn-ssl.ls.apple.com` | `/traffic` | 交通流量（高德/中国版） |

> 中国版地图域名（`gspe19-cn-ssl`、`gspe12-cn-ssl`）通常不需要代理，按需添加即可。

---

## 二、Siri & Spotlight (`smoot.apple.com`)

Siri 建议与 Spotlight 搜索的服务端通信均走 `smoot.apple.com`，新闻小组件的内容也由此域名提供而非 `news-*`。

| 域名 | 路径 | 功能 |
|------|------|------|
| `api.smoot.apple.com` | `/bag` | Siri 建议配置下发 |
| `api*.smoot.apple.com` | — | Siri 建议服务（含新闻小组件内容） |
| `*.smoot.apple.com` | `/bag` | Siri 建议区域设置刷新 |
| `guzzoni.smoot.apple.com` | — | Siri 请求连接（iOS 15+） |
| `guzzoni.apple.com` | — | 询问 Siri 连接（旧版） |

---

## 三、天气 (`weather-data.apple.com`)

| 域名 | 功能 |
|------|------|
| `weather-data.apple.com` | iOS 天气 APP、macOS 天气小组件、地图内天气、部分 Watch |
| `weather-data-origin.apple.com` | iOS 天气小组件、天气 APP 的回退查询 |

> **注意**：不要对 iOS 15–17 天气 app 使用的 `weather-data.apple.com` 做 MitM，会导致功能异常。

---

## 四、TV (`itunes.apple.com`)

| 域名 | 路径 | 功能 |
|------|------|------|
| `uts-api.itunes.apple.com` | `/uts/v3/configitions` | TV app 配置 |

---

## 五、News 新闻 (`news-*.apple.com`)

新闻服务使用 `news-*.apple.com` 系列域名。但值得注意的是，新闻小组件的内容实际由 `api*.smoot.apple.com` 提供，`news-*` 仅服务于 News app 本体。

---
## 六、TestFlight

| 域名 | 功能 |
|------|------|
| `testflight.apple.com` | TestFlight 连接 |

---

## 七、iCloud 专用代理 / Private Relay

| 域名 | 路径 | 功能 |
|------|------|------|
| `doh.dns.apple.com` | `/dns-query` | DoH (DNS over HTTPS) 查询 |

---

## 八、Stash / Surge 代理规则汇总

以下规则按服务分组，可直接用于 Stash 或 Surge 的 `[Rule]` 段：

```yaml
# GeoServices 地理服务
DOMAIN-SUFFIX,ls.apple.com,Proxy

# Siri & Spotlight
DOMAIN-SUFFIX,smoot.apple.com,Proxy
DOMAIN,guzzoni.apple.com,Proxy

# Weather 天气
DOMAIN,weather-data.apple.com,Proxy
DOMAIN,weather-data-origin.apple.com,Proxy

# TV
DOMAIN,uts-api.itunes.apple.com,Proxy

# News
DOMAIN-KEYWORD,news-,Proxy

# TestFlight
DOMAIN,testflight.apple.com,Proxy

# iCloud Private Relay (DoH)
DOMAIN,doh.dns.apple.com,Proxy
```

> 将 `Proxy` 替换为你实际使用的策略组名称。中国版地图域名（`*-cn-ssl.ls.apple.com`）通常走直连，无需代理。

---

## 参考资料

- [NSRingo 项目文档](https://NSRingo.github.io/) — 本文域名信息的主要整理来源
- [Apple - 使用 Apple 产品时需要的网络连接](https://support.apple.com/zh-cn/HT210060)
- [WeatherKit REST API - Apple Developer](https://developer.apple.com/weatherkit/)
- [iCloud 专用代理网络准备 - Apple Developer](https://developer.apple.com/support/prepare-your-network-for-icloud-private-relay/)
