---
layout: post
title: "Cursor Free Plan 接入第三方 BlueRouter 网关调研（WIP）"
date: 2026-05-07 00:00:00 +0800
author: Joseph
categories: [networking]
tags: [agent, proxy, methodology]
---
> **状态**：WIP。这篇是调研笔记，不是完整教程。结论很快就到：**Free Plan 能配置，但 Agent / Composer / Edit 都跑不动，只有 Ask 聊天模式可用**。下面把坑列清楚，给有同样诉求的同学少踩几次。
>
> 相关背景：本站 [Claude Code CLI 第三方网关](/posts/claude-code-cli-third-party-gateway/)、[Codex CLI 第三方网关](/posts/codex-cli-third-party-gateway/)。
>
> 本文所有"网关域名"都用 `https://gateway.example.com` 作占位；实际替换成你的 BlueRouter 部署地址。

---

## TL;DR（先看结论）

| 能力 | Free Plan + BYOK | Pro+ Plan + BYOK |
|------|:---:|:---:|
| Ask / Chat 模式 | ✅ 可用 | ✅ 可用 |
| Agent 模式 | ❌ 后端拦截 | ⚠️ 可用，但有已知 bug |
| Composer / Edit | ❌ 后端拦截 | ⚠️ 可用，但有已知 bug |
| Tab 补全 | ❌（走 Cursor 官方模型） | ❌（走 Cursor 官方模型） |
| Subagents | ❌ 忽略 BYOK，按 Cursor 计费 | ❌ 忽略 BYOK，按 Cursor 计费 |
| 图片 / Vision | ❌ Unauthorized | ❌ Unauthorized（已知 bug） |

**一句话**：如果你只想在 Cursor 里"用 BlueRouter 聊聊天"，Free Plan 够了；如果你要 Agent / Composer 这些 Cursor 真正的卖点，**必须升级到 Pro 才能让 BYOK 生效**，而且还得忍受若干已知 bug。

---

## 一、背景：为什么想走 BlueRouter？

BlueRouter（或任何 OpenAI Compatible Gateway：LiteLLM / OneAPI / Portkey / 自建 vLLM）的诉求基本都一样：

- **统一账单**：一个 Key 背后聚合 OpenAI / Anthropic / Google / DeepSeek / 本地 vLLM
- **审计与脱敏**：企业环境下能加 header、日志、PII 过滤
- **就近路由**：国内网络走本地代理，避开境外直连的不稳定
- **省钱 / 切换**：用 BlueRouter 后端偷偷换成 DeepSeek 或开源模型，省 10 倍 token 成本

Cursor 的 BYOK 入口是 `Settings → Models → API Keys` 里的两个字段：

```
OpenAI API Key:          <你的 BlueRouter Key>
Override OpenAI Base URL: https://gateway.example.com/v1
```

看起来很美好，实际一上手全是坑。下面按顺序拆。

---

## 二、前置：BlueRouter 必须暴露 OpenAI 兼容接口

Cursor 的 BYOK 只有"OpenAI 协议"这一条路——**没有 Anthropic Base URL 覆盖选项**（社区已提了 Feature Request，目前未支持）。

所以 BlueRouter 必须提供：

```
POST https://gateway.example.com/v1/chat/completions
POST https://gateway.example.com/v1/models   # Cursor 探测模型列表时会打
```

先用 curl 冒烟：

```bash
curl -sX POST https://gateway.example.com/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_BLUEROUTER_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role":"user","content":"say OK"}],
    "max_tokens": 20
  }'
```

期望 200 + 标准 OpenAI Chat Completions 响应。curl 不通就别配 Cursor 了，先修网关。

---

## 三、Cursor UI 配置步骤（Free / Pro 通用）

1. 打开 Cursor，`Cmd+Shift+J` → 左侧选 **Models**
2. 滚到底部 **API Keys** 区块
3. 找到 **OpenAI API Key** 行：
   - 勾选 `Override OpenAI Base URL`，填 `https://gateway.example.com/v1`
   - `OpenAI API Key` 填 BlueRouter 签发的 Key
   - 点 **Verify**（Cursor 会 POST `/models` 探活）
4. 在上方 **Model Names** 区块，点 `+ Add model`，手动录入 BlueRouter 后端能路由的模型名，例如：
   - `claude-sonnet-4-6`
   - `gpt-5-5`
   - `deepseek-v4-pro`
5. 保存，回到编辑器，模型下拉里会多出这些 BYOK 模型（通常带小标签标识）

> **Windows 坑**：3.1.14 版本在 Windows 上 "Add model" 后模型瞬间消失，Mac 同账号正常。目前无解，等官方修。

---

## 四、Free Plan 的后端拦截（核心坑）

配置正确的情况下，Free Plan 切到 Agent / Composer / Edit 时，会收到：

```
Agent and Edit rely on custom models that require a paid subscription
```

这不是 bug，是 **by design**。Cursor staff 2026-04-26 在 forum 上明确答复：

> *Hey, the config is set up correctly, but on the Free plan, BYOK (your own API key via Override OpenAI Base URL) doesn't work for Agent and Composer/Edit. This is a by-design limitation, not a bug.*
> — Dean Rie, Cursor Team
> 来源: [External Models Setup, forum.cursor.com](https://forum.cursor.com/t/external-models-setup)

### Free Plan 实际能用的场景

- **Ask / Chat 面板**：右侧侧边栏的"提问"模式，模型下拉选你加的 BYOK 模型，能发消息，能流式返回
- **仅此而已**

Tab 补全走的是 Cursor 自家的 Cursor Tab 模型（Free 额度有限），**BYOK 管不到 Tab**，这条路走不通。

---

## 五、Pro Plan + BYOK 的三个已知坑

升级到 Pro 后 BYOK 能用在 Agent/Composer，但以下坑近期都还在 linear-tracked：

### 5.1 Responses API 格式打到 /chat/completions 端点

Cursor 2.5+ 的 Agent 把 **OpenAI Responses API** 格式的 payload 发到了 `/chat/completions`：

```json
// Cursor 实际发出的 body（错误！这是 Responses API 格式）
{
  "input": [...],                     // 应该是 messages
  "tools": [{"type":"function","name":"..."}],  // 应该嵌套 {"function":{"name":"..."}}
  "reasoning": {"effort": "high"},    // 应该是 reasoning_effort
  "text": {"format": {...}},          // 应该是 response_format
  "store": true, "include": [...], "previous_response_id": "..."
}
```

LiteLLM / Azure OpenAI / vLLM 等标准实现收到这种 payload 直接 400。

**BlueRouter 这边的对策**：
- 在网关里做一层 **Responses → Chat Completions 翻译中间件**，把上述字段改写回标准格式
- 或者直接支持 Responses API 路径（`POST /v1/responses`），并在 Cursor 里把 base URL 写成 `https://gateway.example.com`（不带 `/v1`）让它自动探测

### 5.2 图片 / Vision 全挂

附件里加任何图，BYOK + custom base URL 一律返回：

```
Unauthorized User Openai API key
```

纯文本聊天正常。已 linear-linked，等修。绕不过去，只能纯文本对话。

### 5.3 Subagents 忽略 BYOK

`/multitask`、`@agent` 这些 subagents 始终走 Cursor 官方路径，**即使主 agent 走 BYOK，子 agent 也会从你的 Pro 额度扣费**，不会打到 BlueRouter。

这条对省钱场景影响大：你以为切了 BYOK 成本清零了，实际 subagents 还在啃 Pro 额度。

---

## 六、Anthropic 原生协议不支持

社区 FR：[Missing Anthropic base URL override in Cursor (BYOK)](https://forum.cursor.com/)（2026-04-23 提交）。

目前想用 **Anthropic `/v1/messages` 协议**对接 BlueRouter：

- ❌ 不支持。Cursor 写死只看 `OPENAI_BASE_URL` 这一条路径
- ✅ 变通：让 BlueRouter 同时暴露 OpenAI Chat Completions 兼容端点（把 Anthropic 请求内部翻译成 OpenAI 格式），Cursor 那边还是填 `/v1/chat/completions`

---

## 七、替代方案比较

如果你执意在 Cursor 壳子里用 BlueRouter，以下是决策树：

```
是不是 Pro+ 用户？
├── 否 → BYOK 只能 Ask 模式用。考虑：
│   ├── (A) 换 VS Code + Cline/Continue 扩展：对 BYOK 支持完整
│   ├── (B) 升级 Pro（$20/mo），接受已知 bug
│   └── (C) 不折腾，直接用 Cursor 官方 $20 额度
└── 是 → 网关必须做 Responses API 翻译层
    ├── (A) LiteLLM 前置 monkey-patch（见 forum 方案）
    ├── (B) BlueRouter 自己加翻译中间件
    └── (C) 接受 Cursor 官方订阅的 subagents 扣费漏洞
```

个人推荐：**如果你的核心诉求是"用自己的模型账户 + 完整 Agent 能力"，短期内 Cursor 不是最佳容器**。VS Code + [Cline](https://github.com/cline/cline) / [Continue.dev](https://www.continue.dev/) 对 BYOK 的支持完整得多，Anthropic / OpenAI / 自定义 base URL 都原生支持，且不区分 Free / Pro。Cursor 的价值主要是 Composer + Tab，而这两个都不吃 BYOK。

---

## 八、BlueRouter 实际接入 checklist（给未来的自己）

- [ ] 网关暴露 `POST /v1/chat/completions` + `GET /v1/models`
- [ ] `/models` 返回 JSON Schema 和 OpenAI 一致：`{"data":[{"id":"...","object":"model"}]}`
- [ ] 实现 Responses API → Chat Completions 翻译中间件（覆盖 `input`、`tools`、`reasoning`、`text`、`store`、`include` 等字段）
- [ ] 日志打开 HTTP body dump，方便抓 Cursor 实际请求格式变化
- [ ] 本地用 `curl` 手验 5 秒能通
- [ ] 在 Cursor 里先只测 Ask 模式；Agent 测试放到 Pro 账号再做
- [ ] 监控网关错误率：Cursor Agent 偶尔会发 `{"type":"custom","name":"ApplyPatch"}` 这种非标准 tool，要么透传要么 400

---

## 九、参考资料

- [Cursor Models & Pricing](https://cursor.com/docs/models-and-pricing)（官方定价与使用池）
- [External Models Setup – forum.cursor.com](https://forum.cursor.com/t/external-models-setup)（Free Plan BYOK 限制的官方确认）
- [Cursor Agent sends Responses API format to /chat/completions endpoint](https://forum.cursor.com/t/cursor-agent-sends-responses-api-format-to-chat-completions-endpoint)（Responses API payload mismatch bug）
- [Images/vision completely broken with OpenAI BYOK + custom endpoint override](https://forum.cursor.com/)（Vision BYOK bug）
- [Subagents ignore user's own API key](https://forum.cursor.com/)（Subagents 计费漏洞）
- [Cline (VS Code 扩展)](https://github.com/cline/cline)、[Continue.dev](https://www.continue.dev/)（BYOK 替代容器）

---

## 十、TODO（下一版补）

- [ ] 实测 BlueRouter 的 Responses API 翻译中间件代码（Python + FastAPI 骨架）
- [ ] 在 Pro 账号下录一段 Agent 模式走 BYOK 的完整请求链（HAR + 截图）
- [ ] 测试 `Cursor CLI`（`cursor-agent`）能不能绕过 IDE 限制跑 BYOK
- [ ] 比较 Cline 在相同场景下的稳定性
- [ ] 补充 Azure OpenAI / vLLM / Ollama 三种后端在 BlueRouter 里的配置片段

> 如果你在实际接入中踩到了本文没覆盖的坑，欢迎邮件或 issue 反馈——这篇会持续更新。
