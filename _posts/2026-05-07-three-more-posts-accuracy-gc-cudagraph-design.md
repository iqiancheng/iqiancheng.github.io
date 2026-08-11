---
layout: post
title: "Design Spec — 训推加速三新篇：精度对齐 / Gradient Checkpointing / CUDA Graph"
date: 2026-05-07 00:00:00 +0800
author: Joseph
categories: [ai-ml]
tags: [training, memory, efficiency, methodology]
---
## 1. 定位

延续训推加速系列（已完成 6 篇：CLI toolkit / GPU SOP / CPU SOP / Qwen3 fusion 识别 / Triton 实战 / 以及其他预处理姊妹篇），新增 3 篇**实战技巧深化**：

- **篇 A**：替换 fused kernel 后的精度对齐实战
- **篇 B**：Qwen3-8B dense 下让 Gradient Checkpointing 收益最大化
- **篇 C**：Qwen3-8B dense 下 CUDA Graph 的合理使用

三篇 **standing example 统一用 Qwen3-8B dense**（H100 单卡 bf16 可验证），追求"相对改进量级"而非绝对数字。

## 2. 共享约定

- HF pastel mermaid 配色（与前序一致）
- 每篇 §零 骨架表
- 术语 / 基础术语首次出现链接回 GPU SOP / CPU SOP / CLI toolkit / Qwen3 fusion 前后篇
- 每篇 §八 权威参考 + 相关文章导航
- 标题简洁，避免长标题

## 3. 篇 A：精度对齐实战

**目标读者**：刚写完自定义 Triton kernel、准备接入训练但担心掉点的人

**核心价值**：给出"替换后怎么验证才算对齐"的完整 SOP

**章节**：

| § | 主题 |
|---|---|
| §零 | 骨架 |
| §一 | 为什么替换后可能掉点（7 源：reduction 顺序 / atomic / dtype cast / ε 位置 / RNG / causal mask / master weight） |
| §二 | 三道 Gate 验证法（gradcheck → bf16 数值对照 → 1000 step loss 曲线） |
| §三 | 事后：业务指标 Gate（Eval loss / MMLU / GSM8K / 长链路生成） |
| §四 | 8 个实用技巧（A/B 开关 / Golden 输入 / 层级 hook / 渐进替换 / 对齐 Liger 等） |
| §五 | 业界标答：抄 Liger Kernel 的 CI 策略 |
| §六 | 权威参考 + 相关文章 |

**字数**：~1800 字；1 张 mermaid（三道 Gate 决策图）

## 4. 篇 B：Gradient Checkpointing 实战

**目标读者**：Qwen3 训练显存爆、想用 gradient checkpointing 换算力但不知道怎么配

**核心价值**：把 gradient checkpointing 从"粗粒度一键开关"升级到"按 AI 值选择性开启"

**章节**：

| § | 主题 |
|---|---|
| §零 | 骨架 |
| §一 | 原理：activation 存 vs 重算 tradeoff；显存节省曲线 + 时间代价曲线 |
| §二 | HF transformers 默认实现的瓶颈（`gradient_checkpointing_enable` 粒度粗） |
| §三 | 粒度选择决策（block-level / sub-block / selective） + mermaid 决策树 |
| §四 | Selective Activation Checkpointing 实战（存贵的算便宜的，和前篇 AI 表呼应） |
| §五 | 与 Flash-Attention / Liger 的协同（避免冲突 / 重复存储） |
| §六 | Qwen3-8B benchmark：memory ↓ vs step time ↑ Pareto 曲线；推荐配置 |
| §七 | 陷阱：RNG state / reentrant vs non-reentrant / BF16 异常 |
| §八 | 权威参考 + 相关文章 |

**字数**：~2200 字；2 张 mermaid（决策树 + 时序图）+ benchmark 表

## 5. 篇 C：CUDA Graph 实战

**目标读者**：训练 / 推理想再挤一点 launch overhead 的人

**核心价值**：明确 CUDA Graph 适用边界，别在动态 shape 场景踩坑

**章节**：

| § | 主题 |
|---|---|
| §零 | 骨架 |
| §一 | 原理：kernel launch 开销是 CUDA Graph 的解决目标 |
| §二 | 适用 / 不适用决策（固定 shape ✓ / 动态 shape、control flow ✗） + mermaid 决策树 |
| §三 | 三档使用姿势（手工 `torch.cuda.graph` / `torch.compile reduce-overhead` / CUDA Graph Trees） |
| §四 | 训练场景：Qwen3-8B + 静态 shape + step time -12~20% 实测 |
| §五 | 推理场景：decode 用 / prefill 不用；vLLM & SGLang 的 CUDA Graph 机制对照 |
| §六 | 陷阱：shape lock-in / host sync / memory pool 静态化 / allocator 警告 |
| §七 | 和 Flash-Attention / Liger 的协同 |
| §八 | 权威参考 + 相关文章 |

**字数**：~2000 字；1 张 mermaid（适用决策）+ 1 张时序对比图

## 6. 实施顺序

1. 篇 A（最独立，依赖前篇最少）
2. 篇 B（依赖 AI 概念，引用篇 1 的 §3.4）
3. 篇 C（独立度高，但和篇 B 的 Qwen3 benchmark 数字可互相印证）

完成后统一更新 README + 前序 6 篇末尾的"相关文章"反链（可选）。

## 7. YAGNI（不做的范围）

- MoE 模型对应的 checkpointing / CUDA Graph 差异（留给未来 MoE 专篇）
- 多卡 DP / FSDP 下 CUDA Graph 协同（只讲单卡）
- CUDA Graph C++ 原生 API（只讲 Python / PyTorch 层）
- 精度对齐的具体数学证明（非论文，保留工程实战视角）

## 8. 风险

- **CUDA Graph Trees 是 PyTorch 2.x 相对新特性**，需要声明版本（PyTorch 2.2+）
- **benchmark 数字**依赖硬件（H100），在文中明示
- **Qwen3 源码路径**随 transformers 版本变，加版本号标注
