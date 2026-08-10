---
layout: post
title: "Design Spec — 训推加速系列：看懂 Qwen3 + Fusion 识别 + Triton 实战"
date: 2026-05-07 00:00:00 +0800
author: Joseph
tags: [qwen, triton, kernels, methodology]
---

## 1. 系列定位

两篇姊妹篇，延续训推加速 playbook：

- 前 4 篇：CLI toolkit / GPU SOP / CPU SOP / Speech 三部曲
- **本系列新增 2 篇**，把"训推加速"从"定位问题 + 调参"推进到"理解模型结构 + 自己写 kernel"

两篇均以 **Qwen3 系列**为 standing example（当前最主流开源中文模型）。

## 2. 读者画像

- 已熟悉 `torch.profiler` / `nsys` 基础（看过姊妹 GPU/CPU SOP）
- 想进阶到"看懂模型 → 识别 fusion 机会 → 用或写 Triton kernel"
- 职业：训练 / 推理 / infra 工程师；非研究员

前置知识用 footnote 链接前文，不重讲。

## 3. 篇 1：《看懂 Qwen3 + 识别算子融合机会》

**字数目标**：~2200 字
**核心价值**：教读者用静态（源码 / FX）+ 动态（profile）双管齐下识别 fusion 机会

### 章节

| 章节 | 内容 |
|---|---|
| §零 | 本文骨架表 |
| §一 | 引子：看懂 → 看出机会 → 动手换 |
| §二 静态读 | 2.1 `print(model)` / torchinfo · 2.2 transformers modeling_qwen3.py 源码路径 · 2.3 `torch.fx.symbolic_trace` |
| §三 动态读 | 3.1 `torch.profiler` 采 trace · 3.2 Perfetto 时间线 · 3.3 launch overhead 量化 |
| §四 Fusion 机会的 4 信号 | 小 kernel 密集 / 连续 memory-bound / 重复 pattern / 逻辑本可一步 |
| §五 Qwen3 经典可融合点 | RMSNorm / RoPE / SwiGLU / QKV 合并 / Attention softmax |
| §六 自动 Fusion | torch.compile + Inductor · `TORCH_LOGS=output_code` 看 Triton · before/after benchmark |
| §七 | 自动 fusion 不够时 → 预告篇 2 |
| §八 | 参考资料 |

### 可视化

- mermaid 1：Qwen3 层级图（embedding → N×block → lm_head）
- mermaid 2：Fusion 机会识别决策树（4 信号）
- 外链图：若有高质量可访问的 transformer arch 图则插入，否则不强求

## 4. 篇 2：《从"调包"到"手写"：Triton Kernel 实战》

**字数目标**：~2600 字（含 Backward kernel 增量）
**核心价值**：
- 50% Survey：现成轮子 + 选型决策表
- 50% 实战：Forward kernel → Backward kernel → 集成 → 验证

Backward kernel 是差异化亮点（多数教程只写 forward）。

### 章节

| 章节 | 内容 |
|---|---|
| §零 | 本文骨架表 |
| §一 | 接续篇 1 |
| §二 Survey | 2.1 Flash-Attention · 2.2 Liger Kernel (Qwen/Llama 家族) · 2.3 Unsloth · 2.4 xFormers · 2.5 Apex · 2.6 torch.compile Inductor · 2.7 选型决策表 |
| §三 Triton 一分钟 | 3.1 Triton vs CUDA · 3.2 核心概念 `program_id` / `load` / `store` / `BLOCK_SIZE` |
| §四 Forward kernel | 手写 Qwen3 RMSNorm forward（~50 行 Triton）+ 对照 torch 原生 benchmark |
| §五 **Backward kernel（重头戏）** | 5.1 为什么 backward 难（saved tensors / reduction 顺序 / grad routing） · 5.2 两种策略对比 · 5.3 RMSNorm backward 数学推导 · 5.4 Triton backward kernel 代码（~60 行） · 5.5 `torch.autograd.Function` 封装 · 5.6 正确性验证 + gotcha 表 |
| §六 集成进 Qwen3 训练 | 6.1 替换 RMSNormLayer · 6.2 baseline vs fwd-only vs fwd+bwd 三档 benchmark · 6.3 数值一致性校验 |
| §七 Triton 调优入门 | BLOCK_SIZE / `triton.autotune` / 常见坑（dtype / reduction order / num_warps / shared memory） |
| §八 | 参考资料 |

### 可视化

- mermaid：决策树"先调包后手写"
- 两段完整 Triton 代码块（forward + backward）
- 可能的外链：Liger Kernel 架构图 / Flash-Attention 示意（可访问性验证）

## 5. 两篇共用约定

- 所有 mermaid 使用 HF pastel 配色（与姊妹篇视觉一致）
- 代码 block 标明 language（python / bash / mermaid）
- 术语首次出现链接到 GPU SOP / CPU SOP / CLI toolkit 对应章节
- 每章末尾"权威参考" 3~5 条
- 末尾"相关文章"挂整个训推加速系列导航
- 标题避免过长（参考前面 post 改标题后的节奏）

## 6. 实施顺序

1. 写篇 1 → 本地预览 → 修 mermaid 语法 → commit
2. 写篇 2 → 本地预览 → 代码块测试 → commit
3. 更新 README 索引 + 前文"相关文章"节（反向链接）
4. 统一 push

## 7. 风险 & 注意

- **Triton 版本兼容**：示例代码需基于 Triton 2.2+ / PyTorch 2.3+ 才能跑，前面加版本声明
- **Qwen3 repo 源码路径可能变**：引用时加 commit hash 或 transformers 版本号
- **Backward 数学推导不能太深**：这是博客不是 paper，控制在 3 行公式 + 直觉解释
- **Benchmark 数字**：如果文章里给具体数字，要注明硬件型号（否则可比性差）；倾向给"相对比例"而非绝对数

## 8. 不做的范围（YAGNI）

- MoE 模型的 fusion（Qwen3-MoE 暂不覆盖，留给未来单独一篇）
- Attention kernel 深度实现（留给 Flash-Attention 专篇）
- CUDA 级别的编程（只讲 Triton）
- Inference 场景的 fusion（专注训练，与用户需求一致）
- Distributed 下的 fusion（单卡视角即可）
