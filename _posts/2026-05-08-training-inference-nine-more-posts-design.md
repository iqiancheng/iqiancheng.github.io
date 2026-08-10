---
layout: post
title: "Design Spec — 训推加速系列深化 9 篇（2026-05-08）"
date: 2026-05-08 00:00:00 +0800
author: Joseph
math: true
tags: [inference, training, methodology]
---

---- 作者：Austin
- 状态：draft（outlines approved，待实施）
- 关联：基于 [训推加速技术地图 post](/posts/training-inference-acceleration-map/) 的各行展开深化

## 0. 共享约定（所有 9 篇通用）

- **Standing example**：能用 Qwen3.5 / 3.6 / DeepSeek-V4 等 2025-2026 SOTA 举例
- **HF pastel mermaid**（与姊妹篇视觉一致）：黄 / 蓝 / 绿 / 粉四色
- **图文规范**：每张 `![](url)` 后加**两个空格**换行再写图注；外链 `curl -sI` + 浏览器 `naturalWidth > 0` 双验证
- **时效声明**：每篇顶部注明"最后更新 2026-05-08"
- **数学公式**：必要处用 MathJax `$$...$$`；集合基数用 `\lvert...\rvert`（避免 `#` 字符）
- **Highlight 原则**：每个 table 行加粗 ≤ 3 核心技术 + ≤ 3 代表工具
- **字数目标**：每篇 2000~2500 字（地图篇深化，非入门篇）
- **相关文章**：每篇末尾链回训推加速系列导航 + 本篇的直接上下文

## 1. 篇 G：音乐生成加速（Suno / MusicLM / UniAudio）

**定位**：区别于 TTS 的"歌曲 / 音乐"生成加速栈；2024~2026 这条路线独立成型。

**章节**：
| § | 主题 |
|---|---|
| §一 | 为什么音乐生成和 TTS 不是一回事（旋律 / 伴奏 / 人声多轨） |
| §二 | 两大范式：Audio Token AR（MusicGen / UniAudio / Suno）vs Latent Diffusion（Stable Audio） |
| §三 | 核心基建：高保真 codec（DAC / EnCodec high-rate / BigCodec）|
| §四 | Suno / Udio 路线推测：Audio LLM + RVQ + 长序列训练 |
| §五 | MusicGen / AudioGen / UniAudio 开源方案拆解 |
| §六 | Stable Audio Open 2.0 / AudioLDM2 / Riffusion |
| §七 | 加速专项：长序列注意力 / Speculative Audio Token / 多轨并行生成 |
| §八 | 评测：FAD / CLAP score / 主观 Elo |
| §九 | 权威参考 |

**可视化**：
- mermaid：MusicGen 架构 + codec 层级
- 外链图：MusicGen repo 架构图、Stable Audio overview
- 数学：RVQ loss、FAD 距离公式、long sequence complexity

## 2. 篇 H：视频 & 3D 生成 Diffusion 加速（独立于图像）

**定位**：把视频时间维度和 3D 空间维度的独特挑战拎出来单讲。

**章节**：
| § | 主题 |
|---|---|
| §一 | 视频 ≠ 图像的加速挑战（时间一致性 / attention 立方复杂度 / 显存 N × H × W） |
| §二 | 视频 DiT 的新基建：Wan 2.2 / HunyuanVideo / CogVideoX 架构对比 |
| §三 | **可训练稀疏 Attention**（VSA / SLA / Sliding Window）—— 视频的 2024 关键突破 |
| §四 | Temporal Caching：TeaCache / FBCache / DeepCache 对视频特有用法 |
| §五 | 步数蒸馏：FastVideo / FastWan / 视频版 DMD2 |
| §六 | 3D 生成加速（3D Gaussian Splatting / DreamGaussian / Triplane Diffusion） |
| §七 | 端到端 gen-to-video 管线（FLUX → HunyuanVideo → Wan 2.2） |
| §八 | 评测：FVD / VBench / 主观评分 |
| §九 | 权威参考 |

**可视化**：
- mermaid：视频 DiT 时空 attention 稀疏化示意
- 外链：HunyuanVideo / Wan 2.2 架构图
- 数学：时空 attention 复杂度 O(T²·H²·W²)、VSA sparse ratio 推导

## 3. 篇 I：Robotic VLA 训推加速

**定位**：具身智能 / 机器人 VLA（Vision-Language-Action）的独特加速问题——**实时性 50Hz + 精细操作 + sim2real**。

**章节**：
| § | 主题 |
|---|---|
| §一 | VLA 是什么：视觉 + 语言 + 动作的端到端 |
| §二 | 主流模型：OpenVLA / RT-2 / π0 / RDT-1B / GR00T |
| §三 | VLA 的推理延迟预算（20~50Hz = 20~50ms per action） |
| §四 | 视觉 encoder 加速（SigLIP/DINOv3 INT8 / token pruning） |
| §五 | Action head 轻量化（Flow Matching vs AR token vs 回归头） |
| §六 | 仿真 + 真机的加速：sim2real 一致性保证 |
| §七 | 端侧 VLA：Jetson Thor / 端侧 NPU 部署 |
| §八 | 评测：LIBERO / CALVIN / SimplerEnv / 真机成功率 |
| §九 | 权威参考 |

**可视化**：
- mermaid：VLA 推理 gantt（视觉 + LLM + action 各占多少 ms）
- sequence：闭环控制（传感器 → VLA → 执行器）
- 数学：控制频率 vs 推理延迟约束
- 外链：OpenVLA / π0 架构图（HuggingFace repo）

## 4. 篇 J：图像生成 Diffusion 训推加速（主线深化）

**定位**：和篇 H 区分——**这篇专注图像**（FLUX / SD3 / SDXL / Qwen-Image），篇 H 专注视频 / 3D。

**章节**：
| § | 主题 |
|---|---|
| §一 | Diffusion 训推 2024~2026 时间线 |
| §二 | 三条主线：步数蒸馏（DMD2） + 架构（DiT / MMDiT） + 低精度（FP8 Attention） |
| §三 | Flow Matching 基础：FLUX 为什么快（Rectified Flow + 优质 tokenizer） |
| §四 | DMD2 深度拆解：数学 + 训练 pipeline |
| §五 | Attention 量化：SageAttention vs SVDQuant vs Q-Diffusion |
| §六 | Feature Caching：DeepCache / TeaCache / FBCache 在图像上的经验 |
| §七 | 工程化：ComfyUI / Diffusers / xDiT 部署 |
| §八 | 评测：FID / CLIP score / 主观 Arena |
| §九 | 权威参考 |

**可视化**：
- mermaid：DMD2 训练 pipeline
- 外链：FLUX 官方图、SD3 MMDiT 架构
- 数学：Flow Matching ODE、DMD2 distribution matching loss

## 5. 篇 K：MoE 教师 → Dense 学生 蒸馏范式

**定位**：这是 2024~2026 开源社区最重要的"落地"范式——Qwen3-MoE / DeepSeek-V3 先训练到 SOTA，再蒸馏成消费级 dense。

**章节**：
| § | 主题 |
|---|---|
| §一 | 为什么要蒸馏：MoE 训得好但部署贵、Dense 部署便宜但训不过 MoE |
| §二 | 知识蒸馏基础：Soft label / Hard label / Feature map / Logit matching |
| §三 | On-Policy vs Off-Policy KD：学生生成 vs 教师生成 |
| §四 | Top-K Logit 蒸馏的工程优化（省带宽 + 省磁盘） |
| §五 | Rejection Sampling SFT + 合成数据 pipeline |
| §六 | Teacher 批量 rollout：用 vLLM / SGLang 推理教师 |
| §七 | 具体案例：Qwen3.5-MoE → Qwen3.5-8B-Dense 蒸馏路径推测 |
| §八 | 蒸馏效果评估：能否保留 MoE 教师的推理 / 代码能力 |
| §九 | 权威参考 |

**可视化**：
- mermaid：蒸馏 pipeline sequence（Teacher 生成 → filter → Student SFT → RLHF）
- gantt：三阶段蒸馏时间线
- 数学：KL loss、Top-K logit、RejectionSampling 接受率

## 6. 篇 L：Qwen3.5 / 3.6 MoE 训练加速

**定位**：深入讲 **MoE 训练** 栈——假设 Qwen3.5 / 3.6 MoE 已发布或即将发布，复盘工业级 MoE 训练的真实瓶颈。

**章节**：
| § | 主题 |
|---|---|
| §一 | MoE 训练的 5 大痛点：通信 / 负载均衡 / 显存 / 路由数值 / 收敛 |
| §二 | DeepEP：Expert Parallel 通信优化原理 |
| §三 | Grouped GeMM：多 expert GeMM 的 kernel 级融合 |
| §四 | Aux-Loss-Free Load Balance：DeepSeek-V3 范式 |
| §五 | Expert 量化 + Offload |
| §六 | All-to-all / computation overlap |
| §七 | Qwen3.5 MoE 推测的训练配方（对比 DeepSeek-V3 / Mixtral） |
| §八 | 调参清单：`capacity_factor` / `top_k` / `router_z_loss` |
| §九 | 权威参考 |

**可视化**：
- mermaid：MoE 训练一步的通信 sequence（all-to-all x2）
- gantt：DP × TP × EP 组合下的时间线
- 数学：router 熵约束、aux loss、capacity factor

## 7. 篇 M：端侧多模态端到端加速

**定位**：端侧 + 多模态 + 端到端的**三重约束**下如何落地（Qwen2.5-Omni 手机端 / MiniCPM-o / Gemma 3n）。

**章节**：
| § | 主题 |
|---|---|
| §一 | 端侧多模态的 3 个硬约束：< 3B 参数 + < 5GB 显存 + 实时性 |
| §二 | 视觉侧：SigLIP / DINOv3 INT8 token pruning |
| §三 | 语音侧：Mimi / XCodec 双流 tokenizer 在端侧 |
| §四 | LLM 侧：端侧 3B 以下模型 + INT4 量化 |
| §五 | 多模态 fusion 策略（Q-Former / LLaVA-style / Interleave） |
| §六 | NPU 调度：Apple ANE / 高通 Hexagon / 联发科 APU |
| §七 | 框架选择：MLX / MLC-LLM / ExecuTorch / llama.cpp 多模态扩展 |
| §八 | 案例：MiniCPM-o 在 iPhone 上跑的拆解 |
| §九 | 权威参考 |

**可视化**：
- mermaid：端侧多模态 pipeline（视觉 + 语音 + LLM + TTS）
- gantt：实时性 budget 分解（每个子模块 ms）
- 数学：端侧 KV cache 显存公式 / NPU 吞吐限制
- 外链：MiniCPM-o / Gemma-3n 架构图

## 8. 篇 N：RL 训练基于 Qwen3.5 + vLLM 加速

**定位**：RL 训练里 **rollout 阶段** 占 > 70% 时间——vLLM 成为核心组件。veRL / OpenRLHF / AReal 都用。

**章节**：
| § | 主题 |
|---|---|
| §一 | RL 训练 pipeline 分析：Actor / Critic / Reward / Rollout |
| §二 | 为什么 rollout 是瓶颈（长输出 + 多样本 + 反复推理） |
| §三 | vLLM 作为 rollout engine 的优势 |
| §四 | Rollout-Train 解耦：异步 / 共享权重 / CPU offload |
| §五 | GRPO / DAPO / GSPO 的 vLLM 实战 |
| §六 | Qwen3.5 + veRL 端到端配置（训练 + 推理 + 奖励模型） |
| §七 | 常见陷阱：actor-learner 版本不同步 / KV cache 复用 / reward hack |
| §八 | 评测：sample efficiency / KL budget |
| §九 | 权威参考 |

**可视化**：
- mermaid sequence：Actor / Learner / Rollout / Reward 异步数据流
- gantt：rollout-train 时间线
- 数学：GRPO objective、group-relative advantage、KL penalty
- 外链：veRL / OpenRLHF 架构图

## 9. 篇 O：解码优化专题（Speculative / EAGLE-3 / Chunked Prefill）

**定位**：把技术地图里"解码优化"行展开——从 Speculative Decoding 家族到 Chunked Prefill + Decode 混批。

**章节**：
| § | 主题 |
|---|---|
| §一 | 解码阶段的本质瓶颈：memory-bound 而非 compute-bound |
| §二 | Speculative Decoding 三代：Google 原版 / Medusa / EAGLE |
| §三 | EAGLE-3 深度拆解：为什么比 EAGLE-2 又快 30~50% |
| §四 | DFlash / PLD / SpecInfer / Ouroboros 各变体 |
| §五 | Speculative Decoding 的数学：接受率 / 期望加速比 |
| §六 | Chunked Prefill + Decode 混批（SGLang / TRT-LLM 主流） |
| §七 | KV cache 优化（复用 / 量化 / 驱逐）对 decode 的影响 |
| §八 | 评测：TPOT / ITL 分布 |
| §九 | 权威参考 |

**可视化**：
- mermaid sequence：Speculative 流程（Draft model 生 N token → Target verify）
- gantt：prefill vs decode 时序对比
- 数学：Speculative 期望加速比 $E[\text{speedup}]$ = $\sum p_i \cdot i$、接受率公式
- 外链：EAGLE-3 架构图

## 10. 实施顺序建议

**按优先级 / 社区热度**：

1. **篇 O（解码优化）** —— 覆盖所有推理场景，工程价值最高
2. **篇 K（MoE→Dense 蒸馏）** —— 开源社区最热门范式
3. **篇 J（图像 Diffusion 加速）** —— 用户基数最大
4. **篇 N（RL + vLLM）** —— o1 系 reasoning 爆火后的刚需
5. **篇 H（视频/3D 扩散）** —— 2025 下半年起突破最多
6. **篇 L（Qwen3.5/3.6 MoE 训练）** —— Qwen3.5 发布后写更有针对性
7. **篇 M（端侧多模态）** —— 相对小众但长期价值高
8. **篇 G（音乐生成）** —— 小众专题
9. **篇 I（Robotic VLA）** —— 最窄受众，但标题党效应好

## 11. YAGNI（暂不做）

- 每篇独立教程级 hands-on code walkthrough（太重，保留 code snippet）
- 比较性论文级 benchmark（不实测，只给"相对量级"）
- 跨篇强耦合（每篇应独立可读）
- 实时数字追更（2026 快照即可）
