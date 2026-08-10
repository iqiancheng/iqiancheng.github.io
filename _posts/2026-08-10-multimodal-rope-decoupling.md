---
title: "从 vanilla RoPE 到多模态 RoPE：位置编码的「解耦」之路"
layout: post
date: 2026-08-10 15:30:00 +0800
tags: [llm, rope, multimodal, paper]
math: true
---

RoPE（Rotary Position Embedding）是当下 LLM 的事实标准位置编码——Llama、Qwen、DeepSeek 全在用。但当模型从纯文本走向多模态（图像、视频、音频），RoPE 这套为 1D 序列设计的机制开始失灵。过去两年出现了一批多模态 RoPE 变体（MRoPE、VideoRoPE、Circle-RoPE、HoPE…），表面上看是各种补丁，实际上它们都在攻击同一个根因。

这篇博客把 vanilla RoPE 和三个代表性的多模态变体放在一起深读，梳理出一条清晰的演化脉络：**位置编码如何从「编码距离」走向「解耦语义」**。最后落到我自己的 ASR MTP 研究方向上，看看这些设计原则能迁移什么。

---

## 一、地基：vanilla RoPE 的三个零件

理解所有变体之前，必须先理解 RoFormer（[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)，2021）的核心机制。RoPE 的全部精髓可以压缩成三个零件：

**1. 旋转矩阵 `R(m)`。** 把每个 token 的 query/key 向量按位置 `m` 旋转一个角度。因为旋转可以复合，位置 `m` 的 query 和位置 `n` 的 key 做点积时，旋转角自动变成 `n−m`——**相对位置**。这同时拿到了绝对位置编码的实现简洁和相对位置编码的语义表达力。

![RoPE 实现：query/key 按 2D 块旋转，角度正比于位置](https://ar5iv.labs.arxiv.org/html/2104.09864/assets/x1.png)
*Figure 1. Implementation of Rotary Position Embedding (RoPE). Query/key 向量被拆成 2D 坐标对，每个坐标对按正比于位置的角度旋转。*

**2. 频率公式 `θ_i = base^(−2i/d)`。** 第 `i` 个坐标对的旋转频率随通道索引**单调衰减**。这个公式是后面所有论文的"单点杠杆"——谁改它，谁就改动了整个编码的几何。

**3. 长程衰减。** 高频通道的旋转角随距离增长得极快，远距离 token 的点积振荡并平均掉——RoPE **隐式**地让注意力偏向近邻 token，这个性质是从数学里"白拿"的，不是手设计的。

![RoPE 长程衰减：远距离 token 注意力自然衰减](https://ar5iv.labs.arxiv.org/html/2104.09864/assets/x2.png)
*Figure 2. Long-term decay of RoPE. 旋转编码天然让远距离 token 的注意力衰减，无需显式窗口。*

这三件事——一次旋转、一个频率调度、一个衰减性质——就是整个多模态 RoPE 战场的地基。后面每一篇变体，都在问："这个频率调度 `θ_i` 放到多模态里，还成立吗？"

---

## 二、问题：RoPE 不知道图像是什么

把一张 2D 图像展平成 1D 序列喂进 RoPE，会发生两件坏事。

**坏事一：位置偶然性取代语义。** 图像里语义相邻的两个 patch（比如一只猫的耳朵和眼睛），展平后可能隔得很远，RoPE 距离也就大——模型学会按"位置巧合"而非"语义相关"去注意。Circle-RoPE（[arXiv:2505.16416](https://arxiv.org/abs/2505.16416)，ICML 2026）用一个叫 **Per-Token Distance (PTD)** 的指标把这个病量化了出来：

$$\text{PTD} = \frac{1}{N_{\text{img}} N_{\text{text}}} \sum_{t}\sum_{i} \left| D_{\text{abs}}(t,i) - \bar{D}_t \right|$$

PTD 衡量每个图像 token 到文本 token 的 RoPE 距离偏离均值的程度。PTD=0 意味着文本 token 看所有图像 token "一样远"——这正是跨模态注意力该有的行为。论文证明了 **PTD=0 是消除几何注意力偏差的充分条件**。

![Circle-RoPE 问题示例：位置 8 的图像 token 虽然语义无关，却在 RoPE 距离上离所有文本 token 最近](https://arxiv.org/html/2505.16416v3/x1.png)
*Figure 1. 一个 VQA 示例：图像 token 位置 8 在顺序展平后，对所有文本 token 的 RoPE 距离最小——尽管语义上更接近的图像 token 在别处。*

**坏事二：维度分块把时间轴塞进高频。** Qwen2-VL 的 MRoPE 把通道分成 t/h/w 三块，但 RoPE 频率随通道索引单调衰减，于是**时间轴被迫落在最高频（衰减最快）的通道**上——这是对长程视频建模的强先验惩罚。Revisiting Multimodal RoPE（[arXiv:2510.23095](https://arxiv.org/abs/2510.23095)，Qwen 团队）把这个问题系统化成了三条设计原则：

1. **位置一致性（coherence）**：每个 token 的坐标无歧义，模态区间定义清晰，图文不碰撞。
2. **全频率利用（full frequency utilization）**：每个位置轴 (t, h, w) 都要能访问全频谱，而不是窄带通道。
3. **保留文本先验（preservation of textual priors）**：纯文本输入时，编码必须和 base LLM 的 vanilla RoPE **完全一致**——任何偏离都会侵蚀预训练能力。

这三条原则"事后看都显然"，但**现有的每一种方法都至少违反一条**。这正是系统性分析的价值：把碎片化的补丁收敛成可判定的准则。

---

## 三、三种解法：一个框架、一个几何、一个定理

三个代表性的变体，分别从三个角度回答"怎么解耦"。

### 3.1 MRoPE-I：通道轮询，让每个轴都有全频谱

MRoPE-I（上面那篇 Qwen 论文）的解法最简单也最"元"。它不指定某个维度该用什么频率，而是把通道**round-robin 轮询**分配给 t/h/w，让每个轴都拿到从高频到低频的完整频谱：

![MRoPE-I 频率分配：通道轮询，每个轴都有全频谱](https://arxiv.org/html/2510.23095v3/x9.png)
*Figure 3. 不同多模态 RoPE 的频率分配。MRoPE 把 d 个通道分成三个连续块 (t, h, w)，把时间轴挤进最高频段；MHRoPE 为每个轴分配全频段 head；MRoPE-I 轮询分配通道。*

它还有个配套的关键细节——**spatial-reset**。MRoPE 存在"视觉注意力汇聚"（visual attention sink）：注意力集中在每张图/每帧的左上角，就像 LLM 的注意力汇聚在初始 token 一样。修复方式是每个视觉内容块开始时把空间位置 (h, w) 重置为零，让视觉 sink 对齐 LLM 对"小位置 ID"的偏好。消融显示，没有它视觉 token 的注意力只有 16–22%，有了它跳到 28–32%。

MRoPE-I 是"drop-in"的——**零架构改动**，纯文本时自动退化为 vanilla RoPE（满足原则 3）。在 20+ 基准上，图像平均 +1.6、grounding +2.4，256K 外推时领先 MRoPE+YaRN 约 2 个点。

### 3.2 Circle-RoPE：锥面几何，让图文距离均匀

Circle-RoPE 走的是几何路线。它的核心构造：把 2D 图像网格投影到一个**垂直于文本轴的圆环（annulus）**上，形成锥面结构——文本 token 沿中心轴，所有图像 token 落在底面的圆上，与轴上每一点等距。这样 PTD=0，同时保留图像内部的空间结构：

![Circle-RoPE 四种编码方案对比：锥面几何 (d) 达成 PTD=0 同时保留空间结构](https://arxiv.org/html/2505.16416v3/x2.png)
*Figure 2. 四种 RoPE 编码方案。Hard embedding (a) 展平 1D，PTD 最差；Unordered (b) 所有图像 token 共享一个索引，PTD=0 但丢空间信息；Spatial (c) 2D 网格；Circle-RoPE (d) 把图像索引映射到垂直于文本轴的圆环。*

具体的三步变换叫 **CIP**（Circular Image Token Index Projection）：坐标中心化 → 混合角环形映射（用 Spatial-Origin Angle 和 Grid-Index Angle 的混合，参数 α 控制权衡）→ 目标平面旋转（让圆环平面垂直于文本方向）。

![Circle-RoPE 的 CIP 三步变换：中心化、混合角环形映射、目标平面旋转](https://arxiv.org/html/2505.16416v3/x3.png)
*Figure 3. CIP 变换步骤。(i) 坐标中心化，(ii) 混合角环形映射，(iii) 目标平面旋转。*

但 CIP 纯做会丢掉图像位置的所有"范围"信息（全在同一个圆上）。所以 Circle-RoPE 又加了 **Dual-Frame Fusion (DFF)** 在原始坐标和 CIP 投影间插值（β=0.1 最优），以及 **Alternating Geometry Encoding (AGE)** 在 Transformer 层间交替标准 M-RoPE 和 Circle-RoPE——模型既拿到网格局部性，又拿到无偏的跨模态注意力。

结果：Qwen2.5-VL-3B 上 10 基准平均 +1.33（68.28 vs 66.95），TAM 空间 grounding 的 Func-IoU +3.45。增益不大，但**推理零成本**——变换只作用于位置索引，不碰权重。

### 3.3 HoPE：时间轴退化为零频率，理论最优

HoPE（[arXiv:2505.20444](https://arxiv.org/abs/2505.20444)，NeurIPS 2025）把矛头对准长视频的**时间轴**，并给出了三篇里唯一的**定理**。

先定义目标性质 **semantic preference**：两个语义相同的 token，无论相隔多远，注意力都应偏向它们。HoPE 证明（Theorem 4.1）：**把时间维度的频率设为恰好零（即时间轴退化为 NoPE）时，这个性质最大化**。

为什么零是好的？非零时间频率下，两个时间分离 token 的点积总含一个依赖距离的旋转因子。这个因子是有界振荡，但随上下文增长，它相对语义信号的**噪声占比无界放大**。频率设零，这个噪声彻底消失——注意力只取决于空间和语义对齐。

![HoPE 频率分配对比：时间轴设零频率，达成语义建模上界](https://arxiv.org/html/2505.20444/x4.png)
*Figure 2(c). HoPE 对时间建模使用零频率，在所有频率分配策略中建立语义建模能力上界。*

HoPE 的第二个贡献是**动态时间缩放（DTS）**：训练时从 Γ={0.5, 0.75, 1.0, 1.25, 1.5} 随机采样一个时间步长缩放因子，推理时按任务选（检索用 γ=0.75，理解用 γ=1.5）。这让模型对帧率变化鲁棒，无需任务特定微调。

结果：V-NIAH 长视频检索 63.56%，比最好的 VideoRoPE 高 22.2 个绝对点，且远超 8k 训练上下文仍保持绿色（高准确率）：

![HoPE 在 V-NIAH 长视频检索上远超训练上下文长度仍保持高准确率](https://arxiv.org/html/2505.20444/x5.png)
*Figure 3. 长视频检索（V-NIAH）性能。每帧对应 144 tokens，黑虚线为 8k 训练上下文。HoPE 在远超训练长度处仍保持高准确率（绿色），所有先前方法跌为红色。*

---

## 四、横切对比：三篇不是竞争，是互补

| 维度 | MRoPE-I | Circle-RoPE | HoPE |
|---|---|---|---|
| 攻击的缺陷 | 连续分块把时间轴塞进高频 | 1D 展平产生位置偶然性 | 非零时间频率产生长程噪声 |
| 核心机制 | 通道轮询 + spatial-reset | 锥面几何 (PTD=0, CIP, AGE) | 零频率时间轴 + 动态缩放 |
| 关键信号 | 三条设计原则 | PTD 指标 | Theorem 4.1 |
| 适用场景 | 通用 VLM | 重空间 grounding | 长视频 VLM |
| 推理成本 | 零 | 零 | 零 |

三篇的关系，我用一句话概括：**MRoPE-I 是"元策略"，Circle-RoPE 和 HoPE 是它的两个特化**。

- MRoPE-I 的 round-robin 是框架级的：它不规定某个维度用什么频率，只保证每个维度都有全频谱访问权。Circle-RoPE 回答"图像维度怎么排"，HoPE 回答"时间维度怎么排"——都是这个框架里的具体实例。
- Circle-RoPE 的 PTD 指标 和 MRoPE-I 的"保留文本先验"是同一枚硬币的两面：前者要求图文 token 的 RoPE 距离均匀，后者要求纯文本时与原始 RoPE 完全一致。目标相同——**别让位置编码干扰语义**。
- HoPE 的"时间轴零频率"看起来和 MRoPE-I 的"每个轴都有全频谱"矛盾，其实不矛盾：MRoPE-I 说"每个轴要有访问全频谱的**权利**"，HoPE 说"时间轴在整个频谱里应该**选零**"。两者可以组合——在 MRoPE-I 的轮询框架下，把时间通道的 θ 设为 0。

**更深一层**：这三篇其实代表了位置编码设计的三种方法论——**框架归纳**（MRoPE-I 提炼原则）、**几何构造**（Circle-RoPE 用空间直觉）、**理论推导**（HoPE 用定理）。多模态位置编码这个子领域，罕见地同时出现了这三种风格的解法。

---

## 五、迁移到 ASR MTP

我自己的研究线是给 bluelm_asr 接入 MTP（Multi-Token Prediction）训练/推理加速。MTP 的核心问题之一，恰恰是**音频帧与文本 token 之间的位置编码对齐**。这三篇论文提供了几个可迁移的设计原则：

**1. 频率分配不能是启发式的。** MRoPE 把音频维度塞进高频通道导致长程建模受损——这个教训直接适用。如果 MTP 的 target 处理用 MRoPE 式分块，音频轴同样会被挤进窄带。**round-robin（MRoPE-I 方案）是最小改动的修复**，每个模态维度获得全频谱访问权。

**2. 跨模态距离需要显式解耦。** Circle-RoPE 的 PTD 指标证明：文本 token 到所有图像 token 的距离应均匀，才能消除几何偏差。对 ASR MTP，**音频帧到文本 token 的 RoPE 距离也应该均匀化**——否则模型学到"第 N 帧更靠近第 M 个文本 token"这种位置偶然性，而非语义相关性。PTD 可以当诊断指标直接套用。

**3. 时间轴可能该退化为 NoPE。** HoPE 的定理（θ=0 最优）在 ASR 场景可能同样成立——音频帧之间的相对位置应由内容决定，而非 RoPE 旋转角。如果 MTP 的验收目标是采样接受率，时间位置编码引入的噪声可能比信号多。

**4. 别破坏纯文本能力。** MRoPE-I 明确要求保留文本先验——多模态扩展不能在纯文本输入时干扰 backbone 的文本 RoPE 基线。这条对 ASR MTP 尤其重要，因为 backbone（Qwen2LM / MoeAsr）的文本能力是既定的。

**最值得先做的实验**：在现有 MTP-5 位置编码上，把通道分配从 contiguous block 改成 **round-robin（MRoPE-I）**——改动最小、收益可预期；再实验**时间轴零频率（HoPE）**，看采样接受率是否提升。

---

## 结语

从 vanilla RoPE 的"一次旋转、一个频率、一个衰减"，到多模态时代的三条设计原则、锥面几何、零频率定理——这条脉络的本质，是位置编码从**编码距离**走向**解耦语义**。MRoPE-I 给框架，Circle-RoPE 给几何，HoPE 给定理，三者互补，共同指向同一个结论：**好的多模态位置编码，应该让位置信息服务于语义，而不是干扰语义。**

## 参考

- RoFormer（vanilla RoPE）：[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Revisiting Multimodal RoPE（MRoPE-I）：[arXiv:2510.23095](https://arxiv.org/abs/2510.23095) · [code](https://github.com/JJJYmmm/Multimodal-RoPEs)
- Circle-RoPE：[arXiv:2505.16416](https://arxiv.org/abs/2505.16416) · [code](https://github.com/lose4578/CircleRoPE)
- HoPE：[arXiv:2505.20444](https://arxiv.org/abs/2505.20444) · [code](https://github.com/hrlics/HoPE)
- 相关精读笔记：`artifacts/roformer-rope`、`artifacts/multimodal-rope`、`artifacts/circle-rope`、`artifacts/hope-position-embedding`
