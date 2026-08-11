---
layout: post
title: "训练大模型时，优化器状态到底在吃多少显存"
date: 2026-01-15 10:00:00 +0800
author: Joseph
# categories: [deep-learning, llm]
# tags: [optimizer, adam, zero, distributed-training, memory]
math: true
categories: [ai-ml]
tags: [memory, training, optimizer]
---
最近在做 LLM 训练相关的工作，反复被一个问题困住：明明模型参数只有 7B，为什么光是训练就要 100 多 GB 显存？排查下来，一半的"黑洞"藏在优化器状态里。这篇文章就是我梳理这块知识的记录，从 Adam 的基本原理出发，一路聊到分布式训练下的优化手段。

---

## 从 Adam 的两个动量说起

Adam 之所以好用，核心在于它对每个参数维护了两套独立的历史统计量——也就是常说的一阶动量 $m$ 和二阶动量 $v$。

**一阶动量 $m$** 是梯度的指数移动平均，本质上是在平滑梯度信号，过滤掉单步的噪声，给优化方向一个"惯性"：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

**二阶动量 $v$** 是梯度平方的指数移动平均，记录的是历史梯度的"大小"：

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

最终的参数更新把两者结合起来：

$$\theta \leftarrow \theta - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

分母 $\sqrt{\hat{v}_t}$ 的作用很直观：历史梯度大的参数，分母大，学习率自动缩小；梯度一直很小的参数，学习率反而被放大。这就是所谓的"自适应学习率"。

### 为什么需要 Bias Correction

$m$ 和 $v$ 初始化为 0，训练早期会被零值拖累，估计严重偏低。Bias correction 通过除以 $(1 - \beta^t)$ 来修正这个偏差：

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

训练初期 $t$ 很小，$\beta_1^t$ 接近 1，修正系数很大，拉回了被压低的估计值。随着 $t$ 增大，修正项趋近于 1，自然消失。这个细节看起来不起眼，但不做修正会让前几千步的有效学习率远低于设定值，收敛明显变慢。

---

## 显存到底去哪了

理解了 $m$ 和 $v$ 之后，显存的账就很好算了。

在混合精度（AMP）训练下，标准做法是：前向/反向用 fp16 以节省计算和带宽，但优化器状态必须保持 fp32。原因很实际：$v$ 存的是梯度平方，值域可以跨越十几个数量级，fp16 的动态范围（约 $6 \times 10^{-8}$ 到 $6.5 \times 10^4$）根本压不住，容易溢出或下溢。$m$ 的累积更新也有类似问题，fp16 的精度不足会让小学习率场景下参数更新"消失"。

算下来，每个参数的显存占用是这样的：

| 存储项 | 精度 | bytes/param |
|--------|------|-------------|
| 模型参数（fp16） | fp16 | 2 |
| 梯度（fp16） | fp16 | 2 |
| 参数主副本（fp32） | fp32 | 4 |
| 一阶动量 $m$ | fp32 | 4 |
| 二阶动量 $v$ | fp32 | 4 |
| **合计** | | **16 bytes** |

7B 模型：$7 \times 10^9 \times 16 \approx$ **112 GB**，还没算激活值。而 inference 时只需要 fp16 权重，只要约 14 GB。训练和推理的显存差距接近 8 倍，绝大部分都被优化器状态和梯度贡献了。

这也解释了一件让很多人困惑的事：为什么断点续训必须保存优化器状态？$m$ 和 $v$ 是训练过程积累的"记忆"，丢掉它们重启，等于把历史统计清零，bias correction 会让前几千步的有效学习率极低，需要重新"预热"才能恢复，白白浪费算力。

---

## Adam 的已知缺陷

用 Adam 做大规模训练久了，会踩到一些坑值得提前知道：

**内存重**是最直接的问题，上面已经说了。7B 模型光优化器状态就 56 GB，100B+ 的模型基本不可能在单机上跑。

**对稀疏梯度适应差**。Embedding 层的梯度极度稀疏，大量位置的梯度在大多数 step 都是 0，$v$ 的估计会长期维持在很小的值，等到真正有梯度时，分母过小导致步长爆炸。

**超参数敏感**。$\beta_1, \beta_2, \epsilon, \eta$ 四个参数的组合空间不小，$\epsilon$ 的选取对数值稳定性影响很大，不同任务和模型规模下最优组合差异明显。

### 为什么是 AdamW 而不是 Adam

现在主流 LLM 训练几乎清一色 AdamW，区别在于 Weight Decay 的处理方式。

原版 Adam 把 L2 正则加在梯度上，但梯度会被 $v$ 的自适应缩放处理，实际施加的正则化强度因参数而异，效果不稳定。AdamW 把 Weight Decay 从梯度中解耦出来，直接作用于参数：

$$\theta \leftarrow \theta(1 - \eta\lambda) - \frac{\eta \hat{m}}{\sqrt{\hat{v}}+\epsilon}$$

这样正则化强度与梯度幅度无关，对所有参数统一施加，大模型训练中这个差异对泛化性能的影响相当显著。

---

## 分布式训练下的优化器状态管理

单机放不下，就要考虑怎么切。ZeRO（Zero Redundancy Optimizer）是目前最主流的方案，核心思路是：既然每张卡都存一份完整的优化器状态是冗余的，为什么不切分？

ZeRO 分三个阶段，每个阶段切的东西不同：

**Stage 1** 只切优化器状态（$m$ 和 $v$）。每张卡只维护 $1/N$ 的状态，更新时各卡计算自己负责的参数，用 reduce-scatter 同步梯度，用 all-gather 拿到完整参数。内存节省约 4 倍。

**Stage 2** 在 Stage 1 基础上再切梯度。梯度不再全量存储，只保留各卡负责部分的梯度。内存节省约 8 倍。

**Stage 3** 把参数本身也切了。每张卡只存 $1/N$ 的参数。内存节省约 16 倍，但代价是 forward 和 backward 都需要 all-gather 重建完整参数，通信量上升约 1.5 倍。网络带宽不够的集群上，Stage 3 的训练速度会明显下降，通常需要配合 prefetch 来掩盖通信延迟。

实践中，Stage 1/2 是性价比最高的选择，Stage 3 留给真的放不下的场景。

---

## 更省内存的替代优化器

如果 Adam 太重，有几个方向可以探索：

**Adafactor** 是 Google 提出的内存高效方案，不存完整的 $v$ 矩阵，而是用行向量和列向量的秩一分解来近似：$V \approx r \cdot c^T$。对于大型权重矩阵，内存从 $O(n)$ 降到 $O(\sqrt{n})$，优化器状态减少约一半。代价是近似引入了噪声，收敛稳定性略逊于 Adam，学习率调度也需要特殊处理。T5 家族就是用 Adafactor 训练的。

**Lion** 是 Google Brain 2023 年用进化搜索发现的优化器，只存一阶动量 $m$（省去 $v$），更新方向用 sign 函数：

$$\theta \leftarrow \theta - \eta \cdot \text{sign}(\beta_1 m + (1-\beta_1)g)$$

优化器状态减半至 4 bytes/param，计算量也更小，在图像和语言任务上效果与 AdamW 相当。使用时需要更小的学习率和更强的 Weight Decay，两者都与 Adam 有所不同，直接替换需要重新调参。

---

## 几个容易混淆的点

最后梳理几个经常被搞混的问题。

**Gradient Checkpointing 和优化器状态有关系吗？** 没有直接关系。Gradient Checkpointing 省的是激活值显存，通过在反向传播时重新计算部分激活值来换取内存，与优化器状态是两个独立的维度。所以Gradient Checkpointing 通常也叫做 Activation Checkpointing 或 Activation Recomputation(激活重算)，两者可以同时使用，完整的省内存组合通常是 ZeRO Stage 2/3 + Gradient Checkpointing + bf16。

**Inference 时优化器状态去哪了？** 全部丢掉。推理只需要模型权重，$m$、$v$、fp32 主副本、梯度全部不需要。这也是为什么 GGUF、AWQ 这类推理格式只存权重（甚至量化权重），体积可以比训练 checkpoint 小一个数量级。

**为什么现在很多训练用 bf16 而不是 fp16？** bf16 与 fp32 有相同的指数位数（8 位），动态范围与 fp32 一致，只是精度（尾数位）低一些。fp16 动态范围小，大模型训练中梯度和激活值的数值范围经常超出 fp16 上限，需要 loss scaling 来稳定训练，bf16 则基本不需要这个麻烦。A100 和 H100 都对 bf16 有硬件加速支持，目前基本成了 LLM 训练的默认选择。

---

## 参考资料

以下是这些内容的原始出处，值得深读：

**核心论文**

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba (2014)，Adam 原始论文，Bias Correction 的推导在这里
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter (2017)，AdamW 提出论文，解释了为什么 L2 正则和 Adam 不兼容
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al. (2019)，ZeRO 原始论文，Stage 1/2/3 的通信分析非常详细
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1805.04853) — Shazeer & Stern (2018)
- [Symbolic Discovery of Optimization Algorithms (Lion)](https://arxiv.org/abs/2302.06675) — Chen et al. (2023)

**工程向文章**

- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Megatron-LM 团队关于 Gradient Checkpointing 的实践
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al. (2017)，AMP 的原始论文，fp16 训练中 loss scaling 的来源
- [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://dl.acm.org/doi/10.1145/3394486.3406703) — ZeRO 实现的工程细节

**博客 / 文档**

- [Hugging Face: Model Training Anatomy](https://huggingface.co/docs/transformers/model_memory_anatomy) — 非常好的显存拆解，表格清晰
- [Lilian Weng: An Overview of Large Language Models](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — 包含优化器和训练技巧的系统综述
- [EleutherAI: Transformer Math 101](https://blog.eleuther.ai/transformer-math/) — 参数量、FLOPs、显存的快速估算公式，实用
