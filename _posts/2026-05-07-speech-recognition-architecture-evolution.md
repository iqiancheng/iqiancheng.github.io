---
layout: post
title: "语音识别模型架构演进：从 HMM-GMM 到 Whisper 到 Qwen3-ASR"
date: 2026-05-07 00:00:00 +0800
author: Joseph
categories: [语音, 深度学习]
tags: [speech]
---

> 本文是 [《语音模型预处理流程及常用术语详解》](/2026/05/07/speech-model-preprocessing-glossary.html) 的姊妹篇。前一篇讲"输入怎么来"，这一篇讲"模型怎么吃"。把 2012 ~ 2026 十四年的 ASR 架构演进捋清楚，并且重点拆解最新的 **Qwen3-ASR（2026）**。

---

## 一、代际总览

| 代际 | 时间 | 代表模型 | 核心思想 | 训练数据量级 |
|---|---|---|---|---|
| 1. HMM-GMM | ~2010 前 | HTK / Kaldi 经典方案 | 统计对齐 + GMM 发射概率 | ≤ 千小时 |
| 2. DNN-HMM | 2012–2015 | Kaldi chain、DeepSpeech v1 | DNN 替换 GMM | 千~万小时 |
| 3. 端到端 CTC/AED/RNN-T | 2015–2020 | DeepSpeech 2、LAS、Jasper、Conformer | 抛弃 lexicon/对齐 | 万小时 |
| 4. 自监督预训练 | 2020–2022 | Wav2Vec2、HuBERT、WavLM | 无标签预训练 + 少量微调 | 10 万+ 小时无标签 |
| 5. 大规模弱监督 | 2022–2024 | Whisper、SeamlessM4T | 大规模带字幕的互联网音频 | 68 万+ 小时带转录 |
| 6. Audio-LLM | 2024– | Qwen2-Audio、SALMONN、Gemini Audio、Phi-4-MM | 语音 = 一种模态 token | LLM 规模 |
| 7. ASR-specialized LLM | 2026– | **Qwen3-ASR**、Voxtral | 通用 Audio-LLM 蒸馏出的 ASR 专精版 | 数百万小时 |

每一代都在**扩大端到端的范围**：逐步吞并掉"特征工程 → 声学模型 → 发音词典 → 语言模型 → 后处理"这条流水线上的独立模块，最终变成"audio in, text out"的单一模型。

> **❓ 为什么文本大模型不需要"预处理 processor"？**
>
> 文本 LLM 拿到的是字符串，经过 **tokenizer (BPE / SentencePiece)** 直接变成 `input_ids: List[int]`，每个 int 是词表里的索引——整个过程无损、可逆、CPU 毫秒级。
>
> 语音则是**连续的物理量**：16k 采样 30 秒就是 480,000 个 float，如果直接当 token 塞进 Transformer，序列太长（显存爆）、信息密度太低（10 个采样点才等效 1 个音素的信息量）。所以必须先做"有损压缩 + 离散化"才能喂给 LLM——这个压缩流水线就叫 **audio processor / feature extractor**（Log-Mel、CMVN、SpecAugment 全在这一层）。
>
> 一句话：**文本天生是离散的 token，语音天生是连续的信号，多了一步"连续 → 离散"的桥。**

---

## 二、第一代：HMM-GMM（统计时代）

### 2.1 架构骨架

```
MFCC (39 维) ─→ GMM 发射概率 ─→ HMM 状态序列 ─→ Viterbi 解码
                                                       ↓
                                    发音词典 (lexicon) + n-gram LM
                                                       ↓
                                                  最终文本
```

三个独立训练的模块：

- **声学模型 (AM)**：HMM 状态转移 + GMM 建模每个状态的 MFCC 分布
- **发音词典 (Lexicon)**：单词 → 音素序列映射，纯人工整理
- **语言模型 (LM)**：n-gram 统计 P(word | history)

### 2.2 为什么被淘汰

- **三段式流水线误差累积**：AM 错 → LM 也救不回来
- **人工词典不可扩展**：新增一种语言要语言学家写几个月
- **GMM 假设太强**：每个 HMM 状态的声学特征服从高斯混合，现实语音分布远比这复杂

唯一留下的遗产：**forced alignment（强制对齐）工具链**（HTK、MFA 至今还在给 TTS 训练集打时间戳）。

---

## 三、第二代：DNN-HMM（混合时代，2012–2015）

### 3.1 关键改动

Hinton 2012 那篇 *Deep Neural Networks for Acoustic Modeling* 只动了一个地方：

```
GMM(发射概率) → DNN(发射概率)
```

其它部分（HMM、lexicon、n-gram LM）不变。但精度提升巨大——Switchboard 数据集 WER 从 23% 降到 18%。

### 3.2 Kaldi chain model

Kaldi 的 `nnet3/chain` 把 DNN-HMM 打磨到了工业级：
- LF-MMI 序列训练
- TDNN / TDNN-F 架构
- i-vector 说话人自适应

直到 2020 年前，**Kaldi 仍然是工业 ASR 的主流**（电话客服、呼叫中心）。

---

## 四、第三代：端到端（2015–2020）

三条并行技术路线同时爆发。

### 4.1 CTC 路线

**Connectionist Temporal Classification** 解决了"输入 1000 帧、输出 10 个字母、怎么对齐"这个问题——引入 blank 符号，让模型自己学对齐。

代表作：
- **DeepSpeech / DeepSpeech 2**（百度 2014/2015）：纯 RNN + CTC + 字符级输出
- **Wav2Letter**（Facebook 2016）：纯 CNN + CTC
- **Jasper / QuartzNet**（NVIDIA 2019）：1D 可分离卷积 + CTC，部署性极好

**优点**：结构简单，流式天然。**缺点**：CTC 独立性假设强，无语言模型时容易"音对字错"（详见前篇的 `CHRISTMAUS` vs `CHRISTMAS` 例子）。

### 4.2 AED (Attention Encoder-Decoder) 路线

**LAS (Listen, Attend and Spell)**（Google 2015）是教科书：

```
Encoder (BiLSTM)  ──→  Encoder hidden states
                            │
                            ▼ (cross attention)
Decoder (LSTM, step-by-step)  ──→  "H", "E", "L", ...
```

- 优点：解码器本身就是语言模型，语法顺
- 缺点：整句处理、不流式、数据饥渴

### 4.3 RNN-T (Recurrent Neural Network Transducer)

**Google 2012 提出，2019 年 iPhone / Pixel 上落地**：

```
Encoder     ──→  audio encoding
Predictor   ──→  text encoding (类似 LM)
Joiner      ──→  Encoder ⊕ Predictor → 输出 token 或 blank
```

天然支持**流式 + 语言模型内置**，牺牲训练稳定性。Google、Apple Dictation、字节的实时字幕大量采用。

### 4.4 Conformer（2020 集大成者）

Google 的 Conformer 模块 = **Convolution + Multi-Head Self-Attention + Feed-Forward**，吸收了 CNN 的局部建模和 Transformer 的全局建模优势：

- Conformer-CTC（非流式） / Conformer-Transducer（流式）
- 在 LibriSpeech test-clean 上做到 WER ~2%
- 至今（2026）仍是**公认的强 encoder**，被 Whisper、Paraformer、甚至 Qwen3-ASR 借鉴

---

## 五、第四代：自监督预训练（2020–2022）

核心洞察：**有标签语音贵，无标签语音便宜到几乎免费**。借鉴 BERT / GPT 做自监督预训练。

### 5.1 Wav2Vec 2.0（Meta 2020）

```
Raw waveform (16kHz)
        │
        ▼
7-layer CNN feature extractor  ──→  50 fps latent z
        │
        ▼
Transformer encoder            ──→  contextual c
        │
        ▼
Contrastive loss: 让被 mask 的 z 和对应 c 匹配
```

- **波形直入**：不再依赖 Mel-spec，CNN 前端学出等效特征
- **预训练 + 微调范式**：10 分钟标注数据就能做出可用 ASR
- 开源 checkpoint 引爆后续所有"语音 BERT"

### 5.2 HuBERT（Meta 2021）

改对比学习为 **masked cluster prediction**：先用 k-means 聚类得到 pseudo-label，再像 BERT 那样预测被 mask 的 label。训练更稳定，性能略优于 Wav2Vec2。

### 5.3 WavLM（Microsoft 2021）

在 HuBERT 基础上加 **denoising pretext**：故意往输入混噪声、混另一说话人的语音，让模型学会抗噪声 + 说话人分离。后端接 ASR / 说话人验证 / 情感识别都能打。

**这一代的共同局限**：自监督只给 encoder，还要额外的 decoder 做下游任务。CTC head 无语言模型、AED decoder 要额外微调——工程上仍有缺口。

---

## 六、第五代：大规模弱监督——Whisper 范式（2022）

OpenAI 的 Whisper 跳出"预训练 + 微调"的套路，直接换了个问法：

> 与其用 100 万小时无标签 + 1 万小时标注，能不能用 **68 万小时互联网带字幕**的音频（YouTube / Podcasts / LibriVox 等）直接端到端训？

### 6.1 架构

```
80-维 Log-Mel (30s 固定窗口)
        │
        ▼
Transformer Encoder (凝练声学信息)
        │
        ▼ (cross-attention)
Transformer Decoder (generative, 吐 token)
        │
        ▼
 Text  +  Timestamps  +  Language ID  +  翻译
```

特殊 token 让**同一个模型做多任务**：
- `<|transcribe|>` vs `<|translate|>`
- `<|zh|>` / `<|en|>` / ... 语种提示
- `<|notimestamps|>` 或 `<|0.00|>` 时间戳

![Whisper architecture and multitask training format](https://raw.githubusercontent.com/openai/whisper/main/approach.png)
*图：Whisper 的架构与多任务训练格式。Encoder 吃 Log-Mel，Decoder 吃带"任务类型 / 语种 / 时间戳"的 prompt token，一套权重覆盖转录 / 翻译 / VAD / 语种识别。来源：OpenAI Whisper GitHub repo*

### 6.2 为什么 Whisper 赢了

- **数据规模碾压**：68 万小时 > 所有学术数据集之和
- **真实世界噪声分布**：训练数据本来就带噪，鲁棒性天然好
- **多语言多任务统一**：一个模型解决 99 种语言
- **开源权重**：`whisper-large-v3` 直接可商用

**局限**：
- 固定 30 秒输入，长音频要切片再拼
- 解码慢（自回归）、非流式
- "幻觉"问题严重：静音段会凭空吐字幕

![Whisper language coverage (WER breakdown by language)](https://raw.githubusercontent.com/openai/whisper/main/language-breakdown.svg)
*图：Whisper 在 99 种语言上的 WER 覆盖——这种"广度"只有在训练集跨越 68 万小时多语言互联网音频的前提下才能达成。来源：OpenAI Whisper GitHub repo*

---

## 七、第六代：Audio-LLM（2024–）

Whisper 解决了"转录"，但人类对语音的需求远不止转录——还要理解、对话、情感、歌唱、音乐……于是**把语音变成 LLM 的一种输入模态**。

### 7.1 通用范式

```
Audio (16k waveform)
   │
   ▼
Audio Encoder (通常复用 Whisper encoder)
   │
   ▼  (N audio tokens, e.g. 50 per second)
[audio tokens] + [text prompt] → LLM Decoder → text answer
```

> **❓ 音频是怎么变成 token 的？这个 token 是 `input_ids` 吗？**
>
> **不是同一种 token**。要分清两条路线：
>
> **路线 A：连续嵌入 (continuous embeddings)** ——目前 Audio-LLM 主流做法
>
> ```
> Log-Mel → Audio Encoder (Whisper/Conformer) → [T, D] 连续向量序列
>                                                      ↓
>                         再做一次 Perceiver/MLP resampler 压到固定 N 个 token
>                                                      ↓
>                         [N, D_llm] 的向量 ——— 直接拼在 text embeddings 前面
> ```
>
> 这些 "audio tokens" 本质是 **浮点向量 embedding**，**不经过 LLM 的 `embed_tokens` 层**。它们不是 `input_ids`（int 索引），更像是 vision LLM 里的 `pixel_values → image_embeds` 的角色。
>
> 调用时你会看到：
> ```python
> inputs = processor(audio=wav, text=prompt, return_tensors="pt")
> # inputs 里同时有 input_ids (text) 和 input_features 或 audio_values (audio)
> model.generate(**inputs)
> ```
>
> **路线 B：离散音频 token (discrete audio tokens)** ——TTS / 音乐生成常见，ASR 少见
>
> 用 **RVQ（Residual Vector Quantization）**—— 如 EnCodec、SoundStream、HuBERT discrete units —— 把音频量化成真·整数 codebook index。这种才真的能当 `input_ids` 用。代表：Google AudioLM、Meta VoiceBox、Whisper-Vec、Moshi。
>
> **一句话**：Audio-LLM 的 audio token 绝大多数是**连续向量**（像图片 patch embedding），只有少数端到端 speech-to-speech 架构才用离散的真 token。

> **❓ AuT 是什么？**
>
> **AuT = Audio Tokenizer**，指"把连续音频变成离散 token"的那一层（对应上面"路线 B"）。命名不统一，社区还会叫：
>
> - **Audio Codec / Neural Audio Codec**（侧重压缩）
> - **Discrete Audio Tokens / Speech Tokens**（侧重下游使用）
> - **Semantic Tokens vs Acoustic Tokens**：语义 token（HuBERT 聚类）保留"说了什么"，声学 token（EnCodec RVQ）保留"怎么发音、谁说的"
>
> 主流实现：
>
> | 名字 | 组织 | 类型 | 典型码率 |
> |---|---|---|---|
> | **EnCodec** | Meta | 声学 (RVQ) | 1.5/3/6/12/24 kbps |
> | **SoundStream** | Google | 声学 (RVQ) | 3/6/12 kbps |
> | **HuBERT discrete** | Meta | 语义 (k-means) | ~50 tokens/s |
> | **WavTokenizer** | 2024 | 声学 | 0.5~0.9 kbps |
> | **Mimi** | Kyutai (Moshi) | 语义+声学双流 | 1.1 kbps |
>
> AuT 做到**真·离散**后，LLM 就能在语音上做 next-token prediction——这是 GPT-4o 实时语音对话、Moshi、Qwen2.5-Omni-Speech 背后的关键。

> **❓ Codebook（码本）是什么？它是 ASR 用的还是 TTS 用的？**
>
> **Codebook 是"离散音频 token 的词表"**——一张可学习的查找表（lookup table），形状 `[K, D]`：`K` 个 entry（典型 1024 或 4096），每个是 `D` 维向量（典型 256）。
>
> **离散化流程**：
>
> ```
> 连续向量 z ──→ argmin_k ‖z − codebook[k]‖ ──→ 输出 index k (int)
>                           ↑
>                    （最近邻查码本）
> ```
>
> 解码时：`z_hat = codebook[k]` 查表还原向量 → 经 decoder 网络合成回波形。
>
> **RVQ (Residual Vector Quantization)** 是多层 codebook 递归编码**残差**：第 1 层量化后，把误差 `z − z_hat` 喂给第 2 层再量化，以此类推。典型 8 层 × 1024 entries 就能把 24 kHz 音频压到 6 kbps 还基本听不出失真——这是 EnCodec / SoundStream 的核心。
>
> **主要给谁用？**
>
> | 场景 | 用 codebook 吗 | 理由 |
> |---|---|---|
> | **TTS / 语音克隆** | ✅ 必需 | 要做 autoregressive 生成，必须先离散化（VALL-E、CosyVoice、F5-TTS） |
> | **音乐 / 歌曲生成** | ✅ 必需 | Suno、MusicGen 底层就是 EnCodec codebook |
> | **端到端 speech-in → speech-out** | ✅ 必需 | Moshi / GPT-4o Realtime / Qwen2.5-Omni-Speech 都靠 codebook |
> | **纯 ASR（转录）** | ❌ 不用 | 输出是文本，连续 embedding 直接拼给 LLM decoder 即可，无需离散音频 token |
> | **自监督预训练** (Wav2Vec2 / HuBERT) | ⚠️ 内部隐含 | Wav2Vec2 的 quantizer、HuBERT 的 k-means pseudo-label 本质是 codebook，但只用于 pretext 任务，对下游用户透明 |
>
> **一句话判题**：**codebook 是"生成侧（TTS / 对话）"的核心基础设施，ASR 侧除了预训练阶段以外基本不直接用。**
>
> **两种 codebook 的分化**（在上面"AuT 主流实现"表里已经出现过）：
>
> - **声学 codebook** (acoustic)：EnCodec / SoundStream——优先保留"怎么发音、谁说的、有什么背景音"，码率高，信息丰富
> - **语义 codebook** (semantic)：HuBERT discrete / S3 tokenizer——优先保留"说了什么"，码率低，适合理解类任务
> - **双流 codebook**：Mimi (Moshi)、CosyVoice 2——把语义流和声学流拆开，LLM 只看语义流省算力，合成侧再用声学流补回音色
>
> 这也是为什么 2024 年以后一个 audio codec 论文的核心贡献往往是"**更低码率 + 更少 codebook 层数 + 更好语义保留**"——它直接决定了上层 LLM 能不能用合理的序列长度生成高保真音频。

### 7.2 代表作

| 模型 | 组织 | Audio Encoder | LLM | 特色 |
|---|---|---|---|---|
| **Qwen2-Audio** | 阿里 | Whisper-large | Qwen2-7B | 中文+歌唱+情感 |
| **SALMONN** | 清华+字节 | Whisper + BEATs | Vicuna | 非语音理解强 |
| **Phi-4-Multimodal** | Microsoft | Whisper-v3 | Phi-4 | 小参数规模 |
| **Gemini 1.5/2.5 Audio** | Google | 自研 | Gemini | 一小时级长音频 |
| **GPT-4o Audio** | OpenAI | 自研 + 实时 | GPT-4o | 端到端语音对话 |

### 7.3 优缺点

- ✅ 理解能力爆炸（能回答"说话人情绪如何？背景音乐什么风格？"）
- ✅ 遵循指令（"用一句话总结"、"翻译成法语"）
- ❌ 纯转录精度反而不如 Whisper / Conformer 专精模型
- ❌ 延迟高，成本高

---

## 八、第七代：ASR 专精化回归——Qwen3-ASR（2026）

Audio-LLM 的瑞士军刀定位有代价：当用户**只要高质量转录**时，跑一个 7B+ 的全能模型既慢又贵。**阿里 2026 发布的 Qwen3-ASR 正是对这种分化的回应**——从 Qwen3-Omni 基座蒸馏/裁剪出专做 ASR 的小模型。

### 8.1 关键事实

- **两个尺寸**：Qwen3-ASR-**1.7B** / Qwen3-ASR-**0.6B**
- **基座**：Qwen3-Omni 多模态大模型
- **覆盖**：30 种语言 + 22 种中文方言（闽南、吴语、粤语、陕西、四川…）
- **英语多口音**：印度英语、澳洲英语等
- **伴生对齐模型**：Qwen3-ForcedAligner-0.6B（NAR，支持 5 分钟内任意粒度时间戳）
- **推理栈**：支持 transformers / vLLM 双后端、FlashAttention 2、批推理 / 异步服务 / 流式一体
- **音频类型**：支持普通语音 + 歌唱 + 带背景音乐的歌曲（BGM）——这一点显著超越 Whisper

![Qwen3-ASR architecture](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/qwen3_asr_introduction.png)
*图 1：Qwen3-ASR 系列的能力概览。来源：阿里通义 Qwen 团队 HuggingFace repo*

### 8.2 架构推测（基于 Qwen3-Omni）

官方没完全公开细节，但从 repo 代码结构和 Qwen3-Omni 的论文可以推断：

```
Audio (16kHz)
    │
    ▼
Audio Encoder (Conformer-like, 来自 Qwen3-Omni)
    │  ~50 fps audio tokens
    ▼
Audio ↔ Text adapter (Perceiver-like resampler 或 MLP)
    │
    ▼
Qwen3-base LLM (1.7B / 0.6B) with ASR-specialized SFT
    │
    ▼
Transcription + Language ID + (可选) timestamps
```

**关键设计取舍**：
- 相对 Whisper：抛弃 30 秒固定窗口，支持变长 + 长音频连续转录
- 相对 Qwen2-Audio：LLM 规模从 7B 缩到 1.7B/0.6B，专门为 ASR 微调
- 相对 Conformer 传统方案：保留 LLM decoder 以获得**语言模型感**（减少音对字错）

![Qwen3-ASR architecture overview](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/overview.jpg)
*图：Qwen3-ASR 架构概览——Audio Encoder（继承 Qwen3-Omni 声学前端）+ Audio-Text Adapter + Qwen3 LLM Decoder，额外配套 Qwen3-ForcedAligner 做字级时间戳。来源：阿里通义 Qwen 团队*

### 8.3 流式/离线统一推理

Qwen3-ASR 的一大工程亮点：**同一组权重同时支持 offline 批处理和 streaming 实时流**。通过在训练时混合 chunk-mask 和 full-attention mask，推理时根据业务需求切换策略。

```python
# Offline 模式（最高精度）
results = model.transcribe(audio="long.wav", return_time_stamps=True)

# Streaming 模式（低延迟）
for chunk in model.transcribe_streaming(audio_generator):
    print(chunk.text, end="", flush=True)
```

### 8.4 Forced Aligner 的独立价值

Qwen3-ForcedAligner-0.6B 是个**非自回归 (NAR)** 模型，专做时间戳：给定音频 + 已知文本，输出每个字/词/音素的起止时间。

这对以下场景是刚需：
- TTS 训练集构建（需要字级对齐）
- 视频字幕 SRT 生成（需要行级时间戳）
- 唱词对齐（Qwen3-ASR 本身支持歌曲转录，对齐质量直接决定 KTV/MV 应用体验）

传统方案 MFA (Montreal Forced Aligner) 基于 Kaldi HMM，跨语言扩展难；Qwen3-ForcedAligner 用神经网络直接建模，**支持 11 种语言的任意单位对齐**。

### 8.5 与 Whisper-large-v3 对比（纸面数据）

| 维度 | Whisper-large-v3 | Qwen3-ASR-1.7B |
|---|---|---|
| 参数量 | 1.54B | 1.7B |
| 训练数据 | 680k 小时带字幕 | 未公开具体数字，但应在百万小时级 |
| 语种 | 99 | 30 + 22 中文方言 |
| 长音频 | 30s 窗口切片 | 原生长音频 |
| 流式 | 不支持 | 支持 |
| 带 BGM 音频 | 容易失败 | 显著更强 |
| 中文方言 | 仅普通话 | 22 种方言 |
| 时间戳精度 | 片段级 | 字级（配 Forced Aligner） |

**应用取向**：Whisper 胜在语种覆盖最广；Qwen3-ASR 胜在**中文场景 + 流式 + 歌曲**。国内落地首选前者基本没争议。

---

## 九、2026 年最主流的语音模型架构长什么样？

把前面八节的线索拧成一张"截止本文成稿时的截面图"：

```
┌──────────────────────────────────────────────────────────────┐
│          2026 主流 Speech Model 架构分化图                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   [A] 专精 ASR / 高精度高吞吐                                  │
│       ├── Qwen3-ASR-1.7B / 0.6B   (Alibaba, 2026)             │
│       ├── Voxtral                 (Mistral, 2025)             │
│       ├── Paraformer-v2 / SenseVoice-v2 (Alibaba FunASR)      │
│       └── Whisper-v4 (传闻)        (OpenAI)                    │
│                                                               │
│   [B] 通用 Audio-LLM / 理解 + 对话                              │
│       ├── Qwen2.5-Omni            (Alibaba, 端到端多模态)      │
│       ├── GPT-4o Audio / GPT-5.5  (OpenAI, 实时语音)           │
│       ├── Gemini 2.5 Audio        (Google, 一小时级长音频)     │
│       ├── Phi-4-Multimodal        (Microsoft, 轻量)            │
│       └── SALMONN-v2              (清华/字节)                  │
│                                                               │
│   [C] 端到端语音对话 (speech-in → speech-out)                   │
│       ├── Moshi                   (Kyutai, 双流 full-duplex)   │
│       ├── GPT-4o Realtime API     (OpenAI)                    │
│       └── Qwen2.5-Omni-Speech     (Alibaba)                   │
│                                                               │
│   [D] 音乐 / 歌唱 / TTS                                         │
│       ├── Suno v4 / Udio          (歌曲生成)                   │
│       ├── CosyVoice 2             (Alibaba, 零样本音色)        │
│       └── F5-TTS / GPT-SoVITS     (开源 TTS)                   │
└──────────────────────────────────────────────────────────────┘
```

### 共同骨架

无论是 [A] [B] [C] 哪条路线，**核心架构都可以抽象成三段**：

```
Raw audio → [前端] Log-Mel 或 Raw Waveform + CNN
          → [编码器] Conformer / Whisper-Encoder / 自研 Transformer
          → [解码器 / 对齐]
              ├── CTC / RNN-T              ← [A] 专精 ASR
              ├── LLM Decoder (text-only)   ← [A][B]
              ├── LLM Decoder + AuT         ← [C] 端到端对话
              └── Diffusion / Flow-matching ← [D] 生成
```

### 今天最"主流"的一条技术栈（保守选择）

如果你 2026 年开新项目、只能选一条路线：

```
输入:     16 kHz mono PCM
前端:     Log-Mel 80 × T,  CMVN + SpecAugment
编码器:   Conformer  或  Whisper encoder (蒸馏过的)
对齐:     LLM Decoder (Qwen3 / Llama 3.x) + ASR SFT
训练:     数百万小时弱监督 + 人工高质量精校 + RLHF-for-ASR
推理:     vLLM 后端, FlashAttention 2, bf16,
         支持 streaming / offline 统一
后处理:   Forced Aligner (NAR 神经网络) → 时间戳
         可选 LLM-based 后编辑做标点/ITN
```

这套组合正是 **Qwen3-ASR 架构代表的方向**，也是 Voxtral、CosyVoice 系、Whisper-v4（尚未公开）都在收敛的形态。**2026 年的"主流"，就是从 Audio-LLM 里"蒸馏"出专精模型，同时保留 LLM-style 的语言理解能力**——纯 Conformer-CTC 时代已经翻篇，但"把一切都塞进 70B Omni 模型"也被证明在 ASR 这个子任务上不划算。

---

## 十、架构选型决策导图

```
场景 → 推荐架构
├── 小模型 / 嵌入式关键词唤醒
│   └── QuartzNet / 小型 Conformer-CTC（5M ~ 50M 参数）
├── 电话客服实时转录（8kHz、要流式）
│   └── Conformer-Transducer (RNN-T)  或  Qwen3-ASR-0.6B streaming
├── 大规模离线字幕（准确率优先）
│   ├── 英文 → Whisper-large-v3
│   └── 中文 → Qwen3-ASR-1.7B / Paraformer-large
├── 多语言翻译 + 转录一体
│   └── Whisper / SeamlessM4T
├── 需要理解 + 对话 + 情感
│   └── Qwen2-Audio / GPT-4o / Gemini 2.5 Audio
├── TTS 数据集构建（字级对齐）
│   └── MFA  或  Qwen3-ForcedAligner-0.6B
└── 歌曲/带 BGM 音频转录
    └── Qwen3-ASR（少数能打的开源方案）
```

---

## 十一、几条趋势判断

1. **Audio-LLM 和 ASR-specialized 会长期并存**：全能模型做理解与对话，专精模型做工业级高速高精转录。Qwen3-ASR 这条路线会被复制。
2. **"去 Mel-spec"的尝试会继续但未必成为主流**：Wav2Vec 系证明了可行性，但 Log-Mel 80 维的工程生态（torchaudio、CMVN、SpecAugment）太成熟。短期内 Log-Mel 仍是性价比最高的前端。
3. **流式 + 长音频 + 多方言会是中文 ASR 的核心战场**：Qwen3-ASR 22 方言是给对手（科大讯飞、微软）的一个明确挑战。
4. **Forced Aligner 从 Kaldi HMM 迁移到 NAR Neural 模型**：MFA 的地位在 2 年内可能被颠覆。
5. **ASR 评估从 WER 走向语义评估**：对于 LLM-based ASR 模型，严格的 WER 有时惩罚了"更通顺但不字字一致"的输出。Semantic-WER / NLG-style 评估会兴起。

---

## 十二、参考资料

- [Hugging Face Audio Course – Unit 5 ASR](https://huggingface.co/learn/audio-course/en/chapter5/asr_models)
- [Wav2Vec 2.0 paper (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [HuBERT paper (Hsu et al., 2021)](https://arxiv.org/abs/2106.07447)
- [WavLM paper (Chen et al., 2021)](https://arxiv.org/abs/2110.13900)
- [Conformer paper (Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)
- [Whisper paper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Qwen2-Audio paper](https://arxiv.org/abs/2407.10759)
- [SALMONN paper](https://arxiv.org/abs/2310.13289)
- [Qwen3-ASR-1.7B HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ASR-0.6B HuggingFace](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ForcedAligner-0.6B HuggingFace](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [NVIDIA NeMo — Conformer 实现](https://github.com/NVIDIA/NeMo)
- [阿里 FunASR (Paraformer / SenseVoice)](https://github.com/modelscope/FunASR)
- [前篇：语音模型预处理流程及常用术语详解](/2026/05/07/speech-model-preprocessing-glossary.html)

---

> **一句话总结**：十四年 ASR 演进的主轴只有一条——**端到端的边界不断扩大**，从手工特征 / 发音词典 / n-gram LM 一个接一个被模型吞并。Qwen3-ASR 是这条轴上 2026 年的最新一点，而不是终点。
