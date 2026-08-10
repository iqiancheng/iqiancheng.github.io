---
layout: post
title: "语音模型预处理流程及常用术语详解：从声波到 Log-Mel 到 Audio-LLM"
date: 2026-05-07 00:00:00 +0800
author: Joseph
categories: [语音, 深度学习]
tags: [speech]
---

## 为什么要专门写一篇"预处理 + 术语表"

开源语音模型（Whisper / Conformer / Paraformer / SenseVoice / Qwen2-Audio）的配置文件里动辄出现 `n_mels=80`、`hop_length=160`、`win_length=400`、`sample_rate=16000`、`fbank`、`CMVN`、`SpecAugment`、`frame shift` 这些名词——如果对每一项都只有"大概知道"，调参、排错、换模型时就会反复掉坑里。

这篇把 **语音 → 数字 → 特征 → 模型输入** 的完整链路串一遍，夹带一份工程师向的术语词典。

![Waveform + Spectrogram + Transcription of "Wikipedia"](https://upload.wikimedia.org/wikipedia/commons/6/69/Waveform_spectrogram_and_transcription_of_wikipedia_in_praat.png)
*图 0：同一段"Wikipedia"发音的三层视图——上：时域波形；中：宽带语谱图（能看到共振峰）；下：音素级转录。一整条 ASR 流水线做的事就是把"上层的波形"映射到"下层的文本"。来源：Wikimedia Commons*

目标读者：

- 刚开始做 ASR / TTS / Audio-LLM 调优的开发者
- 在工程里接手"别人写的 `preprocess.py`"但读不顺的人
- 想从传统 ASR 管线迁移到大模型端到端路线的同学

---

## 一、从声波到数字：音频的物理与数字基础

### 1.1 声波三要素

声音的本质是**介质中质点做机械振动产生的疏密波**。描述一段音频的三个原始物理量：

| 物理量 | 直觉 | 数字域对应 |
|---|---|---|
| 频率 (Hz) | 音调高低 | 采样率需 ≥ 2× 最高频率（**奈奎斯特定理**） |
| 振幅 | 音量大小 | 位深 / 量化精度 |
| 持续时间 | 音长 | 样本数 ÷ 采样率 |

人耳频率范围 20 Hz ~ 20 kHz；人声基频主要集中在 80 Hz ~ 8 kHz，这也是**电话采样率 8 kHz、ASR 通用采样率 16 kHz** 的物理来源。

### 1.2 采样 & 量化 = PCM

连续模拟信号通过两步变成数字：

1. **采样 (Sampling)**：按固定时间间隔取样，间隔的倒数就是**采样率 (sample rate)**
2. **量化 (Quantization)**：把每个采样点的幅度值映射到有限的离散数值，决定**位深 (bit depth)**

两步合起来叫 **PCM (Pulse Code Modulation)**。几个常见组合：

| 场景 | 采样率 | 位深 | 声道 | 带宽 |
|---|---|---|---|---|
| 电话 | 8 kHz | 16 bit | mono | 128 kbps |
| ASR / TTS 标配 | 16 kHz | 16 bit | mono | 256 kbps |
| 视频会议 | 48 kHz | 16 bit | mono/stereo | 768+ kbps |
| 音乐 CD | 44.1 kHz | 16 bit | stereo | 1411 kbps |
| 录音棚 / 高保真 | 48 kHz | 24 bit | stereo | 2304 kbps |

**关键事实**：几乎所有**学术 ASR 模型和大多数开源语音大模型都默认 16 kHz + 16 bit + mono**。拿到 44.1 kHz 的 mp3 或 48 kHz 的立体声录音，第一步必然是重采样。

### 1.3 常见音频容器与编码

| 扩展名 | 编码 | 是否有损 | 备注 |
|---|---|---|---|
| `.wav` | 通常 PCM | 否 | 体积大，但加载最快，ASR 训练首选 |
| `.flac` | FLAC | 否（无损压缩） | 约 50% 体积，解码略慢 |
| `.mp3` | MPEG-1 Audio Layer III | 是 | 128 kbps 以下明显损失高频 |
| `.opus` / `.ogg` | Opus | 是 | WebRTC 主流，8~510 kbps 自适应 |
| `.m4a` / `.aac` | AAC | 是 | 苹果生态，iPhone 录音默认 |

**工程建议**：训练数据预处理时一律先解码到 PCM（用 `ffmpeg` / `soundfile` / `torchaudio.load`），后续流程不再纠结编码差异。

---

## 二、预处理流水线（Waveform → Model Input）

下面这张流程图覆盖 95% 的语音模型管道：

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
│ 读取文件  │→ │ 重采样    │→ │ 单声道化  │→ │ 端点/切分  │→ │ 归一化    │
│ .wav/.mp3│   │ 16 kHz   │   │ stereo→  │   │ VAD/trim  │   │ peak/mean│
└──────────┘   └──────────┘   │   mono   │   └───────────┘   └──────────┘
                               └──────────┘          │
                                                     ▼
┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────────┐
│ 模型输入  │← │SpecAugment│← │ Log/CMVN  │← │ STFT → Mel     │
│ 80×T     │   │ (训练)   │   │ 归一化    │   │ 滤波器组       │
└──────────┘   └──────────┘   └───────────┘   └────────────────┘
```

### 2.1 加载 & 重采样

```python
import torchaudio
# torchaudio.load 默认返回 [channels, time] 的 float32 tensor，范围 [-1, 1]
wav, sr = torchaudio.load("input.mp3")    # 例如 sr=44100, shape=[2, N]
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000
```

**重采样算法选择**：`sinc_interp_kaiser`（默认）质量最好但慢；`sinc_interp_hann` 在大批量数据预处理时更划算。项目里如果跑一次就要预处理 TB 级数据，**离线批量 ffmpeg -ar 16000 单次搞完**，比在 `Dataset.__getitem__` 里实时重采样快 10×。

### 2.2 单声道化

```python
if wav.shape[0] > 1:
    wav = wav.mean(dim=0, keepdim=True)   # 双声道取平均
```

有些语音增强任务（说话人分离、AEC 回声消除）会保留多声道，但 ASR/TTS/Audio-LLM 几乎清一色 mono。

### 2.3 时长对齐：pad / trim / chunk

模型对输入长度的假设不同：

- **Whisper**：硬性 30 秒窗口，短的 pad 静音，长的切片
- **Conformer / Paraformer**：变长（attention mask），但会有 max_length（通常 30~60 秒）
- **Wav2Vec2 / HuBERT**：变长，但 CNN 前端对极长音频会爆显存

工程实现：

```python
target_len = 30 * sr   # 480000 samples at 16k
if wav.shape[1] < target_len:
    wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
else:
    wav = wav[:, :target_len]
```

长音频的分片有两种思路：
- **均匀切分**（Whisper `transcribe()` 默认）：按 30s 切，句子会被截断
- **VAD 切分**：用 silero-vad / pyannote 的 VAD 先找静音点再切，语义更完整

### 2.4 端点检测（VAD）与降噪

- **VAD (Voice Activity Detection)**：判断每个时间段是否有人声。开源推荐 [silero-vad](https://github.com/snakers4/silero-vad)，单文件模型，CPU 也能实时
- **降噪 (Denoise)**：经典方法如 spectral subtraction，现代方法用 [RNNoise](https://github.com/xiph/rnnoise)、DeepFilterNet、FRCRN
- **AEC (Acoustic Echo Cancellation)**：实时通话场景常备，RTC 厂商如声网/腾讯云/WebRTC 都有 SDK
- **去混响 (Dereverberation)**：会议室、远场录音的关键，WPE 算法 + 神经网络后处理

### 2.5 振幅归一化

三种常见做法：

```python
# 1. Peak normalization: 把最大振幅缩放到 1.0
wav = wav / wav.abs().max()

# 2. RMS normalization: 基于能量的归一化，更抗 outlier
rms = (wav**2).mean().sqrt()
wav = wav * (target_rms / rms)

# 3. LUFS normalization: 符合 EBU R128 广播标准，用于 TTS 数据
# pip install pyloudnorm
```

**为什么要归一化**：不同录音设备、距离、增益导致音量差 20dB 以上是常态；不归一化训练，模型会把"音量大"和"强类别"错误关联。

---

## 三、时频变换：从波形到 Mel-Spectrogram

直接喂 16k 采样的原始波形给 Transformer 不现实——1 秒就是 16000 个点。必须压缩成**时频表示 (time-frequency representation)**。

### 3.1 STFT（短时傅里叶变换）

思路：把长信号切成短帧（**分帧 framing**），每帧做一次 FFT，得到"每个时刻 × 每个频率"的二维复数矩阵。

关键超参：

| 参数 | 典型值 (16 kHz) | 物理含义 |
|---|---|---|
| `win_length` / frame size | 400 (25 ms) | 每帧多长 |
| `hop_length` / frame shift | 160 (10 ms) | 相邻帧之间的步长（≈ 帧率 100 fps） |
| `n_fft` | 400 或 512 | FFT 点数，通常 = win_length 或更大的 2 的幂 |
| 窗函数 | Hann / Hamming | 减少频谱泄漏 |

**帧率直觉**：hop=10ms 意味着**每秒产生 100 帧特征**，30 秒音频就是 3000 帧——这是大多数 ASR 模型的时间分辨率基准。

![STFT colored spectrogram with 25ms window](https://upload.wikimedia.org/wikipedia/commons/0/0f/STFT_colored_spectrogram_25ms.png)
*图 1：25ms 分帧 + STFT 得到的彩色语谱图。横轴是时间，纵轴是频率，颜色表示能量。分帧越短时间分辨率越高、频率分辨率越低（反之亦然）——这是"测不准"式的 trade-off。来源：Wikimedia Commons*

### 3.2 功率谱 / 幅度谱

STFT 输出是复数 `X(t, f) = a + bi`，两种取实数表示：

- **幅度谱 Magnitude**: `|X| = sqrt(a² + b²)`
- **功率谱 Power**: `|X|² = a² + b²`

相位信息被丢弃——对 ASR 无所谓，但对 vocoder（声码器，如 HiFi-GAN）就是致命的，所以 TTS 后端常常要回归相位或用 Griffin-Lim 估计。

### 3.3 Mel 滤波器组（Mel Filter Bank）

人耳对低频敏感、对高频不敏感（对数感知）。Mel 刻度就是这种感知的数学化：

```
mel(f) = 2595 · log₁₀(1 + f / 700)
```

![Mel vs Hz plot](https://upload.wikimedia.org/wikipedia/commons/a/aa/Mel-Hz_plot.svg)
*图 2：Mel 刻度 vs 物理频率 Hz 的非线性映射。低频（<1kHz）两者几乎线性，高频区 Mel 迅速饱和——这恰好匹配人耳"听不出 8kHz 和 9kHz 区别但能分辨 200Hz 和 300Hz"的感知特点。来源：Wikimedia Commons*

**梅尔滤波器组**：一组三角形窗，低频密、高频疏，覆盖 [0, Nyquist] 范围。典型参数：

| 配置 | `n_mels` | 覆盖范围 |
|---|---|---|
| Whisper | 80 | 0 ~ 8000 Hz |
| Conformer / 大多数 ASR | 80 | 0 ~ 8000 Hz |
| 老式 MFCC-based | 40 | 0 ~ 8000 Hz |
| 高保真 TTS | 128 | 0 ~ 12000 Hz |

流程：`功率谱 (F/2+1 维) × Mel 滤波器组 (n_mels × F/2+1) = Mel 谱 (n_mels 维)`

### 3.4 Log-Mel Spectrogram（最常用的 ASR 输入）

```python
log_mel = torch.log(mel_spec + 1e-10)
```

取对数的动机：**语音能量动态范围极大**（静音 vs 爆破音差 6 个数量级），log 能把分布拉平到模型好学的区间。

**这就是今天 90% 的语音模型真正吃进去的输入：shape 为 `[batch, n_mels=80, T]` 的 log-mel。**

### 3.5 MFCC（梅尔频率倒谱系数）

在 log-mel 基础上再做一次 **DCT（离散余弦变换）**，取前 13~20 个系数，就是 MFCC。

```python
MFCC = DCT(log(mel(|STFT(x)|²)))[:13]
```

历史上 MFCC 是 HMM-GMM 时代的王者（解耦 + 压缩），但深度学习时代**逐渐被 Log-Mel 取代**——因为 CNN/Transformer 不需要人为解耦，原始的 80 维 log-mel 信息更全。

今天还在用 MFCC 的场景：
- 关键词唤醒（KWS）：模型小，MFCC 省算力
- 说话人识别里的 x-vector 特征前端
- 一些嵌入式 SDK

### 3.6 Fbank（Filter Bank）

在 Kaldi 社区的习惯里，**Fbank = log-mel spectrogram**（有时也叫 log filter bank energies）。两者所指特征几乎一致，只是社区命名不同：

- HuggingFace / 学术论文：`log-mel`
- Kaldi / Espnet / 工业 ASR：`fbank`

### 3.7 CMVN（Cepstral/Mel Mean-Variance Normalization）

每个 Mel 维度独立做零均值单位方差归一化：

```python
fbank = (fbank - fbank.mean(dim=-1, keepdim=True)) / (fbank.std(dim=-1) + 1e-5)
```

- **Utterance-level CMVN**：每句话单独算均值方差（在线场景常用）
- **Global CMVN**：用整个训练集统计量（Espnet / Kaldi 传统做法）
- Whisper、Wav2Vec2 这类大模型一般**内置 LayerNorm 替代 CMVN**，不需要再手动做

---

## 四、数据增强（Data Augmentation）

只有原始干净录音不够，模型会对"安静房间里标准口音男声"过拟合。

### 4.1 波形域增强

| 方法 | 作用 | 典型强度 |
|---|---|---|
| Speed Perturbation | 0.9x / 1.0x / 1.1x 三种速度重采样 | 训练集变 3× |
| Pitch Shift | 音高上下 ±2 半音 | 抗说话人变化 |
| Volume / Gain | 音量随机 ±6dB | 抗设备差异 |
| 加性噪声 | 混入 MUSAN / DEMAND 噪声集 | SNR 5~20 dB |
| 混响 (RIR) | 卷积房间冲激响应 | 模拟远场 |
| Codec simulation | 模拟 mp3/opus 压缩损失 | 线上/训练分布对齐 |

### 4.2 谱域增强：SpecAugment（Google 2019）

不是增强波形，而是直接**在 log-mel 上挖空**：

- **Time Mask**：随机把连续 T 帧（横向条）置零或均值
- **Frequency Mask**：随机把连续 F 个 mel bin（纵向条）置零
- **Time Warping**：时间轴非线性扭曲（开销大，很多实现省了）

代码极短，效果惊人——Conformer 在 LibriSpeech 上 WER 下降 1~2 个绝对点主要靠它。

```python
# torchaudio.transforms.FrequencyMasking / TimeMasking
aug = torch.nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
    torchaudio.transforms.TimeMasking(time_mask_param=100),
)
log_mel_aug = aug(log_mel)
```

**实战提示**：SpecAugment 只在训练时开，推理时关。

---

## 五、主流模型的预处理"实际长什么样"

| 模型 | 输入形式 | 采样率 | 特征维度 | 帧率 | 归一化 |
|---|---|---|---|---|---|
| **Whisper** (OpenAI) | Log-Mel | 16k | 80 | 100 fps | 在模型内部做 |
| **Conformer-CTC/AED** (Espnet/NeMo) | Log-Mel / Fbank | 16k | 80 | 100 fps | Global CMVN |
| **Paraformer / SenseVoice** (阿里 FunASR) | Fbank | 16k | 80 | 100 fps | Utterance CMVN |
| **Wav2Vec2 / HuBERT / WavLM** | **原始波形** | 16k | 1 (raw) | 50 fps (CNN 降采样后) | mean-var norm on waveform |
| **Qwen2-Audio / SALMONN** | Whisper encoder → audio tokens | 16k | 80→1280 | 50 fps (2× 下采样) | 同 Whisper |

**两条路线**：

1. **手工特征派 (Log-Mel / Fbank)**：Whisper、Conformer、Paraformer——预处理重、模型可以更小
2. **端到端原始波形派 (raw waveform)**：Wav2Vec2、HuBERT——预处理轻、模型前端有 7 层 CNN 做隐式特征提取

Audio-LLM 时代的主流还是手工特征派，但把 Whisper encoder 的输出当作"audio tokens"喂给 LLM decoder（Qwen2-Audio、Gemini Audio、Phi-4-Multimodal 都是这个范式）。

---

## 六、术语表（按主题分组速查）

### 6.1 信号基础

- **Sample Rate (采样率 / fs)**：每秒采样点数，Hz
- **Bit Depth (位深)**：单个采样点用多少 bit 表示，决定信噪比上限（16-bit ≈ 96 dB SNR）
- **PCM**：Pulse Code Modulation，未压缩的原始数字音频
- **SNR (Signal-to-Noise Ratio)**：信噪比，dB
- **dBFS**：Decibel Full Scale，数字音频中相对满量程的分贝，0 dBFS = 最大值
- **Nyquist Frequency**：采样率的一半，能表达的最高频率
- **Mono / Stereo**：单/双声道

### 6.2 时频变换

- **STFT**：Short-Time Fourier Transform，短时傅里叶变换
- **Window Function**：窗函数，Hann / Hamming / Blackman
- **Frame / Chunk**：一段用来做 FFT 的短音频（典型 25ms）
- **Hop Length / Frame Shift**：相邻帧之间的间隔（典型 10ms）
- **n_fft**：FFT 变换的点数
- **Spectrogram**：语谱图（时间 × 频率的 2D 图）
- **Magnitude / Power Spectrum**：幅度谱 / 功率谱
- **Phase**：相位信息（TTS 声码器需要）

### 6.3 特征

- **Mel Scale**：梅尔刻度，感知相关的非线性频率轴
- **Mel Filter Bank**：梅尔滤波器组
- **Log-Mel Spectrogram**：对 Mel 谱取对数，ASR 最常用输入
- **Fbank**：Filter Bank，Kaldi 社区对 log-mel 的称呼
- **MFCC**：Mel-Frequency Cepstral Coefficients，Mel 基础上的 DCT 系数
- **Delta / Delta-Delta (Δ, ΔΔ)**：MFCC 的一阶/二阶差分，捕获时间动态
- **CMVN**：Cepstral Mean-Variance Normalization，倒谱均值方差归一化
- **Pitch / F0**：基频
- **Formant**：共振峰，元音识别的关键

### 6.4 处理模块

- **VAD (Voice Activity Detection)**：语音活动检测
- **AEC (Acoustic Echo Cancellation)**：回声消除
- **ANS / Denoise**：噪声抑制
- **AGC (Automatic Gain Control)**：自动增益控制
- **Dereverberation**：去混响
- **Beamforming**：麦克风阵列波束成形
- **BSS (Blind Source Separation)**：盲源分离

### 6.5 建模与对齐

- **ASR**：Automatic Speech Recognition，自动语音识别
- **TTS**：Text-to-Speech，语音合成
- **STT**：Speech-to-Text，= ASR
- **KWS**：Keyword Spotting，关键词唤醒
- **CTC (Connectionist Temporal Classification)**：不需要显式对齐的序列损失
- **AED (Attention-based Encoder-Decoder)**：= Seq2Seq，用 attention 做软对齐
- **RNN-T (RNN Transducer)**：流式友好的对齐方案，Google/苹果偏爱
- **Forced Alignment**：强制对齐，已知转录去对齐时间戳，工具 MFA / Gentle
- **Blank Token**：CTC 词表里的特殊 "null" 符号
- **Beam Search**：束搜索解码
- **Greedy Decoding**：每步取最大概率的解码
- **LM / ILM / ELM**：Language Model / Internal LM / External LM
- **Shallow Fusion**：外挂 LM 的浅层融合打分

### 6.6 评估指标

- **WER (Word Error Rate)**：词错误率 = (S + D + I) / N，**英文/西语主指标**
- **CER (Character Error Rate)**：字错误率，**中文主指标**（因为中文没空格分词）
- **SER (Sentence Error Rate)**：句错误率
- **RTF (Real-Time Factor)**：处理 T 秒音频用时 / T，<1 表示能实时
- **Latency / First-Token Latency**：首字延迟，流式 ASR 关键指标
- **MOS (Mean Opinion Score)**：TTS/通话音质主观打分，1~5 分

### 6.7 词表与文本

- **Grapheme**：字形单位（英文字母、中文字）
- **Phoneme**：音素，语音的最小发音单位
- **Lexicon**：发音词典（graph → phone 映射）
- **BPE / SentencePiece**：子词切分
- **Subword Unit**：子词单元
- **Normalization**：文本归一化（数字、缩写、符号展开）
- **ITN (Inverse Text Normalization)**：逆文本归一化（"一百二十三" → "123"）

### 6.8 数据与数据集

- **LibriSpeech** (English, 1000 hours, 朗读)
- **CommonVoice** (多语言众包)
- **AISHELL-1/2/3** (中文普通话)
- **WenetSpeech** (中文 10000 小时)
- **GigaSpeech** (英文 10000 小时)
- **VCTK / LJSpeech** (TTS 单说话人)
- **MUSAN / DEMAND / WHAM!** (噪声增强集)

---

## 七、端到端 Python 例子：30 行内把一段 mp3 变成 Whisper 输入

```python
import torch, torchaudio
import torchaudio.transforms as T

SR = 16000
N_MELS = 80
WIN = 400      # 25ms
HOP = 160      # 10ms

# 1. 加载 + 重采样 + 单声道
wav, sr = torchaudio.load("sample.mp3")
if sr != SR:
    wav = T.Resample(sr, SR)(wav)
if wav.shape[0] > 1:
    wav = wav.mean(0, keepdim=True)

# 2. Peak normalize
wav = wav / wav.abs().max().clamp(min=1e-8)

# 3. 30 秒 pad/trim (Whisper-style)
target = 30 * SR
wav = wav[:, :target] if wav.shape[1] > target else \
      torch.nn.functional.pad(wav, (0, target - wav.shape[1]))

# 4. Log-Mel
mel_extractor = T.MelSpectrogram(
    sample_rate=SR, n_fft=WIN, win_length=WIN,
    hop_length=HOP, n_mels=N_MELS,
    f_min=0, f_max=SR // 2, power=2.0,
)
mel = mel_extractor(wav)              # [1, 80, 3001]
log_mel = torch.log(mel.clamp(min=1e-10))

# 5. 训练时的 SpecAugment
if training:
    log_mel = T.FrequencyMasking(27)(log_mel)
    log_mel = T.TimeMasking(100)(log_mel)

print(log_mel.shape)                  # torch.Size([1, 80, 3001])
# → 直接送进 Whisper encoder
```

**这 30 行就是现代主流 ASR/Audio-LLM 预处理的最小闭环**——从此看 Whisper / FunASR / Espnet 的 preprocess 源码就都能对上号。

---

## 八、Audio-LLM 时代的变化

2024 年后 Audio-LLM（Qwen2-Audio、SALMONN、Phi-4-Multimodal、Gemini 1.5/2.5 audio）的预处理有两个趋势：

1. **复用 Whisper Encoder 作为 audio tokenizer**：log-mel → Whisper encoder → 下采样 → audio embedding（等效于 audio token），直接拼到 LLM 的 prompt 里
2. **多任务统一前端**：同一个音频前端要同时支持 ASR / SER（情感） / SID（说话人） / 音乐理解，Mel-spec 够通用，波形前端（如 HuBERT）也能通过 LoRA 适配

**结论**：即使进入 Audio-LLM 时代，**Log-Mel + SpecAugment + CMVN 这套 2019 年就成熟的前端依然是主流**，底层变化不大，变化的是后端怎么"消费"这些特征。搞懂前面这些术语，看任何最新的 Audio-LLM 论文都不会再卡在第一页。

---

## 九、参考资料

- [Hugging Face Audio Course – Unit 5 ASR Models](https://huggingface.co/learn/audio-course/en/chapter5/asr_models)
- [NVIDIA – Essential Guide to Automatic Speech Recognition Technology](https://developer.nvidia.cn/blog/essential-guide-to-automatic-speech-recognition-technology/)
- [声网博客 – 音频深度学习入门五：ASR](https://www.shengwang.cn/blog/blogdetail/audio-deep-learning-5/)
- [HuggingFace – Audio Datasets Blog](https://huggingface.co/blog/audio-datasets)
- [SpecAugment 原始论文 (Park et al., 2019)](https://arxiv.org/abs/1904.08779)
- [Whisper Paper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Conformer Paper (Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)
- [Kaldi Feature Extraction 文档](https://kaldi-asr.org/doc/feat.html)
- [NVIDIA – Deploy Speech AI Model on GPU](https://developer.nvidia.cn/blog/deploy-speech-ai-model-on-gpu/)
- [A Step-by-Step Guide to Speech Recognition in Python](https://towardsdatascience.com/a-step-by-step-guide-to-speech-recognition-and-audio-signal-processing-in-python-136e37236c24/)
- [Kaggle – Audio Pre-processing Tutorial](https://www.kaggle.com/code/vuppalaadithyasairam/audio-pre-processing-tutorial-in-python)
- [aravindpai/Speech-Recognition Jupyter Notebook](https://github.com/aravindpai/Speech-Recognition/blob/master/Speech%20Recognition.ipynb)
- [Azure DataScienceVM – Deep Learning for Audio Tutorial](https://github.com/Azure/DataScienceVM/blob/main/Tutorials/DeepLearningForAudio/Deep%20Learning%20for%20Audio%20Part%201%20-%20Audio%20Processing.ipynb)

---

> **一句话总结**：搞懂 `sample_rate / win_length / hop_length / n_mels / CMVN / SpecAugment / CTC / WER` 这八个词就已经能看懂 95% 的语音模型配置文件——剩下 5% 是模型架构细节，属于另一篇要写的话题。
