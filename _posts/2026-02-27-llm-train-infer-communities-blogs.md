---
layout: post
title: "大模型训推高质量社区、技术博客与资源索引"
date: 2026-02-27 10:00:00 +0800
author: Joseph
tags: [inference, serving, methodology, llm]
---

> 大模型训练与推理相关的高质量社区、技术博客、官方资源索引。按类型分类，表格形式便于快速查阅。

---

## 一、厂商 / 研究院官方博客

| 来源 | 链接 | 侧重 |
|-----|------|------|
| NVIDIA Technical Blog | [developer.nvidia.com/blog](https://developer.nvidia.com/blog/) | GPU 训练、TensorRT-LLM、推理优化、Megatron |
| Meta AI | [ai.meta.com/blog](https://ai.meta.com/blog/) | Llama 系列、开源模型、训练与部署 |
| PyTorch Forums | [discuss.pytorch.org](https://discuss.pytorch.org/) | PyTorch 的社区 |
| Hugging Face | [huggingface.co/blog](https://huggingface.co/blog) | Transformers、Accelerate、TGI、推理部署 |
| Google AI | [ai.googleblog.com](https://ai.googleblog.com/) | TPU、JAX、高效训练、Gemma |
| Anthropic | [anthropic.com/news](https://www.anthropic.com/news) | 扩展律、可解释性、对齐与安全 |
| Microsoft Research | [research.blog](https://www.microsoft.com/en-us/research/blog/) | ZeRO、DeepSpeed、分布式训练、系统优化 |

---

## 二、框架 / 工具官方博客

| 来源 | 链接 | 侧重 |
|-----|------|------|
| vLLM | [blog.vllm.ai](https://blog.vllm.ai/) | 推理引擎、KV Cache、分布式推理、PagedAttention |
| DeepSpeed | [Microsoft Research 子站](https://www.microsoft.com/en-us/research/project/deepspeed/) | ZeRO 各阶段、3D 并行、大模型训练 |
| PyTorch | [pytorch.org/blog](https://pytorch.org/blog/) | FSDP、分布式、性能与生态 |
| ms-swift | [github.com/microsoft/ms-swift](https://github.com/microsoft/ms-swift) | 微软自研的Swift语言的LLM框架 |
| llama-factory | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 一个用于训练和推理LLM的工具 |
| Pai-Megatron-Patch | [github.com/microsoft/Pai-Megatron-Patch](https://github.com/microsoft/Pai-Megatron-Patch) | 一个用于训练和推理LLM的工具 |
| PEFT | [github.com/huggingface/peft](https://github.com/huggingface/peft) | 后训练微调技术 |
| unsloth | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) | 一个用于训练和推理LLM的工具 |
| liger-kernel | [github.com/bytedance/Liger-Kernel](https://github.com/bytedance/Liger-Kernel) | 一个用于训练和推理LLM的工具 |
| OpenAI Triton Docs | [triton-lang.org/main/getting-started/tutorials/](https://triton-lang.org/main/getting-started/tutorials/) | OpenAI自研的 Triton 深度学习框架的文档 |
| Megatron-LM | [github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | NVIDIA自研的 Megatron-LM 深度学习框架的文档 |
| veRL | [github.com/verl-project/verl/](https://github.com/verl-project/verl/) | 一个用于训练和推理LLM的工具 |
| huggingface docs | [huggingface.co/docs](https://huggingface.co/docs) | Hugging Face 的文档 |

---

## 三、个人 / 社区技术博客

| 来源 | 链接 | 侧重 |
|-----|------|------|
| Lil'Log (Lilian Weng) | [lilianweng.github.io](https://lilianweng.github.io/) | LLM 原理、RLHF、Agent、幻觉与对齐 |
| Sebastian Raschka | [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/) | LLM 综述、前沿解读、《From Scratch》系列 |
| Eugene Yan | [eugeneyan.com](https://eugeneyan.com/) | 大模型应用、数据与工程实践 |
| 九原山 (ninehills) | [github.com/ninehills/blog](https://github.com/ninehills/blog) | LLM 学习路径、推理优化、Embedding 选型 |
| LLM 大模型训练之路 | [wqw547243068.github.io/llm_train](https://wqw547243068.github.io/llm_train) | 预训练、SFT、RLHF、开源模型训练流程 |
| 苏剑林 | [spaces.ac.cn](https://spaces.ac.cn/me.html) | 苏剑林的博客 |

---

## 四、社区与资讯

| 来源 | 链接 | 侧重 |
|-----|------|------|
| Hugging Face Forums | [discuss.huggingface.co](https://discuss.huggingface.co/) | 模型、数据集、训练与部署讨论 |
| r/LocalLLaMA | [reddit.com/r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) | 本地推理、量化、硬件与框架 |
| Papers with Code | [paperswithcode.com](https://paperswithcode.com/) | 论文 + 代码，按任务与模型检索 |
| GPU Mode (YouTube) | [youtube.com/@gpumode](https://www.youtube.com/@gpumode) | GPU、推理与训练工程架构 |

---

## 五、代表性单篇（训推系统向）

| 标题 | 链接 |
|-----|------|
| ZeRO & DeepSpeed: Training 100B+ Models | [Microsoft Research](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) |
| The Ultra-Scale Playbook (HuggingFace) | [ultrascale-playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) |
| Introducing PyTorch Profiler – the new and improved performance tool – PyTorch | [pytorch.org/blog/introducing-pytorch-profiler](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/) |
| Profiling to understand torch.compile performance — PyTorch 2.10 documentation | [pytorch.org/docs/stable/profiler.html](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html) |
| Understanding GPU Memory 1: Visualizing All Allocations over Time – PyTorch | [pytorch.org/blog/understanding-gpu-memory-1](https://pytorch.org/blog/understanding-gpu-memory-1/) |
| Understanding GPU Memory 2: What Is CUDA Kernel Memory? – PyTorch | [pytorch.org/blog/understanding-gpu-memory-2](https://pytorch.org/blog/understanding-gpu-memory-2/) |
| Visualize and understand GPU memory in PyTorch | [huggingface.co/blog/train_memory](https://huggingface.co/blog/train_memory) |
| Automated trace collection and analysis – PyTorch | [pytorch.org/blog/automated-trace-collection-and-analysis/](https://pytorch.org/blog/automated-trace-collection/) |
| Slaying OOMs with PyTorch FSDP and torchao | [parlance-labs.com/education/fine_tuning/slaying_ooms.html](https://parlance-labs.com/education/fine_tuning/slaying_ooms.html) |
| Out of Memory (OOM) — Oumi AI | [oumi.ai/docs/en/latest/faq/oom.html](https://www.oumi.ai/docs/en/latest/faq/oom.html) |
| Inside vLLM: Anatomy of Inference System | [blog.vllm.ai](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html) |
| Mastering LLM Inference Optimization (NVIDIA) | [developer.nvidia.com](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization) |

---

*按需订阅 RSS 或 GitHub Watch，保持更新。*
