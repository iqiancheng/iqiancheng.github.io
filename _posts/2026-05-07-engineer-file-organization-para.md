---
layout: post
title: "工程师的 Downloads 目录整理方法论：PARA + 时间线 + 工程实践"
date: 2026-05-07 00:00:00 +0800
author: Joseph
categories: [tools]
tags: [tooling, workflow]
---
## 先安利一个工具：HoudahSpot + Raycast

在讲方法论之前，先推荐一个我用下来真心觉得 **Powerful** 的 macOS 工具——[**HoudahSpot**](https://www.houdah.com/houdahSpot/)。

它本质上是 Spotlight 的超集，但把 Spotlight 隐藏的能力全部释放出来了：

- **布尔 + 元数据组合查询**：`kind:pdf AND author:"Tiago Forte" AND created:>2024`
- **精细范围限定**：可以把搜索域锁到 `~/Downloads/02_领域_学习/` 子树，避免全盘扫描
- **实时预览 + 内容命中高亮**：不用打开文件就知道命中了哪一段
- **保存查询 Template**：比如 "最近 7 天下载的 pptx"、"文件名含『评审』的 docx"，一键复用
- **全文搜索加密文件内容**（前提是解密后）、Office/PDF/代码文件皆可

再配合 [**Raycast**](https://www.raycast.com/) 调用，效率 max：

![HoudahSpot Raycast Extension](https://www.raycast.com/api/extension-og?handle=felixthehat&name=houdahspot-search)


- 给 HoudahSpot 绑一个 Raycast hotkey（或用 Raycast 的 Script Command 直接调起保存好的 Template）
- Raycast 自身的 File Search、Clipboard History、Window Management 补齐 Spotlight 不擅长的部分
- 两者搭配：**Raycast 做轻量快速跳转，HoudahSpot 做重度检索**

有了这套组合拳后，即使 `~/Downloads` 里堆了几千个文件，我也不再焦虑"找不到"；**真正的瓶颈从"检索"变成了"归档时的分类心智负担"**——这也是下面这篇方法论想解决的核心问题。

> TL;DR: **分类系统负责"让未来的你少做决定"，搜索工具负责"让当下的你少花时间"**。两者缺一不可。

---

## 起因

`~/Downloads` 长期是工程师的"地质层"：几年前下载的论文、临时克隆的开源仓库、各种 `.pptx`、壁纸、浏览器存储快照、以 hash 命名的不知所云的目录……一眼望去有一百多个顶层条目，找任何东西都要 `Cmd+F`。

这篇记录我用来治理它的一套方法——**PARA + 时间线 + 工程师专属子分类**，以及配套的 SOP。核心目标只有一个：**快速找到需要的信息**，而不是追求完美的分类体系。

---

## 一、核心框架：PARA

出自 Tiago Forte 的 *Building a Second Brain*。按"可执行性"分层：

![PARA — Organize by Actionability](https://fortelabs.com/wp-content/uploads/2019/02/20-Organize-by-Actionability-1-1536x1241.png)
*图：PARA 的核心是按"可执行性 (Actionability)"而非"类型 (Topic)"组织信息——离行动最近的在最上层。来源：Forte Labs*


| 层级 | 目录前缀 | 含义 | 工程师映射 |
|------|---------|------|-----------|
| **P**rojects | `01_` | 有明确截止目标的短期项目 | 当前迭代、技术评审、方案设计 |
| **A**reas | `02_` | 长期维护的责任领域 | 技术深度积累、团队管理 |
| **R**esources | `03_` | 可复用的参考资料 | 工具、代码片段、配置模板 |
| **A**rchives | `04_` | 已结束或暂停的内容 | 按年份归档 |

在此基础上加两个工程实践目录：

| 目录 | 作用 |
|------|------|
| `00_收件箱_待整理` | 新下载文件的临时缓冲区 |
| `05_加密文件_待解密` | 企业加密文件集中处理区（DRM/EFS） |

---

## 二、目录命名规范

### 2.1 数字前缀（强制）

```
00_  收件箱 / 缓冲
01_  项目（活跃、有 deadline）
02_  领域（长期学习、无 deadline）
03_  资源（工具、模板、参考）
04_  归档（按年份）
05_  特殊（加密、待处理）
```

**为什么用数字前缀？**

- Finder / `ls` 自动按字典序排序，数字小的始终在最上
- 一眼识别优先级：`01` > `02` > `04`
- 和 [Johnny.Decimal](https://johnnydecimal.com/) 的编号哲学一致

### 2.2 命名格式

```
{序号}_{类别}_{状态/时间}

示例：
01_MoE性能优化_进行中
01_技术评审_0428
02_大模型与深度学习
04_归档_2025
```

---

## 三、工程师专属子分类

### 3.1 项目目录 `01_项目_进行中`

按**当前工作流**组织，每个项目一个子目录，里面按产出物类型分层：

```
01_项目_进行中/
├── 训练框架性能优化_迭代3/
│   ├── 方案设计/
│   ├── 实验数据/
│   ├── 评审材料/
│   └── 会议纪要/
├── 技术评审_图像模型_0218/
│   ├── 评审PPT/
│   ├── 反馈记录/
│   └── 修改版本/
└── 团队OKR_Q2/
    ├── 目标拆解/
    └── 周报汇总/
```

**项目完结 checklist：**

- [ ] 产出物已同步到团队仓库 / 文档系统
- [ ] 有价值的过程文档迁入 `02_领域_学习`
- [ ] 临时文件、中间版本删除
- [ ] 目录整体移入 `04_归档_{年份}`

### 3.2 领域目录 `02_领域_学习`

按**技术栈**组织，长期积累。下面是我当前的划分：

```
02_领域_学习/
├── 大模型与深度学习/
│   ├── 论文精读/              # 论文 PDF + pdfresizer 提取的图
│   └── 开源仓库与PR/          # clone 下来的 repo、PR 分析快照
├── 分布式训练与性能优化/
│   ├── Profiling日志/         # nsys-rep、chrome trace、pid log
│   └── ...                   # CUDA、FSDP、加速脚本、NVIDIA 技术分享
├── 文生图与多模态/
│   └── 代码与实验/
├── 语音技术/
├── Agent与工具链/
│   └── 聊天记录导出/          # ChatGPT / Claude / Cursor 的 history export
├── 教练与管理力/
├── 专利与技术文档/
└── 个人成长/
    └── 简历与演讲/
```

**领域积累原则：**

- 读完的论文、博客放入对应子目录；**整个 `论文ID_标题/` 目录为一个不可拆分的单元**（论文 + pdfresizer 提取图 + 读书笔记一起走）
- 代码片段用 `snippet_描述.sh/py` 命名
- 定期（每季度）回顾，过时内容迁入归档

![Progressive Summarization](https://fortelabs.com/wp-content/uploads/2019/02/24-Progressive-Summarization.png)
*图：Progressive Summarization —— 领域目录里的资料不是"存起来就完事"，而是分层提炼（原文 → 高亮 → 粗体 → 摘要 → 再创作）。读过 ≠ 吸收，提炼过才是自己的。来源：Forte Labs*

### 3.3 资源目录 `03_资源_工具`

按**工具类型**组织，强调可复用：

```
03_资源_工具/
├── 代码与配置/            # shell、vimrc、ssh_config、dockerfile
├── 数据集/                # 评测集、训练样本
├── 软件安装包/            # .app、.dmg、installer zip
├── 字体与图标/            # fontawesome、favicon、icon PNG
├── 图片素材与壁纸/        # wallpapers、keynote 素材
├── 网页存档/              # SiteSucker / wget -m 的离线页面
└── 浏览器存储与扩展/      # .crx、Chrome 扩展 ID 目录
```

### 3.4 归档目录 `04_归档_{年份}`

按**年份**平铺，内部可按主题再分两个大筐：

```
04_归档_2024/
├── browser_storage_快照/     # 2024 年抓取的站点 storage
├── 杂项旧项目/               # 无语义目录（hash 命名、data/、common/）
└── ...                       # 按修改时间自然沉淀的项目目录
```

**归档原则：**

- 项目结束后**整目录移动**，不拆散文件
- 归档内不再精细整理，需要时靠搜索（Spotlight / ripgrep）
- 每年初清理一次，删除明显过时的内容

![How to Organize — Delete](https://fortelabs.com/wp-content/uploads/2019/02/23-How-to-Organize%E2%80%94delete-768x1121.png)  
*图：整理的一半是"删除"。归档不是仓库，而是过滤器——过时的就该被清掉，而不是无限堆积。来源：Forte Labs*

---

## 四、文件命名规范

### 4.1 日期前缀（推荐）

```
{YYYYMMDD}_{描述}_{版本}.{ext}

20240506_训练加速方案_v2.pdf
20240415_分布式训练踩坑记录.md
```

### 4.2 版本控制

```
文件_v1.pdf       # 初稿
文件_v2.pdf       # 修改稿
文件_最终版.pdf   # 定稿（避免 "最终最终版"、"真的最终版"）
文件_{日期}.pdf   # 按日期版本
```

### 4.3 特殊标记

| 标记 | 含义 |
|------|------|
| `[WIP]` | Work In Progress |
| `[DONE]` | 已完成 |
| `[REF]` | 参考资料，非原创 |
| `[TEMP]` | 临时文件，可删除 |

---

## 五、工作流 SOP

### 5.1 每日 / 每周

```
1. 新下载文件 → 先放 00_收件箱_待整理
2. 周五下午  → 清空收件箱（15 分钟）
   - 判断：这是项目文件？领域资料？工具资源？
   - 移动到对应目录
   - 重命名为规范格式
```

### 5.2 每月

```
1. 检查 01_项目_进行中
   - 已完结的项目 → 移入 04_归档
   - 长期停滞的项目 → 评估是否归档

2. 检查 02_领域_学习
   - 清理过时内容
   - 补充新学到的知识点
```

### 5.3 每年

```
1. 创建新的年份归档目录 04_归档_{新年份}
2. 上一年归档整体审查，删除无用内容
3. 备份重要归档到外部存储 / NAS
```

---

## 六、落地原则（踩过的坑）

### 6.1 整体搬移，不拆散"事项/话题"单元

一个 `Megatron-MoE-EA-moe_dev/` 是一个代码工程，一个 `anthropic-claude-chat-history-export-data-2026-0416/` 是一个话题快照——**整个目录为一个不可拆分的单元**。批量整理时千万别 `cd` 进去按后缀拆到各种地方，事后会完全找不回来。

### 6.2 语义不明的目录一律归档，不要留在活跃视野

以下都是我的 Downloads 里真实出现过的"废料"：

- `2e029001587b175df45377476ac9bf3e-*` —— 某个 commit hash 下载
- `data/`、`common/`、`assets/` —— 某项目的残骸
- `crx`、`crx 2` —— Chrome 扩展解压过两次

它们一律丢进 `04_归档_{年份}/杂项旧项目/`。如果未来需要，grep 一下就能捞出来；如果永远不需要，它们也不会污染 `02_领域_学习` 的心智负担。

### 6.3 按"语义"分类，而不是按"文件类型"

**反模式**：`图片/`、`文档/`、`压缩包/`。

**正确做法**：`02_领域_学习/分布式训练与性能优化/Profiling日志/`，里面可以同时有 `.nsys-rep`、`.log`、`.png` 截图、`.md` 笔记。语义相关的文件在一起才有价值。

### 6.4 批量操作时用 `bash + nullglob`，别用 zsh 默认行为

这次整理我踩的一个坑：写了一个 pattern 列表的 bulk move 脚本，在 zsh 下 glob 失败静默跳过。切到 `bash -c 'shopt -s nullglob; ...'` 后才正常。类似：

```bash
bash <<'EOF'
shopt -s nullglob
move_all() {
  local dest="$1"; shift
  for pat in "$@"; do
    for f in $pat; do
      [ -e "$f" ] && mv -- "$f" "$dest/"
    done
  done
}
move_all "02_领域_学习/专利与技术文档" "T20*" "*专利*" "*Patent*"
EOF
```

### 6.5 加密文件单独一个筐

企业 EFS / DRM 加密文件在 `file` 命令下识别为 `data`，解密后才是 `Zip archive` / `Microsoft OOXML`。整理时：

1. 所有加密文件先汇总到 `05_加密文件_待解密/`
2. 批量解密后，用 `file` 筛出已解密的移走：

```bash
find . -maxdepth 1 -type f \( -name "*.pptx" -o -name "*.docx" -o -name "*.xlsx" \) | \
  while read f; do
    file -b "$f" \| grep -qE 'Zip archive\|Microsoft' && echo "DECRYPTED: $f"
  done
```

3. 剩下还是 `data` 的继续留在 `05_` 等下一轮解密。

---

## 七、工具推荐

| 场景 | 工具 | 用法 |
|------|------|------|
| 快速搜索 | `ripgrep` (rg) | `rg "关键词" ~/Downloads/02_领域_学习` |
| 智能跳转 | `zoxide` (z) | `z 02` 直接跳到领域学习目录 |
| 文件预览 | `fzf` | 配合 rg 做交互式搜索 |
| 批量重命名 | `rename` / `mmv` | 统一加日期前缀 |
| 定时整理 | cron + shell | 每小时自动分类新文件 |

---

## 八、反模式清单

| ❌ 反模式 | ✅ 正确做法 |
|-----------|-----------|
| 按文件类型分类（图片 / 文档 / 视频） | 按项目 / 领域分类 |
| 多层嵌套目录（超过 3 层） | 扁平化，靠搜索定位 |
| "新建文件夹 (2)" 这类命名 | 日期 + 描述命名 |
| 桌面堆满临时文件 | 全部先放收件箱 |
| 多个 "最终版" 文件 | 用日期或 v1/v2 版本号 |
| 从不删除 / 归档 | 项目结束立即归档 |
| 按"文件类型"横切归档（.pptx 全放一起） | 按"话题/事项"保持单元完整 |
| 所有文件堆在 Downloads 根目录 | 每月至少 review 一次 |

---

## 九、最终结构长这样

```
~/Downloads/
├── 00_收件箱_待整理/          # < 20 个条目，每周清零
├── 01_项目_进行中/            # 当前在做的事
├── 02_领域_学习/
│   ├── 大模型与深度学习/
│   ├── 分布式训练与性能优化/
│   ├── 文生图与多模态/
│   ├── 语音技术/
│   ├── Agent与工具链/
│   ├── 教练与管理力/
│   ├── 专利与技术文档/
│   └── 个人成长/
├── 03_资源_工具/
│   ├── 代码与配置/
│   ├── 数据集/
│   ├── 软件安装包/
│   ├── 字体与图标/
│   ├── 图片素材与壁纸/
│   ├── 网页存档/
│   └── 浏览器存储与扩展/
├── 04_归档_2024/              # 按年份沉淀，内部粗分 browser_storage / 杂项旧项目
├── 04_归档_2025/
├── 04_归档_2026/
└── 05_加密文件_待解密/        # 企业加密文件的隔离区
```

---

## 十、参考资源

- **PARA 方法**: Tiago Forte — [Building a Second Brain: An Overview](https://fortelabs.com/blog/basboverview/)
- **文件命名**: [Johnny.Decimal](https://johnnydecimal.com/) 编号系统
- **工程师效率**: [*The Pragmatic Programmer*](https://en.wikipedia.org/wiki/The_Pragmatic_Programmer) — 源码组织章节

---

> **整理是手段，不是目的。** 目标是快速找到需要的信息，而不是完美的分类。  
> 如果你的体系让你花 30 分钟决定一个文件该放哪，那它本身就是反模式。
