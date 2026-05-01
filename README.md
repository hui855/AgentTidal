# AgentTidal — AI 长期记忆系统

> 让本地 AI 模型在**模型权重层面**拥有真正的长期记忆，不依赖上下文窗口。

[🌏 English](README_EN.md) | [🌏 中文](README.md)

---

> **⚠️ 免责声明：本项目完全由 AI 生成，仅供娱乐和学习参考。** 作者不对代码的正确性、安全性或适用性做任何保证。使用本项目所产生的任何后果由使用者自行承担。

---

---

## 目录

- [实现原理](#实现原理)
  - [核心思想](#核心思想)
  - [系统架构](#系统架构)
  - [数据流](#数据流)
  - [模块详解](#模块详解)
  - [反遗忘机制](#反遗忘机制)
  - [跨模型记忆继承](#跨模型记忆继承)
- [使用方法](#使用方法)
  - [环境要求](#环境要求)
  - [快速开始](#快速开始)
  - [启动代理（白天）](#启动代理白天)
  - [配置客户端](#配置客户端)
  - [模拟对话（测试用）](#模拟对话测试用)
  - [执行夜间处理](#执行夜间处理)
  - [Web 仪表盘](#web-仪表盘)
  - [安装计划任务](#安装计划任务)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 实现原理

### 核心思想

大语言模型在与用户交互时，所有"记忆"都依赖上下文窗口中的历史消息。这意味着：

- 每次对话都需要重新注入历史
- Token 消耗随记忆量线性增长
- 跨会话记忆需要外部数据库 + prompt 注入
- 更换模型时所有"记忆"丢失

AgentTidal 的核心思路是：**通过夜间自动微调，将白天的对话数据转化为模型的参数知识**。当模型在权重层面记住了用户的信息和偏好后，就不再需要依赖上下文来维持记忆。

### 系统架构

系统分为两个阶段运行：

```
┌─────────────────────────────────────────────────────────────┐
│                    白天 — 交互阶段                            │
│                                                             │
│  用户 ◄────► LM Studio (本地模型)                              │
│                   │                                          │
│                   ▼                                          │
│          AgentTidal Proxy (端口 1235)                          │
│                   │                                          │
│                   ▼                                          │
│          memory/short_term/YYYY-MM-DD.jsonl                   │
│                   (原始对话，按天分片)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                    (凌晨 2:00 自动触发)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    夜间 — 处理阶段                            │
│                                                             │
│   1. 数据清洗 (Cleaner)                                      │
│      ├─ 解析原始 JSONL                                       │
│      ├─ 按时间窗口分组为多轮对话                                │
│      ├─ 过滤短消息、噪音、无意义内容                            │
│      └─ 去重                                                  │
│                                                             │
│   2. 格式化 (Formatter)                                       │
│      ├─ 转为 Alpaca instruction 格式                          │
│      └─ 按 9:1 分割训练/验证集                                │
│                                                             │
│   3. 反遗忘采样                                              │
│      ├─ 从长期数据库读取历史数据                                │
│      └─ 按 20% 比例混合到当前训练集                            │
│                                                             │
│   4. QLoRA 微调 (Trainer)                                    │
│      ├─ 加载基础模型 (4-bit 量化)                             │
│      ├─ 附加 LoRA Adapter                                    │
│      ├─ SFT 训练                                             │
│      └─ 保存 LoRA Adapter + 元数据                            │
│                                                             │
│   5. 长期存储 (Long-Term DB)                                  │
│      ├─ 原始对话 → memory/long_term/raw/                     │
│      ├─ 数据集 → memory/long_term/datasets/                  │
│      ├─ Adapter → memory/long_term/adapters/                 │
│      └─ 元数据 → SQLite (memory.db)                          │
└─────────────────────────────────────────────────────────────┘
```

### 数据流

一条对话从产生到变成模型记忆的完整路径：

```
对话请求 → Proxy 捕获 → short_term/{date}.jsonl
                                     ↓ (凌晨 2:00)
                     ConversationCleaner.clean_file()
                                     ↓
                     格式化 → Alpaca 格式数据集
                                     ↓
                     混合历史数据（反遗忘）
                                     ↓
                     FineTuningEngine.train()
                                     ↓
                     LoRA Adapter → long_term/adapters/
                     原始数据 → long_term/raw/
                     数据集 → long_term/datasets/
                     元数据 → SQLite memory.db
```

加载训练后的 Adapter：

```
LM Studio 加载基础模型 → 合并 LoRA Adapter → 拥有记忆的模型
```

### 模块详解

#### 1. Short-Term Collector — 对话采集

[src/short_term/collector.py](src/short_term/collector.py)

作为**透明 API 代理**运行在 `localhost:1235`，将所有请求转发到 LM Studio（`localhost:1234`），同时自动记录对话内容。

**工作方式：**

- 启动后监听 `127.0.0.1:1235`
- 将所有 `/v1/chat/completions` 请求转发到 LM Studio
- 支持 **流式（streaming）** 和 **非流式** 两种响应模式
- 将每次请求（用户消息 + 模型回复）追加到当天对应的 JSONL 文件中
- 存储位置：`memory/short_term/YYYY-MM-DD.jsonl`

**关键设计：**

- 对客户端完全透明，只需将 API 地址从 `localhost:1234` 改为 `localhost:1235`
- 流式模式下，使用 `BackgroundTasks` 在响应完成后收集完整的回复内容
- 每一条记录包含：时间戳、消息列表、模型回复、Token 用量

#### 2. ConversationCleaner — 数据清洗

[src/processor/cleaner.py](src/processor/cleaner.py)

将原始对话日志解析为结构化的训练数据。

**处理流程：**

1. **加载原始日志** — 读取 JSONL 文件
2. **按会话分组** — 根据时间戳排序，30 分钟内无交互则视为新会话
3. **消息过滤** — 跳过过短（< 5 字符）或过长（> 4096 字符）的消息
4. **会话过滤** — 丢弃不足 2 轮交互的会话
5. **去重** — 基于用户消息哈希去除相似会话
6. **噪音过滤** — 移除纯语气词（"哈"、"嗯"等）、纯标点等无意义内容
7. **事实提取** — 通过正则提取用户陈述的事实（如"我叫XXX"、"我是XXX工程师"）

#### 3. DatasetFormatter — 数据集格式化

[src/processor/formatter.py](src/processor/formatter.py)

将清洗后的对话结构化为适用于 SFT 训练的数据格式。

**数据格式（Alpaca instruction）：**

```json
{
  "instruction": "你好，我叫小明，我是一名软件工程师",
  "output": "你好小明！很高兴认识你，作为一名软件工程师...",
  "system": "这是与用户对话的上下文参考：用户说: 你好，助手说: 你好..."
}
```

**特点：**

- 每一轮用户-助手对话生成一个训练样本
- 历史轮次被编码进 system prompt 作为上下文参考
- 支持 9:1 训练/验证集分割
- 支持多数据集合并与随机采样

#### 4. FineTuningEngine — QLoRA 微调引擎

[src/training/trainer.py](src/training/trainer.py)

核心训练模块，基于 **Unsloth** 实现高效的 QLoRA 微调。

**技术选型：**

- **Unsloth** — 在 8GB VRAM（RTX 4060）上实现高效的 4-bit 微调，训练速度比原生实现快约 2 倍
- **QLoRA** — 4-bit 量化 + LoRA，显存友好
- **SFTTrainer** (TRL) — 标准监督微调流程

**LoRA 配置：**

| 参数 | 值 |
|------|-----|
| r | 16 |
| alpha | 32 |
| dropout | 0.05 |
| 目标模块 | q_proj, k_proj, v_proj, o_proj |

**训练参数：**

| 参数 | 值 |
|------|-----|
| batch size | 2 |
| gradient accumulation | 4 |
| learning rate | 2e-4 |
| epochs | 3 |
| max seq length | 2048 |

**关键设计：**

- `_has_unsloth` 标志实现惰性导入，未安装训练依赖时也能正常使用其他模块
- 训练完成后保存 Adapter + `training_metadata.json`（含样本数、耗时、超参数）
- 支持断点续训（save_total_limit=2）

#### 5. LongTermMemory — 长期记忆数据库

[src/long_term/database.py](src/long_term/database.py)

基于 **SQLite + 文件系统** 的长期记忆管理层。

**数据库表结构：**

```
conversations     — 对话归档记录（日期、路径、质量评分）
adapters          — LoRA Adapter 元数据（模型、路径、训练指标）
knowledge_facts   — 知识事实（用户信息、偏好等结构化记忆）
schedule_log      — 调度日志（每次夜间任务的执行记录）
```

**文件系统布局：**

```
memory/long_term/
├── memory.db          ← SQLite 数据库
├── raw/               ← 原始对话归档（YYYY-MM-DD.jsonl）
├── datasets/          ← 处理后数据集（YYYY-MM-DD_dataset.jsonl）
└── adapters/          ← LoRA Adapter 目录
```

**关键设计：**

- 对话元数据（路径、条数、质量评分）存储在 SQLite，便于查询
- 实际数据（JSONL、模型权重）存储在文件系统，避免数据库膨胀
- `archive_files()` 将短期记忆文件复制到长期存储，保留原始数据
- `get_all_history_dataset_paths()` 获取所有历史数据集路径，供反遗忘采样使用

#### 6. Nightly Scheduler — 夜间调度

[src/scheduler/nightly.py](src/scheduler/nightly.py)

编排完整的夜间处理流水线。

**执行步骤：**

1. 清理目标日期的短期记忆文件
2. 提取事实并存入 `knowledge_facts` 表
3. 格式化对话为训练数据集
4. 从长期数据库采样历史数据（反遗忘机制）
5. 归档对话记录和数据集
6. 执行 QLoRA 微调
7. 记录 Adapter 元数据到数据库

**调用方式：**

```bash
# 完整运行（处理今天的数据）
python -m src.scheduler.nightly

# 干运行（只处理数据，不训练）
python -m src.scheduler.nightly --dry-run

# 处理指定日期
python -m src.scheduler.nightly --date 2026-05-01
```

**Windows 计划任务集成：**

`scripts/install_task.bat` 创建每天凌晨 2:00 自动执行的任务。

#### 7. Web Dashboard — 可视化面板

[src/web/app.py](src/web/app.py) + [src/web/templates/dashboard.html](src/web/templates/dashboard.html)

基于 FastAPI 的本地状态查看和管理界面。

**功能：**

- **概览页** — 代理状态、磁盘用量、训练趋势图
- **对话页** — 按日期浏览原始对话和数据集
- **训练页** — Adapter 历史、质量指标
- **知识页** — 提取的事实展示
- **设置页** — 在线修改配置

**API 端点：**

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/stats` | GET | 获取所有状态数据 |
| `/api/nightly/trigger` | POST | 触发夜间处理 |
| `/api/nightly/dry-run` | POST | 触发干运行 |
| `/api/conversations/{date}` | GET | 获取指定日期的对话 |
| `/api/config` | GET/POST | 读取/修改配置 |

**特点：**

- 纯 SPA 架构，所有界面数据通过 API 动态加载
- 中英文双语界面，支持自由切换
- 概览页 30 秒自动刷新

#### 8. MemoryInheritor — 记忆继承

[src/inheritance/inheritor.py](src/inheritance/inheritor.py)

在更换基础模型时，从长期记忆数据库重建记忆。

**工作原理：**

1. 从数据库获取所有历史数据集路径
2. 使用 `DatasetFormatter.merge_datasets()` 合并所有数据集
3. 对新模型执行完整的 QLoRA 微调
4. 保存新的 Adapter

### 反遗忘机制

在标准的每日微调中，模型可能会在学习新数据时遗忘之前学到的内容（灾难性遗忘）。AgentTidal 通过以下策略缓解：

- **历史重放** — 每次训练时，从长期数据库中采样一定比例（默认 20%）的历史数据，混合到当前训练集中
- **数据集上限** — 总数据集大小限制在 5000 条以内，避免训练时间过长
- **Recency 加权** — 合并数据集时随机采样，近期数据有更高概率被包含

### 跨模型记忆继承

当需要更换基础模型时（如从 Qwen3.5-4B 换到 Qwen2.5-7B），记忆不会丢失：

```
原始模型 (Qwen3.5-4B) → 每日微调 → 累积 LoRA Adapter
                                            ↓
                                         更换模型
                                            ↓
新模型 (Qwen2.5-7B) ← MemoryInheritor ← 长期记忆数据库
                           ↓
                   对所有历史数据重新训练
                           ↓
                   新模型拥有全部记忆
```

---

## 使用方法

### 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.11 |
| 包管理器 | [uv](https://docs.astral.sh/uv/)（推荐）或 pip |
| GPU（训练） | NVIDIA RTX 4060 8GB VRAM 或同等（仅训练需要） |
| 本地模型 | LM Studio（推荐）或 Ollama |

### 快速开始

```bash
# 1. 克隆项目
git clone <repository-url>
cd AgentTidal

# 2. 安装基础依赖
uv sync

# 3. （可选）安装训练依赖
uv sync --extras train

# 4. 初始化环境
scripts\setup_env.bat
```

### 启动代理（白天）

```bash
# 启动透明代理（默认端口 1235）
python -m src.short_term.collector

# 指定端口和 LM Studio 地址
python -m src.short_term.collector --port 1236 --lm-studio-url http://localhost:1234/v1
```

启动后终端显示：

```
🌊 AgentTidal Proxy running on http://localhost:1235
   Forwarding to: http://localhost:1234/v1
   Saving conversations to: memory/short_term
   Configure your client to use http://localhost:1235/v1
```

### 配置客户端

在 LM Studio 或任何 OpenAI 兼容客户端中，将 API 地址改为：

```
http://localhost:1235/v1
```

代理会自动捕获所有对话并保存到 `memory/short_term/` 目录。

### 模拟对话（测试用）

无需启动 LM Studio，直接生成模拟对话数据：

```bash
# 模拟今天的对话
python scripts/simulate_conversations.py

# 模拟过去 3 天的对话
python scripts/simulate_conversations.py --days 3

# 模拟指定日期的对话
python scripts/simulate_conversations.py --date 2026-05-01

# 指定每小时的交互密度
python scripts/simulate_conversations.py --hours 2
```

### 执行夜间处理

```bash
# 干运行（处理数据但不训练，用于验证流程）
python -m src.scheduler.nightly --dry-run

# 完整运行（处理今天的数据并执行微调）
python -m src.scheduler.nightly

# 处理指定日期
python -m src.scheduler.nightly --date 2026-05-01
```

干运行输出示例：

```
==================================================
🌊 AgentTidal Nightly Processing — 2026-05-01
==================================================

[1/4] Cleaning conversations...
   Found 3 conversations after cleaning
   Extracted 2 potential facts

[2/4] Formatting training dataset...
   Train samples: 12, Val samples: 2

[3/4] Preparing training data...
   Added 20 history samples (anti-forgetting)
   Total train samples: 32

[4/4] Fine-tuning...
   [DRY RUN] Skipping training

==================================================
🌊 Dry run complete! 32 samples ready for training.
==================================================
```

### Web 仪表盘

```bash
# 启动 Web 面板
python -m src.web.app
```

打开浏览器访问 `http://localhost:8080`。

仪表盘支持中英文切换，点击侧边栏底部的 **🌐 中文/English** 按钮即可。

### 安装计划任务

以管理员身份运行：

```bash
scripts\install_task.bat
```

这将创建一个名为 `AgentTidal_Nightly` 的 Windows 计划任务，每天凌晨 2:00 执行夜间处理。

---

## 配置说明

配置文件：[config.yaml](config.yaml)

```yaml
lm_studio:
  base_url: "http://localhost:1234/v1"   # LM Studio API 地址
  api_key: ""                            # API 密钥（如需要）

proxy:
  host: "127.0.0.1"                      # 代理监听地址
  port: 1235                             # 代理监听端口

model:
  base: "Qwen/Qwen3.5-4B"               # 基础模型名称
  download_dir: "models/base"            # 模型下载目录
  max_seq_length: 2048                    # 最大序列长度
  load_in_4bit: true                      # 4-bit 量化

train:
  lora_r: 16                              # LoRA 秩
  lora_alpha: 32                          # LoRA alpha
  lora_dropout: 0.05                      # LoRA dropout
  target_modules:                         # LoRA 目标模块
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
  batch_size: 2                           # 训练批次大小
  gradient_accumulation_steps: 4          # 梯度累积步数
  learning_rate: 2.0e-4                   # 学习率
  num_epochs: 3                           # 训练轮数
  max_steps: -1                           # 最大步数（-1 表示由 epochs 决定）
  warmup_steps: 10                        # 预热步数
  history_sample_ratio: 0.2               # 历史数据采样比例（反遗忘）

memory:
  short_term_dir: "memory/short_term"     # 短期记忆目录
  long_term_dir: "memory/long_term"       # 长期记忆目录
  db_path: "memory/long_term/memory.db"   # SQLite 数据库路径
  max_dataset_size: 5000                  # 单次训练最大样本数

cleaner:
  min_turns: 2                            # 最小对话轮数
  min_message_length: 5                   # 最小消息长度
  max_message_length: 4096                # 最大消息长度

scheduler:
  enabled: true                           # 是否启用调度
  time: "02:00"                           # 执行时间
```

---

## 项目结构

```
AgentTidal/
├── README.md                          ← 本文档
├── CLAUDE.md                          ← 项目开发文档
├── pyproject.toml                     ← 项目元数据与依赖
├── config.yaml                        ← 全局配置
├── src/
│   ├── short_term/
│   │   └── collector.py               ← 透明 API 代理（对话采集）
│   ├── processor/
│   │   ├── cleaner.py                 ← 数据清洗与去重
│   │   └── formatter.py               ← Alpaca 格式转换
│   ├── training/
│   │   ├── trainer.py                 ← QLoRA 微调引擎
│   │   └── curriculum.py              ← 反遗忘/课程学习
│   ├── long_term/
│   │   └── database.py                ← 长期记忆数据库
│   ├── inheritance/
│   │   └── inheritor.py               ← 跨模型记忆继承
│   ├── scheduler/
│   │   └── nightly.py                 ← 夜间任务编排
│   └── web/
│       ├── app.py                     ← Web 仪表盘后端
│       └── templates/
│           └── dashboard.html         ← Web 仪表盘前端（SPA）
├── scripts/
│   ├── setup_env.bat                  ← 环境初始化
│   ├── simulate_conversations.py      ← 对话模拟
│   └── install_task.bat               ← 计划任务安装
├── models/
│   └── base/                          ← 基础模型缓存目录
└── memory/
    ├── short_term/                    ← 短期记忆（按日期）
    │   └── YYYY-MM-DD.jsonl
    └── long_term/                     ← 长期记忆
        ├── memory.db                  ← SQLite 数据库
        ├── raw/                       ← 原始对话归档
        ├── datasets/                  ← 处理后的数据集
        └── adapters/                  ← LoRA Adapter
```

---

## 常见问题

**Q: 必须要有 GPU 才能用吗？**

A: 只有训练阶段需要 GPU。对话采集、数据清洗、Web 面板在 CPU 上即可运行。可以使用 `--dry-run` 跳过训练来验证整个数据流程。

**Q: 必须使用 LM Studio 吗？**

A: 代理兼容任何 OpenAI 兼容的 API。可以通过 `--lm-studio-url` 参数指向 Ollama、vLLM 或其他兼容服务。

**Q: 每天微调会不会让模型退化？**

A: 这是持续微调的已知风险。项目通过以下方式缓解：数据清洗过滤低质量内容、历史重放保持已有知识、LoRA 的低秩更新限制对原始模型的影响。

**Q: 4B 参数的模型够用吗？**

A: Qwen3.5-4B 在 8GB VRAM 上微调非常充裕，且作为日常对话模型已经足够。通过继承器可以随时迁移到更大的模型。

**Q: 如何切换语言模型？**

A: 修改 `config.yaml` 中的 `model.base` 字段，然后运行继承器对新模型进行累积训练。
