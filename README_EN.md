# AgentTidal — AI Long-Term Memory System

> Give local AI models true long-term memory at the **model-weight level**, without relying on context windows.

[🌏 English](README_EN.md) | [🌏 中文](README.md)

---

> **⚠️ Disclaimer: This project was entirely AI-generated and is for entertainment and learning purposes only.** The author makes no guarantees regarding the correctness, security, or suitability of the code. Any consequences arising from the use of this project are the sole responsibility of the user.

---

## Table of Contents

- [How It Works](#how-it-works)
  - [Core Idea](#core-idea)
  - [System Architecture](#system-architecture)
  - [Data Flow](#data-flow)
  - [Module Details](#module-details)
  - [Anti-Forgetting Mechanism](#anti-forgetting-mechanism)
  - [Cross-Model Memory Inheritance](#cross-model-memory-inheritance)
- [Usage](#usage)
  - [Requirements](#requirements)
  - [Quick Start](#quick-start)
  - [Start the Proxy (Daytime)](#start-the-proxy-daytime)
  - [Configure Your Client](#configure-your-client)
  - [Simulate Conversations (Testing)](#simulate-conversations-testing)
  - [Run Nightly Processing](#run-nightly-processing)
  - [Web Dashboard](#web-dashboard)
  - [Install Scheduled Task](#install-scheduled-task)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [FAQ](#faq)

---

## How It Works

### Core Idea

When interacting with large language models, all "memory" relies on historical messages within the context window. This means:

- Every conversation requires re-injecting history
- Token consumption grows linearly with memory
- Cross-session memory requires an external database + prompt injection
- All "memory" is lost when switching models

AgentTidal's core approach: **transform daily conversations into parametric knowledge through nightly automatic fine-tuning.** Once the model remembers user information and preferences at the weight level, it no longer depends on context to maintain memory.

### System Architecture

The system operates in two phases:

```
┌─────────────────────────────────────────────────────────────┐
│                    Daytime — Interaction Phase               │
│                                                             │
│  User ◄────► LM Studio (Local Model)                         │
│                   │                                          │
│                   ▼                                          │
│          AgentTidal Proxy (Port 1235)                         │
│                   │                                          │
│                   ▼                                          │
│          memory/short_term/YYYY-MM-DD.jsonl                   │
│                   (Raw conversations, daily shards)           │
└─────────────────────────────────────────────────────────────┘
                              │
                    (Auto-triggered at 2:00 AM)
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Nighttime — Processing Phase                │
│                                                             │
│   1. Data Cleaning (Cleaner)                                 │
│      ├─ Parse raw JSONL                                      │
│      ├─ Group into multi-turn conversations by time window   │
│      ├─ Filter short messages, noise, meaningless content    │
│      └─ Deduplicate                                          │
│                                                             │
│   2. Formatting (Formatter)                                   │
│      ├─ Convert to Alpaca instruction format                 │
│      └─ Split 9:1 train/validation                           │
│                                                             │
│   3. Anti-Forgetting Sampling                                │
│      ├─ Read historical data from long-term database         │
│      └─ Mix 20% into the current training set                │
│                                                             │
│   4. QLoRA Fine-Tuning (Trainer)                             │
│      ├─ Load base model (4-bit quantization)                 │
│      ├─ Attach LoRA Adapter                                  │
│      ├─ SFT training                                         │
│      └─ Save LoRA Adapter + metadata                         │
│                                                             │
│   5. Long-Term Storage (Long-Term DB)                        │
│      ├─ Raw conversations → memory/long_term/raw/            │
│      ├─ Dataset → memory/long_term/datasets/                 │
│      ├─ Adapter → memory/long_term/adapters/                 │
│      └─ Metadata → SQLite (memory.db)                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

The complete path of a conversation from creation to becoming model memory:

```
Request → Proxy captures → short_term/{date}.jsonl
                                     ↓ (at 2:00 AM)
                     ConversationCleaner.clean_file()
                                     ↓
                     Format → Alpaca dataset
                                     ↓
                     Mix historical data (anti-forgetting)
                                     ↓
                     FineTuningEngine.train()
                                     ↓
                     LoRA Adapter → long_term/adapters/
                     Raw data → long_term/raw/
                     Dataset → long_term/datasets/
                     Metadata → SQLite memory.db
```

Loading a trained Adapter:

```
LM Studio loads base model → Merge LoRA Adapter → Model with memory
```

### Module Details

#### 1. Short-Term Collector — Conversation Capture

[src/short_term/collector.py](src/short_term/collector.py)

Runs as a **transparent API proxy** on `localhost:1235`, forwarding all requests to LM Studio (`localhost:1234`) while automatically recording conversations.

**How it works:**

- Listens on `127.0.0.1:1235`
- Forwards all `/v1/chat/completions` requests to LM Studio
- Supports both **streaming** and **non-streaming** response modes
- Appends each request (user message + model reply) to a daily JSONL file
- Storage: `memory/short_term/YYYY-MM-DD.jsonl`

**Key design:**

- Fully transparent to the client — just change the API URL from `localhost:1234` to `localhost:1235`
- Streaming mode uses `BackgroundTasks` to collect the complete response after streaming finishes
- Each record contains: timestamp, message list, model reply, token usage

#### 2. ConversationCleaner — Data Cleaning

[src/processor/cleaner.py](src/processor/cleaner.py)

Parses raw conversation logs into structured training data.

**Pipeline:**

1. **Load raw logs** — Read JSONL files
2. **Group by session** — Sort by timestamp; 30+ minutes of inactivity starts a new session
3. **Filter messages** — Skip messages that are too short (< 5 chars) or too long (> 4096 chars)
4. **Filter sessions** — Discard sessions with fewer than 2 turns
5. **Deduplicate** — Remove similar conversations based on user message hashing
6. **Noise filtering** — Remove pure filler words ("ha", "um"), pure punctuation, etc.
7. **Fact extraction** — Use regex to extract factual statements (e.g., "My name is X", "I am a X engineer")

#### 3. DatasetFormatter — Dataset Formatting

[src/processor/formatter.py](src/processor/formatter.py)

Converts cleaned conversations into training data suitable for SFT.

**Data format (Alpaca instruction):**

```json
{
  "instruction": "Hi, my name is Xiao Ming, I'm a software engineer",
  "output": "Hi Xiao Ming! Nice to meet you. As a software engineer...",
  "system": "Conversation context reference - User said: Hello, Assistant said: Hello..."
}
```

**Features:**

- Each user-assistant turn generates one training sample
- Historical turns are encoded into the system prompt as context reference
- Supports 9:1 train/validation split
- Supports merging multiple datasets with random sampling

#### 4. FineTuningEngine — QLoRA Fine-Tuning Engine

[src/training/trainer.py](src/training/trainer.py)

Core training module, powered by **Unsloth** for efficient QLoRA fine-tuning.

**Technology choices:**

- **Unsloth** — Enables efficient 4-bit fine-tuning on 8GB VRAM (RTX 4060), ~2x faster than native implementation
- **QLoRA** — 4-bit quantization + LoRA, VRAM-friendly
- **SFTTrainer** (TRL) — Standard supervised fine-tuning workflow

**LoRA Configuration:**

| Parameter | Value |
|-----------|-------|
| r | 16 |
| alpha | 32 |
| dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |

**Training Parameters:**

| Parameter | Value |
|-----------|-------|
| batch size | 2 |
| gradient accumulation | 4 |
| learning rate | 2e-4 |
| epochs | 3 |
| max seq length | 2048 |

**Key design:**

- `_has_unsloth` flag enables lazy imports — other modules work fine without training dependencies
- Saves Adapter + `training_metadata.json` (sample count, duration, hyperparameters) after training
- Supports checkpoint resume (save_total_limit=2)

#### 5. LongTermMemory — Long-Term Memory Database

[src/long_term/database.py](src/long_term/database.py)

**SQLite + file system** based long-term memory management.

**Database tables:**

```
conversations     — Conversation archive records (date, paths, quality score)
adapters          — LoRA Adapter metadata (model, path, training metrics)
knowledge_facts   — Knowledge facts (user info, preferences, structured memory)
schedule_log      — Schedule logs (execution records for each nightly run)
```

**File system layout:**

```
memory/long_term/
├── memory.db          ← SQLite database
├── raw/               ← Raw conversation archives (YYYY-MM-DD.jsonl)
├── datasets/          ← Processed datasets (YYYY-MM-DD_dataset.jsonl)
└── adapters/          ← LoRA Adapter directories
```

**Key design:**

- Conversation metadata stored in SQLite for easy querying
- Actual data (JSONL, model weights) stored on the file system to prevent database bloat
- `archive_files()` copies short-term memory files to long-term storage, preserving raw data
- `get_all_history_dataset_paths()` retrieves all historical dataset paths for anti-forgetting sampling

#### 6. Nightly Scheduler — Nightly Orchestration

[src/scheduler/nightly.py](src/scheduler/nightly.py)

Orchestrates the complete nightly processing pipeline.

**Execution steps:**

1. Clean the target date's short-term memory file
2. Extract facts and store in `knowledge_facts` table
3. Format conversations into a training dataset
4. Sample historical data from the long-term database (anti-forgetting)
5. Archive conversation records and datasets
6. Execute QLoRA fine-tuning
7. Record Adapter metadata in the database

**CLI usage:**

```bash
# Full run (process today's data)
python -m src.scheduler.nightly

# Dry run (process data without training)
python -m src.scheduler.nightly --dry-run

# Process a specific date
python -m src.scheduler.nightly --date 2026-05-01
```

**Windows Task Scheduler integration:**

`scripts/install_task.bat` creates a scheduled task that runs daily at 2:00 AM.

#### 7. Web Dashboard — Visualization Panel

[src/web/app.py](src/web/app.py) + [src/web/templates/dashboard.html](src/web/templates/dashboard.html)

A local status viewing and management interface powered by FastAPI.

**Pages:**

- **Overview** — Proxy status, disk usage, training trend chart
- **Conversations** — Browse raw conversations and datasets by date
- **Training** — Adapter history, quality metrics
- **Knowledge** — Display extracted facts
- **Settings** — Modify configuration online

**API Endpoints:**

| Endpoint | Method | Function |
|----------|--------|----------|
| `/api/stats` | GET | Get all status data |
| `/api/nightly/trigger` | POST | Trigger nightly processing |
| `/api/nightly/dry-run` | POST | Trigger dry run |
| `/api/conversations/{date}` | GET | Get conversations for a date |
| `/api/config` | GET/POST | Read/modify configuration |

**Features:**

- Pure SPA architecture, all data loaded dynamically via API
- Bilingual interface (Chinese/English) with free switching
- Overview page auto-refreshes every 30 seconds

#### 8. MemoryInheritor — Memory Inheritance

[src/inheritance/inheritor.py](src/inheritance/inheritor.py)

Rebuilds memory from the long-term database when switching base models.

**How it works:**

1. Retrieve all historical dataset paths from the database
2. Use `DatasetFormatter.merge_datasets()` to merge all datasets
3. Run full QLoRA fine-tuning on the new model
4. Save the new Adapter

### Anti-Forgetting Mechanism

In standard daily fine-tuning, models may forget previously learned content when learning new data (catastrophic forgetting). AgentTidal mitigates this through:

- **History replay** — Each training session samples a proportion (default 20%) of historical data from the long-term database, mixed into the current training set
- **Dataset cap** — Total dataset size limited to 5000 samples to prevent excessively long training
- **Recency weighting** — Random sampling during dataset merging, with a higher probability of including recent data

### Cross-Model Memory Inheritance

When switching base models (e.g., from Qwen3.5-4B to Qwen2.5-7B), memory is not lost:

```
Original Model (Qwen3.5-4B) → Daily fine-tuning → Accumulated LoRA Adapters
                                                           ↓
                                                        Switch model
                                                           ↓
New Model (Qwen2.5-7B) ← MemoryInheritor ← Long-Term Memory Database
                          ↓
                    Retrain on all historical data
                          ↓
                    New model has complete memory
```

---

## Usage

### Requirements

| Item | Requirement |
|------|-------------|
| Python | >= 3.11 |
| Package manager | [uv](https://docs.astral.sh/uv/) (recommended) or pip |
| GPU (training) | NVIDIA RTX 4060 8GB VRAM or equivalent (training only) |
| Local model | LM Studio (recommended) or Ollama |

### Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd AgentTidal

# 2. Install base dependencies
uv sync

# 3. (Optional) Install training dependencies
uv sync --extras train

# 4. Initialize environment
scripts\setup_env.bat
```

### Start the Proxy (Daytime)

```bash
# Start transparent proxy (default port 1235)
python -m src.short_term.collector

# Specify port and LM Studio URL
python -m src.short_term.collector --port 1236 --lm-studio-url http://localhost:1234/v1
```

Console output after starting:

```
🌊 AgentTidal Proxy running on http://localhost:1235
   Forwarding to: http://localhost:1234/v1
   Saving conversations to: memory/short_term
   Configure your client to use http://localhost:1235/v1
```

### Configure Your Client

In LM Studio or any OpenAI-compatible client, change the API URL to:

```
http://localhost:1235/v1
```

The proxy will automatically capture all conversations and save them to `memory/short_term/`.

### Simulate Conversations (Testing)

No need to start LM Studio — generate simulated conversation data directly:

```bash
# Simulate today's conversations
python scripts/simulate_conversations.py

# Simulate the last 3 days
python scripts/simulate_conversations.py --days 3

# Simulate a specific date
python scripts/simulate_conversations.py --date 2026-05-01

# Set the number of interaction hours per day
python scripts/simulate_conversations.py --hours 2
```

### Run Nightly Processing

```bash
# Dry run (process data without training — verify the pipeline)
python -m src.scheduler.nightly --dry-run

# Full run (process today's data and execute fine-tuning)
python -m src.scheduler.nightly

# Process a specific date
python -m src.scheduler.nightly --date 2026-05-01
```

Dry run output example:

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

### Web Dashboard

```bash
# Start the web panel
python -m src.web.app
```

Open your browser and visit `http://localhost:8080`.

The dashboard supports Chinese/English switching — click the **🌐 中文/English** button at the bottom of the sidebar.

### Install Scheduled Task

Run as Administrator:

```bash
scripts\install_task.bat
```

This creates a Windows scheduled task named `AgentTidal_Nightly` that runs nightly at 2:00 AM.

---

## Configuration Reference

Configuration file: [config.yaml](config.yaml)

```yaml
lm_studio:
  base_url: "http://localhost:1234/v1"   # LM Studio API URL
  api_key: ""                            # API key (if needed)

proxy:
  host: "127.0.0.1"                      # Proxy listen address
  port: 1235                             # Proxy listen port

model:
  base: "Qwen/Qwen3.5-4B"               # Base model name
  download_dir: "models/base"            # Model download directory
  max_seq_length: 2048                    # Maximum sequence length
  load_in_4bit: true                      # 4-bit quantization

train:
  lora_r: 16                              # LoRA rank
  lora_alpha: 32                          # LoRA alpha
  lora_dropout: 0.05                      # LoRA dropout
  target_modules:                         # LoRA target modules
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
  batch_size: 2                           # Training batch size
  gradient_accumulation_steps: 4          # Gradient accumulation steps
  learning_rate: 2.0e-4                   # Learning rate
  num_epochs: 3                           # Number of training epochs
  max_steps: -1                           # Max steps (-1 = determined by epochs)
  warmup_steps: 10                        # Warmup steps
  history_sample_ratio: 0.2               # History sampling ratio (anti-forgetting)

memory:
  short_term_dir: "memory/short_term"     # Short-term memory directory
  long_term_dir: "memory/long_term"       # Long-term memory directory
  db_path: "memory/long_term/memory.db"   # SQLite database path
  max_dataset_size: 5000                  # Max samples per training session

cleaner:
  min_turns: 2                            # Minimum conversation turns
  min_message_length: 5                   # Minimum message length
  max_message_length: 4096                # Maximum message length

scheduler:
  enabled: true                           # Enable scheduler
  time: "02:00"                           # Execution time
```

---

## Project Structure

```
AgentTidal/
├── README.md                          ← This document (Chinese)
├── README_EN.md                       ← This document (English)
├── CLAUDE.md                          ← Project development guide
├── pyproject.toml                     ← Project metadata and dependencies
├── config.yaml                        ← Global configuration
├── src/
│   ├── short_term/
│   │   └── collector.py               ← Transparent API proxy
│   ├── processor/
│   │   ├── cleaner.py                 ← Data cleaning and dedup
│   │   └── formatter.py               ← Alpaca format conversion
│   ├── training/
│   │   ├── trainer.py                 ← QLoRA fine-tuning engine
│   │   └── curriculum.py              ← Anti-forgetting / curriculum learning
│   ├── long_term/
│   │   └── database.py                ← Long-term memory database
│   ├── inheritance/
│   │   └── inheritor.py               ← Cross-model memory inheritance
│   ├── scheduler/
│   │   └── nightly.py                 ← Nightly task orchestration
│   └── web/
│       ├── app.py                     ← Web dashboard backend
│       └── templates/
│           └── dashboard.html         ← Web dashboard frontend (SPA)
├── scripts/
│   ├── setup_env.bat                  ← Environment setup
│   ├── simulate_conversations.py      ← Conversation simulation
│   └── install_task.bat               ← Scheduled task installer
├── models/
│   └── base/                          ← Base model cache directory
└── memory/
    ├── short_term/                    ← Short-term memory (by date)
    │   └── YYYY-MM-DD.jsonl
    └── long_term/                     ← Long-term memory
        ├── memory.db                  ← SQLite database
        ├── raw/                       ← Raw conversation archives
        ├── datasets/                  ← Processed datasets
        └── adapters/                  ← LoRA Adapter
```

---

## FAQ

**Q: Do I need a GPU?**

A: Only the training phase requires a GPU. Conversation capture, data cleaning, and the web panel all run on CPU. You can use `--dry-run` to skip training and verify the entire data pipeline.

**Q: Do I have to use LM Studio?**

A: The proxy is compatible with any OpenAI-compatible API. Use the `--lm-studio-url` parameter to point to Ollama, vLLM, or other compatible services.

**Q: Will daily fine-tuning degrade the model?**

A: This is a known risk of continual fine-tuning. The project mitigates it through: data cleaning to filter low-quality content, history replay to preserve existing knowledge, and LoRA's low-rank updates that limit impact on the original model.

**Q: Is a 4B parameter model sufficient?**

A: Qwen3.5-4B is very comfortable for fine-tuning on 8GB VRAM and is more than sufficient for daily conversation. You can always migrate to a larger model using the inheritor.

**Q: How do I switch language models?**

A: Modify the `model.base` field in `config.yaml`, then run the inheritor for cumulative training on the new model.
