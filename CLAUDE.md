# AgentTidal — AI Long-Term Memory System

## Overview
AgentTidal captures daily conversations with a local AI model (via LM Studio), then automatically fine-tunes the model at night using QLoRA/Unsloth to build true long-term memory at the model-weight level.

## Architecture

- **Daytime**: API proxy (port 1235) captures conversations → short-term memory (JSONL)
- **Nighttime**: Clean → Format → QLoRA fine-tune → Long-term memory (SQLite + adapters)

## Key Commands

```bash
# Start the proxy (daytime)
python -m src.short_term.collector

# Simulate test conversations
python scripts/simulate_conversations.py

# Simulate multiple days
python scripts/simulate_conversations.py --days 3

# Run nightly pipeline (dry run, no training)
python -m src.scheduler.nightly --dry-run

# Run nightly pipeline (full)
python -m src.scheduler.nightly

# Run for a specific date
python -m src.scheduler.nightly --date 2026-05-01
```

## Directory Layout

```
src/short_term/collector.py   — API proxy (captures conversations)
src/processor/cleaner.py      — Data cleaning & dedup
src/processor/formatter.py    — Alpaca format conversion
src/training/trainer.py       — QLoRA fine-tuning (Unsloth)
src/long_term/database.py     — SQLite long-term memory
src/scheduler/nightly.py      — Nightly pipeline orchestrator
src/inheritance/inheritor.py  — Cross-model memory transfer
```

## Configuration

Edit `config.yaml` to change model, training parameters, paths.

## Tech Stack
- **Model**: Qwen3.5-4B (via LM Studio API)
- **Fine-tuning**: Unsloth + QLoRA (4-bit, PEFT)
- **Training engine**: Unsloth + TRL SFTTrainer
- **Storage**: SQLite + JSONL + LoRA adapters
- **Proxy**: FastAPI + httpx (sync)
