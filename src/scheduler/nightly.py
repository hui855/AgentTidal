"""Nightly scheduler — orchestrates the full memory pipeline.

Usage:
    python -m src.scheduler.nightly                     # run for today
    python -m src.scheduler.nightly --date 2026-05-01   # run for specific date
    python -m src.scheduler.nightly --dry-run           # simulate without training
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_nightly(target_date: str = None, dry_run: bool = False):
    """Execute the full nightly memory pipeline."""
    config = load_config()
    today = date.today().isoformat() if target_date is None else target_date

    # Import modules
    from src.processor.cleaner import ConversationCleaner
    from src.processor.formatter import DatasetFormatter
    from src.long_term.database import LongTermMemory

    cleaner_config = config.get("cleaner", {})
    cleaner = ConversationCleaner(
        min_turns=cleaner_config.get("min_turns", 2),
        min_msg_length=cleaner_config.get("min_message_length", 5),
        max_msg_length=cleaner_config.get("max_message_length", 4096),
    )
    formatter = DatasetFormatter()
    memory_db = LongTermMemory(
        db_path=config.get("memory", {}).get("db_path", "memory/long_term/memory.db"),
        base_dir=config.get("memory", {}).get("long_term_dir", "memory/long_term"),
    )

    short_term_dir = Path(config.get("memory", {}).get("short_term_dir", "memory/short_term"))
    short_term_file = short_term_dir / f"{today}.jsonl"

    memory_db.log_schedule(today, "nightly_start", "started", f"Processing {short_term_file}")

    print(f"\n{'='*50}")
    print(f"🌊 AgentTidal Nightly Processing — {today}")
    print(f"{'='*50}")

    # Step 1: Clean raw conversations
    print(f"\n[1/4] Cleaning conversations...")
    conversations = cleaner.clean_file(short_term_file)
    print(f"   Found {len(conversations)} conversations after cleaning")

    if not conversations:
        msg = f"No conversations to process for {today}"
        print(f"   {msg}")
        memory_db.log_schedule(today, "nightly_complete", "skipped", msg)
        return

    # Extract facts
    facts = cleaner.extract_facts(conversations)
    print(f"   Extracted {len(facts)} potential facts")

    # Step 2: Format into training dataset
    print(f"\n[2/4] Formatting training dataset...")
    val_split = 0.1
    train_samples, val_samples = formatter.format_conversations(conversations, val_split=val_split)
    print(f"   Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    if not train_samples:
        msg = "No training samples generated"
        print(f"   {msg}")
        memory_db.log_schedule(today, "nightly_complete", "skipped", msg)
        return

    # Save dataset
    raw_target, dataset_target = memory_db.archive_files(short_term_file, today)
    formatter.save_dataset(train_samples + val_samples, dataset_target)
    print(f"   Dataset saved to: {dataset_target}")

    # Save facts
    for fact in facts:
        memory_db.upsert_fact(
            key=fact["key"],
            value=fact["value"],
            category="user_info",
            source=today,
        )

    # Step 3: Anti-forgetting — sample history
    print(f"\n[3/4] Preparing training data...")
    history_ratio = config.get("train", {}).get("history_sample_ratio", 0.2)
    if history_ratio > 0:
        history_paths = memory_db.get_all_history_dataset_paths()
        # Exclude current date's path
        history_paths = [p for p in history_paths if today not in str(p)]
        if history_paths:
            from src.processor.formatter import DatasetFormatter as DF
            hist_formatter = DF()
            max_total = config.get("memory", {}).get("max_dataset_size", 5000)
            current_count = len(train_samples)
            history_budget = min(int(current_count * history_ratio), max_total - current_count)

            if history_budget > 0:
                history_samples = hist_formatter.merge_datasets(history_paths, max_samples=history_budget)
                train_samples = history_samples + train_samples
                print(f"   Added {len(history_samples)} history samples (anti-forgetting)")
                print(f"   Total train samples: {len(train_samples)}")

    # Archive conversation record
    memory_db.archive_conversation(
        date_str=today,
        raw_path=raw_target,
        dataset_path=dataset_target,
        message_count=sum(len(c) for c in conversations),
        quality_score=min(1.0, len(train_samples) / 100),
    )

    # Step 4: Fine-tune
    print(f"\n[4/4] Fine-tuning...")
    if dry_run:
        print(f"   [DRY RUN] Skipping training")
        memory_db.log_schedule(today, "nightly_complete", "dry_run", f"{len(train_samples)} samples ready")
        print(f"\n{'='*50}")
        print(f"🌊 Dry run complete! {len(train_samples)} samples ready for training.")
        print(f"{'='*50}")
        return

    from src.training.trainer import FineTuningEngine

    engine = FineTuningEngine()
    if not engine.available:
        msg = "Training dependencies not installed. Install with: uv sync --extras train"
        print(f"   ERROR: {msg}")
        memory_db.log_schedule(today, "nightly_complete", "failed", msg)
        return

    base_model = config.get("model", {}).get("base", "Qwen/Qwen3.5-4B")
    adapter_name = f"memory_{today}"

    start_time = time.time()
    adapter_path = engine.train(
        train_samples=train_samples,
        val_samples=val_samples,
        output_dir=config.get("memory", {}).get("long_term_dir", "memory/long_term") + "/adapters",
        adapter_name=adapter_name,
    )

    if adapter_path:
        duration = int(time.time() - start_time)
        memory_db.record_adapter(
            date_str=today,
            base_model=base_model,
            adapter_path=adapter_path,
            dataset_path=str(dataset_target),
            train_samples=len(train_samples),
            val_samples=len(val_samples),
            duration=duration,
        )
        memory_db.log_schedule(today, "nightly_complete", "success",
                               f"{len(train_samples)} samples, {duration}s training")

        print(f"\n{'='*50}")
        print(f"🌊 Nightly processing complete!")
        print(f"   Date: {today}")
        print(f"   Conversations: {len(conversations)}")
        print(f"   Training samples: {len(train_samples)}")
        print(f"   Adapter: {adapter_path}")
        print(f"{'='*50}")
    else:
        memory_db.log_schedule(today, "nightly_complete", "failed", "Training returned no adapter")
        print(f"\n   ERROR: Training failed")


def main():
    parser = argparse.ArgumentParser(description="AgentTidal Nightly Scheduler")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Process data without training")
    args = parser.parse_args()

    run_nightly(target_date=args.date, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
