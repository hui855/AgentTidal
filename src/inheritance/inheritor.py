"""Memory inheritance — transfer learned memory to a new model.

Phase 3 implementation. For now, provides the interface contract.
"""

from pathlib import Path
from typing import List, Optional

import yaml


class MemoryInheritor:
    """Transfer accumulated memory from long-term database to a new base model."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def list_available_datasets(self) -> List[Path]:
        """List all historical training datasets available."""
        from src.long_term.database import LongTermMemory

        db = LongTermMemory(
            db_path=self.config.get("memory", {}).get("db_path", "memory/long_term/memory.db"),
            base_dir=self.config.get("memory", {}).get("long_term_dir", "memory/long_term"),
        )
        return db.get_all_history_dataset_paths()

    def inherit_to_new_model(self, new_base_model: str, adapter_name: str = None) -> Optional[str]:
        """Train a new base model on all accumulated memory data.

        Args:
            new_base_model: HuggingFace model name (e.g., "Qwen/Qwen3.5-4B")
            adapter_name: Optional name for the new adapter

        Returns:
            Path to the trained adapter, or None on failure.
        """
        from src.processor.formatter import DatasetFormatter
        from src.training.trainer import FineTuningEngine

        dataset_paths = self.list_available_datasets()
        if not dataset_paths:
            print("No historical datasets found.")
            return None

        print(f"Found {len(dataset_paths)} historical datasets")

        formatter = DatasetFormatter()
        max_samples = self.config.get("memory", {}).get("max_dataset_size", 5000)
        all_samples = formatter.merge_datasets(dataset_paths, max_samples=max_samples)

        if not all_samples:
            print("No samples loaded from datasets.")
            return None

        # Split train/val
        split = int(len(all_samples) * 0.9)
        train_samples = all_samples[:split]
        val_samples = all_samples[split:]

        # Override base model in config for the trainer
        original_base = self.config["model"]["base"]
        self.config["model"]["base"] = new_base_model

        try:
            engine = FineTuningEngine()
            # Override the loaded config
            engine.base_model = new_base_model
            engine.model_config["base"] = new_base_model

            if not engine.available:
                print("Training dependencies not installed.")
                return None

            result = engine.train(
                train_samples=train_samples,
                val_samples=val_samples,
                adapter_name=adapter_name or f"inherited_{new_base_model.replace('/', '_')}",
            )
            return result
        finally:
            self.config["model"]["base"] = original_base
