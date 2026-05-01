"""Data formatter — convert cleaned conversations to training datasets."""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict


class DatasetFormatter:
    """Convert cleaned conversations into Alpaca-format training data."""

    def format_conversations(
        self, conversations: List[List[dict]], val_split: float = 0.1
    ) -> Tuple[List[dict], List[dict]]:
        """Format conversations into Alpaca instruction format.

        Returns (train_data, val_data).
        """
        samples = []
        for conv in conversations:
            samples.extend(self._conv_to_samples(conv))

        if not samples:
            return [], []

        random.shuffle(samples)
        split_idx = max(1, int(len(samples) * (1 - val_split)))
        return samples[:split_idx], samples[split_idx:]

    def _conv_to_samples(self, conversation: List[dict]) -> List[dict]:
        """Convert a multi-turn conversation into Alpaca samples.

        Each user-assistant pair becomes one sample.
        """
        samples = []
        # Build context from previous turns
        context_parts = []
        for turn in conversation:
            user_msg = turn["user_msg"]
            assistant_msg = turn["assistant_msg"]

            # Previous context helps the model understand ongoing conversation
            context = " ".join(context_parts[-4:]) if context_parts else ""
            system_prompt = self._build_system_prompt(context) if context else ""

            # Build system context from accumulated facts
            samples.append({
                "instruction": user_msg,
                "output": assistant_msg,
                "system": system_prompt,
            })

            context_parts.append(f"用户: {user_msg}")
            context_parts.append(f"助手: {assistant_msg}")

        return samples

    def _build_system_prompt(self, context: str) -> str:
        """Build a system prompt from conversation context."""
        if not context:
            return ""
        context_preview = context[:300]
        return f"这是与用户对话的上下文参考：{context_preview}"

    def save_dataset(self, data: List[dict], filepath: Path):
        """Save dataset as JSONL."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_dataset(self, filepath: Path) -> List[dict]:
        """Load a JSONL dataset."""
        data = []
        if not filepath.exists():
            return data
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def merge_datasets(self, dataset_paths: List[Path], max_samples: int = 5000) -> List[dict]:
        """Merge multiple datasets, with recency weighting."""
        all_samples = []
        for path in dataset_paths:
            data = self.load_dataset(path)
            all_samples.extend(data)

        if not all_samples:
            return []

        random.shuffle(all_samples)
        return all_samples[:max_samples]
