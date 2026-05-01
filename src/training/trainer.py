"""QLoRA fine-tuning engine using Unsloth."""

import json
import os
import time
from pathlib import Path
from typing import Optional

import yaml

# These imports are only available when training extras are installed
_has_unsloth = False
try:
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from datasets import Dataset as HFDataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    _has_unsloth = True
except ImportError:
    pass


class FineTuningEngine:
    """QLoRA fine-tuning engine powered by Unsloth."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get("model", {})
        self.train_config = self.config.get("train", {})
        self.base_model = self.model_config.get("base", "Qwen/Qwen3.5-4B")
        self.max_seq_length = self.model_config.get("max_seq_length", 2048)

    @property
    def available(self) -> bool:
        return _has_unsloth

    def load_model(self):
        """Load base model with 4-bit quantization via Unsloth."""
        print(f"Loading model: {self.base_model}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.model_config.get("load_in_4bit", True),
        )

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.train_config.get("lora_r", 16),
            target_modules=self.train_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_alpha=self.train_config.get("lora_alpha", 32),
            lora_dropout=self.train_config.get("lora_dropout", 0.05),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )

        return model, tokenizer

    def prepare_dataset(self, train_samples: list, val_samples: list) -> tuple:
        """Convert samples to Hugging Face datasets."""
        train_dataset = HFDataset.from_list(train_samples)
        val_dataset = HFDataset.from_list(val_samples) if val_samples else None
        return train_dataset, val_dataset

    @staticmethod
    def format_sample(example: dict) -> str:
        """Format one sample into a training text string."""
        system = example.get("system", "")
        instr = example.get("instruction", "")
        output = example.get("output", "")
        if system:
            return f"<|system|>\n{system}\n<|user|>\n{instr}\n<|assistant|>\n{output}"
        return f"<|user|>\n{instr}\n<|assistant|>\n{output}"

    def train(
        self,
        train_samples: list,
        val_samples: list = None,
        output_dir: str = "memory/long_term/adapters",
        adapter_name: str = None,
    ) -> Optional[str]:
        """Run fine-tuning. Returns adapter path or None on failure."""
        if not _has_unsloth:
            print("Training dependencies not installed. Install with: uv pip install agent-tidal[train]")
            return None

        model, tokenizer = self.load_model()

        train_dataset, val_dataset = self.prepare_dataset(train_samples, val_samples)

        if adapter_name is None:
            adapter_name = time.strftime("%Y%m%d_%H%M%S")

        adapter_path = str(Path(output_dir) / adapter_name)
        Path(adapter_path).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=adapter_path,
            per_device_train_batch_size=self.train_config.get("batch_size", 2),
            gradient_accumulation_steps=self.train_config.get("gradient_accumulation_steps", 4),
            warmup_steps=self.train_config.get("warmup_steps", 10),
            num_train_epochs=self.train_config.get("num_epochs", 3),
            max_steps=self.train_config.get("max_steps", -1),
            learning_rate=self.train_config.get("learning_rate", 2.0e-4),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=self.train_config.get("logging_steps", 10),
            save_strategy=self.train_config.get("save_strategy", "steps"),
            save_steps=self.train_config.get("save_steps", 50),
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=50 if val_dataset else None,
            report_to="none",
            save_total_limit=2,
            load_best_model_at_end=True if val_dataset else False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            formatting_func=self.format_sample,
            max_seq_length=self.max_seq_length,
            packing=False,
        )

        print(f"Starting training for {adapter_name}...")
        start_time = time.time()
        trainer.train()
        duration = int(time.time() - start_time)

        # Save final adapter
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        # Save training metadata
        metadata = {
            "adapter_name": adapter_name,
            "base_model": self.base_model,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples) if val_samples else 0,
            "training_duration_seconds": duration,
            "config": self.train_config,
        }
        with open(Path(adapter_path) / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Training complete! Adapter saved to: {adapter_path}")
        print(f"   Duration: {duration}s, Samples: {len(train_samples)}")

        return adapter_path
