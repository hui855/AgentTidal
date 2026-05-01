"""Data cleaner — parse raw conversations, filter noise, deduplicate."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple


class ConversationCleaner:
    """Clean and filter raw conversation logs."""

    def __init__(self, min_turns: int = 2, min_msg_length: int = 5, max_msg_length: int = 4096):
        self.min_turns = min_turns
        self.min_msg_length = min_msg_length
        self.max_msg_length = max_msg_length

    def load_raw(self, filepath: Path) -> List[dict]:
        """Load raw JSONL conversation entries."""
        entries = []
        if not filepath.exists():
            return entries
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries

    def entries_to_conversations(self, entries: List[dict]) -> List[List[dict]]:
        """Group entries into multi-turn conversations (by proximity/timestamp)."""
        if not entries:
            return []

        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda e: e.get("timestamp", 0))

        conversations = []
        current_conv = []
        last_time = 0
        gap_threshold = 1800  # 30 min gap = new conversation

        for entry in sorted_entries:
            ts = entry.get("timestamp", 0)
            messages = entry.get("messages", [])

            # Build a turn object
            turn = {
                "user_msg": self._extract_user_message(messages),
                "assistant_msg": entry.get("response", {}).get("content", ""),
                "model": entry.get("model", ""),
                "timestamp": ts,
                "usage": entry.get("usage", {}),
            }

            # Skip if either message is too short
            if len(turn["user_msg"]) < self.min_msg_length or len(turn["assistant_msg"]) < self.min_msg_length:
                continue
            if len(turn["user_msg"]) > self.max_msg_length or len(turn["assistant_msg"]) > self.max_msg_length:
                continue

            # New conversation if time gap is large
            if current_conv and (ts - last_time > gap_threshold):
                if len(current_conv) >= self.min_turns:
                    conversations.append(current_conv)
                current_conv = []

            current_conv.append(turn)
            last_time = ts

        if len(current_conv) >= self.min_turns:
            conversations.append(current_conv)

        return conversations

    def _extract_user_message(self, messages: List[dict]) -> str:
        """Extract the last user message from the messages array."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle multimodal content arrays
                    texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                    return " ".join(texts)
                return content
        return ""

    def deduplicate(self, conversations: List[List[dict]]) -> List[List[dict]]:
        """Remove near-duplicate conversations."""
        if not conversations:
            return []

        seen_hashes = set()
        result = []

        for conv in conversations:
            # Create a signature from user messages
            user_texts = "|".join(t["user_msg"][:100] for t in conv)
            sig = hash(user_texts)
            if sig not in seen_hashes:
                seen_hashes.add(sig)
                result.append(conv)

        return result

    def clean_file(self, filepath: Path) -> Tuple[List[List[dict]], List[List[dict]]]:
        """Full cleaning pipeline for a raw file. Returns (train_convos, val_convos)."""
        entries = self.load_raw(filepath)
        conversations = self.entries_to_conversations(entries)
        conversations = self.deduplicate(conversations)
        # Filter trivial patterns
        conversations = self._filter_noise(conversations)
        return conversations

    def _filter_noise(self, conversations: List[List[dict]]) -> List[List[dict]]:
        """Remove obvious noise: single words, pure greetings, etc."""
        noise_patterns = [
            re.compile(r"^[哈嘿哦呃嗯好]+$"),
            re.compile(r"^[\.。，,\?!\-_]+$"),
        ]
        filtered = []
        for conv in conversations:
            keep = True
            for turn in conv:
                for pattern in noise_patterns:
                    if pattern.match(turn["user_msg"].strip()) and len(turn["user_msg"].strip()) < 5:
                        keep = False
                        break
                    if pattern.match(turn["assistant_msg"].strip()) and len(turn["assistant_msg"].strip()) < 5:
                        keep = False
                        break
                if not keep:
                    break
            if keep:
                filtered.append(conv)
        return filtered

    def extract_facts(self, conversations: List[List[dict]]) -> List[Dict[str, str]]:
        """Extract potential factual statements from conversations.

        This is a simple heuristic — looks for "我叫" "我是" patterns.
        Phase 2 will use a more sophisticated classifier.
        """
        facts = []
        patterns = [
            (r"(?:我叫|我是|我的名字是?)\s*(.+?)(?:[，。！？\.!,]|$)", "user.name"),
            (r"我(?:是|做|在)\s*(.+?(?:工程师|老师|学生|医生|设计师|运营|产品))", "user.job"),
        ]
        for conv in conversations:
            for turn in conv:
                for pattern, fact_key in patterns:
                    m = re.search(pattern, turn["user_msg"])
                    if m:
                        facts.append({"key": fact_key, "value": m.group(1).strip()[:100]})
        return facts
