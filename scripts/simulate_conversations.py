"""Simulate conversations to test the memory pipeline end-to-end.

Usage:
    python scripts/simulate_conversations.py          # simulate today
    python scripts/simulate_conversations.py --days 3  # simulate 3 days
    python scripts/simulate_conversations.py --date 2026-05-01 --hours 2
"""

import argparse
import json
import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Simulated user persona
USER_NAME = "小明"
USER_JOB = "软件工程师"
USER_HOBBIES = ["编程", "摄影", "跑步"]
MODEL_NAME = "qwen3.5-4b"

SIMULATION_TOPICS = [
    # User info
    [("你好", "你好！今天有什么我可以帮你的吗？"),
     ("我叫{}".format(USER_NAME), "你好{}！很高兴认识你。".format(USER_NAME)),
     ("我是一名{}".format(USER_JOB), "哇，{}！那一定很有趣。你主要做哪方面的开发？".format(USER_JOB))],

    # Tech discussion
    [("你了解Python吗？", "当然！Python是一门非常流行的编程语言，广泛应用于Web开发、数据科学和AI领域。"),
     ("我正在学习FastAPI", "FastAPI是个很好的选择！它性能出色，自动生成API文档，而且基于Python类型提示，开发体验非常好。")],

    # Hobbies
    [("我最近在学摄影", "摄影是个很棒的爱好！你主要拍什么题材？风景还是人像？"),
     ("我喜欢拍风景", "风景摄影很棒！清晨和黄昏的光线最好，黄金时段拍出来的照片特别有质感。")],

    # Daily chat
    [("今天天气怎么样？", "今天是美好的五月天！适合出去走走，呼吸新鲜空气。"),
     ("我今天不想上班", "偶尔休息一下也是可以的，记得给自己充电，保持好心情！")],

    # Deep questions
    [("AI会有意识吗？", "这是一个深刻的哲学问题。目前AI还只是模式匹配和概率预测的工具，没有真正的意识或理解能力。但我们确实在见证一个快速发展的领域。"),
     ("你觉得人类会被AI取代吗？", "AI更可能成为人类的得力助手，而非取代者。重复性工作可能会被自动化，但创造力、同理心和复杂决策仍然是人类的优势。")],
]


def simulate_conversation_session(hours: float = 1.0) -> list:
    """Simulate a conversation session spanning `hours`."""
    entries = []
    start_time = time.time() - random.uniform(0, hours * 3600)
    num_turns = random.randint(2, 6)

    # Pick 1-2 topic threads
    topics = random.sample(SIMULATION_TOPICS, min(random.randint(1, 2), len(SIMULATION_TOPICS)))

    turn_idx = 0
    for topic in topics:
        for user_msg, assistant_msg in topic:
            if turn_idx >= num_turns:
                break

            ts = start_time + turn_idx * random.randint(30, 180)
            messages = [
                {"role": "system", "content": "你是一个友好的AI助手。"},
                {"role": "user", "content": user_msg},
            ]
            entry = {
                "timestamp": ts,
                "model": MODEL_NAME,
                "messages": messages,
                "response": {"content": assistant_msg, "role": "assistant"},
                "usage": {
                    "prompt_tokens": random.randint(20, 100),
                    "completion_tokens": random.randint(30, 150),
                    "total_tokens": random.randint(50, 250),
                },
            }
            entries.append(entry)
            turn_idx += 1

    return entries


def main():
    parser = argparse.ArgumentParser(description="Simulate conversations for testing")
    parser.add_argument("--days", type=int, default=1, help="Number of days to simulate")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--hours", type=float, default=1.0, help="Hours of conversation per day")
    args = parser.parse_args()

    short_term_dir = Path("memory/short_term")
    short_term_dir.mkdir(parents=True, exist_ok=True)

    if args.date:
        target_date = date.fromisoformat(args.date)
        dates = [target_date]
    else:
        base = date.today()
        dates = [base - timedelta(days=i) for i in range(args.days - 1, -1, -1)]

    total_entries = 0
    for d in dates:
        random.seed(str(d))  # Deterministic per date
        entries = simulate_conversation_session(hours=args.hours)
        filepath = short_term_dir / f"{d.isoformat()}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        total_entries += len(entries)
        print(f"  [+] {d.isoformat()}: {len(entries)} conversation turns -> {filepath}")

    print(f"\nTotal: {total_entries} entries across {len(dates)} days")

    # Show summary
    print(f"\nShort-term memory directory: {short_term_dir}")
    for f in sorted(short_term_dir.glob("*.jsonl")):
        count = sum(1 for _ in open(f, "r", encoding="utf-8"))
        print(f"  {f.name}: {count} entries")


if __name__ == "__main__":
    main()
