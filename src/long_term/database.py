"""Long-term memory database — SQLite + file system management."""

import json
import shutil
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Dict


class LongTermMemory:
    """Manages long-term memory storage."""

    def __init__(self, db_path: str = "memory/long_term/memory.db", base_dir: str = "memory/long_term"):
        self.base_dir = Path(base_dir)
        self.db_path = Path(db_path)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database and create tables if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                raw_path TEXT,
                dataset_path TEXT,
                message_count INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0.0,
                archived INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS adapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                base_model TEXT NOT NULL,
                adapter_path TEXT NOT NULL,
                dataset_path TEXT,
                training_loss REAL,
                train_samples INTEGER DEFAULT 0,
                val_samples INTEGER DEFAULT 0,
                training_duration_seconds INTEGER DEFAULT 0,
                status TEXT DEFAULT 'completed',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS knowledge_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT DEFAULT 'general',
                key TEXT NOT NULL UNIQUE,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS schedule_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
        conn.close()

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    # --- Conversations ---

    def archive_conversation(self, date_str: str, raw_path: Path, dataset_path: Path,
                             message_count: int, quality_score: float = 0.0) -> int:
        """Move a day's conversation to long-term storage and record it."""
        conn = self._conn()
        conn.execute(
            """INSERT INTO conversations (date, raw_path, dataset_path, message_count, quality_score, archived)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (date_str, str(raw_path), str(dataset_path), message_count, quality_score),
        )
        conn.commit()
        cursor = conn.execute("SELECT last_insert_rowid()")
        row_id = cursor.fetchone()[0]
        conn.close()
        return row_id

    def get_all_conversation_dates(self) -> List[str]:
        """Get list of dates that have archived conversations."""
        conn = self._conn()
        rows = conn.execute("SELECT DISTINCT date FROM conversations WHERE archived=1 ORDER BY date").fetchall()
        conn.close()
        return [r[0] for r in rows]

    # --- Adapters ---

    def record_adapter(self, date_str: str, base_model: str, adapter_path: str,
                       dataset_path: str, train_samples: int, val_samples: int,
                       duration: int, loss: float = None) -> int:
        conn = self._conn()
        conn.execute(
            """INSERT INTO adapters (date, base_model, adapter_path, dataset_path,
               train_samples, val_samples, training_duration_seconds, training_loss)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_str, base_model, adapter_path, dataset_path,
             train_samples, val_samples, duration, loss),
        )
        conn.commit()
        cursor = conn.execute("SELECT last_insert_rowid()")
        row_id = cursor.fetchone()[0]
        conn.close()
        return row_id

    def get_latest_adapter(self) -> Optional[Dict]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM adapters ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            columns = [d[0] for d in conn.execute("PRAGMA table_info(adapters)").fetchall()]
            # Need a new connection for column info
            conn2 = self._conn()
            columns = [d[0] for d in conn2.execute("PRAGMA table_info(adapters)").fetchall()]
            conn2.close()
            return dict(zip(columns, row))
        return None

    def get_all_adapters(self) -> List[Dict]:
        conn = self._conn()
        columns = [d[0] for d in conn.execute("PRAGMA table_info(adapters)").fetchall()]
        rows = conn.execute("SELECT * FROM adapters ORDER BY id DESC").fetchall()
        conn.close()
        return [dict(zip(columns, r)) for r in rows]

    # --- Knowledge Facts ---

    def upsert_fact(self, key: str, value: str, category: str = "general",
                    confidence: float = 1.0, source: str = ""):
        """Insert or update a knowledge fact."""
        conn = self._conn()
        conn.execute(
            """INSERT INTO knowledge_facts (key, value, category, confidence, source, updated_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET
                   value = excluded.value,
                   confidence = MAX(knowledge_facts.confidence, excluded.confidence),
                   updated_at = datetime('now')""",
            (key, value, category, confidence, source),
        )
        conn.commit()
        conn.close()

    def get_facts(self, category: str = None) -> List[Dict]:
        conn = self._conn()
        if category:
            rows = conn.execute(
                "SELECT * FROM knowledge_facts WHERE category=? ORDER BY confidence DESC",
                (category,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM knowledge_facts ORDER BY confidence DESC"
            ).fetchall()
        columns = [d[0] for d in conn.execute("PRAGMA table_info(knowledge_facts)").fetchall()]
        conn.close()
        return [dict(zip(columns, r)) for r in rows]

    def get_all_history_dataset_paths(self) -> List[Path]:
        """Get all dataset paths for cumulative training."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT dataset_path FROM conversations WHERE archived=1 AND dataset_path IS NOT NULL ORDER BY date"
        ).fetchall()
        conn.close()
        return [Path(r[0]) for r in rows if r[0]]

    # --- Schedule Log ---

    def log_schedule(self, date_str: str, action: str, status: str, message: str = ""):
        conn = self._conn()
        conn.execute(
            "INSERT INTO schedule_log (date, action, status, message) VALUES (?, ?, ?, ?)",
            (date_str, action, status, message),
        )
        conn.commit()
        conn.close()

    # --- File Operations ---

    def archive_files(self, short_term_file: Path, date_str: str) -> tuple:
        """Move raw file and dataset to long-term storage. Returns (raw_archive_path, dataset_archive_path)."""
        raw_dir = self.base_dir / "raw"
        dataset_dir = self.base_dir / "datasets"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        raw_target = raw_dir / f"{date_str}.jsonl"
        dataset_target = dataset_dir / f"{date_str}_dataset.jsonl"

        if short_term_file.exists():
            shutil.copy2(str(short_term_file), str(raw_target))

        return raw_target, dataset_target
