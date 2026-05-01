"""AgentTidal Web Dashboard — visualize and manage the memory pipeline.

Usage:
    python -m src.web.app

Open http://localhost:8080 in your browser.
"""

import json
from datetime import date
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

CONFIG_PATH = "config.yaml"
STATIC_DIR = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="AgentTidal Dashboard")


# ─── Data helpers ────────────────────────────────────────────

def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _short_term_files() -> list:
    base = Path(_load_config()["memory"]["short_term_dir"])
    files = sorted(base.glob("*.jsonl"), reverse=True)
    result = []
    for f in files:
        count = 0
        try:
            with open(f, "r", encoding="utf-8") as fh:
                count = sum(1 for _ in fh)
        except Exception:
            pass
        result.append({"name": f.name, "date": f.stem, "entries": count, "path": str(f)})
    return result


def _long_term_summary() -> dict:
    from src.long_term.database import LongTermMemory
    config = _load_config()
    db = LongTermMemory(
        db_path=config["memory"]["db_path"],
        base_dir=config["memory"]["long_term_dir"],
    )
    conn = db._conn()

    result = {}
    result["conversation_dates"] = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM conversations WHERE archived=1 ORDER BY date DESC"
    ).fetchall()]
    result["total_conversations"] = conn.execute(
        "SELECT COUNT(*) FROM conversations WHERE archived=1"
    ).fetchone()[0]
    result["total_adapters"] = conn.execute(
        "SELECT COUNT(*) FROM adapters"
    ).fetchone()[0]
    result["total_facts"] = conn.execute(
        "SELECT COUNT(*) FROM knowledge_facts"
    ).fetchone()[0]

    latest = conn.execute(
        "SELECT * FROM adapters ORDER BY id DESC LIMIT 1"
    ).fetchone()
    result["latest_adapter"] = list(latest) if latest else None

    logs = conn.execute(
        "SELECT * FROM schedule_log ORDER BY id DESC LIMIT 10"
    ).fetchall()
    result["recent_logs"] = [list(r) for r in logs]

    facts = conn.execute(
        "SELECT * FROM knowledge_facts ORDER BY confidence DESC"
    ).fetchall()
    result["facts"] = [list(r) for r in facts]

    daily = conn.execute(
        "SELECT date, SUM(train_samples) as samples FROM adapters GROUP BY date ORDER BY date"
    ).fetchall()
    result["training_chart"] = {
        "labels": [r[0] for r in daily],
        "samples": [(r[1] or 0) for r in daily],
    }

    conn.close()
    return result


def _proxy_status() -> dict:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", 1235))
        s.close()
        return {"running": True, "port": 1235}
    except ConnectionRefusedError:
        return {"running": False, "port": 1235}


def _disk_usage() -> dict:
    mem_dir = Path(_load_config()["memory"]["long_term_dir"])
    total_bytes = sum(f.stat().st_size for f in mem_dir.rglob("*") if f.is_file())
    short_count = len(list(Path(_load_config()["memory"]["short_term_dir"]).glob("*.jsonl")))
    return {
        "total_mb": round(total_bytes / (1024 * 1024), 1),
        "short_term_files": short_count,
    }


# ─── API Routes ──────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    return {
        "proxy": _proxy_status(),
        "disk": _disk_usage(),
        "short_term": _short_term_files(),
        "long_term": _long_term_summary(),
        "date": date.today().isoformat(),
    }


@app.post("/api/nightly/trigger")
async def api_trigger_nightly():
    from src.scheduler.nightly import run_nightly
    try:
        run_nightly(dry_run=False)
        return {"status": "success", "message": "Nightly processing started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/nightly/dry-run")
async def api_trigger_dryrun():
    from src.scheduler.nightly import run_nightly
    try:
        run_nightly(dry_run=True)
        return {"status": "success", "message": "Dry run complete"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/conversations/{date_str}")
async def api_conversations(date_str: str):
    config = _load_config()
    raw_dir = Path(config["memory"]["long_term_dir"]) / "raw"
    dataset_dir = Path(config["memory"]["long_term_dir"]) / "datasets"
    short_file = Path(config["memory"]["short_term_dir"]) / f"{date_str}.jsonl"

    raw_entries = []
    fp = raw_dir / f"{date_str}.jsonl"
    if not fp.exists():
        fp = short_file
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_entries.append(json.loads(line))

    dataset_entries = []
    dfp = dataset_dir / f"{date_str}_dataset.jsonl"
    if dfp.exists():
        with open(dfp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset_entries.append(json.loads(line))

    return {"date": date_str, "raw": raw_entries, "dataset": dataset_entries}


@app.get("/api/config")
async def api_config():
    return _load_config()


@app.post("/api/config")
async def api_update_config(data: dict):
    config = _load_config()
    for k, v in data.items():
        if k in config and isinstance(v, dict) and isinstance(config[k], dict):
            config[k].update(v)
        elif k in config:
            config[k] = v
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    return {"status": "success"}


# ─── Serve SPA ───────────────────────────────────────────────

_HTML_CACHE: str | None = None


@app.get("/{full_path:path}", response_class=HTMLResponse)
async def serve_spa():
    global _HTML_CACHE
    if _HTML_CACHE is None:
        html_path = TEMPLATES_DIR / "dashboard.html"
        if html_path.exists():
            _HTML_CACHE = html_path.read_text("utf-8")
        else:
            _HTML_CACHE = "<h1>Dashboard template not found</h1>"
    return HTMLResponse(_HTML_CACHE)


def main():
    import uvicorn
    print(f"  AgentTidal Dashboard: http://localhost:8080")
    print(f"  Proxy status: {'Running' if _proxy_status()['running'] else 'Stopped'}")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")


if __name__ == "__main__":
    main()
