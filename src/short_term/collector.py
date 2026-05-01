"""API transparent proxy for LM Studio — captures conversations to short-term memory.

Usage:
    python -m src.short_term.collector

The proxy runs on localhost:1235 and forwards to LM Studio at localhost:1234/v1.
Configure your client to use http://localhost:1235/v1 as the OpenAI base URL.
"""

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

import httpx
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI(title="AgentTidal Proxy")

LM_STUDIO_URL = "http://localhost:1234/v1"
SHORT_TERM_DIR = Path("memory/short_term")
PROXY_PORT = 1235

client = httpx.AsyncClient(base_url=LM_STUDIO_URL, timeout=300.0)


def _get_today_file() -> Path:
    SHORT_TERM_DIR.mkdir(parents=True, exist_ok=True)
    return SHORT_TERM_DIR / f"{date.today().isoformat()}.jsonl"


def _append_to_memory(entry: dict):
    filepath = _get_today_file()
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def _proxy_request(request: Request) -> Response:
    path = request.url.path
    if path.startswith("/v1/"):
        path = path[3:]

    body = await request.body()
    body_str = body.decode("utf-8", errors="replace")

    req_data = None
    if body:
        try:
            req_data = json.loads(body)
        except json.JSONDecodeError:
            req_data = None

    # Forward to LM Studio
    resp = await client.request(
        method=request.method,
        url=path,
        content=body,
        headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")},
        params=request.query_params,
    )

    # Only capture chat completions
    is_chat = path.endswith("/chat/completions") and request.method.upper() == "POST"
    stream = req_data and req_data.get("stream", False)

    if is_chat and stream:
        return await _handle_streaming_chat(req_data, resp)

    if is_chat and req_data:
        resp_body = resp.content.decode("utf-8", errors="replace")
        resp_data = json.loads(resp_body) if resp_body else {}

        _append_to_memory({
            "timestamp": time.time(),
            "model": resp_data.get("model", ""),
            "messages": req_data.get("messages", []),
            "response": {
                "content": resp_data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "role": resp_data.get("choices", [{}])[0].get("message", {}).get("role", "assistant"),
            },
            "usage": resp_data.get("usage", {}),
        })

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")},
    )


async def _handle_streaming_chat(req_data: dict, upstream_resp: Response) -> StreamingResponse:
    """Handle streaming responses — capture full content while streaming to client."""
    collected_content = []

    async def stream():
        async for chunk in upstream_resp.aiter_bytes():
            yield chunk
            for line in chunk.decode("utf-8", errors="replace").split("\n"):
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data and data != "[DONE]":
                        try:
                            chunk_data = json.loads(data)
                            delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                collected_content.append(delta["content"])
                        except json.JSONDecodeError:
                            pass

    async def _capture_on_complete():
        content = "".join(collected_content)
        if content:
            _append_to_memory({
                "timestamp": time.time(),
                "model": req_data.get("model", ""),
                "messages": req_data.get("messages", []),
                "response": {"content": content, "role": "assistant"},
                "usage": {},
            })

    background = BackgroundTasks()
    background.add_task(_capture_on_complete)
    return StreamingResponse(
        stream(),
        media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
        background=background,
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all(request: Request):
    return await _proxy_request(request)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def catch_all_v1(request: Request):
    return await _proxy_request(request)


def main():
    parser = argparse.ArgumentParser(description="AgentTidal LM Studio Proxy")
    parser.add_argument("--port", type=int, default=PROXY_PORT, help="Proxy listen port")
    parser.add_argument("--lm-studio-url", type=str, default=LM_STUDIO_URL, help="LM Studio base URL")
    parser.add_argument("--memory-dir", type=str, default=None, help="Short-term memory directory")
    args = parser.parse_args()

    global LM_STUDIO_URL, SHORT_TERM_DIR, PROXY_PORT, client
    if args.lm_studio_url:
        LM_STUDIO_URL = args.lm_studio_url
    if args.memory_dir:
        SHORT_TERM_DIR = Path(args.memory_dir)
    if args.port:
        PROXY_PORT = args.port

    client = httpx.AsyncClient(base_url=LM_STUDIO_URL, timeout=300.0)

    print(f"🌊 AgentTidal Proxy running on http://localhost:{PROXY_PORT}")
    print(f"   Forwarding to: {LM_STUDIO_URL}")
    print(f"   Saving conversations to: {SHORT_TERM_DIR}")
    print(f"   Configure your client to use http://localhost:{PROXY_PORT}/v1")
    uvicorn.run(app, host="127.0.0.1", port=PROXY_PORT, log_level="info")


if __name__ == "__main__":
    main()
