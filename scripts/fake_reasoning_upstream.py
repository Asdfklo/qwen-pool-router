#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse


def create_app(label: str) -> FastAPI:
    app = FastAPI(title=f"Nyx fake reasoning upstream {label}")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return "llamacpp:prompt_tokens_seconds 1000\nllamacpp:predicted_tokens_seconds 100\n"

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": "qwen-3.6-35b", "object": "model"}]}

    @app.post("/v1/chat/completions", response_model=None)
    async def completions(request: Request) -> JSONResponse | StreamingResponse:
        payload = await request.json()
        response_id = f"chatcmpl-fake-{uuid.uuid4().hex[:10]}"
        if payload.get("stream"):
            async def events() -> AsyncIterator[bytes]:
                chunks = [
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": payload.get("model"),
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": label}, "finish_reason": None}],
                    },
                    {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": payload.get("model"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                    },
                ]
                for chunk in chunks:
                    yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        return JSONResponse(
            {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": label},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                "nyx_received": payload,
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    uvicorn.run(create_app(args.label), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
