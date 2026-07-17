#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json

import httpx

from hermes_router import (
    RouterState,
    load_config,
    reserve_semantic_backend,
    run_semantic_backend_check,
)


async def run(config_path: str, tool_check: bool) -> int:
    config = load_config(config_path)
    state = RouterState(config)
    semantic = config.semantic_health.model_copy(update={"idle_seconds": 0})
    results: dict[str, bool] = {}
    timeout = httpx.Timeout(timeout=semantic.timeout_seconds, connect=config.connect_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for backend in state.backends:
            backend.last_watchdog_status = {"ready": True}
            reserved = await reserve_semantic_backend(state, backend, semantic)
            results[backend.config.name] = bool(
                reserved
                and await run_semantic_backend_check(
                    client,
                    state,
                    backend,
                    semantic,
                    force_tool_check=tool_check,
                )
            )
    print(json.dumps({"tool_check": tool_check, "results": results}, separators=(",", ":")))
    return 0 if results and all(results.values()) else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tool", action="store_true")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.config, args.tool)))


if __name__ == "__main__":
    main()
