#!/usr/bin/env python3
from __future__ import annotations

import argparse
import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:4000")
    args = parser.parse_args()
    with httpx.Client(timeout=10) as client:
        health = client.get(args.base + "/health")
        health.raise_for_status()
        models = client.get(args.base + "/v1/models")
        models.raise_for_status()
    print("router smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
