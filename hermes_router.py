#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sqlite3
import time
from collections import deque
import hashlib
import hmac
from threading import Lock as _ThreadLock
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from reasoning import (
    SUPPORTED_EFFORTS,
    canonical_reasoning_effort,
    parse_model_reasoning_suffix,
    resolve_reasoning_request,
    strip_client_reasoning_fields,
)


LOG = logging.getLogger("hermes-qwen-router")
ROUTER_UI_VERSION = "0.10.1"


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


class BackendConfig(BaseModel):
    name: str
    api_base: str
    watchdog_base: str | None = None
    backend_model: str = "qwen-3.6-35b"
    max_context_tokens: int = 262144
    max_parallel_requests: int = 1
    enabled: bool = True
    preferred_for_long_context: bool = False


class FallbackConfig(BaseModel):
    enabled: bool = False
    name: str = "fallback"
    model: str = ""
    api_base: str = ""
    api_key: str = ""


class DynamicThinkingConfig(BaseModel):
    enabled: bool = False
    classifier_timeout_seconds: float = 20
    default_thinking: bool = False
    max_classifier_prompt_chars: int = 6000
    min_thinking_max_tokens: int = 1024
    thinking_reasoning_budget: int = 128
    classifier_cache_ttl_seconds: float = 300
    classifier_cache_max_entries: int = 1024
    force_think_keywords: list[str] = Field(
        default_factory=lambda: [
            "think deeply",
            "reason carefully",
            "step by step",
            "debug",
            "architecture",
            "design a plan",
            "deep research",
            "benchmark",
            "optimize",
            "refactor",
            "root cause",
            "tradeoff",
            "compare options",
            "analyze this",
            "explain how",
            "help me understand",
            "implement",
            "write a function",
            "write a class",
            "write a script",
            "build a system",
            "design a solution",
            "solve this problem",
            "how does this work",
            "what's happening here",
            "review this code",
            "code review",
            "find the bug",
            "what went wrong",
            "why is this failing",
            "trace the execution",
            "walk through",
            "plan the migration",
            "strategy",
            "pros and cons",
            "evaluate",
            "investigate",
            "troubleshoot",
            "diagnose",
            "performance",
            "latency",
            "throughput",
            "scalability",
            "security audit",
            "threat model",
            "edge cases",
            "what if",
            "consider the",
            "weigh the options",
            "best approach",
            "what are the tradeoffs",
            "impact analysis",
            "breaking change",
            "migration plan",
            "integration",
            "dependency",
            "architecture decision",
            "ADR",
            "technical debt",
            "spike",
            "proof of concept",
            "POC",
            "benchmark test",
            "load test",
            "stress test",
            "regression",
            "root cause analysis",
            "incident",
            "postmortem",
            "technical spec",
            "design doc",
            "RFC",
            "proposal",
            "how to",
            "write me",
            "generate",
            "create a",
            "make a",
            "build me",
            "set up",
            "configure",
            "deploy",
            "migrate",
            "convert",
            "transform",
            "rewrite",
            "redesign",
            "restructure",
            "reorganize",
            "improve",
            "enhance",
            "upgrade",
            "scale",
            "parallelize",
            "concurrent",
            "threading",
            "async",
            "synchronization",
            "lock",
            "deadlock",
            "race condition",
            "memory leak",
            "stack trace",
            "exception",
            "error handling",
            "logging",
            "monitoring",
            "alerting",
            "testing",
            "unit test",
            "integration test",
            "e2e test",
            "CI/CD",
            "pipeline",
            "automation",
            "script",
            "tool",
            "CLI",
            "API",
            "SDK",
            "library",
            "framework",
            "database",
            "schema",
            "query",
            "index",
            "cache",
            "queue",
            "event",
            "webhook",
            "middleware",
            "routing",
            "authentication",
            "authorization",
            "encryption",
            "hashing",
            "serialization",
            "protocol",
            "specification",
            "compliance",
            "audit",
            "governance",
            "policy",
            "procedure",
            "workflow",
            "pipeline",
            "orchestration",
            "deployment",
            "rollback",
            "canary",
            "blue-green",
            "feature flag",
            "A/B test",
            "experiment",
            "hypothesis",
            "data analysis",
            "visualization",
            "dashboard",
            "report",
            "metrics",
            "KPI",
            "SLA",
            "SLO",
            "error budget",
            "on-call",
            "incident response",
            "runbook",
            "playbook",
        ]
    )
    force_no_think_keywords: list[str] = Field(
        default_factory=lambda: [
            "quick answer",
            "briefly",
            "no thinking",
            "don't think",
            "do not think",
            "simple answer",
            "just answer",
            "one word",
            "one-liner",
            "tl;dr",
            "tldr",
            "short answer",
            "concise",
            "in one sentence",
            "yes or no",
            "true or false",
            "hello",
            "hi there",
            "good morning",
            "good evening",
            "good night",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "sure",
            "got it",
            "understood",
            "noted",
            "will do",
            "sounds good",
            "perfect",
            "great",
            "awesome",
            "nice",
            "cool",
            "bye",
            "goodbye",
            "see you",
            "no problem",
            "you're welcome",
            "np",
            "nvm",
            "nevermind",
            "cancel",
            "stop",
            "abort",
            "never mind",
            "forget it",
            "disregard",
            "ignore",
            "skip",
            "pass",
            "next",
            "continue",
            "go on",
            "tell me more",
            "what else",
            "lol",
            "haha",
            "rofl",
            "lmao",
            "brb",
            "afk",
            "ttyl",
            "gtg",
            "omw",
            "idk",
            "imo",
            "imho",
            "fwiw",
            "btw",
            "fyi",
            "same",
            "agreed",
            "seconded",
            "yes",
            "no",
            "maybe",
            "perhaps",
            "definitely",
            "absolutely",
            "certainly",
            "of course",
            "naturally",
            "obviously",
            "clearly",
            "exactly",
            "correct",
            "wrong",
            "true",
            "false",
            "help",
            "status",
            "info",
            "about",
            "version",
            "what is the capital",
            "what year",
            "how many",
            "define",
        ]
    )


class ReasoningConfig(BaseModel):
    # None preserves the existing keyword and slash-command auto mode.
    default_effort: str | None = None
    model_defaults: dict[str, str] = Field(default_factory=dict)
    strip_client_reasoning_fields: bool = True
    expose_reasoning_models: bool = False
    effort_budgets: dict[str, int] = Field(
        default_factory=lambda: {
            "minimal": 128,
            "low": 512,
            "medium": 2048,
            "high": 8192,
            "xhigh": 16384,
        }
    )
    # Optional effort -> backend name(s) map. An absent map means every pooled
    # Qwen replica handles every effort with per-request controls.
    routes: dict[str, str | list[str]] = Field(default_factory=dict)


class SemanticHealthConfig(BaseModel):
    enabled: bool = False
    interval_seconds: float = 900
    initial_delay_seconds: float = 120
    idle_seconds: float = 30
    timeout_seconds: float = 20
    max_tokens: int = 12
    expected_response: str = "NYX_OK"
    tool_check_every: int = 4
    failure_threshold: int = 3
    enforce: bool = False


class RouterConfig(BaseModel):
    public_model_name: str = "qwen-3.6-35b"
    listen_host: str = "0.0.0.0"
    listen_port: int = 4000
    request_timeout_seconds: float = 180
    connect_timeout_seconds: float = 5
    cooldown_seconds: float = 60
    transient_cooldown_seconds: float = 12
    sustained_error_threshold: int = 3
    max_queue_seconds: float = 5
    max_queue_size: int = 64
    retry_after_seconds: int = 2
    request_deadline_seconds: float = 180
    health_cache_seconds: float = 3
    status_cache_seconds: float = 1.5
    telemetry_state_path: str = "/var/lib/hermes-qwen-pool/router-telemetry.json"
    telemetry_db_path: str = "/var/lib/hermes-qwen-pool/router-telemetry.sqlite3"
    telemetry_window_seconds: float = 86400
    telemetry_max_routes: int = 500
    telemetry_persist_interval_seconds: float = 5
    cache_affinity_enabled: bool = True
    cache_affinity_ttl_seconds: float = 300
    cache_affinity_prefix_tokens: int = 200
    circuit_failure_threshold: int = 3
    circuit_open_seconds: float = 30
    circuit_probe_interval_seconds: float = 10
    preferred_backend_names: list[str] = Field(default_factory=lambda: ["precision-qwen"])
    admin_token: str = ""
    backends: list[BackendConfig] = Field(default_factory=list)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    dynamic_thinking: DynamicThinkingConfig = Field(default_factory=DynamicThinkingConfig)
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    semantic_health: SemanticHealthConfig = Field(default_factory=SemanticHealthConfig)


class NodeRegistration(BaseModel):
    name: str
    api_base: str
    watchdog_base: str
    backend_model: str = "qwen-3.6-35b"
    max_context_tokens: int = 262144
    max_parallel_requests: int = 1
    enabled: bool = True
    preferred_for_long_context: bool = False
    node_os: str = ""
    hostname: str = ""


@dataclass
class BackendState:
    config: BackendConfig
    active_requests: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    cooldown_until: float = 0.0
    recent_errors: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    latency_ms_average: float = 0.0
    last_watchdog_status: dict[str, Any] | None = None
    last_health_checked: float = 0.0
    last_reject_reason: str = ""
    prompt_tokens_per_sec: float = 0.0
    generated_tokens_per_sec: float = 0.0
    last_selection_reason: str = ""
    circuit_state: str = "closed"
    circuit_opened_at: float = 0.0
    circuit_probe_in_flight: bool = False
    consecutive_failures: int = 0
    draining: bool = False
    estimated_finish_ms: float = 0.0
    active_leases: dict[str, float] = field(default_factory=dict)
    stale_active_resets: int = 0
    semantic_healthy: bool | None = None
    semantic_checked_at: float = 0.0
    semantic_last_success: float = 0.0
    semantic_last_failure: float = 0.0
    semantic_latency_ms: float = 0.0
    semantic_consecutive_failures: int = 0
    semantic_error: str = ""
    semantic_check_count: int = 0
    semantic_tool_checked_at: float = 0.0
    semantic_tool_healthy: bool | None = None
    semantic_in_progress: bool = False
    semantic_lease_id: str = ""


class RouterState:
    def __init__(self, config: RouterConfig) -> None:
        self.config = config
        self.backends = [BackendState(config=b) for b in config.backends]
        self.fallback_usage_count = 0
        self.started_at = time.time()
        self.total_requests = 0
        self.completed_requests = 0
        self.failed_requests = 0
        self.stream_requests = 0
        self.thinking_classifier_requests = 0
        self.thinking_enabled_requests = 0
        self.thinking_disabled_requests = 0
        self.cancelled_requests = 0
        self.retried_requests = 0
        self.queue_rejected_requests = 0
        self.config_reload_count = 0
        self.last_config_reload_at = 0.0
        self.last_config_reload_error = ""
        self.last_request_activity = time.time()
        self.backend_selection_counts: dict[str, int] = {}
        self.queued_requests: dict[int, dict[str, float | int]] = {}
        self.recent_routes: deque[dict[str, Any]] = deque(maxlen=config.telemetry_max_routes)
        self.telemetry_dirty = False
        self.status_revision = 0
        self._lock = asyncio.Lock()
        self.affinity = SessionAffinity(ttl=config.cache_affinity_ttl_seconds)
        self.classifier_cache = ThinkingClassifierCache(
            ttl=config.dynamic_thinking.classifier_cache_ttl_seconds,
            max_entries=config.dynamic_thinking.classifier_cache_max_entries,
        )

    def backend_by_name(self, name: str) -> BackendState | None:
        return next((b for b in self.backends if b.config.name == name), None)

    async def add_or_update_backend(self, backend_config: BackendConfig) -> None:
        async with self._lock:
            existing = self.backend_by_name(backend_config.name)
            if existing:
                existing.config = backend_config
                existing.last_reject_reason = ""
                existing.cooldown_until = 0
                existing.last_watchdog_status = None
                existing.last_health_checked = 0
            else:
                self.backends.append(BackendState(config=backend_config))

    async def mark_request_started(self, *, stream: bool, priority: int, deadline_at: float) -> int | None:
        async with self._lock:
            self.last_request_activity = time.time()
            if len(self.queued_requests) >= self.config.max_queue_size:
                self.queue_rejected_requests += 1
                self.telemetry_dirty = True
                self.status_revision += 1
                return None
            self.total_requests += 1
            if stream:
                self.stream_requests += 1
            self.queued_requests[self.total_requests] = {
                "started_at": time.time(),
                "priority": max(0, min(9, priority)),
                "deadline_at": deadline_at,
            }
            self.telemetry_dirty = True
            self.status_revision += 1
            return self.total_requests

    async def mark_backend_selected(self, backend: BackendState, request_id: int) -> float:
        async with self._lock:
            name = backend.config.name
            self.backend_selection_counts[name] = self.backend_selection_counts.get(name, 0) + 1
            self.telemetry_dirty = True
            self.status_revision += 1
            queued = self.queued_requests.pop(request_id, None) or {"started_at": time.time()}
            return max(0.0, (time.time() - float(queued["started_at"])) * 1000)

    async def can_attempt(self, request_id: int) -> bool:
        async with self._lock:
            current = self.queued_requests.get(request_id)
            if current is None:
                return False
            current_priority = int(current["priority"])
            current_started = float(current["started_at"])
            return not any(
                int(other["priority"]) > current_priority
                or (int(other["priority"]) == current_priority and float(other["started_at"]) < current_started)
                for other_id, other in self.queued_requests.items()
                if other_id != request_id
            )

    async def mark_retry(self) -> None:
        async with self._lock:
            self.retried_requests += 1
            self.telemetry_dirty = True
            self.status_revision += 1

    async def mark_cancelled(self) -> None:
        async with self._lock:
            self.cancelled_requests += 1
            self.telemetry_dirty = True
            self.status_revision += 1

    async def remove_from_queue(self, request_id: int) -> None:
        async with self._lock:
            self.queued_requests.pop(request_id, None)
            self.status_revision += 1

    async def record_thinking_decision(self, *, enabled: bool, classifier_used: bool) -> None:
        async with self._lock:
            if classifier_used:
                self.thinking_classifier_requests += 1
            if enabled:
                self.thinking_enabled_requests += 1
            else:
                self.thinking_disabled_requests += 1
            self.telemetry_dirty = True
            self.status_revision += 1

    async def record_route(
        self,
        *,
        request_id: int,
        backend_name: str,
        status: str,
        stream: bool,
        latency_ms: float = 0.0,
        queue_ms: float = 0.0,
        classifier_ms: float = 0.0,
        total_ms: float = 0.0,
        routing_reason: str = "",
        thinking_reason: str = "",
        thinking_enabled: bool | None = None,
        error: str = "",
        ttft_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        tokens_per_second: float = 0.0,
        prompt_ms: float = 0.0,
        generation_ms: float = 0.0,
        cache_hit_estimate: bool | None = None,
    ) -> None:
        async with self._lock:
            if status == "success":
                self.completed_requests += 1
            else:
                self.failed_requests += 1
            self.recent_routes.appendleft(
                {
                    "request_id": request_id,
                    "backend_name": backend_name,
                    "status": status,
                    "stream": stream,
                    "latency_ms": round(latency_ms, 2),
                    "backend_ms": round(latency_ms, 2),
                    "queue_ms": round(queue_ms, 2),
                    "classifier_ms": round(classifier_ms, 2),
                    "total_ms": round(total_ms or latency_ms, 2),
                    "routing_reason": routing_reason,
                    "thinking_reason": thinking_reason,
                    "thinking_enabled": thinking_enabled,
                    "error": error,
                    "ttft_ms": round(ttft_ms, 2),
                    "prompt_tokens": max(0, int(prompt_tokens)),
                    "completion_tokens": max(0, int(completion_tokens)),
                    "cached_tokens": max(0, int(cached_tokens)),
                    "tokens_per_second": round(tokens_per_second, 2),
                    "prompt_ms": round(prompt_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "context_tokens": max(0, int(prompt_tokens)),
                    "cache_hit_estimate": cache_hit_estimate,
                    "created_at": time.time(),
                }
            )
            self._prune_routes_locked()
            self.telemetry_dirty = True
            self.status_revision += 1

    def _prune_routes_locked(self) -> None:
        cutoff = time.time() - self.config.telemetry_window_seconds
        retained = [route for route in self.recent_routes if float(route.get("created_at") or 0) >= cutoff]
        self.recent_routes = deque(retained[: self.config.telemetry_max_routes], maxlen=self.config.telemetry_max_routes)

    async def telemetry_payload(self) -> dict[str, Any] | None:
        async with self._lock:
            if not self.telemetry_dirty:
                return None
            self._prune_routes_locked()
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "fallback_usage_count": self.fallback_usage_count,
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests,
                "failed_requests": self.failed_requests,
                "stream_requests": self.stream_requests,
                "thinking_classifier_requests": self.thinking_classifier_requests,
                "thinking_enabled_requests": self.thinking_enabled_requests,
                "thinking_disabled_requests": self.thinking_disabled_requests,
                "cancelled_requests": self.cancelled_requests,
                "retried_requests": self.retried_requests,
                "queue_rejected_requests": self.queue_rejected_requests,
                "backend_selection_counts": self.backend_selection_counts,
                "recent_routes": list(self.recent_routes),
                "cache_affinity": self.affinity.persistence_payload(),
                "classifier_cache": self.classifier_cache.persistence_payload(),
                "backends": {
                    backend.config.name: {
                        "last_success": backend.last_success,
                        "last_failure": backend.last_failure,
                        "recent_errors": list(backend.recent_errors),
                        "latency_ms_average": backend.latency_ms_average,
                        "circuit_state": backend.circuit_state,
                        "circuit_opened_at": backend.circuit_opened_at,
                        "consecutive_failures": backend.consecutive_failures,
                    }
                    for backend in self.backends
                },
            }
            self.telemetry_dirty = False
            return payload

    def restore_telemetry(self, payload: dict[str, Any]) -> None:
        cutoff = time.time() - self.config.telemetry_window_seconds
        routes = [
            route
            for route in payload.get("recent_routes") or []
            if isinstance(route, dict) and float(route.get("created_at") or 0) >= cutoff
        ][: self.config.telemetry_max_routes]
        self.recent_routes = deque(routes, maxlen=self.config.telemetry_max_routes)
        for field_name in (
            "fallback_usage_count", "total_requests", "completed_requests", "failed_requests",
            "stream_requests", "thinking_classifier_requests", "thinking_enabled_requests",
            "thinking_disabled_requests",
            "cancelled_requests", "retried_requests", "queue_rejected_requests",
        ):
            setattr(self, field_name, max(0, int(payload.get(field_name) or 0)))
        counts = payload.get("backend_selection_counts") or {}
        self.backend_selection_counts = {str(name): max(0, int(count)) for name, count in counts.items()}
        self.affinity.restore(payload.get("cache_affinity") or {})
        self.classifier_cache.restore(payload.get("classifier_cache") or {})
        backend_payloads = payload.get("backends") or {}
        for backend in self.backends:
            saved = backend_payloads.get(backend.config.name) or {}
            backend.last_success = float(saved.get("last_success") or 0)
            backend.last_failure = float(saved.get("last_failure") or 0)
            backend.latency_ms_average = max(0.0, float(saved.get("latency_ms_average") or 0))
            backend.circuit_state = str(saved.get("circuit_state") or "closed")
            backend.circuit_opened_at = float(saved.get("circuit_opened_at") or 0)
            backend.consecutive_failures = max(0, int(saved.get("consecutive_failures") or 0))
            backend.recent_errors = deque(
                (float(ts) for ts in saved.get("recent_errors") or [] if float(ts) >= cutoff),
                maxlen=20,
            )
        self.telemetry_dirty = False


def _open_telemetry_db(config: RouterConfig) -> sqlite3.Connection:
    db_path = Path(config.telemetry_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=5)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE IF NOT EXISTS router_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS request_history (
        request_id INTEGER PRIMARY KEY, created_at REAL NOT NULL, backend_name TEXT NOT NULL,
        status TEXT NOT NULL, stream INTEGER NOT NULL, latency_ms REAL NOT NULL,
        queue_ms REAL NOT NULL, classifier_ms REAL NOT NULL, total_ms REAL NOT NULL,
        ttft_ms REAL NOT NULL DEFAULT 0, prompt_tokens INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0, cached_tokens INTEGER NOT NULL DEFAULT 0,
        tokens_per_second REAL NOT NULL DEFAULT 0,
        prompt_ms REAL NOT NULL DEFAULT 0, generation_ms REAL NOT NULL DEFAULT 0,
        cache_hit_estimate INTEGER, routing_reason TEXT NOT NULL, thinking_reason TEXT NOT NULL,
        thinking_enabled INTEGER, error TEXT NOT NULL)"""
    )
    existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(request_history)")}
    for column_name, definition in {
        "cached_tokens": "INTEGER NOT NULL DEFAULT 0",
        "prompt_ms": "REAL NOT NULL DEFAULT 0",
        "generation_ms": "REAL NOT NULL DEFAULT 0",
    }.items():
        if column_name not in existing_columns:
            connection.execute(f"ALTER TABLE request_history ADD COLUMN {column_name} {definition}")
    return connection


def load_telemetry_state(config: RouterConfig) -> dict[str, Any] | None:
    try:
        with _open_telemetry_db(config) as connection:
            row = connection.execute("SELECT value FROM router_state WHERE key='snapshot'").fetchone()
            if row:
                payload = json.loads(row[0])
                columns = [description[0] for description in connection.execute("SELECT * FROM request_history LIMIT 0").description]
                routes = connection.execute(
                    "SELECT * FROM request_history WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                    (time.time() - config.telemetry_window_seconds, config.telemetry_max_routes),
                ).fetchall()
                payload["recent_routes"] = [dict(zip(columns, route)) for route in routes]
                for route in payload["recent_routes"]:
                    for field_name in ("stream", "thinking_enabled", "cache_hit_estimate"):
                        if route[field_name] is not None:
                            route[field_name] = bool(route[field_name])
                return payload
        legacy_path = Path(config.telemetry_state_path)
        if legacy_path.exists():
            payload = json.loads(legacy_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        return None
    except Exception as exc:
        structured_log("telemetry_restore_failed", error=type(exc).__name__)
        return None


def write_telemetry_state(config: RouterConfig, payload: dict[str, Any]) -> None:
    routes = payload.pop("recent_routes", [])
    with _open_telemetry_db(config) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO router_state(key, value) VALUES('snapshot', ?)",
            (json.dumps(payload, separators=(",", ":")),),
        )
        for route in routes:
            connection.execute(
                """INSERT OR REPLACE INTO request_history
                (request_id, created_at, backend_name, status, stream, latency_ms, queue_ms,
                classifier_ms, total_ms, ttft_ms, prompt_tokens, completion_tokens, cached_tokens,
                tokens_per_second, prompt_ms, generation_ms, cache_hit_estimate, routing_reason, thinking_reason,
                thinking_enabled, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    route.get("request_id"), route.get("created_at", time.time()), route.get("backend_name", ""),
                    route.get("status", ""), int(bool(route.get("stream"))), route.get("latency_ms", 0),
                    route.get("queue_ms", 0), route.get("classifier_ms", 0), route.get("total_ms", 0),
                    route.get("ttft_ms", 0), route.get("prompt_tokens", 0), route.get("completion_tokens", 0),
                    route.get("cached_tokens", 0),
                    route.get("tokens_per_second", 0),
                    route.get("prompt_ms", 0), route.get("generation_ms", 0),
                    None if route.get("cache_hit_estimate") is None else int(bool(route.get("cache_hit_estimate"))),
                    route.get("routing_reason", ""), route.get("thinking_reason", ""),
                    None if route.get("thinking_enabled") is None else int(bool(route.get("thinking_enabled"))),
                    route.get("error", ""),
                ),
            )
        connection.execute("DELETE FROM request_history WHERE created_at < ?", (time.time() - config.telemetry_window_seconds,))
        connection.execute(
            "DELETE FROM request_history WHERE request_id NOT IN "
            "(SELECT request_id FROM request_history ORDER BY created_at DESC LIMIT ?)",
            (config.telemetry_max_routes,),
        )


async def persist_telemetry(state: RouterState, config: RouterConfig) -> None:
    payload = await state.telemetry_payload()
    if payload is None:
        return
    try:
        await asyncio.to_thread(write_telemetry_state, config, payload)
    except Exception as exc:
        state.telemetry_dirty = True
        structured_log("telemetry_persist_failed", error=type(exc).__name__)


async def telemetry_persist_loop(state: RouterState, config: RouterConfig) -> None:
    while True:
        await asyncio.sleep(config.telemetry_persist_interval_seconds)
        await persist_telemetry(state, config)


class SessionAffinity:
    """Maps conversation prefixes to backends for KV cache affinity."""

    def __init__(self, ttl: float = 300) -> None:
        self._ttl = ttl
        self._lock = _ThreadLock()
        self._map: dict[str, tuple[str, float]] = {}  # key -> (backend_name, timestamp)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _extract_prefix_key(self, payload: dict[str, Any], explicit_key: str = "") -> str | None:
        """Hash the conversation prefix to generate a routing key."""
        if explicit_key:
            return hashlib.sha256(f"session:{explicit_key}".encode("utf-8")).hexdigest()[:16]
        messages = payload.get("messages", [])
        if not messages:
            return None
        # Take system prompt + first user message as the stable prefix
        prefix_parts = []
        token_budget = 200  # ~200 tokens ≈ 800 chars
        char_budget = token_budget * 4
        char_count = 0
        for msg in messages[:3]:  # system + first user + first assistant
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle content blocks (multimodal)
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            if not content:
                continue
            prefix_parts.append(f"{role}:{content[:char_budget - char_count]}")
            char_count += len(content)
            if char_count >= char_budget:
                break
        if not prefix_parts:
            return None
        raw = "|".join(prefix_parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def lookup(self, payload: dict[str, Any], explicit_key: str = "") -> str | None:
        """Return the pinned backend name, or None if no affinity."""
        key = self._extract_prefix_key(payload, explicit_key)
        if not key:
            return None
        with self._lock:
            entry = self._map.get(key)
            if entry is None:
                self.misses += 1
                return None
            backend_name, ts = entry
            if time.time() - ts > self._ttl:
                del self._map[key]
                self.evictions += 1
                self.misses += 1
                return None
            self.hits += 1
            return backend_name

    def record(self, payload: dict[str, Any], backend_name: str, explicit_key: str = "") -> None:
        """Record that this prefix was routed to this backend."""
        key = self._extract_prefix_key(payload, explicit_key)
        if not key:
            return
        with self._lock:
            self._map[key] = (backend_name, time.time())

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        removed = 0
        with self._lock:
            expired = [k for k, (_, ts) in self._map.items() if now - ts > self._ttl]
            for k in expired:
                del self._map[k]
                removed += 1
            self.evictions += removed
        return removed

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._map)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._map),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "ttl_seconds": self._ttl,
            }

    def persistence_payload(self) -> dict[str, Any]:
        cutoff = time.time() - self._ttl
        with self._lock:
            entries = [(key, name, ts) for key, (name, ts) in self._map.items() if ts >= cutoff]
            return {
                "entries": entries[-2048:],
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def restore(self, payload: dict[str, Any]) -> None:
        cutoff = time.time() - self._ttl
        with self._lock:
            self._map = {
                str(key): (str(name), float(ts))
                for key, name, ts in (payload.get("entries") or [])[-2048:]
                if float(ts) >= cutoff
            }
            self.hits = max(0, int(payload.get("hits") or 0))
            self.misses = max(0, int(payload.get("misses") or 0))
            self.evictions = max(0, int(payload.get("evictions") or 0))


class ThinkingClassifierCache:
    """TTL-based LRU cache for thinking classifier results.

    Keyed by SHA256 of the classification prompt text. Avoids redundant LLM
    classification calls for repeated or similar prompts.
    """

    def __init__(self, ttl: float = 300, max_entries: int = 1024) -> None:
        self._ttl = ttl
        self._max_entries = max_entries
        self._lock = _ThreadLock()
        self._map: dict[str, tuple[bool, float]] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _make_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def lookup(self, text: str) -> bool | None:
        key = self._make_key(text)
        with self._lock:
            entry = self._map.get(key)
            if entry is None:
                self.misses += 1
                return None
            thinking, ts = entry
            if time.time() - ts > self._ttl:
                del self._map[key]
                self.evictions += 1
                self.misses += 1
                return None
            self.hits += 1
            return thinking

    def record(self, text: str, thinking: bool) -> None:
        key = self._make_key(text)
        with self._lock:
            if len(self._map) >= self._max_entries:
                oldest_key = min(self._map, key=lambda k: self._map[k][1])
                del self._map[oldest_key]
                self.evictions += 1
            self._map[key] = (thinking, time.time())

    def evict_expired(self) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            expired = [k for k, (_, ts) in self._map.items() if now - ts > self._ttl]
            for k in expired:
                del self._map[k]
                removed += 1
            self.evictions += removed
        return removed

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._map)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._map),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "ttl_seconds": self._ttl,
                "max_entries": self._max_entries,
            }

    def persistence_payload(self) -> dict[str, Any]:
        cutoff = time.time() - self._ttl
        with self._lock:
            entries = [(key, thinking, ts) for key, (thinking, ts) in self._map.items() if ts >= cutoff]
            return {
                "entries": entries[-self._max_entries:],
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }

    def restore(self, payload: dict[str, Any]) -> None:
        cutoff = time.time() - self._ttl
        with self._lock:
            self._map = {
                str(key): (bool(thinking), float(ts))
                for key, thinking, ts in (payload.get("entries") or [])[-self._max_entries:]
                if float(ts) >= cutoff
            }
            self.hits = max(0, int(payload.get("hits") or 0))
            self.misses = max(0, int(payload.get("misses") or 0))
            self.evictions = max(0, int(payload.get("evictions") or 0))


def validate_config(config: RouterConfig) -> None:
    errors: list[str] = []
    if not (1 <= config.listen_port <= 65535):
        errors.append("listen_port must be between 1 and 65535")
    if config.max_queue_seconds <= 0:
        errors.append("max_queue_seconds must be positive")
    if config.max_queue_size < 1:
        errors.append("max_queue_size must be at least 1")
    if config.retry_after_seconds < 1:
        errors.append("retry_after_seconds must be at least 1")
    if not config.backends:
        errors.append("at least one backend is required")

    backend_names = [backend.name for backend in config.backends]
    known_backends = set(backend_names)
    if len(backend_names) != len(known_backends):
        errors.append("backend names must be unique")
    for backend in config.backends:
        if not backend.api_base.startswith(("http://", "https://")):
            errors.append(f"backend {backend.name!r} api_base must use http:// or https://")
        if backend.max_parallel_requests < 1:
            errors.append(f"backend {backend.name!r} max_parallel_requests must be at least 1")
        if backend.max_context_tokens < 1:
            errors.append(f"backend {backend.name!r} max_context_tokens must be positive")

    default_effort = config.reasoning.default_effort
    if default_effort is not None and canonical_reasoning_effort(default_effort) is None:
        errors.append(f"reasoning.default_effort {default_effort!r} is invalid")
    for model, effort in config.reasoning.model_defaults.items():
        if canonical_reasoning_effort(effort) is None:
            errors.append(f"reasoning.model_defaults[{model!r}]={effort!r} is invalid")
    for effort, budget in config.reasoning.effort_budgets.items():
        if effort not in SUPPORTED_EFFORTS or effort == "none":
            errors.append(f"reasoning effort budget key {effort!r} is invalid")
        if budget < 0:
            errors.append(f"reasoning effort budget {effort!r} must not be negative")
    for effort, routes in config.reasoning.routes.items():
        if effort not in SUPPORTED_EFFORTS:
            errors.append(f"reasoning route effort {effort!r} is invalid")
        route_names = [routes] if isinstance(routes, str) else routes
        missing = sorted(set(route_names) - known_backends)
        if missing:
            errors.append(f"reasoning route {effort!r} references unknown backends: {', '.join(missing)}")

    semantic = config.semantic_health
    if semantic.enabled and semantic.interval_seconds < 60:
        errors.append("semantic_health.interval_seconds must be at least 60 when enabled")
    if semantic.initial_delay_seconds < 0 or semantic.idle_seconds < 0:
        errors.append("semantic health delays must not be negative")
    if not (1 <= semantic.timeout_seconds <= 120):
        errors.append("semantic_health.timeout_seconds must be between 1 and 120")
    if not (4 <= semantic.max_tokens <= 64):
        errors.append("semantic_health.max_tokens must be between 4 and 64")
    if not semantic.expected_response.strip():
        errors.append("semantic_health.expected_response must not be empty")
    if semantic.tool_check_every < 0:
        errors.append("semantic_health.tool_check_every must not be negative")
    if semantic.failure_threshold < 1:
        errors.append("semantic_health.failure_threshold must be at least 1")

    if errors:
        raise ValueError("invalid router config:\n- " + "\n- ".join(errors))


def load_config(path: str) -> RouterConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    config = RouterConfig.model_validate(expand_env(raw))
    validate_config(config)
    return config


def persist_backend(config_path: str, backend: BackendConfig) -> None:
    if not config_path:
        return
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    backends = raw.setdefault("backends", [])
    item = backend.model_dump()
    for idx, existing in enumerate(backends):
        if existing.get("name") == backend.name:
            backends[idx] = item
            break
    else:
        backends.append(item)
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)


def llama_health_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base.rstrip("/") + "/health"


def structured_log(event: str, **fields: Any) -> None:
    LOG.info(json.dumps({"event": event, **fields}, separators=(",", ":"), default=str))


def active_request_stale_seconds(config: RouterConfig) -> float:
    return max(config.request_deadline_seconds, config.request_timeout_seconds) + max(60.0, config.retry_after_seconds)


def acquire_backend_lease(backend: BackendState, request_id: int, *, purpose: str = "request") -> str:
    lease_id = f"{purpose}:{request_id}:{time.monotonic_ns()}"
    backend.active_leases[lease_id] = time.time()
    backend.active_requests = len(backend.active_leases)
    return lease_id


def release_backend_lease(backend: BackendState, lease_id: str | None) -> None:
    if lease_id and lease_id in backend.active_leases:
        backend.active_leases.pop(lease_id, None)
    elif backend.active_leases:
        oldest = min(backend.active_leases, key=backend.active_leases.get)
        backend.active_leases.pop(oldest, None)
    backend.active_requests = len(backend.active_leases)


def reap_stale_active_leases(backend: BackendState, config: RouterConfig, *, now: float | None = None) -> int:
    if not backend.active_leases:
        if backend.active_requests != 0:
            backend.active_requests = 0
            backend.stale_active_resets += 1
            structured_log(
                "stale_active_requests_reset",
                backend=backend.config.name,
                reason="missing_active_lease",
            )
            return 1
        return 0
    checked_at = time.time() if now is None else now
    stale_after = active_request_stale_seconds(config)
    stale = [
        lease_id
        for lease_id, started_at in backend.active_leases.items()
        if checked_at - started_at > stale_after
    ]
    for lease_id in stale:
        backend.active_leases.pop(lease_id, None)
    if stale:
        backend.active_requests = len(backend.active_leases)
        backend.stale_active_resets += len(stale)
        structured_log(
            "stale_active_requests_reset",
            backend=backend.config.name,
            count=len(stale),
            stale_after_seconds=round(stale_after, 2),
        )
    else:
        backend.active_requests = len(backend.active_leases)
    return len(stale)


async def fetch_watchdog(client: httpx.AsyncClient, backend: BackendState, cache_seconds: float) -> dict[str, Any]:
    now = time.time()
    if backend.last_watchdog_status is not None and now - backend.last_health_checked < cache_seconds:
        return backend.last_watchdog_status
    cfg = backend.config
    if not cfg.watchdog_base:
        status = {"ready": True, "reason": "watchdog_not_configured", "gpu": {}, "llama": {"healthy": True}}
    else:
        try:
            resp = await client.get(cfg.watchdog_base.rstrip("/") + "/ready")
            resp.raise_for_status()
            status = resp.json()
        except Exception as exc:
            status = {"ready": False, "reason": "watchdog_unreachable", "error": type(exc).__name__}
    backend.last_watchdog_status = status
    reason = str(status.get("reason") or "")
    backend.draining = reason in {"manual_drain", "game_detected", "game_cooldown", "draining"}
    if status.get("ready", False):
        backend.last_reject_reason = ""
    backend.last_health_checked = now
    return status


async def llama_healthy(client: httpx.AsyncClient, backend: BackendState) -> bool:
    try:
        resp = await client.get(llama_health_url(backend.config.api_base))
        return resp.status_code < 500
    except Exception:
        return False


async def fetch_llama_metrics(client: httpx.AsyncClient, api_base: str) -> tuple[float, float]:
    """Fetch prompt_tokens_seconds and predicted_tokens_seconds from llama.cpp /metrics."""
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    metrics_url = base.rstrip("/") + "/metrics"
    try:
        resp = await client.get(metrics_url, timeout=3.0)
        resp.raise_for_status()
        text = resp.text
        prompt = 0.0
        predicted = 0.0
        for line in text.splitlines():
            m = re.match(r"^llamacpp:prompt_tokens_seconds\s+(\S+)", line)
            if m:
                try:
                    prompt = float(m.group(1))
                except ValueError:
                    pass
            m = re.match(r"^llamacpp:predicted_tokens_seconds\s+(\S+)", line)
            if m:
                try:
                    predicted = float(m.group(1))
                except ValueError:
                    pass
        return prompt, predicted
    except Exception:
        return 0.0, 0.0


def recent_error_penalty(backend: BackendState) -> float:
    now = time.time()
    while backend.recent_errors and now - backend.recent_errors[0] > 300:
        backend.recent_errors.popleft()
    return float(len(backend.recent_errors) * 500)


def estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    text = json.dumps(payload.get("messages", []), ensure_ascii=False)
    return max(1, len(text) // 4)


def score_backend(
    backend: BackendState,
    watchdog: dict[str, Any],
    *,
    prompt_tokens: int = 1,
    expected_output_tokens: int = 512,
    long_context: bool = False,
) -> float:
    gpu = watchdog.get("gpu") or {}
    prompt_ms = prompt_tokens / max(backend.prompt_tokens_per_sec, 250.0) * 1000
    generation_ms = expected_output_tokens / max(backend.generated_tokens_per_sec, 20.0) * 1000
    request_ms = max(backend.latency_ms_average, prompt_ms + generation_ms)
    backend.estimated_finish_ms = request_ms * (backend.active_requests + 1)
    score = 100000.0 - backend.estimated_finish_ms
    score -= float(gpu.get("memory_used_percent") or 0) * 5
    score -= float(gpu.get("utilization_percent") or 0) * 3
    score -= recent_error_penalty(backend)
    if watchdog.get("reason") == "game_detected" or watchdog.get("process_matches"):
        score -= 10000
    if long_context and backend.config.preferred_for_long_context:
        score += 500
    return score


async def rejection_reason(client: httpx.AsyncClient, backend: BackendState, config: RouterConfig) -> str | None:
    now = time.time()
    cfg = backend.config
    if not cfg.enabled:
        return "disabled"
    semantic = config.semantic_health
    if (
        semantic.enforce
        and backend.semantic_healthy is False
        and backend.semantic_consecutive_failures >= semantic.failure_threshold
    ):
        return "semantic_unhealthy"
    if backend.draining:
        return "draining"
    if backend.circuit_state == "open":
        if now - backend.circuit_opened_at < config.circuit_open_seconds:
            return "circuit_open"
        if backend.circuit_probe_in_flight:
            return "circuit_probe_in_flight"
        backend.circuit_state = "half_open"
    if backend.cooldown_until > now:
        return "cooldown"
    reap_stale_active_leases(backend, config, now=now)
    if backend.active_requests >= cfg.max_parallel_requests:
        return "busy"
    if len(backend.recent_errors) >= 5 and now - backend.recent_errors[-1] < config.cooldown_seconds:
        return "too_many_recent_failures"
    watchdog = await fetch_watchdog(client, backend, config.health_cache_seconds)
    if not watchdog.get("ready", False):
        return str(watchdog.get("reason") or "watchdog_not_ready")
    if not await llama_healthy(client, backend):
        return "llama_health_failed"
    return None


async def choose_backend(
    client: httpx.AsyncClient,
    state: RouterState,
    payload: dict[str, Any],
    request_id: int,
    exclude: set[str] | None = None,
    affinity_key: str = "",
    allowed_backend_names: set[str] | None = None,
) -> tuple[BackendState, str] | None:
    exclude = exclude or set()
    long_context = estimate_prompt_tokens(payload) >= 32768
    prompt_tokens = estimate_prompt_tokens(payload)
    expected_output_tokens = max(1, int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 512))

    # --- Phase 1: Try cache affinity (pin to same backend for KV reuse) ---
    pinned_name: str | None = None
    if state.config.cache_affinity_enabled:
        pinned_name = state.affinity.lookup(payload, affinity_key)

    async with state._lock:
        # Phase 1: Try pinned backend first
        if (
            pinned_name
            and pinned_name not in exclude
            and (allowed_backend_names is None or pinned_name in allowed_backend_names)
        ):
            pinned = state.backend_by_name(pinned_name)
            if pinned:
                reason = await rejection_reason(client, pinned, state.config)
                if not reason:
                    pinned.last_reject_reason = ""
                    lease_id = acquire_backend_lease(pinned, request_id)
                    pinned.last_selection_reason = "affinity"
                    pinned.circuit_probe_in_flight = pinned.circuit_state == "half_open"
                    state.status_revision += 1
                    state.affinity.record(payload, pinned.config.name, affinity_key)
                    structured_log(
                        "affinity_hit",
                        backend=pinned.config.name,
                        active_request_count=pinned.active_requests,
                    )
                    return pinned, lease_id
                else:
                    structured_log("affinity_miss", backend=pinned_name, reason=reason)

        # Phase 2: choose the node with the lowest estimated completion time.
        choices: list[tuple[float, BackendState]] = []
        for backend in state.backends:
            if backend.config.name in exclude:
                continue
            if allowed_backend_names is not None and backend.config.name not in allowed_backend_names:
                continue
            reason = await rejection_reason(client, backend, state.config)
            if reason:
                backend.last_reject_reason = reason
                structured_log("rejected_backend", backend=backend.config.name, reason=reason)
                continue
            backend.last_reject_reason = ""
            watchdog = backend.last_watchdog_status or {}
            score = score_backend(
                backend,
                watchdog,
                prompt_tokens=prompt_tokens,
                expected_output_tokens=expected_output_tokens,
                long_context=long_context,
            )
            if backend.config.name in state.config.preferred_backend_names:
                score += 250.0
            choices.append((score, backend))
        if not choices:
            return None
        choices.sort(key=lambda item: item[0], reverse=True)
        selected = choices[0][1]
        lease_id = acquire_backend_lease(selected, request_id)
        selected.circuit_probe_in_flight = selected.circuit_state == "half_open"
        selected.last_selection_reason = "long_context" if long_context and selected.config.preferred_for_long_context else "estimated_completion"
        state.status_revision += 1
        # Only record affinity on cache miss (no prior pin), so a temporary
        # rejection doesn't permanently overwrite the original affinity mapping.
        if state.config.cache_affinity_enabled and pinned_name is None:
            state.affinity.record(payload, selected.config.name, affinity_key)
        structured_log(
            "selected_backend",
            backend=selected.config.name,
            score=choices[0][0],
            active_request_count=selected.active_requests,
        )
        return selected, lease_id


def backend_snapshot(backend: BackendState) -> dict[str, Any]:
    cfg = backend.config
    watchdog = backend.last_watchdog_status or {}
    gpu = watchdog.get("gpu") or {}
    llama = watchdog.get("llama") or {}
    now = time.time()
    active_ages = [max(0.0, now - started_at) for started_at in backend.active_leases.values()]
    return {
        "name": cfg.name,
        "enabled": cfg.enabled,
        "api_base": cfg.api_base,
        "watchdog_base": cfg.watchdog_base,
        "backend_model": cfg.backend_model,
        "max_context_tokens": cfg.max_context_tokens,
        "max_parallel_requests": cfg.max_parallel_requests,
        "preferred_for_long_context": cfg.preferred_for_long_context,
        "active_requests": backend.active_requests,
        "active_request_age_seconds": round(max(active_ages, default=0.0), 2),
        "stale_active_resets": backend.stale_active_resets,
        "draining": backend.draining,
        "circuit_state": backend.circuit_state,
        "consecutive_failures": backend.consecutive_failures,
        "estimated_finish_ms": round(backend.estimated_finish_ms, 2),
        "cooldown_until": backend.cooldown_until,
        "last_success": backend.last_success,
        "last_failure": backend.last_failure,
        "recent_error_count": len(backend.recent_errors),
        "latency_ms_average": round(backend.latency_ms_average, 2),
        "last_reject_reason": backend.last_reject_reason,
        "ready": bool(watchdog.get("ready", False)),
        "ready_reason": watchdog.get("reason"),
        "process_matches": watchdog.get("process_matches") or [],
        "gpu": gpu,
        "llama": llama,
        "watchdog_checked_at": watchdog.get("checked_at"),
        "prompt_tokens_per_sec": round(backend.prompt_tokens_per_sec, 1),
        "generated_tokens_per_sec": round(backend.generated_tokens_per_sec, 1),
        "semantic_healthy": backend.semantic_healthy,
        "semantic_checked_at": backend.semantic_checked_at,
        "semantic_last_success": backend.semantic_last_success,
        "semantic_last_failure": backend.semantic_last_failure,
        "semantic_latency_ms": round(backend.semantic_latency_ms, 2),
        "semantic_consecutive_failures": backend.semantic_consecutive_failures,
        "semantic_error": backend.semantic_error,
        "semantic_check_count": backend.semantic_check_count,
        "semantic_tool_checked_at": backend.semantic_tool_checked_at,
        "semantic_tool_healthy": backend.semantic_tool_healthy,
        "semantic_in_progress": backend.semantic_in_progress,
    }


def router_snapshot(state: RouterState, config: RouterConfig) -> dict[str, Any]:
    connected = [b for b in state.backends if b.config.enabled]
    ready = [b for b in state.backends if (b.last_watchdog_status or {}).get("ready", False)]
    token_backends = [b for b in state.backends if b.prompt_tokens_per_sec > 0]
    overall_prompt = round(sum(b.prompt_tokens_per_sec for b in token_backends) / max(len(token_backends), 1), 1)
    overall_gen = round(sum(b.generated_tokens_per_sec for b in token_backends) / max(len(token_backends), 1), 1)
    return {
        "ok": True,
        "model": config.public_model_name,
        "started_at": state.started_at,
        "uptime_seconds": round(time.time() - state.started_at, 1),
        "node_count": len(state.backends),
        "connected_nodes": len(connected),
        "ready_nodes": len(ready),
        "fallback_enabled": config.fallback.enabled,
        "fallback_name": config.fallback.name,
        "fallback_model": config.fallback.model,
        "fallback_usage_count": state.fallback_usage_count,
        "total_requests": state.total_requests,
        "completed_requests": state.completed_requests,
        "failed_requests": state.failed_requests,
        "cancelled_requests": state.cancelled_requests,
        "retried_requests": state.retried_requests,
        "queue_rejected_requests": state.queue_rejected_requests,
        "config_reload_count": state.config_reload_count,
        "last_config_reload_at": state.last_config_reload_at,
        "last_config_reload_error": state.last_config_reload_error,
        "max_queue_size": config.max_queue_size,
        "max_queue_seconds": config.max_queue_seconds,
        "retry_after_seconds": config.retry_after_seconds,
        "queued_requests": len(state.queued_requests),
        "oldest_queue_seconds": round(max((time.time() - float(item["started_at"]) for item in state.queued_requests.values()), default=0.0), 2),
        "stream_requests": state.stream_requests,
        "dynamic_thinking_enabled": config.dynamic_thinking.enabled,
        "semantic_health_enabled": config.semantic_health.enabled,
        "semantic_health_enforced": config.semantic_health.enforce,
        "semantic_health_interval_seconds": config.semantic_health.interval_seconds,
        "thinking_classifier_requests": state.thinking_classifier_requests,
        "thinking_enabled_requests": state.thinking_enabled_requests,
        "thinking_disabled_requests": state.thinking_disabled_requests,
        "backend_selection_counts": state.backend_selection_counts,
        "backends": [backend_snapshot(b) for b in state.backends],
        "recent_routes": list(state.recent_routes),
        "cache_affinity": state.affinity.stats() if state.config.cache_affinity_enabled else None,
        "classifier_cache": state.classifier_cache.stats() if config.dynamic_thinking.enabled else None,
        "overall_prompt_tokens_per_sec": overall_prompt,
        "overall_generated_tokens_per_sec": overall_gen,
    }


async def refresh_backend_statuses(client: httpx.AsyncClient, state: RouterState, config: RouterConfig) -> None:
    for backend in state.backends:
        await fetch_watchdog(client, backend, config.health_cache_seconds)
        prompt, predicted = await fetch_llama_metrics(client, backend.config.api_base)
        backend.prompt_tokens_per_sec = prompt
        backend.generated_tokens_per_sec = predicted
        async with state._lock:
            if reap_stale_active_leases(backend, config):
                state.telemetry_dirty = True
                state.status_revision += 1


def semantic_canary_payload(backend: BackendState, config: SemanticHealthConfig, *, tool_check: bool) -> dict[str, Any]:
    if tool_check:
        return {
            "model": backend.config.backend_model,
            "stream": False,
            "temperature": 0,
            "max_tokens": min(64, max(16, config.max_tokens)),
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": "Call nyx_health_ping exactly once with status set to ok.\n\n/no_think",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "nyx_health_ping",
                        "description": "Return semantic health status.",
                        "parameters": {
                            "type": "object",
                            "properties": {"status": {"type": "string", "enum": ["ok"]}},
                            "required": ["status"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "nyx_health_ping"}},
        }
    return {
        "model": backend.config.backend_model,
        "stream": False,
        "temperature": 0,
        "max_tokens": config.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {
                "role": "user",
                "content": f"Reply with exactly {config.expected_response}.\n\n/no_think",
            }
        ],
    }


def validate_semantic_response(payload: dict[str, Any], config: SemanticHealthConfig, *, tool_check: bool) -> None:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise ValueError("missing_choices")
    message = choices[0].get("message") or {}
    if tool_check:
        calls = message.get("tool_calls") or []
        names = [str((call.get("function") or {}).get("name") or "") for call in calls if isinstance(call, dict)]
        if "nyx_health_ping" not in names:
            raise ValueError("missing_health_tool_call")
        return
    content = content_to_text(message.get("content"))
    if config.expected_response.casefold() not in content.casefold():
        raise ValueError("unexpected_canary_response")


async def reserve_semantic_backend(state: RouterState, backend: BackendState, config: SemanticHealthConfig) -> bool:
    async with state._lock:
        now = time.time()
        if state.queued_requests or now - state.last_request_activity < config.idle_seconds:
            return False
        recovered = sum(reap_stale_active_leases(candidate, state.config, now=now) for candidate in state.backends)
        if recovered:
            state.telemetry_dirty = True
            state.status_revision += 1
        if any(candidate.active_requests > 0 for candidate in state.backends):
            return False
        if backend.semantic_in_progress or not backend.config.enabled or backend.draining:
            return False
        if not (backend.last_watchdog_status or {}).get("ready", False):
            return False
        backend.semantic_in_progress = True
        backend.semantic_lease_id = acquire_backend_lease(
            backend,
            backend.semantic_check_count + 1,
            purpose="semantic",
        )
        state.status_revision += 1
        return True


async def run_semantic_backend_check(
    client: httpx.AsyncClient,
    state: RouterState,
    backend: BackendState,
    config: SemanticHealthConfig,
    *,
    force_tool_check: bool | None = None,
) -> bool:
    tool_check = (
        force_tool_check
        if force_tool_check is not None
        else config.tool_check_every > 0 and (backend.semantic_check_count + 1) % config.tool_check_every == 0
    )
    started = time.perf_counter()
    success = False
    error = ""
    cancelled = False
    try:
        url = backend.config.api_base.rstrip("/") + "/chat/completions"
        response = await client.post(
            url,
            json=semantic_canary_payload(backend, config, tool_check=tool_check),
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        validate_semantic_response(response.json(), config, tool_check=tool_check)
        success = True
    except asyncio.CancelledError:
        cancelled = True
        error = "CancelledError:semantic check cancelled"
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"

    checked_at = time.time()
    latency_ms = (time.perf_counter() - started) * 1000
    async with state._lock:
        release_backend_lease(backend, backend.semantic_lease_id)
        backend.semantic_lease_id = ""
        backend.semantic_in_progress = False
        if not cancelled:
            backend.semantic_checked_at = checked_at
            backend.semantic_latency_ms = latency_ms
            backend.semantic_check_count += 1
            if tool_check:
                backend.semantic_tool_checked_at = checked_at
                backend.semantic_tool_healthy = success
            if success:
                backend.semantic_healthy = True
                backend.semantic_last_success = checked_at
                backend.semantic_consecutive_failures = 0
                backend.semantic_error = ""
            else:
                backend.semantic_healthy = False
                backend.semantic_last_failure = checked_at
                backend.semantic_consecutive_failures += 1
                backend.semantic_error = error[:240]
        state.status_revision += 1
    if cancelled:
        structured_log("semantic_health_check_cancelled", backend=backend.config.name)
        raise asyncio.CancelledError
    structured_log(
        "semantic_health_check",
        backend=backend.config.name,
        success=success,
        tool_check=tool_check,
        latency_ms=round(latency_ms, 2),
        error=error,
    )
    return success


async def semantic_health_loop(app: FastAPI, config: RouterConfig) -> None:
    await asyncio.sleep(max(0.0, config.semantic_health.initial_delay_seconds))
    while True:
        semantic = config.semantic_health
        if not semantic.enabled:
            await asyncio.sleep(5)
            continue
        state: RouterState = app.state.router_state
        client: httpx.AsyncClient = app.state.http
        now = time.time()
        due = [
            backend
            for backend in state.backends
            if now - backend.semantic_checked_at >= semantic.interval_seconds
        ]
        for backend in due:
            if await reserve_semantic_backend(state, backend, semantic):
                await run_semantic_backend_check(client, state, backend, semantic)
        await asyncio.sleep(min(5.0, max(1.0, semantic.interval_seconds / 20)))


def dashboard_html(title: str) -> str:
    safe_title = escape(title)
    ui_version = escape(ROUTER_UI_VERSION)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0A0A0A;
      --panel: #0F0F0F;
      --panel-2: #1A1A1A;
      --panel-3: #1A1A1A;
      --text: #FAFAFA;
      --muted: #737373;
      --line: #262626;
      --good: #73bf69;
      --warn: #f2cc0c;
      --bad: #f2495c;
      --accent: #FF3D00;
      --accent-foreground: #0A0A0A;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ overflow-x: clip; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Inter Tight", Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      letter-spacing: -0.01em;
      background-color: var(--bg);
      background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.82' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E"),
        linear-gradient(180deg, #0A0A0A 0%, #111111 100%);
    }}
    button {{ font: inherit; }}
    .app {{ min-height: 100vh; }}
    aside {{ border-left: 1px solid var(--line); background: rgba(10,10,10,0.98); padding: 32px 24px; position: fixed; inset: 0 0 0 auto; width: min(340px, 86vw); transform: translateX(100%); transition: transform 200ms cubic-bezier(0.25,0,0,1); z-index: 30; }}
    body.menu-open aside {{ transform: translateX(0); }}
    .drawer-backdrop {{ position: fixed; inset: 0; background: rgba(10,10,10,0.72); border: 0; opacity: 0; pointer-events: none; transition: opacity 150ms cubic-bezier(0.25,0,0,1); z-index: 20; }}
    body.menu-open .drawer-backdrop {{ opacity: 1; pointer-events: auto; }}
    main {{ padding: 48px 64px 64px; max-width: 1480px; width: 100%; margin: 0 auto; }}
    .brand {{ display: flex; gap: 12px; align-items: center; margin-bottom: 24px; }}
    .logo {{ width: 44px; height: 44px; background: var(--accent); display: grid; place-items: center; color: var(--accent-foreground); font-weight: 900; letter-spacing: -0.06em; }}
    h1 {{ margin: 0; font-size: 1.25rem; letter-spacing: -0.04em; line-height: 1; }}
    .sub {{ color: var(--muted); font-size: 0.875rem; line-height: 1.6; margin-top: 6px; }}
    .navcard {{ border-top: 1px solid var(--line); background: transparent; padding: 18px 0; margin-bottom: 0; }}
    .navtitle {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 9px; font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace; }}
    .pill {{ border: 1px solid var(--line); padding: 11px 12px; background: transparent; color: var(--muted); font-size: 0.875rem; min-height: 44px; display: inline-flex; align-items: center; }}
    .menu-button {{ background: transparent; color: var(--text); border: 0; border-bottom: 2px solid var(--accent); padding: 10px 0; min-height: 44px; text-transform: uppercase; letter-spacing: 0.12em; cursor: pointer; }}
    .menu-button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; margin-bottom: 18px; }}
    .titleblock h2 {{ margin: 0; font-size: clamp(3rem, 7vw, 6.5rem); letter-spacing: -0.06em; line-height: 0.9; max-width: 900px; text-transform: uppercase; }}
    .titleblock p {{ margin: 22px 0 0; color: var(--muted); max-width: 680px; font-size: 1.125rem; line-height: 1.6; }}
    .grid {{ display: grid; gap: 12px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 56px 0 12px; }}
    .two {{ grid-template-columns: minmax(0, 1fr) minmax(360px, 0.7fr); }}
    .card {{ border: 1px solid var(--line); padding: 24px; background: transparent; box-shadow: none; min-width: 0; }}
    .cardhead {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 18px; border-bottom: 1px solid var(--line); padding-bottom: 14px; }}
    .card h3 {{ margin: 0; font-size: 1.5rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1.1; }}
    .metric {{ border: 1px solid var(--line); padding: 24px 20px; }}
    .metric .label {{ color: var(--muted); font-size: 0.75rem; letter-spacing: 0.05em; font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace; }}
    .metric .value {{ font-size: clamp(2.5rem, 5vw, 5.8rem); font-weight: 800; letter-spacing: -0.06em; line-height: 0.9; margin-top: 20px; }}
    .metric .duration-value {{ display: flex; align-items: baseline; flex-wrap: nowrap; white-space: nowrap; }}
    .metric-unit {{ color: var(--muted); font-size: 0.34em; font-weight: 700; letter-spacing: 0.02em; line-height: 1; margin-left: 0.22em; }}
    .metric .hint {{ color: var(--muted); font-size: 0.875rem; margin-top: 14px; min-height: 18px; line-height: 1.5; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 16px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    th {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700; font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace; }}
    td {{ font-size: 0.88rem; }}
    code {{ color: var(--accent); }}
    .status {{ display: inline-flex; align-items: center; gap: 7px; padding: 5px 0; font-weight: 700; font-size: 0.75rem; border-bottom: 2px solid currentColor; background: transparent; text-transform: uppercase; letter-spacing: 0.1em; }}
    .status::before {{ content: ""; width: 7px; height: 7px; background: currentColor; }}
    .ready {{ color: var(--good); }}
    .busy {{ color: var(--warn); }}
    .down {{ color: var(--bad); }}
    .muted {{ color: var(--muted); }}
    .bar {{ height: 10px; background: var(--panel-2); border: 1px solid var(--line); overflow: hidden; margin-top: 9px; }}
    .fill {{ height: 100%; background: linear-gradient(90deg, var(--good), var(--warn) 68%, var(--accent)); }}
    .node-gauges {{ display: grid; grid-template-columns: 1.5fr 1fr 0.8fr; gap: 14px; min-width: 320px; }}
    .gauge-label {{ color: var(--muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; font-family: "JetBrains Mono", "Fira Code", ui-monospace, monospace; }}
    .gauge-value {{ margin-top: 6px; font-weight: 700; }}
    .route {{ display: grid; grid-template-columns: 86px 1fr auto; gap: 10px; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--line); }}
    .route:last-child {{ border-bottom: 0; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    .empty {{ color: var(--muted); padding: 32px; text-align: center; border: 1px solid var(--line); }}
    .kv {{ display: grid; grid-template-columns: 1fr auto; gap: 8px; padding: 7px 0; border-bottom: 1px solid var(--line); font-size: 0.84rem; }}
    .kv:last-child {{ border-bottom: 0; }}
    footer {{ color: var(--muted); margin-top: 14px; font-size: 0.82rem; }}
    .three {{ grid-template-columns: 1fr 1fr 1fr; }}
    .aff-bar {{ height: 8px; background: var(--panel-2); border: 1px solid var(--line); overflow: hidden; margin-top: 4px; }}
    .aff-fill {{ height: 100%; background: var(--good); }}
    .thinking-bar {{ display: flex; height: 16px; border: 1px solid var(--line); overflow: hidden; }}
    .thinking-seg {{ height: 100%; transition: width 0.3s ease; }}
    .window-controls {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 24px 0 0; }}
    .window-button {{ background: transparent; color: var(--muted); border: 0; border-bottom: 1px solid var(--line); padding: 8px 0; cursor: pointer; font-family: ui-monospace, monospace; }}
    .window-button.active {{ color: var(--accent); border-bottom: 2px solid var(--accent); }}
    @media (max-width: 1200px) {{
      .metrics {{ grid-template-columns: repeat(4, 1fr); }}
    }}
    @media (max-width: 980px) {{
      aside {{ width: min(330px, 90vw); }}
      main {{ padding: 24px; }}
      .topbar {{ flex-direction: column; }}
      .titleblock h2 {{ font-size: clamp(2.75rem, 14vw, 4.7rem); }}
      .metrics {{ grid-template-columns: repeat(3, 1fr); }}
      .two {{ grid-template-columns: 1fr; }}
      .three {{ grid-template-columns: repeat(3, 1fr); }}
      table {{ display: block; overflow-x: auto; }}
    }}
    @media (max-width: 600px) {{
      .metrics {{ grid-template-columns: repeat(2, 1fr); }}
      .two, .three {{ grid-template-columns: 1fr; }}
      .card {{ padding: 20px; }}
      .cardhead {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body>
<div class="app">
  <button class="drawer-backdrop" id="drawerBackdrop" aria-label="Close NYX router menu"></button>
  <aside id="routerDrawer" role="dialog" aria-modal="true" aria-label="NYX router menu" aria-hidden="true" inert tabindex="-1">
    <div class="brand">
      <div class="logo">NYX</div>
      <div><h1>NYX_Router</h1><div class="sub">llama.cpp router telemetry</div></div>
    </div>
    <div class="navcard">
      <div class="navtitle">Scrape Target</div>
      <div class="pill mono">GET /api/status</div>
    </div>
    <div class="navcard">
      <div class="navtitle">OpenAI Compatible</div>
      <div class="pill mono">POST /v1/chat/completions</div>
    </div>
    <div class="navcard">
      <div class="navtitle">NYX Router</div>
      <div class="sub">Version {ui_version}</div>
    </div>
    <div class="pill" id="updated">Waiting for scrape...</div>
  </aside>
  <main>
    <div class="topbar">
      <div class="titleblock">
        <h2>NYX<br>Pool</h2>
        <p id="clusterSummary">Loading cluster state.</p>
      </div>
      <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
        <div class="pill mono" id="modelName">model=-</div>
        <button class="menu-button" id="menuButton" type="button" aria-expanded="false" aria-controls="routerDrawer">Menu</button>
      </div>
    </div>

    <div class="window-controls" aria-label="Telemetry time window">
      <button class="window-button" data-window="900000">15m</button>
      <button class="window-button" data-window="3600000">1h</button>
      <button class="window-button" data-window="21600000">6h</button>
      <button class="window-button active" data-window="86400000">24h</button>
    </div>

    <section class="grid metrics">
      <div class="card metric"><div class="label">Nodes Connected</div><div class="value" id="nodes">-</div><div class="hint" id="ready">- ready</div></div>
      <div class="card metric"><div class="label">Active Requests</div><div class="value" id="active">-</div><div class="hint">live slot pressure</div></div>
      <div class="card metric"><div class="label">Queued Requests</div><div class="value" id="queued">-</div><div class="hint" id="queueHint">oldest wait</div></div>
      <div class="card metric"><div class="label">Total Requests</div><div class="value" id="total">-</div><div class="hint" id="success">- completed</div></div>
      <div class="card metric"><div class="label">Memory Pressure</div><div class="value" id="gpuUsed">-</div><div class="hint" id="gpuHint">across ready nodes</div></div>
      <div class="card metric"><div class="label">Prompt Prefill</div><div class="value" id="promptSpeed">-</div><div class="hint" id="promptSpeedHint">avg prompt tok/s</div></div>
      <div class="card metric"><div class="label">Token Gen</div><div class="value" id="genSpeed">-</div><div class="hint" id="genSpeedHint">avg gen tok/s</div></div>
      <div class="card metric"><div class="label">Cache Hit Rate</div><div class="value" id="cacheHitRate">-</div><div class="hint" id="cacheHitRateHint">KV affinity hits</div></div>
      <div class="card metric"><div class="label">Average Latency</div><div class="value duration-value" id="avgLatency">-</div><div class="hint" id="avgLatencyHint">mean across recent routes</div></div>
      <div class="card metric"><div class="label">Avg Time to First Token</div><div class="value duration-value" id="avgTtft">-</div><div class="hint" id="avgTtftHint">streaming requests</div></div>
      <div class="card metric"><div class="label">Measured Generation</div><div class="value" id="measuredTps">-</div><div class="hint" id="measuredTpsHint">response token throughput</div></div>
      <div class="card metric"><div class="label">Error Rate</div><div class="value" id="errorRate">-</div><div class="hint" id="errorRateHint">failed / total</div></div>
    </section>

    <section class="grid">
      <div class="card">
      <div class="cardhead"><h3>Node Targets</h3><span class="muted mono">watchdog + llama health</span></div>
      <table>
        <thead><tr><th>Status</th><th>Target</th><th>Model</th><th>Context</th><th>Slots</th><th>Requests</th><th>Memory</th><th>Prompt tok/s</th><th>Gen tok/s</th><th>Last Route</th></tr></thead>
        <tbody id="backendRows"></tbody>
      </table>
    </div>
  </section>

  <section class="grid three" style="margin-top:12px">
    <div class="card"><div class="cardhead"><h3>Latency Breakdown</h3><span class="muted mono">queue → classifier → backend → total</span></div><div id="latencyBreakdown"></div></div>
    <div class="card"><div class="cardhead"><h3>Routing Reasons</h3><span class="muted mono">why each node was selected</span></div><div id="routingReasons"></div></div>
    <div class="card"><div class="cardhead"><h3>Classifier Performance</h3><span class="muted mono">calls, cache, latency</span></div><div id="classifierPerformance"></div></div>
  </section>

  <section class="card" style="margin-top:12px">
    <div class="cardhead"><h3>Per-Node Latency Trend</h3><span class="muted mono">last 20 requests per backend</span></div>
    <div id="latencyTrend" class="empty">No latency data yet.</div>
  </section>

  <section class="grid two" style="margin-top:12px">
    <div class="card">
      <div class="cardhead"><h3>Routing Mix</h3><span class="muted mono">selected_backend_total</span></div>
      <div id="routingMix" class="empty">No routed requests yet.</div>
    </div>
    <div class="card">
      <div class="cardhead"><h3>Metadata</h3></div>
      <div id="metadata"></div>
      <div id="cacheAffinity"></div>
    </div>
  </section>

  <section class="grid two" style="margin-top:12px">
    <div class="card">
      <div class="cardhead"><h3>Latency Histogram</h3><span class="muted mono">p50 / p95 / p99 per backend</span></div>
      <div id="latencyHistogram" class="empty">No latency data yet.</div>
    </div>
    <div class="card">
      <div class="cardhead"><h3>Thinking Mode</h3><span class="muted mono">dynamic_thinking breakdown</span></div>
      <div id="thinkingMode" class="empty">Dynamic thinking is disabled.</div>
    </div>
  </section>

  <section class="card" style="margin-top:16px">
    <div class="cardhead"><h3>Recent Routing Log</h3><span class="muted mono">Recent Requests</span></div>
    <div id="routes" class="empty">No recent routes yet.</div>
  </section>
  <footer>Polls <code>/api/status</code> every two seconds.</footer>
  </main>
</div>
<script>
const fmt = new Intl.NumberFormat();
const compact = new Intl.NumberFormat(undefined, {{maximumFractionDigits: 1}});
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({{
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}}[char]));
const two = (n) => String(Math.max(0, Math.floor(n))).padStart(2, "0");
const uptime = (seconds) => {{
  const totalMinutes = Math.max(0, Math.floor(Number(seconds || 0) / 60));
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;
  return `${{days}}d:${{two(hours)}}h:${{two(minutes)}}m`;
}};
const gb = (mb) => {{
  const n = Number(mb);
  if (!Number.isFinite(n)) return "n/a";
  return `${{(n / 1024).toFixed(2)}} GB`;
}};
const duration = (milliseconds) => {{
  const value = Number(milliseconds);
  if (!Number.isFinite(value) || value <= 0) return "n/a";
  if (value < 1000) return `${{Math.round(value)}} ms`;
  if (value < 10000) return `${{(value / 1000).toFixed(2)}} s`;
  return `${{(value / 1000).toFixed(1)}} s`;
}};
const metricDuration = (milliseconds) => {{
  const value = Number(milliseconds);
  if (!Number.isFinite(value) || value <= 0) return "-";
  let number;
  let unit;
  if (value < 1000) {{
    number = Math.round(value);
    unit = "ms";
  }} else if (value < 10000) {{
    number = (value / 1000).toFixed(2);
    unit = "s";
  }} else {{
    number = (value / 1000).toFixed(1);
    unit = "s";
  }}
  return `<span>${{number}}</span><span class="metric-unit">${{unit}}</span>`;
}};
const age = (ts) => {{
  if (!ts) return "never";
  const s = Math.max(0, Math.round(Date.now()/1000 - ts));
  if (s < 60) return `${{s}}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${{m}}m ago`;
  return `${{Math.round(m / 60)}}h ago`;
}};
const statusClass = (b) => {{
  if (!b.enabled || b.last_reject_reason || !b.ready) return "down";
  if (b.active_requests >= b.max_parallel_requests) return "busy";
  return "ready";
}};
const displayStatusReason = (reason) => {{
  if (["llama_unhealthy", "llama_health_failed", "watchdog_unreachable"].includes(reason)) return "OFFLINE";
  return reason;
}};
const statusText = (b) => {{
  if (!b.enabled) return "disabled";
  if (b.last_reject_reason) return displayStatusReason(b.last_reject_reason);
  if (!b.ready) return displayStatusReason(b.ready_reason) || "not ready";
  if (b.active_requests >= b.max_parallel_requests) return "busy";
  return "ready";
}};
function pct(value) {{
  const n = Number(value || 0);
  return Math.max(0, Math.min(100, n));
}}
function pctl(sorted, p) {{
  if (!sorted.length) return 0;
  const i = Math.ceil(sorted.length * p / 100) - 1;
  return sorted[Math.max(0, Math.min(i, sorted.length - 1))];
}}
const trendColors = ["#FF3D00", "#4FC3F7", "#81C784", "#FFB74D", "#BA68C8", "#F06292", "#AED581", "#FF8A65"];
let selectedWindowMs = 86400000;
let latestData = null;
function miniChart(latencies, color, w, h) {{
  if (latencies.length < 2) return "";
  const max = Math.max(...latencies, 1);
  const step = w / (latencies.length - 1);
  const pts = latencies.map((v, i) => `${{(i * step).toFixed(1)}},${{(h - (v / max) * (h - 6) - 3).toFixed(1)}}`);
  return `<svg viewBox="0 0 ${{w}} ${{h}}" width="100%" height="${{h}}" preserveAspectRatio="none" style="display:block"><polyline points="${{pts.join(' ')}}" fill="none" stroke="${{color}}" stroke-width="1.5"/></svg>`;
}}
function render(data) {{
  latestData = data;
  const cutoff = Date.now() / 1000 - selectedWindowMs / 1000;
  const routes = (data.recent_routes || []).filter(route => Number(route.created_at || 0) >= cutoff);
  const completed = routes.filter(route => route.status === "success").length;
  const failed = routes.filter(route => route.status !== "success").length;
  const total = routes.length;
  const active = data.backends.reduce((sum, b) => sum + Number(b.active_requests || 0), 0);
  const maxSlots = data.backends.reduce((sum, b) => sum + Number(b.max_parallel_requests || 0), 0);
  const clusterLoad = maxSlots ? Math.round(active / maxSlots * 100) : 0;
  const gpuSamples = data.backends.map(b => Number(b.gpu && b.gpu.memory_used_percent)).filter(n => Number.isFinite(n));
  const maxGpu = gpuSamples.length ? Math.max(...gpuSamples) : 0;
  document.getElementById("nodes").textContent = `${{data.connected_nodes}}/${{data.node_count}}`;
  document.getElementById("ready").textContent = `${{data.ready_nodes}} ready`;
  document.getElementById("active").textContent = active;
  document.getElementById("queued").textContent = fmt.format(data.queued_requests || 0);
  document.getElementById("queueHint").textContent = data.queued_requests ? `oldest · ${{duration((data.oldest_queue_seconds || 0) * 1000)}}` : "no waiting requests";
  document.getElementById("total").textContent = fmt.format(total);
  document.getElementById("success").textContent = `${{fmt.format(completed)}} completed, ${{fmt.format(failed)}} failed`;
  document.getElementById("gpuUsed").textContent = gpuSamples.length ? `${{compact.format(maxGpu)}}%` : "n/a";
  document.getElementById("gpuHint").textContent = gpuSamples.length ? "max node VRAM used" : "no nvidia-smi data";
  const genSpeed = data.overall_generated_tokens_per_sec || 0;
  const promptSpeed = data.overall_prompt_tokens_per_sec || 0;
  document.getElementById("promptSpeed").textContent = promptSpeed > 0 ? promptSpeed.toFixed(0) : "n/a";
  document.getElementById("promptSpeedHint").textContent = promptSpeed > 0 ? `across ${{data.ready_nodes}} nodes` : "unavailable";
  document.getElementById("genSpeed").textContent = genSpeed > 0 ? genSpeed.toFixed(0) : "n/a";
  document.getElementById("genSpeedHint").textContent = genSpeed > 0 ? `across ${{data.ready_nodes}} nodes` : "unavailable";
  const ca = data.cache_affinity || {{}};
  const caTotal = (ca.hits || 0) + (ca.misses || 0);
  const caRate = caTotal > 0 ? ((ca.hits || 0) / caTotal * 100) : 0;
  document.getElementById("cacheHitRate").textContent = caTotal > 0 ? `${{caRate.toFixed(0)}}%` : "n/a";
  document.getElementById("cacheHitRateHint").textContent = caTotal > 0 ? `${{fmt.format(ca.hits || 0)}} hits / ${{fmt.format(caTotal)}} total` : "no affinity data";
  const avgLat = routes.length > 0 ? routes.reduce((s, r) => s + (r.total_ms || r.latency_ms || 0), 0) / routes.length : 0;
  document.getElementById("avgLatency").innerHTML = metricDuration(avgLat);
  document.getElementById("avgLatencyHint").textContent = routes.length ? `mean · ${{routes.length}} recent routes` : "no recent routes";
  const ttftRoutes = routes.filter(r => Number(r.ttft_ms || 0) > 0);
  const avgTtft = ttftRoutes.length ? ttftRoutes.reduce((s, r) => s + Number(r.ttft_ms), 0) / ttftRoutes.length : 0;
  document.getElementById("avgTtft").innerHTML = metricDuration(avgTtft);
  document.getElementById("avgTtftHint").textContent = ttftRoutes.length ? `mean · ${{ttftRoutes.length}} streams` : "no streaming samples";
  const measuredRoutes = routes.filter(r => Number(r.tokens_per_second || 0) > 0);
  const measuredTps = measuredRoutes.length ? measuredRoutes.reduce((s, r) => s + Number(r.tokens_per_second), 0) / measuredRoutes.length : 0;
  document.getElementById("measuredTps").textContent = measuredTps > 0 ? measuredTps.toFixed(1) : "n/a";
  document.getElementById("measuredTpsHint").textContent = measuredRoutes.length ? `mean · ${{measuredRoutes.length}} responses` : "no usage samples";
  const errPct = total > 0 ? (failed / total * 100) : 0;
  document.getElementById("errorRate").textContent = total > 0 ? `${{errPct.toFixed(1)}}%` : "n/a";
  document.getElementById("errorRateHint").textContent = `${{fmt.format(failed)}} / ${{fmt.format(total)}}`;
  document.getElementById("updated").textContent = `Updated ${{new Date().toLocaleTimeString()}} · uptime ${{uptime(data.uptime_seconds)}}`;
  document.getElementById("modelName").textContent = `model=${{data.model}}`;
  document.getElementById("clusterSummary").textContent = `${{data.ready_nodes}} Node${{data.ready_nodes === 1 ? "" : "s"}} Online. Cluster Load at ${{clusterLoad}}%.`;
  document.getElementById("metadata").innerHTML = [
    ["Router Uptime", uptime(data.uptime_seconds)],
    ["Total Stream Requests", fmt.format(data.stream_requests || 0)],
    ["Router Failures", fmt.format(data.failed_requests || 0)],
    ["Retries / Failovers", fmt.format(data.retried_requests || 0)],
    ["Client Cancellations", fmt.format(data.cancelled_requests || 0)],
    ["Queue Rejections", fmt.format(data.queue_rejected_requests || 0)],
    ["Config Reloads", fmt.format(data.config_reload_count || 0)],
    ["Semantic Checks", data.semantic_health_enabled ? `every ${{uptime(data.semantic_health_interval_seconds)}}` : "disabled"],
    ["Fallback Usage Count", fmt.format(data.fallback_usage_count || 0)],
    ["Avg Prompt tok/s", promptSpeed > 0 ? promptSpeed.toFixed(1) : "n/a"],
    ["Avg Gen tok/s", genSpeed > 0 ? genSpeed.toFixed(1) : "n/a"]
  ].map(([k,v]) => `<div class="kv"><span class="mono muted">${{k}}</span><strong>${{v}}</strong></div>`).join("");

  document.getElementById("backendRows").innerHTML = data.backends.map((b) => {{
    const gpuUsed = pct(b.gpu && b.gpu.memory_used_percent);
    const gpuUtil = pct(b.gpu && b.gpu.utilization_percent);
    const gpuTemp = b.gpu && b.gpu.temperature_c != null ? `${{b.gpu.temperature_c}}°C` : "n/a";
    const gpuMem = b.gpu && b.gpu.memory_used_mb != null ? `${{gb(b.gpu.memory_used_mb)}} / ${{gb(b.gpu.memory_total_mb)}}` : "n/a";
    const promptTps = b.prompt_tokens_per_sec > 0 ? b.prompt_tokens_per_sec.toFixed(1) : "n/a";
    const genTps = b.generated_tokens_per_sec > 0 ? b.generated_tokens_per_sec.toFixed(1) : "n/a";
    const semanticText = b.semantic_healthy == null
      ? "semantic pending"
      : (b.semantic_healthy
        ? `semantic ok · ${{age(b.semantic_checked_at)}}`
        : `semantic failed · ${{esc(b.semantic_error || "unknown")}}`);
    const nodeRoutes = routes.filter(route => route.backend_name === b.name);
    const nodeSuccess = nodeRoutes.filter(route => route.status === "success").length;
    const nodeFailed = nodeRoutes.length - nodeSuccess;
    const lastRoute = routes.find((route) => route.backend_name === b.name);
    const lastRouteText = b.active_requests > 0
      ? "active now"
      : (lastRoute ? age(lastRoute.created_at) : "never");
    const lastRouteDetail = b.active_requests > 0
      ? `${{b.active_requests}} in flight · oldest ${{duration(Number(b.active_request_age_seconds || 0) * 1000)}}`
      : (lastRoute ? `${{lastRoute.status}} · #${{lastRoute.request_id}}` : "no route history");
    const staleResetText = Number(b.stale_active_resets || 0) > 0 ? ` · recovered ${{fmt.format(b.stale_active_resets)}} stale` : "";
    return `<tr>
      <td><span class="status ${{statusClass(b)}}">${{esc(statusText(b))}}</span></td>
      <td><strong>${{esc(b.name)}}</strong><div class="muted mono">${{esc(b.watchdog_base || "watchdog=none")}}</div><div class="muted mono">${{esc(b.api_base)}}</div><div class="muted mono">${{semanticText}}</div></td>
      <td>${{esc(b.backend_model)}}</td>
      <td>${{fmt.format(b.max_context_tokens || 0)}}</td>
      <td>${{b.active_requests}}/${{b.max_parallel_requests}}<div class="muted">avg ${{b.latency_ms_average || 0}} ms</div><div class="muted">ETA ${{duration(b.estimated_finish_ms)}} · circuit ${{esc(b.circuit_state)}}${{staleResetText}}</div></td>
      <td><strong>${{fmt.format(nodeRoutes.length)}}</strong><div class="muted">${{nodeSuccess}} ok · ${{nodeFailed}} failed</div></td>
      <td><div class="node-gauges">
        <div><div class="gauge-label">Memory</div><div class="gauge-value">${{gpuMem}}</div><div class="bar"><div class="fill" style="width:${{gpuUsed}}%"></div></div></div>
        <div><div class="gauge-label">Util</div><div class="gauge-value">${{gpuUtil.toFixed(1)}}%</div><div class="bar"><div class="fill" style="width:${{gpuUtil}}%"></div></div></div>
        <div><div class="gauge-label">Temp</div><div class="gauge-value">${{gpuTemp}}</div></div>
      </div></td>
      <td><div class="mono">${{promptTps}}</div></td>
      <td><div class="mono">${{genTps}}</div></td>
      <td>${{lastRouteText}}<div class="muted">${{esc(lastRouteDetail)}}</div></td>
    </tr>`;
  }}).join("");

  const trendGroups = {{}};
  for (const r of data.recent_routes) {{
    if (!trendGroups[r.backend_name]) trendGroups[r.backend_name] = [];
    if (r.latency_ms > 0) trendGroups[r.backend_name].push(r.latency_ms);
  }}
  const trendEntries = Object.entries(trendGroups);
  if (trendEntries.length) {{
    document.getElementById("latencyTrend").innerHTML = trendEntries.map(([name, lats], idx) => {{
      const recent = lats.slice(0, 20).reverse();
      const color = trendColors[idx % trendColors.length];
      const latest = recent.length ? recent[recent.length - 1].toFixed(0) : "0";
      return `<div style="display:flex;align-items:center;gap:16px;padding:10px 0;border-bottom:1px solid var(--line)">
        <div style="min-width:110px"><strong>${{esc(name)}}</strong><div class="muted mono" style="font-size:0.75rem">${{recent.length}} samples</div></div>
        <div style="flex:1">${{miniChart(recent, color, 400, 36)}}</div>
        <div class="mono" style="min-width:70px;text-align:right;font-size:0.85rem">${{latest}} ms</div>
      </div>`;
    }}).join("");
  }} else {{
    document.getElementById("latencyTrend").innerHTML = '<div class="empty">No latency data yet.</div>';
  }}

  const counts = routes.reduce((out, route) => {{ out[route.backend_name] = (out[route.backend_name] || 0) + 1; return out; }}, {{}});
  const totalSelections = Object.values(counts).reduce((a, b) => a + Number(b || 0), 0);
  document.getElementById("routingMix").innerHTML = totalSelections ? Object.entries(counts).map(([name, count]) => {{
    const width = totalSelections ? Number(count) / totalSelections * 100 : 0;
        return `<div style="margin-bottom:14px"><div style="display:flex;justify-content:space-between;gap:12px"><strong>${{esc(name)}}</strong><span>${{fmt.format(count)}} requests</span></div><div class="bar"><div class="fill" style="width:${{width}}%"></div></div></div>`;
  }}).join("") : `<div class="empty">No routed requests yet.</div>`;

  const mean = (field) => {{
    const samples = routes.filter(route => route[field] != null && Number.isFinite(Number(route[field])));
    return samples.length ? samples.reduce((sum, route) => sum + Number(route[field]), 0) / samples.length : 0;
  }};
  document.getElementById("latencyBreakdown").innerHTML = [
    ["Queue", duration(mean("queue_ms"))],
    ["Classifier", duration(mean("classifier_ms"))],
    ["Backend", duration(mean("backend_ms"))],
    ["End to End", duration(mean("total_ms"))]
  ].map(([k,v]) => `<div class="kv"><span class="mono muted">${{k}}</span><strong>${{v}}</strong></div>`).join("");

  const reasons = routes.reduce((out, route) => {{ const key = route.routing_reason || "legacy"; out[key] = (out[key] || 0) + 1; return out; }}, {{}});
  document.getElementById("routingReasons").innerHTML = Object.keys(reasons).length ? Object.entries(reasons).sort((a,b) => b[1]-a[1]).map(([reason,count]) => `<div class="kv"><span class="mono muted">${{esc(reason)}}</span><strong>${{count}}</strong></div>`).join("") : '<div class="empty">No routing data yet.</div>';

  const cacheRoutes = routes.filter(route => route.thinking_reason === "classifier_cache_hit");
  const classifierRoutes = routes.filter(route => String(route.thinking_reason || "").startsWith("classifier") && route.thinking_reason !== "classifier_cache_hit");
  const classifierAvg = classifierRoutes.length ? classifierRoutes.reduce((sum, route) => sum + Number(route.classifier_ms || 0), 0) / classifierRoutes.length : 0;
  document.getElementById("classifierPerformance").innerHTML = [
    ["Model Calls", classifierRoutes.length],
    ["Cache Hits", cacheRoutes.length],
    ["Avoided Calls", routes.length - classifierRoutes.length],
    ["Average Latency", duration(classifierAvg)]
  ].map(([k,v]) => `<div class="kv"><span class="mono muted">${{k}}</span><strong>${{v}}</strong></div>`).join("");

  const aff = data.cache_affinity;
  if (aff) {{
    const totalLookups = (aff.hits || 0) + (aff.misses || 0);
    const hitRate = totalLookups > 0 ? ((aff.hits || 0) / totalLookups * 100) : 0;
    document.getElementById("cacheAffinity").innerHTML = `
      <div style="border-top:1px solid var(--line);margin-top:14px;padding-top:14px">
        <div class="navtitle" style="margin-bottom:10px">Cache Affinity</div>
        <div class="kv"><span class="mono muted">Entries</span><strong>${{fmt.format(aff.entries || 0)}}</strong></div>
        <div class="kv"><span class="mono muted">Hits</span><strong>${{fmt.format(aff.hits || 0)}}</strong></div>
        <div class="kv"><span class="mono muted">Misses</span><strong>${{fmt.format(aff.misses || 0)}}</strong></div>
        <div class="kv"><span class="mono muted">Evictions</span><strong>${{fmt.format(aff.evictions || 0)}}</strong></div>
        <div class="kv"><span class="mono muted">TTL</span><strong>${{aff.ttl_seconds || 0}}s</strong></div>
        <div style="margin-top:10px">
          <div style="display:flex;justify-content:space-between;font-size:0.84rem"><span class="muted">Hit Rate</span><strong>${{hitRate.toFixed(1)}}%</strong></div>
          <div class="aff-bar"><div class="aff-fill" style="width:${{hitRate.toFixed(1)}}%"></div></div>
        </div>
      </div>`;
  }}

  const backendGroups = {{}};
  for (const r of data.recent_routes) {{
    if (r.latency_ms <= 0) continue;
    if (!backendGroups[r.backend_name]) backendGroups[r.backend_name] = [];
    backendGroups[r.backend_name].push(r.latency_ms);
  }}
  const backendNames = Object.keys(backendGroups).filter(n => backendGroups[n].length > 0);
  if (backendNames.length) {{
    const allP99 = backendNames.map(n => {{
      const sorted = [...backendGroups[n]].sort((a, b) => a - b);
      return pctl(sorted, 99);
    }});
    const maxP99 = Math.max(...allP99, 1);
    document.getElementById("latencyHistogram").innerHTML = backendNames.map((name, i) => {{
      const sorted = [...backendGroups[name]].sort((a, b) => a - b);
      const p50 = pctl(sorted, 50);
      const p95 = pctl(sorted, 95);
      const p99 = allP99[i];
      const barWidth = (p99 / maxP99 * 100).toFixed(1);
      return `<div style="margin-bottom:16px">
        <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:4px">
          <strong>${{esc(name)}}</strong>
          <span class="muted" style="font-size:0.8rem">${{sorted.length}} samples</span>
        </div>
        <div style="display:grid;grid-template-columns:72px 72px 72px 1fr;gap:8px;align-items:center;font-size:0.85rem">
          <div><div class="gauge-label">p50</div><div class="mono">${{p50.toFixed(0)}} ms</div></div>
          <div><div class="gauge-label">p95</div><div class="mono">${{p95.toFixed(0)}} ms</div></div>
          <div><div class="gauge-label">p99</div><div class="mono">${{p99.toFixed(0)}} ms</div></div>
          <div class="bar" style="margin-top:0"><div class="fill" style="width:${{barWidth}}%"></div></div>
        </div>
      </div>`;
    }}).join("");
  }} else {{
    document.getElementById("latencyHistogram").innerHTML = '<div class="empty">No latency data yet.</div>';
  }}

  if (data.dynamic_thinking_enabled) {{
    const thinking = routes.filter(route => route.thinking_enabled === true).length;
    const nonThinking = routes.filter(route => route.thinking_enabled === false).length;
    const classifier = routes.filter(route => String(route.thinking_reason || "").startsWith("classifier")).length;
    const totalT = thinking + nonThinking;
    const thinkPct = totalT > 0 ? (thinking / totalT * 100) : 0;
    const noThinkPct = totalT > 0 ? (nonThinking / totalT * 100) : 0;
    document.getElementById("thinkingMode").innerHTML = `
      <div class="thinking-bar">
        <div class="thinking-seg" style="width:${{thinkPct.toFixed(1)}}%;background:var(--accent)"></div>
        <div class="thinking-seg" style="width:${{noThinkPct.toFixed(1)}}%;background:var(--muted);opacity:0.3"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:10px;font-size:0.85rem;flex-wrap:wrap;gap:8px">
        <div><span style="display:inline-block;width:8px;height:8px;background:var(--accent);margin-right:6px;vertical-align:middle"></span>Thinking: ${{fmt.format(thinking)}} (${{thinkPct.toFixed(1)}}%)</div>
        <div><span style="display:inline-block;width:8px;height:8px;background:var(--muted);opacity:0.3;margin-right:6px;vertical-align:middle"></span>No-Think: ${{fmt.format(nonThinking)}} (${{noThinkPct.toFixed(1)}}%)</div>
      </div>
      <div style="margin-top:12px;font-size:0.85rem"><span class="muted">Classifier calls:</span> <strong>${{fmt.format(classifier)}}</strong></div>
    `;
  }} else {{
    document.getElementById("thinkingMode").innerHTML = '<div class="muted" style="padding:24px;text-align:center">Dynamic thinking is disabled.</div>';
  }}

  document.getElementById("routes").innerHTML = data.recent_routes.length ? data.recent_routes.map((r) => {{
    const cls = r.status === "success" ? "ready" : "down";
    const perf = [
      r.ttft_ms > 0 ? `TTFT ${{duration(r.ttft_ms)}}` : "",
      r.prompt_ms > 0 ? `prefill ${{duration(r.prompt_ms)}}` : "",
      r.generation_ms > 0 ? `generate ${{duration(r.generation_ms)}}` : "",
      r.tokens_per_second > 0 ? `${{Number(r.tokens_per_second).toFixed(1)}} tok/s` : "",
      r.prompt_tokens > 0 ? `${{fmt.format(r.prompt_tokens)}} ctx tok` : "",
      r.cached_tokens > 0 ? `${{fmt.format(r.cached_tokens)}} cached tok` : "",
      r.cache_hit_estimate === true ? "cache affinity" : ""
    ].filter(Boolean).join(" · ");
    return `<div class="route"><span class="status ${{cls}}">${{esc(r.status)}}</span><div><strong>${{esc(r.backend_name)}}</strong><div class="muted mono">#${{esc(r.request_id)}} · ${{esc(r.routing_reason || "legacy")}} · queue ${{duration(r.queue_ms)}} · classify ${{duration(r.classifier_ms)}} · backend ${{duration(r.backend_ms || r.latency_ms)}}</div>${{perf ? `<div class="muted mono">${{perf}}</div>` : ""}}</div><div class="mono">${{duration(r.total_ms || r.latency_ms)}}</div></div>`;
  }}).join("") : `<div class="empty">No recent routes yet.</div>`;
}}
const menuButton = document.getElementById("menuButton");
const drawerBackdrop = document.getElementById("drawerBackdrop");
function setMenu(open) {{
  document.body.classList.toggle("menu-open", open);
  menuButton.setAttribute("aria-expanded", open ? "true" : "false");
  const drawer = document.getElementById("routerDrawer");
  drawer.toggleAttribute("inert", !open);
  drawer.setAttribute("aria-hidden", open ? "false" : "true");
  if (open) drawer.focus();
  else if (document.activeElement === drawer || drawer.contains(document.activeElement)) menuButton.focus();
}}
menuButton.addEventListener("click", () => setMenu(!document.body.classList.contains("menu-open")));
drawerBackdrop.addEventListener("click", () => setMenu(false));
window.addEventListener("keydown", (event) => {{
  if (event.key === "Escape") setMenu(false);
}});
document.querySelectorAll(".window-button").forEach(button => button.addEventListener("click", () => {{
  selectedWindowMs = Number(button.dataset.window);
  document.querySelectorAll(".window-button").forEach(item => item.classList.toggle("active", item === button));
  if (latestData) render(latestData);
}}));
async function refresh() {{
  try {{
    const response = await fetch("/api/status", {{cache: "no-store"}});
    if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
    render(await response.json());
  }} catch (err) {{
    document.getElementById("updated").textContent = `Status unavailable: ${{err.message}}`;
  }}
}}
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


async def release_backend(
    state: RouterState,
    backend: BackendState,
    *,
    success: bool | None,
    lease_id: str | None = None,
    latency_ms: float = 0.0,
) -> None:
    async with state._lock:
        release_backend_lease(backend, lease_id)
        now = time.time()
        state.last_request_activity = now
        if success is True:
            backend.last_success = now
            backend.consecutive_failures = 0
            backend.circuit_state = "closed"
            backend.circuit_opened_at = 0.0
            backend.circuit_probe_in_flight = False
            if latency_ms > 0:
                if backend.latency_ms_average <= 0:
                    backend.latency_ms_average = latency_ms
                else:
                    backend.latency_ms_average = backend.latency_ms_average * 0.8 + latency_ms * 0.2
        elif success is False:
            backend.last_failure = now
            backend.recent_errors.append(now)
            backend.consecutive_failures += 1
            backend.circuit_probe_in_flight = False
            if backend.consecutive_failures >= state.config.circuit_failure_threshold:
                backend.circuit_state = "open"
                backend.circuit_opened_at = now
                structured_log(
                    "circuit_opened",
                    backend=backend.config.name,
                    consecutive_failures=backend.consecutive_failures,
                    seconds=state.config.circuit_open_seconds,
                )
            # Graduated cooldown: short for transient, full for sustained failures
            recent_count = len([t for t in backend.recent_errors if now - t < 300])
            if recent_count >= state.config.sustained_error_threshold:
                cooldown = state.config.cooldown_seconds
                structured_log(
                    "cooldown_applied",
                    backend=backend.config.name,
                    level="sustained",
                    seconds=cooldown,
                    recent_errors=recent_count,
                )
            else:
                cooldown = state.config.transient_cooldown_seconds
                structured_log(
                    "cooldown_applied",
                    backend=backend.config.name,
                    level="transient",
                    seconds=cooldown,
                    recent_errors=recent_count,
                )
            backend.cooldown_until = now + cooldown
        state.telemetry_dirty = True
        state.status_revision += 1


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def message_text_for_classification(payload: dict[str, Any], max_chars: int) -> str:
    messages = payload.get("messages") or []
    lines: list[str] = []
    if isinstance(messages, list):
        for message in messages[-8:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "unknown")
            text = content_to_text(message.get("content"))
            if text:
                lines.append(f"{role}: {text}")
    text = "\n\n".join(lines)
    if len(text) > max_chars:
        return text[-max_chars:]
    return text


def payload_contains_control_tag(payload: dict[str, Any]) -> bool:
    text = message_text_for_classification(payload, 12000).lower()
    return "/think" in text or "/no_think" in text or "/nothink" in text


_THINK_PATTERNS = [
    (re.compile(r"\b(debug|troubleshoot|diagnos(?:e|is|ing)|investigat(?:e|ion|ing)|root\s+cause)\b"), "diagnosis"),
    (re.compile(r"\b(refactor|optimi[sz]e|benchmark|profil(?:e|ing)|audit|threat\s+model)\b"), "engineering_analysis"),
    (re.compile(r"\b(architect(?:ure)?|design|plan|strateg(?:y|ize)|trade-?offs?|pros?\s+(?:and|&)\s+cons?)\b"), "planning"),
    (re.compile(r"\b(compare|evaluate|analy[sz]e|review|research|reason\s+through|think\s+(?:deeply|carefully))\b"), "analysis"),
    (re.compile(r"\b(implement|build|deploy|migrate|integrate|configure|set\s+up)\b"), "implementation"),
    (re.compile(r"\b(fix|solve|prove|derive|calculate)\b.{0,80}\b(code|bug|error|equation|problem|issue|failure)\b"), "problem_solving"),
    (re.compile(r"\b(write|create|generate)\b.{0,40}\b(function|class|script|program|api|query|test|component|service|system)\b"), "code_generation"),
    (re.compile(r"\b(explain\s+(?:how|why)|walk\s+(?:me\s+)?through|step\s+by\s+step|how\s+(?:should|would|can)\s+(?:i|we))\b"), "multi_step_explanation"),
]

_TECHNICAL_CONTEXT = re.compile(
    r"\b(code|python|javascript|typescript|sql|api|database|server|service|router|network|linux|docker|systemd|"
    r"kubernetes|latency|throughput|cache|queue|thread|async|security|schema|deployment|function|class|exception)\b"
)
_MULTISTEP_CONTEXT = re.compile(r"\b(first|then|after|before|multiple|several|end[- ]to[- ]end|edge\s+cases?|constraints?)\b")

_NO_THINK_PATTERNS = [
    (re.compile(r"^(hi|hey|yo|sup|hiya|howdy)\s*[!.]*$", re.IGNORECASE), "greeting"),
    (re.compile(r"^(thanks?|thank\s*you|thx|ty|tysm|tia)\s*[!.]*$", re.IGNORECASE), "thanks"),
    (re.compile(r"^(ok|okay|k|kk|sure|yep|yeah|yup|nah|nope)\s*[!.]*$", re.IGNORECASE), "ack"),
    (re.compile(r"^(bye|goodbye|see\s*ya|later|cya|ttyl|gn|gm|ge)\s*[!.]*$", re.IGNORECASE), "farewell"),
    (re.compile(r"^(lol|haha|hehe|rofl|lmao|smh|omg|wtf|bruh)\s*[!.]*$", re.IGNORECASE), "reaction"),
    (re.compile(r"^(yes|no|maybe|idk|dunno)\s*[!.]*$", re.IGNORECASE), "boolean"),
    (re.compile(r"^(got\s*it|understood|noted|ack|will\s*do|copy|roger)\s*[!.]*$", re.IGNORECASE), "ack2"),
    (re.compile(r"^(cool|nice|great|awesome|sweet|rad|sick|lit|fire)\s*[!.]*$", re.IGNORECASE), "reaction2"),
]


def keyword_thinking_decision(payload: dict[str, Any], config: DynamicThinkingConfig) -> tuple[bool | None, str]:
    messages = payload.get("messages") or []
    text = ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            text = content_to_text(message.get("content")).strip().lower()
            break
    if re.search(r"(?:^|\s)/(?:no_think|nothink)\b", text):
        return False, "explicit_no_think"
    if re.search(r"(?:^|\s)/think\b", text):
        return True, "explicit_think"
    for keyword in config.force_no_think_keywords:
        if re.search(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", text):
            return False, f"keyword_no_think:{keyword}"
    for pattern, label in _NO_THINK_PATTERNS:
        if pattern.search(text):
            return False, f"pattern_no_think:{label}"
    for keyword in config.force_think_keywords:
        if re.search(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", text):
            return True, f"keyword_think:{keyword}"
    for pattern, label in _THINK_PATTERNS:
        if pattern.search(text):
            return True, f"pattern_think:{label}"
    score = 0
    reasons = []
    if _TECHNICAL_CONTEXT.search(text):
        score += 1
        reasons.append("technical")
    if _MULTISTEP_CONTEXT.search(text):
        score += 1
        reasons.append("multi_step")
    if len(text) >= 500:
        score += 1
        reasons.append("long_prompt")
    if text.count("\n") >= 8 or "```" in text:
        score += 1
        reasons.append("structured_input")
    if score >= 2:
        return True, f"score_think:{'+'.join(reasons)}"
    return None, ""


def reasoning_backend_filter(config: RouterConfig, effort: str | None) -> tuple[set[str] | None, bool]:
    if effort is None or not config.reasoning.routes:
        return None, False
    configured = config.reasoning.routes.get(effort)
    if configured is None:
        return None, True
    names = {configured} if isinstance(configured, str) else {str(name) for name in configured}
    available = {backend.name for backend in config.backends if backend.enabled}
    selected = names & available
    return (selected, False) if selected else (None, True)


def append_text_to_last_user_message(payload: dict[str, Any], suffix: str) -> dict[str, Any]:
    out = dict(payload)
    messages = list(out.get("messages") or [])
    out["messages"] = messages
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        updated = dict(message)
        content = updated.get("content")
        if isinstance(content, str):
            updated["content"] = content.rstrip() + suffix
        elif isinstance(content, list):
            updated["content"] = list(content) + [{"type": "text", "text": suffix.strip()}]
        else:
            updated["content"] = suffix.strip()
        messages[idx] = updated
        return out
    messages.append({"role": "user", "content": suffix.strip()})
    return out


def apply_thinking_control(
    payload: dict[str, Any],
    enabled: bool,
    config: DynamicThinkingConfig | None = None,
    reasoning_budget: int | None = None,
    force_reasoning_budget: bool = False,
) -> dict[str, Any]:
    out = dict(payload)
    kwargs = dict(out.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = enabled
    if enabled:
        kwargs.setdefault("preserve_thinking", True)
    out["chat_template_kwargs"] = kwargs
    if enabled and config and config.min_thinking_max_tokens > 0:
        for token_field in ("max_tokens", "max_completion_tokens"):
            current = out.get(token_field)
            if current is None:
                continue
            try:
                if int(current) < config.min_thinking_max_tokens:
                    out[token_field] = config.min_thinking_max_tokens
            except (TypeError, ValueError):
                continue
    if force_reasoning_budget:
        if enabled and reasoning_budget is not None and reasoning_budget >= 0:
            out["reasoning_budget"] = reasoning_budget
        elif not enabled:
            out.pop("reasoning_budget", None)
    elif enabled and config and config.thinking_reasoning_budget >= 0 and "reasoning_budget" not in out:
        out["reasoning_budget"] = config.thinking_reasoning_budget
    if not payload_contains_control_tag(out):
        out = append_text_to_last_user_message(out, "\n\n/think" if enabled else "\n\n/no_think")
    return out


def build_thinking_classifier_payload(payload: dict[str, Any], backend: BackendConfig, config: DynamicThinkingConfig) -> dict[str, Any]:
    prompt_text = message_text_for_classification(payload, config.max_classifier_prompt_chars)
    classifier_payload: dict[str, Any] = {
        "model": backend.backend_model,
        "stream": False,
        "temperature": 0,
        "max_tokens": 4,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a routing classifier. Decide whether the user's request needs slower, deeper model "
                    "thinking. Reply with exactly THINK or NO_THINK. Use THINK for coding, debugging, planning, "
                    "math, multi-step analysis, research synthesis, architecture, optimization, or ambiguous tasks. "
                    "Use NO_THINK for greetings, simple factual answers, short rewrites, formatting, or casual chat."
                ),
            },
            {"role": "user", "content": prompt_text or "No user text provided."},
        ],
    }
    return apply_thinking_control(classifier_payload, False, config)


async def decide_thinking(
    client: httpx.AsyncClient,
    backend: BackendState,
    payload: dict[str, Any],
    config: DynamicThinkingConfig,
    classifier_cache: "ThinkingClassifierCache | None" = None,
) -> tuple[bool, bool, str]:
    if not config.enabled:
        return config.default_thinking, False, "disabled"
    keyword_decision, keyword_reason = keyword_thinking_decision(payload, config)
    if keyword_decision is not None:
        return keyword_decision, False, keyword_reason
    return config.default_thinking, False, "rules_default"


def rewrite_payload(
    payload: dict[str, Any],
    model: str,
    thinking_enabled: bool | None = None,
    dynamic_thinking: DynamicThinkingConfig | None = None,
    reasoning_effort: str | None = None,
    reasoning: ReasoningConfig | None = None,
) -> dict[str, Any]:
    out = (
        strip_client_reasoning_fields(payload)
        if reasoning and reasoning.strip_client_reasoning_fields
        else dict(payload)
    )
    out["model"] = model
    if out.get("stream"):
        stream_options = dict(out.get("stream_options") or {})
        stream_options["include_usage"] = True
        out["stream_options"] = stream_options
    if reasoning_effort is not None:
        budget = reasoning.effort_budgets.get(reasoning_effort) if reasoning else None
        out = apply_thinking_control(
            out,
            reasoning_effort != "none",
            dynamic_thinking,
            reasoning_budget=budget,
            force_reasoning_budget=True,
        )
    elif thinking_enabled is not None:
        out = apply_thinking_control(out, thinking_enabled, dynamic_thinking)
    return out


def auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def response_metrics(payload: dict[str, Any], elapsed_ms: float) -> dict[str, float | int]:
    usage = payload.get("usage") or {}
    timings = payload.get("timings") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
    completion_tokens = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    prompt_ms = float(timings.get("prompt_ms") or 0)
    generation_ms = float(timings.get("predicted_ms") or 0)
    measured_generation_ms = generation_ms or max(0.0, elapsed_ms - prompt_ms)
    tokens_per_second = completion_tokens / measured_generation_ms * 1000 if measured_generation_ms > 0 else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "prompt_ms": prompt_ms,
        "generation_ms": generation_ms,
        "tokens_per_second": tokens_per_second,
    }


async def post_with_disconnect(
    client: httpx.AsyncClient,
    request: Request,
    url: str,
    payload: dict[str, Any],
    deadline_at: float,
) -> httpx.Response:
    task = asyncio.create_task(client.post(url, json=payload))
    try:
        while not task.done():
            if time.time() >= deadline_at:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                raise RequestDeadlineExceeded
            if await request.is_disconnected():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                raise asyncio.CancelledError("client disconnected")
            await asyncio.sleep(0.1)
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
        raise


class RequestDeadlineExceeded(Exception):
    pass


async def forward_non_stream(
    client: httpx.AsyncClient,
    request: Request,
    state: RouterState,
    backend: BackendState,
    lease_id: str,
    payload: dict[str, Any],
    request_id: int,
    request_started: float,
    queue_ms: float,
    classifier_ms: float,
    routing_reason: str,
    thinking_reason: str,
    thinking_enabled: bool | None = None,
    dynamic_thinking: DynamicThinkingConfig | None = None,
    deadline_at: float = float("inf"),
    reasoning_effort: str | None = None,
    reasoning: ReasoningConfig | None = None,
) -> JSONResponse:
    started = time.perf_counter()
    url = backend.config.api_base.rstrip("/") + "/chat/completions"
    try:
        resp = await post_with_disconnect(
            client,
            request,
            url,
            rewrite_payload(
                payload,
                backend.config.backend_model,
                thinking_enabled,
                dynamic_thinking,
                reasoning_effort,
                reasoning,
            ),
            deadline_at,
        )
        if resp.status_code >= 500:
            structured_log("backend_5xx", backend=backend.config.name, status_code=resp.status_code)
            await release_backend(state, backend, success=False, lease_id=lease_id)
            await state.record_route(
                request_id=request_id,
                backend_name=backend.config.name,
                status="error",
                stream=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                queue_ms=queue_ms, classifier_ms=classifier_ms,
                total_ms=(time.perf_counter() - request_started) * 1000,
                routing_reason=routing_reason, thinking_reason=thinking_reason, thinking_enabled=thinking_enabled,
                error=f"backend_5xx:{resp.status_code}",
            )
            return JSONResponse({"retryable_backend_error": True, "status_code": resp.status_code}, status_code=599)
        latency_ms = (time.perf_counter() - started) * 1000
        response_payload = resp.json()
        metrics = response_metrics(response_payload, latency_ms)
        await release_backend(state, backend, success=True, lease_id=lease_id, latency_ms=latency_ms)
        await state.record_route(
            request_id=request_id,
            backend_name=backend.config.name,
            status="success",
            stream=False,
            latency_ms=latency_ms,
            queue_ms=queue_ms, classifier_ms=classifier_ms,
            total_ms=(time.perf_counter() - request_started) * 1000,
            routing_reason=routing_reason, thinking_reason=thinking_reason, thinking_enabled=thinking_enabled,
            cache_hit_estimate=routing_reason == "affinity",
            **metrics,
        )
        structured_log("non_stream_finished", backend=backend.config.name, latency_ms=round(latency_ms, 2))
        return JSONResponse(response_payload, status_code=resp.status_code)
    except RequestDeadlineExceeded:
        latency_ms = (time.perf_counter() - started) * 1000
        await release_backend(state, backend, success=None, lease_id=lease_id)
        await state.record_route(
            request_id=request_id, backend_name=backend.config.name, status="error", stream=False,
            latency_ms=latency_ms, queue_ms=queue_ms, classifier_ms=classifier_ms,
            total_ms=(time.perf_counter() - request_started) * 1000,
            routing_reason=routing_reason, thinking_reason=thinking_reason,
            thinking_enabled=thinking_enabled, error="request_deadline_exceeded",
            cache_hit_estimate=routing_reason == "affinity",
        )
        return JSONResponse(
            {"error": {"message": "Request deadline exceeded", "type": "request_timeout"}},
            status_code=504,
        )
    except asyncio.CancelledError:
        latency_ms = (time.perf_counter() - started) * 1000
        await release_backend(state, backend, success=None, lease_id=lease_id)
        await state.mark_cancelled()
        await state.record_route(
            request_id=request_id, backend_name=backend.config.name, status="cancelled", stream=False,
            latency_ms=latency_ms, queue_ms=queue_ms, classifier_ms=classifier_ms,
            total_ms=(time.perf_counter() - request_started) * 1000,
            routing_reason=routing_reason, thinking_reason=thinking_reason,
            thinking_enabled=thinking_enabled, error="client_disconnect",
            cache_hit_estimate=routing_reason == "affinity",
        )
        raise
    except httpx.TimeoutException:
        structured_log("backend_timeout", backend=backend.config.name)
        await release_backend(state, backend, success=False, lease_id=lease_id)
        await state.record_route(
            request_id=request_id,
            backend_name=backend.config.name,
            status="error",
            stream=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            queue_ms=queue_ms, classifier_ms=classifier_ms,
            total_ms=(time.perf_counter() - request_started) * 1000,
            routing_reason=routing_reason, thinking_reason=thinking_reason, thinking_enabled=thinking_enabled,
            error="timeout",
        )
        return JSONResponse({"retryable_backend_error": True, "reason": "timeout"}, status_code=599)
    except httpx.ConnectError:
        structured_log("backend_connection_failed", backend=backend.config.name)
        await release_backend(state, backend, success=False, lease_id=lease_id)
        await state.record_route(
            request_id=request_id,
            backend_name=backend.config.name,
            status="error",
            stream=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            queue_ms=queue_ms, classifier_ms=classifier_ms,
            total_ms=(time.perf_counter() - request_started) * 1000,
            routing_reason=routing_reason, thinking_reason=thinking_reason, thinking_enabled=thinking_enabled,
            error="connection_failed",
        )
        return JSONResponse({"retryable_backend_error": True, "reason": "connection_failed"}, status_code=599)
    except Exception as exc:
        error = f"{type(exc).__name__}:{exc}"
        structured_log("backend_unexpected_error", backend=backend.config.name, error=error[:240])
        await release_backend(state, backend, success=False, lease_id=lease_id)
        await state.record_route(
            request_id=request_id,
            backend_name=backend.config.name,
            status="error",
            stream=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            queue_ms=queue_ms, classifier_ms=classifier_ms,
            total_ms=(time.perf_counter() - request_started) * 1000,
            routing_reason=routing_reason, thinking_reason=thinking_reason, thinking_enabled=thinking_enabled,
            error=error[:240],
        )
        return JSONResponse({"retryable_backend_error": True, "reason": type(exc).__name__}, status_code=599)


async def forward_stream(
    client: httpx.AsyncClient,
    request: Request,
    state: RouterState,
    backend: BackendState,
    lease_id: str,
    payload: dict[str, Any],
    request_id: int,
    request_started: float,
    queue_ms: float,
    classifier_ms: float,
    routing_reason: str,
    thinking_reason: str,
    thinking_enabled: bool | None = None,
    dynamic_thinking: DynamicThinkingConfig | None = None,
    affinity_key: str = "",
    deadline_at: float = float("inf"),
    reasoning_effort: str | None = None,
    reasoning: ReasoningConfig | None = None,
    allowed_backend_names: set[str] | None = None,
) -> StreamingResponse:
    async def body() -> AsyncIterator[bytes]:
        current: BackendState | None = backend
        current_lease_id: str | None = lease_id
        current_routing_reason = routing_reason
        excluded: set[str] = set()
        while current is not None:
            url = current.config.api_base.rstrip("/") + "/chat/completions"
            started = time.perf_counter()
            success = False
            cancelled = False
            emitted = False
            error_reason = ""
            ttft_ms = 0.0
            final_payload: dict[str, Any] = {}
            try:
                async with client.stream(
                    "POST",
                    url,
                    json=rewrite_payload(
                        payload,
                        current.config.backend_model,
                        thinking_enabled,
                        dynamic_thinking,
                        reasoning_effort,
                        reasoning,
                    ),
                ) as resp:
                    if resp.status_code >= 500:
                        structured_log("backend_5xx", backend=current.config.name, status_code=resp.status_code)
                        error_reason = f"backend_5xx:{resp.status_code}"
                    else:
                        structured_log("stream_started", backend=current.config.name)
                        async for chunk in resp.aiter_bytes():
                            if time.time() >= deadline_at:
                                raise RequestDeadlineExceeded
                            if await request.is_disconnected():
                                raise asyncio.CancelledError("client disconnected")
                            if chunk and not emitted:
                                emitted = True
                                ttft_ms = (time.perf_counter() - started) * 1000
                            if b'"usage"' in chunk or b'"timings"' in chunk:
                                for line in chunk.decode("utf-8", errors="ignore").splitlines():
                                    value = line.removeprefix("data:").strip()
                                    if value.startswith("{"):
                                        with suppress(json.JSONDecodeError):
                                            candidate = json.loads(value)
                                            if candidate.get("usage") or candidate.get("timings"):
                                                final_payload = candidate
                            yield chunk
                        success = True
            except asyncio.CancelledError:
                cancelled = True
                error_reason = "client_disconnect"
            except RequestDeadlineExceeded:
                cancelled = True
                error_reason = "request_deadline_exceeded"
            except httpx.TimeoutException:
                structured_log("backend_timeout", backend=current.config.name)
                error_reason = "timeout"
            except httpx.ConnectError:
                structured_log("backend_connection_failed", backend=current.config.name)
                error_reason = "connection_failed"
            except Exception as exc:
                error_reason = f"{type(exc).__name__}:{exc}"[:240]
                structured_log("backend_unexpected_error", backend=current.config.name, error=error_reason)

            latency_ms = (time.perf_counter() - started) * 1000
            await release_backend(
                state,
                current,
                success=None if cancelled else success,
                lease_id=current_lease_id,
                latency_ms=latency_ms if success else 0,
            )
            current_lease_id = None
            metrics = response_metrics(final_payload, latency_ms)
            await state.record_route(
                request_id=request_id,
                backend_name=current.config.name,
                status="cancelled" if error_reason == "client_disconnect" else ("success" if success else "error"),
                stream=True,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                queue_ms=queue_ms, classifier_ms=classifier_ms,
                total_ms=(time.perf_counter() - request_started) * 1000,
                routing_reason=current_routing_reason, thinking_reason=thinking_reason, thinking_enabled=thinking_enabled,
                error="" if success else error_reason,
                cache_hit_estimate=current_routing_reason == "affinity",
                **metrics,
            )
            structured_log("stream_finished", backend=current.config.name, success=success, latency_ms=round(latency_ms, 2))
            if cancelled:
                if error_reason == "client_disconnect":
                    await state.mark_cancelled()
                    raise asyncio.CancelledError("client disconnected")
                yield b'data: {"error":"request_deadline_exceeded"}\n\n'
                return
            if success:
                return
            if emitted:
                yield b'data: {"error":"backend_stream_interrupted"}\n\n'
                return
            excluded.add(current.config.name)
            await state.mark_retry()
            choice = await choose_backend(
                client,
                state,
                payload,
                request_id,
                exclude=excluded,
                affinity_key=affinity_key,
                allowed_backend_names=allowed_backend_names,
            )
            if choice is not None:
                current, current_lease_id = choice
                await state.mark_backend_selected(current, request_id)
                current_routing_reason = current.last_selection_reason
            else:
                current = None
        yield b'data: {"error":"backend_unavailable"}\n\n'

    return StreamingResponse(body(), media_type="text/event-stream")


async def forward_fallback(client: httpx.AsyncClient, state: RouterState, payload: dict[str, Any]) -> JSONResponse | StreamingResponse:
    fb = state.config.fallback
    if not fb.enabled or not fb.api_base or not fb.model:
        return JSONResponse(
            {"error": {"message": "No healthy Qwen backend and no fallback configured", "type": "backend_unavailable"}},
            status_code=503,
        )
    async with state._lock:
        state.fallback_usage_count += 1
        state.telemetry_dirty = True
    structured_log("fallback_used", fallback=fb.name, count=state.fallback_usage_count)
    url = fb.api_base.rstrip("/") + "/chat/completions"
    payload = rewrite_payload(payload, fb.model)
    headers = auth_headers(fb.api_key)
    if payload.get("stream"):
        async def body() -> AsyncIterator[bytes]:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        return StreamingResponse(body(), media_type="text/event-stream")
    resp = await client.post(url, json=payload, headers=headers)
    return JSONResponse(resp.json(), status_code=resp.status_code)


RESTART_REQUIRED_CONFIG_FIELDS = (
    "listen_host",
    "listen_port",
    "request_timeout_seconds",
    "connect_timeout_seconds",
    "telemetry_state_path",
    "telemetry_db_path",
)


async def apply_config_reload(app: FastAPI, current: RouterConfig, updated: RouterConfig) -> list[str]:
    changed_fields = [
        name
        for name in RouterConfig.model_fields
        if getattr(current, name) != getattr(updated, name)
    ]
    restart_changes = [name for name in RESTART_REQUIRED_CONFIG_FIELDS if name in changed_fields]
    if restart_changes:
        raise ValueError("restart required for: " + ", ".join(restart_changes))

    state: RouterState = app.state.router_state
    async with state._lock:
        updated_names = {backend.name for backend in updated.backends}
        active_removed = [
            backend.config.name
            for backend in state.backends
            if backend.config.name not in updated_names and backend.active_requests > 0
        ]
        if active_removed:
            raise ValueError("cannot remove active backends: " + ", ".join(active_removed))

        existing = {backend.config.name: backend for backend in state.backends}
        reloaded_backends: list[BackendState] = []
        for backend_config in updated.backends:
            backend = existing.get(backend_config.name)
            if backend is None:
                backend = BackendState(config=backend_config)
            else:
                endpoint_changed = (
                    backend.config.api_base != backend_config.api_base
                    or backend.config.backend_model != backend_config.backend_model
                )
                watchdog_changed = backend.config.watchdog_base != backend_config.watchdog_base
                backend.config = backend_config
                backend.last_reject_reason = ""
                if endpoint_changed or watchdog_changed:
                    backend.last_watchdog_status = None
                    backend.last_health_checked = 0
                if endpoint_changed:
                    backend.semantic_healthy = None
                    backend.semantic_checked_at = 0
                    backend.semantic_consecutive_failures = 0
                    backend.semantic_error = ""
            reloaded_backends.append(backend)

        if current.cache_affinity_ttl_seconds != updated.cache_affinity_ttl_seconds:
            affinity_payload = state.affinity.persistence_payload()
            state.affinity = SessionAffinity(ttl=updated.cache_affinity_ttl_seconds)
            state.affinity.restore(affinity_payload)
        classifier_settings_changed = (
            current.dynamic_thinking.classifier_cache_ttl_seconds
            != updated.dynamic_thinking.classifier_cache_ttl_seconds
            or current.dynamic_thinking.classifier_cache_max_entries
            != updated.dynamic_thinking.classifier_cache_max_entries
        )
        if classifier_settings_changed:
            classifier_payload = state.classifier_cache.persistence_payload()
            state.classifier_cache = ThinkingClassifierCache(
                ttl=updated.dynamic_thinking.classifier_cache_ttl_seconds,
                max_entries=updated.dynamic_thinking.classifier_cache_max_entries,
            )
            state.classifier_cache.restore(classifier_payload)

        for field_name in RouterConfig.model_fields:
            setattr(current, field_name, getattr(updated, field_name))
        state.config = current
        state.backends = reloaded_backends
        state.config_reload_count += 1
        state.last_config_reload_at = time.time()
        state.last_config_reload_error = ""
        state.telemetry_dirty = True
        state.status_revision += 1

    app.state.status_snapshot = None
    structured_log("config_reloaded", changed_fields=changed_fields, backend_count=len(updated.backends))
    return changed_fields


async def reload_config_from_disk(app: FastAPI, config_path: str, current: RouterConfig) -> bool:
    state: RouterState = app.state.router_state
    try:
        updated = await asyncio.to_thread(load_config, config_path)
        await apply_config_reload(app, current, updated)
        return True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        async with state._lock:
            state.last_config_reload_at = time.time()
            state.last_config_reload_error = error[:500]
            state.status_revision += 1
        app.state.status_snapshot = None
        structured_log("config_reload_failed", error=error)
        return False


async def config_reload_loop(app: FastAPI, config_path: str, current: RouterConfig) -> None:
    while True:
        await app.state.config_reload_event.wait()
        app.state.config_reload_event.clear()
        await reload_config_from_disk(app, config_path, current)


def create_app(config: RouterConfig, config_path: str = "") -> FastAPI:
    timeout = httpx.Timeout(
        timeout=config.request_timeout_seconds,
        connect=config.connect_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.router_state = RouterState(config)
        app.state.config = config
        saved_telemetry = load_telemetry_state(config)
        if saved_telemetry is not None:
            try:
                app.state.router_state.restore_telemetry(saved_telemetry)
            except Exception as exc:
                structured_log("telemetry_restore_failed", error=type(exc).__name__)
        app.state.http = httpx.AsyncClient(timeout=timeout)
        app.state.status_snapshot = None
        app.state.status_snapshot_at = 0.0
        app.state.status_snapshot_revision = -1
        app.state.status_refresh_lock = asyncio.Lock()
        app.state.config_reload_event = asyncio.Event()
        telemetry_task = asyncio.create_task(telemetry_persist_loop(app.state.router_state, config))
        semantic_task = asyncio.create_task(semantic_health_loop(app, config))
        reload_task = asyncio.create_task(config_reload_loop(app, config_path, config))
        loop = asyncio.get_running_loop()
        signal_installed = False
        if config_path:
            try:
                loop.add_signal_handler(signal.SIGHUP, app.state.config_reload_event.set)
                signal_installed = True
            except (NotImplementedError, RuntimeError):
                structured_log("config_reload_signal_unavailable")
        try:
            yield
        finally:
            if signal_installed:
                loop.remove_signal_handler(signal.SIGHUP)
            for task in (reload_task, semantic_task, telemetry_task):
                task.cancel()
            for task in (reload_task, semantic_task, telemetry_task):
                with suppress(asyncio.CancelledError):
                    await task
            await persist_telemetry(app.state.router_state, config)
            await app.state.http.aclose()

    app = FastAPI(title="Hermes NYX Pool Router", lifespan=lifespan)

    async def cached_router_snapshot() -> dict[str, Any]:
        state: RouterState = app.state.router_state
        now = time.monotonic()
        cached = app.state.status_snapshot
        if (
            cached is not None
            and app.state.status_snapshot_revision == state.status_revision
            and now - app.state.status_snapshot_at < config.status_cache_seconds
        ):
            return cached

        async with app.state.status_refresh_lock:
            now = time.monotonic()
            cached = app.state.status_snapshot
            if (
                cached is not None
                and app.state.status_snapshot_revision == state.status_revision
                and now - app.state.status_snapshot_at < config.status_cache_seconds
            ):
                return cached
            client: httpx.AsyncClient = app.state.http
            await refresh_backend_statuses(client, state, config)
            async with state._lock:
                route_count = len(state.recent_routes)
                state._prune_routes_locked()
                if len(state.recent_routes) != route_count:
                    state.telemetry_dirty = True
                cached = router_snapshot(state, config)
            app.state.status_snapshot = cached
            app.state.status_snapshot_at = time.monotonic()
            app.state.status_snapshot_revision = state.status_revision
            return cached

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        return HTMLResponse(dashboard_html(f"{config.public_model_name} router"))

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return await cached_router_snapshot()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return await cached_router_snapshot()

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        context_length = max((backend.max_context_tokens for backend in config.backends if backend.enabled), default=262144)
        model_ids = [config.public_model_name]
        if config.reasoning.expose_reasoning_models:
            model_ids.extend(f"{config.public_model_name}:{effort}" for effort in SUPPORTED_EFFORTS)
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "hermes-qwen-pool",
                    "context_length": context_length,
                }
                for model_id in model_ids
            ],
        }

    @app.post("/admin/register-node")
    async def register_node(request: Request) -> JSONResponse:
        if not config.admin_token:
            return JSONResponse({"error": "node registration is disabled"}, status_code=503)
        auth = request.headers.get("Authorization", "")
        expected = f"Bearer {config.admin_token}"
        if not hmac.compare_digest(auth, expected):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        payload = await request.json()
        registration = NodeRegistration.model_validate(payload)
        backend_config = BackendConfig(
            name=registration.name,
            api_base=registration.api_base,
            watchdog_base=registration.watchdog_base,
            backend_model=registration.backend_model,
            max_context_tokens=registration.max_context_tokens,
            max_parallel_requests=registration.max_parallel_requests,
            enabled=registration.enabled,
            preferred_for_long_context=registration.preferred_for_long_context,
        )
        state: RouterState = app.state.router_state
        await state.add_or_update_backend(backend_config)
        app.state.status_snapshot = None
        try:
            persist_backend(config_path, backend_config)
        except Exception as exc:
            structured_log("registration_persist_failed", backend=backend_config.name, error=type(exc).__name__)
            return JSONResponse(
                {"ok": False, "registered": backend_config.name, "persisted": False, "error": type(exc).__name__},
                status_code=500,
            )
        structured_log(
            "node_registered",
            backend=backend_config.name,
            api_base=backend_config.api_base,
            watchdog_base=backend_config.watchdog_base,
            node_os=registration.node_os,
            hostname=registration.hostname,
        )
        return JSONResponse({"ok": True, "registered": backend_config.name, "persisted": bool(config_path)})

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        request_started = time.perf_counter()
        payload = dict(await request.json())
        requested_model = str(payload.get("model") or config.public_model_name)
        base_model_for_default, _ = parse_model_reasoning_suffix(requested_model)
        model_default = config.reasoning.model_defaults.get(base_model_for_default)
        reasoning_resolution = resolve_reasoning_request(
            payload,
            requested_model,
            model_default=model_default,
            global_default=config.reasoning.default_effort,
        )
        payload["model"] = reasoning_resolution.base_model
        allowed_backend_names, route_fallback = reasoning_backend_filter(
            config,
            reasoning_resolution.effort,
        )
        reasoning_fallback = reasoning_resolution.used_fallback or route_fallback
        if reasoning_resolution.warning:
            structured_log(
                "reasoning_warning",
                requested_model=requested_model,
                raw_reasoning=reasoning_resolution.raw,
                warning=reasoning_resolution.warning,
            )
        state: RouterState = app.state.router_state
        client: httpx.AsyncClient = app.state.http
        priority_name = request.headers.get("x-nyx-priority", "normal").strip().lower()
        priority = {"low": 1, "normal": 5, "high": 9}.get(priority_name, 5)
        with suppress(ValueError):
            priority = max(0, min(9, int(priority_name)))
        timeout_ms = request.headers.get("x-request-timeout-ms", "")
        request_timeout = config.request_deadline_seconds
        with suppress(ValueError):
            request_timeout = min(request_timeout, max(0.1, float(timeout_ms) / 1000))
        request_deadline_at = time.time() + request_timeout
        deadline = min(time.time() + config.max_queue_seconds, request_deadline_at)
        affinity_key = (
            request.headers.get("x-hermes-session-id", "")
            or request.headers.get("x-session-id", "")
            or str(payload.get("user") or "")
        )
        request_id = await state.mark_request_started(
            stream=bool(payload.get("stream")),
            priority=priority,
            deadline_at=request_deadline_at,
        )
        if request_id is None:
            return JSONResponse(
                {"error": {"message": "Router queue is full", "type": "queue_full"}},
                status_code=429,
                headers={"Retry-After": str(config.retry_after_seconds)},
            )
        excluded: set[str] = set()
        while time.time() <= deadline or (excluded and len(excluded) < len(state.backends)):
            if await request.is_disconnected():
                await state.remove_from_queue(request_id)
                await state.mark_cancelled()
                raise asyncio.CancelledError("client disconnected while queued")
            if not excluded and not await state.can_attempt(request_id):
                await asyncio.sleep(0.05)
                continue
            choice = await choose_backend(
                client,
                state,
                payload,
                request_id,
                exclude=excluded,
                affinity_key=affinity_key,
                allowed_backend_names=allowed_backend_names,
            )
            if choice is None:
                await asyncio.sleep(0.1)
                continue
            backend, lease_id = choice
            queue_ms = await state.mark_backend_selected(backend, request_id)
            routing_reason = backend.last_selection_reason
            classifier_started = time.perf_counter()
            if reasoning_resolution.effort is not None:
                thinking_enabled = reasoning_resolution.effort != "none"
                classifier_used = False
                thinking_reason = (
                    f"reasoning_effort:{reasoning_resolution.effort}:"
                    f"{reasoning_resolution.source}"
                )
            else:
                thinking_enabled, classifier_used, thinking_reason = await decide_thinking(
                    client,
                    backend,
                    payload,
                    config.dynamic_thinking,
                    classifier_cache=state.classifier_cache,
                )
            classifier_ms = (time.perf_counter() - classifier_started) * 1000
            await state.record_thinking_decision(enabled=thinking_enabled, classifier_used=classifier_used)
            structured_log(
                "thinking_decision",
                backend=backend.config.name,
                enabled=thinking_enabled,
                classifier_used=classifier_used,
                reason=thinking_reason,
            )
            structured_log(
                "reasoning_route",
                requested_model=requested_model,
                base_model=reasoning_resolution.base_model,
                raw_reasoning=reasoning_resolution.raw,
                reasoning_source=reasoning_resolution.source,
                normalized_effort=reasoning_resolution.effort or "auto",
                selected_upstream=backend.config.name,
                selected_upstream_url=backend.config.api_base,
                fallback_used=reasoning_fallback,
                streaming=bool(payload.get("stream")),
            )
            if payload.get("stream"):
                return await forward_stream(
                    client,
                    request,
                    state,
                    backend,
                    lease_id,
                    payload,
                    request_id,
                    request_started,
                    queue_ms,
                    classifier_ms,
                    routing_reason,
                    thinking_reason,
                    thinking_enabled=thinking_enabled,
                    dynamic_thinking=config.dynamic_thinking,
                    affinity_key=affinity_key,
                    deadline_at=request_deadline_at,
                    reasoning_effort=reasoning_resolution.effort,
                    reasoning=config.reasoning,
                    allowed_backend_names=allowed_backend_names,
                )
            resp = await forward_non_stream(
                client,
                request,
                state,
                backend,
                lease_id,
                payload,
                request_id,
                request_started,
                queue_ms,
                classifier_ms,
                routing_reason,
                thinking_reason,
                thinking_enabled=thinking_enabled,
                dynamic_thinking=config.dynamic_thinking,
                deadline_at=request_deadline_at,
                reasoning_effort=reasoning_resolution.effort,
                reasoning=config.reasoning,
            )
            if resp.status_code == 599:
                excluded.add(backend.config.name)
                await state.mark_retry()
                continue
            return resp
        await state.remove_from_queue(request_id)
        await state.record_route(
            request_id=request_id,
            backend_name="none",
            status="error",
            stream=bool(payload.get("stream")),
            error="backend_unavailable",
            queue_ms=(time.perf_counter() - request_started) * 1000,
            total_ms=(time.perf_counter() - request_started) * 1000,
            routing_reason="unavailable",
        )
        fallback = await forward_fallback(client, state, payload)
        if fallback.status_code == 503:
            fallback.headers["Retry-After"] = str(config.retry_after_seconds)
        return fallback

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hermes Qwen replica pool router")
    parser.add_argument("--config", default=os.getenv("HERMES_ROUTER_CONFIG", "router/config.yaml.example"))
    parser.add_argument("--check-config", action="store_true", help="validate configuration and exit")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
    args = parse_args()
    config = load_config(args.config)
    if args.check_config:
        print(
            json.dumps(
                {
                    "ok": True,
                    "config": str(Path(args.config).resolve()),
                    "listen": f"{config.listen_host}:{config.listen_port}",
                    "backends": [backend.name for backend in config.backends],
                    "semantic_health_enabled": config.semantic_health.enabled,
                },
                separators=(",", ":"),
            )
        )
        return
    import uvicorn

    uvicorn.run(create_app(config, args.config), host=config.listen_host, port=config.listen_port)


if __name__ == "__main__":
    main()
