from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


SUPPORTED_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")

_ALIASES = {
    "off": "none",
    "false": "none",
    "0": "none",
    "fast": "none",
    "no": "none",
    "disabled": "none",
    "min": "minimal",
    "tiny": "minimal",
    "normal": "medium",
    "default": "medium",
    "standard": "medium",
    "deep": "high",
    "strong": "high",
    "max": "xhigh",
    "extreme": "xhigh",
    "maximum": "xhigh",
}


@dataclass(frozen=True)
class ReasoningResolution:
    base_model: str
    effort: str | None
    raw: Any = None
    source: str = "dynamic_rules"
    used_fallback: bool = False
    warning: str | None = None


def canonical_reasoning_effort(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    return normalized if normalized in SUPPORTED_EFFORTS else None


def normalize_reasoning_effort(
    value: Any,
    default_effort: Any = "medium",
    *,
    source: str = "request",
) -> ReasoningResolution:
    effort = canonical_reasoning_effort(value)
    if effort is not None:
        return ReasoningResolution(base_model="", effort=effort, raw=value, source=source)

    fallback = canonical_reasoning_effort(default_effort) or "medium"
    if value is None:
        return ReasoningResolution(base_model="", effort=fallback, source="configured_default")

    return ReasoningResolution(
        base_model="",
        effort=fallback,
        raw=value,
        source=source,
        used_fallback=True,
        warning=f"invalid reasoning effort {value!r}; using {fallback}",
    )


def parse_model_reasoning_suffix(model: str) -> tuple[str, str | None]:
    base_model, separator, suffix = model.rpartition(":")
    if not separator or not base_model:
        return model, None
    effort = canonical_reasoning_effort(suffix)
    return (base_model, effort) if effort is not None else (model, None)


def extract_reasoning_hint(payload: dict[str, Any]) -> tuple[Any, str] | None:
    direct_paths = (
        ("reasoningEffort",),
        ("reasoning_effort",),
        ("reasoning", "effort"),
        ("reasoning", "level"),
        ("providerOptions", "openai", "reasoningEffort"),
        ("providerOptions", "openai", "reasoning_effort"),
        ("providerOptions", "nyx", "reasoningEffort"),
        ("providerOptions", "nyx", "reasoning_effort"),
        ("providerOptions", "openaiCompatible", "reasoningEffort"),
        ("providerOptions", "openaiCompatible", "reasoning_effort"),
        ("providerOptions", "openai-compatible", "reasoningEffort"),
        ("providerOptions", "openai-compatible", "reasoning_effort"),
        ("extra_body", "reasoningEffort"),
        ("extra_body", "reasoning_effort"),
        ("extra_body", "reasoning", "effort"),
    )
    for path in direct_paths:
        current: Any = payload
        found = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                found = False
                break
            current = current[key]
        if found:
            return current, ".".join(path)
    return None


def resolve_reasoning_request(
    payload: dict[str, Any],
    requested_model: str,
    *,
    model_default: Any = None,
    global_default: Any = None,
) -> ReasoningResolution:
    base_model, suffix_effort = parse_model_reasoning_suffix(requested_model)
    if suffix_effort is not None:
        return ReasoningResolution(
            base_model=base_model,
            effort=suffix_effort,
            raw=requested_model.rpartition(":")[2],
            source="model_suffix",
        )

    hint = extract_reasoning_hint(payload)
    if hint is not None:
        raw, source = hint
        fallback = model_default if canonical_reasoning_effort(model_default) else global_default
        normalized = normalize_reasoning_effort(raw, fallback or "medium", source=source)
        return ReasoningResolution(
            base_model=base_model,
            effort=normalized.effort,
            raw=normalized.raw,
            source=normalized.source,
            used_fallback=normalized.used_fallback,
            warning=normalized.warning,
        )

    for value, source in ((model_default, "model_default"), (global_default, "global_default")):
        effort = canonical_reasoning_effort(value)
        if effort is not None:
            return ReasoningResolution(base_model=base_model, effort=effort, source=source)

    # An unconfigured router keeps its existing keyword and slash-command behavior.
    return ReasoningResolution(base_model=base_model, effort=None)


def _strip_keys(container: Any, keys: tuple[str, ...]) -> None:
    if not isinstance(container, dict):
        return
    for key in keys:
        container.pop(key, None)


def strip_client_reasoning_fields(payload: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(payload)
    _strip_keys(out, ("reasoningEffort", "reasoning_effort"))

    reasoning = out.get("reasoning")
    _strip_keys(reasoning, ("effort", "level"))
    if isinstance(reasoning, dict) and not reasoning:
        out.pop("reasoning", None)

    provider_options = out.get("providerOptions")
    if isinstance(provider_options, dict):
        for provider in ("openai", "nyx", "openaiCompatible", "openai-compatible"):
            provider_config = provider_options.get(provider)
            _strip_keys(provider_config, ("reasoningEffort", "reasoning_effort"))
            if isinstance(provider_config, dict) and not provider_config:
                provider_options.pop(provider, None)
        if not provider_options:
            out.pop("providerOptions", None)

    extra_body = out.get("extra_body")
    if isinstance(extra_body, dict):
        _strip_keys(extra_body, ("reasoningEffort", "reasoning_effort"))
        nested_reasoning = extra_body.get("reasoning")
        _strip_keys(nested_reasoning, ("effort", "level"))
        if isinstance(nested_reasoning, dict) and not nested_reasoning:
            extra_body.pop("reasoning", None)
        if not extra_body:
            out.pop("extra_body", None)

    return out
