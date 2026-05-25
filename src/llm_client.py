"""
Groq wrapper with disk caching, retry, and JSON-schema enforcement.

The hackathon mandates Llama via Groq (was Gemini until the spec changed).
Groq supports `response_format={"type": "json_object"}` which guarantees valid
JSON, but it does NOT enforce a schema — so we inject the schema into the
system prompt as text and validate required keys on the parsed output,
retrying with the error message appended so the model can self-correct.

Every call is keyed on SHA256(model + temperature + system + prompt + schema)
so re-runs from cold are deterministic and re-runs from warm cost zero API
calls. We ship the cache in the submission ZIP so judges' first run works
offline.

If the GROQ_API_KEY env var isn't set we still return cached responses; we
only raise on a true cache miss with no key configured. That lets us iterate
locally during development without burning quota.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import config

try:
    from groq import Groq
except ImportError:
    Groq = None  # type: ignore


_call_counter = {"calls": 0, "errors": 0, "synthetic_hits": 0, "real_hits": 0}
_client: Optional[Any] = None


@dataclass
class LLMResult:
    payload: Dict[str, Any]
    cache_hit: bool
    raw_text: str
    prompt_hash: str
    model: str


def _hash(model: str, temperature: float, system: str, prompt: str,
          schema_str: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(f"|{temperature:.2f}|".encode())
    h.update(system.encode())
    h.update(prompt.encode())
    h.update(schema_str.encode())
    return h.hexdigest()[:24]


def _cache_path(prompt_hash: str):
    return config.LLM_CACHE / f"{prompt_hash}.json"


def _read_cache(prompt_hash: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(prompt_hash)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            # Track whether this hit came from the synthetic seed cache or
            # from a real LLM response. pipeline.run_all surfaces a warning
            # at the end of a run if any synthetic entries were hit.
            if data.get("source") == "synthetic":
                _call_counter["synthetic_hits"] += 1
            else:
                _call_counter["real_hits"] += 1
            return data
        except json.JSONDecodeError:
            return None
    return None


def _write_cache(prompt_hash: str, response_text: str, payload: Dict[str, Any]) -> None:
    p = _cache_path(prompt_hash)
    p.write_text(json.dumps({
        "source": "groq",
        "raw_text": response_text,
        "payload": payload,
    }, indent=2), encoding="utf-8")


def _ensure_client():
    global _client
    if _client is not None:
        return _client
    if Groq is None:
        raise RuntimeError(
            "groq is not installed. `pip install groq`"
        )
    api_key = config.get_api_key()
    if not api_key:
        raise RuntimeError(
            f"{config.LLM_API_KEY_ENV} is not set; cannot make live LLM calls."
        )
    _client = Groq(api_key=api_key)
    return _client


def _schema_required_keys(schema: Dict[str, Any]) -> list[str]:
    """Return the top-level required keys from a JSON schema dict, or [] if
    none declared. Used for post-hoc validation since Groq doesn't enforce
    schemas natively."""
    req = schema.get("required")
    return list(req) if isinstance(req, list) else []


def _format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    """Render the schema as compact pretty JSON for injection into the system
    prompt. Llama follows explicit schema text far better than implicit
    field-by-field descriptions."""
    return json.dumps(schema, indent=2, sort_keys=False)


def _build_messages(system: str, prompt: str, schema: Dict[str, Any],
                    correction_note: str = "") -> list[Dict[str, str]]:
    """Compose the chat messages. The schema is appended to the system role
    so Groq's json_object mode has a structural target, and any correction
    note from a previous failed attempt is appended to the user prompt."""
    sys_full = (
        f"{system}\n\n"
        "OUTPUT FORMAT — STRICT\n"
        "Return a single JSON object that conforms to this JSON Schema. "
        "Do not wrap it in markdown fences. Do not add prose around it. "
        "All keys listed under `required` MUST be present.\n\n"
        f"JSON SCHEMA:\n{_format_schema_for_prompt(schema)}"
    )
    user_full = prompt
    if correction_note:
        user_full = (
            f"{prompt}\n\n"
            f"NOTE — your previous attempt failed validation: {correction_note}\n"
            "Return a corrected JSON object."
        )
    return [
        {"role": "system", "content": sys_full},
        {"role": "user", "content": user_full},
    ]


def call_json(prompt: str,
              schema: Dict[str, Any],
              *,
              system: str = "",
              temperature: float = config.LLM_TEMPERATURE_DEFAULT,
              model: str = config.LLM_MODEL,
              force: bool = False) -> LLMResult:
    """Call the LLM with a JSON response schema. Returns parsed payload.

    Caches on (model, temperature, system, prompt, schema). The schema is
    injected into the system prompt for Llama (no native schema enforcement),
    and validated post-hoc against required top-level keys.
    """
    schema_str = json.dumps(schema, sort_keys=True)
    prompt_hash = _hash(model, temperature, system, prompt, schema_str)
    cached = _read_cache(prompt_hash)
    if cached and not force:
        return LLMResult(
            payload=cached["payload"],
            cache_hit=True,
            raw_text=cached.get("raw_text", ""),
            prompt_hash=prompt_hash,
            model=model,
        )

    client = _ensure_client()
    required_keys = _schema_required_keys(schema)
    correction_note = ""
    last_err: Optional[Exception] = None
    last_text: str = ""

    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            messages = _build_messages(system, prompt, schema, correction_note)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
            )
            _call_counter["calls"] += 1
            _warn_on_budget()
            text = (resp.choices[0].message.content or "").strip()
            last_text = text

            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                correction_note = f"output was not valid JSON ({exc.msg})"
                last_err = exc
                continue

            missing = [k for k in required_keys if k not in payload]
            if missing:
                correction_note = (
                    f"missing required top-level keys: {missing}. "
                    "Include every key listed under `required` in the schema."
                )
                last_err = ValueError(correction_note)
                continue

            _write_cache(prompt_hash, text, payload)
            return LLMResult(
                payload=payload, cache_hit=False, raw_text=text,
                prompt_hash=prompt_hash, model=model,
            )
        except Exception as exc:  # noqa: BLE001
            # Network / rate-limit / SDK errors — back off and retry the
            # whole call, not just the validation loop.
            last_err = exc
            _call_counter["errors"] += 1
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(
        f"LLM call failed after {config.LLM_MAX_RETRIES} retries: {last_err}. "
        f"Last raw output (truncated): {last_text[:300]!r}"
    )


def counter_state() -> dict:
    """Read the in-process counter — calls made, errors hit, and the
    real-vs-synthetic cache-hit split. pipeline.run_all surfaces this."""
    return dict(_call_counter)


_budget_warned = {"80": False, "100": False}


def _warn_on_budget() -> None:
    """Surface a one-shot warning at 80% and 100% of the daily call budget.
    Groq's free tier is rate-limited per-minute (RPM + TPM) rather than per
    day, so this is a soft heads-up — the hard backpressure comes from 429s
    in the retry loop."""
    n = _call_counter["calls"]
    if n >= config.DAILY_CALL_BUDGET and not _budget_warned["100"]:
        _budget_warned["100"] = True
        print(
            f"[llm_client] WARNING: hit DAILY_CALL_BUDGET={config.DAILY_CALL_BUDGET}. "
            "Watch for 429 rate-limit errors from Groq."
        )
    elif n >= int(config.DAILY_CALL_BUDGET * 0.8) and not _budget_warned["80"]:
        _budget_warned["80"] = True
        print(
            f"[llm_client] heads up: {n} calls made ({n*100//config.DAILY_CALL_BUDGET}% "
            f"of DAILY_CALL_BUDGET={config.DAILY_CALL_BUDGET})."
        )
