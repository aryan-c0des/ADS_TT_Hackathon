"""
Gemini wrapper with disk caching, retry, and JSON-schema enforcement.

Every call is keyed on SHA256(model + temperature + system + prompt + schema) so
re-runs from cold are deterministic and re-runs from warm cost zero API calls.
We ship the cache in the submission ZIP so judges' first run works offline.

If the GEMINI_API_KEY env var isn't set we still return cached responses; we
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
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore


_call_counter = {"calls": 0, "errors": 0, "synthetic_hits": 0, "real_hits": 0}


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
            # from a real Gemini response. pipeline.run_all surfaces a
            # warning at the end of a run if any synthetic entries were hit.
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
        "source": "gemini",
        "raw_text": response_text,
        "payload": payload,
    }, indent=2), encoding="utf-8")


def _ensure_configured(model: str):
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed. `pip install google-generativeai`"
        )
    api_key = config.get_api_key()
    if not api_key:
        raise RuntimeError(
            f"{config.GEMINI_API_KEY_ENV} is not set; cannot make live LLM calls."
        )
    genai.configure(api_key=api_key)


def call_json(prompt: str,
              schema: Dict[str, Any],
              *,
              system: str = "",
              temperature: float = config.GEMINI_TEMPERATURE_DEFAULT,
              model: str = config.GEMINI_MODEL,
              force: bool = False) -> LLMResult:
    """Call Gemini with a JSON response schema. Returns parsed payload.

    Caches on (model, temperature, system, prompt, schema).
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

    _ensure_configured(model)
    cfg = {
        "response_mime_type": "application/json",
        "response_schema": schema,
        "temperature": temperature,
        "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
    }
    last_err: Optional[Exception] = None
    for attempt in range(config.GEMINI_MAX_RETRIES):
        try:
            m = genai.GenerativeModel(  # type: ignore[attr-defined]
                model_name=model,
                system_instruction=system or None,
                generation_config=cfg,
            )
            resp = m.generate_content(prompt)
            _call_counter["calls"] += 1
            _warn_on_budget()
            text = (resp.text or "").strip()
            payload = json.loads(text)
            _write_cache(prompt_hash, text, payload)
            return LLMResult(
                payload=payload, cache_hit=False, raw_text=text,
                prompt_hash=prompt_hash, model=model,
            )
        except json.JSONDecodeError as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            _call_counter["errors"] += 1
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gemini call failed after retries: {last_err}")


def counter_state() -> dict:
    """Read the in-process counter — calls made, errors hit, and the
    real-vs-synthetic cache-hit split. pipeline.run_all surfaces this."""
    return dict(_call_counter)


_budget_warned = {"80": False, "100": False}


def _warn_on_budget() -> None:
    """Surface a one-shot warning at 80% and 100% of the daily call budget
    so the user knows BEFORE the next call returns 429 RESOURCE_EXHAUSTED.
    Quota counters reset at UTC midnight on Gemini free tier."""
    n = _call_counter["calls"]
    if n >= config.DAILY_CALL_BUDGET and not _budget_warned["100"]:
        _budget_warned["100"] = True
        print(
            f"[llm_client] WARNING: hit DAILY_CALL_BUDGET={config.DAILY_CALL_BUDGET}. "
            "Further calls likely to return 429 RESOURCE_EXHAUSTED until UTC midnight."
        )
    elif n >= int(config.DAILY_CALL_BUDGET * 0.8) and not _budget_warned["80"]:
        _budget_warned["80"] = True
        print(
            f"[llm_client] heads up: {n} calls made ({n*100//config.DAILY_CALL_BUDGET}% "
            f"of DAILY_CALL_BUDGET={config.DAILY_CALL_BUDGET})."
        )
