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
import os
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


def call_text(prompt: str, *, system: str = "",
              temperature: float = config.GEMINI_TEMPERATURE_DEFAULT,
              model: str = config.GEMINI_MODEL,
              force: bool = False) -> LLMResult:
    """Plain-text completion (used for one-off LLM-judge or anchor location)."""
    prompt_hash = _hash(model, temperature, system, prompt, "<text>")
    cached = _read_cache(prompt_hash)
    if cached and not force:
        return LLMResult(
            payload={"text": cached.get("raw_text", "")},
            cache_hit=True,
            raw_text=cached.get("raw_text", ""),
            prompt_hash=prompt_hash, model=model,
        )
    _ensure_configured(model)
    cfg = {
        "temperature": temperature,
        "max_output_tokens": config.GEMINI_MAX_OUTPUT_TOKENS,
    }
    m = genai.GenerativeModel(  # type: ignore[attr-defined]
        model_name=model,
        system_instruction=system or None,
        generation_config=cfg,
    )
    resp = m.generate_content(prompt)
    _call_counter["calls"] += 1
    text = (resp.text or "").strip()
    _write_cache(prompt_hash, text, {"text": text})
    return LLMResult(
        payload={"text": text}, cache_hit=False, raw_text=text,
        prompt_hash=prompt_hash, model=model,
    )


def locate_anchors(filename: str, brand: str, text: str) -> dict:
    """Ask Gemini for (start_anchor, end_anchor) substrings that bracket
    the brand's section in the policy. Used by segment_brand as a fallback."""
    schema = {
        "type": "object",
        "properties": {
            "start_anchor": {"type": "string"},
            "end_anchor":   {"type": "string"},
            "confidence":   {"type": "number"},
        },
        "required": ["start_anchor", "end_anchor"],
    }
    system = (
        "You are a clinical document analyst. Given a Prior Authorization policy, "
        "you return two short verbatim substrings that mark the start and end of "
        "the section that governs the requested brand for plaque psoriasis."
    )
    prompt = (
        f"Brand: {brand}\nFilename: {filename}\n\n"
        "Return two 15-30 character verbatim substrings from the policy. "
        "start_anchor = the first words of the section. end_anchor = the first "
        "words of whatever comes AFTER the section. Both must be exact substrings "
        "(copy-pasteable). If the section runs to the end of the policy, return "
        'end_anchor as the empty string "".\n\nPOLICY TEXT:\n'
        + text[:60_000]
    )
    return call_json(prompt, schema, system=system).payload


def counter_state() -> dict:
    return dict(_call_counter)
