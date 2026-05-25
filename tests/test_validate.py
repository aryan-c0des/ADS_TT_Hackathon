"""
Unit tests for the business-rule cross-checks in src/validate.py.

The validator owns three risky behaviours that judges grade directly:
  - Normalising free-form LLM outputs into the canonical CSV cell values
  - Auto-flipping Reauthorization Required when a duration is present
  - Coercing Quantity Limits to 'Not specified' when no labelled evidence exists

These tests pin those behaviours so a future prompt change can't silently
move the answers around.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import step_graph, validate  # noqa: E402


# ---------------------------------------------------------------------------
# _normalise_age
# ---------------------------------------------------------------------------
def test_normalise_age_explicit_threshold():
    assert validate._normalise_age("18 years or older") == ">=18"
    assert validate._normalise_age(">=6") == ">=6"
    assert validate._normalise_age("adult patients (18+)") == ">=18"


def test_normalise_age_fda_label():
    assert validate._normalise_age("FDA labelled age") == "FDA labelled age"
    assert validate._normalise_age("Per FDA label") == "FDA labelled age"


def test_normalise_age_empty_or_none():
    assert validate._normalise_age("") == "No"
    assert validate._normalise_age("NA") == "No"
    assert validate._normalise_age("None") == "No"


# ---------------------------------------------------------------------------
# _normalise_yes_no
# ---------------------------------------------------------------------------
def test_normalise_yes_no_canonical():
    assert validate._normalise_yes_no("Yes") == "Yes"
    assert validate._normalise_yes_no("y") == "Yes"
    assert validate._normalise_yes_no("true") == "Yes"
    assert validate._normalise_yes_no("1") == "Yes"
    assert validate._normalise_yes_no("No") == "No"
    assert validate._normalise_yes_no("n") == "No"
    assert validate._normalise_yes_no("false") == "No"


def test_normalise_yes_no_falls_back_to_default():
    assert validate._normalise_yes_no("maybe") == "No"
    assert validate._normalise_yes_no("") == "No"
    assert validate._normalise_yes_no("maybe", default="Yes") == "Yes"


# ---------------------------------------------------------------------------
# _normalise_months — covers the bug where '30 days' became '30' (months)
# ---------------------------------------------------------------------------
def test_normalise_months_integer_passes_through():
    assert validate._normalise_months("12") == "12"
    assert validate._normalise_months("6 months") == "6"
    assert validate._normalise_months("12-month authorization") == "12"


def test_normalise_months_day_strings_always_convert():
    # Regression: the old code only converted when n >= 60, so '30 days'
    # stayed as '30' and was interpreted as 30 months downstream.
    assert validate._normalise_months("30 days") == "1"
    assert validate._normalise_months("45 days") == "2"
    assert validate._normalise_months("90 days") == "3"
    assert validate._normalise_months("183 days") == "6"
    assert validate._normalise_months("365 days") == "12"


def test_normalise_months_year_strings_convert():
    assert validate._normalise_months("1 year") == "12"
    assert validate._normalise_months("2 years") == "24"


def test_normalise_months_week_strings_convert():
    # 4 weeks ≈ 1 month; 52 weeks ≈ 12
    assert validate._normalise_months("4 weeks") == "1"
    assert validate._normalise_months("52 weeks") == "12"


def test_normalise_months_unspecified_and_na():
    assert validate._normalise_months("Unspecified") == "Unspecified"
    assert validate._normalise_months("") == "Unspecified"
    assert validate._normalise_months("NA") == "NA"
    assert validate._normalise_months("n/a") == "NA"
    assert validate._normalise_months("some words no digits") == "Unspecified"


# ---------------------------------------------------------------------------
# validate() cross-checks — the high-risk auto-fixes
# ---------------------------------------------------------------------------
def _make_extracted(scalars=None, step_data=None, text_fields=None):
    return SimpleNamespace(
        scalars=scalars or {},
        step_data=step_data or {},
        text_fields=text_fields or {},
    )


def _empty_count():
    return step_graph.CountResult("NA", "NA", "NA", [], [], False)


def test_reauth_required_auto_flips_yes_when_duration_present():
    extracted = _make_extracted(
        scalars={
            "age": {"value": ">=18"},
            "tb_test_required": {"value": "Yes"},
            "initial_authorization_duration_months": {"value": "6"},
            "reauthorization_duration_months": {"value": "12"},
            "reauthorization_required": {"value": "No"},  # contradicts duration
        }
    )
    o = validate.validate(extracted, _empty_count())
    assert o.fixed["Reauthorization Required"] == "Yes"
    assert any("auto-flipped" in f for f in o.flags)


def test_reauth_required_left_no_when_no_duration_no_text():
    extracted = _make_extracted(
        scalars={
            "age": {"value": ">=18"},
            "tb_test_required": {"value": "No"},
            "initial_authorization_duration_months": {"value": "Unspecified"},
            "reauthorization_duration_months": {"value": "NA"},
            "reauthorization_required": {"value": "No"},
        }
    )
    o = validate.validate(extracted, _empty_count())
    assert o.fixed["Reauthorization Required"] == "No"


def test_quantity_limits_coerced_when_evidence_unsupported():
    # LLM emitted a value but the evidence snippet doesn't contain the
    # "quantity limit" keyword family → coerce to "Not specified"
    extracted = _make_extracted(
        scalars={"age": {"value": ">=18"}, "tb_test_required": {"value": "No"},
                 "initial_authorization_duration_months": {"value": "Unspecified"},
                 "reauthorization_duration_months": {"value": "NA"},
                 "reauthorization_required": {"value": "No"}},
        text_fields={
            "reauthorization_requirements": {"value": "NA", "evidence": ""},
            "specialist_types": {"value": "NA", "evidence": ""},
            "quantity_limits": {
                "value": "patient must take 1 tablet daily",
                "evidence": "1 tablet daily for treatment of plaque psoriasis",
            },
        },
    )
    o = validate.validate(extracted, _empty_count())
    assert o.fixed["Quantity Limits"] == "Not specified"
    assert any("quantity_limits" in f for f in o.flags)


def test_quantity_limits_kept_when_evidence_has_the_label():
    extracted = _make_extracted(
        scalars={"age": {"value": ">=18"}, "tb_test_required": {"value": "No"},
                 "initial_authorization_duration_months": {"value": "Unspecified"},
                 "reauthorization_duration_months": {"value": "NA"},
                 "reauthorization_required": {"value": "No"}},
        text_fields={
            "reauthorization_requirements": {"value": "NA", "evidence": ""},
            "specialist_types": {"value": "NA", "evidence": ""},
            "quantity_limits": {
                "value": "1 vial per 84 days",
                "evidence": "Quantity Limit: 1 vial per 84 days",
            },
        },
    )
    o = validate.validate(extracted, _empty_count())
    assert o.fixed["Quantity Limits"] == "1 vial per 84 days"


def test_step_count_with_empty_text_triggers_reprompt_flag():
    count = step_graph.CountResult(brands=2, generics="NA", phototherapy="No",
                                   brand_trace=[], generic_trace=[], photo_present=False)
    extracted = _make_extracted(
        scalars={"age": {"value": ">=18"}, "tb_test_required": {"value": "No"},
                 "initial_authorization_duration_months": {"value": "Unspecified"},
                 "reauthorization_duration_months": {"value": "NA"},
                 "reauthorization_required": {"value": "No"}},
        step_data={"step_therapy_text": ""},
    )
    o = validate.validate(extracted, count)
    assert "step_therapy" in o.needs_reprompt
    assert any("step count > 0" in f for f in o.flags)


if __name__ == "__main__":
    import traceback
    tests = [
        test_normalise_age_explicit_threshold,
        test_normalise_age_fda_label,
        test_normalise_age_empty_or_none,
        test_normalise_yes_no_canonical,
        test_normalise_yes_no_falls_back_to_default,
        test_normalise_months_integer_passes_through,
        test_normalise_months_day_strings_always_convert,
        test_normalise_months_year_strings_convert,
        test_normalise_months_week_strings_convert,
        test_normalise_months_unspecified_and_na,
        test_reauth_required_auto_flips_yes_when_duration_present,
        test_reauth_required_left_no_when_no_duration_no_text,
        test_quantity_limits_coerced_when_evidence_unsupported,
        test_quantity_limits_kept_when_evidence_has_the_label,
        test_step_count_with_empty_text_triggers_reprompt_flag,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
