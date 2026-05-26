"""
Unit tests for the deterministic step counter.

The Reference sheet in PA_Business_Rules.xlsx provides ONE fully worked
example. The hand-built graph below mirrors that policy. If the counter
returns the expected values, the AND/OR/UNION logic is correct.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import step_graph  # noqa: E402


def test_reference_example():
    """Reference sheet: universal-Yesintek AND OR(biologic, OR(photo, generic))
    must resolve to brands=1, generics=1, phototherapy=No.

    Path-selection rule encoded in step_graph._Path.sort_key:
      - Outer OR: A=(1 brand) vs B=(inner-OR resolved). Both 1 total step → tie
        broken by photo-first (B's chosen child is generic, 0 photo) and then
        brand-fewer (B has 0 brand). B wins.
      - Inner OR: photo (0,0,0,1) vs generic (0,1,0,0). Both 1 step → tie
        broken by photo (generic wins because it has 0 photo).
      - Universal Yesintek contributes 1 brand via AND.
      Final: 1 brand + 1 generic, phototherapy=No.
    """
    graph = {
        "universal_branch": [
            {
                "logic": "LEAF",
                "drug_or_category": "Yesintek (ustekinumab-kfce)",
                "class": "BRANDED_BIOLOGIC",
                "is_mandatory": True,
            }
        ],
        "indication_branch": [
            {
                "logic": "OR",
                "drug_or_category": "indication-specific (PsO)",
                "class": "OTHER",
                "is_mandatory": True,
                "children": [
                    {
                        "logic": "LEAF",
                        "drug_or_category": "biologic or targeted synthetic (Sotyktu, Otezla)",
                        "class": "BRANDED_BIOLOGIC",
                        "is_mandatory": True,
                    },
                    {
                        "logic": "OR",
                        "drug_or_category": "phototherapy OR generic systemic",
                        "class": "OTHER",
                        "is_mandatory": True,
                        "children": [
                            {
                                "logic": "LEAF",
                                "drug_or_category": "phototherapy (UVB, PUVA)",
                                "class": "PHOTOTHERAPY",
                                "is_mandatory": False,
                            },
                            {
                                "logic": "LEAF",
                                "drug_or_category": "methotrexate / cyclosporine / acitretin",
                                "class": "GENERIC_SYSTEMIC",
                                "is_mandatory": False,
                            },
                        ],
                    },
                ],
            }
        ],
    }
    r = step_graph.count_steps(graph)
    assert r.brands == 1, f"Expected brands=1, got {r.brands}"
    assert r.generics == 1, f"Expected generics=1, got {r.generics}"
    assert r.phototherapy == "No", f"Expected photo=No, got {r.phototherapy}"


def test_or_picks_single_coherent_path_not_per_class_min():
    """Regression: the old counter computed brand-min and generic-min INDEPENDENTLY
    inside the same OR, which lets an OR with [A=(1 brand,0 gen), B=(0 brand,1 gen)]
    resolve to (0, 0). The new counter must pick A or B and report that path's
    counts coherently."""
    graph = {
        "universal_branch": [],
        "indication_branch": [
            {
                "logic": "OR",
                "children": [
                    {"logic": "LEAF", "drug_or_category": "Stelara",
                     "class": "BRANDED_BIOLOGIC", "is_mandatory": True},
                    {"logic": "LEAF", "drug_or_category": "methotrexate",
                     "class": "GENERIC_SYSTEMIC", "is_mandatory": True},
                ],
            }
        ],
    }
    r = step_graph.count_steps(graph)
    # Both branches are 1 total step; tie broken by photo (both 0) then brand (B has 0).
    # Patient picks generic. Result: brand=0 → 'NA', generic=1.
    assert r.brands == "NA", f"Expected brands=NA, got {r.brands}"
    assert r.generics == 1, f"Expected generics=1, got {r.generics}"


def test_empty_graph_returns_na():
    r = step_graph.count_steps({"universal_branch": [], "indication_branch": []})
    assert r.brands == "NA"
    assert r.generics == "NA"
    assert r.phototherapy == "NA"


def test_phototherapy_under_or_is_not_mandatory():
    """OR(photo, methotrexate): phototherapy is NOT mandatory (it's under
    an OR alternative), so phototherapy='No'. The chosen path is the generic
    branch (sort_key tie-broken by photo-count → methotrexate wins), so
    generics=1."""
    graph = {
        "universal_branch": [],
        "indication_branch": [
            {
                "logic": "OR",
                "children": [
                    {"logic": "LEAF", "drug_or_category": "phototherapy",
                     "class": "PHOTOTHERAPY", "is_mandatory": True},
                    {"logic": "LEAF", "drug_or_category": "methotrexate",
                     "class": "GENERIC_SYSTEMIC", "is_mandatory": True},
                ],
            }
        ],
    }
    r = step_graph.count_steps(graph)
    assert r.phototherapy == "No"
    assert r.generics == 1, f"Expected generics=1 (methotrexate path chosen), got {r.generics}"
    assert r.brands == "NA"


def test_phototherapy_mandatory_when_at_top_level():
    graph = {
        "universal_branch": [],
        "indication_branch": [
            {"logic": "LEAF", "drug_or_category": "phototherapy",
             "class": "PHOTOTHERAPY", "is_mandatory": True}
        ],
    }
    r = step_graph.count_steps(graph)
    assert r.phototherapy == "Yes"
    # Phototherapy is NEVER counted in brands or generics
    assert r.brands == "NA"
    assert r.generics == "NA"


def test_branded_step_counting_via_whitelist():
    # LLM mis-labels Stelara as GENERIC; the whitelist must override.
    graph = {
        "universal_branch": [],
        "indication_branch": [
            {"logic": "LEAF", "drug_or_category": "Stelara",
             "class": "GENERIC_SYSTEMIC", "is_mandatory": True},
            {"logic": "LEAF", "drug_or_category": "methotrexate",
             "class": "GENERIC_SYSTEMIC", "is_mandatory": True},
        ],
    }
    r = step_graph.count_steps(graph)
    assert r.brands == 1, f"Stelara should reclassify to BRANDED_BIOLOGIC, got {r.brands}"
    assert r.generics == 1


def test_topical_counts_as_generic():
    graph = {
        "universal_branch": [],
        "indication_branch": [
            {"logic": "LEAF", "drug_or_category": "topical corticosteroid",
             "class": "TOPICAL", "is_mandatory": True}
        ],
    }
    r = step_graph.count_steps(graph)
    assert r.generics == 1
    assert r.brands == "NA"


def test_biologic_leaf_with_tar_substring_not_misclassified():
    """A leaf described as 'a biologic or targeted synthetic drug
    (e.g., Sotyktu, Otezla)' contains the substring 'tar' inside the
    word 'targeted'. The old classifier substring-matched 'tar' (a
    coal-tar TOPICAL keyword) BEFORE the biologic whitelist, so this
    leaf was misclassified as TOPICAL → counted as a generic step.

    Per CLAUDE.md rule #4 the brand whitelist is the source of truth:
    'Sotyktu' and 'Otezla' are both in BRAND_WHITELIST_BIOLOGIC, so the
    leaf must reconcile to BRANDED_BIOLOGIC and contribute +1 brand.

    Regression-tests the real STELARA case at 148593-4960549.pdf where
    the universal branch is empty and the bug surfaces directly."""
    graph = {
        "universal_branch": [],
        "indication_branch": [
            {"logic": "LEAF",
             "drug_or_category": "a biologic or targeted synthetic drug (e.g., Sotyktu, Otezla)",
             "class": "BRANDED_BIOLOGIC",
             "is_mandatory": True},
        ],
    }
    r = step_graph.count_steps(graph)
    assert r.brands == 1, f"Sotyktu/Otezla leaf must count as 1 brand, got brands={r.brands}"
    assert r.generics == "NA", f"No generic step in this graph; got generics={r.generics}"


def test_word_boundary_prevents_false_topical_match():
    """Defensive: ensure substring-style false positives never sneak in.
    'targeted' contains 'tar' (a TOPICAL keyword), 'started' contains
    'tar', etc. None of these should classify as TOPICAL."""
    from src.step_graph import classify_drug_name
    assert classify_drug_name("targeted immune modulator") == "OTHER", \
        "'targeted' must not classify as TOPICAL via 'tar' substring"
    assert classify_drug_name("started therapy") == "OTHER", \
        "'started' must not classify as TOPICAL via 'tar' substring"
    # And confirm the real coal-tar topical still works
    assert classify_drug_name("coal tar preparation") == "TOPICAL", \
        "Real 'tar' as a standalone word must still classify as TOPICAL"


if __name__ == "__main__":
    import traceback
    tests = [
        test_reference_example,
        test_or_picks_single_coherent_path_not_per_class_min,
        test_empty_graph_returns_na,
        test_phototherapy_under_or_is_not_mandatory,
        test_phototherapy_mandatory_when_at_top_level,
        test_branded_step_counting_via_whitelist,
        test_topical_counts_as_generic,
        test_biologic_leaf_with_tar_substring_not_misclassified,
        test_word_boundary_prevents_false_topical_match,
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
