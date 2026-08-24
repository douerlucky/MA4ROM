"""Regression tests for the bounded enum-to-Class LLM output contract."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealValue.real_value_enhancement_agent import (  # noqa: E402
    _enum_llm_candidate_catalog,
    _merge_enum_llm_assignments,
    _real_value_type_mapping,
)


NS = "http://example.test/ontology#"


def test_compact_candidate_ids_expand_without_long_uri_output() -> None:
    candidate = NS + "Membership"
    catalog = _enum_llm_candidate_catalog(
        [{"uri": candidate, "local_name": "Membership", "score": 1.0}]
    )
    values = [f"status-{index}" for index in range(20)]
    response = {
        "candidate_assignments": {f"v{index}": "c0" for index in range(20)},
        "unmapped_value_ids": [],
        "confidence": "medium",
        "reason": "synthetic compact response",
    }

    merged = _merge_enum_llm_assignments(
        response=response,
        review_values=values,
        candidate_catalog=catalog,
        allowed={candidate},
        current_class_uri=None,
        ontology=None,
        bool_hint_classes=[],
        locked_by_rule=set(),
    )

    assert merged == {value: candidate for value in values}


def test_candidate_catalog_retains_ontology_annotations() -> None:
    candidate = NS + "CompactArtifact"
    catalog = _enum_llm_candidate_catalog(
        [{"uri": candidate, "local_name": "CompactArtifact", "score": 1.0}],
        ontology={
            "class_annotations": {
                candidate: {
                    "labels": ["Compact artifact"],
                    "comments": ["Contains only a reduced representation."],
                }
            }
        },
    )

    assert catalog == [
        {
            "id": "c0",
            "local_name": "CompactArtifact",
            "uri": candidate,
            "labels": ["Compact artifact"],
            "comments": ["Contains only a reduced representation."],
        }
    ]


def test_enum_mapping_prompt_uses_short_ids_for_large_domains() -> None:
    candidate = NS + "Membership"
    values = [f"status-{index}" for index in range(20)]
    profiles = {value: [{"type": value}] for value in values}
    response = {
        "candidate_assignments": {f"v{index}": "c0" for index in range(20)},
        "unmapped_value_ids": [],
        "confidence": "medium",
        "reason": "synthetic compact response",
    }

    with patch(
        "RealValue.real_value_enhancement_agent._call_llm",
        return_value=response,
    ) as call:
        result = _real_value_type_mapping(
            table_name="membership_rows",
            type_col="type",
            value_profiles=profiles,
            class_candidates=[
                {"uri": candidate, "local_name": "Membership", "score": 1.0}
            ],
            current_class_uri=None,
        )

    assert result["value_to_class"] == {value: candidate for value in values}
    assert result["unmapped_values"] == []
    prompt = call.call_args.args[0]
    assert '"candidate_assignments"' in prompt
    assert '"v19"' in prompt
    assert '"c0"' in prompt
    assert "候选顺序和相同的 matcher 分数都不是语义证据" in prompt
