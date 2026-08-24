"""Structural guard for key-only relation tables misclassified as entities."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from candidate_generation import generate_candidates
from DPMapping.data_property_mapping_agent import run_data_property_mapping


NS = "https://synthetic.invalid/ontology#"


def uri(local: str) -> str:
    return NS + local


def ontology(*, include_relation_class: bool = False) -> dict:
    classes = [uri("Person"), uri("ProgramCommitteeChair")]
    if include_relation_class:
        classes.append(uri("markConflictOfInterest"))
    return {
        "classes": classes,
        "object_properties": {
            uri("markConflictOfInterest"): {
                "domain": [uri("Person")],
                "range": [uri("Person")],
            },
            uri("reviews"): {
                "domain": [uri("Person")],
                "range": [uri("Person")],
            },
        },
        "datatype_properties": {},
        "subclass_of": {},
        "ancestors_of": {},
        "union_members": {},
        "incompatible_classes": {},
    }


def key_only_schema(*, include_payload: bool = False) -> dict:
    columns = {"left_id": "integer", "right_id": "integer"}
    if include_payload:
        columns["created_at"] = "timestamp"
    return {
        "markConflictOfInterest": {
            "columns": columns,
            "primary_key": ["left_id", "right_id"],
            "foreign_keys": [],
        }
    }


def test_key_only_composite_table_with_unique_op_name_abstains_as_relation():
    candidates = generate_candidates(
        key_only_schema(),
        {"markConflictOfInterest": "SE"},
        ontology(),
    )
    entry = candidates["markConflictOfInterest"]

    assert entry["pattern"] == "SR"
    assert entry["table_kind"] == "key_only_relation"
    assert entry["relation_status"] == "abstained_incomplete_physical_endpoints"
    assert entry["fk1"]["column"] is None
    assert entry["fk2"]["column"] is None
    assert [item["uri"] for item in entry["sr_prop_candidates"]] == [
        uri("markConflictOfInterest")
    ]

    alignment = run_data_property_mapping(candidates, ontology=ontology())
    assert alignment["markConflictOfInterest"]["pattern"] == "SR"
    assert alignment["markConflictOfInterest"]["relation_status"] == (
        "abstained_incomplete_physical_endpoints"
    )
    assert "class_uri" not in alignment["markConflictOfInterest"]


def test_exact_ontology_class_prevents_relation_override():
    candidates = generate_candidates(
        key_only_schema(),
        {"markConflictOfInterest": "SE"},
        ontology(include_relation_class=True),
    )
    assert candidates["markConflictOfInterest"]["pattern"] == "SE"


def test_payload_column_prevents_key_only_relation_override():
    candidates = generate_candidates(
        key_only_schema(include_payload=True),
        {"markConflictOfInterest": "SE"},
        ontology(),
    )
    assert candidates["markConflictOfInterest"]["pattern"] == "SE"

