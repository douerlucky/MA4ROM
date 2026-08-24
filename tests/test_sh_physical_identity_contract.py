"""Regression tests for schema-backed SH identity resolution."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealValue.real_value_enhancement_agent import _repair_invalid_sh_class_pair
from r2rml_generator import build_entity_identity_contracts


NS = "https://synthetic.invalid/ontology#"


def _schema() -> dict:
    return {
        "parent_rows": {
            "columns": {"parent_id": "integer"},
            "primary_key": ["parent_id"],
            "foreign_keys": [],
        },
        "child_rows": {
            "columns": {"child_id": "integer", "value": "text"},
            "primary_key": ["child_id"],
            "foreign_keys": [
                {
                    "constraint_name": "child_parent_fk",
                    "fk_arity": 1,
                    "column": "child_id",
                    "references_table": "parent_rows",
                    "references_column": "parent_id",
                }
            ],
        },
    }


def test_same_class_pair_keeps_physical_parent_identity() -> None:
    parent = NS + "Entity"
    child = {
        "pattern": "SH",
        "sub_class_uri": parent,
        "parent_class_uri": parent,
        "columns": {"child_id": {"role": "sh_inherited_pk"}},
    }
    alignment = {
        "parent_rows": {"pattern": "SE", "class_uri": parent},
        "child_rows": child,
    }
    candidates = {
        "parent_table": "parent_rows",
        "parent_class_candidates": [{"uri": parent, "score": 1.0}],
    }
    ontology = {"subclass_of": {}, "ancestors_of": {}}

    repaired = _repair_invalid_sh_class_pair(
        "child_rows",
        child,
        candidates,
        ontology,
        enriched_schema=_schema(),
        alignment=alignment,
    )

    assert repaired["parent_class_uri"] == parent
    assert repaired["parent_table"] == "parent_rows"
    assert repaired["sh_class_validation"]["status"] == "valid_physical_shared_class"

    contracts = build_entity_identity_contracts(alignment, _schema(), ontology)
    assert contracts["child_rows"]["root_table"] == "parent_rows"
    assert contracts["child_rows"]["identity_columns"] == ["child_id"]


def test_non_owl_parent_pair_uses_fk_evidence_without_ancestor_rewrite() -> None:
    child_class = NS + "RoleEntity"
    parent_class = NS + "Entity"
    child = {
        "pattern": "SH",
        "sub_class_uri": child_class,
        "parent_class_uri": parent_class,
        "columns": {"child_id": {"role": "sh_inherited_pk"}},
    }
    alignment = {
        "parent_rows": {"pattern": "SE", "class_uri": parent_class},
        "child_rows": child,
    }
    candidates = {
        "parent_table": "parent_rows",
        "parent_class_candidates": [{"uri": parent_class, "score": 1.0}],
    }
    ontology = {
        "subclass_of": {},
        "ancestors_of": {},
        "incompatible_classes": {},
    }

    repaired = _repair_invalid_sh_class_pair(
        "child_rows",
        child,
        candidates,
        ontology,
        enriched_schema=_schema(),
        alignment=alignment,
    )
    assert repaired["parent_class_uri"] == parent_class
    assert repaired["sh_class_validation"]["status"] == (
        "valid_physical_identity_non_owl_subclass"
    )

    contracts = build_entity_identity_contracts(alignment, _schema(), ontology)
    assert contracts["child_rows"]["validated_inheritance"] is True
    assert contracts["child_rows"]["semantic_relation_verified"] is False

