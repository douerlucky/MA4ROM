"""Regression tests for constraint-preserving FK and DP candidate handling."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DPMapping import data_property_mapping_agent as dp_mapping
from DPMapping.data_property_mapping_agent import _match_SE
from candidate_generation import _handle_SR, _prepare_fk_metadata, generate_candidates


NS = "https://synthetic.invalid/ontology#"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


def u(local: str) -> str:
    return NS + local


def ontology() -> dict:
    country, city = u("Country"), u("City")
    return {
        "classes": [country, city],
        "object_properties": {},
        "datatype_properties": {
            u("countryValue"): {"domain": [country], "range": [XSD_STRING]},
            u("cityValue"): {"domain": [city], "range": [XSD_STRING]},
        },
        "subclass_of": {},
        "ancestors_of": {},
        "incompatible_classes": {country: [city], city: [country]},
    }


def test_dp_mapping_does_not_select_explicitly_disjoint_top_candidate():
    ont = ontology()
    result = _match_SE(
        "country",
        {
            "pattern": "SE",
            "table_class_candidates": [{"uri": u("Country"), "score": 1.0}],
            "columns": {
                "value": {
                    "role": "data_attr",
                    "candidates": [
                        {"uri": u("cityValue"), "score": 0.95, "domain": [u("City")]},
                        {"uri": u("countryValue"), "score": 0.60, "domain": [u("Country")]},
                    ],
                }
            },
        },
        ontology=ont,
    )
    assert result["columns"]["value"]["prop_uri"] == u("countryValue")


def test_dp_mapping_rejects_unrelated_named_domain_without_disjoint_axiom():
    entity, other = u("Entity"), u("Other")
    prop = u("otherValue")
    ont = {
        "classes": [entity, other],
        "object_properties": {},
        "datatype_properties": {
            prop: {"domain": [other], "range": [XSD_STRING]},
        },
        "ancestors_of": {entity: [], other: []},
        "union_members": {},
        "incompatible_classes": {entity: [], other: []},
    }
    result = _match_SE(
        "records",
        {
            "pattern": "SE",
            "table_class_candidates": [{"uri": entity, "score": 1.0}],
            "columns": {
                "value": {
                    "role": "data_attr",
                    "column_type": "text",
                    "candidates": [
                        {
                            "uri": prop,
                            "local_name": "otherValue",
                            "score": 0.99,
                            "domain": [other],
                            "range": [XSD_STRING],
                        }
                    ],
                }
            },
        },
        ontology=ont,
    )
    assert result["columns"]["value"]["prop_uri"] is None


def test_dp_mapping_accepts_property_declared_on_class_ancestor():
    parent, child = u("Parent"), u("Child")
    prop = u("label")
    ont = {
        "classes": [parent, child],
        "object_properties": {},
        "datatype_properties": {
            prop: {"domain": [parent], "range": [XSD_STRING]},
        },
        "ancestors_of": {parent: [], child: [parent]},
        "union_members": {},
        "incompatible_classes": {parent: [], child: []},
    }
    result = _match_SE(
        "records",
        {
            "pattern": "SE",
            "table_class_candidates": [{"uri": child, "score": 1.0}],
            "columns": {
                "label": {
                    "role": "data_attr",
                    "column_type": "text",
                    "candidates": [
                        {
                            "uri": prop,
                            "local_name": "label",
                            "score": 0.99,
                            "domain": [parent],
                            "range": [XSD_STRING],
                        }
                    ],
                }
            },
        },
        ontology=ont,
    )
    assert result["columns"]["label"]["prop_uri"] == prop


def test_object_property_named_column_is_not_forced_into_dp_lane():
    entity, target = u("Entity"), u("Target")
    literal = u("label")
    ont = {
        "classes": [entity, target],
        "object_properties": {
            u("assignedByReviewer"): {
                "domain": [entity],
                "range": [target],
            }
        },
        "datatype_properties": {
            literal: {"domain": [entity], "range": [XSD_STRING]},
        },
        "ancestors_of": {entity: [], target: []},
        "union_members": {},
        "incompatible_classes": {entity: [], target: []},
    }
    result = _match_SE(
        "records",
        {
            "pattern": "SE",
            "table_class_candidates": [{"uri": entity, "score": 1.0}],
            "columns": {
                "assigned_by_reviewer": {
                    "role": "data_attr",
                    "column_type": "integer",
                    "candidates": [
                        {
                            "uri": literal,
                            "local_name": "label",
                            "score": 0.99,
                            "domain": [entity],
                            "range": [XSD_STRING],
                        }
                    ],
                }
            },
        },
        ontology=ont,
    )
    assert result["columns"]["assigned_by_reviewer"]["prop_uri"] is None


def test_obviously_incompatible_sql_xsd_range_is_filtered_before_selection():
    entity, count = u("Entity"), u("count")
    xsd_int = "http://www.w3.org/2001/XMLSchema#int"
    ont = {
        "classes": [entity],
        "object_properties": {},
        "datatype_properties": {
            count: {"domain": [entity], "range": [xsd_int]},
        },
        "ancestors_of": {entity: []},
        "union_members": {},
        "incompatible_classes": {entity: []},
    }
    result = _match_SE(
        "records",
        {
            "pattern": "SE",
            "table_class_candidates": [{"uri": entity, "score": 1.0}],
            "columns": {
                "count": {
                    "role": "data_attr",
                    "column_type": "text",
                    "candidates": [
                        {
                            "uri": count,
                            "local_name": "count",
                            "score": 1.0,
                            "domain": [entity],
                            "range": [xsd_int],
                        }
                    ],
                }
            },
        },
        ontology=ont,
    )
    assert result["columns"]["count"]["prop_uri"] is None


def test_dp_provider_explicit_null_remains_abstained(monkeypatch):
    entity = u("Entity")
    candidates = [
        {
            "uri": u("label"),
            "local_name": "label",
            "score": 0.65,
            "domain": [entity],
            "range": [XSD_STRING],
        },
        {
            "uri": u("description"),
            "local_name": "description",
            "score": 0.60,
            "domain": [entity],
            "range": [XSD_STRING],
        },
    ]
    ont = {
        "object_properties": {},
        "datatype_properties": {
            candidate["uri"]: {
                "domain": candidate["domain"],
                "range": candidate["range"],
            }
            for candidate in candidates
        },
        "ancestors_of": {entity: []},
        "union_members": {},
    }
    monkeypatch.setattr(
        dp_mapping,
        "_call_llm",
        lambda *_args, **_kwargs: {
            "selected_uri": None,
            "reason": "insufficient semantic evidence",
        },
    )

    selected, confidence = dp_mapping._resolve_data_attr(
        "records",
        "opaque_value",
        entity,
        candidates,
        sql_type="text",
        ontology=ont,
    )

    assert selected is None
    assert confidence == "low"


def test_fk_memberships_keep_scalar_and_composite_constraints():
    schema = {
        "owner": {
            "columns": {"id": "text", "code": "text", "name": "text"},
            "primary_key": ["id"],
            "foreign_keys": [
                {
                    "constraint_name": "owner_target_composite",
                    "fk_arity": 2,
                    "column": "code",
                    "references_table": "target",
                    "references_column": "code",
                },
                {
                    "constraint_name": "owner_target_scalar",
                    "fk_arity": 1,
                    "column": "code",
                    "references_table": "country",
                    "references_column": "code",
                },
            ],
        },
        "target": {
            "columns": {"code": "text", "part": "text"},
            "primary_key": ["code", "part"],
            "foreign_keys": [],
        },
        "country": {
            "columns": {"code": "text"},
            "primary_key": ["code"],
            "foreign_keys": [],
        },
    }
    selected, groups = _prepare_fk_metadata(
        schema["owner"]["foreign_keys"], schema, "owner"
    )
    assert set(groups) == {"owner_target_composite", "owner_target_scalar"}
    assert selected["code"]["ref_table"] == "country"
    assert len(selected["code"]["fk_memberships"]) == 2
    assert len(selected["code"]["composite_memberships"]) == 1


def test_sr_uses_one_representative_per_constraint():
    fk_cols = {
        "left": {"column": "left", "ref_table": "A", "ref_col": "id"},
        "left_code": {"column": "left_code", "ref_table": "A", "ref_col": "code"},
        "right": {"column": "right", "ref_table": "B", "ref_col": "id"},
        "right_code": {"column": "right_code", "ref_table": "B", "ref_col": "code"},
    }
    constraints = {
        "left_fk": [fk_cols["left"], fk_cols["left_code"]],
        "right_fk": [fk_cols["right"], fk_cols["right_code"]],
    }
    result = _handle_SR(
        "link",
        {"left": "text", "left_code": "text", "right": "text", "right_code": "text"},
        set(),
        fk_cols,
        [u("A"), u("B")],
        {},
        {"A": {"primary_key": ["id", "code"]}, "B": {"primary_key": ["id", "code"]}},
        ontology={"classes": [u("A"), u("B")], "object_properties": {}},
        fk_constraints=constraints,
    )
    assert result["fk1"]["ref_table"] == "A"
    assert result["fk2"]["ref_table"] == "B"
