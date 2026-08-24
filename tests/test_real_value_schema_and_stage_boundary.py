"""Synthetic guards for RealValue SQL safety and paper-stage ownership."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
from typing import Any

from psycopg2 import sql


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealValue import real_value_enhancement_agent as real_value


def _identifier_values(fragment: Any) -> list[str]:
    if isinstance(fragment, sql.Identifier):
        return list(fragment.strings)
    if isinstance(fragment, sql.Composed):
        values: list[str] = []
        for child in fragment._wrapped:
            values.extend(_identifier_values(child))
        return values
    return []


class _Cursor:
    def __init__(self, distinct_rows=None) -> None:
        self.executions: list[tuple[Any, Any]] = []
        self.description = [("kind",), ("payload",)]
        self._fetchall_calls = 0
        self.distinct_rows = distinct_rows or [("A",)]

    def execute(self, query: Any, params: Any = None) -> None:
        self.executions.append((query, params))

    def fetchall(self):
        self._fetchall_calls += 1
        if self._fetchall_calls == 1:
            return self.distinct_rows
        return [("A", 1)]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _Connection:
    def __init__(self, distinct_rows=None) -> None:
        self.cursor_instance = _Cursor(distinct_rows)

    def cursor(self) -> _Cursor:
        return self.cursor_instance

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


class _RelationCursor:
    def __init__(self, group_rows, linked_rows) -> None:
        self.group_rows = group_rows
        self.linked_rows = list(linked_rows)
        self.executions: list[tuple[Any, Any]] = []

    def execute(self, query: Any, params: Any = None) -> None:
        self.executions.append((query, params))

    def fetchall(self):
        return list(self.group_rows)

    def fetchone(self):
        return (self.linked_rows.pop(0),)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _RelationConnection:
    def __init__(self, group_rows, linked_rows) -> None:
        self.cursor_instance = _RelationCursor(group_rows, linked_rows)
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> _RelationCursor:
        return self.cursor_instance

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _relation_context_fixture():
    ns = "http://example.test/ontology#"
    base = ns + "BaseRecord"
    subtype = ns + "AuthoredRecord"
    agent = ns + "Agent"
    role = ns + "AuthorRole"
    source_table = "entity_rows"
    target_table = "agent_rows"
    type_column = "category_code"
    relation_column = "authoredBy"
    schema = {
        source_table: {
            "columns": {
                "entity_id": "integer",
                type_column: "integer",
                relation_column: "integer",
            },
            "primary_key": ["entity_id"],
            "foreign_keys": [],
        },
        target_table: {
            "columns": {"agent_id": "integer"},
            "primary_key": ["agent_id"],
            "foreign_keys": [],
        },
    }
    alignment = {
        source_table: {
            "pattern": "SE",
            "class_uri": base,
            "class_confidence": "high",
            "columns": {
                "entity_id": {"role": "pk"},
                type_column: {"role": "discriminator"},
                relation_column: {
                    "role": "data_attr",
                    "prop_uri": None,
                    "confidence": "low",
                },
            },
        },
        target_table: {
            "pattern": "SE",
            "class_uri": agent,
            "class_confidence": "high",
            "columns": {"agent_id": {"role": "pk"}},
        },
    }
    ontology = {
        "object_properties": {
            ns + relation_column: {
                "domain": [subtype],
                "range": [role],
            }
        },
        "children_of": {base: [subtype], agent: [role]},
        "descendants_of": {base: [subtype], agent: [role]},
        "subclass_of": {subtype: [base], role: [agent]},
        "ancestors_of": {subtype: [base], role: [agent]},
        "incompatible_classes": {},
    }
    return {
        "base": base,
        "subtype": subtype,
        "source_table": source_table,
        "type_column": type_column,
        "relation_column": relation_column,
        "schema": schema,
        "alignment": alignment,
        "ontology": ontology,
    }


def test_distinct_profiles_schema_qualify_hostile_identifiers(monkeypatch) -> None:
    connection = _Connection()
    monkeypatch.setattr(real_value, "_get_conn", lambda: connection)
    table = 'records"; DROP TABLE audit; --'
    column = 'kind"; SELECT pg_sleep(1); --'

    values, profiles, value_domain = real_value._fetch_distinct_value_profiles(
        table, column, per_value_limit=2, max_values=4
    )

    assert values == ["A"]
    assert profiles == {"A": [{"kind": "A", "payload": 1}]}
    assert value_domain == {
        "complete": True,
        "truncated": False,
        "observed_distinct": 1,
        "max_values": 4,
    }
    first_query, first_params = connection.cursor_instance.executions[0]
    second_query, second_params = connection.cursor_instance.executions[1]
    assert isinstance(first_query, sql.Composed)
    assert _identifier_values(first_query) == [column, real_value.DB_SCHEMA_NAME, table, column]
    assert first_params == (5,)
    assert _identifier_values(second_query) == [real_value.DB_SCHEMA_NAME, table, column]
    assert second_params == ("A", 2)


def test_distinct_profiles_reports_truncated_value_domain(monkeypatch) -> None:
    connection = _Connection([(value,) for value in "ABCDE"])
    monkeypatch.setattr(real_value, "_get_conn", lambda: connection)

    values, profiles, value_domain = real_value._fetch_distinct_value_profiles(
        "records", "kind", per_value_limit=2, max_values=4
    )

    assert values == ["A", "B", "C", "D"]
    assert set(profiles) == {"A", "B", "C", "D"}
    assert value_domain == {
        "complete": False,
        "truncated": True,
        "observed_distinct": 4,
        "max_values": 4,
    }


def test_every_real_value_execute_builds_sql_composition() -> None:
    source = inspect.getsource(real_value)
    tree = ast.parse(source)
    unsafe = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "execute" or not node.args:
            continue
        if isinstance(node.args[0], (ast.JoinedStr, ast.Constant)):
            unsafe.append(node.lineno)
    assert unsafe == []


def test_real_value_cannot_select_object_property_uri(monkeypatch) -> None:
    monkeypatch.setattr(
        real_value,
        "_get_conn",
        lambda: (_ for _ in ()).throw(AssertionError("retired OP path touched the database")),
    )
    assert real_value._build_exact_object_property_discriminator_evidence(
        table_name="records",
        type_col="kind",
        group_values=["A"],
        class_candidates=[{"uri": "http://example.test/Record"}],
        enriched_schema={"records": {"columns": {"kind": "text"}}},
        alignment={},
        ontology={"object_properties": {"http://example.test/hasPeer": {}}},
    ) == {}
    assert "object_property_evidence" not in inspect.signature(
        real_value._score_enum_value_by_rules
    ).parameters
    assert "object_property_evidence" not in inspect.signature(
        real_value._real_value_type_mapping
    ).parameters


def test_all_null_data_attr_without_name_or_unique_type_evidence_abstains(
    monkeypatch,
) -> None:
    ns = "https://synthetic.invalid/ontology#"
    entity = ns + "Entity"
    label = ns + "label"
    description = ns + "description"
    xsd_string = "http://www.w3.org/2001/XMLSchema#string"
    alignment = {
        "records": {
            "pattern": "SE",
            "class_uri": entity,
            "class_confidence": "high",
            "columns": {
                "opaque_value": {
                    "role": "data_attr",
                    "prop_uri": label,
                    "confidence": "low",
                }
            },
        }
    }
    candidates = {
        "records": {
            "pattern": "SE",
            "table_class_candidates": [
                {"uri": entity, "local_name": "Entity", "score": 1.0}
            ],
            "columns": {
                "opaque_value": {
                    "role": "data_attr",
                    "column_type": "text",
                    "candidates": [
                        {
                            "uri": label,
                            "local_name": "label",
                            "score": 0.8,
                            "domain": [entity],
                            "range": [xsd_string],
                        },
                        {
                            "uri": description,
                            "local_name": "description",
                            "score": 0.7,
                            "domain": [entity],
                            "range": [xsd_string],
                        },
                    ],
                }
            },
        }
    }
    ontology = {
        "classes": [entity],
        "object_properties": {},
        "datatype_properties": {
            label: {"domain": [entity], "range": [xsd_string]},
            description: {"domain": [entity], "range": [xsd_string]},
        },
        "ancestors_of": {entity: []},
        "children_of": {entity: []},
        "union_members": {},
        "incompatible_classes": {entity: []},
    }

    monkeypatch.setattr(
        real_value,
        "fetch_sample_rows",
        lambda *_args, **_kwargs: [{"opaque_value": None}],
    )
    monkeypatch.setattr(
        real_value,
        "_build_fk_semantic_context",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        real_value,
        "_fetch_non_null_counts",
        lambda *_args, **_kwargs: {"opaque_value": 0},
    )
    monkeypatch.setattr(
        real_value,
        "_call_llm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("all-NULL abstention must not call the provider")
        ),
    )

    result = real_value.run_real_value_enhancement(
        alignment,
        {
            "records": {
                "table_low": False,
                "columns_low": ["opaque_value"],
                "identity_dp_low": [],
            }
        },
        candidates,
        ontology=ontology,
        enriched_schema={
            "records": {
                "columns": {"opaque_value": "text"},
                "primary_key": [],
                "foreign_keys": [],
            }
        },
    )

    column = result["records"]["columns"]["opaque_value"]
    assert column["prop_uri"] is None
    assert column["confidence"] == "low"
    assert column["abstain_reason"] == "all_null_without_name_or_type_evidence"


def test_real_value_provider_abstention_does_not_restore_old_dp(monkeypatch) -> None:
    ns = "https://synthetic.invalid/ontology#"
    entity = ns + "Entity"
    label = ns + "label"
    description = ns + "description"
    xsd_string = "http://www.w3.org/2001/XMLSchema#string"
    alignment = {
        "records": {
            "pattern": "SE",
            "class_uri": entity,
            "class_confidence": "high",
            "columns": {
                "opaque_value": {
                    "role": "data_attr",
                    "prop_uri": label,
                    "confidence": "low",
                }
            },
        }
    }
    candidate_list = [
        {
            "uri": label,
            "local_name": "label",
            "score": 0.65,
            "domain": [entity],
            "range": [xsd_string],
        },
        {
            "uri": description,
            "local_name": "description",
            "score": 0.60,
            "domain": [entity],
            "range": [xsd_string],
        },
    ]
    candidates = {
        "records": {
            "pattern": "SE",
            "table_class_candidates": [{"uri": entity, "score": 1.0}],
            "columns": {
                "opaque_value": {
                    "role": "data_attr",
                    "column_type": "text",
                    "candidates": candidate_list,
                }
            },
        }
    }
    ontology = {
        "classes": [entity],
        "object_properties": {},
        "datatype_properties": {
            label: {"domain": [entity], "range": [xsd_string]},
            description: {"domain": [entity], "range": [xsd_string]},
        },
        "ancestors_of": {entity: []},
        "children_of": {entity: []},
        "union_members": {},
        "incompatible_classes": {entity: []},
    }
    monkeypatch.setattr(
        real_value,
        "fetch_sample_rows",
        lambda *_args, **_kwargs: [{"opaque_value": "payload"}],
    )
    monkeypatch.setattr(
        real_value,
        "_build_fk_semantic_context",
        lambda **_kwargs: {},
    )
    monkeypatch.setattr(
        real_value,
        "_call_llm",
        lambda *_args, **_kwargs: {
            "selected_uri": None,
            "confidence": "low",
            "reason": "insufficient semantic evidence",
        },
    )

    result = real_value.run_real_value_enhancement(
        alignment,
        {
            "records": {
                "table_low": False,
                "columns_low": ["opaque_value"],
                "identity_dp_low": [],
            }
        },
        candidates,
        ontology=ontology,
        enriched_schema={
            "records": {
                "columns": {"opaque_value": "text"},
                "primary_key": [],
                "foreign_keys": [],
            }
        },
    )

    column = result["records"]["columns"]["opaque_value"]
    assert column["prop_uri"] is None
    assert column["confidence"] == "low"
    assert column["abstain_reason"] == "provider_abstained"


def test_invalid_sh_pair_is_repaired_from_owl_hierarchy() -> None:
    ns = "https://synthetic.invalid/ontology#"
    child = ns + "Child"
    parent = ns + "Parent"
    entry = {
        "pattern": "SH",
        "sub_class_uri": child,
        "parent_class_uri": child,
    }
    repaired = real_value._repair_invalid_sh_class_pair(
        "child_rows",
        entry,
        {"sub_class_candidates": [{"uri": child, "score": 1.0}]},
        {
            "subclass_of": {child: [parent]},
            "ancestors_of": {child: [parent]},
        },
    )
    assert repaired["sub_class_uri"] == child
    assert repaired["parent_class_uri"] == parent
    assert repaired["sh_class_validation"]["status"] == "repaired_from_owl_subclass"


def test_complete_contrastive_relation_context_supports_only_its_domain(monkeypatch) -> None:
    fixture = _relation_context_fixture()
    connection = _RelationConnection(
        group_rows=[("0", 4, 0), ("1", 4, 4)],
        linked_rows=[4],
    )
    monkeypatch.setattr(real_value, "_get_conn", lambda: connection)

    evidence = real_value._build_relation_domain_class_evidence(
        table_name=fixture["source_table"],
        type_col=fixture["type_column"],
        group_values=["0", "1"],
        class_candidates=[{"uri": fixture["subtype"]}],
        current_class_uri=fixture["base"],
        enriched_schema=fixture["schema"],
        alignment=fixture["alignment"],
        ontology=fixture["ontology"],
        value_domain_complete=True,
    )

    assert set(evidence) == {"1"}
    assert evidence["1"][0]["class_uri"] == fixture["subtype"]
    assert evidence["1"][0]["evidence_source"] == "validated_relation_domain_context"
    assert evidence["1"][0]["total"] == 4
    assert evidence["1"][0]["nonnull"] == 4
    assert evidence["1"][0]["linked"] == 4
    assert "property_uri" not in evidence["1"][0]
    assert connection.closed is True
    assert all(
        isinstance(query, sql.Composed)
        for query, _params in connection.cursor_instance.executions
    )


def test_relation_context_abstains_without_null_peer_contrast(monkeypatch) -> None:
    fixture = _relation_context_fixture()
    connection = _RelationConnection(
        group_rows=[("0", 4, 4), ("1", 4, 4)],
        linked_rows=[],
    )
    monkeypatch.setattr(real_value, "_get_conn", lambda: connection)

    evidence = real_value._build_relation_domain_class_evidence(
        table_name=fixture["source_table"],
        type_col=fixture["type_column"],
        group_values=["0", "1"],
        class_candidates=[{"uri": fixture["subtype"]}],
        current_class_uri=fixture["base"],
        enriched_schema=fixture["schema"],
        alignment=fixture["alignment"],
        ontology=fixture["ontology"],
        value_domain_complete=True,
    )

    assert evidence == {}
    assert len(connection.cursor_instance.executions) == 1


def test_relation_context_abstains_on_partial_range_identity(monkeypatch) -> None:
    fixture = _relation_context_fixture()
    connection = _RelationConnection(
        group_rows=[("0", 4, 0), ("1", 4, 4)],
        linked_rows=[3],
    )
    monkeypatch.setattr(real_value, "_get_conn", lambda: connection)

    evidence = real_value._build_relation_domain_class_evidence(
        table_name=fixture["source_table"],
        type_col=fixture["type_column"],
        group_values=["0", "1"],
        class_candidates=[{"uri": fixture["subtype"]}],
        current_class_uri=fixture["base"],
        enriched_schema=fixture["schema"],
        alignment=fixture["alignment"],
        ontology=fixture["ontology"],
        value_domain_complete=True,
    )

    assert evidence == {}


def test_relation_context_requires_complete_enum_domain(monkeypatch) -> None:
    fixture = _relation_context_fixture()
    monkeypatch.setattr(
        real_value,
        "_get_conn",
        lambda: (_ for _ in ()).throw(AssertionError("incomplete domain queried the database")),
    )

    evidence = real_value._build_relation_domain_class_evidence(
        table_name=fixture["source_table"],
        type_col=fixture["type_column"],
        group_values=["0", "1"],
        class_candidates=[{"uri": fixture["subtype"]}],
        current_class_uri=fixture["base"],
        enriched_schema=fixture["schema"],
        alignment=fixture["alignment"],
        ontology=fixture["ontology"],
        value_domain_complete=False,
    )

    assert evidence == {}


def test_relation_domain_class_lock_cannot_be_overridden_by_llm(monkeypatch) -> None:
    fixture = _relation_context_fixture()
    other_subtype = fixture["base"] + "Variant"
    ontology = dict(fixture["ontology"])
    ontology["children_of"] = {
        fixture["base"]: [fixture["subtype"], other_subtype]
    }
    ontology["subclass_of"] = {
        **ontology["subclass_of"],
        other_subtype: [fixture["base"]],
    }
    ontology["ancestors_of"] = {
        **ontology["ancestors_of"],
        other_subtype: [fixture["base"]],
    }
    monkeypatch.setattr(
        real_value,
        "_call_llm",
        lambda _prompt: {
            "value_to_class": {
                "0": other_subtype,
                # The fallback is not authorized to change a context lock.
                "1": other_subtype,
            },
            "confidence": "high",
            "reason": "synthetic fallback",
        },
    )

    result = real_value._real_value_type_mapping(
        table_name=fixture["source_table"],
        type_col=fixture["type_column"],
        value_profiles={
            "0": [{fixture["type_column"]: 0}],
            "1": [{fixture["type_column"]: 1}],
        },
        class_candidates=[
            {"uri": fixture["subtype"], "local_name": "AuthoredRecord", "score": 0.55},
            {"uri": other_subtype, "local_name": "BaseRecordVariant", "score": 0.55},
        ],
        relation_domain_evidence={
            "1": [{
                "class_uri": fixture["subtype"],
                "evidence_source": "validated_relation_domain_context",
            }]
        },
        current_class_uri=fixture["base"],
        descendant_uris=[fixture["subtype"], other_subtype],
        ontology=ontology,
    )

    assert result["value_to_class"]["1"] == fixture["subtype"]
    assert result["value_to_class"]["0"] == other_subtype
