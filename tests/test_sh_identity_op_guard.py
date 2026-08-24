"""Regression coverage for SH identity/relationship separation."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
MODULE_PATH = PROJECT_ROOT / "OPMapping" / "equivalence_op_module.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r2rml_generator import generate_r2rml


def _load_equiv_without_runtime_clients():
    saved = {
        name: sys.modules.get(name)
        for name in ("config", "utils.db_utils", "utils.llm_client")
    }
    config = types.ModuleType("config")
    config.DB_SCHEMA_NAME = "public"
    config.ONTOLOGY_PATH = "unused.ttl"
    config.OUTPUT_DIR = "unused-output"
    db_utils = types.ModuleType("utils.db_utils")
    db_utils.get_connection = lambda: None
    llm_client = types.ModuleType("utils.llm_client")
    llm_client.call_llm = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("this regression test must not call an external LLM")
    )
    sys.modules.update(
        {
            "config": config,
            "utils.db_utils": db_utils,
            "utils.llm_client": llm_client,
        }
    )
    try:
        spec = importlib.util.spec_from_file_location(
            "_sh_identity_op_guard_under_test", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


EQUIV = _load_equiv_without_runtime_clients()
NS = "https://synthetic.invalid/ontology#"


def u(local: str) -> str:
    return NS + local


class _Connection:
    def close(self):
        pass


def _metrics(*_args, **_kwargs):
    return {
        "source_distinct": 2,
        "target_distinct": 2,
        "intersection": 2,
        "union": 2,
        "manual_jaccard": 1.0,
        "source_in_target": 1.0,
        "target_in_source": 1.0,
        "evidence_type": "equivalence_column",
    }


def _ontology() -> dict:
    return {
        "classes": [u("Parent"), u("Child"), u("Owner")],
        "subclass_of": {u("Child"): [u("Parent")]},
        "ancestors_of": {u("Child"): [u("Parent")]},
        "object_properties": {
            u("linksToParent"): {
                "domain": [u("Child")],
                "range": [u("Parent")],
            },
            u("hasOwner"): {
                "domain": [u("Child")],
                "range": [u("Owner")],
            },
        },
        "datatype_properties": {},
    }


def _scalar_fixture() -> tuple[dict, dict]:
    alignment = {
        "parent_rows": {
            "pattern": "SE",
            "class_uri": u("Parent"),
            "columns": {"id": {"role": "pk"}},
        },
        "child_rows": {
            "pattern": "SH",
            "sub_class_uri": u("Child"),
            "parent_class_uri": u("Parent"),
            "parent_table": "parent_rows",
            "columns": {
                "id": {"role": "sh_inherited_pk"},
                "owner_id": {"role": "fk_obj", "ref_table": "owner_rows"},
            },
        },
        "owner_rows": {
            "pattern": "SE",
            "class_uri": u("Owner"),
            "columns": {"id": {"role": "pk"}},
        },
    }
    schema = {
        "parent_rows": {
            "columns": {"id": "integer"},
            "primary_key": ["id"],
            "foreign_keys": [],
        },
        "child_rows": {
            "columns": {"id": "integer", "owner_id": "integer"},
            "primary_key": ["id"],
            "foreign_keys": [
                {
                    "constraint_name": "child_identity_fk",
                    "fk_arity": 1,
                    "column": "id",
                    "references_table": "parent_rows",
                    "references_column": "id",
                },
                {
                    "constraint_name": "child_owner_fk",
                    "fk_arity": 1,
                    "column": "owner_id",
                    "references_table": "owner_rows",
                    "references_column": "id",
                },
            ],
        },
        "owner_rows": {
            "columns": {"id": "integer"},
            "primary_key": ["id"],
            "foreign_keys": [],
        },
    }
    return alignment, schema


def test_equiv_op_filters_only_validated_scalar_identity_fk() -> None:
    alignment, schema = _scalar_fixture()
    with (
        patch.object(EQUIV, "get_connection", return_value=_Connection()),
        patch.object(EQUIV, "_overlap_metrics", side_effect=_metrics) as overlap,
    ):
        tasks, evidence = EQUIV._build_fk_tasks(
            schema, alignment, _ontology(), "public"
        )

    assert [task["key"] for task in tasks] == ["child_rows.owner_id"]
    assert [row["source_column"] for row in evidence] == ["owner_id"]
    assert overlap.call_count == 1


def test_r2rml_rejects_stale_identity_op_but_keeps_real_fk() -> None:
    alignment, schema = _scalar_fixture()
    mapping = generate_r2rml(
        final_alignment=alignment,
        op_mapping_full={
            "step1": {
                "child_rows.id": {
                    "object_prop_uri": u("linksToParent"),
                    "scenario_type": "fk_obj",
                },
                "child_rows.owner_id": {
                    "object_prop_uri": u("hasOwner"),
                    "scenario_type": "fk_obj",
                },
            },
            "step2_orphans": {},
        },
        enriched_schema=schema,
        ontology=_ontology(),
        base_url="https://synthetic.invalid/data/",
        prefix="syn",
    )

    child = mapping.split("# === child_rows (SH) ===", 1)[1].split(
        "# === parent_rows", 1
    )[0]
    assert "syn:linksToParent" not in child
    assert "syn:hasOwner" in child
    assert (
        'rr:template "https://synthetic.invalid/data/parent_rows/{id}"'
        in child
    )
    assert (
        'rr:template "https://synthetic.invalid/data/owner_rows/{owner_id}"'
        in child
    )


def test_complete_composite_identity_fk_is_filtered_before_op() -> None:
    alignment, schema = _scalar_fixture()
    alignment["parent_rows"]["columns"] = {
        "id_a": {"role": "pk"},
        "id_b": {"role": "pk"},
    }
    alignment["child_rows"]["columns"] = {
        "child_a": {"role": "sh_inherited_pk"},
        "child_b": {"role": "sh_inherited_pk"},
        "owner_id": {"role": "fk_obj", "ref_table": "owner_rows"},
    }
    schema["parent_rows"] = {
        "columns": {"id_a": "integer", "id_b": "integer"},
        "primary_key": ["id_a", "id_b"],
        "foreign_keys": [],
    }
    schema["child_rows"] = {
        "columns": {
            "child_a": "integer",
            "child_b": "integer",
            "owner_id": "integer",
        },
        "primary_key": ["child_a", "child_b"],
        "foreign_keys": [
            {
                "constraint_name": "child_identity_fk",
                "fk_arity": 2,
                "column": "child_a",
                "references_table": "parent_rows",
                "references_column": "id_a",
            },
            {
                "constraint_name": "child_identity_fk",
                "fk_arity": 2,
                "column": "child_b",
                "references_table": "parent_rows",
                "references_column": "id_b",
            },
            {
                "constraint_name": "child_owner_fk",
                "fk_arity": 1,
                "column": "owner_id",
                "references_table": "owner_rows",
                "references_column": "id",
            },
        ],
    }

    with (
        patch.object(EQUIV, "get_connection", return_value=_Connection()),
        patch.object(EQUIV, "_overlap_metrics", side_effect=_metrics) as overlap,
    ):
        tasks, _evidence = EQUIV._build_fk_tasks(
            schema, alignment, _ontology(), "public"
        )

    assert [task["key"] for task in tasks] == ["child_rows.owner_id"]
    assert overlap.call_count == 1

    mapping = generate_r2rml(
        final_alignment=alignment,
        op_mapping_full={
            "step1": {
                "child_rows.__fk__child_identity_fk": {
                    "object_prop_uri": u("linksToParent"),
                    "scenario_type": "fk_obj_composite",
                },
                "child_rows.owner_id": {
                    "object_prop_uri": u("hasOwner"),
                    "scenario_type": "fk_obj",
                },
            },
            "step2_orphans": {},
        },
        enriched_schema=schema,
        ontology=_ontology(),
        base_url="https://synthetic.invalid/data/",
        prefix="syn",
    )
    child = mapping.split("# === child_rows (SH) ===", 1)[1].split(
        "# === parent_rows", 1
    )[0]
    assert "syn:linksToParent" not in child
    assert "syn:hasOwner" in child
