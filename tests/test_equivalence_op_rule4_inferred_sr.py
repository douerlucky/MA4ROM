import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
MODULE_PATH = PROJECT_ROOT / "OPMapping" / "equivalence_op_module.py"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_module_without_runtime_clients():
    """Load the pure task builder without database/provider side effects."""
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
        AssertionError("LLM must not be called by this test")
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
            "_rule4_inferred_sr_under_test", MODULE_PATH
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


EQUIV = _load_module_without_runtime_clients()
NS = "https://synthetic.invalid/ontology#"


class _Connection:
    def close(self):
        pass


def _fk(column, target, constraint, *, arity=1, target_column="id"):
    return {
        "column": column,
        "references_table": target,
        "references_column": target_column,
        "constraint_name": constraint,
        "fk_arity": arity,
    }


def _metrics(source_in_target=1.0):
    return {
        "source_distinct": 2,
        "target_distinct": 2,
        "intersection": 2 if source_in_target == 1.0 else 1,
        "union": 2,
        "manual_jaccard": source_in_target,
        "source_in_target": source_in_target,
        "target_in_source": source_in_target,
        "evidence_type": (
            "equivalence_column" if source_in_target == 1.0 else "weak_or_conflicting"
        ),
    }


def _fixture(join_info):
    left = NS + "Left"
    right = NS + "Right"
    alignment = {
        "association": {"pattern": "SE", "class_uri": NS + "Association"},
        "left_entity": {"pattern": "SE", "class_uri": left},
        "right_entity": {"pattern": "SE", "class_uri": right},
    }
    schema = {
        "association": join_info,
        "left_entity": {
            "columns": {"id": "integer"},
            "primary_key": ["id"],
            "foreign_keys": [],
        },
        "right_entity": {
            "columns": {"id": "integer"},
            "primary_key": ["id"],
            "foreign_keys": [],
        },
    }
    ontology = {
        "classes": [left, right, NS + "Association"],
        "ancestors_of": {},
        "subclass_of": {},
        "object_properties": {},
    }
    return alignment, schema, ontology


def _build(join_info, overlap):
    alignment, schema, ontology = _fixture(join_info)
    with (
        patch.object(EQUIV, "get_connection", return_value=_Connection()),
        patch.object(EQUIV, "_overlap_metrics", side_effect=overlap) as overlap_mock,
    ):
        tasks, evidence = EQUIV._build_sr_tasks(
            alignment, schema, ontology, "public"
        )
    return tasks, evidence, overlap_mock


class Rule4InferredSRTests(unittest.TestCase):
    def test_name_only_object_property_is_not_retrieved(self):
        left = NS + "Left"
        right = NS + "Right"
        unrelated = NS + "Unrelated"
        task = {
            "name_hint": "right",
            "source_table": "left_entity",
            "source_column": "right_id",
            "target_table": "right_entity",
            "target_column": "id",
            "domain_class_uri": left,
            "range_class_uri": right,
            "schema_matching": [],
        }
        ontology = {
            "object_properties": {
                NS + "right": {
                    "domain": [unrelated],
                    "range": [unrelated],
                }
            },
            "classes": [left, right, unrelated],
            "ancestors_of": {},
            "subclass_of": {},
            "union_members": {},
            "incompatible_classes": {},
        }
        assert EQUIV._filter_endpoint_candidates(task, ontology, {"ops": {}}) == []

    def test_one_coarse_endpoint_can_reach_llm_with_strong_fk_role_evidence(self):
        source = NS + "CoarseSource"
        target = NS + "Target"
        declared_domain = NS + "UnmodeledDomain"
        prop = NS + "hasTarget"
        task = {
            "name_hint": "hasTarget target",
            "source_table": "source",
            "source_column": "target_id",
            "target_table": "target",
            "target_column": "id",
            "domain_class_uri": source,
            "range_class_uri": target,
            "schema_matching": [
                {
                    "source_table": "source",
                    "source_column": "target_id",
                    "target_table": "target",
                    "target_column": "id",
                    "constraint_name": "source_target_fk",
                    "source_in_target": 1.0,
                    "evidence_type": "equivalence_column",
                }
            ],
        }
        ontology = {
            "object_properties": {
                prop: {"domain": [declared_domain], "range": [target]}
            },
            "classes": [source, target, declared_domain],
            "ancestors_of": {},
            "subclass_of": {},
            "union_members": {},
            "incompatible_classes": {},
        }
        candidates = EQUIV._filter_endpoint_candidates(task, ontology, {"ops": {}})
        self.assertEqual([row["uri"] for row in candidates], [prop])
        self.assertTrue(candidates[0]["weak_endpoint_match"])
        self.assertEqual(
            candidates[0]["endpoint_evidence_tier"],
            "one_endpoint_plus_strong_fk_role",
        )

    def test_two_complete_scalar_pk_fk_endpoints_with_ind_create_one_task(self):
        join_info = {
            "columns": {"left_id": "integer", "right_id": "integer"},
            "primary_key": ["left_id", "right_id"],
            "foreign_keys": [
                _fk("right_id", "right_entity", "fk_right"),
                _fk("left_id", "left_entity", "fk_left"),
            ],
        }
        def overlap_at_boundary(_conn, _source_table, source_column, *_args):
            return _metrics(0.95 if source_column == "right_id" else 1.0)

        tasks, evidence, overlap = _build(join_info, overlap_at_boundary)

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["task_type"], "sr_relation_inferred")
        self.assertEqual(task["key"], "association::left_id__right_id")
        self.assertEqual(
            [row["source_column"] for row in task["schema_matching"]],
            ["left_id", "right_id"],
        )
        self.assertEqual(len(evidence), 2)
        self.assertEqual(overlap.call_count, 2)

    def test_non_pk_foreign_keys_on_entity_table_do_not_form_relation(self):
        join_info = {
            "columns": {
                "id": "integer",
                "left_id": "integer",
                "right_id": "integer",
            },
            "primary_key": ["id"],
            "foreign_keys": [
                _fk("left_id", "left_entity", "fk_left"),
                _fk("right_id", "right_entity", "fk_right"),
            ],
        }
        tasks, evidence, overlap = _build(join_info, lambda *_args, **_kwargs: _metrics())

        self.assertEqual(tasks, [])
        self.assertEqual(evidence, [])
        overlap.assert_not_called()

    def test_three_pk_fk_endpoint_constraints_abstain_as_nary(self):
        join_info = {
            "columns": {
                "left_id": "integer",
                "right_id": "integer",
                "third_id": "integer",
            },
            "primary_key": ["left_id", "right_id", "third_id"],
            "foreign_keys": [
                _fk("left_id", "left_entity", "fk_left"),
                _fk("right_id", "right_entity", "fk_right"),
                _fk("third_id", "left_entity", "fk_third"),
            ],
        }
        tasks, evidence, overlap = _build(join_info, lambda *_args, **_kwargs: _metrics())

        self.assertEqual(tasks, [])
        self.assertEqual(evidence, [])
        overlap.assert_not_called()

    def test_two_rows_of_same_composite_constraint_are_not_paired(self):
        join_info = {
            "columns": {
                "left_a": "integer",
                "left_b": "integer",
                "right_id": "integer",
            },
            "primary_key": ["left_a", "left_b", "right_id"],
            "foreign_keys": [
                _fk("left_a", "left_entity", "fk_left_composite", arity=2),
                _fk(
                    "left_b",
                    "left_entity",
                    "fk_left_composite",
                    arity=2,
                    target_column="version",
                ),
                _fk("right_id", "right_entity", "fk_right"),
            ],
        }
        tasks, evidence, overlap = _build(join_info, lambda *_args, **_kwargs: _metrics())

        self.assertEqual(tasks, [])
        self.assertEqual(evidence, [])
        overlap.assert_not_called()

    def test_incomplete_composite_constraint_abstains(self):
        join_info = {
            "columns": {"left_a": "integer", "right_id": "integer"},
            "primary_key": ["left_a", "right_id"],
            "foreign_keys": [
                _fk("left_a", "left_entity", "fk_left_composite", arity=2),
                _fk("right_id", "right_entity", "fk_right"),
            ],
        }
        tasks, evidence, overlap = _build(join_info, lambda *_args, **_kwargs: _metrics())

        self.assertEqual(tasks, [])
        self.assertEqual(evidence, [])
        overlap.assert_not_called()

    def test_both_endpoints_must_pass_source_in_target_threshold(self):
        join_info = {
            "columns": {"left_id": "integer", "right_id": "integer"},
            "primary_key": ["left_id", "right_id"],
            "foreign_keys": [
                _fk("left_id", "left_entity", "fk_left"),
                _fk("right_id", "right_entity", "fk_right"),
            ],
        }

        def overlap(_conn, _source_table, source_column, *_args):
            return _metrics(0.94 if source_column == "right_id" else 1.0)

        tasks, evidence, overlap_mock = _build(join_info, overlap)

        self.assertEqual(tasks, [])
        self.assertEqual(len(evidence), 2)
        self.assertEqual(overlap_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
