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
    """Load pure helpers without importing database or provider clients."""
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
            "_partial_sr_value_attr_under_test", MODULE_PATH
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


def uri(local: str) -> str:
    return NS + local


class _Connection:
    def close(self):
        pass


def _overlap(*_args, **_kwargs):
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


def _fixture():
    owner = uri("Owner")
    target_a = uri("TargetA")
    target_b = uri("TargetB")
    relation = uri("linksTo")
    final_alignment = {
        "entity_links": {
            "pattern": "SE",
            "table_kind": "value_attr",
            "class_uri": owner,
            "fk": {"column": "owner_id", "ref_table": "entities"},
            "value_column": "target_iri",
            "columns": {},
        },
        "entities": {"pattern": "SE", "class_uri": owner},
    }
    enriched_schema = {
        "entity_links": {
            "columns": {"owner_id": "integer", "target_iri": "text"},
            "primary_key": ["owner_id", "target_iri"],
            "foreign_keys": [
                {
                    "column": "owner_id",
                    "references_table": "entities",
                    "references_column": "id",
                    "constraint_name": "fk_owner",
                }
            ],
        },
        "entities": {
            "columns": {"id": "integer"},
            "primary_key": ["id"],
            "foreign_keys": [],
        },
    }
    ontology = {
        "classes": [owner, target_a, target_b],
        "ancestors_of": {},
        "subclass_of": {},
        "union_members": {},
        "incompatible_classes": {},
        "object_properties": {
            relation: {"domain": [], "range": []},
        },
    }
    restriction = {
        "class": owner,
        "class_local": "Owner",
        "class_tables": ["entities"],
        "values": [target_a, target_b],
        "values_local": ["TargetA", "TargetB"],
    }
    metadata = {
        "ops": {
            relation: {
                "local_name": "linksTo",
                "domain": [],
                "range": [],
                "construction": [],
                "inverse_of": [],
                "subproperty_of": [],
                "restrictions": [restriction],
            }
        }
    }
    return final_alignment, enriched_schema, ontology, metadata, relation


class PartialValueAttributeTests(unittest.TestCase):
    def _build_task(self, iri: bool):
        final_alignment, enriched_schema, ontology, metadata, relation = _fixture()
        profile = {
            "partial_value_term_type": "iri" if iri else None,
            "partial_value_iri_ratio": 1.0 if iri else 0.0,
            "partial_value_observed_distinct": 2,
        }
        with (
            patch.object(EQUIV, "get_connection", return_value=_Connection()),
            patch.object(EQUIV, "_overlap_metrics", side_effect=_overlap),
            patch.object(EQUIV, "_partial_value_iri_profile", return_value=profile),
        ):
            tasks, _ = EQUIV._build_sr_tasks(
                final_alignment, enriched_schema, ontology, "public"
            )
        self.assertEqual(len(tasks), 1)
        return (
            tasks[0],
            final_alignment,
            enriched_schema,
            ontology,
            metadata,
            relation,
        )

    def test_restriction_only_polymorphic_iri_endpoint_is_persisted(self):
        task, alignment, schema, ontology, metadata, relation = self._build_task(True)
        candidates = EQUIV._partial_sr_endpoint_candidates(
            task, ontology, metadata, alignment, schema
        )

        self.assertEqual([candidate["uri"] for candidate in candidates], [relation])
        selected = candidates[0]
        self.assertEqual(selected["partial_endpoint_evidence"], "restriction")
        self.assertEqual(selected["partial_value_term_type"], "iri")
        self.assertEqual(
            selected["partial_counterpart_class_uris"],
            [uri("TargetA"), uri("TargetB")],
        )
        self.assertEqual(selected["partial_sr_subject_column"], "owner_id")
        self.assertEqual(selected["partial_sr_object_column"], "target_iri")
        self.assertEqual(selected["partial_sr_subject_ref_table"], "entities")
        self.assertEqual(selected["partial_sr_object_ref_table"], "")

        contract = EQUIV._selected_sr_endpoints(
            task, selected, selected["sr_direction"]
        )
        self.assertEqual(contract["partial_value_column"], "target_iri")
        self.assertEqual(contract["partial_value_term_type"], "iri")
        self.assertEqual(contract["sr_subject_column"], "owner_id")
        self.assertEqual(contract["sr_object_column"], "target_iri")

    def test_lexical_match_without_endpoint_evidence_is_rejected(self):
        task, alignment, schema, ontology, metadata, relation = self._build_task(True)
        metadata["ops"][relation]["restrictions"] = []

        self.assertEqual(
            EQUIV._partial_sr_endpoint_candidates(
                task, ontology, metadata, alignment, schema
            ),
            [],
        )

    def test_literal_value_without_unique_entity_key_is_not_an_object(self):
        task, alignment, schema, ontology, metadata, _ = self._build_task(False)

        self.assertIsNone(task["partial_value_term_type"])
        self.assertEqual(
            EQUIV._partial_sr_endpoint_candidates(
                task, ontology, metadata, alignment, schema
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
