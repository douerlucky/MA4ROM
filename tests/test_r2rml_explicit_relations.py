import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r2rml_generator import generate_r2rml, generate_sr_mapping


NS = "https://synthetic.invalid/ontology#"


def uri(local_name: str) -> str:
    return NS + local_name


def ontology() -> dict:
    return {
        "classes": [
            uri("ChildEntity"),
            uri("ParentEntity"),
            uri("Entity"),
            uri("LeftEntity"),
            uri("RightEntity"),
        ],
        "object_properties": {
            uri("linksToParent"): {
                "domain": [uri("ChildEntity")],
                "range": [uri("ParentEntity")],
            },
            uri("relatesEndpoints"): {
                "domain": [uri("LeftEntity")],
                "range": [uri("RightEntity")],
            },
        },
        "datatype_properties": {},
    }


def schema() -> dict:
    return {
        "child_rows": {
            "columns": {"child_id": "integer", "kind": "text"},
            "primary_key": ["child_id"],
            "foreign_keys": [
                {
                    "column": "child_id",
                    "references_table": "parent_rows",
                    "references_column": "parent_id",
                }
            ],
        },
        "entity_rows": {
            "columns": {
                "entity_id": "integer",
                "left_id": "integer",
                "right_id": "integer",
            },
            "primary_key": ["entity_id"],
            "foreign_keys": [],
        },
        "parent_rows": {
            "columns": {"parent_id": "integer"},
            "primary_key": ["parent_id"],
            "foreign_keys": [],
        },
        "left_rows": {
            "columns": {"left_id": "integer"},
            "primary_key": ["left_id"],
            "foreign_keys": [],
        },
        "right_rows": {
            "columns": {"right_id": "integer"},
            "primary_key": ["right_id"],
            "foreign_keys": [],
        },
    }


def alignment() -> dict:
    return {
        "child_rows": {
            "pattern": "SH",
            "sub_class_uri": uri("ChildEntity"),
            "columns": {
                "child_id": {"role": "sh_inherited_pk"},
                "kind": {"role": "discriminator", "prop_uri": None},
            },
        },
        "entity_rows": {
            "pattern": "SE",
            "class_uri": uri("Entity"),
            "columns": {"entity_id": {"role": "pk"}},
        },
        "parent_rows": {
            "pattern": "SE",
            "class_uri": uri("ParentEntity"),
            "columns": {"parent_id": {"role": "pk"}},
        },
        "left_rows": {
            "pattern": "SE",
            "class_uri": uri("LeftEntity"),
            "columns": {"left_id": {"role": "pk"}},
        },
        "right_rows": {
            "pattern": "SE",
            "class_uri": uri("RightEntity"),
            "columns": {"right_id": {"role": "pk"}},
        },
    }


def inferred_relation(*, include_object_ref=True, mixed_source=False) -> dict:
    return {
        "object_prop_uri": uri("relatesEndpoints"),
        "scenario_type": "sr_relation_inferred",
        "sr_subject_column": "left_id",
        "sr_object_column": "right_id",
        "sr_subject_ref_table": "left_rows",
        "sr_object_ref_table": "right_rows" if include_object_ref else None,
        "schema_matching": [
            {
                "source_table": "entity_rows",
                "source_column": "left_id",
                "target_table": "left_rows",
            },
            {
                "source_table": "other_rows" if mixed_source else "entity_rows",
                "source_column": "right_id",
                "target_table": "right_rows",
            },
        ],
    }


class ExplicitRelationSerializationTests(unittest.TestCase):
    def test_partial_sr_rejects_missing_explicit_reference_endpoint(self):
        entry = {
            "pattern": "SR",
            "relation_kind": "partial_fk",
            "fk1": {
                "column": "left_id",
                "ref_table": "left_rows",
            },
            "fk2": {
                "column": "right_id",
                "ref_table": None,
            },
            "domain_class_uri": uri("LeftEntity"),
            "range_class_uri": uri("RightEntity"),
        }
        selected = {
            "bridge_rows": {
                "object_prop_uri": uri("relatesEndpoints"),
                "sr_subject_column": "left_id",
                "sr_object_column": "right_id",
                "sr_subject_ref_table": "left_rows",
                "sr_object_ref_table": None,
            }
        }

        mapping = generate_sr_mapping(
            "bridge_rows",
            entry,
            {"bridge_rows": {"columns": {"left_id": "integer", "right_id": "integer"}}},
            selected,
            "https://synthetic.invalid/data/",
            {"syn": NS},
            {"left_rows": "left_rows", "right_rows": "right_rows"},
            {
                uri("LeftEntity"): "left_rows",
                uri("RightEntity"): "right_rows",
            },
        )

        self.assertIn("FK 引用表信息不完整", mapping)
        self.assertNotIn("syn:relatesEndpoints", mapping)

    def test_sh_inherited_pk_fk_uses_explicit_step1_object_property(self):
        mapping = generate_r2rml(
            final_alignment=alignment(),
            op_mapping_full={
                "step1": {
                    "child_rows.child_id": {
                        "object_prop_uri": uri("linksToParent"),
                        "scenario_type": "fk_obj",
                    }
                },
                "step2_orphans": {},
            },
            enriched_schema=schema(),
            ontology=ontology(),
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )

        self.assertIn("syn:linksToParent", mapping)
        self.assertIn(
            'rr:template "https://synthetic.invalid/data/parent_rows/{child_id}"',
            mapping,
        )

    def test_entity_table_serializes_fully_evidenced_inferred_relation(self):
        mapping = generate_r2rml(
            final_alignment=alignment(),
            op_mapping_full={
                "step1": {"opaque-contract-id": inferred_relation()},
                "step2_orphans": {},
            },
            enriched_schema=schema(),
            ontology=ontology(),
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )

        self.assertIn("syn:relatesEndpoints", mapping)
        self.assertIn(
            'SELECT \\"left_id\\" , \\"right_id\\" FROM \\"entity_rows\\"',
            mapping,
        )
        self.assertIn(
            'rr:template "https://synthetic.invalid/data/left_rows/{left_id}"',
            mapping,
        )
        self.assertIn(
            'rr:template "https://synthetic.invalid/data/right_rows/{right_id}"',
            mapping,
        )

    def test_inferred_relation_rejects_missing_or_cross_table_evidence(self):
        for selected in (
            inferred_relation(include_object_ref=False),
            inferred_relation(mixed_source=True),
        ):
            with self.subTest(selected=selected):
                mapping = generate_r2rml(
                    final_alignment=alignment(),
                    op_mapping_full={
                        "step1": {"opaque-contract-id": selected},
                        "step2_orphans": {},
                    },
                    enriched_schema=schema(),
                    ontology=ontology(),
                    base_url="https://synthetic.invalid/data/",
                    prefix="syn",
                )
                self.assertNotIn("syn:relatesEndpoints", mapping)

    def test_sh_entity_and_type_maps_share_validated_inherited_identity(self):
        document = uri("Document")
        paper = uri("Paper")
        paper_variant = uri("PaperVariant")
        reviewer = uri("Reviewer")
        aligned = {
            "document_rows": {
                "pattern": "SE",
                "class_uri": document,
                "columns": {"id": {"role": "pk"}},
            },
            "paper_rows": {
                "pattern": "SH",
                "sub_class_uri": paper,
                "parent_class_uri": document,
                "parent_table": "document_rows",
                "columns": {
                    "id": {"role": "sh_inherited_pk"},
                    "reviewer_id": {"role": "fk_obj"},
                    "kind": {"role": "discriminator"},
                },
                "type_assertions": [{
                    "kind": "enum",
                    "column": "kind",
                    "value_to_class": {"draft": paper_variant},
                }],
            },
            "reviewer_rows": {
                "pattern": "SE",
                "class_uri": reviewer,
                "columns": {"reviewer_id": {"role": "pk"}},
            },
        }
        schema = {
            "document_rows": {
                "columns": {"id": "integer"},
                "primary_key": ["id"],
                "foreign_keys": [],
            },
            "paper_rows": {
                "columns": {
                    "id": "integer",
                    "reviewer_id": "integer",
                    "kind": "text",
                },
                "primary_key": ["id", "reviewer_id"],
                "foreign_keys": [
                    {
                        "column": "id",
                        "references_table": "document_rows",
                        "references_column": "id",
                    },
                    {
                        "column": "reviewer_id",
                        "references_table": "reviewer_rows",
                        "references_column": "reviewer_id",
                    },
                ],
            },
            "reviewer_rows": {
                "columns": {"reviewer_id": "integer"},
                "primary_key": ["reviewer_id"],
                "foreign_keys": [],
            },
        }
        ontology_data = {
            "classes": [document, paper, paper_variant, reviewer],
            "children_of": {document: [paper], paper: [paper_variant]},
            "subclass_of": {paper: [document], paper_variant: [paper]},
            "ancestors_of": {
                paper: [document],
                paper_variant: [paper, document],
            },
            "object_properties": {},
            "datatype_properties": {},
        }

        mapping = generate_r2rml(
            final_alignment=aligned,
            op_mapping_full={"step1": {}, "step2_orphans": {}},
            enriched_schema=schema,
            ontology=ontology_data,
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )

        paper_section = mapping.split("# === paper_rows (SH) ===", 1)[1]
        type_section = mapping.split("# === paper_rows (Type Assertions) ===", 1)[1]
        self.assertIn(
            'rr:template "https://synthetic.invalid/data/document_rows/{id}"',
            paper_section,
        )
        self.assertIn(
            'rr:template "https://synthetic.invalid/data/document_rows/{id}"',
            type_section,
        )
        self.assertNotIn("{id}__{reviewer_id}", paper_section)
        self.assertNotIn("{id}__{reviewer_id}", type_section)


if __name__ == "__main__":
    unittest.main()
