"""Synthetic guards for identity columns with independent RDF semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r2rml_generator import generate_r2rml


NS = "https://synthetic.invalid/ontology#"
ALT_NS = "https://alternate.invalid/ontology#"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


def uri(local_name: str, namespace: str = NS) -> str:
    return namespace + local_name


def ontology_with(*datatype_properties: tuple[str, list[str]]) -> dict:
    entity = uri("Entity")
    target = uri("Target")
    other = uri("Other")
    return {
        "classes": [entity, target, other],
        "object_properties": {
            uri("linksToTarget"): {
                "domain": [entity],
                "range": [target],
            }
        },
        "datatype_properties": {
            prop_uri: {"domain": domains, "range": [XSD_STRING]}
            for prop_uri, domains in datatype_properties
        },
        "subclass_of": {},
        "ancestors_of": {},
        "union_members": {},
        "incompatible_classes": {
            entity: [other],
            other: [entity],
            target: [],
        },
    }


def target_alignment() -> dict:
    return {
        "pattern": "SE",
        "class_uri": uri("Target"),
        "columns": {"target_id": {"role": "pk"}},
    }


def target_schema() -> dict:
    return {
        "columns": {"target_id": "integer"},
        "primary_key": ["target_id"],
        "foreign_keys": [],
    }


class PrimaryKeyDualSemanticTests(unittest.TestCase):
    def test_composite_identity_keeps_explicit_fk_op_and_non_fk_name_dp(self):
        alignment = {
            "records": {
                "pattern": "SE",
                "class_uri": uri("Entity"),
                # Legacy artifacts can still say role=pk for the PK+FK column;
                # physical schema evidence must preserve the selected OP.
                "columns": {
                    "target_id": {"role": "pk"},
                    "name": {
                        "role": "pk",
                        "data_prop_uri": uri("name"),
                        "data_prop_confidence": "high",
                    },
                },
            },
            "targets": target_alignment(),
        }
        schema = {
            "records": {
                "columns": {"target_id": "integer", "name": "text"},
                "primary_key": ["target_id", "name"],
                "foreign_keys": [
                    {
                        "column": "target_id",
                        "references_table": "targets",
                        "references_column": "target_id",
                    }
                ],
            },
            "targets": target_schema(),
        }
        ontology = ontology_with(
            (uri("name"), [uri("Entity")]),
            # This tempting exact DP must never steal a physical FK column.
            (uri("target_id"), [uri("Entity")]),
        )

        mapping = generate_r2rml(
            final_alignment=alignment,
            op_mapping_full={
                "step1": {
                    "records.target_id": {
                        "object_prop_uri": uri("linksToTarget"),
                    }
                },
                "step2_orphans": {},
            },
            enriched_schema=schema,
            ontology=ontology,
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )

        records = mapping.split("# === records (SE) ===", 1)[1].split(
            "# === targets (SE) ===", 1
        )[0]
        self.assertIn(
            'rr:template "https://synthetic.invalid/data/records/{target_id}__{name}"',
            records,
        )
        self.assertIn("syn:linksToTarget", records)
        self.assertIn("syn:name", records)
        self.assertNotIn("syn:target_id", records)

    def test_serializer_requires_an_explicit_identity_dp_decision(self):
        schema = {
            "records": {
                "columns": {"name": "text"},
                "primary_key": ["name"],
                "foreign_keys": [],
            }
        }
        alignment = {
            "records": {
                "pattern": "SE",
                "class_uri": uri("Entity"),
                "columns": {
                    "name": {
                        "role": "pk",
                        "data_prop_uri": uri("name"),
                        "data_prop_confidence": "high",
                    }
                },
            }
        }

        unique_mapping = generate_r2rml(
            final_alignment=alignment,
            op_mapping_full={"step1": {}, "step2_orphans": {}},
            enriched_schema=schema,
            # A blank-node domain models an anonymous intersection/complement
            # such as those used by geospatial ontologies.
            ontology=ontology_with((uri("name"), ["anonymous-domain"])),
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )
        self.assertIn("syn:name", unique_mapping)

        no_decision_mapping = generate_r2rml(
            final_alignment={
                "records": {
                    "pattern": "SE",
                    "class_uri": uri("Entity"),
                    "columns": {"name": {"role": "pk"}},
                }
            },
            op_mapping_full={"step1": {}, "step2_orphans": {}},
            enriched_schema=schema,
            ontology=ontology_with(
                (uri("name"), [uri("Entity")]),
                (uri("name", ALT_NS), [uri("Entity")]),
            ),
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )
        self.assertNotIn("rr:predicate syn:name", no_decision_mapping)
        self.assertNotIn(f"<{uri('name', ALT_NS)}>", no_decision_mapping)

    def test_exact_name_with_incompatible_domain_is_rejected(self):
        mapping = generate_r2rml(
            final_alignment={
                "records": {
                    "pattern": "SE",
                    "class_uri": uri("Entity"),
                    "columns": {"name": {"role": "pk"}},
                }
            },
            op_mapping_full={"step1": {}, "step2_orphans": {}},
            enriched_schema={
                "records": {
                    "columns": {"name": "text"},
                    "primary_key": ["name"],
                    "foreign_keys": [],
                }
            },
            ontology=ontology_with((uri("name"), [uri("Other")])),
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )

        self.assertNotIn("rr:predicate syn:name", mapping)


if __name__ == "__main__":
    unittest.main()
