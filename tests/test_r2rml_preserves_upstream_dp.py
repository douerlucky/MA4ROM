"""Regression guard: R2RML serializes, but never re-selects, a DP URI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from r2rml_generator import generate_r2rml


NS = "https://synthetic.invalid/ontology#"
SELECTED_NS = "https://selected.invalid/ontology#"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


class UpstreamDatatypePropertySelectionTests(unittest.TestCase):
    def test_serializer_preserves_selected_uri_despite_name_and_domain_pressure(self):
        entity = NS + "Entity"
        incompatible_entity = NS + "IncompatibleEntity"
        selected_uri = SELECTED_NS + "upstreamSelectedProperty"
        tempting_uri = NS + "temptingProperty"

        alignment = {
            "records": {
                "pattern": "SE",
                "class_uri": entity,
                "columns": {
                    "record_id": {"role": "pk"},
                    # The column name exactly matches another ontology DP, while
                    # the selected DP has a deliberately incompatible domain.
                    # The serializer must still preserve the upstream decision.
                    "tempting_property": {
                        "role": "data_attr",
                        "prop_uri": selected_uri,
                        "confidence": "low",
                    },
                },
            }
        }
        schema = {
            "records": {
                "columns": {
                    "record_id": "integer",
                    "tempting_property": "text",
                },
                "primary_key": ["record_id"],
                "foreign_keys": [],
            }
        }
        ontology = {
            "classes": [entity, incompatible_entity],
            "object_properties": {},
            "datatype_properties": {
                selected_uri: {
                    "domain": [incompatible_entity],
                    "range": [XSD_STRING],
                },
                tempting_uri: {
                    "domain": [entity],
                    "range": [XSD_STRING],
                },
            },
            "subclass_of": {},
            "ancestors_of": {},
            "union_members": {},
            "incompatible_classes": {
                entity: [incompatible_entity],
                incompatible_entity: [entity],
            },
        }

        mapping = generate_r2rml(
            final_alignment=alignment,
            op_mapping_full={"step1": {}, "step2_orphans": {}},
            enriched_schema=schema,
            ontology=ontology,
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )

        self.assertIn(f"rr:predicate <{selected_uri}>", mapping)
        self.assertNotIn(f"rr:predicate syn:temptingProperty", mapping)
        self.assertIn('rr:column "tempting_property"', mapping)


if __name__ == "__main__":
    unittest.main()
