"""Guards for explicit semantic identifiers and constraint-level FK output."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from candidate_generation import _handle_SE
from DPMapping.data_property_mapping_agent import (
    _match_SE,
    _precompute_confidence,
    collect_low_confidence_data_property_mappings,
)
from r2rml_generator import generate_r2rml


NS = "https://synthetic.invalid/ontology#"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"


def uri(local: str) -> str:
    return NS + local


def ontology() -> dict:
    country = uri("Country")
    city = uri("City")
    return {
        "classes": [country, city],
        "object_properties": {
            uri("capital"): {"domain": [country], "range": [city]},
            uri("locatedIn"): {"domain": [city], "range": [country]},
        },
        "datatype_properties": {
            uri("countryCode"): {"domain": [country], "range": [XSD_STRING]},
            uri("carCode"): {"domain": [country], "range": [XSD_STRING]},
            uri("otherCode"): {"domain": [country], "range": [XSD_STRING]},
        },
        "subclass_of": {},
        "ancestors_of": {},
        "union_members": {},
        "incompatible_classes": {country: [], city: []},
    }


def composite_schema() -> dict:
    return {
        "country": {
            "columns": {
                "code": "text",
                "capital_name": "text",
                "capital_province": "text",
            },
            "primary_key": ["code"],
            "foreign_keys": [
                {
                    "constraint_name": "country_capital_fk",
                    "fk_arity": 3,
                    "column": "capital_name",
                    "references_table": "city",
                    "references_column": "name",
                },
                {
                    "constraint_name": "country_capital_fk",
                    "fk_arity": 3,
                    "column": "code",
                    "references_table": "city",
                    "references_column": "country",
                },
                {
                    "constraint_name": "country_capital_fk",
                    "fk_arity": 3,
                    "column": "capital_province",
                    "references_table": "city",
                    "references_column": "province",
                },
            ],
        },
        "city": {
            "columns": {"name": "text", "country": "text", "province": "text"},
            "primary_key": ["name", "country", "province"],
            "foreign_keys": [],
        },
    }


class SemanticIdentityTests(unittest.TestCase):
    def test_inclusive_paper_threshold_is_not_lost_to_float_rounding(self):
        candidates = [
            {"uri": uri("carCode"), "score": 0.7},
            {"uri": uri("otherCode"), "score": 0.5},
        ]
        self.assertEqual(_precompute_confidence(candidates), "high")

    def test_pk_fk_keeps_independent_dp_and_op_candidate_tracks(self):
        ont = ontology()
        result = _handle_SE(
            "country",
            {"code": "text"},
            {"code"},
            {
                "code": {
                    "ref_table": "city",
                    "ref_col": "country",
                    "constraint_name": "country_capital_fk",
                    "fk_arity": 3,
                }
            },
            ont["classes"],
            ont["object_properties"],
            ont["datatype_properties"],
            ont,
        )
        column = result["columns"]["code"]
        self.assertEqual(column["role"], "fk_obj")
        self.assertTrue(column["identity_part"])
        self.assertEqual(column["constraint_name"], "country_capital_fk")
        self.assertEqual(column["fk_arity"], 3)
        self.assertTrue(column["candidates"])
        self.assertEqual(column["dp_candidates"][0]["uri"], uri("countryCode"))
        self.assertTrue(
            column["dp_candidates"][0]["identity_evidence"]["unique"]
        )

    def test_generic_id_does_not_enter_unrelated_dp_track(self):
        person = uri("Person")
        result = _handle_SE(
            "Person",
            {"ID": "integer"},
            {"ID"},
            {},
            [person],
            {},
            {
                uri("email"): {"domain": [person], "range": [XSD_STRING]},
                uri("siteURL"): {"domain": [person], "range": [XSD_STRING]},
            },
            {
                "classes": [person],
                "datatype_properties": {
                    uri("email"): {"domain": [person], "range": [XSD_STRING]},
                    uri("siteURL"): {"domain": [person], "range": [XSD_STRING]},
                },
                "ancestors_of": {},
            },
        )

        column = result["columns"]["ID"]
        self.assertEqual(column["role"], "pk")
        self.assertEqual(column["candidates"], [])

    def test_identity_dp_requires_independent_lexical_ontology_evidence(self):
        evidence = {
            "kind": "strong_identifier_lexical_ontology",
            "lexical_rule": "class_qualified_identifier_name",
            "lexical_score": 0.98,
            "domain_score": 1.0,
            "unique": True,
        }
        base = {
            "pattern": "SE",
            "table_class_candidates": [
                {"uri": uri("Country"), "score": 1.0},
                {"uri": uri("City"), "score": 0.5},
            ],
            "columns": {
                "code": {
                    "role": "fk_obj",
                    "identity_part": True,
                    "ref_table": "city",
                    "ref_col": "country",
                    "constraint_name": "country_capital_fk",
                    "fk_arity": 3,
                    "ref_class_candidates": [
                        {"uri": uri("City"), "score": 1.0},
                        {"uri": uri("Country"), "score": 0.5},
                    ],
                    "candidates": [{"uri": uri("capital"), "score": 1.0}],
                    "dp_candidates": [
                        {
                            "uri": uri("countryCode"),
                            "score": 0.7,
                            "identity_evidence": evidence,
                        },
                    ],
                }
            },
        }
        high = _match_SE("country", base)
        high_column = high["columns"]["code"]
        self.assertEqual(high_column["data_prop_uri"], uri("countryCode"))
        self.assertEqual(high_column["data_prop_confidence"], "high")

        # A high weighted score can come entirely from the table domain.  It
        # must not authorize a generic identity column to become an unrelated
        # literal property, nor route that guess to ContextEnhanced/LLM.
        base["columns"]["code"]["dp_candidates"] = [
            {"uri": uri("email"), "score": 0.99, "domain_score": 1.0},
        ]
        abstained = _match_SE("country", base)
        abstained_column = abstained["columns"]["code"]
        self.assertIsNone(abstained_column["data_prop_uri"])
        self.assertEqual(abstained_column["dp_candidates"], [])
        report = collect_low_confidence_data_property_mappings({"country": abstained})
        self.assertNotIn("country", report)


class CompositeForeignKeyOutputTests(unittest.TestCase):
    def _alignment(self) -> dict:
        return {
            "country": {
                "pattern": "SE",
                "class_uri": uri("Country"),
                "columns": {
                    "code": {
                        "role": "fk_obj",
                        "identity_part": True,
                        "data_prop_uri": uri("carCode"),
                        "data_prop_confidence": "high",
                    },
                    "capital_name": {"role": "fk_obj"},
                    "capital_province": {"role": "fk_obj"},
                },
            },
            "city": {
                "pattern": "SE",
                "class_uri": uri("City"),
                "columns": {
                    "name": {"role": "pk"},
                    "country": {"role": "pk"},
                    "province": {"role": "pk"},
                },
            },
        }

    def test_complete_composite_fk_emits_one_parent_join_and_literal_dp(self):
        mapping = generate_r2rml(
            final_alignment=self._alignment(),
            op_mapping_full={
                "step1": {
                    "country.__fk__country_capital_fk": {
                        "object_prop_uri": uri("capital")
                    }
                },
                "step2_orphans": {},
            },
            enriched_schema=composite_schema(),
            ontology=ontology(),
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )
        country = mapping.split("# === country (SE) ===", 1)[1].split(
            "# === city (SE) ===", 1
        )[0]
        self.assertEqual(country.count("syn:capital"), 1)
        self.assertIn("rr:parentTriplesMap <#cityMapping>", country)
        for child, parent in (
            ("capital_name", "name"),
            ("code", "country"),
            ("capital_province", "province"),
        ):
            self.assertIn(
                f'rr:child "{child}" ; rr:parent "{parent}"',
                country,
            )
        self.assertIn("syn:carCode", country)
        self.assertNotIn("/city/{capital_name}", country)
        self.assertNotIn("/city/{code}", country)

    def test_incomplete_or_conflicting_composite_fk_abstains(self):
        schema = composite_schema()
        schema["country"]["foreign_keys"].pop()
        mapping = generate_r2rml(
            final_alignment=self._alignment(),
            op_mapping_full={
                "step1": {
                    "country.capital_name": {"object_prop_uri": uri("capital")},
                    "country.code": {"object_prop_uri": uri("locatedIn")},
                },
                "step2_orphans": {},
            },
            enriched_schema=schema,
            ontology=ontology(),
            base_url="https://synthetic.invalid/data/",
            prefix="syn",
        )
        country = mapping.split("# === country (SE) ===", 1)[1].split(
            "# === city (SE) ===", 1
        )[0]
        self.assertNotIn("rr:parentTriplesMap <#cityMapping>", country)
        self.assertNotIn("/city/{capital_name}", country)
        self.assertNotIn("/city/{code}", country)
        self.assertIn("syn:carCode", country)


if __name__ == "__main__":
    unittest.main()
