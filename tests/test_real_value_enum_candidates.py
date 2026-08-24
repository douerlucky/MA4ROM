import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealValue.real_value_enhancement_agent import (
    _ambiguous_single_value_leaf_classes,
    _discover_data_attr_enum_type_assertion,
    _expand_enum_class_candidates,
    _unique_lexical_enum_class,
    _real_value_type_mapping,
    _residual_direct_branch_mapping,
    run_real_value_enhancement,
)


NS = "http://example.test/ontology#"
OTHER_NS = "http://external.test/types#"


def uri(local_name: str) -> str:
    return NS + local_name


def synthetic_ontology() -> dict:
    root = uri("Artifact")
    base = uri("CurrentArtifact")
    child = uri("CurrentArtifactVariant")
    precision = uri("PrecisionSensor")
    acronym = uri("XMLArtifact")
    disjoint = uri("ForbiddenArtifact")
    alpha_one = uri("AlphaOne")
    alpha_two = uri("AlphaTwo")
    leaf_one = uri("CandidateLeafOne")
    leaf_two = uri("CandidateLeafTwo")
    children_of = {
        root: [base, precision, acronym, disjoint, alpha_one, alpha_two],
        base: [child, leaf_one, leaf_two],
    }
    return {
        "classes": [
            root,
            base,
            child,
            precision,
            acronym,
            disjoint,
            alpha_one,
            alpha_two,
            leaf_one,
            leaf_two,
        ],
        "children_of": children_of,
        "subclass_of": {
            base: [root],
            child: [base],
            precision: [root],
            acronym: [root],
            disjoint: [root],
            alpha_one: [root],
            alpha_two: [root],
            leaf_one: [base],
            leaf_two: [base],
        },
        "ancestors_of": {
            base: [root],
            child: [base, root],
            precision: [root],
            acronym: [root],
            disjoint: [root],
            alpha_one: [root],
            alpha_two: [root],
            leaf_one: [base, root],
            leaf_two: [base, root],
        },
        "incompatible_classes": {
            base: [disjoint],
            disjoint: [base, child],
            child: [disjoint],
        },
    }


class EnumCandidateExpansionTests(unittest.TestCase):
    def test_unique_semantic_prefix_can_lock_a_leaf_without_dataset_lookup(self):
        base = uri("Facility")
        leaf = uri("FSUFacility")
        sibling = uri("FPSOFacility")
        ontology = {
            "children_of": {base: [leaf, sibling]},
            "ancestors_of": {leaf: [base], sibling: [base]},
            "subclass_of": {leaf: [base], sibling: [base]},
            "incompatible_classes": {},
        }
        selected, audit = _unique_lexical_enum_class(
            "FSU",
            [
                {"uri": leaf, "local_name": "FSUFacility"},
                {"uri": sibling, "local_name": "FPSOFacility"},
            ],
            base,
            ontology,
        )
        self.assertEqual(selected, leaf)
        self.assertEqual(audit["reason"], "unique_strong_lexical_class_evidence")

    def test_ambiguous_prefix_abstains(self):
        base = uri("Facility")
        first = uri("AlphaFacility")
        second = uri("AlphaUnit")
        ontology = {
            "children_of": {base: [first, second]},
            "ancestors_of": {first: [base], second: [base]},
            "subclass_of": {first: [base], second: [base]},
            "incompatible_classes": {},
        }
        selected, audit = _unique_lexical_enum_class(
            "ALPHA",
            [
                {"uri": first, "local_name": "AlphaFacility"},
                {"uri": second, "local_name": "AlphaUnit"},
            ],
            base,
            ontology,
        )
        self.assertIsNone(selected)
        self.assertEqual(audit["reason"], "ambiguous_or_weak_lexical_candidates")

    def test_strong_matcher_can_retain_sibling_under_trusted_parent(self):
        sibling = uri("PrecisionSensor")
        candidates, diagnostics = _expand_enum_class_candidates(
            current_class_uri=uri("CurrentArtifact"),
            class_candidates=[
                {
                    "uri": sibling,
                    "local_name": "PrecisionSensor",
                    "score": 0.95,
                }
            ],
            ontology=synthetic_ontology(),
            return_diagnostics=True,
        )

        by_uri = {candidate["uri"]: candidate for candidate in candidates}
        self.assertIn(sibling, by_uri)
        self.assertEqual(
            by_uri[sibling]["admission_evidence"]["kind"],
            "strong_matcher",
        )
        self.assertEqual(
            diagnostics["admitted_sibling_candidates"][0]["common_ancestor"],
            uri("Artifact"),
        )

    def test_unique_real_value_lexical_evidence_recalls_missing_sibling(self):
        sibling = uri("PrecisionSensor")
        candidates, _ = _expand_enum_class_candidates(
            current_class_uri=uri("CurrentArtifact"),
            class_candidates=[],
            ontology=synthetic_ontology(),
            evidence_values=["PRECISION SENSOR"],
            return_diagnostics=True,
        )

        by_uri = {candidate["uri"]: candidate for candidate in candidates}
        self.assertIn(sibling, by_uri)
        self.assertEqual(by_uri[sibling]["source"], "enum_value_lexical_evidence")

        acronym_candidates, _ = _expand_enum_class_candidates(
            current_class_uri=uri("CurrentArtifact"),
            class_candidates=[],
            ontology=synthetic_ontology(),
            evidence_values=["XML"],
            return_diagnostics=True,
        )
        self.assertIn(uri("XMLArtifact"), {candidate["uri"] for candidate in acronym_candidates})

    def test_disjoint_and_foreign_namespace_candidates_remain_excluded(self):
        disjoint = uri("ForbiddenArtifact")
        foreign = OTHER_NS + "PrecisionSensor"
        candidates, diagnostics = _expand_enum_class_candidates(
            current_class_uri=uri("CurrentArtifact"),
            class_candidates=[
                {"uri": disjoint, "local_name": "ForbiddenArtifact", "score": 0.99},
                {"uri": foreign, "local_name": "PrecisionSensor", "score": 0.99},
            ],
            ontology=synthetic_ontology(),
            evidence_values=["FORBIDDEN ARTIFACT", "PRECISION SENSOR"],
            return_diagnostics=True,
        )

        result_uris = {candidate["uri"] for candidate in candidates}
        self.assertNotIn(disjoint, result_uris)
        self.assertNotIn(foreign, result_uris)
        reasons = {item["uri"]: item["reason"] for item in diagnostics["excluded_candidates"]}
        self.assertEqual(reasons[disjoint], "ontology_disjoint_with_known_class")
        self.assertEqual(reasons[foreign], "namespace_mismatch")

    def test_weak_or_ambiguous_evidence_does_not_blindly_expand_siblings(self):
        weak_sibling = uri("PrecisionSensor")
        candidates, diagnostics = _expand_enum_class_candidates(
            current_class_uri=uri("CurrentArtifact"),
            class_candidates=[
                {"uri": weak_sibling, "local_name": "PrecisionSensor", "score": 0.4}
            ],
            ontology=synthetic_ontology(),
            evidence_values=["ALPHA"],
            return_diagnostics=True,
        )

        result_uris = {candidate["uri"] for candidate in candidates}
        self.assertNotIn(weak_sibling, result_uris)
        self.assertNotIn(uri("AlphaOne"), result_uris)
        self.assertNotIn(uri("AlphaTwo"), result_uris)
        self.assertTrue(diagnostics["ambiguous_value_evidence"])
        self.assertIn(
            "insufficient_sibling_evidence",
            {item["reason"] for item in diagnostics["excluded_candidates"]},
        )


class DataAttributeEnumDiscoveryTests(unittest.TestCase):
    def _ontology(self):
        root = uri("Asset")
        base = uri("OperationalAsset")
        solar = uri("SolarPowerAsset")
        wind = uri("WindPowerAsset")
        emergency = uri("EmergencyResponseUnit")
        marked_acronym = uri("RDAsset")
        alpha_service = uri("AlphaService")
        alpha_system = uri("AlphaSystem")
        return base, solar, wind, emergency, {
            "classes": [
                root,
                base,
                solar,
                wind,
                emergency,
                marked_acronym,
                alpha_service,
                alpha_system,
            ],
            "children_of": {
                root: [
                    base,
                    solar,
                    wind,
                    emergency,
                    marked_acronym,
                    alpha_service,
                    alpha_system,
                ],
            },
            "subclass_of": {
                base: [root],
                solar: [root],
                wind: [root],
                emergency: [root],
                marked_acronym: [root],
                alpha_service: [root],
                alpha_system: [root],
            },
            "ancestors_of": {
                base: [root],
                solar: [root],
                wind: [root],
                emergency: [root],
                marked_acronym: [root],
                alpha_service: [root],
                alpha_system: [root],
            },
            "incompatible_classes": {},
        }

    @staticmethod
    def _profiles(values):
        return {
            value: [{"asset_kind": value}] * 4
            for value in values
        }

    def test_spacing_hyphen_and_acronym_values_add_only_type_assertions(self):
        base, solar, wind, emergency, ontology = self._ontology()
        column_entry = {
            "role": "data_attr",
            "prop_uri": uri("assetKindLabel"),
            "confidence": "low",
        }
        before = dict(column_entry)

        assertion, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="asset_kind",
            column_entry=column_entry,
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles=self._profiles(
                ["SOLAR POWER", "WIND-POWER", "ERU", "R&D"]
            ),
            value_domain={"complete": True, "truncated": False},
        )

        self.assertEqual(column_entry, before)
        self.assertEqual(assertion["kind"], "enum")
        self.assertEqual(assertion["column"], "asset_kind")
        self.assertEqual(
            assertion["value_to_class"],
            {
                "SOLAR POWER": solar,
                "WIND-POWER": wind,
                "ERU": emergency,
                "R&D": uri("RDAsset"),
            },
        )
        self.assertEqual(assertion["unmapped_values"], [])
        self.assertEqual(diagnostics["reason"], "unique_ontology_family_lexical_matches")

    def test_alphanumeric_label_is_allowed_but_numeric_codes_abstain(self):
        base = uri("Device")
        type_two = uri("Type2Device")
        ontology = {
            "children_of": {base: [type_two]},
            "subclass_of": {type_two: [base]},
            "ancestors_of": {type_two: [base]},
            "incompatible_classes": {},
        }
        entry = {"role": "data_attr", "prop_uri": uri("deviceCode")}

        assertion, _ = _discover_data_attr_enum_type_assertion(
            column_name="device_code",
            column_entry=entry,
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles={
                "TYPE-2": [{"device_code": "TYPE-2"}] * 3,
                "71": [{"device_code": "71"}] * 3,
            },
            value_domain={"complete": True},
        )

        self.assertEqual(assertion["value_to_class"], {"TYPE-2": type_two})
        self.assertEqual(assertion["unmapped_values"], ["71"])

    def test_ambiguous_values_and_free_text_abstain(self):
        base, _solar, _wind, _emergency, ontology = self._ontology()
        entry = {"role": "data_attr", "prop_uri": uri("description")}

        ambiguous, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="category_label",
            column_entry=entry,
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles=self._profiles(["ALPHA", "UNKNOWN"]),
            value_domain={"complete": True},
        )
        self.assertIsNone(ambiguous)
        self.assertIn("ALPHA", diagnostics["ambiguous_values"])

        free_text, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="description",
            column_entry=entry,
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles=self._profiles([
                "This is a long unstructured description containing many independent words",
                "Another unstructured paragraph that should remain a literal data property",
            ]),
            value_domain={"complete": True},
        )
        self.assertIsNone(free_text)
        self.assertEqual(diagnostics["reason"], "free_text_like_values")

    def test_incomplete_or_high_cardinality_profile_abstains(self):
        base, _solar, _wind, _emergency, ontology = self._ontology()
        entry = {"role": "data_attr", "prop_uri": uri("assetKindLabel")}

        incomplete, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="asset_kind",
            column_entry=entry,
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles=self._profiles(["SOLAR POWER", "WIND POWER"]),
            value_domain={"complete": False, "truncated": True},
        )
        self.assertIsNone(incomplete)
        self.assertEqual(diagnostics["reason"], "incomplete_value_domain")

        many_values = ["SOLAR POWER", "WIND POWER"] + [
            f"UNKNOWN-{idx}" for idx in range(20)
        ]
        high_cardinality, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="asset_kind",
            column_entry=entry,
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles=self._profiles(many_values),
            value_domain={"complete": True},
        )
        self.assertIsNone(high_cardinality)
        self.assertEqual(diagnostics["reason"], "not_low_cardinality")

    def test_unconfirmed_base_or_non_data_attribute_abstains(self):
        base, _solar, _wind, _emergency, ontology = self._ontology()
        profiles = self._profiles(["SOLAR POWER", "WIND POWER"])

        unconfirmed, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="asset_kind",
            column_entry={"role": "data_attr", "prop_uri": uri("assetKindLabel")},
            current_class_uri=base,
            current_class_confidence="low",
            class_candidates=[],
            ontology=ontology,
            value_profiles=profiles,
            value_domain={"complete": True},
        )
        self.assertIsNone(unconfirmed)
        self.assertEqual(diagnostics["reason"], "unconfirmed_table_class")

        discriminator, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="asset_kind",
            column_entry={"role": "discriminator", "prop_uri": None},
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles=profiles,
            value_domain={"complete": True},
        )
        self.assertIsNone(discriminator)
        self.assertEqual(diagnostics["reason"], "not_data_attribute")

    def test_sibling_with_multiple_direct_common_parents_abstains(self):
        root_one = uri("RootOne")
        root_two = uri("RootTwo")
        base = uri("CurrentAsset")
        sibling = uri("SolarPowerAsset")
        ontology = {
            "children_of": {
                root_one: [base, sibling],
                root_two: [base, sibling],
            },
            "subclass_of": {
                base: [root_one, root_two],
                sibling: [root_one, root_two],
            },
            "ancestors_of": {
                base: [root_one, root_two],
                sibling: [root_one, root_two],
            },
            "incompatible_classes": {},
        }
        assertion, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="asset_kind",
            column_entry={"role": "data_attr", "prop_uri": uri("assetKind")},
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles=self._profiles(["SOLAR POWER", "UNKNOWN"]),
            value_domain={"complete": True},
        )

        self.assertIsNone(assertion)
        self.assertIn(
            "non_unique_direct_common_parent",
            {item["reason"] for item in diagnostics["excluded_scope_candidates"]},
        )

    def test_disjoint_sibling_and_conflicting_samples_abstain(self):
        root = uri("Asset")
        base = uri("CurrentAsset")
        sibling = uri("SolarPowerAsset")
        disjoint_ontology = {
            "children_of": {root: [base, sibling]},
            "subclass_of": {base: [root], sibling: [root]},
            "ancestors_of": {base: [root], sibling: [root]},
            "incompatible_classes": {base: [sibling], sibling: [base]},
        }
        assertion, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="asset_kind",
            column_entry={"role": "data_attr", "prop_uri": uri("assetKind")},
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=disjoint_ontology,
            value_profiles={
                "SOLAR POWER": [{"asset_kind": "SOLAR POWER"}] * 3,
                "UNKNOWN": [{"asset_kind": "UNKNOWN"}] * 3,
            },
            value_domain={"complete": True},
        )
        self.assertIsNone(assertion)
        self.assertIn(
            "ontology_disjoint_with_known_class",
            {
                item["reason"]
                for item in diagnostics["candidate_expansion"]["excluded_candidates"]
            },
        )

        base, solar, wind, _emergency, ontology = self._ontology()
        assertion, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="asset_kind",
            column_entry={"role": "data_attr", "prop_uri": uri("assetKind")},
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles={
                "SOLAR POWER": [{"asset_kind": "WIND-POWER"}] * 3,
                "WIND-POWER": [{"asset_kind": "WIND-POWER"}] * 3,
            },
            value_domain={"complete": True},
        )
        self.assertEqual(assertion["value_to_class"], {"WIND-POWER": wind})
        self.assertNotIn("SOLAR POWER", assertion["value_to_class"])
        self.assertIn("SOLAR POWER", diagnostics["conflicting_values"])
        self.assertEqual(
            diagnostics["abstain_reasons"]["SOLAR POWER"],
            "conflicting_profile_samples",
        )

    def test_non_string_values_are_not_interpreted_as_labels(self):
        base = uri("Device")
        type_two = uri("Type2Device")
        ontology = {
            "children_of": {base: [type_two]},
            "subclass_of": {type_two: [base]},
            "ancestors_of": {type_two: [base]},
            "incompatible_classes": {},
        }
        assertion, diagnostics = _discover_data_attr_enum_type_assertion(
            column_name="device_code",
            column_entry={"role": "data_attr", "prop_uri": uri("deviceCode")},
            current_class_uri=base,
            current_class_confidence="high",
            class_candidates=[],
            ontology=ontology,
            value_profiles={
                "2": [{"device_code": 2}] * 3,
                "TYPE-2": [{"device_code": "TYPE-2"}] * 3,
            },
            value_domain={"complete": True},
        )
        self.assertEqual(assertion["value_to_class"], {"TYPE-2": type_two})
        self.assertEqual(diagnostics["abstain_reasons"]["2"], "non_string_value")

    def test_pipeline_appends_assertion_without_reclassifying_the_dp_column(self):
        base, solar, wind, _emergency, ontology = self._ontology()
        prop_uri = uri("assetKindLabel")
        alignment = {
            "asset_records": {
                "pattern": "SE",
                "class_uri": base,
                "class_confidence": "high",
                "columns": {
                    "asset_kind": {
                        "role": "data_attr",
                        "prop_uri": prop_uri,
                        "confidence": "low",
                    }
                },
            }
        }
        candidates = {
            "asset_records": {
                "table_class_candidates": [
                    {"uri": base, "local_name": "OperationalAsset", "score": 1.0}
                ],
                "columns": {
                    "asset_kind": {
                        "role": "data_attr",
                        "column_type": "character varying(32)",
                        "candidates": [
                            {
                                "uri": prop_uri,
                                "local_name": "has_a_asset_kind",
                                "score": 0.9,
                            }
                        ],
                    }
                },
            }
        }
        sample_rows = [
            {"asset_kind": "SOLAR POWER"},
            {"asset_kind": "WIND-POWER"},
            {"asset_kind": "SOLAR POWER"},
        ]
        profiles = {
            "SOLAR POWER": [{"asset_kind": "SOLAR POWER"}] * 4,
            "WIND-POWER": [{"asset_kind": "WIND-POWER"}] * 4,
        }

        with (
            patch(
                "RealValue.real_value_enhancement_agent.fetch_sample_rows",
                return_value=sample_rows,
            ),
            patch(
                "RealValue.real_value_enhancement_agent._fetch_distinct_value_profiles",
                return_value=(
                    sorted(profiles),
                    profiles,
                    {"complete": True, "truncated": False},
                ),
            ),
            patch(
                "RealValue.real_value_enhancement_agent._call_llm",
                side_effect=AssertionError("strong lexical discovery must not call the LLM"),
            ),
        ):
            result = run_real_value_enhancement(
                alignment,
                {
                    "asset_records": {
                        "table_low": False,
                        "columns_low": ["asset_kind"],
                    }
                },
                candidates,
                ontology=ontology,
                enriched_schema={},
            )

        column = result["asset_records"]["columns"]["asset_kind"]
        self.assertEqual(column["role"], "data_attr")
        self.assertEqual(column["prop_uri"], prop_uri)
        self.assertEqual(
            result["asset_records"]["type_assertions"][0]["value_to_class"],
            {"SOLAR POWER": solar, "WIND-POWER": wind},
        )


class SingletonNumericEnumTests(unittest.TestCase):
    def candidates(self) -> list[dict]:
        return [
            {
                "uri": uri("CandidateLeafOne"),
                "local_name": "CandidateLeafOne",
                "score": 0.99,
                "source": "matcher_candidate",
            },
            {
                "uri": uri("CandidateLeafTwo"),
                "local_name": "CandidateLeafTwo",
                "score": 0.98,
                "source": "matcher_candidate",
            },
        ]

    def test_two_non_disjoint_high_matcher_leaves_still_abstain(self):
        leaves, diagnostics = _ambiguous_single_value_leaf_classes(
            values=["7"],
            class_candidates=self.candidates(),
            current_class_uri=uri("CurrentArtifact"),
            ontology=synthetic_ontology(),
        )
        self.assertEqual(leaves, [])
        self.assertEqual(
            diagnostics["reason"],
            "non_disjointness_does_not_prove_class_intersection",
        )

        result = _real_value_type_mapping(
            table_name="entity_records",
            type_col="category_code",
            value_profiles={"7": [{"category_code": 7}]},
            class_candidates=self.candidates(),
            current_class_uri=uri("CurrentArtifact"),
            descendant_uris=[uri("CandidateLeafOne"), uri("CandidateLeafTwo")],
            ontology=synthetic_ontology(),
        )
        self.assertEqual(result["value_to_class"], {})
        self.assertEqual(result["unmapped_values"], [7])

    def test_joined_rows_enable_bounded_single_class_llm_review(self):
        joined_context = {
            "pk_column": "id",
            "groups": {
                "7": [
                    {
                        "category_code": 7,
                        "outgoing_fk_rows": {
                            "owner_id->owners.id": [{"id": owner_id}]
                        },
                        "incoming_relation_rows": {},
                    }
                    for owner_id in (42, 43, 44)
                ]
            }
        }
        selected = uri("CandidateLeafOne")
        with patch(
            "RealValue.real_value_enhancement_agent._call_llm",
            return_value={
                "candidate_assignments": {"v0": "c0"},
                "unmapped_value_ids": [],
                "confidence": "medium",
                "reason": "joined context selects one direct subtype",
            },
        ) as call:
            result = _real_value_type_mapping(
                table_name="entity_records",
                type_col="category_code",
                value_profiles={"7": [{"category_code": 7, "owner_id": 42}]},
                # The base Class is intentionally first.  The local catalog
                # must remove it so c0 denotes the first direct subtype.
                class_candidates=[
                    {
                        "uri": uri("CurrentArtifact"),
                        "local_name": "CurrentArtifact",
                        "score": 1.0,
                    },
                    *self.candidates(),
                ],
                fk_context={
                    "pk_column": "id",
                    "coverage_by_value": {
                        "outgoing:owner_id->owners.id": {
                            "7": {"total": 3, "linked": 3, "ratio": 1.0}
                        }
                    },
                },
                group_context=joined_context,
                current_class_uri=uri("CurrentArtifact"),
                current_class_confidence="high",
                descendant_uris=[
                    uri("CandidateLeafOne"),
                    uri("CandidateLeafTwo"),
                ],
                ontology=synthetic_ontology(),
                value_domain_complete=True,
            )

        self.assertEqual(result["value_to_class"], {"7": selected})
        self.assertEqual(result["unmapped_values"], [])
        self.assertEqual(call.call_count, 1)

    def test_joined_singleton_retries_null_contract_once(self):
        joined_context = {
            "pk_column": "id",
            "groups": {
                "7": [
                    {
                        "category_code": 7,
                        "outgoing_fk_rows": {
                            "owner_id->owners.id": [{"id": owner_id}]
                        },
                        "incoming_relation_rows": {},
                    }
                    for owner_id in (42, 43, 44)
                ]
            },
        }
        selected = uri("CandidateLeafTwo")
        with patch(
            "RealValue.real_value_enhancement_agent._call_llm",
            side_effect=[
                {
                    "candidate_assignments": {"v0": None},
                    "unmapped_value_ids": ["v0"],
                    "confidence": "low",
                    "reason": "synthetic contract violation",
                },
                {
                    "candidate_assignments": {"v0": "c1"},
                    "unmapped_value_ids": [],
                    "confidence": "medium",
                    "reason": "stable joined roles support the second subtype",
                },
            ],
        ) as call:
            result = _real_value_type_mapping(
                table_name="entity_records",
                type_col="category_code",
                value_profiles={"7": [{"category_code": 7, "owner_id": 42}]},
                class_candidates=[
                    {
                        "uri": uri("CurrentArtifact"),
                        "local_name": "CurrentArtifact",
                        "score": 1.0,
                    },
                    *self.candidates(),
                ],
                fk_context={
                    "pk_column": "id",
                    "coverage_by_value": {
                        "outgoing:owner_id->owners.id": {
                            "7": {"total": 3, "linked": 3, "ratio": 1.0}
                        }
                    },
                },
                group_context=joined_context,
                current_class_uri=uri("CurrentArtifact"),
                current_class_confidence="high",
                descendant_uris=[
                    uri("CandidateLeafOne"),
                    uri("CandidateLeafTwo"),
                ],
                ontology=synthetic_ontology(),
                value_domain_complete=True,
            )

        self.assertEqual(result["value_to_class"], {"7": selected})
        self.assertEqual(result["unmapped_values"], [])
        self.assertEqual(call.call_count, 2)
        self.assertIn("必须从候选直接子类中选择且只选择一个", call.call_args_list[0].args[0])
        self.assertIn("不得返回 null", call.call_args_list[1].args[0])

    def test_joined_singleton_never_falls_back_to_candidate_order(self):
        joined_context = {
            "pk_column": "id",
            "groups": {
                "7": [
                    {
                        "category_code": 7,
                        "outgoing_fk_rows": {
                            "owner_id->owners.id": [{"id": owner_id}]
                        },
                        "incoming_relation_rows": {},
                    }
                    for owner_id in (42, 43, 44)
                ]
            },
        }
        invalid = {
            "candidate_assignments": {"v0": None},
            "unmapped_value_ids": ["v0"],
            "confidence": "low",
            "reason": "synthetic repeated contract violation",
        }
        with patch(
            "RealValue.real_value_enhancement_agent._call_llm",
            side_effect=[invalid, invalid],
        ) as call:
            result = _real_value_type_mapping(
                table_name="entity_records",
                type_col="category_code",
                value_profiles={"7": [{"category_code": 7, "owner_id": 42}]},
                class_candidates=[
                    {
                        "uri": uri("CurrentArtifact"),
                        "local_name": "CurrentArtifact",
                        "score": 1.0,
                    },
                    *self.candidates(),
                ],
                fk_context={
                    "pk_column": "id",
                    "coverage_by_value": {
                        "outgoing:owner_id->owners.id": {
                            "7": {"total": 3, "linked": 3, "ratio": 1.0}
                        }
                    },
                },
                group_context=joined_context,
                current_class_uri=uri("CurrentArtifact"),
                current_class_confidence="high",
                descendant_uris=[
                    uri("CandidateLeafOne"),
                    uri("CandidateLeafTwo"),
                ],
                ontology=synthetic_ontology(),
                value_domain_complete=True,
            )

        self.assertEqual(result["value_to_class"], {})
        self.assertEqual(result["unmapped_values"], [7])
        self.assertEqual(call.call_count, 2)
        self.assertIn("未使用 top1", result["reason"])

    def test_joined_singleton_rejects_base_uri_and_multi_label_output(self):
        joined_context = {
            "pk_column": "id",
            "groups": {
                "7": [
                    {
                        "category_code": 7,
                        "outgoing_fk_rows": {
                            "owner_id->owners.id": [{"id": owner_id}]
                        },
                        "incoming_relation_rows": {},
                    }
                    for owner_id in (42, 43, 44)
                ]
            }
        }
        candidates = [
            {
                "uri": uri("CurrentArtifact"),
                "local_name": "CurrentArtifact",
                "score": 1.0,
            },
            *self.candidates(),
        ]

        for response in (
            {"value_to_class": {"7": uri("CurrentArtifact")}},
            {"candidate_assignments": {"v0": ["c0", "c1"]}},
        ):
            with self.subTest(response=response), patch(
                "RealValue.real_value_enhancement_agent._call_llm",
                return_value=response,
            ):
                result = _real_value_type_mapping(
                    table_name="entity_records",
                    type_col="category_code",
                    value_profiles={"7": [{"category_code": 7, "owner_id": 42}]},
                    class_candidates=candidates,
                    fk_context={
                        "pk_column": "id",
                        "coverage_by_value": {
                            "outgoing:owner_id->owners.id": {
                                "7": {"total": 3, "linked": 3, "ratio": 1.0}
                            }
                        },
                    },
                    group_context=joined_context,
                    current_class_uri=uri("CurrentArtifact"),
                    descendant_uris=[
                        uri("CandidateLeafOne"),
                        uri("CandidateLeafTwo"),
                    ],
                    ontology=synthetic_ontology(),
                )

            self.assertEqual(result["value_to_class"], {})
            self.assertEqual(result["unmapped_values"], [7])

    def test_inherited_identity_join_alone_does_not_authorize_review(self):
        identity_only_context = {
            "pk_column": "id",
            "groups": {
                "7": [
                    {
                        "category_code": 7,
                        "outgoing_fk_rows": {
                            "id->base_records.id": [{"id": entity_id}]
                        },
                        "incoming_relation_rows": {},
                    }
                    for entity_id in (1, 2, 3)
                ]
            },
        }
        with patch(
            "RealValue.real_value_enhancement_agent._call_llm",
            side_effect=AssertionError(
                "inherited base identity must not authorize subtype review"
            ),
        ):
            result = _real_value_type_mapping(
                table_name="entity_records",
                type_col="category_code",
                value_profiles={"7": [{"category_code": 7}]},
                class_candidates=self.candidates(),
                fk_context={
                    "pk_column": "id",
                    "coverage_by_value": {
                        "outgoing:id->base_records.id": {
                            "7": {"total": 3, "linked": 3, "ratio": 1.0}
                        }
                    },
                },
                group_context=identity_only_context,
                current_class_uri=uri("CurrentArtifact"),
                descendant_uris=[
                    uri("CandidateLeafOne"),
                    uri("CandidateLeafTwo"),
                ],
                ontology=synthetic_ontology(),
            )

        self.assertEqual(result["value_to_class"], {})
        self.assertEqual(result["unmapped_values"], [7])

    def test_sampled_join_cannot_hide_mixed_full_bucket_coverage(self):
        sampled_context = {
            "pk_column": "id",
            "groups": {
                "7": [
                    {
                        "category_code": 7,
                        "outgoing_fk_rows": {},
                        "incoming_relation_rows": {
                            "subtype_rows.id->entity_records.id": [{"id": entity_id}]
                        },
                    }
                    for entity_id in (1, 2, 3)
                ]
            },
        }
        with patch(
            "RealValue.real_value_enhancement_agent._call_llm",
            side_effect=AssertionError(
                "three lucky sample rows must not override mixed full-bucket coverage"
            ),
        ):
            result = _real_value_type_mapping(
                table_name="entity_records",
                type_col="category_code",
                value_profiles={"7": [{"category_code": 7}]},
                class_candidates=self.candidates(),
                fk_context={
                    "pk_column": "id",
                    "coverage_by_value": {
                        "subtype_rows.id": {
                            "7": {"total": 100, "linked": 57, "ratio": 0.57}
                        }
                    },
                },
                group_context=sampled_context,
                current_class_uri=uri("CurrentArtifact"),
                descendant_uris=[
                    uri("CandidateLeafOne"),
                    uri("CandidateLeafTwo"),
                ],
                ontology=synthetic_ontology(),
            )

        self.assertEqual(result["value_to_class"], {})
        self.assertEqual(result["unmapped_values"], [7])

    def test_insufficient_singleton_structural_gap_abstains_without_llm(self):
        with patch(
            "RealValue.real_value_enhancement_agent._call_llm",
            side_effect=AssertionError("ambiguous SH evidence must not reach the LLM"),
        ):
            result = _real_value_type_mapping(
                table_name="entity_records",
                type_col="category_code",
                value_profiles={"7": [{"category_code": 7}]},
                class_candidates=self.candidates(),
                sh_value_evidence={
                    "7": [
                        {
                            "class_uri": uri("CandidateLeafOne"),
                            "ratio": 1.0,
                            "total": 5,
                            "evidence_source": "validated_sh_identity",
                        },
                        {
                            "class_uri": uri("CandidateLeafTwo"),
                            "ratio": 0.85,
                            "total": 5,
                            "evidence_source": "validated_sh_identity",
                        },
                    ]
                },
                current_class_uri=uri("CurrentArtifact"),
                descendant_uris=[
                    uri("CandidateLeafOne"),
                    uri("CandidateLeafTwo"),
                ],
                ontology=synthetic_ontology(),
            )

        self.assertEqual(result["value_to_class"], {})
        self.assertEqual(result["unmapped_values"], [7])

    def test_unique_strong_sh_instance_evidence_still_maps_one_class(self):
        selected = uri("CandidateLeafOne")
        result = _real_value_type_mapping(
            table_name="entity_records",
            type_col="category_code",
            value_profiles={"7": [{"category_code": 7}]},
            class_candidates=self.candidates(),
            sh_value_evidence={
                "7": [
                    {
                        "class_uri": selected,
                        "ratio": 1.0,
                        "total": 5,
                        "evidence_source": "validated_sh_identity",
                    },
                    {
                        "class_uri": uri("CandidateLeafTwo"),
                        "ratio": 0.2,
                        "total": 5,
                        "evidence_source": "validated_sh_identity",
                    },
                ]
            },
            current_class_uri=uri("CurrentArtifact"),
            descendant_uris=[uri("CandidateLeafOne"), uri("CandidateLeafTwo")],
            ontology=synthetic_ontology(),
        )
        self.assertEqual(result["value_to_class"], {"7": selected})
        self.assertEqual(result["unmapped_values"], [])

    def test_verified_sh_context_does_not_authorize_multi_leaf_mapping(self):
        base = uri("Report")
        abstract = uri("ReportAbstract")
        full = uri("ReportFullVersion")
        family_ontology = {
            "children_of": {base: [abstract, full]},
            "subclass_of": {abstract: [base], full: [base]},
            "ancestors_of": {abstract: [base], full: [base]},
            "incompatible_classes": {},
        }
        result = _real_value_type_mapping(
            table_name="subclass_rows",
            type_col="category_code",
            value_profiles={"2": [{"category_code": 2}]},
            class_candidates=[
                {"uri": abstract, "local_name": "ReportAbstract", "score": 1.0},
                {"uri": full, "local_name": "ReportFullVersion", "score": 1.0},
            ],
            current_class_uri=base,
            descendant_uris=[abstract, full],
            ontology=family_ontology,
        )
        self.assertEqual(result["value_to_class"], {})
        self.assertEqual(result["unmapped_values"], [2])

    def test_table_class_is_confirmed_before_singleton_type_abstention(self):
        document = uri("Document")
        paper = uri("Paper")
        abstract = uri("PaperAbstract")
        full = uri("PaperFullVersion")
        alignment = {
            "Document": {
                "pattern": "SE",
                "class_uri": document,
                "class_confidence": "high",
                "columns": {"ID": {"role": "pk"}},
            },
            "Paper": {
                "pattern": "SH",
                "sub_class_uri": paper,
                "parent_class_uri": document,
                "class_confidence": "low",
                "columns": {
                    "ID": {"role": "sh_inherited_pk"},
                    "Reviewer": {"role": "fk_obj"},
                    "TYPE": {"role": "discriminator"},
                },
                "type_assertions": [{
                    "column": "TYPE",
                    "kind": "enum",
                    "value_to_class": None,
                    "class_candidates": [
                        {"uri": paper, "local_name": "Paper", "score": 1.0},
                        {"uri": abstract, "local_name": "PaperAbstract", "score": 1.0},
                        {"uri": full, "local_name": "PaperFullVersion", "score": 1.0},
                    ],
                }],
            },
        }
        candidates = {
            "Paper": {
                "sub_class_candidates": [
                    {"uri": paper, "local_name": "Paper", "score": 1.0},
                    {"uri": abstract, "local_name": "PaperAbstract", "score": 1.0},
                    {"uri": full, "local_name": "PaperFullVersion", "score": 1.0},
                ],
                "columns": {},
            }
        }
        schema = {
            "Document": {
                "columns": {"ID": "integer"},
                "primary_key": ["ID"],
                "foreign_keys": [],
            },
            "Paper": {
                "columns": {
                    "ID": "integer",
                    "Reviewer": "integer",
                    "TYPE": "integer",
                },
                "primary_key": ["ID", "Reviewer"],
                "foreign_keys": [{
                    "column": "ID",
                    "references_table": "Document",
                    "references_column": "ID",
                }],
            },
        }
        ontology_data = {
            "children_of": {document: [paper], paper: [abstract, full]},
            "subclass_of": {
                paper: [document],
                abstract: [paper],
                full: [paper],
            },
            "ancestors_of": {
                paper: [document],
                abstract: [paper, document],
                full: [paper, document],
            },
            "incompatible_classes": {},
        }
        with (
            patch(
                "RealValue.real_value_enhancement_agent.fetch_sample_rows",
                return_value=[{"TYPE": 2}],
            ),
            patch(
                "RealValue.real_value_enhancement_agent._real_value_table_class",
                return_value={"selected_uri": paper, "confidence": "high", "reason": "synthetic"},
            ),
            patch(
                "RealValue.real_value_enhancement_agent._fetch_distinct_value_profiles",
                return_value=(
                    ["2"],
                    {"2": [{"TYPE": 2}]},
                    {"complete": True, "truncated": False},
                ),
            ),
            patch("RealValue.real_value_enhancement_agent._build_fk_semantic_context", return_value={}),
            patch("RealValue.real_value_enhancement_agent._build_type_group_context", return_value={}),
            patch("RealValue.real_value_enhancement_agent._build_sh_value_evidence", return_value={}),
            patch(
                "RealValue.real_value_enhancement_agent._build_relation_domain_class_evidence",
                return_value={},
            ),
        ):
            result = run_real_value_enhancement(
                alignment,
                {"Paper": {"table_low": True, "columns_low": []}},
                candidates,
                ontology=ontology_data,
                enriched_schema=schema,
            )

        self.assertEqual(result["Paper"]["class_confidence"], "high")
        self.assertEqual(result["Paper"]["type_assertions"][0]["value_to_class"], {})
        self.assertEqual(result["Paper"]["type_assertions"][0]["unmapped_values"], [2])


class ResidualDirectBranchTests(unittest.TestCase):
    def _fixture(self):
        base = uri("Report")
        abstract = uri("ReportAbstract")
        full = uri("ReportFullVersion")
        candidates = [
            {"uri": abstract, "local_name": "ReportAbstract", "score": 1.0},
            {"uri": full, "local_name": "ReportFullVersion", "score": 1.0},
        ]
        ontology = {
            "children_of": {base: [abstract, full]},
            "subclass_of": {abstract: [base], full: [base]},
        }
        evidence = {
            "1": [{
                "class_uri": abstract,
                "ratio": 1.0,
                "total": 5,
                "evidence_source": "validated_sh_identity",
            }]
        }
        return base, abstract, full, candidates, ontology, evidence

    def test_complete_value_domain_allows_unique_residual_branch(self):
        base, abstract, full, candidates, ontology, evidence = self._fixture()

        residual = _residual_direct_branch_mapping(
            values=["1", "2"],
            value_to_class={"1": abstract},
            class_candidates=candidates,
            current_class_uri=base,
            class_subclass_of=ontology["subclass_of"],
            sh_value_evidence=evidence,
            value_domain_complete=True,
            ontology=ontology,
        )

        self.assertEqual(residual, ("2", full))

    def test_truncated_value_domain_abstains_from_residual_branch(self):
        base, abstract, _full, candidates, ontology, evidence = self._fixture()

        residual = _residual_direct_branch_mapping(
            values=["1", "2"],
            value_to_class={"1": abstract},
            class_candidates=candidates,
            current_class_uri=base,
            class_subclass_of=ontology["subclass_of"],
            sh_value_evidence=evidence,
            value_domain_complete=False,
            ontology=ontology,
        )

        self.assertEqual(residual, (None, None))

    def test_complete_residual_is_locked_before_llm_fallback(self):
        base, abstract, full, candidates, ontology, evidence = self._fixture()

        with patch(
            "RealValue.real_value_enhancement_agent._call_llm",
            side_effect=AssertionError("a determined residual partition must not call the LLM"),
        ):
            result = _real_value_type_mapping(
                table_name="entity_rows",
                type_col="category_code",
                value_profiles={
                    "1": [{"category_code": 1}] * 5,
                    "2": [{"category_code": 2}] * 5,
                },
                class_candidates=candidates,
                current_class_uri=base,
                descendant_uris=[abstract, full],
                class_subclass_of=ontology["subclass_of"],
                sh_value_evidence=evidence,
                value_domain_complete=True,
                ontology=ontology,
            )

        self.assertEqual(
            result["value_to_class"],
            {"1": abstract, "2": full},
        )
        self.assertEqual(result["unmapped_values"], [])


if __name__ == "__main__":
    unittest.main()
