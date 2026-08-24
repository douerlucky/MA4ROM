"""Generic SH parent-selection guards; no dataset-specific fixtures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from candidate_generation import _handle_SE, _handle_SH, _resolve_table_class


NS = "https://synthetic.invalid/ontology#"


def uri(local_name: str) -> str:
    return NS + local_name


def entity_table(*foreign_keys: dict) -> dict:
    return {
        "columns": {"entity_id": "integer"},
        "primary_key": ["entity_id"],
        "foreign_keys": list(foreign_keys),
    }


def parent_table() -> dict:
    return {
        "columns": {"entity_id": "integer"},
        "primary_key": ["entity_id"],
        "foreign_keys": [],
    }


def identity_fk(target_table: str) -> dict:
    return {
        "column": "entity_id",
        "references_table": target_table,
        "references_column": "entity_id",
    }


def sh_candidates(table_name: str, schema: dict, ontology: dict) -> dict:
    table = schema[table_name]
    physical_fks = table["foreign_keys"]
    fk_cols = {
        fk["column"]: {
            **fk,
            "ref_table": fk["references_table"],
            "ref_col": fk["references_column"],
        }
        for fk in physical_fks
    }
    return _handle_SH(
        table_name,
        table["columns"],
        set(table["primary_key"]),
        fk_cols,
        ontology["classes"],
        {},
        {},
        schema,
        ontology,
    )


class ShParentSelectionTests(unittest.TestCase):
    def test_normalized_exact_table_name_resolves_before_fuzzy_candidates(self):
        exact_uri = uri("DirectParentRows")

        resolved, method = _resolve_table_class(
            "direct_parent_rows",
            [uri("ParentRows"), exact_uri, uri("DirectParent")],
        )

        self.assertEqual(resolved, exact_uri)
        self.assertEqual(method, "exact")

    def test_exact_table_class_and_direct_superclass_beat_token_tie_and_ancestor(self):
        child_table = "ProjectRoleLead"
        direct_table = "ProjectRoleMember"
        ancestor_table = "Agent"
        classes = [
            # Put the broad token match first to reproduce the former tie bug.
            uri("ProjectRole"),
            uri("ProjectRoleLead"),
            uri("ProjectRoleMember"),
            uri("Agent"),
        ]
        ontology = {
            "classes": classes,
            "subclass_of": {
                uri("ProjectRoleLead"): [uri("ProjectRoleMember")],
                uri("ProjectRoleMember"): [uri("Agent")],
            },
            "ancestors_of": {
                uri("ProjectRoleLead"): [uri("ProjectRoleMember"), uri("Agent")],
                uri("ProjectRoleMember"): [uri("Agent")],
            },
        }
        schema = {
            child_table: entity_table(
                identity_fk(ancestor_table),
                identity_fk(direct_table),
            ),
            direct_table: parent_table(),
            ancestor_table: parent_table(),
        }

        result = sh_candidates(child_table, schema, ontology)

        self.assertEqual(result["sub_class_candidates"][0]["uri"], uri(child_table))
        self.assertEqual(result["parent_table"], direct_table)
        self.assertEqual(
            result["parent_class_candidates"][0]["uri"],
            uri(direct_table),
        )
        self.assertEqual(result["columns"]["entity_id"]["role"], "sh_inherited_pk")

    def test_nearest_available_ancestor_wins_when_direct_parent_has_no_physical_fk(self):
        child_table = "SpecialRecord"
        near_table = "BaseRecord"
        far_table = "RootRecord"
        middle_class = uri("MiddleRecord")
        ontology = {
            "classes": [
                uri(child_table),
                middle_class,
                uri(near_table),
                uri(far_table),
            ],
            "subclass_of": {
                uri(child_table): [middle_class],
                middle_class: [uri(near_table)],
                uri(near_table): [uri(far_table)],
            },
            "ancestors_of": {
                uri(child_table): [middle_class, uri(near_table), uri(far_table)],
                middle_class: [uri(near_table), uri(far_table)],
                uri(near_table): [uri(far_table)],
            },
        }
        schema = {
            child_table: entity_table(
                identity_fk(far_table),
                identity_fk(near_table),
            ),
            near_table: parent_table(),
            far_table: parent_table(),
        }

        result = sh_candidates(child_table, schema, ontology)

        self.assertEqual(result["parent_table"], near_table)

    def test_equal_direct_parents_are_rejected_instead_of_using_fk_order(self):
        child_table = "DualRole"
        left_table = "LeftRole"
        right_table = "RightRole"
        ontology = {
            "classes": [uri(child_table), uri(left_table), uri(right_table)],
            "subclass_of": {
                uri(child_table): [uri(left_table), uri(right_table)],
            },
            "ancestors_of": {
                uri(child_table): [uri(left_table), uri(right_table)],
            },
        }
        schema = {
            child_table: entity_table(
                identity_fk(left_table),
                identity_fk(right_table),
            ),
            left_table: parent_table(),
            right_table: parent_table(),
        }

        result = sh_candidates(child_table, schema, ontology)

        self.assertEqual(result["parent_table"], "")
        self.assertEqual(result["parent_class_candidates"], [])
        # Parent inheritance is rejected, but the physical FK keeps its
        # independent relation semantics and remains part of subject identity.
        self.assertEqual(result["columns"]["entity_id"]["role"], "fk_obj")
        self.assertTrue(result["columns"]["entity_id"]["identity_part"])
        self.assertTrue(result["columns"]["entity_id"]["inheritance_ambiguous"])

    def test_unique_structural_parent_remains_available_without_hierarchy_metadata(self):
        child_table = "SparseChild"
        parent = "SparseParent"
        ontology = {
            "classes": [uri(child_table), uri(parent)],
            "subclass_of": {},
            "ancestors_of": {},
        }
        schema = {
            child_table: entity_table(identity_fk(parent)),
            parent: parent_table(),
        }

        result = sh_candidates(child_table, schema, ontology)

        self.assertEqual(result["parent_table"], parent)
        self.assertEqual(result["columns"]["entity_id"]["role"], "sh_inherited_pk")

    def test_child_candidate_with_parent_semantics_beats_unrelated_lexical_match(self):
        child_table = "politics"
        parent = "country"
        political_body = uri("PoliticalBody")
        country = uri("Country")
        ontology = {
            "classes": [uri("Location"), political_body, country],
            "subclass_of": {country: [political_body]},
            "ancestors_of": {country: [political_body]},
            "incompatible_classes": {},
        }
        schema = {
            child_table: entity_table(identity_fk(parent)),
            parent: parent_table(),
        }
        result = sh_candidates(child_table, schema, ontology)
        self.assertNotEqual(
            result["sub_class_candidates"][0]["uri"], uri("Location")
        )


class IdentityPredicateCandidateTests(unittest.TestCase):
    def test_se_pk_fk_keeps_object_role_and_identity_metadata(self):
        child = uri("Child")
        parent = uri("Parent")
        relation = uri("linksToParent")
        result = _handle_SE(
            "Child",
            {"parent_id": "integer"},
            {"parent_id"},
            {
                "parent_id": {
                    "ref_table": "Parent",
                    "ref_col": "parent_id",
                }
            },
            [child, parent],
            {
                relation: {
                    "domain": [child],
                    "range": [parent],
                }
            },
            {},
            {
                "classes": [child, parent],
                "object_properties": {
                    relation: {
                        "domain": [child],
                        "range": [parent],
                    }
                },
            },
        )

        column = result["columns"]["parent_id"]
        self.assertEqual(column["role"], "fk_obj")
        self.assertTrue(column["identity_part"])
        self.assertTrue(column["candidates"])

    def test_se_non_fk_pk_preserves_datatype_candidates(self):
        entity = uri("NamedEntity")
        name = uri("name")
        result = _handle_SE(
            "NamedEntity",
            {"name": "text"},
            {"name"},
            {},
            [entity],
            {},
            {name: {"domain": [entity], "range": []}},
            {
                "classes": [entity],
                "datatype_properties": {
                    name: {"domain": [entity], "range": []},
                },
            },
        )

        column = result["columns"]["name"]
        self.assertEqual(column["role"], "pk")
        self.assertTrue(column["identity_part"])
        self.assertEqual(column["candidates"][0]["uri"], name)


if __name__ == "__main__":
    unittest.main()
