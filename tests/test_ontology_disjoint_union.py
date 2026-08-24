"""Regression tests for OWL disjoint-union hierarchy reasoning."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.ontology_utils import read_ontology  # noqa: E402


class DisjointUnionHierarchyTests(unittest.TestCase):
    def test_class_labels_and_comments_are_preserved_for_semantic_review(self):
        ttl = """
        @prefix : <https://synthetic.invalid#>.
        @prefix owl: <http://www.w3.org/2002/07/owl#>.
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
        :Compact a owl:Class;
            rdfs:label "Compact representation"@en;
            rdfs:comment "Contains only a reduced representation."@en.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.ttl"
            path.write_text(ttl, encoding="utf-8")
            ontology = read_ontology(str(path))

        annotations = ontology["class_annotations"][
            "https://synthetic.invalid#Compact"
        ]
        self.assertEqual(annotations["labels"], ["Compact representation"])
        self.assertEqual(
            annotations["comments"],
            ["Contains only a reduced representation."],
        )

    def test_members_are_subclasses_of_disjoint_union_parent(self):
        ttl = """
        @prefix : <https://synthetic.invalid#>.
        @prefix owl: <http://www.w3.org/2002/07/owl#>.
        :Parent a owl:Class; owl:disjointUnionOf (:Left :Right).
        :Left a owl:Class.
        :Right a owl:Class.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.ttl"
            path.write_text(ttl, encoding="utf-8")
            ontology = read_ontology(str(path))

        parent = "https://synthetic.invalid#Parent"
        left = "https://synthetic.invalid#Left"
        right = "https://synthetic.invalid#Right"
        self.assertIn(parent, ontology["ancestors_of"][left])
        self.assertIn(parent, ontology["ancestors_of"][right])
        self.assertIn(left, ontology["children_of"][parent])
        self.assertIn(right, ontology["children_of"][parent])

    def test_existing_rdfs_edges_and_disjointness_are_preserved(self):
        ttl = """
        @prefix : <https://synthetic.invalid#>.
        @prefix owl: <http://www.w3.org/2002/07/owl#>.
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
        :Parent a owl:Class; owl:disjointUnionOf (:Left :Right).
        :Left a owl:Class; rdfs:subClassOf :Base.
        :Right a owl:Class.
        :Base a owl:Class.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ontology.ttl"
            path.write_text(ttl, encoding="utf-8")
            ontology = read_ontology(str(path))

        left = "https://synthetic.invalid#Left"
        base = "https://synthetic.invalid#Base"
        parent = "https://synthetic.invalid#Parent"
        self.assertIn(base, ontology["ancestors_of"][left])
        self.assertIn(parent, ontology["ancestors_of"][left])


if __name__ == "__main__":
    unittest.main()
