"""The discriminator shortcut must be structural, never dataset-name based."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DPMapping.data_property_mapping_agent import _is_discriminator


def test_role_detection_uses_generic_tokens_only() -> None:
    # A domain-specific column name must remain a normal data attribute here.
    # RealValue may later add an rdf:type assertion from measured values and
    # ontology evidence; the DP stage must not bake in a dataset vocabulary.
    assert not _is_discriminator("wlbDiscoveryWellbore")
    assert not _is_discriminator("plotSymbol")

    assert _is_discriminator("well_type")
    assert _is_discriminator("facility_kind")
    assert _is_discriminator("category")

