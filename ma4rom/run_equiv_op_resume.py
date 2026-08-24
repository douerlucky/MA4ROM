#!/usr/bin/env python3
"""Resume the pipeline from ObjectProperty mapping to final R2RML.

This entry point intentionally does not rerun FKCompletion / DP Mapping / RealValue enhancement.
It reads existing intermediate files:
  - output/<db>/final_alignment.json
  - output/<db>/enriched_schema.json

Then it runs the active equivalence-column OP module and generates the final TTL.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required intermediate file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run only the active equivalence-column OP module and R2RML generation."
    )
    parser.add_argument(
        "--database",
        help="Database/output folder name. Equivalent to MAMG_CURRENT_DATABASE.",
    )
    parser.add_argument(
        "--min-endpoint-score",
        type=float,
        default=None,
        help="Minimum domain/range endpoint score for candidate pruning.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=None,
        help="Sleep seconds between LLM calls.",
    )
    args = parser.parse_args()

    if args.database:
        os.environ["MAMG_CURRENT_DATABASE"] = args.database

    # Import after env vars are set, because config.py is evaluated at import time.
    from config import (  # noqa: WPS433
        CURRENT_DATABASE,
        DB_SCHEMA_NAME,
        MAPPING_BASE_URL,
        ONTOLOGY_PATH,
        OUTPUT_DIR,
        OUTPUT_MAPPING_FILENAME,
    )
    from OPMapping.equivalence_op_module import run_equivalence_op_module  # noqa: WPS433
    from r2rml_generator import generate_r2rml  # noqa: WPS433
    from utils.ontology_utils import read_ontology  # noqa: WPS433

    output_dir = Path(OUTPUT_DIR)
    final_alignment = _read_json(output_dir / "final_alignment.json")
    enriched_schema = _read_json(output_dir / "enriched_schema.json")
    ontology = read_ontology(ONTOLOGY_PATH)

    print("=" * 60)
    print("Resume from OP mapping")
    print("=" * 60)
    print(f"  database: {CURRENT_DATABASE}")
    print(f"  output:   {output_dir}")
    print("  input:    final_alignment.json + enriched_schema.json")
    print("  OP mapping: equivalence-column module")

    op_mapping_step1 = run_equivalence_op_module(
        final_alignment=final_alignment,
        ontology=ontology,
        enriched_schema=enriched_schema,
        schema_name=DB_SCHEMA_NAME,
        output_dir=OUTPUT_DIR,
        ontology_path=ONTOLOGY_PATH,
        min_endpoint_score=args.min_endpoint_score,
        sleep_seconds=args.sleep,
    )
    op_mapping_full = {
        "step1": op_mapping_step1,
        "step2_orphans": [],
    }
    (output_dir / "op_mapping_step1_result.json").write_text(
        json.dumps(op_mapping_step1, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "op_mapping_step2_result.json").write_text(
        json.dumps(
            {
                "orphan_matches": [],
                "skipped": True,
                "reason": "Legacy OP Step2 orphan completion has been removed.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "op_mapping_full_result.json").write_text(
        json.dumps(op_mapping_full, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    r2rml = generate_r2rml(
        final_alignment=final_alignment,
        op_mapping_full=op_mapping_full,
        enriched_schema=enriched_schema,
        ontology=ontology,
        base_url=MAPPING_BASE_URL,
        prefix=CURRENT_DATABASE.replace("_", ""),
    )
    ttl_path = output_dir / OUTPUT_MAPPING_FILENAME
    ttl_path.write_text(r2rml, encoding="utf-8")

    print("\nDone.")
    print(f"  OP results: {output_dir / 'equivalence_op_module_predictions.json'}")
    print(f"  OP shape:   {output_dir / 'op_mapping_step1_result.json'}")
    print(f"  TTL:        {ttl_path}")
    print(f"  TTL chars:  {len(r2rml)}")


if __name__ == "__main__":
    main()
