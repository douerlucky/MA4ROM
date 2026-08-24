#!/usr/bin/env python3
"""Replay mapping stages from saved schema/pattern intermediates.

This is intentionally narrower than the full pipeline.  The input directory is
always immutable and the output directory must be a distinct new namespace.  A
saved DP alignment is reused when available; RealValue is then either resumed
from an explicit table checkpoint or honestly recomputed from that alignment.
An explicit immutable EquivOP checkpoint may also be staged into the new
namespace.  The OP module revalidates every saved task by semantic signature,
so only matching completed work is reused before OP and R2RML finish normally.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import shutil
from pathlib import Path


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing replay input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage_equivalence_op_checkpoint(
    source: Path | None,
    destination: Path,
) -> dict[str, object]:
    """Atomically seed a new namespace from an immutable EquivOP checkpoint.

    The checkpoint contains task signatures rather than a trusted final OP
    result.  ``run_equivalence_op_module`` will rebuild the current task list
    and reuse only entries whose semantic signatures still match.
    """
    if source is None:
        return {
            "reused": False,
            "source_checkpoint": None,
            "source_checkpoint_sha256": None,
            "source_completed_count": 0,
            "source_task_count": 0,
            "destination_checkpoint": str(destination),
        }

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing EquivOP continuation checkpoint: {source}")
    if source == destination:
        raise RuntimeError("EquivOP continuation source and destination must differ")
    if destination.exists():
        raise RuntimeError(
            "Refusing to overwrite an existing EquivOP checkpoint: "
            + str(destination)
        )

    payload = source.read_bytes()
    try:
        checkpoint = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "EquivOP continuation checkpoint is malformed: " + str(source)
        ) from exc
    completed = checkpoint.get("completed")
    task_count = checkpoint.get("task_count")
    if not isinstance(completed, dict) or not completed:
        raise RuntimeError(
            "EquivOP continuation checkpoint has no completed task entries: "
            + str(source)
        )
    if not isinstance(task_count, int) or task_count < len(completed):
        raise RuntimeError(
            "EquivOP continuation checkpoint has an invalid task_count: "
            + str(source)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    source_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    destination_sha256 = _sha256(destination)
    if destination_sha256 != source_sha256:
        raise RuntimeError("EquivOP checkpoint copy failed its SHA-256 validation")
    return {
        "reused": True,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha256,
        "source_completed_count": len(completed),
        "source_task_count": task_count,
        "destination_checkpoint": str(destination),
        "staged_checkpoint_sha256": destination_sha256,
    }


def _reuse_completed_real_value_checkpoint(
    *,
    source_checkpoint: Path,
    input_dir: Path,
    destination_checkpoint: Path,
    current_implementation: Path,
) -> tuple[dict, dict[str, object]]:
    """Reuse a completed RealValue result without replaying paid decisions.

    A completed checkpoint is a final stage boundary, not a partial iterator.
    Reusing its exact result avoids recomputing an input signature whose
    ontology collections may have nondeterministic list order across Python
    processes.  Every stable boundary is still bound and verified below.
    """
    source_checkpoint = source_checkpoint.expanduser().resolve()
    input_dir = input_dir.expanduser().resolve()
    destination_checkpoint = destination_checkpoint.expanduser().resolve()
    if source_checkpoint.parent != input_dir:
        raise RuntimeError(
            "RealValue checkpoint must come from the same immutable replay namespace"
        )
    checkpoint = _read_json(source_checkpoint)
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("Unsupported completed RealValue checkpoint schema")
    if checkpoint.get("kind") != "ma4rom_real_value_table_checkpoint":
        raise RuntimeError("Unexpected completed RealValue checkpoint kind")
    if checkpoint.get("status") != "completed":
        raise RuntimeError("RealValue direct reuse requires a completed checkpoint")
    current_implementation_sha256 = _sha256(current_implementation)
    if checkpoint.get("implementation_sha256") != current_implementation_sha256:
        raise RuntimeError(
            "Completed RealValue checkpoint implementation hash changed"
        )
    completed = checkpoint.get("completed") or {}
    class_order = checkpoint.get("class_table_order") or []
    table_order = checkpoint.get("table_order") or []
    if (
        completed.get("class_confirmation") != class_order
        or completed.get("table_refinement") != table_order
    ):
        raise RuntimeError("Completed RealValue checkpoint has incomplete phase prefixes")
    result = checkpoint.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("Completed RealValue checkpoint lacks its final alignment")
    source_alignment_path = input_dir / "final_alignment.json"
    if _read_json(source_alignment_path) != result:
        raise RuntimeError(
            "Completed RealValue checkpoint result differs from source final_alignment"
        )
    if destination_checkpoint.exists():
        raise RuntimeError(
            "Refusing to overwrite an existing RealValue checkpoint destination"
        )

    source_sha256 = _sha256(source_checkpoint)
    destination_document = deepcopy(checkpoint)
    destination_document["resumption"] = {
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_namespace_mutated": False,
        "mode": "completed_result_reuse",
    }
    _write_json(destination_checkpoint, destination_document)
    return deepcopy(result), {
        "mode": "completed_checkpoint_result_reuse",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": source_sha256,
        "source_final_alignment": str(source_alignment_path),
        "source_final_alignment_sha256": _sha256(source_alignment_path),
        "destination_checkpoint": str(destination_checkpoint),
        "destination_checkpoint_sha256": _sha256(destination_checkpoint),
        "implementation_sha256": current_implementation_sha256,
        "completed_counts": {
            "class_confirmation": len(class_order),
            "table_refinement": len(table_order),
        },
    }


def _validate_equivalence_op_resume_context(
    *,
    source_checkpoint: Path,
    input_dir: Path,
    final_alignment: dict,
    current_op_module: Path,
    current_model: str,
    current_thinking: str,
    current_provider_models: list[str],
    related_source_files: dict[str, Path],
) -> dict[str, object]:
    """Refuse OP reuse unless its semantic construction context is unchanged."""
    source_checkpoint = source_checkpoint.expanduser().resolve()
    input_dir = input_dir.expanduser().resolve()
    if source_checkpoint.parent != input_dir:
        raise RuntimeError(
            "EquivOP checkpoint must come from the same immutable replay namespace"
        )

    source_alignment_path = input_dir / "final_alignment.json"
    source_provenance_path = input_dir / "paper_snapshot_provenance.json"
    source_alignment = _read_json(source_alignment_path)
    if source_alignment != final_alignment:
        raise RuntimeError(
            "EquivOP checkpoint context mismatch: resumed final_alignment differs "
            "from the source namespace"
        )

    provenance = _read_json(source_provenance_path)
    source_hashes = provenance.get("source_files_sha256") or {}
    source_op_hash = source_hashes.get(
        "ma4rom/OPMapping/equivalence_op_module.py"
    )
    current_op_hash = _sha256(current_op_module)
    if not source_op_hash or source_op_hash != current_op_hash:
        raise RuntimeError(
            "EquivOP checkpoint context mismatch: OP implementation hash changed"
        )
    source_model = str(provenance.get("model") or "")
    if source_model != str(current_model or ""):
        raise RuntimeError(
            "EquivOP checkpoint context mismatch: provider model changed"
        )
    runtime = provenance.get("runtime_reliability") or {}
    if str(runtime.get("thinking") or "") != str(current_thinking or ""):
        raise RuntimeError(
            "EquivOP checkpoint context mismatch: thinking mode changed"
        )
    if list(runtime.get("provider_candidate_models") or []) != list(
        current_provider_models
    ):
        raise RuntimeError(
            "EquivOP checkpoint context mismatch: provider fallback set changed"
        )
    related_hashes: dict[str, str] = {}
    for provenance_name, current_path in related_source_files.items():
        expected = source_hashes.get(provenance_name)
        observed = _sha256(current_path)
        if not expected or expected != observed:
            raise RuntimeError(
                "EquivOP checkpoint context mismatch: related source changed: "
                + provenance_name
            )
        related_hashes[provenance_name] = observed

    checkpoint = _read_json(source_checkpoint)
    saved_entries = list((checkpoint.get("completed") or {}).values())
    signatures = [
        str(entry.get("task_signature") or "")
        for entry in saved_entries
        if isinstance(entry, dict)
    ]
    if (
        len(signatures) != len(saved_entries)
        or any(not signature for signature in signatures)
        or len(set(signatures)) != len(signatures)
    ):
        raise RuntimeError(
            "EquivOP checkpoint contains missing or duplicate task signatures"
        )
    if any(
        (entry.get("entry") or {}).get("terminal_error")
        or (entry.get("prediction") or {}).get("terminal_error")
        or (entry.get("entry") or {}).get("error")
        or (entry.get("prediction") or {}).get("error")
        or (entry.get("entry") or {}).get("fallback_used")
        or (entry.get("prediction") or {}).get("fallback_used")
        for entry in saved_entries
        if isinstance(entry, dict)
    ):
        raise RuntimeError(
            "EquivOP checkpoint contains failed task entries; explicit recovery is required"
        )

    return {
        "source_final_alignment": str(source_alignment_path),
        "source_final_alignment_sha256": _sha256(source_alignment_path),
        "source_provenance": str(source_provenance_path),
        "source_op_module_sha256": source_op_hash,
        "current_op_module_sha256": current_op_hash,
        "model": source_model,
        "thinking": current_thinking,
        "provider_candidate_models": list(current_provider_models),
        "related_source_sha256": related_hashes,
        "unique_task_signatures": len(signatures),
    }


def _audit_equivalence_op_task_reuse(
    *,
    op_module,
    checkpoint_path: Path,
    final_alignment: dict,
    ontology: dict,
    enriched_schema: dict,
    schema_name: str,
    output_dir: Path,
) -> dict[str, object]:
    """Prove actual task-signature reuse before the first paid OP request."""
    checkpoint = _read_json(checkpoint_path)
    saved_entries = list((checkpoint.get("completed") or {}).values())
    saved_by_signature = op_module._checkpoint_entries_by_signature(checkpoint)
    fk_tasks, _ = op_module._build_fk_tasks(
        enriched_schema,
        final_alignment,
        ontology,
        schema_name,
    )
    sr_tasks, _ = op_module._build_sr_tasks(
        final_alignment,
        enriched_schema,
        ontology,
        schema_name,
    )
    tasks = fk_tasks + sr_tasks
    current_signatures = [
        op_module._task_checkpoint_signature(task) for task in tasks
    ]
    matched_indices = [
        index
        for index, signature in enumerate(current_signatures, start=1)
        if signature in saved_by_signature
    ]
    source_task_count = int(checkpoint.get("task_count") or 0)
    if source_task_count != len(tasks):
        raise RuntimeError(
            "EquivOP checkpoint task-count mismatch before paid continuation"
        )
    if len(saved_by_signature) != len(saved_entries):
        raise RuntimeError(
            "EquivOP checkpoint signatures are missing or non-unique"
        )
    if len(matched_indices) != len(saved_entries):
        raise RuntimeError(
            "EquivOP checkpoint did not match every saved task in current context"
        )
    first_new_task_index = next(
        (
            index
            for index in range(1, len(tasks) + 1)
            if index not in set(matched_indices)
        ),
        None,
    )
    audit = {
        "status": "validated_before_paid_requests",
        "task_count": len(tasks),
        "source_completed_count": len(saved_entries),
        "matched_task_count": len(matched_indices),
        "mismatched_signature_count": 0,
        "rejected_error_terminal_or_fallback_count": 0,
        "matched_task_indices": matched_indices,
        "first_new_task_index": first_new_task_index,
        "planned_new_task_count": len(tasks) - len(matched_indices),
    }
    _write_json(output_dir / "equivalence_op_resume_audit.json", audit)
    return audit


def _load_or_build_dp_inputs(
    *,
    input_dir: Path,
    enriched_schema: dict,
    pattern_result: dict,
    ontology: dict,
    generate_candidates,
    run_data_property_mapping,
    rebuild_dp: bool = False,
) -> tuple[dict, dict, bool, bool, Path, Path]:
    """Prefer saved DP boundaries; rebuild only the artefact that is absent."""

    candidates_path = input_dir / "dp_mapping_candidates.json"
    if candidates_path.is_file() and not rebuild_dp:
        candidates = _read_json(candidates_path)
        reused_candidates = True
    else:
        candidates = generate_candidates(enriched_schema, pattern_result, ontology)
        reused_candidates = False

    alignment_path = input_dir / "dp_mapping_alignment.json"
    if alignment_path.is_file() and not rebuild_dp:
        alignment = _read_json(alignment_path)
        reused_alignment = True
    else:
        alignment = run_data_property_mapping(candidates, ontology=ontology)
        reused_alignment = False
    return (
        candidates,
        alignment,
        reused_candidates,
        reused_alignment,
        candidates_path,
        alignment_path,
    )


def _prepare_new_output_namespace(
    input_dir: Path,
    output_dir: Path,
    *,
    allowed_existing_files: frozenset[str] = frozenset(),
) -> None:
    """Create a continuation namespace without touching its source.

    A camera-ready parent runner creates its dataset record and generation log
    before starting this child.  Those runner-owned files may be admitted only
    through the explicit allow-list; direct replay remains empty-directory
    only.  Directories, symlinks, or any other prior artifact are rejected.
    """

    if (
        output_dir == input_dir
        or output_dir.is_relative_to(input_dir)
        or input_dir.is_relative_to(output_dir)
    ):
        raise RuntimeError(
            "Replay output must be a separate namespace, not the source or its parent/child"
        )
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        unexpected = [
            path
            for path in existing
            if path.name not in allowed_existing_files or not path.is_file()
        ]
        if unexpected:
            raise RuntimeError(
                "Refusing to reuse a replay output namespace containing "
                "non-runner artifacts: "
                + ", ".join(str(path) for path in sorted(unexpected))
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate MA4ROM downstream stages from saved schema/pattern inputs."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Immutable prior namespace containing schema/pattern and optional DP files.",
    )
    parser.add_argument(
        "--database",
        required=True,
        help="Logical dataset/database name used to select its ontology and mapping prefix.",
    )
    parser.add_argument(
        "--source-db-schema",
        required=True,
        help="PostgreSQL source schema used for RealValue sampling and generated mapping execution.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="New continuation namespace; defaults to config.OUTPUT_DIR.",
    )
    parser.add_argument(
        "--resume-real-value-checkpoint",
        default=None,
        help=(
            "Explicit immutable RealValue table checkpoint to continue.  If omitted, "
            "RealValue restarts from the saved DP alignment."
        ),
    )
    parser.add_argument(
        "--resume-equivalence-op-checkpoint",
        default=None,
        help=(
            "Explicit immutable EquivOP task checkpoint to continue in the new "
            "namespace; semantic task signatures are revalidated before reuse."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Disable all external LLM requests; use existing conservative fallbacks only.",
    )
    parser.add_argument(
        "--rebuild-dp",
        action="store_true",
        help=(
            "Rebuild candidates and DP alignment from immutable schema/pattern inputs "
            "instead of reusing saved pre-repair DP artefacts."
        ),
    )
    return parser.parse_args()


def run_replay(args: argparse.Namespace) -> Path:
    """Execute one downstream replay and return its generated mapping path."""
    # Apply generic runtime identity before importing config or any MA4ROM
    # module.  Both values are explicit CLI evidence; neither is inferred from
    # a dataset name or target score.
    os.environ["MAMG_CURRENT_DATABASE"] = args.database
    os.environ["MAMG_DB_SCHEMA"] = args.source_db_schema
    if args.offline:
        # Set before importing MA4ROM modules, whose config/client are loaded
        # at import time.  No key is read, persisted, or printed.
        os.environ["MAMG_LLM_OFFLINE"] = "true"

    from config import (  # noqa: WPS433
        CURRENT_DATABASE,
        DB_SCHEMA_NAME,
        EQUIV_OP_LLM_SLEEP_SECONDS,
        EQUIV_OP_MIN_ENDPOINT_SCORE,
        LLM_FALLBACK_MODELS,
        LLM_MODEL,
        LLM_THINKING_ENABLED,
        MAPPING_BASE_URL,
        ONTOLOGY_PATH,
        OUTPUT_DIR,
        OUTPUT_MAPPING_FILENAME,
    )
    from candidate_generation import generate_candidates  # noqa: WPS433
    from DPMapping.data_property_mapping_agent import (  # noqa: WPS433
        collect_low_confidence_data_property_mappings,
        run_data_property_mapping,
    )
    from OPMapping import equivalence_op_module as op_module  # noqa: WPS433
    from RealValue.real_value_enhancement_agent import run_real_value_enhancement  # noqa: WPS433
    from r2rml_generator import generate_r2rml  # noqa: WPS433
    from utils.db_utils import get_conn  # noqa: WPS433
    from utils.ontology_utils import read_ontology  # noqa: WPS433

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(OUTPUT_DIR).resolve()
    runner_managed_output = bool(getattr(args, "runner_managed_output", False))
    allowed_existing_files = (
        frozenset(
            {
                "dataset_record.json",
                "generation.log",
                "paper_snapshot_provenance.json",
            }
        )
        if runner_managed_output
        else frozenset()
    )
    _prepare_new_output_namespace(
        input_dir,
        output_dir,
        allowed_existing_files=allowed_existing_files,
    )

    resume_checkpoint = (
        Path(args.resume_real_value_checkpoint).expanduser().resolve()
        if args.resume_real_value_checkpoint
        else None
    )
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(f"Missing RealValue continuation checkpoint: {resume_checkpoint}")
    resume_op_checkpoint = (
        Path(args.resume_equivalence_op_checkpoint).expanduser().resolve()
        if getattr(args, "resume_equivalence_op_checkpoint", None)
        else None
    )
    if resume_op_checkpoint is not None and not resume_op_checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing EquivOP continuation checkpoint: {resume_op_checkpoint}"
        )
    if resume_op_checkpoint is not None:
        if resume_checkpoint is None:
            raise RuntimeError(
                "EquivOP continuation requires a completed RealValue checkpoint"
            )
        if bool(args.rebuild_dp):
            raise RuntimeError(
                "EquivOP continuation cannot be combined with DP rebuilding"
            )
        if (
            resume_checkpoint.parent != input_dir
            or resume_op_checkpoint.parent != input_dir
        ):
            raise RuntimeError(
                "RealValue and EquivOP checkpoints must share the replay input namespace"
            )

    schema_path = input_dir / "enriched_schema.json"
    pattern_path = input_dir / "pattern_result.json"
    enriched_schema = _read_json(schema_path)
    pattern_result = _read_json(pattern_path)
    ontology = read_ontology(ONTOLOGY_PATH)

    # The scenario name and PostgreSQL schema are not always the same.  A
    # forced but nonexistent schema makes OP equivalence evidence silently
    # empty, so fail before producing a deceptively successful replay.
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = %s AND table_type = 'BASE TABLE'",
                (DB_SCHEMA_NAME,),
            )
            configured_tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    required_tables = set(enriched_schema)
    missing_tables = sorted(required_tables - configured_tables)
    if missing_tables:
        raise RuntimeError(
            "Configured PostgreSQL schema does not contain the replay input tables: "
            f"schema={DB_SCHEMA_NAME!r}, missing={missing_tables[:8]}. "
            "Use the schema printed by setup_dataset.py; do not assume it is the scenario name."
        )

    print("=" * 72)
    print("MA4ROM downstream replay")
    print(f"  database: {CURRENT_DATABASE}")
    print(f"  schema:   {DB_SCHEMA_NAME}")
    print(f"  input:    {input_dir}")
    print(f"  output:   {output_dir}")
    print(f"  offline:  {args.offline}")
    print("=" * 72)

    _write_json(output_dir / "enriched_schema.json", enriched_schema)
    _write_json(output_dir / "pattern_result.json", pattern_result)

    (
        candidates,
        alignment,
        reused_candidates,
        reused_alignment,
        candidates_path,
        alignment_path,
    ) = _load_or_build_dp_inputs(
        input_dir=input_dir,
        enriched_schema=enriched_schema,
        pattern_result=pattern_result,
        ontology=ontology,
        generate_candidates=generate_candidates,
        run_data_property_mapping=run_data_property_mapping,
        rebuild_dp=bool(args.rebuild_dp),
    )
    _write_json(output_dir / "dp_mapping_candidates.json", candidates)

    _write_json(output_dir / "dp_mapping_alignment.json", alignment)
    low_confidence = collect_low_confidence_data_property_mappings(alignment)
    _write_json(output_dir / "dp_mapping_low_confidence.json", low_confidence)

    real_value_checkpoint = output_dir / "real_value_table_checkpoint.json"
    real_value_reuse_evidence: dict[str, object] = {}
    source_checkpoint_document = (
        _read_json(resume_checkpoint) if resume_checkpoint is not None else {}
    )
    if resume_checkpoint is not None and source_checkpoint_document.get(
        "status"
    ) == "completed":
        final_alignment, real_value_reuse_evidence = (
            _reuse_completed_real_value_checkpoint(
                source_checkpoint=resume_checkpoint,
                input_dir=input_dir,
                destination_checkpoint=real_value_checkpoint,
                current_implementation=(
                    Path(__file__).resolve().parent
                    / "RealValue"
                    / "real_value_enhancement_agent.py"
                ),
            )
        )
    else:
        final_alignment = run_real_value_enhancement(
            alignment,
            low_confidence,
            candidates,
            ontology=ontology,
            enriched_schema=enriched_schema,
            checkpoint_path=real_value_checkpoint,
            resume_checkpoint_path=resume_checkpoint,
        )
    _write_json(output_dir / "final_alignment.json", final_alignment)

    # Seed only an explicitly supplied immutable checkpoint.  The OP module
    # reconstructs current tasks and accepts saved entries by semantic
    # signature, so changed/mismatched tasks are paid and evaluated afresh.
    op_checkpoint_path = output_dir / "equivalence_op_module_checkpoint.json"
    op_resume_context = (
        _validate_equivalence_op_resume_context(
            source_checkpoint=resume_op_checkpoint,
            input_dir=input_dir,
            final_alignment=final_alignment,
            current_op_module=(
                Path(__file__).resolve().parent
                / "OPMapping"
                / "equivalence_op_module.py"
            ),
            current_model=LLM_MODEL,
            current_thinking=(
                "enabled" if bool(LLM_THINKING_ENABLED) else "disabled"
            ),
            current_provider_models=[LLM_MODEL, *list(LLM_FALLBACK_MODELS)],
            related_source_files={
                "ma4rom/config.py": Path(__file__).resolve().parent / "config.py",
                "ma4rom/utils/db_utils.py": (
                    Path(__file__).resolve().parent / "utils" / "db_utils.py"
                ),
                "ma4rom/utils/ontology_utils.py": (
                    Path(__file__).resolve().parent / "utils" / "ontology_utils.py"
                ),
            },
        )
        if resume_op_checkpoint is not None
        else {}
    )
    op_checkpoint_evidence = _stage_equivalence_op_checkpoint(
        resume_op_checkpoint,
        op_checkpoint_path,
    )
    # Keep the exact source OP implementation hash for semantic compatibility,
    # while replacing only its persistence primitive with this module's
    # flush+fsync+replace writer.
    op_module._write_json = lambda data, path: _write_json(Path(path), data)
    op_task_reuse_audit = (
        _audit_equivalence_op_task_reuse(
            op_module=op_module,
            checkpoint_path=op_checkpoint_path,
            final_alignment=final_alignment,
            ontology=ontology,
            enriched_schema=enriched_schema,
            schema_name=DB_SCHEMA_NAME,
            output_dir=output_dir,
        )
        if resume_op_checkpoint is not None
        else {}
    )
    if resume_op_checkpoint is not None:
        os.environ["MAMG_EQUIV_OP_RESUME"] = "true"
    op_step1 = op_module.run_equivalence_op_module(
        final_alignment=final_alignment,
        ontology=ontology,
        enriched_schema=enriched_schema,
        schema_name=DB_SCHEMA_NAME,
        output_dir=str(output_dir),
        ontology_path=ONTOLOGY_PATH,
        min_endpoint_score=EQUIV_OP_MIN_ENDPOINT_SCORE,
        sleep_seconds=EQUIV_OP_LLM_SLEEP_SECONDS,
    )
    op_full = {"step1": op_step1, "step2_orphans": []}
    _write_json(output_dir / "op_mapping_step1_result.json", op_step1)
    _write_json(
        output_dir / "op_mapping_step2_result.json",
        {
            "orphan_matches": [],
            "skipped": True,
            "reason": "Replay uses the active equivalence-column OP module only.",
        },
    )
    _write_json(output_dir / "op_mapping_full_result.json", op_full)

    mapping = generate_r2rml(
        final_alignment=final_alignment,
        op_mapping_full=op_full,
        enriched_schema=enriched_schema,
        ontology=ontology,
        base_url=MAPPING_BASE_URL,
        prefix=CURRENT_DATABASE.replace("_", ""),
    )
    mapping_path = output_dir / OUTPUT_MAPPING_FILENAME
    mapping_path.write_text(mapping, encoding="utf-8")

    # Preserve the exact ontology used for evaluation alongside the generated
    # mapping.  The evaluator creates ontology.properties itself.
    ontology_copy = output_dir / f"rodi_{CURRENT_DATABASE}_generated_ontology.ttl"
    shutil.copyfile(ONTOLOGY_PATH, ontology_copy)
    input_hashes = {
        "enriched_schema.json": _sha256(schema_path),
        "pattern_result.json": _sha256(pattern_path),
    }
    if candidates_path.is_file():
        input_hashes["dp_mapping_candidates.json"] = _sha256(candidates_path)
    if alignment_path.is_file():
        input_hashes["dp_mapping_alignment.json"] = _sha256(alignment_path)

    _write_json(
        output_dir / "replay_manifest.json",
        {
            "database": CURRENT_DATABASE,
            "schema": DB_SCHEMA_NAME,
            "offline": bool(args.offline),
            "input_dir": str(input_dir),
            "input_namespace_policy": "read_only",
            "input_sha256": input_hashes,
            "ontology_path": ONTOLOGY_PATH,
            "ontology_sha256": _sha256(Path(ONTOLOGY_PATH)),
            "reused_candidates": reused_candidates,
            "reused_alignment": reused_alignment,
            "rebuild_dp": bool(args.rebuild_dp),
            "real_value": {
                "mode": (
                    "checkpoint_continuation"
                    if resume_checkpoint is not None
                    else (
                        "recomputed_from_saved_dp_alignment"
                        if reused_alignment
                        else "fresh_downstream_replay"
                    )
                ),
                "source_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
                "source_checkpoint_sha256": _sha256(resume_checkpoint) if resume_checkpoint else None,
                "destination_checkpoint": str(real_value_checkpoint),
                "destination_checkpoint_sha256": _sha256(real_value_checkpoint),
                "completed_reuse": real_value_reuse_evidence,
            },
            "equivalence_op": {
                **op_checkpoint_evidence,
                "compatibility": op_resume_context,
                "task_reuse_audit": op_task_reuse_audit,
                "destination_checkpoint_sha256": (
                    _sha256(op_checkpoint_path)
                    if op_checkpoint_path.is_file()
                    else None
                ),
            },
            "reused_op_checkpoint": bool(op_checkpoint_evidence["reused"]),
        },
    )
    print(f"Replay completed: {mapping_path}")
    return mapping_path


def main() -> None:
    run_replay(parse_args())


if __name__ == "__main__":
    main()
