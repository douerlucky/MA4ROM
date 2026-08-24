"""Namespace safety checks for parent-managed downstream replay."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
MODULE_PATH = PROJECT_ROOT / "replay_from_intermediate.py"


def _load_replay():
    spec = importlib.util.spec_from_file_location("_replay_namespace_test", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_replay_rejects_any_nonempty_destination(tmp_path: Path) -> None:
    replay = _load_replay()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (destination / "generation.log").write_text("prior", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-runner artifacts"):
        replay._prepare_new_output_namespace(source, destination)


def test_runner_replay_allows_only_declared_regular_files(tmp_path: Path) -> None:
    replay = _load_replay()
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (destination / "dataset_record.json").write_text("{}", encoding="utf-8")
    (destination / "generation.log").write_text("", encoding="utf-8")
    (destination / "paper_snapshot_provenance.json").write_text("{}", encoding="utf-8")

    replay._prepare_new_output_namespace(
        source,
        destination,
        allowed_existing_files=frozenset(
            {
                "dataset_record.json",
                "generation.log",
                "paper_snapshot_provenance.json",
            }
        ),
    )

    (destination / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected.json"):
        replay._prepare_new_output_namespace(
            source,
            destination,
            allowed_existing_files=frozenset(
                {
                    "dataset_record.json",
                    "generation.log",
                    "paper_snapshot_provenance.json",
                }
            ),
        )


def test_equivalence_op_checkpoint_is_copied_without_mutating_source(
    tmp_path: Path,
) -> None:
    replay = _load_replay()
    source = tmp_path / "source" / "equivalence_op_module_checkpoint.json"
    destination = tmp_path / "destination" / source.name
    source.parent.mkdir()
    source.write_text(
        '{"task_count": 3, "completed": {"1": {"task_signature": "one"}}}',
        encoding="utf-8",
    )
    source_before = source.read_bytes()

    evidence = replay._stage_equivalence_op_checkpoint(source, destination)

    assert evidence["reused"] is True
    assert evidence["source_completed_count"] == 1
    assert destination.read_bytes() == source_before
    assert source.read_bytes() == source_before


def test_equivalence_op_checkpoint_rejects_empty_or_existing_destination(
    tmp_path: Path,
) -> None:
    replay = _load_replay()
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    source.write_text('{"task_count": 1, "completed": {}}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="no completed task"):
        replay._stage_equivalence_op_checkpoint(source, destination)

    source.write_text(
        '{"task_count": 1, "completed": {"1": {"task_signature": "one"}}}',
        encoding="utf-8",
    )
    destination.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        replay._stage_equivalence_op_checkpoint(source, destination)


def test_equivalence_op_resume_requires_identical_alignment_code_and_model(
    tmp_path: Path,
) -> None:
    replay = _load_replay()
    input_dir = tmp_path / "source"
    input_dir.mkdir()
    checkpoint = input_dir / "equivalence_op_module_checkpoint.json"
    checkpoint.write_text(
        '{"task_count": 1, "completed": {"1": {'
        '"task_signature": "one", "entry": {}, "prediction": {}}}}',
        encoding="utf-8",
    )
    alignment = {"table": {"class_uri": "urn:Class"}}
    replay._write_json(input_dir / "final_alignment.json", alignment)
    op_module = tmp_path / "equivalence_op_module.py"
    op_module.write_text("# stable implementation\n", encoding="utf-8")
    replay._write_json(
        input_dir / "paper_snapshot_provenance.json",
        {
            "model": "deepseek-v4-flash",
            "runtime_reliability": {
                "thinking": "disabled",
                "provider_candidate_models": ["deepseek-v4-flash"],
            },
            "source_files_sha256": {
                "ma4rom/OPMapping/equivalence_op_module.py": replay._sha256(
                    op_module
                )
            },
        },
    )

    evidence = replay._validate_equivalence_op_resume_context(
        source_checkpoint=checkpoint,
        input_dir=input_dir,
        final_alignment=alignment,
        current_op_module=op_module,
        current_model="deepseek-v4-flash",
        current_thinking="disabled",
        current_provider_models=["deepseek-v4-flash"],
        related_source_files={},
    )
    assert evidence["unique_task_signatures"] == 1

    with pytest.raises(RuntimeError, match="final_alignment differs"):
        replay._validate_equivalence_op_resume_context(
            source_checkpoint=checkpoint,
            input_dir=input_dir,
            final_alignment={"changed": True},
            current_op_module=op_module,
            current_model="deepseek-v4-flash",
            current_thinking="disabled",
            current_provider_models=["deepseek-v4-flash"],
            related_source_files={},
        )
