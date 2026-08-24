"""Offline regression tests for RealValue table checkpoint/continuation."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "ma4rom"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RealValue import real_value_enhancement_agent as real_value  # noqa: E402
from RealValue.real_value_checkpoint import (  # noqa: E402
    CHECKPOINT_KIND,
    PHASE_CLASS,
    PHASE_TABLE,
    RealValueCheckpointError,
    RealValueCheckpointSession,
    input_signature,
    portable_input_signature,
    sha256_file,
)
from utils.ontology_utils import read_ontology  # noqa: E402


class _Interrupted(BaseException):
    pass


def _inputs():
    alignment = {
        "alpha": {
            "pattern": "SE",
            "class_uri": "http://example.test/Base",
            "class_confidence": "low",
            "columns": {},
        },
        "beta": {
            "pattern": "SE",
            "class_uri": "http://example.test/Base",
            "class_confidence": "low",
            "columns": {},
        },
    }
    low = {
        "alpha": {"table_low": True, "columns_low": [], "identity_dp_low": []},
        "beta": {"table_low": True, "columns_low": [], "identity_dp_low": []},
    }
    candidates = {
        name: {
            "table_class_candidates": [
                {"uri": f"http://example.test/{name.title()}", "score": 1.0}
            ]
        }
        for name in alignment
    }
    schema = {
        name: {"columns": {"id": "integer"}, "primary_key": ["id"], "foreign_keys": []}
        for name in alignment
    }
    return alignment, low, candidates, schema


def test_session_requires_a_distinct_new_checkpoint_namespace(tmp_path: Path) -> None:
    alignment, low, candidates, schema = _inputs()
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(RealValueCheckpointError, match="must differ"):
        RealValueCheckpointSession(
            alignment=alignment,
            low_conf_report=low,
            candidates=candidates,
            ontology={},
            enriched_schema=schema,
            force_all_context=False,
            implementation_sha256="implementation-v1",
            class_table_order=["alpha", "beta"],
            table_order=["alpha", "beta"],
            checkpoint_path=source,
            resume_checkpoint_path=source,
        )


def test_real_value_resumes_after_last_atomically_completed_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alignment, low, candidates, schema = _inputs()
    source = tmp_path / "failed-run" / "real_value_table_checkpoint.json"
    continuation = tmp_path / "new-run" / "real_value_table_checkpoint.json"

    monkeypatch.setattr(real_value, "_repair_invalid_sh_class_pair", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(real_value, "fetch_sample_rows", lambda *_args, **_kwargs: [{"id": 1}])

    calls: list[str] = []

    def interrupt_on_beta(**kwargs):
        table_name = kwargs["table_name"]
        calls.append(table_name)
        if table_name == "beta":
            raise _Interrupted()
        kwargs["result"][table_name]["class_uri"] = "http://example.test/Alpha"
        kwargs["result"][table_name]["class_confidence"] = "high"

    monkeypatch.setattr(real_value, "_refine_table_class_if_needed", interrupt_on_beta)
    with pytest.raises(_Interrupted):
        real_value.run_real_value_enhancement(
            alignment,
            low,
            candidates,
            ontology={},
            enriched_schema=schema,
            checkpoint_path=source,
        )

    assert calls == ["alpha", "beta"]
    checkpoint = json.loads(source.read_text(encoding="utf-8"))
    assert checkpoint["kind"] == CHECKPOINT_KIND
    assert checkpoint["status"] == "running"
    assert checkpoint["completed"][PHASE_CLASS] == ["alpha"]
    assert checkpoint["completed"][PHASE_TABLE] == []
    assert checkpoint["result"]["alpha"]["class_uri"].endswith("/Alpha")
    immutable_source = source.read_bytes()

    resumed_calls: list[str] = []

    def finish_beta(**kwargs):
        table_name = kwargs["table_name"]
        resumed_calls.append(table_name)
        kwargs["result"][table_name]["class_uri"] = f"http://example.test/{table_name.title()}"
        kwargs["result"][table_name]["class_confidence"] = "high"

    monkeypatch.setattr(real_value, "_refine_table_class_if_needed", finish_beta)
    result = real_value.run_real_value_enhancement(
        alignment,
        low,
        candidates,
        ontology={},
        enriched_schema=schema,
        checkpoint_path=continuation,
        resume_checkpoint_path=source,
    )

    # Alpha's completed Class decision came from the source checkpoint; only
    # Beta's unfinished Class atom is executed by the continuation.
    assert resumed_calls == ["beta"]
    assert result["alpha"]["class_uri"].endswith("/Alpha")
    assert result["beta"]["class_uri"].endswith("/Beta")
    assert source.read_bytes() == immutable_source

    resumed = json.loads(continuation.read_text(encoding="utf-8"))
    assert resumed["status"] == "completed"
    assert resumed["completed"][PHASE_CLASS] == ["alpha", "beta"]
    assert resumed["completed"][PHASE_TABLE] == ["alpha", "beta"]
    assert resumed["resumption"]["source_namespace_mutated"] is False


def test_checkpoint_rejects_changed_semantic_inputs(tmp_path: Path) -> None:
    alignment, low, candidates, schema = _inputs()
    source = tmp_path / "source.json"
    session = RealValueCheckpointSession(
        alignment=alignment,
        low_conf_report=low,
        candidates=candidates,
        ontology={},
        enriched_schema=schema,
        force_all_context=False,
        implementation_sha256="implementation-v1",
        class_table_order=["alpha", "beta"],
        table_order=["alpha", "beta"],
        checkpoint_path=source,
    )
    result = session.initial_result
    for _index, _item in session.iter_phase(
        PHASE_CLASS,
        list(low.items()),
        table_name=lambda item: item[0],
        result=lambda: result,
    ):
        pass
    for _index, _item in session.iter_phase(
        PHASE_TABLE,
        list(low.items()),
        table_name=lambda item: item[0],
        result=lambda: result,
    ):
        pass

    changed_alignment = json.loads(json.dumps(alignment))
    changed_alignment["alpha"]["class_uri"] = "http://example.test/Different"
    with pytest.raises(RealValueCheckpointError, match="inputs or implementation differ"):
        RealValueCheckpointSession(
            alignment=changed_alignment,
            low_conf_report=low,
            candidates=candidates,
            ontology={},
            enriched_schema=schema,
            force_all_context=False,
            implementation_sha256="implementation-v1",
            class_table_order=["alpha", "beta"],
            table_order=["alpha", "beta"],
            checkpoint_path=tmp_path / "new.json",
            resume_checkpoint_path=source,
        )


def test_portable_signature_removes_only_rdflib_blank_node_uuid_prefix() -> None:
    alignment, low, candidates, schema = _inputs()
    ontology_a = {
        "classes": ["http://example.test/Named", "n" + "a" * 32 + "b7"],
        "union_members": {
            "n" + "a" * 32 + "b7": ["http://example.test/Named"]
        },
    }
    ontology_b = {
        "classes": ["http://example.test/Named", "n" + "f" * 32 + "b7"],
        "union_members": {
            "n" + "f" * 32 + "b7": ["http://example.test/Named"]
        },
    }
    kwargs = {
        "alignment": alignment,
        "low_conf_report": low,
        "candidates": candidates,
        "enriched_schema": schema,
        "force_all_context": False,
        "implementation_sha256": "implementation-v1",
    }

    assert input_signature(ontology=ontology_a, **kwargs) != input_signature(
        ontology=ontology_b, **kwargs
    )
    assert portable_input_signature(
        ontology=ontology_a, **kwargs
    ) == portable_input_signature(ontology=ontology_b, **kwargs)

    ontology_b["classes"][0] = "http://example.test/DifferentNamedClass"
    assert portable_input_signature(
        ontology=ontology_a, **kwargs
    ) != portable_input_signature(ontology=ontology_b, **kwargs)


def test_legacy_bnode_checkpoint_bridge_requires_persisted_inputs_and_provenance(
    tmp_path: Path,
) -> None:
    alignment, low, candidates, schema = _inputs()
    source_dir = tmp_path / "legacy-source"
    source_dir.mkdir()
    source = source_dir / "real_value_table_checkpoint.json"
    continuation = tmp_path / "new-run" / "real_value_table_checkpoint.json"
    ontology_path = PROJECT_ROOT / "input" / "npd_atomic_tests" / "ontology.ttl"
    implementation = sha256_file(
        PROJECT_ROOT / "RealValue" / "real_value_enhancement_agent.py"
    )

    RealValueCheckpointSession(
        alignment=alignment,
        low_conf_report=low,
        candidates=candidates,
        ontology=read_ontology(str(ontology_path)),
        enriched_schema=schema,
        force_all_context=False,
        implementation_sha256=implementation,
        class_table_order=["alpha", "beta"],
        table_order=["alpha", "beta"],
        checkpoint_path=source,
    )
    legacy_document = json.loads(source.read_text(encoding="utf-8"))
    legacy_document.pop("portable_input_signature")
    source.write_text(json.dumps(legacy_document), encoding="utf-8")

    for name, value in {
        "dp_mapping_alignment.json": alignment,
        "dp_mapping_candidates.json": candidates,
        "enriched_schema.json": schema,
    }.items():
        (source_dir / name).write_text(json.dumps(value), encoding="utf-8")

    repository_root = PROJECT_ROOT.parent
    base_commit = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relevant_files = (
        "ma4rom/RealValue/real_value_enhancement_agent.py",
        "ma4rom/DPMapping/data_property_mapping_agent.py",
        "ma4rom/utils/ontology_utils.py",
        "ma4rom/config.py",
        "ma4rom/utils/candidate_ranking.py",
        "ma4rom/utils/db_utils.py",
        "ma4rom/utils/llm_client.py",
    )
    provenance = {
        "database": "npd_atomic_tests",
        "base_paper_commit": base_commit,
        "context_enhancement_mode": "confidence",
        "snapshot_root": str(repository_root),
        "source_root": str(PROJECT_ROOT),
        "source_files_sha256": {
            relative: sha256_file(repository_root / relative)
            for relative in relevant_files
        },
    }
    (source_dir / "paper_snapshot_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )

    resumed = RealValueCheckpointSession(
        alignment=alignment,
        low_conf_report=low,
        candidates=candidates,
        ontology=read_ontology(str(ontology_path)),
        enriched_schema=schema,
        force_all_context=False,
        implementation_sha256=implementation,
        class_table_order=["alpha", "beta"],
        table_order=["alpha", "beta"],
        checkpoint_path=continuation,
        resume_checkpoint_path=source,
    )
    assert resumed.resume_compatibility["mode"] == (
        "legacy_rdflib_blank_node_portability_bridge"
    )

    (source_dir / "dp_mapping_alignment.json").write_text(
        json.dumps({"changed": True}), encoding="utf-8"
    )
    with pytest.raises(RealValueCheckpointError, match="inputs or implementation differ"):
        RealValueCheckpointSession(
            alignment=alignment,
            low_conf_report=low,
            candidates=candidates,
            ontology=read_ontology(str(ontology_path)),
            enriched_schema=schema,
            force_all_context=False,
            implementation_sha256=implementation,
            class_table_order=["alpha", "beta"],
            table_order=["alpha", "beta"],
            checkpoint_path=tmp_path / "rejected" / "real_value_table_checkpoint.json",
            resume_checkpoint_path=source,
        )


def test_old_namespace_without_checkpoint_reuses_dp_alignment_for_real_value(
    tmp_path: Path,
) -> None:
    replay_path = PROJECT_ROOT / "replay_from_intermediate.py"
    spec = importlib.util.spec_from_file_location("_real_value_replay_test", replay_path)
    assert spec and spec.loader
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)

    old_namespace = tmp_path / "old-failed-run"
    old_namespace.mkdir()
    saved_candidates = {"table_a": {"table_class_candidates": []}}
    saved_alignment = {"table_a": {"pattern": "SE", "columns": {}}}
    (old_namespace / "dp_mapping_candidates.json").write_text(
        json.dumps(saved_candidates), encoding="utf-8"
    )
    (old_namespace / "dp_mapping_alignment.json").write_text(
        json.dumps(saved_alignment), encoding="utf-8"
    )
    before = {
        path.name: path.read_bytes()
        for path in old_namespace.iterdir()
        if path.is_file()
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("saved DP artefacts must not be recomputed")

    candidates, alignment, reused_candidates, reused_alignment, *_paths = (
        replay._load_or_build_dp_inputs(
            input_dir=old_namespace,
            enriched_schema={"table_a": {}},
            pattern_result={"table_a": "SE"},
            ontology={},
            generate_candidates=forbidden,
            run_data_property_mapping=forbidden,
        )
    )

    assert candidates == saved_candidates
    assert alignment == saved_alignment
    assert reused_candidates is True
    assert reused_alignment is True
    assert {
        path.name: path.read_bytes()
        for path in old_namespace.iterdir()
        if path.is_file()
    } == before


def test_rebuild_dp_ignores_saved_pre_repair_artifacts(tmp_path: Path) -> None:
    replay_path = PROJECT_ROOT / "replay_from_intermediate.py"
    spec = importlib.util.spec_from_file_location("_real_value_rebuild_test", replay_path)
    assert spec and spec.loader
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)

    input_dir = tmp_path / "old-run"
    input_dir.mkdir()
    (input_dir / "dp_mapping_candidates.json").write_text(
        json.dumps({"stale": True}), encoding="utf-8"
    )
    (input_dir / "dp_mapping_alignment.json").write_text(
        json.dumps({"stale": True}), encoding="utf-8"
    )
    calls: list[str] = []

    def build_candidates(*_args, **_kwargs):
        calls.append("candidates")
        return {"fresh": "candidates"}

    def build_alignment(*_args, **_kwargs):
        calls.append("alignment")
        return {"fresh": "alignment"}

    candidates, alignment, reused_candidates, reused_alignment, *_ = (
        replay._load_or_build_dp_inputs(
            input_dir=input_dir,
            enriched_schema={"table_a": {}},
            pattern_result={"table_a": "SE"},
            ontology={},
            generate_candidates=build_candidates,
            run_data_property_mapping=build_alignment,
            rebuild_dp=True,
        )
    )

    assert calls == ["candidates", "alignment"]
    assert candidates == {"fresh": "candidates"}
    assert alignment == {"fresh": "alignment"}
    assert reused_candidates is False
    assert reused_alignment is False


def test_replay_refuses_source_or_nested_output_namespace(tmp_path: Path) -> None:
    replay_path = PROJECT_ROOT / "replay_from_intermediate.py"
    spec = importlib.util.spec_from_file_location("_real_value_replay_path_test", replay_path)
    assert spec and spec.loader
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)

    source = tmp_path / "failed-run"
    source.mkdir()
    with pytest.raises(RuntimeError, match="separate namespace"):
        replay._prepare_new_output_namespace(source, source)
    with pytest.raises(RuntimeError, match="separate namespace"):
        replay._prepare_new_output_namespace(source, source / "continuation")

    sibling = tmp_path / "new-run"
    replay._prepare_new_output_namespace(source, sibling)
    assert sibling.is_dir()
