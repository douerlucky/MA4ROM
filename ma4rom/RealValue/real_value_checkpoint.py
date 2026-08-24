"""Crash-safe, table-granular checkpoints for the RealValue stage.

The checkpoint records the two ordering-sensitive passes used by RealValue:
table-Class confirmation first, then per-table value/column refinement.  A
continuation reads one checkpoint as immutable input and writes progress to a
different path.  This prevents a failed experiment namespace from being
silently mutated while still avoiding repeated paid decisions after an
interruption.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_KIND = "ma4rom_real_value_table_checkpoint"
PHASE_CLASS = "class_confirmation"
PHASE_TABLE = "table_refinement"
PHASES = (PHASE_CLASS, PHASE_TABLE)

# RDFLib deliberately gives blank nodes process-local identifiers.  Turtle
# parses currently use ``n<uuid>b<ordinal>``; the UUID changes on every parse
# while the parser-local ordinal is stable for the same immutable graph.  A
# process-local identifier must never make an otherwise identical checkpoint
# unusable in a later process.
_RDFLIB_BNODE_ID = re.compile(r"^n[0-9a-fA-F]{32}(b[0-9]+)$")


class RealValueCheckpointError(ValueError):
    """The checkpoint cannot be safely applied to the requested replay."""


def _canonicalize(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation.

    Ontology helpers occasionally expose sets or tuples.  Sorting their
    canonical representations keeps the input fingerprint stable without
    depending on Python hash iteration order.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonicalize(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable_ontology_value(value: Any) -> Any:
    """Normalize only RDFLib's process-local blank-node UUID prefix.

    Named URIs, labels, comments, list order, and every other ontology value
    remain part of the signature.  Keeping the parser-local ordinal preserves
    references between anonymous expressions without treating a random UUID as
    ontology semantics.
    """

    if isinstance(value, str):
        match = _RDFLIB_BNODE_ID.fullmatch(value)
        return f"_:rdflib:{match.group(1)}" if match else value
    if isinstance(value, dict):
        return {
            str(_portable_ontology_value(key)): _portable_ontology_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_portable_ontology_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_portable_ontology_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(_portable_ontology_value(item) for item in value)
    return value


def _contains_rdflib_blank_node(value: Any) -> bool:
    """Return whether a nested ontology payload contains a process-local ID."""

    if isinstance(value, str):
        return _RDFLIB_BNODE_ID.fullmatch(value) is not None
    if isinstance(value, dict):
        return any(
            _contains_rdflib_blank_node(key)
            or _contains_rdflib_blank_node(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_rdflib_blank_node(item) for item in value)
    return False


def portable_input_signature(
    *,
    alignment: dict,
    low_conf_report: dict,
    candidates: dict,
    ontology: dict | None,
    enriched_schema: dict | None,
    force_all_context: bool,
    implementation_sha256: str,
) -> str:
    """Fingerprint RealValue inputs without RDFLib's random BNode UUIDs."""

    return sha256_json(
        {
            "alignment": alignment,
            "low_conf_report": low_conf_report,
            "candidates": candidates,
            "ontology": _portable_ontology_value(ontology or {}),
            "enriched_schema": enriched_schema or {},
            "force_all_context": bool(force_all_context),
            "implementation_sha256": implementation_sha256,
        }
    )


def input_signature(
    *,
    alignment: dict,
    low_conf_report: dict,
    candidates: dict,
    ontology: dict | None,
    enriched_schema: dict | None,
    force_all_context: bool,
    implementation_sha256: str,
) -> str:
    """Fingerprint every semantic input that can affect RealValue output."""

    return sha256_json(
        {
            "alignment": alignment,
            "low_conf_report": low_conf_report,
            "candidates": candidates,
            "ontology": ontology or {},
            "enriched_schema": enriched_schema or {},
            "force_all_context": bool(force_all_context),
            "implementation_sha256": implementation_sha256,
        }
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RealValueCheckpointError(f"RealValue checkpoint does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RealValueCheckpointError(f"Malformed RealValue checkpoint JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RealValueCheckpointError(f"RealValue checkpoint must be a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Publish one complete checkpoint with replace-on-success semantics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validated_prefix(
    *,
    checkpoint: dict[str, Any],
    phase: str,
    expected_order: Sequence[str],
    path: Path,
) -> list[str]:
    completed = checkpoint.get("completed")
    if not isinstance(completed, dict):
        raise RealValueCheckpointError(f"Checkpoint lacks completed phases: {path}")
    value = completed.get(phase)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RealValueCheckpointError(f"Malformed completed phase {phase!r}: {path}")
    expected_prefix = list(expected_order[: len(value)])
    if value != expected_prefix:
        raise RealValueCheckpointError(
            f"Checkpoint phase {phase!r} is not a contiguous table-order prefix: {path}"
        )
    return list(value)


T = TypeVar("T")


class RealValueCheckpointSession:
    """Own one destination checkpoint and optionally continue a source one."""

    def __init__(
        self,
        *,
        alignment: dict,
        low_conf_report: dict,
        candidates: dict,
        ontology: dict | None,
        enriched_schema: dict | None,
        force_all_context: bool,
        implementation_sha256: str,
        class_table_order: Sequence[str],
        table_order: Sequence[str],
        checkpoint_path: str | Path | None,
        resume_checkpoint_path: str | Path | None = None,
    ) -> None:
        self.class_table_order = list(class_table_order)
        self.table_order = list(table_order)
        self.signature = input_signature(
            alignment=alignment,
            low_conf_report=low_conf_report,
            candidates=candidates,
            ontology=ontology,
            enriched_schema=enriched_schema,
            force_all_context=force_all_context,
            implementation_sha256=implementation_sha256,
        )
        self.portable_signature = portable_input_signature(
            alignment=alignment,
            low_conf_report=low_conf_report,
            candidates=candidates,
            ontology=ontology,
            enriched_schema=enriched_schema,
            force_all_context=force_all_context,
            implementation_sha256=implementation_sha256,
        )
        self.input_component_sha256 = {
            "alignment": sha256_json(alignment),
            "low_conf_report": sha256_json(low_conf_report),
            "candidates": sha256_json(candidates),
            "portable_ontology": sha256_json(
                _portable_ontology_value(ontology or {})
            ),
            "enriched_schema": sha256_json(enriched_schema or {}),
            "force_all_context": sha256_json(bool(force_all_context)),
            "implementation": implementation_sha256,
        }
        self._alignment = alignment
        self._candidates = candidates
        self._ontology = ontology or {}
        self._enriched_schema = enriched_schema or {}
        self._force_all_context = bool(force_all_context)
        self.implementation_sha256 = implementation_sha256
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        self.resume_checkpoint_path = (
            Path(resume_checkpoint_path).expanduser().resolve()
            if resume_checkpoint_path
            else None
        )
        if self.resume_checkpoint_path is not None and self.checkpoint_path is None:
            raise RealValueCheckpointError(
                "A resumed RealValue run must write a checkpoint in its new namespace"
            )
        if (
            self.checkpoint_path is not None
            and self.resume_checkpoint_path is not None
            and self.checkpoint_path == self.resume_checkpoint_path
        ):
            raise RealValueCheckpointError(
                "Continuation checkpoint must differ from the immutable source checkpoint"
            )
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            raise FileExistsError(
                f"Refusing to reuse an existing RealValue checkpoint destination: {self.checkpoint_path}"
            )

        self.source_sha256: str | None = None
        self.resume_compatibility: dict[str, Any] = {
            "mode": "fresh_checkpoint",
        }
        self.completed = {PHASE_CLASS: [], PHASE_TABLE: []}
        self.initial_result = deepcopy(alignment)
        if self.resume_checkpoint_path is not None:
            self._load_resume_source()

        if self.checkpoint_path is not None:
            initial_status = (
                "completed"
                if (
                    self.completed[PHASE_CLASS] == self.class_table_order
                    and self.completed[PHASE_TABLE] == self.table_order
                )
                else "running"
            )
            self._publish(status=initial_status, result=self.initial_result)

    def _load_resume_source(self) -> None:
        assert self.resume_checkpoint_path is not None
        checkpoint = _read_json_object(self.resume_checkpoint_path)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RealValueCheckpointError(
                f"Unsupported RealValue checkpoint schema: {self.resume_checkpoint_path}"
            )
        if checkpoint.get("kind") != CHECKPOINT_KIND:
            raise RealValueCheckpointError(
                f"Unexpected RealValue checkpoint kind: {self.resume_checkpoint_path}"
            )
        if checkpoint.get("input_signature") == self.signature:
            self.resume_compatibility = {"mode": "exact_input_signature"}
        elif checkpoint.get("portable_input_signature") == self.portable_signature:
            self.resume_compatibility = {
                "mode": "portable_blank_node_signature",
                "source_legacy_input_signature": checkpoint.get("input_signature"),
            }
        elif checkpoint.get("portable_input_signature") is None:
            evidence = self._validate_legacy_blank_node_resume(checkpoint)
            if evidence is None:
                raise RealValueCheckpointError(
                    "RealValue checkpoint inputs or implementation differ from the requested continuation"
                )
            self.resume_compatibility = evidence
        else:
            raise RealValueCheckpointError(
                "RealValue checkpoint inputs or implementation differ from the requested continuation"
            )
        if checkpoint.get("class_table_order") != self.class_table_order:
            raise RealValueCheckpointError("RealValue class-pass table order changed")
        if checkpoint.get("table_order") != self.table_order:
            raise RealValueCheckpointError("RealValue refinement table order changed")
        result = checkpoint.get("result")
        if not isinstance(result, dict):
            raise RealValueCheckpointError(
                f"RealValue checkpoint lacks a partial alignment: {self.resume_checkpoint_path}"
            )
        self.completed = {
            PHASE_CLASS: _validated_prefix(
                checkpoint=checkpoint,
                phase=PHASE_CLASS,
                expected_order=self.class_table_order,
                path=self.resume_checkpoint_path,
            ),
            PHASE_TABLE: _validated_prefix(
                checkpoint=checkpoint,
                phase=PHASE_TABLE,
                expected_order=self.table_order,
                path=self.resume_checkpoint_path,
            ),
        }
        if self.completed[PHASE_TABLE] and len(self.completed[PHASE_CLASS]) != len(
            self.class_table_order
        ):
            raise RealValueCheckpointError(
                "RealValue refinement progress exists before Class confirmation completed"
            )
        self.initial_result = deepcopy(result)
        self.source_sha256 = sha256_file(self.resume_checkpoint_path)

    def _validate_legacy_blank_node_resume(
        self,
        checkpoint: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Certify one legacy partial checkpoint with a non-portable signature.

        Version-1 checkpoints written before ``portable_input_signature``
        hashed RDFLib blank-node UUIDs.  The old aggregate cannot be recreated
        in another process.  A legacy bridge is allowed only when the source
        namespace's persisted semantic inputs are byte-for-value identical,
        all semantic source modules recorded by the source provenance are
        unchanged, and the ontology actually contains the known unstable
        RDFLib identifier form.  Any named-ontology or mapping change still
        fails closed.
        """

        assert self.resume_checkpoint_path is not None
        if checkpoint.get("implementation_sha256") != self.implementation_sha256:
            return None
        if not _contains_rdflib_blank_node(self._ontology):
            return None

        source_dir = self.resume_checkpoint_path.parent
        persisted_inputs = {
            "dp_mapping_alignment.json": self._alignment,
            "dp_mapping_candidates.json": self._candidates,
            "enriched_schema.json": self._enriched_schema,
        }
        artifact_sha256: dict[str, str] = {}
        for name, expected in persisted_inputs.items():
            path = source_dir / name
            try:
                observed = _read_json_object(path)
            except RealValueCheckpointError:
                return None
            if observed != expected:
                return None
            artifact_sha256[name] = sha256_file(path)

        provenance_path = source_dir / "paper_snapshot_provenance.json"
        try:
            provenance = _read_json_object(provenance_path)
        except RealValueCheckpointError:
            return None
        source_root = Path(__file__).resolve().parents[1]
        recorded_root = provenance.get("source_root")
        if not recorded_root or Path(recorded_root).expanduser().resolve() != source_root:
            return None
        recorded_hashes = provenance.get("source_files_sha256")
        if not isinstance(recorded_hashes, dict):
            return None
        if (
            self._force_all_context
            or provenance.get("context_enhancement_mode") != "confidence"
        ):
            return None

        snapshot_root_value = provenance.get("snapshot_root")
        database = provenance.get("database")
        base_commit = provenance.get("base_paper_commit")
        if (
            not snapshot_root_value
            or Path(snapshot_root_value).expanduser().resolve() != source_root.parent
            or not isinstance(database, str)
            or not database
            or database in {".", ".."}
            or "/" in database
            or "\\" in database
            or not isinstance(base_commit, str)
            or not base_commit
        ):
            return None
        ontology_path = source_root / "input" / database / "ontology.ttl"
        if not ontology_path.is_file():
            return None
        try:
            committed_ontology = subprocess.run(
                [
                    "git",
                    "-C",
                    str(source_root.parent),
                    "show",
                    f"{base_commit}:ma4rom/input/{database}/ontology.ttl",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        current_ontology_bytes = ontology_path.read_bytes()
        if current_ontology_bytes != committed_ontology:
            return None
        try:
            from utils.ontology_utils import read_ontology  # noqa: WPS433

            reparsed_ontology = read_ontology(str(ontology_path))
        except Exception:
            return None
        if _portable_ontology_value(reparsed_ontology) != _portable_ontology_value(
            self._ontology
        ):
            return None

        # This module is the portability fix itself, so its hash is expected to
        # differ from the legacy provenance.  Every semantic production module
        # must remain byte-identical to the source run.
        checkpoint_module = "ma4rom/RealValue/real_value_checkpoint.py"
        verified_source_sha256: dict[str, str] = {}
        for relative, expected_hash in recorded_hashes.items():
            if not isinstance(relative, str) or not relative.startswith("ma4rom/"):
                continue
            if relative == checkpoint_module:
                continue
            current_path = source_root.parent / relative
            if not current_path.is_file():
                return None
            observed_hash = sha256_file(current_path)
            if observed_hash != expected_hash:
                return None
            verified_source_sha256[relative] = observed_hash

        if (
            verified_source_sha256.get(
                "ma4rom/RealValue/real_value_enhancement_agent.py"
            )
            != self.implementation_sha256
            or "ma4rom/DPMapping/data_property_mapping_agent.py"
            not in verified_source_sha256
            or "ma4rom/utils/ontology_utils.py" not in verified_source_sha256
            or "ma4rom/config.py" not in verified_source_sha256
            or "ma4rom/utils/candidate_ranking.py"
            not in verified_source_sha256
            or "ma4rom/utils/db_utils.py" not in verified_source_sha256
            or "ma4rom/utils/llm_client.py" not in verified_source_sha256
        ):
            return None

        return {
            "mode": "legacy_rdflib_blank_node_portability_bridge",
            "source_legacy_input_signature": checkpoint.get("input_signature"),
            "portable_input_signature": self.portable_signature,
            "source_artifact_sha256": artifact_sha256,
            "source_provenance": str(provenance_path),
            "source_provenance_sha256": sha256_file(provenance_path),
            "ontology_path": str(ontology_path),
            "ontology_sha256": hashlib.sha256(current_ontology_bytes).hexdigest(),
            "ontology_git_commit": base_commit,
            "verified_semantic_source_files": len(verified_source_sha256),
            "source_namespace_mutated": False,
        }

    def _document(self, *, status: str, result: dict) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND,
            "status": status,
            "input_signature": self.signature,
            "portable_input_signature": self.portable_signature,
            "input_component_sha256": deepcopy(self.input_component_sha256),
            "implementation_sha256": self.implementation_sha256,
            "class_table_order": self.class_table_order,
            "table_order": self.table_order,
            "completed": deepcopy(self.completed),
            "completed_counts": {
                PHASE_CLASS: len(self.completed[PHASE_CLASS]),
                PHASE_TABLE: len(self.completed[PHASE_TABLE]),
            },
            "resumption": {
                "source_checkpoint": (
                    str(self.resume_checkpoint_path)
                    if self.resume_checkpoint_path is not None
                    else None
                ),
                "source_checkpoint_sha256": self.source_sha256,
                "source_namespace_mutated": False,
                "compatibility": deepcopy(self.resume_compatibility),
            },
            "result": deepcopy(result),
        }

    def _publish(self, *, status: str, result: dict) -> None:
        if self.checkpoint_path is None:
            return
        atomic_write_json(self.checkpoint_path, self._document(status=status, result=result))

    def iter_phase(
        self,
        phase: str,
        items: Sequence[T],
        *,
        table_name: Callable[[T], str],
        result: Callable[[], dict],
    ) -> Iterator[tuple[int, T]]:
        """Yield unfinished items and checkpoint after each body completes.

        A generator is intentional: ``continue`` in the caller resumes this
        generator, so even empty/no-op tables are committed.  If a BaseException
        escapes the table body, execution never reaches the commit and that table
        is correctly retried by the continuation.
        """

        if phase not in PHASES:
            raise RealValueCheckpointError(f"Unknown RealValue checkpoint phase: {phase}")
        expected_order = self.class_table_order if phase == PHASE_CLASS else self.table_order
        if [table_name(item) for item in items] != expected_order:
            raise RealValueCheckpointError(f"RealValue {phase} work order changed after validation")
        start = len(self.completed[phase])
        for zero_index in range(start, len(items)):
            item = items[zero_index]
            yield zero_index + 1, item
            expected_table = expected_order[zero_index]
            if table_name(item) != expected_table:
                raise RealValueCheckpointError(f"Unexpected RealValue table at index {zero_index}")
            self.completed[phase].append(expected_table)
            all_done = (
                self.completed[PHASE_CLASS] == self.class_table_order
                and self.completed[PHASE_TABLE] == self.table_order
            )
            self._publish(status="completed" if all_done else "running", result=result())

    def finish(self, result: dict) -> None:
        if self.completed[PHASE_CLASS] != self.class_table_order:
            raise RealValueCheckpointError("RealValue Class-confirmation phase did not complete")
        if self.completed[PHASE_TABLE] != self.table_order:
            raise RealValueCheckpointError("RealValue table-refinement phase did not complete")
        self._publish(status="completed", result=result)

    def summary(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "resumed_from": (
                str(self.resume_checkpoint_path) if self.resume_checkpoint_path else None
            ),
            "source_checkpoint_sha256": self.source_sha256,
            "completed": deepcopy(self.completed),
        }
