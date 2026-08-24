#!/usr/bin/env python3
"""Offline integrity checks for the compact MA4ROM public release."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from rdflib import Graph


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = REPO_ROOT / "ma4rom" / "input"
RESULTS_ROOT = REPO_ROOT / "results"
TABLE2_ROOT = RESULTS_ROOT / "paper" / "table2"
DS_ARTEFACT_ROOT = TABLE2_ROOT / "deepseek-v4-flash"
GPT4O_ARTEFACT_ROOT = TABLE2_ROOT / "gpt-4o"
TABLE6_ROOT = RESULTS_ROOT / "paper" / "table6"
FK_REMOVAL_ROOT = TABLE6_ROOT / "fk-removal"

EXPECTED_DS_F1 = {
    "cmt_renamed": 0.8966,
    "conference_renamed": 0.9487,
    "sigkdd_renamed": 0.9649,
    "cmt_structured": 0.8966,
    "conference_structured": 0.8156,
    "sigkdd_structured": 0.9649,
    "sigkdd_mixed": 0.9649,
    "conference_nofks": 0.8462,
    "cmt_denormalized": 0.7734,
    "mondial_rel": 0.3446,
    "npd_atomic_tests": 0.3765,
}
EXPECTED_GPT4O_F1 = {
    "cmt_renamed": 0.9310,
    "conference_renamed": 0.9487,
    "sigkdd_renamed": 0.9649,
    "cmt_structured": 0.7931,
    "conference_structured": 0.8412,
    "sigkdd_structured": 0.9649,
    "sigkdd_mixed": 0.9649,
    "conference_nofks": 0.8718,
    "cmt_denormalized": 0.5200,
    "mondial_rel": 0.5784,
    "npd_atomic_tests": 0.3637,
}
EXPECTED_FK_REMOVAL_F1 = {
    "cmt_renamed": 0.8621,
    "conference_renamed": 0.8974,
    "sigkdd_renamed": 0.8959,
    "cmt_structured": 0.8621,
    "conference_structured": 0.8205,
    "sigkdd_structured": 0.9649,
    "sigkdd_mixed": 0.8959,
    "cmt_denormalized": 0.5600,
    "mondial_rel": 0.2885,
    "npd_atomic_tests": 0.3617,
}

FORBIDDEN_PARTS = {
    ".idea",
    ".vscode",
    ".pytest_cache",
    "__pycache__",
    "backup",
    "backups",
    "logs",
    "output",
    "tmp",
}
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".html",
    ".json",
    ".md",
    ".py",
    ".qpair",
    ".toml",
    ".ttl",
    ".txt",
    ".yml",
    ".yaml",
}
# Split the literals so this scanner does not flag its own source code.
PERSONAL_PATH_PATTERNS = (
    b"/Users/" + b"douer_" + b"lucky",
    b".codex/" + b"worktrees",
)
SECRET_PATTERN = re.compile(rb"sk-[A-Za-z0-9_-]{20,}")


def fail(message: str) -> None:
    raise AssertionError(message)


def parse_f1(path: Path) -> float:
    match = re.search(
        r"Average F1 Score:\s*([0-9]+(?:\.[0-9]+)?)",
        path.read_text(encoding="utf-8"),
    )
    if not match:
        fail(f"Cannot parse F1 from {path.relative_to(REPO_ROOT)}")
    return float(match.group(1))


def check_inputs() -> None:
    actual = {path.name for path in INPUT_ROOT.iterdir() if path.is_dir()}
    if actual != set(EXPECTED_DS_F1):
        fail(f"Input scenario set differs: {sorted(actual ^ set(EXPECTED_DS_F1))}")

    for scenario in sorted(EXPECTED_DS_F1):
        directory = INPUT_ROOT / scenario
        required = [directory / "ontology.ttl", directory / "dump.sql"]
        missing = [path.name for path in required if not path.is_file()]
        queries = sorted((directory / "queries").glob("*.qpair"))
        if missing or not queries:
            fail(f"Incomplete input {scenario}: missing={missing}, queries={len(queries)}")
        Graph().parse(directory / "ontology.ttl", format="turtle")


def read_summary(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row["scenario"]: float(row["f1"]) for row in rows}


def check_artefact_set(root: Path, expected_f1: dict[str, float]) -> None:
    actual = {path.name for path in root.iterdir() if path.is_dir()}
    if actual != set(expected_f1):
        fail(f"Result scenario set differs in {root}: {sorted(actual ^ set(expected_f1))}")

    for scenario, expected in sorted(expected_f1.items()):
        directory = root / scenario
        for name in ("f1.txt", "mapping.ttl", "ontology.ttl"):
            if not (directory / name).is_file():
                fail(f"Missing {scenario}/{name}")
        observed = parse_f1(directory / "f1.txt")
        if abs(observed - expected) > 1e-9:
            fail(f"F1 mismatch for {scenario}: {observed} != {expected}")
        Graph().parse(directory / "mapping.ttl", format="turtle")
        Graph().parse(directory / "ontology.ttl", format="turtle")

        compact = directory / "per_query_metrics.json"
        if compact.exists():
            records = json.loads(compact.read_text(encoding="utf-8"))
            if not isinstance(records, list) or not records:
                fail(f"Invalid compact metrics for {scenario}")
            allowed = {"id", "precision", "recall", "f1"}
            for record in records:
                if set(record) != allowed:
                    fail(f"Unexpected compact metric fields for {scenario}: {set(record)}")


def check_results() -> None:
    summaries = (
        (TABLE2_ROOT / "summary.csv", EXPECTED_DS_F1),
        (TABLE2_ROOT / "gpt4o_summary.csv", EXPECTED_GPT4O_F1),
    )
    for path, expected in summaries:
        if read_summary(path) != expected:
            fail(f"{path.relative_to(REPO_ROOT)} does not match its canonical F1 index")

    check_artefact_set(DS_ARTEFACT_ROOT, EXPECTED_DS_F1)
    check_artefact_set(GPT4O_ARTEFACT_ROOT, EXPECTED_GPT4O_F1)
    check_artefact_set(FK_REMOVAL_ROOT, EXPECTED_FK_REMOVAL_F1)

    baseline_path = RESULTS_ROOT / "baselines" / "llm4vkg" / "table2.csv"
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    if len(baseline_rows) != 12 or baseline_rows[-1]["scenario"] != "macro_average":
        fail("LLM4VKG reported-only score table is incomplete")
    if float(baseline_rows[-1]["f1"]) != 0.6073:
        fail("LLM4VKG reported macro F1 differs from Table 2")
    if any(row["evidence_status"] != "reported_only" for row in baseline_rows):
        fail("LLM4VKG scores must remain explicitly reported-only")

    fk_summary = read_summary(TABLE6_ROOT / "summary.csv")
    for scenario, expected in EXPECTED_FK_REMOVAL_F1.items():
        if abs(fk_summary[scenario] - expected) > 1e-9:
            fail(f"FK-removal summary mismatch for {scenario}")

    with (TABLE6_ROOT / "fk-recovery" / "summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        recovery_rows = list(csv.DictReader(handle))
    aggregate = recovery_rows[-1]
    if aggregate["database"] != "micro_aggregate":
        fail("FK-recovery micro aggregate is missing")
    if abs(float(aggregate["f1"]) - 0.7127429806) > 1e-10:
        fail("FK-recovery micro F1 mismatch")

    for table_number in (3, 4, 5, 6):
        directory = RESULTS_ROOT / "paper" / f"table{table_number}"
        if not (directory / "summary.csv").is_file():
            fail(f"Missing Table {table_number} summary")
        json.loads((directory / "provenance.json").read_text(encoding="utf-8"))


def check_checksums() -> None:
    checksum_path = RESULTS_ROOT / "checksums.sha256"
    listed: set[Path] = set()
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = RESULTS_ROOT / relative
        if not path.is_file():
            fail(f"Checksum target is missing: {relative}")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != digest:
            fail(f"Checksum mismatch: {relative}")
        listed.add(path.resolve())

    expected = {
        path.resolve()
        for path in RESULTS_ROOT.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if listed != expected:
        missing = sorted(str(path.relative_to(REPO_ROOT)) for path in expected - listed)
        extra = sorted(str(path.relative_to(REPO_ROOT)) for path in listed - expected)
        fail(f"Checksum coverage differs; missing={missing}, extra={extra}")


def check_public_tree() -> None:
    for path in REPO_ROOT.rglob("*"):
        relative = path.relative_to(REPO_ROOT)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_symlink():
            fail(f"Public tree contains a symlink: {relative}")
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            fail(f"Public tree contains a forbidden path: {relative}")
        if path.name == ".DS_Store" or path.suffix == ".pyc":
            fail(f"Public tree contains a cache file: {relative}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            continue
        content = path.read_bytes()
        if any(pattern in content for pattern in PERSONAL_PATH_PATTERNS):
            fail(f"Public tree contains a personal absolute path: {relative}")
        if path.name != ".env.example" and SECRET_PATTERN.search(content):
            fail(f"Public tree contains a likely API key: {relative}")


def main() -> int:
    checks = (
        ("inputs", check_inputs),
        ("results", check_results),
        ("checksums", check_checksums),
        ("public tree", check_public_tree),
    )
    for label, check in checks:
        check()
        print(f"[ok] {label}")
    print("Release verification passed without database, network, or LLM access.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Release verification failed: {exc}", file=sys.stderr)
        raise
