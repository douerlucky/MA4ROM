#!/usr/bin/env python3
"""Validate or load the packaged RODI datasets for MA4ROM.

The repository contains ma4rom/input/<scenario>/dump.sql and queries/.
This helper verifies the dataset layout and can load one or more scenarios into
local PostgreSQL databases using psql. Each scenario is loaded into a database
with the same name as the scenario, matching ma4rom/config.py.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "ma4rom" / "input"


@dataclass(frozen=True)
class Scenario:
    name: str
    path: Path
    ontology: Path
    dump: Path
    queries: Path
    schema: str


def read_dump_schema(dump_path: Path, fallback: str) -> str:
    if not dump_path.exists():
        return fallback
    head = dump_path.read_text(encoding="utf-8", errors="ignore")[:4000]
    match = re.search(r"CREATE\s+SCHEMA\s+([^;]+);", head, flags=re.IGNORECASE)
    if not match:
        return fallback
    return match.group(1).strip().strip('"')


def discover_scenarios(dataset_dir: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    for path in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        dump = path / "dump.sql"
        scenarios.append(
            Scenario(
                name=path.name,
                path=path,
                ontology=path / "ontology.ttl",
                dump=dump,
                queries=path / "queries",
                schema=read_dump_schema(dump, path.name),
            )
        )
    return scenarios


def validate_scenarios(scenarios: list[Scenario]) -> list[str]:
    errors: list[str] = []
    for scenario in scenarios:
        if not scenario.ontology.exists():
            errors.append(f"{scenario.name}: missing ontology.ttl")
        if not scenario.dump.exists():
            errors.append(f"{scenario.name}: missing dump.sql")
        if not scenario.queries.is_dir():
            errors.append(f"{scenario.name}: missing queries/")
        elif not any(scenario.queries.glob("*.qpair")):
            errors.append(f"{scenario.name}: queries/ contains no .qpair files")
    return errors


def resolve_targets(selection: str, scenarios: list[Scenario]) -> list[Scenario]:
    if selection == "all":
        return scenarios
    wanted = {item.strip() for item in selection.split(",") if item.strip()}
    by_name = {scenario.name: scenario for scenario in scenarios}
    missing = sorted(wanted - set(by_name))
    if missing:
        raise ValueError(f"Unknown scenario(s): {', '.join(missing)}")
    return [by_name[name] for name in sorted(wanted)]


def psql_env(password: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if password:
        env["PGPASSWORD"] = password
    return env


def run_psql(args: argparse.Namespace, database: str, extra: list[str]) -> None:
    cmd = [
        "psql",
        "--set", "ON_ERROR_STOP=1",
        "--host", args.host,
        "--port", str(args.port),
        "--username", args.user,
        "--dbname", database,
        *extra,
    ]
    if args.dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True, env=psql_env(args.password))


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def database_exists(args: argparse.Namespace, database: str) -> bool:
    cmd = [
        "psql",
        "--tuples-only",
        "--no-align",
        "--host", args.host,
        "--port", str(args.port),
        "--username", args.user,
        "--dbname", args.maintenance_db,
        "--command", f"SELECT 1 FROM pg_database WHERE datname = {quote_literal(database)};",
    ]
    if args.dry_run:
        print("DRY-RUN:", " ".join(cmd))
        return False
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=psql_env(args.password),
    )
    return result.stdout.strip() == "1"


def load_scenario(args: argparse.Namespace, scenario: Scenario) -> None:
    db_name = args.database_prefix + scenario.name
    print(f"\n==> Loading {scenario.name} into PostgreSQL database {db_name!r}")

    if args.drop_existing:
        run_psql(args, args.maintenance_db, ["--command", f"DROP DATABASE IF EXISTS {quote_ident(db_name)};"])
    elif database_exists(args, db_name):
        print(f"    Database {db_name!r} already exists; use --drop-existing to recreate it.")
        return

    run_psql(args, args.maintenance_db, ["--command", f"CREATE DATABASE {quote_ident(db_name)};"])
    run_psql(args, db_name, ["--file", str(scenario.dump)])
    env = f"MAMG_CURRENT_DATABASE={scenario.name}"
    if scenario.schema != scenario.name:
        env += f" MAMG_DB_SCHEMA={scenario.schema}"
    print(f"    Loaded schema: {scenario.schema}")
    print(f"    Done. Run MA4ROM with: {env} python ma4rom/main.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and load MA4ROM/RODI datasets.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--check", action="store_true", help="Only validate dataset files.")
    parser.add_argument("--list", action="store_true", help="List available scenarios.")
    parser.add_argument("--load", metavar="SCENARIOS", help="Load comma-separated scenarios or 'all' into PostgreSQL.")
    parser.add_argument("--drop-existing", action="store_true", help="Drop target databases before loading.")
    parser.add_argument("--database-prefix", default="", help="Optional prefix for created PostgreSQL databases.")
    parser.add_argument("--host", default=os.environ.get("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PGPORT", "5432")))
    parser.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD"))
    parser.add_argument("--maintenance-db", default=os.environ.get("PGDATABASE", "postgres"))
    parser.add_argument("--dry-run", action="store_true", help="Print psql commands without executing them.")
    args = parser.parse_args()

    scenarios = discover_scenarios(args.dataset_dir)
    errors = validate_scenarios(scenarios)
    if args.list:
        for scenario in scenarios:
            print(scenario.name)
        if not args.check and not args.load:
            return 0
    if errors:
        print("Dataset validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Dataset validation passed: {len(scenarios)} scenarios found.")

    if args.check or not args.load:
        if args.load is None and not args.check and not args.list:
            print("No load requested. Use --load all to import PostgreSQL databases.")
        return 0

    if shutil.which("psql") is None:
        print("psql was not found on PATH. Install PostgreSQL client tools first.", file=sys.stderr)
        return 1

    targets = resolve_targets(args.load, scenarios)
    for scenario in targets:
        load_scenario(args, scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
