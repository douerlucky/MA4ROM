from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from rdflib import BNode, Graph, OWL, RDF, RDFS

from config import DB_SCHEMA_NAME, ONTOLOGY_PATH, OUTPUT_DIR
from utils.db_utils import get_connection
from utils.llm_client import call_llm
from utils.ontology_utils import _expanded_class_terms, are_classes_disjoint, local_name


CONSTRUCTION_RE = re.compile(
    r"Foreign\s+key\s+([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\s*=>\s*([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)",
    re.IGNORECASE,
)

# A value column is treated as an object IRI only when nearly all observed
# non-empty values are syntactically absolute IRIs.  This is deliberately a
# data-shape rule: it does not depend on a table, dataset, or ontology name.
PARTIAL_VALUE_IRI_MIN_RATIO = 0.90
ABSOLUTE_IRI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:[^\s]+$")
INFERRED_SR_MIN_SOURCE_IN_TARGET = 0.95


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _relation_key(table: str, column: str) -> str:
    return f"{table}.{column}"


def _constraint_relation_key(table: str, constraint_name: str) -> str:
    """Stable Step1 key for one physical composite FK constraint."""
    return f"{table}.__fk__{constraint_name}"


def _split_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    return {p.lower() for p in re.split(r"[^A-Za-z0-9]+", spaced) if p}


def _meaningful_tokens(text: str | None) -> set[str]:
    stop = {
        "fk",
        "id",
        "uri",
        "to",
        "by",
        "of",
        "has",
        "is",
        "the",
        "a",
        "an",
        "inv",
    }
    return {token for token in _split_tokens(text) if token not in stop and len(token) > 1}


def _read_json_if_exists(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _construction_predicate(graph: Graph):
    for predicate in set(graph.predicates()):
        if local_name(str(predicate)) == "construction":
            return predicate
    return None


def _table_predicate(graph: Graph):
    for predicate in set(graph.predicates()):
        if local_name(str(predicate)) == "table":
            return predicate
    return None


def _parse_ontology_metadata(ontology_path: str) -> dict[str, Any]:
    graph = Graph()
    graph.parse(ontology_path, format="turtle")
    construction_pred = _construction_predicate(graph)
    table_pred = _table_predicate(graph)

    ops: dict[str, dict[str, Any]] = {}
    for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
        uri = str(prop)
        raw_domains = list(graph.objects(prop, RDFS.domain))
        raw_ranges = list(graph.objects(prop, RDFS.range))
        ops[uri] = {
            "uri": uri,
            "local_name": local_name(uri),
            "domain": _expanded_class_terms(graph, raw_domains),
            "range": _expanded_class_terms(graph, raw_ranges),
            "construction": [str(x) for x in graph.objects(prop, construction_pred)] if construction_pred else [],
            "inverse_of": sorted(
                {str(x) for x in graph.objects(prop, OWL.inverseOf)}
                | {str(s) for s in graph.subjects(OWL.inverseOf, prop)}
            ),
            "subproperty_of": [str(x) for x in graph.objects(prop, RDFS.subPropertyOf)],
            "restrictions": [],
        }

    class_tables: dict[str, list[str]] = {}
    if table_pred:
        for cls, _, table_value in graph.triples((None, table_pred, None)):
            if not isinstance(cls, BNode):
                class_tables.setdefault(str(cls), []).append(str(table_value))

    for cls, _, restriction in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(cls, BNode) or not isinstance(restriction, BNode):
            continue
        for prop in graph.objects(restriction, OWL.onProperty):
            prop_uri = str(prop)
            if prop_uri not in ops:
                continue
            raw_values = []
            for pred in [OWL.someValuesFrom, OWL.allValuesFrom, OWL.onClass]:
                raw_values.extend(graph.objects(restriction, pred))
            # Expand unions while the owning RDF graph is still available.
            # Blank-node IDs are parse-local, so deferring this work to the
            # separately parsed ontology dict would silently lose the range.
            values = _expanded_class_terms(graph, raw_values)
            ops[prop_uri]["restrictions"].append(
                {
                    "class": str(cls),
                    "class_local": local_name(str(cls)),
                    "class_tables": class_tables.get(str(cls), []),
                    "values": values,
                    "values_local": [local_name(v) for v in values],
                }
            )
    return {"ops": ops}


def _normalize_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_absolute_iri(value: object) -> bool:
    """Conservatively recognize an absolute IRI stored in a scalar column."""
    text = _normalize_value(value)
    if not text or not ABSOLUTE_IRI_RE.fullmatch(text):
        return False
    scheme, remainder = text.split(":", 1)
    # A one-letter scheme is overwhelmingly likely to be a platform path.
    if len(scheme) == 1 or not remainder:
        return False
    if scheme.lower() in {"http", "https"}:
        return remainder.startswith("//") and bool(remainder[2:].split("/", 1)[0])
    if scheme.lower() == "urn":
        return ":" in remainder and not remainder.startswith(":")
    return True


def _partial_value_iri_profile(
    conn,
    table: str,
    column: str,
    schema_name: str,
) -> dict[str, Any]:
    """Profile a value endpoint without using lexical table/column hints."""
    values = _fetch_distinct_values(conn, table, column, schema_name)
    observed = len(values)
    iri_count = sum(1 for value in values if _is_absolute_iri(value))
    iri_ratio = iri_count / observed if observed else 0.0
    return {
        "partial_value_term_type": (
            "iri" if observed and iri_ratio >= PARTIAL_VALUE_IRI_MIN_RATIO else None
        ),
        "partial_value_iri_ratio": round(iri_ratio, 6),
        "partial_value_observed_distinct": observed,
    }


def _fetch_distinct_values(conn, table: str, column: str, schema_name: str) -> set[str]:
    query = f'SELECT DISTINCT "{column}" FROM "{schema_name}"."{table}" WHERE "{column}" IS NOT NULL'
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return {
                normalized
                for (value,) in cur.fetchall()
                if (normalized := _normalize_value(value)) is not None
            }
    except Exception as exc:
        print(f"  [WARN] 读取 {table}.{column} 失败: {exc}")
        conn.rollback()
        return set()


def _overlap_metrics(
    conn,
    source_table: str,
    source_column: str,
    target_table: str,
    target_column: str,
    schema_name: str,
) -> dict[str, Any]:
    source_values = _fetch_distinct_values(conn, source_table, source_column, schema_name)
    target_values = _fetch_distinct_values(conn, target_table, target_column, schema_name)
    intersection = len(source_values & target_values)
    union = len(source_values | target_values)
    source_distinct = len(source_values)
    target_distinct = len(target_values)
    source_in_target = intersection / source_distinct if source_distinct else 0.0
    target_in_source = intersection / target_distinct if target_distinct else 0.0
    manual_jaccard = intersection / union if union else 0.0
    if source_in_target >= 0.95 and target_in_source >= 0.95:
        evidence_type = "equivalence_column"
    elif source_in_target >= 0.95:
        evidence_type = "inclusion_column_source_to_target"
    elif manual_jaccard >= 0.80:
        evidence_type = "high_overlap_column"
    else:
        evidence_type = "weak_or_conflicting"
    return {
        "source_distinct": source_distinct,
        "target_distinct": target_distinct,
        "intersection": intersection,
        "union": union,
        "manual_jaccard": manual_jaccard,
        "source_in_target": source_in_target,
        "target_in_source": target_in_source,
        "evidence_type": evidence_type,
    }


def _edge_from_fk(source_table: str, fk: dict[str, Any]) -> dict[str, str] | None:
    source_column = fk.get("column")
    target_table = fk.get("references_table") or fk.get("ref_table")
    target_column = fk.get("references_column") or fk.get("ref_col")
    if not source_column or not target_table or not target_column:
        return None
    return {
        "source_table": source_table,
        "source_column": source_column,
        "target_table": target_table,
        "target_column": target_column,
        "constraint_name": fk.get("constraint_name") or "",
    }


def _rule4_inferred_sr_edges(
    table_name: str,
    table_info: dict[str, Any],
) -> list[dict[str, str]]:
    """Return the two scalar FK-key endpoints permitted by Rule 4.

    An inferred relation is structurally justified only by ``FKKeys(t)``:
    foreign-key constraints whose complete source key is also part of the
    table primary key. FK metadata is stored one row per column, so grouping
    by the physical constraint is essential; pairing two rows of one
    composite FK would fabricate a binary relation.

    The inferred-SR serializer currently carries one ordered source column per
    endpoint. Composite or incompletely described constraints therefore
    abstain here instead of creating a task that cannot be serialized
    truthfully downstream.
    """
    primary_keys = {
        str(column)
        for column in (table_info.get("primary_key") or [])
        if column
    }
    foreign_keys = [
        fk
        for fk in (table_info.get("foreign_keys") or [])
        if isinstance(fk, dict)
    ]
    if not primary_keys or not foreign_keys:
        return []

    constraint_groups: dict[str, list[dict[str, Any]]] = {}
    for fk in foreign_keys:
        constraint_name = str(fk.get("constraint_name") or "").strip()
        # Without a stable constraint id, multiple column rows cannot be
        # distinguished from separate scalar endpoints. Fresh schema reads
        # always provide this metadata, so ambiguity should fail closed.
        if not constraint_name:
            return []
        constraint_groups.setdefault(constraint_name, []).append(fk)

    fk_key_groups: list[list[dict[str, Any]]] = []
    for rows in constraint_groups.values():
        source_columns = [str(row.get("column") or "") for row in rows]
        if not all(source_columns) or len(set(source_columns)) != len(source_columns):
            return []

        declared_arities: set[int] = set()
        for row in rows:
            try:
                arity = int(row.get("fk_arity"))
            except (TypeError, ValueError):
                return []
            if arity <= 0:
                return []
            declared_arities.add(arity)
        if len(declared_arities) != 1:
            return []
        declared_arity = next(iter(declared_arities))
        if len(rows) != declared_arity:
            # A truncated composite constraint is not a scalar endpoint.
            return []

        columns_in_pk = [column in primary_keys for column in source_columns]
        if any(columns_in_pk) and not all(columns_in_pk):
            # Never project a partially-PK composite constraint down to the
            # one row that happens to intersect the primary key.
            return []
        if all(columns_in_pk):
            fk_key_groups.append(rows)

    # Rule 4 is binary. Zero/one groups do not form a relation, while three
    # or more groups describe an n-ary association and must not be paired.
    if len(fk_key_groups) != 2:
        return []

    edges: list[dict[str, str]] = []
    for rows in fk_key_groups:
        if len(rows) != 1:
            # The serializer has no ordered composite-identity contract yet.
            return []
        edge = _edge_from_fk(table_name, rows[0])
        if not edge:
            return []
        edges.append(edge)

    if edges[0]["source_column"] == edges[1]["source_column"]:
        return []
    return sorted(
        edges,
        key=lambda edge: (
            edge["constraint_name"].lower(),
            edge["source_column"].lower(),
            edge["target_table"].lower(),
        ),
    )


def _find_fk_reference_column(
    enriched_schema: dict[str, Any],
    table_name: str,
    column_name: str,
    ref_table: str | None = None,
) -> str:
    table_info = enriched_schema.get(table_name) or enriched_schema.get(table_name.lower()) or {}
    for fk in table_info.get("foreign_keys") or []:
        fk_col = fk.get("column")
        fk_ref_table = fk.get("references_table") or fk.get("ref_table")
        if fk_col != column_name:
            continue
        if ref_table and fk_ref_table and fk_ref_table != ref_table:
            continue
        return fk.get("references_column") or fk.get("ref_col") or ""
    return ""


def _find_fk_reference_table(
    enriched_schema: dict[str, Any],
    table_name: str,
    column_name: str,
) -> str:
    """Look up a physical FK target when an older alignment omitted it."""
    table_info = enriched_schema.get(table_name) or enriched_schema.get(table_name.lower()) or {}
    for fk in table_info.get("foreign_keys") or []:
        if fk.get("column") == column_name:
            return fk.get("references_table") or fk.get("ref_table") or ""
    return ""


def _find_fk_constraint_name(
    enriched_schema: dict[str, Any],
    table_name: str,
    column_name: str,
    ref_table: str | None = None,
) -> str:
    table_info = enriched_schema.get(table_name) or enriched_schema.get(table_name.lower()) or {}
    for fk in table_info.get("foreign_keys") or []:
        fk_col = fk.get("column")
        fk_ref_table = fk.get("references_table") or fk.get("ref_table")
        if fk_col != column_name:
            continue
        if ref_table and fk_ref_table and fk_ref_table != ref_table:
            continue
        return fk.get("constraint_name") or ""
    return ""


def _class_from_alignment_entry(entry: dict[str, Any] | None) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in ("class_uri", "sub_class_uri", "parent_class_uri"):
        value = entry.get(key)
        if value:
            return value
    return ""


def _table_class(table: str, final_alignment: dict[str, Any], ontology: dict[str, Any]) -> str:
    """Read the ClassMapping result without reopening semantic selection.

    ``EquivOP`` is downstream of Class/DP mapping in the paper pipeline.  A
    second table-name/DP-domain guess here can change the endpoint classes and
    admit a different ObjectProperty than the one justified by the preceding
    stage.  The optional ontology argument remains for compatibility with
    older callers, but is intentionally not consulted for fallback guesses.
    """
    lower_map = {str(k).lower(): v for k, v in final_alignment.items()}
    entry = final_alignment.get(table) or lower_map.get(table.lower())
    return _class_from_alignment_entry(entry)


def _aligned_table_for_class(
    class_uri: str | None,
    final_alignment: dict[str, Any],
    enriched_schema: dict[str, Any],
    ontology: dict[str, Any],
) -> str:
    """Resolve an entity table for a class using the alignment hierarchy.

    SR tables sometimes contain one physical FK and one value column.  The
    latter still denotes an entity, but it has no FK metadata from which to
    obtain a table.  We therefore reverse the already-produced ClassMapping
    (including SH parent classes) and use the nearest compatible aligned class
    as the target table.  This keeps the inference data/ontology driven and
    avoids relation-table-specific fallbacks.
    """
    if not class_uri:
        return ""
    target = str(class_uri)
    ancestors = ontology.get("ancestors_of", {}) or {}
    direct_parents = ontology.get("subclass_of", {}) or {}

    def _distance(child: str, ancestor: str) -> int:
        if child == ancestor:
            return 0
        queue = [(child, 0)]
        seen = {child}
        while queue:
            current, depth = queue.pop(0)
            # ``ancestors_of`` in older snapshots is a flattened list.  Prefer
            # direct subclass edges when available so a nearer parent (User)
            # outranks a transitive root (Person).
            parents = direct_parents.get(current)
            if parents is None:
                parents = ancestors.get(current, []) or []
            for parent in parents:
                if parent in seen:
                    continue
                if parent == ancestor:
                    return depth + 1
                seen.add(parent)
                queue.append((parent, depth + 1))
        return 10_000

    candidates: list[tuple[int, int, int, str]] = []
    for table_name, entry in sorted(final_alignment.items(), key=lambda item: str(item[0]).lower()):
        if not isinstance(entry, dict) or entry.get("pattern") == "SR":
            continue
        if table_name not in enriched_schema and table_name.lower() not in {
            str(k).lower() for k in enriched_schema
        }:
            continue
        classes: list[tuple[str, int]] = []
        if entry.get("pattern") == "SH":
            # A SH row's direct subclass is a stronger table identity than its
            # inherited parent class.  Keep that distinction for deterministic
            # tie-breaking (Reviewer should prefer User over Author's parent
            # User mapping, for example).
            classes.extend((x, role) for role, x in enumerate(
                (entry.get("sub_class_uri"), entry.get("parent_class_uri"))
            ) if x)
        else:
            classes.extend((x, 0) for x in (entry.get("class_uri"),) if x)
        for aligned_class, class_role in classes:
            if aligned_class == target or local_name(aligned_class).lower() == local_name(target).lower():
                candidates.append((0, 0, class_role, str(table_name)))
                continue
            # If the requested class is a subclass of the aligned table class,
            # the table is a valid parent-table representation (e.g. Reviewer
            # backed by User).  The opposite direction is also retained for
            # schemas where a dedicated subtype table is available.
            d_target_to_aligned = _distance(target, aligned_class)
            d_aligned_to_target = _distance(aligned_class, target)
            distance = min(d_target_to_aligned, d_aligned_to_target)
            if distance < 10_000:
                # Prefer the table whose aligned class is the nearest ancestor;
                # a dedicated exact/subtype table wins through the distance
                # tuple and deterministic table-name tie-break.
                relation_penalty = 0 if d_target_to_aligned <= d_aligned_to_target else 1
                candidates.append((distance, relation_penalty, class_role, str(table_name)))
    if candidates:
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].lower()))
        return candidates[0][3]

    # Last generic fallback: exact normalized class/table names.  This is only
    # used when an alignment entry is absent, never as a dataset-specific map.
    target_name = _norm(local_name(target))
    for table_name in sorted(enriched_schema, key=lambda value: str(value).lower()):
        if _norm(str(table_name)) == target_name:
            return str(table_name)
    return ""


def _primary_key_for_table(
    table_name: str,
    enriched_schema: dict[str, Any],
    preferred_column: str | None = None,
) -> str:
    """Return the target key column used by a partial SR value endpoint."""
    info = enriched_schema.get(table_name) or enriched_schema.get(str(table_name).lower()) or {}
    columns = info.get("columns") or {}
    if preferred_column and preferred_column in columns:
        return preferred_column
    primary_key = info.get("primary_key") or []
    if primary_key:
        return str(primary_key[0])
    for column in columns:
        if str(column).lower() == "id":
            return str(column)
    return "ID"


def _build_fk_tasks(
    enriched_schema: dict[str, Any],
    final_alignment: dict[str, Any],
    ontology: dict[str, Any],
    schema_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Rule-3 tasks at physical FK-constraint granularity.

    A scalar FK remains one column task.  A composite FK is one relation, not
    one relation per join component, and is admitted only when its metadata is
    complete and its referenced columns cover the target primary identity.
    """
    tasks: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    # R2RML's validated identity contract is the authority for deciding
    # whether a physical SH FK is an inheritance join rather than a semantic
    # relation.  Import lazily to keep the pure module import lightweight.
    from r2rml_generator import (
        _complete_physical_fk_group,
        _fk_contract_matches,
        build_sh_identity_fk_consumption_contracts,
    )

    consumed_identity_fks = build_sh_identity_fk_consumption_contracts(
        final_alignment,
        enriched_schema,
        ontology,
    )
    conn = get_connection()
    try:
        for table_name, info in sorted(enriched_schema.items()):
            foreign_keys = [
                fk for fk in (info.get("foreign_keys") or [])
                if isinstance(fk, dict)
            ]
            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
            unnamed: list[dict[str, Any]] = []
            for fk in foreign_keys:
                constraint_name = str(fk.get("constraint_name") or "").strip()
                if constraint_name:
                    grouped.setdefault((table_name, constraint_name), []).append(fk)
                else:
                    unnamed.append(fk)

            constraint_groups = list(grouped.items())
            constraint_groups.extend(
                [((table_name, f"__scalar__{index}"), [fk]) for index, fk in enumerate(unnamed)]
            )

            for (_source_table, group_name), rows in constraint_groups:
                physical_group = _complete_physical_fk_group(rows)
                if physical_group and any(
                    _fk_contract_matches(physical_group, identity_contract)
                    for identity_contract in consumed_identity_fks.get(
                        table_name, []
                    )
                ):
                    # The FK has already collapsed this SH row onto its
                    # parent's canonical subject IRI.  Emitting an OP task for
                    # the same columns would turn identity into a self-edge.
                    continue
                edges = [_edge_from_fk(table_name, fk) for fk in rows]
                edges = [edge for edge in edges if edge]
                arities = {
                    int(fk.get("fk_arity") or 1)
                    for fk in rows
                    if str(fk.get("fk_arity") or "1").isdigit()
                }
                expected_arity = max(arities) if arities else len(rows)
                is_composite = expected_arity > 1 or len(rows) > 1

                relation_key = (
                    _constraint_relation_key(table_name, group_name)
                    if is_composite and not group_name.startswith("__scalar__")
                    else (
                        _relation_key(table_name, edges[0]["source_column"])
                        if edges else f"{table_name}.__invalid_fk__{group_name}"
                    )
                )

                match_rows = []
                for edge in edges:
                    metrics = _overlap_metrics(
                        conn,
                        edge["source_table"],
                        edge["source_column"],
                        edge["target_table"],
                        edge["target_column"],
                        schema_name,
                    )
                    match_row = {**edge, **metrics, "relation_key": relation_key}
                    match_rows.append(match_row)
                    evidence_rows.append(match_row)

                if not edges:
                    continue

                target_tables = {edge["target_table"] for edge in edges}
                source_columns = [edge["source_column"] for edge in edges]
                target_columns = [edge["target_column"] for edge in edges]
                composite_complete = True
                if is_composite:
                    target_table = next(iter(target_tables)) if len(target_tables) == 1 else ""
                    target_info = enriched_schema.get(target_table, {}) or {}
                    target_identity = [
                        str(column)
                        for column in (target_info.get("primary_key") or [])
                        if column
                    ]
                    composite_complete = bool(
                        not group_name.startswith("__scalar__")
                        and len(arities) <= 1
                        and len(edges) == expected_arity
                        and len(source_columns) == len(set(source_columns))
                        and len(target_columns) == len(set(target_columns))
                        and len(target_tables) == 1
                        and target_identity
                        and set(target_columns) == set(target_identity)
                    )
                    if not composite_complete:
                        for row in match_rows:
                            row["task_status"] = "abstained_incomplete_composite_fk"
                        continue

                first = edges[0]
                domain_class = _table_class(table_name, final_alignment, ontology)
                range_class = _table_class(first["target_table"], final_alignment, ontology)
                if not domain_class or not range_class:
                    continue

                if is_composite:
                    target_identity = list(
                        (enriched_schema.get(first["target_table"], {}) or {}).get(
                            "primary_key", []
                        )
                    )
                    edge_by_target = {edge["target_column"]: edge for edge in edges}
                    ordered_edges = [edge_by_target[column] for column in target_identity]
                    source_column = ",".join(
                        edge["source_column"] for edge in ordered_edges
                    )
                    target_column = ",".join(target_identity)
                    name_hint = " ".join(
                        [group_name, *(edge["source_column"] for edge in ordered_edges)]
                    )
                    task_type = "fk_obj_composite"
                else:
                    ordered_edges = edges
                    source_column = first["source_column"]
                    target_column = first["target_column"]
                    name_hint = first["source_column"]
                    task_type = "fk_obj"

                tasks.append(
                    {
                        "task_type": task_type,
                        "key": relation_key,
                        "name_hint": name_hint,
                        "source_table": first["source_table"],
                        "source_column": source_column,
                        "source_columns": [
                            edge["source_column"] for edge in ordered_edges
                        ],
                        "target_table": first["target_table"],
                        "target_column": target_column,
                        "target_columns": [
                            edge["target_column"] for edge in ordered_edges
                        ],
                        "constraint_name": (
                            group_name if is_composite else first.get("constraint_name", "")
                        ),
                        "domain_class_uri": domain_class,
                        "range_class_uri": range_class,
                        "schema_matching": match_rows,
                    }
                )
    finally:
        conn.close()
    return tasks, evidence_rows


def _build_value_attr_partial_sr_task(
    conn,
    table_name: str,
    entry: dict[str, Any],
    final_alignment: dict[str, Any],
    enriched_schema: dict[str, Any],
    ontology: dict[str, Any],
    schema_name: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Promote only a structural one-FK/one-value table to partial-SR.

    ``table_kind=value_attr`` is normally a literal multivalue table.  Some
    relational layouts, however, store an object endpoint in that scalar
    column.  This helper only creates a *candidate task*: the OP stage still
    needs declared/restriction endpoint evidence, and a non-IRI value still
    needs one unambiguous entity table/key materialization.
    """
    if entry.get("table_kind") != "value_attr":
        return None, []
    table_info = enriched_schema.get(table_name) or enriched_schema.get(table_name.lower()) or {}
    columns = table_info.get("columns") or {}
    column_names = list(columns) if isinstance(columns, dict) else list(columns or [])
    value_column = str(entry.get("value_column") or "")
    if len(column_names) != 2 or not value_column or value_column not in column_names:
        return None, []

    physical_fks = [
        fk
        for fk in (table_info.get("foreign_keys") or [])
        if fk.get("column")
        and (fk.get("references_table") or fk.get("ref_table"))
    ]
    physical_columns = {str(fk.get("column")) for fk in physical_fks}
    if len(physical_fks) != 1 or len(physical_columns) != 1:
        return None, []
    physical_fk = physical_fks[0]
    physical_column = str(physical_fk.get("column"))
    if physical_column == value_column or set(column_names) != {physical_column, value_column}:
        return None, []
    aligned_fk_column = str((entry.get("fk") or {}).get("column") or physical_column)
    if aligned_fk_column != physical_column:
        return None, []

    ref_table = str(
        physical_fk.get("references_table")
        or physical_fk.get("ref_table")
        or (entry.get("fk") or {}).get("ref_table")
        or ""
    )
    if not ref_table:
        return None, []
    ref_column = str(
        physical_fk.get("references_column")
        or physical_fk.get("ref_col")
        or _primary_key_for_table(ref_table, enriched_schema)
    )
    physical_class = _table_class(ref_table, final_alignment, ontology)
    if not physical_class:
        return None, []

    physical_row = {
        "source_table": table_name,
        "source_column": physical_column,
        "target_table": ref_table,
        "target_column": ref_column,
        "constraint_name": physical_fk.get("constraint_name") or "",
        "relation_key": table_name.lower(),
        **_overlap_metrics(
            conn,
            table_name,
            physical_column,
            ref_table,
            ref_column,
            schema_name,
        ),
    }
    value_profile = _partial_value_iri_profile(
        conn, table_name, value_column, schema_name
    )
    value_row = {
        "source_table": table_name,
        "source_column": value_column,
        "target_table": "",
        "target_column": "",
        "constraint_name": "",
        "relation_key": table_name.lower(),
        "partial_endpoint": True,
        "target_class_uri": "",
        **value_profile,
    }
    physical_index = column_names.index(physical_column)
    value_index = column_names.index(value_column)
    ordered_rows = [physical_row, value_row] if physical_index == 0 else [value_row, physical_row]
    task = {
        "task_type": "sr_relation_partial_value_attr",
        "key": table_name,
        "name_hint": table_name,
        "source_table": table_name,
        "source_column": ordered_rows[0]["source_column"],
        "target_table": ordered_rows[1]["target_table"],
        "target_column": ordered_rows[1]["target_column"],
        "domain_class_uri": physical_class,
        "range_class_uri": "",
        "schema_matching": ordered_rows,
        "partial_value_column": value_column,
        "partial_endpoint_invariant": True,
        "partial_physical_row_index": physical_index,
        "partial_value_row_index": value_index,
        "partial_physical_class_uri": physical_class,
        **value_profile,
    }
    evidence = [
        physical_row,
        {
            **value_row,
            "physical_endpoint_class_uri": physical_class,
            "inference_status": "value_attr_deferred_to_op_endpoint_evidence",
        },
    ]
    return task, evidence


def _build_sr_tasks(
    final_alignment: dict[str, Any],
    enriched_schema: dict[str, Any],
    ontology: dict[str, Any],
    schema_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    seen = set()
    conn = get_connection()
    try:
        for table_name, entry in sorted(final_alignment.items()):
            if not isinstance(entry, dict):
                continue
            if entry.get("table_kind") == "value_attr":
                task, value_attr_evidence = _build_value_attr_partial_sr_task(
                    conn,
                    table_name,
                    entry,
                    final_alignment,
                    enriched_schema,
                    ontology,
                    schema_name,
                )
                evidence_rows.extend(value_attr_evidence)
                if task:
                    seen.add(table_name.lower())
                    tasks.append(task)
                continue
            if entry.get("pattern") != "SR":
                continue
            fk1 = entry.get("fk1") or {}
            fk2 = entry.get("fk2") or {}
            endpoints = []
            for fk in [fk1, fk2]:
                col = fk.get("column")
                if not col:
                    continue
                ref_table = (
                    fk.get("ref_table")
                    or fk.get("references_table")
                    or _find_fk_reference_table(enriched_schema, table_name, col)
                )
                ref_col = (
                    fk.get("ref_col")
                    or fk.get("references_column")
                    or _find_fk_reference_column(enriched_schema, table_name, col, ref_table)
                    or "ID"
                )
                # A partial SR has a real FK on one side and a value column on
                # the other.  Keep the value side in the endpoint list for
                # ordering, but resolve its target table/key from the aligned
                # endpoint class below instead of dropping the whole task.
                if not ref_table:
                    endpoints.append(
                        {
                            "source_table": table_name,
                            "source_column": col,
                            "target_table": "",
                            "target_column": ref_col,
                            "constraint_name": fk.get("constraint_name") or "",
                            "relation_key": table_name.lower(),
                            "partial_endpoint": True,
                            "target_class_uri": "",
                        }
                    )
                    continue
                metrics = _overlap_metrics(conn, table_name, col, ref_table, ref_col, schema_name)
                match_row = {
                    "source_table": table_name,
                    "source_column": col,
                    "target_table": ref_table,
                    "target_column": ref_col,
                    "constraint_name": fk.get("constraint_name")
                    or _find_fk_constraint_name(enriched_schema, table_name, col, ref_table),
                    "relation_key": table_name.lower(),
                    **metrics,
                }
                evidence_rows.append(match_row)
                endpoints.append((fk, match_row))

            has_partial_endpoint = any(
                isinstance(item, dict) and item.get("partial_endpoint") for item in endpoints
            )
            # Treat a relation as partial only when an endpoint is actually
            # missing a referenced table.  Some schema-enrichment runs recover
            # the second FK after ClassMapping was produced; those rows should
            # naturally continue through the ordinary full-FK path.
            if has_partial_endpoint:
                # Exactly one endpoint must be physical.  If the alignment is
                # incomplete, leave the ordinary inferred-SR path untouched.
                physical = [item for item in endpoints if isinstance(item, tuple)]
                partial = [item for item in endpoints if isinstance(item, dict) and item.get("partial_endpoint")]
                if len(physical) != 1 or len(partial) != 1:
                    continue

                _physical_fk, physical_row = physical[0]
                partial_row = partial[0]
                value_profile = _partial_value_iri_profile(
                    conn,
                    table_name,
                    partial_row["source_column"],
                    schema_name,
                )
                partial_row.update(value_profile)
                physical_is_first = endpoints.index(physical[0]) == 0
                # A relation table's own ``domain_class_uri`` / ``range_class_uri``
                # are only a tentative ClassMapping result.  They must *not*
                # determine the missing side of a partial SR: an erroneous
                # relation-table alignment would otherwise turn both endpoints
                # into the same Class before OP direction is even considered.
                #
                # The physical FK is the invariant anchor.  Its referenced
                # entity table supplies a trustworthy endpoint Class; the
                # selected OP's declared domain/range will later resolve the
                # value endpoint and decide the semantic subject/object order.
                physical_class = _table_class(
                    physical_row["target_table"], final_alignment, ontology
                )
                if not physical_class:
                    evidence_rows.append(
                        {
                            **partial_row,
                            "inference_status": "physical_endpoint_class_unresolved",
                        }
                    )
                    continue

                # Preserve the raw relation-column order.  The partial-SR
                # candidate builder below records the OP-derived order
                # explicitly, so consumers never need to trust this raw order
                # as the ontology direction.
                ordered_rows = [physical_row, partial_row] if physical_is_first else [partial_row, physical_row]
                physical_row_index = 0 if physical_is_first else 1
                value_row_index = 1 - physical_row_index
                evidence_rows.append(
                    {
                        **partial_row,
                        "physical_endpoint_class_uri": physical_class,
                        "inference_status": "deferred_to_op_endpoint_invariant",
                    }
                )
                seen.add(table_name.lower())
                tasks.append(
                    {
                        "task_type": "sr_relation_partial",
                        "key": table_name,
                        "name_hint": table_name,
                        "source_table": table_name,
                        "source_column": ordered_rows[0]["source_column"],
                        "target_table": ordered_rows[1]["target_table"],
                        "target_column": ordered_rows[1]["target_column"],
                        # These fields are intentionally neutral placeholders
                        # for legacy task consumers.  Partial candidates are
                        # built from ``partial_physical_class_uri`` plus OP
                        # endpoint declarations, not from SR final_alignment.
                        "domain_class_uri": physical_class,
                        "range_class_uri": "",
                        "schema_matching": ordered_rows,
                        "partial_value_column": partial_row["source_column"],
                        "partial_endpoint_invariant": True,
                        "partial_physical_row_index": physical_row_index,
                        "partial_value_row_index": value_row_index,
                        "partial_physical_class_uri": physical_class,
                        **value_profile,
                    }
                )
                continue

            # Ordinary full-FK SR path remains unchanged.
            if len(endpoints) < 2 or not all(isinstance(item, tuple) for item in endpoints):
                continue
            domain_class = (
                _table_class(endpoints[0][1]["target_table"], final_alignment, ontology)
                or endpoints[0][0].get("domain_class_hint")
                or entry.get("domain_class_uri")
            )
            range_class = (
                _table_class(endpoints[1][1]["target_table"], final_alignment, ontology)
                or endpoints[1][0].get("range_class_hint")
                or entry.get("range_class_uri")
            )
            if not domain_class or not range_class:
                continue
            seen.add(table_name.lower())
            tasks.append(
                {
                    "task_type": "sr_relation",
                    "key": table_name,
                    "name_hint": table_name,
                    "source_table": table_name,
                    "source_column": endpoints[0][1]["source_column"],
                    "target_table": endpoints[1][1]["target_table"],
                    "target_column": endpoints[1][1]["target_column"],
                    "domain_class_uri": domain_class,
                    "range_class_uri": range_class,
                    "schema_matching": [x[1] for x in endpoints],
                }
            )
        for table_name, info in sorted(enriched_schema.items()):
            if table_name.lower() in seen:
                continue
            edges = _rule4_inferred_sr_edges(table_name, info)
            if len(edges) != 2:
                continue

            endpoints = []
            for edge in edges:
                metrics = _overlap_metrics(
                    conn,
                    edge["source_table"],
                    edge["source_column"],
                    edge["target_table"],
                    edge["target_column"],
                    schema_name,
                )
                match_row = {
                    **edge,
                    **metrics,
                    "relation_key": table_name.lower(),
                }
                evidence_rows.append(match_row)
                endpoints.append(match_row)

            # IND is the value-level half of Rule 4. Both FK-key columns must
            # be contained in the referenced key domain; one strong endpoint
            # cannot license a relation whose other endpoint is unsupported.
            if any(
                float(row.get("source_in_target") or 0.0)
                < INFERRED_SR_MIN_SOURCE_IN_TARGET
                for row in endpoints
            ):
                continue

            left, right = edges
            domain_class = _table_class(left["target_table"], final_alignment, ontology)
            range_class = _table_class(right["target_table"], final_alignment, ontology)
            if not domain_class or not range_class:
                continue
            seen.add(table_name.lower())
            tasks.append(
                {
                    "task_type": "sr_relation_inferred",
                    "key": f"{table_name}::{left['source_column']}__{right['source_column']}",
                    "name_hint": table_name,
                    "source_table": table_name,
                    "source_column": left["source_column"],
                    "target_table": right["target_table"],
                    "target_column": right["target_column"],
                    "domain_class_uri": domain_class,
                    "range_class_uri": range_class,
                    "schema_matching": endpoints,
                }
            )
    finally:
        conn.close()
    return tasks, evidence_rows


def _expand_endpoint_values(
    values: list[str] | tuple[str, ...] | None,
    ontology: dict[str, Any],
) -> list[str]:
    """Expand OWL union expressions used by a property endpoint.

    ``read_ontology`` stores anonymous and named ``owl:unionOf`` expressions
    in ``union_members``.  Endpoint matching must recurse through those
    expressions: a property whose domain is ``unionOf(River, Lake)`` is a
    valid candidate for either table class.  Opaque intersection/complement
    blank nodes are intentionally omitted because treating them as a free
    disjunction would admit semantically unrelated relations.
    """
    union_members = ontology.get("union_members", {}) or {}
    known_classes = set(ontology.get("classes", []) or [])
    expanded: list[str] = []
    seen: set[str] = set()

    def visit(value: object, stack: set[str]) -> None:
        text = str(value or "")
        if not text or text in stack:
            return
        members = union_members.get(text)
        if members:
            for member in members:
                visit(member, {*stack, text})
            return
        # A named class may be represented by a URI not explicitly typed in
        # the source ontology; retain URI-shaped terms for compatibility with
        # legacy snapshots.  Bare blank-node identifiers are opaque.
        if text not in known_classes and not text.startswith(("http://", "https://", "urn:")):
            return
        if text not in seen:
            seen.add(text)
            expanded.append(text)

    for value in values or []:
        visit(value, set())
    return expanded


def _class_match_kind(hint: str, prop_values: list[str], ontology: dict[str, Any]) -> str:
    if not hint:
        return "missing_hint"
    if not prop_values:
        return "op_has_no_declared_class"
    expanded = _expand_endpoint_values(prop_values, ontology)
    if not expanded:
        return "opaque_class_expression"
    for value in expanded:
        if hint == value or local_name(hint).lower() == local_name(value).lower():
            return "exact"
    for value in expanded:
        if value in ontology.get("ancestors_of", {}).get(hint, []):
            return "table_class_is_subclass_of_op_class"
    for value in expanded:
        if hint in ontology.get("ancestors_of", {}).get(value, []):
            return "table_class_is_parent_of_op_class"
    return "conflict"


def _class_match_explanation(side: str, hint: str, prop_values: list[str], ontology: dict[str, Any]) -> str:
    kind = _class_match_kind(hint, prop_values, ontology)
    if kind == "exact":
        return f"{side}: exact {local_name(hint)}"
    if kind == "table_class_is_subclass_of_op_class":
        return f"{side}: compatible because table class {local_name(hint)} is subclass of OP class {[local_name(v) for v in prop_values]}"
    if kind == "table_class_is_parent_of_op_class":
        return f"{side}: OP class is more specific than table class; use only with strong name/role evidence"
    if kind == "op_has_no_declared_class":
        return f"{side}: OP has no declared class"
    if kind == "opaque_class_expression":
        return f"{side}: OP class expression is opaque (intersection/complement); no endpoint claim"
    return f"{side}: conflict with {[local_name(v) for v in prop_values]}"


def _class_compatible(hint: str, candidate: str, ontology: dict[str, Any]) -> bool:
    if not hint or not candidate:
        return False
    ancestors = ontology.get("ancestors_of", {})
    hint_values = _expand_endpoint_values([hint], ontology) or [hint]
    candidate_values = _expand_endpoint_values([candidate], ontology) or [candidate]
    for hint_value in hint_values:
        for candidate_value in candidate_values:
            if are_classes_disjoint(hint_value, candidate_value, ontology):
                continue
            if hint_value == candidate_value or local_name(hint_value).lower() == local_name(candidate_value).lower():
                return True
            if (
                candidate_value in ancestors.get(hint_value, [])
                or hint_value in ancestors.get(candidate_value, [])
            ):
                return True
    return False


def _restriction_endpoint_support(
    domain_hint: str,
    range_hint: str,
    op_meta: dict[str, Any],
    ontology: dict[str, Any],
) -> dict[str, Any]:
    """Return restriction evidence whose endpoints close under class hierarchy."""
    hits = []
    for item in op_meta.get("restrictions", []) or []:
        cls = item.get("class") or ""
        values = item.get("values") or []
        domain_ok = _class_compatible(domain_hint, cls, ontology)
        range_ok = any(_class_compatible(range_hint, value, ontology) for value in values)
        if domain_ok and range_ok:
            hits.append(
                {
                    "class": item.get("class_local") or local_name(cls),
                    "values": item.get("values_local") or [local_name(v) for v in values],
                }
            )
    return {"hits": hits[:3]}


def _matching_constructions(task: dict[str, Any], op_meta: dict[str, Any]) -> list[str]:
    edge_norms = [
        tuple(_norm(row.get(k)) for k in ("source_table", "source_column", "target_table", "target_column"))
        for row in task.get("schema_matching", [])
    ]
    construction_hits = []
    for text in op_meta.get("construction", []) or []:
        for match in CONSTRUCTION_RE.finditer(text):
            if tuple(_norm(x) for x in match.groups()) in edge_norms:
                construction_hits.append(match.group(0))
    return construction_hits


def _class_endpoint_compatible(hint: str, prop_values: list[str], ontology: dict[str, Any]) -> bool:
    return _class_match_kind(hint, prop_values, ontology) in {
        "exact",
        "table_class_is_subclass_of_op_class",
        "table_class_is_parent_of_op_class",
    }


def _has_strong_fk_evidence(task: dict[str, Any]) -> bool:
    """Whether the physical FK evidence is strong enough for a weak candidate.

    A one-sided endpoint match is useful for schemas whose table Class is a
    deliberately coarse or missing ontology class.  It is safe to expose to
    the LLM only when the database values themselves establish the join.
    """
    return any(
        row.get("evidence_type") == "equivalence_column"
        or float(row.get("source_in_target") or 0.0) >= 0.95
        for row in task.get("schema_matching", []) or []
    )


def _weak_role_match(
    task: dict[str, Any],
    op_local_name: str,
) -> bool:
    """Require an explicit relation/column role signal for weak admission."""
    op_tokens = _meaningful_tokens(op_local_name)
    if not op_tokens:
        return False
    role_tokens = set()
    for value in (
        task.get("name_hint"),
        task.get("source_table"),
        task.get("source_column"),
        task.get("target_table"),
        task.get("target_column"),
        task.get("constraint_name"),
    ):
        role_tokens |= _meaningful_tokens(value)
    for row in task.get("schema_matching", []) or []:
        role_tokens |= _meaningful_tokens(row.get("constraint_name"))
    return bool(op_tokens & role_tokens) or _norm(op_local_name) in _norm(
        " ".join(str(value or "") for value in (
            task.get("name_hint"),
            task.get("source_table"),
            task.get("source_column"),
            task.get("target_table"),
            task.get("target_column"),
        ))
    )


def _filter_endpoint_candidates(
    task: dict[str, Any],
    ontology: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    name_hint = task["name_hint"]
    domain_hint = task["domain_class_uri"]
    range_hint = task["range_class_uri"]
    table_tokens = _meaningful_tokens(task.get("source_table"))
    column_tokens = _meaningful_tokens(task.get("source_column"))
    target_tokens = _meaningful_tokens(task.get("target_table")) | _meaningful_tokens(task.get("target_column"))
    constraint_tokens = set()
    for row in task.get("schema_matching", []) or []:
        constraint_tokens |= _meaningful_tokens(row.get("constraint_name"))
    rows = []
    for uri, info in ontology.get("object_properties", {}).items():
        local = local_name(uri)
        op_meta = metadata.get("ops", {}).get(uri, {})
        restriction_support = _restriction_endpoint_support(domain_hint, range_hint, op_meta, ontology)
        construction_hits = _matching_constructions(task, op_meta)
        op_tokens = _meaningful_tokens(local)
        declared_endpoint_match = (
            _class_endpoint_compatible(domain_hint, info.get("domain", []), ontology)
            and _class_endpoint_compatible(range_hint, info.get("range", []), ontology)
        )
        restriction_endpoint_match = bool(restriction_support["hits"])
        construction_match = bool(construction_hits)
        name_pattern_match = bool(
            op_tokens
            & (table_tokens | column_tokens | target_tokens | constraint_tokens)
        ) or _norm(local) in _norm(
            " ".join(
                [
                    str(name_hint),
                    str(task.get("source_table")),
                    str(task.get("source_column")),
                    str(task.get("target_table")),
                    str(task.get("target_column")),
                    " ".join(row.get("constraint_name", "") for row in task.get("schema_matching", []) or []),
                ]
            )
        )
        # Tier-2 retrieval: if the table Class is coarse/under-specified, one
        # endpoint may be unavailable even though the physical FK and role
        # name are decisive.  Admit this candidate for LLM adjudication only;
        # it is never selected by the deterministic fallback.  Explicit OWL
        # disjointness still blocks the candidate.
        declared_domains = list(info.get("domain", []) or [])
        declared_ranges = list(info.get("range", []) or [])
        explicit_disjoint = any(
            are_classes_disjoint(domain_hint, value, ontology)
            for value in declared_domains
        ) or any(
            are_classes_disjoint(range_hint, value, ontology)
            for value in declared_ranges
        )
        domain_side_match = _class_endpoint_compatible(domain_hint, declared_domains, ontology)
        range_side_match = _class_endpoint_compatible(range_hint, declared_ranges, ontology)
        weak_endpoint_match = bool(
            not declared_endpoint_match
            and not restriction_endpoint_match
            and not construction_match
            and not explicit_disjoint
            and _has_strong_fk_evidence(task)
            and (domain_side_match or range_side_match)
            and (name_pattern_match or _weak_role_match(task, local))
        )
        # A name match is context for the chooser, never retrieval evidence.
        # Admitting a candidate from a table/column token alone is what lets
        # unrelated ontology properties become inferred SR relations.  Keep
        # only an explicit endpoint closure, an OWL restriction closure, or an
        # exact physical sql:construction edge; all three are data/ontology
        # evidence independent of dataset names and target scores.
        if not (
            declared_endpoint_match
            or restriction_endpoint_match
            or construction_match
            or weak_endpoint_match
        ):
            continue

        rows.append(
            {
                "uri": uri,
                "local_name": local,
                "domain": info.get("domain", []),
                "range": info.get("range", []),
                "declared_endpoint_match": declared_endpoint_match,
                "restriction_endpoint_match": restriction_endpoint_match,
                "restriction_endpoint_hits": restriction_support["hits"],
                "construction_match": construction_match,
                "weak_endpoint_match": weak_endpoint_match,
                "endpoint_evidence_tier": (
                    "declared_or_restriction_or_construction"
                    if not weak_endpoint_match
                    else "one_endpoint_plus_strong_fk_role"
                ),
                "matching_construction": construction_hits[:2],
                "name_pattern_match": name_pattern_match,
                "domain_match_kind": _class_match_kind(domain_hint, info.get("domain", []), ontology),
                "range_match_kind": _class_match_kind(range_hint, info.get("range", []), ontology),
                "domain_closure_explanation": _class_match_explanation("domain", domain_hint, info.get("domain", []), ontology),
                "range_closure_explanation": _class_match_explanation("range", range_hint, info.get("range", []), ontology),
                "name_exact_match": _norm(name_hint) == _norm(local),
                "table_role_tokens": sorted(table_tokens & op_tokens),
                "column_role_tokens": sorted(column_tokens & op_tokens),
                "target_role_tokens": sorted(target_tokens & op_tokens),
                "constraint_role_tokens": sorted(constraint_tokens & op_tokens),
                "_task": task,
            }
        )
    return rows


def _reverse_sr_task(task: dict[str, Any]) -> dict[str, Any]:
    rows = task.get("schema_matching") or []
    if len(rows) < 2:
        return task
    left, right = rows[0], rows[1]
    reversed_task = dict(task)
    reversed_task["source_column"] = right.get("source_column")
    reversed_task["target_table"] = left.get("target_table")
    reversed_task["target_column"] = left.get("target_column")
    reversed_task["domain_class_uri"] = task.get("range_class_uri")
    reversed_task["range_class_uri"] = task.get("domain_class_uri")
    reversed_task["schema_matching"] = [right, left]
    reversed_task["sr_direction"] = "reversed"
    return reversed_task


def _endpoint_classes(values: list[str], ontology: dict[str, Any]) -> list[str]:
    """Expand named/anonymous union members deterministically."""
    return sorted(
        set(_expand_endpoint_values(values, ontology)),
        key=lambda uri: (local_name(uri).lower(), uri),
    )


def _uri_namespace(uri: str | None) -> str:
    text = str(uri or "")
    if "#" in text:
        return text.rsplit("#", 1)[0] + "#"
    if "/" in text:
        return text.rsplit("/", 1)[0] + "/"
    if text.lower().startswith("urn:") and ":" in text[4:]:
        return text.rsplit(":", 1)[0] + ":"
    return ""


def _partial_class_compatible(
    known_class: str,
    endpoint_class: str,
    ontology: dict[str, Any],
) -> bool:
    """Match endpoint classes without cross-namespace local-name leakage."""
    if not known_class or not endpoint_class:
        return False
    ancestors = ontology.get("ancestors_of", {}) or {}
    known_values = _expand_endpoint_values([known_class], ontology) or [known_class]
    endpoint_values = _expand_endpoint_values([endpoint_class], ontology) or [endpoint_class]
    for known_value in known_values:
        for endpoint_value in endpoint_values:
            if are_classes_disjoint(known_value, endpoint_value, ontology):
                continue
            if known_value == endpoint_value:
                return True
            if (
                endpoint_value in (ancestors.get(known_value, []) or [])
                or known_value in (ancestors.get(endpoint_value, []) or [])
            ):
                # An explicit ontology hierarchy edge is valid even when an
                # ontology imports its parent class from another namespace.
                return True
            if (
                _uri_namespace(known_value) == _uri_namespace(endpoint_value)
                and local_name(known_value).lower() == local_name(endpoint_value).lower()
            ):
                return True
    return False


def _partial_endpoint_assignments(
    physical_class: str,
    op_info: dict[str, Any],
    op_meta: dict[str, Any],
    ontology: dict[str, Any],
) -> list[dict[str, Any]]:
    """Resolve physical endpoint roles from declarations and OWL restrictions."""
    assignments: list[dict[str, Any]] = []
    domains = _endpoint_classes(list(op_info.get("domain", []) or []), ontology)
    ranges = _endpoint_classes(list(op_info.get("range", []) or []), ontology)
    if domains and ranges:
        if any(_partial_class_compatible(physical_class, value, ontology) for value in domains):
            assignments.append(
                {
                    "physical_role": "domain",
                    "counterpart_values": ranges,
                    "endpoint_evidence": "declared",
                    "restriction": None,
                }
            )
        if any(_partial_class_compatible(physical_class, value, ontology) for value in ranges):
            assignments.append(
                {
                    "physical_role": "range",
                    "counterpart_values": domains,
                    "endpoint_evidence": "declared",
                    "restriction": None,
                }
            )

    for restriction in op_meta.get("restrictions", []) or []:
        owner_class = str(restriction.get("class") or "")
        value_classes = _endpoint_classes(
            list(restriction.get("values", []) or []), ontology
        )
        if not owner_class or not value_classes:
            continue
        if _partial_class_compatible(physical_class, owner_class, ontology):
            assignments.append(
                {
                    "physical_role": "domain",
                    "counterpart_values": value_classes,
                    "endpoint_evidence": "restriction",
                    "restriction": restriction,
                }
            )
        if any(
            _partial_class_compatible(physical_class, value, ontology)
            for value in value_classes
        ):
            assignments.append(
                {
                    "physical_role": "range",
                    "counterpart_values": [owner_class],
                    "endpoint_evidence": "restriction",
                    "restriction": restriction,
                }
            )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for assignment in assignments:
        key = (
            str(assignment["physical_role"]),
            str(assignment["endpoint_evidence"]),
            tuple(sorted(str(value) for value in assignment["counterpart_values"])),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(assignment)
    return deduped


def _partial_counterpart_options(
    endpoint_values: list[str],
    final_alignment: dict[str, Any],
    enriched_schema: dict[str, Any],
    ontology: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Return concrete ``(Class, table, key)`` options for an OP endpoint.

    The SR relation table itself is deliberately absent from this lookup.  Its
    ClassMapping may be the source of the original error; only entity-table
    mappings are used to materialize the endpoint required by the selected OP.
    """
    options: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for endpoint_class in _endpoint_classes(endpoint_values, ontology):
        target_table = _aligned_table_for_class(
            endpoint_class, final_alignment, enriched_schema, ontology
        )
        if not target_table:
            continue
        aligned_class = _table_class(target_table, final_alignment, ontology)
        if aligned_class and not _partial_class_compatible(
            aligned_class, endpoint_class, ontology
        ):
            continue
        target_column = _primary_key_for_table(target_table, enriched_schema)
        target_info = (
            enriched_schema.get(target_table)
            or enriched_schema.get(str(target_table).lower())
            or {}
        )
        target_columns = target_info.get("columns") or {}
        target_column_names = (
            list(target_columns.keys())
            if isinstance(target_columns, dict)
            else list(target_columns)
        )
        known_keys = {
            str(column).lower()
            for column in (
                target_column_names
                + list(target_info.get("primary_key") or [])
            )
        }
        if not target_column or str(target_column).lower() not in known_keys:
            continue
        key = (endpoint_class, target_table, target_column)
        if key not in seen:
            seen.add(key)
            options.append(key)
    return options


def _partial_sr_endpoint_candidates(
    task: dict[str, Any],
    ontology: dict[str, Any],
    metadata: dict[str, Any],
    final_alignment: dict[str, Any],
    enriched_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build partial-SR candidates from a physical FK and OP endpoints.

    The physical endpoint is invariant.  Its semantic role must be supported
    by a declared endpoint or an OWL restriction closure; lexical similarity
    never admits a candidate by itself.  A scalar IRI endpoint can be emitted
    directly, including polymorphic restriction ranges.  Other scalar values
    require one unambiguous aligned entity table and a real key column.
    """
    rows = task.get("schema_matching") or []
    try:
        physical_index = int(task.get("partial_physical_row_index"))
        value_index = int(task.get("partial_value_row_index"))
    except (TypeError, ValueError):
        return []
    if (
        len(rows) != 2
        or physical_index not in (0, 1)
        or value_index not in (0, 1)
        or physical_index == value_index
    ):
        return []

    physical_row = dict(rows[physical_index])
    value_row = dict(rows[value_index])
    physical_class = str(task.get("partial_physical_class_uri") or "")
    if not physical_class or not physical_row.get("target_table"):
        return []
    value_term_type = str(task.get("partial_value_term_type") or "")
    value_is_iri = value_term_type == "iri"

    candidates: list[dict[str, Any]] = []
    for uri, info in sorted((ontology.get("object_properties", {}) or {}).items()):
        op_meta = metadata.get("ops", {}).get(uri, {})
        assignments = _partial_endpoint_assignments(
            physical_class, info, op_meta, ontology
        )
        for assignment in assignments:
            physical_role = str(assignment["physical_role"])
            endpoint_evidence = str(assignment["endpoint_evidence"])
            counterpart_classes = _endpoint_classes(
                list(assignment["counterpart_values"]), ontology
            )
            if not counterpart_classes:
                continue

            if value_is_iri:
                # The source value itself is the object IRI.  No counterpart
                # table is required, so a union/restriction endpoint can stay
                # polymorphic without inventing a single relational target.
                materializations = [
                    (counterpart_class, "", "")
                    for counterpart_class in counterpart_classes
                ]
            else:
                options = _partial_counterpart_options(
                    counterpart_classes,
                    final_alignment,
                    enriched_schema,
                    ontology,
                )
                by_entity_key: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
                for option in options:
                    by_entity_key.setdefault(
                        (option[1].lower(), option[2].lower()), []
                    ).append(option)
                # A non-IRI scalar cannot safely become an object solely from
                # an OP/table name.  Require one entity table/key target.
                if len(by_entity_key) != 1:
                    continue
                materializations = sorted(
                    next(iter(by_entity_key.values())),
                    key=lambda option: (
                        local_name(option[0]).lower(),
                        option[0],
                    ),
                )[:1]

            for counterpart_class, counterpart_table, counterpart_key in materializations:
                value_endpoint = {
                    **value_row,
                    "target_table": counterpart_table,
                    "target_column": counterpart_key,
                    "target_class_uri": counterpart_class,
                    "inferred_target_table": bool(counterpart_table),
                    "partial_value_term_type": value_term_type or None,
                }

                if physical_role == "domain":
                    subject_row, object_row = physical_row, value_endpoint
                    domain_class, range_class = physical_class, counterpart_class
                    subject_raw_index = physical_index
                else:
                    subject_row, object_row = value_endpoint, physical_row
                    domain_class, range_class = counterpart_class, physical_class
                    subject_raw_index = value_index

                # ``sr_direction`` is retained for legacy consumers, but the
                # explicit endpoint fields below are authoritative.  It means
                # whether the semantic subject is first or second in the raw
                # relation-table column order.
                raw_direction = "normal" if subject_raw_index == 0 else "reversed"
                effective_task = {
                    **task,
                    "source_column": subject_row.get("source_column"),
                    "target_table": object_row.get("target_table"),
                    "target_column": object_row.get("target_column"),
                    "domain_class_uri": domain_class,
                    "range_class_uri": range_class,
                    "schema_matching": [subject_row, object_row],
                    "sr_direction": raw_direction,
                    "partial_value_column": task.get("partial_value_column"),
                    "partial_value_term_type": value_term_type or None,
                }

                # Reuse the common explanations, then enforce the provenance
                # of the assignment which admitted this OP.  A lexical match
                # cannot satisfy either endpoint-evidence branch.
                scored = next(
                    (
                        row
                        for row in _filter_endpoint_candidates(
                            effective_task, ontology, metadata
                        )
                        if row.get("uri") == uri
                        and (
                            (
                                endpoint_evidence == "declared"
                                and row.get("declared_endpoint_match")
                            )
                            or (
                                endpoint_evidence == "restriction"
                                and row.get("restriction_endpoint_match")
                            )
                        )
                    ),
                    None,
                )
                if not scored:
                    continue

                candidates.append(
                    {
                        **scored,
                        "partial_endpoint_invariant": True,
                        "partial_physical_endpoint_role": physical_role,
                        "partial_endpoint_evidence": endpoint_evidence,
                        "partial_effective_domain_class_uri": domain_class,
                        "partial_effective_range_class_uri": range_class,
                        "partial_value_column": task.get("partial_value_column"),
                        "partial_value_term_type": value_term_type or None,
                        "partial_value_iri_ratio": task.get(
                            "partial_value_iri_ratio"
                        ),
                        "partial_counterpart_class_uris": counterpart_classes,
                        "partial_sr_subject_column": subject_row.get("source_column"),
                        "partial_sr_object_column": object_row.get("source_column"),
                        "partial_sr_subject_ref_table": subject_row.get("target_table"),
                        "partial_sr_object_ref_table": object_row.get("target_table"),
                        "partial_counterpart_table": counterpart_table,
                        "partial_counterpart_key": counterpart_key,
                        "sr_direction": raw_direction,
                        "_task": effective_task,
                    }
                )

    # A multi-valued OWL endpoint can yield several equivalent table paths.
    # Keep one deterministic representation per OP/direction; the endpoint
    # closure and explicit columns stay intact and no dataset name is involved.
    candidates.sort(
        key=lambda row: (
            -int(bool(row.get("declared_endpoint_match"))),
            -int(bool(row.get("restriction_endpoint_match"))),
            -int(row.get("partial_value_term_type") == "iri"),
            -int(row.get("domain_match_kind") == "exact"),
            -int(row.get("range_match_kind") == "exact"),
            str(row.get("uri") or ""),
            str(row.get("sr_direction") or ""),
            str(row.get("partial_counterpart_table") or "").lower(),
        )
    )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in candidates:
        key = (str(row.get("uri") or ""), str(row.get("sr_direction") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _sr_direction_candidates(
    task: dict[str, Any],
    ontology: dict[str, Any],
    metadata: dict[str, Any],
    final_alignment: dict[str, Any] | None = None,
    enriched_schema: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if task.get("partial_endpoint_invariant"):
        if final_alignment is None or enriched_schema is None:
            return []
        return _partial_sr_endpoint_candidates(
            task,
            ontology,
            metadata,
            final_alignment,
            enriched_schema,
        )
    normal_task = {**task, "sr_direction": task.get("sr_direction") or "normal"}
    reversed_task = _reverse_sr_task(task)
    return _filter_endpoint_candidates(normal_task, ontology, metadata) + _filter_endpoint_candidates(
        reversed_task, ontology, metadata
    )


def _selected_sr_endpoints(
    task: dict[str, Any],
    selected_candidate: dict[str, Any] | None,
    selected_direction: str,
) -> dict[str, Any]:
    """Return the endpoint fields persisted for an accepted SR selection."""
    if selected_candidate and selected_candidate.get("partial_endpoint_invariant"):
        return {
            "domain_class_uri": selected_candidate.get("partial_effective_domain_class_uri") or "",
            "range_class_uri": selected_candidate.get("partial_effective_range_class_uri") or "",
            "sr_direction": selected_candidate.get("sr_direction") or selected_direction,
            "sr_subject_column": selected_candidate.get("partial_sr_subject_column"),
            "sr_object_column": selected_candidate.get("partial_sr_object_column"),
            "sr_subject_ref_table": selected_candidate.get("partial_sr_subject_ref_table"),
            "sr_object_ref_table": selected_candidate.get("partial_sr_object_ref_table"),
            "partial_endpoint_invariant": True,
            "partial_physical_endpoint_role": selected_candidate.get(
                "partial_physical_endpoint_role"
            ),
            "partial_endpoint_evidence": selected_candidate.get(
                "partial_endpoint_evidence"
            ),
            "partial_value_column": selected_candidate.get("partial_value_column")
            or task.get("partial_value_column"),
            "partial_value_term_type": selected_candidate.get(
                "partial_value_term_type"
            )
            or task.get("partial_value_term_type"),
            "partial_counterpart_class_uris": selected_candidate.get(
                "partial_counterpart_class_uris"
            )
            or [],
        }

    rows = task.get("schema_matching") or []
    row0 = rows[0] if len(rows) >= 1 else {}
    row1 = rows[1] if len(rows) >= 2 else row0
    return {
        "domain_class_uri": task.get("domain_class_uri") or "",
        "range_class_uri": task.get("range_class_uri") or "",
        "sr_direction": selected_direction or task.get("sr_direction"),
        "sr_subject_column": (
            row1.get("source_column")
            if selected_direction == "reversed"
            else task.get("source_column")
        ),
        "sr_object_column": (
            row0.get("source_column")
            if selected_direction == "reversed"
            else row1.get("source_column")
        ),
        "sr_subject_ref_table": (
            row1.get("target_table")
            if selected_direction == "reversed"
            else row0.get("target_table")
        ),
        "sr_object_ref_table": (
            row0.get("target_table")
            if selected_direction == "reversed"
            else row1.get("target_table")
        ),
        "partial_endpoint_invariant": False,
        "partial_physical_endpoint_role": None,
        "partial_endpoint_evidence": None,
        "partial_value_column": None,
        "partial_value_term_type": None,
        "partial_counterpart_class_uris": [],
    }


def _compact_op_info(op_uri: str, meta: dict[str, Any], scored: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    op_meta = meta["ops"].get(op_uri, {})
    construction_hits = _matching_constructions(task, op_meta)
    compact = {
        "uri": op_uri,
        "local_name": op_meta.get("local_name") or local_name(op_uri),
        "sr_direction": task.get("sr_direction"),
        "candidate_domain_class": local_name(task.get("domain_class_uri")),
        "candidate_range_class": local_name(task.get("range_class_uri")),
        "domain_local": [local_name(x) for x in op_meta.get("domain", [])],
        "range_local": [local_name(x) for x in op_meta.get("range", [])],
        "declared_endpoint_match": scored.get("declared_endpoint_match", False),
        "restriction_endpoint_match": scored.get("restriction_endpoint_match", False),
        "restriction_endpoint_hits": scored.get("restriction_endpoint_hits", []),
        "name_pattern_match": scored.get("name_pattern_match", False),
        "domain_match_kind": scored.get("domain_match_kind"),
        "range_match_kind": scored.get("range_match_kind"),
        "domain_closure_explanation": scored.get("domain_closure_explanation"),
        "range_closure_explanation": scored.get("range_closure_explanation"),
        "name_exact_match": scored.get("name_exact_match", False),
        "table_role_tokens": scored.get("table_role_tokens", []),
        "column_role_tokens": scored.get("column_role_tokens", []),
        "target_role_tokens": scored.get("target_role_tokens", []),
        "constraint_role_tokens": scored.get("constraint_role_tokens", []),
        "exact_sql_construction_match": bool(construction_hits),
        "matching_construction": construction_hits[:2],
        "inverse_of": [local_name(x) for x in op_meta.get("inverse_of", [])],
        "subproperty_of": [local_name(x) for x in op_meta.get("subproperty_of", [])],
        "restrictions": [
            {
                "class": x.get("class_local"),
                "class_tables": x.get("class_tables", [])[:2],
                "values": x.get("values_local", [])[:3],
            }
            for x in op_meta.get("restrictions", [])[:3]
        ],
    }
    # Endpoint materialization is not presentation-only metadata for a partial
    # SR: R2RML must consume exactly the OP-derived subject/object pair that
    # survived candidate filtering.  Preserve it through the compact prompt
    # representation so a later selection cannot fall back to stale alignment
    # endpoints.
    for key in (
        "partial_endpoint_invariant",
        "partial_physical_endpoint_role",
        "partial_endpoint_evidence",
        "partial_effective_domain_class_uri",
        "partial_effective_range_class_uri",
        "partial_value_column",
        "partial_value_term_type",
        "partial_value_iri_ratio",
        "partial_counterpart_class_uris",
        "partial_sr_subject_column",
        "partial_sr_object_column",
        "partial_sr_subject_ref_table",
        "partial_sr_object_ref_table",
        "partial_counterpart_table",
        "partial_counterpart_key",
    ):
        if key in scored:
            compact[key] = scored[key]
    return compact


def _build_prompt(task: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    is_partial_sr = bool(task.get("partial_endpoint_invariant"))
    payload = {
        "task": "从候选 ontology ObjectProperty 里选择当前数据库关系最合适的 OP。只能选候选 URI 或 null。",
        "hard_rules": [
            "这是从零 OP 模块：候选来自 ontology 全部 ObjectProperty 的 domain/range endpoint closure，不来自旧 OP top-k。",
            "等价列/包含列证据说明当前数据库列真实值如何连接目标主键/候选键；它用来确认关系端点和方向，不等于 OP 名称匹配。",
            "FK constraint name 和表/列名只用于模式匹配召回候选；最终选择必须结合 endpoint、等价列/IND、construction/restriction 证据。",
            "普通 FK：source table class 是 domain，referenced table class 是 range。",
            "SR 关系表：关系表名是 role/name hint；候选里会同时给出 normal/reversed 两种方向，必须根据 endpoint 和证据选择方向。",
            "如果 ClassMapping top-1 与 FK constraint/table role 明显冲突，不要机械相信 top-1；说明冲突并选择证据链最闭合的候选。",
            "如果两个 OP endpoint 都兼容，优先选择 relation/table/column role 更具体的 OP。",
            "如果一个候选是另一个的 subPropertyOf，且名称/角色支持，选更具体的子属性。",
            "如果 endpoint 方向相反，只能在 inverseOf 明确支持时选择相应方向；否则不要硬选。",
            "如果所有候选都明显冲突，返回 null。",
        ],
        "relation": {
            "key": task["key"],
            "task_type": task["task_type"],
            "name_hint": task["name_hint"],
            "domain_class": local_name(task["domain_class_uri"]),
            "range_class": local_name(task["range_class_uri"]),
        },
        "schema_matching_evidence": [
            {
                "source": f"{row['source_table']}.{row['source_column']}",
                "target": f"{row['target_table']}.{row['target_column']}",
                "fk_constraint_name": row.get("constraint_name"),
                "manual_jaccard": row.get("manual_jaccard"),
                "source_in_target_IND": row.get("source_in_target"),
                "target_in_source": row.get("target_in_source"),
                "source_distinct": row.get("source_distinct"),
                "target_distinct": row.get("target_distinct"),
                "intersection": row.get("intersection"),
                "evidence_type": row.get("evidence_type"),
            }
            for row in task.get("schema_matching", [])
        ],
        "candidate_object_properties": candidates,
        "output_schema": {
            "selected_uri": "candidate uri or null",
            "selected_direction": "normal|reversed|null; only required for SR candidates",
            "selected_local_name": "local name or empty",
            "confidence": "high|medium|low",
            "reason": "简短中文理由，必须提到 endpoint、name/role、等价列/IND 证据是否支持",
        },
    }
    if is_partial_sr:
        payload["hard_rules"].append(
            "partial SR：一端是真实 FK。每个候选已经根据 OP domain/range 给出唯一的 "
            "partial_sr_subject/object_column；不得使用关系表旧 ClassMapping 改写它。"
        )
        payload["relation"]["partial_endpoint_invariant"] = {
            "physical_endpoint_class": local_name(task.get("partial_physical_class_uri")),
            "physical_raw_row_index": task.get("partial_physical_row_index"),
            "value_raw_row_index": task.get("partial_value_row_index"),
            "value_column": task.get("partial_value_column"),
            "value_term_type": task.get("partial_value_term_type"),
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _selected_is_candidate(selected_uri: str, candidates: list[dict[str, Any]]) -> bool:
    return bool(selected_uri) and selected_uri in {c["uri"] for c in candidates}


def _task_checkpoint_signature(task: dict[str, Any]) -> str:
    """Stable identity for safely reusing an OP task across task-list changes."""
    payload = {
        "task_type": str(task.get("task_type") or ""),
        "key": str(task.get("key") or ""),
        "source": f"{task.get('source_table')}.{task.get('source_column')}",
        "target": f"{task.get('target_table')}.{task.get('target_column')}",
        "domain_class_uri": str(task.get("domain_class_uri") or ""),
        "range_class_uri": str(task.get("range_class_uri") or ""),
    }
    if task.get("partial_endpoint_invariant"):
        # A recovered task must not reuse a checkpoint written before the
        # scalar endpoint's object/literal contract was known.
        payload.update(
            {
                "partial_value_column": str(task.get("partial_value_column") or ""),
                "partial_value_term_type": str(
                    task.get("partial_value_term_type") or ""
                ),
                "partial_physical_class_uri": str(
                    task.get("partial_physical_class_uri") or ""
                ),
            }
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _saved_checkpoint_signature(saved: dict[str, Any]) -> str:
    """Read a task identity from both current and legacy checkpoint entries."""
    signature = saved.get("task_signature")
    if signature:
        return str(signature)
    prediction = saved.get("prediction") or {}
    entry = saved.get("entry") or {}
    payload = {
        "task_type": str(prediction.get("task_type") or entry.get("scenario_type") or ""),
        "key": str(prediction.get("relation_key") or ""),
        "source": str(prediction.get("source") or ""),
        "target": str(prediction.get("target") or ""),
        "domain_class_uri": str(entry.get("domain_class_uri") or ""),
        "range_class_uri": str(entry.get("range_class_uri") or ""),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _checkpoint_entries_by_signature(checkpoint: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index completed work by semantic identity instead of mutable list index."""
    indexed: dict[str, dict[str, Any]] = {}
    for saved in (checkpoint.get("completed") or {}).values():
        if not isinstance(saved, dict):
            continue
        signature = _saved_checkpoint_signature(saved)
        # A later occurrence is the one that historically won the result-key
        # collision, so it is the safest deterministic legacy choice.
        indexed[signature] = saved
    return indexed


def _endpoint_match_weight(kind: str | None) -> float:
    return {
        "exact": 1.0,
        "table_class_is_subclass_of_op_class": 0.78,
        "table_class_is_parent_of_op_class": 0.52,
        "op_has_no_declared_class": 0.12,
    }.get(str(kind or ""), 0.0)


def _deterministic_candidate_fallback(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Pick the strongest endpoint-constrained candidate after all LLM retries fail.

    This is a generic recovery path, not a relation-name exception: it only
    ranks candidates which already passed the ontology endpoint filter.  Exact
    endpoint closure dominates, then name/role/construction evidence breaks
    ties.  Keeping this fallback means a transient provider failure does not
    silently erase a structurally valid relationship from a mapping.
    """
    if not candidates:
        return None

    # Do not manufacture an OP merely because it is lexically nearby.  A
    # recovery result must have explicit endpoint/restriction support; otherwise
    # preserving the missing result is safer and leaves an auditable failure.
    structurally_supported = [
        candidate
        for candidate in candidates
        if candidate.get("declared_endpoint_match")
        or candidate.get("restriction_endpoint_match")
        or candidate.get("matching_construction")
    ]
    if not structurally_supported:
        return None

    def score(candidate: dict[str, Any]) -> tuple[float, str, str, str]:
        endpoint = (
            _endpoint_match_weight(candidate.get("domain_match_kind"))
            + _endpoint_match_weight(candidate.get("range_match_kind"))
        )
        role_hits = sum(
            len(candidate.get(field) or [])
            for field in (
                "table_role_tokens",
                "column_role_tokens",
                "target_role_tokens",
                "constraint_role_tokens",
            )
        )
        evidence = (
            5.0 * endpoint
            + (1.2 if candidate.get("declared_endpoint_match") else 0.0)
            + (1.0 if candidate.get("restriction_endpoint_match") else 0.0)
            + (1.0 if candidate.get("matching_construction") else 0.0)
            + (2.4 if candidate.get("name_exact_match") else 0.0)
            + (0.18 * role_hits)
        )
        return (
            evidence,
            str(candidate.get("local_name") or "").lower(),
            str(candidate.get("uri") or ""),
            str(candidate.get("sr_direction") or "normal"),
        )

    # Reverse sort the score, then use URI/direction deterministically for an
    # exact tie.  The negated lexical fields avoid any dependency on dict order.
    return sorted(
        structurally_supported,
        key=lambda candidate: (
            -score(candidate)[0],
            score(candidate)[1],
            score(candidate)[2],
            score(candidate)[3],
        ),
    )[0]


def _strong_equivalence_support(task: dict[str, Any]) -> bool:
    rows = task.get("schema_matching") or []
    if not rows:
        return False
    return any(
        row.get("evidence_type") == "equivalence_column"
        or float(row.get("source_in_target") or 0.0) >= 0.95
        for row in rows
    )


def run_equivalence_op_module(
    final_alignment: dict[str, Any],
    ontology: dict[str, Any],
    enriched_schema: dict[str, Any],
    *,
    schema_name: str = DB_SCHEMA_NAME,
    output_dir: str = OUTPUT_DIR,
    ontology_path: str = ONTOLOGY_PATH,
    min_endpoint_score: float | None = None,
    sleep_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the clean OP module and return a OP-step1-compatible result."""
    min_endpoint_score = min_endpoint_score if min_endpoint_score is not None else float(os.getenv("MAMG_EQUIV_OP_MIN_ENDPOINT_SCORE", "0.5"))
    sleep_seconds = sleep_seconds if sleep_seconds is not None else float(os.getenv("MAMG_EQUIV_OP_LLM_SLEEP", "0.15"))

    fk_tasks, fk_evidence = _build_fk_tasks(enriched_schema, final_alignment, ontology, schema_name)
    sr_tasks, sr_evidence = _build_sr_tasks(final_alignment, enriched_schema, ontology, schema_name)
    tasks = fk_tasks + sr_tasks
    metadata = _parse_ontology_metadata(ontology_path)

    _write_json(
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "schema_name": schema_name,
            "evidence_rows": fk_evidence + sr_evidence,
        },
        Path(output_dir) / "schema_matching_equivalence_op.json",
    )

    result: dict[str, Any] = {}
    prediction_rows = []
    checkpoint_path = Path(output_dir) / "equivalence_op_module_checkpoint.json"
    completed: dict[str, dict[str, Any]] = {}
    completed_by_signature: dict[str, dict[str, Any]] = {}
    retry_terminal = os.getenv("MAMG_EQUIV_OP_RETRY_TERMINAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    recover_terminal = os.getenv("MAMG_EQUIV_OP_RECOVER_TERMINAL", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if os.getenv("MAMG_EQUIV_OP_RESUME", "").strip().lower() in {"1", "true", "yes"}:
        checkpoint = _read_json_if_exists(checkpoint_path) or {}
        completed_by_signature = _checkpoint_entries_by_signature(checkpoint)
        # Checkpoints are keyed by list position in legacy runs.  Adding a new
        # structural task (such as a partial SR) changes positions, but not the
        # semantic identity of finished work.  Re-index it before processing
        # so we never apply a saved prediction to a different relation.
        for idx, task in enumerate(tasks, start=1):
            saved = completed_by_signature.get(_task_checkpoint_signature(task))
            if saved:
                completed[str(idx)] = saved
        if completed:
            print(
                f"  恢复 OP 检查点: {len(completed)}/{len(tasks)} 条语义一致任务已完成"
            )
    system = (
        "You are an OBDA ontology mapping expert. "
        "Return strict JSON only. Do not use Markdown. "
        "Select ObjectProperty only from candidates."
    )

    print(f"  等价列 OP 模块任务数: FK={len(fk_tasks)}, SR={len(sr_tasks)}, total={len(tasks)}")
    for idx, task in enumerate(tasks, start=1):
        saved = completed.get(str(idx))
        saved_error = (saved or {}).get("prediction", {}).get("error")
        task_signature = _task_checkpoint_signature(task)
        if saved and not ((retry_terminal or recover_terminal) and saved_error):
            result[task["key"]] = saved["entry"]
            prediction_rows.append(saved["prediction"])
            print(f"  [{idx}/{len(tasks)}] EquivOP resume {task['key']}")
            continue
        if saved and saved_error:
            action = "recover terminal" if recover_terminal else "retry terminal"
            print(f"  [{idx}/{len(tasks)}] EquivOP {action} {task['key']}")
        if str(task.get("task_type", "")).startswith("sr_relation"):
            filtered_candidates = _sr_direction_candidates(
                task,
                ontology,
                metadata,
                final_alignment,
                enriched_schema,
            )
        else:
            filtered_candidates = _filter_endpoint_candidates(task, ontology, metadata)
        candidates = [
            _compact_op_info(item["uri"], metadata, item, item.get("_task") or task)
            for item in filtered_candidates
            if item.get("uri")
        ]
        print(f"  [{idx}/{len(tasks)}] EquivOP selecting {task['key']} candidates={len(candidates)}")
        selected_uri = ""
        selected_direction = ""
        response: dict[str, Any]
        error = None
        terminal_error = None
        fallback_used = False
        if candidates:
            if saved_error and recover_terminal:
                fallback = _deterministic_candidate_fallback(candidates)
                terminal_error = str(saved_error)
                if fallback:
                    selected_uri = str(fallback.get("uri") or "")
                    selected_direction = str(fallback.get("sr_direction") or "")
                    fallback_used = True
                    response = {
                        "recovery": "deterministic_endpoint_fallback",
                        "reason": "LLM retries were exhausted; selected the strongest ontology-endpoint-constrained candidate.",
                        "confidence": "low",
                    }
                else:
                    response = {"error": terminal_error}
                    error = terminal_error
            else:
                try:
                    response = call_llm(_build_prompt(task, candidates), system=system, prefer_fast=False)
                    selected_uri = str(response.get("selected_uri") or "")
                    selected_direction = str(response.get("selected_direction") or "")
                    if not _selected_is_candidate(selected_uri, candidates):
                        selected_uri = ""
                except Exception as exc:
                    terminal_error = str(exc)
                    fallback = _deterministic_candidate_fallback(candidates)
                    if fallback:
                        selected_uri = str(fallback.get("uri") or "")
                        selected_direction = str(fallback.get("sr_direction") or "")
                        fallback_used = True
                        response = {
                            "recovery": "deterministic_endpoint_fallback",
                            "reason": "LLM retries were exhausted; selected the strongest ontology-endpoint-constrained candidate.",
                            "confidence": "low",
                        }
                    else:
                        response = {"error": terminal_error}
                        error = terminal_error
        else:
            response = {"selected_uri": None, "reason": "No endpoint-compatible ontology ObjectProperty candidates."}

        selected_candidate = None
        if selected_uri:
            uri_candidates = [c for c in candidates if c["uri"] == selected_uri]
            direction_candidates = [
                c for c in uri_candidates
                if (
                    not selected_direction
                    or not c.get("sr_direction")
                    or c.get("sr_direction") == selected_direction
                )
            ]
            selected_candidate = direction_candidates[0] if direction_candidates else None
            if task.get("partial_endpoint_invariant"):
                # A partial SR's direction is a semantic consequence of its
                # physical endpoint and the OP declaration.  If the OP admits
                # several concrete directions, require an explicit matching
                # direction instead of taking dict/list order as evidence.
                # With one viable materialization, normalize an inconsistent
                # or omitted model token to that ontology-derived direction.
                if len(uri_candidates) == 1:
                    selected_candidate = uri_candidates[0]
                elif selected_candidate is None or not selected_direction:
                    selected_uri = ""
                    selected_direction = ""
            if selected_candidate and selected_candidate.get("sr_direction"):
                selected_direction = selected_candidate["sr_direction"]

        selected_local = local_name(selected_uri) if selected_uri else ""
        is_sr_task = str(task.get("task_type", "")).startswith("sr_relation")
        sr_endpoints = _selected_sr_endpoints(
            task,
            selected_candidate,
            selected_direction,
        ) if is_sr_task else {}
        entry = {
            "object_prop_uri": selected_uri or None,
            "confidence": response.get("confidence") or ("medium" if selected_uri else "low"),
            "method": "equivalence_column_pattern_matching_llm",
            "scenario_type": task["task_type"],
            "domain_class_uri": (
                sr_endpoints.get("domain_class_uri")
                if is_sr_task else task["domain_class_uri"]
            ),
            "range_class_uri": (
                sr_endpoints.get("range_class_uri")
                if is_sr_task else task["range_class_uri"]
            ),
            "name_hint": task["name_hint"],
            "sr_direction": sr_endpoints.get("sr_direction") if is_sr_task else None,
            "sr_subject_column": sr_endpoints.get("sr_subject_column") if is_sr_task else None,
            "sr_object_column": sr_endpoints.get("sr_object_column") if is_sr_task else None,
            "sr_subject_ref_table": sr_endpoints.get("sr_subject_ref_table") if is_sr_task else None,
            "sr_object_ref_table": sr_endpoints.get("sr_object_ref_table") if is_sr_task else None,
            "partial_endpoint_invariant": sr_endpoints.get("partial_endpoint_invariant", False),
            "partial_physical_endpoint_role": sr_endpoints.get("partial_physical_endpoint_role"),
            "partial_endpoint_evidence": sr_endpoints.get("partial_endpoint_evidence"),
            "partial_value_column": sr_endpoints.get("partial_value_column"),
            "partial_value_term_type": sr_endpoints.get("partial_value_term_type"),
            "partial_counterpart_class_uris": sr_endpoints.get(
                "partial_counterpart_class_uris", []
            ),
            "schema_matching": task.get("schema_matching", []),
            "candidates_used": candidates,
            "llm_response": response,
            "error": error,
            "terminal_error": terminal_error,
            "fallback_used": fallback_used,
        }
        result[task["key"]] = entry
        prediction = {
            "relation_key": task["key"],
            "task_type": task["task_type"],
            "source": f"{task.get('source_table')}.{task.get('source_column')}",
            "target": f"{task.get('target_table')}.{task.get('target_column')}",
            "llm_selected_uri": selected_uri,
            "llm_selected_local_name": selected_local,
            "llm_selected_direction": selected_direction,
            "candidate_count": len(candidates),
            "schema_matching": task.get("schema_matching", []),
            "llm_response": response,
            "error": error,
            "terminal_error": terminal_error,
            "fallback_used": fallback_used,
        }
        prediction_rows.append(prediction)
        completed[str(idx)] = {
            "task_signature": task_signature,
            "entry": entry,
            "prediction": prediction,
        }
        _write_json(
            {"task_count": len(tasks), "completed": completed},
            checkpoint_path,
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    _write_json(
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "setting": "clean OP module: hard domain/range filtering + equivalence/inclusion column evidence + ontology endpoint candidates + LLM",
            "candidate_policy": "no weighted OP scoring; candidates are retained by endpoint closure, restriction closure, or exact sql:construction",
            "summary": {
                "total": len(prediction_rows),
                "answered": sum(1 for row in prediction_rows if row.get("llm_selected_uri")),
                "no_answer": sum(1 for row in prediction_rows if not row.get("llm_selected_uri")),
            },
            "results": prediction_rows,
        },
        Path(output_dir) / "equivalence_op_module_predictions.json",
    )
    return result
