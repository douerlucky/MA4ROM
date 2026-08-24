"""
real_value_enhancement_agent.py  ——  真实值上下文增强

真实值增强的职责：
  ✓ 表级 Class 低置信 → 拉数据重判 Class
  ✓ data_attr 列低置信 → 拉数据值重判 DatatypeProperty
  ✓ fk_obj 列低置信 → 拉数据重判 range Class URI（不判 ObjectProperty）
  ✓ SR 表 domain/range Class 低置信 → 拉数据重判两端 Class
  ✗ 不判断任何 ObjectProperty（那是 OP 映射的职责）
"""

import json
import re
import copy
import sys
from pathlib import Path
from psycopg2 import sql

# 兼容 PyCharm profiler / run_path：确保项目根目录在 sys.path 里
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.llm_client import call_llm as _call_llm
from utils.db_utils import get_conn as _get_conn, fetch_sample_rows, _qualified_table
from utils.ontology_utils import are_classes_disjoint
from utils.candidate_ranking import (
    filter_semantically_admissible_datatype_candidates,
    filter_strong_identity_datatype_candidates,
    has_direct_datatype_name_evidence,
    has_unique_datatype_range_evidence,
)
from config import (
    REAL_VALUE_ANCESTOR_MAX_DEPTH,
    REAL_VALUE_BOOL_MAX_VALUES,
    REAL_VALUE_ENUM_DISTINCT_MAX_FOR_CODE,
    REAL_VALUE_ENUM_MAX_VALUES,
    REAL_VALUE_ENUM_NUMERIC_RATIO_THRESHOLD,
    REAL_VALUE_ENUM_PER_VALUE_LIMIT,
    REAL_VALUE_ENUM_SAMPLE_DISTINCT_RATIO_THRESHOLD,
    REAL_VALUE_FK_CONTEXT_MAX_INCOMING,
    REAL_VALUE_RULE_FALLBACK_DISTINCT_MAX,
    REAL_VALUE_RULE_FALLBACK_REPEATED_RATIO,
    REAL_VALUE_RULE_STRUCT_SIGNAL_THRESHOLD,
    REAL_VALUE_SAMPLE_ROWS_LIMIT,
    REAL_VALUE_TYPE_HIGH_GAP,
    REAL_VALUE_TYPE_HIGH_SCORE,
    REAL_VALUE_TYPE_MEDIUM_GAP,
    REAL_VALUE_TYPE_MEDIUM_SCORE,
    REAL_VALUE_TYPE_WEAK_SCORE,
    REAL_VALUE_SINGLE_VALUE_SH_MIN_ENTITIES,
    REAL_VALUE_SINGLE_VALUE_SH_MIN_GAP,
    REAL_VALUE_SINGLE_VALUE_SH_MIN_RATIO,
    DB_SCHEMA_NAME,
)


def _ns_from_uri(uri: str) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[0] + "#"
    return uri.rsplit("/", 1)[0] + "/"


def _fetch_distinct_value_profiles(
    table_name: str,
    col_name: str,
    per_value_limit: int = REAL_VALUE_ENUM_PER_VALUE_LIMIT,
    max_values: int = REAL_VALUE_ENUM_MAX_VALUES,
) -> tuple[list[str], dict[str, list[dict]], dict]:
    """
    先 DISTINCT 再分值抽样：
      1) 多读取一个 distinct 值，显式判断值域是否被 max_values 截断
      2) 每个值抽样若干行（受 per_value_limit 限制）

    第三个返回值是值域完整性证据。调用方只能在 ``complete``
    为 True 时把已观测值当作闭合枚举分区。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT DISTINCT {column} FROM {table} "
                    "WHERE {column} IS NOT NULL ORDER BY 1 LIMIT %s"
                ).format(
                    column=sql.Identifier(col_name),
                    table=_qualified_table(table_name, DB_SCHEMA_NAME),
                ),
                (max_values + 1,),
            )
            probed_vals = [row[0] for row in cur.fetchall()]
            truncated = len(probed_vals) > max_values
            vals = probed_vals[:max_values]

            value_profiles = {}
            for v in vals:
                cur.execute(
                    sql.SQL(
                        "SELECT * FROM {table} WHERE {column} = %s "
                        "ORDER BY random() LIMIT %s"
                    ).format(
                        table=_qualified_table(table_name, DB_SCHEMA_NAME),
                        column=sql.Identifier(col_name),
                    ),
                    (v, per_value_limit),
                )
                cols = [desc[0] for desc in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                value_profiles[str(v)] = rows

            return [str(v) for v in vals], value_profiles, {
                "complete": not truncated,
                "truncated": truncated,
                "observed_distinct": len(vals),
                "max_values": max_values,
            }
    except Exception as e:
        print(f"  [WARN] DISTINCT+分值抽样失败 {table_name}.{col_name}: {e}")
        conn.rollback()
        return [], {}, {
            "complete": False,
            "truncated": None,
            "observed_distinct": 0,
            "max_values": max_values,
            "reason": "distinct_profile_query_failed",
        }
    finally:
        conn.close()


def _fetch_non_null_counts(
    table_name: str,
    column_names: list[str],
) -> dict[str, int]:
    """Read full-table non-NULL counts for weak DP columns in one query.

    A five-row context sample cannot prove a column is empty.  This aggregate
    is issued only for low-confidence data attributes whose loaded sample has
    no value, and failure returns ``unknown`` rather than a false abstention.
    """
    ordered_columns = list(dict.fromkeys(col for col in column_names if col))
    if not ordered_columns:
        return {}
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            projections = sql.SQL(", ").join(
                sql.SQL("COUNT({column})").format(column=sql.Identifier(col))
                for col in ordered_columns
            )
            cur.execute(
                sql.SQL("SELECT {projections} FROM {table}").format(
                    projections=projections,
                    table=_qualified_table(table_name, DB_SCHEMA_NAME),
                )
            )
            row = cur.fetchone()
            if row is None:
                return {}
            return {
                column: int(count or 0)
                for column, count in zip(ordered_columns, row)
            }
    except Exception as exc:
        print(f"  [WARN] 非空计数失败 {table_name}: {exc}")
        conn.rollback()
        return {}
    finally:
        conn.close()


def _trim_context_row(row: dict | None, max_items: int = 12) -> dict:
    if not isinstance(row, dict):
        return {}
    out = {}
    for idx, (k, v) in enumerate(row.items()):
        if idx >= max_items:
            break
        out[k] = v
    return out


def _repair_invalid_sh_class_pair(
    table_name: str,
    entry: dict,
    table_candidates: dict | None,
    ontology: dict | None,
    enriched_schema: dict | None = None,
    alignment: dict | None = None,
) -> dict:
    """Validate an SH pair, giving physical inherited-key evidence priority.

    The table-per-subclass contract is a *relational* fact: a PK/FK constraint
    whose child columns are the inherited PK and whose referenced columns are
    the parent PK.  It remains valid even when an LLM gives the same class on
    both sides, or when the ontology does not contain a direct ``subClassOf``
    edge (both occur in real schemas).  Earlier code treated those outputs as
    invalid and replaced the parent with an arbitrary OWL ancestor, splitting
    the parent's IRI namespace.  We now preserve the physically evidenced
    parent class/table and only use the old OWL repair as a compatibility
    fallback when no schema is supplied (e.g. legacy unit tests).

    ``enriched_schema`` and ``alignment`` are optional to keep this helper
    backwards compatible.  The production path always supplies both.
    """
    if not isinstance(entry, dict) or entry.get("pattern") != "SH":
        return entry
    ontology = ontology or {}
    subclass_of = ontology.get("subclass_of", {}) or {}
    ancestors_of = ontology.get("ancestors_of", {}) or {}
    child = entry.get("sub_class_uri")
    parent = entry.get("parent_class_uri")

    def _fk_groups(table_info: dict) -> list[dict]:
        """Group physical FK members without collapsing composite constraints."""
        groups: dict[tuple, dict] = {}
        for index, raw_fk in enumerate((table_info or {}).get("foreign_keys", []) or []):
            if not isinstance(raw_fk, dict):
                continue
            source = raw_fk.get("column")
            target_table = (
                raw_fk.get("ref_table")
                or raw_fk.get("references_table")
                or raw_fk.get("target_table")
            )
            target = (
                raw_fk.get("ref_col")
                or raw_fk.get("references_column")
                or raw_fk.get("target_column")
            )
            if not source or not target_table:
                continue
            # Named constraints and declared arity are stable across schema
            # readers.  A nameless arity-1 FK is intentionally kept separate
            # so two independent references to one table cannot be merged.
            constraint = raw_fk.get("constraint_name")
            try:
                arity = int(raw_fk.get("fk_arity") or 1)
            except (TypeError, ValueError):
                arity = 1
            if constraint:
                key = ("constraint", str(constraint), str(target_table))
            elif arity > 1:
                key = ("anonymous-composite", str(target_table), arity)
            else:
                key = ("anonymous-single", index)
            group = groups.setdefault(
                key,
                {
                    "ref_table": str(target_table),
                    "source_columns": [],
                    "target_columns": [],
                },
            )
            group["source_columns"].append(str(source))
            if target:
                group["target_columns"].append(str(target))
        return list(groups.values())

    def _physical_inheritance() -> dict | None:
        if not enriched_schema:
            return None
        child_info = (enriched_schema or {}).get(table_name, {}) or {}
        columns = child_info.get("columns", {}) or {}
        child_pk = set(child_info.get("primary_key", []) or [])
        inherited = {
            col
            for col, info in (entry.get("columns", {}) or {}).items()
            if isinstance(info, dict) and info.get("role") == "sh_inherited_pk"
        }
        if not inherited or not inherited.issubset(child_pk):
            return None

        groups = [
            group for group in _fk_groups(child_info)
            if set(group.get("source_columns", [])) == inherited
        ]
        declared_parent = (table_candidates or {}).get("parent_table") or entry.get("parent_table")
        if declared_parent:
            groups = [group for group in groups if group.get("ref_table") == declared_parent]
        # More than one equally valid parent is an ambiguity, not permission to
        # pick one by score or table name.
        if len(groups) != 1:
            return None
        group = groups[0]
        parent_table = group.get("ref_table")
        parent_info = (enriched_schema or {}).get(parent_table, {}) or {}
        parent_pk = set(parent_info.get("primary_key", []) or [])
        target_cols = set(group.get("target_columns", []) or [])
        if not parent_pk or target_cols != parent_pk:
            return None
        if any(col not in columns for col in inherited):
            return None
        return {
            "parent_table": parent_table,
            "child_columns": sorted(inherited),
            "parent_columns": sorted(parent_pk),
        }

    physical = _physical_inheritance()
    if physical:
        # The aligned parent table is the strongest available semantic anchor.
        # Use its class when present; this repairs noisy LLM parent classes
        # without consulting a dataset name, query id, or target score.
        parent_table = physical["parent_table"]
        parent_entry = (alignment or {}).get(parent_table, {}) or {}
        if parent_entry.get("pattern") == "SH":
            physical_parent_class = parent_entry.get("sub_class_uri")
        else:
            physical_parent_class = parent_entry.get("class_uri")
        parent_candidates = [
            c.get("uri")
            for c in ((table_candidates or {}).get("parent_class_candidates", []) or [])
            if isinstance(c, dict) and c.get("uri")
        ]
        if physical_parent_class:
            parent = physical_parent_class
        elif parent and parent_candidates and parent not in parent_candidates:
            # No aligned parent entry is available (e.g. a partial unit
            # fixture); retain a provider class only when it is among the
            # physical parent-table candidates.
            parent = parent_candidates[0] if len(parent_candidates) == 1 else None
        child_candidates = (table_candidates or {}).get("sub_class_candidates", []) or []
        allowed_children = {
            c.get("uri") for c in child_candidates
            if isinstance(c, dict) and c.get("uri")
        }
        if child_candidates and child not in allowed_children:
            child = child_candidates[0].get("uri")
            entry["sub_class_uri"] = child
        entry["parent_table"] = parent_table
        entry["parent_class_uri"] = parent
        if child and parent and not are_classes_disjoint(child, parent, ontology):
            entry["sh_class_validation"] = {
                "status": (
                    "valid_physical_shared_class"
                    if child == parent
                    else "valid_physical_identity_non_owl_subclass"
                ),
                "physical_evidence": physical,
                "semantic_subclass_verified": bool(
                    parent == child or parent in ancestors_of.get(child, [])
                ),
            }
            return entry
        # A disjoint class pair is an explicit semantic conflict; do not
        # collapse identities even though the relational shape looks similar.
        entry["parent_class_uri"] = None
        entry["sh_class_validation"] = {
            "status": "abstained_physical_semantic_conflict",
            "physical_evidence": physical,
            "original_sub_class_uri": child,
            "original_parent_class_uri": parent,
        }
        return entry

    # In production, missing physical evidence is a reason to abstain rather
    # than invent an OWL ancestor.  Keep the historical OWL-only behavior when
    # callers omit ``enriched_schema`` for backwards-compatible fixtures.
    if enriched_schema is not None:
        entry["parent_class_uri"] = None
        entry["sh_class_validation"] = {
            "status": "abstained_without_physical_inheritance",
            "original_sub_class_uri": child,
            "original_parent_class_uri": parent,
        }
        return entry

    def ancestors(uri: str | None) -> list[str]:
        if not uri:
            return []
        direct = list(subclass_of.get(uri, []) or [])
        flattened = list(ancestors_of.get(uri, []) or [])
        return direct + [value for value in flattened if value not in direct]

    def valid_pair(child_uri: str | None, parent_uri: str | None) -> bool:
        return bool(
            child_uri
            and parent_uri
            and child_uri != parent_uri
            and parent_uri in ancestors(child_uri)
        )

    if valid_pair(child, parent):
        entry.setdefault("sh_class_validation", {"status": "valid"})
        return entry

    child_candidates = []
    if child:
        child_candidates.append({"uri": child, "score": 1.0})
    child_candidates.extend(
        candidate
        for candidate in (table_candidates or {}).get("sub_class_candidates", []) or []
        if isinstance(candidate, dict) and candidate.get("uri")
    )
    deduped: dict[str, dict] = {}
    for candidate in child_candidates:
        uri = candidate.get("uri")
        if uri and (
            uri not in deduped
            or float(candidate.get("score", 0.0) or 0.0)
            > float(deduped[uri].get("score", 0.0) or 0.0)
        ):
            deduped[uri] = candidate
    ranked_children = sorted(
        deduped.values(),
        key=lambda candidate: (
            -float(candidate.get("score", 0.0) or 0.0),
            str(candidate.get("uri") or ""),
        ),
    )
    if child and child in deduped:
        ranked_children = [deduped[child]] + [
            candidate for candidate in ranked_children if candidate.get("uri") != child
        ]

    parent_candidates = [
        candidate
        for candidate in (table_candidates or {}).get("parent_class_candidates", []) or []
        if isinstance(candidate, dict) and candidate.get("uri")
    ]
    parent_score = {
        candidate.get("uri"): float(candidate.get("score", 0.0) or 0.0)
        for candidate in parent_candidates
    }
    selected_child = None
    selected_parent = None
    for candidate in ranked_children:
        child_uri = candidate.get("uri")
        available = [
            ancestor
            for ancestor in ancestors(child_uri)
            if ancestor
            and not str(ancestor).endswith("#Thing")
            and not str(ancestor).endswith("/Thing")
        ]
        if not available:
            continue
        available.sort(
            key=lambda uri: (
                -int(uri in parent_score),
                -parent_score.get(uri, 0.0),
                0 if uri in (subclass_of.get(child_uri, []) or []) else 1,
                str(uri),
            )
        )
        selected_child = child_uri
        selected_parent = available[0]
        break

    if selected_child and selected_parent:
        entry["sub_class_uri"] = selected_child
        entry["parent_class_uri"] = selected_parent
        entry["sh_class_validation"] = {
            "status": "repaired_from_owl_subclass",
            "original_sub_class_uri": child,
            "original_parent_class_uri": parent,
            "selected_sub_class_uri": selected_child,
            "selected_parent_class_uri": selected_parent,
        }
    else:
        entry["parent_class_uri"] = None
        entry["sh_class_validation"] = {
            "status": "abstained_invalid_pair",
            "original_sub_class_uri": child,
            "original_parent_class_uri": parent,
        }
    return entry


def _build_type_group_context(
    table_name: str,
    type_col: str,
    value_profiles: dict[str, list[dict]],
    enriched_schema: dict | None,
    fk_context: dict | None,
    per_value_rows: int = 3,
    ref_rows_limit: int = 2,
) -> dict:
    """
    TYPE 值 -> 当前表样本行 -> FK 引用行 / incoming 关系行。
    这是给 LLM 的 instance-level group context，不依赖具体数据集名称。
    """
    pk_col = _first_pk(enriched_schema or {}, table_name)
    outgoing = (fk_context or {}).get("outgoing_fks", []) or []
    incoming = (fk_context or {}).get("incoming_fks", []) or []
    context = {
        "table": table_name,
        "type_column": type_col,
        "pk_column": pk_col,
        "groups": {},
    }

    try:
        conn = _get_conn()
    except Exception as e:
        context["warning"] = f"无法连接数据库补充 FK 上下文: {e}"
        return context

    try:
        with conn.cursor() as cur:
            for raw_value, rows in (value_profiles or {}).items():
                group_rows = []
                for row in (rows or [])[:per_value_rows]:
                    row_ctx = {"self": _trim_context_row(row)}

                    fk_refs = {}
                    for fk in outgoing:
                        col = fk.get("column")
                        ref_table = fk.get("ref_table")
                        ref_col = fk.get("ref_col") or _first_pk(enriched_schema or {}, ref_table)
                        if not col or not ref_table or not ref_col or row.get(col) is None:
                            continue
                        try:
                            cur.execute(
                                sql.SQL(
                                    "SELECT * FROM {table} WHERE {column} = %s LIMIT %s"
                                ).format(
                                    table=_qualified_table(ref_table, DB_SCHEMA_NAME),
                                    column=sql.Identifier(ref_col),
                                ),
                                (row.get(col), int(ref_rows_limit)),
                            )
                            cols = [d[0] for d in cur.description]
                            fk_refs[f"{col}->{ref_table}.{ref_col}"] = [
                                _trim_context_row(dict(zip(cols, r))) for r in cur.fetchall()
                            ]
                        except Exception:
                            conn.rollback()

                    incoming_refs = {}
                    if pk_col and row.get(pk_col) is not None:
                        for rel in incoming:
                            rel_table = rel.get("from_table")
                            rel_col = rel.get("from_column")
                            if not rel_table or not rel_col:
                                continue
                            try:
                                cur.execute(
                                    sql.SQL(
                                        "SELECT * FROM {table} WHERE {column} = %s LIMIT %s"
                                    ).format(
                                        table=_qualified_table(rel_table, DB_SCHEMA_NAME),
                                        column=sql.Identifier(rel_col),
                                    ),
                                    (row.get(pk_col), int(ref_rows_limit)),
                                )
                                cols = [d[0] for d in cur.description]
                                incoming_refs[f"{rel_table}.{rel_col}->{table_name}.{pk_col}"] = [
                                    _trim_context_row(dict(zip(cols, r))) for r in cur.fetchall()
                                ]
                            except Exception:
                                conn.rollback()

                    if fk_refs:
                        row_ctx["outgoing_fk_rows"] = fk_refs
                    if incoming_refs:
                        row_ctx["incoming_relation_rows"] = incoming_refs
                    group_rows.append(row_ctx)

                context["groups"][str(raw_value)] = group_rows
    finally:
        conn.close()

    return context


def _collect_descendants(root_uri: str, children_of: dict) -> list[str]:
    if not root_uri or not children_of:
        return []
    out = []
    queue = [root_uri]
    seen = {root_uri}
    while queue:
        cur = queue.pop(0)
        for child in children_of.get(cur, []):
            if child in seen:
                continue
            seen.add(child)
            out.append(child)
            queue.append(child)
    return out


def _local_name(uri: str) -> str:
    if not uri:
        return ""
    return uri.split("#")[-1].split("/")[-1]


def _norm_token(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _strip_dp_name_wrapper(local_name: str) -> str:
    """
    DatatypeProperty 常见命名：has_a_name / has_an_email / has_the_first_name。
    真实值增强只用它做保守锁定，避免实例值把同名列误改成别的属性。
    """
    norm = _norm_token(local_name)
    for prefix in ("hasa", "hasan", "hasthe", "has"):
        if norm.startswith(prefix) and len(norm) > len(prefix):
            return norm[len(prefix):]
    return norm


def _dp_range_xsd(ontology: dict | None, prop_uri: str | None) -> str | None:
    if not ontology or not prop_uri:
        return None
    ranges = ((ontology.get("datatype_properties", {}) or {}).get(prop_uri, {}) or {}).get("range", []) or []
    for r in ranges:
        local = _local_name(r).lower()
        if local:
            return local
    return None


def _sql_type_compatible_with_dp(sql_type: str, ontology: dict | None, prop_uri: str | None) -> bool:
    xsd = _dp_range_xsd(ontology, prop_uri)
    if not xsd:
        return True

    st = (sql_type or "").lower()
    is_num_sql = any(k in st for k in ("int", "numeric", "decimal", "real", "double", "float"))
    is_bool_sql = "bool" in st
    is_date_sql = "date" in st or "time" in st

    is_num_xsd = xsd in {
        "int", "integer", "decimal", "float", "double",
        "nonnegativeinteger", "unsignedlong", "unsignedint",
    }
    is_bool_xsd = xsd == "boolean"
    is_date_xsd = xsd in {"date", "datetime"}

    if is_num_xsd and not is_num_sql:
        return False
    if is_bool_xsd and not is_bool_sql:
        return False
    if is_date_xsd and not is_date_sql:
        return False
    return True


def _find_schema_locked_dp(col_name: str, col_cands: list, sql_type: str, ontology: dict | None) -> str | None:
    norm_col = _norm_token(col_name)
    if not norm_col:
        return None

    # 第一优先级：列名和属性 local name 完全一致。
    for c in col_cands:
        uri = c.get("uri")
        if not uri or not _sql_type_compatible_with_dp(sql_type, ontology, uri):
            continue
        if norm_col == _norm_token(c.get("local_name") or _local_name(uri)):
            return uri

    # 第二优先级：has_a/has_an/has_the 等本体属性命名包装后等于列名。
    for c in col_cands:
        uri = c.get("uri")
        if not uri or not _sql_type_compatible_with_dp(sql_type, ontology, uri):
            continue
        local = c.get("local_name") or _local_name(uri)
        if norm_col == _strip_dp_name_wrapper(local):
            return uri

    return None


def _leaf_descendants(descendant_uris: list[str] | None, children_of: dict) -> list[str]:
    leaves = []
    descendant_set = set(descendant_uris or [])
    for uri in descendant_uris or []:
        children = [c for c in children_of.get(uri, []) if c in descendant_set]
        if not children:
            leaves.append(uri)
    return leaves


def _first_pk(enriched_schema: dict, table_name: str) -> str | None:
    pks = (enriched_schema.get(table_name, {}) or {}).get("primary_key", []) or []
    return pks[0] if pks else None


def _iter_foreign_keys(table_info: dict) -> list[dict]:
    out = []
    for fk in (table_info or {}).get("foreign_keys", []) or []:
        col = fk.get("column")
        ref_table = fk.get("ref_table") or fk.get("references_table")
        ref_col = fk.get("ref_col") or fk.get("references_column")
        if col and ref_table:
            out.append({"column": col, "ref_table": ref_table, "ref_col": ref_col})
    return out


def _uri_local_name(uri: str | None) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rsplit("/", 1)[-1]


def _norm_entity_name(text: str) -> str:
    raw = (text or "").lower()
    return "".join(ch for ch in raw if ch.isalnum())


def _add_class_hint(class_hints: dict, uri: str | None, score: float, source: str) -> None:
    if not uri or not isinstance(uri, str) or not uri.startswith("http"):
        return
    s = float(score or 0.0)
    prev = class_hints.get(uri)
    if prev is None or s > prev.get("score", 0.0):
        class_hints[uri] = {
            "uri": uri,
            "local_name": _uri_local_name(uri),
            "score": round(s, 3),
            "source": source,
        }


def _relation_semantic_hints_from_candidates(
    rel_table: str, fk_col: str, candidates: dict
) -> dict:
    """
    从候选集中抽取 FK 语义提示：
      - relation_hints: 关系名提示（local_name）
      - class_hints: 与该 FK 侧相关的 class 提示（带分数）
    """
    entry = (candidates or {}).get(rel_table, {}) or {}
    pattern = entry.get("pattern")
    rel_hints = []
    rel_seen = set()
    class_hints = {}

    if pattern == "SR":
        fk1_col = ((entry.get("fk1") or {}) or {}).get("column")
        fk2_col = ((entry.get("fk2") or {}) or {}).get("column")
        side = "domain" if fk_col == fk1_col else "range" if fk_col == fk2_col else None

        for c in entry.get("sr_prop_candidates", [])[:5]:
            local = c.get("local_name") or _uri_local_name(c.get("uri"))
            if local and local not in rel_seen:
                rel_seen.add(local)
                rel_hints.append(local)

            if side:
                for cls_uri in c.get(side, []) or []:
                    _add_class_hint(
                        class_hints,
                        cls_uri,
                        c.get("score", 0.0),
                        source=f"sr_{side}:{local or ''}",
                    )
            else:
                for cls_uri in (c.get("domain", []) or []) + (c.get("range", []) or []):
                    _add_class_hint(
                        class_hints,
                        cls_uri,
                        c.get("score", 0.0) * 0.7,
                        source=f"sr_any:{local or ''}",
                    )
    else:
        col_entry = (entry.get("columns", {}) or {}).get(fk_col, {}) or {}
        role = col_entry.get("role")
        for c in col_entry.get("candidates", [])[:5]:
            local = c.get("local_name") or _uri_local_name(c.get("uri"))
            if local and local not in rel_seen:
                rel_seen.add(local)
                rel_hints.append(local)

            # 非 SR 表中，FK 通常对应 relation 的 range 端；保留 domain 作为弱提示兜底
            for cls_uri in c.get("range", []) or []:
                _add_class_hint(
                    class_hints,
                    cls_uri,
                    c.get("score", 0.0),
                    source=f"range:{local or ''}",
                )
            for cls_uri in c.get("domain", []) or []:
                _add_class_hint(
                    class_hints,
                    cls_uri,
                    c.get("score", 0.0) * 0.45,
                    source=f"domain:{local or ''}",
                )

        # SH 继承表的 inherited PK 指向父类表时，用子类候选作为强提示
        if pattern == "SH" and role in ("sh_inherited_pk", "pk"):
            for c in entry.get("sub_class_candidates", [])[:5]:
                uri = c.get("uri")
                if uri:
                    _add_class_hint(
                        class_hints,
                        uri,
                        c.get("score", 0.0) or 0.7,
                        source="sh_subclass",
                    )
                    local = c.get("local_name") or _uri_local_name(uri)
                    if local and local not in rel_seen:
                        rel_seen.add(local)
                        rel_hints.append(local)

    return {
        "relation_hints": rel_hints,
        "class_hints": sorted(
            class_hints.values(), key=lambda x: x.get("score", 0.0), reverse=True
        ),
    }


def _build_fk_semantic_context(
    table_name: str,
    enriched_schema: dict | None,
    candidates: dict | None,
    implicit_relations: dict | None = None,
    group_col: str | None = None,
    group_values: list[str] | None = None,
    max_incoming: int = REAL_VALUE_FK_CONTEXT_MAX_INCOMING,
) -> dict:
    """
    基于 schema/FK 图构建语义上下文，供真实值增强的 TYPE/BOOL 判断使用。
    包含：
      - 当前表 outgoing FK
      - 指向当前表的 incoming FK（含关系候选提示）
      - 若给定 group_col/group_values，则给出按取值分组的 incoming 覆盖率
    """
    if not enriched_schema or table_name not in enriched_schema:
        return {}

    def _is_discriminator_fk(src_table: str, src_col: str) -> bool:
        col_entry = (((candidates or {}).get(src_table, {}) or {}).get("columns", {}) or {}).get(src_col, {}) or {}
        return col_entry.get("role") == "discriminator"

    table_info = enriched_schema.get(table_name, {}) or {}
    pk_col = _first_pk(enriched_schema, table_name)

    outgoing = []
    for fk in _iter_foreign_keys(table_info):
        if _is_discriminator_fk(table_name, fk["column"]):
            continue
        outgoing.append({
            "column": fk["column"],
            "ref_table": fk["ref_table"],
            "ref_col": fk.get("ref_col"),
            "source": "schema_fk",
        })

    incoming = []
    incoming_seen = set()
    for rel_table, rel_info in (enriched_schema or {}).items():
        for fk in _iter_foreign_keys(rel_info):
            if fk.get("ref_table") != table_name:
                continue
            if _is_discriminator_fk(rel_table, fk.get("column")):
                continue
            rel_sem_hints = _relation_semantic_hints_from_candidates(
                rel_table, fk.get("column"), candidates or {}
            )
            k = (rel_table, fk.get("column"), fk.get("ref_col"))
            incoming_seen.add(k)
            incoming.append({
                "from_table": rel_table,
                "from_column": fk.get("column"),
                "to_table": table_name,
                "to_column": fk.get("ref_col"),
                "from_pattern": ((candidates or {}).get(rel_table, {}) or {}).get("pattern"),
                "relation_hints": rel_sem_hints.get("relation_hints", []),
                "class_hints": rel_sem_hints.get("class_hints", []),
                "source": "schema_fk",
            })

    for edge in (implicit_relations or {}).get("edges", []) or []:
        src_table = edge.get("source_table")
        src_col = edge.get("source_column")
        tgt_table = edge.get("target_table")
        tgt_col = edge.get("target_column")
        if not src_table or not src_col or not tgt_table:
            continue

        if src_table == table_name:
            if _is_discriminator_fk(src_table, src_col):
                continue
            outgoing.append({
                "column": src_col,
                "ref_table": tgt_table,
                "ref_col": tgt_col,
                "source": "implicit",
                "evidence_score": edge.get("evidence_score"),
            })

        if tgt_table != table_name:
            continue
        if _is_discriminator_fk(src_table, src_col):
            continue
        k = (src_table, src_col, tgt_col)
        if k in incoming_seen:
            continue
        incoming_seen.add(k)
        rel_sem_hints = _relation_semantic_hints_from_candidates(
            src_table, src_col, candidates or {}
        )
        incoming.append({
            "from_table": src_table,
            "from_column": src_col,
            "to_table": table_name,
            "to_column": tgt_col,
            "from_pattern": ((candidates or {}).get(src_table, {}) or {}).get("pattern"),
            "relation_hints": rel_sem_hints.get("relation_hints", []),
            "class_hints": rel_sem_hints.get("class_hints", []),
            "source": "implicit",
            "evidence_score": edge.get("evidence_score"),
        })

    incoming = incoming[:max_incoming]

    coverage_by_value = {}
    if pk_col and group_col and group_values and (incoming or outgoing):
        try:
            conn = _get_conn()
        except Exception as e:
            print(f"  [WARN] 无法连接数据库，跳过分组覆盖率统计 {table_name}.{group_col}: {e}")
            conn = None
        if not conn:
            return {
                "table": table_name,
                "pk_column": pk_col,
                "outgoing_fks": outgoing,
                "incoming_fks": incoming,
                "group_col": group_col,
                "coverage_by_value": coverage_by_value,
            }
        try:
            with conn.cursor() as cur:
                totals_by_value = {}
                for raw_val in group_values:
                    v = str(raw_val)
                    cur.execute(
                        sql.SQL(
                            "SELECT COUNT(DISTINCT {pk}) FROM {table} "
                            "WHERE CAST({group_col} AS TEXT) = %s"
                        ).format(
                            pk=sql.Identifier(pk_col),
                            table=_qualified_table(table_name, DB_SCHEMA_NAME),
                            group_col=sql.Identifier(group_col),
                        ),
                        (v,),
                    )
                    totals_by_value[v] = int(cur.fetchone()[0] or 0)

                # Measure outgoing joins as full-bucket evidence too.  This is
                # required for denormalized relations stored directly on the
                # entity row; sampled rows alone can overstate a mixed group.
                for rel in outgoing:
                    source_col = rel.get("column")
                    ref_table = rel.get("ref_table")
                    ref_col = rel.get("ref_col") or _first_pk(
                        enriched_schema or {}, ref_table
                    )
                    if not source_col or not ref_table or not ref_col:
                        continue
                    rel_key = f"outgoing:{source_col}->{ref_table}.{ref_col}"
                    coverage_by_value[rel_key] = {}
                    for raw_val in group_values:
                        v = str(raw_val)
                        total = totals_by_value.get(v, 0)
                        if total == 0:
                            coverage_by_value[rel_key][v] = {
                                "total": 0,
                                "linked": 0,
                                "ratio": 0.0,
                            }
                            continue
                        cur.execute(
                            sql.SQL(
                                "SELECT COUNT(DISTINCT source.{pk}) "
                                "FROM {source_table} AS source "
                                "WHERE CAST(source.{group_col} AS TEXT) = %s "
                                "AND source.{source_col} IS NOT NULL "
                                "AND EXISTS ("
                                "SELECT 1 FROM {ref_table} AS ref "
                                "WHERE ref.{ref_col} = source.{source_col}"
                                ")"
                            ).format(
                                pk=sql.Identifier(pk_col),
                                source_table=_qualified_table(table_name, DB_SCHEMA_NAME),
                                group_col=sql.Identifier(group_col),
                                source_col=sql.Identifier(source_col),
                                ref_table=_qualified_table(ref_table, DB_SCHEMA_NAME),
                                ref_col=sql.Identifier(ref_col),
                            ),
                            (v,),
                        )
                        linked = int(cur.fetchone()[0] or 0)
                        coverage_by_value[rel_key][v] = {
                            "total": total,
                            "linked": linked,
                            "ratio": round(linked / total, 4),
                        }

                for rel in incoming:
                    rel_table = rel["from_table"]
                    rel_col = rel["from_column"]
                    rel_key = f"{rel_table}.{rel_col}"
                    coverage_by_value[rel_key] = {}
                    for raw_val in group_values:
                        v = str(raw_val)
                        total = totals_by_value.get(v, 0)

                        if total == 0:
                            coverage_by_value[rel_key][v] = {"total": 0, "linked": 0, "ratio": 0.0}
                            continue

                        # linked entities in current group via this incoming FK
                        cur.execute(
                            sql.SQL(
                                "SELECT COUNT(DISTINCT source.{pk}) "
                                "FROM {source_table} AS source "
                                "WHERE CAST(source.{group_col} AS TEXT) = %s "
                                "AND EXISTS ("
                                "SELECT 1 FROM {relation_table} AS relation "
                                "WHERE relation.{relation_col} = source.{pk}"
                                ")"
                            ).format(
                                pk=sql.Identifier(pk_col),
                                source_table=_qualified_table(table_name, DB_SCHEMA_NAME),
                                group_col=sql.Identifier(group_col),
                                relation_table=_qualified_table(rel_table, DB_SCHEMA_NAME),
                                relation_col=sql.Identifier(rel_col),
                            ),
                            (v,),
                        )
                        linked = int(cur.fetchone()[0] or 0)
                        ratio = round(linked / total, 4) if total else 0.0
                        coverage_by_value[rel_key][v] = {
                            "total": total,
                            "linked": linked,
                            "ratio": ratio,
                        }
        except Exception as e:
            print(f"  [WARN] 构建 FK 语义上下文失败 {table_name}.{group_col}: {e}")
            conn.rollback()
        finally:
            conn.close()

    return {
        "table": table_name,
        "pk_column": pk_col,
        "outgoing_fks": outgoing,
        "incoming_fks": incoming,
        "group_col": group_col,
        "coverage_by_value": coverage_by_value,
    }


def _class_is_compatible_with_base(
    candidate_uri: str | None,
    current_class_uri: str | None,
    ontology: dict | None,
) -> bool:
    """Check a candidate against the table's known class and all its ancestors."""
    if not candidate_uri or not current_class_uri:
        return True
    anchors = [current_class_uri]
    anchors.extend((ontology or {}).get("ancestors_of", {}).get(current_class_uri, []) or [])
    return not any(
        are_classes_disjoint(anchor_uri, candidate_uri, ontology)
        for anchor_uri in anchors
    )


def _enum_value_class_lexical_score(raw_value: str, local_name: str) -> float:
    """Return conservative, auditable value-to-Class lexical evidence."""
    raw_text = str(raw_value or "").strip()
    value_norm = _norm_entity_name(raw_text)
    local_norm = _norm_entity_name(local_name)
    raw_compact = re.sub(r"[^A-Za-z0-9]+", "", raw_text)
    marked_two_letter_acronym = bool(
        len(raw_compact) == 2
        and raw_compact == raw_compact.upper()
        and raw_compact.isalpha()
        and re.search(r"[^A-Za-z0-9\s]", raw_text)
    )
    # Bare numbers, booleans and tiny fragments carry no reusable semantics.
    # Alphanumeric labels such as ``TYPE-2`` remain eligible.  A two-letter
    # acronym is considered only when an explicit separator marks it as one
    # (for example R&D); opaque two-character codes still abstain.
    if (
        (len(value_norm) < 3 and not marked_two_letter_acronym)
        or not any(ch.isalpha() for ch in value_norm)
        or value_norm in {"true", "false", "yes", "no"}
        or not local_norm
    ):
        return 0.0
    if value_norm == local_norm:
        return 1.0

    if (
        (3 <= len(raw_compact) <= 8 or marked_two_letter_acronym)
        and raw_compact == raw_compact.upper()
        and any(ch.isalpha() for ch in raw_compact)
    ):
        # Support both an uppercase semantic prefix (XML -> XMLArtifact) and
        # a genuine initialism (ERU -> EmergencyResponseUnit).  Uniqueness is
        # enforced by the caller, so a shared acronym never becomes a fact.
        split_local = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(local_name))
        split_local = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", split_local)
        local_parts = [
            token
            for token in re.split(r"[^A-Za-z0-9]+", split_local)
            if token
        ]
        initialism = "".join(token[0] for token in local_parts).lower()
        if local_norm.startswith(value_norm) or value_norm == initialism:
            return 0.94

    if value_norm in local_norm:
        coverage = len(value_norm) / len(local_norm)
        if len(value_norm) >= 5 and coverage >= 0.35:
            return min(0.98, 0.72 + 0.26 * coverage)

    value_tokens = _tokenize_semantic_name(raw_text)
    local_tokens = _tokenize_semantic_name(local_name)
    token_score = _token_jaccard(value_tokens, local_tokens)
    return token_score if token_score >= 0.75 else 0.0


def _expand_enum_class_candidates(
    current_class_uri: str,
    class_candidates: list[dict],
    ontology: dict | None,
    *,
    evidence_values: list[str] | None = None,
    return_diagnostics: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """
    Build evidence-backed Class candidates for an enum discriminator.

    The table-level Class is a useful anchor, but it is not a closed-world
    declaration.  A discriminator may refine rows to a sibling Class under a
    trusted, direct common ancestor.  Such a sibling is admitted only when it
    already has strong matcher evidence, explicit instance evidence, or a
    unique strong lexical match to an observed enum value.  Namespace and OWL
    disjointness remain hard constraints; an unverified sibling is never added
    merely because it exists in the ontology.
    """
    scored: list[dict] = []
    seen: set[str] = set()
    excluded: list[dict] = []
    admitted_siblings: list[dict] = []
    ambiguous_value_evidence: list[dict] = []
    current_ns = _ns_from_uri(current_class_uri)
    children_of = (ontology or {}).get("children_of", {})
    subclass_of = (ontology or {}).get("subclass_of", {})
    descendants = _collect_descendants(current_class_uri, children_of)
    known_subtree = set(descendants)
    if current_class_uri:
        known_subtree.add(current_class_uri)

    # Only a direct, named parent in the same ontology namespace is a trusted
    # sibling anchor.  Expanding through owl:Thing or a cross-namespace upper
    # ontology would make the candidate space effectively unconstrained.
    trusted_ancestors: list[str] = []
    sibling_common_ancestor: dict[str, str] = {}
    for ancestor_uri in (subclass_of.get(current_class_uri, []) or []):
        if (
            not ancestor_uri
            or ancestor_uri.endswith("#Thing")
            or ancestor_uri.endswith("/Thing")
            or (current_ns and _ns_from_uri(ancestor_uri) != current_ns)
        ):
            continue
        ancestor_descendants = set(_collect_descendants(ancestor_uri, children_of))
        if current_class_uri not in ancestor_descendants:
            continue
        trusted_ancestors.append(ancestor_uri)
        for uri in ancestor_descendants - known_subtree:
            sibling_common_ancestor.setdefault(uri, ancestor_uri)

    def matcher_evidence(candidate: dict) -> tuple[bool, float, str | None]:
        """Require an explicit strong matcher score, not the legacy default."""
        if "score" not in candidate:
            return False, 0.0, None
        score = float(candidate.get("score", 0.0) or 0.0)
        auxiliary_scores = {
            key: float(candidate.get(key, 0.0) or 0.0)
            for key in (
                "syntax_score",
                "token_score",
                "name_score",
                "lexical_score",
                "value_score",
                "real_value_score",
                "instance_score",
                "evidence_score",
            )
            if candidate.get(key) is not None
        }
        best_aux_key = max(auxiliary_scores, key=auxiliary_scores.get, default=None)
        best_aux = auxiliary_scores.get(best_aux_key, 0.0)
        strong = (
            score >= REAL_VALUE_TYPE_HIGH_SCORE
            or (
                score >= REAL_VALUE_TYPE_MEDIUM_SCORE
                and best_aux >= REAL_VALUE_TYPE_MEDIUM_SCORE
            )
        )
        return strong, max(score, best_aux), best_aux_key

    # Admit ontology-discovered siblings only when one observed value points
    # uniquely to that Class.  A generic value matching several siblings is
    # recorded as ambiguous and does not expand the candidate set.
    lexical_sibling_evidence: dict[str, dict] = {}
    sibling_uris = sorted(sibling_common_ancestor)
    for raw_value in evidence_values or []:
        ranked = sorted(
            (
                (_enum_value_class_lexical_score(str(raw_value), _uri_local_name(uri)), uri)
                for uri in sibling_uris
            ),
            reverse=True,
        )
        ranked = [(score, uri) for score, uri in ranked if score >= 0.75]
        if not ranked:
            continue
        top_score, top_uri = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if second_score >= top_score - REAL_VALUE_TYPE_MEDIUM_GAP:
            ambiguous_value_evidence.append(
                {
                    "value": str(raw_value),
                    "candidate_uris": [uri for score, uri in ranked if score >= top_score - REAL_VALUE_TYPE_MEDIUM_GAP],
                    "reason": "ambiguous_sibling_lexical_evidence",
                }
            )
            continue
        previous = lexical_sibling_evidence.get(top_uri)
        if previous is None or top_score > previous["score"]:
            lexical_sibling_evidence[top_uri] = {
                "value": str(raw_value),
                "score": round(top_score, 4),
            }

    def add_candidate(
        uri: str | None,
        local_name: str | None,
        score: float,
        source: str,
        *,
        evidence: dict | None = None,
        original: dict | None = None,
    ) -> None:
        if not uri or uri in seen:
            return
        if current_ns and _ns_from_uri(uri) != current_ns:
            excluded.append({"uri": uri, "reason": "namespace_mismatch", "source": source})
            return
        is_sibling = uri in sibling_common_ancestor
        if current_class_uri and uri not in known_subtree and not is_sibling:
            excluded.append({"uri": uri, "reason": "no_trusted_common_ancestor", "source": source})
            return
        if is_sibling and not evidence:
            excluded.append({"uri": uri, "reason": "insufficient_sibling_evidence", "source": source})
            return
        if not _class_is_compatible_with_base(uri, current_class_uri, ontology):
            excluded.append({"uri": uri, "reason": "ontology_disjoint_with_known_class", "source": source})
            return
        seen.add(uri)
        item = {
            "uri": uri,
            "local_name": local_name or _uri_local_name(uri),
            "score": score,
            # Keep provenance for the singleton ambiguity guard below.  A
            # candidate is still evidence for ranking, never an assertion.
            "source": source,
        }
        for key in ("syntax_score", "token_score", "name_score"):
            if original and original.get(key) is not None:
                item[key] = original[key]
        if evidence:
            item["admission_evidence"] = evidence
        scored.append(item)
        if is_sibling:
            admitted_siblings.append(
                {
                    "uri": uri,
                    "common_ancestor": sibling_common_ancestor[uri],
                    "source": source,
                    "evidence": evidence,
                }
            )

    for candidate in class_candidates or []:
        uri = candidate.get("uri")
        evidence = None
        if uri in sibling_common_ancestor:
            strong, evidence_score, evidence_key = matcher_evidence(candidate)
            if strong:
                evidence = {
                    "kind": "strong_matcher",
                    "score": round(evidence_score, 4),
                    "supporting_score": evidence_key,
                }
            elif uri in lexical_sibling_evidence:
                evidence = {
                    "kind": "enum_value_lexical_match",
                    **lexical_sibling_evidence[uri],
                }
        add_candidate(
            uri,
            candidate.get("local_name"),
            float(candidate.get("score", 0.6) or 0.6),
            "matcher_candidate",
            evidence=evidence,
            original=candidate,
        )

    # Descendants are a structurally valid specialization of the known base;
    # the downstream ambiguity guard still prevents unsupported assertions.
    for child_uri in descendants:
        add_candidate(child_uri, _uri_local_name(child_uri), 0.55, "known_descendant")

    # Real enum values may provide the missing evidence for a sibling that the
    # table-name matcher never proposed.
    for sibling_uri, evidence in lexical_sibling_evidence.items():
        add_candidate(
            sibling_uri,
            _uri_local_name(sibling_uri),
            max(0.55, float(evidence["score"])),
            "enum_value_lexical_evidence",
            evidence={"kind": "enum_value_lexical_match", **evidence},
        )

    if current_class_uri:
        add_candidate(current_class_uri, _uri_local_name(current_class_uri), 0.6, "known_base_class")

    admitted_uris = {item["uri"] for item in admitted_siblings}
    diagnostics = {
        "known_base_class": current_class_uri,
        # Backward-compatible prompt field: only evidence-admitted siblings
        # are added to the structurally known subtree.
        "allowed_subtree": sorted(known_subtree | admitted_uris),
        "trusted_sibling_ancestors": sorted(trusted_ancestors),
        "admitted_sibling_candidates": admitted_siblings,
        "ambiguous_value_evidence": ambiguous_value_evidence,
        "excluded_candidates": excluded,
    }
    if return_diagnostics:
        return scored, diagnostics
    return scored


def _discover_data_attr_enum_type_assertion(
    *,
    column_name: str,
    column_entry: dict,
    current_class_uri: str | None,
    current_class_confidence: str | None,
    class_candidates: list[dict],
    ontology: dict | None,
    value_profiles: dict[str, list[dict]],
    value_domain: dict | None,
) -> tuple[dict | None, dict]:
    """
    Discover an additional rdf:type assertion from a literal enum column.

    This is deliberately orthogonal to DP alignment: it never edits the
    column's role or ``prop_uri``.  A value is mapped only when its lexical
    form uniquely identifies a descendant, or a narrowly evidence-admitted
    direct sibling, of the already confirmed table Class.  Opaque codes and
    ambiguous labels remain literals only.
    """
    values = sorted(
        {
            str(value).strip()
            for value in (value_profiles or {})
            if str(value).strip()
        }
    )
    diagnostics = {
        "column": column_name,
        "reason": "",
        "distinct": len(values),
        "lexical_matches": {},
        "ambiguous_values": [],
        "opaque_values": [],
        "conflicting_values": [],
        "abstain_reasons": {},
    }

    if (column_entry or {}).get("role") != "data_attr":
        diagnostics["reason"] = "not_data_attribute"
        return None, diagnostics
    if not current_class_uri or current_class_confidence != "high":
        diagnostics["reason"] = "unconfirmed_table_class"
        return None, diagnostics
    if not values:
        diagnostics["reason"] = "no_observed_values"
        return None, diagnostics

    # Sentence-like text is not an enum even if it happens to mention a Class
    # label.  Keep the gate structural and independent of column/table names.
    word_counts = [
        len([part for part in re.split(r"\s+", value.strip()) if part])
        for value in values
    ]
    if any(len(value) > 64 for value in values) or any(count > 6 for count in word_counts):
        diagnostics["reason"] = "free_text_like_values"
        return None, diagnostics

    expanded_candidates, expansion_diagnostics = _expand_enum_class_candidates(
        current_class_uri=current_class_uri,
        class_candidates=class_candidates or [],
        ontology=ontology,
        evidence_values=values,
        return_diagnostics=True,
    )
    descendant_uris = set(
        _collect_descendants(
            current_class_uri,
            (ontology or {}).get("children_of", {}),
        )
    )
    subclass_of = (ontology or {}).get("subclass_of", {}) or {}
    children_of = (ontology or {}).get("children_of", {}) or {}
    current_ns = _ns_from_uri(current_class_uri)
    current_named_parents = {
        parent_uri
        for parent_uri in subclass_of.get(current_class_uri, []) or []
        if parent_uri
        and not parent_uri.endswith("#Thing")
        and not parent_uri.endswith("/Thing")
        and (not current_ns or _ns_from_uri(parent_uri) == current_ns)
    }

    # `_expand_enum_class_candidates` is the single source of sibling
    # admission.  Narrow it further here: profile discovery accepts only a
    # lexical-evidence sibling sharing exactly one direct named parent with the
    # confirmed table Class.  Multiple inheritance is not guessed through.
    admitted_by_uri = {
        item.get("uri"): item
        for item in expansion_diagnostics.get("admitted_sibling_candidates", [])
        if item.get("uri")
    }
    safe_sibling_info: dict[str, dict] = {}
    excluded_scope_candidates: list[dict] = []
    for sibling_uri, admission in admitted_by_uri.items():
        evidence = admission.get("evidence") or {}
        sibling_named_parents = {
            parent_uri
            for parent_uri in subclass_of.get(sibling_uri, []) or []
            if parent_uri
            and not parent_uri.endswith("#Thing")
            and not parent_uri.endswith("/Thing")
            and (not current_ns or _ns_from_uri(parent_uri) == current_ns)
        }
        shared_parents = sorted(current_named_parents & sibling_named_parents)
        direct_parent_is_structural = bool(
            len(shared_parents) == 1
            and current_class_uri in (children_of.get(shared_parents[0], []) or [])
            and sibling_uri in (children_of.get(shared_parents[0], []) or [])
        )
        if evidence.get("kind") != "enum_value_lexical_match":
            excluded_scope_candidates.append({
                "uri": sibling_uri,
                "reason": "sibling_without_value_lexical_evidence",
            })
            continue
        if len(shared_parents) != 1:
            excluded_scope_candidates.append({
                "uri": sibling_uri,
                "reason": "non_unique_direct_common_parent",
                "shared_parents": shared_parents,
            })
            continue
        if not direct_parent_is_structural:
            excluded_scope_candidates.append({
                "uri": sibling_uri,
                "reason": "common_parent_not_directly_asserted",
                "shared_parents": shared_parents,
            })
            continue
        if not _class_is_compatible_with_base(
            sibling_uri, current_class_uri, ontology
        ):
            excluded_scope_candidates.append({
                "uri": sibling_uri,
                "reason": "ontology_disjoint_with_confirmed_class",
            })
            continue
        safe_sibling_info[sibling_uri] = {
            "shared_direct_parent": shared_parents[0],
            "admission_evidence": evidence,
        }

    allowed_class_uris = descendant_uris | set(safe_sibling_info)
    ontology_family_candidates = [
        candidate
        for candidate in expanded_candidates
        if candidate.get("uri") in allowed_class_uris
        and _class_is_compatible_with_base(
            candidate.get("uri"), current_class_uri, ontology
        )
    ]
    diagnostics["candidate_expansion"] = expansion_diagnostics
    diagnostics["ontology_family_candidate_count"] = len(ontology_family_candidates)
    diagnostics["admitted_sibling_candidates"] = safe_sibling_info
    diagnostics["excluded_scope_candidates"] = excluded_scope_candidates
    for ambiguous in expansion_diagnostics.get("ambiguous_value_evidence", []) or []:
        ambiguous_value = str(ambiguous.get("value", "")).strip()
        if not ambiguous_value:
            continue
        if ambiguous_value not in diagnostics["ambiguous_values"]:
            diagnostics["ambiguous_values"].append(ambiguous_value)
        diagnostics["abstain_reasons"][ambiguous_value] = (
            "ambiguous_lexical_candidates"
        )

    conflicting_values: set[str] = set()
    non_string_values: set[str] = set()
    for profile_value, rows in (value_profiles or {}).items():
        profile_text = str(profile_value).strip()
        row_text_values: set[str] = set()
        for row in rows or []:
            if not isinstance(row, dict) or column_name not in row:
                continue
            row_value = row.get(column_name)
            if row_value is None:
                continue
            if not isinstance(row_value, str):
                non_string_values.add(profile_text)
            row_text_values.add(str(row_value).strip())
        if row_text_values and row_text_values != {profile_text}:
            conflicting_values.add(profile_text)
    diagnostics["conflicting_values"] = sorted(conflicting_values)

    min_score = 0.82
    min_gap = max(0.14, REAL_VALUE_TYPE_HIGH_GAP)
    value_to_class: dict[str, str] = {}
    for value in values:
        if value in diagnostics["ambiguous_values"]:
            continue
        if value in conflicting_values:
            diagnostics["abstain_reasons"][value] = "conflicting_profile_samples"
            continue
        if value in non_string_values:
            diagnostics["opaque_values"].append(value)
            diagnostics["abstain_reasons"][value] = "non_string_value"
            continue
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", value):
            diagnostics["opaque_values"].append(value)
            diagnostics["abstain_reasons"][value] = "numeric_string_code"
            continue
        ranked = sorted(
            [
                {
                    "uri": candidate.get("uri"),
                    "score": _enum_value_class_lexical_score(
                        value,
                        candidate.get("local_name")
                        or _uri_local_name(candidate.get("uri")),
                    ),
                }
                for candidate in ontology_family_candidates
                if candidate.get("uri")
            ],
            key=lambda item: (-item["score"], item["uri"]),
        )
        top = ranked[0] if ranked else {"uri": None, "score": 0.0}
        second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        gap = float(top["score"]) - float(second_score)
        if float(top["score"]) >= min_score and gap >= min_gap:
            value_to_class[value] = top["uri"]
            diagnostics["lexical_matches"][value] = {
                "class_uri": top["uri"],
                "score": round(float(top["score"]), 4),
                "margin": round(gap, 4),
                "candidate_scope": (
                    "descendant"
                    if top["uri"] in descendant_uris
                    else "evidence_admitted_sibling"
                ),
                "shared_direct_parent": (
                    safe_sibling_info.get(top["uri"], {})
                    .get("shared_direct_parent")
                ),
            }
        elif float(top["score"]) >= min_score:
            diagnostics["ambiguous_values"].append(value)
            diagnostics["abstain_reasons"][value] = "ambiguous_lexical_candidates"
        else:
            diagnostics["abstain_reasons"][value] = "no_strong_lexical_match"

    # Different raw labels that collapse to one normalized code must not imply
    # conflicting Classes.  This is rare in clean SQL enums, but explicit here
    # so punctuation/case normalization cannot conceal contradictory samples.
    classes_by_normalized_value: dict[str, set[str]] = {}
    raw_values_by_normalized_value: dict[str, set[str]] = {}
    for raw_value, class_uri in value_to_class.items():
        normalized = _norm_entity_name(raw_value)
        classes_by_normalized_value.setdefault(normalized, set()).add(class_uri)
        raw_values_by_normalized_value.setdefault(normalized, set()).add(raw_value)
    for normalized, class_uris in classes_by_normalized_value.items():
        if len(class_uris) <= 1:
            continue
        for raw_value in raw_values_by_normalized_value.get(normalized, set()):
            value_to_class.pop(raw_value, None)
            diagnostics["lexical_matches"].pop(raw_value, None)
            diagnostics["conflicting_values"].append(raw_value)
            diagnostics["abstain_reasons"][raw_value] = (
                "normalized_value_maps_to_conflicting_classes"
            )

    # The incomplete-domain gate is intentionally after lexical diagnostics;
    # callers can cheaply pre-screen a small in-memory sample before issuing a
    # bounded DISTINCT profile query.
    if not bool((value_domain or {}).get("complete")):
        diagnostics["reason"] = "incomplete_value_domain"
        return None, diagnostics
    if len(values) < 2 or len(values) > REAL_VALUE_ENUM_DISTINCT_MAX_FOR_CODE:
        diagnostics["reason"] = "not_low_cardinality"
        return None, diagnostics
    if not ontology_family_candidates:
        diagnostics["reason"] = "no_safe_ontology_family_class_space"
        return None, diagnostics
    if not value_to_class:
        diagnostics["reason"] = "no_unique_strong_lexical_match"
        return None, diagnostics

    unmapped_values = [value for value in values if value not in value_to_class]
    assertion = {
        "column": column_name,
        "kind": "enum",
        "value_to_class": value_to_class,
        "unmapped_values": unmapped_values,
        "class_candidates": ontology_family_candidates,
        "confidence": "high" if not unmapped_values else "medium",
        "reason": (
            "低基数字符串数据属性的观测值与已确认表类的后代，"
            "或唯一可信直接父类下由词法证据准入的同级类，形成唯一强匹配；"
            "保留原 DatatypeProperty，并追加独立 rdf:type 断言。"
        ),
        "evidence_source": "data_profile_unique_ontology_family_lexical_match",
        "evidence": {
            "source": "data_profile_unique_ontology_family_lexical_match",
            "matches": copy.deepcopy(diagnostics["lexical_matches"]),
            "value_domain_complete": True,
        },
    }
    diagnostics["reason"] = "unique_ontology_family_lexical_matches"
    diagnostics["unmapped_values"] = unmapped_values
    return assertion, diagnostics


# 真实值增强函数
def _real_value_table_class(
    table_name: str,
    class_cands: list,
    sample_rows: list,
    *,
    force_llm: bool = False,
) -> dict:
    """表级 Class 重判：用真实数据判断这张表对应哪个 Class。"""
    table_norm = _norm_entity_name(table_name)
    if table_norm.endswith("s") and len(table_norm) > 3:
        table_norm_alt = table_norm[:-1]
    else:
        table_norm_alt = table_norm

    if not force_llm:
        for c in class_cands or []:
            uri = c.get("uri")
            local = c.get("local_name") or _uri_local_name(uri)
            ln = _norm_entity_name(local)
            if not ln:
                continue
            if ln == table_norm or ln == table_norm_alt:
                return {
                    "selected_uri": uri,
                    "confidence": "high",
                    "reason": "表名与候选类名在规范化后精确匹配，优先锁定该 Class。",
                }

    prompt = f"""
## 任务
为表 `{table_name}` 找到本体中最对应的 OWL Class。
之前仅凭表名置信度不足，现在结合真实数据重新判断。

## 真实数据样本（{len(sample_rows)} 行）
{json.dumps(sample_rows, indent=2, ensure_ascii=False, default=str)}

## 候选 Class（按相似度预排序）
{json.dumps(class_cands, indent=2, ensure_ascii=False)}

## 判断要点
- 观察数据内容，判断它更像哪个 Class 的实例
- 数据为空时，仅凭候选名称语义判断

## 输出格式（严格 JSON）
{{
  "selected_uri": "选中的 URI（必须来自候选列表，或 null）",
  "confidence": "high / medium / low",
  "reason": "一句话理由，说明数据如何支持这个判断"
}}
"""
    return _call_llm(prompt)


def _real_value_data_attr(
    table_name: str,
    col_name: str,
    class_uri: str,
    col_cands: list,
    col_values: list,
    *,
    row_context: list[dict] | None = None,
    fk_context: dict | None = None,
    identity_part: bool = False,
) -> dict:
    """数据属性列重判：用真实值判断对应哪个 DatatypeProperty。"""
    prompt = f"""
## 任务
表 `{table_name}`（Class: {class_uri}）中，列 `{col_name}` 是数据属性列。
之前置信度不足，现在结合真实数据值重新判断对应哪个 DatatypeProperty。

## 该列的真实数据值样本
{json.dumps(col_values, ensure_ascii=False, default=str)}

## 同行 schema context
{json.dumps(row_context or [], ensure_ascii=False, indent=2, default=str)}

## FK / 关联表 context
{json.dumps(fk_context or {}, ensure_ascii=False, indent=2, default=str)}

## 结构角色
- identity_part: {bool(identity_part)}
- identity_part 只表示该列参与 subject identity；它既不自动证明，也不排除
  独立的 DatatypeProperty 语义。

## 候选 DatatypeProperty（按相似度+domain排序）
{json.dumps(col_cands, indent=2, ensure_ascii=False)}

## 判断要点
- 观察数据值的格式和语义（日期？姓名？描述文字？布尔？数字？）
- 结合列名和数据值共同判断
- 结合同行字段和真实 FK 图判断，不能只凭值格式猜测
- 若候选中无合适项，选 null

## 输出格式（严格 JSON）
{{
  "selected_uri": "选中的 URI（必须来自候选列表，或 null）",
  "confidence": "high / medium / low",
  "reason": "一句话理由"
}}
"""
    return _call_llm(prompt)

def _real_value_fk_range_class(table_name: str, col_name: str,
                        ref_table: str, ref_class_cands: list,
                        col_values: list) -> dict:
    """
    FK列 range Class 重判：用真实 FK 值辅助确认引用表对应哪个 Class。
    """
    prompt = f"""
## 任务
表 `{table_name}` 中，FK列 `{col_name}` 引用表 `{ref_table}`。
之前 range Class 置信度不足，现在结合 FK 值重新确认引用表对应哪个 OWL Class。

## 该列的真实 FK 值样本（ID值）
{json.dumps(col_values, ensure_ascii=False, default=str)}

## 引用表的 Class 候选
{json.dumps(ref_class_cands, indent=2, ensure_ascii=False)}

## 判断要点
- FK 值本身是 ID，语义有限，重点看引用表名与候选 Class 名的语义匹配
- 选最符合引用表语义的 Class URI

## 输出格式（严格 JSON）
{{
  "selected_uri": "选中的 Class URI（必须来自候选列表，或 null）",
  "confidence": "high / medium / low",
  "reason": "一句话理由"
}}
"""
    return _call_llm(prompt)


def _real_value_sr_classes(table_name: str, fk1: dict, fk2: dict,
                    sample_rows: list, alignment_entry: dict) -> dict:
    """
    SR 表 domain/range Class 重判。
    只确认两端的 Class URI，不判断 ObjectProperty。
    """
    current_domain = alignment_entry.get("domain_class_uri")
    current_range  = alignment_entry.get("range_class_uri")

    prompt = f"""
## 任务
关联表 `{table_name}` 是纯关联表（SR Pattern），连接两个实体。
之前 domain/range Class 确认置信度不足，现在结合真实数据重新确认两端的 Class。

## 真实数据样本（{len(sample_rows)} 行）
{json.dumps(sample_rows, indent=2, ensure_ascii=False, default=str)}

## FK 关联信息
- FK1: 列 `{fk1.get('column')}` 引用表 `{fk1.get('ref_table')}` → 当前 domain 推断: {current_domain}
- FK2: 列 `{fk2.get('column')}` 引用表 `{fk2.get('ref_table')}` → 当前 range 推断: {current_range}

## 判断要点
- 通过数据值确认 FK1 对应哪个 Class（domain），FK2 对应哪个 Class（range）
- 如果当前推断合理，可以保持不变
- 只输出 Class URI，不要判断 ObjectProperty

## 输出格式（严格 JSON）
{{
  "domain_class_uri": "domain 端 Class URI（保持或修正）",
  "range_class_uri": "range 端 Class URI（保持或修正）",
  "confidence": "high / medium / low",
  "reason": "一句话理由"
}}
"""
    return _call_llm(prompt)


def _tokenize_semantic_name(text: str | None) -> set[str]:
    if not text:
        return set()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    tokens = set()
    for tok in re.split(r"[^a-z0-9]+", text.lower()):
        if not tok:
            continue
        tokens.add(tok)
        if len(tok) > 3 and tok.endswith("s"):
            tokens.add(tok[:-1])
    return tokens


def _token_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom else 0.0


def _ancestor_distance(
    child_uri: str,
    ancestor_uri: str,
    subclass_of: dict | None,
    max_depth: int = REAL_VALUE_ANCESTOR_MAX_DEPTH,
) -> int | None:
    if not child_uri or not ancestor_uri or not subclass_of:
        return None
    if child_uri == ancestor_uri:
        return 0
    queue = [(child_uri, 0)]
    seen = {child_uri}
    while queue:
        cur, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for p in (subclass_of.get(cur, []) or []):
            if p == ancestor_uri:
                return depth + 1
            if p in seen:
                continue
            seen.add(p)
            queue.append((p, depth + 1))
    return None


def _is_true_like(v) -> bool | None:
    s = str(v).strip().lower()
    if s in {"1", "t", "true", "y", "yes"}:
        return True
    if s in {"0", "f", "false", "n", "no"}:
        return False
    return None


def _is_likely_enum_discriminator(values: list[str], value_profiles: dict[str, list[dict]]) -> bool:
    distinct = len(values)
    if distinct < 2 or distinct > 30:
        return False

    sample_rows = sum(len(rows or []) for rows in (value_profiles or {}).values())
    sample_distinct_ratio = _safe_ratio(distinct, max(sample_rows, distinct))

    short_codes = sum(1 for v in values if len(str(v)) <= 16 and " " not in str(v))
    numeric_like = sum(1 for v in values if re.fullmatch(r"-?\d+", str(v)))
    short_ratio = _safe_ratio(short_codes, distinct)
    numeric_ratio = _safe_ratio(numeric_like, distinct)

    if short_ratio < 0.8:
        return False
    if numeric_ratio >= REAL_VALUE_ENUM_NUMERIC_RATIO_THRESHOLD:
        return True
    return sample_distinct_ratio <= REAL_VALUE_ENUM_SAMPLE_DISTINCT_RATIO_THRESHOLD and distinct <= REAL_VALUE_ENUM_DISTINCT_MAX_FOR_CODE


def _enum_structure_signal(
    values: list[str],
    fk_context: dict | None,
) -> float:
    """
    估计“枚举值是否携带结构语义”的强度。
    基于 incoming FK 的按值覆盖率差异（coverage_by_value）计算，返回 0~1。
    """
    if not values or not fk_context:
        return 0.0

    cov = (fk_context or {}).get("coverage_by_value", {}) or {}
    if not cov:
        return 0.0

    best = 0.0
    value_set = {str(v) for v in values}
    for rel_key, by_val in cov.items():
        if not isinstance(by_val, dict):
            continue
        ratios = []
        totals = 0
        for v, stat in by_val.items():
            if str(v) not in value_set:
                continue
            if not isinstance(stat, dict):
                continue
            total = int(stat.get("total", 0) or 0)
            ratio = float(stat.get("ratio", 0.0) or 0.0)
            if total <= 0:
                continue
            totals += total
            ratios.append(ratio)

        if len(ratios) < 2:
            continue

        gap = max(ratios) - min(ratios)
        active = sum(1 for r in ratios if r >= 0.2)
        active_ratio = _safe_ratio(active, len(ratios))
        sample_factor = min(1.0, totals / 20.0)
        signal = min(1.0, (0.7 * gap + 0.3 * active_ratio) * sample_factor)
        best = max(best, signal)

    return round(best, 4)


def _should_rule_first_for_enum(
    values: list[str],
    value_profiles: dict[str, list[dict]],
    fk_context: dict | None,
    descendant_uris: list[str] | None = None,
    sh_value_evidence: dict | None = None,
) -> tuple[bool, dict]:
    """
    规则轨道触发门：
      1) 先满足“低基数枚举”基础条件
      2) 再满足结构证据强，或“码值风格 + 存在子类空间”
    """
    is_enum = _is_likely_enum_discriminator(values, value_profiles)
    distinct = len(values or [])
    desc_cnt = len(descendant_uris or [])
    sample_rows = sum(len(rows or []) for rows in (value_profiles or {}).values())
    sample_distinct_ratio = _safe_ratio(distinct, max(sample_rows, distinct))
    repeated_ratio = max(0.0, 1.0 - sample_distinct_ratio)

    short_codes = sum(1 for v in values if len(str(v)) <= 16 and " " not in str(v))
    numeric_like = sum(1 for v in values if re.fullmatch(r"-?\d+", str(v)))
    short_ratio = _safe_ratio(short_codes, max(distinct, 1))
    numeric_ratio = _safe_ratio(numeric_like, max(distinct, 1))
    code_like = short_ratio >= 0.8 and numeric_ratio >= 0.4

    struct_signal = _enum_structure_signal(values, fk_context)
    struct_strong = struct_signal >= REAL_VALUE_RULE_STRUCT_SIGNAL_THRESHOLD
    fallback_signal = (
        code_like
        and repeated_ratio >= REAL_VALUE_RULE_FALLBACK_REPEATED_RATIO
        and desc_cnt >= 1
        and distinct <= REAL_VALUE_RULE_FALLBACK_DISTINCT_MAX
    )
    # A singleton numeric code is normally not semantically interpretable.
    # It may still be resolved without an LLM when a SH child provides strong,
    # instance-level coverage evidence (for example, every matching parent
    # row occurs in one child table).  This is deliberately based on measured
    # data coverage rather than a dataset/table name.
    single_value_sh_support = []
    if distinct == 1:
        only_value = str(values[0])
        for evidence in (sh_value_evidence or {}).get(only_value, []) or []:
            if evidence.get("evidence_source") != "validated_sh_identity":
                continue
            ratio = float(evidence.get("ratio", 0.0) or 0.0)
            total = int(evidence.get("total", 0) or 0)
            if ratio >= REAL_VALUE_SINGLE_VALUE_SH_MIN_RATIO:
                single_value_sh_support.append(
                    {
                        "class_uri": evidence.get("class_uri"),
                        "ratio": ratio,
                        "total": total,
                    }
                )
    single_value_strong = bool(single_value_sh_support)
    apply_rule = bool(
        (is_enum and (struct_strong or fallback_signal))
        or single_value_strong
    )

    diagnostics = {
        "is_enum_like": is_enum,
        "distinct": distinct,
        "sample_rows": sample_rows,
        "repeated_ratio": round(repeated_ratio, 4),
        "short_ratio": round(short_ratio, 4),
        "numeric_ratio": round(numeric_ratio, 4),
        "descendant_count": desc_cnt,
        "structure_signal": struct_signal,
        "single_value_sh_support": single_value_sh_support,
        "single_value_strong": single_value_strong,
        "rule_first": apply_rule,
    }
    return apply_rule, diagnostics


def _score_enum_value_by_rules(
    value: str,
    rows: list[dict],
    class_candidates: list[dict],
    fk_context: dict | None,
    type_col: str | None = None,
    bool_hint_classes: list[str] | None = None,
    bool_assertions: list[dict] | None = None,
    class_ancestors: dict | None = None,
    class_subclass_of: dict | None = None,
    sh_value_evidence: dict | None = None,
    current_class_uri: str | None = None,
    descendant_uris: list[str] | None = None,
    all_values: list[str] | None = None,
) -> list[dict]:
    bool_hint_set = set(bool_hint_classes or [])
    descendant_set = set(descendant_uris or [])
    coverage = (fk_context or {}).get("coverage_by_value", {}) or {}
    incoming = (fk_context or {}).get("incoming_fks", []) or []

    profiles = []
    for c in class_candidates or []:
        uri = c.get("uri")
        if not uri:
            continue
        local = c.get("local_name") or _uri_local_name(uri)
        base = float(c.get("score", 0.0) or 0.0)
        prior = 0.25 + 0.75 * min(base, 1.0)
        if uri in descendant_set:
            prior *= 1.06
        if current_class_uri and descendant_set and uri == current_class_uri:
            prior *= 0.35
        if uri in bool_hint_set:
            prior *= 0.9
        profiles.append({
            "uri": uri,
            "local_name": local,
            "tokens": _tokenize_semantic_name(local),
            "base": base,
            "score": prior * 0.28,
        })

    # -1) 枚举值本身的语义证据：
    #     DEVELOPMENT -> DevelopmentWellbore, FORMATION -> Formation,
    #     SEMISUB STEEL -> SemisubSteelFacility。
    value_tokens = _tokenize_semantic_name(value)
    value_norm = _norm_entity_name(value)
    value_truth = _is_true_like(value)
    col_tokens = _tokenize_semantic_name(type_col)
    col_norm = _norm_entity_name(type_col or "")
    for p in profiles:
        local_norm = _norm_entity_name(p.get("local_name"))
        sim = _token_jaccard(value_tokens, p["tokens"])
        if sim > 0:
            p["score"] += 0.42 + 0.58 * sim
        if value_norm and local_norm and (value_norm == local_norm or value_norm in local_norm):
            p["score"] += 0.72

        # YES/NO 这种布尔型枚举：true-like 值用列名语义补充。
        if value_truth is True and col_tokens:
            col_sim = _token_jaccard(col_tokens, p["tokens"])
            if col_sim > 0:
                p["score"] += 0.32 + 0.46 * col_sim
            if col_norm and local_norm and (col_norm in local_norm or local_norm in col_norm):
                p["score"] += 0.68

    # 0) TYPE 专用证据 A：布尔判别列（column -> true_class）对父类的回溯支持
    #    目的：把 Author_of_xxx 这类细粒度信号，稳定回传到 Author，而不是被 Speaker 之类关系名盖过去。
    bool_assertions = bool_assertions or []
    class_ancestors = class_ancestors or {}
    class_subclass_of = class_subclass_of or {}
    for ba in bool_assertions:
        bcol = ba.get("column")
        true_class = ba.get("true_class_uri")
        if not bcol or not true_class:
            continue
        local_true = _uri_local_name(true_class)

        total = 0
        tcnt = 0
        for row in rows or []:
            if not isinstance(row, dict) or bcol not in row:
                continue
            total += 1
            tv = _is_true_like(row.get(bcol))
            if tv is True:
                tcnt += 1
        if total == 0:
            continue
        true_ratio = _safe_ratio(tcnt, total)
        if true_ratio <= 0.0:
            continue

        for p in profiles:
            p_uri = p.get("uri")
            if not p_uri:
                continue
            # exact 子类命中（弱）
            if p_uri == true_class:
                p["score"] += 0.22 * true_ratio
                continue

            # 祖先命中（按层级距离衰减）
            if p_uri in (class_ancestors.get(true_class, []) or []):
                dist = _ancestor_distance(true_class, p_uri, class_subclass_of)
                if dist is None:
                    continue
                p["score"] += (0.34 / max(1, dist)) * true_ratio

    # 0.5) TYPE 专用证据 B：SH 子类成员证据（按值分组）
    #      若某个值几乎都出现在某个 SH 子表中，则该值应强支持该子类。
    sh_bucket = (sh_value_evidence or {}).get(str(value), []) or []
    for ev in sh_bucket:
        cls_uri = ev.get("class_uri")
        ratio = float(ev.get("ratio", 0.0) or 0.0)
        if not cls_uri:
            continue
        for p in profiles:
            p_uri = p.get("uri")
            if not p_uri:
                continue
            if p_uri == cls_uri:
                if ratio > 0:
                    p["score"] += 1.05 * ratio
                else:
                    # 当前 value 在该 SH 子类中完全不出现，给负证据
                    p["score"] -= 0.32
                continue
            # 对同一值下“明显非该 SH 子类”的候选做轻惩罚，防止 Review 泄漏到别的 type 值
            if ratio >= 0.7 and p_uri != cls_uri:
                p["score"] -= 0.18 * ratio

    # 1) 结构证据：按 value 分组覆盖率 × relation/class 语义匹配
    for rel in incoming:
        rel_table = rel.get("from_table")
        rel_col = rel.get("from_column")
        if not rel_table or not rel_col:
            continue

        rel_key = f"{rel_table}.{rel_col}"
        rel_cov = (coverage.get(rel_key) or {}).get(str(value), {}) or {}
        ratio = float(rel_cov.get("ratio", 0.0) or 0.0)
        peer_ratios = []
        for peer_value in (all_values or [str(value)]):
            if str(peer_value) == str(value):
                continue
            peer_stat = (coverage.get(rel_key) or {}).get(str(peer_value), {}) or {}
            peer_ratios.append(float(peer_stat.get("ratio", 0.0) or 0.0))
        contrasting_peer_ratio = max(peer_ratios, default=0.0)

        src = rel.get("source")
        edge_score = 1.0
        if src == "implicit":
            edge_score = max(0.55, min(1.0, float(rel.get("evidence_score", 0.7) or 0.7)))

        rel_hints = rel.get("relation_hints", []) or []
        class_hints = rel.get("class_hints", []) or []

        for p in profiles:
            best_match = 0.0

            for h in class_hints:
                h_uri = h.get("uri")
                h_local = h.get("local_name")
                h_score = float(h.get("score", 0.0) or 0.0)

                if h_uri and h_uri == p["uri"]:
                    best_match = max(best_match, 0.72 + 0.28 * min(h_score, 1.0))

                sim = _token_jaccard(p["tokens"], _tokenize_semantic_name(h_local))
                if sim > 0:
                    best_match = max(best_match, 0.2 + 0.55 * sim + 0.25 * min(h_score, 1.0))

            for hint in rel_hints:
                sim = _token_jaccard(p["tokens"], _tokenize_semantic_name(hint))
                if sim > 0:
                    best_match = max(best_match, 0.12 + 0.58 * sim)

            if ratio > 0 and best_match > 0:
                p["score"] += ratio * edge_score * best_match

            # Contrastive negative evidence matters for coded discriminators:
            # if the same relation covers another TYPE value strongly but does
            # not cover this value, the candidate it semantically names should
            # not win merely because it has a broad relation hint.
            if (
                ratio <= 0.05
                and contrasting_peer_ratio >= 0.70
                and best_match >= 0.62
            ):
                p["score"] -= (
                    0.42 * edge_score * best_match * contrasting_peer_ratio
                )

    # 2) 值样本中的布尔列协同信号（不依赖列名前缀）
    bool_cols = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for col, v in row.items():
            t = _is_true_like(v)
            if t is None:
                continue
            stat = bool_cols.setdefault(col, {"true": 0, "total": 0})
            stat["total"] += 1
            if t:
                stat["true"] += 1

    for col, stat in bool_cols.items():
        true_ratio = _safe_ratio(stat["true"], stat["total"])
        if true_ratio < 0.5:
            continue
        col_tokens = _tokenize_semantic_name(col)
        col_norm = _norm_entity_name(col)
        for p in profiles:
            sim = _token_jaccard(p["tokens"], col_tokens)
            if sim > 0:
                p["score"] += (0.14 + 0.24 * sim) * true_ratio
            if col_norm and col_norm == _norm_entity_name(p["local_name"]):
                p["score"] += 0.45 * true_ratio

    profiles.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return profiles


def _unique_lexical_enum_class(
    value: str,
    class_candidates: list[dict],
    current_class_uri: str | None,
    ontology: dict | None,
) -> tuple[str | None, dict]:
    """Lock one unambiguous ontology-admitted lexical Class match.

    Exact local-name and unique uppercase-prefix evidence (for example
    ``FSU`` → ``FSUFacility``) comes from the observed value itself.  It is
    safe to use before an LLM review only when the candidate is compatible
    with the confirmed table Class and the runner-up is clearly separated.
    Ambiguous values remain unresolved so this rule cannot manufacture a
    class from a weak lexical coincidence.
    """
    ranked = []
    for candidate in class_candidates or []:
        uri = candidate.get("uri")
        if not uri or not _class_is_compatible_with_base(
            uri, current_class_uri, ontology
        ):
            continue
        local = candidate.get("local_name") or _uri_local_name(uri)
        score = _enum_value_class_lexical_score(str(value), str(local))
        if score > 0.0:
            ranked.append((score, str(uri), str(local)))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    if not ranked:
        return None, {"value": str(value), "reason": "no_lexical_candidate"}
    top = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if top[0] < 0.90 or second_score >= top[0] - REAL_VALUE_TYPE_HIGH_GAP:
        return None, {
            "value": str(value),
            "reason": "ambiguous_or_weak_lexical_candidates",
            "candidates": [
                {"uri": uri, "local_name": local, "score": round(score, 4)}
                for score, uri, local in ranked[:5]
            ],
        }
    return top[1], {
        "value": str(value),
        "reason": "unique_strong_lexical_class_evidence",
        "local_name": top[2],
        "score": round(top[0], 4),
        "runner_up_score": round(second_score, 4),
    }


def _is_numeric_enum_value(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", str(value)))


def _strong_single_value_sh_class(
    value: str,
    sh_value_evidence: dict | None,
    allowed_classes: set[str] | None = None,
) -> tuple[str | None, dict]:
    """Return a uniquely and sufficiently supported SH class, if one exists."""
    supports = []
    for evidence in ((sh_value_evidence or {}).get(str(value), []) or []):
        if evidence.get("evidence_source") != "validated_sh_identity":
            continue
        uri = evidence.get("class_uri")
        if not uri or (allowed_classes and uri not in allowed_classes):
            continue
        supports.append(
            {
                "class_uri": uri,
                "ratio": float(evidence.get("ratio", 0.0) or 0.0),
                "total": int(evidence.get("total", 0) or 0),
            }
        )
    supports.sort(key=lambda item: item["ratio"], reverse=True)
    if not supports:
        return None, {"supports": []}
    top = supports[0]
    second_ratio = supports[1]["ratio"] if len(supports) > 1 else 0.0
    strong = (
        top["ratio"] >= REAL_VALUE_SINGLE_VALUE_SH_MIN_RATIO
        and top["total"] >= REAL_VALUE_SINGLE_VALUE_SH_MIN_ENTITIES
        and top["ratio"] - second_ratio >= REAL_VALUE_SINGLE_VALUE_SH_MIN_GAP
    )
    return (
        top["class_uri"] if strong else None,
        {
            "supports": supports,
            "top_ratio": top["ratio"],
            "second_ratio": second_ratio,
            "strong": strong,
        },
    )


def _has_strong_sh_support(value: str, class_uri: str | None, sh_value_evidence: dict | None) -> bool:
    selected, _ = _strong_single_value_sh_class(
        value,
        sh_value_evidence,
        allowed_classes={class_uri} if class_uri else set(),
    )
    return bool(selected == class_uri)


def _has_materialized_joined_context(
    value: str,
    group_context: dict | None,
) -> bool:
    """Whether a value bucket contains at least one real joined relation row.

    A singleton numeric code has no lexical meaning and no within-column peer
    for contrast.  Paper Algorithm 1 can still send it to ContextEnhanced when
    the sampled entity rows actually join to another table.  Merely knowing
    that an FK exists is insufficient: this gate requires materialized rows in
    the captured outgoing/incoming context.
    """
    rows = ((group_context or {}).get("groups", {}) or {}).get(str(value), []) or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("outgoing_fk_rows", "incoming_relation_rows"):
            relations = row.get(key, {}) or {}
            if any(values for values in relations.values()):
                return True
    return False


def _single_value_joined_context_support(
    value: str,
    group_context: dict | None,
    fk_context: dict | None,
) -> tuple[bool, dict]:
    """Measure whether one opaque enum bucket has stable subtype context.

    A single joined row is enough to prove that ContextEnhanced has data, but
    not enough to assert that every entity in the bucket shares one subtype.
    Require one non-identity relation signature to cover the full value bucket
    at the same conservative sample/ratio used by the physical SH guard.  The
    table's PK join is excluded because it only restates inherited base
    identity.  Sample rows are retained for audit, never used as coverage
    proof; otherwise a heterogeneous group can look homogeneous by chance.
    """
    rows = ((group_context or {}).get("groups", {}) or {}).get(str(value), []) or []
    pk_column = (group_context or {}).get("pk_column")
    signature_counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        outgoing = row.get("outgoing_fk_rows", {}) or {}
        for signature, joined_rows in outgoing.items():
            source_column = str(signature).split("->", 1)[0]
            if source_column == pk_column or not joined_rows:
                continue
            key = f"outgoing:{signature}"
            signature_counts[key] = signature_counts.get(key, 0) + 1
        incoming = row.get("incoming_relation_rows", {}) or {}
        for signature, joined_rows in incoming.items():
            if not joined_rows:
                continue
            key = f"incoming:{signature}"
            signature_counts[key] = signature_counts.get(key, 0) + 1

    sample_count = len(rows)
    sample_supports = [
        {
            "signature": signature,
            "matched_rows": matched_rows,
            "sample_rows": sample_count,
            "ratio": round(matched_rows / sample_count, 4) if sample_count else 0.0,
        }
        for signature, matched_rows in sorted(signature_counts.items())
    ]

    coverage_supports = []
    pk_column = (fk_context or {}).get("pk_column") or pk_column
    coverage_by_value = (fk_context or {}).get("coverage_by_value", {}) or {}
    for signature, by_value in sorted(coverage_by_value.items()):
        if not isinstance(by_value, dict):
            continue
        if str(signature).startswith("outgoing:"):
            source_column = str(signature).split(":", 1)[1].split("->", 1)[0]
            if source_column == pk_column:
                continue
        stat = by_value.get(str(value), {}) or {}
        total = int(stat.get("total", 0) or 0)
        linked = int(stat.get("linked", 0) or 0)
        ratio = float(stat.get("ratio", 0.0) or 0.0)
        coverage_supports.append({
            "signature": signature,
            "total_entities": total,
            "linked_entities": linked,
            "ratio": ratio,
        })
    strong_supports = [
        item
        for item in coverage_supports
        if item["total_entities"] >= REAL_VALUE_SINGLE_VALUE_SH_MIN_ENTITIES
        and item["ratio"] >= REAL_VALUE_SINGLE_VALUE_SH_MIN_RATIO
    ]
    return bool(strong_supports), {
        "sample_rows": sample_count,
        "sample_supports": sample_supports,
        "coverage_supports": coverage_supports,
        "strong_supports": strong_supports,
        "min_entities": REAL_VALUE_SINGLE_VALUE_SH_MIN_ENTITIES,
        "min_ratio": REAL_VALUE_SINGLE_VALUE_SH_MIN_RATIO,
    }


def _direct_enum_subclass_candidates(
    class_candidates: list[dict] | None,
    current_class_uri: str | None,
    ontology: dict | None,
) -> tuple[list[dict], dict]:
    """Bound discriminator review to named direct subclasses of the table Class.

    MapEnumVal maps one observed enum value to one Class.  Once the table Class
    is confirmed, a TYPE discriminator refines that Class; it must not select
    the already asserted base Class or an unrelated ontology node.
    """
    if not current_class_uri:
        return [], {
            "current_class_uri": None,
            "direct_child_uris": [],
            "eligible_candidate_uris": [],
            "reason": "missing_confirmed_table_class",
        }

    children_of = (ontology or {}).get("children_of", {}) or {}
    subclass_of = (ontology or {}).get("subclass_of", {}) or {}
    direct_children = set(children_of.get(current_class_uri, []) or [])
    if not direct_children:
        # Some ontology loaders expose only child -> direct parent adjacency.
        direct_children = {
            str(candidate_uri)
            for candidate_uri, parents in subclass_of.items()
            if current_class_uri in (parents or [])
        }

    eligible: list[dict] = []
    seen: set[str] = set()
    for candidate in class_candidates or []:
        if not isinstance(candidate, dict):
            continue
        uri = candidate.get("uri")
        if (
            not uri
            or uri in seen
            or uri not in direct_children
            or not _class_is_compatible_with_base(uri, current_class_uri, ontology)
        ):
            continue
        seen.add(uri)
        eligible.append(candidate)

    return eligible, {
        "current_class_uri": current_class_uri,
        "direct_child_uris": sorted(direct_children),
        "eligible_candidate_uris": [item.get("uri") for item in eligible],
        "reason": (
            "bounded_to_direct_subclasses"
            if eligible
            else "no_eligible_direct_subclass_candidate"
        ),
    }


def _is_boolean_asserted_class(class_uri: str | None, bool_hint_classes: list[str] | None) -> bool:
    return bool(class_uri and class_uri in set(bool_hint_classes or []))


def _aligned_table_classes(entry: dict | None) -> list[str]:
    """Return the named ontology classes represented by one aligned table."""
    entry = entry or {}
    if entry.get("pattern") == "SH":
        values = [entry.get("sub_class_uri"), entry.get("parent_class_uri")]
    else:
        values = [entry.get("class_uri")]
    return [str(value) for value in values if value]


def _range_storage_rank(
    table_class_uri: str | None,
    range_class_uri: str | None,
    ontology: dict | None,
) -> tuple[int, int] | None:
    """Rank a mapped table class as a physical container for an OP range.

    Exact range tables are preferred, followed by a table for a narrower
    subtype, and finally a table for the nearest named supertype.  The latter
    is common in denormalized schemas where role instances share their parent
    entity table.  Only explicit ontology paths are considered.
    """
    if not table_class_uri or not range_class_uri:
        return None
    if table_class_uri == range_class_uri:
        return (0, 0)

    subclass_of = (ontology or {}).get("subclass_of", {}) or {}
    subtype_distance = _ancestor_distance(
        table_class_uri,
        range_class_uri,
        subclass_of,
    )
    if subtype_distance is not None:
        return (1, subtype_distance)

    supertype_distance = _ancestor_distance(
        range_class_uri,
        table_class_uri,
        subclass_of,
    )
    if supertype_distance is not None:
        return (2, supertype_distance)
    return None


def _build_exact_object_property_discriminator_evidence(
    table_name: str,
    type_col: str,
    group_values: list[str],
    class_candidates: list[dict],
    enriched_schema: dict | None,
    alignment: dict | None,
    ontology: dict | None,
) -> dict[str, list[dict]]:
    """Compatibility shim: ObjectProperty selection belongs to OPMapping.

    RealValue follows paper Section 4.2 and may use joined relational context,
    but it must not select ontology ObjectProperty URIs or feed them into Class
    scores. This inert entry point remains only for offline caller compatibility.
    """
    return {}


def _build_relation_domain_class_evidence(
    table_name: str,
    type_col: str,
    group_values: list[str],
    class_candidates: list[dict],
    current_class_uri: str | None,
    enriched_schema: dict | None,
    alignment: dict | None,
    ontology: dict | None,
    *,
    value_domain_complete: bool,
) -> dict[str, list[dict]]:
    """Build a conservative joined-context signal for enum-to-Class mapping.

    This helper never selects or emits an ObjectProperty mapping.  It uses a
    uniquely exact relation name and its ontology domain/range only as schema
    context, as described by the paper's ContextEnhanced step.  A Class is
    asserted only when all of the following independent conditions hold:

    * the observed enum value domain is complete and has a contrasting peer;
    * one source column exactly names one ontology ObjectProperty;
    * that property has one named domain which is a candidate strict subtype
      of the already confirmed table Class, and one named range;
    * the column is non-null for every row in this bucket and null for every
      peer bucket; and
    * every value resolves to one uniquely nearest, high-confidence mapped
      table capable of storing the declared range.

    The evidence is Class-only.  Downstream OPMapping remains solely
    responsible for deciding whether the source column maps to a property.
    """
    values = sorted({str(value) for value in group_values})
    if (
        not value_domain_complete
        or len(values) < 2
        or not current_class_uri
        or not enriched_schema
        or not ontology
    ):
        return {}

    source_info = (enriched_schema or {}).get(table_name, {}) or {}
    source_columns = source_info.get("columns", {}) or {}
    if type_col not in source_columns:
        return {}

    source_alignment = ((alignment or {}).get(table_name, {}) or {})
    if source_alignment.get("class_confidence") != "high":
        return {}

    allowed = {
        str(candidate.get("uri"))
        for candidate in (class_candidates or [])
        if candidate.get("uri")
    }
    descendants = set(
        (ontology or {}).get("descendants_of", {}).get(current_class_uri, []) or []
    )
    if not descendants:
        descendants = set(
            _collect_descendants(
                current_class_uri,
                (ontology or {}).get("children_of", {}) or {},
            )
        )

    # Local-name equality is exact after punctuation/case normalization.  A
    # collision across ontology namespaces is ambiguity, not extra support.
    props_by_name: dict[str, list[tuple[str, dict]]] = {}
    for prop_uri, prop_info in ((ontology or {}).get("object_properties", {}) or {}).items():
        normalized = _norm_token(_uri_local_name(prop_uri))
        if normalized:
            props_by_name.setdefault(normalized, []).append((prop_uri, prop_info or {}))

    source_pks = set(source_info.get("primary_key", []) or [])
    evidence: dict[str, list[dict]] = {value: [] for value in values}
    conn = None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            for column_name in source_columns:
                if column_name == type_col or column_name in source_pks:
                    continue
                column_alignment = (source_alignment.get("columns", {}) or {}).get(
                    column_name,
                    {},
                ) or {}
                if column_alignment.get("role") in {
                    "pk",
                    "sh_inherited_pk",
                    "discriminator",
                }:
                    continue
                # A settled DatatypeProperty is contrary schema evidence.  A
                # low-confidence/null DP decision may still be corrected by
                # joined identity evidence without changing the DP mapping.
                if (
                    column_alignment.get("role") == "data_attr"
                    and column_alignment.get("prop_uri")
                    and column_alignment.get("confidence") == "high"
                ):
                    continue

                matching_props = props_by_name.get(_norm_token(column_name), [])
                if len(matching_props) != 1:
                    continue
                prop_uri, prop_info = matching_props[0]

                raw_domains = list(dict.fromkeys(prop_info.get("domain", []) or []))
                raw_ranges = list(dict.fromkeys(prop_info.get("range", []) or []))
                if (
                    len(raw_domains) != 1
                    or len(raw_ranges) != 1
                    or not str(raw_domains[0]).startswith("http")
                    or not str(raw_ranges[0]).startswith("http")
                ):
                    continue
                domain_uri = str(raw_domains[0])
                range_uri = str(raw_ranges[0])
                if (
                    domain_uri not in allowed
                    or domain_uri not in descendants
                    or not _class_is_compatible_with_base(
                        domain_uri,
                        current_class_uri,
                        ontology,
                    )
                ):
                    continue

                # A scalar source column cannot prove a composite target
                # identity.  For every table retain only its closest compatible
                # aligned Class so SH parent/subclass annotations do not create
                # duplicate target candidates.
                target_options: list[dict] = []
                for target_table in sorted(enriched_schema):
                    target_info = (enriched_schema.get(target_table, {}) or {})
                    target_pks = [
                        pk
                        for pk in (target_info.get("primary_key", []) or [])
                        if pk in (target_info.get("columns", {}) or {})
                    ]
                    if len(target_pks) != 1:
                        continue
                    target_alignment = ((alignment or {}).get(target_table, {}) or {})
                    if target_alignment.get("class_confidence") != "high":
                        continue
                    ranked_classes = []
                    for target_class in _aligned_table_classes(target_alignment):
                        rank = _range_storage_rank(target_class, range_uri, ontology)
                        if rank is not None:
                            ranked_classes.append((rank, target_class))
                    if not ranked_classes:
                        continue
                    rank, target_class = min(ranked_classes)
                    target_options.append(
                        {
                            "rank": rank,
                            "table": target_table,
                            "pk": target_pks[0],
                            "class_uri": target_class,
                        }
                    )
                if not target_options:
                    continue

                cur.execute(
                    sql.SQL(
                        "SELECT CAST({type_column} AS TEXT), COUNT(*), COUNT({column}) "
                        "FROM {source_table} WHERE {type_column} IS NOT NULL "
                        "GROUP BY CAST({type_column} AS TEXT)"
                    ).format(
                        type_column=sql.Identifier(type_col),
                        column=sql.Identifier(column_name),
                        source_table=_qualified_table(table_name, DB_SCHEMA_NAME),
                    )
                )
                group_stats = {
                    str(value): (int(total or 0), int(nonnull or 0))
                    for value, total, nonnull in cur.fetchall()
                }
                if set(group_stats) != set(values):
                    continue

                for value in values:
                    total, nonnull = group_stats[value]
                    if (
                        total < REAL_VALUE_SINGLE_VALUE_SH_MIN_ENTITIES
                        or nonnull != total
                    ):
                        continue
                    peer_nonnull = [
                        group_stats[peer][1]
                        for peer in values
                        if peer != value
                    ]
                    if not peer_nonnull or any(count != 0 for count in peer_nonnull):
                        continue

                    fully_linked_targets = []
                    for option in target_options:
                        cur.execute(
                            sql.SQL(
                                "SELECT COUNT(*) FROM {source_table} AS source "
                                "WHERE CAST(source.{type_column} AS TEXT) = %s "
                                "AND source.{column} IS NOT NULL "
                                "AND EXISTS ("
                                "SELECT 1 FROM {target_table} AS target "
                                "WHERE CAST(target.{target_pk} AS TEXT) "
                                "= CAST(source.{column} AS TEXT)"
                                ")"
                            ).format(
                                source_table=_qualified_table(table_name, DB_SCHEMA_NAME),
                                type_column=sql.Identifier(type_col),
                                column=sql.Identifier(column_name),
                                target_table=_qualified_table(option["table"], DB_SCHEMA_NAME),
                                target_pk=sql.Identifier(option["pk"]),
                            ),
                            (value,),
                        )
                        linked = int((cur.fetchone() or [0])[0] or 0)
                        if linked == nonnull:
                            fully_linked_targets.append({**option, "linked": linked})
                    if not fully_linked_targets:
                        continue

                    best_rank = min(option["rank"] for option in fully_linked_targets)
                    nearest = [
                        option
                        for option in fully_linked_targets
                        if option["rank"] == best_rank
                    ]
                    if len(nearest) != 1:
                        continue
                    target = nearest[0]
                    evidence[value].append(
                        {
                            "class_uri": domain_uri,
                            "evidence_source": "validated_relation_domain_context",
                            "source_column": column_name,
                            "relation_local_name": _uri_local_name(prop_uri),
                            "range_class_uri": range_uri,
                            "target_table": target["table"],
                            "target_pk": target["pk"],
                            "target_class_uri": target["class_uri"],
                            "total": total,
                            "nonnull": nonnull,
                            "linked": target["linked"],
                            "peer_nonnull": 0,
                        }
                    )
    except Exception as exc:
        print(
            f"  [WARN] 构建 relation-domain Class 上下文失败 "
            f"{table_name}.{type_col}: {exc}"
        )
        if conn:
            conn.rollback()
        return {}
    finally:
        if conn:
            conn.close()

    return {value: rows for value, rows in evidence.items() if rows}


def _strong_relation_domain_class(
    value: str,
    relation_domain_evidence: dict | None,
    allowed_classes: set[str] | None = None,
) -> tuple[str | None, dict]:
    """Return one Class only when every strong relation signal agrees."""
    supports = [
        item
        for item in ((relation_domain_evidence or {}).get(str(value), []) or [])
        if item.get("evidence_source") == "validated_relation_domain_context"
        and item.get("class_uri")
        and (
            not allowed_classes
            or item.get("class_uri") in allowed_classes
        )
    ]
    supported_classes = sorted({item["class_uri"] for item in supports})
    return (
        supported_classes[0] if len(supported_classes) == 1 else None,
        {
            "supports": supports,
            "supported_classes": supported_classes,
            "strong": len(supported_classes) == 1,
        },
    )


def _ambiguous_single_value_leaf_classes(
    values: list[str],
    class_candidates: list[dict],
    current_class_uri: str | None,
    ontology: dict | None,
) -> tuple[list[str], dict]:
    """Never infer a multi-label Class assertion from matcher ambiguity.

    OWL's open-world semantics do not make two classes intersect merely because
    their disjointness was not declared.  MapEnumVal produces one Class per
    value; an opaque singleton numeric value therefore needs a unique strong
    instance/context signal or must remain unmapped.  The helper is retained as
    a compatibility shim for callers of the former shortcut, but it can no
    longer authorize any Class assertion.
    """
    return [], {
        "eligible": False,
        "leaves": [],
        "reason": "non_disjointness_does_not_prove_class_intersection",
    }


def _residual_direct_branch_mapping(
    *,
    values: list[str],
    value_to_class: dict,
    class_candidates: list[dict],
    current_class_uri: str | None,
    class_subclass_of: dict | None,
    sh_value_evidence: dict | None,
    value_domain_complete: bool = False,
    ontology: dict | None = None,
) -> tuple[str | None, str | None]:
    """Complete a closed enum partition only from physical branch evidence.

    If all but one direct child branch have a uniquely measured SH membership
    mapping, and values cover the complete direct-child partition, the final
    code/branch pair is structurally determined.  No table name, query ID, or
    target F1 participates in this rule.
    """
    if not value_domain_complete:
        return None, None
    if not current_class_uri or len(values or []) < 2:
        return None, None
    if any(not _is_numeric_enum_value(value) for value in values):
        return None, None

    candidate_uris = {
        candidate.get("uri") for candidate in class_candidates or []
        if candidate.get("uri")
    }
    direct_children = {
        candidate_uri
        for candidate_uri in candidate_uris
        if current_class_uri in ((class_subclass_of or {}).get(candidate_uri, []) or [])
    }
    if len(direct_children) != len(values):
        return None, None

    ontology_children = set(
        ((ontology or {}).get("children_of", {}) or {}).get(current_class_uri, [])
        or []
    )
    if not ontology_children:
        ontology_children = {
            candidate_uri
            for candidate_uri, parents in ((class_subclass_of or {}).items())
            if current_class_uri in (parents or [])
        }
    if not ontology_children or direct_children != ontology_children:
        return None, None

    assigned: dict[str, str] = {}
    for raw_value, class_uri in (value_to_class or {}).items():
        if isinstance(class_uri, str) and class_uri in direct_children:
            assigned[str(raw_value)] = class_uri
    if len(assigned) != len(values) - 1 or len(set(assigned.values())) != len(assigned):
        return None, None
    if not all(
        _has_strong_sh_support(raw_value, class_uri, sh_value_evidence)
        for raw_value, class_uri in assigned.items()
    ):
        return None, None

    remaining_values = [value for value in values if value not in assigned]
    remaining_children = sorted(direct_children - set(assigned.values()))
    if len(remaining_values) != 1 or len(remaining_children) != 1:
        return None, None
    return remaining_values[0], remaining_children[0]


def _enum_llm_candidate_catalog(
    class_candidates: list[dict] | None,
    ontology: dict | None = None,
) -> list[dict]:
    """Give the enum prompt short, stable candidate ids.

    Long ontology URIs are useful as input evidence but are a poor output
    protocol: a mapping for a moderately sized enum can consume the whole
    completion budget before the JSON object is closed.  The model therefore
    chooses ``c0``, ``c1`` ... and the caller expands those ids locally.
    """
    catalog: list[dict] = []
    seen: set[str] = set()
    for candidate in class_candidates or []:
        if not isinstance(candidate, dict):
            continue
        uri = candidate.get("uri")
        if not uri or uri in seen:
            continue
        seen.add(uri)
        annotation = (
            ((ontology or {}).get("class_annotations", {}) or {}).get(uri, {})
            or {}
        )
        catalog_entry = {
            "id": f"c{len(catalog)}",
            "local_name": candidate.get("local_name") or str(uri).rsplit("#", 1)[-1],
            "uri": uri,
        }
        labels = [str(value) for value in (annotation.get("labels", []) or []) if value]
        comments = [str(value) for value in (annotation.get("comments", []) or []) if value]
        if labels:
            catalog_entry["labels"] = labels
        if comments:
            catalog_entry["comments"] = comments
        catalog.append(catalog_entry)
    return catalog


def _enum_llm_value_id(value_ref: object, review_values: list[str]) -> str | None:
    """Resolve a compact ``vN`` reference while retaining legacy value keys."""
    if isinstance(value_ref, str) and value_ref in review_values:
        return value_ref
    if isinstance(value_ref, int):
        index = value_ref
    elif isinstance(value_ref, str) and value_ref.startswith("v") and value_ref[1:].isdigit():
        index = int(value_ref[1:])
    else:
        return None
    if 0 <= index < len(review_values):
        return review_values[index]
    return None


def _enum_llm_candidate_uri(candidate_ref: object, catalog: list[dict]) -> str | None:
    """Resolve compact candidate ids and tolerate the old URI response format."""
    if candidate_ref is None:
        return None
    if isinstance(candidate_ref, int):
        index = candidate_ref
    elif isinstance(candidate_ref, str) and candidate_ref.startswith("c") and candidate_ref[1:].isdigit():
        index = int(candidate_ref[1:])
    else:
        text = str(candidate_ref)
        for candidate in catalog:
            if text in {candidate.get("uri"), candidate.get("local_name")}:
                return candidate.get("uri")
        return text
    if 0 <= index < len(catalog):
        return catalog[index].get("uri")
    return None


def _merge_enum_llm_assignments(
    response: dict | None,
    review_values: list[str],
    candidate_catalog: list[dict],
    allowed: set[str],
    current_class_uri: str | None,
    ontology: dict | None,
    bool_hint_classes: list[str] | None,
    locked_by_rule: set[str],
) -> dict[str, str]:
    """Validate both compact and legacy enum responses.

    The compact protocol is deliberately evidence-neutral: it only changes
    how a provider names an already supplied candidate.  All URI membership,
    OWL compatibility and discriminator guards remain local and unchanged.
    """
    if not isinstance(response, dict):
        return {}

    assignments: dict[str, object] = {}
    legacy = response.get("value_to_class")
    if isinstance(legacy, dict):
        assignments.update(legacy)

    compact = (
        response.get("candidate_assignments")
        or response.get("value_to_candidate")
        or response.get("assignments")
    )
    if isinstance(compact, dict):
        assignments.update(compact)
    elif isinstance(compact, list):
        for item in compact:
            if not isinstance(item, dict):
                continue
            value_ref = item.get("value_id", item.get("value"))
            assignments[value_ref] = item.get("candidate_id", item.get("candidate"))

    merged: dict[str, str] = {}
    for value_ref, candidate_ref in assignments.items():
        value = _enum_llm_value_id(value_ref, review_values)
        if value is None or value in locked_by_rule:
            continue
        uri = _enum_llm_candidate_uri(candidate_ref, candidate_catalog)
        if (
            uri in allowed
            and _class_is_compatible_with_base(uri, current_class_uri, ontology)
            and not (
                _is_numeric_enum_value(value)
                and _is_boolean_asserted_class(uri, bool_hint_classes)
            )
        ):
            merged[value] = uri
    return merged


def _real_value_type_mapping(
    table_name: str,
    type_col: str,
    value_profiles: dict[str, list[dict]],
    class_candidates: list[dict],
    bool_hint_classes: list[str] | None = None,
    bool_assertions: list[dict] | None = None,
    class_ancestors: dict | None = None,
    class_subclass_of: dict | None = None,
    sh_value_evidence: dict | None = None,
    relation_domain_evidence: dict | None = None,
    current_class_uri: str | None = None,
    current_class_confidence: str | None = None,
    descendant_uris: list[str] | None = None,
    fk_context: dict | None = None,
    group_context: dict | None = None,
    ontology: dict | None = None,
    candidate_constraints: dict | None = None,
    value_domain_complete: bool = False,
) -> dict:
    """
    TYPE 列重判：规则优先，LLM 兜底。
    仅对判别列（低基数枚举）启用规则映射；非判别列保留 LLM 主判。
    """
    values = sorted((str(k) for k in (value_profiles or {}).keys()), key=lambda x: (len(x), x))
    if not values:
        return {
            "value_to_class": {},
            "unmapped_values": [],
            "confidence": "low",
            "reason": f"{type_col} 无有效取值样本",
        }

    allowed = {c.get("uri") for c in class_candidates if c.get("uri")}
    value_to_class: dict[str, str] = {}
    locked_by_rule: set[str] = set()
    abstained_values: set[str] = set()
    ranking_snapshot = {}
    mapped_by_rule = 0

    rule_first, rule_diag = _should_rule_first_for_enum(
        values=values,
        value_profiles=value_profiles,
        fk_context=fk_context,
        descendant_uris=descendant_uris,
        sh_value_evidence=sh_value_evidence,
    )

    # Always compute the deterministic ranking.  It is both the rule input and
    # the audit context shown to an LLM fallback; previously it was absent when
    # the enum gate did not trigger, which made low-cardinality numeric values
    # effectively random.
    rankings_by_value = {}
    for v in values:
        ranking = _score_enum_value_by_rules(
            value=v,
            rows=value_profiles.get(v, []) or [],
            class_candidates=class_candidates,
            fk_context=fk_context,
            type_col=type_col,
            bool_hint_classes=bool_hint_classes,
            bool_assertions=bool_assertions,
            class_ancestors=class_ancestors,
            class_subclass_of=class_subclass_of,
            sh_value_evidence=sh_value_evidence,
            current_class_uri=current_class_uri,
            descendant_uris=descendant_uris,
            all_values=values,
        )
        rankings_by_value[v] = ranking
        ranking_snapshot[v] = [
            {
                "uri": r.get("uri"),
                "local_name": r.get("local_name"),
                "score": round(r.get("score", 0.0), 3),
            }
            for r in ranking[:5]
        ]

    # A lone numeric code has no lexical or contrastive meaning.  A uniquely
    # measured physical SH branch may lock it deterministically.  Otherwise,
    # Paper Algorithm 1 still permits ContextEnhanced when sampled rows contain
    # a real join: the provider then receives only named direct subclasses and
    # may select at most one.  Without a materialized join, abstain locally.
    singleton_numeric = len(values) == 1 and _is_numeric_enum_value(values[0])
    singleton_review_candidates: list[dict] = []
    if singleton_numeric:
        only_value = values[0]
        selected_uri, singleton_audit = _strong_single_value_sh_class(
            only_value,
            sh_value_evidence,
            allowed_classes=allowed,
        )
        if selected_uri:
            return {
                "value_to_class": {only_value: selected_uri},
                "unmapped_values": [],
                "confidence": "high",
                "reason": (
                    "单值数字编码仅由唯一高覆盖、足够样本且明显领先的 "
                    "validated SH identity 证据锁定"
                ),
            }

        has_any_joined_context = _has_materialized_joined_context(
            only_value,
            group_context,
        )
        has_joined_context, joined_context_audit = (
            _single_value_joined_context_support(
                only_value,
                group_context,
                fk_context,
            )
        )
        singleton_review_candidates, candidate_audit = (
            _direct_enum_subclass_candidates(
                class_candidates,
                current_class_uri,
                ontology,
            )
        )
        rule_diag["singleton_numeric_context"] = {
            "value": only_value,
            "has_materialized_joined_context": has_any_joined_context,
            "has_stable_joined_context": has_joined_context,
            "joined_context": joined_context_audit,
            "sh_evidence": singleton_audit,
            "candidate_scope": candidate_audit,
        }
        if not has_joined_context or not singleton_review_candidates:
            normalized = int(only_value) if only_value.isdigit() else only_value
            missing_evidence = (
                "缺少稳定覆盖的非身份 joined relation rows"
                if not has_joined_context
                else "缺少当前表 Class 的可用直接子类候选"
            )
            return {
                "value_to_class": {},
                "unmapped_values": [normalized],
                "confidence": "low",
                "reason": (
                    f"单值数字编码{missing_evidence}，保持 unmapped；"
                    f"audit={json.dumps(rule_diag['singleton_numeric_context'], ensure_ascii=False)}"
                ),
            }

        # Matcher scores cannot turn an opaque number into a fact.  The strong
        # physical rule above may lock a result; the remaining joined case must
        # follow the paper's bounded LLM selection path.
        rule_first = False

    # Direct lexical evidence is independent of provider wording.  Lock only
    # unique exact/prefix matches; all other values still go through measured
    # relation/SH context and (if necessary) the constrained LLM review.
    lexical_locked: dict[str, dict] = {}
    for value in values:
        selected_uri, lexical_audit = _unique_lexical_enum_class(
            value,
            class_candidates,
            current_class_uri,
            ontology,
        )
        if selected_uri and selected_uri in allowed:
            value_to_class[value] = selected_uri
            locked_by_rule.add(value)
            mapped_by_rule += 1
            lexical_locked[value] = {
                "selected_uri": selected_uri,
                **lexical_audit,
            }
    rule_diag["lexical_locked_values"] = lexical_locked

    # A uniquely named relation column can provide Class evidence through its
    # ontology domain, but only after the builder has validated complete and
    # contrastive real-value coverage plus a fully resolved range identity.
    # This is a Class-only context lock; it does not assert an OP mapping.
    relation_locked = 0
    relation_diagnostics = {}
    for value in values:
        relation_uri, diagnostic = _strong_relation_domain_class(
            value,
            relation_domain_evidence,
            allowed_classes=allowed,
        )
        relation_diagnostics[value] = diagnostic
        if relation_uri:
            value_to_class[value] = relation_uri
            locked_by_rule.add(value)
            abstained_values.discard(value)
            mapped_by_rule += 1
            relation_locked += 1
    rule_diag["relation_domain_context"] = relation_diagnostics

    if rule_first:
        for v in values:
            if v in locked_by_rule:
                continue
            ranking = rankings_by_value.get(v, [])
            if not ranking:
                continue
            top = ranking[0]
            sec = ranking[1] if len(ranking) > 1 else {"score": 0.0}
            top_score = float(top.get("score", 0.0) or 0.0)
            gap = top_score - float(sec.get("score", 0.0) or 0.0)
            top_uri = top.get("uri")
            if top_uri not in allowed:
                continue
            if _is_numeric_enum_value(v) and _is_boolean_asserted_class(top_uri, bool_hint_classes):
                continue

            high = top_score >= REAL_VALUE_TYPE_HIGH_SCORE and gap >= REAL_VALUE_TYPE_HIGH_GAP
            medium = top_score >= REAL_VALUE_TYPE_MEDIUM_SCORE and gap >= REAL_VALUE_TYPE_MEDIUM_GAP
            weak = (
                not _is_numeric_enum_value(v)
                and top_score >= REAL_VALUE_TYPE_WEAK_SCORE
                and gap >= 0.0
            )
            if high or medium or weak:
                value_to_class[v] = top_uri
                mapped_by_rule += 1
                if high:
                    locked_by_rule.add(v)

    # Residual completion is a deterministic, evidence-gated rule.  Apply it
    # before asking an LLM so a fully determined partition neither consumes a
    # request nor disappears when the provider is unavailable.  The helper
    # requires a complete observed value domain, the complete ontology direct-
    # child set, and strong physical SH identity evidence for every assigned
    # branch; the sole remaining value/Class pair is therefore locked.
    residual_value, residual_class = _residual_direct_branch_mapping(
        values=values,
        value_to_class=value_to_class,
        class_candidates=class_candidates,
        current_class_uri=current_class_uri,
        class_subclass_of=class_subclass_of,
        sh_value_evidence=sh_value_evidence,
        value_domain_complete=value_domain_complete,
        ontology=ontology,
    )
    if residual_value and residual_class:
        value_to_class[residual_value] = residual_class
        locked_by_rule.add(residual_value)
        abstained_values.discard(residual_value)
        mapped_by_rule += 1

    # 非高置信锁定值交给 LLM 结合分组上下文复核；高置信规则值不可覆盖。
    review_values = [
        v for v in values
        if v not in locked_by_rule and v not in abstained_values
    ]
    llm_reason = ""
    llm_conf = "medium"
    singleton_contract_required = False
    singleton_contract_retry_used = False
    singleton_contract_unresolved = False
    if review_values:
        rule_mode_text = "规则先验 + 分组上下文 LLM复核" if rule_first else "分组上下文 LLM主判"
        llm_class_candidates = (
            singleton_review_candidates
            if singleton_numeric
            else class_candidates
        )
        candidate_catalog = _enum_llm_candidate_catalog(
            llm_class_candidates,
            ontology=ontology,
        )
        llm_allowed = {
            candidate.get("uri")
            for candidate in llm_class_candidates or []
            if candidate.get("uri")
        }
        review_profiles = [
            {
                "id": f"v{index}",
                "value": value,
                "sample_rows": value_profiles.get(value, []),
            }
            for index, value in enumerate(review_values)
        ]
        locked_map = {v: uri for v, uri in value_to_class.items() if v in locked_by_rule}
        tentative_map = {v: uri for v, uri in value_to_class.items() if v not in locked_by_rule}
        # Paper Algorithm 1 defines MapEnumVal over every validated enum-value
        # input.  A singleton numeric bucket reaches this point only after the
        # local gate has proved stable, non-identity joined context and bounded
        # the candidates to direct children of the confirmed table Class.
        # That is materially different from an opaque code with no evidence:
        # the provider must now make one contextual choice, while the latter
        # already returned unmapped before any paid call.
        singleton_contract_required = bool(
            singleton_numeric
            and len(review_values) == 1
            and singleton_review_candidates
            and value_domain_complete
            and current_class_confidence == "high"
        )
        selection_policy = (
            "本值已经通过稳定 joined-context 门，属于论文 MapEnumVal 的有效输入。"
            "必须从候选直接子类中选择且只选择一个；null/unmapped 违反本轮输出契约。"
            "若多个候选都可兼容，应选择最能解释已物化业务关系丰富度、角色和阶段的类。"
            if singleton_contract_required
            else "若上下文证据仍不足，可以返回 null 并放入 unmapped_values。"
        )
        prompt = f"""
## 任务
表 `{table_name}` 中，列 `{type_col}` 是类型判别列（discriminator）。
当前采用：{rule_mode_text}。
请根据每个 TYPE 值的分组样本、FK 上下文、incoming/outgoing 关系证据，为每个待复核值选择最合适的 Class。
{selection_policy}

## TYPE 待复核取值与本表样本（只使用 value_id）
{json.dumps(review_profiles, ensure_ascii=False, indent=2, default=str)}

## 分组上下文大表（本行样本 + FK引用行 + incoming关系行）
{json.dumps(group_context or {}, ensure_ascii=False, indent=2, default=str)}

## 高置信规则锁定映射（不可改）
{json.dumps(locked_map, ensure_ascii=False, indent=2)}

## 规则暂定映射（必须重新审查，可以改）
{json.dumps(tentative_map, ensure_ascii=False, indent=2)}

## 规则先验（用于参考）
{json.dumps(ranking_snapshot, ensure_ascii=False, indent=2)}

## SH 子表按 TYPE 值的实例覆盖证据（用于参考）
{json.dumps(sh_value_evidence or {}, ensure_ascii=False, indent=2, default=str)}

## 本体层级/候选过滤审计
{json.dumps(candidate_constraints or {}, ensure_ascii=False, indent=2, default=str)}

## 规则触发诊断（用于参考）
{json.dumps(rule_diag, ensure_ascii=False, indent=2)}

## FK 语义上下文（来自真实 schema）
{json.dumps(fk_context or {}, ensure_ascii=False, indent=2, default=str)}

## 候选 Class（必须从这里选）
{json.dumps(candidate_catalog, ensure_ascii=False, indent=2)}

## 该表已由布尔判别列直接表达的语义类（仅供约束参考）
{json.dumps(bool_hint_classes or [], ensure_ascii=False, indent=2)}

## 要求
1. 只为待复核取值输出映射，不要输出高置信锁定值。
2. 数字 TYPE 值本身没有语义，必须主要依据分组样本和 FK/关系上下文判断。
3. 如果某个 Class 已经由布尔列直接表达，除非分组上下文强支持，不要再用 TYPE 重复映射到该 Class。
4. 遵守上面的 MapEnumVal 选择策略；只有未通过稳定 joined-context 门的普通复核值才允许 unmapped。
5. 不得选择候选列表之外、或与已知表 Class 不相容的 URI；被过滤的候选不能恢复。
6. 对只有一个数值编码且既没有强实例覆盖、也没有实际 joined relation rows
   的情况，必须放入 unmapped_values；若 joined context 存在，也最多选择一个类。
7. 只允许输出候选列表中的 id；本地程序会验证 id 对应的 URI。
8. 数值大小、候选顺序和相同的 matcher 分数都不是语义证据；不得据此选择。
9. 必须把候选的 label/comment 与已物化的 joined relation 逐项对照。对于表示
   同一实体不同完整度、阶段或活跃角色的兄弟类，丰富且稳定覆盖的业务关系支持
   能解释这些关系的完整/活跃状态；关系缺失才支持精简、占位或仅部分内容状态。
10. 单值编码的 reason 必须指出至少一个能区分候选的关系事实与候选语义；不得沿用
    规则先验、数值大小或候选顺序。已通过稳定 joined-context 门的值必须据此单选，
    其他值若做不到才返回 null 并放入 unmapped_values。

## 输出格式（严格 JSON，必须使用短 ID）
{{
  "candidate_assignments": {{
    "v0": "c0",
    "v1": null
  }},
  "unmapped_value_ids": ["v1"],
  "confidence": "high / medium / low",
  "reason": "一句话说明"
}}

candidate_assignments 的 key 只能是上面给出的 value_id（v0、v1…），value 只能是候选 Class 的 id（c0、c1…）或 null；不要在输出中重复长 URI。
"""
        m = _call_llm(prompt) or {}
        llm_reason = m.get("reason", "")
        llm_conf = m.get("confidence", "medium")
        merged_llm = _merge_enum_llm_assignments(
            response=m,
            review_values=review_values,
            candidate_catalog=candidate_catalog,
            allowed=llm_allowed,
            current_class_uri=current_class_uri,
            ontology=ontology,
            bool_hint_classes=bool_hint_classes,
            locked_by_rule=locked_by_rule,
        )

        # A provider may still emit null despite the total-function contract.
        # Retry the protocol once with the same evidence and candidate set.
        # Never recover by choosing top1/c0 locally: that would turn candidate
        # order into hidden supervision and make the score non-auditable.
        required_value = review_values[0] if singleton_contract_required else None
        if required_value is not None and required_value not in merged_llm:
            singleton_contract_retry_used = True
            retry_prompt = f"""
## MapEnumVal 输出契约重试
表 `{table_name}` 的判别列 `{type_col}` 已通过稳定非身份 joined-context 门。
当前只有一个待映射值 v0。论文的 MapEnumVal 对该有效输入必须返回一个 Class。

## 值与样本
{json.dumps(review_profiles, ensure_ascii=False, indent=2, default=str)}

## 已物化 joined context
{json.dumps(group_context or {}, ensure_ascii=False, indent=2, default=str)}

## FK/关系覆盖审计
{json.dumps(rule_diag.get("singleton_numeric_context", {}), ensure_ascii=False, indent=2, default=str)}

## 候选直接子类
{json.dumps(candidate_catalog, ensure_ascii=False, indent=2)}

请比较候选 label/comment 与业务关系的丰富度、角色和阶段，选择最能解释这些关系的
一个候选。数字大小、候选顺序、matcher 并列分数都不是证据。不得返回 null，不得
选择候选外 URI，也不得返回多个类。

严格输出：
{{
  "candidate_assignments": {{"v0": "c0"}},
  "unmapped_value_ids": [],
  "confidence": "high / medium / low",
  "reason": "指出区分候选的关系事实与候选语义"
}}
"""
            retry_response = _call_llm(retry_prompt) or {}
            retry_merged = _merge_enum_llm_assignments(
                response=retry_response,
                review_values=review_values,
                candidate_catalog=candidate_catalog,
                allowed=llm_allowed,
                current_class_uri=current_class_uri,
                ontology=ontology,
                bool_hint_classes=bool_hint_classes,
                locked_by_rule=locked_by_rule,
            )
            if required_value in retry_merged:
                merged_llm.update(retry_merged)
                llm_reason = retry_response.get("reason", llm_reason)
                llm_conf = retry_response.get("confidence", llm_conf)
            else:
                singleton_contract_unresolved = True
                llm_reason = (
                    "provider 两次未满足 MapEnumVal 单选契约；保持 unmapped，"
                    "未使用 top1、候选顺序或本地猜测"
                )

        value_to_class = dict(locked_map)
        value_to_class.update(merged_llm)

    explicit_unmapped = {v for v in values if v not in value_to_class}

    norm_unmapped = []
    for x in sorted(explicit_unmapped):
        if x.isdigit():
            norm_unmapped.append(int(x))
        else:
            norm_unmapped.append(x)

    if not norm_unmapped and mapped_by_rule >= max(1, len(values) - 1):
        conf = "high"
    elif len(value_to_class) >= 1:
        conf = "medium"
    else:
        conf = llm_conf if review_values else "low"

    reason_parts = []

    if mapped_by_rule:
        reason_parts.append(f"规则映射 {mapped_by_rule} 个取值")
    if relation_locked:
        reason_parts.append(
            f"其中 {relation_locked} 个由完整分组行与 relation-domain 上下文锁定"
        )
    if abstained_values:
        reason_parts.append(
            "对缺少强实例证据的单值编码保持 unmapped，保留已知上位 Class"
        )
    if residual_value and residual_class:
        reason_parts.append("由完整 SH 分支覆盖推得唯一剩余枚举分支")
    if not rule_first and review_values:
        reason_parts.append("该列未触发规则轨道，采用 LLM 主判")
    if review_values:
        reason_parts.append("非锁定取值由 LLM 结合分组上下文复核")
    if singleton_contract_required and not singleton_contract_unresolved:
        reason_parts.append("稳定 joined-context enum 按 MapEnumVal 契约完成单一直接子类选择")
    if singleton_contract_retry_used:
        reason_parts.append("provider 首次违反单选契约，已使用同证据重试一次")
    if singleton_contract_unresolved:
        reason_parts.append("provider 两次违反单选契约，安全保持 unmapped")
    if llm_reason:
        reason_parts.append(llm_reason)

    return {
        "value_to_class": value_to_class,
        "unmapped_values": norm_unmapped,
        "confidence": conf,
        "reason": "；".join(reason_parts) if reason_parts else "",
    }


def _real_value_boolean_mapping(
    table_name: str,
    bool_col: str,
    value_profiles: dict[str, list[dict]],
    class_candidates: list[dict],
    fk_context: dict | None = None,
) -> dict:
    """
    用 LLM 判断布尔判别列：true 时最可能代表哪个语义类。
    返回:
    {
      "true_class_uri": "...#Program_Chair" 或 null,
      "confidence": "high|medium|low",
      "reason": "..."
    }
    """
    prompt = f"""
## 任务
表 `{table_name}` 中，列 `{bool_col}` 是布尔判别列（boolean discriminator）。
请判断当 `{bool_col} = true` 时，对应的最合适 OWL Class（若无法确定可返回 null）。

## 取值样本（按 true/false 分组）
{json.dumps(value_profiles, ensure_ascii=False, indent=2, default=str)}

## FK 语义上下文（来自真实 schema）
{json.dumps(fk_context or {}, ensure_ascii=False, indent=2, default=str)}

## 候选 Class（必须从这里选）
{json.dumps(class_candidates, ensure_ascii=False, indent=2)}

## 输出格式（严格 JSON）
{{
  "selected_true_class_uri": "Class URI（必须来自候选列表，或 null）",
  "confidence": "high / medium / low",
  "reason": "一句话说明"
}}
"""
    m = _call_llm(prompt)
    allowed = {c.get("uri") for c in class_candidates if c.get("uri")}
    uri = m.get("selected_true_class_uri")
    if uri not in allowed:
        uri = None
    return {
        "true_class_uri": uri,
        "confidence": m.get("confidence", "medium"),
        "reason": m.get("reason", ""),
    }


def _build_sh_value_evidence(
    table_name: str,
    type_col: str,
    enriched_schema: dict | None,
    candidates: dict | None,
    alignment: dict | None,
    group_values: list[str] | None,
    ontology: dict | None = None,
) -> dict:
    """
    构建 TYPE 值 -> SH 子类证据:
      value -> [{class_uri, ratio, child_table}]
    ratio = 在该 value 下，出现在子类表中的实体占比。
    """
    if not enriched_schema or table_name not in (enriched_schema or {}):
        return {}
    parent_info = (enriched_schema or {}).get(table_name, {}) or {}
    parent_columns = parent_info.get("columns", {}) or {}
    parent_pk = [
        column for column in parent_info.get("primary_key", []) or []
        if column in parent_columns
    ]
    parent_entry = ((alignment or {}).get(table_name, {}) or {})
    parent_class = (
        parent_entry.get("sub_class_uri")
        if parent_entry.get("pattern") == "SH"
        else parent_entry.get("class_uri")
    )
    if not parent_pk or not parent_class or not group_values:
        return {}

    out: dict[str, list[dict]] = {str(v): [] for v in group_values}
    conn = None
    try:
        conn = _get_conn()
    except Exception as e:
        print(f"  [WARN] 无法连接数据库，跳过 SH 值证据 {table_name}.{type_col}: {e}")
        return out

    try:
        with conn.cursor() as cur:
            # Only a final aligned SH table with a complete physical inherited
            # identity can prove membership. Candidate guesses or a single FK
            # component cannot authorize enum type assertions.
            for child_table, child_alignment in (alignment or {}).items():
                if child_table == table_name:
                    continue
                child_alignment = ((alignment or {}).get(child_table, {}) or {})
                if child_alignment.get("pattern") != "SH":
                    continue
                if child_alignment.get("parent_class_uri") != parent_class:
                    continue
                child_class = child_alignment.get("sub_class_uri")
                if not child_class:
                    continue
                if (
                    ontology is not None
                    and parent_class not in ((ontology or {}).get("ancestors_of", {}).get(child_class, []) or [])
                    and parent_class not in ((ontology or {}).get("subclass_of", {}).get(child_class, []) or [])
                ):
                    continue
                declared_parent = child_alignment.get("parent_table")
                if declared_parent and declared_parent != table_name:
                    continue

                child_info = (enriched_schema.get(child_table, {}) or {})
                child_columns = child_info.get("columns", {}) or {}
                child_pk = {
                    column for column in child_info.get("primary_key", []) or []
                    if column in child_columns
                }
                inherited_markers = {
                    column
                    for column, column_info in (child_alignment.get("columns", {}) or {}).items()
                    if isinstance(column_info, dict)
                    and column_info.get("role") == "sh_inherited_pk"
                }
                if (
                    not child_pk
                    or not inherited_markers
                    or not inherited_markers.issubset(child_pk)
                ):
                    continue

                parent_to_child: dict[str, str] = {}
                ambiguous_parent_columns: set[str] = set()
                for fk in child_info.get("foreign_keys", []) or []:
                    child_column = fk.get("column")
                    parent_column = (
                        fk.get("ref_col")
                        or fk.get("references_column")
                        or fk.get("target_column")
                        or ""
                    )
                    parent_table = (
                        fk.get("ref_table")
                        or fk.get("references_table")
                        or fk.get("target_table")
                        or ""
                    )
                    if (
                        child_column not in inherited_markers
                        or parent_table != table_name
                        or parent_column not in parent_pk
                    ):
                        continue
                    prior = parent_to_child.get(parent_column)
                    if prior is not None and prior != child_column:
                        ambiguous_parent_columns.add(parent_column)
                    else:
                        parent_to_child[parent_column] = child_column
                if (
                    ambiguous_parent_columns
                    or set(parent_to_child) != set(parent_pk)
                    or set(parent_to_child.values()) != inherited_markers
                ):
                    continue
                identity_pairs = [
                    (parent_column, parent_to_child[parent_column])
                    for parent_column in parent_pk
                ]

                # 每个值单独统计，避免复杂 SQL 方言差异
                for v in group_values:
                    vs = str(v)
                    cur.execute(
                        sql.SQL(
                            "SELECT COUNT(*) "
                            "FROM {parent_table} AS parent "
                            "WHERE CAST(parent.{type_column} AS TEXT) = %s"
                        ).format(
                            parent_table=_qualified_table(table_name, DB_SCHEMA_NAME),
                            type_column=sql.Identifier(type_col),
                        ),
                        (vs,),
                    )
                    total = int(cur.fetchone()[0] or 0)
                    if total == 0:
                        continue

                    cur.execute(
                        sql.SQL(
                            "SELECT COUNT(*) "
                            "FROM {parent_table} AS parent "
                            "WHERE CAST(parent.{type_column} AS TEXT) = %s "
                            "AND EXISTS ("
                            "SELECT 1 FROM {child_table} AS child "
                            "WHERE {identity_join}"
                            ")"
                        ).format(
                            parent_table=_qualified_table(table_name, DB_SCHEMA_NAME),
                            type_column=sql.Identifier(type_col),
                            child_table=_qualified_table(child_table, DB_SCHEMA_NAME),
                            identity_join=sql.SQL(" AND ").join(
                                sql.SQL("child.{child_column} = parent.{parent_column}").format(
                                    child_column=sql.Identifier(child_column),
                                    parent_column=sql.Identifier(parent_column),
                                )
                                for parent_column, child_column in identity_pairs
                            ),
                        ),
                        (vs,),
                    )
                    linked = int(cur.fetchone()[0] or 0)
                    ratio = round(_safe_ratio(linked, total), 4)
                    out[vs].append(
                        {
                            "class_uri": child_class,
                            "child_table": child_table,
                            "ratio": ratio,
                            "linked": linked,
                            "total": total,
                            "evidence_source": "validated_sh_identity",
                        }
                    )
    except Exception as e:
        print(f"  [WARN] 构建 SH 值证据失败 {table_name}.{type_col}: {e}")
        conn.rollback()
    finally:
        conn.close()

    return out



def _refine_table_class_if_needed(
    *,
    table_name: str,
    pattern: str,
    table_low: bool,
    table_candidates: dict | None,
    sample_rows: list[dict],
    result: dict,
    force_all_context: bool,
) -> None:
    """Confirm a low-confidence base Class before refining its discriminator."""
    if not table_low or pattern not in ("SE", "SH"):
        return
    table_candidates = table_candidates or {}
    class_candidates = (
        table_candidates.get("sub_class_candidates", [])
        if pattern == "SH"
        else table_candidates.get("table_class_candidates", [])
    )
    if not class_candidates:
        return
    try:
        re_match = _real_value_table_class(
            table_name,
            class_candidates,
            sample_rows,
            force_llm=force_all_context,
        )
        allowed = {candidate.get("uri") for candidate in class_candidates if candidate.get("uri")}
        new_uri = re_match.get("selected_uri")
        if new_uri not in allowed:
            new_uri = class_candidates[0].get("uri")
        if pattern == "SH":
            result[table_name]["sub_class_uri"] = new_uri
        else:
            result[table_name]["class_uri"] = new_uri
        result[table_name]["class_confidence"] = re_match.get("confidence", "medium")
        print(f"  表级 Class 重判 → {new_uri} [{result[table_name]['class_confidence']}]")
        print(f"  理由: {re_match.get('reason', '')}")
    except Exception as exc:
        print(f"  [WARN] 表级真实值增强失败: {exc}，保留原结果")


# 主函数
def run_real_value_enhancement(
    alignment: dict,
    low_conf_report: dict,
    candidates: dict,
    ontology: dict | None = None,
    enriched_schema: dict | None = None,
    force_all_context: bool = False,
    checkpoint_path: str | Path | None = None,
    resume_checkpoint_path: str | Path | None = None,
) -> dict:
    """
    真实值上下文增强主函数。
    只处理 Class 和 DatatypeProperty 的低置信条目。
    fk_obj 列只重判 range Class，不判 ObjectProperty。
    SR 表只重判 domain/range Class，不判 ObjectProperty。
    """
    # RealValue is intentionally a two-pass stage: all low-confidence table
    # Classes settle before any table's values/columns are refined.  Persist the
    # two ordered prefixes separately so a continuation preserves that semantic
    # boundary instead of re-running completed paid decisions.
    from RealValue.real_value_checkpoint import (  # local import keeps helpers pure in unit tests
        PHASE_CLASS,
        PHASE_TABLE,
        RealValueCheckpointSession,
        sha256_file,
    )

    table_tasks = list(low_conf_report.items())
    class_tasks = [
        (table_name, low_info)
        for table_name, low_info in table_tasks
        if low_info.get("table_low", False)
        and alignment.get(table_name, {}).get("pattern", "SE") in ("SE", "SH")
    ]
    checkpoint = RealValueCheckpointSession(
        alignment=alignment,
        low_conf_report=low_conf_report,
        candidates=candidates,
        ontology=ontology,
        enriched_schema=enriched_schema,
        force_all_context=force_all_context,
        implementation_sha256=sha256_file(Path(__file__)),
        class_table_order=[table_name for table_name, _ in class_tasks],
        table_order=[table_name for table_name, _ in table_tasks],
        checkpoint_path=checkpoint_path,
        resume_checkpoint_path=resume_checkpoint_path,
    )
    result = checkpoint.initial_result
    # Validate provider-produced SH endpoints before using them to build
    # identity/enum context.  This is a structural OWL check, not a dataset
    # or score lookup, and it prevents ``child == parent`` responses from
    # poisoning every downstream mapping for an inherited table.
    for table_name, table_entry in result.items():
        _repair_invalid_sh_class_pair(
            table_name,
            table_entry,
            (candidates or {}).get(table_name, {}),
            ontology,
            enriched_schema=enriched_schema,
            alignment=result,
        )
    total = len(table_tasks)
    # 按流程约束：隐式关系挖掘仅在 OP 映射阶段执行，真实值增强仅使用显式 FK 上下文。
    implicit_relations = None
    sample_rows_cache: dict[str, list[dict]] = {}

    # Stage boundary: settle every low-confidence table Class before any
    # discriminator is refined.  Parent enum evidence may depend on a child SH
    # table that appears later in dictionary order (or several levels later in
    # an inheritance chain); per-table confirmation is therefore too late.
    # This pass changes only Class selections and reuses the same samples in
    # the value/column pass below, so it adds no duplicate database sampling.
    for class_idx, (table_name, low_info) in checkpoint.iter_phase(
        PHASE_CLASS,
        class_tasks,
        table_name=lambda item: item[0],
        result=lambda: result,
    ):
        table_entry = result.get(table_name, {})
        table_cands = candidates.get(table_name, {})
        pattern = table_entry.get("pattern", "SE")
        sample_rows = fetch_sample_rows(
            table_name,
            limit=REAL_VALUE_SAMPLE_ROWS_LIMIT,
            schema_name=DB_SCHEMA_NAME,
        )
        sample_rows_cache[table_name] = sample_rows
        if not sample_rows:
            continue
        print(
            f"\n[RealValue Class {class_idx}/{len(class_tasks)}] "
            f"{table_name} (Pattern: {pattern})"
        )
        _refine_table_class_if_needed(
            table_name=table_name,
            pattern=pattern,
            table_low=bool(low_info.get("table_low")),
            table_candidates=table_cands,
            sample_rows=sample_rows,
            result=result,
            force_all_context=force_all_context,
        )

    for idx, (table_name, low_info) in checkpoint.iter_phase(
        PHASE_TABLE,
        table_tasks,
        table_name=lambda item: item[0],
        result=lambda: result,
    ):
        table_entry = result.get(table_name, {})
        table_cands = candidates.get(table_name, {})
        pattern     = table_entry.get("pattern", "SE")
        table_low   = low_info.get("table_low", False)
        columns_low = low_info.get("columns_low", [])
        identity_dp_low = low_info.get("identity_dp_low", [])

        print(f"\n[RealValue {idx}/{total}] {table_name} (Pattern: {pattern})")

        # 拉真实数据
        if table_name in sample_rows_cache:
            sample_rows = sample_rows_cache[table_name]
        else:
            sample_rows = fetch_sample_rows(
                table_name,
                limit=REAL_VALUE_SAMPLE_ROWS_LIMIT,
                schema_name=DB_SCHEMA_NAME,
            )
        if not sample_rows:
            print(f"  空表，跳过真实值增强")
            continue
        print(f"  拉到 {len(sample_rows)} 行数据")

        current_class_uri = (
            result[table_name].get("sub_class_uri")
            if pattern == "SH"
            else result[table_name].get("class_uri")
        )

        type_assertions = result[table_name].get("type_assertions", [])
        if type_assertions:
            base_class_cands = (
                table_cands.get("sub_class_candidates", [])
                if pattern == "SH"
                else table_cands.get("table_class_candidates", [])
            )
            bool_hint_classes = [
                ta.get("true_class_uri")
                for ta in type_assertions
                if ta.get("kind") == "boolean" and ta.get("true_class_uri")
            ]
            bool_assertions = [
                {"column": ta.get("column"), "true_class_uri": ta.get("true_class_uri")}
                for ta in type_assertions
                if ta.get("kind") == "boolean" and ta.get("column") and ta.get("true_class_uri")
            ]
            class_ancestors = (ontology or {}).get("ancestors_of", {})
            class_subclass_of = (ontology or {}).get("subclass_of", {})
            for ta in type_assertions:
                type_col = ta.get("column")
                kind = ta.get("kind")
                if not type_col or kind not in ("enum", "boolean"):
                    continue

                class_cands = ta.get("class_candidates", [])
                if not class_cands:
                    class_cands, _ = _expand_enum_class_candidates(
                        current_class_uri=current_class_uri,
                        class_candidates=base_class_cands,
                        ontology=ontology,
                        return_diagnostics=True,
                    )
                    ta["class_candidates"] = class_cands

                if kind == "enum":
                    _, value_profiles, value_domain = _fetch_distinct_value_profiles(
                        table_name, type_col
                    )
                    if not value_profiles:
                        value_profiles = {
                            str(row.get(type_col)): [row]
                            for row in sample_rows if row.get(type_col) is not None
                        }
                        # A small fallback sample cannot prove a closed value domain.
                        value_domain["complete"] = False

                    expanded_class_cands, candidate_constraints = _expand_enum_class_candidates(
                        current_class_uri=current_class_uri,
                        class_candidates=class_cands,
                        ontology=ontology,
                        evidence_values=sorted(value_profiles.keys()),
                        return_diagnostics=True,
                    )
                    descendant_uris = _collect_descendants(
                        current_class_uri,
                        (ontology or {}).get("children_of", {}),
                    )
                    fk_context = _build_fk_semantic_context(
                        table_name=table_name,
                        enriched_schema=enriched_schema,
                        candidates=candidates,
                        implicit_relations=implicit_relations,
                        group_col=type_col,
                        group_values=sorted(value_profiles.keys()),
                    )
                    group_context = _build_type_group_context(
                        table_name=table_name,
                        type_col=type_col,
                        value_profiles=value_profiles,
                        enriched_schema=enriched_schema,
                        fk_context=fk_context,
                    )
                    sh_value_evidence = _build_sh_value_evidence(
                        table_name=table_name,
                        type_col=type_col,
                        enriched_schema=enriched_schema,
                        candidates=candidates,
                        alignment=result,
                        group_values=sorted(value_profiles.keys()),
                        ontology=ontology,
                    )
                    relation_domain_evidence = _build_relation_domain_class_evidence(
                        table_name=table_name,
                        type_col=type_col,
                        group_values=sorted(value_profiles.keys()),
                        class_candidates=expanded_class_cands,
                        current_class_uri=current_class_uri,
                        enriched_schema=enriched_schema,
                        alignment=result,
                        ontology=ontology,
                        value_domain_complete=bool(value_domain.get("complete")),
                    )
                    try:
                        re_map = _real_value_type_mapping(
                            table_name=table_name,
                            type_col=type_col,
                            value_profiles=value_profiles,
                            class_candidates=expanded_class_cands,
                            bool_hint_classes=bool_hint_classes,
                            bool_assertions=bool_assertions,
                            class_ancestors=class_ancestors,
                            class_subclass_of=class_subclass_of,
                            sh_value_evidence=sh_value_evidence,
                            relation_domain_evidence=relation_domain_evidence,
                            current_class_uri=current_class_uri,
                            current_class_confidence=result[table_name].get("class_confidence"),
                            descendant_uris=descendant_uris,
                            fk_context=fk_context,
                            group_context=group_context,
                            ontology=ontology,
                            candidate_constraints=candidate_constraints,
                            value_domain_complete=bool(value_domain.get("complete")),
                        )
                    except Exception as exc:
                        # Context enhancement is a refinement.  Preserve the
                        # preceding Class/DP decision if all local retries of
                        # this one discriminator request fail.
                        print(
                            f"  [WARN] TYPE 列 {type_col} 真实值增强失败: {exc}，保留原映射"
                        )
                        continue
                    ta["value_to_class"] = re_map.get("value_to_class", {})
                    ta["unmapped_values"] = re_map.get("unmapped_values", [])
                    ta["confidence"] = re_map.get("confidence", "medium")
                    ta["reason"] = re_map.get("reason", "")
                    ta["class_candidates"] = expanded_class_cands
                    print(
                        f"  TYPE 列 {type_col} 的值→Class 映射重判完成，"
                        f"映射了 {len(ta['value_to_class'])} 个值，未映射 {len(ta['unmapped_values'])} 个值"
                    )
                else:
                    # BOOL discriminator：基于样本+FK语义上下文重判 true_class_uri
                    _, bool_profiles, _ = _fetch_distinct_value_profiles(
                        table_name,
                        type_col,
                        per_value_limit=REAL_VALUE_ENUM_PER_VALUE_LIMIT,
                        max_values=REAL_VALUE_BOOL_MAX_VALUES,
                    )
                    if not bool_profiles:
                        bool_profiles = {"true": [], "false": []}
                        for row in sample_rows:
                            val = row.get(type_col)
                            if val is None:
                                continue
                            key = str(val).lower()
                            if key in {"t", "true", "1"}:
                                bool_profiles["true"].append(row)
                            elif key in {"f", "false", "0"}:
                                bool_profiles["false"].append(row)

                    fk_context = _build_fk_semantic_context(
                        table_name=table_name,
                        enriched_schema=enriched_schema,
                        candidates=candidates,
                        implicit_relations=implicit_relations,
                        group_col=type_col,
                        group_values=sorted(bool_profiles.keys()),
                    )
                    try:
                        re_map = _real_value_boolean_mapping(
                            table_name=table_name,
                            bool_col=type_col,
                            value_profiles=bool_profiles,
                            class_candidates=class_cands,
                            fk_context=fk_context,
                        )
                    except Exception as exc:
                        print(
                            f"  [WARN] BOOL 列 {type_col} 真实值增强失败: {exc}，保留原映射"
                        )
                        continue

                    ta["true_class_uri"] = re_map.get("true_class_uri")
                    ta["confidence"] = re_map.get("confidence", "medium")
                    ta["reason"] = re_map.get("reason", "布尔判别列映射为 true 时的语义类")
                    if ta.get("true_class_uri"):
                        bool_hint_classes.append(ta["true_class_uri"])
                    print(
                        f"  BOOL 列 {type_col} 语义类补全 -> {ta.get('true_class_uri')} [{ta.get('confidence')}]"
                    )

        # A literal enum can carry two orthogonal semantics: its ordinary
        # DatatypeProperty value and an additional row Class.  Candidate
        # generation historically discovered only columns named like TYPE;
        # profile low-cardinality data attributes as well, but issue a bounded
        # DISTINCT query only when the already loaded sample has a plausible
        # ontology-family lexical signal.
        if pattern in ("SE", "SH"):
            existing_assertion_columns = {
                ta.get("column")
                for ta in result[table_name].get("type_assertions", []) or []
                if ta.get("column")
            }
            base_class_cands = (
                table_cands.get("sub_class_candidates", [])
                if pattern == "SH"
                else table_cands.get("table_class_candidates", [])
            )
            current_class_confidence = result[table_name].get("class_confidence")
            for profile_col, profile_entry in (
                result[table_name].get("columns", {}) or {}
            ).items():
                if (
                    not isinstance(profile_entry, dict)
                    or profile_entry.get("role") != "data_attr"
                    or profile_col in existing_assertion_columns
                ):
                    continue

                sample_profiles: dict[str, list[dict]] = {}
                for row in sample_rows:
                    value = row.get(profile_col)
                    if value is None:
                        continue
                    sample_profiles.setdefault(str(value), []).append(row)

                _, precheck = _discover_data_attr_enum_type_assertion(
                    column_name=profile_col,
                    column_entry=profile_entry,
                    current_class_uri=current_class_uri,
                    current_class_confidence=current_class_confidence,
                    class_candidates=base_class_cands,
                    ontology=ontology,
                    value_profiles=sample_profiles,
                    value_domain={"complete": False, "source": "sample_precheck"},
                )
                if not precheck.get("lexical_matches"):
                    continue

                _, value_profiles, value_domain = _fetch_distinct_value_profiles(
                    table_name,
                    profile_col,
                    per_value_limit=REAL_VALUE_ENUM_PER_VALUE_LIMIT,
                    max_values=REAL_VALUE_ENUM_DISTINCT_MAX_FOR_CODE,
                )
                assertion, _discovery = _discover_data_attr_enum_type_assertion(
                    column_name=profile_col,
                    column_entry=profile_entry,
                    current_class_uri=current_class_uri,
                    current_class_confidence=current_class_confidence,
                    class_candidates=base_class_cands,
                    ontology=ontology,
                    value_profiles=value_profiles,
                    value_domain=value_domain,
                )
                if not assertion:
                    continue

                result[table_name].setdefault("type_assertions", []).append(assertion)
                existing_assertion_columns.add(profile_col)
                print(
                    f"  列 {profile_col} 保留 DatatypeProperty，并由数据画像追加 "
                    f"{len(assertion['value_to_class'])} 个高置信 rdf:type 映射"
                )

        # ── SR 表：重判两端 Class ──
        if pattern == "SR" and table_low:
            fk1 = table_entry.get("fk1", {})
            fk2 = table_entry.get("fk2", {})
            try:
                re_match = _real_value_sr_classes(table_name, fk1, fk2, sample_rows, table_entry)
                result[table_name]["domain_class_uri"] = re_match.get("domain_class_uri")
                result[table_name]["range_class_uri"]  = re_match.get("range_class_uri")
                result[table_name]["confidence"]       = re_match.get("confidence", "medium")
                print(f"  SR domain→{re_match.get('domain_class_uri')} range→{re_match.get('range_class_uri')} [{re_match.get('confidence')}]")
                print(f"  理由: {re_match.get('reason', '')}")
            except Exception as e:
                print(f"  [WARN] SR 真实值增强失败: {e}，保留原结果")
            continue

        if not columns_low and not identity_dp_low:
            continue

        # 列级重判
        # 预提取列值
        col_values_cache = {
            col: [row.get(col) for row in sample_rows if row.get(col) is not None]
            for col in dict.fromkeys([*columns_low, *identity_dp_low])
        }
        empty_sample_data_attrs = [
            col
            for col in columns_low
            if not col_values_cache.get(col)
            and (
                result[table_name].get("columns", {}).get(col, {}) or {}
            ).get("role") == "data_attr"
        ]
        non_null_counts = _fetch_non_null_counts(
            table_name,
            empty_sample_data_attrs,
        )

        table_fk_context = _build_fk_semantic_context(
            table_name=table_name,
            enriched_schema=enriched_schema,
            candidates=candidates,
            implicit_relations=implicit_relations,
        )

        # 取当前表的 class_uri 作为上下文
        current_class_uri = (
            result[table_name].get("sub_class_uri")
            if pattern == "SH"
            else result[table_name].get("class_uri")
        )

        for col_name in columns_low:
            col_entry       = result[table_name].get("columns", {}).get(col_name, {})
            col_cands_entry = table_cands.get("columns", {}).get(col_name, {})
            role            = col_entry.get("role", "data_attr")
            col_cands       = col_cands_entry.get("candidates", [])
            col_vals        = col_values_cache.get(col_name, [])

            # discriminator 列跳过
            if role == "discriminator":
                continue

            if not col_cands:
                print(f"  列 {col_name}: 无候选集，跳过")
                continue

            try:
                if role == "fk_obj":
                    # 只重判 range Class，不判 ObjectProperty
                    ref_table = col_entry.get("ref_table", "")
                    ref_class_cands = col_cands_entry.get("ref_class_candidates", [])
                    if not ref_class_cands:
                        # 从候选集的 range 字段尝试构造
                        ref_class_cands = [
                            {"uri": r, "local_name": r.split("#")[-1], "score": 0.5}
                            for c in col_cands for r in c.get("range", [])
                        ]
                    re_match = _real_value_fk_range_class(
                        table_name, col_name, ref_table, ref_class_cands, col_vals
                    )
                    allowed = {c.get("uri") for c in ref_class_cands if c.get("uri")}
                    new_uri = re_match.get("selected_uri")
                    if new_uri not in allowed:
                        new_uri = ref_class_cands[0].get("uri") if ref_class_cands else None
                    result[table_name]["columns"][col_name]["range_class_uri"] = new_uri
                    result[table_name]["columns"][col_name]["confidence"]      = re_match.get("confidence", "medium")
                    print(f"  列 {col_name} (fk_obj) range Class 重判 → {new_uri} [{re_match.get('confidence')}]")

                else:  # data_attr
                    col_type = col_cands_entry.get("column_type", "")
                    raw_col_cands = list(col_cands)
                    table_result = result[table_name]
                    sh_validation_status = str(
                        (table_result.get("sh_class_validation") or {}).get(
                            "status", ""
                        )
                    )
                    class_anchor_confirmed = bool(
                        table_result.get("class_confidence") == "high"
                        or (
                            pattern == "SH"
                            and (
                                sh_validation_status.startswith("valid")
                                or sh_validation_status.startswith("repaired")
                            )
                        )
                    )
                    domain_hints = []
                    if class_anchor_confirmed:
                        domain_hints.append(current_class_uri)
                        if pattern == "SH":
                            domain_hints.append(
                                table_result.get("parent_class_uri")
                            )
                    type_evidence = has_unique_datatype_range_evidence(
                        col_type,
                        raw_col_cands,
                        ontology,
                    )
                    col_cands = filter_semantically_admissible_datatype_candidates(
                        raw_col_cands,
                        column_name=col_name,
                        domain_hints=[hint for hint in domain_hints if hint],
                        sql_type=col_type,
                        ontology=ontology,
                    )
                    target = result[table_name]["columns"][col_name]
                    if not col_cands:
                        target["prop_uri"] = None
                        target["confidence"] = "low"
                        target["abstain_reason"] = (
                            "no_semantically_admissible_datatype_candidate"
                        )
                        print(
                            f"  列 {col_name} (data_attr) 无 domain/range/角色兼容候选，"
                            "保持 abstain"
                        )
                        continue

                    name_evidence = has_direct_datatype_name_evidence(
                        col_name,
                        col_cands,
                    )
                    # An all-NULL sample contains no instance semantics.  Do
                    # not turn a domain-weighted shortlist into a fabricated
                    # DP unless schema naming or a uniquely discriminating
                    # physical type supplies independent positive evidence.
                    if (
                        non_null_counts.get(col_name) == 0
                        and not name_evidence
                        and not type_evidence
                    ):
                        target["prop_uri"] = None
                        target["confidence"] = "low"
                        target["abstain_reason"] = (
                            "all_null_without_name_or_type_evidence"
                        )
                        print(
                            f"  列 {col_name} (data_attr) 样本全空且无名称/唯一类型证据，"
                            "保持 abstain"
                        )
                        continue

                    # 列名与候选属性名/常见 has_a_* 包装匹配时锁定，避免真实值增强用实例值误改 schema 语义。
                    locked_uri = None
                    if not force_all_context:
                        locked_uri = _find_schema_locked_dp(
                            col_name,
                            col_cands,
                            col_type,
                            ontology,
                        )
                    if locked_uri:
                        result[table_name]["columns"][col_name]["prop_uri"] = locked_uri
                        result[table_name]["columns"][col_name]["confidence"] = "high"
                        print(f"  列 {col_name} (data_attr) schema 名称锁定 → {locked_uri} [high]")
                        continue

                    re_match = _real_value_data_attr(
                        table_name,
                        col_name,
                        current_class_uri,
                        col_cands,
                        col_vals,
                        row_context=sample_rows,
                        fk_context=table_fk_context,
                    )
                    allowed = {c.get("uri") for c in col_cands if c.get("uri")}
                    new_uri = re_match.get("selected_uri")
                    old_uri = col_entry.get("prop_uri")
                    if new_uri is None:
                        target["prop_uri"] = None
                        target["confidence"] = "low"
                        target["abstain_reason"] = "provider_abstained"
                        print(
                            f"  列 {col_name} (data_attr) 提供者显式 abstain，"
                            "不使用 top-1/旧 URI 回填"
                        )
                        continue
                    if new_uri not in allowed:
                        target["prop_uri"] = None
                        target["confidence"] = "low"
                        target["abstain_reason"] = (
                            "provider_selected_outside_candidate_set"
                        )
                        print(
                            f"  列 {col_name} (data_attr) 提供者返回了候选集外 URI，"
                            "保持 abstain"
                        )
                        continue
                    if (
                        old_uri in allowed
                        and new_uri != old_uri
                        and not _sql_type_compatible_with_dp(col_type, ontology, new_uri)
                        and _sql_type_compatible_with_dp(col_type, ontology, old_uri)
                    ):
                        print(f"  列 {col_name} (data_attr) 真实值增强类型不兼容，保留原映射 → {old_uri}")
                        new_uri = old_uri
                    result[table_name]["columns"][col_name]["prop_uri"]   = new_uri
                    result[table_name]["columns"][col_name]["confidence"] = re_match.get("confidence", "medium")
                    print(f"  列 {col_name} (data_attr) 重判 → {new_uri} [{re_match.get('confidence')}]")

                print(f"  理由: {re_match.get('reason', '')}")

            except Exception as e:
                print(f"  [WARN] 列 {col_name} 真实值增强失败: {e}，保留原结果")

        # semantic identifier 是独立于 identity/FK 的 literal 轨道，但真实
        # 值或 LLM 不能凭空建立这条语义。这里只复核候选生成阶段留下的
        # 唯一 lexical+ontology 证据；其余 identity 列明确 abstain。
        for col_name in identity_dp_low:
            col_entry = result[table_name].get("columns", {}).get(col_name, {})
            col_cands_entry = table_cands.get("columns", {}).get(col_name, {})
            raw_candidates = (
                col_cands_entry.get("dp_candidates", [])
                or col_entry.get("dp_candidates", [])
                or (
                    col_cands_entry.get("candidates", [])
                    if col_cands_entry.get("role") == "pk"
                    else []
                )
            )
            dp_candidates = filter_strong_identity_datatype_candidates(
                raw_candidates
            )
            target = result[table_name]["columns"][col_name]
            if len(dp_candidates) != 1:
                target["data_prop_uri"] = None
                target["data_prop_confidence"] = "low"
                target["dp_candidates"] = []
                print(
                    f"  列 {col_name} (semantic identity) 缺少唯一独立语义证据，"
                    "保持 abstain"
                )
                continue
            selected = dp_candidates[0].get("uri")
            target["dp_candidates"] = dp_candidates
            target["data_prop_uri"] = selected
            target["data_prop_confidence"] = "high"
            print(
                f"  列 {col_name} (semantic identity) 结构证据锁定 → "
                f"{selected} [high]"
            )

    return result


# 主程序
if __name__ == "__main__":
    from utils.db_utils import read_schema
    from utils.ontology_utils import read_ontology
    from utils.merge_fks import merge_fks_into_schema, merge_llm_fks_into_schema
    from FKCompletion_agent import allocate_targets_and_shooters, discover_implicit_foreign_keys
    from classify_agent import classify_rule, classify_agent, find_difference, cal_num_fks, battle_layer
    from candidate_generation import generate_candidates
    try:
        from data_property_mapping_agent import run_data_property_mapping, collect_low_confidence_data_property_mappings
    except ModuleNotFoundError:
        from DPMapping.data_property_mapping_agent import run_data_property_mapping, collect_low_confidence_data_property_mappings

    from config import ONTOLOGY_PATH, OUTPUT_DIR   # ← 路径从 config 读取
    import os

    # 读取schema
    schema = read_schema()

    # FK补全（IND）
    allocation = allocate_targets_and_shooters(schema)
    discovered_fks = discover_implicit_foreign_keys(allocation)   # schema_name 从 config 自动读取
    enriched_schema = merge_fks_into_schema(schema, discovered_fks)

    # 分类（规则 + LLM battle）
    rule_result  = classify_rule(enriched_schema)
    agent_result = classify_agent(enriched_schema)
    fks_count    = cal_num_fks(enriched_schema)
    diff         = find_difference(rule_result, agent_result)
    pattern_result = battle_layer(diff, rule_result, agent_result, fks_count, enriched_schema)

    # ④ 把 LLM 推断的 FK 也合并进 schema（修复 SH parent_class_uri = null 的问题）
    enriched_schema = merge_llm_fks_into_schema(enriched_schema, agent_result)

    # ⑤ 候选生成
    ontology   = read_ontology(ONTOLOGY_PATH)
    candidates = generate_candidates(enriched_schema, pattern_result, ontology)

    # ⑥ LLM Matcher
    alignment   = run_data_property_mapping(candidates, ontology=ontology)
    low_conf    = collect_low_confidence_data_property_mappings(alignment)

    print(f"\n低置信条目数: {len(low_conf)} 张表")

    # ⑦ 真实值上下文增强（本文件）
    final_alignment = run_real_value_enhancement(
        alignment,
        low_conf,
        candidates,
        ontology=ontology,
        enriched_schema=enriched_schema,
    )

    # ⑧ 输出结果
    print("\n\n=== 最终 Alignment（真实值增强后）===")
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json.dump(final_alignment,
              open(os.path.join(OUTPUT_DIR, "final_alignment.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False, default=str)
    json.dump(enriched_schema,
              open(os.path.join(OUTPUT_DIR, "enriched_schema.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
