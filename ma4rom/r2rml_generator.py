"""
r2rml_generator.py  ——  R2RML 映射生成器（纯模板拼接，不用 LLM）

修复清单（相比 v1）：
  ✓ Bug1: rr:column 用普通双引号 "col"，不用 SQL 转义 \\"col\\"
  ✓ Bug2: _Inv 列的 objectMap IRI 模板指向正确的 range 表（从本体 OP domain 推断）
  ✓ Bug3: prop_uri 为 null/None/"null" 时正确跳过
  ✓ Bug4: 孤儿列 OP 的 objectMap 用正确的 range 表 IRI
  ✓ Bug5: 建立全局 class→table 反查映射，确保 IRI 模板一致
"""

import argparse
import json
import re
from pathlib import Path

from utils.ontology_utils import are_classes_disjoint

# ============================================================
#  工具函数
# ============================================================

def _local_name(uri: str) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rstrip("/").split("/")[-1]


def _get_namespace(uri: str) -> str:
    if "#" in uri:
        return uri.split("#")[0] + "#"
    return uri.rsplit("/", 1)[0] + "/"


def _is_valid_uri(uri) -> bool:
    """检查 URI 是否有效（非 null/None/空）"""
    if uri is None:
        return False
    if isinstance(uri, str) and uri.strip().lower() in ("null", "none", ""):
        return False
    return True


def _sql_col_in_query(col: str) -> str:
    """SQL 查询中的列名：需要 PostgreSQL 双引号转义"""
    return f'\\"{col}\\"'


def _sql_cols_in_query(cols: list) -> str:
    return " , ".join(_sql_col_in_query(c) for c in cols)


def _rr_column(col: str) -> str:
    """rr:column 的值：普通双引号，不转义"""
    return f'"{col}"'


def _iri_template(base_url: str, table_name: str, pk_col: str) -> str:
    table_lower = table_name.lower().replace(" ", "_")
    return f"{base_url}{table_lower}/{{{pk_col}}}"


def _predicate_str(uri: str, prefix_map: dict) -> str:
    """生成谓词字符串。含特殊字符用完整 IRI"""
    if not _is_valid_uri(uri):
        return None

    local = _local_name(uri)
    ns = _get_namespace(uri)

    if "-" in local or "." in local or " " in local:
        return f"<{uri}>"

    for prefix, namespace in prefix_map.items():
        if ns == namespace:
            return f"{prefix}:{local}"

    return f"<{uri}>"


def _xsd_type_from_sql(sql_type: str) -> str:
    """从 SQL 类型推断 XSD 类型（仅作 fallback）"""
    if not sql_type:
        return "xsd:string"
    t = sql_type.lower()
    if "int" in t:
        return "xsd:int"          # Ontop 严格校验，必须用 xsd:int 而非 xsd:integer
    elif t in ("date",):
        return "xsd:date"
    elif "timestamp" in t or "datetime" in t:
        return "xsd:dateTime"
    elif "float" in t or "double" in t or "numeric" in t or "decimal" in t:
        return "xsd:decimal"
    elif "bool" in t:
        return "xsd:boolean"
    else:
        return "xsd:string"


def build_dp_range_map(ontology: dict) -> dict:
    """
    构建 DatatypeProperty URI → XSD 类型 的映射。
    优先使用本体中声明的 range 类型，确保与 Ontop 校验一致。
    """
    dp_range = {}
    XSD_PREFIX_MAP = {
        "http://www.w3.org/2001/XMLSchema#string": "xsd:string",
        "http://www.w3.org/2001/XMLSchema#int": "xsd:int",
        "http://www.w3.org/2001/XMLSchema#integer": "xsd:integer",
        "http://www.w3.org/2001/XMLSchema#date": "xsd:date",
        "http://www.w3.org/2001/XMLSchema#dateTime": "xsd:dateTime",
        "http://www.w3.org/2001/XMLSchema#decimal": "xsd:decimal",
        "http://www.w3.org/2001/XMLSchema#boolean": "xsd:boolean",
        "http://www.w3.org/2001/XMLSchema#float": "xsd:float",
        "http://www.w3.org/2001/XMLSchema#double": "xsd:double",
        "http://www.w3.org/2001/XMLSchema#nonNegativeInteger": "xsd:nonNegativeInteger",
        "http://www.w3.org/2001/XMLSchema#unsignedLong": "xsd:unsignedLong",
        "http://www.w3.org/2001/XMLSchema#unsignedInt": "xsd:unsignedInt",
        "http://www.w3.org/2001/XMLSchema#anyURI": "xsd:anyURI",
    }
    for dp_uri, dp_info in ontology.get("datatype_properties", {}).items():
        ranges = dp_info.get("range", [])
        for r in ranges:
            if r in XSD_PREFIX_MAP:
                dp_range[dp_uri] = XSD_PREFIX_MAP[r]
                break
    return dp_range


def _xsd_type(sql_type: str, prop_uri: str = None, dp_range_map: dict = None) -> str:
    """
    确定 XSD 类型。优先查本体声明的 range，查不到再从 SQL 类型推断。
    这样能避免 Ontop 的 MappingOntologyMismatchException。
    """
    # 优先：本体声明的 range 类型
    if prop_uri and dp_range_map and prop_uri in dp_range_map:
        return dp_range_map[prop_uri]
    # Fallback：从 SQL 类型推断
    return _xsd_type_from_sql(sql_type)


def _is_sql_xsd_compatible(sql_type: str, xsd_type: str) -> bool:
    """
    避免明显不兼容的类型映射（如字符串列 -> xsd:nonNegativeInteger）。
    仅做保守过滤，不做复杂推断。
    """
    st = (sql_type or "").lower()
    xt = (xsd_type or "").lower()

    is_num_sql = any(k in st for k in ("int", "numeric", "decimal", "real", "double", "float"))
    is_bool_sql = "bool" in st
    is_date_sql = "date" in st or "time" in st

    is_num_xsd = any(
        k in xt for k in (
            "xsd:int", "xsd:integer", "xsd:decimal", "xsd:float", "xsd:double",
            "xsd:nonnegativeinteger", "xsd:unsignedlong", "xsd:unsignedint",
        )
    )
    is_bool_xsd = "xsd:boolean" in xt
    is_date_xsd = "xsd:date" in xt or "xsd:datetime" in xt

    if is_num_xsd and not is_num_sql:
        return False
    if is_bool_xsd and not is_bool_sql:
        return False
    if is_date_xsd and not is_date_sql:
        return False
    return True


def _canonical_iri_base_name(table_name: str) -> str:
    """
    将物理表名规范化为资源路径基名（通用规则，无数据集硬编码）。
    """
    t = (table_name or "").strip().lower().replace(" ", "_")
    if not t:
        return "resource"
    # 避免连续下划线/首尾下划线
    t = "_".join([seg for seg in t.split("_") if seg])
    return t or "resource"


# ============================================================
#  全局映射表：class_uri → table_name（用于 IRI 模板一致性）
# ============================================================

def build_class_to_table_map(final_alignment: dict) -> dict:
    """
    构建 class_uri → table_name 映射。
    用于：当我们知道某个 OP 的 range 是 conference#Committee，
    就能查到对应表是 Committee，从而构造 IRI template = committee/{ID}
    """
    c2t = {}
    for table_name, entry in final_alignment.items():
        pattern = entry.get("pattern", "SE")
        if pattern == "SH":
            cls = entry.get("sub_class_uri")
            parent_cls = entry.get("parent_class_uri")
            if cls:
                c2t[cls] = table_name
            # 不覆盖父类已有的映射
            if parent_cls and parent_cls not in c2t:
                c2t[parent_cls] = table_name
        elif pattern == "SR":
            continue
        else:
            cls = entry.get("class_uri")
            if cls:
                c2t[cls] = table_name
    return c2t


def _fk_reference_table(fk: dict) -> str:
    """Return the referenced physical table from schema variants."""
    return str(
        fk.get("ref_table")
        or fk.get("references_table")
        or fk.get("target_table")
        or ""
    )


def _fk_reference_column(fk: dict) -> str:
    """Return the referenced physical column from schema variants."""
    return str(
        fk.get("ref_col")
        or fk.get("references_column")
        or fk.get("target_column")
        or ""
    )


def _entry_represents_class(entry: dict, class_uri: str | None) -> bool:
    """Whether an aligned table is the exact physical representation of a Class."""
    if not class_uri:
        return False
    entry = entry or {}
    if entry.get("pattern") == "SH":
        return entry.get("sub_class_uri") == class_uri
    return entry.get("pattern", "SE") == "SE" and entry.get("class_uri") == class_uri


def _physical_sh_parent_tables(
    table_name: str,
    entry: dict,
    enriched_schema: dict,
) -> list[str]:
    """Infer unique parent tables from an inherited PK/FK constraint.

    This is deliberately independent of ontology names.  It is used as a
    structural tie-breaker when multiple aligned tables expose the same Class
    URI (a common consequence of noisy SH matching).
    """
    info = (enriched_schema or {}).get(table_name, {}) or {}
    child_pk = set(info.get("primary_key", []) or [])
    inherited = {
        col
        for col, col_info in (entry or {}).get("columns", {}).items()
        if isinstance(col_info, dict) and col_info.get("role") == "sh_inherited_pk"
    }
    if not inherited or not inherited.issubset(child_pk):
        return []

    groups: dict[tuple, dict] = {}
    for index, fk in enumerate(info.get("foreign_keys", []) or []):
        if not isinstance(fk, dict) or not fk.get("column"):
            continue
        ref_table = _fk_reference_table(fk)
        ref_col = _fk_reference_column(fk)
        if not ref_table:
            continue
        constraint = fk.get("constraint_name")
        try:
            arity = int(fk.get("fk_arity") or 1)
        except (TypeError, ValueError):
            arity = 1
        if constraint:
            key = ("constraint", str(constraint), ref_table)
        elif arity > 1:
            key = ("anonymous-composite", ref_table, arity)
        else:
            key = ("anonymous-single", index)
        group = groups.setdefault(
            key,
            {"ref_table": ref_table, "source": set(), "target": set()},
        )
        group["source"].add(str(fk.get("column")))
        if ref_col:
            group["target"].add(ref_col)

    parents = set()
    for group in groups.values():
        if group["source"] != inherited:
            continue
        parent_info = (enriched_schema or {}).get(group["ref_table"], {}) or {}
        parent_pk = set(parent_info.get("primary_key", []) or [])
        if parent_pk and group["target"] == parent_pk:
            parents.add(group["ref_table"])
    return sorted(parents)


def _semantic_sh_parent_candidates(
    table_name: str,
    entry: dict,
    final_alignment: dict,
    enriched_schema: dict,
) -> list[str]:
    """Return an unambiguous parent representation for a SH entry.

    ``parent_table`` is a useful upstream hint, but it is not trusted unless
    its aligned Class equals ``parent_class_uri``.  Multiple tables can carry
    the same Class in a noisy alignment; collapsing an identity through an
    arbitrary one of them would be less safe than retaining the full PK.
    """
    parent_class_uri = (entry or {}).get("parent_class_uri")
    if not parent_class_uri:
        return []
    candidates = {
        candidate_table
        for candidate_table, candidate_entry in (final_alignment or {}).items()
        if candidate_table != table_name
        and candidate_table in (enriched_schema or {})
        and _entry_represents_class(candidate_entry or {}, parent_class_uri)
    }
    declared_parent_table = (entry or {}).get("parent_table")
    if declared_parent_table:
        # The physical FK parent is a stronger witness than the fact that
        # several aligned tables happen to share a Class URI.  Restrict the
        # candidate set to that declared table when it is semantically aligned;
        # otherwise abstain rather than selecting an arbitrary same-class
        # table.  The FK/PK contract is still checked by the caller below.
        if declared_parent_table not in candidates:
            return []
        return [declared_parent_table]
    physical_parents = _physical_sh_parent_tables(
        table_name, entry, enriched_schema
    )
    if len(physical_parents) == 1:
        # The FK witness is sufficient to disambiguate same-class tables, but
        # still require that the referenced table carries the chosen class.
        return [physical_parents[0]] if physical_parents[0] in candidates else []
    return sorted(candidates)


def _sh_subclass_relation_is_verified(entry: dict, ontology: dict | None) -> bool:
    """Check the semantic half of a SH inheritance contract when available."""
    child_class_uri = (entry or {}).get("sub_class_uri")
    parent_class_uri = (entry or {}).get("parent_class_uri")
    if not child_class_uri or not parent_class_uri:
        return False
    if child_class_uri == parent_class_uri:
        return True
    if ontology is None:
        return True
    ancestors_of = (ontology or {}).get("ancestors_of", {}) or {}
    if parent_class_uri in (ancestors_of.get(child_class_uri, []) or []):
        return True
    pending = list((ontology or {}).get("subclass_of", {}).get(child_class_uri, []) or [])
    visited: set[str] = set()
    while pending:
        candidate = pending.pop()
        if candidate in visited:
            continue
        visited.add(candidate)
        if candidate == parent_class_uri:
            return True
        pending.extend((ontology or {}).get("subclass_of", {}).get(candidate, []) or [])
    return False


def build_entity_identity_contracts(
    final_alignment: dict,
    enriched_schema: dict,
    ontology: dict | None = None,
) -> dict[str, dict]:
    """Build one recursive, physical-FK-validated identity contract per table."""
    alignment = final_alignment or {}
    schema = enriched_schema or {}
    resolved: dict[str, dict] = {}
    resolving: set[str] = set()

    def fallback(table_name: str) -> dict:
        return {
            "root_table": table_name,
            "identity_columns": _find_row_id_cols(schema, table_name),
            "parent_table": None,
            "validated_inheritance": False,
        }

    def resolve(table_name: str) -> dict | None:
        if table_name in resolved:
            return resolved[table_name]
        if table_name in resolving:
            return None

        entry = alignment.get(table_name, {}) or {}
        if entry.get("pattern", "SE") != "SH":
            contract = fallback(table_name)
            resolved[table_name] = contract
            return contract

        resolving.add(table_name)
        try:
            semantic_verified = _sh_subclass_relation_is_verified(entry, ontology)
            child_class_uri = (entry or {}).get("sub_class_uri")
            parent_class_uri = (entry or {}).get("parent_class_uri")
            # A relational inherited-PK contract is authoritative for the
            # subject identity even when an ontology has no explicit
            # subClassOf edge (or the matcher selected the parent's class for
            # both sides).  Only an explicit OWL disjointness contradiction
            # blocks identity collapse; a merely missing/non-direct edge must
            # not send the child to an unrelated ancestor.
            if (
                child_class_uri
                and parent_class_uri
                and are_classes_disjoint(child_class_uri, parent_class_uri, ontology)
            ):
                contract = fallback(table_name)
                resolved[table_name] = contract
                return contract
            parent_candidates = _semantic_sh_parent_candidates(
                table_name, entry, alignment, schema
            )
            if len(parent_candidates) != 1:
                contract = fallback(table_name)
                resolved[table_name] = contract
                return contract
            parent_table = parent_candidates[0]
            parent_contract = resolve(parent_table)
            if not parent_contract or not parent_contract.get("identity_columns"):
                contract = fallback(table_name)
                resolved[table_name] = contract
                return contract
            if (
                (alignment.get(parent_table, {}) or {}).get("pattern") == "SH"
                and not parent_contract.get("validated_inheritance")
            ):
                contract = fallback(table_name)
                resolved[table_name] = contract
                return contract

            child_info = schema.get(table_name, {}) or {}
            child_columns = child_info.get("columns", {}) or {}
            child_pk = [
                column for column in child_info.get("primary_key", []) or []
                if column in child_columns
            ]
            inherited_markers = {
                column
                for column, column_info in (entry.get("columns", {}) or {}).items()
                if isinstance(column_info, dict)
                and column_info.get("role") == "sh_inherited_pk"
            }
            if (
                not child_pk
                or not inherited_markers
                or not inherited_markers.issubset(set(child_pk))
            ):
                contract = fallback(table_name)
                resolved[table_name] = contract
                return contract

            parent_to_child: dict[str, str] = {}
            ambiguous_parent_columns: set[str] = set()
            for fk in child_info.get("foreign_keys", []) or []:
                child_column = fk.get("column")
                parent_column = _fk_reference_column(fk)
                if (
                    child_column not in inherited_markers
                    or _fk_reference_table(fk) != parent_table
                    or parent_column not in parent_contract["identity_columns"]
                ):
                    continue
                existing = parent_to_child.get(parent_column)
                if existing is not None and existing != child_column:
                    ambiguous_parent_columns.add(parent_column)
                    continue
                parent_to_child[parent_column] = child_column

            parent_identity_columns = parent_contract["identity_columns"]
            if (
                ambiguous_parent_columns
                or set(parent_to_child) != set(parent_identity_columns)
            ):
                contract = fallback(table_name)
                resolved[table_name] = contract
                return contract
            child_identity_columns = [
                parent_to_child[parent_column]
                for parent_column in parent_identity_columns
            ]
            if set(child_identity_columns) != inherited_markers:
                contract = fallback(table_name)
                resolved[table_name] = contract
                return contract

            contract = {
                "root_table": parent_contract["root_table"],
                "identity_columns": child_identity_columns,
                "parent_table": parent_table,
                "validated_inheritance": True,
                "semantic_relation_verified": semantic_verified,
            }
            resolved[table_name] = contract
            return contract
        finally:
            resolving.discard(table_name)

    for table_name, entry in alignment.items():
        if (entry or {}).get("pattern") != "SR":
            resolve(table_name)
    return resolved


def _complete_physical_fk_group(rows: list[dict]) -> dict | None:
    """Normalize one complete physical FK constraint without semantic guesses.

    Schema metadata is stored one row per join member.  A caller may suppress
    or serialize a relation only after every declared member is present and
    all members point to one target table.  Anonymous composite constraints
    are deliberately rejected because their rows cannot be grouped safely.
    """
    members = [row for row in (rows or []) if isinstance(row, dict)]
    if not members or len(members) != len(rows or []):
        return None

    constraint_names = {
        str(row.get("constraint_name") or "").strip()
        for row in members
        if str(row.get("constraint_name") or "").strip()
    }
    if len(constraint_names) > 1:
        return None
    constraint_name = next(iter(constraint_names), "")

    declared_arities: set[int] = set()
    for row in members:
        try:
            arity = int(row.get("fk_arity") or 1)
        except (TypeError, ValueError):
            return None
        if arity <= 0:
            return None
        declared_arities.add(arity)
    if len(declared_arities) != 1:
        return None
    expected_arity = next(iter(declared_arities))
    if expected_arity != len(members):
        return None
    if expected_arity > 1 and not constraint_name:
        return None

    source_columns = [str(row.get("column") or "") for row in members]
    target_tables = [_fk_reference_table(row) for row in members]
    target_columns = [_fk_reference_column(row) for row in members]
    if (
        not all(source_columns)
        or len(source_columns) != len(set(source_columns))
        or not all(target_tables)
        or len(set(target_tables)) != 1
        or not all(target_columns)
        or len(target_columns) != len(set(target_columns))
    ):
        return None

    return {
        "constraint_name": constraint_name,
        "source_columns": source_columns,
        "target_table": target_tables[0],
        "target_columns": target_columns,
        "join_pairs": list(zip(source_columns, target_columns)),
        "is_composite": expected_arity > 1,
    }


def _physical_fk_groups(table_info: dict) -> list[list[dict]]:
    """Group FK metadata by constraint while keeping scalars independent."""
    named: dict[str, list[dict]] = {}
    scalar: list[list[dict]] = []
    for row in (table_info or {}).get("foreign_keys", []) or []:
        if not isinstance(row, dict):
            continue
        constraint_name = str(row.get("constraint_name") or "").strip()
        try:
            arity = int(row.get("fk_arity") or 1)
        except (TypeError, ValueError):
            arity = -1
        if constraint_name:
            named.setdefault(constraint_name, []).append(row)
        elif arity == 1:
            scalar.append([row])
        # Anonymous composite rows are ambiguous and intentionally omitted.
    return [*named.values(), *scalar]


def build_sh_identity_fk_consumption_contracts(
    final_alignment: dict,
    enriched_schema: dict,
    ontology: dict | None = None,
    identity_contracts: dict[str, dict] | None = None,
) -> dict[str, list[dict]]:
    """Identify FK constraints already consumed by validated SH identity.

    A SH row and every validated ancestor row denote the same RDF subject.
    An FK whose *complete* source key is the inherited identity and whose
    target is one of those ancestor identities is therefore an identity join,
    not an ObjectProperty assertion.  The check is structural and ontology
    driven; no dataset, table spelling, query, or expected score is involved.
    """
    contracts = identity_contracts or build_entity_identity_contracts(
        final_alignment, enriched_schema, ontology
    )
    consumed: dict[str, list[dict]] = {}

    for table_name, entry in (final_alignment or {}).items():
        if (entry or {}).get("pattern") != "SH":
            continue
        table_contract = (contracts or {}).get(table_name) or {}
        if not table_contract.get("validated_inheritance"):
            continue

        inherited_columns = {
            str(column)
            for column, column_info in ((entry or {}).get("columns", {}) or {}).items()
            if isinstance(column_info, dict)
            and column_info.get("role") == "sh_inherited_pk"
        }
        if (
            not inherited_columns
            or inherited_columns
            != set(table_contract.get("identity_columns") or [])
        ):
            continue

        # Follow only validated parent contracts.  This includes redundant
        # direct root FKs (child -> root) as well as the immediate parent FK.
        identity_ancestors: set[str] = set()
        cursor = table_contract.get("parent_table")
        seen: set[str] = set()
        while cursor and cursor not in seen:
            seen.add(cursor)
            identity_ancestors.add(cursor)
            parent_entry = (final_alignment or {}).get(cursor, {}) or {}
            parent_contract = (contracts or {}).get(cursor) or {}
            if parent_entry.get("pattern") == "SH" and not parent_contract.get(
                "validated_inheritance"
            ):
                break
            cursor = parent_contract.get("parent_table")

        if not identity_ancestors:
            continue

        for rows in _physical_fk_groups(
            (enriched_schema or {}).get(table_name, {}) or {}
        ):
            physical = _complete_physical_fk_group(rows)
            if not physical:
                continue
            target_table = physical["target_table"]
            target_identity = set(
                ((contracts or {}).get(target_table) or {}).get(
                    "identity_columns", []
                )
                or _find_row_id_cols(enriched_schema, target_table)
            )
            if (
                target_table not in identity_ancestors
                or set(physical["source_columns"]) != inherited_columns
                or not target_identity
                or set(physical["target_columns"]) != target_identity
            ):
                continue
            consumed.setdefault(table_name, []).append(physical)

    return consumed


def _fk_contract_matches(left: dict, right: dict) -> bool:
    """Compare normalized FK contracts independent of metadata row order."""
    left_constraint = str((left or {}).get("constraint_name") or "")
    right_constraint = str((right or {}).get("constraint_name") or "")
    if left_constraint and right_constraint and left_constraint != right_constraint:
        return False
    left_target = (left or {}).get("target_table") or (left or {}).get("ref_table")
    right_target = (right or {}).get("target_table") or (right or {}).get("ref_table")
    left_targets = (left or {}).get("target_columns") or [
        parent for _child, parent in ((left or {}).get("join_pairs") or [])
    ]
    right_targets = (right or {}).get("target_columns") or [
        parent for _child, parent in ((right or {}).get("join_pairs") or [])
    ]
    return bool(
        left_target == right_target
        and set((left or {}).get("source_columns") or [])
        == set((right or {}).get("source_columns") or [])
        and set(left_targets) == set(right_targets)
    )


def build_table_to_iri_base(
    final_alignment: dict,
    enriched_schema: dict,
    identity_contracts: dict[str, dict] | None = None,
    ontology: dict | None = None,
) -> dict:
    """Map each entity table to the root chosen by its identity contract."""
    contracts = identity_contracts or build_entity_identity_contracts(
        final_alignment, enriched_schema, ontology
    )
    return {
        table_name: _canonical_iri_base_name(
            (contracts.get(table_name) or {}).get("root_table", table_name)
        )
        for table_name, entry in (final_alignment or {}).items()
        if (entry or {}).get("pattern", "SE") != "SR"
    }


def resolve_range_table(
    op_uri: str,
    ontology: dict,
    class_to_table: dict,
    direction: str = "normal",
    table_to_iri: dict | None = None,
) -> str:
    """
    从本体 OP 的 domain/range 声明，推断 FK 指向的表名。
    - normal 方向：range 端是目标表
    - inverse 方向：domain 端是目标表（因为反向了）
    返回 IRI base 名（优先继承链归一化后的 table_to_iri），找不到返回 None
    """
    op_info = ontology.get("object_properties", {}).get(op_uri, {})
    if not op_info:
        return None

    if direction == "inverse":
        targets = op_info.get("domain", [])
    else:
        targets = op_info.get("range", [])

    for target_cls in targets:
        if target_cls in class_to_table:
            tbl = class_to_table[target_cls]
            return (table_to_iri or {}).get(tbl, _canonical_iri_base_name(tbl))
        # 尝试本地名匹配
        target_local = _local_name(target_cls).lower()
        for cls_uri, tbl in class_to_table.items():
            if _local_name(cls_uri).lower() == target_local:
                return (table_to_iri or {}).get(tbl, _canonical_iri_base_name(tbl))
    return None


def _find_row_id_cols(enriched_schema, table_name) -> list[str]:
    """
    选择可用于 subject IRI 的标识列（按优先级）：
    1) 主键列（可复合）
    2) 外键列（去重后，按表字段顺序）
    3) 常见标识列（id/code/name/number）
    4) 失败则返回空（调用方决定 skip，避免生成错误 URI）
    """
    table_info = enriched_schema.get(table_name, {}) or {}
    cols_dict = table_info.get("columns", {}) or {}
    all_cols = list(cols_dict.keys())

    pks = table_info.get("primary_key", []) or []
    if pks:
        return [c for c in pks if c in cols_dict]

    fk_cols = []
    seen = set()
    for fk in table_info.get("foreign_keys", []) or []:
        c = fk.get("column")
        if not c or c in seen or c not in cols_dict:
            continue
        seen.add(c)
        fk_cols.append(c)
    if fk_cols:
        order = {c: i for i, c in enumerate(all_cols)}
        fk_cols.sort(key=lambda c: order.get(c, 10**9))
        return fk_cols

    # 最后兜底：只选“像标识符”的列，不再退化为全表列拼接。
    # 使用 token/后缀边界，避免原先 substring 规则将任意含 ``id``/``no``
    # 的普通单词误当成标识列。
    id_like_cols = []
    for c in all_cols:
        tokenized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(c))
        tokens = {
            token.lower()
            for token in re.split(r"[^A-Za-z0-9]+", tokenized)
            if token
        }
        compact = re.sub(r"[^a-z0-9]", "", str(c).lower())
        has_identifier_token = bool(tokens & {"id", "code", "name", "number", "no"})
        has_identifier_suffix = any(
            compact.endswith(suffix) and len(compact) > len(suffix)
            for suffix in ("id", "code", "name", "number")
        )
        if has_identifier_token or has_identifier_suffix:
            id_like_cols.append(c)
    if id_like_cols:
        return id_like_cols
    return []


def _validated_subject_identity_cols(
    table_name: str,
    entry: dict,
    enriched_schema: dict,
    final_alignment: dict | None = None,
    identity_contracts: dict[str, dict] | None = None,
) -> list[str]:
    """Return the stable subject key for an aligned table.

    A validated SH child inherits the canonical identity of its parent entity;
    extra child PK components remain relationship attributes.  The main map
    and discriminator maps share this resolver so their IRI templates cannot
    drift apart.
    """
    contracts = identity_contracts
    if contracts is None and final_alignment:
        contracts = build_entity_identity_contracts(final_alignment, enriched_schema)
    identity_columns = ((contracts or {}).get(table_name) or {}).get(
        "identity_columns"
    ) or []
    return list(identity_columns) or _find_row_id_cols(enriched_schema, table_name)


def _make_subject_template(base_url: str, iri_base: str, id_cols: list[str]) -> str | None:
    if not id_cols:
        return None
    if len(id_cols) == 1:
        return f"{base_url}{iri_base}/{{{id_cols[0]}}}"
    joined = "__".join(f"{{{c}}}" for c in id_cols)
    return f"{base_url}{iri_base}/{joined}"


def _make_subject_template_for_table(
    base_url: str, table_name: str, iri_base: str, id_cols: list[str], cols: list[str]
) -> str | None:
    """
    统一走通用模板（不做数据集硬编码）。
    URI 语义由前序 DP Mapping/OP Mapping 输出与表主键结构共同决定。
    """
    return _make_subject_template(base_url, iri_base, id_cols)


def _find_fk_ref_table(enriched_schema, table_name, col_name) -> str:
    """从 enriched_schema 的 foreign_keys 查找 FK 列引用的目标表"""
    fks = enriched_schema.get(table_name, {}).get("foreign_keys", [])
    for fk in fks:
        if fk.get("column") == col_name:
            return fk.get("ref_table") or fk.get("references_table") or ""
    return ""


def _get_op_mapping_entry(op_mapping_step1: dict, key: str) -> dict:
    entry = (op_mapping_step1 or {}).get(key)
    if entry:
        return entry
    key_lower = key.lower()
    for existing_key, existing_entry in (op_mapping_step1 or {}).items():
        if str(existing_key).lower() == key_lower:
            return existing_entry or {}
    return {}


def _infer_fk_object_property_uri(
    table_name: str,
    col_name: str,
    domain_class_uri: str | None,
    ref_table: str,
    ontology: dict,
    class_to_table: dict,
    op_mapping_step1: dict,
) -> str | None:
    """Return exactly the ObjectProperty selected by upstream OP Step1."""
    op_mapping_key = f"{table_name}.{col_name}"
    op_mapping_info = _get_op_mapping_entry(op_mapping_step1, op_mapping_key)
    op_uri = (op_mapping_info or {}).get("object_prop_uri")
    return op_uri if _is_valid_uri(op_uri) else None


def _refine_one_sided_op_mapping_op(op_mapping_info: dict) -> str | None:
    """Backward-compatible accessor; output stage never rewrites Step1."""
    op_uri = (op_mapping_info or {}).get("object_prop_uri")
    return op_uri if _is_valid_uri(op_uri) else None


def _allow_orphan_object_pom(orphan_info: dict, table_name: str, col_name: str, enriched_schema: dict) -> bool:
    """
    控制 Step2 orphan -> ObjectProperty 的落盘条件，避免把普通数据列误映射成对象关系。
    仅在以下情形允许：
    1) orphan 明确带有反向关系信号（is_inv=True）
    2) 或该列在 schema 中确实是 FK 列（结构关系信号）
    """
    if (orphan_info or {}).get("is_inv") is True:
        return True
    return bool(_find_fk_ref_table(enriched_schema, table_name, col_name))


def _collect_columns(table_name, enriched_schema):
    return list(enriched_schema.get(table_name, {}).get("columns", {}).keys())


def _find_class_uri_by_table(class_to_table: dict, table_name: str) -> str | None:
    """从 class->table 反查某表对应的 class URI。"""
    for cls_uri, tbl in (class_to_table or {}).items():
        if tbl == table_name:
            return cls_uri
    return None


# ============================================================
#  POM（PredicateObjectMap）生成辅助
# ============================================================

def _make_pom_datatype(pred_str: str, col_name: str, xsd_type: str) -> list:
    """生成数据属性的 predicateObjectMap"""
    return [
        f'    rr:predicateObjectMap [',
        f'        rr:predicate {pred_str} ;',
        f'        rr:objectMap [ rr:column {_rr_column(col_name)} ; rr:datatype {xsd_type} ]',
        f'    ]',
    ]


def _make_pom_object(pred_str: str, obj_template: str) -> list:
    """生成对象属性的 predicateObjectMap"""
    return [
        f'    rr:predicateObjectMap [',
        f'        rr:predicate {pred_str} ;',
        f'        rr:objectMap [ rr:template "{obj_template}" ]',
        f'    ]',
    ]


def _make_pom_parent_join(
    pred_str: str,
    parent_table: str,
    join_pairs: list[tuple[str, str]],
) -> list:
    """Generate one standard R2RML object map for a complete FK constraint."""
    lines = [
        "    rr:predicateObjectMap [",
        f"        rr:predicate {pred_str} ;",
        "        rr:objectMap [",
        f"            rr:parentTriplesMap <#{parent_table}Mapping> ;",
    ]
    for index, (child_column, parent_column) in enumerate(join_pairs):
        suffix = " ;" if index < len(join_pairs) - 1 else ""
        lines.append(
            "            rr:joinCondition [ "
            f"rr:child {_rr_column(child_column)} ; "
            f"rr:parent {_rr_column(parent_column)} ]{suffix}"
        )
    lines.extend(["        ]", "    ]"])
    return lines


def _constraint_relation_key(table: str, constraint_name: str) -> str:
    return f"{table}.__fk__{constraint_name}"


def _composite_fk_contracts(
    table_name: str,
    enriched_schema: dict,
    identity_contracts: dict[str, dict] | None = None,
) -> tuple[list[dict], set[str]]:
    """Validate composite FK groups and order joins by target identity.

    Every source column declared as part of a composite FK is returned in the
    blocked set, even when the group is invalid.  This prevents a malformed or
    incomplete constraint from falling through to scalar object templates.
    """
    table_info = (enriched_schema or {}).get(table_name, {}) or {}
    rows = [
        fk for fk in (table_info.get("foreign_keys", []) or [])
        if isinstance(fk, dict)
    ]
    groups: dict[str, list[dict]] = {}
    blocked_columns: set[str] = set()
    for fk in rows:
        constraint_name = str(fk.get("constraint_name") or "").strip()
        try:
            arity = int(fk.get("fk_arity") or 1)
        except (TypeError, ValueError):
            arity = 1
        if constraint_name:
            groups.setdefault(constraint_name, []).append(fk)
        elif arity > 1 and fk.get("column"):
            blocked_columns.add(str(fk["column"]))

    contracts = []
    for constraint_name, members in groups.items():
        declared_arities = set()
        for member in members:
            if member.get("fk_arity") is None:
                continue
            try:
                declared_arities.add(int(member.get("fk_arity")))
            except (TypeError, ValueError):
                declared_arities.add(-1)
        expected_arity = (
            next(iter(declared_arities))
            if len(declared_arities) == 1
            else len(members)
        )
        is_composite = expected_arity > 1 or len(members) > 1
        if not is_composite:
            continue

        source_columns = [str(member.get("column") or "") for member in members]
        blocked_columns.update(column for column in source_columns if column)
        ref_tables = {
            _fk_reference_table(member) for member in members
            if _fk_reference_table(member)
        }
        ref_columns = [
            str(_fk_reference_column(member) or "") for member in members
        ]
        ref_table = next(iter(ref_tables)) if len(ref_tables) == 1 else ""
        target_identity = list(
            ((identity_contracts or {}).get(ref_table, {}) or {}).get(
                "identity_columns", []
            )
            or _find_row_id_cols(enriched_schema, ref_table)
        )
        valid = bool(
            len(declared_arities) <= 1
            and expected_arity == len(members)
            and all(source_columns)
            and len(source_columns) == len(set(source_columns))
            and all(ref_columns)
            and len(ref_columns) == len(set(ref_columns))
            and ref_table
            and target_identity
            and set(ref_columns) == set(target_identity)
        )
        if not valid:
            continue

        member_by_parent = {
            str(_fk_reference_column(member)): member for member in members
        }
        ordered_members = [member_by_parent[column] for column in target_identity]
        contracts.append(
            {
                "constraint_name": constraint_name,
                "ref_table": ref_table,
                "source_columns": [
                    str(member.get("column")) for member in ordered_members
                ],
                "target_columns": [
                    str(_fk_reference_column(member))
                    for member in ordered_members
                ],
                "join_pairs": [
                    (
                        str(member.get("column")),
                        str(_fk_reference_column(member)),
                    )
                    for member in ordered_members
                ],
            }
        )
    return contracts, blocked_columns


def _selected_composite_fk_uri(
    table_name: str,
    contract: dict,
    op_mapping_step1: dict,
) -> str | None:
    """Read one explicit constraint decision, with safe legacy consolidation."""
    constraint_key = _constraint_relation_key(
        table_name, contract.get("constraint_name", "")
    )
    constraint_entry = _get_op_mapping_entry(op_mapping_step1, constraint_key)
    constraint_uri = (constraint_entry or {}).get("object_prop_uri")
    if _is_valid_uri(constraint_uri):
        return constraint_uri

    legacy_uris = {
        (_get_op_mapping_entry(op_mapping_step1, f"{table_name}.{column}") or {}).get(
            "object_prop_uri"
        )
        for column in contract.get("source_columns", [])
    }
    legacy_uris = {uri for uri in legacy_uris if _is_valid_uri(uri)}
    return next(iter(legacy_uris)) if len(legacy_uris) == 1 else None


def _explicit_identity_dp_pom(
    col_name: str,
    col_info: dict,
    col_types: dict,
    prefix_map: dict,
    dp_range_map: dict | None,
) -> list | None:
    """Serialize only a DP URI explicitly selected before R2RML generation."""
    prop_uri = (col_info or {}).get("data_prop_uri")
    if not _is_valid_uri(prop_uri):
        return None
    pred = _predicate_str(prop_uri, prefix_map)
    if not pred:
        return None
    sql_type = col_types.get(col_name, "character varying")
    xsd = _xsd_type(sql_type, prop_uri, dp_range_map)
    if not _is_sql_xsd_compatible(sql_type, xsd):
        return None
    return _make_pom_datatype(pred, col_name, xsd)


def _make_pom_object_column_iri(pred_str: str, col_name: str) -> list:
    """Generate an object POM when the relational value is already an IRI."""
    return [
        f'    rr:predicateObjectMap [',
        f'        rr:predicate {pred_str} ;',
        f'        rr:objectMap [ rr:column {_rr_column(col_name)} ; rr:termType rr:IRI ]',
        f'    ]',
    ]


def _partial_value_relation_contract(
    table_name: str,
    entry: dict,
    enriched_schema: dict,
    op_mapping_step1: dict,
) -> dict | None:
    """Validate the explicit upstream contract for a polymorphic IRI column.

    This path deliberately performs no name-based inference.  It is enabled
    only after the OP stage has established endpoint closure and the value
    profiler has established that the value column contains absolute IRIs.
    The value endpoint may be either the semantic subject or object; direction
    is taken exclusively from the explicit SR endpoint fields.
    """
    selected = _get_op_mapping_entry(op_mapping_step1, table_name)
    if not selected or selected.get("partial_endpoint_invariant") is not True:
        return None
    if str(selected.get("partial_value_term_type") or "").lower() != "iri":
        return None
    required_endpoint_fields = {
        "sr_subject_column",
        "sr_object_column",
        "sr_subject_ref_table",
        "sr_object_ref_table",
    }
    if not required_endpoint_fields.issubset(selected):
        return None

    op_uri = selected.get("object_prop_uri")
    value_col = selected.get("partial_value_column")
    subject_col = selected.get("sr_subject_column")
    object_col = selected.get("sr_object_column")
    subject_ref = selected.get("sr_subject_ref_table")
    object_ref = selected.get("sr_object_ref_table")
    fk_info = entry.get("fk", {}) or {}
    fk_col = fk_info.get("column")
    physical_ref = fk_info.get("ref_table") or fk_info.get("references_table")
    entry_value_col = entry.get("value_column")
    table_columns = (enriched_schema.get(table_name, {}) or {}).get("columns", {}) or {}

    if not _is_valid_uri(op_uri):
        return None
    if not fk_col or not physical_ref or not value_col or not entry_value_col:
        return None
    if entry_value_col != value_col or fk_col == value_col:
        return None
    if fk_col not in table_columns or value_col not in table_columns:
        return None

    def _same_ref(left, right) -> bool:
        return bool(left and right and str(left).lower() == str(right).lower())

    if subject_col == fk_col and object_col == value_col:
        if not _same_ref(subject_ref, physical_ref) or _same_ref(object_ref, physical_ref):
            return None
        direction = "physical_subject"
    elif subject_col == value_col and object_col == fk_col:
        if not _same_ref(object_ref, physical_ref) or _same_ref(subject_ref, physical_ref):
            return None
        direction = "value_iri_subject"
    else:
        return None

    physical_role = str(selected.get("partial_physical_endpoint_role") or "").lower()
    if physical_role:
        expected_role = "domain" if direction == "physical_subject" else "range"
        if physical_role != expected_role:
            return None
    return {
        "object_prop_uri": op_uri,
        "direction": direction,
        "physical_column": fk_col,
        "physical_ref_table": physical_ref,
        "value_column": value_col,
        "sr_subject_ref_table": subject_ref,
        "sr_object_ref_table": object_ref,
    }


def _explicit_inferred_sr_contracts(
    table_name: str,
    enriched_schema: dict,
    op_mapping_step1: dict,
) -> list[dict]:
    """Return fully evidenced relation mappings attached to an entity table.

    The OP stage can discover a relation between two columns even when the
    source table is classified as SE/SH rather than as a dedicated SR table.
    Consume only that explicit upstream contract: both endpoint columns and
    referenced tables must be present, and schema-matching evidence must tie
    each endpoint back to the same source table. No names or score thresholds
    are inferred at serialization time.
    """
    table_schema = enriched_schema.get(table_name, {}) or {}
    table_columns = table_schema.get("columns", {}) or {}
    contracts = []
    seen = set()

    for _mapping_key, selected in sorted((op_mapping_step1 or {}).items()):
        if not isinstance(selected, dict):
            continue
        if selected.get("scenario_type") != "sr_relation_inferred":
            continue

        op_uri = selected.get("object_prop_uri")
        subject_col = selected.get("sr_subject_column")
        object_col = selected.get("sr_object_column")
        subject_ref = selected.get("sr_subject_ref_table")
        object_ref = selected.get("sr_object_ref_table")
        if not all((subject_col, object_col, subject_ref, object_ref)):
            continue
        if not _is_valid_uri(op_uri) or subject_col == object_col:
            continue
        if subject_col not in table_columns or object_col not in table_columns:
            continue

        evidence = selected.get("schema_matching")
        if not isinstance(evidence, list) or not evidence:
            continue
        if any(
            str(item.get("source_table") or "").lower() != table_name.lower()
            for item in evidence
            if isinstance(item, dict)
        ):
            continue

        def _endpoint_is_evidenced(column: str, ref_table: str) -> bool:
            return any(
                isinstance(item, dict)
                and str(item.get("source_table") or "").lower() == table_name.lower()
                and str(item.get("source_column") or "").lower() == column.lower()
                and str(item.get("target_table") or "").lower() == ref_table.lower()
                for item in evidence
            )

        if not _endpoint_is_evidenced(subject_col, subject_ref):
            continue
        if not _endpoint_is_evidenced(object_col, object_ref):
            continue

        identity = (op_uri, subject_col, object_col, subject_ref, object_ref)
        if identity in seen:
            continue
        seen.add(identity)
        contracts.append(
            {
                "object_prop_uri": op_uri,
                "subject_column": subject_col,
                "object_column": object_col,
                "subject_ref_table": subject_ref,
                "object_ref_table": object_ref,
            }
        )
    return contracts


def generate_explicit_inferred_sr_mappings(
    table_name: str,
    enriched_schema: dict,
    op_mapping_step1: dict,
    base_url: str,
    prefix_map: dict,
    table_to_iri: dict,
    identity_contracts: dict[str, dict] | None = None,
) -> list[str]:
    """Serialize explicit inferred-SR contracts as auxiliary TriplesMaps."""
    blocks = []
    for contract in _explicit_inferred_sr_contracts(
        table_name=table_name,
        enriched_schema=enriched_schema,
        op_mapping_step1=op_mapping_step1,
    ):
        subject_col = contract["subject_column"]
        object_col = contract["object_column"]
        subject_ref = contract["subject_ref_table"]
        object_ref = contract["object_ref_table"]
        predicate = _predicate_str(contract["object_prop_uri"], prefix_map)
        if not predicate:
            continue

        # This upstream contract carries one source column per endpoint.  It
        # cannot truthfully serialize a composite canonical identity without
        # an explicit ordered key mapping, so skip rather than fabricate one.
        if identity_contracts is not None:
            subject_contract = identity_contracts.get(subject_ref) or {}
            object_contract = identity_contracts.get(object_ref) or {}
            if (
                len(subject_contract.get("identity_columns") or []) != 1
                or len(object_contract.get("identity_columns") or []) != 1
            ):
                continue

        subject_iri_base = table_to_iri.get(
            subject_ref, _canonical_iri_base_name(subject_ref)
        )
        object_iri_base = table_to_iri.get(
            object_ref, _canonical_iri_base_name(object_ref)
        )
        subject_template = f"{base_url}{subject_iri_base}/{{{subject_col}}}"
        object_template = f"{base_url}{object_iri_base}/{{{object_col}}}"
        sql = (
            f"SELECT {_sql_cols_in_query([subject_col, object_col])} "
            f"FROM {_sql_col_in_query(table_name)}"
        )
        mapping_id = _safe_id(
            f"{table_name}_{subject_col}_{object_col}_"
            f"{_local_name(contract['object_prop_uri'])}_InferredRelation"
        )
        lines = [
            f"<#{mapping_id}> a rr:TriplesMap ;",
            f'    rr:logicalTable [ rr:sqlQuery "{sql}" ] ;',
            f'    rr:subjectMap [ rr:template "{subject_template}" ] ;',
        ]
        lines.extend(_make_pom_object(predicate, object_template))
        lines[-1] += " ."
        blocks.append("\n".join(lines) + "\n")
    return blocks


def _dp_domain_matches_class(
    table_class_uri: str | None,
    domain_uri: str | None,
    ontology: dict,
) -> bool:
    """Return whether a named DP domain can describe the table Class."""
    if not table_class_uri or not domain_uri:
        return False
    if domain_uri in {"http://www.w3.org/2002/07/owl#Thing", "http://www.w3.org/2000/01/rdf-schema#Resource"}:
        return True
    if table_class_uri == domain_uri:
        return True
    if are_classes_disjoint(table_class_uri, domain_uri, ontology):
        return False
    return domain_uri in (ontology.get("ancestors_of", {}).get(table_class_uri, []) or [])


def _dp_domain_compatible(
    table_class_uri: str | None,
    dp_info: dict,
    ontology: dict,
) -> tuple[bool, bool]:
    """
    Return ``(compatible, has_only_unknown_expression)`` for a DP domain.

    Union class expressions are evaluated member-wise.  A blank-node domain
    that cannot be expanded is retained only as a weak fallback; an explicit
    incompatible named domain is never allowed to win on name similarity.
    """
    domains = dp_info.get("domain", []) or []
    if not domains:
        return True, True
    union_members = ontology.get("union_members", {}) or {}
    unknown_only = True
    for domain_uri in domains:
        members = union_members.get(domain_uri, []) or []
        if members:
            unknown_only = False
            if any(_dp_domain_matches_class(table_class_uri, member, ontology) for member in members):
                return True, False
            continue
        if not str(domain_uri).startswith("http"):
            continue
        unknown_only = False
        if _dp_domain_matches_class(table_class_uri, domain_uri, ontology):
            return True, False
    return False, unknown_only


def _dp_domain_hard_disjoint(
    table_class_uri: str | None,
    dp_info: dict,
    ontology: dict,
) -> bool:
    """Return True only when every resolvable DP domain is explicitly disjoint.

    A plain domain mismatch is not negative proof: the table Class is itself an
    upstream prediction and can be wrong. Blank-node expressions that cannot
    be expanded are likewise insufficient. This guard is intentionally
    stricter than :func:`_dp_domain_compatible` because replacing an already
    selected predicate is a destructive output-stage decision.
    """
    if not _is_valid_uri(table_class_uri):
        return False

    domains = dp_info.get("domain", []) or []
    if not domains:
        return False

    union_members = ontology.get("union_members", {}) or {}
    saw_named_domain = False
    for domain_uri in domains:
        members = union_members.get(domain_uri, []) or []
        if members:
            targets = members
        elif str(domain_uri).startswith("http"):
            targets = [domain_uri]
        else:
            # An unresolved expression may still contain the table Class.
            return False

        if not targets:
            return False
        for target_uri in targets:
            if not str(target_uri).startswith("http"):
                return False
            saw_named_domain = True
            if not are_classes_disjoint(table_class_uri, target_uri, ontology):
                return False
    return saw_named_domain


def _record_dp_refinement_audit(
    audit: list[dict] | None,
    *,
    action: str,
    col_name: str,
    current_prop_uri: str | None,
    selected_prop_uri: str | None,
    current_confidence: str | None,
    domain_class_uri: str | None,
    reason: str,
) -> None:
    """Persist a compact decision record and mirror it into ``generation.log``."""
    record = {
        "action": action,
        "column": col_name,
        "current_prop_uri": current_prop_uri,
        "selected_prop_uri": selected_prop_uri,
        "current_confidence": current_confidence,
        "domain_class_uri": domain_class_uri,
        "reason": reason,
    }
    if audit is not None:
        audit.append(record)
    print("[R2RML-DP-AUDIT] " + json.dumps(record, ensure_ascii=False, sort_keys=True))


def _refine_datatype_prop_uri(
    col_name: str,
    current_prop_uri: str | None,
    domain_class_uri: str | None,
    ontology: dict,
    current_confidence: str | None = None,
    audit: list[dict] | None = None,
) -> str | None:
    """Serialize an upstream DP selection without reopening alignment.

    The R2RML agent owns syntax, not semantic selection.  A valid URI is
    therefore returned byte-for-byte for every confidence level.  Ontology
    domain checks are retained solely as audit information; a missing/invalid
    URI is skipped instead of being guessed from the column name.
    """
    if not _is_valid_uri(current_prop_uri):
        return None

    current_info = (ontology.get("datatype_properties", {}) or {}).get(
        current_prop_uri, {}
    ) or {}
    if not current_info:
        return current_prop_uri

    compatible, unknown_domain = _dp_domain_compatible(
        domain_class_uri, current_info, ontology
    )
    hard_disjoint = _dp_domain_hard_disjoint(
        domain_class_uri, current_info, ontology
    )
    if hard_disjoint:
        action = "preserved_selected_hard_disjoint"
        reason = "output stage preserves the upstream DP despite an audited disjoint domain"
    elif not compatible and not unknown_domain:
        action = "preserved_selected_domain_mismatch"
        reason = "output stage records but does not resolve a named domain mismatch"
    elif not compatible and unknown_domain:
        action = "preserved_selected_unknown_domain"
        reason = "output stage records but does not interpret an anonymous domain"
    else:
        return current_prop_uri

    _record_dp_refinement_audit(
        audit,
        action=action,
        col_name=col_name,
        current_prop_uri=current_prop_uri,
        selected_prop_uri=current_prop_uri,
        current_confidence=current_confidence,
        domain_class_uri=domain_class_uri,
        reason=reason,
    )
    return current_prop_uri


def _sql_literal(value):
    """将 Python 值转成 SQL 字面量字符串（用于 WHERE 条件）"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    s = str(value).replace("'", "''")
    return f"'{s}'"


def _safe_id(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _flatten_class_values(value_to_class: dict) -> list[str]:
    out = []
    for value in (value_to_class or {}).values():
        if isinstance(value, list):
            out.extend(value)
        else:
            out.append(value)
    return [v for v in out if _is_valid_uri(v)]


def _enum_assertions_cover_subclasses(entry: dict, base_class_uri: str, ontology: dict) -> bool:
    """
    Avoid asserting a base SH class when discriminator mappings already assert
    subclasses for all mapped enum values; RDFS reasoning will infer the base.
    """
    if not _is_valid_uri(base_class_uri):
        return False
    ancestors_of = (ontology or {}).get("ancestors_of", {})
    for ta in entry.get("type_assertions", []) or []:
        if ta.get("kind") != "enum":
            continue
        mapping = ta.get("value_to_class") or {}
        if not mapping or ta.get("unmapped_values"):
            continue
        class_values = _flatten_class_values(mapping)
        if class_values and all(
            cls != base_class_uri and base_class_uri in (ancestors_of.get(cls, []) or [])
            for cls in class_values
        ):
            return True
    return False


def _leaf_descendants_for_class(base_class_uri: str, ontology: dict) -> list[str]:
    children_of = (ontology or {}).get("children_of", {})
    descendants = []
    queue = list(children_of.get(base_class_uri, []) or [])
    seen = set()
    while queue:
        uri = queue.pop(0)
        if uri in seen:
            continue
        seen.add(uri)
        descendants.append(uri)
        queue.extend(children_of.get(uri, []) or [])
    desc_set = set(descendants)
    return [
        uri for uri in descendants
        if not [child for child in children_of.get(uri, []) or [] if child in desc_set]
    ]


def _is_risky_single_value_subclass_assertion(
    entry: dict,
    ta: dict,
    ontology: dict,
) -> bool:
    """
    A single discriminator value over a broad class hierarchy is not enough to
    specialize every row into one arbitrary subclass. Candidate breadth alone
    is never assertion evidence; upstream semantic selection must decide.
    """
    return False


# ============================================================
#  按 Pattern 生成 TriplesMap
# ============================================================

def generate_value_attr_mapping(
    table_name, entry, enriched_schema,
    ontology, base_url, prefix_map, table_to_iri, dp_range_map=None,
    prop_uri_map=None, op_mapping_step1=None,
):
    """
    处理 has_xxx(EntityFK, VALUE) 这类多值数据属性表。
    表级仍是 SE，但 subject 使用被引用实体 IRI，object 使用 literal VALUE。
    """
    fk_info = entry.get("fk", {}) or {}
    fk_col = fk_info.get("column")
    ref_table = fk_info.get("ref_table") or fk_info.get("references_table")
    value_col = entry.get("value_column")
    if not fk_col or not ref_table or not value_col:
        return f"# SKIP: {table_name} (SE value_attr) 缺少 FK/value 列信息\n"

    partial_relation = _partial_value_relation_contract(
        table_name=table_name,
        entry=entry,
        enriched_schema=enriched_schema,
        op_mapping_step1=op_mapping_step1 or {},
    )
    if partial_relation:
        pred = _predicate_str(partial_relation["object_prop_uri"], prefix_map)
        if not pred:
            return f"# SKIP: {table_name} (SE partial value relation) 谓词无法序列化\n"
        physical_col = partial_relation["physical_column"]
        iri_value_col = partial_relation["value_column"]
        sql = (
            f'SELECT {_sql_cols_in_query([physical_col, iri_value_col])} '
            f'FROM {_sql_col_in_query(table_name)}'
        )
        physical_ref = partial_relation["physical_ref_table"]
        ref_iri_base = table_to_iri.get(physical_ref, _canonical_iri_base_name(physical_ref))
        physical_template = f"{base_url}{ref_iri_base}/{{{physical_col}}}"
        lines = [
            f'<#{table_name}Mapping> a rr:TriplesMap ;',
            f'    rr:logicalTable [ rr:sqlQuery "{sql}" ] ;',
        ]
        if partial_relation["direction"] == "physical_subject":
            lines.append(f'    rr:subjectMap [ rr:template "{physical_template}" ] ;')
            pom = _make_pom_object_column_iri(pred, iri_value_col)
        else:
            lines.append(
                f'    rr:subjectMap [ rr:column {_rr_column(iri_value_col)} ; '
                f'rr:termType rr:IRI ] ;'
            )
            pom = _make_pom_object(pred, physical_template)
        pom[-1] += " ."
        lines.extend(pom)
        return "\n".join(lines) + "\n"

    prop_uri = entry.get("prop_uri")
    prop_uri = _refine_datatype_prop_uri(
        col_name=table_name,
        current_prop_uri=prop_uri,
        domain_class_uri=entry.get("class_uri"),
        ontology=ontology,
        current_confidence=entry.get("prop_confidence") or entry.get("confidence"),
    )
    if not _is_valid_uri(prop_uri):
        return f"# SKIP: {table_name} (SE value_attr) 无 prop_uri\n"

    pred = _predicate_str(prop_uri, prefix_map)
    if not pred:
        return f"# SKIP: {table_name} (SE value_attr) 谓词无法序列化\n"

    cols = [fk_col, value_col]
    sql = f'SELECT {_sql_cols_in_query(cols)} FROM {_sql_col_in_query(table_name)}'
    ref_iri_base = table_to_iri.get(ref_table, _canonical_iri_base_name(ref_table))
    subject_template = f"{base_url}{ref_iri_base}/{{{fk_col}}}"

    col_types = enriched_schema.get(table_name, {}).get("columns", {})
    sql_type = col_types.get(value_col, entry.get("value_column_type") or "character varying")
    xsd = _xsd_type(sql_type, prop_uri, dp_range_map)
    if not _is_sql_xsd_compatible(sql_type, xsd):
        return f"# SKIP: {table_name} (SE value_attr) SQL/XSD 类型不兼容\n"

    lines = [
        f'<#{table_name}Mapping> a rr:TriplesMap ;',
        f'    rr:logicalTable [ rr:sqlQuery "{sql}" ] ;',
        f'    rr:subjectMap [ rr:template "{subject_template}" ] ;',
    ]
    pom = _make_pom_datatype(pred, value_col, xsd)
    pom[-1] += " ."
    lines.extend(pom)
    return "\n".join(lines) + "\n"


def generate_se_mapping(
    table_name, entry, enriched_schema, op_mapping_step1, op_mapping_step2_orphans,
    ontology, base_url, prefix_map, class_to_table, table_to_iri, dp_range_map=None,
    prop_uri_map=None, final_alignment=None, identity_contracts=None,
):
    if entry.get("table_kind") == "value_attr":
        return generate_value_attr_mapping(
            table_name=table_name,
            entry=entry,
            enriched_schema=enriched_schema,
            ontology=ontology,
            base_url=base_url,
            prefix_map=prefix_map,
            table_to_iri=table_to_iri,
            dp_range_map=dp_range_map,
            prop_uri_map=prop_uri_map,
            op_mapping_step1=op_mapping_step1,
        )

    class_uri = entry.get("class_uri")
    if not _is_valid_uri(class_uri):
        return f"# SKIP: {table_name} 无 class_uri\n"

    id_cols = _find_row_id_cols(enriched_schema, table_name)
    cols = _collect_columns(table_name, enriched_schema)
    sql = f'SELECT {_sql_cols_in_query(cols)} FROM {_sql_col_in_query(table_name)}'
    iri_base = table_to_iri.get(table_name, table_name.lower())
    subject_template = _make_subject_template_for_table(
        base_url, table_name, iri_base, id_cols, cols
    )
    if not subject_template:
        return f"# SKIP: {table_name} (SE) 无可用标识列，无法构造 subject IRI\n"
    class_pred = _predicate_str(class_uri, prefix_map)

    lines = [
        f'<#{table_name}Mapping> a rr:TriplesMap ;',
        f'    rr:logicalTable [ rr:sqlQuery "{sql}" ] ;',
        f'    rr:subjectMap [ rr:template "{subject_template}" ; rr:class {class_pred} ] ;',
    ]

    columns = entry.get("columns", {})
    col_types = enriched_schema.get(table_name, {}).get("columns", {})
    poms = []
    mapped_cols = set()

    composite_contracts, composite_fk_columns = _composite_fk_contracts(
        table_name,
        enriched_schema,
        identity_contracts,
    )
    for contract in composite_contracts:
        op_uri = _selected_composite_fk_uri(
            table_name,
            contract,
            op_mapping_step1,
        )
        pred = _predicate_str(op_uri, prefix_map)
        if pred:
            poms.append(
                _make_pom_parent_join(
                    pred,
                    contract["ref_table"],
                    contract["join_pairs"],
                )
            )

    for col_name, col_info in columns.items():
        role = col_info.get("role")
        if role == "discriminator":
            continue

        fk_ref_table = _find_fk_ref_table(enriched_schema, table_name, col_name)
        if fk_ref_table:
            if col_name not in composite_fk_columns:
                op_uri = _infer_fk_object_property_uri(
                    table_name=table_name,
                    col_name=col_name,
                    domain_class_uri=class_uri,
                    ref_table=fk_ref_table,
                    ontology=ontology,
                    class_to_table=class_to_table,
                    op_mapping_step1=op_mapping_step1,
                )
                if _is_valid_uri(op_uri):
                    pred = _predicate_str(op_uri, prefix_map)
                    if pred:
                        ref_iri_base = table_to_iri.get(fk_ref_table, _canonical_iri_base_name(fk_ref_table))
                        obj_tmpl = f"{base_url}{ref_iri_base}/{{{col_name}}}"
                        poms.append(_make_pom_object(pred, obj_tmpl))
                        mapped_cols.add(col_name)
            literal_pom = _explicit_identity_dp_pom(
                col_name,
                col_info,
                col_types,
                prefix_map,
                dp_range_map,
            )
            if literal_pom:
                poms.append(literal_pom)
                mapped_cols.add(col_name)
            continue

        # Non-FK primary-key columns are considered by the conservative
        # identity+DatatypeProperty pass below.  They remain identity-only in
        # this ordinary column loop.
        if role in ("pk", "sh_inherited_pk"):
            literal_pom = _explicit_identity_dp_pom(
                col_name,
                col_info,
                col_types,
                prefix_map,
                dp_range_map,
            )
            if literal_pom:
                poms.append(literal_pom)
                mapped_cols.add(col_name)
            continue

        if role == "data_attr":
            prop_uri = col_info.get("prop_uri")
            prop_uri = _refine_datatype_prop_uri(
                col_name=col_name,
                current_prop_uri=prop_uri,
                domain_class_uri=class_uri,
                ontology=ontology,
                current_confidence=col_info.get("confidence") or col_info.get("prop_confidence"),
            )

            # 先检查 Step 2 孤儿列是否补全了 ObjectProperty
            orphan_key = f"{table_name}.{col_name}"
            if (not _is_valid_uri(prop_uri)) and orphan_key in op_mapping_step2_orphans:
                orphan_info = op_mapping_step2_orphans[orphan_key]
                op_uri = orphan_info.get("object_prop_uri")
                if _is_valid_uri(op_uri):
                    if not _allow_orphan_object_pom(orphan_info, table_name, col_name, enriched_schema):
                        continue
                    # 防御：布尔列不生成 ObjectProperty POM（OP Step2 可能误匹配）
                    sql_type_check = col_types.get(col_name, "")
                    if "bool" in sql_type_check.lower():
                        continue
                    pred = _predicate_str(op_uri, prefix_map)
                    if pred:
                        direction = orphan_info.get("direction", "normal")
                        # 优先从本体查 range 表名
                        range_tbl = resolve_range_table(
                            op_uri, ontology, class_to_table, direction, table_to_iri=table_to_iri
                        )
                        if not range_tbl:
                            # fallback：从 orphan_info 里的 range_class_uri 查 class_to_table
                            range_cls = orphan_info.get("range_class_uri") if direction == "normal" \
                                        else orphan_info.get("domain_class_uri")
                            if range_cls and range_cls in class_to_table:
                                rt = class_to_table[range_cls]
                                range_tbl = table_to_iri.get(rt, _canonical_iri_base_name(rt))
                        if range_tbl:
                            obj_tmpl = f"{base_url}{range_tbl}/{{{col_name}}}"
                        else:
                            # 最后 fallback：跳过，不生成错误的 POM
                            continue
                        poms.append(_make_pom_object(pred, obj_tmpl))
                        mapped_cols.add(col_name)
                    continue

            # 普通 DatatypeProperty
            if not _is_valid_uri(prop_uri):
                continue

            pred = _predicate_str(prop_uri, prefix_map)
            if not pred:
                continue
            sql_type = col_types.get(col_name, "character varying")
            xsd = _xsd_type(sql_type, prop_uri, dp_range_map)
            if not _is_sql_xsd_compatible(sql_type, xsd):
                continue
            poms.append(_make_pom_datatype(pred, col_name, xsd))
            mapped_cols.add(col_name)

        elif role == "fk_obj":
            # 常规路径已在上方 FK 通用分支处理，这里保持兜底兼容
            ref_table = col_info.get("ref_table") or _find_fk_ref_table(enriched_schema, table_name, col_name)
            op_uri = _infer_fk_object_property_uri(
                table_name=table_name,
                col_name=col_name,
                domain_class_uri=class_uri,
                ref_table=ref_table,
                ontology=ontology,
                class_to_table=class_to_table,
                op_mapping_step1=op_mapping_step1,
            )
            if not _is_valid_uri(op_uri) or not ref_table:
                continue

            pred = _predicate_str(op_uri, prefix_map)
            if not pred:
                continue

            ref_iri_base = table_to_iri.get(ref_table, _canonical_iri_base_name(ref_table))
            obj_tmpl = f"{base_url}{ref_iri_base}/{{{col_name}}}"
            poms.append(_make_pom_object(pred, obj_tmpl))
            mapped_cols.add(col_name)

    # 拼接 POM，用分号分隔，最后一个用句号
    if not poms:
        # 只有 subjectMap，去掉最后的分号
        result = "\n".join(lines)
        if result.endswith(" ;"):
            result = result[:-2] + " ."
        return result + "\n"

    all_pom_lines = []
    for i, pom in enumerate(poms):
        separator = " ;" if i < len(poms) - 1 else " ."
        pom_with_sep = pom.copy()
        pom_with_sep[-1] = pom_with_sep[-1] + separator
        all_pom_lines.extend(pom_with_sep)

    lines.extend(all_pom_lines)
    return "\n".join(lines) + "\n"


def generate_sh_mapping(
    table_name, entry, enriched_schema,op_mapping_step1, op_mapping_step2_orphans,
    ontology, base_url, prefix_map, class_to_table, table_to_iri, dp_range_map=None,
    prop_uri_map=None, final_alignment=None, identity_contracts=None,
    identity_fk_consumption_contracts=None,
):
    sub_class_uri = entry.get("sub_class_uri")
    if not _is_valid_uri(sub_class_uri):
        return f"# SKIP: {table_name} (SH) 无 sub_class_uri\n"

    id_cols = _validated_subject_identity_cols(
        table_name=table_name,
        entry=entry,
        enriched_schema=enriched_schema,
        final_alignment=final_alignment,
        identity_contracts=identity_contracts,
    )
    cols = _collect_columns(table_name, enriched_schema)
    sql = f'SELECT {_sql_cols_in_query(cols)} FROM {_sql_col_in_query(table_name)}'
    iri_base = table_to_iri.get(table_name, table_name.lower())
    subject_template = _make_subject_template_for_table(
        base_url, table_name, iri_base, id_cols, cols
    )
    if not subject_template:
        return f"# SKIP: {table_name} (SH) 无可用标识列，无法构造 subject IRI\n"
    sub_pred = _predicate_str(sub_class_uri, prefix_map)
    omit_base_class = _enum_assertions_cover_subclasses(entry, sub_class_uri, ontology)
    subject_map = f'    rr:subjectMap [ rr:template "{subject_template}"'
    if not omit_base_class:
        subject_map += f" ; rr:class {sub_pred}"
    subject_map += " ] ;"

    lines = [
        f'<#{table_name}Mapping> a rr:TriplesMap ;',
        f'    rr:logicalTable [ rr:sqlQuery "{sql}" ] ;',
        subject_map,
    ]

    columns = entry.get("columns", {})
    col_types = enriched_schema.get(table_name, {}).get("columns", {})
    poms = []
    if identity_fk_consumption_contracts is None:
        identity_fk_consumption_contracts = (
            build_sh_identity_fk_consumption_contracts(
                final_alignment or {table_name: entry},
                enriched_schema,
                ontology,
                identity_contracts,
            )
        )
    consumed_identity_fks = identity_fk_consumption_contracts.get(
        table_name, []
    )

    composite_contracts, composite_fk_columns = _composite_fk_contracts(
        table_name,
        enriched_schema,
        identity_contracts,
    )
    for contract in composite_contracts:
        if any(
            _fk_contract_matches(contract, identity_contract)
            for identity_contract in consumed_identity_fks
        ):
            continue
        op_uri = _selected_composite_fk_uri(
            table_name,
            contract,
            op_mapping_step1,
        )
        pred = _predicate_str(op_uri, prefix_map)
        if pred:
            poms.append(
                _make_pom_parent_join(
                    pred,
                    contract["ref_table"],
                    contract["join_pairs"],
                )
            )

    for col_name, col_info in columns.items():
        role = col_info.get("role")
        if role == "discriminator":
            continue

        fk_ref_table = _find_fk_ref_table(enriched_schema, table_name, col_name)
        if fk_ref_table:
            if col_name not in composite_fk_columns:
                selected_scalar_fk = None
                for physical_rows in _physical_fk_groups(
                    enriched_schema.get(table_name, {}) or {}
                ):
                    physical_fk = _complete_physical_fk_group(physical_rows)
                    if (
                        physical_fk
                        and not physical_fk.get("is_composite")
                        and physical_fk.get("source_columns") == [col_name]
                        and physical_fk.get("target_table") == fk_ref_table
                    ):
                        selected_scalar_fk = physical_fk
                        break
                identity_fk_consumed = bool(
                    selected_scalar_fk
                    and any(
                        _fk_contract_matches(
                            selected_scalar_fk, identity_contract
                        )
                        for identity_contract in consumed_identity_fks
                    )
                )
                op_uri = _infer_fk_object_property_uri(
                    table_name=table_name,
                    col_name=col_name,
                    domain_class_uri=sub_class_uri,
                    ref_table=fk_ref_table,
                    ontology=ontology,
                    class_to_table=class_to_table,
                    op_mapping_step1=op_mapping_step1,
                )
                if not identity_fk_consumed and _is_valid_uri(op_uri):
                    pred = _predicate_str(op_uri, prefix_map)
                    if pred:
                        ref_iri_base = table_to_iri.get(fk_ref_table, _canonical_iri_base_name(fk_ref_table))
                        obj_tmpl = f"{base_url}{ref_iri_base}/{{{col_name}}}"
                        poms.append(_make_pom_object(pred, obj_tmpl))
            literal_pom = _explicit_identity_dp_pom(
                col_name,
                col_info,
                col_types,
                prefix_map,
                dp_range_map,
            )
            if literal_pom:
                poms.append(literal_pom)
            continue

        # An inherited/ordinary PK can also be a physical FK. That case was
        # handled above using an explicit Step1 OP; a non-FK identifier still
        # remains subject identity only and must not become a literal POM.
        if role in ("sh_inherited_pk", "pk"):
            literal_pom = _explicit_identity_dp_pom(
                col_name,
                col_info,
                col_types,
                prefix_map,
                dp_range_map,
            )
            if literal_pom:
                poms.append(literal_pom)
            continue

        if role == "data_attr":
            prop_uri = col_info.get("prop_uri")
            prop_uri = _refine_datatype_prop_uri(
                col_name=col_name,
                current_prop_uri=prop_uri,
                domain_class_uri=sub_class_uri,
                ontology=ontology,
                current_confidence=col_info.get("confidence") or col_info.get("prop_confidence"),
            )
            orphan_key = f"{table_name}.{col_name}"

            if (not _is_valid_uri(prop_uri)) and orphan_key in op_mapping_step2_orphans:
                orphan_info = op_mapping_step2_orphans[orphan_key]
                op_uri = orphan_info.get("object_prop_uri")
                if _is_valid_uri(op_uri):
                    if not _allow_orphan_object_pom(orphan_info, table_name, col_name, enriched_schema):
                        continue
                    pred = _predicate_str(op_uri, prefix_map)
                    if pred:
                        direction = orphan_info.get("direction", "normal")
                        range_tbl = resolve_range_table(
                            op_uri, ontology, class_to_table, direction, table_to_iri=table_to_iri
                        )
                        if range_tbl:
                            obj_tmpl = f"{base_url}{range_tbl}/{{{col_name}}}"
                        else:
                            # fallback：从 orphan_info range_class_uri 查表名
                            range_cls = orphan_info.get("range_class_uri") if direction == "normal" \
                                        else orphan_info.get("domain_class_uri")
                            if range_cls and range_cls in class_to_table:
                                rt = class_to_table[range_cls]
                                range_tbl = table_to_iri.get(rt, _canonical_iri_base_name(rt))
                                obj_tmpl = f"{base_url}{range_tbl}/{{{col_name}}}"
                            else:
                                continue  # 实在找不到，跳过
                        poms.append(_make_pom_object(pred, obj_tmpl))
                    continue

            if not _is_valid_uri(prop_uri):
                continue

            pred = _predicate_str(prop_uri, prefix_map)
            if not pred:
                continue
            sql_type = col_types.get(col_name, "character varying")
            xsd = _xsd_type(sql_type, prop_uri, dp_range_map)
            if not _is_sql_xsd_compatible(sql_type, xsd):
                continue
            poms.append(_make_pom_datatype(pred, col_name, xsd))

        elif role == "fk_obj":
            # 常规路径已在上方 FK 通用分支处理，这里保持兜底兼容
            ref_table = col_info.get("ref_table") or _find_fk_ref_table(enriched_schema, table_name, col_name)
            op_uri = _infer_fk_object_property_uri(
                table_name=table_name,
                col_name=col_name,
                domain_class_uri=sub_class_uri,
                ref_table=ref_table,
                ontology=ontology,
                class_to_table=class_to_table,
                op_mapping_step1=op_mapping_step1,
            )
            if not _is_valid_uri(op_uri) or not ref_table:
                continue
            pred = _predicate_str(op_uri, prefix_map)
            if not pred:
                continue
            ref_iri_base = table_to_iri.get(ref_table, _canonical_iri_base_name(ref_table))
            obj_tmpl = f"{base_url}{ref_iri_base}/{{{col_name}}}"
            poms.append(_make_pom_object(pred, obj_tmpl))

    if not poms:
        result = "\n".join(lines)
        if result.endswith(" ;"):
            result = result[:-2] + " ."
        return result + "\n"

    all_pom_lines = []
    for i, pom in enumerate(poms):
        separator = " ;" if i < len(poms) - 1 else " ."
        pom_with_sep = pom.copy()
        pom_with_sep[-1] = pom_with_sep[-1] + separator
        all_pom_lines.extend(pom_with_sep)

    lines.extend(all_pom_lines)
    return "\n".join(lines) + "\n"


def generate_sr_mapping(
    table_name, entry, enriched_schema, op_mapping_step1,
    base_url, prefix_map, table_to_iri, class_to_table=None
):
    op_uri = None
    op_mapping_info = _get_op_mapping_entry(op_mapping_step1, table_name)
    if op_mapping_info:
        op_uri = op_mapping_info.get("object_prop_uri")

    if not _is_valid_uri(op_uri):
        return f"# SKIP: {table_name} (SR) 无 ObjectProperty\n"

    fk1 = entry.get("fk1", {})
    fk2 = entry.get("fk2", {})
    fk1_col = fk1.get("column")
    fk2_col = fk2.get("column")
    fk1_ref = fk1.get("ref_table") or fk1.get("references_table") or ""
    fk2_ref = fk2.get("ref_table") or fk2.get("references_table") or ""
    # The equivalence OP module records the selected SR endpoints explicitly.
    # Prefer those fields over the original fk1/fk2 order: partial SRs have no
    # physical FK on one side, and a reversed OP must swap both columns and
    # referenced tables consistently. Otherwise a complete physical full-FK
    # entry remains readable without reopening semantic endpoint selection.
    mapped_subject_col = (op_mapping_info or {}).get("sr_subject_column")
    mapped_object_col = (op_mapping_info or {}).get("sr_object_column")
    mapped_subject_ref = (op_mapping_info or {}).get("sr_subject_ref_table")
    mapped_object_ref = (op_mapping_info or {}).get("sr_object_ref_table")
    if mapped_subject_col and mapped_object_col:
        fk1_col = mapped_subject_col
        fk2_col = mapped_object_col
        fk1_ref = mapped_subject_ref or ""
        fk2_ref = mapped_object_ref or ""
    elif op_mapping_info and op_mapping_info.get("sr_direction") == "reversed":
        fk1_col, fk2_col = fk2_col, fk1_col
        fk1_ref, fk2_ref = fk2_ref, fk1_ref

    if not fk1_col or not fk2_col:
        return f"# SKIP: {table_name} (SR) FK 列信息不完整\n"

    # R2RML is a serializer, not an endpoint selector. A partial relation must
    # arrive with both reference tables fixed by OPMapping's explicit contract;
    # missing endpoints are rejected instead of being guessed from Class maps.
    if not fk1_ref or not fk2_ref:
        return f"# SKIP: {table_name} (SR) FK 引用表信息不完整\n"

    cols = [fk1_col, fk2_col]
    sql = f'SELECT {_sql_cols_in_query(cols)} FROM {_sql_col_in_query(table_name)}'

    fk1_iri_base = table_to_iri.get(fk1_ref, _canonical_iri_base_name(fk1_ref))
    fk2_iri_base = table_to_iri.get(fk2_ref, _canonical_iri_base_name(fk2_ref))
    subject_template = f"{base_url}{fk1_iri_base}/{{{fk1_col}}}"
    object_template = f"{base_url}{fk2_iri_base}/{{{fk2_col}}}"

    pred = _predicate_str(op_uri, prefix_map)
    lines = [f'<#{table_name}Mapping> a rr:TriplesMap ;']
    lines.append(f'    rr:logicalTable [ rr:sqlQuery "{sql}" ] ;')
    # SR 是关系映射，不应在此处额外声明 subject rdf:type，避免按关系行重复打类型
    lines.append(f'    rr:subjectMap [ rr:template "{subject_template}" ] ;')

    lines.extend(_make_pom_object(pred, object_template))
    lines[-1] += " ."
    return "\n".join(lines) + "\n"


def generate_type_assertion_mappings(
    table_name, entry, enriched_schema, base_url, prefix_map, table_to_iri, ontology=None,
    final_alignment=None, identity_contracts=None,
):
    """
    为 discriminator/type_assertions 生成附加 rdf:type TriplesMap。
    支持:
      - enum: TYPE=值 -> rdf:type Class
      - boolean: is_x=true -> rdf:type Class
    """
    assertions = entry.get("type_assertions", []) or []
    if not assertions:
        return []

    id_cols = _validated_subject_identity_cols(
        table_name=table_name,
        entry=entry,
        enriched_schema=enriched_schema,
        final_alignment=final_alignment,
        identity_contracts=identity_contracts,
    )
    iri_base = table_to_iri.get(table_name, _canonical_iri_base_name(table_name))
    cols = _collect_columns(table_name, enriched_schema)
    subject_template = _make_subject_template_for_table(
        base_url, table_name, iri_base, id_cols, cols
    )
    if not subject_template:
        return []
    blocks = []

    for ta in assertions:
        kind = ta.get("kind")
        col = ta.get("column")
        if not col:
            continue

        if kind == "enum":
            if _is_risky_single_value_subclass_assertion(entry, ta, ontology or {}):
                continue
            mapping = ta.get("value_to_class") or {}
            for raw_val, class_value in mapping.items():
                class_uris = class_value if isinstance(class_value, list) else [class_value]
                sql_val = _sql_literal(int(raw_val) if str(raw_val).isdigit() else raw_val)
                select_cols = _sql_cols_in_query(id_cols) if id_cols else _sql_col_in_query(col)
                sql = (
                    f'SELECT {select_cols} FROM {_sql_col_in_query(table_name)} '
                    f'WHERE {_sql_col_in_query(col)} = {sql_val}'
                )
                for class_uri in class_uris:
                    if not _is_valid_uri(class_uri):
                        continue
                    class_term = _predicate_str(class_uri, prefix_map)
                    if not class_term:
                        continue
                    map_id = _safe_id(f"{table_name}_{col}_{raw_val}_{_local_name(class_uri)}_Type")
                    lines = [
                        f"<#{map_id}> a rr:TriplesMap ;",
                        f'    rr:logicalTable [ rr:sqlQuery "{sql}" ] ;',
                        f'    rr:subjectMap [ rr:template "{subject_template}" ] ;',
                        f"    rr:predicateObjectMap [",
                        f"        rr:predicate rdf:type ;",
                        f"        rr:objectMap [ rr:constant {class_term} ]",
                        f"    ] .",
                    ]
                    blocks.append("\n".join(lines) + "\n")

        elif kind == "boolean":
            class_uri = ta.get("true_class_uri")
            if not _is_valid_uri(class_uri):
                continue
            class_term = _predicate_str(class_uri, prefix_map)
            if not class_term:
                continue
            select_cols = _sql_cols_in_query(id_cols) if id_cols else _sql_col_in_query(col)
            sql = (
                f'SELECT {select_cols} FROM {_sql_col_in_query(table_name)} '
                f'WHERE {_sql_col_in_query(col)} = true'
            )
            map_id = _safe_id(f"{table_name}_{col}_true_Type")
            lines = [
                f"<#{map_id}> a rr:TriplesMap ;",
                f'    rr:logicalTable [ rr:sqlQuery "{sql}" ] ;',
                f'    rr:subjectMap [ rr:template "{subject_template}" ] ;',
                f"    rr:predicateObjectMap [",
                f"        rr:predicate rdf:type ;",
                f"        rr:objectMap [ rr:constant {class_term} ]",
                f"    ] .",
            ]
            blocks.append("\n".join(lines) + "\n")

    return blocks


# ============================================================
#  主生成函数
# ============================================================

def generate_r2rml(
    final_alignment, op_mapping_full, enriched_schema, ontology,
    base_url="http://example.com/", prefix="ont"
):
    classes = ontology.get("classes", [])
    namespace = _get_namespace(classes[0]) if classes else ""

    prefix_map = {
        prefix: namespace,
        "rr": "http://www.w3.org/ns/r2rml#",
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
    }

    op_mapping_step1 = op_mapping_full.get("step1", {})
    op_mapping_step2_orphans = op_mapping_full.get("step2_orphans", {})

    # 构建全局映射表
    class_to_table = build_class_to_table_map(final_alignment)
    identity_contracts = build_entity_identity_contracts(
        final_alignment, enriched_schema, ontology
    )
    identity_fk_consumption_contracts = (
        build_sh_identity_fk_consumption_contracts(
            final_alignment,
            enriched_schema,
            ontology,
            identity_contracts,
        )
    )
    table_to_iri = build_table_to_iri_base(
        final_alignment, enriched_schema, identity_contracts, ontology
    )
    dp_range_map = build_dp_range_map(ontology)

    # 前缀
    parts = [
        f"@prefix rr: <{prefix_map['rr']}> .",
        f"@prefix rdf: <{prefix_map['rdf']}> .",
        f"@prefix rdfs: <{prefix_map['rdfs']}> .",
        f"@prefix xsd: <{prefix_map['xsd']}> .",
    ]
    if namespace:
        parts.append(f"@prefix {prefix}: <{namespace}> .")
    parts.append(f"@base <{base_url}> .")
    parts.append("")

    counts = {"SE": 0, "SH": 0, "SR": 0, "INFERRED_SR": 0, "SKIP": 0}

    # SH
    for table_name, entry in final_alignment.items():
        if entry.get("pattern") != "SH":
            continue
        mapping = generate_sh_mapping(
            table_name, entry, enriched_schema,op_mapping_step1, op_mapping_step2_orphans,
            ontology, base_url, prefix_map, class_to_table, table_to_iri, dp_range_map,
            prop_uri_map=None,
            final_alignment=final_alignment,
            identity_contracts=identity_contracts,
            identity_fk_consumption_contracts=identity_fk_consumption_contracts,
        )
        parts.append(f"# === {table_name} (SH) ===")
        parts.append(mapping)
        counts["SH"] += 1
        inferred_relations = generate_explicit_inferred_sr_mappings(
            table_name=table_name,
            enriched_schema=enriched_schema,
            op_mapping_step1=op_mapping_step1,
            base_url=base_url,
            prefix_map=prefix_map,
            table_to_iri=table_to_iri,
            identity_contracts=identity_contracts,
        )
        if inferred_relations:
            parts.append(f"# === {table_name} (Explicit Inferred Relations) ===")
            parts.extend(inferred_relations)
            counts["INFERRED_SR"] += len(inferred_relations)

    # SR
    for table_name, entry in final_alignment.items():
        if entry.get("pattern") != "SR":
            continue
        mapping = generate_sr_mapping(
            table_name, entry, enriched_schema, op_mapping_step1,
            base_url, prefix_map, table_to_iri, class_to_table
        )
        parts.append(f"# === {table_name} (SR) ===")
        parts.append(mapping)
        counts["SR"] += 1

    # SE
    for table_name, entry in final_alignment.items():
        pattern = entry.get("pattern", "SE")
        if pattern != "SE":
            continue

        mapping = generate_se_mapping(
            table_name, entry, enriched_schema, op_mapping_step1, op_mapping_step2_orphans,
            ontology, base_url, prefix_map, class_to_table, table_to_iri, dp_range_map,
            prop_uri_map=None,
            final_alignment=final_alignment,
            identity_contracts=identity_contracts,
        )
        parts.append(f"# === {table_name} ({pattern}) ===")
        parts.append(mapping)
        counts["SE"] += 1
        inferred_relations = generate_explicit_inferred_sr_mappings(
            table_name=table_name,
            enriched_schema=enriched_schema,
            op_mapping_step1=op_mapping_step1,
            base_url=base_url,
            prefix_map=prefix_map,
            table_to_iri=table_to_iri,
            identity_contracts=identity_contracts,
        )
        if inferred_relations:
            parts.append(f"# === {table_name} (Explicit Inferred Relations) ===")
            parts.extend(inferred_relations)
            counts["INFERRED_SR"] += len(inferred_relations)

    # discriminator/type_assertions 附加 rdf:type 映射（SE/SH）
    for table_name, entry in final_alignment.items():
        if entry.get("pattern") not in ("SE", "SH"):
            continue
        type_blocks = generate_type_assertion_mappings(
            table_name=table_name,
            entry=entry,
            enriched_schema=enriched_schema,
            base_url=base_url,
            prefix_map=prefix_map,
            table_to_iri=table_to_iri,
            ontology=ontology,
            final_alignment=final_alignment,
            identity_contracts=identity_contracts,
        )
        if not type_blocks:
            continue
        parts.append(f"# === {table_name} (Type Assertions) ===")
        parts.extend(type_blocks)

    print(
        f"\nR2RML 生成完成: SE={counts['SE']}, SH={counts['SH']}, "
        f"SR={counts['SR']}, INFERRED_SR={counts['INFERRED_SR']}"
    )
    return "\n".join(parts)


# ============================================================
#  主程序
# ============================================================

def _main() -> None:
    from utils.ontology_utils import read_ontology

    parser = argparse.ArgumentParser(description="Generate R2RML from upstream mapping artifacts.")
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--op-mapping", required=True, type=Path)
    parser.add_argument("--ontology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()

    with args.alignment.open("r", encoding="utf-8") as f:
        final_alignment = json.load(f)
    with args.schema.open("r", encoding="utf-8") as f:
        enriched_schema = json.load(f)
    with args.op_mapping.open("r", encoding="utf-8") as f:
        op_mapping_full = json.load(f)

    ontology = read_ontology(str(args.ontology))

    print(f"已加载 final_alignment: {len(final_alignment)} 张表")
    print(f"已加载 enriched_schema: {len(enriched_schema)} 张表")

    r2rml = generate_r2rml(
        final_alignment=final_alignment,
        op_mapping_full=op_mapping_full,
        enriched_schema=enriched_schema,
        ontology=ontology,
        base_url=args.base_url,
        prefix=args.prefix,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(r2rml, encoding="utf-8")

    print(f"\n映射已保存到 {args.output}")
    print(f"文件大小: {len(r2rml)} 字符")


if __name__ == "__main__":
    _main()
