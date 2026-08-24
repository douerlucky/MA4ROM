"""
Data Property Mapping 前的 Pattern 约束候选生成
根据每张表的Pattern类型（SE/SH/SR），为每张表和每列
生成本体候选集合，缩小后续LLM匹配的搜索空间。

输入:
  - enriched_schema: FKCompletion后的schema对象
  - pattern_result:  classify_agent输出的 {table: SE/SH/SR}
  - ontology:        read_ontology()解析出的本体对象

输出:
  - candidates: 每张表/每列的候选集合（dict结构）

列级关系与身份说明:
  LLM4VKG/Calvanese 原文把 SRm 作为独立 mapping pattern；本系统当前将
  表级输出收敛为 SE/SH/SR，因此 SE/SH 表内的物理 FK 列会保留关系约束
  元数据。身份、关系和 literal 语义彼此正交：PK∩FK 既是 identity_part，
  也是 FK constraint member；若本体检索还给出独立的 DatatypeProperty
  证据，则通过 dp_candidates 显式交给 DP/RealValue 阶段裁决。输出阶段
  不得从列名重新猜测这些语义。
"""

import json
from utils.ontology_utils import (
    are_classes_disjoint,
    local_name as _local_name,
)
from utils.candidate_ranking import (
    rank_class_candidates as _rank_class_candidates,
    rank_object_prop_candidates as _rank_object_prop_candidates,
    rank_datatype_prop_candidates as _rank_datatype_prop_candidates,
    rank_identity_datatype_prop_candidates as _rank_identity_datatype_prop_candidates,
)


def _norm_identifier(value: str | None) -> str:
    """Compare relational key names without punctuation/case noise."""
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _rank_table_class_candidates(table_name: str, classes: list, top_k: int = 5) -> list:
    """Rank Class candidates with an exact normalized table-name tie-break.

    Token overlap intentionally gives broad names and their specializations the
    same score in some schemas.  For a physical table, however, a unique exact
    local-name match is stronger evidence than that token tie.  Keep this
    preference local to table-to-Class resolution so column candidate behavior
    is unchanged.
    """
    ranked = _rank_class_candidates(
        table_name,
        classes,
        top_k=max(len(classes), top_k),
    )
    normalized_table = _norm_identifier(table_name)
    ranked.sort(
        key=lambda candidate: (
            0
            if _norm_identifier(candidate.get("local_name")) == normalized_table
            else 1,
            -float(candidate.get("score", 0.0)),
            -float(candidate.get("syntax_score", 0.0)),
            -float(candidate.get("token_score", 0.0)),
            str(candidate.get("uri", "")),
        )
    )
    return ranked[:top_k]


def _resolve_table_class(table_name: str, classes: list) -> tuple[str | None, str]:
    """Resolve a table to one Class without breaking semantic ties arbitrarily."""
    ranked = _rank_table_class_candidates(
        table_name,
        classes,
        top_k=max(len(classes), 1),
    )
    normalized_table = _norm_identifier(table_name)
    exact = [
        candidate
        for candidate in ranked
        if _norm_identifier(candidate.get("local_name")) == normalized_table
    ]
    if len(exact) == 1:
        return exact[0].get("uri"), "exact"
    if len(exact) > 1:
        return None, "ambiguous_exact"
    if not ranked or float(ranked[0].get("score", 0.0)) <= 0.0:
        return None, "unresolved"

    top_signature = tuple(
        float(ranked[0].get(field, 0.0))
        for field in ("score", "syntax_score", "token_score")
    )
    tied = [
        candidate
        for candidate in ranked
        if tuple(
            float(candidate.get(field, 0.0))
            for field in ("score", "syntax_score", "token_score")
        )
        == top_signature
    ]
    if len(tied) != 1:
        return None, "ambiguous_fuzzy"
    return tied[0].get("uri"), "fuzzy"


def _sh_child_parent_compatible(
    child_uri: str | None,
    parent_uri: str | None,
    ontology: dict | None,
) -> bool:
    """Check whether an SH child candidate can denote the physical parent.

    OWL permits either a concrete child→parent edge or a shared/broader class
    contract.  A physical inherited PK is still the decisive evidence; this
    helper only removes candidates that are explicitly unrelated/disjoint.
    """
    if not child_uri or not parent_uri:
        return False
    if child_uri == parent_uri:
        return True
    ontology = ontology or {}
    if are_classes_disjoint(child_uri, parent_uri, ontology):
        return False
    ancestors = ontology.get("ancestors_of", {}) or {}
    return bool(
        parent_uri in (ancestors.get(child_uri, []) or [])
        or child_uri in (ancestors.get(parent_uri, []) or [])
    )


def _ancestor_distance(
    child_uri: str | None,
    ancestor_uri: str | None,
    ontology: dict | None,
) -> int | None:
    """Return the shortest named-class parent distance, if one is known."""
    if not child_uri or not ancestor_uri or child_uri == ancestor_uri:
        return None
    subclass_of = ((ontology or {}).get("subclass_of", {}) or {})
    frontier = [(child_uri, 0)]
    visited = {child_uri}
    while frontier:
        current, distance = frontier.pop(0)
        for parent in subclass_of.get(current, []) or []:
            if parent == ancestor_uri:
                return distance + 1
            if parent not in visited:
                visited.add(parent)
                frontier.append((parent, distance + 1))

    # Older ontology snapshots may expose only flattened ancestry.  It proves
    # compatibility but cannot safely claim that one ancestor is nearer.
    if ancestor_uri in (((ontology or {}).get("ancestors_of", {}) or {}).get(child_uri, []) or []):
        return 10**6
    return None


def _unique_best_parent_candidate(parent_candidates: list[dict]) -> tuple[dict | None, set[str]]:
    """Choose one inheritance FK or conservatively reject a genuine tie."""
    if not parent_candidates:
        return None, set()

    semantic_candidates = [
        candidate
        for candidate in parent_candidates
        if candidate.get("ancestor_distance") is not None
    ]
    pool = semantic_candidates or parent_candidates
    for candidate in pool:
        distance = candidate.get("ancestor_distance")
        candidate["selection_rank"] = (
            1 if distance is not None else 0,
            -(distance if distance is not None else 10**9),
            candidate.get("class_resolution_score", 0),
            candidate.get("structural_score", 0),
        )

    best_rank = max(candidate["selection_rank"] for candidate in pool)
    tied = [candidate for candidate in pool if candidate["selection_rank"] == best_rank]
    physical_identities = {
        (
            candidate["fk"].get("column"),
            candidate["fk"].get("ref_table"),
            candidate["fk"].get("ref_col"),
        )
        for candidate in tied
    }
    if len(physical_identities) != 1:
        return None, {
            candidate["fk"].get("column")
            for candidate in tied
            if candidate["fk"].get("column")
        }
    return tied[0], set()


def _fk_constraint_key(fk: dict, ordinal: int) -> str:
    """Build a stable key for one physical FK constraint."""
    name = str(fk.get("constraint_name") or "").strip()
    table = str(fk.get("ref_table") or fk.get("references_table") or "").strip()
    return name or f"__unnamed_fk_{ordinal}_{table}"


def _prepare_fk_metadata(
    fks: list[dict],
    enriched_schema: dict,
    table_name: str,
) -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Keep every FK membership while exposing one safe column representative.

    A column may belong to both a composite constraint and a scalar FK.  The
    former must remain a constraint-level relation; the latter is the only
    representative suitable for the legacy per-column DP/OP interface.
    """
    normalized: list[tuple[int, dict]] = []
    for ordinal, raw in enumerate(fks or []):
        fk = dict(raw or {})
        if "references_table" in fk and "ref_table" not in fk:
            fk["ref_table"] = fk.get("references_table")
        if "references_column" in fk and "ref_col" not in fk:
            fk["ref_col"] = fk.get("references_column")
        fk.setdefault("constraint_name", "")
        try:
            fk["fk_arity"] = int(fk.get("fk_arity") or 1)
        except (TypeError, ValueError):
            fk["fk_arity"] = 1
        fk["_fk_group_key"] = _fk_constraint_key(fk, ordinal)
        normalized.append((ordinal, fk))

    groups: dict[str, list[dict]] = {}
    for _, fk in normalized:
        groups.setdefault(fk["_fk_group_key"], []).append(fk)

    target_pk_by_table = {
        table: set(info.get("primary_key", []) or [])
        for table, info in (enriched_schema or {}).items()
        if isinstance(info, dict)
    }
    by_column: dict[str, list[tuple[int, dict]]] = {}
    for ordinal, fk in normalized:
        if fk.get("column"):
            by_column.setdefault(fk["column"], []).append((ordinal, fk))

    selected: dict[str, dict] = {}
    for column, memberships in by_column.items():
        def rank(item: tuple[int, dict]) -> tuple:
            ordinal, fk = item
            ref_table = fk.get("ref_table") or ""
            ref_col = fk.get("ref_col") or ""
            target_pk = ref_col in target_pk_by_table.get(ref_table, set())
            group_size = len(groups.get(fk["_fk_group_key"], []))
            return (
                0 if int(fk.get("fk_arity") or 1) == 1 and group_size == 1 else 1,
                0 if target_pk else 1,
                int(fk.get("fk_arity") or 1),
                ordinal,
            )

        _, representative = min(memberships, key=rank)
        representative = dict(representative)
        representative["fk_memberships"] = [dict(fk) for _, fk in memberships]
        representative["composite_memberships"] = [
            dict(fk)
            for _, fk in memberships
            if int(fk.get("fk_arity") or 1) > 1
            or len(groups.get(fk["_fk_group_key"], [])) > 1
        ]
        representative.pop("_fk_group_key", None)
        selected[column] = representative

    clean_groups = {
        key: [
            {k: v for k, v in fk.items() if k != "_fk_group_key"}
            for fk in members
        ]
        for key, members in groups.items()
    }
    return selected, clean_groups

def generate_candidates(
    enriched_schema: dict,
    pattern_result: dict,
    ontology: dict,
    disabled_patterns: set[str] | None = None,
) -> dict:
    """
    为每张表和其每列，根据 Pattern 约束生成候选集合。
    返回结构示例:
    {
      "Paper": {
        "pattern": "SE",
        "table_class_candidates": [...],
        "columns": {
          "has_a_paper_title": {
            "role": "data_attr",
            "candidates": [...]
          },
          "has_an_abstract": {
            "role": "fk_obj",
            "ref_table": "Abstract",
            "candidates": [...]
          }
        }
      },
      "has_members": {
        "pattern": "SR",
        "sr_prop_candidates": [...],
        "fk1": {...},
        "fk2": {...}
      }
    }
    """
    classes        = ontology["classes"]
    object_props   = ontology["object_properties"]
    datatype_props = ontology["datatype_properties"]

    disabled_patterns = {p.upper() for p in (disabled_patterns or set())}
    candidates = {}

    for table_name, table_info in enriched_schema.items():
        pattern = pattern_result.get(table_name, "SE")
        pattern_norm = (pattern or "SE")
        if pattern_norm.upper() in disabled_patterns:
            continue
        cols    = table_info.get("columns", {})
        pks     = set(table_info.get("primary_key", []))
        fks     = table_info.get("foreign_keys", [])

        fk_cols, fk_constraints = _prepare_fk_metadata(
            fks,
            enriched_schema,
            table_name,
        )

        column_op_enabled = "COLUMN_OP" not in disabled_patterns

        # SE：主实体表；也包括 has_xxx(EntityFK, VALUE) 这种多值数据属性表
        if pattern == "SE":
            relation_evidence = _key_only_relation_evidence(
                table_name,
                cols,
                pks,
                classes,
                object_props,
                ontology,
            )
            if relation_evidence:
                entry = _handle_key_only_relation(
                    table_name,
                    cols,
                    pks,
                    fk_cols,
                    classes,
                    object_props,
                    enriched_schema,
                    ontology,
                    fk_constraints,
                    relation_evidence,
                )
            elif _is_value_attr_table(cols, pks, fk_cols):
                entry = _handle_value_attr_table(
                    table_name, cols, pks, fk_cols,
                    classes, datatype_props,
                    ontology=ontology,
                )
            else:
                entry = _handle_SE(
                    table_name, cols, pks, fk_cols,
                    classes, object_props, datatype_props,
                    ontology,
                    enable_column_op=column_op_enabled,
                    fk_constraints=fk_constraints,
                )

        #SH：子类继承表
        elif pattern == "SH":
            entry = _handle_SH(
                table_name, cols, pks, fk_cols,
                classes, object_props, datatype_props,
                enriched_schema,
                ontology,
                enable_column_op=column_op_enabled,
                fk_constraints=fk_constraints,
            )

        #SR：纯关联表
        elif pattern == "SR":
            entry = _handle_SR(
                table_name, cols, pks, fk_cols,
                classes, object_props,
                enriched_schema,
                ontology,
                fk_constraints=fk_constraints,
            )

        else:
            entry = {"pattern": pattern, "note": "未知 Pattern，跳过"}

        # Keep the complete physical constraints available to downstream
        # diagnostics/replay; OP mapping still reads the enriched schema as
        # its authoritative source.
        entry["fk_constraints"] = fk_constraints
        candidates[table_name] = entry

    return candidates

# 各 Pattern 处理函数

def _is_literal_value_type(col_type: str | None) -> bool:
    t = (col_type or "").lower()
    return any(k in t for k in (
        "char", "text", "string", "date", "time", "bool", "json", "xml"
    ))


def _is_value_attr_table(cols: dict, pks: set, fk_cols: dict) -> bool:
    """
    多值数据属性表：一列 FK 指向实体，一列 literal 值，两列通常共同组成 PK。
    这不是 SR/SRm；表级仍为 SE，生成时挂到被引用实体的 DataProperty 上。
    """
    col_names = set(cols.keys())
    if len(col_names) != 2 or pks != col_names or len(fk_cols) != 1:
        return False
    value_cols = [c for c in col_names if c not in fk_cols]
    return len(value_cols) == 1 and _is_literal_value_type(cols.get(value_cols[0]))


def _key_only_relation_evidence(
    table_name: str,
    cols: dict,
    pks: set,
    classes: list,
    object_props: dict,
    ontology: dict | None,
) -> dict | None:
    """Detect a relation-shaped table that must not become an SE entity.

    A table consisting solely of a composite key can encode an n-ary/bridge
    relation even when FK discovery is incomplete.  We only override an SE
    classification when the table name uniquely and strongly names an OWL
    ObjectProperty while the ontology supplies no comparable Class name.
    Missing endpoints remain an explicit abstention; they are never guessed
    from key order or evaluator behavior.
    """
    if len(pks) < 2 or set(cols) != set(pks):
        return None

    class_candidates = _rank_class_candidates(
        table_name,
        classes,
        top_k=max(1, len(classes or [])),
    )
    normalized_table = _norm_identifier(table_name)
    if any(
        _norm_identifier(candidate.get("local_name")) == normalized_table
        for candidate in class_candidates
    ):
        return None

    op_candidates = _rank_object_prop_candidates(
        table_name,
        object_props,
        domain_hint=None,
        range_hint=None,
        top_k=max(1, len(object_props or {})),
        ontology=ontology,
    )
    op_candidates.sort(
        key=lambda candidate: (
            float(candidate.get("name_score", 0.0) or 0.0),
            float(candidate.get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    if not op_candidates:
        return None
    top = op_candidates[0]
    top_name_score = float(top.get("name_score", 0.0) or 0.0)
    second_name_score = (
        float(op_candidates[1].get("name_score", 0.0) or 0.0)
        if len(op_candidates) > 1
        else 0.0
    )
    top_class_score = max(
        (float(item.get("score", 0.0) or 0.0) for item in class_candidates),
        default=0.0,
    )
    exact_properties = [
        candidate
        for candidate in op_candidates
        if _norm_identifier(candidate.get("local_name")) == normalized_table
    ]
    unique_exact_property = bool(
        len(exact_properties) == 1
        and exact_properties[0].get("uri") == top.get("uri")
    )
    if (
        top_name_score < 0.90
        or (
            not unique_exact_property
            and top_name_score - second_name_score < 0.20
        )
        or (
            top_class_score >= 0.75
            and top_name_score - top_class_score < 0.20
        )
    ):
        return None

    return {
        "kind": "key_only_composite_relation",
        "object_property_candidate": top,
        "object_property_name_score": top_name_score,
        "object_property_runner_up_score": second_name_score,
        "best_class_score": top_class_score,
        "endpoint_policy": "physical_fk_only; abstain_when_incomplete",
    }


def _handle_key_only_relation(
    table_name,
    cols,
    pks,
    fk_cols,
    classes,
    object_props,
    enriched_schema,
    ontology,
    fk_constraints,
    structural_evidence,
):
    """Create an SR candidate while retaining an explicit endpoint guard."""
    entry = _handle_SR(
        table_name,
        cols,
        pks,
        fk_cols,
        classes,
        object_props,
        enriched_schema,
        ontology,
        fk_constraints=fk_constraints,
    )
    strong_uri = structural_evidence["object_property_candidate"].get("uri")
    entry["sr_prop_candidates"] = [
        candidate
        for candidate in entry.get("sr_prop_candidates", [])
        if candidate.get("uri") == strong_uri
    ]
    complete_endpoints = sum(
        1
        for endpoint in (entry.get("fk1", {}), entry.get("fk2", {}))
        if endpoint.get("column") and endpoint.get("ref_table")
    )
    entry.update(
        {
            "table_kind": "key_only_relation",
            "source_pattern": "SE",
            "relation_kind": (
                "key_only_relation"
                if complete_endpoints == 2
                else "key_only_endpoint_abstain"
            ),
            "relation_status": (
                "candidate"
                if complete_endpoints == 2
                else "abstained_incomplete_physical_endpoints"
            ),
            "structural_relation_evidence": structural_evidence,
        }
    )
    return entry


def _handle_value_attr_table(
    table_name,
    cols,
    pks,
    fk_cols,
    classes,
    datatype_props,
    ontology=None,
):
    fk_col, fk_info = next(iter(fk_cols.items()))
    value_col = next(c for c in cols if c != fk_col)
    ref_table = fk_info.get("ref_table", "")
    owner_class_candidates = _rank_class_candidates(
        ref_table,
        classes,
        top_k=3,
    )
    owner_class_uri = owner_class_candidates[0]["uri"] if owner_class_candidates else None

    # 属性名优先来自表名：has_an_email(Person, VALUE) 表达的是 :has_an_email。
    prop_candidates = _rank_datatype_prop_candidates(
        table_name,
        datatype_props,
        domain_hint=owner_class_uri,
        ontology=ontology,
    )
    if not prop_candidates:
        prop_candidates = _rank_datatype_prop_candidates(
            value_col,
            datatype_props,
            domain_hint=owner_class_uri,
            ontology=ontology,
        )

    return {
        "pattern": "SE",
        "table_kind": "value_attr",
        "fk": {
            "column": fk_col,
            "ref_table": ref_table,
            "owner_class_candidates": owner_class_candidates,
            "column_type": cols.get(fk_col),
        },
        "value_column": value_col,
        "value_column_type": cols.get(value_col),
        "property_candidates": prop_candidates,
    }

def _handle_SE(table_name, cols, pks, fk_cols,
               classes, object_props, datatype_props,
               ontology=None, enable_column_op=True, fk_constraints=None):
    """
    SE: 表 → Class；非FK列 → dataProperty；FK列 → objectProperty
    数据库里最常见的表，如 Person、Paper
    """
    # 1. 表 → Class 候选
    table_cls_candidates = _rank_class_candidates(table_name, classes)
    best_class_uri = table_cls_candidates[0]["uri"] if table_cls_candidates else None

    col_entries = {}
    for col_name in cols:
        if col_name in fk_cols:
            # Identity and predicate semantics are orthogonal.  A column that
            # belongs to the physical PK still needs ObjectProperty candidates
            # when it is also a FK; subject identity continues to come from the
            # schema primary_key contract downstream.
            if not enable_column_op:
                col_entries[col_name] = {
                    "role": "fk_disabled",
                    "identity_part": col_name in pks,
                    "ref_table": fk_cols[col_name].get("ref_table", ""),
                    "candidates": [],
                    "column_type": cols.get(col_name),
                    "fk_memberships": list(
                        fk_cols[col_name].get("fk_memberships", []) or []
                    ),
                    "composite_memberships": list(
                        fk_cols[col_name].get("composite_memberships", []) or []
                    ),
                }
                continue
            # FK 列 → objectProperty 候选
            ref_table = fk_cols[col_name].get("ref_table", "")
            ref_col = fk_cols[col_name].get("ref_col", "")
            ref_class_candidates = _rank_class_candidates(ref_table, classes, top_k=3)
            best_ref_class = ref_class_candidates[0]["uri"] if ref_class_candidates else None

            op_candidates = _rank_object_prop_candidates(
                col_name, object_props,
                domain_hint=best_class_uri,
                range_hint=best_ref_class,
                ontology=ontology,
            )
            identity_part = col_name in pks
            dp_candidates = (
                _rank_identity_datatype_prop_candidates(
                    col_name,
                    datatype_props,
                    domain_hint=best_class_uri,
                    ontology=ontology,
                )
                if identity_part
                else []
            )
            col_entries[col_name] = {
                "role": "fk_obj",
                "identity_part": identity_part,
                "ref_table": ref_table,
                "ref_col": ref_col,
                "constraint_name": fk_cols[col_name].get("constraint_name", ""),
                "fk_arity": fk_cols[col_name].get("fk_arity", 1),
                "ref_class_candidates": ref_class_candidates,
                "candidates": op_candidates,
                "dp_candidates": dp_candidates,
                "column_type": cols.get(col_name),
                "fk_memberships": list(
                    fk_cols[col_name].get("fk_memberships", []) or []
                ),
                "composite_memberships": list(
                    fk_cols[col_name].get("composite_memberships", []) or []
                ),
            }
        elif col_name in pks:
            # A non-FK identifier can additionally denote a DatatypeProperty,
            # but generic identity wording is not literal-property evidence.
            # The identity ranker returns at most one candidate backed by both
            # explicit lexical semantics and an ontology-compatible domain.
            dp_candidates = _rank_identity_datatype_prop_candidates(
                col_name,
                datatype_props,
                domain_hint=best_class_uri,
                ontology=ontology,
            )
            col_entries[col_name] = {
                "role": "pk",
                "identity_part": True,
                "candidates": dp_candidates,
                "column_type": cols.get(col_name)
            }
        else:
            # 普通数据列 → dataProperty 候选
            dp_candidates = _rank_datatype_prop_candidates(
                col_name, datatype_props,
                domain_hint=best_class_uri,
                ontology=ontology,
            )
            col_entries[col_name] = {
                "role": "data_attr",
                "candidates": dp_candidates,
                # 为 BOOL/TYPE 判别列预留 class 候选（供 DP 映射/真实值增强使用）
                "class_candidates": _rank_class_candidates(col_name, classes, top_k=20),
                "column_type": cols.get(col_name)
            }

    return {
        "pattern": "SE",
        "table_class_candidates": table_cls_candidates,
        "columns": col_entries
    }


def _handle_SH(table_name, cols, pks, fk_cols,
               classes, object_props, datatype_props,
               enriched_schema,
               ontology=None, enable_column_op=True, fk_constraints=None):
    """
    SH: 表 → 子类（subClassOf 某父类）；
        PK 列继承父类 IRI，不单独映射；
        非PK非FK列 → dataProperty；
        非PK的FK列 → objectProperty。
    """
    # Find the *inherited* FK, not merely any FK that happens to be part of a
    # composite primary key.  A SH child can have role/association keys in the
    # PK as well (e.g. ``(entity_id, reviewer_id)``); treating every PK member
    # as inherited silently discards legitimate object properties.
    table_info = (enriched_schema or {}).get(table_name, {}) or {}
    physical_fks = list(table_info.get("foreign_keys", []) or [])
    for fk in physical_fks:
        if "ref_table" not in fk:
            fk["ref_table"] = fk.get("references_table") or ""
        if "ref_col" not in fk:
            fk["ref_col"] = fk.get("references_column") or ""

    # Compute the child candidate before choosing an inherited parent.  A
    # relational subtype can expose several same-named PK FKs (one to a
    # direct parent and another to a transitive ancestor); OWL hierarchy then
    # safely breaks an otherwise structural tie in favour of the direct
    # superclass.
    sub_class_candidates = _rank_table_class_candidates(table_name, classes)
    best_sub_class, _child_resolution = _resolve_table_class(table_name, classes)

    parent_candidates = []
    for fk in physical_fks:
        source_col = fk.get("column")
        ref_table = fk.get("ref_table") or ""
        ref_col = fk.get("ref_col") or ""
        if source_col not in pks or not ref_table:
            continue
        ref_pks = set(((enriched_schema or {}).get(ref_table, {}) or {}).get("primary_key", []) or [])
        # Same-named PK-to-PK FK is the strongest generic relational witness
        # for table-per-subclass inheritance.  Referencing a target PK still
        # contributes a weak fallback for schemas that rename inherited keys.
        same_key_name = bool(
            ref_col and _norm_identifier(source_col) == _norm_identifier(ref_col)
        )
        target_pk_match = bool(ref_col and ref_col in ref_pks)
        structural_score = (4 if same_key_name and target_pk_match else 2 if target_pk_match else 0)
        if structural_score:
            ref_class, resolution = _resolve_table_class(ref_table, classes)
            parent_candidates.append({
                "fk": fk,
                "structural_score": structural_score,
                "class_resolution_score": {
                    "exact": 2,
                    "fuzzy": 1,
                }.get(resolution, 0),
                "ancestor_distance": _ancestor_distance(
                    best_sub_class,
                    ref_class,
                    ontology,
                ),
            })

    selected_parent, ambiguous_parent_columns = _unique_best_parent_candidate(
        parent_candidates
    )
    inherited_fk = selected_parent["fk"] if selected_parent else None

    parent_table = (inherited_fk or {}).get("ref_table", "")
    inherited_pk_columns = {
        candidate["fk"].get("column")
        for candidate in parent_candidates
        for fk in [candidate["fk"]]
        if fk.get("ref_table") == parent_table
        and _norm_identifier(fk.get("column")) == _norm_identifier(fk.get("ref_col"))
    }
    if not inherited_pk_columns and inherited_fk and inherited_fk.get("column"):
        inherited_pk_columns.add(inherited_fk["column"])

    # 父类候选
    parent_class_candidates = []
    if parent_table:
        parent_class_candidates = _rank_table_class_candidates(
            parent_table,
            classes,
            top_k=3,
        )

    # A coarse lexical child candidate (e.g. ``Location`` for a political
    # subtype) must not outrank a candidate that is structurally compatible
    # with the physical parent entity.  Keep all candidates for LLM review,
    # but place compatible ones first for deterministic/offline fallback.
    parent_hint = (
        parent_class_candidates[0].get("uri")
        if parent_class_candidates
        else None
    )
    if parent_hint and sub_class_candidates:
        compatible = [
            candidate
            for candidate in sub_class_candidates
            if _sh_child_parent_compatible(
                candidate.get("uri"), parent_hint, ontology
            )
        ]
        incompatible = [
            candidate
            for candidate in sub_class_candidates
            if candidate not in compatible
        ]
        if compatible:
            sub_class_candidates = compatible + incompatible
            best_sub_class = sub_class_candidates[0].get("uri")

    # SH data properties are usually declared on the inherited entity class,
    # while the child table name may be ambiguous (or the ontology may expose
    # a measurement/reified subclass with no DP domain declarations).  Keep
    # both pieces of evidence in candidate ranking; the downstream matcher can
    # still choose a child-specific property when its domain is explicit.
    parent_domain_uri = (
        parent_class_candidates[0].get("uri")
        if parent_class_candidates
        else None
    )
    dp_domain_hints = [
        hint for hint in (best_sub_class, parent_domain_uri) if hint
    ]

    col_entries = {}
    for col_name in cols:
        if col_name in inherited_pk_columns:
            # SH 的 PK 是继承用的，不需要独立属性候选
            col_entries[col_name] = {
                "role": "sh_inherited_pk",
                "identity_part": True,
                "note": "使用父类 IRI 模板，不单独映射",
                "candidates": [],
                "column_type": cols.get(col_name),
                "fk_memberships": list(
                    fk_cols.get(col_name, {}).get("fk_memberships", []) or []
                ),
            }
        elif col_name in fk_cols:
            if not enable_column_op:
                col_entries[col_name] = {
                    "role": "fk_disabled",
                    "identity_part": col_name in pks,
                    "inheritance_ambiguous": col_name in ambiguous_parent_columns,
                    "ref_table": fk_cols[col_name].get("ref_table", ""),
                    "candidates": [],
                    "column_type": cols.get(col_name),
                    "fk_memberships": list(
                        fk_cols[col_name].get("fk_memberships", []) or []
                    ),
                    "composite_memberships": list(
                        fk_cols[col_name].get("composite_memberships", []) or []
                    ),
                }
                continue
            # 非PK的FK列 → objectProperty
            ref_table = fk_cols[col_name].get("ref_table", "")
            ref_col = fk_cols[col_name].get("ref_col", "")
            ref_class_candidates = _rank_class_candidates(ref_table, classes, top_k=3)
            best_ref_class = ref_class_candidates[0]["uri"] if ref_class_candidates else None
            op_candidates = _rank_object_prop_candidates(
                col_name, object_props,
                domain_hint=best_sub_class,
                range_hint=best_ref_class,
                ontology=ontology,
            )
            identity_part = col_name in pks
            dp_candidates = (
                _rank_identity_datatype_prop_candidates(
                    col_name,
                    datatype_props,
                    domain_hint=best_sub_class,
                    domain_hints=dp_domain_hints,
                    ontology=ontology,
                )
                if identity_part
                else []
            )
            col_entries[col_name] = {
                "role": "fk_obj",
                "identity_part": identity_part,
                "inheritance_ambiguous": col_name in ambiguous_parent_columns,
                "ref_table": ref_table,
                "ref_col": ref_col,
                "constraint_name": fk_cols[col_name].get("constraint_name", ""),
                "fk_arity": fk_cols[col_name].get("fk_arity", 1),
                "ref_class_candidates": ref_class_candidates,
                "candidates": op_candidates,
                "dp_candidates": dp_candidates,
                "column_type": cols.get(col_name),
                "fk_memberships": list(
                    fk_cols[col_name].get("fk_memberships", []) or []
                ),
                "composite_memberships": list(
                    fk_cols[col_name].get("composite_memberships", []) or []
                ),
            }
        elif col_name in pks:
            # Preserve a literal track only when it has evidence independent
            # of the column's subject-identity role.
            dp_candidates = _rank_identity_datatype_prop_candidates(
                col_name,
                datatype_props,
                domain_hint=best_sub_class,
                domain_hints=dp_domain_hints,
                ontology=ontology,
            )
            col_entries[col_name] = {
                "role": "pk",
                "identity_part": True,
                "candidates": dp_candidates,
                "column_type": cols.get(col_name),
            }
        else:
            # 普通数据列 → dataProperty
            dp_candidates = _rank_datatype_prop_candidates(
                col_name, datatype_props,
                domain_hint=best_sub_class,
                domain_hints=dp_domain_hints,
                ontology=ontology,
            )
            col_entries[col_name] = {
                "role": "data_attr",
                "candidates": dp_candidates,
                "class_candidates": _rank_class_candidates(col_name, classes, top_k=20),
                "column_type": cols.get(col_name)
            }

    return {
        "pattern": "SH",
        "sub_class_candidates": sub_class_candidates,
        "parent_table": parent_table,
        "parent_class_candidates": parent_class_candidates,
        "columns": col_entries
    }


def _handle_SR(table_name, cols, pks, fk_cols,
               classes, object_props,
               enriched_schema,
               ontology=None, fk_constraints=None):
    """
    SR: 整张表 → 一个 objectProperty。
        domain ≈ 第一个FK引用表的Class；
        range  ≈ 第二个FK引用表的Class。
    """
    if fk_constraints:
        # One representative per physical constraint, preserving member order.
        fk_list = [members[0] for members in fk_constraints.values() if members]
    else:
        fk_list = list(fk_cols.values())
    partial_fk = False
    value_col = None

    domain_hint, range_hint = None, None
    fk1_info, fk2_info = None, None

    if len(fk_list) >= 2:
        fk1_info = fk_list[0]
        fk2_info = fk_list[1]
        ref1 = fk1_info.get("ref_table", "")
        ref2 = fk2_info.get("ref_table", "")
        ref1_cls = _rank_class_candidates(ref1, classes, top_k=1)
        ref2_cls = _rank_class_candidates(ref2, classes, top_k=1)
        domain_hint = ref1_cls[0]["uri"] if ref1_cls else None
        range_hint  = ref2_cls[0]["uri"] if ref2_cls else None
    elif len(fk_list) == 1:
        fk1_info = fk_list[0]
        ref1 = fk1_info.get("ref_table", "")
        ref1_cls = _rank_class_candidates(ref1, classes, top_k=1)
        domain_hint = ref1_cls[0]["uri"] if ref1_cls else None
        if len(cols) == 2:
            partial_fk = True
            non_fk_cols = [c for c in cols if c not in fk_cols]
            value_col = non_fk_cols[0] if non_fk_cols else None

    # 整张表 → objectProperty 候选
    op_candidates = _rank_object_prop_candidates(
        table_name, object_props,
        domain_hint=domain_hint,
        range_hint=range_hint,
        ontology=ontology,
    )

    return {
        "pattern": "SR",
        "relation_kind": "partial_fk" if partial_fk else "full_fk",
        "fk1": {
            "column": fk1_info["column"] if fk1_info else None,
            "ref_table": fk1_info.get("ref_table") if fk1_info else None,
            "domain_class_hint": domain_hint
        },
        "fk2": {
            "column": fk2_info["column"] if fk2_info else value_col,
            "ref_table": fk2_info.get("ref_table") if fk2_info else None,
            "range_class_hint": range_hint
        },
        "partial_value_column": value_col,
        "sr_prop_candidates": op_candidates
    }


if __name__ == "__main__":
    from utils.db_utils import read_schema
    from utils.ontology_utils import read_ontology
    from FKCompletion_agent import allocate_targets_and_shooters, discover_implicit_foreign_keys
    from utils.merge_fks import merge_fks_into_schema

    ONTOLOGY_PATH = "input/conference_nofks/ontology.ttl"

    # 1. 读取并补全 schema
    schema = read_schema()
    allocation = allocate_targets_and_shooters(schema)
    discovered_fks = discover_implicit_foreign_keys(allocation, schema_name="public")
    enriched_schema = merge_fks_into_schema(schema, discovered_fks)

    # 2. 读取本体
    ontology = read_ontology(ONTOLOGY_PATH)

    # 3. 使用 classify_agent 输出的 pattern
    pattern_result = {
        'Abstract': 'SH', 'Accepted_contribution': 'SH',
        'Active_conference_partici': 'SH', 'Call_for_paper': 'SH',
        'Call_for_participation': 'SH', 'Camera_ready_contribution': 'SH',
        'Chair': 'SH', 'Co-chair': 'SH',
        'Committee': 'SE', 'Committee_member': 'SH',
        'Conference': 'SE', 'Conference_announcement': 'SH',
        'Conference_applicant': 'SH', 'Conference_contribution': 'SH',
        'Conference_contributor': 'SH', 'Conference_document': 'SE',
        'Conference_fees': 'SE', 'Conference_part': 'SE',
        'Conference_participant': 'SH', 'Conference_proceedings': 'SE',
        'Conference_volume': 'SH', 'Conference_www': 'SH',
        'Contribution_1th-author': 'SH', 'Contribution_co-author': 'SH',
        'Early_paid_applicant': 'SH', 'Extended_abstract': 'SH',
        'Important_dates': 'SE', 'Information_for_participa': 'SH',
        'Invited_speaker': 'SH', 'Invited_talk': 'SH',
        'Late_paid_applicant': 'SH', 'Organization': 'SE',
        'Organizer': 'SH', 'Organizing_committee': 'SH',
        'Paid_applicant': 'SH', 'Paper': 'SH',
        'Passive_conference_partic': 'SH', 'Person': 'SE',
        'Poster': 'SH', 'Presentation': 'SH',
        'Program_committee': 'SH', 'Publisher': 'SE',
        'Registeered_applicant': 'SH', 'Regular_author': 'SH',
        'Regular_contribution': 'SH', 'Rejected_contribution': 'SH',
        'Review': 'SH', 'Review_expertise': 'SE',
        'Review_preference': 'SE', 'Reviewed_contribution': 'SH',
        'Reviewer': 'SH', 'Steering_committee': 'SH',
        'Submitted_contribution': 'SH', 'Topic': 'SE',
        'Track': 'SH', 'Track-workshop_chair': 'SH',
        'Tutorial': 'SH', 'Workshop': 'SH',
        'Written_contribution': 'SH',
        'belongs_to_reviewers': 'SE',
        'contributes': 'SR',
        'has_a_committee_co-chair': 'SR',
        'has_a_track-workshop-tuto': 'SR',
        'has_an_email': 'SE',
        'has_members': 'SR',
        'invited_by': 'SR'
    }

    # 4. 生成候选
    candidates = generate_candidates(enriched_schema, pattern_result, ontology)

    # 5. 打印结果
    demo_tables = ["Paper", "has_members", "Organizer", "has_an_email", "belongs_to_reviewers"]
    for t in demo_tables:
        if t in candidates:
            print(f"\n{'='*60}")
            print(f"表: {t}")
            print(json.dumps(candidates[t], indent=2, ensure_ascii=False))

    # 输出完整结果
    with open("DPMapping/candidates_output.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print("\n\n完整候选集已保存到 candidates_output.json")
