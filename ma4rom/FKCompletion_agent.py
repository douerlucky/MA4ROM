import json
from typing import Dict, Any

import psycopg2

from utils.db_utils import read_schema, DB_CONFIG
from utils.candidate_ranking import rank_class_candidates, rank_object_prop_candidates
from utils.llm_client import call_llm
from utils.name_similarity import name_overlap
from utils.ontology_utils import hint_match, local_name, read_ontology
from config import (
    DB_SCHEMA_NAME,
    ONTOLOGY_PATH,
    FK_COMPLETION_ONTOLOGY_FALLBACK_MIN_SCORE,
    FK_COMPLETION_LLM_EMPTY_MIN_NAME_SCORE,
    FK_COMPLETION_EXCLUDED_COLUMNS,
    FK_COMPLETION_EXCLUDED_TYPES,
    FK_COMPLETION_IND_THRESHOLD,
    FK_COMPLETION_OP_NAME_MIN_SCORE,
    FK_COMPLETION_OP_SIDE_MIN_SCORE,
    FK_COMPLETION_ROLE_CLASS_MIN_SCORE,
    FK_COMPLETION_STRICT_OP_NAME_MIN_SCORE,
)        # schema 名与阈值从 config 读取


def allocate_targets_and_shooters(schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    将全库的列严格划分为靶点集合target与射手shooter集合。
    target是单列主键（候选被引用端）
    shooter非主键、非已知FK、且类型不过滤的列（候选引用端）
    """
    targets = {}
    shooters = {}

    for table_name, table_info in schema.items():
        columns = table_info.get("columns", {})
        pks = table_info.get("primary_key", [])
        fks = table_info.get("foreign_keys", [])

        existing_fk_cols = {fk["column"] for fk in fks}

        # 1. 甄别靶点 Targets（单列主键）
        is_single_pk = (len(pks) == 1)

        if is_single_pk:
            pk_col = pks[0]
            targets[f"{table_name}.{pk_col}"] = {
                "table": table_name,
                "column": pk_col,
                "type": columns.get(pk_col)
            }

        # 2. 甄别射手 Shooters
        for col_name, col_type in columns.items():
            if col_name in existing_fk_cols:
                continue
            if col_name.lower() in FK_COMPLETION_EXCLUDED_COLUMNS:
                continue
            if col_type in FK_COMPLETION_EXCLUDED_TYPES:
                continue

            shooters[f"{table_name}.{col_name}"] = {
                "table": table_name,
                "column": col_name,
                "type": col_type
            }

    return {
        "targets": targets,
        "shooters": shooters
    }


def fetch_column_data_as_set(conn, schema_name, table, column):
    """
    执行 SQL 去重查询，将一列的真实数据拉入内存变成 Set。
    """
    query = f'SELECT DISTINCT "{column}" FROM "{schema_name}"."{table}" WHERE "{column}" IS NOT NULL;'
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return set([row[0] for row in cur.fetchall()])
    except Exception as e:
        print(f"读取 {table}.{column} 失败: {e}")
        conn.rollback()
        return set()


def fetch_table_row_count(conn, schema_name, table):
    query = f'SELECT COUNT(*) FROM "{schema_name}"."{table}";'
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return int(cur.fetchone()[0])
    except Exception as e:
        print(f"读取 {table} 行数失败: {e}")
        conn.rollback()
        return None


def _normalize_class_lookup_name(name: str) -> str:
    text = (name or "").strip().lower().replace("-", "_")
    text = "_".join(seg for seg in text.split("_") if seg)
    if text.endswith("ies") and len(text) > 3:
        return text[:-3] + "y"
    if text.endswith("s") and not text.endswith("ss") and len(text) > 1:
        return text[:-1]
    return text


def _class_uri_by_local_name(ontology: dict) -> dict[str, str]:
    by_name = {}
    for uri in ontology.get("classes", []):
        local = uri.split("#")[-1]
        for key in {local.lower(), _normalize_class_lookup_name(local)}:
            by_name[key] = uri
            by_name[key.replace("_", "")] = uri
    return by_name


def _class_uri_for_table(table_name: str, ontology: dict) -> str | None:
    key = _normalize_class_lookup_name(table_name)
    by_name = _class_uri_by_local_name(ontology)
    return by_name.get(key) or by_name.get(key.replace("_", ""))


def _infer_role_class_uri(column_name: str, ontology: dict) -> str | None:
    """
    从非 ID 列名推断它语义上指向的本体 Class。
    例如 hasAuthor -> Author，Reviewer -> Reviewer。
    """
    by_local = _class_uri_by_local_name(ontology)
    exact = by_local.get((column_name or "").lower())
    if exact:
        return exact

    candidates = rank_class_candidates(
        column_name,
        ontology.get("classes", []),
        top_k=1,
    )
    if not candidates:
        return None
    best = candidates[0]
    if best.get("score", 0) < FK_COMPLETION_ROLE_CLASS_MIN_SCORE:
        return None
    return best.get("uri")


def _is_class_ancestor(child_uri: str | None, ancestor_uri: str | None, ontology: dict) -> bool:
    if not child_uri or not ancestor_uri:
        return False
    return ancestor_uri in ontology.get("ancestors_of", {}).get(child_uri, [])


def _class_match_score(hint_uri: str | None, candidate_uri: str | None, ontology: dict) -> float:
    if not hint_uri or not candidate_uri:
        return 0.0
    if hint_uri == candidate_uri:
        return 1.0
    if _is_class_ancestor(hint_uri, candidate_uri, ontology):
        return 0.85
    if _is_class_ancestor(candidate_uri, hint_uri, ontology):
        return 0.75
    return 0.0


def _is_same_or_subclass(child_uri: str | None, parent_uri: str | None, ontology: dict) -> bool:
    if not child_uri or not parent_uri:
        return False
    return child_uri == parent_uri or _is_class_ancestor(child_uri, parent_uri, ontology)


def _mark_targets(candidates: list[dict], support: str) -> list[dict]:
    for cand in candidates:
        cand["semantic_support"] = support
    return candidates


def _nearest_targets(candidates: list[dict]) -> list[dict]:
    """
    多个本体支持候选同时成立时，优先保留值域最具体的父表。
    若最小 target_size 并列，则全部保留，支持同一列多物理 FK。
    """
    if not candidates:
        return []
    min_size = min(c.get("target_size", 0) for c in candidates)
    return [c for c in candidates if c.get("target_size", 0) == min_size]


def _prune_more_general_semantic_targets(candidates: list[dict], ontology: dict) -> list[dict]:
    """
    同一列的候选里，如果 A 是 B 的父类且两者语义分接近，保留更具体的 B。
    例如 Administrator 可物理落到 User/Person 时，优先 User，过滤 Person。
    """
    kept = []
    for cand in candidates:
        cand_cls = cand.get("_target_class")
        cand_score = cand.get("semantic_score", 0)
        has_more_specific = False
        for other in candidates:
            if other is cand:
                continue
            other_cls = other.get("_target_class")
            other_score = other.get("semantic_score", 0)
            if other_score + 0.05 < cand_score:
                continue
            if _is_same_or_subclass(other_cls, cand_cls, ontology) and other_cls != cand_cls:
                has_more_specific = True
                break
        if not has_more_specific:
            kept.append(cand)
    return kept


def _expand_named_class_values(values: list, ontology: dict) -> list[str]:
    expanded = []
    union_members = ontology.get("union_members", {})
    for value in values or []:
        members = union_members.get(value)
        if members:
            expanded.extend(members)
        else:
            expanded.append(value)
    return expanded


def _has_table_for_class(class_uri: str, table_names: set[str], ontology: dict) -> bool:
    by_local = _class_uri_by_local_name(ontology)
    return any(by_local.get((table_name or "").lower()) == class_uri for table_name in table_names)


def _has_direct_target_for_class(class_uri: str, targets: dict, ontology: dict) -> bool:
    return any(
        _class_uri_for_table(target_info["table"], ontology) == class_uri
        for target_info in targets.values()
    )


def _target_endpoint_match_score(
    target_uri: str,
    endpoint_values: list,
    targets: dict,
    table_names: set[str],
    ontology: dict,
) -> tuple[float, str | None]:
    """
    严格按 OP 端点类判断 target。
    若 OP 端点类有直接物理表，则只允许直接表，不自动退到父类表。
    若没有直接表，才允许退到最近可用父类表，例如 Administrator -> User。
    """
    best_score = 0.0
    best_endpoint = None
    for endpoint_uri in _expand_named_class_values(endpoint_values, ontology):
        if not endpoint_uri:
            continue
        if target_uri == endpoint_uri:
            return 1.0, endpoint_uri
        if _has_direct_target_for_class(endpoint_uri, targets, ontology) or _has_table_for_class(endpoint_uri, table_names, ontology):
            continue
        if _is_class_ancestor(endpoint_uri, target_uri, ontology):
            if 0.85 > best_score:
                best_score = 0.85
                best_endpoint = endpoint_uri
        if _is_class_ancestor(target_uri, endpoint_uri, ontology):
            if 0.85 > best_score:
                best_score = 0.85
                best_endpoint = endpoint_uri
    return best_score, best_endpoint


def _best_op_side_evidence(
    op_uri: str,
    op_info: dict,
    target_uri: str,
    targets: dict,
    table_names: set[str],
    ontology: dict,
) -> dict | None:
    domain_score, domain_endpoint = _target_endpoint_match_score(
        target_uri,
        op_info.get("domain", []),
        targets,
        table_names,
        ontology,
    )
    range_score, range_endpoint = _target_endpoint_match_score(
        target_uri,
        op_info.get("range", []),
        targets,
        table_names,
        ontology,
    )
    if domain_score >= range_score:
        side = "domain"
        side_score = domain_score
        endpoint_uri = domain_endpoint
    else:
        side = "range"
        side_score = range_score
        endpoint_uri = range_endpoint
    if side_score <= 0 or not endpoint_uri:
        return None
    return {
        "op": op_uri,
        "op_local": local_name(op_uri),
        "side": side,
        "endpoint_class": local_name(endpoint_uri),
        "side_score": round(side_score, 3),
    }


def _rank_ontology_fallback_targets(
    source_info: dict,
    targets: dict,
    ontology: dict,
    candidate_limit: int,
    table_names: set[str] | None = None,
) -> list[dict]:
    """
    为 IND 召回失败的空列/空表列生成本体支撑 FK 候选。
    支撑来源包括：
      - 列名对应的 role Class，目标表是该 Class 或其物理父类；
      - 列名本身像 ObjectProperty，source table class -> target class；
      - 表名像 ObjectProperty，列承担该 OP 的 domain/range 角色。
    """
    source_table = source_info["table"]
    source_column = source_info["column"]
    source_type = source_info.get("type")
    source_class = _class_uri_for_table(source_table, ontology)
    role_class = _infer_role_class_uri(source_column, ontology)
    if table_names is None:
        table_names = {info["table"] for info in targets.values()}

    candidates = []
    for target_info in targets.values():
        if source_type != target_info.get("type"):
            continue
        if source_table == target_info.get("table"):
            continue

        target_class = _class_uri_for_table(target_info["table"], ontology)
        if not target_class:
            continue

        role_score = _class_match_score(role_class, target_class, ontology)
        evidences = []
        score = 0.0

        for op_uri, op_info in ontology.get("object_properties", {}).items():
            op_local = local_name(op_uri)

            # 情况 1：普通实体表中的列名就是 OP，例如 Paper.acceptedBy。
            col_op_name_score = name_overlap(source_column, op_local)
            if source_class and col_op_name_score >= FK_COMPLETION_STRICT_OP_NAME_MIN_SCORE:
                domain_score = hint_match(source_class, op_info.get("domain", []), ontology=ontology)
                range_score, range_endpoint = _target_endpoint_match_score(
                    target_class,
                    op_info.get("range", []),
                    targets,
                    table_names,
                    ontology,
                )
                if (
                    domain_score >= FK_COMPLETION_OP_SIDE_MIN_SCORE
                    and range_score >= FK_COMPLETION_OP_SIDE_MIN_SCORE
                    and range_endpoint
                ):
                    op_score = (col_op_name_score + domain_score + range_score) / 3
                    evidences.append({
                        "type": "column_object_property",
                        "op": op_local,
                        "endpoint_class": local_name(range_endpoint),
                        "name_score": round(col_op_name_score, 3),
                        "domain_score": round(domain_score, 3),
                        "range_score": round(range_score, 3),
                    })
                    score = max(score, op_score)

            # 情况 2：关系表/动作表名就是 OP，例如 detailsEnteredBy.Administrator。
            table_op_name_score = name_overlap(source_table, op_local)
            if table_op_name_score >= FK_COMPLETION_STRICT_OP_NAME_MIN_SCORE:
                side_evidence = _best_op_side_evidence(
                    op_uri,
                    op_info,
                    target_class,
                    targets,
                    table_names,
                    ontology,
                )
                if side_evidence and side_evidence["side_score"] >= FK_COMPLETION_OP_SIDE_MIN_SCORE:
                    # 列名必须对应 OP 的 domain/range 端点角色，不能只靠父类或相似 OP 硬补。
                    endpoint_name = side_evidence.get("endpoint_class", "")
                    role_or_name_score = max(
                        name_overlap(source_column, endpoint_name),
                        name_overlap(source_column, target_info["table"])
                        if target_class == _class_uri_by_local_name(ontology).get(source_column.lower())
                        else 0,
                    )
                    if role_or_name_score >= FK_COMPLETION_STRICT_OP_NAME_MIN_SCORE:
                        op_score = (
                            table_op_name_score
                            + side_evidence["side_score"]
                            + role_or_name_score
                        ) / 3
                        evidences.append({
                            "type": "table_object_property",
                            **side_evidence,
                            "table_name_score": round(table_op_name_score, 3),
                            "role_or_name_score": round(role_or_name_score, 3),
                        })
                        score = max(score, op_score)

        if score < FK_COMPLETION_ONTOLOGY_FALLBACK_MIN_SCORE:
            continue

        candidates.append({
            "table": target_info["table"],
            "column": target_info["column"],
            "type": target_info.get("type"),
            "semantic_score": round(score, 4),
            "evidence": evidences,
            "_target_class": target_class,
        })

    candidates.sort(key=lambda x: (-x["semantic_score"], x["table"]))
    candidates = _prune_more_general_semantic_targets(candidates, ontology)
    for cand in candidates:
        cand.pop("_target_class", None)
    return candidates[:candidate_limit]


def _select_ontology_supported_targets(
    s_info: dict,
    candidate_targets: list[dict],
    ontology: dict,
) -> list[dict]:
    """
    IND 负责召回候选；本体负责判断哪些候选真的有语义支撑。
    判断顺序：
      1. ID 列：source table 是 target table 子类时，补继承 FK。
      2. 非 ID 列：列名推断 role Class，若有同名目标表，优先只补同名目标。
      3. 非 ID 列：若 role Class 没有物理表，则补最近的本体父类目标。
      4. OP 兜底：ObjectProperty 的 domain/range 支持该 source/target。
      5. 都没有本体证据时，回退 IND 排序第一名。
    """
    column_name = s_info["column"]
    source_class_uri = _class_uri_for_table(s_info["table"], ontology)

    if column_name.lower() == "id":
        inheritance_targets = [
            c for c in candidate_targets
            if _is_subclass_table(s_info["table"], c["target"]["table"], ontology)
        ]
        if inheritance_targets:
            return _mark_targets(inheritance_targets, "table_inheritance")
        # ID->ID 只用于继承/子类表；没有本体证明 source 是 target 子类时，
        # 不能退回 IND 最优候选，否则会产生 parent.ID -> child.ID 的反向外键。
        return []

    role_class_uri = _infer_role_class_uri(column_name, ontology)
    if role_class_uri:
        direct_targets = [
            c for c in candidate_targets
            if _class_uri_for_table(c["target"]["table"], ontology) == role_class_uri
        ]
        if direct_targets:
            return _mark_targets(direct_targets, "direct_role_class")

        ancestor_targets = [
            c for c in candidate_targets
            if _is_class_ancestor(
                role_class_uri,
                _class_uri_for_table(c["target"]["table"], ontology),
                ontology,
            )
        ]
        nearest_ancestors = _nearest_targets(ancestor_targets)
        if nearest_ancestors:
            return _mark_targets(nearest_ancestors, "role_class_ancestor")

    op_supported = []
    for cand in candidate_targets:
        target_class_uri = _class_uri_for_table(cand["target"]["table"], ontology)
        op_candidates = rank_object_prop_candidates(
            column_name,
            ontology.get("object_properties", {}),
            domain_hint=source_class_uri,
            range_hint=target_class_uri,
            top_k=1,
            ontology=ontology,
        )
        if not op_candidates:
            continue
        best_op = op_candidates[0]
        if (
            best_op.get("name_score", 0) >= FK_COMPLETION_OP_NAME_MIN_SCORE
            and best_op.get("domain_score", 0) >= FK_COMPLETION_OP_SIDE_MIN_SCORE
            and best_op.get("range_score", 0) >= FK_COMPLETION_OP_SIDE_MIN_SCORE
        ):
            cand["op_uri"] = best_op.get("uri")
            op_supported.append(cand)

    if op_supported:
        # OP 兜底只证明最优 IND 候选有语义支撑；多目标交给 class/继承规则处理。
        # 否则像 writtenBy 这类宽 range 容易把祖先表也一起误补出来。
        return _mark_targets([op_supported[0]], "object_property")

    return _mark_targets([candidate_targets[0]], "ind_best")


def _filter_reciprocal_pk_fks(discovered_fks: dict) -> dict:
    """
    IND 在同值域的继承表上可能推出 A.ID->B.ID 和 B.ID->A.ID。
    若本体能判断 A 是 B 的子类，则只保留 A->B，删除反向边。
    """
    try:
        ontology = read_ontology(ONTOLOGY_PATH)
    except Exception:
        return discovered_fks

    by_local = _class_uri_by_local_name(ontology)
    ancestors = ontology.get("ancestors_of", {})
    fk_pairs = {}
    for table, fks in discovered_fks.items():
        for fk in fks:
            if fk.get("column") != "ID" or fk.get("ref_col") != "ID":
                continue
            fk_pairs[(table, fk.get("ref_table"))] = fk

    to_remove = set()
    for src, ref in list(fk_pairs):
        if (ref, src) not in fk_pairs:
            continue
        src_uri = by_local.get(_normalize_class_lookup_name(src))
        ref_uri = by_local.get(_normalize_class_lookup_name(ref))
        if not src_uri or not ref_uri:
            continue
        if ref_uri in ancestors.get(src_uri, []):
            to_remove.add((ref, src))
        elif src_uri in ancestors.get(ref_uri, []):
            to_remove.add((src, ref))

    if not to_remove:
        return discovered_fks

    filtered = {}
    for table, fks in discovered_fks.items():
        kept = [
            fk for fk in fks
            if not (fk.get("column") == "ID" and fk.get("ref_col") == "ID" and (table, fk.get("ref_table")) in to_remove)
        ]
        if kept:
            filtered[table] = kept
    return filtered

#是否一个类是另一个类的祖先
def _is_subclass_table(source_table: str, target_table: str, ontology: dict) -> bool:
    source_uri = _class_uri_for_table(source_table, ontology)
    target_uri = _class_uri_for_table(target_table, ontology)

    if not source_uri or not target_uri:
        return False

    return target_uri in ontology.get("ancestors_of", {}).get(source_uri, [])


def discover_implicit_foreign_keys(allocation_result, schema_name: str = None):
    """
    核心 IND 碰撞。shooter 里的值有多少比例能在 target 里找到
    IND 包含依赖度 = 交集大小 / 射手大小，阈值 0.95

    参数:
        allocation_result: allocate_targets_and_shooters() 的输出
        schema_name:       PostgreSQL schema 名；未传入时从 config.DB_SCHEMA_NAME 读取
    """
    if schema_name is None:
        schema_name = DB_SCHEMA_NAME    #从 config 读取

    targets = allocation_result["targets"]
    shooters = allocation_result["shooters"]

    discovered_fks = {}
    ontology = read_ontology(ONTOLOGY_PATH)

    conn = psycopg2.connect(**DB_CONFIG)

    # 预加载所有靶点数据
    target_cache = {} #候选父表主键集合
    for t_key, t_info in targets.items():
        t_set = fetch_column_data_as_set(conn, schema_name, t_info["table"], t_info["column"])
        if len(t_set) > 0:
            target_cache[t_key] = {
                "info": t_info,
                "data": t_set,
                "size": len(t_set)
            }

    print(f"成功建立 {len(target_cache)} 个有效靶点。开始执行射击碰撞\n")

    #s_info里面有table名、column名和type，s_set是当前列的具体的真实值
    for s_key, s_info in shooters.items():
        s_set = fetch_column_data_as_set(conn, schema_name, s_info["table"], s_info["column"])
        if len(s_set) == 0:
            continue

        candidate_targets = []  # 当前 shooter 列满足 IND 阈值的候选目标

        for t_key, t_cache in target_cache.items():
            if (
                s_info["table"] == t_cache["info"]["table"]
                and s_info["column"] == t_cache["info"]["column"]
            ):
                continue
            if s_info["type"] != t_cache["info"]["type"]:
                continue

            t_set = t_cache["data"]
            intersection = s_set.intersection(t_set)
            ind_score = len(intersection) / len(s_set)

            if ind_score >= FK_COMPLETION_IND_THRESHOLD:
                candidate_targets.append({
                    "target": t_cache["info"],
                    "ind_score": ind_score,
                    "target_size": t_cache["size"],
                })

        candidate_targets.sort(key=lambda x: (-x["ind_score"], x["target_size"]))
        if not candidate_targets:
            continue

        selected_targets = _select_ontology_supported_targets(
            s_info,
            candidate_targets,
            ontology,
        )
        if not selected_targets:
            continue

        table_name = s_info["table"]
        if table_name not in discovered_fks:
            discovered_fks[table_name] = []

        for selected in selected_targets:
            best_target = selected["target"]
            best_ind = selected["ind_score"]

            # 最终补出来的外键字典
            discovered_fks[table_name].append({
                "column": s_info["column"],
                "ref_table": best_target["table"],
                "ref_col": best_target["column"],
                "ind_score": round(best_ind, 4),
                "semantic_support": selected.get("semantic_support", "ind_best"),
            })
            print(f"找回外键！ {s_key} ---> {best_target['table']}.{best_target['column']} (IND: {best_ind:.1%})")

    conn.close()
    return _filter_reciprocal_pk_fks(discovered_fks)


def _existing_fk_columns(discovered_fks: dict) -> dict[str, set[str]]:
    existing = {}
    for table, fks in (discovered_fks or {}).items():
        existing[table] = {fk.get("column") for fk in fks if fk.get("column")}
    return existing


def _rank_empty_fk_targets(source_info: dict, targets: dict, candidate_limit: int) -> list[dict]:
    candidates = []
    source_name = f"{source_info['table']}.{source_info['column']}"
    for target_info in targets.values():
        if source_info.get("type") != target_info.get("type"):
            continue
        if source_info.get("table") == target_info.get("table"):
            continue
        table_score = name_overlap(source_info["column"], target_info["table"])
        full_score = name_overlap(source_name, f"{target_info['table']}.{target_info['column']}")
        score = max(table_score, full_score)
        candidates.append({
            "table": target_info["table"],
            "column": target_info["column"],
            "type": target_info.get("type"),
            "name_score": round(score, 4),
        })
    candidates.sort(key=lambda x: x["name_score"], reverse=True)
    return candidates[:candidate_limit]


def discover_empty_semantic_foreign_keys(
    allocation_result,
    discovered_fks: dict | None = None,
    schema_name: str = None,
    candidate_limit: int = 20,
):
    """
    对 IND 无法处理的空表做语义 FK 补全。

    只处理整张表行数为 0 的 source 表；非空表里的空列仍交给下游语义流程，
    避免 LLM 因缺少实例证据过度猜测。
    """
    if schema_name is None:
        schema_name = DB_SCHEMA_NAME

    targets = allocation_result["targets"]
    shooters = allocation_result["shooters"]
    existing_cols = _existing_fk_columns(discovered_fks or {})

    conn = psycopg2.connect(**DB_CONFIG)
    empty_by_table = {}
    try:
        table_row_count = {}
        for s_key, s_info in shooters.items():
            if s_info["column"] in existing_cols.get(s_info["table"], set()):
                continue
            if s_info["table"] not in table_row_count:
                table_row_count[s_info["table"]] = fetch_table_row_count(conn, schema_name, s_info["table"])
            if table_row_count[s_info["table"]] != 0:
                continue
            s_set = fetch_column_data_as_set(conn, schema_name, s_info["table"], s_info["column"])
            if len(s_set) > 0:
                continue
            target_candidates = _rank_empty_fk_targets(s_info, targets, candidate_limit=candidate_limit)
            if not target_candidates:
                continue
            empty_by_table.setdefault(s_info["table"], []).append({
                "column": s_info["column"],
                "type": s_info.get("type"),
                "target_candidates": target_candidates,
            })
    finally:
        conn.close()

    if not empty_by_table:
        return {}

    table_names = sorted({info["table"] for info in targets.values()} | set(empty_by_table.keys()))
    llm_fks = {}

    for table_name, columns in empty_by_table.items():
        prompt = f"""
你是关系数据库 Schema 外键补全专家。当前数据库缺失外键，且当前 source 表是空表，以下 source 列没有任何实例值，无法用 IND 包含依赖判断。
请只基于表名、列名、类型和候选目标表语义，选择高可信的外键；不确定就跳过。

## 全部表名
{json.dumps(table_names, ensure_ascii=False)}

## 当前 source 表
{table_name}

## 空 source 列及候选目标
{json.dumps(columns, ensure_ascii=False, indent=2)}

## 约束
- 只能从每个 source 列给出的 target_candidates 中选择 references_table/references_column。
- 若 source 列不像外键，或候选目标不明确，不要输出该列。
- 对二元关系表/空关系表，若列名就是实体角色名，应优先恢复到对应实体表主键。
- 输出 JSON 对象，不要 Markdown。

## 输出格式
{{
  "foreign_keys": [
    {{
      "column": "source列名",
      "references_table": "目标表名",
      "references_column": "目标列名",
      "confidence": "high|medium|low",
      "reason": "一句话理由"
    }}
  ]
}}
"""
        try:
            parsed = call_llm(
                prompt=prompt,
                system="You are a database schema foreign-key completion expert. Output JSON only.",
                prefer_fast=False,
            )
        except Exception as e:
            print(f"  [WARN] LLM 空列 FK 补全失败 {table_name}: {e}")
            continue

        candidate_by_col = {
            col["column"]: {
                (cand["table"], cand["column"]): cand.get("name_score", 0)
                for cand in col.get("target_candidates", [])
            }
            for col in columns
        }
        for fk in parsed.get("foreign_keys", []) if isinstance(parsed, dict) else []:
            if not isinstance(fk, dict):
                continue
            col = fk.get("column")
            ref_table = fk.get("references_table")
            ref_col = fk.get("references_column") or "ID"
            confidence = str(fk.get("confidence", "")).lower()
            if confidence not in {"high", "medium"}:
                continue
            target_score = candidate_by_col.get(col, {}).get((ref_table, ref_col))
            if target_score is None:
                continue
            if target_score < FK_COMPLETION_LLM_EMPTY_MIN_NAME_SCORE:
                continue
            llm_fks.setdefault(table_name, []).append({
                "column": col,
                "ref_table": ref_table,
                "ref_col": ref_col,
                "source": "LLM_EMPTY",
                "confidence": confidence,
                "name_score": target_score,
                "reason": fk.get("reason", ""),
            })
            print(f"LLM补全空表外键: {table_name}.{col} ---> {ref_table}.{ref_col} ({confidence}, name={target_score:.2f})")

    return llm_fks


def discover_ontology_llm_fallback_foreign_keys(
    allocation_result,
    discovered_fks: dict | None = None,
    schema_name: str = None,
    candidate_limit: int = 8,
):
    """
    IND 无法召回时的保守回退：只处理无非空实例值的 source 列。

    流程：
      1. 对每个 IND 未补到的空列，用本体 OP/domain/range 和 role class 生成候选；
      2. 把候选交给 LLM 选择；
      3. LLM 只能选择候选里的目标，且只接受 high/medium。

    这样 LLM 不是自由猜 FK，而是在本体支撑候选里做裁决。
    """
    if schema_name is None:
        schema_name = DB_SCHEMA_NAME

    targets = allocation_result["targets"]
    shooters = allocation_result["shooters"]
    existing_cols = _existing_fk_columns(discovered_fks or {})
    ontology = read_ontology(ONTOLOGY_PATH)
    table_names = {
        info["table"]
        for info in list(targets.values()) + list(shooters.values())
        if info.get("table")
    }

    conn = psycopg2.connect(**DB_CONFIG)
    fallback_by_table = {}
    try:
        for s_key, s_info in shooters.items():
            if s_info["column"] in existing_cols.get(s_info["table"], set()):
                continue

            s_set = fetch_column_data_as_set(conn, schema_name, s_info["table"], s_info["column"])
            if len(s_set) > 0:
                continue

            target_candidates = _rank_ontology_fallback_targets(
                s_info,
                targets,
                ontology,
                candidate_limit=candidate_limit,
                table_names=table_names,
            )
            if not target_candidates:
                continue

            fallback_by_table.setdefault(s_info["table"], []).append({
                "column": s_info["column"],
                "type": s_info.get("type"),
                "target_candidates": target_candidates,
            })
    finally:
        conn.close()

    if not fallback_by_table:
        return {}

    llm_fks = {}
    for table_name, columns in fallback_by_table.items():
        prompt = f"""
你是数据库外键补全专家。现在 IND 无法判断这些 source 列，因为它们没有非空实例值。
下面每个候选 FK 已经由本体 ObjectProperty 的 domain/range、列名 role class、表名 OP 关系筛过。
请只在候选中选择你认为应该补成物理/逻辑外键的项；如果不确定就跳过。

## source 表
{table_name}

## 待判断列与本体候选
{json.dumps(columns, ensure_ascii=False, indent=2)}

## 约束
- 只能从 target_candidates 中选择 references_table/references_column。
- 优先选择本体 OP 的 domain/range 与列名角色同时支持的候选。
- 不要因为某个表名出现在 OP 中就机械选择所有列；不确定就跳过。
- 输出 JSON 对象，不要 Markdown。

## 输出格式
{{
  "foreign_keys": [
    {{
      "column": "source列名",
      "references_table": "目标表名",
      "references_column": "目标列名",
      "confidence": "high|medium|low",
      "reason": "一句话说明引用的本体证据"
    }}
  ]
}}
"""
        try:
            parsed = call_llm(
                prompt=prompt,
                system="You are a conservative ontology-aware database foreign-key completion expert. Output JSON only.",
                prefer_fast=False,
            )
        except Exception as e:
            print(f"  [WARN] 本体+LLM FK 回退失败 {table_name}: {e}")
            continue

        candidate_by_col = {
            col["column"]: {
                (cand["table"], cand["column"]): cand
                for cand in col.get("target_candidates", [])
            }
            for col in columns
        }
        for fk in parsed.get("foreign_keys", []) if isinstance(parsed, dict) else []:
            if not isinstance(fk, dict):
                continue
            col = fk.get("column")
            ref_table = fk.get("references_table")
            ref_col = fk.get("references_column") or "ID"
            confidence = str(fk.get("confidence", "")).lower()
            if confidence not in {"high", "medium"}:
                continue

            candidate = candidate_by_col.get(col, {}).get((ref_table, ref_col))
            if candidate is None:
                continue

            llm_fks.setdefault(table_name, []).append({
                "column": col,
                "ref_table": ref_table,
                "ref_col": ref_col,
                "source": "LLM_ONTOLOGY",
                "confidence": confidence,
                "semantic_score": candidate.get("semantic_score"),
                "reason": fk.get("reason", ""),
            })
            print(
                f"本体+LLM补全外键: {table_name}.{col} ---> "
                f"{ref_table}.{ref_col} ({confidence}, semantic={candidate.get('semantic_score')})"
            )

    return llm_fks


if __name__ == "__main__":
    schema = read_schema()

    allocation_result = allocate_targets_and_shooters(schema)
    print(json.dumps(allocation_result, indent=4, ensure_ascii=False))

    fks = discover_implicit_foreign_keys(allocation_result)
    ontology_fks = discover_ontology_llm_fallback_foreign_keys(
        allocation_result,
        discovered_fks=fks,
    )
    for table_name, fk_list in ontology_fks.items():
        fks.setdefault(table_name, []).extend(fk_list)

    print("\n逻辑外键修复字典:")
    print(json.dumps(fks, indent=4, ensure_ascii=False))
