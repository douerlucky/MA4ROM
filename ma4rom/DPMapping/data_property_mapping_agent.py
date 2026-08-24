"""
data_property_mapping_agent.py  ——  Data Property Mapping 阶段

职责：
  表  → Class URI（SE/SH）
  数据属性列 → DatatypeProperty URI
  FK列 / SR表 → 只确认 domain_class_uri + range_class_uri，保留候选集
"""

import json
import re
from config import (
    DP_MAPPING_BOOL_CLASS_BASE_WEIGHT,
    DP_MAPPING_BOOL_CLASS_EXACT_BOOST,
    DP_MAPPING_CONF_HIGH_GAP,
    DP_MAPPING_CONF_HIGH_TOP1,
    DP_MAPPING_CONF_MEDIUM_GAP,
    DP_MAPPING_CONF_MEDIUM_TOP1,
    DP_MAPPING_LOW_CONF_MIN_LOW_COLS,
    DP_MAPPING_LOW_CONF_RATIO_THRESHOLD,
)
from utils.llm_set import client
from utils.llm_client import call_llm as _call_llm
from utils.ontology_utils import are_classes_disjoint
from utils.candidate_ranking import (
    filter_semantically_admissible_datatype_candidates,
    filter_strong_identity_datatype_candidates,
)

# 工具函数
_DISCRIMINATOR_NAMES = {"type", "kind", "category", "status", "flag", "discriminator"}
_DISCRIMINATOR_SUFFIXES = (
    "type",
    "kind",
    "level",
    "typemain",
    "typepart",
    "typeen",
    "typeno",
)
_LLM_BUDGET_EXHAUSTED = False
_CONFIDENCE_EPSILON = 1e-9


def _mark_quota_if_needed(exc: Exception) -> None:
    """
    若检测到配额/余额耗尽，切换到“无 LLM”降级模式，
    避免后续每列继续触发失败请求并放大成本与延迟。
    """
    global _LLM_BUDGET_EXHAUSTED
    msg = str(exc).lower()
    if (
        "insufficient_quota" in msg
        or "quota" in msg
        or "pre-consumed quota" in msg
        or "need quota" in msg
    ):
        _LLM_BUDGET_EXHAUSTED = True


def _ns_from_uri(uri: str) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[0] + "#"
    return uri.rsplit("/", 1)[0] + "/"


def _class_uri_from_bool_col(col_name: str, ns: str) -> str | None:
    if not col_name or not col_name.lower().startswith("is_") or not ns:
        return None
    local = col_name[3:]
    if not local:
        return None
    return f"{ns}{local}"


def _is_boolean_coltype(col_type: str | None) -> bool:
    t = (col_type or "").strip().lower()
    return t == "boolean" or t == "bool"


def _uri_local_name(uri: str | None) -> str:
    if not uri:
        return ""
    if "#" in uri:
        return uri.split("#")[-1]
    return uri.rsplit("/", 1)[-1]


def _tokenize_name(text: str) -> set[str]:
    return {x for x in re.split(r"[^a-z0-9]+", (text or "").lower()) if x}


def _norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _boolean_class_candidates(
    col_name: str,
    class_uri: str,
    table_class_cands: list,
    extra_class_cands: list | None = None,
) -> list:
    """
    给布尔判别列准备 class 候选：
      - 复用当前表已有 class 候选（通用策略）
      - 用列名 token 与 class local_name token 重叠做轻量重排
    """
    col_tokens = _tokenize_name(col_name)
    if col_tokens:
        col_tokens = {t for t in col_tokens if t not in {"is", "has", "flag", "status", "type"}}

    scored = []
    seen = {}
    merged = list(table_class_cands or []) + list(extra_class_cands or [])

    for c in merged:
        uri = c.get("uri")
        if not uri:
            continue
        local = c.get("local_name") or _uri_local_name(uri)

        cls_tokens = _tokenize_name(local)
        overlap = len(col_tokens & cls_tokens) / len(col_tokens) if col_tokens else 0.0
        base_score = float(c.get("score", 0.0))
        score = max(base_score * DP_MAPPING_BOOL_CLASS_BASE_WEIGHT, overlap)

        # 关键规则：列名与类名规范化后完全一致时，强提升到首位
        # 例如 program_chair -> Program_Chair
        if _norm_name(col_name) and _norm_name(col_name) == _norm_name(local):
            score = max(score, DP_MAPPING_BOOL_CLASS_EXACT_BOOST)

        prev = seen.get(uri)
        if not prev or score > prev["score"]:
            seen[uri] = {"uri": uri, "local_name": local, "score": round(score, 3)}

    scored = list(seen.values())
    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return scored


def _enum_class_candidates(table_name: str, class_uri: str, table_class_cands: list) -> list:
    """
    为 TYPE 列准备 class 候选。
    仅使用当前表已有 class 候选（通用策略，不做领域硬编码）。
    """
    cands = []
    seen = set()

    for c in table_class_cands or []:
        uri = c.get("uri")
        if not uri or uri in seen:
            continue
        seen.add(uri)
        cands.append({"uri": uri, "local_name": c.get("local_name"), "score": c.get("score", 0.0)})

    return cands

def _is_discriminator(col_name: str) -> bool:
    """判别列：TYPE/Kind/Level 等枚举分类列。"""
    if not col_name:
        return False
    normalized = col_name.lower().replace("_", "")
    if normalized in {n.replace("_", "") for n in _DISCRIMINATOR_NAMES}:
        return True
    if normalized.endswith(_DISCRIMINATOR_SUFFIXES):
        return True
    return False

#在大模型开口说话之前，先用数学统计的方法算一算
def _precompute_confidence(candidates: list) -> str:
    if not candidates:
        return "low"
    scores = [c.get("score", 0) for c in candidates]
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0
    gap = top1 - top2
    # 论文阈值是闭区间。二进制浮点会把 0.7 - 0.5 表示成
    # 0.19999999999999996，因此比较时只补偿机器舍入误差。
    if (
        top1 + _CONFIDENCE_EPSILON >= DP_MAPPING_CONF_HIGH_TOP1
        and gap + _CONFIDENCE_EPSILON >= DP_MAPPING_CONF_HIGH_GAP
    ):
        return "high"
    elif (
        top1 + _CONFIDENCE_EPSILON >= DP_MAPPING_CONF_MEDIUM_TOP1
        and gap + _CONFIDENCE_EPSILON >= DP_MAPPING_CONF_MEDIUM_GAP
    ):
        return "medium"
    return "low"


def _named_domain_values(candidate: dict) -> list[str]:
    """Return only named OWL classes from a DP candidate's domain."""
    values = candidate.get("domain", []) or []
    return [
        str(value)
        for value in values
        if isinstance(value, str)
        and (value.startswith("http://") or value.startswith("https://") or value.startswith("urn:"))
    ]


def _candidate_is_hard_disjoint(
    candidate: dict,
    domain_hints: list[str] | None,
    ontology: dict | None,
) -> bool:
    """Reject a DP only when every usable subject hint is OWL-incompatible.

    Anonymous restriction domains remain ``unknown`` and are not rejected.
    Multiple hints are used for SH rows: a property declared on either the
    concrete child or its inherited parent is semantically admissible.
    """
    if not ontology or not domain_hints:
        return False
    named_domains = _named_domain_values(candidate)
    if not named_domains:
        return False
    usable_hints = [hint for hint in domain_hints if hint]
    if not usable_hints:
        return False
    return all(
        all(are_classes_disjoint(hint, domain, ontology) for domain in named_domains)
        for hint in usable_hints
    )


def _safe_dp_candidates(
    candidates: list,
    domain_hints: list[str] | None,
    ontology: dict | None,
) -> list:
    """Keep the upstream order while removing provably disjoint candidates."""
    return [
        candidate
        for candidate in (candidates or [])
        if not _candidate_is_hard_disjoint(candidate, domain_hints, ontology)
    ]


def _resolve_identity_data_attr(
    col_cands: list,
    *,
    domain_hints: list[str] | None = None,
    ontology: dict | None = None,
) -> tuple[str | None, str]:
    """Accept only a uniquely proven literal meaning for an identity column.

    Natural identifiers can carry a literal ontology meaning in addition to
    forming the RDF subject.  Generic keys cannot: domain score, values, or an
    LLM answer alone are insufficient.  Candidate generation therefore
    attaches lexical+ontology provenance, and this second gate validates it
    before accepting exactly one property.
    """
    evidenced = filter_strong_identity_datatype_candidates(col_cands)
    safe_candidates = _safe_dp_candidates(evidenced, domain_hints, ontology)
    if len(safe_candidates) == 1:
        return safe_candidates[0].get("uri"), "high"
    return None, "low"

# ============================================================
#  基础匹配函数（Class / DatatypeProperty / range Class）
# ============================================================

def _resolve_class(table_name: str, class_cands: list) -> tuple:
    """表 → Class URI。返回 (uri, confidence)"""
    conf = _precompute_confidence(class_cands)
    if not class_cands:
        return None, "low"
    # 高置信直接走 top1，减少不必要 LLM 开销与时延
    if conf == "high":
        return class_cands[0]["uri"], "high"
    if _LLM_BUDGET_EXHAUSTED:
        return class_cands[0]["uri"], "low"
    prompt = f"""
## 任务
为关系表 `{table_name}` 找到本体中最对应的 OWL Class。

## 候选 Class（已按相似度排序）
{json.dumps(class_cands, indent=2, ensure_ascii=False)}

## 输出格式（严格 JSON）
{{
  "selected_uri": "选中的 URI（必须来自候选列表）",
  "reason": "一句话理由"
}}
"""
    try:
        m = _call_llm(prompt)
        selected = m.get("selected_uri")
        allowed = {c.get("uri") for c in class_cands if c.get("uri")}
        if selected in allowed:
            return selected, conf
        return class_cands[0]["uri"], "medium"
    except Exception as e:
        _mark_quota_if_needed(e)
        print(f"  [WARN] {table_name} Class 匹配失败: {e}")
        return class_cands[0]["uri"], "low"


def _resolve_data_attr(
    table_name: str,
    col_name: str,
    class_uri: str,
    col_cands: list,
    *,
    domain_hints: list[str] | None = None,
    class_anchor_confirmed: bool = True,
    sql_type: str | None = None,
    ontology: dict | None = None,
) -> tuple:
    """数据属性列 → DatatypeProperty URI。返回 (uri, confidence)"""
    # RDFS domain is a hard semantic gate only after the represented table
    # Class has itself been confirmed.  A noisy low-confidence Class must not
    # erase an otherwise plausible DP candidate.
    hints = (
        [hint for hint in ([class_uri] + list(domain_hints or [])) if hint]
        if class_anchor_confirmed
        else []
    )
    safe_candidates = filter_semantically_admissible_datatype_candidates(
        col_cands,
        column_name=col_name,
        domain_hints=hints,
        sql_type=sql_type,
        ontology=ontology,
    )
    conf = _precompute_confidence(safe_candidates)
    if not safe_candidates:
        return None, "low"
    # 高置信直接走 top1，避免逐列调用 LLM
    if conf == "high":
        return safe_candidates[0]["uri"], "high"
    if _LLM_BUDGET_EXHAUSTED:
        return safe_candidates[0]["uri"], "low"

    # 列名与属性 local_name 精确匹配时直接锁定，避免 LLM 误改
    norm_col = re.sub(r"[^a-z0-9]", "", (col_name or "").lower())
    for c in safe_candidates:
        local = c.get("local_name") or ""
        norm_local = re.sub(r"[^a-z0-9]", "", local.lower())
        if norm_col and norm_col == norm_local and c.get("uri"):
            return c["uri"], "high"

    prompt = f"""
## 任务
表 `{table_name}`（Class: {class_uri}）中，列 `{col_name}` 是数据属性列。
找到本体中最对应的 OWL DatatypeProperty。

## 候选属性（已按相似度+domain排序）
{json.dumps(safe_candidates, indent=2, ensure_ascii=False)}

## 输出格式（严格 JSON）
{{
  "selected_uri": "选中的 URI（必须来自候选列表，或 null）",
  "reason": "一句话理由"
}}
"""
    try:
        m = _call_llm(prompt)
        selected = m.get("selected_uri")
        allowed = {c.get("uri") for c in safe_candidates if c.get("uri")}
        if selected in allowed:
            return selected, conf
        # ``null`` is an explicit semantic abstention, not a transport
        # failure.  Replacing it with top-1 silently converts uncertainty into
        # a fabricated assertion; an out-of-pool URI is equally inadmissible.
        return None, "low"
    except Exception as e:
        _mark_quota_if_needed(e)
        print(f"  [WARN] {table_name}.{col_name} DatatypeProperty 匹配失败: {e}")
        return safe_candidates[0]["uri"], "low"


def _resolve_range_class(table_name: str, col_name: str,
                         ref_table: str, ref_class_cands: list) -> tuple:
    """
    FK列：只确认 range Class URI。
    不判断 ObjectProperty。
    返回 (range_class_uri, confidence)
    """
    conf = _precompute_confidence(ref_class_cands)
    if not ref_class_cands:
        return None, "low"
    # 高置信直接走 top1，减少时延
    if conf == "high":
        return ref_class_cands[0]["uri"], "high"
    if _LLM_BUDGET_EXHAUSTED:
        return ref_class_cands[0]["uri"], "low"
    prompt = f"""
## 任务
表 `{table_name}` 中，FK列 `{col_name}` 引用表 `{ref_table}`。
找到引用表在本体中最对应的 OWL Class（即该关系的 range Class）。

## 候选 Class
{json.dumps(ref_class_cands, indent=2, ensure_ascii=False)}

## 输出格式（严格 JSON）
{{
  "selected_uri": "选中的 Class URI（必须来自候选列表，或 null）",
  "reason": "一句话理由"
}}
"""
    try:
        m = _call_llm(prompt)
        selected = m.get("selected_uri")
        allowed = {c.get("uri") for c in ref_class_cands if c.get("uri")}
        if selected in allowed:
            return selected, conf
        return ref_class_cands[0]["uri"], "medium"
    except Exception as e:
        _mark_quota_if_needed(e)
        print(f"  [WARN] {table_name}.{col_name} range Class 匹配失败: {e}")
        return ref_class_cands[0]["uri"], "low"

# ============================================================
#  各 Pattern 处理函数
# ============================================================

def _match_SE(
    table_name: str,
    table_candidates: dict,
    ontology: dict | None = None,
) -> dict:
    if table_candidates.get("table_kind") == "value_attr":
        fk_info = table_candidates.get("fk", {})
        ref_table = fk_info.get("ref_table", "")
        owner_cands = fk_info.get("owner_class_candidates", [])
        owner_class_uri, owner_conf = _resolve_class(ref_table or table_name, owner_cands)
        prop_uri, prop_conf = _resolve_data_attr(
            ref_table or table_name,
            table_name,
            owner_class_uri,
            table_candidates.get("property_candidates", []),
            class_anchor_confirmed=owner_conf == "high",
            sql_type=table_candidates.get("value_column_type"),
            ontology=ontology,
        )
        return {
            "pattern": "SE",
            "table_kind": "value_attr",
            "class_uri": owner_class_uri,
            "class_confidence": owner_conf,
            "fk": {
                "column": fk_info.get("column"),
                "ref_table": ref_table,
            },
            "value_column": table_candidates.get("value_column"),
            "value_column_type": table_candidates.get("value_column_type"),
            "prop_uri": prop_uri,
            "prop_confidence": prop_conf,
            "columns": {}
        }

    result = {"pattern": "SE", "columns": {}}

    class_cands = table_candidates.get("table_class_candidates", [])
    class_uri, class_conf = _resolve_class(table_name, class_cands)
    result["class_uri"] = class_uri
    result["class_confidence"] = class_conf

    for col_name, col_info in table_candidates.get("columns", {}).items():
        role = col_info.get("role")
        col_type = col_info.get("column_type")

        if role == "pk":
            dp_candidates = filter_strong_identity_datatype_candidates(
                col_info.get("candidates", [])
            )
            data_prop_uri, data_prop_confidence = _resolve_identity_data_attr(
                dp_candidates,
                domain_hints=[class_uri] if class_uri else None,
                ontology=ontology,
            )
            result["columns"][col_name] = {
                "role": "pk",
                "identity_part": True,
                "dp_candidates": dp_candidates,
                "data_prop_uri": data_prop_uri,
                "data_prop_confidence": data_prop_confidence,
            }
            continue

        if role == "fk_disabled":
            result["columns"][col_name] = {
                "role": "fk_disabled",
                "identity_part": bool(col_info.get("identity_part")),
                "ref_table": col_info.get("ref_table", ""),
            }
            continue

        is_bool_discriminator = role == "data_attr" and _is_boolean_coltype(col_type)
        if _is_discriminator(col_name) or is_bool_discriminator:
            result["columns"][col_name] = {"role": "discriminator", "prop_uri": None}

            result.setdefault("type_assertions", [])
            if is_bool_discriminator:
                bool_class_cands = _boolean_class_candidates(
                    col_name,
                    class_uri,
                    class_cands,
                    extra_class_cands=col_info.get("class_candidates", []),
                )
                guessed_uri = bool_class_cands[0]["uri"] if bool_class_cands else None
                result["type_assertions"].append({
                    "column": col_name,
                    "kind": "boolean",
                    "true_class_uri": guessed_uri,
                    "class_candidates": bool_class_cands,
                    "confidence": "medium" if guessed_uri else "low"
                })
            else:
                # TYPE 类型特调：先放候选，真正值->类映射交给真实值增强
                result["type_assertions"].append({
                    "column": col_name,
                    "kind": "enum",
                    "value_to_class": None,  # 交给真实值增强补
                    "class_candidates": _enum_class_candidates(table_name, class_uri, class_cands),
                    "confidence": "low"
                })
            continue

        col_cands = col_info.get("candidates", [])

        if role == "data_attr":
            prop_uri, conf = _resolve_data_attr(
                table_name,
                col_name,
                class_uri,
                col_cands,
                class_anchor_confirmed=class_conf == "high",
                sql_type=col_type,
                ontology=ontology,
            )
            result["columns"][col_name] = {
                "role": "data_attr",
                "prop_uri": prop_uri,
                "confidence": conf
            }

        elif role == "fk_obj":
            ref_table = col_info.get("ref_table", "")
            ref_class_cands = col_info.get("ref_class_candidates", [])
            range_uri, conf = _resolve_range_class(table_name, col_name, ref_table, ref_class_cands)
            identity_part = bool(col_info.get("identity_part"))
            dp_candidates = (
                filter_strong_identity_datatype_candidates(
                    col_info.get("dp_candidates", [])
                )
                if identity_part
                else []
            )
            data_prop_uri, data_prop_confidence = _resolve_identity_data_attr(
                dp_candidates,
                domain_hints=[class_uri] if class_uri else None,
                ontology=ontology,
            )
            result["columns"][col_name] = {
                "role": "fk_obj",
                "identity_part": identity_part,
                "domain_class_uri": class_uri,    # 当前表 Class
                "range_class_uri": range_uri,      # 引用表 Class
                "ref_table": ref_table,
                "ref_col": col_info.get("ref_col", ""),
                "constraint_name": col_info.get("constraint_name", ""),
                "fk_arity": col_info.get("fk_arity", 1),
                "op_candidates": col_cands,        # 完整候选集，留给 OP 映射
                "confidence": conf,
                "dp_candidates": dp_candidates,
                "data_prop_uri": data_prop_uri,
                "data_prop_confidence": data_prop_confidence,
            }

    return result


def _match_SH(
    table_name: str,
    table_candidates: dict,
    ontology: dict | None = None,
) -> dict:
    result = {"pattern": "SH", "columns": {}}

    sub_cands    = table_candidates.get("sub_class_candidates", [])
    parent_cands = table_candidates.get("parent_class_candidates", [])
    parent_table = table_candidates.get("parent_table", "")
    sub_conf     = _precompute_confidence(sub_cands)
    parent_conf  = _precompute_confidence(parent_cands)

    if sub_conf == "high" and parent_conf == "high":
        sub_class_uri = sub_cands[0]["uri"] if sub_cands else None
        parent_class_uri = parent_cands[0]["uri"] if parent_cands else None
    elif _LLM_BUDGET_EXHAUSTED:
        sub_class_uri = sub_cands[0]["uri"] if sub_cands else None
        parent_class_uri = parent_cands[0]["uri"] if parent_cands else None
        sub_conf = "low"
    else:
        prompt = f"""
## 任务
表 `{table_name}` 是继承表（SH Pattern）。
分别找到本体中对应的子类 URI 和父类 URI。

## 子类候选（当前表）
{json.dumps(sub_cands, indent=2, ensure_ascii=False)}

## 父类候选（父表 `{parent_table}`）
{json.dumps(parent_cands, indent=2, ensure_ascii=False)}

## 输出格式（严格 JSON）
{{
  "sub_class_uri": "子类 URI（来自子类候选，或 null）",
  "parent_class_uri": "父类 URI（来自父类候选，或 null）",
  "reason": "一句话理由"
}}
"""
        try:
            m = _call_llm(prompt)
            sub_class_uri    = m.get("sub_class_uri")
            parent_class_uri = m.get("parent_class_uri")
        except Exception as e:
            _mark_quota_if_needed(e)
            print(f"  [WARN] {table_name} SH Class 匹配失败: {e}")
            sub_class_uri    = sub_cands[0]["uri"] if sub_cands else None
            parent_class_uri = parent_cands[0]["uri"] if parent_cands else None
            sub_conf = "low"

    # Treat the prompt's candidate-list restriction as an executable
    # invariant.  In particular, never let a provider invent an OWL ancestor
    # that is not the class of the physically referenced parent table; the
    # schema-backed identity resolver can then preserve a same-class pair.
    allowed_sub = {c.get("uri") for c in sub_cands if c.get("uri")}
    allowed_parent = {c.get("uri") for c in parent_cands if c.get("uri")}
    if sub_class_uri not in allowed_sub:
        sub_class_uri = sub_cands[0].get("uri") if sub_cands else None
        sub_conf = "medium" if sub_class_uri else "low"
    if parent_class_uri not in allowed_parent:
        parent_class_uri = parent_cands[0].get("uri") if parent_cands else None

    result["sub_class_uri"]    = sub_class_uri
    result["parent_class_uri"] = parent_class_uri
    # Preserve the structural parent-table witness.  The R2RML identity
    # resolver validates the PK/FK contract against the schema; retaining this
    # hint prevents duplicate same-class tables from making the parent
    # representation ambiguous.
    if parent_table:
        result["parent_table"] = parent_table
    result["class_confidence"] = sub_conf

    for col_name, col_info in table_candidates.get("columns", {}).items():
        role = col_info.get("role")
        col_type = col_info.get("column_type")

        if role == "sh_inherited_pk":
            result["columns"][col_name] = {
                "role": "sh_inherited_pk",
                "identity_part": True,
            }
            continue

        if role == "pk":
            dp_candidates = filter_strong_identity_datatype_candidates(
                col_info.get("candidates", [])
            )
            data_prop_uri, data_prop_confidence = _resolve_identity_data_attr(
                dp_candidates,
                domain_hints=[sub_class_uri, parent_class_uri],
                ontology=ontology,
            )
            result["columns"][col_name] = {
                "role": "pk",
                "identity_part": True,
                "dp_candidates": dp_candidates,
                "data_prop_uri": data_prop_uri,
                "data_prop_confidence": data_prop_confidence,
            }
            continue

        if role == "fk_disabled":
            result["columns"][col_name] = {
                "role": "fk_disabled",
                "identity_part": bool(col_info.get("identity_part")),
                "ref_table": col_info.get("ref_table", ""),
            }
            continue

        is_bool_discriminator = role == "data_attr" and _is_boolean_coltype(col_type)
        if _is_discriminator(col_name) or is_bool_discriminator:
            result["columns"][col_name] = {"role": "discriminator", "prop_uri": None}
            result.setdefault("type_assertions", [])
            if is_bool_discriminator:
                bool_class_cands = _boolean_class_candidates(
                    col_name,
                    sub_class_uri,
                    sub_cands,
                    extra_class_cands=col_info.get("class_candidates", []),
                )
                guessed_uri = bool_class_cands[0]["uri"] if bool_class_cands else None
                result["type_assertions"].append({
                    "column": col_name,
                    "kind": "boolean",
                    "true_class_uri": guessed_uri,
                    "class_candidates": bool_class_cands,
                    "confidence": "medium" if guessed_uri else "low"
                })
            else:
                result["type_assertions"].append({
                    "column": col_name,
                    "kind": "enum",
                    "value_to_class": None,
                    "class_candidates": _enum_class_candidates(table_name, sub_class_uri, sub_cands),
                    "confidence": "low"
                })
            continue

        col_cands = col_info.get("candidates", [])

        if role == "data_attr":
            prop_uri, conf = _resolve_data_attr(
                table_name,
                col_name,
                sub_class_uri,
                col_cands,
                domain_hints=[parent_class_uri] if parent_class_uri else None,
                class_anchor_confirmed=sub_conf == "high",
                sql_type=col_type,
                ontology=ontology,
            )
            result["columns"][col_name] = {
                "role": "data_attr",
                "prop_uri": prop_uri,
                "confidence": conf
            }

        elif role == "fk_obj":
            ref_table = col_info.get("ref_table", "")
            ref_class_cands = col_info.get("ref_class_candidates", [])
            range_uri, conf = _resolve_range_class(table_name, col_name, ref_table, ref_class_cands)
            identity_part = bool(col_info.get("identity_part"))
            dp_candidates = (
                filter_strong_identity_datatype_candidates(
                    col_info.get("dp_candidates", [])
                )
                if identity_part
                else []
            )
            data_prop_uri, data_prop_confidence = _resolve_identity_data_attr(
                dp_candidates,
                domain_hints=[sub_class_uri, parent_class_uri],
                ontology=ontology,
            )
            result["columns"][col_name] = {
                "role": "fk_obj",
                "identity_part": identity_part,
                "domain_class_uri": sub_class_uri,
                "range_class_uri": range_uri,
                "ref_table": ref_table,
                "ref_col": col_info.get("ref_col", ""),
                "constraint_name": col_info.get("constraint_name", ""),
                "fk_arity": col_info.get("fk_arity", 1),
                "op_candidates": col_cands,
                "confidence": conf,
                "dp_candidates": dp_candidates,
                "data_prop_uri": data_prop_uri,
                "data_prop_confidence": data_prop_confidence,
            }

    return result


def _match_SR(table_name: str, table_candidates: dict) -> dict:
    """
    SR 表：只确认 domain_class_uri + range_class_uri。
    ObjectProperty 完全交给 OP 映射。
    """
    fk1      = table_candidates.get("fk1", {})
    fk2      = table_candidates.get("fk2", {})
    op_cands = table_candidates.get("sr_prop_candidates", [])
    conf     = _precompute_confidence(op_cands)

    domain_class_uri = fk1.get("domain_class_hint")
    range_class_uri  = fk2.get("range_class_hint")

    # FK 补全失败时，从候选集 domain/range 字段补救
    if not domain_class_uri and op_cands:
        domains = op_cands[0].get("domain", [])
        domain_class_uri = domains[0] if domains else None
    if not range_class_uri and op_cands:
        ranges = op_cands[0].get("range", [])
        range_class_uri = ranges[0] if ranges else None

    return {
        "pattern": "SR",
        "table_kind": table_candidates.get("table_kind"),
        "relation_kind": table_candidates.get("relation_kind", "full_fk"),
        "relation_status": table_candidates.get("relation_status"),
        "structural_relation_evidence": table_candidates.get(
            "structural_relation_evidence"
        ),
        "domain_class_uri": domain_class_uri,
        "range_class_uri": range_class_uri,
        "fk1": fk1,
        "fk2": fk2,
        "partial_value_column": table_candidates.get("partial_value_column"),
        "op_candidates": op_cands,    # ← 完整候选集，留给 OP 映射
        "confidence": conf
    }


# ============================================================
#  主函数
# ============================================================

def run_data_property_mapping(
    candidates: dict,
    ontology: dict | None = None,
) -> dict:
    alignment = {}
    total = len(candidates)

    for i, (table_name, table_cands) in enumerate(candidates.items(), 1):
        pattern = table_cands.get("pattern", "SE")
        print(f"[{i}/{total}] DP Mapping: {table_name} (Pattern: {pattern})")

        if pattern == "SE":
            result = _match_SE(table_name, table_cands, ontology=ontology)
        elif pattern == "SH":
            result = _match_SH(table_name, table_cands, ontology=ontology)
        elif pattern == "SR":
            result = _match_SR(table_name, table_cands)
        else:
            result = {"pattern": pattern, "note": "未知 Pattern，跳过"}

        alignment[table_name] = result
        print(f"  → 完成")

    return alignment


def collect_low_confidence_data_property_mappings(alignment: dict) -> dict:
    """
    收集 DP 映射低置信条目，供真实值增强处理。

    各 pattern 的低置信含义：
      SE/SH: class_confidence == "low"、普通列低置信达到阈值，或
             semantic-identifier 的 DP 轨道未达到 high
      SR:    confidence == "low"（指 domain/range Class 置信度）
    """
    # 宽表成本保护：避免“只要有 1 列 low 就整表进入真实值增强”
    # 仅当低置信列达到数量/比例阈值时才触发表级重判
    min_low_cols = DP_MAPPING_LOW_CONF_MIN_LOW_COLS
    low_ratio_threshold = DP_MAPPING_LOW_CONF_RATIO_THRESHOLD
    low_conf_report = {}

    for table_name, entry in alignment.items():
        pattern = entry.get("pattern", "SE")

        # SR 表：低置信指 domain/range Class 确认不足
        if pattern == "SR":
            if entry.get("confidence") == "low":
                low_conf_report[table_name] = {"table_low": True, "columns_low": []}
            continue

        # SE / SH 表：表级 + 列级
        table_low = entry.get("class_confidence") == "low"
        cols_low = [
            col for col, info in entry.get("columns", {}).items()
            if isinstance(info, dict)
            # Algorithm 1 has only a high shortcut; every non-high decision
            # belongs to ContextEnhanced, including our diagnostic "medium".
            and info.get("confidence") != "high"
            and info.get("role") in ("data_attr", "fk_obj")
        ]
        identity_dp_low = [
            col
            for col, info in entry.get("columns", {}).items()
            if isinstance(info, dict)
            and bool(info.get("identity_part"))
            and bool(info.get("dp_candidates"))
            and info.get("data_prop_confidence") != "high"
        ]

        type_low = False
        for ta in entry.get("type_assertions", []) or []:
            if ta.get("confidence") == "low":
                type_low = True
                break
            if ta.get("kind") == "enum" and not ta.get("value_to_class"):
                type_low = True
                break
            if ta.get("kind") == "boolean" and not ta.get("true_class_uri"):
                type_low = True
                break

        total_cols = sum(
            1 for _c, info in entry.get("columns", {}).items()
            if isinstance(info, dict) and info.get("role") in ("data_attr", "fk_obj")
        )
        cols_low_enough = (
            len(cols_low) >= min_low_cols
            or (total_cols > 0 and (len(cols_low) / total_cols) >= low_ratio_threshold)
        )

        if table_low or cols_low_enough or type_low or identity_dp_low:
            low_conf_report[table_name] = {
                "table_low": table_low,
                "columns_low": cols_low,
                "identity_dp_low": identity_dp_low,
            }

    return low_conf_report


def collect_all_context_data_property_mappings(alignment: dict) -> dict:
    """
    效率实验的全量上下文条件。

    对 SE/SH 表重判 Class 和全部 data_attr/fk_obj 列；对 SR 表重判两端 Class。
    fk_obj 在 DP 阶段只重判 range Class，不判断 ObjectProperty。
    """
    report = {}
    for table_name, entry in alignment.items():
        pattern = entry.get("pattern", "SE")
        if pattern == "SR":
            report[table_name] = {"table_low": True, "columns_low": []}
            continue

        context_columns = [
            col
            for col, info in entry.get("columns", {}).items()
            if isinstance(info, dict) and info.get("role") in {"data_attr", "fk_obj"}
        ]
        report[table_name] = {
            "table_low": True,
            "columns_low": context_columns,
            "identity_dp_low": [
                col
                for col, info in entry.get("columns", {}).items()
                if isinstance(info, dict)
                and bool(info.get("identity_part"))
                and bool(info.get("dp_candidates"))
            ],
        }
    return report
