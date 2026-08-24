"""
utils/candidate_ranking.py  ——  候选集排序工具函数

candidate_generation.py 和当前映射流程都需要对 Class / ObjectProperty /
DatatypeProperty 候选进行相似度排序，统一放这里避免重复。

用法：
    from utils.candidate_ranking import (
        rank_class_candidates,
        rank_object_prop_candidates,
        rank_datatype_prop_candidates,
    )
"""

import re

from config import (
    DP_MAPPING_CANDIDATE_DOMAIN_WEIGHT,
    DP_MAPPING_CANDIDATE_TEXT_WEIGHT,
)
from utils.name_similarity import name_overlap
from utils.ontology_utils import local_name, hint_match


STOP_TOKENS = {
    "hst", "all", "inc", "npdid", "ncs", "totalt", "poly", "petreg", "id"
}


_NAMED_URI_PREFIXES = ("http://", "https://", "urn:")
_XSD_NUMERIC_NAMES = {
    "byte",
    "decimal",
    "double",
    "float",
    "int",
    "integer",
    "long",
    "negativeinteger",
    "nonnegativeinteger",
    "nonpositiveinteger",
    "positiveinteger",
    "short",
    "unsignedbyte",
    "unsignedint",
    "unsignedlong",
    "unsignedshort",
}


def _normalised_schema_name(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _datatype_property_info(candidate: dict, ontology: dict | None) -> dict:
    uri = (candidate or {}).get("uri")
    return (
        ((ontology or {}).get("datatype_properties", {}) or {}).get(uri, {})
        if uri
        else {}
    ) or {}


def _candidate_terms(candidate: dict, field: str, ontology: dict | None) -> list[str]:
    values = (candidate or {}).get(field)
    if not values:
        values = _datatype_property_info(candidate, ontology).get(field, [])
    return [str(value) for value in (values or []) if value is not None]


def _subject_class_is_within_domain(
    subject_class_uri: str,
    domain_class_uri: str,
    ontology: dict | None,
) -> bool:
    """Return whether using the DP cannot narrow the represented table Class.

    RDFS domain is an entailment, not a UI hint.  A property declared on an
    ancestor of the represented Class is safe; a property declared only on a
    child or an unrelated Class would silently re-type every emitted row.
    """
    if subject_class_uri == domain_class_uri:
        return True
    if domain_class_uri == "http://www.w3.org/2002/07/owl#Thing":
        return True
    ancestors = ((ontology or {}).get("ancestors_of", {}) or {}).get(
        subject_class_uri, []
    )
    if domain_class_uri in ancestors:
        return True
    # Manually assembled ontology fixtures may retain a named union endpoint
    # without copying its members into the candidate record.
    union_members = ((ontology or {}).get("union_members", {}) or {}).get(
        domain_class_uri, []
    )
    return any(
        member == subject_class_uri or member in ancestors
        for member in union_members
    )


def datatype_candidate_domain_is_compatible(
    candidate: dict,
    domain_hints: list[str] | None,
    ontology: dict | None,
) -> bool:
    """Reject a named DP domain unless a confirmed subject Class fits it.

    Anonymous class expressions remain unknown and therefore admissible.  SH
    callers may supply both their concrete child and physically validated
    parent so a property declared on that inherited parent remains usable even
    when the ontology omitted the corresponding subclass edge.
    """
    hints = [str(hint) for hint in (domain_hints or []) if hint]
    if not hints:
        return True
    named_domains = [
        value
        for value in _candidate_terms(candidate, "domain", ontology)
        if value.startswith(_NAMED_URI_PREFIXES)
    ]
    if not named_domains:
        return True
    return any(
        _subject_class_is_within_domain(hint, domain, ontology)
        for hint in hints
        for domain in named_domains
    )


def _sql_type_family(sql_type: str | None) -> str:
    value = str(sql_type or "").strip().lower()
    if not value:
        return "unknown"
    if "bool" in value:
        return "boolean"
    if "date" in value or "time" in value:
        return "temporal"
    if any(
        token in value
        for token in ("int", "numeric", "decimal", "real", "double", "float")
    ):
        return "numeric"
    if any(token in value for token in ("bytea", "binary", "blob")):
        return "binary"
    if any(
        token in value
        for token in ("char", "text", "string", "json", "xml", "uuid")
    ):
        return "text"
    return "unknown"


def _xsd_range_family(range_uri: str) -> str:
    name = local_name(range_uri).lower()
    if name in _XSD_NUMERIC_NAMES:
        return "numeric"
    if name == "boolean":
        return "boolean"
    if name in {"date", "datetime", "time", "duration"}:
        return "temporal"
    if name in {"base64binary", "hexbinary"}:
        return "binary"
    if name in {
        "anyuri",
        "language",
        "name",
        "ncname",
        "normalizedstring",
        "qname",
        "string",
        "token",
    }:
        return "text"
    return "unknown"


def datatype_candidate_range_is_compatible(
    candidate: dict,
    sql_type: str | None,
    ontology: dict | None,
) -> bool:
    """Filter only clearly impossible SQL/XSD family combinations.

    Text/URI targets accept lexical conversion, and integer-backed booleans
    are common in relational schemas, so those are deliberately not rejected.
    Numeric, temporal, boolean and binary ontology ranges otherwise require a
    compatible physical family.
    """
    source_family = _sql_type_family(sql_type)
    if source_family == "unknown":
        return True
    target_families = {
        family
        for family in (
            _xsd_range_family(value)
            for value in _candidate_terms(candidate, "range", ontology)
        )
        if family != "unknown"
    }
    if not target_families or "text" in target_families:
        return True
    for target_family in target_families:
        if target_family == source_family:
            return True
        if target_family == "boolean" and source_family == "numeric":
            return True
    return False


def column_matches_object_property_name(
    column_name: str,
    ontology: dict | None,
) -> bool:
    """Detect a relation-shaped column before it is forced into the DP lane."""
    column_norm = _normalised_schema_name(column_name)
    if not column_norm:
        return False
    return any(
        column_norm == _normalised_schema_name(local_name(uri))
        for uri in ((ontology or {}).get("object_properties", {}) or {})
    )


def has_direct_datatype_name_evidence(
    column_name: str,
    candidates: list[dict] | None,
) -> bool:
    """Return whether a candidate explicitly names the physical column."""
    column_norm = _normalised_schema_name(column_name)
    if not column_norm:
        return False
    for candidate in candidates or []:
        candidate_name = candidate.get("local_name") or local_name(
            candidate.get("uri", "")
        )
        candidate_norm = _normalised_schema_name(candidate_name)
        if candidate_norm == column_norm:
            return True
        for prefix in ("hasa", "hasan", "hasthe", "has"):
            if (
                candidate_norm.startswith(prefix)
                and candidate_norm[len(prefix):] == column_norm
            ):
                return True
    return False


def has_unique_datatype_range_evidence(
    sql_type: str | None,
    candidates: list[dict] | None,
    ontology: dict | None,
) -> bool:
    """A physical type is positive evidence only when it uniquely filters a pool."""
    pool = list(candidates or [])
    if len(pool) < 2:
        return False
    compatible = [
        candidate
        for candidate in pool
        if datatype_candidate_range_is_compatible(candidate, sql_type, ontology)
    ]
    return len(compatible) == 1


def filter_semantically_admissible_datatype_candidates(
    candidates: list[dict] | None,
    *,
    column_name: str,
    domain_hints: list[str] | None,
    sql_type: str | None,
    ontology: dict | None,
) -> list[dict]:
    """Apply structural DP gates before a score or provider can select one."""
    if column_matches_object_property_name(column_name, ontology):
        return []
    return [
        candidate
        for candidate in (candidates or [])
        if datatype_candidate_domain_is_compatible(
            candidate, domain_hints, ontology
        )
        and datatype_candidate_range_is_compatible(candidate, sql_type, ontology)
    ]


def _camel_split_tokens(name: str) -> set[str]:
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return {t.lower() for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) >= 2}


def _table_tokens(table_name: str) -> set[str]:
    raw = {t.lower() for t in re.split(r"[^a-z0-9]+", table_name.lower()) if len(t) >= 2}
    return raw - STOP_TOKENS


def _token_overlap_score(name: str, class_local_name: str) -> float:
    t_toks = _table_tokens(name)
    c_toks = _camel_split_tokens(class_local_name)
    if not t_toks or not c_toks:
        return 0.0
    covered = 0
    for tt in t_toks:
        hit = False
        for ct in c_toks:
            if tt == ct:
                hit = True
                break
            if len(tt) >= 4 and len(ct) >= 4 and (tt.startswith(ct) or ct.startswith(tt)):
                hit = True
                break
        if hit:
            covered += 1
    return covered / len(t_toks)


def rank_class_candidates(name: str, classes: list, top_k: int = 5) -> list:
    """
    在所有 OWL Class 里，按名称相似度对 name 排序，返回 top_k。

    参数:
        name:    待匹配的表名或列名
        classes: 本体 class URI 列表（来自 read_ontology()["classes"]）
        top_k:   返回候选数量

    返回:
        [{"uri": ..., "local_name": ..., "score": float}, ...]
    """
    scored = []
    for uri in classes:
        lname = local_name(uri)
        syntax_score = name_overlap(name, lname)
        token_score = _token_overlap_score(name, lname)
        score = max(syntax_score, token_score)
        scored.append({
            "uri": uri,
            "local_name": lname,
            "score": round(score, 3),
            "syntax_score": round(syntax_score, 3),
            "token_score": round(token_score, 3),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rank_object_prop_candidates(
    name: str,
    object_props: dict,
    domain_hint: str,
    range_hint: str,
    top_k: int = 5,
    ontology: dict | None = None,
) -> list:
    """
    在 ObjectProperty 里做“双赛道”排序：
      - 语法赛道: syntax_score（名称相似）
      - 语义赛道: semantic_score（domain/range 语义可满足）

    最终输出保留 top_k，但由两赛道共同入围（避免互相湮灭）：
      1) 分别计算 syntax_top_k 与 semantic_top_k
      2) 合并去重
      3) 在候选上附带两赛道排名与来源
      4) 再截断到 top_k

    参数:
        name:         待匹配的表名或列名
        object_props: 本体 ObjectProperty dict（来自 read_ontology()["object_properties"]）
        domain_hint:  DP 映射确认的 domain Class URI（可为 None）
        range_hint:   DP 映射确认的 range Class URI（可为 None）
        top_k:        返回候选数量
        ontology:     read_ontology() 的完整结果（用于 subclass/union 语义匹配）

    返回:
        [{"uri", "local_name", "score", "name_score",
          "domain", "range", "domain_score", "range_score"}, ...]
    """
    scored = []
    for uri, info in object_props.items():
        lname = local_name(uri)
        syntax_score = name_overlap(name, lname)
        prop_domains = info.get("domain", [])
        prop_ranges  = info.get("range", [])
        domain_score = hint_match(domain_hint, prop_domains, ontology=ontology)
        range_score  = hint_match(range_hint, prop_ranges, ontology=ontology)

        semantic_score = (domain_score + range_score) / 2.0
        total = syntax_score * 0.5 + semantic_score * 0.5

        scored.append({
            "uri":          uri,
            "local_name":   lname,
            "score":        round(total, 3),
            "name_score":   round(syntax_score, 3),  # 兼容旧字段
            "syntax_score": round(syntax_score, 3),
            "semantic_score": round(semantic_score, 3),
            "domain":       prop_domains,
            "range":        prop_ranges,
            "domain_score": round(domain_score, 3),
            "range_score":  round(range_score, 3),
        })

    # 语法赛道排名
    syntax_sorted = sorted(
        scored,
        key=lambda x: (x["syntax_score"], x["semantic_score"], x["score"]),
        reverse=True,
    )
    for idx, c in enumerate(syntax_sorted, 1):
        c["syntax_rank"] = idx

    # 语义赛道排名
    semantic_sorted = sorted(
        scored,
        key=lambda x: (x["semantic_score"], x["domain_score"], x["range_score"], x["syntax_score"]),
        reverse=True,
    )
    for idx, c in enumerate(semantic_sorted, 1):
        c["semantic_rank"] = idx

    syntax_pool = {c["uri"] for c in syntax_sorted[:top_k]}
    semantic_pool = {c["uri"] for c in semantic_sorted[:top_k]}
    pool = syntax_pool | semantic_pool

    selected = [c for c in scored if c["uri"] in pool]
    for c in selected:
        in_s = c["uri"] in syntax_pool
        in_m = c["uri"] in semantic_pool
        if in_s and in_m:
            c["track_source"] = "both"
        elif in_s:
            c["track_source"] = "syntax"
        else:
            c["track_source"] = "semantic"

    ordered = sorted(
        selected,
        key=lambda x: (
            0 if x["track_source"] == "both" else 1,
            min(x["syntax_rank"], x["semantic_rank"]),
            x["syntax_rank"] + x["semantic_rank"],
            -x["semantic_score"],
            -x["syntax_score"],
            -x["score"],
        ),
    )

    # 保证两赛道头部候选至少有机会进入最终 top_k
    final = []
    seen = set()

    def _push(candidate):
        if not candidate:
            return
        uri = candidate.get("uri")
        if not uri or uri in seen:
            return
        seen.add(uri)
        final.append(candidate)

    syntax_best = next((c for c in syntax_sorted if c["uri"] in pool), None)
    semantic_best = next((c for c in semantic_sorted if c["uri"] in pool), None)
    _push(syntax_best)
    _push(semantic_best)

    for c in ordered:
        if len(final) >= top_k:
            break
        _push(c)

    if len(final) < top_k:
        for c in syntax_sorted:
            if len(final) >= top_k:
                break
            _push(c)

    return final[:top_k]


def rank_datatype_prop_candidates(
    name: str,
    datatype_props: dict,
    domain_hint: str,
    top_k: int = 5,
    ontology: dict | None = None,
    domain_hints: list[str] | None = None,
) -> list:
    """
    在 DatatypeProperty 里，综合名称相似度 + domain 匹配度排序。

    得分公式：
        total = 名称相似度 × λ_text + domain 匹配 × λ_dom

    参数:
        name:           待匹配的表名或列名
        datatype_props: 本体 DatatypeProperty dict（来自 read_ontology()）
        domain_hint:    DP 映射确认的 domain Class URI（可为 None）
        top_k:          返回候选数量

    返回:
        [{"uri", "local_name", "score", "name_score",
          "domain", "domain_score"}, ...]
    """
    scored = []
    all_domain_hints = [hint for hint in ([domain_hint] + list(domain_hints or [])) if hint]
    for uri, info in datatype_props.items():
        lname = local_name(uri)
        name_score   = name_overlap(name, lname)
        prop_domains = info.get("domain", [])
        # Use the parsed OWL hierarchy when it is available.  In particular,
        # members of ``owl:disjointUnionOf`` are legitimate subclasses of the
        # union parent; treating them as unrelated makes the DP candidate pool
        # dominated by lexical accidents.
        domain_score = max(
            (
                hint_match(hint, prop_domains, ontology=ontology)
                for hint in all_domain_hints
            ),
            default=0.3 if not all_domain_hints else 0.0,
        )
        total = (
            name_score * DP_MAPPING_CANDIDATE_TEXT_WEIGHT
            + domain_score * DP_MAPPING_CANDIDATE_DOMAIN_WEIGHT
        )
        scored.append({
            "uri":          uri,
            "local_name":   lname,
            "score":        round(total, 3),
            "name_score":   round(name_score, 3),
            "domain":       prop_domains,
            "domain_score": round(domain_score, 3),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


_IDENTITY_DP_EVIDENCE_KIND = "strong_identifier_lexical_ontology"
_IDENTITY_DP_MIN_LEXICAL_SCORE = 0.95
_IDENTITY_DP_MIN_DOMAIN_SCORE = 0.85


def _ordered_name_tokens(name: str | None) -> list[str]:
    """Split SQL/OWL names while preserving token order.

    Identity columns need a deliberately stricter lexical test than ordinary
    attributes.  Character overlap is useful for recall, but a generic ``ID``
    must not become ``email`` or ``siteURL`` merely because those properties
    share the table's ontology domain.
    """
    text = str(name or "")
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", text)
    return [
        token.lower()
        for token in re.split(r"[^A-Za-z0-9]+", text)
        if token
    ]


def _property_core_tokens(local_name_value: str | None) -> list[str]:
    """Remove only conventional predicate wrappers from a property name."""
    tokens = _ordered_name_tokens(local_name_value)
    if tokens and tokens[0] == "has":
        tokens = tokens[1:]
        if tokens and tokens[0] in {"a", "an", "the"}:
            tokens = tokens[1:]
    return tokens


def _identity_lexical_score(
    column_name: str,
    property_local_name: str,
    domain_hints: list[str] | None,
) -> tuple[float, str]:
    """Measure explicit identifier wording, never broad name similarity.

    Two forms count as independent lexical evidence:

    * the property core names the column exactly (``orcid`` / ``hasORCID``);
    * the property explicitly qualifies it with the represented Class
      (``Person.ID`` / ``personId``).

    A suffix-only coincidence such as ``ID`` / ``submissionId`` is not enough
    unless ``Submission`` is one of the actual domain hints.
    """
    column_tokens = _ordered_name_tokens(column_name)
    property_tokens = _property_core_tokens(property_local_name)
    if not column_tokens or not property_tokens:
        return 0.0, "missing_tokens"
    if property_tokens == column_tokens:
        return 1.0, "exact_identifier_name"

    for hint in domain_hints or []:
        class_tokens = _ordered_name_tokens(local_name(hint))
        if not class_tokens:
            continue
        if property_tokens in (
            [*class_tokens, *column_tokens],
            [*column_tokens, *class_tokens],
        ):
            return 0.98, "class_qualified_identifier_name"
    return 0.0, "no_explicit_identifier_name"


def is_strong_identity_datatype_candidate(candidate: dict | None) -> bool:
    """Validate the provenance attached by the identity candidate gate."""
    if not isinstance(candidate, dict) or not candidate.get("uri"):
        return False
    evidence = candidate.get("identity_evidence")
    if not isinstance(evidence, dict):
        return False
    return bool(
        evidence.get("kind") == _IDENTITY_DP_EVIDENCE_KIND
        and evidence.get("unique") is True
        and float(evidence.get("lexical_score", 0.0) or 0.0)
        >= _IDENTITY_DP_MIN_LEXICAL_SCORE
        and float(evidence.get("domain_score", 0.0) or 0.0)
        >= _IDENTITY_DP_MIN_DOMAIN_SCORE
    )


def filter_strong_identity_datatype_candidates(candidates: list | None) -> list:
    """Keep only independently proven literal semantics for identity columns."""
    return [
        candidate
        for candidate in (candidates or [])
        if is_strong_identity_datatype_candidate(candidate)
    ]


def rank_identity_datatype_prop_candidates(
    name: str,
    datatype_props: dict,
    domain_hint: str | None,
    *,
    ontology: dict | None = None,
    domain_hints: list[str] | None = None,
) -> list:
    """Return at most one strongly proven DatatypeProperty for a PK/identity.

    Identity and literal semantics are orthogonal, but the literal side needs
    evidence of its own.  The ordinary weighted rank can be high solely from
    a matching domain, so it is not an admissible gate for generic keys.  This
    resolver requires both explicit identifier wording and a declared,
    ontology-compatible domain, then rejects a genuinely ambiguous tie.
    """
    hints = [hint for hint in ([domain_hint] + list(domain_hints or [])) if hint]
    ranked = rank_datatype_prop_candidates(
        name,
        datatype_props,
        domain_hint,
        top_k=max(1, len(datatype_props or {})),
        ontology=ontology,
        domain_hints=domain_hints,
    )
    eligible = []
    for candidate in ranked:
        domain_score = float(candidate.get("domain_score", 0.0) or 0.0)
        if domain_score < _IDENTITY_DP_MIN_DOMAIN_SCORE:
            continue
        lexical_score, lexical_rule = _identity_lexical_score(
            name,
            candidate.get("local_name") or local_name(candidate.get("uri", "")),
            hints,
        )
        if lexical_score < _IDENTITY_DP_MIN_LEXICAL_SCORE:
            continue
        eligible.append(
            {
                **candidate,
                "identity_evidence": {
                    "kind": _IDENTITY_DP_EVIDENCE_KIND,
                    "lexical_rule": lexical_rule,
                    "lexical_score": lexical_score,
                    "domain_score": domain_score,
                    "unique": False,
                },
            }
        )

    eligible.sort(
        key=lambda candidate: (
            float(candidate["identity_evidence"]["lexical_score"]),
            float(candidate["identity_evidence"]["domain_score"]),
            float(candidate.get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )
    if not eligible:
        return []

    top = eligible[0]
    if len(eligible) > 1:
        second = eligible[1]
        lexical_gap = (
            float(top["identity_evidence"]["lexical_score"])
            - float(second["identity_evidence"]["lexical_score"])
        )
        domain_gap = (
            float(top["identity_evidence"]["domain_score"])
            - float(second["identity_evidence"]["domain_score"])
        )
        # Two equally explicit properties remain semantically ambiguous.  A
        # full exact-vs-inherited domain step (1.0 vs 0.85) is the only safe
        # ontology tie-break admitted here.
        if lexical_gap < 0.05 and domain_gap < 0.15 - 1e-9:
            return []

    top["identity_evidence"]["unique"] = True
    return [top]
