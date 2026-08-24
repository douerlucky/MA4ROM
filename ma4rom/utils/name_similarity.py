import re


#归一化
def _normalize(name: str) -> str:
    """
    把驼峰、下划线、连字符统一转成小写拼接字符串。
    用于字符级重叠计算。

    "has_a_paper_title" → "hasapapertitle"
    "hasTitle"          → "hastitle"
    """
    parts = re.split(r'[_\-\s]+', name)
    expanded = []
    for p in parts:
        tokens = re.sub(r'([A-Z])', r' \1', p).split()
        expanded.extend(tokens)
    return "".join(t.lower() for t in expanded if t)


def _tokenize(name: str) -> set:
    """
    把属性名拆成小写 token 集合，用于 Jaccard 计算。

    "has_an_ISBN"  → {"has", "an", "isbn"}   ← 全大写缩写作为整体
    "hasTitle"     → {"has", "title"}         ← 驼峰正常拆
    "Paper"        → {"paper"}

    全大写词（如 ISBN、URL、DOI）直接保留为一个 token，不按字母逐个拆分，避免 ISBN → {i, s, b, n}的错误。
    """
    parts = re.split(r'[_\-\s]+', name)
    tokens = []
    for p in parts:
        if p.isupper() and len(p) > 1:
            # 全大写缩写整体保留
            tokens.append(p)
        else:
            # 驼峰拆分hasTitle
            sub = re.sub(r'([A-Z])', r' \1', p).split()
            tokens.extend(sub)
    return {t.lower() for t in tokens if t}


def _char_overlap(a: str, b: str) -> float:
    """
    字符级重叠比例（带频次约束）
    统计两个字符串的字符多重集交集占较长串的比例，避免
    “只要字符出现过就计数”造成的虚高。
    """
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    ca = {}
    cb = {}
    for ch in na:
        ca[ch] = ca.get(ch, 0) + 1
    for ch in nb:
        cb[ch] = cb.get(ch, 0) + 1
    common = 0
    for ch, cnt in ca.items():
        common += min(cnt, cb.get(ch, 0))
    return common / max(len(na), len(nb))


def _jaccard_tokens(a: str, b: str) -> float:
    """
    Jaccard相似度
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    先拆词再比较
    """
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union if union > 0 else 0.0

#名称重叠度
def name_overlap(a: str, b: str) -> float:
    """
    融合字符级重叠 + Jaccard token 相似度，取两者最大值。

    字符级计算：擅长处理缩写、局部匹配（如 "conf"和"conference"）
    Jaccard计算：擅长处理大小写差异、词序无关（如 "has_an_ISBN" vs "has_an_isbn"）

    两个计算结果取max：两种方法互补，任一命中即得高分，返回值 [0.0, 1.0]，越大越相似。
    """
    return max(_char_overlap(a, b), _jaccard_tokens(a, b))


def explain_similarity(a: str, b: str) -> dict:
    """
    返回两个名称之间的详细相似度分析

    返回示例：
    {
      "a": "has_an_ISBN",
      "b": "has_an_isbn",
      "tokens_a": ["has", "an", "isbn"],
      "tokens_b": ["has", "an", "isbn"],
      "intersection": ["has", "an", "isbn"],
      "union": ["has", "an", "isbn"],
      "jaccard": 1.0,
      "char_overlap": 0.917,
      "final_score": 1.0
    }
    """
    ta, tb = _tokenize(a), _tokenize(b)
    inter = ta & tb
    union = ta | tb

    jac   = len(inter) / len(union) if union else 0.0
    char  = _char_overlap(a, b)
    final = max(char, jac)

    return {
        "a":            a,
        "b":            b,
        "tokens_a":     sorted(ta),
        "tokens_b":     sorted(tb),
        "intersection": sorted(inter),
        "union":        sorted(union),
        "jaccard":      round(jac,  4),
        "char_overlap": round(char, 4),
        "final_score":  round(final, 4),
        "winner":       "jaccard" if jac >= char else "char_overlap"
    }

if __name__ == "__main__":
    import json

    test_pairs = [
        # ── Bug 场景：大小写不同 ──────────────────────────────
        ("has_an_ISBN",        "has_an_isbn"),       # 期望: 1.0
        ("has_an_ISBN",        "has_an_volume"),      # 期望: 低分，区分出来

        # ── 驼峰 vs 下划线 ────────────────────────────────────
        ("hasTitle",           "has_title"),          # 期望: 1.0
        ("paperTitle",         "paper_title"),        # 期望: 1.0

        # ── 完全不相关 ────────────────────────────────────────
        ("has_an_ISBN",        "has_authors"),        # 期望: 低分
        ("Conference",         "Person"),             # 期望: 0.0 或接近 0

        # ── 部分匹配 ──────────────────────────────────────────
        ("conference_title",   "title"),              # 期望: 中等
        ("has_a_paper_title",  "hasTitle"),           # 期望: 中等偏高
    ]

    print("=" * 70)
    print(f"{'名称 A':<28} {'名称 B':<28} {'Jaccard':>8} {'Char':>6} {'Final':>7}")
    print("=" * 70)

    for a, b in test_pairs:
        r = explain_similarity(a, b)
        print(f"{a:<28} {b:<28} {r['jaccard']:>8.4f} {r['char_overlap']:>6.4f} {r['final_score']:>7.4f}")

    print()
    print("── 详细分析（Bug修复案例）──")
    detail = explain_similarity("has_an_ISBN", "has_an_isbn")
    print(json.dumps(detail, indent=2, ensure_ascii=False))

    print()
    print("── 详细分析（区分能力验证）──")
    detail2 = explain_similarity("has_an_ISBN", "has_an_volume")
    print(json.dumps(detail2, indent=2, ensure_ascii=False))
