# 只有这些"语义模糊的通用列名"才需要名称验证
AMBIGUOUS_COLUMNS = {"type", "status", "kind", "category", "flag", "code", "level"}


def merge_fks_into_schema(schema: dict, discovered_fks: dict) -> dict:
    """
    将 IND 算法找到的隐式外键合并进 schema。
    discovered_fks 格式（来自 FKCompletion_agent）:
    {
      "Paper": [
        {"column": "has_an_abstract", "ref_table": "Abstract", "ref_col": "ID", "ind_score": 1.0}
      ]
    }
    """
    import copy
    enriched = copy.deepcopy(schema)

    for table, fk_list in discovered_fks.items():
        if table not in enriched:
            continue

        existing_cols = {fk["column"] for fk in enriched[table]["foreign_keys"]}

        for fk in fk_list:
            if fk["column"] in existing_cols:
                continue
            if fk["ref_table"] == table:  # 自引用过滤
                continue

            # 只对通用列名做名称关联检查，其他列名直接放行
            col_lower = fk["column"].lower().replace("_", "")
            ref_lower = fk["ref_table"].lower().replace("_", "")

            is_ambiguous = any(word in col_lower for word in AMBIGUOUS_COLUMNS)
            if is_ambiguous:
                if ref_lower not in col_lower and col_lower not in ref_lower:
                    continue  # 通用列名 + 名称无关联 → 过滤

            enriched[table]["foreign_keys"].append({
                "column": fk["column"],
                "ref_table": fk["ref_table"],
                "ref_col": fk["ref_col"],
                "source": fk.get("source", "IND"),
                "ind_score": fk.get("ind_score"),
                "confidence": fk.get("confidence"),
                "reason": fk.get("reason"),
            })

    return enriched


def merge_llm_fks_into_schema(schema: dict, agent_result: dict) -> dict:
    """
    将 classify_agent 推断出的隐式外键合并进 schema。
    只处理 IND 没找到、但 LLM 语义推断出来的 FK（即 source="LLM" 的条目）。

    agent_result 格式（来自 classify_agent / battle_layer 后的结果）:
    {
      "Organizer": {
        "type": "SH",
        "inferred_fks": [
          {"column": "ID", "references_table": "Person", "references_column": "ID"}
        ],
        "reason": "..."
      }
    }

    合并策略：
    - 已有物理 FK 或 IND 已补全的列 → 跳过，不覆盖
    - LLM 推断的 FK 列 → 直接写入，source 标记为 "LLM"
    - TYPE / STATUS 等通用歧义列 → 同样过滤，防止误判
    """
    import copy
    enriched = copy.deepcopy(schema)

    for table, info in agent_result.items():
        if table not in enriched:
            continue

        inferred_fks = info.get("inferred_fks", [])
        if not inferred_fks:
            continue

        existing_cols = {fk["column"] for fk in enriched[table]["foreign_keys"]}

        for fk in inferred_fks:
            col = fk.get("column")
            ref_table = fk.get("references_table")
            ref_col = fk.get("references_column", "ID")

            if not col or not ref_table:
                continue
            if col in existing_cols:
                # 已经有这列的 FK 了（IND 或物理 FK），不覆盖
                continue
            if ref_table == table:
                # 自引用跳过
                continue
            if ref_table not in enriched:
                # LLM 瞎编了一个不存在的表
                print(f"  [WARN] LLM 推断的 FK {table}.{col} → {ref_table} 目标表不存在，跳过")
                continue

            # 通用列名过滤（和 IND 保持同样的标准）
            col_lower = col.lower().replace("_", "")
            ref_lower = ref_table.lower().replace("_", "")
            is_ambiguous = any(word in col_lower for word in AMBIGUOUS_COLUMNS)
            if is_ambiguous:
                if ref_lower not in col_lower and col_lower not in ref_lower:
                    print(f"  [SKIP] {table}.{col} 是通用列名且与 {ref_table} 名称无关，跳过")
                    continue

            enriched[table]["foreign_keys"].append({
                "column": col,
                "ref_table": ref_table,
                "ref_col": ref_col,
                "source": "LLM"
            })
            print(f"  [LLM FK] {table}.{col} → {ref_table}.{ref_col}")

    return enriched
