import psycopg2

from utils.llm_client import call_llm
from utils.merge_fks import merge_fks_into_schema
from utils.db_utils import read_schema, DB_CONFIG
from config import CLASSIFY_BATTLE_MATCH_THRESHOLD, CLASSIFY_SAMPLE_ROWS_LIMIT
import json
from json import JSONDecoder


"""
表级分类规则

表级分类只保留三种：
  SH:  单列PK且该PK是FK（子类继承表）
  SR:  关系表。包括两边都显式声明 FK 的纯关联表，也包括 RODI
       renamed 中“复合 PK 二列表、只有一边 FK、另一边是关系端 ID”的
       partial-FK 关系表。
  SE:  其余（主实体表 + 带属性的关联表 SRR）

注意：LLM4VKG/Calvanese 原文把 SRm 作为独立 mapping pattern。
本系统当前为了把表级分类收敛到 SE/SH/SR，只在实现内部把 SE/SH 表中的
非 PK FK 列标成 fk_obj，用来生成 FK→ObjectProperty；不再输出 SRm 表级标签。
"""


def _is_literal_value_type(col_type: str | None) -> bool:
    """True 表示该列更像 literal 值列，而不是对象 ID 端点。"""
    t = (col_type or "").lower()
    return any(k in t for k in (
        "char", "text", "string", "date", "time", "bool", "json", "xml"
    ))


def _is_partial_fk_sr(cols: set, pks: set, fk_cols: set, fk_groups: list[dict], columns: dict) -> bool:
    """
    partial-FK SR 只覆盖“另一端仍是对象 ID”的二元关系表。
    若非 FK 端是字符串/日期等 literal 值，则它是多值数据属性表，应归入 SE。
    """
    if not (len(cols) == 2 and pks == cols and len(fk_groups) == 1 and len(fk_cols & pks) == 1):
        return False
    non_fk_cols = list(cols - fk_cols)
    if len(non_fk_cols) != 1:
        return False
    return not _is_literal_value_type(columns.get(non_fk_cols[0]))


def classify_rule(schema):
    def _is_sr_like(cols: set, pks: set, fk_cols: set, fk_groups: list[dict], columns: dict) -> bool:
        if not cols:
            return False
        # 标准 SR：两端都显式声明 FK。
        if fk_cols == cols and len(fk_groups) == 2 and (len(pks) == 0 or pks.issubset(fk_cols)):
            return True
        # partial-FK SR：二列表复合 PK 只有一端物理 FK，另一端仍是对象 ID。
        # literal VALUE 表（如 has_an_email(Person, VALUE)）不是 SR/SRm，而是 SE 内部 value_attr。
        if _is_partial_fk_sr(cols, pks, fk_cols, fk_groups, columns):
            return True
        return False

    def _group_fk_constraints(table_info: dict) -> list[dict]:
        """
        将逐列 FK 记录按约束分组，便于判断“整组列是否形成同一个 FK”。
        返回元素形如：
          {"columns": [...], "ref_table": "...", "ref_columns": [...]}
        """
        groups = {}
        for i, fk in enumerate(table_info.get("foreign_keys", []) or []):
            cname = fk.get("constraint_name") or f"__anon_fk_{i}"
            g = groups.setdefault(cname, {
                "columns": [],
                "ref_table": fk.get("references_table") or fk.get("ref_table"),
                "ref_columns": []
            })
            col = fk.get("column")
            if col and col not in g["columns"]:
                g["columns"].append(col)
            ref_col = fk.get("references_column") or fk.get("ref_col")
            if ref_col and ref_col not in g["ref_columns"]:
                g["ref_columns"].append(ref_col)
        return list(groups.values())

    result = {}

    for table, info in schema.items():
        cols = set(info["columns"].keys())
        pks = set(info["primary_key"])
        fks = info["foreign_keys"]
        fk_cols = set(f["column"] for f in fks)
        fk_groups = _group_fk_constraints(info)

        # SH：子表主键整体是 FK，或复合 PK 中包含 ID->父表ID 的继承键。
        # 后一种覆盖 RODI denormalized 里 Paper(ID, Reviewer) 这种“继承ID + 业务键”的形态。
        has_pk_as_whole_fk = any(set(g.get("columns", [])) == pks for g in fk_groups) if pks else False
        has_id_parent_fk = "ID" in pks and any(
            f.get("column") == "ID"
            and (f.get("references_column") or f.get("ref_col")) == "ID"
            and (f.get("references_table") or f.get("ref_table")) != table
            for f in fks
        )
        if pks and (has_pk_as_whole_fk or has_id_parent_fk):
            result[table] = "SH"

        # SR：二元关系表。允许 partial-FK 关系表，避免 RODI renamed 中
        # 只有一端物理 FK 的关系表被误判为 SE。
        elif _is_sr_like(cols, pks, fk_cols, fk_groups, info.get("columns", {})):
            result[table] = "SR"

        # SE: 主实体表（PK不是FK）
        else:
            result[table] = "SE"

    return result


def classify_agent(schema: dict) -> dict:
    """
    用 LLM 推断每张表的 Mapping Pattern 类型

    LLM 只判断 SE / SH / SR 三种。SRm 不是表级分类。
    """

    def _parse_json_object(raw_text: str):
        """
        容错 JSON 解析：
        1) 去 code fence
        2) 从首个 '{' 开始尝试 raw_decode，允许前后有杂质文本
        """
        text = (raw_text or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        decoder = JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
        raise json.JSONDecodeError("No valid JSON object found in model output", text, 0)

    def _normalize_entry(table_name: str, entry: dict, fallback_type: str) -> dict:
        """保证每个表都有可用结构，避免单点 JSON 问题导致全流程中断。"""
        if not isinstance(entry, dict):
            return {
                "type": fallback_type,
                "inferred_fks": [],
                "reason": "LLM输出缺失或解析失败，回退规则类型"
            }

        tp = str(entry.get("type", "")).upper()
        if tp not in {"SE", "SH", "SR"}:
            tp = fallback_type

        inferred_fks = []
        for fk in entry.get("inferred_fks", []) if isinstance(entry.get("inferred_fks"), list) else []:
            if not isinstance(fk, dict):
                continue
            col = fk.get("column")
            ref_table = fk.get("references_table")
            ref_col = fk.get("references_column") or "ID"
            if col and ref_table:
                inferred_fks.append({
                    "column": col,
                    "references_table": ref_table,
                    "references_column": ref_col
                })

        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "无有效理由文本"

        return {
            "type": tp,
            "inferred_fks": inferred_fks,
            "reason": reason
        }

    # DeepSeek reasoner 在超长输出上容易截断；分批可显著降低 JSON 断裂概率
    batch_size = 10
    table_list = list(schema.keys())
    fallback_types = classify_rule(schema)
    result = {}

    total_batches = (len(table_list) + batch_size - 1) // batch_size

    for bi, start in enumerate(range(0, len(table_list), batch_size), start=1):
        batch_tables = table_list[start:start + batch_size]
        batch_schema = {t: schema[t] for t in batch_tables}

        prompt = f"""
你是关系数据库 Schema 分析专家。请只对给定批次表进行分类（SE/SH/SR）。

## 类型定义（简版）
- SE: 主实体或带属性关系表
- SH: 主键（可复合）整体等于某个 FK 列集（指向父表）
- SR: 二元关系表（所有列都是 FK，且无额外属性列）

## 强约束
- 若不满足 SH 的结构硬条件（PK=某个 FK 列集），禁止输出 SH。
- 若不满足 SR 的结构硬条件（全列FK + 二元关系），禁止输出 SR。

## 全部表名（用于跨表语义）
{json.dumps(table_list, ensure_ascii=False)}

## 本批次待分类 Schema
{json.dumps(batch_schema, ensure_ascii=False)}

## 输出（严格 JSON 对象）
{{
  "TableName": {{
    "type": "SE|SH|SR",
    "inferred_fks": [{{"column":"列名","references_table":"目标表","references_column":"ID"}}],
    "reason": "一句话理由"
  }}
}}
"""

        parsed = {}
        try:
            parsed = call_llm(
                prompt=prompt,
                system="You are a database schema analysis expert. Output JSON only.",
                prefer_fast=False,
            )
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception as e:
            print(f"  [WARN] classify_agent 批次 {bi}/{total_batches} 解析失败: {e}，该批回退规则类型")
            parsed = {}

        for table_name in batch_tables:
            result[table_name] = _normalize_entry(
                table_name,
                parsed.get(table_name),
                fallback_types.get(table_name, "SE")
            )

        print(f"  classify_agent 进度: {bi}/{total_batches}（{len(batch_tables)} 张表）")

    return result


# 寻找到规则和agent不同之处
def find_difference(rule_result: dict, agent_result: dict) -> dict:
    """
    比较 classify_rule 和 classify_agent 的分类结果。
    """
    diff = {}

    all_tables = set(rule_result.keys()) | set(agent_result.keys())

    for table in all_tables:
        rule_type = rule_result.get(table)
        agent_type = agent_result.get(table, {}).get("type")

        if rule_type != agent_type:
            diff[table] = {
                "rule": rule_type,
                "agent": agent_type
            }

    return diff


# 统计并返回外键总个数
def cal_num_fks(schema: dict) -> int:
    total = 0
    for table_info in schema.values():
        total += len(table_info.get("foreign_keys", []))
    return total


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


# B-1: 取 5 行真实数据样本
def fetch_sample_rows(table_name: str, limit: int = CLASSIFY_SAMPLE_ROWS_LIMIT) -> list[dict]:
    """从数据库取指定表的少量真实数据"""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(f'SELECT * FROM "{table_name}" LIMIT %s', (limit,))
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]
    finally:
        cur.close()
        conn.close()


# B-2: 带数据样本让 LLM 重新判断
def re_classify_with_data(
    table_name: str,
    table_schema: dict,
    sample_rows: list[dict],
    all_table_names: list[str],
    rule_type: str,
    agent_type: str
) -> dict:
    """
    给 LLM 看真实数据，让其重新推断该表的类型和 inferred_fks
    返回格式: {"type": "SH|SR|SE", "inferred_fks": [...], "reason": "..."}
    注意：只输出 SE / SH / SR。
    """
    prompt = f"""
你是关系数据库 Schema 分析专家，熟悉 Calvanese et al. 2023 的 VKG Mapping Patterns。

## 任务
对表 `{table_name}` 进行最终裁决。规则引擎和语义分析存在分歧，请你结合真实数据重新判断。

## 三种类型定义
- **SE**: 主实体表，PK 不是 FK，代表一类独立实体（含带属性的关联表）
- **SH**: 子类继承表，且必须满足“主键（可复合）整体等于某个 FK 列集（指向父表）”
- **SR**: 纯关联表，所有列都是 FK，表示二元实体关系，无额外数据属性

## 当前分歧
- 规则引擎判断: **{rule_type}**（基于显式 FK，但数据库中无 FK 声明）
- 语义分析判断: **{agent_type}**（基于语义推断）

## 表结构
{json.dumps(table_schema, indent=2)}

## 真实数据样本（{len(sample_rows)} 行）
{json.dumps(sample_rows, indent=2, default=str)}

## 数据库中所有表名（供推断跨表引用）
{json.dumps(all_table_names, indent=2)}

## 裁决硬规则
- 仅当满足 **PK 列集 = 某个 FK 列集** 时，才允许输出 SH。
- 若不满足上述结构条件，即使语义看起来像下位词，也不要输出 SH。
- 仅当满足“全列都是 FK、且是二元关系（两组 FK）”时，才允许输出 SR。

严格输出 JSON，不要任何解释：
{{
  "type": "SE|SH|SR",
  "inferred_fks": [
    {{"column": "列名", "references_table": "目标表名", "references_column": "ID"}}
  ],
  "reason": "一句话理由，说明为什么这样判断"
}}
"""
    parsed = call_llm(
        prompt=prompt,
        system="You are a database schema analysis expert. Output JSON only.",
        prefer_fast=False,
    )
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("No valid JSON object in re_classify_with_data output", str(parsed), 0)
    return parsed


# B-3: SQL 验证 inferred_fks 的 match_rate
def _schema_ref_col(schema: dict, table_name: str, col_name: str) -> str | None:
    """
    从已读取 schema 中找到列的真实 FK 引用列，避免默认 ID 造成误判。
    """
    info = (schema or {}).get(table_name, {}) or {}
    for fk in info.get("foreign_keys", []) or []:
        if fk.get("column") == col_name:
            return fk.get("references_column") or fk.get("ref_col")
    return None


def validate_match_rate(table_name: str, inferred_fks: list[dict], schema: dict | None = None) -> float:
    """
    验证 inferred_fks 的匹配率
    对每个 FK 分别计算 match_rate，返回所有 FK 的平均值

    match_rate = 能在目标表找到匹配的行数 / 总行数
    """
    if not inferred_fks:
        return 0.0

    conn = get_connection()
    cur = conn.cursor()
    rates = []

    try:
        seen = set()
        for fk in inferred_fks:
            col = fk["column"]
            ref_table = fk["references_table"]
            ref_col = fk.get("references_column")
            if not ref_col:
                ref_col = _schema_ref_col(schema or {}, table_name, col) or "ID"

            key = (col, ref_table, ref_col)
            if key in seen:
                continue
            seen.add(key)

            try:
                sql = f"""
                    SELECT 
                        COUNT(*) AS total,
                        COUNT(t2."{ref_col}") AS matched
                    FROM "{table_name}" t1
                    LEFT JOIN "{ref_table}" t2 
                        ON t1."{col}" = t2."{ref_col}"
                """
                cur.execute(sql)
                row = cur.fetchone()
                total, matched = row[0], row[1]

                if total == 0:
                    rates.append(0.0)
                else:
                    rates.append(matched / total)

            except Exception as e:
                print(f"  [WARN] 验证 {table_name}.{col} → {ref_table}.{ref_col} 失败: {e}")
                conn.rollback()
                rates.append(0.0)

    finally:
        cur.close()
        conn.close()

    return sum(rates) / len(rates) if rates else 0.0


def battle_layer(diff: dict, rule_result, agent_result, fks_count, schema):
    """
    Battle Layer

    冲突只在 SE / SH / SR 之间。
    """

    # 先基于规则结果初始化最终结果
    final_result = dict(rule_result)

    # 类型 A：完全没有 FK 证据，直接信 LLM
    if fks_count == 0:
        print("[Battle] fks_count=0，属于冲突类型 A，直接采用 LLM 结果")
        for table in diff:
            agent_type = agent_result.get(table, {}).get("type")
            if agent_type:
                final_result[table] = agent_type
                print(f"  {table}: {rule_result.get(table)} → {agent_type} (LLM直接胜出)")
        return final_result

    # 类型 B：有 FK 但存在冲突，两步裁决
    print(f"[Battle] fks_count={fks_count}，属于冲突类型 B，开始两步裁决")
    all_table_names = list(schema.keys())

    def _group_fk_constraints(table_info: dict) -> list[dict]:
        groups = {}
        for i, fk in enumerate(table_info.get("foreign_keys", []) or []):
            cname = fk.get("constraint_name") or f"__anon_fk_{i}"
            g = groups.setdefault(cname, {"columns": []})
            col = fk.get("column")
            if col and col not in g["columns"]:
                g["columns"].append(col)
        return list(groups.values())

    for table, conflict in diff.items():
        rule_type = conflict["rule"]
        agent_type = conflict["agent"]
        print(f"\n[Battle] 处理冲突表: {table}  规则={rule_type}  LLM={agent_type}")

        table_info = schema[table]
        table_cols = set(table_info.get("columns", {}).keys())
        table_pk = set(table_info.get("primary_key", []))
        table_fk = {f.get("column") for f in table_info.get("foreign_keys", [])}
        table_has_physical_fk = bool(table_info.get("foreign_keys", []))
        fk_groups = _group_fk_constraints(table_info)

        # SH：PK 整体是 FK，或复合 PK 中包含 ID->父表ID 的继承键
        struct_is_sh = bool(table_pk and any(set(g.get("columns", [])) == table_pk for g in fk_groups))
        struct_is_sh = struct_is_sh or bool(
            "ID" in table_pk and any(
                f.get("column") == "ID"
                and (f.get("references_column") or f.get("ref_col")) == "ID"
                and (f.get("references_table") or f.get("ref_table")) != table
                for f in table_info.get("foreign_keys", []) or []
            )
        )
        # SR：标准全 FK 关系表，或 renamed 中的 partial-FK 二列表关系表。
        struct_is_sr = bool(
            (
                table_cols
                and table_fk == table_cols
                and len(fk_groups) == 2
                and (len(table_pk) == 0 or table_pk.issubset(table_fk))
            )
            or _is_partial_fk_sr(
                table_cols,
                table_pk,
                table_fk,
                fk_groups,
                table_info.get("columns", {}),
            )
        )

        # 先做结构锁定：当规则已满足论文硬条件时，直接采用规则结果（避免语义误提）
        if rule_type == "SH" and struct_is_sh:
            print("  Step0: 结构满足 SH 硬条件，直接锁定 SH")
            final_result[table] = "SH"
            continue
        if rule_type == "SR" and struct_is_sr:
            print("  Step0: 结构满足 SR 硬条件，直接锁定 SR")
            final_result[table] = "SR"
            continue
        # 结构不满足 SH/SR 且表内已存在物理 FK 时，按论文保持 SE，避免语义误提升
        if rule_type == "SE" and table_has_physical_fk and (not struct_is_sh) and (not struct_is_sr):
            print("  Step0: 结构不满足 SH/SR（且有物理FK），保持 SE")
            final_result[table] = "SE"
            continue

        # 第一步：取 5 行真实数据
        try:
            sample_rows = fetch_sample_rows(table, limit=CLASSIFY_SAMPLE_ROWS_LIMIT)
            print(f"  Step1: 取到 {len(sample_rows)} 行样本数据")
        except Exception as e:
            print(f"  Step1: 取样本失败 ({e})，回退到规则结果")
            final_result[table] = rule_type
            continue

        # 空表处理
        if len(sample_rows) == 0:
            agent_type_direct = agent_result.get(table, {}).get("type")
            print(f"  Step1: 空表，无法SQL验证，直接信LLM语义判断 → {agent_type_direct}")
            final_result[table] = agent_type_direct
            continue

        # 第二步：喂给 LLM 重新判断
        try:
            re_judgment = re_classify_with_data(
                table_name=table,
                table_schema=schema[table],
                sample_rows=sample_rows,
                all_table_names=all_table_names,
                rule_type=rule_type,
                agent_type=agent_type
            )
            re_type = str(re_judgment.get("type", "")).strip().upper()
            re_fks = re_judgment.get("inferred_fks", [])
            if re_type not in {"SE", "SH", "SR"}:
                print(f"  Step2: LLM 返回非法类型 `{re_type}`，回退规则结果")
                final_result[table] = rule_type
                continue
            if not isinstance(re_fks, list):
                re_fks = []
            print(f"  Step2: LLM 重新判断 → {re_type}，推断 FK 数量: {len(re_fks)}")
            print(f"         理由: {re_judgment.get('reason', '')}")
        except Exception as e:
            print(f"  Step2: LLM 重新判断失败 ({e})，回退到规则结果")
            final_result[table] = rule_type
            continue

        # 结构硬约束优先：满足 SR/SH 的硬条件时，不允许回退成 SE
        if re_type == "SE" and struct_is_sr:
            print("  Step3: 表结构满足 SR 硬条件，拒绝回退到 SE，保持 SR")
            final_result[table] = "SR"
            continue
        if re_type == "SE" and struct_is_sh:
            print("  Step3: 表结构满足 SH 硬条件，拒绝回退到 SE，保持 SH")
            final_result[table] = "SH"
            continue

        # 反向硬约束：不满足硬条件时，禁止被 LLM 提升为 SH / SR
        if re_type == "SH" and not struct_is_sh:
            print("  Step3: 不满足 SH 硬条件（PK=某FK列集），拒绝提升，回退规则")
            final_result[table] = rule_type
            continue
        if re_type == "SR" and not struct_is_sr:
            print("  Step3: 不满足 SR 硬条件（全列FK+二元关系），拒绝提升，回退规则")
            final_result[table] = rule_type
            continue

        # 如果 LLM 重新判断后也认为是 SE，直接采纳
        if re_type == "SE" or not re_fks:
            print(f"  Step3: LLM 重新判断也认为是 SE，裁决 → {re_type}")
            final_result[table] = re_type
            continue

        # 第三步：SQL 验证 inferred_fks
        if re_type == "SH":
            pk_cols = set(schema[table].get("primary_key", []))
            re_fks_to_validate = [f for f in re_fks if f["column"] in pk_cols]
            if not re_fks_to_validate:
                print(f"  Step3: SH 但 LLM 未给 PK 列推断 FK，回退规则 → {rule_type}")
                final_result[table] = rule_type
                continue
        else:
            re_fks_to_validate = re_fks

        try:
            # SR 场景下，如果结构上已满足“全列FK + 二元关系”，直接采纳 SR
            if re_type == "SR":
                if struct_is_sr:
                    print("  Step3: 结构满足纯关系表条件（含无PK），直接采纳 SR")
                    final_result[table] = "SR"
                    continue

            match_rate = validate_match_rate(table, re_fks_to_validate, schema=schema)
            print(f"  Step3: SQL 验证 match_rate = {match_rate:.2%}")
        except Exception as e:
            print(f"  Step3: SQL 验证失败 ({e})，回退到规则结果")
            final_result[table] = rule_type
            continue

        # 裁决
        threshold = CLASSIFY_BATTLE_MATCH_THRESHOLD
        if match_rate >= threshold:
            print(f"  裁决: match_rate={match_rate:.2%} ≥ {threshold}，LLM 胜出 → {re_type}")
            final_result[table] = re_type
        else:
            print(f"  裁决: match_rate={match_rate:.2%} < {threshold}，规则胜出 → {rule_type}")
            final_result[table] = rule_type

    return final_result


if __name__ == "__main__":
    schema = read_schema()

    from config import DB_SCHEMA_NAME, USE_IND_FK_COMPLETION

    if USE_IND_FK_COMPLETION:
        # ① FK 补全
        from FKCompletion_agent import allocate_targets_and_shooters, discover_implicit_foreign_keys

        allocation = allocate_targets_and_shooters(schema)
        discovered_fks = discover_implicit_foreign_keys(allocation, schema_name=DB_SCHEMA_NAME)

        # ② 合并回 schema
        enriched_schema = merge_fks_into_schema(schema, discovered_fks)
    else:
        print("USE_IND_FK_COMPLETION=False，跳过 IND 外键补全，仅使用物理外键。")
        enriched_schema = schema

    # ③ 分类（用补全后的 schema）
    rule_result = classify_rule(enriched_schema)
    agent_result = classify_agent(enriched_schema)

    # ④ Battle
    fks_count = cal_num_fks(enriched_schema)
    diff = find_difference(rule_result, agent_result)
    final = battle_layer(diff, rule_result, agent_result, fks_count, enriched_schema)

    # ⑤ 打印结果
    print("\n最终分类结果:")
    for t, p in sorted(final.items()):
        print(f"  {t}: {p}")

    # 统计各类型数量
    from collections import Counter
    counts = Counter(final.values())
    print(f"\n类型统计: {dict(counts)}")
