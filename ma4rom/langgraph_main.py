"""
将原 main.py 的串行流程重构为可路由的多 Agent 图：
  1) 数据清洗 + nofks 判定
  2) ClassifyAgent 分类（SE/SR/SH）
  3) Data Property Mapping + Real-Value 上下文增强
  4) Object Property Mapping（SMD/SSD/SLD + 隐式关系补全）
  5) R2RML 生成

特点：
  PlannerAgent 每一步根据当前状态决定是否执行/跳过下一步
  对 nofks 自动插入 IND 外键补全步骤
  中间结果按规范化阶段命名落盘
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DPMapping"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RealValue"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "OPMapping"))

from config import (  # noqa: E402
    AUTO_ENABLE_IND_ON_SPARSE_FKS,
    AUTO_ENABLE_IND_ON_NOFKS,
    CURRENT_DATABASE,
    DB_SCHEMA_NAME,
    DEFAULT_DATABASE,
    IND_SPARSE_FK_THRESHOLD,
    HYPER_LEVELS_BY_SELECT,
    HyperChild,
    HyperSelect,
    MAPPING_BASE_URL,
    MAX_REAL_VALUE_TABLES,
    ONTOLOGY_PATH,
    OUTPUT_DIR,
    OUTPUT_MAPPING_FILENAME,
    EQUIV_OP_LLM_SLEEP_SECONDS,
    EQUIV_OP_MIN_ENDPOINT_SCORE,
    isHyper,
    USE_IND_FK_COMPLETION,
    USE_LLM_EMPTY_FK_COMPLETION,
    USE_LLM_ONTOLOGY_FK_FALLBACK,
    USE_LLM_FK_MERGE,
)
from utils.db_utils import read_schema
from utils.ontology_utils import read_ontology
from utils.merge_fks import merge_fks_into_schema, merge_llm_fks_into_schema

from FKCompletion_agent import (
    allocate_targets_and_shooters,
    discover_empty_semantic_foreign_keys,
    discover_implicit_foreign_keys,
    discover_ontology_llm_fallback_foreign_keys,
)
from classify_agent import (
    battle_layer,
    cal_num_fks,
    classify_agent,
    classify_rule,
    find_difference,
)
from candidate_generation import generate_candidates
from DPMapping.data_property_mapping_agent import (
    collect_low_confidence_data_property_mappings,
    run_data_property_mapping,
)
from RealValue.real_value_enhancement_agent import run_real_value_enhancement

from OPMapping.equivalence_op_module import run_equivalence_op_module

from r2rml_generator import generate_r2rml

os.makedirs(OUTPUT_DIR, exist_ok=True)


class PipelineState(TypedDict, total=False):
    started_at: float
    step_status: Dict[str, bool]
    decision_log: List[Dict[str, str]]
    next_action: str

    schema: Dict[str, Any]
    ontology: Dict[str, Any]
    enriched_schema: Dict[str, Any]

    physical_fk_count: int
    should_run_ind: bool

    discovered_fks: Dict[str, Any]
    pattern_result: Dict[str, Any]
    agent_result: Dict[str, Any]
    candidates: Dict[str, Any]
    alignment: Dict[str, Any]
    low_conf: Dict[str, Any]
    final_alignment: Dict[str, Any]
    scenarios: Dict[str, Any]
    op_mapping_step1_result: Dict[str, Any]
    op_mapping_full: Dict[str, Any]

    output_ttl: str


def _save_json(data: Any, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  → 已保存: {path}")


def _banner(step_num: str, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Step {step_num}: {title}")
    print(f"{'=' * 60}\n")


def _trim_low_confidence(low_conf: dict, max_tables: int) -> dict:
    if max_tables <= 0:
        return {}
    if len(low_conf) <= max_tables:
        return low_conf

    ranked = sorted(
        low_conf.items(),
        key=lambda kv: (
            1 if kv[1].get("table_low") else 0,
            len(kv[1].get("columns_low", [])),
        ),
        reverse=True,
    )
    return dict(ranked[:max_tables])


def _mark_done(state: PipelineState, key: str) -> Dict[str, bool]:
    status = dict(state.get("step_status", {}))
    status[key] = True
    return status


def _append_decision(state: PipelineState, agent: str, decision: str, reason: str) -> List[Dict[str, str]]:
    logs = list(state.get("decision_log", []))
    logs.append({"agent": agent, "decision": decision, "reason": reason})
    return logs


#Agents
def init_agent(_: PipelineState) -> PipelineState:
    _banner("0", "初始化上下文")

    schema = read_schema()
    ontology = read_ontology(ONTOLOGY_PATH)
    physical_fk_count = sum(
        len((tbl_info or {}).get("foreign_keys", []) or [])
        for tbl_info in (schema or {}).values()
    )
    should_run_ind = (
        USE_IND_FK_COMPLETION
        or (AUTO_ENABLE_IND_ON_NOFKS and physical_fk_count == 0)
        or (
            AUTO_ENABLE_IND_ON_SPARSE_FKS
            and physical_fk_count <= IND_SPARSE_FK_THRESHOLD
        )
    )

    print(f"  目标数据库 : {CURRENT_DATABASE}")
    print(f"  本体路径   : {ONTOLOGY_PATH}")
    print(f"  输出目录   : {OUTPUT_DIR}")
    print(f"  Schema: {len(schema)} 张表")
    print(f"  本体 Classes: {len(ontology['classes'])}")
    print(f"  本体 ObjectProperties: {len(ontology['object_properties'])}")
    print(f"  本体 DatatypeProperties: {len(ontology['datatype_properties'])}")
    print(f"  物理 FK 数量: {physical_fk_count}")
    if should_run_ind:
        if physical_fk_count == 0:
            print("  nofks 判定成立：将自动执行 IND 外键补全")
        elif physical_fk_count <= IND_SPARSE_FK_THRESHOLD:
            print(
                f"  sparse-fks 判定成立（物理 FK <= {IND_SPARSE_FK_THRESHOLD}）："
                "将自动执行 IND 外键补全"
            )

    return {
        "started_at": time.time(),
        "step_status": {"init": True},
        "decision_log": [],
        "schema": schema,
        "ontology": ontology,
        "enriched_schema": schema,
        "physical_fk_count": physical_fk_count,
        "should_run_ind": should_run_ind,
    }


def planner_agent(state: PipelineState) -> PipelineState:
    step_status = state.get("step_status", {})

    if not step_status.get("fk_completion"):
        if state.get("should_run_ind"):
            decision = "fk_completion_agent"
            reason = "启用 IND，或命中 nofks/sparse-fks 自动补全条件"
        else:
            decision = "classify_agent_node"
            reason = "IND 关闭，直接进入分类"
            step_status = _mark_done(state, "fk_completion")
        return {
            "next_action": decision,
            "step_status": step_status,
            "decision_log": _append_decision(state, "PlannerAgent", decision, reason),
        }

    if not step_status.get("classify"):
        return {
            "next_action": "classify_agent_node",
            "decision_log": _append_decision(state, "PlannerAgent", "classify_agent_node", "进入表模式分类"),
        }

    if not step_status.get("merge_llm_fk"):
        return {
            "next_action": "merge_fk_agent",
            "decision_log": _append_decision(state, "PlannerAgent", "merge_fk_agent", "处理 LLM FK 回灌策略"),
        }

    if not step_status.get("dp_candidates"):
        return {
            "next_action": "dp_candidate_agent",
            "decision_log": _append_decision(state, "PlannerAgent", "dp_candidate_agent", "生成 DP 映射候选"),
        }

    if not step_status.get("dp_mapping"):
        return {
            "next_action": "dp_mapping_agent",
            "decision_log": _append_decision(state, "PlannerAgent", "dp_mapping_agent", "执行 DP 映射"),
        }

    if not step_status.get("real_value"):
        low_conf = state.get("low_conf", {})
        if not low_conf:
            step_status = _mark_done(state, "real_value")
            return {
                "next_action": "op_scenario_agent",
                "step_status": step_status,
                "final_alignment": state.get("alignment", {}),
                "decision_log": _append_decision(
                    state,
                    "PlannerAgent",
                    "skip_real_value",
                    "无低置信条目，跳过真实值增强",
                ),
            }

        return {
            "next_action": "real_value_agent_node",
            "decision_log": _append_decision(state, "PlannerAgent", "real_value_agent_node", "存在低置信条目，执行真实值增强"),
        }

    if not step_status.get("op_scenario"):
        return {
            "next_action": "op_scenario_agent",
            "decision_log": _append_decision(state, "PlannerAgent", "op_scenario_agent", "进入 OP 映射场景判定"),
        }

    if not step_status.get("op_mapping_step1"):
        return {
            "next_action": "op_step1_agent",
            "decision_log": _append_decision(state, "PlannerAgent", "op_step1_agent", "执行 OP 映射 Step1"),
        }

    if not step_status.get("op_mapping_step2"):
        return {
            "next_action": "op_step2_agent",
            "decision_log": _append_decision(state, "PlannerAgent", "op_step2_agent", "执行 OP 映射 Step2"),
        }

    if not step_status.get("r2rml"):
        return {
            "next_action": "r2rml_agent",
            "decision_log": _append_decision(state, "PlannerAgent", "r2rml_agent", "生成最终 R2RML"),
        }

    return {
        "next_action": "finish",
        "decision_log": _append_decision(state, "PlannerAgent", "finish", "所有步骤完成"),
    }


def planner_route(state: PipelineState) -> str:
    return state.get("next_action", "finish")


def fk_completion_agent(state: PipelineState) -> PipelineState:
    _banner("1", "数据清洗与FK补全（IND）")

    enriched_schema = state.get("enriched_schema", {})  # 当前 schema；首次进入时等于原始 schema，补全后会写回 state
    physical_fk_count = state.get("physical_fk_count", 0) #数据库 schema 中原本显式声明的 foreign key 数量

    if not USE_IND_FK_COMPLETION:
        if physical_fk_count == 0 and AUTO_ENABLE_IND_ON_NOFKS:
            print("  检测到 nofks 场景（物理 FK=0），自动启用 IND 补全。")
        elif (
            physical_fk_count <= IND_SPARSE_FK_THRESHOLD
            and AUTO_ENABLE_IND_ON_SPARSE_FKS
        ):
            print(
                f"  检测到 sparse-fks 场景（物理 FK={physical_fk_count} <= "
                f"{IND_SPARSE_FK_THRESHOLD}），自动启用 IND 补全。"
            )

    allocation = allocate_targets_and_shooters(enriched_schema) #返回{targets： shooters:}
    print(f"  靶点(Targets): {len(allocation['targets'])}")
    print(f"  射手(Shooters): {len(allocation['shooters'])}")

    discovered_fks = discover_implicit_foreign_keys(
        allocation, schema_name=DB_SCHEMA_NAME
    )
    if USE_LLM_EMPTY_FK_COMPLETION:
        semantic_fks = discover_empty_semantic_foreign_keys(
            allocation,
            discovered_fks=discovered_fks,
            schema_name=DB_SCHEMA_NAME,
        )
        for table_name, fk_list in semantic_fks.items():
            discovered_fks.setdefault(table_name, []).extend(fk_list)
    if USE_LLM_ONTOLOGY_FK_FALLBACK:
        ontology_fks = discover_ontology_llm_fallback_foreign_keys(
            allocation,
            discovered_fks=discovered_fks,
            schema_name=DB_SCHEMA_NAME,
        )
        for table_name, fk_list in ontology_fks.items():
            discovered_fks.setdefault(table_name, []).extend(fk_list)
    total_discovered = sum(len(v) for v in discovered_fks.values())
    print(f"  IND 发现外键: {total_discovered} 条")

    enriched_schema = merge_fks_into_schema(enriched_schema, discovered_fks)

    return {
        "discovered_fks": discovered_fks,
        "enriched_schema": enriched_schema,
        "step_status": _mark_done(state, "fk_completion"),
    }


def classify_agent_node(state: PipelineState) -> PipelineState:
    _banner("2", "ClassifyAgent：SE/SR/SH 分类")

    enriched_schema = state.get("enriched_schema", {})

    rule_result = classify_rule(enriched_schema)
    agent_result = classify_agent(enriched_schema)
    fks_count = cal_num_fks(enriched_schema)

    diff = find_difference(rule_result, agent_result)
    pattern_result = battle_layer(
        diff, rule_result, agent_result, fks_count, enriched_schema
    )

    print(f"  规则 vs LLM 分歧: {len(diff)} 张表")
    print(f"  当前 FK 总数: {fks_count}")
    _save_json(pattern_result, "pattern_result.json")

    return {
        "pattern_result": pattern_result,
        "agent_result": agent_result,
        "step_status": _mark_done(state, "classify"),
    }


def merge_fk_agent(state: PipelineState) -> PipelineState:
    _banner("3", "可选LLM FK回灌")

    enriched_schema = state.get("enriched_schema", {})

    if USE_LLM_FK_MERGE:
        enriched_schema = merge_llm_fks_into_schema(
            enriched_schema,
            state.get("agent_result", {}),
        )
        print(f"  合并后外键总数: {cal_num_fks(enriched_schema)}")
    else:
        print("  USE_LLM_FK_MERGE=False，跳过 LLM FK 回灌")

    _save_json(enriched_schema, "enriched_schema.json")
    return {
        "enriched_schema": enriched_schema,
        "step_status": _mark_done(state, "merge_llm_fk"),
    }


def dp_candidate_agent(state: PipelineState) -> PipelineState:
    _banner("4", "DP 映射候选生成")

    candidates = generate_candidates(
        state.get("enriched_schema", {}),
        state.get("pattern_result", {}),
        state.get("ontology", {}),
    )
    print(f"  为 {len(candidates)} 张表生成候选")
    _save_json(candidates, "dp_mapping_candidates.json")

    return {
        "candidates": candidates,
        "step_status": _mark_done(state, "dp_candidates"),
    }


def dp_mapping_agent(state: PipelineState) -> PipelineState:
    _banner("5", "DP 映射")

    alignment = run_data_property_mapping(
        state.get("candidates", {}),
        ontology=state.get("ontology", {}),
    )
    low_conf = collect_low_confidence_data_property_mappings(alignment)
    print(f"  低置信条目: {len(low_conf)} 张表（进入真实值增强）")

    _save_json(alignment, "dp_mapping_alignment.json")

    return {
        "alignment": alignment,
        "low_conf": low_conf,
        "step_status": _mark_done(state, "dp_mapping"),
    }


def real_value_agent_node(state: PipelineState) -> PipelineState:
    _banner("6", "真实值上下文增强")

    final_alignment = run_real_value_enhancement(
        state.get("alignment", {}),
        state.get("low_conf", {}),
        state.get("candidates", {}),
        ontology=state.get("ontology", {}),
        enriched_schema=state.get("enriched_schema", {}),
        checkpoint_path=os.path.join(OUTPUT_DIR, "real_value_table_checkpoint.json"),
    )

    _save_json(final_alignment, "final_alignment.json")

    return {
        "final_alignment": final_alignment,
        "step_status": _mark_done(state, "real_value"),
    }


def op_scenario_agent(state: PipelineState) -> PipelineState:
    _banner("7", "Object Property Mapping：构建新 OP 输入")
    scenarios = {
        "_skipped": True,
        "reason": "OP tasks are built directly from final Class/DP alignment, enriched FK schema, real-value evidence, and ontology endpoint constraints.",
    }
    _save_json(scenarios, "op_mapping_scenarios.json")
    return {
        "scenarios": scenarios,
        "step_status": _mark_done(state, "op_scenario"),
    }


def op_step1_agent(state: PipelineState) -> PipelineState:
    _banner("8", "Equivalence OP Module：等价列证据 + Endpoint 约束 + LLM")
    final_alignment = state.get("final_alignment", {}) or state.get("alignment", {})
    op_mapping_step1_result = run_equivalence_op_module(
        final_alignment=final_alignment,
        ontology=state.get("ontology", {}),
        enriched_schema=state.get("enriched_schema", {}),
        schema_name=DB_SCHEMA_NAME,
        output_dir=OUTPUT_DIR,
        ontology_path=ONTOLOGY_PATH,
        min_endpoint_score=EQUIV_OP_MIN_ENDPOINT_SCORE,
        sleep_seconds=EQUIV_OP_LLM_SLEEP_SECONDS,
    )
    _save_json(op_mapping_step1_result, "op_mapping_step1_result.json")
    return {
        "op_mapping_step1_result": op_mapping_step1_result,
        "step_status": _mark_done(state, "op_mapping_step1"),
    }


def op_step2_agent(state: PipelineState) -> PipelineState:
    _banner("9", "Object Property Mapping：收束结果")
    step2_result = {
        "orphan_matches": [],
        "skipped": True,
        "reason": "Legacy OP Step2 orphan completion has been removed; the active OP pipeline is the equivalence-column module only.",
    }
    _save_json(step2_result, "op_mapping_step2_result.json")
    op_mapping_full = {
        "step1": state.get("op_mapping_step1_result", {}),
        "step2_orphans": [],
    }
    _save_json(op_mapping_full, "op_mapping_full_result.json")
    return {
        "op_mapping_full": op_mapping_full,
        "step_status": _mark_done(state, "op_mapping_step2"),
    }


def r2rml_agent(state: PipelineState) -> PipelineState:
    _banner("10", "R2RML 映射生成")

    final_alignment = state.get("final_alignment", {}) or state.get("alignment", {})

    r2rml = generate_r2rml(
        final_alignment=final_alignment,
        op_mapping_full=state.get("op_mapping_full", {}),
        enriched_schema=state.get("enriched_schema", {}),
        ontology=state.get("ontology", {}),
        base_url=MAPPING_BASE_URL,
        prefix=CURRENT_DATABASE.replace("_", ""),
    )

    output_ttl = os.path.join(OUTPUT_DIR, OUTPUT_MAPPING_FILENAME)
    with open(output_ttl, "w", encoding="utf-8") as f:
        f.write(r2rml)

    print(f"  → 映射已保存: {output_ttl}")
    print(f"  → 文件大小: {len(r2rml)} 字符")

    return {
        "output_ttl": output_ttl,
        "step_status": _mark_done(state, "r2rml"),
    }


def finish_agent(state: PipelineState) -> PipelineState:
    elapsed = time.time() - float(state.get("started_at", time.time()))
    print(f"\n{'=' * 60}")
    print(f"  LangGraph Pipeline 完成！总耗时: {elapsed:.1f} 秒")
    print(f"  输出目录: {OUTPUT_DIR}/")
    print(f"  最终映射: {state.get('output_ttl', os.path.join(OUTPUT_DIR, OUTPUT_MAPPING_FILENAME))}")
    print(f"{'=' * 60}")
    return state


# Graph

def build_graph():
    builder = StateGraph(PipelineState)

    builder.add_node("init_agent", init_agent) #读 schema / ontology，判断要不要 IND 补 FK
    builder.add_node("planner_agent", planner_agent) 
    builder.add_node("fk_completion_agent", fk_completion_agent) #IND 补全 missing FK，生成 enriched_schema
    builder.add_node("classify_agent_node", classify_agent_node) #规则 + LLM + battle layer，得到 SE/SH/SR pattern_result
    builder.add_node("merge_fk_agent", merge_fk_agent) #把 LLM 推断 FK 回灌 schema
    builder.add_node("dp_candidate_agent", dp_candidate_agent) # 根据 pattern 生成 Class / DP / OP 候选
    builder.add_node("dp_mapping_agent", dp_mapping_agent) # 选择 Class 和 Data Property，FK 只确定 range
    builder.add_node("real_value_agent_node", real_value_agent_node) # 对低置信 type-like / 弱语义列做真实值增强
    builder.add_node("op_scenario_agent", op_scenario_agent) # 把需要 OP 的关系整理成 scenarios
    builder.add_node("op_step1_agent", op_step1_agent) # name + domain + range 选择 ObjectProperty
    builder.add_node("op_step2_agent", op_step2_agent) # 孤儿列补全

    builder.add_node("r2rml_agent", r2rml_agent) # 根据 final_alignment + OP 结果生成 ttl mapping
    builder.add_node("finish_agent", finish_agent)

    builder.set_entry_point("init_agent")
    builder.add_edge("init_agent", "planner_agent")

    builder.add_conditional_edges(
        "planner_agent",
        planner_route,
        {
            "fk_completion_agent": "fk_completion_agent",
            "classify_agent_node": "classify_agent_node",
            "merge_fk_agent": "merge_fk_agent",
            "dp_candidate_agent": "dp_candidate_agent",
            "dp_mapping_agent": "dp_mapping_agent",
            "real_value_agent_node": "real_value_agent_node",
            "op_scenario_agent": "op_scenario_agent",
            "op_step1_agent": "op_step1_agent",
            "op_step2_agent": "op_step2_agent",
            "r2rml_agent": "r2rml_agent",
            "finish": "finish_agent",
        },
    )

    for node_name in [
        "fk_completion_agent",
        "classify_agent_node",
        "merge_fk_agent",
        "dp_candidate_agent",
        "dp_mapping_agent",
        "real_value_agent_node",
        "op_scenario_agent",
        "op_step1_agent",
        "op_step2_agent",
        "r2rml_agent",
    ]:
        builder.add_edge(node_name, "planner_agent")

    builder.add_edge("finish_agent", END)
    return builder.compile()


def run_pipeline() -> PipelineState:
    graph = build_graph()
    return graph.invoke({})


def _run_hyper_experiments() -> None:
    """
    直接从 langgraph_main.py 入口串行运行同一组超参数的不同挡位。

    说明：
      - 用户平时就是直接启动 langgraph_main.py，因此把 hyper 入口也放在这里。
      - 每个挡位独立起一个子进程，避免 config 常量在同一进程内被缓存。
      - 子进程通过环境变量指定 HyperSelect / HyperLevel / CURRENT_DATABASE。
    """
    levels = HYPER_LEVELS_BY_SELECT.get(HyperSelect)
    if not levels:
        raise ValueError(
            f"未知 HyperSelect={HyperSelect!r}，可选: {', '.join(sorted(HYPER_LEVELS_BY_SELECT))}"
        )

    print(f"\n{'=' * 72}")
    print(f"  Hyper Experiment: {HyperSelect}")
    print(f"  Database        : {DEFAULT_DATABASE}")
    print(f"  Levels          : {', '.join(levels)}")
    print(f"{'=' * 72}")

    script_path = os.path.abspath(__file__)
    workdir = os.path.dirname(script_path)
    for idx, level in enumerate(levels, start=1):
        env = os.environ.copy()
        env["MAMG_HYPER_CHILD"] = "1"
        env["MAMG_HYPER_SELECT"] = HyperSelect
        env["MAMG_HYPER_LEVEL"] = level
        env["MAMG_CURRENT_DATABASE"] = DEFAULT_DATABASE

        print(f"\n[{idx}/{len(levels)}] Running {HyperSelect} -> {level}")
        subprocess.run(
            [sys.executable, "-u", script_path],
            cwd=workdir,
            env=env,
            check=True,
        )

    print(f"\n{'=' * 72}")
    print("  Hyper Experiment Finished")
    print(f"{'=' * 72}")


def main() -> None:
    if isHyper and not HyperChild:
        _run_hyper_experiments()
        return
    run_pipeline()


if __name__ == "__main__":
    main()
