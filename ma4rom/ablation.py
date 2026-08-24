from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from config import CONTEXT_ENHANCEMENT_MODE, isAblation


# ============================================================
# Ablation switches
# ============================================================

ABLATION_PRESET = os.environ.get("MAMG_ABLATION_PRESET", "column_op").strip()
# isAblation=True 时可选原组件消融，以及 context_none/context_all/context_confidence。
ABLATION_NAME = ABLATION_PRESET if isAblation else "full"
REAL_VALUE_CONTEXT_MODE = CONTEXT_ENHANCEMENT_MODE
CONTEXT_EXPERIMENT_VERSION = 3

def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


CONTEXT_PREPARE_ONLY = _env_bool("MAMG_CONTEXT_PREPARE_ONLY") is True
CONTEXT_CHECKPOINT_DIR = os.environ.get("MAMG_CONTEXT_CHECKPOINT_DIR", "").strip()


# 调整为 true 代表打开，也可用 MAMG_ALL_TEST=1 从命令行临时打开。
ALL_TEST = False# True 时批量跑除 npd_atomic_tests 外的 RODI 基础场景；npd_atomic_tests 太慢，单独跑
ALL_TEST_DATABASES = [
    # renamed 系列
    "cmt_renamed",
    "conference_renamed",
    "sigkdd_renamed",
    # structured 系列
    "cmt_structured",
    "conference_structured",
    "sigkdd_structured",
    # mixed / combined case
    "sigkdd_mixed",
    # missing FK
    "conference_nofks",
    # denormalized
    "cmt_denormalized",
    # geodata
    "mondial_rel",
]

# w/o Pattern Recognition: force every table to SE.
WO_PATTERN_RECOGNITION = False

# w/o FK / Implicit Relation Completion: skip IND FK completion and disable
# implicit relation evidence mining in OP mapping scenario judgement.
# OP 映射智能体
WO_FK_IMPLICIT_RELATION_COMPLETION = False

# w/o RealValue Enhancement: use DP mapping alignment directly as final alignment.
# 纯 DP 映射
WO_REAL_VALUE_ENHANCEMENT = False

# w/o OP Mapping Reasoning: skip OP Step0/Step1/Step2 and prevent R2RML OP fallbacks.
WO_OP_MAPPING_REASONING = False

# w/o DP Mapping LLM Matching: keep DP candidate/rule logic, but force top-1 fallback
# instead of LLM selection. The pipeline still produces a complete mapping.
WO_DP_MAPPING_LLM_MATCHING = False

# w/o specific mapping-pattern components: remove the corresponding pattern
# handling from candidate generation onward. column_op disables column-level
# FK→ObjectProperty handling in SE/SH tables.
WO_SH_PATTERN = False
WO_SR_PATTERN = False
WO_SE_PATTERN = False
WO_COLUMN_OP = False


def _apply_ablation_preset() -> None:
    """用一个 preset 同时设置实验名和开关，避免手动改错。"""
    global ABLATION_NAME
    global WO_PATTERN_RECOGNITION
    global WO_FK_IMPLICIT_RELATION_COMPLETION
    global WO_REAL_VALUE_ENHANCEMENT
    global WO_OP_MAPPING_REASONING
    global WO_DP_MAPPING_LLM_MATCHING
    global WO_SH_PATTERN
    global WO_SR_PATTERN
    global WO_SE_PATTERN
    global WO_COLUMN_OP
    global REAL_VALUE_CONTEXT_MODE

    if not isAblation:
        ABLATION_NAME = "full"
        WO_PATTERN_RECOGNITION = False
        WO_FK_IMPLICIT_RELATION_COMPLETION = False
        WO_REAL_VALUE_ENHANCEMENT = False
        WO_OP_MAPPING_REASONING = False
        WO_DP_MAPPING_LLM_MATCHING = False
        WO_SH_PATTERN = False
        WO_SR_PATTERN = False
        WO_SE_PATTERN = False
        WO_COLUMN_OP = False
        REAL_VALUE_CONTEXT_MODE = CONTEXT_ENHANCEMENT_MODE
        return

    if ABLATION_PRESET == "custom":
        return

    presets = {
        "pattern": {"WO_PATTERN_RECOGNITION": True},
        "fk_implicit": {"WO_FK_IMPLICIT_RELATION_COMPLETION": True},
        "real_value": {"WO_REAL_VALUE_ENHANCEMENT": True},
        "op_mapping": {"WO_OP_MAPPING_REASONING": True},
        "dp_mapping_matching": {"WO_DP_MAPPING_LLM_MATCHING": True},
        "sh": {"WO_SH_PATTERN": True},
        "sr": {"WO_SR_PATTERN": True},
        "se": {"WO_SE_PATTERN": True},
        "column_op": {"WO_COLUMN_OP": True},
        "context_none": {"REAL_VALUE_CONTEXT_MODE": "none"},
        "context_all": {"REAL_VALUE_CONTEXT_MODE": "all"},
        "context_confidence": {"REAL_VALUE_CONTEXT_MODE": "confidence"},
        "context_base": {"REAL_VALUE_CONTEXT_MODE": "none"},
    }
    if ABLATION_PRESET not in presets:
        raise ValueError(
            f"未知 ABLATION_PRESET={ABLATION_PRESET!r}，"
            "可选: pattern / fk_implicit / real_value / op_mapping / dp_mapping_matching / "
            "sh / sr / se / column_op / context_none / context_all / "
            "context_confidence / context_base / custom"
        )

    ABLATION_NAME = ABLATION_PRESET
    WO_PATTERN_RECOGNITION = False
    WO_FK_IMPLICIT_RELATION_COMPLETION = False
    WO_REAL_VALUE_ENHANCEMENT = False
    WO_OP_MAPPING_REASONING = False
    WO_DP_MAPPING_LLM_MATCHING = False
    WO_SH_PATTERN = False
    WO_SR_PATTERN = False
    WO_SE_PATTERN = False
    WO_COLUMN_OP = False
    REAL_VALUE_CONTEXT_MODE = CONTEXT_ENHANCEMENT_MODE
    for key, value in presets[ABLATION_PRESET].items():
        globals()[key] = value


_apply_ablation_preset()

if isAblation:
    for _env_name, _switch_name in {
        "MAMG_WO_PATTERN_RECOGNITION": "WO_PATTERN_RECOGNITION",
        "MAMG_WO_FK_IMPLICIT_RELATION_COMPLETION": "WO_FK_IMPLICIT_RELATION_COMPLETION",
        "MAMG_WO_REAL_VALUE_ENHANCEMENT": "WO_REAL_VALUE_ENHANCEMENT",
        "MAMG_WO_OP_MAPPING_REASONING": "WO_OP_MAPPING_REASONING",
        "MAMG_WO_DP_MAPPING_LLM_MATCHING": "WO_DP_MAPPING_LLM_MATCHING",
        "MAMG_WO_SH_PATTERN": "WO_SH_PATTERN",
        "MAMG_WO_SR_PATTERN": "WO_SR_PATTERN",
        "MAMG_WO_SE_PATTERN": "WO_SE_PATTERN",
        "MAMG_WO_COLUMN_OP": "WO_COLUMN_OP",
    }.items():
        _value = _env_bool(_env_name)
        if _value is not None:
            globals()[_switch_name] = _value

    ABLATION_NAME = os.environ.get("MAMG_ABLATION_NAME", ABLATION_NAME)
    REAL_VALUE_CONTEXT_MODE = os.environ.get(
        "MAMG_CONTEXT_ENHANCEMENT_MODE",
        REAL_VALUE_CONTEXT_MODE,
    ).strip().lower()


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "OPMapping") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "OPMapping"))
if str(PROJECT_ROOT / "DPMapping") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "DPMapping"))

from config import (  # noqa: E402
    CURRENT_DATABASE,
    DB_SCHEMA_NAME,
    MAPPING_BASE_URL,
    ONTOLOGY_PATH,
    OUTPUT_MAPPING_FILENAME,
    USE_LLM_FK_MERGE,
    USE_LLM_EMPTY_FK_COMPLETION,
    USE_LLM_ONTOLOGY_FK_FALLBACK,
)
from utils.db_utils import read_schema  # noqa: E402
from utils.merge_fks import merge_fks_into_schema, merge_llm_fks_into_schema  # noqa: E402
from utils.ontology_utils import read_ontology  # noqa: E402

from FKCompletion_agent import (  # noqa: E402
    allocate_targets_and_shooters,
    discover_empty_semantic_foreign_keys,
    discover_implicit_foreign_keys,
    discover_ontology_llm_fallback_foreign_keys,
)
from classify_agent import (  # noqa: E402
    battle_layer,
    cal_num_fks,
    classify_agent,
    classify_rule,
    find_difference,
)
from candidate_generation import generate_candidates  # noqa: E402
import DPMapping.data_property_mapping_agent as data_property_mapping_agent  # noqa: E402
from RealValue.real_value_enhancement_agent import run_real_value_enhancement  # noqa: E402
from OPMapping.equivalence_op_module import run_equivalence_op_module  # noqa: E402
from utils.llm_metrics import (  # noqa: E402
    diff_llm_metrics,
    reset_llm_metrics,
    snapshot_llm_metrics,
)
import r2rml_generator  # noqa: E402


if isAblation:
    ABLATION_ROOT_DIR = PROJECT_ROOT / "output" / "ablation" / ABLATION_NAME
    ABLATION_OUTPUT_DIR = ABLATION_ROOT_DIR / CURRENT_DATABASE
else:
    ABLATION_ROOT_DIR = PROJECT_ROOT / "output"
    ABLATION_OUTPUT_DIR = ABLATION_ROOT_DIR / CURRENT_DATABASE


def _banner(step: str) -> None:
    print(f"\n{'=' * 72}\n{step}\n{'=' * 72}")


def _save_json(data: Any, filename: str) -> None:
    ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = ABLATION_OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> saved {path}")


def _current_switches() -> dict[str, bool | str]:
    return {
        "isAblation": isAblation,
        "ABLATION_PRESET": ABLATION_PRESET,
        "ABLATION_NAME": ABLATION_NAME,
        "CURRENT_DATABASE": CURRENT_DATABASE,
        "WO_PATTERN_RECOGNITION": WO_PATTERN_RECOGNITION,
        "WO_FK_IMPLICIT_RELATION_COMPLETION": WO_FK_IMPLICIT_RELATION_COMPLETION,
        "WO_REAL_VALUE_ENHANCEMENT": WO_REAL_VALUE_ENHANCEMENT,
        "WO_OP_MAPPING_REASONING": WO_OP_MAPPING_REASONING,
        "WO_DP_MAPPING_LLM_MATCHING": WO_DP_MAPPING_LLM_MATCHING,
        "WO_SH_PATTERN": WO_SH_PATTERN,
        "WO_SR_PATTERN": WO_SR_PATTERN,
        "WO_SE_PATTERN": WO_SE_PATTERN,
        "WO_COLUMN_OP": WO_COLUMN_OP,
        "REAL_VALUE_CONTEXT_MODE": REAL_VALUE_CONTEXT_MODE,
        "CONTEXT_EXPERIMENT_VERSION": CONTEXT_EXPERIMENT_VERSION,
        "CONTEXT_PREPARE_ONLY": CONTEXT_PREPARE_ONLY,
        "CONTEXT_CHECKPOINT_DIR": CONTEXT_CHECKPOINT_DIR,
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_id(alignment: dict, candidates: dict, enriched_schema: dict) -> str:
    payload = json.dumps(
        {
            "alignment": alignment,
            "candidates": candidates,
            "enriched_schema": enriched_schema,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_id_from_dir(checkpoint_dir: Path) -> str:
    return _checkpoint_id(
        _load_json(checkpoint_dir / "dp_mapping_alignment.json"),
        _load_json(checkpoint_dir / "dp_mapping_candidates.json"),
        _load_json(checkpoint_dir / "enriched_schema.json"),
    )


def _merge_llm_metrics(prefix: dict, branch: dict) -> dict:
    numeric_keys = (
        "api_attempts",
        "llm_calls",
        "failed_attempts",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_input_token_calls",
    )
    merged = {
        key: int(prefix.get(key, 0)) + int(branch.get(key, 0))
        for key in numeric_keys
    }
    merged["models"] = {}
    for source in (prefix.get("models", {}), branch.get("models", {})):
        for model, values in source.items():
            target = merged["models"].setdefault(
                model,
                {
                    "llm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )
            for key in target:
                target[key] += int(values.get(key, 0))
    return merged


def _load_context_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    prefix_summary = _load_json(checkpoint_dir / "context_prefix_summary.json")
    checkpoint = {
        "prefix_summary": prefix_summary,
        "schema": _load_json(checkpoint_dir / "schema.json"),
        "ontology": read_ontology(ONTOLOGY_PATH),
        "discovered_fks": _load_json(checkpoint_dir / "discovered_fks.json"),
        "pattern_result": _load_json(checkpoint_dir / "pattern_result.json"),
        "agent_result": _load_json(checkpoint_dir / "classify_agent_result.json"),
        "enriched_schema": _load_json(checkpoint_dir / "enriched_schema.json"),
        "candidates": _load_json(checkpoint_dir / "dp_mapping_candidates.json"),
        "disabled_pattern_tables": _load_json(checkpoint_dir / "disabled_pattern_tables.json"),
        "alignment": _load_json(checkpoint_dir / "dp_mapping_alignment.json"),
        "low_conf": _load_json(checkpoint_dir / "dp_mapping_low_confidence.json"),
    }
    actual_id = _checkpoint_id_from_dir(checkpoint_dir)
    expected_id = prefix_summary.get("context_checkpoint_id")
    if not expected_id or actual_id != expected_id:
        raise RuntimeError(
            "冻结 checkpoint 内容与指纹不一致，拒绝继续运行。"
        )
    return checkpoint


def _disabled_pattern_components() -> set[str]:
    disabled = set()
    if WO_SH_PATTERN:
        disabled.add("SH")
    if WO_SR_PATTERN:
        disabled.add("SR")
    if WO_SE_PATTERN:
        disabled.add("SE")
    if WO_COLUMN_OP:
        disabled.add("COLUMN_OP")
    return disabled


def _count_patterns(pattern_result: dict) -> dict[str, int]:
    counts = {"SE": 0, "SH": 0, "SR": 0, "OTHER": 0}
    for pattern in pattern_result.values():
        key = pattern or "SE"
        if key in counts:
            counts[key] += 1
        else:
            counts["OTHER"] += 1
    return counts


def _merge_optional_llm_fks(enriched_schema: dict, agent_result: dict) -> dict:
    if not USE_LLM_FK_MERGE:
        return enriched_schema
    if WO_FK_IMPLICIT_RELATION_COMPLETION:
        print("  LLM FK merge skipped because FK/implicit completion is ablated.")
        return enriched_schema
    return merge_llm_fks_into_schema(enriched_schema, agent_result)


def _run_fk_completion_if_enabled(schema: dict) -> tuple[dict, dict]:
    if WO_FK_IMPLICIT_RELATION_COMPLETION:
        print("  FKCompletion disabled: using only physical foreign keys.")
        return schema, {}

    allocation = allocate_targets_and_shooters(schema)
    discovered_fks = discover_implicit_foreign_keys(
        allocation,
        schema_name=DB_SCHEMA_NAME,
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
    enriched_schema = merge_fks_into_schema(schema, discovered_fks)
    return enriched_schema, discovered_fks


def _run_pattern_recognition(enriched_schema: dict) -> tuple[dict, dict]:
    if WO_PATTERN_RECOGNITION:
        pattern_result = {table_name: "SE" for table_name in enriched_schema}
        print("  Pattern Recognition disabled: all tables forced to SE.")
        return pattern_result, {}

    rule_result = classify_rule(enriched_schema)
    agent_result = classify_agent(enriched_schema)
    diff = find_difference(rule_result, agent_result)
    pattern_result = battle_layer(
        diff,
        rule_result,
        agent_result,
        cal_num_fks(enriched_schema),
        enriched_schema,
    )
    print(f"  pattern conflicts resolved by Battle Layer: {len(diff)}")
    return pattern_result, agent_result


def _run_dp_mapping(
    candidates: dict,
    ontology: dict | None = None,
) -> tuple[dict, dict]:
    old_budget_flag = data_property_mapping_agent._LLM_BUDGET_EXHAUSTED
    if WO_DP_MAPPING_LLM_MATCHING:
        data_property_mapping_agent._LLM_BUDGET_EXHAUSTED = True
        print("  DP Mapping LLM Matching disabled: agent will use top-1 fallbacks.")
    try:
        alignment = data_property_mapping_agent.run_data_property_mapping(
            candidates,
            ontology=ontology,
        )
        low_conf = data_property_mapping_agent.collect_low_confidence_data_property_mappings(alignment)
        return alignment, low_conf
    finally:
        data_property_mapping_agent._LLM_BUDGET_EXHAUSTED = old_budget_flag


def _run_real_value_if_enabled(
    alignment: dict,
    low_conf: dict,
    candidates: dict,
    ontology: dict,
    enriched_schema: dict,
) -> tuple[dict, dict]:
    if WO_REAL_VALUE_ENHANCEMENT or REAL_VALUE_CONTEXT_MODE == "none":
        print("  RealValue enhancement disabled: using DP mapping alignment directly.")
        return alignment, {}

    if REAL_VALUE_CONTEXT_MODE == "all":
        selected = data_property_mapping_agent.collect_all_context_data_property_mappings(alignment)
        print(
            "  Context mode=all: "
            f"{len(selected)} tables and "
            f"{sum(len(v.get('columns_low', [])) for v in selected.values())} DP-stage columns."
        )
    else:
        selected = low_conf
        print(
            "  Context mode=confidence: "
            f"{len(selected)} low-confidence tables selected."
        )

    if not selected:
        print("  No low-confidence DP entries: RealValue enhancement naturally skipped.")
        return alignment, {}
    return (
        run_real_value_enhancement(
            alignment,
            selected,
            candidates,
            ontology=ontology,
            enriched_schema=enriched_schema,
            force_all_context=REAL_VALUE_CONTEXT_MODE == "all",
        ),
        selected,
    )


def _raise_r2rml_op_fallback_thresholds():
    keys = [
        "R2RML_OP_FALLBACK_MIN_SCORE",
        "R2RML_OP_FALLBACK_MIN_NAME_SCORE",
        "R2RML_OP_FALLBACK_MIN_SIDE_SCORE",
        "R2RML_PK_OP_MIN_SCORE",
        "R2RML_PK_OP_MIN_NAME_SCORE",
        "R2RML_PK_OP_MIN_SIDE_SCORE",
    ]
    old_values = {key: getattr(r2rml_generator, key) for key in keys}
    for key in keys:
        setattr(r2rml_generator, key, 2.0)
    return old_values


def _restore_attrs(module, values: dict[str, Any]) -> None:
    for key, value in values.items():
        setattr(module, key, value)


def _run_op_mapping_if_enabled(final_alignment: dict, ontology: dict, enriched_schema: dict) -> dict:
    if WO_OP_MAPPING_REASONING:
        print("  OP Mapping disabled: Step0/Step1/Step2 skipped.")
        return {"step1": {}, "step2_orphans": []}

    if WO_FK_IMPLICIT_RELATION_COMPLETION:
        print("  WO_FK_IMPLICIT_RELATION_COMPLETION does not affect the active OP module; OP uses only the enriched schema already produced upstream.")

    scenarios = {
        "_skipped": True,
        "reason": "Legacy OP scenario judgement has been removed; ablation runs the equivalence-column OP module directly.",
    }
    _save_json(scenarios, "op_mapping_scenarios.json")

    op_mapping_step1_result = run_equivalence_op_module(
        final_alignment=final_alignment,
        ontology=ontology,
        enriched_schema=enriched_schema,
        schema_name=DB_SCHEMA_NAME,
        output_dir=str(ABLATION_OUTPUT_DIR),
        ontology_path=ONTOLOGY_PATH,
    )
    _save_json(op_mapping_step1_result, "op_mapping_step1_result.json")

    op_mapping_step2_result = {
        "orphan_matches": [],
        "skipped": True,
        "reason": "Legacy OP Step2 orphan completion has been removed in ablation mode as well.",
    }
    _save_json(op_mapping_step2_result, "op_mapping_step2_result.json")

    return {
        "step1": op_mapping_step1_result,
        "step2_orphans": [],
    }


def _generate_mapping(
    final_alignment: dict,
    op_mapping_full: dict,
    enriched_schema: dict,
    ontology: dict,
) -> str:
    old_r2rml_values = None
    try:
        if WO_OP_MAPPING_REASONING:
            old_r2rml_values = _raise_r2rml_op_fallback_thresholds()
            print("  R2RML OP fallbacks disabled to keep w/o OP mapping pure.")

        return r2rml_generator.generate_r2rml(
            final_alignment=final_alignment,
            op_mapping_full=op_mapping_full,
            enriched_schema=enriched_schema,
            ontology=ontology,
            base_url=MAPPING_BASE_URL,
            prefix=CURRENT_DATABASE.replace("_", ""),
        )
    finally:
        if old_r2rml_values is not None:
            _restore_attrs(r2rml_generator, old_r2rml_values)


def run_ablation() -> dict[str, Any]:
    started_at = time.time()
    reset_llm_metrics()
    ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_label = f"Ablation: {ABLATION_NAME}" if isAblation else "Full pipeline"
    _banner(run_label)
    print(f"  database: {CURRENT_DATABASE}")
    print(f"  root:     {ABLATION_ROOT_DIR}")
    print(f"  output:   {ABLATION_OUTPUT_DIR}")
    print("  switches:")
    for key, value in _current_switches().items():
        print(f"    {key}: {value}")

    _save_json(_current_switches(), "ablation_switches.json")

    prefix_summary: dict[str, Any] = {}
    if CONTEXT_CHECKPOINT_DIR:
        checkpoint_dir = Path(CONTEXT_CHECKPOINT_DIR).expanduser().resolve()
        _banner(f"0-4. Load frozen context checkpoint: {checkpoint_dir}")
        checkpoint = _load_context_checkpoint(checkpoint_dir)
        prefix_summary = checkpoint["prefix_summary"]
        schema = checkpoint["schema"]
        ontology = checkpoint["ontology"]
        discovered_fks = checkpoint["discovered_fks"]
        pattern_result = checkpoint["pattern_result"]
        agent_result = checkpoint["agent_result"]
        enriched_schema = checkpoint["enriched_schema"]
        candidates = checkpoint["candidates"]
        disabled_pattern_tables = checkpoint["disabled_pattern_tables"]
        alignment = checkpoint["alignment"]
        low_conf = checkpoint["low_conf"]
        disabled_patterns = set(prefix_summary.get("disabled_pattern_components", []))
        pattern_counts = prefix_summary.get("pattern_counts") or _count_patterns(pattern_result)

        # Copy the frozen inputs into every branch output for auditability.
        for data, filename in (
            (schema, "schema.json"),
            (discovered_fks, "discovered_fks.json"),
            (pattern_result, "pattern_result.json"),
            (agent_result, "classify_agent_result.json"),
            (enriched_schema, "enriched_schema.json"),
            (candidates, "dp_mapping_candidates.json"),
            (disabled_pattern_tables, "disabled_pattern_tables.json"),
            (alignment, "dp_mapping_alignment.json"),
            (low_conf, "dp_mapping_low_confidence.json"),
        ):
            _save_json(data, filename)
        print(
            "  Frozen prefix loaded: "
            f"calls={prefix_summary.get('prefix_llm_metrics', {}).get('llm_calls', 0)}, "
            f"input_tokens={prefix_summary.get('prefix_llm_metrics', {}).get('input_tokens', 0)}"
        )
    else:
        _banner("0. Load schema and ontology")
        schema = read_schema()
        ontology = read_ontology(ONTOLOGY_PATH)
        _save_json(schema, "schema.json")
        print(f"  tables={len(schema)}, classes={len(ontology.get('classes', []))}")

        _banner("1. FK / implicit completion")
        enriched_schema, discovered_fks = _run_fk_completion_if_enabled(schema)
        _save_json(discovered_fks, "discovered_fks.json")

        _banner("2. Pattern recognition")
        pattern_result, agent_result = _run_pattern_recognition(enriched_schema)
        enriched_schema = _merge_optional_llm_fks(enriched_schema, agent_result)
        _save_json(pattern_result, "pattern_result.json")
        _save_json(agent_result, "classify_agent_result.json")
        _save_json(enriched_schema, "enriched_schema.json")

        _banner("3. DP mapping candidates")
        disabled_patterns = _disabled_pattern_components()
        if disabled_patterns:
            print(f"  Disabled pattern components: {', '.join(sorted(disabled_patterns))}")
        pattern_counts = _count_patterns(pattern_result)
        candidates = generate_candidates(
            enriched_schema,
            pattern_result,
            ontology,
            disabled_patterns=disabled_patterns,
        )
        disabled_pattern_tables = {
            table: pattern
            for table, pattern in pattern_result.items()
            if (pattern or "SE").upper() in {p.upper() for p in disabled_patterns}
        }
        if disabled_pattern_tables:
            print(f"  skipped tables by component ablation: {len(disabled_pattern_tables)}")
        _save_json(candidates, "dp_mapping_candidates.json")
        _save_json(disabled_pattern_tables, "disabled_pattern_tables.json")

        _banner("4. DP mapping")
        alignment, low_conf = _run_dp_mapping(candidates, ontology=ontology)
        _save_json(alignment, "dp_mapping_alignment.json")
        _save_json(low_conf, "dp_mapping_low_confidence.json")

        if CONTEXT_PREPARE_ONLY:
            prefix_summary = {
                **_current_switches(),
                "output_dir": str(ABLATION_OUTPUT_DIR),
                "num_tables": len(schema),
                "num_patterns": len(pattern_result),
                "pattern_counts": pattern_counts,
                "disabled_pattern_components": sorted(disabled_patterns),
                "num_disabled_pattern_tables": len(disabled_pattern_tables),
                "num_discovered_fks": sum(len(v) for v in discovered_fks.values()),
                "num_low_conf_tables": len(low_conf),
                "prefix_llm_metrics": snapshot_llm_metrics(),
                "prefix_elapsed_seconds": round(time.time() - started_at, 2),
                # Compute after JSON persistence so Python-only value types
                # cannot make the in-memory and reloaded fingerprints differ.
                "context_checkpoint_id": _checkpoint_id_from_dir(
                    ABLATION_OUTPUT_DIR
                ),
                "status": "prepared",
            }
            _save_json(prefix_summary, "context_prefix_summary.json")
            _save_json(prefix_summary, "ablation_summary.json")
            print("\nFrozen context checkpoint prepared.")
            return prefix_summary

    _banner("5. RealValue enhancement")
    real_value_started_at = time.time()
    real_value_metrics_before = snapshot_llm_metrics()
    final_alignment, context_targets = _run_real_value_if_enabled(
        alignment,
        low_conf,
        candidates,
        ontology,
        enriched_schema,
    )
    real_value_elapsed = time.time() - real_value_started_at
    real_value_metrics = diff_llm_metrics(
        real_value_metrics_before,
        snapshot_llm_metrics(),
    )
    real_value_metrics.update(
        {
            "runtime_seconds": round(real_value_elapsed, 2),
            "selected_tables": len(context_targets),
            "selected_columns": sum(
                len(v.get("columns_low", [])) for v in context_targets.values()
            ),
        }
    )
    _save_json(context_targets, "context_enhancement_targets.json")
    _save_json(real_value_metrics, "context_efficiency_metrics.json")
    _save_json(final_alignment, "final_alignment.json")

    _banner("6. OP mapping reasoning")
    op_mapping_full = _run_op_mapping_if_enabled(final_alignment, ontology, enriched_schema)
    _save_json(op_mapping_full, "op_mapping_full_result.json")

    _banner("7. R2RML generation")
    r2rml = _generate_mapping(final_alignment, op_mapping_full, enriched_schema, ontology)
    output_ttl = ABLATION_OUTPUT_DIR / OUTPUT_MAPPING_FILENAME
    output_ttl.write_text(r2rml, encoding="utf-8")
    print(f"  -> saved {output_ttl}")

    root_ttl = ABLATION_ROOT_DIR / OUTPUT_MAPPING_FILENAME
    ABLATION_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    root_ttl.write_text(r2rml, encoding="utf-8")
    print(f"  -> saved {root_ttl}")

    branch_metrics = snapshot_llm_metrics()
    branch_elapsed = round(time.time() - started_at, 2)
    prefix_metrics = prefix_summary.get("prefix_llm_metrics", {})
    prefix_elapsed = float(prefix_summary.get("prefix_elapsed_seconds", 0.0))
    total_metrics = (
        _merge_llm_metrics(prefix_metrics, branch_metrics)
        if prefix_summary
        else branch_metrics
    )
    total_elapsed = round(prefix_elapsed + branch_elapsed, 2)
    summary = {
        **_current_switches(),
        "ablation_root_dir": str(ABLATION_ROOT_DIR),
        "output_dir": str(ABLATION_OUTPUT_DIR),
        "output_mapping": str(output_ttl),
        "root_output_mapping": str(root_ttl),
        "num_tables": len(schema),
        "num_patterns": len(pattern_result),
        "pattern_counts": pattern_counts,
        "disabled_pattern_components": sorted(disabled_patterns),
        "num_disabled_pattern_tables": len(disabled_pattern_tables),
        "num_discovered_fks": sum(len(v) for v in discovered_fks.values()),
        "num_low_conf_tables": len(low_conf),
        "context_efficiency": real_value_metrics,
        "common_prefix_llm_metrics": prefix_metrics,
        "branch_llm_metrics": branch_metrics,
        "pipeline_llm_metrics": total_metrics,
        "common_prefix_elapsed_seconds": prefix_elapsed,
        "branch_elapsed_seconds": branch_elapsed,
        "context_checkpoint_id": prefix_summary.get("context_checkpoint_id"),
        "num_op_mapping_step1_relations": len(op_mapping_full.get("step1", {})),
        "elapsed_seconds": total_elapsed,
        "status": "completed",
    }
    _save_json(summary, "ablation_summary.json")
    print("\nAblation completed.")
    return summary


def main() -> None:
    if ALL_TEST and os.environ.get("MAMG_ABLATION_CHILD") != "1":
        run_all_tests()
        return

    try:
        run_ablation()
    except Exception as exc:
        ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        error_summary = {
            **_current_switches(),
            "ablation_root_dir": str(ABLATION_ROOT_DIR),
            "output_dir": str(ABLATION_OUTPUT_DIR),
            "status": "failed",
            "error": repr(exc),
        }
        _save_json(error_summary, "ablation_summary.json")
        raise


def run_all_tests() -> None:
    """
    批量运行短场景消融。

    每个数据库用独立 Python 子进程运行，避免 config.py 和各模块 import 后
    缓存 CURRENT_DATABASE 导致串库。npd_atomic_tests 不在这里跑，建议单独测。
    """
    root_dir = PROJECT_ROOT / "output" / "ablation" / ABLATION_NAME
    root_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    results = []
    print(f"\n{'=' * 72}")
    print(f"Batch Ablation: {ABLATION_NAME}")
    print(f"Output root: {root_dir}")
    print(f"Databases: {', '.join(ALL_TEST_DATABASES)}")
    print(f"{'=' * 72}\n")

    for idx, db_name in enumerate(ALL_TEST_DATABASES, 1):
        print(f"\n{'#' * 72}")
        print(f"[{idx}/{len(ALL_TEST_DATABASES)}] Running {db_name}")
        print(f"{'#' * 72}\n")

        env = os.environ.copy()
        env["MAMG_ABLATION_CHILD"] = "1"
        env["MAMG_CURRENT_DATABASE"] = db_name

        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "ablation.py")],
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        summary_path = root_dir / db_name / "ablation_summary.json"
        summary = {
            "CURRENT_DATABASE": db_name,
            "returncode": proc.returncode,
            "summary_path": str(summary_path),
        }
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summary.update(json.load(f))
            except Exception as exc:
                summary["summary_read_error"] = repr(exc)
        else:
            summary["status"] = "failed"
            summary["error"] = "summary file not found"
        results.append(summary)

        if proc.returncode != 0:
            print(f"[WARN] {db_name} failed with returncode={proc.returncode}; continuing.")

    aggregate = {
        "ABLATION_NAME": ABLATION_NAME,
        "ALL_TEST_DATABASES": ALL_TEST_DATABASES,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "results": results,
    }
    aggregate_path = root_dir / "all_test_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nBatch ablation completed. Summary: {aggregate_path}")


if __name__ == "__main__":
    main()
